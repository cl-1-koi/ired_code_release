# IRED Sudoku Calibration Report

Date: 2026-08-11

## Executive result

The released IRED Sudoku model is approximately reproduced on the metric the
paper most likely reports, but not on strict whole-board solving.  The 50k
seed-42 checkpoint obtains 97.7861% unknown-cell accuracy on standard
validation and 63.4313% on RRN hard, versus 99.4% and 62.1% in Table 4.  The
paper does not define which released evaluator field generated its table, but
the numerical match and the field named `accuracy` strongly indicate
unknown-cell accuracy.

Our stronger strict valid/clue-consistent rates are 76.9% standard and 4.4667%
hard.  These are not paper-metric reproduction failures; they show how often
small cell-level errors prevent a complete valid solution.

The most important new result is that deterministic inference hides useful
coverage.  Adding the implementation's reverse-process posterior noise and
selecting the minimum learned-energy candidate over four trajectories raises
strict solving on fixed 256-board panels:

| Dataset | One trajectory | Four: oracle any | Four: min energy | Gain |
| --- | ---: | ---: | ---: | ---: |
| Standard validation | 80.86% (207/256) | 87.89% (225/256) | 87.89% | +7.03 pp |
| RRN hard | 5.47% (14/256) | 10.16% (26/256) | 10.16% | +4.69 pp |

Minimum learned-energy selection equals oracle-any selection in both panels:
every extra solved candidate was selected.  This makes learned energy useful as
a within-puzzle meta-policy even though one deterministic descent path is weak.

## Metric correction

The released `sudoku_accuracy` function returns three distinct quantities:

- `accuracy`: unknown-cell accuracy;
- `consistency`: strict row/column/box validity rate;
- `board_accuracy`: a weak sum-constraint fraction inherited from SATNet.

The paper calls its Table 4 quantity only "accuracy."  Previous program notes
compared its 99.4%/62.1% directly to strict whole-board solves.  That comparison
was apples-to-oranges and has been explicitly withdrawn in the reproduction
specification.  Strict validity remains the application-facing primary metric.

## Frozen checkpoint and compute results

All large evaluations use EMA weights.  Increasing training past the paper's
50k endpoint sharply worsens held-out performance:

| Step | Standard strict | Standard unknown-cell | Hard strict | Hard unknown-cell |
| ---: | ---: | ---: | ---: | ---: |
| 50k | 76.90% | 97.79% | 4.4667% | 63.43% |
| 100k | 34.70% | 93.79% | 0.7556% | 59.77% |
| 300k | 28.90% | 92.71% | 0.1222% | 55.09% |

Those rows use 20 inner steps on all 1,000 standard and 18,000 hard boards.
The sealed inner-step sweeps show that more deterministic optimization does not
repair the later checkpoints.  At 50k, hard strict solving rises only from
3.85% at one inner step to 4.62% at 80.  At 100k and 300k it is flat or worse.
This rules out "train longer" and "take more deterministic gradient steps" as
primary remedies.

The endpoint metric and the paper's claimed inference-scaling mechanism must
therefore be separated.  Although the likely Table 4 hard endpoint is matched,
our hard unknown-cell accuracy slightly *falls* from 64.21% at one inner step
to 63.47% at 80; it does not reproduce Figure 6's claimed substantial gain
from extra optimization.  The stochastic population result below restores a
positive compute curve by expanding coverage rather than repeating the same
deterministic descent.

## Learned landscape calibration

Controlled corruptions change only non-clue cells at exact edit distances
1/2/4/8/16, use matched Gaussian noise, and test diffusion landscapes
`t=0,3,6,9` on 128 standard and 128 hard boards.

- At `t=0`, both 50k and 300k rank every valid board below its paired corrupt
  board at every tested distance.
- Ranking alone is misleading.  On standard EMA, corrective-gradient cosine at
  `t=0` falls from 0.131/0.366 (distance 1/16) at 50k to 0.024/-0.002 at 300k.
  On hard it falls from 0.110/0.335 to 0.008/0.020.
- At the noisiest `t=9`, ranking stays near chance and corrective cosine near
  zero for every checkpoint.
- At intermediate `t=3`, ranking is nearly perfect and discrete Hamming repair
  is strong.  At `t=6`, ordering is distance-dependent and degrades materially
  on the 300k hard checkpoint.
- Official step acceptance guarantees learned-energy descent, not progress
  toward the valid paired state.  Several panels lower learned energy while
  increasing continuous paired-state MSE.

The model therefore learns useful coarse ordering without a uniformly
corrective vector field.  Continued finite-dataset training preserves the easy
ordering test while damaging the geometry used by iterative search.

## Search trajectories and population coverage

The released prediction algorithm is deterministic after its initial Gaussian
state.  On the 50k checkpoint, four different initial states collapse to one
decoded result on 96.9% of standard boards and 81.2% of hard boards in the
32-board trajectory panel.  Mean unique decoded results are only 1.03 and 1.19.

The declared reverse-noise diagnostic changes inference, not weights.  On the
256-board scale-up:

- four trajectories are diverse on 20.3% of standard boards and 94.9% of hard
  boards;
- hard puzzles average 3.38 unique decoded boards across four trajectories;
- within-board learned-energy/error Spearman correlation is positive where
  diversity makes it defined (0.65 against exact conflict on hard);
- two trajectories already improve strict solves to 85.16% standard and 7.42%
  hard; four improve them to 87.89% and 10.16%.

This is evidence for stochastic population inference plus energy selection,
not evidence that the paper's deterministic Algorithm 2 was reproduced with
noise.  The arm is always labeled
`released_code_with_reverse_noise_diagnostic`.

## Exact-objective control

A box-preserving simulated annealer uses exact row/column duplicate energy and
never changes clues.  It solves 16/16 standard boards quickly.  On the first 16
hard boards, the original schedule solves 6/16; deeper incremental search solves
12/16, and the best temperature/cooling arm solves 14/16.  The remaining states
sit at exact conflict energy 2 even after extensive swaps.

This validates strict metrics and shows that exact constraint energy is not by
itself sufficient: search moves and annealing schedule still determine whether
local barriers are crossed.  It also cautions against attributing every
iterative failure to representation.

## Decision and next experiments

1. Keep 50k as the current best checkpoint; do not resume it blindly.
2. Complete the already-running fixed-panel stochastic compute curve at
   restart prefixes 8 and 16.  This tests whether population coverage compounds
   or saturates.
3. If the curve continues upward, scale the stochastic panel and treat energy
   selection over a candidate population as the immediate inference recipe.
4. The smallest training intervention is search-state negative exposure from
   the 50k checkpoint: contrast valid/noisy states against actual failed chain
   states at the same landscape, with a held-out early-stop gate.  The released
   5% random digit corruption already teaches pairwise ordering but does not
   guarantee corrective geometry along the sampler's own state distribution.
   Concretely, start with two detached energy-optimization steps on each Sudoku
   contrastive negative.  This ports the released continuous-task negative
   refinement into the Sudoku branch instead of introducing a new optimizer.
5. Keep that arm separate from procedural-data expansion.  If search-state
   exposure improves train-like but not RRN-hard geometry, the next test is
   fresh generated Sudoku training data rather than additional epochs over the
   same 9,000 boards.

## Provenance

- Calibration contract and pair diagnostics: commit `780137a`.
- Restart diversity and within-board correlation correction: commits `e88c06a`
  and `63b23d8`.
- Reverse-noise diagnostic: commit `5dd1145`.
- Incremental/Numba exact annealer: commit `ab73928`.
- Metric-interpretation addendum: commit `bea147c`.
- Pair results root: `/home/ubuntu/ired-sudoku-data/calibration-780137a/pairs`.
- Deterministic trajectory root:
  `/home/ubuntu/ired-sudoku-data/calibration-63b23d8/search`.
- Stochastic 256-board root:
  `/home/ubuntu/ired-sudoku-data/calibration-5dd1145/search-noise-n256`.
- Exact controls: `/home/ubuntu/ired-sudoku-data/calibration-exact-ab73928-deep`
  and `/home/ubuntu/ired-sudoku-data/calibration-exact-schedule-ab73928`.

Every learned-model summary records source revision, checkpoint SHA-256,
manifest seal, dataset slice, seed, and inference configuration.  Remote hard
results were synced to these durable local roots before analysis.
