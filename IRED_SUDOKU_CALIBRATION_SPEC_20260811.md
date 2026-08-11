# IRED Sudoku Energy Calibration Contract

Date: 2026-08-11

## Purpose

The existing reproduction measures final solve rate.  That is insufficient to
distinguish a weak learned constraint landscape from a capable landscape that
is searched badly.  This calibration ladder freezes checkpoints and tests the
energy function, its local vector field, and inference search separately.

The released IRED Sudoku CNN and the S4 transformer adaptation are different
systems.  Results from this contract must identify the architecture and may
not be pooled across them.  This first implementation targets the released
IRED CNN checkpoints because those are the paper-facing reproduction.

## Frozen inputs

- Standard validation: the final 1,000 examples of the official SATNet data.
- Hard test: the 18,000-example RRN test set.
- Initial checkpoint ladder: released-code seed 42 at 50k, 100k, and 300k.
- Both raw and EMA weights are evaluated.  Published final evaluation uses EMA.
- Controlled corruptions alter only non-clue cells and use exact edit distances
  `1,2,4,8,16` where a puzzle has enough blanks.
- Diffusion landscapes are `t=0,3,6,9`.  Each valid/corrupt pair receives the
  same Gaussian noise and the same clue clamp.

Every output records the checkpoint and manifest hashes, dataset slice, seed,
weight source, and source revision.  Diagnostics never update weights.

## C0: exact-objective search sanity

A box-preserving discrete simulated annealer uses only Sudoku constraints:
clues and box legality are invariants, while row/column duplicate count is the
energy.  This is not a learned baseline and does not establish IRED quality. It
checks that search, restart accounting, and strict validity metrics can recover
solutions when the objective is exact.

Report solve rate, proposals, accepted proposals, restarts, final energy, and
wall time separately on fixed standard-validation and hard-test slices.

## C1: learned-energy ordering

For every dataset, checkpoint, weight source, landscape, and edit distance,
report

- `P(E(valid) < E(corrupt))` with a Wilson interval;
- the mean and median margin `E(corrupt) - E(valid)`;
- the exact Sudoku conflict-energy margin as a non-learned control.

Chance ordering is 0.5.  A landscape that cannot order nearby valid/corrupt
pairs cannot be expected to support reliable local repair.

## C2: corrective vector field

At each corrupt state, compute the negative learned-energy gradient after
masking clue dimensions.  Report

- cosine similarity with the vector from corrupt state to paired valid state;
- fraction of positive cosines and gradient norm;
- official-step-size proposal acceptance;
- change in paired-state MSE and non-clue Hamming error after the accepted
  proposal.

Energy decrease without correctness improvement is an explicit failure mode,
not evidence of reasoning.

## C3: inference trajectory and restarts

Instrument the released sampler without changing its update rule.  At every
landscape report proposal acceptance, energy, gradient norm, strict solve rate,
unknown-cell accuracy, and exact conflict energy.  Generate a fixed number of
independent restarts once and report, for each prefix `R`:

- first-restart solve rate;
- oracle-any-restart solve rate;
- solve rate when selecting the minimum learned-energy restart;
- rank correlation between final learned energy and exact conflict/error.

This distinguishes generation coverage from learned-energy selection quality.
Existing sealed inner-step sweeps at 50k, 100k, and 300k remain the primary
large-sample compute-scaling panel; the trajectory panel explains their shape.

If independent initial states collapse to the same decoded board, a declared
diagnostic may enable the implementation's reverse-process posterior noise.
That arm is reported separately as `released_code_with_reverse_noise_diagnostic`;
it is not pooled with or described as the released deterministic sampler.

## Decision rules

1. If C0 fails, repair the search/metric harness before changing training.
2. If C1 fails, the training objective or generalization is the bottleneck;
   more inference steps are not the primary intervention.
3. If C1 passes but C2 fails, investigate gradient geometry and step-size
   calibration.
4. If C1/C2 pass but oracle restarts improve while energy selection does not,
   improve selection/calibration rather than representation.
5. If energy selection works but coverage saturates, investigate stochastic
   search, schedules, and training exposure.
6. Compare 50k/100k/300k and raw/EMA before choosing regularization.  A
   worsening landscape with training is evidence for overfitting or objective
   drift; it is not a reason to train longer blindly.

No new training arm is authorized by this document.  The next training change
must be the smallest intervention justified by these frozen diagnostics.
