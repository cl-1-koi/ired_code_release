# IRED Sudoku Search-Negative Intervention

Date: 2026-08-11

## Question

Can contrastive exposure to short model-generated search states improve the
corrective geometry of the 50k IRED Sudoku energy landscape without sacrificing
its reproduced unknown-cell accuracy or strict solve rate?

The frozen calibration found that the 50k model orders valid versus corrupted
states well, but its deterministic sampler collapses coverage and some accepted
energy-descending steps move away from paired valid states.  The released code
already refines contrastive negatives with two energy-gradient steps for
continuous tasks.  The Sudoku branch instead uses only one-shot 5% random digit
corruption.  This experiment ports that existing mechanism to Sudoku.

## Paired arms

Both arms resume the same signed released-code seed-42 checkpoint at step
50,000, including model, Adam, EMA, shuffled batch permutation, and all RNG
states.  Both use the released 1x1 final convolution, batch 64, learning rate
`1e-4`, Adam betas `(0.9,0.99)`, gradient clip 1.0, and the same finite 9,000
training boards.

- `SN-C0`: `sudoku_negative_opt_steps=0`; exact released-loss continuation.
- `SN-T2`: `sudoku_negative_opt_steps=2`; after the same random digit
  corruption and matched-noise construction, take two detached, accepted
  energy-gradient steps before applying the contrastive energy loss.

The extra steps do not backpropagate through the negative-generation path.
They make the negative lower-energy and therefore harder.  No reverse-process
noise is used during training and no inference setting changes between arms.

## Ladder

1. Resume each arm from 50,000 to 50,200.  Require finite loss components and
   gradients, a resumable signed checkpoint, and measured throughput.
2. Run the fixed 128-board standard/hard learned-energy pair panel on raw and
   EMA weights.  Compare ranking, corrective cosine, accepted paired-state MSE,
   and Hamming repair to the frozen 50k baseline.
3. Run a fixed 256-board standard/hard deterministic strict evaluation.  The
   paper-facing unknown-cell metric and strict validity are both mandatory.
4. Continue both healthy arms to 51,000, repeat the pair panel, and evaluate all
   1,000 standard-validation boards.  Use a fixed 256-board hard gate for fast
   paired feedback.
5. Continue to 55k/60k only if `SN-T2` improves corrective geometry or strict
   solve rate without a material loss in unknown-cell accuracy relative to
   `SN-C0`.

## Decision rules

- A loss/gradient non-finite, non-emission, or manifest mismatch stops the arm.
- Pairwise ranking without improved corrective cosine/repair is not a win; the
  intervention targets geometry rather than an already-saturated ordering test.
- A standard or hard unknown-cell drop of more than 0.5 percentage points versus
  the paired control blocks continuation unless strict solving improves enough
  to justify an explicitly reviewed tradeoff.
- If treatment and control both worsen similarly, the finite-data continuation
  effect dominates; do not infer that refinement caused the degradation.
- If treatment improves train-like geometry but not RRN-hard geometry, stop and
  test fresh procedural Sudoku data as a separate arm.
- Reverse-noise population inference remains a frozen evaluation method.  It is
  not mixed into this training comparison.

## Provenance and implementation

The opt-in implementation is commit `04ea293`.  Default zero preserves the
released reproduction path.  The setting and variant name are included in the
training manifest; resuming with a different value fails manifest verification.
Treatment lineage is labeled `warm_start_search_state_negative_intervention`.
Outputs and checkpoints remain outside Git.
