# IRED Sudoku Data-Coverage Bridge

Date: 2026-08-11

## Question

Can broader clue-distribution coverage stabilize the transient RRN-hard gain
from search-state negative refinement without sacrificing its large SATNet
validation gain?

The finite SATNet training set has 9,000 boards with 31--42 clues (mean 36.22).
RRN train/validation/test span 17--34 clues (mean 25.5).  In the 51k hard gate,
neither arm solved any of the 114 boards with 17--24 clues.  On a larger fixed
2,048-board hard panel, treatment improved from 96/2,048 at the base 50k state
to 145/2,048 at 50,400, then regressed to 108/2,048 at 51k.  The transient
generalized gain and subsequent reversal justify a separate data-coverage
experiment rather than more epochs over the same 9,000 easy boards.

## Frozen inputs and paired arms

Both arms warm-start directly from the signed released-loss seed-42 checkpoint
at step 50,000.  They do not start from the partially overfit `SN-T2` lineage.
The new finite training set contains all 9,000 SATNet training boards plus the
first 9,000 rows of the official 180,000-row RRN training split.  The RRN slice
has the same 17--34 clue support as the full split.  File hashes and row
selection are sealed in each manifest.

Changing the dataset from 9,000 to 18,000 examples requires an explicit batcher
reset.  Model, Adam, EMA, and process RNG state come from the same parent; both
arms receive the same newly seeded shuffled batch order.

- `DC-C0`: released one-shot random corruption,
  `sudoku_negative_opt_steps=0`.
- `DC-T2`: the same mixed batches with two detached accepted energy-gradient
  steps applied to each contrastive negative.

All other architecture, optimizer, EMA, loss-scale, and inference settings
remain unchanged.  This is a coverage bridge, not a claim of procedural or
infinite-data training.

## Ladder

1. Resume both arms to 50,200.  Require signed resumable checkpoints, finite
   telemetry, and record the treatment's pre-clip gradient distribution.
2. Evaluate all 1,000 SATNet validation boards and a fixed 2,048-board slice of
   **RRN validation**, not RRN test.  Report strict solving, unknown-cell
   accuracy, and clue-count buckets.
3. Run the raw/EMA energy-pair panel on 128 SATNet-validation and 128
   RRN-validation boards.
4. Continue to 51k only if at least one mixed-data arm improves RRN-validation
   without losing more than 0.5 percentage points of SATNet unknown-cell
   accuracy relative to its matched prior-data counterpart.
5. Use the full 18,000-board RRN test split once, only after selecting an
   endpoint from validation.  Do not tune on the repeatedly inspected first
   2,048 test boards.

## Attribution rules

- Compare `DC-C0` with `SN-C0` to estimate the data-coverage effect.
- Compare `DC-T2` with `DC-C0` to estimate search-negative refinement under
  matched coverage.
- If both mixed-data arms improve RRN validation, limited clue-distribution
  coverage was the primary bottleneck.
- If only `DC-T2` improves, the refinement becomes a candidate mechanism, but
  it still needs the untouched full-test gate.
- If SATNet improves while RRN validation does not, stop: neither extra finite
  data nor this hard-negative construction has produced general reasoning.
- A large raw/EMA split or persistent gradient clipping is an optimization
  warning even when EMA metrics improve; preserve the best EMA checkpoint and
  do not extrapolate a longer run.

## Provenance

The training variant is `mixed-satnet-rrn9k`.  It seals the official RRN train
SHA-256, `first_9000` row selection, parent checkpoint/manifest hashes, parent
and child batcher sizes, and the explicit lineage-start batcher reset.  RRN
validation support is added to the frozen evaluators; outputs remain outside
Git.
