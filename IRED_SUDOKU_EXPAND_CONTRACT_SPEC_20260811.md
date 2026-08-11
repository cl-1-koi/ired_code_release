# IRED Sudoku Expand-Then-Consolidate Test

Date: 2026-08-11

## Question

Can the transient policy/energy improvement created by search-state negatives
be retained while broader data and the released loss consolidate it, or does
any additional optimization erase the gain?

`SN-T2` is an aggressive expansion step: it raises standard and RRN solving
quickly, but pre-clip gradients are large and performance peaks early.  The
direct 50/50 data bridge from the original 50k checkpoint fails at 200 steps.
This experiment changes the order: expand first on the familiar distribution,
then switch objectives and data.

## Frozen parent and paired arms

Both arms start from the signed `SN-T2` checkpoint at step 50,400, including
model, Adam, EMA, and process RNG.  Before training, the parent is evaluated on
all 1,000 SATNet-validation boards and a fixed 2,048-board RRN-validation
slice.  Both child arms use the same 18,000-board mixed set from the data-
coverage contract and the same explicit reset batch order.

- `EC-C0`: switch off search-state refinement
  (`sudoku_negative_opt_steps=0`) and consolidate with the released
  one-shot-corruption loss.
- `EC-T2`: keep two-step search-state refinement under the identical mixed-data
  switch.  This is the matched control for whether consolidation specifically
  requires removing the high-gradient pressure.

No learning-rate, optimizer, EMA, architecture, or inference setting changes.
The child target is step 51,400; the first supervised stop is 50,600 (200 child
steps).

## Gates

1. Require finite telemetry, signed resumable checkpoints, and record raw/EMA
   separation plus pre-clip gradient norms.
2. Evaluate full SATNet validation and the fixed 2,048-board RRN-validation
   slice with the same seed/batch protocol as the parent.
3. A child passes only if it retains at least 95% of the parent's strict-solve
   improvement over the original 50k checkpoint on both distributions and
   loses no more than 0.5 percentage points of parent unknown-cell accuracy.
4. If `EC-C0` passes and `EC-T2` fails, objective switching is the mechanism;
   continue `EC-C0` in 200-step windows.
5. If both fail, the abrupt data shift or optimizer momentum is destructive.
   Stop before changing learning rate or optimizer; those become separate
   reannealing arms.
6. If both pass, broader data rather than objective switching is sufficient;
   continue the less volatile arm.

RRN test remains untouched by this selection.  The first full 18,000-board test
evaluation occurs only after an endpoint is selected on RRN validation.
