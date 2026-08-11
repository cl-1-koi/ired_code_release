# IRED Sudoku Reannealing Isolation

Date: 2026-08-11

## Trigger

Both expand-then-consolidate children passed the SATNet-validation gate but
failed the fixed 2,048-board RRN-validation gate.  `EC-C0` was close: it
retained 57 of the parent's 62 additional strict solves (91.9%, versus the
declared 95% requirement) and stayed within the 0.5-point unknown-cell
tolerance.  `EC-T2` was weaker.  The prior contract says to stop and separate
optimizer momentum from distribution-shift abruptness before changing either.

## Frozen parent and control

Both new arms restart from the same signed `SN-T2` step-50,400 checkpoint,
including the frozen model and EMA.  Both use the released `C0` loss, reset the
batcher at the lineage boundary, train exactly 200 child steps to 50,600, and
retain the existing learning rate, Adam betas, clipping, precision, seed, and
batch size.  `EC-C0` is the already-completed control: retained Adam state and
an abrupt 50/50 SATNet/RRN mixture.

## Paired isolation arms

- `ER-O`: use the same abrupt 9,000 SATNet + 9,000 RRN training set as
  `EC-C0`, but initialize a fresh Adam state at the lineage boundary.  This
  changes optimizer state only.
- `ER-G`: retain the parent's Adam state, but use 9,000 SATNet + the first
  3,000 RRN rows (75/25 mixture) for the first 200-step bridge.  This changes
  distribution-shift amplitude only.  It is not yet a 50/50 consolidation.

The code must record both the exact RRN row count and whether optimizer state
was reset in the signed child manifest.  A reset flag applies only while
starting a new lineage; resuming that child must restore its own optimizer.

## Gate

Evaluate both arms on all 1,000 SATNet-validation boards and the identical
2,048-board RRN-validation panel (seed 314159, batch 64, 20 inner-loop steps).
Use the same gate as the preceding experiment: retain at least 95% of the
parent's strict-solve improvement over the original 50,000-step checkpoint on
both distributions, with no more than 0.5 percentage points of parent
unknown-cell accuracy lost.

- If only `ER-O` passes, optimizer momentum is the primary destructive factor.
- If only `ER-G` passes, abrupt distribution shift is primary; advance the
  mixture in separately gated 25-point increments.
- If both pass, use a 2x2 follow-up to test whether their effects combine, then
  prefer the lower-intervention path.
- If neither passes, neither simple reannealing mechanism explains the loss;
  freeze the parent and inspect short-horizon trajectory/gradient geometry
  before further optimization.

Do not touch the 18,000-board hard-test split until an endpoint passes this
gate.
