# IRED Sudoku reproduction contract (2026-08-09)

## Question

Can the released IRED Sudoku method reproduce its reported standard-difficulty
and harder-distribution results when run on the official data and model path?
This benchmark comes before diagnosing the separate AP-MD and insertion-only
failures.

## Frozen provenance

- Paper: *Learning Iterative Reasoning through Energy Diffusion* (ICML 2024),
  SHA-256 `f8348abcc56307cd6cd502c4786cb0dab9dc11feaa5bc2d012661bca2051373c`.
- Upstream code: `yilundu/ired_code_release` commit
  `3d74b85fab7fcf5e28aaf15e9ed3bf51c1a1d545`.
- SATNet standard features SHA-256:
  `8349ff2a210f6a5bffc052dbddbde6b3461ef893122d19b375fc5dd2f444fbef`.
- SATNet standard labels SHA-256:
  `596fa31210b143fed37252d2694caeb297e18998a8127d2c9a58897552da5ec8`.
- Standard split: first 9,000 puzzles for training, final 1,000 for validation.
- RRN harder split: official 18,000-example test CSV, SHA-256
  `1b14d68f58de6f64b3db4032eaf4ea68783471aceb8ff10306b92d0479c6bc99`.

The runtime writes a sealed manifest containing source and data hashes. Training
outputs and checkpoints stay outside Git.

## Frozen training arm

- Released `SudokuEBM`: 384-channel convolutional energy model with six residual
  blocks and a scalar squared-output energy.
- Released diffusion/energy objective: 10 diffusion landscapes, 5% discrete-cell
  corruption negatives, energy-loss scale 0.05.
- Batch 64; Adam learning rate `1e-4`, betas `(0.9, 0.99)`; gradient clip 1.0;
  FP32; EMA 0.995 updated every 10 optimizer steps.
- Seed 42 and resumable shuffled-epoch batches.
- Production target: 50,000 optimizer steps, matching the paper. The public
  `train.py` says 1.3M, so 1.3M is a declared extension only if 50k fails.

## Sampling and metrics

- Sampling uses 10 landscapes and 20 accepted energy-gradient proposals per
  landscape, as in the released Sudoku path.
- The released validation function indexed `samples[-1]`, accidentally evaluating
  only the last board of a batch. This reproduction passes the entire batch.
- The released `sudoku_score` checks only row/column/box digit *sums*. It is
  retained as `published_satnet_constraint_fraction`, but cannot establish that a
  board is a Sudoku solution.
- Primary metric: fraction of boards that are both strictly valid (each digit once
  in every row, column, and box) and clue-consistent.
- Also report exact reference solution, strict validity, clue consistency, all-cell
  accuracy, unknown-cell accuracy, and the published weak score. Store one record
  per evaluated board and a 95% Wilson interval for the primary rate.

## Execution ladder

1. `R0`: train the sealed trajectory through step 200; require finite losses,
   nonzero throughput, a resumable checkpoint, and a 16-board standard-validation
   sampler smoke test. Accuracy at step 200 is diagnostic, not a gate.
2. `R1`: immediately resume the same state to 50,000 steps on the same healthy pod.
   Do not restart or modify the arm after seeing R0 metrics.
3. Evaluate all 1,000 standard-validation boards. If operationally useful, shard
   the fixed evaluation over additional pods without changing seed/index mapping.
4. Evaluate all 18,000 harder RRN boards and compare both strict and published
   metrics to the paper's 99.4% and 62.1% reported figures.
5. If 50k fails, inspect learning curves and evaluator behavior first. The already
   declared next arm is an exact resume to the public code's 1.3M-step target; it
   is not a silent reinterpretation of the 50k reproduction.

## RunPod policy for this ladder

Once R0 is healthy, the queued next experiment is `R1 resume-to-50k`; the pod
continues immediately. After R1, the queued work is standard evaluation, then hard
evaluation. A healthy ordinary pod may remain briefly between these declared
steps: time-to-next-result matters more than saving cents. Artifacts must be hashed
and copied to durable local storage before any eventual termination.
