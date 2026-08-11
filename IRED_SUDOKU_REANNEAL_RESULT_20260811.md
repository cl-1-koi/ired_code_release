# IRED Sudoku Expand/Contract and Reannealing Result

Date: 2026-08-11

All comparisons use the frozen seed-314159 sampler with 20 inner-loop steps.
Standard validation contains 1,000 boards; RRN validation is the fixed first
2,048 boards.  The original 50,000-step model and the selected `SN-T2` 50,400
parent were evaluated on the same RRN panel before applying the gate.

| Endpoint | Intervention after 50,400 | Standard strict | RRN strict | RRN unknown-cell |
|---|---|---:|---:|---:|
| Base 50k | none | 769/1000 | 115/2048 | 63.919% |
| SN-T2 parent | frozen reference | 921/1000 | 177/2048 | 65.265% |
| EC-C0 | 50/50 mix, released loss, retained Adam | 925/1000 | 172/2048 | 64.911% |
| EC-T2 | 50/50 mix, continued search negatives | 922/1000 | 164/2048 | 64.689% |
| ER-O | 50/50 mix, released loss, fresh Adam | 926/1000 | 170/2048 | 65.005% |
| ER-G | 75/25 mix, released loss, retained Adam | 923/1000 | 167/2048 | 64.971% |

The declared RRN gate required at least 95% of the parent's 62-solve lift over
base: at least 174 strict solves, plus no more than 0.5 percentage points of
parent unknown-cell accuracy lost.  Every child passes the standard gate, but
none reaches 174 RRN solves.  `EC-C0` is closest at 172 and passes the
unknown-cell tolerance.  Continued search-negative pressure, a fresh Adam
state, and the gentler 75/25 bridge are all worse on strict RRN solves.

This rules out the simple claims that retained optimizer momentum alone or the
size of the first distribution shift alone caused the loss.  It does not show
that their effects are zero: confidence intervals overlap and EC-C0 misses the
deterministic gate by only two boards.  Under the frozen contract, however,
there is no selected endpoint and the untouched 18,000-board hard test remains
closed.

The next experiment must freeze the 50,400 parent and measure short-horizon
geometry before optimizing again: per-objective SATNet/RRN gradient cosine and
norm, optimizer-preconditioned update direction, EMA displacement, and sampler
state changes on fixed boards.  A new optimization arm should be declared only
after that diagnostic identifies a separable mechanism.

RunPod `0idy3tz5h7pwcd` used one secure-cloud A40 at $0.44/hour.  Live and
durable closure audits passed 32/32 references for the expand/contract gate and
35/35 for the reannealing isolation.  The ordinary pod was then terminated and
verified absent from the account inventory.
