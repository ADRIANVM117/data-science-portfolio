# Current Research Methodology

## Prediction problem
Information available:
r0 ... r52, corresponding to 09:30–14:00.

Target:
reod = direction of return from 14:00–16:00.

## Dataset structure
Training:
503 chronological days
1,829 equities
dense financial panel

Competition deployment:
different temporal period
different equity universe

## Validation protocol
See EXP_000.

Expanding walk-forward:
Fold 1: 0–302 -> 303–352
...

Fixed equity partition:
E_dev = 1463
E_holdout = 366
seed = 20260908

Primary evaluation:
Joint OOS = future days × unseen equities

Diagnostic:
Temporal OOS = future days × seen equities

## Information boundaries
ID = key only
day = partition only
equity = partition only

Competition test forbidden for research/model selection.

## Current benchmark
Majority baseline:
Temporal OOS accuracy = 0.4147 ± 0.0092
Joint OOS accuracy = 0.4124 ± 0.0117