# EXP_009 - Recency-Weighted Movement Intensity

## Status

Frozen. Not implemented or executed.

## Post-execution factual addendum

The scientific specification below remains the frozen pre-execution contract.
It was subsequently implemented, validated, and executed once using training
data only. The frozen PRIMARY verdict was **FAIL**. Result records are stored
under `experiments/results/EXP_009_*`, including the manifest, preprocessing
parameters, metrics, incremental-AUC, decision, confusion-matrix, summary,
and metadata artifacts. Competition-test data were not accessed.

## Scientific question

Among observations with complete availability of the 12-position recent window
`W`, does a pre-specified monotonic recency-weighted movement-intensity
statistic add Joint OOS Neutral-vs-Directional discrimination beyond the
aggregate unweighted recent intensity established in EXP_008?

The motivating concept is **recency of movement intensity**. Unweighted RMS is
invariant to permutations of the same squared magnitudes inside W. For the
same multiset of magnitudes, the pre-specified conceptual ordering is:

```text
G = [2,2,2,2,2,2,2,2,20,20,20,20]
J = [2,2,2,2,2,2,20,20,20,20,2,2]
H = [20,20,20,20,2,2,2,2,2,2,2,2]

G > J > H
```

This is not a hypothesis about return sign, momentum, order flow, liquidity,
volatility-regime identification, causality, an optimal decay function, or
final ternary challenge utility.

## Target, isolation population, and frozen validation

Use the EXP_002/EXP_008 binary target:

```text
Z = 0  if reod = 0          # neutral
Z = 1  if reod in {-1, +1}  # directional
```

Construct normal frozen EXP_000 subsets first, then independently restrict
each Fit, Temporal OOS, and Joint OOS subset to:

```text
N_obs,W = 12
```

where `N_obs,W` counts observed values in W. Inclusion does not use target
values. The complete-W restriction is deliberate: the pre-contract audit
found all 4,094 possible partial W masks, so a weighted statistic on partial
rows could encode positional availability as well as return recency.

Reuse EXP_000 exactly:

- frozen `E_dev=1463` / `E_holdout=366` partition, seed `20260908`;
- four expanding folds below;
- no holdout equity in `fit_f`;
- Joint OOS as primary and Temporal OOS as diagnostic;
- no purging or embargo; and
- competition-test data fully excluded.

| Fold | Train days | OOS days |
|---|---|---|
| 1 | 0-302 | 303-352 |
| 2 | 0-352 | 353-402 |
| 3 | 0-402 | 403-452 |
| 4 | 0-452 | 453-502 |

```text
fit_f         = train_days_f x E_dev
Temporal OOS  = oos_days_f x E_dev
Joint OOS     = oos_days_f x E_holdout
```

The structural audit found retained Joint OOS samples of approximately
13.2k-13.4k rows, 347-355 equities, 49-50 days, and both binary classes in
every fold. Results apply conditionally to complete-W observations and must
not be generalized automatically to EXP_008's full deployment population.

## Positional window and semantics caveat

```text
W = {r41, r42, ..., r52}
```

W is a positional window of 12 chronologically ordered observations. The
official challenge description is internally inconsistent about absolute
endpoints of the 53 five-minute returns. EXP_009 assigns no exact clock-time
endpoint to `r41`, `r52`, or W.

Every retained W value must be observed and finite. Consequently,
`q = 1[N_obs,W = 0]` is identically zero and must not enter a model.

## Frozen intensity definitions

For every retained row, define the aggregate unweighted intensity:

```text
I_recent = sqrt((1 / 12) * sum_{k=1}^{12} r_{40+k}^2)
```

Freeze the exact recency weights:

| Return | r41 | r42 | r43 | r44 | r45 | r46 | r47 | r48 | r49 | r50 | r51 | r52 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Weight | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |

```text
weight vector = [1, 2, ..., 12]
weight sum    = 78

I_recency = sqrt(sum_{k=1}^{12} k * r_{40+k}^2 / 78)
```

The weights are strictly positive, strictly increasing in positional recency,
parameter-free, and the simplest frozen monotonic rank weighting consistent
with `G > J > H`. They are not claimed to estimate a true market decay.

EXP_009 must not test alternative windows, exponential weights, EWMA,
alternative linear slopes, half-lives, optimized weights, alternative decay
functions, or other intensity transformations.

## Representations and fit-only preprocessing

```text
missing_ratio = number of NaNs among r0,...,r52 / 53

C_009 = [missing_ratio, I_recent_z]
D_009 = [missing_ratio, I_recent_z, I_recency_z]
```

`missing_ratio` is exactly the EXP_002/EXP_008 feature and remains unscaled.

For each fold, after the complete-W restriction:

1. calculate target-free raw `I_recent` and `I_recency` row-wise;
2. fit one `StandardScaler` on retained Fit raw `I_recent` only;
3. fit a separate `StandardScaler` on retained Fit raw `I_recency` only;
4. transform retained Fit, Temporal OOS, and Joint OOS using unchanged Fit
   scalers; and
5. use the transformed columns `I_recent_z` and `I_recency_z`.

No imputation is allowed because W is complete. C_009 and D_009 must retain
identical IDs, targets, membership, and ordering. The only treatment
difference is `I_recency_z`.

No positional masks, signs, raw returns, maximum-return features, maximum
positions, slopes, rolling/EWMA features, volatility estimators, or manually
engineered interactions may enter either model.

## Fixed model and baseline

Fit independent C_009 and D_009 models in each fold using exactly the
EXP_002/EXP_008 Logistic Regression configuration:

```python
LogisticRegression(
    solver="lbfgs",
    penalty="l2",
    C=1.0,
    fit_intercept=True,
    class_weight=None,
    max_iter=1000,
    tol=1e-4,
    random_state=20260908,
)
```

No tuning, class-weight change, threshold change, or model-family comparison
is permitted.

Determine the binary majority baseline from retained Fit `Z` only. Class order
is `[0, 1]`, so a tie selects `Z=0`. Apply that class to corresponding retained
OOS subsets. Probabilities refer to `P(Z=1)`; the fixed threshold is `0.5`,
with equality predicting `Z=1`.

## Evaluation

For both representations, each fold, and retained Temporal/Joint OOS, report:

- ROC-AUC from `P(Z=1)`;
- Balanced Accuracy, Accuracy, neutral recall, and directional recall;
- confusion matrix with labels `[0, 1]`; and
- Accuracy and Balanced Accuracy deltas versus the retained-Fit majority
  baseline.

Both classes must be present in each retained Fit, Temporal OOS, and Joint OOS
subset. Otherwise EXP_009 stops without a scientific PASS/FAIL verdict.

```text
delta_recency_auc_f = AUC(D_009,f) - AUC(C_009,f)
```

Report fold values plus means and sample standard deviations (`ddof=1`) across
four folds as descriptive stability summaries, not IID inference.

## Pre-specified decisions

### Primary - incremental recency-weighted OOS discrimination

PRIMARY PASS requires all Joint OOS conditions:

1. `AUC(D_009) > 0.50` in all four folds.
2. Mean Joint `AUC(D_009) > 0.50`.
3. `delta_recency_auc_f > 0` in all four folds.
4. Mean Joint `delta_recency_auc > 0`.

No fold-level failure may be rescued by an average. No arbitrary minimum
effect-size threshold is imposed.

### Secondary - fixed-threshold classification for D_009

At threshold `0.5`, secondary evidence requires, in every Joint OOS fold:

1. Balanced Accuracy > `0.50`;
2. neutral recall > `0`; and
3. directional recall > `0`;

plus mean Joint delta Balanced Accuracy versus the retained-Fit majority
baseline > `0`. Accuracy is descriptive. Secondary cannot override primary.

### Temporal OOS diagnostic

Report Temporal OOS AUC for C_009 and D_009, plus `AUC(D_009)-AUC(C_009)`.
Temporal results are diagnostic only and cannot alter either verdict.

## Integrity checks

Implementation must verify:

- exact EXP_000 partition and folds, with normal splitting before filtering;
- every retained row has exactly 12 observed, finite W values;
- no target controls retention or feature construction;
- both classes occur in retained Fit, Temporal OOS, and Joint OOS;
- C_009 and D_009 have identical rows, targets, and ordering;
- exact weights `[1,2,...,12]` with sum `78`;
- finite raw intensity values for every retained row;
- each scaler is fit only on retained Fit raw intensity;
- both OOS subsets are transform-only;
- exact unscaled EXP_002 `missing_ratio`;
- IDs, day, equity, q, masks, and prohibited features are absent;
- identical model specification except feature dimension;
- no tuning or alternate feature/weight evaluation; and
- no competition-test access.

An integrity/evaluability failure stops EXP_009 without silent fallback or
scientific PASS/FAIL conclusion.

## Required result record

Persist `experiments/results/EXP_009_*` artifacts sufficient to audit:

- manifests before/after complete-W filtering;
- both Fit-only scaler parameters;
- C_009/D_009 metrics and confusion matrices by fold/subset;
- per-fold incremental AUC;
- primary P1-P4 and secondary decision records;
- Temporal OOS diagnostics;
- exact weight vector and sum; and
- runtime/environment metadata.

## Interpretation boundaries

If primary PASSes, the supported conclusion is only:

> Among observations with complete availability of `r41,...,r52`, the
> pre-specified monotonic recency-weighted movement-intensity statistic adds
> consistent Joint OOS Neutral-vs-Directional discrimination beyond aggregate
> unweighted recent movement intensity under the fixed linear model.

A PASS does not establish a true linear decay mechanism, optimal weights,
exponential-decay validity, a volatility regime, momentum, sign prediction,
order-flow or liquidity mechanisms, causality, improvement on partial-W rows,
full-population improvement, or ternary-task improvement.

If primary FAILs, the supported conclusion is only:

> Under the complete-W isolation population, frozen linear recency weighting,
> and fixed Logistic Regression specification, EXP_009 did not provide
> consistent Joint OOS evidence that recency-sensitive movement intensity adds
> discrimination beyond aggregate recent movement intensity.

A FAIL must not trigger different weights, exponential decay, half-lives,
windows, shock/trend features, or nonlinear models within EXP_009. Those
questions require separately motivated future contracts.
