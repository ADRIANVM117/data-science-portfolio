# EXP_008 - Recent Movement Intensity Incremental Signal

## Status

Completed (execution-status reconciliation recorded after execution).

## Execution-status reconciliation

The scientific specification below was frozen before execution and is retained
unchanged. A subsequent documentation review verified the authoritative
`experiments/results/EXP_008_*` artifacts: the fold manifest, fit-only
preprocessing parameters, metrics, incremental-AUC record, Joint decision,
summary, confusion matrices, and metadata. Those artifacts show that EXP_008
was implemented and executed once using training data only; the competition
test was not accessed.

The historical Joint OOS primary result was a PASS. Condition B ROC-AUC was
`0.680477`, `0.686750`, `0.690781`, and `0.698219` in folds 1--4, while the
corresponding B-minus-C intensity AUC deltas were `+0.076415`, `+0.102034`,
`+0.094798`, and `+0.099619`. The mean B ROC-AUC was `0.689057` and the mean
incremental intensity AUC was `+0.093217`. This status addendum reconciles
documentation only; it does not retrospectively alter the frozen scientific
contract or historical result artifacts.

## Scientific question

Among the full Neutral-vs-Directional population, does movement intensity in
the final 12 chronologically ordered return observations add Joint OOS
discrimination beyond both:

1. the global missingness signal established in EXP_002; and
2. positional information that the entire recent 12-observation window is
   unavailable?

This experiment tests incremental Neutral-vs-Directional predictive
information only. It does not test a volatility regime, causality, directional
sign prediction, a hierarchical ternary classifier, or final challenge utility.

## Target, population, and frozen validation

Reuse EXP_002 exactly:

```text
Z = 0  if reod = 0          # neutral
Z = 1  if reod in {-1, +1}  # directional
```

Use the full EXP_002 population. Do not condition on directional outcomes and
do not apply the EXP_003-EXP_007 evaluability rule.

Reuse the frozen EXP_000 protocol exactly:

- fixed equity partition: `E_dev=1463`, `E_holdout=366`, seed `20260908`;
- no holdout equity in `fit_f`;
- Joint OOS as the primary deployment proxy;
- Temporal OOS as diagnostic only;
- no purging or embargo; and
- competition-test data excluded completely.

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

The six structurally unusual sessions `{112, 134, 229, 314, 438, 469}` remain
naturally in whichever frozen subsets contain them. No session filtering is
permitted.

## Positional recent window and data-semantics caveat

Define the window positionally:

```text
W = {r41, r42, ..., r52}
```

`W` contains exactly the final 12 chronologically ordered return observations.
At the dataset's stated five-minute granularity, it is a 60-minute
recent-history representation by observation count.

The official challenge documentation in this repository is internally
inconsistent about the absolute endpoints of the 53 five-minute returns.
Therefore EXP_008 defines recency positionally and does **not** claim that
`W` is exactly any particular clock-time interval. No exact clock-time
endpoint is assigned to `r41` or `r52`.

## Raw intensity and complete-window unavailability

For every row, define:

```text
N_obs,W = sum_{t in W} 1[r_t is observed]
q       = 1[N_obs,W = 0]
```

For rows with `N_obs,W > 0`, define raw recent movement intensity as:

```text
I_recent = sqrt(sum_{t in W, r_t observed}(r_t^2) / N_obs,W)
```

Observed zero returns are observed values: they count in `N_obs,W` and
contribute `0^2` to the numerator. NaNs contribute to neither the numerator
nor denominator. For `N_obs,W = 0`, raw `I_recent` is undefined and must be
represented as missing before the fold-specific imputation described below.

`q` is explicit positional missingness information. It is not volatility or
intensity. The preceding target-free structural audit established that `q` is
not generally determined by scalar global `missing_ratio`.

## Frozen representations

All three conditions must use identical rows, targets, subset membership, and
row order within every fold. The only differences are the frozen predictors.

```text
missing_ratio = (number of NaNs among r0,...,r52) / 53
```

This is exactly the EXP_002 feature.

| Condition | Predictors |
|---|---|
| A - global missingness control | `[missing_ratio]` |
| C - positional availability control | `[missing_ratio, q]` |
| B - intensity condition | `[missing_ratio, q, I_recent_z]` |

The key scientific comparison is **B versus C**. `C versus A` is a diagnostic
for incremental positional complete-window unavailability; it is not evidence
for recent movement intensity.

`ID`, `day`, `equity`, raw `I_recent`, return values, masks other than `q`,
and all other features are prohibited. `missing_ratio` and `q` remain in their
original scales and are not standardized.

## Fit-only intensity preprocessing

Within each fold, after constructing the normal frozen subsets:

1. Compute raw `I_recent` from `W` without imputation wherever `N_obs,W > 0`.
2. From `fit_f` rows with `N_obs,W > 0` only, calculate the ordinary sample
   median `fit_intensity_median` of raw `I_recent`.
3. Impute raw undefined intensity in `fit_f`, Temporal OOS, and Joint OOS with
   that same fit-only median.
4. Fit `StandardScaler` on the resulting imputed intensity column of `fit_f`
   only.
5. Transform fit and both OOS intensity columns with that unchanged scaler.

Call the resulting predictor `I_recent_z`. No clipping, winsorization, log
transformation, RobustScaler, alternative scaling, or OOS-derived
preprocessing is permitted.

## Fixed model and baseline

Fit separate independent models for A, C, and B in every fold, each using the
exact EXP_002 Logistic Regression configuration:

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

There is no tuning, alternative model, or changed threshold. Probabilities
refer to `P(Z=1)` and class predictions use exactly threshold `0.5`; equality
at `0.5` predicts `Z=1`.

For each fold, derive the binary majority baseline only from `Z` in `fit_f`.
The class order is `[0, 1]`, so a tie selects `Z=0`. Apply that fixed
fit-derived class to both OOS subsets.

## Evaluation

For each condition, fold, and Temporal/Joint OOS subset, report:

- ROC-AUC from probabilities for `Z=1`;
- Balanced Accuracy;
- Accuracy;
- recall neutral (`Z=0`);
- recall directional (`Z=1`);
- confusion matrix with labels `[0, 1]`;
- delta Accuracy versus the fit-derived majority baseline; and
- delta Balanced Accuracy versus the fit-derived majority baseline.

Both classes must be present in every evaluated OOS subset. Otherwise ROC-AUC,
recalls, and Balanced Accuracy are not evaluable and the experiment stops
without a scientific PASS/FAIL verdict.

For each Joint OOS fold, calculate:

```text
delta_intensity_auc_f = AUC(B_f) - AUC(C_f)
delta_q_auc_f         = AUC(C_f) - AUC(A_f)
```

Report fold-level values and means with sample standard deviations (`ddof=1`)
over four folds. These are descriptive stability summaries, not IID
inference.

Temporal OOS reports `AUC(A)`, `AUC(C)`, `AUC(B)`, `C-A`, and `B-C`; it is
diagnostic only and cannot affect a verdict.

## Pre-specified decisions

### Primary - incremental recent-intensity OOS discrimination

EXP_008 has primary evidence only if all of the following hold on Joint OOS:

1. `AUC(B) > 0.50` in all four folds.
2. Mean Joint `AUC(B) > 0.50`.
3. `delta_intensity_auc_f > 0` in all four folds.
4. Mean Joint `delta_intensity_auc > 0`.

No minimum effect-size threshold is added. The overall primary verdict is
FAIL if any condition fails.

### Positional-missingness diagnostic

`delta_q_auc = AUC(C) - AUC(A)` is reported for both Joint and Temporal OOS.
It does not enter the primary verdict and must not be interpreted as evidence
for movement intensity.

### Secondary - fixed-threshold classification for B

At the fixed threshold `0.5`, condition B has secondary evidence only if, in
every Joint OOS fold:

1. `Balanced Accuracy(B) > 0.50`;
2. neutral recall is strictly positive; and
3. directional recall is strictly positive;

and mean Joint delta Balanced Accuracy of B versus the fit-derived majority
baseline is strictly positive. Accuracy is descriptive. A secondary result
does not override a primary FAIL.

## Structural and information-boundary integrity checks

Implementation must assert that:

- `W` is exactly ordered `r41,...,r52` and has 12 positions;
- `q=1` if and only if all 12 original W returns are NaN;
- observed zeros are not missing and are included in `N_obs,W`;
- raw `I_recent` uses observed W returns only;
- raw `I_recent` is finite for every `N_obs,W > 0` row and undefined before
  imputation for every `N_obs,W = 0` row;
- `fit_intensity_median` is finite and uses only `fit_f` rows with
  `N_obs,W > 0`;
- neither OOS subset affects the median or scaler parameters;
- `StandardScaler` is fit only on imputed `fit_f` intensity, while OOS is
  transform-only;
- `missing_ratio` exactly matches the EXP_002 implementation and remains in
  `[0, 1]`;
- A, C, and B have identical IDs, targets, and row ordering for each subset;
- their predictor counts are exactly 1, 2, and 3, respectively;
- ID, day, and equity are not predictors;
- frozen folds and equity partition are reused exactly, with no holdout equity
  in `fit_f`; and
- no competition-test row is accessed.

Any integrity or evaluability failure stops the experiment without a silent
fallback, representation change, or PASS/FAIL conclusion.

## Required result record

Persist `experiments/results/EXP_008_*` artifacts sufficient to reconstruct:

- frozen metadata and reproducibility information;
- fold/subset sizes, binary class counts, and `q` counts;
- fit-only intensity medians and scaler parameters;
- all A/C/B fold metrics and confusion matrices;
- Joint and Temporal `B-C` and `C-A` AUC deltas;
- aggregate means and sample standard deviations;
- each primary and secondary condition and the resulting verdicts; and
- integrity-check outcomes.

## Interpretation boundaries

A primary PASS supports only this statement:

> Under the frozen linear model, positional recent window, RMS definition,
> missingness controls, and OOS validation protocol, recent observed movement
> intensity provides incremental Joint OOS Neutral-vs-Directional
> discrimination beyond global missingness and complete recent-window
> unavailability.

A PASS does not establish a volatility regime or persistence, causality,
optimality of RMS, the 12-observation window, five-minute frequency, or any
clock-time mapping. It also does not establish sign predictability, final
ternary-task improvement, nonlinear-model superiority, or that alternative
windows/statistics perform similarly.

A FAIL does not prove that movement intensity contains no information. It
must not trigger testing alternative windows, RMS variants, clipping, log
transforms, volatility statistics, or model families within EXP_008. Any such
question requires a separately motivated future experiment contract.
