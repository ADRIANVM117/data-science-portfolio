# EXP_005 — Conditional Positional Path Signal

## Status

Completed.

## Research question

Among oracle-known directional and evaluable observations, do the values of
the 53 temporally positioned 09:30–14:00 returns provide OOS sign
discrimination beyond the information contained in positional missingness
alone, under a fixed linear-additive model?

## Population and target

For each normal frozen EXP_000 subset, define:

```text
D = {reod ∈ {-1, +1}}
S = 0  if reod = -1
S = 1  if reod = +1
```

Using only `r0...r52`, define:

```text
N_obs = Σ 1[r_j is observed]
```

The effective research population is:

```text
D ∩ {N_obs >= 1}
```

Construct the normal frozen subsets first, then condition on `D`, then apply
the fixed `N_obs >= 1` evaluability rule. This matches EXP_003 and EXP_004.
All-NaN directional rows are non-evaluable and are excluded only at this
stage.

## Frozen validation protocol

Reuse EXP_000 exactly:

- frozen `E_dev=1463` / `E_holdout=366` equity partition, seed `20260908`;
- four expanding temporal folds;
- normal splitting before directional conditioning and evaluability;
- Joint OOS as primary evaluation;
- Temporal OOS as diagnostic evaluation;
- no purging or embargo; and
- no competition-test access.

| Fold | Train days | OOS days |
|---|---|---|
| 1 | 0–302 | 303–352 |
| 2 | 0–352 | 353–402 |
| 3 | 0–402 | 403–452 |
| 4 | 0–452 | 453–502 |

```text
fit_f         = train_days_f × E_dev
Temporal OOS  = oos_days_f × E_dev
Joint OOS     = oos_days_f × E_holdout
```

## Representations

`A_mask` and `B_path_mask` must use exactly the same retained rows in every
corresponding fit, Temporal OOS, and Joint OOS subset.

### A_mask — availability control

```text
A_mask = [m0, ..., m52]
m_j = 1 if r_j is NaN, else 0
```

`A_mask` uses no learned preprocessing.

### B_path_mask — primary representation

```text
B_path_mask = [
  53 positional returns after fit-only median imputation and fit-only
  StandardScaler transformation,
  53 original positional binary masks
]
```

For each fold:

1. Construct the 53 masks from original data before imputation.
2. Compute each positional median using evaluable directional `fit_f` only.
3. Assert that all 53 fit medians are finite.
4. If any fit median is non-finite, stop EXP_005 as an
   integrity/evaluability failure. Do not substitute zero, drop the position,
   derive a replacement from OOS, or produce a scientific PASS/FAIL verdict.
5. Impute `fit_f`, Temporal OOS, and Joint OOS using the fit medians only.
6. Fit `StandardScaler` on the 53 imputed return columns of `fit_f` only.
7. Transform `fit_f` and both OOS subsets without refitting.
8. Append the original 53 masks unchanged and unscaled.

## Forbidden predictors

Neither representation may include:

- `R_obs`;
- EXP_004 persistence `P`;
- `N_plus`, `N_minus`, `N_zero`, or `N_obs` as predictors;
- missingness ratios or summaries;
- rolling or window features;
- runs or transitions;
- technical indicators;
- `ID`, `day`, or `equity`; or
- any additional engineered predictor.

## Fixed model

Fit the same binary Logistic Regression separately for `A_mask` and
`B_path_mask`, using evaluable directional `fit_f` only:

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

No hyperparameter tuning, alternative classifier, nonlinear model, RNN, or
ensemble is permitted. Class predictions use the fixed probability threshold
`0.5` for `S=1` / positive sign.

## Baseline

For each fold, determine the majority sign only from evaluable directional
`fit_f`. The class order is `[0, 1]`; a tie selects `S=0` (negative). Predict
that fixed class on the corresponding evaluable OOS subset.

## Evaluation

For `A_mask`, `B_path_mask`, and the majority baseline, report every fold for
Temporal OOS and Joint OOS:

- ROC-AUC from predicted `P(S=1 | X, D)` when probabilities are available;
- Balanced Accuracy;
- Accuracy;
- recall negative (`S=0`);
- recall positive (`S=1`);
- confusion matrix with labels ordered `[0, 1]`;
- model minus majority-baseline Accuracy; and
- model minus majority-baseline Balanced Accuracy.

Also report, fold by fold and in mean, the Joint OOS ROC-AUC delta:

```text
AUC(B_path_mask) - AUC(A_mask)
```

EXP_003 and EXP_004 are frozen historical/descriptive references only. Their
fold-level and mean ROC-AUC differences may be reported where useful, but no
PASS/FAIL condition is defined relative to them and no retrospective material
improvement threshold is introduced.

Report means and sample standard deviations (`ddof=1`) across folds as
descriptive stability summaries only, not IID inference.

## Pre-specified decisions

### Primary — incremental positional-return OOS discrimination

EXP_005 supports evidence that positional return values add OOS sign
information beyond availability only if all conditions hold:

1. `B_path_mask` Joint ROC-AUC is strictly greater than `0.50` in all four
   folds.
2. Mean `B_path_mask` Joint ROC-AUC is strictly greater than `0.50`.
3. In every Joint OOS fold,
   `AUC(B_path_mask) - AUC(A_mask)` is strictly greater than `0`.
4. Mean Joint OOS `AUC(B_path_mask) - AUC(A_mask)` is strictly greater than
   `0`.

### Secondary — fixed-threshold classification for B_path_mask

At the fixed threshold `0.5`, `B_path_mask` has fixed-threshold
sign-classification evidence only if, in every Joint OOS fold:

1. Balanced Accuracy is strictly greater than `0.50`.
2. Recall negative is strictly positive.
3. Recall positive is strictly positive.

Mean Joint OOS delta Balanced Accuracy versus the fit-derived majority
baseline must also be strictly positive.

Primary and secondary criteria are independent. Accuracy and delta Accuracy
are diagnostics, not pass/fail conditions. `A_mask` performance is reported
separately as a scientific control.

## Executed results

All frozen integrity assertions passed. The competition test was not accessed.
The fit-derived majority sign was negative in all four folds. All 53 fit
medians were finite in every fold; median imputation and `StandardScaler` were
fit-only, OOS subsets were transform-only, masks remained binary and unscaled,
and `A_mask` and `B_path_mask` used identical retained rows.

### Joint OOS

| Fold | A_mask ROC-AUC | B_path_mask ROC-AUC | B_path_mask - A_mask |
|---|---:|---:|---:|
| 1 | 0.497363 | 0.518273 | +0.020910 |
| 2 | 0.506395 | 0.471215 | -0.035180 |
| 3 | 0.505435 | 0.507501 | +0.002066 |
| 4 | 0.498217 | 0.488085 | -0.010132 |

- A_mask mean ROC-AUC: `0.501852`; sample standard deviation: `0.004720`.
- B_path_mask mean ROC-AUC: `0.496269`; sample standard deviation: `0.020857`.
- Mean B_path_mask minus A_mask ROC-AUC: `-0.005584`; sample standard
  deviation: `0.023502`.
- B_path_mask mean Balanced Accuracy: `0.496191`.
- B_path_mask mean Accuracy: `0.495880`.
- B_path_mask mean recall negative: `0.643586`; mean recall positive:
  `0.348797`.
- B_path_mask mean delta Balanced Accuracy versus majority: `-0.003809`.
- B_path_mask mean delta Accuracy versus majority: `-0.008209`.

Pre-specified verdicts:

- PRIMARY incremental positional-return OOS discrimination: **FAIL**.
- SECONDARY fixed-threshold classification: **FAIL**.

### Temporal OOS diagnostic

- A_mask mean ROC-AUC: `0.501863`.
- B_path_mask mean ROC-AUC: `0.491700`.

## Integrity checks

Before reporting results, the implementation must verify:

- exact reuse of EXP_000 folds and frozen equity partition;
- normal frozen splitting before directional conditioning and evaluability;
- all retained rows satisfy `reod ∈ {-1, +1}` and `N_obs >= 1`;
- `A_mask` and `B_path_mask` have exactly the same retained row identities and
  order within every subset;
- masks are constructed from original returns before imputation, are binary,
  and remain unscaled;
- all 53 medians are finite and are fitted using only evaluable directional
  `fit_f` rows;
- any non-finite fit median stops the experiment as an
  integrity/evaluability failure without a scientific verdict;
- imputation and StandardScaler fitting use only evaluable directional
  `fit_f`, with unchanged transforms applied to both OOS subsets;
- `A_mask` has exactly 53 mask features and `B_path_mask` exactly 106 declared
  features, with no forbidden predictors;
- no holdout equity enters retained `fit_f`;
- both sign classes occur in retained fit and every evaluated OOS subset;
- the predicted probability is the `S=1` column and the threshold is exactly
  `0.5`; and
- the competition test is not accessed.

## Interpretation boundaries and complexity

A positive primary result supports OOS evidence that positional return values
add linear-additive sign information beyond positional missingness within the
oracle-directional population. It does not establish sequential dynamics,
causality, deployable sign prediction, ternary challenge improvement, trading
utility, superiority of RNN/CatBoost/LightGBM, or competition-test
generalization.

A failure supports only:

> No OOS evidence under the frozen criterion that the 53 positional return
> values add sign discrimination beyond positional missingness through this
> fixed linear-additive Logistic Regression representation.

A failure must not be interpreted as evidence that the full path, nonlinear
interactions, temporal ordering, or sequential models contain no sign
information.

- If `B_path_mask` passes and consistently beats `A_mask`, richer interaction
  models become scientifically motivated.
- If `B_path_mask` discriminates but does not consistently beat `A_mask`, the
  joint representation has signal but incremental return-value information is
  not established.
- If `A_mask` discriminates while `B_path_mask` adds nothing, availability
  deserves further investigation before attributing sign information to
  returns.
- If `A_mask` and `B_path_mask` both fail, the linear-positional hypothesis is
  closed. Any next experiment must pose a separately frozen nonlinear or
  sequence-specific hypothesis rather than modifying EXP_005 post hoc.

## Interpretation boundary after execution

EXP_005 found no OOS evidence under the frozen criterion that the 53
positional return values add sign discrimination beyond positional missingness
through the fixed linear-additive Logistic Regression representation. A_mask
itself also remained approximately non-discriminative for conditional sign.

Together, EXP_003 failed with cumulative observed return, EXP_004 failed with
directional sign dominance, and EXP_005 failed to establish incremental sign
information when all 53 return positions were preserved under a linear-
additive model. The current evidence rejects these tested simple/global and
linear-positional representations as useful conditional sign discriminators
under the frozen protocol.

This does not establish that the full path, nonlinear interactions, temporal
ordering, or sequential dynamics contain no sign information, nor that RNNs
or boosting cannot work.
