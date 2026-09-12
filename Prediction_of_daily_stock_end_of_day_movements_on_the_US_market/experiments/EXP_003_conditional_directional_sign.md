# EXP_003 — Conditional Directional Sign from Cumulative Observed Return

## Status

Completed.

The protocol and decision rules were specified before execution. EXP_003 was
executed on training data only; the competition test was not accessed.

## Research question

Among observations known ex post to be directional, does cumulative observed
intraday return available by 14:00 contain out-of-sample predictive information
about the sign of the subsequent 14:00–16:00 directional move?

This is deliberately an oracle/conditional research question:

```text
P(sign | directional, observed path)
```

It is not a deployable classifier: at 14:00, the eventual directional status
is not known. EXP_003 must not combine EXP_002 predictions with this
conditional analysis.

## Conditional target and effective research population

For each frozen subset, define directional rows:

```text
D = {reod ∈ {-1, +1}}
S = 0  if reod = -1
S = 1  if reod = +1
```

Define, using only `r0...r52`:

```text
n_obs = Σ 1[r_j is observed]
R_obs = Σ r_j over observed return cells, only if n_obs >= 1
```

The effective research population is:

```text
D ∩ {n_obs >= 1}
```

Rows with `n_obs=0` are non-evaluable for EXP_003 and must be excluded only
after the normal frozen EXP_000 subset is constructed and after conditioning
on `D`. This is a fixed evaluability rule, not a learned or target-optimized
filter. `n_obs` and missingness are not predictive features.

`R_obs` must be described as **cumulative observed return**, not as the true
09:30–14:00 cumulative return. NaN intervals are skipped, never imputed. No
observed return, missingness feature, mask, return window, volatility,
technical indicator, identifier, `day`, `equity`, or other feature may enter
the model.

## Frozen validation protocol

- Reuse exactly EXP_000's frozen equity partition: `E_dev=1463`,
  `E_holdout=366`, seed `20260908`.
- Reuse exactly its four expanding temporal folds:

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

For each fold, construct the normal frozen subsets first, then apply `D`, then
apply the fixed `n_obs >= 1` evaluability rule. Do not redefine folds or the
equity partition after conditioning. Joint OOS is primary and Temporal OOS is
diagnostic. No purging or embargo is added. The competition test is prohibited
from all stages of this experiment.

## Fit-only scaling and fixed model

Fit `StandardScaler` only on evaluable directional rows in `fit_f`, then apply
it unchanged to Temporal OOS and Joint OOS. This fixed fit-only scaling is
required because `R_obs` is expressed in basis points and logistic regression
with L2 regularization is scale-sensitive. It adds no predictive feature.

Fit binary logistic regression on the standardized sole feature using only
evaluable directional `fit_f` rows:

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

No hyperparameter search, tuning, model substitution, class weighting, or
additional preprocessing is allowed. Class predictions use the fixed
probability threshold `0.5` for `S=1` / positive sign.

Report the coefficient `beta` for standardized `R_obs` in each fold and
whether its sign is consistent across all four folds. This diagnostic is not
part of either PASS/FAIL decision.

```text
beta > 0: relationship consistent with continuation / momentum
beta < 0: relationship consistent with reversal
```

OOS performance, not coefficient sign, determines evidence.

## Baseline

For every fold, determine the majority sign class exclusively from evaluable
directional `fit_f` rows. Class order is `[0, 1]`; a tie selects `S=0`
(negative). Predict that fixed class on the corresponding evaluable OOS subset.

## Evaluation

For the model and baseline, report every fold separately for Temporal OOS and
Joint OOS:

- ROC-AUC from predicted `P(S=1 | R_obs, D)`;
- Balanced Accuracy;
- Accuracy;
- recall negative (`S=0`);
- recall positive (`S=1`);
- confusion matrix with labels ordered `[0, 1]`;
- model minus majority-baseline Accuracy;
- model minus majority-baseline Balanced Accuracy.

Also report `beta` by fold and sign consistency across the four folds. Report
means and sample standard deviations (`ddof=1`) across folds as descriptive
stability summaries only, not IID inference.

For transparency, report for every fit, Temporal OOS, and Joint OOS subset:

- directional rows before applying `n_obs` evaluability;
- all-NaN directional rows excluded (`n_obs=0`);
- evaluable directional rows retained;
- negative and positive sign counts before and after evaluability.

## Pre-specified decisions

### Primary — OOS sign discrimination

EXP_003 has promising conditional OOS sign-discrimination evidence only if:

1. Joint OOS ROC-AUC is strictly greater than `0.50` in every one of the four
   folds; and
2. mean Joint OOS ROC-AUC is strictly greater than `0.50`.

### Secondary — fixed-threshold sign classification

EXP_003 has fixed-threshold conditional sign-classification evidence only if,
in every Joint OOS fold:

1. Balanced Accuracy is strictly greater than `0.50`;
2. recall negative is strictly positive; and
3. recall positive is strictly positive;

and mean Joint OOS delta Balanced Accuracy versus the fit-derived majority
baseline is strictly positive.

Accuracy and delta Accuracy are diagnostics, not pass/fail conditions. The
primary and secondary decisions remain separate: primary discrimination may
pass while fixed-threshold classification fails.

## Integrity checks

Before reporting results, the implementation must verify:

- exact reuse of EXP_000 folds and frozen equity partition;
- normal frozen subsets are constructed before `D` and evaluability filtering;
- all retained rows satisfy `reod ∈ {-1, +1}` and `n_obs >= 1`;
- no neutral row is fitted or evaluated;
- no holdout equity enters retained `fit_f`;
- `R_obs` uses exactly the observed cells of `r0...r52`, with no imputation;
- no missingness information, `n_obs`, or non-declared feature enters model
  fitting;
- scaler fitting uses only retained `fit_f` rows;
- both sign classes occur in retained fit and every evaluated OOS subset;
- predicted probability is the `S=1` column and the threshold is exactly
  `0.5`;
- the fit scaler has finite positive scale; and
- the competition test is not accessed.

## Interpretation boundary and limitation

A positive result supports conditional/oracle OOS sign information in
cumulative observed return. It does not establish deployable sign prediction,
causality, trading utility, missingness mechanisms, improvement on the original
ternary challenge, or validity of a hierarchical ternary architecture. The
503 labelled days remain an imperfect internal proxy for deployment in a
future period and a new equity universe.

## Results

All real-data integrity checks passed. Both sign classes were present in every
retained fit and evaluated OOS subset. All-NaN exclusions were negligible
relative to subset sizes: between zero and five directional rows per subset.
The fit-derived majority sign was negative (`S=0`) in all four folds.

### Joint OOS

| Fold | ROC-AUC | Standardized `R_obs` beta |
|---|---:|---:|
| 1 | 0.493825 | +0.003504 |
| 2 | 0.491287 | +0.003135 |
| 3 | 0.494237 | +0.002918 |
| 4 | 0.499332 | +0.002774 |

Joint OOS mean results:

- ROC-AUC: `0.494671` (sample standard deviation `0.003371`)
- Balanced Accuracy: `0.499977`
- Accuracy: `0.504064`
- Recall negative: `0.999953`
- Recall positive: `0.000000`
- Delta Balanced Accuracy versus majority baseline: `-0.000023`
- Delta Accuracy versus majority baseline: `-0.000025`

**PRIMARY OOS sign-discrimination verdict: FAIL.**

**SECONDARY fixed-threshold sign-classification verdict: FAIL.**

All standardized `R_obs` coefficients were positive, and the pre-specified
beta-sign consistency diagnostic was `True`. Those positive coefficient signs
must not be interpreted as evidence of predictive momentum because the
pre-specified OOS discrimination criterion failed.

### Temporal OOS diagnostic

- Mean ROC-AUC: `0.499747`
- Mean Balanced Accuracy: `0.500015`
- Mean Accuracy: `0.506131`

## Interpretation boundary after execution

EXP_003 found no OOS evidence that cumulative observed return alone
discriminates the subsequent sign among oracle-known directional outcomes under
the frozen validation protocol. This does not establish that the full intraday
path `r0...r52` contains no sign information, nor that momentum or reversal is
impossible in other path representations, horizons, or conditional structures.
