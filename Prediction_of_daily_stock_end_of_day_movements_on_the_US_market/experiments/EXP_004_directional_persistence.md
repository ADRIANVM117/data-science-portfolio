# EXP_004 — Conditional Directional Sign from Directional Persistence

## Status

Completed.

## Research question

Among observations known ex post to be directional, does directional
persistence in the observed 09:30–14:00 return path contain out-of-sample
predictive information about the sign of the subsequent 14:00–16:00
directional move?

This is an oracle/conditional research question:

```text
P(S = 1 | P, D)
```

where:

```text
D = {reod ∈ {-1, +1}}
S = 0  if reod = -1
S = 1  if reod = +1
```

It is not deployable by itself because `D` is known only ex post. EXP_004 is a
distinct representation test from EXP_003: it tests observed sign dominance,
not cumulative observed return.

## Conditional population and representation

For each row, using only `r0...r52`, define:

```text
N_plus  = number of observed returns with r_j > 0
N_minus = number of observed returns with r_j < 0
N_zero  = number of observed returns with r_j = 0
N_obs   = N_plus + N_minus + N_zero
```

NaNs are not observed and do not contribute to any count. Observed zeros count
in `N_obs` but contribute neither to `N_plus` nor `N_minus`.

For `N_obs >= 1`, define the sole predictive feature:

```text
P = (N_plus - N_minus) / N_obs
```

Equivalently, `P` is the mean of `sign(r_j)` over observed intervals, with
`sign(0)=0`. It lies in `[-1, 1]`.

The effective research population is:

```text
D ∩ {N_obs >= 1}
```

Rows with `N_obs=0` are non-evaluable and must be excluded only after the
normal frozen EXP_000 subset is constructed and after conditioning on `D`.
This is a fixed evaluability rule, not a learned or target-optimized filter.

`P` is the only predictor. Do not add cumulative observed return, its count
components, missingness features or masks, temporal ordering, run structure,
sign-transition counts, volatility, windows, technical indicators, IDs, `day`,
`equity`, or any other feature. EXP_004 does not test sign-sequence continuity.

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

Construct normal frozen subsets first; then apply `D`; then apply the fixed
`N_obs >= 1` evaluability rule. Do not redefine folds or the equity partition
after conditioning. Joint OOS is primary; Temporal OOS is diagnostic. No
purging or embargo is added. The competition test is prohibited from all
stages.

## Fixed model

No scaling or learned preprocessing is used: `P` is naturally bounded in
`[-1,1]` and has a stable semantic scale across folds.

Fit binary logistic regression using only evaluable directional `fit_f` rows:

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
threshold adjustment is allowed. Class predictions use the fixed probability
threshold `0.5` for `S=1` / positive sign.

Report the fitted coefficient `beta` for `P` in each fold and whether its sign
is consistent across all four folds. This is diagnostic only:

```text
beta > 0: positive / continuation-consistent
beta < 0: negative / reversal-consistent
beta = 0: zero
```

Coefficient-sign consistency is `True` only if all four fold coefficients are
strictly positive or all four are strictly negative. Any zero coefficient
makes it `False`. Coefficient diagnostics must not affect either PASS/FAIL
decision.

## Baseline

For each fold, determine the majority sign exclusively from evaluable
directional `fit_f` rows. Class order is `[0, 1]`; a tie selects `S=0`
(negative). Predict that fixed class on the corresponding evaluable OOS subset.

## Evaluation

For the model and baseline, report every fold separately for Temporal OOS and
Joint OOS:

- ROC-AUC from predicted `P(S=1 | P, D)`;
- Balanced Accuracy;
- Accuracy;
- recall negative (`S=0`);
- recall positive (`S=1`);
- confusion matrix with labels ordered `[0, 1]`;
- model minus majority-baseline Accuracy;
- model minus majority-baseline Balanced Accuracy.

Report beta by fold and the sign-consistency diagnostic. Report means and
sample standard deviations (`ddof=1`) across folds as descriptive stability
summaries only, not IID inference.

For every fit, Temporal OOS, and Joint OOS subset, report:

- directional rows before evaluability filtering;
- all-NaN directional rows excluded (`N_obs=0`);
- evaluable directional rows retained;
- negative/positive sign counts before and after evaluability.

## Pre-specified decisions

### Primary — OOS sign discrimination

EXP_004 has promising conditional OOS sign-discrimination evidence only if:

1. Joint OOS ROC-AUC is strictly greater than `0.50` in every one of the four
   folds; and
2. mean Joint OOS ROC-AUC is strictly greater than `0.50`.

### Secondary — fixed-threshold sign classification

EXP_004 has fixed-threshold conditional sign-classification evidence only if,
in every Joint OOS fold:

1. Balanced Accuracy is strictly greater than `0.50`;
2. recall negative is strictly positive; and
3. recall positive is strictly positive;

and mean Joint OOS delta Balanced Accuracy versus the fit-derived majority
baseline is strictly positive.

Accuracy and delta Accuracy are diagnostics, not pass/fail conditions. The
primary and secondary decisions remain separate: primary discrimination may
pass while fixed-threshold classification fails.

## Executed results

All real-data integrity assertions passed. The competition test was not
accessed. The fit-derived majority sign was negative in all four folds, and
all-NaN exclusions were negligible relative to subset sizes.

### Joint OOS

- ROC-AUC by fold: `0.499966`, `0.509310`, `0.495915`, `0.492626`.
- Mean ROC-AUC: `0.499454`; sample standard deviation: `0.007224`.
- Mean Balanced Accuracy: `0.498369`.
- Mean Accuracy: `0.501954`.
- Mean recall negative: `0.973249`; mean recall positive: `0.023488`.
- Mean delta Balanced Accuracy versus the majority baseline: `-0.001631`.
- Mean delta Accuracy versus the majority baseline: `-0.002135`.

Pre-specified verdicts:

- PRIMARY OOS sign-discrimination: **FAIL**.
- SECONDARY fixed-threshold sign-classification: **FAIL**.

### Coefficient diagnostic

`beta(P)` was negative in all four folds and coefficient-sign consistency was
`True`. These negative coefficient signs are not evidence of predictive
reversal because the frozen OOS discrimination criterion failed.

### Temporal OOS diagnostic

Mean ROC-AUC was `0.496793`, mean Balanced Accuracy was `0.498580`, and mean
Accuracy was `0.504156`.

## Integrity checks

Before reporting results, the implementation must verify:

- exact reuse of EXP_000 folds and frozen equity partition;
- normal frozen subsets are constructed before `D` and evaluability filtering;
- all retained rows satisfy `reod ∈ {-1, +1}` and `N_obs >= 1`;
- no neutral row is fitted or evaluated;
- no holdout equity enters retained `fit_f`;
- exact `N_plus`, `N_minus`, `N_zero`, and `N_obs` definitions;
- observed zero returns enter `N_obs`; NaNs do not;
- all-NaN rows are excluded and never assigned `P=0`;
- `P` is finite and bounded in `[-1,1]`;
- the predictive matrix contains exactly `P`, with no component count or
  missingness feature;
- both sign classes occur in retained fit and every evaluated OOS subset;
- predicted probability is the `S=1` column and the threshold is exactly
  `0.5`; and
- the competition test is not accessed.

## Interpretation boundary and limitation

A positive result supports conditional/oracle OOS sign information in
directional persistence only. It does not establish deployable sign prediction,
causality, trading utility, full-path sign information, or validity of a
hierarchical ternary architecture. The 503 labelled days remain an imperfect
internal proxy for deployment in a future period and a new equity universe.

EXP_004 found no OOS evidence that the frozen one-dimensional
directional-persistence representation
`P = (N_plus - N_minus) / N_obs` discriminates subsequent positive versus
negative outcomes among oracle-known directional observations. This does not
establish that intraday path ordering, timing, run structure, recent-state
information, or the complete `r0...r52` path contains no sign information.
