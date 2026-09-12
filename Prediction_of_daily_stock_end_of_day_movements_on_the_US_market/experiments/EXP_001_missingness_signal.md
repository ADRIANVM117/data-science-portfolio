# EXP_001 — Missingness Signal

## Status

Completed.

The protocol and decision rule were specified before execution. EXP_001 was
run on training data only; the competition test was not accessed.

## Research question

Does missingness information available by 14:00 provide out-of-sample
predictive information about `reod`, particularly for identifying the neutral
regime, under the frozen EXP_000 validation protocol?

## Scope and fixed protocol

- Reuse exactly the frozen `E_dev` / `E_holdout` partition (seed `20260908`)
  and all four expanding walk-forward folds from `EXP_000`.
- For every fold, use the same `fit_f`, `Temporal OOS`, and `Joint OOS`
  definitions from `EXP_000`. Joint OOS is primary; Temporal OOS is a
  diagnostic.
- Use only the training dataset. The competition test is prohibited from
  feature construction, fitting, evaluation, model selection, and hypothesis
  evaluation.
- `ID`, `day`, and `equity` are keys / partition variables only. The only
  predictive information is the original missingness pattern of `r0` through
  `r52`, available by 14:00.

## Feature representations

Each representation is evaluated separately. A missing value is encoded as
`1`; an observed return, including an observed zero, is encoded as `0`.
No return value is used and no missing value is imputed.

| Variant | Features | Dimension |
|---|---|---:|
| M1 | `missing_ratio`: fraction of missing values across `r0...r52` | 1 |
| M2 | Mean missingness in mechanically fixed windows: early `r0...r16`, mid `r17...r34`, late `r35...r52` | 3 |
| M3 | Complete binary mask `m0...m52`, one indicator per return position | 53 |

These transformations are deterministic row-wise functions: they learn no
parameters and are constructed independently in each subset without using
`reod`.

## Fixed classifier

Fit the same multinomial logistic regression separately for every variant and
fold, using `fit_f` only:

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

`scikit-learn 1.6.1` deprecates the explicit `multi_class` argument. It is
therefore omitted: with `lbfgs` and three classes, the installed estimator
uses its multinomial behavior. `lbfgs` is appropriate for these small dense
feature matrices, and L2 regularization provides a fixed guard against
correlation among mask positions in M3. All inputs lie in `[0, 1]`, so this
experiment applies no learned scaling. No hyperparameter tuning is allowed.

## Evaluation

For each variant, fold, and OOS subset, report Accuracy, Macro-F1, Balanced
Accuracy, and the confusion matrix with labels ordered `[-1, 0, 1]`.
Compare each result fold-by-fold with the corresponding `EXP_000`
majority-class baseline. Also report the mean and sample standard deviation
(`ddof=1`) across folds as descriptive stability summaries only, not iid
inference.

## Pre-specified decision criterion

A variant is considered **promising** only if, in **each of the four Joint
OOS folds**, it simultaneously:

1. has Accuracy strictly greater than the corresponding EXP_000
   majority-baseline Accuracy;
2. has Balanced Accuracy strictly greater than `1/3`; and
3. has strictly positive recall for both directional classes, `-1` and `+1`.

The mean Joint OOS Accuracy improvement relative to the baseline must also be
positive. Macro-F1 and confusion matrices remain diagnostics for understanding
how performance is distributed between the three classes. This is a
pre-specified EXP_001 decision rule, not a statistical-significance test; no
iid inference is drawn from the four folds.

## Integrity checks required before reporting results

- Verify that the frozen equity partition and every fold boundary exactly
  match EXP_000.
- Verify temporal separation, `E_dev` / `E_holdout` disjointness, and absence
  of holdout equities from every `fit_f`.
- Verify that feature columns are exactly the declared M1, M2, or M3
  representation; values must be binary for M3 and in `[0, 1]` for M1/M2.
- Verify that no `reod`, `ID`, `day`, `equity`, return value, competition-test
  row, or OOS row enters model fitting.
- Verify class counts and evaluation-row counts for `fit_f`, Temporal OOS,
  and Joint OOS in every fold.
- Compute the majority comparison exclusively from the executed EXP_000
  fold-level baseline artifacts, preserving its fit-only definition.

## Limitation

This experiment tests predictive association under the internal Joint OOS
proxy; it does not establish whether missingness originates from liquidity,
technical data availability, or dataset construction. It also cannot exactly
reproduce the distinct future period and wholly new universe of the competition
test.

## Results

No variant qualified as promising under the pre-specified Joint OOS criterion.

- **M1:** reproduced the EXP_000 majority baseline in all Joint OOS folds
  (mean Accuracy `0.412362`, delta `0.000000`).
- **M2:** improved Joint OOS Accuracy in all four folds, but only by a mean
  `0.000312`. It achieved positive recall for class `-1`, but zero recall for
  class `+1` in every Joint OOS fold; it therefore failed the directional
  criterion.
- **M3:** had higher mean Joint OOS Macro-F1 (`0.214066`) and Balanced
  Accuracy (`0.335448`) than the majority baseline diagnostics, but its
  Accuracy was lower than the baseline in all four Joint OOS folds (mean
  delta `-0.001324`).

M3 also differed across OOS subsets: its mean Accuracy delta was `+0.001548`
on Temporal OOS and `-0.001324` on Joint OOS. This is a recorded diagnostic,
not a revision of the protocol.

Detailed executed artifacts are stored in `experiments/results/` as
`EXP_001_*`.
