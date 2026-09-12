# EXP_002 — Missingness: Neutral vs Directional

## Status

Completed.

The protocol and both decision rules were specified before execution. EXP_002
was executed on training data only; the competition test was not accessed.

## Research question

Does `missing_ratio`, available by 14:00, contain out-of-sample predictive
information for distinguishing neutral from directional end-of-day outcomes
under the frozen EXP_000 validation protocol?

Define the binary target:

```text
Z = 0  if reod = 0        # neutral
Z = 1  if reod ∈ {-1, +1} # directional
```

This hypothesis arose after EXP_001 and the subsequent descriptive
missingness investigation. EXP_001 remains unchanged.

## Scope and fixed protocol

- Reuse exactly EXP_000's frozen equity partition: `E_dev=1463`,
  `E_holdout=366`, seed `20260908`.
- Reuse exactly its four expanding folds and subset definitions:

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

- Joint OOS is primary; Temporal OOS is diagnostic.
- The sessions `{112, 134, 229, 314, 438, 469}` must not be excluded. They
  remain naturally in whichever frozen fit or OOS subset contains them.
- No purging or embargo is introduced; EXP_000's assumption that target
  horizons are contained within sessions remains unchanged.
- The competition test is prohibited from fitting, feature construction,
  model selection, evaluation, and hypothesis evaluation.
- `ID`, `day`, and `equity` are keys or partition variables only.

## Representation

Use M1 only:

```text
missing_ratio = (1 / 53) × Σ 1[r_j is NaN],  j = 0,...,52
```

This is the sole predictive feature. It is a deterministic row-wise value in
`[0, 1]` and learns no parameters. Do not use observed returns, a full binary
mask, early/mid/late summaries, IDs, `day`, `equity`, imputation, or any other
feature engineering.

## Fixed model

Fit binary logistic regression on `fit_f` only:

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

No scaling, hyperparameter search, tuning, model substitution, or threshold
adjustment is permitted. Class predictions use the pre-specified probability
threshold `0.5` for `Z=1` (directional).

## Baseline

For every fold, determine the majority binary class exclusively from `Z` in
`fit_f` and predict that class for both corresponding OOS subsets. Do not
assume which binary class is the majority.

## Evaluation

For the model and baseline, report each fold separately for Temporal OOS and
Joint OOS:

- Balanced Accuracy;
- Accuracy;
- recall neutral (`Z=0`);
- recall directional (`Z=1`);
- confusion matrix with labels ordered `[0, 1]`;
- model ROC-AUC from predicted directional probabilities;
- model minus baseline Accuracy;
- model minus baseline Balanced Accuracy.

Report mean and sample standard deviation (`ddof=1`) across folds as
descriptive stability summaries only, not IID inference. Accuracy and delta
Accuracy are diagnostics, not pass/fail conditions.

## Pre-specified decisions

### Primary — OOS discrimination evidence

This asks whether `missing_ratio` has neutral-versus-directional ranking
information independently of the fixed classification threshold.

EXP_002 has **promising OOS discrimination evidence** only if:

1. ROC-AUC is strictly greater than `0.50` in every one of the four Joint OOS
   folds; and
2. mean Joint OOS ROC-AUC is strictly greater than `0.50`.

Passing this criterion supports only promising OOS discrimination under the
tested representation and model.

### Secondary — fixed-threshold classification evidence

This asks whether such discrimination is already expressed as useful binary
decisions at the fixed threshold `0.5`.

EXP_002 has **fixed-threshold classification evidence** only if, in every
Joint OOS fold:

1. Balanced Accuracy is strictly greater than `0.50`;
2. recall neutral is strictly positive; and
3. recall directional is strictly positive;

and mean Joint OOS delta Balanced Accuracy relative to the fold-specific
majority baseline is strictly positive.

The primary and secondary decisions are intentionally separate. For example,
OOS discrimination may pass while fixed-threshold classification fails. A
secondary failure must not be interpreted as absence of all predictive
information.

## Integrity checks

Before reporting results, the implementation must verify:

- exact reuse of EXP_000 fold boundaries and frozen equity partition;
- temporal separation, `E_dev`/`E_holdout` disjointness, and no holdout equity
  in `fit_f`;
- `Z` is derived exactly as declared and `missing_ratio` is the sole feature,
  with all values in `[0, 1]`;
- only `fit_f` enters model fitting and majority-baseline selection;
- no structurally unusual session is filtered;
- no competition-test row is accessed;
- row counts and binary class counts are reported for every fit and OOS
  subset; and
- both binary classes occur in every OOS subset. Otherwise the relevant
  Balanced Accuracy, recall, and/or ROC-AUC decision is not evaluable and the
  experiment must not be declared passing.

## Interpretation boundary and limitation

**Predictive information** means whether `missing_ratio` discriminates neutral
from directional outcomes in the internal Joint OOS proxy.

**Decision utility** means whether this information improves the original
ternary challenge objective or a future hierarchical decision system.
EXP_002 evaluates predictive information only. It neither establishes
causality nor validates a production exclusion rule for the six structurally
unusual sessions. The 503 labelled days remain an imperfect internal proxy for
deployment on a distinct future period and a wholly new equity universe.

## Results

All real-data integrity checks passed. The binary majority class derived from
`fit_f` was directional (`Z=1`) in all folds, and both binary classes were
present in every evaluated OOS subset. The six structurally unusual sessions
remained naturally in their frozen folds; they were not filtered. The
competition test was not accessed.

### Joint OOS

| Fold | ROC-AUC |
|---|---:|
| 1 | 0.604098 |
| 2 | 0.584693 |
| 3 | 0.595994 |
| 4 | 0.598582 |

Joint OOS mean results:

- ROC-AUC: `0.595842`
- Balanced Accuracy: `0.578116`
- Accuracy: `0.639120`
- Recall neutral: `0.229846`
- Recall directional: `0.926386`
- Delta Balanced Accuracy versus binary majority baseline: `+0.078116`
- Delta Accuracy versus binary majority baseline: `+0.051482`

**Primary OOS discrimination verdict: PASS.** ROC-AUC exceeded `0.50` in all
four Joint OOS folds and its mean exceeded `0.50`.

**Secondary fixed-threshold classification verdict: PASS.** Balanced Accuracy
exceeded `0.50`, both recalls were positive, and the mean Balanced Accuracy
delta versus the fold-specific majority baseline was positive in every Joint
OOS fold as specified.

The threshold-`0.5` classifier was asymmetric: mean Joint OOS neutral recall
was `0.229846`, while directional recall was `0.926386`. These results must
not be described as strong neutral detection.

### Temporal OOS diagnostic

- Mean ROC-AUC: `0.600211`
- Mean Balanced Accuracy: `0.586526`
- Mean Accuracy: `0.641073`

## Interpretation boundary after execution

EXP_002 provides evidence that `missing_ratio` contains OOS discriminatory
information for neutral versus directional outcomes under the frozen
validation protocol. It does not establish causality, that missingness is a
liquidity measure, directional-sign predictability, improvement on the
original ternary challenge, or validity of a hierarchical ternary
architecture.
