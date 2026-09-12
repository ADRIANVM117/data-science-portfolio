# EXP_006 — Nonlinear Conditional Positional Path Signal

## Status

Frozen, not yet executed.

## Scientific question

Among oracle-known directional and evaluable observations, do the positional
return values in `r0...r52` add Joint OOS sign discrimination beyond
positional availability when modeled through fixed nonlinear tree
interactions?

## Population, target, and validation

Reuse EXP_005 exactly:

```text
D = {reod ∈ {-1, +1}}
S = 0  if reod = -1
S = 1  if reod = +1
N_obs = Σ 1[r_j is observed]
effective population = D ∩ {N_obs >= 1}
```

Construct the normal frozen EXP_000 subsets first, then apply directional
conditioning and the `N_obs >= 1` evaluability rule.

Reuse EXP_000 exactly:

- frozen `E_dev=1463` / `E_holdout=366` partition, seed `20260908`;
- four expanding folds;
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

Reuse EXP_005 exactly. `A_mask` and `B_path_mask` must evaluate exactly the
same retained rows in every corresponding subset.

### A_mask

```text
A_mask = [m0, ..., m52]
m_j = 1 iff r_j is NaN
```

It has exactly 53 binary predictors and no learned preprocessing.

### B_path_mask

```text
B_path_mask = [
  53 fit-median-imputed and fit-StandardScaler-transformed positional returns,
  53 original unscaled binary masks
]
```

It has exactly 106 predictors. For each fold:

1. Construct masks from original returns before imputation.
2. Compute 53 positional medians from evaluable directional `fit_f` only.
3. Assert all 53 medians are finite.
4. A non-finite fit median is an integrity/evaluability failure: stop without
   zero replacement, position dropping, OOS-derived replacement, or a
   scientific PASS/FAIL verdict.
5. Impute fit and OOS subsets using fit medians only.
6. Fit `StandardScaler` on the 53 imputed fit-return columns only.
7. Transform OOS subsets without refitting.
8. Append original masks unchanged and unscaled.

Do not use native HistGradientBoosting NaN handling.

No additional features may enter either representation, including global
return summaries, missingness summaries, counts, windows, runs, transitions,
technical indicators, IDs, `day`, or `equity`.

## Fixed model

This contract freezes scikit-learn version `1.6.1`. Fit the following exact
estimator separately for A_mask and B_path_mask:

```python
HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0,
    max_features=1.0,
    max_bins=255,
    categorical_features="from_dtype",
    monotonic_cst=None,
    interaction_cst=None,
    warm_start=False,
    early_stopping=False,
    scoring="loss",
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-7,
    verbose=0,
    random_state=20260908,
    class_weight=None,
)
```

The only deliberate non-default methodological choices are:

- `early_stopping=False`, avoiding an internal validation and capacity-
  selection mechanism; and
- `random_state=20260908`, ensuring reproducibility.

`validation_fraction`, `n_iter_no_change`, `tol`, and `scoring` are inert
when early stopping is disabled, but remain explicit to freeze estimator
state. No tuning, alternative tree model, RNN, ensemble, SHAP, feature
selection, or feature-importance-driven redesign is permitted.

## Baseline and evaluation

Use the EXP_005 fit-derived majority-sign baseline: it is determined only
from evaluable directional `fit_f`, with class order `[0, 1]` and ties assigned
to `S=0`.

Verify fitted classes explicitly. Compute ROC-AUC using the probability for
`S=1`, and use a fixed classification threshold of `0.5`.

For each representation, fold, and Temporal/Joint OOS subset, report:

- ROC-AUC;
- Balanced Accuracy;
- Accuracy;
- recall negative and recall positive;
- confusion matrix with labels `[0, 1]`;
- delta Accuracy versus majority baseline; and
- delta Balanced Accuracy versus majority baseline.

Report fold-level and mean Joint OOS:

```text
delta_AUC = AUC(B_path_mask) - AUC(A_mask)
```

Means and sample standard deviations (`ddof=1`) are descriptive stability
summaries only, not IID inference.

## Pre-specified decisions

### Primary — incremental nonlinear positional-return OOS discrimination

EXP_006 supports evidence that positional return values add
nonlinear/interacting OOS sign information beyond availability only if all
conditions hold:

1. B_path_mask Joint ROC-AUC is strictly greater than `0.50` in all four
   folds.
2. Mean B_path_mask Joint ROC-AUC is strictly greater than `0.50`.
3. `AUC(B_path_mask) - AUC(A_mask)` is strictly greater than zero in every
   Joint fold.
4. Mean Joint delta_AUC is strictly greater than zero.

No arbitrary minimum improvement threshold is used.

### Secondary — fixed-threshold classification for B_path_mask

At threshold `0.5`, B_path_mask has fixed-threshold classification evidence
only if, in every Joint OOS fold:

1. Balanced Accuracy is strictly greater than `0.50`.
2. Recall negative is strictly positive.
3. Recall positive is strictly positive.

Mean Joint delta Balanced Accuracy versus the fit-derived majority baseline
must also be strictly positive. Primary and secondary criteria are
independent. Accuracy and delta Accuracy are diagnostics.

## Role of EXP_005

The A_mask-versus-B_path_mask comparison is the primary within-experiment
attribution. EXP_005 Logistic Regression results are frozen historical
references only. B_path_mask differences between EXP_005 and EXP_006 may be
reported descriptively as model-capacity evidence, but EXP_005 metrics must
not enter EXP_006 PASS/FAIL functions, set hyperparameters or thresholds,
trigger feature changes, or become a tuning target.

## Integrity checks

Before reporting results, verify:

- exact reuse of the EXP_000 folds and equity partition;
- normal splitting before conditioning/evaluability;
- retained rows meet `D ∩ {N_obs >= 1}`;
- A_mask and B_path_mask row identities and order match exactly;
- masks precede imputation, are binary, and remain unscaled;
- medians and scaler are fit only on evaluable directional `fit_f`;
- every fit median is finite;
- OOS preprocessing is transform-only;
- B_path_mask has exactly 106 declared finite features and A_mask exactly 53;
- no holdout equity enters fit;
- both sign classes exist in fit and every evaluated OOS subset;
- predicted probability is the `S=1` column and threshold is exactly `0.5`;
- installed scikit-learn version is exactly `1.6.1`; and
- the competition test is not accessed.

Any integrity/evaluability failure stops the experiment without silently
repairing the representation or producing a scientific PASS/FAIL verdict.

## Interpretation boundaries and scientific capability

If the primary criterion passes, the supported conclusion is:

> Under the frozen protocol and fixed HistGradientBoosting learner, adding
> positional return values to positional availability provides consistent
> Joint OOS sign discrimination, compatible with predictive information
> expressed through nonlinear/interacting functions that the EXP_005
> linear-additive representation did not establish.

Use incremental OOS predictive evidence/performance language, not causal
attribution. A pass does not establish temporal sequence dynamics, causality,
deployable sign prediction, ternary challenge improvement, trading utility,
RNN superiority, general boosting superiority, or competition-period
generalization.

If the primary criterion fails, the supported conclusion is only:

> No OOS evidence under the frozen criterion that positional return values add
> sign discrimination beyond positional availability through this fixed
> HistGradientBoosting representation.

A failure does not establish that nonlinear information, temporal ordering,
sequential models, RNNs, or the challenge itself cannot contain useful
information.

EXP_006 tests a fixed nonlinear tabular learner's ability to exploit
thresholds and interactions among temporally positioned returns and masks. It
does not test whether explicitly modeling chronological/sequential dynamics
adds predictive information.
