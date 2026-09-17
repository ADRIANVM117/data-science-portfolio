# D007 — Raw Positional Returns Beyond Missingness

## Status

Completed — Discovery only — PRIMARY SCREEN: PASS.

## Scientific question

Within the physical Discovery partition, do `r0...r52` add consistent OOS ternary probability information beyond `m0...m52` when both are modeled by the same frozen nonlinear learner?

## Scope, target, and folds

Use only physical Discovery: `days 0–352 × E_dev`. The full ternary target is `Y ∈ {-1, 0, +1}`. No directional conditioning, D004 gate, target-based row filtering, full-training fallback, days `353–502`, `E_holdout`, or competition-test data are permitted.

| Fold | Fit | Validation |
| --- | --- | --- |
| 1 | 0–202 | 203–252 |
| 2 | 0–252 | 253–302 |
| 3 | 0–302 | 303–352 |

## Frozen representations

```text
M   = [m0, ..., m52]
R+M = [r0, ..., r52, m0, ..., m52]
m_t = 1[r_t is NaN]
```

`M` is the scientific control. No engineered feature may enter either matrix, including recent intensity, D006 `O`, cumulative return, sign dominance, rolling or volatility statistics, embeddings, cross-sectional features, ranks, identities, or SHAP-derived features.

For `R+M`, finite observed returns are preserved exactly; observed zero remains zero; `NaN` remains native; and masks are built from original returns. There is no imputation, scaling, clipping, winsorization, or normalization. A non-NaN non-finite return is an integrity failure.

M and R+M must use identical rows, IDs, targets, ordering, folds, learner, hyperparameters, seed, training policy, class handling, and metrics. Only the permitted information set differs. Reporting probability order is `[-1,0,+1]`.

## Frozen learner

Require XGBoost `3.4.1` and use independently for M and R+M:

```python
XGBClassifier(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    n_estimators=300, learning_rate=0.05, max_depth=3,
    min_child_weight=50, subsample=0.8, colsample_bytree=1.0,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
    tree_method="hist", n_jobs=1, random_state=20260908, verbosity=0,
)
```

No early stopping, tuning, alternate seed, class weights, calibration, ensemble, alternate learner, or validation-driven capacity selection is allowed.

## Evaluation and frozen Discovery screen

The primary metric is multiclass log loss. For each Validation fold:

```text
delta_log_loss = log_loss(M) - log_loss(R+M)
```

Positive delta means raw return values improve ternary probability quality over masks alone. The primary screen passes only if `delta_log_loss > 0` in all three folds and its mean is strictly positive. No minimum effect-size threshold is defined. The screen addresses existence and consistency of incremental information, not material usefulness.

For M and R+M, report log loss, negative/positive one-vs-rest tail AUC, MacroTailAUC, ternary accuracy, Macro-F1, balanced accuracy, class recalls, and confusion matrices in `[-1,0,+1]` order. A fit-majority ternary baseline is descriptive only and is not the scientific control.

## Interpretation and governance

A primary pass establishes only consistent incremental Discovery OOS ternary probability information from raw returns beyond missingness under this frozen XGBoost probe. It does not establish material predictive value, economic or directional alpha, deployability, official-score improvement, or lockbox generalization. A failure does not establish that the raw path contains no signal.

If D007 passes, Human + Sol must first interpret its magnitude and diagnostics. Only then may a separately authorized SHAP interpretation phase explain the already-fitted frozen model to generate later hypotheses; it cannot modify D007 or recycle its Validation result as confirmation. If D007 fails, there is no SHAP rescue, alternate XGBoost specification, replacement learner, feature engineering, alternate seed, or calibration rescue under D007.

Result artifacts will record membership, class counts, model specification, M/R+M metrics, deltas, screen decision, confusion matrices, majority baseline, and protected-boundary metadata. They may be written only after separately authorized real execution.

## Technical incident history

The first separately authorized execution attempt terminated before valid
result generation or artifact persistence. Its probability-coherence assertion
used a fixed row-sum tolerance of `1e-12`; valid XGBoost `float32` outputs had
machine-epsilon-scale deviations, with maximum absolute deviation
`1.1920928955e-07`. No predictive result was inspected. Human + Sol approved
an assertion-only, dtype-aware repair: `max(1e-12, 2 * eps(dtype))`, retaining
`rtol=0.0`. It neither clips nor renormalizes probabilities and does not alter
the scientific specification. Any later real D007 execution requires separate
authorization.

## Executed result

The first execution attempt was a `TECHNICAL EXECUTION FAILURE — NO SCIENTIFIC
RESULT`: the fixed `atol=1e-12` probability row-sum assertion rejected valid
float32 machine-epsilon-scale deviations before any predictive result was
generated or inspected. Human + Sol approved an assertion-only dtype-aware
repair, which passed the required tests without changing model, features,
folds, seed, metrics, or screen. The subsequent authorized run is D007's first
scientifically interpretable execution.

The frozen primary screen passed:

| Fold | log loss M | log loss R+M | delta log loss |
| ---: | ---: | ---: | ---: |
| 1 | 1.029137 | 0.999890 | +0.029247 |
| 2 | 1.045175 | 1.021698 | +0.023477 |
| 3 | 1.032531 | 1.006671 | +0.025861 |
| Mean | 1.035615 | 1.009420 | +0.026195 |

The sample standard deviation of delta was `0.002899`. Delta log loss was
strictly positive in every fold and on average.

The frozen D007 probe provides evidence of consistent incremental Discovery
OOS ternary probability information from raw return values beyond missingness
under this fixed XGBoost learner.

This does not establish economic or directional alpha, deployability,
competition-score improvement, lockbox generalization, a causal mechanism, or
material usefulness under any post-hoc threshold.

Secondary diagnostics were descriptive only, but moved in the same direction:

| Metric | M mean | R+M mean |
| --- | ---: | ---: |
| MacroTailAUC | 0.566113 | 0.618777 |
| AUC -1 | 0.563384 | 0.607790 |
| AUC +1 | 0.568841 | 0.629764 |
| Accuracy | 0.413778 | 0.464439 |
| Macro-F1 | 0.224811 | 0.410687 |
| Balanced Accuracy | 0.339889 | 0.426883 |
| Recall -1 | 0.050036 | 0.322349 |
| Recall 0 | 0.967384 | 0.750523 |
| Recall +1 | 0.002248 | 0.207777 |

Both one-vs-rest tail AUCs improved in every fold. These metrics were not part
of the primary screen and remain descriptive.

Scikit-learn emitted probability-row-sum warnings while calculating log loss,
consistent with the documented float32 behavior. No adaptation was made and
execution completed successfully. Only physical Discovery `days 0–352 × E_dev`
were used; days `353–502`, `E_holdout`, the competition test, and full-training
fallbacks were untouched.

No SHAP or feature-importance analysis has been run, no tuning or alternate
learner was tried, and no post-result feature engineering occurred. No D008 is
authorized. A separately governed Human + Sol design review is required before
any bounded post-D007 interpretation phase.
