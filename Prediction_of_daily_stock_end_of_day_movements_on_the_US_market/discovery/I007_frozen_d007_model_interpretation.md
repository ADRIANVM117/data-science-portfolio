# I007A — Frozen D007 Main-Effect Model Interpretation

## Status

Frozen — implementation/testing authorized — real reconstruction and real SHAP not yet authorized.

## Purpose and scope

I007A is a descriptive, hypothesis-generation interpretation of the already
frozen D007 `R+M` learner. It does not refit, compare, validate, improve, or
rescue D007. It may use only physical Discovery (`days 0–352 × E_dev`), its
three frozen folds, and Validation observations. Days `353–502`, `E_holdout`,
the competition test, and all interaction-SHAP analyses are forbidden.

Scientific question: conditional on exact reconstruction of the frozen D007
`R+M` models, how does TreeSHAP allocate each model's raw-margin behavior
among the 53 raw-return and 53 missingness-mask inputs?

This is not an attribution of unique information, causality, economic
mechanism, predictive validity, or confirmation.

## Exact reconstruction gate

The original estimators were not persisted. For each fold, reconstruct only
the D007 `R+M` model with the exact physical Discovery loader, fold splitter,
row order, target encoding, native-NaN `R+M` matrix, XGBoost `3.4.1`
specification, and seed from D007. Before any SHAP call, reconstruct its
Validation multiclass log loss and compare it with the persisted D007 `R+M`
value in `discovery/results/D007_metrics.csv`.

The absolute tolerance is frozen at `1e-8` (`rtol=0`). This is strict relative
to the persisted double-precision metric while allowing only immaterial
deterministic numerical representation differences. All three gates must pass
before any real SHAP computation; a mismatch stops I007A.

## Frozen SHAP configuration and additivity gate

```python
shap.TreeExplainer(
    reconstructed_model,
    data=None,
    feature_perturbation="tree_path_dependent",
    model_output="raw",
    feature_names=list(RM_COLUMNS),
)
```

Explain Validation samples only. The expected SHAP-value shape is
`(n_rows, 106, 3)`, with output indices `0 → -1`, `1 → 0`, `2 → +1`.
For every explained row and output, verify raw-margin additivity:

```text
base_value + sum(feature SHAP values) = XGBoost output_margin
```

with `np.allclose(rtol=1e-5, atol=1e-5)`. A failed shape, finite-value, or
additivity check stops I007A; values must not be clipped, renormalized, or
otherwise altered.

The `tree_path_dependent` convention describes the fitted trees' path-based
attribution allocation. With temporally dependent adjacent returns, native
NaNs, and masks that directly record each return's availability, it does not
identify unique conditional contributions or an intervention on one feature.

## Deterministic Validation sampling

For fold `f`, sample `min(2000, n_validation)` positions without replacement
using `np.random.default_rng(20260908 + f)`, then restore ascending source-row
order. Sampling never uses `Y`; the identical sampled positions and ordering
are used for every output and every I007A summary. Persist fold, source row
position, and `ID`. No interaction sample exists in I007A.

## Main-effect summaries

For feature `f` and output class `c`, define

```text
A[f,c] = mean_i |SHAP[i,f,c]|.
```

Report this by fold and output class and as an unweighted three-fold macro
mean. Also report the raw-return allocation share
`sum(A[r0:r52,c]) / sum(A[:,c])` and the mask allocation share analogously.
These are attribution allocations, not measures of unique predictive
information.

For each output separately, rank all 106 features by `A` in each fold and
report fold rank, median rank, rank range, the three pairwise Spearman rank
correlations, Top-10 feature sets, pairwise Top-10 intersection count and
Jaccard overlap, the three-fold intersection, and feature membership in
0/1/2/3 fold Top-10 sets. Exact ties in `A` are broken by the frozen
`RM_COLUMNS` order, ensuring exactly ten Top-10 features per fold. A feature is called stable Top-10 only if it is in
the Top-10 in all three folds; this is descriptive.

## Limited raw-feature orientation diagnostics

For output `-1`, analyze only raw return features in its stable Top-10 set;
for output `+1`, analogously analyze only its own stable Top-10 set. For each
eligible feature/output/fold, use observed values only and report Spearman
correlation between raw feature value and its same-output SHAP value, plus
ten deterministic equal-count bins. Ties are ordered by original sampled row
position before splitting into ten contiguous equal-count-as-possible bins;
therefore exactly ten nonempty bins are used when at least ten observed rows
exist. Report bin count, raw-value minimum/maximum/mean, and mean SHAP.
If fewer than two distinct observed values exist, Spearman is `NaN` with an
`undefined_constant` status. No continuation/reversal label is assigned.

## Outputs and boundaries

Expected real-execution artifacts are reconstruction gates, sample manifest,
additivity checks, allocation summaries, rank/stability and Top-10 summaries,
and the limited orientation tables plus metadata. No SHAP interaction values,
dependence plots, feature selection, new predictive metric, or model change
is allowed.

Any hypothesis generated here must be separately frozen before a later
Discovery or Confirmation test.
