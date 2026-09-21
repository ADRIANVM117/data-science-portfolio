# I007A — Frozen D007 Main-Effect Model Interpretation

## Status

Completed — descriptive interpretation only — Discovery only.

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

## Execution record and frozen outputs

I007A completed one valid reconstruction-and-interpretation run after a prior
technical persistence failure. The first attempt passed reconstruction,
SHAP-shape, and additivity gates but stopped during artifact assembly with
`KeyError: 'median_rank'`; no interpretation output was reviewed. The
subsequent repair changed only the output-assembly merge identity to
`[output_index, output_label, feature]` with `many_to_one` validation. It did
not alter the reconstructed D007 model, data, sampling, SHAP configuration, or
scientific specification. The valid run completed in `337.362784500001`
seconds.

All three deterministic D007 reconstruction gates passed (`atol=1e-8`,
`rtol=0`): the absolute Validation log-loss differences for folds 1--3 were
`1.1102230246251565e-16`, `0`, and `0`, respectively. TreeSHAP used the
frozen `tree_path_dependent`, raw-margin configuration on deterministic
Validation samples of 2,000 rows per fold (seeds `20260909`, `20260910`, and
`20260911`). The returned SHAP arrays had shape `(2000, 106, 3)` and
raw-margin additivity passed for every fold; maximum absolute residuals were
`1.3156678e-06`, `1.2066084e-06`, and `1.0592630e-06`.

### Level 1 — raw-return versus explicit-mask allocation

For every output and fold, the persisted main-effect allocation was
`R_share = 1.0` and `M_share = 0.0`. Thus the fitted D007 `R+M` XGBoost model
assigned no main-effect TreeSHAP attribution to the explicit mask columns.
This does **not** show that missingness is irrelevant: raw return features
retain native `NaN` values, so a return feature can jointly represent an
observed value versus missing/nonmissing through XGBoost native missing-value
routing. Explicit masks can consequently be redundant conditional on that raw
representation. These allocation shares are not unique-information shares.

### Level 2 — positional attribution stability

Pairwise fold rank correlations for mean absolute SHAP attribution were high:

| Output | Fold 1/2 | Fold 1/3 | Fold 2/3 |
| --- | ---: | ---: | ---: |
| -1 | 0.9867 | 0.9798 | 0.9871 |
| 0 | 0.9852 | 0.9757 | 0.9885 |
| +1 | 0.9791 | 0.9552 | 0.9773 |

The frozen three-fold stable Top-10 intersections were:

| Output | Stable raw-return features |
| --- | --- |
| -1 | `r33`, `r46`, `r48`, `r52` |
| 0 | `r1`, `r44`, `r46`, `r47`, `r48`, `r49`, `r50`, `r51`, `r52` |
| +1 | `r24`, `r50`, `r51`, `r52` |

All stable features were raw returns; `r52` was the only stable feature shared
by outputs `-1` and `+1`. The feature ranking was highly stable across the
three Discovery folds, with late-path positions especially recurrent for the
neutral output. This is an attribution-stability observation, not evidence
that any individual position is a standalone predictive feature or alpha.

### Level 2B — persisted same-output orientation summaries

The limited, frozen same-output Spearman signs were consistent across the
three folds for each eligible feature. For output `-1`, the correlations for
`r33`, `r46`, `r48`, and `r52` were positive in every fold:

| Output | Feature | Fold 1 | Fold 2 | Fold 3 |
| --- | --- | ---: | ---: | ---: |
| -1 | `r33` | 0.609998 | 0.427964 | 0.327036 |
| -1 | `r46` | 0.653751 | 0.589030 | 0.375846 |
| -1 | `r48` | 0.670851 | 0.520716 | 0.367108 |
| -1 | `r52` | 0.549878 | 0.685541 | 0.564990 |

For output `+1`, the correlations for `r24`, `r50`, `r51`, and `r52` were,
respectively, negative, negative, positive, and negative in every fold:

| Output | Feature | Fold 1 | Fold 2 | Fold 3 |
| --- | --- | ---: | ---: | ---: |
| +1 | `r24` | -0.876181 | -0.875888 | -0.816547 |
| +1 | `r50` | -0.698553 | -0.606309 | -0.046533 |
| +1 | `r51` | 0.389434 | 0.171472 | 0.025380 |
| +1 | `r52` | -0.645446 | -0.598277 | -0.638255 |

Magnitude varied across folds, particularly for `r50` and `r51`. Within this
fitted multiclass model's raw-margin attribution, higher observed `r52` tends
to correspond to a higher SHAP contribution to raw output `-1` and a lower
SHAP contribution to raw output `+1`. This is fitted-model behavior only; no
economic, directional, continuation, reversal, or causal label is assigned.

## Interpretation boundary and next-step status

I007A is descriptive and hypothesis-generating. It does not establish
standalone predictive value or incremental individual information for `r33`,
`r46`, `r48`, `r52`, `r24`, `r50`, or `r51`; causal relevance; unique feature
importance; an economic mechanism; or deployable directional alpha. The
results do not distinguish whether D007's value is largely explained by
established recent intensity/availability, raw positional values incremental
to those structures, nonlinear combinations not reducible to main effects, or
native-NaN availability routing within raw features.

No interaction analysis was performed. Any next question must be separately
designed and frozen; no D008 is authorized by I007A. Only physical Discovery
was used: days `353–502`, `E_holdout`, and competition-test data remain
untouched.
