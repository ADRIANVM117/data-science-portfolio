# D008 — Raw Path Beyond Availability + Established Recent Intensity

## Status

Completed — PRIMARY SCREEN PASS — Discovery only.

## Scientific question

Within physical Discovery, do raw positional return values provide consistent
out-of-sample ternary predictive information beyond a shared control containing
full positional availability and the previously established EXP_008 recent-
movement-intensity feature, under the same frozen nonlinear learner?

## Scope, target, and frozen folds

Use only physical Discovery: `days 0–352 × E_dev`. The target is full ternary
`Y ∈ {-1, 0, +1}`. Days `353–502`, `E_holdout`, competition-test data, and any
full-training fallback are forbidden.

| Fold | Fit days | Validation days | Expected Fit rows | Expected Validation rows |
| ---: | --- | --- | ---: | ---: |
| 1 | 0–202 | 203–252 | 271,897 | 67,031 |
| 2 | 0–252 | 253–302 | 338,928 | 66,989 |
| 3 | 0–302 | 303–352 | 405,917 | 66,899 |

## Frozen representations

```text
M = [m0, ..., m52], where m_t = 1[r_t is NaN]
R = [r0, ..., r52]
```

`M` uses exact D007 semantics and order. Observed zero is observed (`m_t=0`);
no preprocessing is applied. `R` uses exact D007 raw-return semantics and
order: finite observed values and zeros are preserved, NaNs remain native, and
every non-NaN non-finite value is an integrity failure. No imputation,
scaling, clipping, winsorization, or normalization is applied to `R`.

Define positional recent window `W=[r41,...,r52]`, with no exact clock-time
claim. For each row:

```text
N_obs,W = sum_{t in W} 1[r_t is observed]
raw I_recent = sqrt(sum_{observed t in W}(r_t^2) / N_obs,W), if N_obs,W > 0
raw I_recent = NaN,                                             if N_obs,W = 0
```

Observed zeros count in `N_obs,W` and contribute zero squared movement. NaNs
contribute to neither component. `q` is not a D008 predictor.

Within each fold, reproduce EXP_008 preprocessing exactly: compute raw
intensity from untouched returns; take the ordinary median over defined Fit
raw intensities only; impute undefined Fit and Validation intensities with
that Fit-only median; fit `StandardScaler` on imputed Fit intensity only; and
transform Fit and Validation without refitting. The shared resulting feature
is `I_recent_z`. No clipping, winsorization, log transform, or alternative
normalization is allowed.

```text
C = [m0, ..., m52, I_recent_z]                   # 54 predictors
P = [m0, ..., m52, I_recent_z, r0, ..., r52]      # 107 predictors
```

`C` and `P` must retain identical rows, IDs, targets, order, folds, and
identical `I_recent_z` values. They differ only in raw positional returns.
Do not add `missing_ratio`, `q`, SHAP-selected columns, interactions,
engineered summaries, D006 features, alternate windows, recency weighting, or
cross-sectional features. The rationale is frozen: full `M` deterministically
contains the availability state from which scalar `missing_ratio` and `q` can
be derived; D008 controls full positional availability rather than reproducing
EXP_008's original learner-facing vector exactly.

## Frozen learner

Require XGBoost `3.4.1`. Fit independent instances for `C` and `P` using:

```python
XGBClassifier(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    n_estimators=300, learning_rate=0.05, max_depth=3,
    min_child_weight=50, subsample=0.8, colsample_bytree=1.0,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
    tree_method="hist", n_jobs=1, random_state=20260908, verbosity=0,
)
```

No early stopping, tuning, alternate seed, class weighting, calibration,
ensemble, or alternate learner is allowed. Class encoding and probability
order are D007's `[0,1,2] → [-1,0,+1]` convention.

## Evaluation and primary screen

The primary metric is multiclass log loss. For each Validation fold:

```text
delta_log_loss = log_loss(C) - log_loss(P)
```

Positive delta means `P` has better ternary probability quality. PRIMARY PASS
requires strictly positive delta in all three folds and strictly positive mean
delta. No effect-size threshold is frozen. A pass establishes consistency
under this frozen screen only; material/economic usefulness is a separate
post-execution interpretation question.

For both arms, report D007's descriptive metrics: MacroTailAUC, one-vs-rest
tail AUCs for `-1` and `+1`, ternary accuracy, Macro-F1, balanced accuracy,
per-class recalls, and confusion matrices in `[-1,0,+1]` order. Persist the
Fit-majority ternary hard-label baseline as context only; it is not the
scientific control and no secondary metric creates a decision criterion.

## Interpretation boundaries and pre-result threats

If PRIMARY PASS, the permitted conclusion is that raw positional returns add
consistent incremental Discovery OOS ternary predictive utility beyond the
shared full-positional-availability plus established-recent-intensity control
under this frozen XGBoost learner. It does not establish individual unique
information, SHAP causation, directional alpha, information-theoretic
uniqueness, or an economic mechanism.

If PRIMARY FAIL, D008 did not find consistent incremental utility under this
learner and representation. It does not establish no residual path signal,
that the path is useless, that D007 was false, or that intensity fully explains
D007; finite-sample, model-capacity, representation, and interaction limits
remain possible.

Known pre-result threats: `I_recent_z` is intentionally derived from part of
`R` and shared; `P` encodes availability redundantly through explicit `M` and
native NaNs in `R`; dimensionality differs by design; and identical numerical
hyperparameters can allocate practical capacity differently across feature
spaces. D008 therefore measures incremental utility under the frozen learner
and representation interface, not universal or information-theoretic
uniqueness.

## Required integrity checks and artifacts

Require exact W/intensity semantics, fit-only preprocessing, finite observed
returns, native-NaN preservation, exact 54/107 schemas, aligned rows/IDs/
targets/order, D007 model/version/probability semantics, frozen Discovery
folds and expected counts, and pre-execution artifact freshness. Tests must
not create real result artifacts.

An authorized future run may write only:

```text
D008_fold_manifest.csv
D008_preprocessing_parameters.csv
D008_model_spec.json
D008_metrics.csv
D008_incremental_log_loss.csv
D008_summary.json
D008_confusion_matrices.json
D008_majority_baseline.csv
D008_metadata.json
```

## Executed result

One authorized real execution completed with:

```text
python discovery/run_d008_raw_path_beyond_availability_intensity.py
```

It exited with status `0` in approximately `406.35` seconds. All three frozen
Discovery folds completed and all nine authorized artifacts persisted. The
contract SHA-256 remained
`C8576CD6EA1CCA65A2B886560C65CFDD38CC44ED9F7E89C2061B76674812CDC8`.
Five scikit-learn probability-row-sum warnings occurred during log-loss
calculation; there was no error, adaptation, repair, or rerun. Protected
boundaries remained intact. The persisted preprocessing artifact does not
contain Validation raw-`I_recent` NaN counts; this is a reporting omission
only, and no post-run calculation or rerun was authorized to recover them.

| Fold | log loss C | log loss P | delta log loss C−P |
| ---: | ---: | ---: | ---: |
| 1 | 1.0018695850120722 | 0.9982946876583179 | +0.0035748973537543804 |
| 2 | 1.0205728750298875 | 1.0198104887499693 | +0.0007623862799182035 |
| 3 | 1.0068907621206846 | 1.0039140843873360 | +0.0029766777333486427 |
| Mean | 1.0097777407208814 | 1.0073397535985410 | +0.002437987122340409 |

The sample standard deviation of the three deltas was
`0.001481619153461506`. Every fold delta and the mean delta were strictly
positive; therefore the frozen PRIMARY SCREEN **PASS**ed.

Under the frozen XGBoost learner, raw positional return values provided
consistent incremental Discovery OOS ternary predictive utility beyond the
shared full-positional-availability plus established-recent-intensity control
representation. The residual improvement is modest in magnitude.

### Descriptive secondary results

| Metric | C mean | P mean | P−C mean |
| --- | ---: | ---: | ---: |
| MacroTailAUC | 0.619066 | 0.619871 | +0.000805 |
| AUC -1 | 0.607099 | 0.608547 | +0.001448 |
| AUC +1 | 0.631033 | 0.631196 | +0.000163 |
| Accuracy | 0.451771 | 0.463925 | +0.012154 |
| Macro-F1 | 0.348076 | 0.413834 | +0.065757 |
| Balanced Accuracy | 0.415403 | 0.429522 | +0.014119 |
| Recall -1 | 0.490856 | 0.360556 | -0.130299 |
| Recall 0 | 0.744146 | 0.729032 | -0.015114 |
| Recall +1 | 0.011209 | 0.198979 | +0.187770 |

These secondary results are descriptive and do not alter the primary decision.
Tail-ranking metrics changed only modestly after adding `R`, whereas hard
multiclass behavior changed more visibly: the control almost never selected
`+1`, while `P` increased `+1` recall with lower `-1` recall. This is compatible
with residual raw-path information affecting ternary probability allocation,
but it is not evidence that a directional/sign mechanism has been isolated.

### Relation to D007 and unresolved explanations

D007 showed `M+R` beyond `M` with mean incremental log-loss improvement of
approximately `+0.026195`. D008 asked the stricter comparison
`M+I_recent_z+R` versus `M+I_recent_z`, retaining a smaller, consistently
positive mean improvement of `+0.002437987122340409`. Providing established
recent intensity explicitly to both arms substantially reduced the remaining
raw-path advantage, while a smaller consistent residual remained. This is a
descriptive program comparison, not a variance, signal, information, or
causal decomposition.

The residual remains unresolved. Possible explanations include residual
nonlinear directional/class-allocation structure, other magnitude/path-shape
information not summarized by `I_recent_z`, native-NaN routing plus duplicated
availability representation, finite-sample/model-capacity effects, or
interactions not identifiable from current descriptive metrics. No SHAP,
interaction analysis, alternate control, feature selection, tuning, or new
model was run. No D009 is authorized.
