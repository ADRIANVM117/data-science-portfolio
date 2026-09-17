# D005 — Gated Positional Path Conditional-Sign Signal

## Status

Completed — `DESCRIPTIVE_CANDIDATE_SCREEN_NOT_MET` — Discovery only.

## Post-execution factual record

One authorized execution completed across all three frozen Discovery folds.
All integrity assertions passed, and only physical `days 0–352 × E_dev` were
used; days `353–502`, `E_holdout`, full-training fallbacks, and competition
test data were not accessed. No adaptation or alternative specification was
evaluated.

Validation ROC-AUC was `0.547320`, `0.499176`, and `0.531055` in folds 1–3,
respectively; mean/sample standard deviation was `0.525850` / `0.024490`.
The frozen descriptive screen required AUC strictly above `0.50` in every
fold, so it was not met because Fold 2 was below `0.50`. Secondary
fixed-threshold metrics are recorded in `docs/DISCOVERY_LOG.md` and did not
alter that screen.

## Scientific question

Within oracle-known directional observations routed by the independently
frozen D004 Neutral-versus-Directional gate, does the EXP_005 positional own
return-path representation discriminate the subsequent sign in the physical
Discovery population?

Formally, D005 studies:

```text
P(Y = +1 | Y != 0, G = 1, X)
```

where `G` is the frozen D004 gate and `X` is the frozen 106-column positional
own-path representation below. This is a conditional, model-defined Discovery
question. It is not a confirmation experiment, a claim about a high-volatility
or causal regime, a new feature claim, or a claim that the D004 gate itself
contains conditional-sign information.

## Protected scope and chronological folds

Use only the physical Discovery partition:

```text
days 0–352 × E_dev
```

Do not access days `353–502`, `E_holdout`, raw full-training fallbacks, or
competition-test data. Use exactly the materialized physical Discovery loader
and these frozen expanding folds:

| Fold | Fit days | Validation days |
| --- | --- | --- |
| 1 | 0–202 | 203–252 |
| 2 | 0–252 | 253–302 |
| 3 | 0–302 | 303–352 |

## Independently frozen D004 gate

For each fold, reconstruct the D004 Neutral-versus-Directional model exactly;
it is not refit, redesigned, calibrated, thresholded, or selected using any
conditional-sign outcome.

```text
Z = 0 if reod = 0
Z = 1 if reod ∈ {-1, +1}

W = {r41, ..., r52}
missing_ratio = count_NaN(r0,...,r52) / 53
N_obs,W = count observed returns in W
q = 1[N_obs,W = 0]
I_recent = sqrt(sum(observed r_t^2) / N_obs,W), if N_obs,W > 0

B_D004 = [missing_ratio, q, I_recent_z]
```

`I_recent` is median-imputed and standardized with the Fit-only D004
transformer; `missing_ratio` and `q` remain unscaled. The D004 model is the
frozen Logistic Regression:

```python
LogisticRegression(
    solver="lbfgs", penalty="l2", C=1.0, fit_intercept=True,
    class_weight=None, max_iter=1000, tol=1e-4, random_state=20260908,
)
```

Fit directional priors are computed from the complete Fit ternary labels:

```text
pi_minus = n_fit(reod=-1) / [n_fit(reod=-1) + n_fit(reod=+1)]
pi_plus  = n_fit(reod=+1) / [n_fit(reod=-1) + n_fit(reod=+1)]
pi_max   = max(pi_minus, pi_plus)
p_D_star = 1 / (1 + pi_max)
```

Let `p_D` be the raw `predict_proba` column associated with `Z=1`. For both
Fit and Validation rows define, with exact comparison and no tolerance:

```text
G = 1[p_D > p_D_star]
```

An equality routes to `G=0`. No alternative gate, threshold, probability
calibration, margin, ranking, or selection rule is permitted. Validation sign
labels must not enter gate construction, threshold construction, D004
preprocessing, or D004 fitting.

## Conditional-sign population and operation order

For each normal physical Discovery Fold subset, perform the following order:

```text
1. Construct the frozen D004 gate G from that fold's Fit-only D004 model.
2. Condition on G=1 and reod ∈ {-1,+1}.
3. Apply the inherited all-NaN own-path evaluability rule:
   N_obs = count observed values in r0,...,r52; retain N_obs >= 1.
4. Define the binary sign target S:
   S=0 for reod=-1; S=1 for reod=+1.
5. Fit sign-path preprocessing and the D005 sign classifier on retained Fit
   rows only; evaluate the retained Validation rows.
```

Rows with `G=1`, directional `reod`, and all 53 own returns missing are
non-evaluable and excluded only at step 3. They receive no synthetic zero
path. The count of these exclusions must be reported separately for Fit and
Validation. This is inherited EXP_005 evaluability, not a learned or
target-optimized filter. No other filtering is allowed.

The effective research population is therefore:

```text
{physical Discovery rows} ∩ {G=1} ∩ {reod ∈ {-1,+1}} ∩ {N_obs >= 1}.
```

Oracle knowledge of `reod ∈ {-1,+1}` is allowed solely because D005 is a
conditional-sign study. It must never be represented as a deployable ternary
or competition decision procedure.

## Frozen sign representation

Use exactly EXP_005's own positional path-and-mask representation, with no
additional feature:

```text
X = [x_r0, ..., x_r52, m0, ..., m52]
```

- `m_j = 1[r_j is NaN]` is constructed from the original retained own return
  path before imputation. All 53 masks are binary, original, and unscaled.
- For each return position `r_j`, calculate the median only on retained Fit
  rows, require that all 53 medians are finite, impute Fit and Validation
  with these Fit medians, then fit `StandardScaler` only on imputed retained
  Fit returns.
- `x_r0,...,x_r52` are the 53 transformed positional returns. Validation is
  transform-only. The full matrix has exactly 106 columns in this order:
  scaled returns followed by original masks.

The sign representation contains no `I_recent`, `p_D`, gate margin,
directional priors, `q`, missing-ratio scalar, ranks, cross-sectional values,
weights, adjacency variables, peak features, engineered returns, temporal
summaries, or other predictors. Gate membership defines population only and
is not a model input.

Fit the positional transformer only on:

```text
Fit ∩ {G=1} ∩ {reod ∈ {-1,+1}} ∩ {N_obs >=1}.
```

## Frozen sign model, baseline, and evaluation

Fit exactly one D005 sign classifier per fold:

```python
LogisticRegression(
    solver="lbfgs", penalty="l2", C=1.0, fit_intercept=True,
    class_weight=None, max_iter=1000, tol=1e-4, random_state=20260908,
)
```

Use the fitted `S=1` probability for ROC-AUC. Require both sign classes in
the retained Fit and Validation subsets; otherwise fail closed as an
evaluability/integrity failure. For descriptive threshold diagnostics, use
the fixed threshold `0.5`, with equality predicting `S=1`, and confusion
matrix label order `[0, 1]` (`[-1, +1]`).

The primary Discovery statistic is retained-Validation ROC-AUC. Report each
fold, its mean, and sample standard deviation (`ddof=1`). The descriptive
candidate screen is met only if Validation ROC-AUC is strictly greater than
`0.50` in all three folds. This is not confirmation or an inferential test.

Secondary diagnostics (which cannot rescue or alter the screen) are Accuracy,
Balanced Accuracy, recall negative, recall positive, and the `[0,1]`
confusion matrix at the frozen `0.5` threshold. A Fit-majority sign baseline
may be reported descriptively only; if reported, it is derived from retained
Fit `S` alone with tie order `[0,1]`, so an exact tie selects negative.

## Required integrity record

For every Fold and each Fit/Validation subset, report:

- total physical subset rows before the gate;
- `G=1` rows;
- `G=1` directional rows before all-NaN evaluability;
- all-NaN own-path rows excluded;
- evaluable retained sign rows;
- retained negative and positive counts and proportions;
- `pi_minus`, `pi_plus`, `pi_max`, and `p_D_star` derived from Fit only;
- unique Validation days and equities for the retained population.

Assert the physical Discovery-only boundary, exact folds, no raw-data
fallback, E_dev-only membership, no protected-region or competition access,
Fit-only D004 preprocessing and priors, strict gate comparison, retained
row/index alignment, finite 53 Fit medians, Fit-only positional scaler,
transform-only Validation processing, 106-column predictor schema, binary
unscaled masks, both sign classes where required, correct `S=1` probability
column, and frozen threshold/model parameters.

## Interpretation boundaries

A positive Discovery screen could support only this narrow statement:

> Under the independently frozen D004 gate and the specified 106-column
> positional own-path Logistic Regression, the retained routed directional
> Discovery population showed conditional-sign discrimination.

It would not establish deployable sign alpha, a high-intensity or causal
regime, the validity of selection on `G`, ternary usefulness, official
competition improvement, trading alpha, calibration, confirmation, or
generalization beyond the specified D004-routed population. Because `G` is a
feature-defined selection event, conditioning on it can alter distributions
and relationships through selection/collider effects even in the absence of
new sign information.

A failed screen would not refute D004's Neutral-versus-Directional result,
other sign representations, or sign information outside this frozen gated
population.

No SHAP, coefficient/feature inspection, alternative gate, calibration,
different sign model, XGBoost/HGB/GRU, inversion, threshold adaptation, or
post-result redesign is permitted under D005.

## Implementation governance

Before a real execution, prepare a minimal reuse-first implementation plan,
implement only after approval, pass synthetic/integrity and relevant
regression tests, then obtain a separate authorization for exactly one real
Discovery execution.
