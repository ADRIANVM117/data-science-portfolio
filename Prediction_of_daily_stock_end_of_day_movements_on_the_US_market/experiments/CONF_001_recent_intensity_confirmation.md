# CONF_001 - Recent Movement Intensity Internal Confirmation

## Status

Completed -- CONFIRMATION CRITERION SATISFIED.

## Post-execution factual addendum

The scientific specification below was frozen before execution and remains
unchanged. CONF_001 was executed once across all three pre-authorized Internal
Confirmation blocks using training data only. The competition test was not
accessed.

Joint Confirmation AUC results were:

| Block | AUC(C) | AUC(B) | B minus C |
|---:|---:|---:|---:|
| 1 | 0.584716 | 0.686750 | +0.102034 |
| 2 | 0.595982 | 0.690781 | +0.094798 |
| 3 | 0.598600 | 0.698219 | +0.099619 |

All three Joint deltas were strictly positive. Their mean was `+0.098817`
and their sample standard deviation was `0.003684`; therefore the frozen
recurrence criterion was satisfied. The historical EXP_008 mean Joint
B-minus-C delta was `+0.093217`. This addendum records factual execution
results only and does not revise the frozen hypothesis, representation,
preprocessing, learner, or decision rule.

## Governance and lineage

This is **Post-EXP_009 Internal Confirmation**, not a pristine holdout and
not the competition lockbox. Days 353--502 are prospectively protected from
the current Discovery program, but outcomes from these dates were historically
exposed by EXP_000--EXP_009. The competition test remains the final lockbox.

Documented lineage:

```text
EXP_002 -> EXP_008 -> EXP_009 -> D001 -> D001 SHAP -> D002 -> CONF_001 freeze
```

CONF_001 is one candidate from that adaptive historical program; it is not
presented as the only hypothesis previously considered.

## Scientific hypothesis

On the full Neutral-versus-Directional population, recent observed movement
intensity over positional window `W = {r41, ..., r52}` adds Joint Internal-
Confirmation ROC-AUC discrimination beyond global missingness and complete
recent-window unavailability. No causal or economic mechanism is claimed.

## Target and population

```text
Z = 0  if reod = 0                 # neutral
Z = 1  if reod in {-1, +1}         # directional
```

Use the full EXP_002 / EXP_008 population. No complete-W or other
Discovery-derived population restriction is permitted.

## Frozen representations

`W = {r41, ..., r52}` is a positional window of twelve chronologically
ordered observations; no exact clock-time endpoint is asserted.

```text
N_obs,W  = sum(1[r_t observed] for t in W)
q        = 1[N_obs,W = 0]
I_recent = sqrt(sum(r_t^2 for observed t in W) / N_obs,W), if N_obs,W > 0
```

Observed zeros count as observed values and contribute zero to the numerator.
NaNs enter neither numerator nor denominator. Raw `I_recent` is undefined
when `q=1`.

```text
C = [missing_ratio, q]
B = [missing_ratio, q, I_recent_z]
missing_ratio = NaNs among r0,...,r52 divided by 53
```

`missing_ratio` and `q` remain unscaled. Within each block, the median of
defined raw intensity is fitted only on Fit rows, used to impute undefined
intensity in Fit and both OOS subsets, and `StandardScaler` is fitted only on
the resulting Fit intensity. Both OOS subsets are transform-only. C and B
must have identical rows, IDs, targets, and order; B differs only by
`I_recent_z`.

## Frozen learner

Fit independent C and B models per block:

```python
LogisticRegression(
    solver="lbfgs", penalty="l2", C=1.0, fit_intercept=True,
    class_weight=None, max_iter=1000, tol=1e-4, random_state=20260908,
)
```

No tuning, early stopping, class-weight change, or alternate learner is
permitted. A fit-derived majority baseline and threshold-0.5 metrics may be
stored only as diagnostics.

## Frozen expanding Internal Confirmation procedure

| Block | Fit | Temporal diagnostic | Joint primary |
|---|---|---|---|
| 1 | days 0--352 x E_dev | days 353--402 x E_dev | days 353--402 x E_holdout |
| 2 | days 0--402 x E_dev | days 403--452 x E_dev | days 403--452 x E_holdout |
| 3 | days 0--452 x E_dev | days 453--502 x E_dev | days 453--502 x E_holdout |

The frozen EXP_000 equity partition is reused. Earlier confirmation dates
may enter later Fits only through this pre-specified expanding rule. No
observed Confirmation outcome may change the hypothesis, population,
representation, preprocessing, learner, metric, retraining policy, or
decision rule.

## Primary metric and criterion

For each Joint Confirmation block:

```text
delta_auc_k = ROC_AUC(B)_k - ROC_AUC(C)_k
```

Joint ROC-AUC is the sole primary metric. Temporal AUC is diagnostic only;
threshold metrics cannot affect the verdict.

**CONFIRMATION CRITERION SATISFIED** if and only if
`delta_auc_k > 0` in all three Joint Confirmation blocks. Always report the
three deltas, their mean, and their sample standard deviation (`ddof=1`).
This is a recurrence criterion, not an effect-size threshold. Effect size is
interpreted separately and no post-result minimum is permitted.

## Integrity requirements

Before real execution, assert exact partition and blocks; fit/Temporal/Joint
boundaries; no competition-test access; Fit-only median and scaler; exact W
and q semantics; identical C/B row identity and targets; B's sole additional
predictor; exact Logistic Regression parameters; both classes in every
required subset; deterministic behavior; and absence of Discovery-only
features.

## Interpretation boundaries

If satisfied, the allowed conclusion is:

> Under the frozen full-population linear specification, recent observed
> movement intensity added recurrent positive Joint Internal-Confirmation
> Neutral-versus-Directional discrimination beyond global missingness and
> complete recent-window unavailability.

The result may then be described relative to historical EXP_008 as broadly
comparable, materially attenuated, or heterogeneous without defining those
categories numerically after execution. If any block has `delta_auc <= 0`,
CONF_001 is **not confirmed**; report the block structure and stop. Do not
rescue the result through alternate windows, complete-W restriction, recency
weights, adjacency, H_peak, SHAP-selected regions, transformations, models,
or thresholds.

Success does not authorize competition-test access. Neither outcome
establishes causality, an economic mechanism, sign predictability, or final
ternary-task utility.
