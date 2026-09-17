# D004 — Hierarchical Recent-Intensity Ternary Decision

## Status

Completed — `DESCRIPTIVE_CANDIDATE_SCREEN_MET` (Discovery only).

## Post-execution factual record

The specification below was frozen before one authorized complete execution
across the three Discovery folds. All runtime integrity assertions passed. The
run used only physical `days 0–352 × E_dev`; days `353–502`, `E_holdout`, and
competition-test data were not accessed.

| Fold | Candidate accuracy | Fit-majority accuracy | Delta accuracy | N-v-D AUC | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.452552 | 0.420641 | +0.031911 | 0.687598 | 0.220988 | 0.632411 |
| 2 | 0.450790 | 0.410769 | +0.040021 | 0.679077 | 0.219797 | 0.630398 |
| 3 | 0.425388 | 0.404954 | +0.020434 | 0.679254 | 0.219541 | 0.630528 |
| Mean | 0.442910 | 0.412121 | +0.030789 | 0.681976 | 0.220109 | 0.631112 |
| Sample standard deviation | 0.015200 | 0.007931 | 0.009842 | 0.004869 | 0.000772 | 0.001126 |

`delta_accuracy > 0` held in all three folds, so the frozen descriptive
candidate screen is met. This is not a confirmatory PASS/FAIL rule and does
not establish official competition improvement, calibrated probabilities,
conditional-sign alpha, trading alpha, or lockbox generalization.

In every fold `pi_minus > pi_plus`, so the frozen no-sign-alpha rule routed
all directional predictions to `-1`; it never predicted both directional
classes. The fit-majority baseline was class `0` in all folds. Mean directional
routing was `0.248842`; the implied thresholds were `0.657898`, `0.661508`,
and `0.659623` (mean `0.659676`, sample standard deviation `0.001805`).

## Scope and scientific question

The official competition scoring metric remains unresolved. D004 adopts
ordinary ternary hard-class accuracy as an **internal Discovery metric** only;
it is not competition confirmation or an official-score estimate.

Within the physical Discovery region only, can the already-supported
Neutral-versus-Directional recent-intensity model be converted into a ternary
hard-label decision rule that improves ordinary accuracy over the Fit-majority
ternary baseline, without introducing a conditional-sign model?

D004 tests a decision layer, not a new alpha feature.

Only the physical Discovery partition `days 0–352 × E_dev` may be used. Days
`353–502`, `E_holdout`, and competition-test data are prohibited.

## Population and chronological folds

Use the full ternary population. Do not condition on direction, restrict to
complete windows, or exclude rows based on targets.

| Fold | Fit days | Validation days |
| --- | --- | --- |
| 1 | 0–202 | 203–252 |
| 2 | 0–252 | 253–302 |
| 3 | 0–302 | 303–352 |

All rows must come from the physical Discovery partition.

## Neutral-versus-Directional model

For fitting the intensity model only, define:

```text
Z = 0 if reod = 0
Z = 1 if reod is -1 or +1
```

Reuse exactly the authoritative EXP_008 / CONF_001 condition B:

```text
B = [missing_ratio, q, I_recent_z]
W = {r41, ..., r52}
missing_ratio = NaNs among r0,...,r52 divided by 53
N_obs,W = number of observed W returns
q = 1[N_obs,W = 0]
I_recent = sqrt(sum(observed r_t^2) / N_obs,W), when N_obs,W > 0
```

Observed zeros remain observed and contribute zero. NaNs enter neither the
intensity numerator nor denominator. `I_recent` is undefined before
imputation when `q=1`; impute it with the Fit-only median of defined Fit
intensities. Fit `StandardScaler` only on imputed Fit intensity. Transform
Validation with those unchanged Fit parameters. `missing_ratio` and `q`
remain unscaled.

Fit exactly:

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

Define `p_D(x)` as the `predict_proba` column whose fitted class is `1`.
Do not recalibrate probabilities, tune the model, or tune thresholds.

## Fit-only directional priors and ternary probabilities

For each Fit subset, calculate:

```text
pi_minus = n(reod = -1) / [n(reod = -1) + n(reod = +1)]
pi_plus  = n(reod = +1) / [n(reod = -1) + n(reod = +1)]
```

Assert `pi_minus + pi_plus = 1` within numerical precision. Validation labels
must not affect either prior.

For every Validation row, calculate:

```text
P_minus = p_D * pi_minus
P_zero  = 1 - p_D
P_plus  = p_D * pi_plus
```

Assert rowwise that these sum to one within numerical precision.

## Frozen ternary decision rule

Let:

```text
pi_max = max(pi_minus, pi_plus)
p_D_star = 1 / (1 + pi_max)
```

Predict neutral unless:

```text
p_D * pi_max > 1 - p_D
```

equivalently, unless `p_D > p_D_star`. Comparisons are exact, with no
tolerance. At an exact neutral-versus-directional tie, predict `0`.

Once direction strictly wins:

- predict `-1` if `pi_minus > pi_plus`;
- predict `+1` if `pi_plus > pi_minus`;
- if `pi_minus == pi_plus` exactly, predict `-1` as a fixed arbitrary
  directional tie convention.

Do not use a generic array-order `argmax` if it conflicts with this rule.
This no-sign-alpha architecture normally predicts only the Fit-majority
directional class within a fold; that is an intended consequence, not a bug.

## Baseline and evaluation

For each fold, predict the Fit-majority ternary class for every Validation
row. If Fit class counts tie, apply deterministic class order `[-1, 0, +1]`.
The Fit-prior-only hard-class control is mathematically identical under this
same tie convention.

Primary internal Discovery metric: ordinary ternary hard-class accuracy.

```text
delta_accuracy = accuracy_hierarchical - accuracy_fit_majority_baseline
```

Report whether `delta_accuracy > 0` in each Discovery fold and its mean. This
is a descriptive candidate screen, not a stronger PASS/FAIL rule.

For candidate and baseline, report accuracy, Balanced Accuracy, Macro-F1,
ternary confusion matrix, and recall for `-1`, `0`, and `+1`.

For the candidate, additionally report Fit `pi_minus`, Fit `pi_plus`,
`p_D_star`, Validation routing fractions to neutral and directional, the
directional prediction fractions for `-1` and `+1`, whether both directional
classes are ever predicted within a fold, and N-v-D ROC-AUC of `p_D` as a
diagnostic only.

## Calibration diagnostics

Do not fit a calibration model. For raw `p_D` on each Validation subset,
report descriptive Brier score and log loss for `Z`, plus a fixed ten-bin
reliability table:

```text
[0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
```

For each bin, report row count, mean predicted `p_D`, and empirical
directional frequency. These diagnostics cannot trigger recalibration,
threshold changes, alternative bins, or any D004 repair.

## Integrity and interpretation boundaries

Assert physical Discovery-only input; the exact three folds; E_dev-only rows;
no protected-region or competition input; Fit-only intensity median/scaler;
exact B representation; fit-only priors; valid probability bounds and
rowwise ternary-probability sums; and Fit-only baseline selection.

If accuracy improves, the permitted statement is only:

> Under the frozen hierarchical no-sign-alpha decision rule, the
> recent-intensity N-v-D signal produced positive Discovery OOS ternary
> accuracy improvement over the Fit-majority baseline.

It does not establish conditional-sign alpha, official competition-score
improvement, calibrated probabilities, trading alpha, or competition-test
generalization. If accuracy does not improve, distinguish N-v-D predictive
discrimination from usefulness of this particular hard-class decision rule;
do not treat it as refuting EXP_008 or CONF_001.

## Implementation governance

Before any real execution: propose the minimal files and implementation plan,
verify the protected-region boundary, implement and pass synthetic/unit tests,
then obtain Human + Sol approval for exactly one real Discovery execution.
