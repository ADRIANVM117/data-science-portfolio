# D006 — Ternary Tail Intensity × Magnitude-Allocation Interaction

## Status

 Completed — Discovery only — both frozen candidate screens failed.

## Question and directional hypothesis

Within full ternary physical Discovery data, holding established recent
movement intensity and availability fixed, does allocation of recent squared
movement between negative and positive returns add tail information, and does
its contribution depend on intensity?

The hypothesis is frozen as **continuation of recent magnitude allocation**:
higher `O` should shift relative tail risk toward `Y=+1`; lower `O` toward
`Y=-1`, all else equal. This is not sign dominance or cumulative return.

## Scope, target, and folds

Use physical Discovery only: `days 0–352 × E_dev`, with full ternary target
`Y=reod ∈ {-1,0,+1}`. Do not condition on `Y != 0`, use D004 gating, or drop
rows for availability. Prohibit days `353–502`, `E_holdout`, competition data,
and raw full-training fallbacks.

| Fold | Fit | Validation |
| --- | --- | --- |
| 1 | 0–202 | 203–252 |
| 2 | 0–252 | 253–302 |
| 3 | 0–302 | 303–352 |

## Frozen information

For `W={r41,...,r52}` and observed finite values only:

```text
N_obs,W = count observed returns in W
q       = 1[N_obs,W = 0]
E_plus  = sum(r_t² for observed r_t > 0)
E_minus = sum(r_t² for observed r_t < 0)
E_total = E_plus + E_minus
I_recent = sqrt(E_total / N_obs,W), when N_obs,W > 0
```

Observed zeros count in `N_obs,W` but add zero energy; NaNs enter neither
counts nor energies. Non-NaN non-finite returns are integrity failures.

```text
O = (E_plus - E_minus) / E_total, when E_total > 0
O = 0, when E_total = 0
```

The deterministic zero convention covers all-zero and all-missing windows;
`I_recent` and `q` retain their established distinct semantics. `O` is the
sole new orientation: it is in `[-1,1]`, invariant to positive rescaling, and
changes sign under path sign reversal. No alternative orientation, window,
intensity estimator, path variable, sign count, cumulative return,
cross-sectional value, rank, or engineered feature is allowed.

## Preprocessing and nested representations

Construct all structural quantities target-free. Per fold, fit the median of
defined Fit `I_recent` only; impute undefined intensity with it; fit
`StandardScaler` on imputed Fit intensity only; transform Validation unchanged
to `I_recent_z`. Keep `missing_ratio`, `q`, and `O` unscaled. Build the sole
interaction after Fit-only scaling:

```text
I_recent_z_x_O = I_recent_z × O

C0 = [missing_ratio, q, I_recent_z]
C1 = [missing_ratio, q, I_recent_z, O]
C2 = [missing_ratio, q, I_recent_z, O, I_recent_z_x_O]
```

All C0/C1/C2 rows, targets, ordering, and learned preprocessing are identical.
With `O` included, the centered interaction differs from a rescaled raw
`I_recent × O` only by a multiple of `O`. `I_recent² × O` is forbidden: it is
signed squared energy per observed position.

## Model and coherent probabilities

Fit one multinomial Logistic Regression for each C0/C1/C2 matrix:

```python
LogisticRegression(
    solver="lbfgs", penalty="l2", C=1.0, fit_intercept=True,
    class_weight=None, max_iter=1000, tol=1e-4, random_state=20260908,
)
```

Omit deprecated `multi_class`; with three classes and `lbfgs`, use the
installed supported multinomial behavior. Require `classes_ == [-1,0,1]` and
probabilities that are finite, non-negative, and sum rowwise to one. Separate
tail binary models are forbidden because they need not be coherent.

## Metrics and Discovery screens

For every representation and Validation fold:

```text
tail_auc_minus = AUC(1[Y=-1], p_minus)
tail_auc_plus  = AUC(1[Y=+1], p_plus)
macro_tail_auc = (tail_auc_minus + tail_auc_plus) / 2
```

Report `C1-C0` and `C2-C1` macro-tail-AUC deltas. Multiclass log loss, argmax
accuracy, Macro-F1, class recalls, and confusion matrices are diagnostics;
accuracy is internal and the official metric remains unresolved.

C1’s allocation screen requires C1 macro-tail AUC > `0.50` in all three
folds, C1-C0 > `0` in all three, and positive mean C1-C0. C2’s interaction
screen requires C2 macro-tail AUC > `0.50` in all three, C2-C1 > `0` in all
three, and positive mean C2-C1. C2-C0 is descriptive only. These are
Discovery screens, not confirmation or significance tests.

## Frozen probability orientation diagnostic

For every fitted C1/C2 model and Validation row, keep all non-orientation
inputs fixed and set:

```text
O' = -O
I_recent_z_x_O' = -I_recent_z_x_O
T_i = [p_plus(X_i)-p_minus(X_i)]
      - [p_plus(X_i with O')-p_minus(X_i with O')]
```

The fold diagnostic is `mean(T_i)`. C1 reverses only `O`; C2 reverses `O` and
its interaction. This is one fixed model-prediction diagnostic, not a raw
coefficient, target, threshold, model, or feature search.

- strictly positive in all three folds: continuation-oriented;
- strictly negative in all three: opposite-oriented;
- otherwise: no stable orientation.

If a screen passes with continuation orientation, evidence is compatible with
the frozen hypothesis. If it passes with opposite orientation, continuation
is falsified; it is not a reversal success without a new contract. If
orientation is unstable, no stable mechanism support exists. If a screen
fails, its D006 hypothesis closes regardless of diagnostics or coefficients.

## Falsifiability and governance

No C1/C2 gain means no evidence allocation helps; C1-only gain indicates
allocation without interaction; C2-only gain is interaction-only and cautious;
both gains are compatible with allocation and moderation; unstable folds fail.
A negative result rejects only this frozen specification, not all sign
information. D006 follows EXP_008, D004, and D005 and remains Discovery-only.
No alternate transform, window, threshold, learner, calibration, coefficient
inspection, or post-result redesign is allowed. Synthetic/integrity and
regression tests must pass before one separately authorized execution.

## Executed result

One authorized execution completed across the three frozen physical Discovery
folds. All integrity assertions passed. No post-result adaptation, alternate
`O`, interaction, window, model, threshold, preprocessing, or metric was
evaluated. Days `353–502`, `E_holdout`, and competition-test data were not
accessed.

### C1 directional-allocation screen — FAIL

Mean MacroTailAUC was `0.606022` for C0 and `0.603691` for C1. The frozen
C1-minus-C0 MacroTailAUC deltas were `-0.003255`, `-0.005768`, and
`+0.002029` in folds 1–3, respectively; their mean was `-0.002331`.
Although C1 MacroTailAUC exceeded `0.50` in every fold, its incremental
criterion was not met in every fold and its mean increment was not positive.

### C2 intensity-interaction screen — FAIL

Mean C2 MacroTailAUC was `0.603488`. The frozen C2-minus-C1 MacroTailAUC
deltas were `-0.001131`, `-0.000170`, and `+0.000694`, with mean
`-0.000202`. Although C2 MacroTailAUC exceeded `0.50` in every fold, its
incremental criterion was not met in every fold and its mean increment was
not positive.

### Orientation diagnostic

The pre-specified probability-based diagnostic was continuation-oriented in
fold 1, opposite-oriented in fold 2, and continuation-oriented in fold 3;
the overall classification was `unstable`. Fold 2 is not evidence for a
reversal hypothesis.

### Bounded conclusion

Under the frozen recent-window intensity representation, frozen directional
squared-magnitude-allocation representation, and simple multinomial
interaction model, D006 found no consistent Discovery OOS evidence that
directional magnitude allocation adds to intensity or that its effect depends
on intensity. This does not state that direction is unpredictable.

C0 remained descriptively informative for tail risk: its MacroTailAUCs were
`0.611054`, `0.604537`, and `0.602475` (mean `0.606022`), with both negative-
and positive-tail AUCs above `0.50` in every fold. This is descriptive evidence
for established intensity/availability information about tail or large-movement
risk, not directional alpha. C0's hard argmax predictions had `+1` recall of
zero in all three folds; this is a property of this frozen multinomial model
and argmax decision rule, not evidence that positive-tail outcomes are
inherently unpredictable.
