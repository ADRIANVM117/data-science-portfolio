# D003 — Incremental Cross-Sectional Relative Path for Conditional Sign

## Status

Completed — `DESCRIPTIVE_CANDIDATE_SCREEN_NOT_MET`.

## Execution record (valid reexecution)

The first authorized execution stopped before producing any valid result because
the D003 evaluation helper used an unsupported keyword argument. It is recorded
as `TECHNICAL_EXECUTION_FAILURE_BEFORE_VALID_RESULT`, not as scientific
evidence. The call site was repaired to the established supported interface,
covered by an end-to-end synthetic test, and the historical regression suites
passed before one authorized valid reexecution.

The valid reexecution used only the physical Discovery partition
`days 0–352 × E_dev`; it did not access days `353–502`, `E_holdout`, or the
competition test. All frozen runtime integrity assertions passed.

### Joint Discovery validation results

| Fold | A own-path AUC | B own + relative-path AUC | B − A AUC |
| --- | ---: | ---: | ---: |
| 1 | 0.531467 | 0.543880 | +0.012414 |
| 2 | 0.485030 | 0.421540 | −0.063490 |
| 3 | 0.517804 | 0.497722 | −0.020082 |
| Mean | 0.511434 | 0.487714 | −0.023719 |
| Sample standard deviation | 0.023865 | 0.061781 | 0.038082 |

The frozen descriptive candidate screen required positive B-minus-A AUC in
all three folds and a positive mean delta. Both conditions were not met;
therefore the screen is `DESCRIPTIVE_CANDIDATE_SCREEN_NOT_MET`. This is a
Discovery result only and does not establish that cross-sectional relative
paths contain no information under another representation or learner.

For B, mean validation metrics were: ROC-AUC `0.487714`, balanced accuracy
`0.499071`, accuracy `0.492104`, negative recall `0.556386`, and positive
recall `0.441755`.

## Discovery status and information-set assumption

D003 is Discovery only, using the physical target-blind boundary
`days 0--352 x E_dev`. It is not Confirmation and cannot establish an OOS
deployment claim. The competition test, `E_holdout`, and days 353--502 are
excluded.

For this challenge-specific research design, freeze the assumption that all
input rows `r0,...,r52` from the same day are available as a contemporaneous
cross-section when predictions for that day are constructed. This is not a
claim about live-trading feed availability.

## Scientific question

Among oracle-known directional observations in Discovery, does adding an
equity's contemporaneous cross-sectional relative return path improve
conditional-sign discrimination beyond that equity's own positional path under
the same fixed linear-additive learner?

```text
D = {reod in {-1, +1}}
S = 0 if reod = -1
S = 1 if reod = +1
```

## Target-free common and relative paths

Before target conditioning, for each same-day position:

```text
m_d,t = median_j(r_d,j,t) over all observed Discovery E_dev rows on day d
u_d,i,t = r_d,i,t - m_d,t, when r_d,i,t is observed
```

The all-row median includes equity `i` when its return is observed. The
target-free audit found an all-row versus leave-one-out median mean absolute
difference of approximately `0.009038` bps; leave-one-out is prohibited in
D003.

If own `r_d,i,t` is missing, `u_d,i,t` is missing. The identity
`isna(u_t) == isna(r_t)` must be asserted before target conditioning. An
observed residual of zero is a valid observed value, never a missing-value
encoding.

Use all 53 positions. Cross-sectional ranks, percentiles, MAD, dispersion,
SHAP-selected positions, recency weights, adjacency, H_peak, and all other
cross-sectional features are prohibited.

## Population and representations

Construct normal chronological subsets, then apply the scientifically
identical EXP_005 directional/evaluability rule: retain `D` rows with at least
one observed own return. The relative path has the identical availability mask
by construction; no second relative mask is included.

```text
A = [53 fit-imputed/scaled own returns, 53 original own masks]
B = [A, 53 separately fit-imputed/scaled relative returns]
```

Thus A has 106 predictors, B has 159, and B differs only by the 53 relative
numeric positions. IDs, day, equity, raw returns, raw relatives, masks beyond
the own mask, ranks, common components, and all other variables are excluded.

## Fit-only preprocessing and learner

Use two independent EXP_005-compatible `PositionalPathTransformer` objects:

1. fit own positional medians and a StandardScaler on evaluable directional
   Fit own returns only;
2. fit relative positional medians and a separate StandardScaler on evaluable
   directional Fit relative returns only.

Both OOS subsets are transform-only. Every positional median must be finite;
there is no zero fallback, feature dropping, or OOS-derived parameter.
Original own masks are binary and unscaled.

Fit independent A and B models using exactly:

```python
LogisticRegression(
    solver="lbfgs", penalty="l2", C=1.0, fit_intercept=True,
    class_weight=None, max_iter=1000, tol=1e-4, random_state=20260908,
)
```

No tuning, class weighting, nonlinear learner, XGBoost, HGB, GRU, or model
alternative is permitted.

## Frozen Discovery chronology

| Fold | Fit days | Validation days |
|---:|---|---|
| 1 | 0--202 | 203--252 |
| 2 | 0--252 | 253--302 |
| 3 | 0--302 | 303--352 |

All rows remain inside the physical Discovery `E_dev` boundary.

## Evaluation and descriptive candidate rule

For each Validation fold report `AUC(A)`, `AUC(B)`, and
`delta_auc = AUC(B) - AUC(A)`. Report B Accuracy, Balanced Accuracy, negative
recall, and positive recall as descriptive diagnostics.

D003 is not confirmatory. A material candidate for Human + Sol review
requires positive delta AUC in all three Discovery folds and a separately
human-reviewed magnitude that is not merely numerically trivial. No numerical
materiality threshold or automatic PASS/FAIL verdict is defined.

If results are unstable, record them and stop. Do not search alternative
ranks, windows, transformations, models, or regimes inside D003.

## Integrity requirements

Assert physical Discovery files only; days at most 352; `E_dev` only; no
competition test; same-day target-free medians; exact residual/missingness
identity; 53 positions; separate Fit-only own/relative transformers; aligned
A/B rows and targets; B's sole addition; exact learner parameters; frozen
folds; and both sign classes in Fit/Validation.

## Interpretation limits

If B exceeds A, the permitted Discovery statement is only:

> Under the fixed exploratory linear-additive learner, contemporaneous
> cross-sectional relative returns added Discovery OOS conditional-sign
> discrimination beyond the own positional path.

This does not establish cross-sectional momentum or reversal, market-neutral
alpha, causal peer effects, live-feed validity, or confirmation. If B is not
materially better than A, the conclusion is limited to this relative-return
representation and learner.
