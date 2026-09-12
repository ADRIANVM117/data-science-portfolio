# EXP_000 — Validation Baseline


## Status

Completed.

The protocol was specified before execution. The majority-class baseline
has now been executed using training data only. The competition test
was not accessed.

## Research question

Can we obtain a reasonable out-of-sample estimate of performance when
deployment requires generalizing simultaneously to future days and completely
new equities?

## Known facts and assumptions

### Dataset facts

- Training contains 503 `day` identifiers (`0`–`502`) and 1,829 equities.
- The competition test belongs to a distinct period and contains different
  equities.
- The competition test is completely excluded from model selection, feature
  selection, hyperparameter selection, preprocessing decisions, and hypothesis
  evaluation.

### Protocol assumption

- According to external evidence in the official challenge material, `day`
  represents chronological order.

## Fixed equity partition

Create one reproducible partition of training equity IDs and freeze it for
EXP_000 and subsequent experiments using this protocol.

- `E_dev`: approximately 80% of equity IDs.
- `E_holdout`: approximately 20% of equity IDs.
- Random seed: `20260908`.
- The selection must not use returns, `reod`, missingness, future coverage, or
  model results.

## Temporal folds

| Fold | Train days | OOS days |
|---|---|---|
| 1 | 0–302 | 303–352 |
| 2 | 0–352 | 353–402 |
| 3 | 0–402 | 403–452 |
| 4 | 0–452 | 453–502 |

For each fold `f`:

```text
fit_f         = train_days_f × E_dev
Temporal OOS  = oos_days_f × E_dev
Joint OOS     = oos_days_f × E_holdout
```

`Equity OOS` is not part of EXP_000.

## Information boundaries

- `ID` is a key only.
- `day` is a partition key only.
- `equity` is a partition key only.
- Initial predictive inputs are `r0` through `r52`.
- Any preprocessing that learns parameters must be fitted exclusively on
  `fit_f`, then applied without refitting to both OOS subsets.
- EXP_000 uses no purging or embargo because target horizons are contained
  within their respective sessions and do not overlap across days.

## Evaluation

- **Primary deployment proxy:** Joint OOS.
- **Diagnostic:** Temporal OOS.
- **Primary metric:** accuracy.
- **Diagnostics:** Macro-F1, Balanced Accuracy, and confusion matrix.
- Report every fold separately for both OOS subsets.
- Also report the mean and standard deviation across folds; do not hide
  fold-level results behind an aggregate.

## Baseline

For each fold, construct a majority-class baseline using the target
distribution in `fit_f` only. Do not use the global target distribution.

## Integrity checks required before reporting results

The eventual implementation must verify and report:

- temporal separation between `fit_f` and both OOS subsets;
- disjointness of `E_dev` and `E_holdout`;
- complete absence of `E_holdout` equities from `fit_f`;
- complete exclusion of the competition test;
- row counts and class distributions for `fit_f`, Temporal OOS, and Joint OOS
  in every fold.

## Limitation

The 503 labelled training days cannot exactly reproduce deployment against an
approximately 505-day period and a wholly new equity universe. Joint OOS is an
internal proxy designed to stress temporal and cross-sectional generalization
simultaneously.

## Results

EXP_000 completed successfully across all four folds.

Majority-class baseline:
- Temporal OOS accuracy: 0.4147 ± 0.0092
- Joint OOS accuracy: 0.4124 ± 0.0117

The majority class selected from `fit_f` was class `0` in all four folds.

These results establish the OOS baseline under the frozen validation
protocol. They do not constitute evidence of predictive signal.

Detailed fold-level results are stored in:
`experiments/results/`.