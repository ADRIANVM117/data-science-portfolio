# EXP_007 — Sequential Chronology Signal

## Status

Completed.

## Scientific question and hypothesis

Among oracle-known directional and evaluable observations, does presenting the
same intraday return/availability information in its natural chronological
organization provide Joint OOS conditional-sign discrimination beyond a
pre-specified order-destroyed control, under a fixed small GRU?

The hypothesis is that conditional sign information, if present, may depend on
the chronological evolution of the intraday path — direction, local
persistence, recency, and state evolution — rather than only on the global or
tabular positional summaries tested in EXP_003 through EXP_006.

This experiment tests sequence-specific information. It does not test final
ternary challenge utility, a deployable hierarchical classifier, causality, or
the competition period.

## Population, target, and frozen validation

Reuse the effective population, operation order, frozen equity partition, and
four expanding temporal folds of EXP_003 through EXP_006.

```text
D = {reod in {-1, +1}}
S = 0  if reod = -1
S = 1  if reod = +1
N_obs = sum_j 1[r_j is observed]
effective population = D intersection {N_obs >= 1}
```

The normal EXP_000 split is always constructed first, followed by directional
conditioning and then the fixed evaluability exclusion. All-NaN rows are
therefore non-evaluable; they are not represented as zero-return paths.

Reuse EXP_000 exactly:

- frozen `E_dev=1463` / `E_holdout=366` equity partition, seed `20260908`;
- no holdout equity in `fit_f`;
- no purging or embargo;
- competition test fully excluded;
- Joint OOS as the primary deployment proxy; and
- Temporal OOS as a diagnostic only.

| Fold | Train days | OOS days |
|---|---|---|
| 1 | 0–302 | 303–352 |
| 2 | 0–352 | 353–402 |
| 3 | 0–402 | 403–452 |
| 4 | 0–452 | 453–502 |

```text
fit_f         = train_days_f x E_dev
Temporal OOS  = oos_days_f x E_dev
Joint OOS     = oos_days_f x E_holdout
```

The fit-derived majority-sign baseline is computed exclusively from evaluable
directional `fit_f`. Its class order is `[0, 1]`; a tie selects `S=0`.

## Sequential representation and fit-only preprocessing

For every retained row, the sequential input is exactly:

```text
x_t = [z_t, m_t], for t = 0, ..., 52
```

where:

- `m_t = 1` iff original `r_t` is missing, otherwise `0`;
- `z_t` is the positional return after fit-median imputation and fit-only
  standardization; and
- the tensor has shape `53 x 2` with `float32` values.

This preprocessing is compatible with EXP_005/EXP_006 and is applied within
each fold in this order:

1. Construct the 53 original binary masks before any imputation.
2. Fit one median for each positional return using evaluable directional
   `fit_f` only.
3. Require every fit median to be finite. A non-finite median is an
   integrity/evaluability failure; no zero fallback, feature dropping, or
   OOS-derived replacement is allowed.
4. Impute fit, Temporal OOS, and Joint OOS returns using only the fit medians.
5. Fit `StandardScaler` on the 53 imputed return columns of `fit_f` only.
6. Transform both OOS subsets without refitting.
7. Keep masks original, binary, and unscaled.

No IDs, `day`, `equity`, global missingness summaries, count features, return
summaries, engineered features, or native NaNs may enter the model.

## Paired chronological conditions

The two conditions use identical retained rows, labels, fit-only preprocessing,
model architecture, initialization, optimization budget, and minibatch order.

### Real chronological condition

```text
[(z_0, m_0), ..., (z_52, m_52)]
```

### Permuted condition

Use this single fixed global permutation for every row and every fold:

```text
PERMUTATION_SEED = 20260908

pi = [13, 44, 24, 4, 37, 23, 17, 50, 12, 6, 46, 7, 32, 5, 2, 33,
      21, 40, 41, 51, 35, 11, 18, 3, 16, 49, 42, 30, 31, 14, 19, 20,
      15, 48, 45, 47, 27, 39, 29, 9, 43, 10, 1, 25, 22, 28, 26, 8,
      34, 38, 36, 0, 52]

[(z_pi(0), m_pi(0)), ..., (z_pi(52), m_pi(52))]
```

Its frozen descriptive structural diagnostics are:

- length `53` and `53` unique elements;
- it is not the identity permutation;
- it has `1` fixed point;
- it retains `3` originally adjacent pairs (`|pi[t+1] - pi[t]| = 1`); and
- its longest contiguous original-order run, in either direction, has length
  `2`.

The permutation is generated before execution using
`numpy.random.default_rng(20260908).permutation(53)`. It must not depend on
rows, fold, labels, returns, OOS data, or any observed result. Returns and
masks are always permuted jointly as pairs. Per-row permutations and
return-only permutations with masks fixed are prohibited.

The fixed global permutation is invertible and therefore does not remove
row-level information in an information-theoretic sense. The experiment tests
whether presenting the same paired information in its natural chronological
organization is more useful to the frozen GRU's sequential inductive bias than
presenting it in one pre-specified artificial global order.

## Frozen model

Fit one independent model for Real and one for Permuted in each fold:

```text
GRU(
    input_size=2,
    hidden_size=16,
    num_layers=1,
    batch_first=True,
    dropout=0.0,
    bidirectional=False,
)
final hidden state -> Linear(16, 1)
```

There are exactly `977` trainable parameters. No attention, positional
embedding, convolution, residual connection, alternate pooling, auxiliary
loss, architecture search, or model-family comparison is permitted.

## Frozen deterministic training protocol

```text
Python:                  3.13.2
NumPy:                   2.2.3
PyTorch:                 2.12.0+cpu
Device:                  CPU only
Tensor dtype:            torch.float32
Deterministic algorithms: torch.use_deterministic_algorithms(True)
CPU threads:             1
Interop threads:         1

MODEL_SEED:              20260908
BATCH_ORDER_SEED:        20260909

Loss:                    BCEWithLogitsLoss(reduction="mean")
Optimizer:               Adam
Learning rate:           0.001
Betas:                   (0.9, 0.999)
Epsilon:                 1e-8
Weight decay:            0.0
AMSGrad:                 False
Batch size:              512
Epochs:                  10
Learning-rate scheduler: none
Gradient clipping:       none
Early stopping:          none
Probability threshold:   0.5
```

Before constructing each fold's initial model template, reset Python `random`,
NumPy, and PyTorch RNGs with `MODEL_SEED`. Copy that exact initial `state_dict`
to separately instantiated Real and Permuted models. Use separate optimizers
with identical frozen settings.

For each fold and epoch, deterministically generate a fit-row ordering from
`BATCH_ORDER_SEED`; apply the identical index sequence and minibatches to both
conditions. The two models therefore receive the same number of updates.

The 10-epoch budget is pre-specified and is not claimed to be optimal. Fit-only
training diagnostics are descriptive and may identify gross optimization
failure, but they cannot trigger changes to epochs, architecture, optimizer,
learning rate, or any other frozen choice after execution. Any such
modification requires a new experiment contract.

For each condition and each epoch, report fit-only descriptive diagnostics:

- sample-weighted mean BCE loss over training updates;
- ROC-AUC on the complete `fit_f` using the probability of `S=1`; and
- Balanced Accuracy on the complete `fit_f` at threshold `0.5`.

These diagnostics may not be used for early stopping, checkpoint selection,
hyperparameter changes, or any post-execution protocol change.

## Evaluation and primary comparison

For each condition, fold, and OOS subset, report:

- ROC-AUC using the probability for `S=1`;
- Balanced Accuracy;
- Accuracy;
- recall negative and recall positive;
- confusion matrix with label order `[0, 1]`;
- delta Accuracy versus the fold-specific majority baseline; and
- delta Balanced Accuracy versus the fold-specific majority baseline.

Both sign classes must be present in every evaluated OOS subset; otherwise the
experiment stops as an integrity/evaluability failure.

For each Joint OOS fold, the primary paired quantity is:

```text
delta_AUC_f = AUC(real_f) - AUC(permuted_f)
```

Means and sample standard deviations (`ddof=1`) across folds are descriptive
stability summaries, not IID inference.

### Primary — OOS chronological-organization evidence

EXP_007 has primary evidence that natural chronological organization is more
useful than the frozen artificial order to this GRU only if all hold:

1. Real Joint ROC-AUC is strictly greater than `0.50` in all four folds.
2. Mean Real Joint ROC-AUC is strictly greater than `0.50`.
3. `delta_AUC_f` is strictly greater than zero in all four Joint folds.
4. Mean Joint `delta_AUC` is strictly greater than zero.

No arbitrary minimum effect size is imposed.

### Secondary — fixed-threshold sign classification

At threshold `0.5`, Real has secondary fixed-threshold classification evidence
only if, in all four Joint OOS folds:

1. Balanced Accuracy is strictly greater than `0.50`.
2. Recall negative is strictly positive.
3. Recall positive is strictly positive.

Mean Joint delta Balanced Accuracy versus the fit-derived majority baseline
must also be strictly positive. This secondary result is independent of the
primary chronological comparison. Accuracy and delta Accuracy are diagnostics,
not pass/fail conditions.

Temporal OOS results are diagnostic only and cannot alter either verdict.

## Paired-control integrity assertions

Before and during execution, verify:

- exact reuse of EXP_000 folds and frozen equity partition;
- normal split before directional conditioning and evaluability;
- retained rows satisfy `D intersection {N_obs >= 1}`;
- all 53 fit medians are finite and learned only from evaluable directional
  `fit_f`;
- OOS imputation/scaling is transform-only;
- masks are constructed before imputation, remain binary and unscaled, and
  each representation is finite after preprocessing;
- Real and Permuted have exactly identical retained row IDs, labels, and row
  order in each corresponding subset;
- for every row and time index, the Permuted pair equals the Real pair at the
  corresponding `pi` index;
- the Permuted row preserves the multiset of its 53 Real `(z_t, m_t)` pairs;
- the exact permutation vector and its frozen diagnostics hold;
- Real and Permuted initial model tensors are identical before training;
- model and optimizer objects are independent despite identical initial state
  and optimizer settings;
- minibatch index order and update count match between Real and Permuted for
  every fold and epoch;
- both sign classes exist in fit and every evaluated OOS subset;
- probabilities use the `S=1` output, the threshold is exactly `0.5`, and
  confusion matrices use labels `[0, 1]`; and
- competition test data are not accessed.

Any integrity/evaluability failure stops the experiment without a silent
repair, substitute representation, or scientific PASS/FAIL verdict.

## Interpretation matrix and boundaries

| Outcome | Supported statement | Not supported |
|---|---|---|
| Real passes primary criterion and consistently exceeds Permuted | Under this frozen GRU and control, natural chronological organization provides Joint OOS evidence of incremental conditional-sign discrimination. | Causality, universal sequence-model superiority, deployable trading utility, or proof that chronology is the only source of information. |
| Both Real and Permuted exceed `0.50` similarly | The frozen GRU can use some retained paired information, but this design does not establish that the natural chronological order is the source. | Incremental evidence for chronology. |
| Both fail | This frozen sequential representation and GRU provide no OOS evidence under the stated criteria. | Absence of all sequential or conditional-sign information. |
| Permuted exceeds Real | Natural ordering was not useful to this model/control under the frozen protocol. | That chronological information is absent; the architecture may not exploit it. |

A primary pass is incremental OOS predictive-performance evidence only. It
does not establish causality, that missingness is liquidity, directional-sign
predictability in the original ternary task, final challenge improvement,
trading utility, competition-period generalization, or validity of a
hierarchical ternary architecture.

A failure does not establish that the full path has no sign information, that
another sequential representation or model cannot contain information, or that
the challenge is impossible.

EXP_003 through EXP_006 are historical context only. Their results must not
be used to tune EXP_007's architecture, epochs, optimization, permutation,
threshold, preprocessing, or success criteria.

## Executed results

All four frozen folds completed. All runtime integrity assertions passed:
fit medians were finite, Real and Permuted retained identical rows and labels,
their initial model states and minibatch schedules were paired, both OOS sign
classes were present, and the competition test was not accessed. The frozen
CPU runtime was `6869.885` seconds.

### Joint OOS

| Fold | Real ROC-AUC | Permuted ROC-AUC | Real minus Permuted |
|---|---:|---:|---:|
| 1 | 0.516058 | 0.491894 | +0.024164 |
| 2 | 0.490858 | 0.497683 | -0.006825 |
| 3 | 0.511161 | 0.503669 | +0.007493 |
| 4 | 0.473412 | 0.496952 | -0.023540 |

- Mean Real ROC-AUC: `0.497872`; sample standard deviation: `0.019620`.
- Mean Permuted ROC-AUC: `0.497549`; sample standard deviation: `0.004824`.
- Mean Real-minus-Permuted ROC-AUC: `+0.000323`; sample standard deviation:
  `0.020333`.

Frozen primary conditions:

1. Real Joint ROC-AUC > `0.50` in all four folds: **FAIL**.
2. Mean Real Joint ROC-AUC > `0.50`: **FAIL**.
3. Real-minus-Permuted ROC-AUC > `0` in all four folds: **FAIL**.
4. Mean Real-minus-Permuted ROC-AUC > `0`: **PASS**.

**PRIMARY OOS chronological-organization evidence: FAIL.**

### Secondary Joint OOS classification

- Mean Balanced Accuracy: `0.496540`.
- Mean Accuracy: `0.495585`.
- Mean recall negative: `0.641334`.
- Mean recall positive: `0.351746`.
- Mean delta Balanced Accuracy versus the fit-derived majority baseline:
  `-0.003460`.

**SECONDARY fixed-threshold sign classification: FAIL.**

### Temporal OOS diagnostic

- Mean Real ROC-AUC: `0.502139`.
- Mean Permuted ROC-AUC: `0.497690`.
- Mean Real-minus-Permuted ROC-AUC: `+0.004449`.

### Training diagnostics

For both Real and Permuted in all four folds, full-fit BCE decreased from
epoch 1 to epoch 10. Real full-fit ROC-AUC at epoch 10 ranged from
approximately `0.5295` to `0.5355`. These fit-only observations are
descriptive and did not alter any frozen choice.

## Interpretation boundary after execution

EXP_007 found no consistent Joint OOS evidence that presenting the 53
return/missingness pairs in their natural chronological organization provides
incremental conditional-sign discrimination to the frozen GRU relative to the
pre-specified artificial global ordering. Real-sequence performance varied
materially across folds and averaged approximately chance. The paired
chronology advantage was positive in two folds and negative in two folds, with
mean delta AUC approximately zero.

This result does not prove that chronological information is absent from
intraday returns; that sequence models cannot predict conditional sign; or
that GRU is an inappropriate architecture. It does not justify interpreting
fold variation as regime dependence, tuning the GRU, changing epochs, trying
additional seeds, or replacing the model with an LSTM or Transformer under
EXP_007. It also establishes nothing about final ternary competition
performance.
