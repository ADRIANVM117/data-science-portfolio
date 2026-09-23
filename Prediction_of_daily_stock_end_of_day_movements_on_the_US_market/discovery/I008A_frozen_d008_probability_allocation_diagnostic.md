# I008A — Frozen D008 Probability-Allocation Diagnostic

## Status

Frozen — implementation/testing authorized; real execution requires separate authorization.

## Question

How does the already-frozen D008 candidate `P` change ternary predictive probabilities relative to control `C`, and where does the already-established D008 log-loss improvement arise? This is a descriptive interpretation of D008, not D009 and not a new predictive experiment.

## Frozen reconstruction

Reconstruct the six D008 models (three Discovery folds × `C`/`P`) exactly:

```text
C = [m0,...,m52, I_recent_z]
P = [m0,...,m52, I_recent_z, r0,...,r52]
```

Use only physical Discovery (`days 0–352 × E_dev`), the frozen D008 folds, exact fit-only intensity preprocessing, XGBoost 3.4.1 specification, seed, class encoding `[0,1,2]` mapped to `[-1,0,+1]`, and D008 probability extraction. No tuning, recalibration, alternate seed, class weighting, or new features are permitted.

## Reproduction gate

Before any diagnostic, every fold and arm must satisfy:

```text
abs(reconstructed_log_loss - persisted_D008_log_loss) <= 1e-8
```

with `rtol=0`. The gate also requires expected Validation count, identical C/P IDs and ordering, persisted ordered-ID fingerprint, frozen schemas and shared C block, class mapping, aligned complete labels, fit-only intensity parameters matching persisted D008 values, and finite/nonnegative `(n_validation,3)` probabilities. Raw probability row sums must satisfy `rtol=0`, `atol=max(1e-12, 2 * eps(raw_probability_dtype))`.

Any gate failure stops execution before interpretation artifacts are produced. The tolerance cannot be loosened during execution.

## Probability definitions

For every Validation row and arm `X ∈ {C,P}`:

```text
p_D^X = p_X(-1) + p_X(+1)
q_plus^X = p_X(+1) / p_D^X, only if p_D^X > 0
delta_p_D = p_D^P - p_D^C
delta_q_plus = q_plus^P - q_plus^C, only if both allocations are defined
```

`p_D` is calculated directly from tail columns, never as `1-p(0)`. Undefined allocation receives no fallback and is counted explicitly.

For realized class `y_i`, use the same per-row clipping semantics as frozen `sklearn.metrics.log_loss`:

```text
g_i = loss_i(C) - loss_i(P)
```

Conceptually this is `log(p_P(y_i)/p_C(y_i))` after only the library-required numerical clipping. Raw probabilities are never renormalized or diagnostic-clipped. Per fold, `mean(g_i)` must equal both persisted and reconstructed `log_loss(C)-log_loss(P)` within pre-specified `atol=1e-10`, `rtol=0`.

## Frozen outputs

Level 1, per fold: mean/median and mean/median absolute `delta_p_D`; plus `n_defined` and the same four summaries for `delta_q_plus`.

Level 2, per fold and true class `[-1,0,+1]`: `n_rows`, mean/median `g_i`, fraction `g_i>0`, mean `delta_p_D`, `n_delta_q_defined`, and mean defined `delta_q_plus`.

Level 3, per fold: exhaustive C-argmax → P-argmax transitions in long format, both unconditional and conditioned on each true class. Argmax is NumPy’s first-index tie convention in ordered labels `[-1,0,+1]`.

## Boundaries

I008A may describe frozen-model probability movement, directional allocation, realized-class gain/loss, and hard transitions only. It cannot establish alpha, causality, economic mechanism, individual raw-feature causation, momentum, reversal, SHAP validation, competition improvement, or independent confirmation.

## Artifacts and lifecycle

Exactly these artifacts are allowed:

- `I008A_reproduction_gate.csv`
- `I008A_probability_movement.csv`
- `I008A_realized_class_gain.csv`
- `I008A_prediction_transitions.csv`
- `I008A_metadata.json`

States are `fresh` (none), `incomplete` (strict subset), and `complete` (all). The runner refuses incomplete and complete states. It never cleans artifacts automatically. Probabilities and diagnostics remain in memory until all six reproduction gates pass.

## Completed execution

Status: `Completed — descriptive interpretation only`.

The sole authorized command, `python discovery/run_i008a_frozen_d008_probability_allocation_diagnostic.py`, completed in `505.6745423999382` seconds. All three folds, six C/P reconstructions, six gates, and five artifacts completed. No warning or error was emitted or persisted. The long-running execution interface did not expose a separate numeric exit code. D008 hashes and protected boundaries were unchanged.

Every reconstruction passed at `atol=1e-8`, `rtol=0`. Encoded classes `[0,1,2]`, labels `[-1,0,+1]`, float32 dtype-aware row-sum checks, schemas, shared C block, ID/order identity, and fit-only preprocessing all passed. The runtime maximum row-sum deviation was not persisted separately. Absolute reconstruction-loss differences for Fold 1 C/P, Fold 2 C/P, Fold 3 C/P were `0.0`, `1.1102230246251565e-16`, `2.220446049250313e-16`, `0.0`, `0.0`, and `0.0`.

The conservation identity passed at `atol=1e-10`. Mean `g_i` was `0.003574897353754461`, `0.0007623862799180481`, and `0.002976677733348173` in folds 1–3; maximum discrepancies from D008 C-minus-P loss deltas were `8.07e-17`, `1.55e-16`, and `4.70e-16`.

### Level 1

| Fold | mean ΔpD | median ΔpD | mean |ΔpD| | median |ΔpD| | n Δq+ | mean Δq+ | median Δq+ | mean |Δq+| | median |Δq+| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.0008472873 | -0.0012768507 | 0.0313142878 | 0.0263352543 | 67031 | +0.0005453382 | -0.0004149005 | 0.0367795688 | 0.0294492619 |
| 2 | -0.0009991550 | -0.0016109049 | 0.0313057318 | 0.0263290703 | 66989 | -0.0006271334 | -0.0010733286 | 0.0345438806 | 0.0276943870 |
| 3 | -0.0012976261 | -0.0021584332 | 0.0308857285 | 0.0259556174 | 66899 | +0.0009749742 | +0.0001481274 | 0.0300162443 | 0.0241585149 |

Signed average changes were small, while absolute row-level changes were materially larger. Mean `delta_q_plus` signs were positive, negative, positive across folds; I008A therefore does not support a simple global directional-allocation shift. These probability coordinates are not directly commensurate causal quantities.

### Level 2

| Fold | Y | mean g | median g | frac g>0 | mean ΔpD | mean Δq+ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -1 | -0.0021048946 | -0.0053427558 | 0.4822805977 | +0.0012357877 | -0.0008043401 |
| 1 | 0 | +0.0095425683 | +0.0063256926 | 0.5296850617 | -0.0039389919 | -0.0000638246 |
| 1 | +1 | +0.0004724739 | -0.0076129544 | 0.4776803311 | +0.0015450844 | +0.0026244493 |
| 2 | -1 | -0.0052944938 | -0.0095164728 | 0.4658031088 | +0.0012713426 | +0.0008329155 |
| 2 | 0 | +0.0097018878 | +0.0105202399 | 0.5510411745 | -0.0050008077 | -0.0009899632 |
| 2 | +1 | -0.0056733499 | -0.0107909234 | 0.4637101195 | +0.0023947061 | -0.0017790273 |
| 3 | -1 | -0.0058041156 | -0.0098115066 | 0.4604818489 | -0.0004358037 | +0.0005490548 |
| 3 | 0 | +0.0098942572 | +0.0075916769 | 0.5365250452 | -0.0031451687 | -0.0002267208 |
| 3 | +1 | +0.0017507104 | -0.0043361472 | 0.4848103625 | +0.0002977884 | +0.0028559284 |

The clearest stable class-specific component occurred for true neutral rows: mean gain was positive and mean directional-mass change negative in every fold. True `-1` mean gain was negative in every fold; true `+1` gain was small and sign-inconsistent. The pre-I008A simple directional-allocation explanation did not receive clean support.

### Level 3 and limits

Full unconditional and true-class-conditioned transitions are in `I008A_prediction_transitions.csv`. Counts for `C=-1 -> P=+1` were `7806`, `8329`, and `8324`. They do not demonstrate consistently improved `+1` probability quality: true-`+1` gains were positive, negative, positive and small.

I008A is descriptive frozen-model interpretation only. It does not establish sign alpha, neutral prediction, causality, an economic mechanism, raw-position causation, momentum, reversal, SHAP validation, or competition-test performance. No next experiment is authorized; Human + Sol program-level review is required.
