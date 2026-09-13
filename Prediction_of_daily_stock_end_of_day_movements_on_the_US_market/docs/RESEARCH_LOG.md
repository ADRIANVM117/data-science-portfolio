# Research Log

## 2026-09-08 — Dataset geometry audit

Observed:
- 843,299 training observations
- 503 chronological days
- 1,829 equities
- panel density 91.66%
- 82.72% equities present all 503 days

Implication:
Rows cannot be treated as independent random observations.

Decision:
Random train/validation split rejected.


## 2026-09-09 — Validation design

Problem:
Competition deployment contains both new period and new equities.

Considered:
- temporal split only
- equity holdout only
- temporal + equity OOS
- purged/embargo CV

Decision:
Use expanding temporal validation + frozen equity holdout.

Reason:
Joint OOS approximates both dimensions of deployment shift.

Rejected:
Equity OOS on training days because labels from other equities
of the same session are already known to the fitted model.


## 2026-09-10 — EXP_000

Result:
Majority baseline Joint OOS = 41.24% ± 1.17%.

Interpretation:
Establishes OOS floor; does not demonstrate predictive signal.

Next question:
Does missingness contain generalizable predictive information?


## 2026-09-10 â€” EXP_001 Missingness Signal

Result:
- M1 reproduced the EXP_000 majority baseline.
- M2 produced a minimal mean Joint OOS Accuracy improvement (`+0.000312`) but
  failed the pre-specified directional criterion: recall for `+1` was zero in
  every Joint OOS fold.
- M3 increased the Joint OOS Macro-F1 and Balanced Accuracy diagnostics, but
  reduced Accuracy in all four Joint OOS folds relative to EXP_000.
- M3 behaved differently across OOS subsets: its mean Accuracy delta was
  positive on Temporal OOS and negative on Joint OOS.

Decision:
No M1, M2, or M3 representation qualified as promising under the
pre-specified Joint OOS rule.

Scope:
This does not establish that missingness has no predictive information in
general. The conclusion is limited to the three representations, fixed
logistic-regression model, and decision criterion tested in EXP_001.

Post-EXP_001 hypothesis (not an EXP_001 conclusion):
Missingness may contain more information about neutral versus directional
outcomes than about the directional sign, `-1` versus `+1`.


## 2026-09-10 — Post-EXP_001 descriptive missingness investigation

Scope:
Training-data descriptive audits only. No competition test, model training,
validation-protocol change, or alteration of EXP_001 was performed.

Observed:
- The exact `n_missing` audit showed a broad descriptive association between
  higher missingness and a larger neutral-class proportion, with an extreme
  anomaly at `n_missing=11`.
- A target-free structural audit traced that anomaly to one dominant exact
  mask: `r0...r41` observed and `r42...r52` NaN. It contains 7,779 rows,
  or 61.05% of all `n_missing=11` rows.
- Target-free day/equity diagnostics showed this mask in exactly six of 503
  sessions (`112, 134, 229, 314, 438, 469`), affecting 1,642 of 1,829
  equities. It is not persistent by equity; within each of those sessions it
  is the modal mask and covers a large fraction of the cross-section.
- The generalized session-level audit found that these are exactly the six
  sessions whose modal mask contains missingness. The modal mask is fully
  observed in the other 497 sessions.
- Excluding those six sessions using that target-free structural definition
  changed `P(reod=0 | n_missing=11)` from 76.52% to 39.70%. The broader
  descriptive relationship between high missingness and a greater neutral
  proportion remained outside those sessions.

Interpretation:
Missingness should not be interpreted simply as an equity-level liquidity
proxy. Training contains at least one strong session-level missingness
mechanism, while a broader residual association with neutral-versus-
directional target composition remains descriptively visible.

Boundary:
This does not establish causality or OOS predictive information and does not
change EXP_001.

Post-EXP_001 hypothesis, not yet tested:
Missingness available by 14:00 may contain more OOS information about whether
the subsequent return is neutral versus directional than about the sign of a
directional move.


## 2026-09-10 — EXP_002 Missingness: Neutral vs Directional

Protocol:
The pre-specified EXP_002 binary target was `Z=0` for neutral `reod=0` and
`Z=1` for directional `reod in {-1,+1}`. It used only `missing_ratio` under
the frozen EXP_000 folds and equity partition. The six structurally unusual
sessions remained naturally in their frozen subsets and were not filtered.

Integrity:
All real-data integrity checks passed. The competition test was not accessed.
Both binary classes were present in every evaluated OOS subset. The fit-only
binary majority class was directional in all four folds.

Joint OOS result:
- ROC-AUC by fold: `0.604098`, `0.584693`, `0.595994`, `0.598582`.
- Mean ROC-AUC: `0.595842`.
- Mean Balanced Accuracy: `0.578116`.
- Mean Accuracy: `0.639120`.
- Mean recall neutral: `0.229846`.
- Mean recall directional: `0.926386`.
- Mean delta Balanced Accuracy versus the majority baseline: `+0.078116`.
- Mean delta Accuracy versus the majority baseline: `+0.051482`.

Pre-specified verdicts:
- Primary OOS discrimination evidence: PASS.
- Secondary fixed-threshold classification evidence: PASS.

Temporal OOS diagnostic:
Mean ROC-AUC `0.600211`, Balanced Accuracy `0.586526`, and Accuracy
`0.641073`.

Interpretation boundary:
EXP_002 provides evidence that `missing_ratio` contains OOS discriminatory
information for neutral versus directional outcomes under the frozen protocol.
It does not establish causality, that missingness is a liquidity measure,
directional-sign predictability, improvement on the original ternary
challenge, or validity of a hierarchical ternary architecture. The fixed
threshold classifier is asymmetric and should not be described as strong
neutral detection: mean Joint OOS recall was `0.229846` for neutral and
`0.926386` for directional.


## 2026-09-10 — EXP_003 Conditional Directional Sign

Protocol:
EXP_003 evaluated the oracle-conditional directional population using only
the one-dimensional cumulative observed return `R_obs`, after the frozen
EXP_000 split, directional conditioning, and the fixed `n_obs >= 1`
evaluability rule. The competition test was not accessed.

Integrity:
All real-data integrity checks passed. Both sign classes were present in every
retained fit and evaluated OOS subset. All-NaN exclusions were negligible
relative to subset sizes, ranging from zero to five directional rows per
subset. The fit-derived majority sign was negative in all four folds.

Joint OOS result:
- ROC-AUC by fold: `0.493825`, `0.491287`, `0.494237`, `0.499332`.
- Mean ROC-AUC: `0.494671` with sample standard deviation `0.003371`.
- Mean Balanced Accuracy: `0.499977`.
- Mean Accuracy: `0.504064`.
- Mean recall negative: `0.999953`; mean recall positive: `0.000000`.
- Mean delta Balanced Accuracy: `-0.000023`; mean delta Accuracy:
  `-0.000025`.

Pre-specified verdicts:
- PRIMARY OOS sign-discrimination: FAIL.
- SECONDARY fixed-threshold sign-classification: FAIL.

Coefficient diagnostic:
Standardized `R_obs` beta was positive in every fold: `+0.003504`,
`+0.003135`, `+0.002918`, and `+0.002774`. Beta-sign consistency was `True`.
These coefficient signs are not evidence of predictive momentum because the
pre-specified OOS discrimination criterion failed.

Temporal OOS diagnostic:
Mean ROC-AUC `0.499747`, Balanced Accuracy `0.500015`, and Accuracy
`0.506131`.

Interpretation boundary:
EXP_003 found no OOS evidence that cumulative observed return alone
discriminates subsequent sign among oracle-known directional outcomes under
the frozen validation protocol. This does not establish that the full
intraday path `r0...r52` has no sign information, nor that momentum or
reversal is impossible in other path representations, horizons, or
conditional structures.


## 2026-09-11 â€” EXP_004 Conditional Directional Persistence

Protocol:
EXP_004 evaluated the oracle-conditional directional population using only
the frozen directional-persistence summary
`P = (N_plus - N_minus) / N_obs`, after the frozen EXP_000 split,
directional conditioning, and the fixed `N_obs >= 1` evaluability rule. The
competition test was not accessed.

Integrity:
All real-data integrity assertions passed. All-NaN exclusions were negligible
relative to subset sizes. The fit-derived majority sign was negative in all
four folds.

Joint OOS result:
- ROC-AUC by fold: `0.499966`, `0.509310`, `0.495915`, `0.492626`.
- Mean ROC-AUC: `0.499454` with sample standard deviation `0.007224`.
- Mean Balanced Accuracy: `0.498369`.
- Mean Accuracy: `0.501954`.
- Mean recall negative: `0.973249`; mean recall positive: `0.023488`.
- Mean delta Balanced Accuracy: `-0.001631`; mean delta Accuracy:
  `-0.002135`.

Pre-specified verdicts:
- PRIMARY OOS sign-discrimination: FAIL.
- SECONDARY fixed-threshold sign-classification: FAIL.

Coefficient diagnostic:
`beta(P)` was negative in all four folds and coefficient-sign consistency was
`True`. The negative coefficient signs are not evidence of predictive reversal
because the frozen OOS discrimination criterion failed.

Temporal OOS diagnostic:
Mean ROC-AUC `0.496793`, Balanced Accuracy `0.498580`, and Accuracy
`0.504156`.

Interpretation boundary:
EXP_004 found no OOS evidence that the frozen one-dimensional
directional-persistence representation
`P = (N_plus - N_minus) / N_obs` discriminates subsequent positive versus
negative outcomes among oracle-known directional observations. This does not
establish that intraday path ordering, timing, run structure, recent-state
information, or the complete `r0...r52` path contains no sign information.

Combined research observation:
EXP_003 and EXP_004 tested two different simple global path summaries —
cumulative observed return and directional sign dominance — and neither
produced OOS conditional sign discrimination under the frozen protocol. Both
summaries discard most or all temporal ordering information.


## 2026-09-11 — EXP_005 Conditional Positional Path Signal

Protocol:
EXP_005 used the oracle-directional, evaluable population and compared the
53-position availability control `A_mask` with `B_path_mask`: 53 fit-median-
imputed and fit-StandardScaler-transformed returns plus the original 53
unscaled masks. It reused the frozen EXP_000 protocol. The competition test
was not accessed.

Integrity:
All frozen integrity assertions passed. The fit-derived majority sign was
negative in all four folds. All 53 fit medians were finite in every fold;
median imputation and StandardScaler were fit-only, OOS was transform-only,
masks remained binary and unscaled, and A_mask/B_path_mask retained identical
rows.

Joint OOS result:
- A_mask ROC-AUC by fold: `0.497363`, `0.506395`, `0.505435`, `0.498217`.
  Mean `0.501852`; sample standard deviation `0.004720`.
- B_path_mask ROC-AUC by fold: `0.518273`, `0.471215`, `0.507501`,
  `0.488085`. Mean `0.496269`; sample standard deviation `0.020857`.
- B_path_mask minus A_mask ROC-AUC deltas: `+0.020910`, `-0.035180`,
  `+0.002066`, `-0.010132`. Mean `-0.005584`; sample standard deviation
  `0.023502`.
- B_path_mask mean Balanced Accuracy: `0.496191`; Accuracy: `0.495880`.
- B_path_mask mean recall negative: `0.643586`; mean recall positive:
  `0.348797`.
- B_path_mask mean delta Balanced Accuracy versus majority: `-0.003809`;
  mean delta Accuracy: `-0.008209`.

Pre-specified verdicts:
- PRIMARY incremental positional-return OOS discrimination: FAIL.
- SECONDARY fixed-threshold classification: FAIL.

Temporal OOS diagnostic:
A_mask mean ROC-AUC `0.501863`; B_path_mask mean ROC-AUC `0.491700`.

Interpretation boundary:
EXP_005 found no OOS evidence under the frozen criterion that the 53
positional return values add sign discrimination beyond positional missingness
through the fixed linear-additive Logistic Regression representation. A_mask
itself also remained approximately non-discriminative for conditional sign.

Combined research observation:
EXP_003 failed with cumulative observed return, EXP_004 failed with
directional sign dominance, and EXP_005 failed to establish incremental sign
information when all 53 return positions were preserved under a linear-
additive model. The current evidence rejects these tested simple/global and
linear-positional representations as useful conditional sign discriminators
under the frozen protocol.

This does not establish that the full path contains no sign information, that
nonlinear interactions or sequential dynamics contain no sign information, or
that RNNs or boosting cannot work.


## 2026-09-11 — EXP_006 Nonlinear Conditional Positional Path Signal

Protocol:
EXP_006 reused the oracle-directional evaluable population and the exact
EXP_005 A_mask/B_path_mask representations, but replaced Logistic Regression
with the frozen HistGradientBoostingClassifier. The competition test was not
accessed.

Integrity:
All frozen integrity assertions passed. Scikit-learn version was `1.6.1`, the
exact frozen HGB parameters were used, and `early_stopping=False`. The
fit-derived majority sign was negative in all four folds. All 53 fit medians
were finite; preprocessing was fit-only, OOS preprocessing was transform-only,
masks remained binary and unscaled, and A/B retained identical rows.

Joint OOS result:
- A_mask ROC-AUC by fold: `0.498276`, `0.497753`, `0.509464`, `0.497879`.
  Mean `0.500843`; sample standard deviation `0.005752`.
- B_path_mask ROC-AUC by fold: `0.492204`, `0.488405`, `0.481283`,
  `0.500209`. Mean `0.490525`; sample standard deviation `0.007885`.
- B_path_mask minus A_mask ROC-AUC deltas: `-0.006072`, `-0.009348`,
  `-0.028182`, `+0.002330`. Mean `-0.010318`; sample standard deviation
  `0.012885`.
- B_path_mask mean Balanced Accuracy: `0.492994`; Accuracy: `0.492568`.
- B_path_mask mean recall negative: `0.595291`; mean recall positive:
  `0.390696`.
- B_path_mask mean delta Balanced Accuracy versus majority: `-0.007006`;
  mean delta Accuracy: `-0.011521`.

Pre-specified verdicts:
- PRIMARY nonlinear incremental positional-return OOS discrimination: FAIL.
- SECONDARY fixed-threshold classification: FAIL.

Temporal OOS diagnostic:
A_mask mean ROC-AUC `0.500360`; B_path_mask mean ROC-AUC `0.491904`.

Interpretation boundary:
EXP_006 found no OOS evidence under the frozen criterion that positional
return values add conditional sign discrimination beyond positional
availability through the fixed HistGradientBoosting representation. The
nonlinear/interacting learner did not recover the conditional sign signal that
was absent under the EXP_005 linear-additive learner.

Combined EXP_003 through EXP_006 observation:
- Cumulative observed return did not discriminate conditional sign OOS.
- Directional sign dominance did not discriminate conditional sign OOS.
- Preserving all 53 positions under Logistic Regression did not establish
  conditional sign discrimination.
- Fixed nonlinear tree interactions over the same positional representation
  also did not establish conditional sign discrimination.

This does not establish that no conditional sign information exists, that
nonlinear information cannot exist under any model, that temporal ordering
contains no information, that sequence models or RNNs cannot work, or that the
challenge is impossible.


## 2026-09-11 — EXP_007 Sequential Chronology Signal

Protocol:
EXP_007 used the same oracle-known directional/evaluable population and frozen
EXP_000 validation protocol as EXP_003 through EXP_006. It compared the same
53 fit-preprocessed return/missingness pairs under their natural chronological
order (Real) against one frozen global artificial ordering (Permuted) using a
paired, deterministic one-layer GRU. The competition test was not accessed.

Scientific question:
Among oracle-known directional/evaluable observations, does presenting the
same 53 return/missingness pairs in their natural chronological order provide
incremental Joint OOS sign discrimination to the frozen GRU relative to one
fixed artificial global ordering?

Integrity:
All four folds completed and all frozen runtime assertions passed. Real and
Permuted used identical retained rows, labels, initial states, and minibatch
orders. All fit medians were finite; OOS preprocessing was transform-only; and
both sign classes were present in every evaluated subset.

Joint OOS result:
- Fold 1: Real AUC `0.516058`, Permuted AUC `0.491894`, delta `+0.024164`.
- Fold 2: Real AUC `0.490858`, Permuted AUC `0.497683`, delta `-0.006825`.
- Fold 3: Real AUC `0.511161`, Permuted AUC `0.503669`, delta `+0.007493`.
- Fold 4: Real AUC `0.473412`, Permuted AUC `0.496952`, delta `-0.023540`.
- Mean Real AUC: `0.497872`; sample standard deviation: `0.019620`.
- Mean Permuted AUC: `0.497549`; sample standard deviation: `0.004824`.
- Mean delta AUC: `+0.000323`; sample standard deviation: `0.020333`.

Pre-specified verdicts:
- PRIMARY OOS chronological-organization evidence: FAIL. Real AUC was not
  above `0.50` in all folds, mean Real AUC was not above `0.50`, and the paired
  delta was not positive in all folds. Only mean delta AUC was positive.
- SECONDARY fixed-threshold classification: FAIL. Mean Joint Balanced Accuracy
  was `0.496540`, Accuracy `0.495585`, recall negative `0.641334`, recall
  positive `0.351746`, and delta Balanced Accuracy versus majority
  `-0.003460`.

Temporal OOS diagnostic:
Mean Real AUC `0.502139`, Mean Permuted AUC `0.497690`, and mean delta AUC
`+0.004449`.

Training diagnostic:
Full-fit BCE declined from epoch 1 to epoch 10 for both conditions in every
fold. Real full-fit ROC-AUC at epoch 10 ranged approximately from `0.5295` to
`0.5355`; there was no evidence of gross optimization failure. These fit-only
diagnostics did not translate into stable Joint OOS sign discrimination and
did not alter the frozen experiment.

Conclusion boundary:
EXP_007 found no consistent Joint OOS evidence that presenting the 53
return/missingness pairs in their natural chronological organization provides
incremental conditional-sign discrimination to the frozen GRU relative to the
pre-specified artificial global ordering. Real-sequence performance varied
materially across folds and averaged approximately chance. The paired
chronology advantage was positive in two folds and negative in two folds, with
mean delta AUC approximately zero.

This does not prove that chronological information is absent from intraday
returns, that sequence models cannot predict conditional sign, or that GRU is
an inappropriate architecture. It does not justify interpreting fold variation
as regime dependence, tuning the GRU, changing epochs, trying additional
seeds, replacing it with an LSTM or Transformer under EXP_007, or making any
claim about final ternary competition performance.

Research-program synthesis, EXP_002 through EXP_007:
- EXP_002 produced consistent OOS evidence that missingness contains
  information for Neutral versus Directional discrimination.
- EXP_003 through EXP_007 have not produced consistent OOS evidence for
  conditional sign discrimination using cumulative return, global sign
  dominance, positional linear relationships, positional nonlinear tree
  interactions, or the frozen sequential chronology test.
- This is an empirical asymmetry in the research program, not a general claim
  that magnitude/activity is predictable while direction is impossible.


## 2026-09-12 — Post-EXP_009 Program-Level Research Protocol

EXP_009 completed with PRIMARY FAIL. After EXP_009, the project adopted a
prospective Discovery/Confirmation protocol. Candidate B was selected from a
target-blind structural panel audit, not from predictive performance:

```text
Discovery:              days 0-352 x E_dev
Internal Confirmation:  days 353-502
```

Internal Confirmation is one frozen three-block expanding procedure:

```text
0-352 -> 353-402
0-402 -> 403-452
0-452 -> 453-502
```

`E_holdout` is reserved from Discovery. Competition-test data remain locked.
Future SHAP work is exploratory and hypothesis-generating only, not
confirmatory evidence. This governance change addresses program-level
researcher-selection risk created by repeated OOS exposure across EXP_000-009,
even where individual experiments were leakage-safe. The Internal
Confirmation Zone is prospective after EXP_009 and is not historically
pristine.
