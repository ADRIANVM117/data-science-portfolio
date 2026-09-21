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

## 2026-09-14 — EXP_008 documentation reconciliation and CONF_001 freeze

The authoritative `EXP_008_*` result artifacts were verified. EXP_008 was
executed once on training data only after its scientific contract had been
frozen; its former “not implemented or executed” status was a documentation
error and has been reconciled without changing its specification or results.
The historical Joint OOS B-minus-C recent-intensity AUC deltas were positive
in all four folds (`+0.076415`, `+0.102034`, `+0.094798`, `+0.099619`), with
mean `+0.093217`.

`CONF_001_recent_intensity_confirmation.md` was frozen before execution. It
carries the documented lineage `EXP_002 -> EXP_008 -> EXP_009 -> D001 ->
D001 SHAP -> D002 -> CONF_001`. It is an Internal Confirmation candidate
under the post-EXP-009 governance protocol, not a pristine holdout and not a
competition lockbox.

## 2026-09-14 — CONF_001 Recent Movement Intensity Internal Confirmation

CONF_001 was executed once after its contract was frozen, using the
post-EXP-009 three-block Internal Confirmation procedure. It retained the
full EXP_002 / EXP_008 Neutral-versus-Directional population and compared
`C=[missing_ratio, q]` against `B=[missing_ratio, q, I_recent_z]` with
Fit-only median imputation and scaling of intensity. The zone is not
historically pristine; the competition test remained unaccessed.

Joint Confirmation results were:

| Block | AUC(C) | AUC(B) | B minus C |
|---:|---:|---:|---:|
| 1 | 0.584716 | 0.686750 | +0.102034 |
| 2 | 0.595982 | 0.690781 | +0.094798 |
| 3 | 0.598600 | 0.698219 | +0.099619 |

The mean Joint incremental AUC was `+0.098817` with sample standard
deviation `0.003684`. Every frozen Joint block delta was strictly positive;
therefore **CONFIRMATION CRITERION SATISFIED**. The historical EXP_008 mean
Joint B-minus-C delta was `+0.093217`; effect magnitude remains separate from
the recurrence criterion.

Temporal AUC deltas were `+0.095721`, `+0.089183`, and `+0.095656` for blocks
1--3 and are diagnostic only. No scientific adaptation occurred between
blocks. This result supports only the frozen full-population linear claim of
recurrent positive Joint Internal-Confirmation discrimination beyond global
missingness and complete recent-window unavailability; it does not establish
causality, an economic mechanism, sign predictability, or final ternary-task
utility.

## 2026-09-14 — Discovery session close after D005

### D004 — Hierarchical Recent-Intensity Ternary Decision

D004 converted the frozen recent-intensity Neutral-versus-Directional model
into a hierarchical ternary hard-class decision. Fit-only directional priors
supplied the deliberately unresolved sign branch. Relative to the Fit-majority
ternary baseline, Discovery Validation accuracy improved in every frozen fold:
candidate accuracy mean/sample SD was `0.442910` / `0.015200`, baseline was
`0.412121` / `0.007931`, and the mean delta was `+0.030789` / `0.009842`.

The Fit directional prior favored `-1` in all three folds. Consequently, D004
never predicted `+1`; this was a property of the frozen no-sign-alpha rule,
not conditional-sign evidence. D004 establishes only Discovery-level decision
value from the N-v-D gate under that fixed hierarchy, not sign alpha.

### Post-D004 routed-population diagnosis

The exact D004 positive accuracy deltas decomposed into routed true `-1` rows
minus routed true `0` rows in every fold:

| Fold | Routed | True −1 | True 0 | True +1 | Net gain | Delta accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17,500 | 6,311 | 4,172 | 7,017 | +2,139 | +0.031911 |
| 2 | 15,431 | 6,299 | 3,618 | 5,514 | +2,681 | +0.040021 |
| 3 | 17,066 | 5,541 | 4,174 | 7,351 | +1,367 | +0.020434 |

Positive decision contribution was not confined to an extreme `p_D` tail:
fixed positive-margin regions near the frozen boundary also contributed
positively in all folds. This diagnosis concerns the ternary routing decision;
it is not evidence that the gate discriminates `-1` from `+1`.

### Gated-sign structural feasibility audit

The target-free reconstruction of the frozen D004 gate left a structurally
broad potential conditional-sign population. Eligible Validation rows were
`13,328`, `11,813`, and `12,892` in folds 1–3, respectively—approximately
30%–34% of the original directional Validation population. Coverage was
`49 days / 1,278 equities`, `50 / 1,205`, and `49 / 1,294`.

`G` is feature-derived. Conditioning on `G=1` can induce selection effects,
so any gated-sign result is Discovery-only and cannot automatically be called
a high-intensity sign regime.

### D005 — Gated Positional Path Conditional-Sign Signal

D005 deliberately changed only the population through the independently
frozen D004 gate. It reused EXP_005's own 106-column positional path (53
Fit-preprocessed positional returns plus 53 original masks) and fixed Logistic
Regression. Validation sign ROC-AUCs were `0.547320`, `0.499176`, and
`0.531055`; mean/sample SD was `0.525850` / `0.024490`.

The frozen Discovery screen required AUC strictly above `0.50` in all three
folds. Fold 2 did not meet it, so
`DESCRIPTIVE_CANDIDATE_SCREEN_MET = NO`. Threshold-0.5 secondary diagnostics
were descriptive only: accuracy was `0.509679`, `0.515449`, and `0.494803`;
Balanced Accuracy was `0.522245`, `0.502406`, and `0.518672`; negative/
positive recalls were `0.759468/0.285022`, `0.698682/0.306130`, and
`0.688684/0.348660`. They do not rescue the failed primary screen.

The correct bounded conclusion is: the frozen D004 gate did not make the
previously specified positional own-path Logistic Regression consistently OOS
discriminative for conditional sign. This does not mean sign is impossible or
inherently unpredictable, that high intensity destroys sign information, that
Fold 2 is a reversal, that EXP_005 was wrong, or that D005 confirms absence
of sign alpha.

### Program-level empirical synthesis

Recent movement intensity has produced strong and relatively stable
Neutral-versus-Directional discrimination, with Discovery-level ternary
decision value in D004. In contrast, no frozen conditional-sign hypothesis
tested so far has produced consistent OOS discrimination. The tested families
include cumulative observed return; sign dominance/persistence; positional own
path with linear Logistic Regression; nonlinear positional tree interactions;
sequence chronology with a GRU; contemporaneous cross-sectional relative path;
and the D004-gated positional own path.

This is empirical evidence about these frozen hypotheses, not proof that sign
information does not exist.

### Governance and current state

- Discovery remains adaptive and hypothesis-generating; D005 arose after D004
  and is not confirmation.
- CONF_001 satisfied its frozen internal criterion, but reused observations
  historically exposed through EXP_008; it is not independent statistical
  replication.
- Days `353–502` are not program-level pristine because of EXP_000–009
  exposure. `E_holdout` has not been used in current Discovery.
- The competition test remains the untouched lockbox; it has not been opened,
  and no decision has been made to spend it.
- `RECENT_MOVEMENT_INTENSITY`: strong historical/Discovery evidence for
  Neutral-versus-Directional ranking.
- `TERNARY_DECISION`: D004 shows positive Discovery OOS decision value under
  the frozen hierarchical no-sign-alpha rule.
- `CONDITIONAL_SIGN`: no consistently successful frozen OOS hypothesis so far.
- `CALIBRATION`: descriptive imperfections in `p_D` are known; no
  recalibration experiment is authorized.
- `LOCKBOX`: untouched and reserved.

### Next Human + Sol decision

Determine which scientific question now deserves additional Discovery degrees
of freedom before any new experiment is frozen. Unresolved possibilities—not
an authorized queue—include a fundamentally different conditional-sign
information source, deeper study of the robust N-v-D intensity signal,
probability calibration/decision architecture, and when the system is mature
enough to justify spending the competition lockbox.

Today's session is closed after D005 interpretation: no D006, additional
target-conditioned diagnostic, model execution, or protected-data access was
authorized by this documentation step.

## 2026-09-15 — D006 Ternary Tail Intensity × Magnitude-Allocation Interaction

D006 completed once on the physical Discovery partition only. All integrity
assertions passed; days `353–502`, `E_holdout`, and the competition test were
untouched. No post-result adaptation, alternate directional allocation,
interaction, window, model, threshold, preprocessing, or metric was tried.

The frozen directional-allocation hypothesis failed: C0/C1 mean MacroTailAUC
was `0.606022` / `0.603691`; C1-minus-C0 fold deltas were `-0.003255`,
`-0.005768`, and `+0.002029`, with mean `-0.002331`. The frozen interaction
hypothesis also failed: C2 mean MacroTailAUC was `0.603488`; C2-minus-C1 fold
deltas were `-0.001131`, `-0.000170`, and `+0.000694`, with mean `-0.000202`.
The pre-specified probability-based orientation was continuation in folds 1
and 3 and opposite in fold 2, therefore overall unstable; Fold 2 is not
evidence for a reversal hypothesis.

The bounded result is that, under the frozen recent-window intensity,
directional squared-magnitude-allocation, and simple multinomial interaction
representations, no consistent Discovery OOS evidence showed that directional
magnitude allocation adds to intensity or that its effect depends on
intensity. It does not say that direction is unpredictable.

Separately, C0 MacroTailAUCs were `0.611054`, `0.604537`, and `0.602475`
(mean `0.606022`), with both negative- and positive-tail AUCs above `0.50` in
all folds. This is descriptive evidence that established intensity/availability
contains tail or large-movement-risk information, not directional alpha. C0's
hard argmax `+1` recall was zero in all folds, a property of that frozen model
and decision rule rather than evidence that positive-tail outcomes are
inherently unpredictable.

### Program-level synthesis after D006

- Recent movement intensity has repeatedly shown useful information for
  Neutral-versus-Directional and tail-risk discrimination.
- D004 showed Discovery-level ternary decision value from the intensity gate.
- Tested directional hypotheses have not produced consistent frozen OOS
  evidence: cumulative return, sign dominance, positional path under Logistic
  Regression, positional path under fixed HGB, GRU chronology,
  cross-sectional relative path, gated positional path, and signed magnitude
  allocation with its intensity interaction.
- This does not establish that sign or direction is impossible to predict.
- It does establish that continued ad-hoc variation of own-path directional
  summaries would create substantial feature-shopping risk.

No D007 is authorized. The next step is a Human + Sol program-level scientific
review before any new Discovery experiment.

## 2026-09-16 — D007 Raw Positional Returns Beyond Missingness

D007 is completed, Discovery-only, with PRIMARY SCREEN PASS. The first attempt
was a technical execution failure with no scientific result: a fixed
`atol=1e-12` row-sum assertion rejected valid float32 XGBoost probabilities at
machine-epsilon scale before result generation. No predictive result was
inspected. Human + Sol approved a dtype-aware assertion-only repair, which
passed required tests; model, features, folds, seed, metrics, and screen were
unchanged. The subsequent authorized rerun is the first scientifically
interpretable D007 execution. Scikit-learn emitted documented float32
probability-row-sum warnings during log-loss calculation; no adaptation was
made and execution completed.

The scientific question was whether raw `r0...r52` add consistent ternary
probability information beyond `m0...m52` under the same fixed XGBoost learner.
Validation log-loss deltas `log_loss(M)-log_loss(R+M)` were `+0.029247`,
`+0.023477`, and `+0.025861`; mean `+0.026195`, sample SD `0.002899`. Mean
log loss was `1.035615` for M and `1.009420` for R+M. Every delta was strictly
positive, so the frozen primary screen passed.

The bounded conclusion is: the frozen D007 probe provides evidence of
consistent incremental Discovery OOS ternary probability information from raw
return values beyond missingness under this fixed XGBoost learner. This does
not establish economic alpha, directional alpha, deployability, competition
improvement, lockbox generalization, causal mechanism, or material usefulness
under a post-hoc threshold.

Secondary diagnostics are descriptive only: mean MacroTailAUC `0.566113 ->
0.618777`, AUC -1 `0.563384 -> 0.607790`, AUC +1 `0.568841 -> 0.629764`,
Accuracy `0.413778 -> 0.464439`, Macro-F1 `0.224811 -> 0.410687`, and Balanced
Accuracy `0.339889 -> 0.426883`. Both tail AUCs improved in every fold; mean
recalls -1/0/+1 were `0.050036/0.967384/0.002248` for M and
`0.322349/0.750523/0.207777` for R+M. These were not primary-screen criteria.

Program interpretation: missingness alone retains predictive information, but
D007 establishes that raw observed return values add consistent incremental
ternary probability information beyond it under this frozen flexible learner.
Earlier directional failures therefore should not be summarized as evidence
that the raw path lacks directional or tail-specific information; a more
defensible statement is that the hand-crafted or restricted representations
then tested did not consistently extract it. D007 does not establish a stable
directional mechanism or identify relevant positions, signs, thresholds,
interactions, or missingness-return combinations.

No SHAP, feature-importance inspection, tuning, alternate learner, or
post-result feature engineering has been performed. D007 is closed and frozen;
no D008 is authorized. The next step is a Human + Sol design review for a
bounded post-D007 model-interpretation phase that may generate future
hypotheses but cannot improve or revalidate D007 on the same folds. The run
used only physical Discovery `days 0–352 × E_dev`; days `353–502`, `E_holdout`,
and competition-test data remain untouched.

## 2026-09-16 — I007A Frozen D007 Main-Effect Interpretation

I007A completed as a bounded descriptive interpretation of the successful
D007 `R+M` model, using physical Discovery only. Deterministic reconstruction
of all three D007 models passed the frozen log-loss gate and raw-margin
TreeSHAP additivity passed. No interactions, refitting for improvement,
predictive re-evaluation, protected-region access, or competition-test access
occurred.

Across every output and fold, TreeSHAP main-effect allocation was entirely to
raw returns (`R_share=1.0`, `M_share=0.0`). This must not be read as
missingness being irrelevant: raw returns retain native `NaN`s, and XGBoost can
encode observed versus unavailable values through native missing routing.
Explicit masks may therefore be redundant conditional on native-NaN raw
features. Attribution allocation is not a unique-information decomposition.

Attribution rankings were highly stable. Stable Top-10 sets were
`{-1: r33, r46, r48, r52}`, `{0: r1, r44, r46, r47, r48, r49, r50, r51, r52}`,
and `{+1: r24, r50, r51, r52}`; only `r52` overlapped the two tail sets.
Limited frozen same-output orientation tables showed a consistent Spearman
sign across folds for each eligible stable tail feature, while magnitude
stability varied. These facts do not establish individual incremental signal,
causal relevance, an economic mechanism, or deployable directional alpha.

### Program-level synthesis through I007A

- D007 produced consistent Discovery OOS incremental ternary probability
  information from raw returns beyond masks under its fixed XGBoost learner.
- I007A found highly stable positional main-effect attributions across folds;
  late positions recurred especially for the neutral output.
- EXP_003--EXP_007 failures do not show that the raw path contains no
  information: they show that their frozen hand-crafted or restricted
  representations and learners did not consistently extract it, whereas D007
  did under a flexible raw-path learner.
- I007A does not resolve whether the D007 result is explained by established
  intensity/availability, incremental raw positional values, nonlinear
  combinations, or native-NaN routing. These remain competing explanations.

The prospective question is whether raw positional paths provide consistent
Discovery OOS ternary predictive information beyond an appropriately frozen
control for established recent-intensity/availability structure. Its precise
control remains undecided. No D008, interaction analysis, or new model is
authorized by this documentation entry.

## 2026-09-20 — D008 Raw Path Beyond Availability + Established Recent Intensity

D008 is completed, Discovery-only, with PRIMARY SCREEN PASS. It compared
`C=[m0,...,m52,I_recent_z]` against
`P=[m0,...,m52,I_recent_z,r0,...,r52]`: full positional availability and exact
fit-only EXP_008 recent intensity were shared, while untouched native-NaN raw
returns were added only to P. `q`, `missing_ratio`, SHAP-selected features, and
other engineered summaries were excluded.

The one authorized execution completed all three frozen Discovery folds with
exit status `0` in approximately `406.35` seconds. All nine artifacts
persisted and the frozen contract SHA-256 remained unchanged. Five sklearn
probability-row-sum warnings occurred during log-loss calculation; there was
no error, adaptation, repair, or rerun. The preprocessing artifact omits
Validation raw-`I_recent` NaN counts; this is a reporting omission only, and
no post-run computation was authorized. Protected boundaries remained intact.

Fold log-loss deltas `log_loss(C)-log_loss(P)` were
`+0.0035748973537543804`, `+0.0007623862799182035`, and
`+0.0029766777333486427`; mean `+0.002437987122340409`, sample SD
`0.001481619153461506`. Because every delta and the mean were strictly
positive, the frozen primary screen passed. The bounded conclusion is that,
under the frozen XGBoost learner, raw positional returns provided consistent
incremental Discovery OOS ternary predictive utility beyond the shared
full-positional-availability plus established-recent-intensity control. The
residual improvement is modest in magnitude.

Secondary metrics remain descriptive. Tail-ranking moved only slightly
(MacroTailAUC `0.619066 -> 0.619871`), while hard multiclass allocation changed
more visibly: `+1` recall was `0.011209 -> 0.198979` and `-1` recall was
`0.490856 -> 0.360556`. This is compatible with residual raw-path information
affecting ternary probability allocation; it is not evidence that a
directional/sign mechanism has been isolated.

### Program-level synthesis after D008

- Availability/missingness has reproducible Neutral-versus-Directional
  information.
- EXP_008 established recent movement intensity as a strong N-v-D
  representation.
- Multiple restricted or hand-crafted sign/path representations failed under
  their frozen contracts.
- D007 showed incremental ternary information from raw returns plus masks
  beyond masks alone under fixed XGBoost; I007A showed stable model-specific
  positional attribution structure without identifying the mechanism.
- D008 showed that explicit recent-intensity control greatly reduced, but did
  not eliminate, the incremental raw-path advantage under the frozen XGBoost
  comparison.

The remaining object is a smaller but consistent residual raw-path
contribution beyond full availability plus established intensity. It may
reflect residual nonlinear directional/class-allocation structure, other
magnitude/path-shape information, native-NaN routing with duplicated
availability, finite-sample/model-capacity effects, or unresolved
interactions. These are competing explanations, not conclusions.

The prospective question is whether the residual D008 improvement primarily
reflects changes in tail-event detection or probability allocation among
directional classes once availability and recent intensity are controlled. No
diagnostic design is frozen, no D009 is authorized, and no new training, SHAP,
interaction analysis, alternate control, feature selection, or tuning is
authorized.
