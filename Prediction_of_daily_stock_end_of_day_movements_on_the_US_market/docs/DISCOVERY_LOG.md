# Discovery Log

This registry records material researcher search paths under the post-EXP_009
program-level protocol, including negative and abandoned ideas. It is not a
record of every trivial plot, temporary debugging action, or implementation
detail. EXP_000 through EXP_009 predate this governance regime; see
`docs/RESEARCH_LOG.md` and the corresponding experiment contracts for that
pre-protocol research history.

## Template

## DXXX — Short candidate name

Status:
`PROPOSED` / `EXPLORING` / `REJECTED_IN_DISCOVERY` /
`RETAINED_FOR_FREEZE` / `MERGED` / `ABANDONED`

Date:

Scientific idea:

Origin:  
economic reasoning / statistical reasoning / EDA / SHAP / model behavior /
literature / other

Discovery scope:  
days 0-352 x E_dev

Information inspected:

Candidate representation or mechanism:

Why it might matter:

Discovery evidence:

Contradictory evidence:

Researcher decisions made after seeing evidence:

Decision:

Reason:

Eligible for confirmation:  
YES / NO

If YES:  
Link to the subsequently frozen experiment contract once one exists.

## D001 — Path structure beyond recent RMS

Status:
`EXPLORING`

Date:
2026-09-12

Scientific idea:
Explore whether the complete 12-position recent magnitude path has Discovery
discrimination structure beyond a scalar recent RMS representation for
Neutral-versus-Directional outcomes.

Origin:
Statistical reasoning and model behavior following EXP_008/EXP_009.

Discovery scope:
days 0-352 x E_dev

Information inspected:
Complete-W availability, target labels within Discovery only, and the
authorized C/P representations. No Internal Confirmation, `E_holdout`, or
competition-test data are permitted.

Candidate representation or mechanism:
`C_D001 = [I_recent]`; `P_D001 = [abs(r41), ..., abs(r52)]`, each on rows
with all 12 W positions observed.

Why it might matter:
Aggregate RMS discards the positional organization of magnitudes inside W.

Discovery evidence:
Predictive Discovery comparison executed once on 2026-09-12. Validation
ROC-AUCs were:

| Fold | C_D001 | P_D001 | P minus C |
|---:|---:|---:|---:|
| 1 | 0.651871 | 0.652191 | +0.000319 |
| 2 | 0.642132 | 0.644425 | +0.002293 |
| 3 | 0.648623 | 0.649474 | +0.000851 |

Mean Validation ROC-AUC was 0.647542 for C_D001 and 0.648697 for P_D001.
Mean P-minus-C Validation ROC-AUC was +0.001154 (sample standard deviation
0.001021). Environment: Python 3.13.2, NumPy 2.2.3, pandas 2.3.1,
scikit-learn 1.6.1, and XGBoost 3.4.1.

Approved exploratory interpretation:  
Under the fixed exploratory XGBoost learner, the positional magnitude
representation produced only a very small and consistent Discovery OOS
improvement over scalar recent RMS. Most of the observed discrimination
available to this comparison appears reproducible from aggregate recent
movement intensity. This does not establish statistical sufficiency of RMS,
does not establish that the path contains no additional information, and does
not create a confirmatory hypothesis from the predictive delta alone.

Execution integrity note:  
Only Discovery-scope rows entered the retained populations, fitted models, or
reported metrics. However, the current chunked loader parses the source CSV
before filtering each chunk. Under a strict raw-data-access interpretation,
it therefore does not establish that days 353-502 were never parsed. No
rerun or repair is authorized here. Prospectively, this caveat was addressed
on 2026-09-13 by target-blind physical partition materialization under a
separately authorized infrastructure operation. It does not change the
historical D001 execution.

Reproducibility note:  
The post-fit serialization defect in `D001_incremental_auc.csv` was repaired
without refitting. It now records separate Fold 1, Fold 2, Fold 3, aggregate
mean, and aggregate sample-standard-deviation rows using the authoritative
persisted `D001_metrics.csv` values.

Contradictory evidence:  
Fit-minus-Validation ROC-AUC gaps for C_D001 were +0.002901, +0.012265, and
+0.003703 across folds 1-3; corresponding P_D001 gaps were +0.009552,
+0.015949, and +0.008603. These are Discovery diagnostics only.

Researcher decisions made after seeing evidence:  
No feature, model, parameter, threshold, or representation change was made.
Before the bounded SHAP pass documented below, no real-data SHAP,
feature-importance, tree, or interaction analysis was performed. Limited SHAP
is approved only as hypothesis-generation
analysis. A SHAP pattern may be logged as a material candidate only if it is
repeated across all or nearly all Discovery folds, has qualitatively similar
dependence behavior in those folds, and, when an interaction is claimed,
recurs as the same feature pair or compact positional region across folds.
Mean absolute SHAP alone, a pattern isolated to one fold, or a one-off
interaction is insufficient. This is a qualitative stability rule; no
post-hoc numerical threshold is authorized.

Bounded SHAP interpretation pass (2026-09-13):
The original fitted P_D001 models were not persisted. Under a separately
authorized SHAP-only reconstruction, the three frozen P models were rebuilt
from the physical Discovery partition and reproduced their authoritative
Validation AUCs exactly: Fold 1 `0.6521908605`, Fold 2 `0.6444253326`, Fold 3
`0.6494738303`. SHAP used version `0.52.0` with XGBoost `3.4.1` and exactly
`TreeExplainer(data=None, feature_perturbation="tree_path_dependent",
model_output="raw", feature_names=PATH_COLUMNS)`. Each fold explained 2,000
deterministically selected Validation rows; interactions used a nested 500-row
sample. Values are raw-margin attributions, not probability attributions.

Global cross-fold SHAP result:

| Position | Fold 1 mean abs SHAP (rank) | Fold 2 | Fold 3 | Cross-fold mean |
|---|---:|---:|---:|---:|
| abs(r41) | 0.05102 (12) | 0.06275 (11) | 0.05741 (12) | 0.05706 |
| abs(r42) | 0.07157 (8) | 0.06814 (9) | 0.06400 (9) | 0.06791 |
| abs(r43) | 0.06548 (10) | 0.06923 (8) | 0.06825 (8) | 0.06765 |
| abs(r44) | 0.08005 (4) | 0.07969 (3) | 0.08099 (1) | 0.08024 |
| abs(r45) | 0.05717 (11) | 0.05577 (12) | 0.06320 (10) | 0.05871 |
| abs(r46) | 0.08419 (1) | 0.07523 (6) | 0.07411 (3) | 0.07784 |
| abs(r47) | 0.08173 (2) | 0.07674 (4) | 0.07389 (4) | 0.07745 |
| abs(r48) | 0.07896 (5) | 0.07995 (2) | 0.07337 (6) | 0.07743 |
| abs(r49) | 0.08021 (3) | 0.08265 (1) | 0.08098 (2) | 0.08128 |
| abs(r50) | 0.06879 (9) | 0.06608 (10) | 0.06120 (11) | 0.06536 |
| abs(r51) | 0.07577 (6) | 0.07229 (7) | 0.07163 (7) | 0.07323 |
| abs(r52) | 0.07239 (7) | 0.07588 (5) | 0.07352 (5) | 0.07393 |

Rank correlations were `0.804` (folds 1/2), `0.881` (1/3), and `0.874`
(2/3). Attribution was not concentrated at a single terminal position: the
most recurrent prominence was a compact middle-to-late region. `abs(r49)` was
the only position ranked in the top three in all folds (ranks 3, 1, 2). Its
raw-margin SHAP dependence had positive Spearman correlations with abs(r49)
of `0.862`, `0.850`, and `0.846` across folds; this is descriptively
consistent with an approximately monotonic fitted-model relationship, not a
fitted threshold or a causal claim.

Interaction result:
The strongest cross-fold mean-absolute interaction was adjacent pair
abs(r45)-abs(r46), `0.006229`, with ranks 7, 2, 2 across folds. Other
recurrently high adjacent pairs were abs(r46)-abs(r47), ranks 1, 7, 9, and
abs(r47)-abs(r48), ranks 10, 3, 1. The early adjacent pair abs(r41)-abs(r42)
also ranked 2, 5, 6. These are model-attribution patterns under dependent
positional magnitudes, not evidence of unique interactions or mechanisms.

Candidate decision (superseding the earlier pre-SHAP status below):
`MATERIAL_D001_CANDIDATE_REQUIRES_HUMAN_SOL_INTERPRETATION`. The candidate is
only the observed recurring structure: a compact r45-r49 magnitude region,
including a stable positive `abs(r49)` contribution and recurrent adjacent
attributions. It is not an engineered feature, a threshold, a causal claim,
or a confirmation hypothesis. The small original P-minus-C Discovery AUC
increment remains an important limitation. A separately frozen hypothesis is
still required before Internal Confirmation.

## D002 — Local organization of recent movement intensity

Status:  
`PROPOSED`

Date:  
2026-09-13

Scientific idea:  
Within the complete-W physical Discovery population, assess descriptively
whether the local temporal organization and adjacency of absolute movement
magnitudes may distinguish Neutral-versus-Directional behavior beyond their
aggregate recent RMS level.

Origin:  
D001 TreeSHAP. The idea was generated after observing stable positional ranks,
the approximate r44-r49 prominence region, and recurrent high-ranked adjacent
interactions. It is explicitly Discovery-generated rather than pre-specified.

Discovery scope:  
days 0-352 x E_dev, physical Discovery partition only; complete-W rows only.

Information inspected:  
No D002 target-conditioned statistic, model, predictive metric, Internal
Confirmation observation, E_holdout observation, or competition-test row has
been inspected.

Candidate representation or mechanism:  
Primary local-organization diagnostic: `A_adj = sum(a_t * a_(t+1)) / E` for
`E > 0`, otherwise zero, where `a_t = abs(r_t)` and `E = sum(a_t^2)` over
the complete r41-r52 window. Secondary interpretation diagnostic only:
`H_peak = max(a_t^2) / E` for `E > 0`, otherwise zero. `H_peak` cannot become
a competing feature selected by target association.

Why it might matter:  
D001 suggested a recurrent compact positional region and adjacent attribution
patterns, but did not establish local structure beyond aggregate RMS, an
economic mechanism, or a useful engineered feature.

Discovery evidence:  
No target-conditioned D002 local-organization evidence has been inspected.

Target-blind RMS matching design and quality audit (2026-09-14):
The earlier decile-stratification proposal was rejected before any
target-conditioned organization analysis because meaningful residual RMS
imbalance could remain within coarse strata, while RMS is already known to be
related to the target. D002 therefore uses deterministic local 1:1 RMS
matching: Neutral (`Z=0`) is the anchor; Directional (`Z=1`) is the donor and
may be reused; matching is performed independently within `0-202`, `203-252`,
`253-302`, and `303-352`; nearest `I_recent` donor is selected; ties resolve
by lower donor RMS and then lower donor ID. No caliper is used, so the
quality distribution is reported rather than improved by target-informed
discarding. The pooled result is the union of these within-block matches.

All complete-W rows were retained as matching candidates. There were 1,455
zero-energy rows in pooled Discovery (854, 177, 220, and 204 by the four
blocks). They are not excluded: zero-RMS anchors match exact zero-RMS donors
when available, otherwise their nearest positive-RMS donor. The future
organization convention remains `A_adj = H_peak = 0` at `E=0`; no
target-conditioned organization outcome for these rows was inspected.

Matching quality only:
Pooled Discovery produced 125,961 pairs, retaining all 125,961 Neutral
anchors and 81,966 unique Directional donors (35.37% of Directional rows).
The pooled mean and median absolute RMS differences were `0.000737` and
`0.000106`; p90/p95/p99 were `0.000771`, `0.001458`, and `0.007671`; the
maximum was `4.007026`. The block-level mean absolute RMS differences were
`0.000390`, `0.001001`, `0.001179`, and `0.001473`; corresponding median
differences were `0.000061`, `0.000260`, `0.000254`, and `0.000263`. Exact
RMS-match frequencies were 1.56%, 0.72%, 0.86%, and 0.92%. These are control
quality diagnostics only, not scientific results for `A_adj` or `H_peak`.

Future D002 estimand:
For each frozen pair, `Delta_A_adj = A_adj_directional - A_adj_neutral`.
The later Discovery-only descriptive report must give its mean, median,
positive fraction, and distribution summary pooled and separately for blocks
1-3. `H_peak` is secondary interpretation only and cannot rescue a weak
primary `A_adj` pattern. No PASS/FAIL threshold is defined.

Decision:
Matching-quality audit completed. Stop for Human + Sol review before any
target-conditioned D002 local-organization analysis.

D002 paired organization result (2026-09-14):
The frozen pair artifact was joined without rematching, a caliper, or any
post-result exclusion. Primary `Delta_A_adj = A_adj_directional -
A_adj_neutral` was negative in each primary Discovery block: Block 1 mean
`-0.011554`, median `-0.007176`, positive/zero/negative fractions `0.477851`,
`0.011102`, `0.511047`; Block 2 mean `-0.005602`, median `-0.000322`,
fractions `0.487862`, `0.011478`, `0.500659`; Block 3 mean `-0.004998`,
median `0.000000`, fractions `0.490677`, `0.012999`, `0.496324`. Pooled mean
was `-0.007166`, median `-0.000189`, with fractions `0.486539`, `0.013075`,
and `0.500385`. The historical 0-202 block was also negative (mean
`-0.006978`, median `0.000000`).

Secondary `Delta_H_peak` was near zero and not directionally stable across
the primary blocks: means were `+0.005028`, `-0.002790`, and `-0.003089`.
It does not alter the primary interpretation. Zero-energy-pair diagnostics
reported 663 neutral-zero pairs, 666 directional-zero pairs, and 663 pairs
with both sides zero in pooled Discovery; none were removed.

Factual interpretation:
At closely matched recent RMS intensity, this primary adjacency diagnostic did
not show a material descriptive organization difference between Directional
and Neutral observations. Although the mean primary delta was negative in all
three blocks, its magnitude was small, the medians were near zero, and the
positive fractions remained near one half. This is Discovery-only descriptive
evidence; no IID p-value, causal mechanism, predictive claim, or confirmation
claim is made.

Status after execution:
`EXPLORING_NO_MATERIAL_ORGANIZATION_PATTERN` (superseding the earlier
`PROPOSED` status). No Confirmation contract is created.

Eligible for confirmation:
NO — no D002 diagnostic or confirmation hypothesis has been frozen.

Decision:  
Remain EXPLORING. Stop after the frozen predictive comparison pending Human +
Sol review; no candidate mechanism has been generated.

Reason:  
The comparison may generate interpretable positional-path hypotheses; it is
not confirmatory evidence.

Model-representation interpretation limitation:  
Although `I_recent` is deterministic from `P_D001`, a finite depth-limited
tree ensemble receiving the twelve components may not reconstruct RMS as
efficiently as a direct scalar input. Therefore P greater than C would show
greater Discovery OOS discrimination under this learner, not unique
incremental information conditional on RMS; P less than or equal to C would
not prove path structure contains no additional information.

Eligible for confirmation:  
NO — no material candidate hypothesis has yet been generated or frozen.

## Target-free cross-sectional information audit (post-CONF_001)

Status:  
`EXPLORING`

Discovery scope:  
Only the physical target-blind partition `days 0-352 x E_dev`:
`472,816` input rows, `353` days, and `1,463` equities. No label file,
target, Internal Confirmation day, `E_holdout` row, or competition-test row
was opened.

Target-free structural facts:  
Same-day position coverage has median `0.88814` across `18,709` day-position
cells. Four Discovery days (`112`, `134`, `229`, `314`) have zero coverage at
`r42,...,r52`; every other day-position cell has at least two observed
equities. Same-day MAD has median `7.17` bps and is nonzero in every
evaluable cell. Exact all-row versus leave-one-out medians differ by mean
absolute `0.009038` bps across observed comparisons (maximum `0.555` bps).

Candidate information taxonomy:  
Same-day median common movement, an equity-minus-median relative return,
cross-sectional percentile rank, and an aggregate relative path are
structurally distinct from the own-row information used in EXP_003--EXP_007.
No predictive representation, missingness rule, self-inclusion convention,
or target-conditioned result has been selected.

Generated candidate questions:  
Whether a pre-specified contemporaneous relative observed path contains
conditional-sign information beyond an equity's own path; whether a
positionwise relative state does so; and whether percentile rank differs from
median-residual information. These are questions only, not frozen hypotheses.

Decision:  
No D003 is created. The next target-conditioned Discovery action requires
Human + Sol review, including an explicit same-day data-availability
assumption and a frozen missing-position/self-inclusion convention.

Artifact:  
`discovery/CROSS_SECTIONAL_TARGET_FREE_AUDIT.md` and the corresponding
`discovery/results/CROSS_SECTIONAL_TARGET_FREE_*` tables.

## D003 — Incremental Cross-Sectional Relative Path for Conditional Sign

Status:  
`COMPLETED — DESCRIPTIVE_CANDIDATE_SCREEN_NOT_MET`

Discovery scope:  
Physical Discovery boundary `days 0-352 x E_dev` only. D003 is Discovery,
not Internal Confirmation and not a competition-lockbox evaluation.

Frozen question:  
Among oracle-known directional observations, does adding an equity's
same-day all-row-median relative return path improve conditional-sign
discrimination beyond its own 53-position path under the fixed EXP_005
linear-additive Logistic Regression?

Frozen representation:  
`A = [fit-imputed/scaled own path, original own mask]`; `B = [A,
separately fit-imputed/scaled 53-position relative path]`. The common
component is the all-row same-day median, target-free and including the
equity itself when observed. Relative and own missingness must be identical;
no second mask, rank, MAD, dispersion, recency statistic, or local-structure
feature is allowed.

Frozen chronology and candidate rule:  
Fit/Validation days are `0-202/203-252`, `0-252/253-302`, and
`0-302/303-352`. A Human + Sol material-candidate review requires positive
B-minus-A Validation AUC in all three Discovery folds plus separate human
assessment that the magnitude is not numerically trivial. This is not a
confirmatory PASS/FAIL rule.

Technical execution provenance:  
The initial authorized run stopped before a valid result because D003 called
the established sign-evaluation helper with an unsupported keyword argument.
It is classified as `TECHNICAL_EXECUTION_FAILURE_BEFORE_VALID_RESULT` and is
not scientific evidence. A minimal call-site repair was validated with a new
end-to-end synthetic test and the relevant historical regression suites before
one authorized valid reexecution.

Valid D003 result:  
All frozen integrity assertions passed in the single valid reexecution. Joint
Discovery AUCs for A/B/B-minus-A were respectively: Fold 1 `0.531467` /
`0.543880` / `+0.012414`; Fold 2 `0.485030` / `0.421540` / `−0.063490`; Fold
3 `0.517804` / `0.497722` / `−0.020082`. Means were `0.511434`, `0.487714`,
and `−0.023719`; the B-minus-A sample standard deviation was `0.038082`.

Decision:  
`DESCRIPTIVE_CANDIDATE_SCREEN_NOT_MET`. The frozen screen required a positive
B-minus-A Validation AUC in all three Discovery folds and a positive mean
delta; neither condition held. This Discovery result does not establish that
cross-sectional relative paths contain no information under other frozen
representations or learners.

For B, validation mean metrics were ROC-AUC `0.487714`, balanced accuracy
`0.499071`, accuracy `0.492104`, negative recall `0.556386`, and positive
recall `0.441755`.

Access boundary:  
The valid execution used only the physical Discovery partition `days 0–352 ×
E_dev`. Days `353–502`, `E_holdout`, and competition-test data remain
unaccessed.

## D004 — Hierarchical Recent-Intensity Ternary Decision

Status:  
`FROZEN — DISCOVERY ONLY — NOT EXECUTED`

Scientific idea:  
Evaluate a fixed no-conditional-sign-alpha ternary decision layer built from
the supported recent-intensity Neutral-versus-Directional probability and
Fit-only directional class priors.

Discovery scope:  
Physical `days 0–352 × E_dev` only. Days `353–502`, `E_holdout`, and the
competition test are prohibited.

Frozen representation and decision:  
Reuse EXP_008 / CONF_001 condition B
`[missing_ratio, q, I_recent_z]`, with its Fit-only intensity preprocessing
and fixed Logistic Regression. Convert raw directional probability to ternary
probabilities using Fit-only directional priors. Direction must strictly beat
neutral; exact neutral/directional ties predict `0`; a strict directional tie
between `-1` and `+1` predicts `-1` deterministically.

Internal objective:  
Ordinary ternary hard-class accuracy versus the Fit-majority ternary baseline.
This is an internal Discovery metric only: the official competition metric
remains unresolved. No target-conditioned D004 result has been observed.

Execution result:  
One authorized execution completed across all three frozen Discovery folds.
All runtime integrity assertions passed; only physical `days 0–352 × E_dev`
were used. Days `353–502`, `E_holdout`, and competition-test data were not
accessed.

| Fold | Hierarchical accuracy | Fit-majority accuracy | Delta accuracy | N-v-D AUC |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.452552 | 0.420641 | +0.031911 | 0.687598 |
| 2 | 0.450790 | 0.410769 | +0.040021 | 0.679077 |
| 3 | 0.425388 | 0.404954 | +0.020434 | 0.679254 |

Candidate accuracy mean/sample SD: `0.442910` / `0.015200`; baseline mean/
sample SD: `0.412121` / `0.007931`; delta mean/sample SD: `+0.030789` /
`0.009842`. The fixed Discovery screen (`delta_accuracy > 0` in every fold)
is met.

Routing fact:  
`pi_minus > pi_plus` in every Fit subset. Consequently, the frozen
no-sign-alpha rule predicted only `-1` whenever it routed a row directionally;
it never predicted both directional signs. This is a direct property of the
frozen rule, not a conditional-sign result.

Status after execution:  
`COMPLETED — DESCRIPTIVE_CANDIDATE_SCREEN_MET`. This remains Discovery-only
internal hard-class accuracy evidence; it does not establish official
competition performance, probability calibration, conditional-sign alpha,
trading alpha, or lockbox generalization.

Post-D004 descriptive diagnosis:  
Because D004 did not persist row-level probabilities or predictions, the
frozen model was reconstructed exactly on the same physical Discovery inputs
solely to recover them. Candidate and baseline accuracies, plus all three
frozen thresholds, reconciled to the persisted D004 artifacts exactly.
No alternate threshold, calibration method, sign model, feature, or routing
rule was tested.

The D004 net accuracy gains exactly decomposed into routed true-negative rows
minus routed true-neutral rows: Fold 1 `6311 - 4172 = +2139` (`+0.031911`);
Fold 2 `6299 - 3618 = +2681` (`+0.040021`); Fold 3 `5541 - 4174 = +1367`
(`+0.020434`). Routed true-positive rows were respectively `7017`, `5514`,
and `7351`; they remain errors under both the neutral baseline and D004's
fixed `-1` directional branch.

The gain/cost ratios were `1.513`, `1.741`, and `1.328`, so the same factual
routing pattern occurred in all folds: the frozen routed subset contained
more true `-1` than true `0` observations. The closest fixed positive-margin
region `(0.00, 0.05]` supplied `1146`, `1688`, and `909` net correct gains;
positive contribution also remained in the next two fixed regions in every
fold. Only Fold 3's farthest fixed region `(0.20, +inf)` had a negative net
contribution (`-37`). These fixed regions are descriptive and were not used
to select a replacement threshold.

Interpretation boundary:  
The observed N-v-D AUCs describe ranking of Directional versus Neutral.
D004 routing describes the frozen gate's hard-label behavior. Neither result
shows that `p_D` distinguishes `-1` from `+1` among directional observations:
the frozen architecture predicted only `-1` whenever routed directionally.
Existing raw-probability reliability deviations could plausibly matter for a
numerical decision boundary, but no calibration repair was fitted or tested.

## Post-D004 gated-sign structural feasibility audit

Status:  
`TARGET-FREE GATE / CLASS-BALANCE FEASIBILITY AUDIT COMPLETED`

Scope and gate:  
Only physical `days 0–352 × E_dev` were used. The gate was reconstructed
exactly from the frozen D004 N-v-D model, Fit-only priors, and strict
Fit-derived threshold. Validation sign labels were not used to define, alter,
or select the gate. No conditional-sign model, sign AUC, sign feature
relationship, alternative threshold, or calibration method was evaluated.

Validation gate size and eligible conditional-sign population:  
Fold 1 routed `17,500 / 67,031` rows (`26.11%`), with `13,328` eligible
directional rows (`6,311` negative, `7,017` positive), retaining `34.32%` of
the original Validation directional population. Fold 2 routed `15,431 /
66,989` (`23.04%`), with `11,813` eligible rows (`6,299` negative, `5,514`
positive), retaining `29.93%`. Fold 3 routed `17,066 / 66,899` (`25.51%`),
with `12,892` eligible rows (`5,541` negative, `7,351` positive), retaining
`32.39%`.

Panel coverage:  
The eligible Validation population spans `49`, `50`, and `49` represented
days and `1,278`, `1,205`, and `1,294` equities in folds 1--3, respectively.
Its per-day median row counts are `234`, `203`, and `221`; per-equity medians
are `7`, `7`, and `7`. The observed -1/+1 balances are `47.35%/52.65%`,
`53.32%/46.68%`, and `42.98%/57.02%`.

Structural assessment:  
Without setting a numerical sufficiency threshold, the gated eligible
population is broad across sessions and equities and has substantial counts
in each frozen Validation fold. It is therefore structurally viable for
Human + Sol to consider designing a separately frozen **Discovery** gated
conditional-sign experiment. This is feasibility evidence only, not evidence
that conditional sign is predictable.

Selection warning:  
`G` is a deterministic function of features through the independently frozen
D004 N-v-D architecture. Conditioning on `G=1` can change feature
distributions and relationships through selection/collider effects even if no
new sign information exists. Any future positive result could initially
support only specified-model Discovery discrimination within this frozen
routed population; it could not casually be called sign predictability in a
high-intensity regime.

## D005 — Gated Positional Path Conditional-Sign Signal

Status:  
`COMPLETED — DESCRIPTIVE_CANDIDATE_SCREEN_NOT_MET — DISCOVERY ONLY`

Motivation and scope:
The post-D004 target-free structural audit established that the independently
frozen D004 gate leaves a broad, class-balanced directional/evaluable
population across all three physical Discovery Validation folds. D005 is a
separate conditional-sign question on that already specified population; it
does not modify D004 or reinterpret its ternary decision result.

Frozen question and representation:
Within `G=1` and oracle-known `reod ∈ {-1,+1}`, D005 will test the exact
EXP_005 106-column own positional path representation—53 Fit-median-imputed,
Fit-standardized returns plus 53 original binary masks—with the frozen linear
Logistic Regression. The D004 gate is rebuilt using its frozen N-v-D model,
Fit-only preprocessing, Fit-only directional priors, and strict threshold;
it is population selection only, never a sign predictor.

Interpretation boundary:
Any later positive Discovery screen could support only specified-model
conditional-sign discrimination within the D004-routed Discovery population.
It cannot establish deployable sign alpha, a high-intensity/causal regime,
ternary usefulness, calibration, trading alpha, or confirmation. No D005
model, performance metric, target-conditioned sign relationship, or
protected-region data has been accessed at freeze time.

Execution result:

One authorized complete execution was performed across all three frozen
Discovery folds. All runtime integrity assertions passed. The execution used
only the physical Discovery partition `days 0–352 × E_dev`; days `353–502`,
`E_holdout`, full-training fallbacks, and competition-test data were not
accessed. No adaptation between folds or alternative gate, representation,
model, probability orientation, threshold, or feature specification was
evaluated. All-NaN own-path exclusions were zero in both Fit and Validation
subsets in every fold.

| Fold | Validation eligible rows | Validation ROC-AUC | Accuracy | Balanced Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 13,328 | 0.547320 | 0.509679 | 0.522245 |
| 2 | 11,813 | 0.499176 | 0.515449 | 0.502406 |
| 3 | 12,892 | 0.531055 | 0.494803 | 0.518672 |

Secondary frozen-threshold diagnostics were descriptive only and did not
alter the primary screen. Negative/positive recalls were `0.759468` /
`0.285022`, `0.698682` / `0.306130`, and `0.688684` / `0.348660` in folds
1–3. The corresponding confusion matrices, in `[S=0(-1), S=1(+1)]` order,
were `[[4793,1518],[5017,2000]]`, `[[4401,1898],[3826,1688]]`, and
`[[3816,1725],[4788,2563]]`.

The Validation AUC mean/sample standard deviation was `0.525850` /
`0.024490`. The frozen descriptive candidate screen required AUC strictly
above `0.50` in all three folds. Fold 2 did not meet that condition;
therefore `DESCRIPTIVE_CANDIDATE_SCREEN_MET = NO`.

The Fit-derived D004 gate quantities reproduced the frozen D004 records:
`(pi_minus, pi_plus, p_D_star)` were `(0.519992, 0.480008, 0.657898)`,
`(0.511699, 0.488301, 0.661508)`, and `(0.516017, 0.483983, 0.659623)` in
folds 1–3. Gate validation fractions were `0.261077`, `0.230350`, and
`0.255101`. These are population-construction facts, not sign evidence.

Interpretation boundary:

D005 did not produce consistent Discovery OOS conditional-sign discrimination
under the frozen D004-routed population, EXP_005 positional own-path
representation, and fixed Logistic Regression. This does not invalidate the
D004 Neutral-versus-Directional result or establish a reversal signal. It
does not rule out other conditional-sign representations or populations, and
it provides no confirmation, competition, ternary-utility, calibration, or
trading claim.

## D006 — Ternary Tail Intensity × Magnitude-Allocation Interaction

Status:
`COMPLETED — BOTH FROZEN CANDIDATE SCREENS FAILED — DISCOVERY ONLY`

D006 was generated after EXP_008, D004, and D005. It preserves established
recent intensity and availability information, adds one normalized signed
squared-energy allocation `O`, and compares nested multinomial C0/C1/C2
representations. The frozen continuation orientation expects positive `O` to
shift relative tail probability toward `+1` and negative `O` toward `-1`.

A fixed predicted-probability sign-reversal contrast—not raw coefficient
signs—will classify continuation, opposite, or unstable orientation. D006 is
not a D004 gate variant, a sign-dominance test, or evidence of new raw path
information.

Execution result:

One authorized execution completed across all three frozen physical Discovery
folds. All integrity assertions passed; no post-result adaptation or alternate
`O`, interaction, window, model, threshold, preprocessing, or metric was
tried. Days `353–502`, `E_holdout`, and competition-test data were untouched.

The directional-allocation screen failed. Mean MacroTailAUC was `0.606022`
for C0 and `0.603691` for C1. C1-minus-C0 deltas were `-0.003255`,
`-0.005768`, and `+0.002029`; mean `-0.002331`. Thus the frozen signed
squared-magnitude allocation `O` did not add consistent Discovery OOS
tail-ranking information beyond established intensity/availability.

The interaction screen failed. Mean C2 MacroTailAUC was `0.603488`; C2-minus-
C1 deltas were `-0.001131`, `-0.000170`, and `+0.000694`; mean `-0.000202`.
Thus the frozen `I_recent_z × O` interaction did not provide consistent
Discovery OOS evidence that directional allocation changes predictive meaning
with intensity.

The frozen probability orientation diagnostic was continuation in folds 1 and
3 and opposite in fold 2, hence overall `unstable`. The opposite Fold 2 result
is not evidence for a reversal hypothesis.

C0 remained a separate descriptive result: MacroTailAUCs were `0.611054`,
`0.604537`, and `0.602475` (mean `0.606022`); both negative- and positive-tail
AUCs exceeded `0.50` in every fold. This is further descriptive evidence that
the established intensity/availability representation contains tail or
large-movement-risk information, not directional alpha. C0 argmax `+1` recall
was zero in all three folds, a property of this frozen multinomial model and
decision rule rather than evidence that positive-tail outcomes are inherently
unpredictable.

Decision:

Under the frozen recent-window intensity representation, directional
squared-magnitude-allocation representation, and simple multinomial
interaction model, D006 found no consistent Discovery OOS evidence that
directional magnitude allocation adds to intensity or that its effect depends
on intensity. This rejects only the frozen specification and does not say that
direction is unpredictable.

## D007 — Raw Positional Returns Beyond Missingness

Status:
`COMPLETED — PRIMARY SCREEN PASS — DISCOVERY ONLY`

Scientific question:
Within physical Discovery, do `r0...r52` add consistent OOS ternary
probability information beyond `m0...m52` under the same frozen nonlinear
learner?

Technical execution history:
The first attempt ended before scientific-result generation because the
probability-coherence assertion used `atol=1e-12` on float32 XGBoost outputs.
Target-blind diagnosis found machine-epsilon-scale row-sum deviation only; no
predictive result was inspected. Human + Sol approved a dtype-aware
assertion-only repair, which passed all required tests without changing the
scientific specification. The subsequent authorized run is the first
scientifically interpretable D007 execution. Scikit-learn emitted documented
float32 probability-row-sum warnings during log-loss calculation; no adaptation
was made and the run completed.

Discovery evidence:

| Fold | log loss M | log loss R+M | delta log loss |
| ---: | ---: | ---: | ---: |
| 1 | 1.029137 | 0.999890 | +0.029247 |
| 2 | 1.045175 | 1.021698 | +0.023477 |
| 3 | 1.032531 | 1.006671 | +0.025861 |

Mean M/R+M log loss was `1.035615` / `1.009420`; mean delta was `+0.026195`
with sample SD `0.002899`. The frozen screen passed because the delta was
strictly positive in every fold and on average.

Secondary metrics were descriptive, not screen criteria, but showed the same
direction: mean MacroTailAUC `0.566113 -> 0.618777`, AUC -1 `0.563384 ->
0.607790`, AUC +1 `0.568841 -> 0.629764`, Accuracy `0.413778 -> 0.464439`,
Macro-F1 `0.224811 -> 0.410687`, and Balanced Accuracy `0.339889 ->
0.426883`. Both tail AUCs improved in every fold. Mean recalls -1/0/+1 were
`0.050036/0.967384/0.002248` for M and `0.322349/0.750523/0.207777` for R+M.

Decision and boundary:
The frozen D007 probe provides evidence of consistent incremental Discovery
OOS ternary probability information from raw return values beyond missingness
under this fixed XGBoost learner. It does not establish economic or directional
alpha, deployability, competition improvement, lockbox generalization, causal
mechanism, or material usefulness under a post-hoc threshold. Earlier
directional failures should not be summarized as evidence that the raw path
lacks directional or tail-specific information; they instead show that the
tested hand-crafted or restricted representations did not consistently extract
it. D007 does not identify which positions, signs, thresholds, interactions,
or missingness-return combinations drive the result.

No SHAP, feature importance, tuning, alternate learner, or post-result feature
engineering has been performed. No D008 is authorized. The next authorized
action is Human + Sol design review for a bounded interpretation phase of the
already-fitted frozen R+M models. Only physical Discovery was used; days
`353–502`, `E_holdout`, and competition-test data remain untouched.

## 2026-09-16 — I007A Frozen D007 Main-Effect Interpretation

I007A completed as a bounded, descriptive interpretation of the already frozen
D007 `R+M` XGBoost models. It used only physical Discovery and deterministic
2,000-row Validation samples per fold. Exact deterministic reconstruction
passed in every fold (Validation log-loss differences at or below `1.11e-16`),
as did raw-margin TreeSHAP additivity. No interaction SHAP, model change,
predictive re-evaluation, protected-region access, or competition-test access
occurred.

Main-effect TreeSHAP allocated `R_share=1.0` and `M_share=0.0` for every
output and fold. This means explicit masks received no main-effect attribution
in the fitted `R+M` model; it does not make missingness irrelevant because raw
returns retain native `NaN` values and can encode availability through native
missing-value routing. The shares are not unique-information shares.

Feature-attribution rankings were highly stable across Discovery folds. The
stable Top-10 raw-return intersections were `{-1: r33, r46, r48, r52}`,
`{0: r1, r44, r46, r47, r48, r49, r50, r51, r52}`, and
`{+1: r24, r50, r51, r52}`. Only `r52` was stable for both tail outputs.
Frozen same-output orientation summaries had consistent signs across folds
for all eligible stable tail features, though magnitudes varied, especially
for `r50` and `r51`. These are fitted-model main-effect attribution facts, not
standalone-feature evidence, causal findings, economic mechanisms, or
directional alpha.

I007A leaves competing explanations unresolved: D007 may primarily reflect
established intensity/availability, raw values may add beyond it, nonlinear
combinations may be essential, and native-NaN routing may account for some raw
attribution. No interaction analysis was performed. Any next Discovery
question requires separate Human + Sol design and freezing; no D008 is
authorized.

## 2026-09-20 — D008 Raw Path Beyond Availability + Established Recent Intensity

D008 completed one authorized physical-Discovery execution with PRIMARY SCREEN
PASS. The command `python discovery/run_d008_raw_path_beyond_availability_intensity.py`
exited `0` after approximately `406.35` seconds. All three frozen folds and
all nine artifacts completed; the contract SHA-256 was unchanged. Five
scikit-learn probability-row-sum warnings occurred during log-loss calculation,
with no error, adaptation, repair, or rerun. Days `353–502`, `E_holdout`, and
competition-test data were untouched. The missing Validation raw-`I_recent`
NaN counts in the preprocessing artifact are a reporting omission only; no
post-run calculation was authorized.

D008 compared `C=[m0,...,m52,I_recent_z]` (54 predictors) with
`P=[m0,...,m52,I_recent_z,r0,...,r52]` (107 predictors). `M` is full positional
availability, `I_recent_z` is the exact fit-only EXP_008 intensity transform,
and `R` retains untouched raw returns with native NaNs. `q`, `missing_ratio`,
SHAP-selected features, and other engineered summaries were absent.

Validation log-loss deltas `C-P` were `+0.0035748973537543804`,
`+0.0007623862799182035`, and `+0.0029766777333486427`; the mean was
`+0.002437987122340409` with sample SD `0.001481619153461506`. All were
strictly positive, so the frozen primary screen passed. Under the frozen
XGBoost learner, raw positional returns provided consistent incremental
Discovery OOS ternary predictive utility beyond the shared full-positional-
availability plus established-recent-intensity control. The residual
improvement is modest in magnitude.

Secondary descriptive means showed only small tail-ranking changes
(MacroTailAUC `0.619066 -> 0.619871`), while hard multiclass behavior changed
more visibly: `+1` recall moved `0.011209 -> 0.198979` and `-1` recall moved
`0.490856 -> 0.360556`. This is compatible with changes in ternary probability
allocation, not isolated directional/sign evidence.

D007's `M+R` versus `M` mean log-loss improvement was approximately
`+0.026195`; D008's stricter intensity-controlled residual was
`+0.002437987122340409`. This comparison is descriptive and is not an
information, variance, signal, or causal decomposition. Remaining competing
explanations include residual nonlinear class-allocation/path-shape structure,
native-NaN routing with duplicated availability, finite-sample/model-capacity
effects, and unresolved interactions. No D009 is authorized.
