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
