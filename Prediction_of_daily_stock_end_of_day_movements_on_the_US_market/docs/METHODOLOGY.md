# Current Research Methodology

## Prediction problem
Information available:
r0 ... r52, corresponding to 09:30–14:00.

Target:
reod = direction of return from 14:00–16:00.

## Dataset structure
Training:
503 chronological days
1,829 equities
dense financial panel

Competition deployment:
different temporal period
different equity universe

## Validation protocol
See EXP_000.

Expanding walk-forward:
Fold 1: 0–302 -> 303–352
...

Fixed equity partition:
E_dev = 1463
E_holdout = 366
seed = 20260908

Primary evaluation:
Joint OOS = future days × unseen equities

Diagnostic:
Temporal OOS = future days × seen equities

## Information boundaries
ID = key only
day = partition only
equity = partition only

Competition test forbidden for research/model selection.

## Post-EXP_009 program-level research protocol

### Purpose and historical limitation

This prospective protocol separates hypothesis generation from hypothesis
confirmation at the research-program level:

```text
Discovery -> Hypothesis Freeze -> Internal Confirmation -> Final Lockbox
```

EXP_000 through EXP_009 already exposed the research program to OOS outcomes
over days 303-502. Therefore the Internal Confirmation Zone below is **not**
historically pristine or untouched. This protocol governs new work beginning
after EXP_009; it does not erase prior researcher exposure.

### Discovery

The frozen Discovery scope is:

```text
days 0-352 x E_dev
```

Discovery may use EDA, target-conditioned exploratory analysis, flexible
models, XGBoost or other exploratory learners, SHAP, conditional plots,
candidate transformations, interaction discovery, and exploratory feature
engineering. Discovery findings are hypothesis-generating only: a high
Discovery AUC or attractive SHAP pattern is not confirmatory evidence.

Discovery must not access days 353-502 for new exploratory analysis, any
`E_holdout` observations for new exploratory analysis, or competition-test
data.

Discovery now has a physically materialized authorized partition in
`data/discovery/`. Its membership is target-blind: `0 <= day <= 352` and
`equity in E_dev`, determined only from `ID`, `day`, `equity`, and the frozen
EXP_000 equity partition. The one-time parsing of the raw training files to
build that partition is controlled infrastructure access, not analytical
exposure. Ordinary Discovery analysis must read the physical partition only;
Internal Confirmation and `E_holdout` observations remain analytically
inaccessible.

### Discovery logging

Material Discovery candidates must be recorded in `docs/DISCOVERY_LOG.md`,
including failures and abandoned candidates. A material candidate is one
seriously inspected as a possible predictive mechanism, feature,
transformation, interaction, model behavior, or later confirmation candidate.
This does not require logging trivial plots, temporary debugging actions, or
implementation details. Its purpose is to preserve the researcher search path
and make researcher degrees of freedom visible.

### Freeze gate

Before a new hypothesis accesses Internal Confirmation, freeze its scientific
question, target, population, feature definitions, preprocessing, model,
hyperparameters, retraining policy, metrics, PASS/FAIL criteria, exact
confirmation blocks, and allowed interpretation. Discovery-driven redesign of
that same confirmatory test is prohibited after this gate.

### Internal Confirmation

The frozen Post-EXP_009 Internal Confirmation Zone is days 353-502, evaluated
as one pre-specified three-block procedure:

```text
Fit 0-352 -> Evaluate 353-402
Fit 0-402 -> Evaluate 403-452
Fit 0-452 -> Evaluate 453-502
```

Earlier confirmation observations may enter later fits only through this
pre-frozen expanding chronological retraining rule. Block 1 or Block 2
outcomes must not change the hypothesis, features, transformations,
preprocessing policy, model family, hyperparameters, threshold, metrics, or
PASS/FAIL criteria for that procedure. No SHAP-driven redesign may use
Internal Confirmation observations or outcomes for the same hypothesis.

```text
Temporal Confirmation = future confirmation days x E_dev
Joint Confirmation    = future confirmation days x E_holdout
```

Joint Confirmation remains the stronger internal deployment proxy because it
requires both future-time and unseen-equity generalization.

### Final lockbox

Competition-test data remain inaccessible during Discovery and Internal
Confirmation. No lockbox-opening rule is defined here. Opening it requires a
separate future Human + Sol authorization and a frozen final-stage protocol.

### Distinct research risks

- **Data leakage:** future or unavailable information enters training,
  preprocessing, feature construction, or prediction.
- **Researcher / selection leakage:** observed evaluation outcomes influence
  later scientific or model choices while those evaluations are still
  represented as confirmatory evidence.
- **Multiple testing / researcher degrees of freedom:** trying many
  hypotheses, features, or models increases the opportunity to find apparently
  strong results by chance.

Avoiding ordinary train/test leakage does not, by itself, solve
researcher-selection or multiple-testing risk.

### SHAP governance

SHAP is permitted in Discovery as an exploratory model-interpretation and
hypothesis-generation tool. It describes behavior or attributions of the
fitted exploratory model under its representation; it does not itself
establish causal or economic mechanism, independent predictive validity,
confirmatory evidence, or OOS generalization. A SHAP-discovered pattern must
become an explicitly defined hypothesis before Internal Confirmation.

## Current benchmark
Majority baseline:
Temporal OOS accuracy = 0.4147 ± 0.0092
Joint OOS accuracy = 0.4124 ± 0.0117
