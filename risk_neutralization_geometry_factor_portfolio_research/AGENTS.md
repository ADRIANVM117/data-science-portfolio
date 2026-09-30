# AGENTS.md

## Project

`risk_neutralization_geometry_factor_portfolio_research`

This repository implements a research experiment connecting
characteristic-based portfolio representation with geometric
factor decomposition.

The authoritative research specification is:

`docs/METHODOLOGY.md`

All agents MUST read that document before modifying research code,
data transformations, experiments, or documentation.

If implementation convenience conflicts with the methodology,
the methodology takes precedence.

---

## Current research stage

| Stage | Status |
| --- | --- |
| Stage 0 — Data Feasibility | **CLOSED** |
| Stage 1A — Characteristic Prerequisite Resolution | **CLOSED** |
| Stage 1B — Timing and Representation Conventions | **CLOSED** |
| Stage 1C — Characteristic Specification | **CLOSED** |
| Stage 2 — Freeze universe | **NEXT — NOT STARTED** |

### Stage 1C record

Stage 1C does **not** attempt to discover, optimize, or prove the globally
best Value, Quality, Growth, or Momentum definition. Its narrow purpose is to
operationalize one economically defensible, point-in-time feasible,
transparent, and reproducible specification for each previously defined
characteristic concept, sufficient for the downstream geometric portfolio
experiment.

A Stage 1C characteristic specification may be frozen only when it is:

- economically coherent with the previously defined concept;
- constructible under the frozen point-in-time conventions;
- reproducible and auditable;
- free of an obvious defect that would invalidate the experiment; and
- sufficiently distinct from the other selected characteristics for this
  experiment.

Do **not** select a specification based on future portfolio returns, Sharpe
ratio, backtest performance, downstream geometric results, or an exhaustive
search across candidate definitions.

Characteristic construction is an input to the main experiment, not the main
research question:

```text
X_char
  -> w_raw
  -> F_risk
  -> risk subspace S
  -> geometric decomposition
  -> economic interpretation
```

The stage froze seven raw components under Value, Quality, Growth, and
Momentum. The eventual contents of `F_risk` remain **PENDING**; Stage 1C did
not freeze risk exposures, `X_char`, `g(X)`, weights, or geometric results.

---

## Current status and boundaries

Stage 1C is complete. Stage 2 — Freeze universe is the next research stage,
but it has **not** begun and requires explicit research authorization.

The following records the permissions that applied during Stage 1C:

- operationalize and document a candidate specification only after it satisfies
  the Stage 1C decision criterion;
- inspect already-audited raw-field provenance, point-in-time eligibility,
  coverage, and economically relevant edge cases;
- compare the already documented candidate representations without using
  downstream performance; and
- create transparent research documentation and, when separately authorized,
  small reusable validation or implementation components.

No current authorization exists to define the universe, construct `X_char`,
choose normalization or missing/outlier treatment, define `g(X)`, construct
weights or `F_risk`, or perform a geometric result.

---

## Explicitly forbidden during Stage 1C

Do NOT:

- construct the final characteristic matrix;
- calculate or implement a characteristic before its specification is frozen
  through the documented research process;
- construct portfolio weights;
- define \(g(X)\);
- construct \(F_{\mathrm{risk}}\);
- perform factor neutralization;
- optimize a portfolio;
- introduce Markowitz mean-variance optimization;
- introduce Black-Litterman;
- construct an efficient frontier;
- maximize Sharpe ratio;
- train return-prediction models;
- perform ML or reinforcement learning;
- backtest a trading strategy;
- infer alpha from observed returns;
- silently expand the research scope.

In particular, do not use future portfolio returns, Sharpe ratio, backtest
performance, downstream geometric results, or exhaustive candidate search to
choose a characteristic definition.

If a potentially useful extension is discovered, document it as a
proposal rather than implementing it.

---

## Research objects

Maintain the conceptual distinction between:

### Characteristic matrix

\[
X_{\mathrm{char},t}
\]

This will eventually describe companies using economically
interpretable characteristics.

### Risk-exposure matrix

\[
F_{\mathrm{risk},t}
\]

This will eventually represent selected exposure directions.

These objects MUST NOT be silently merged or treated as equivalent.

Neither object is constructed during Stage 1C.

---

## Data provider

The planned primary provider is Alpha Vantage.

API credentials MUST NOT be:

- committed to Git;
- written into notebooks;
- written into source files;
- written into documentation;
- printed in logs.

Use environment variables or an equivalent secrets mechanism.

A `.env` file, if used locally, MUST be excluded through `.gitignore`.

Provide `.env.example` only with placeholder values.

---

## Raw data policy

Raw provider responses must remain distinguishable from derived data.

Never overwrite raw data with transformed data.

Suggested conceptual separation:

`data/raw/`

`data/interim/`

`data/processed/`

During Stage 0, prefer preserving small raw audit samples when useful
for reproducibility.

Do not commit large provider datasets unless explicitly approved.

---

## Point-in-time discipline

This is a critical research requirement.

Do not assume:

- fiscal period end date;
- API observation timestamp;
- earnings announcement date;
- SEC filing date;
- data availability date

are equivalent.

For every fundamental dataset, identify which timestamps are actually
provided and what they mean.

Where the provider does not expose enough information to establish
historical availability, explicitly mark the limitation.

Never manufacture an availability date.

Never use future information to fill earlier observations.

Never silently forward-fill fundamental information across unknown
publication boundaries.

---

## Characteristic governance

During Stage 1C, Value, Quality, Growth, and Momentum remain candidate
economic families until each individual specification is explicitly frozen.

Do not decide that, for example,

`Value = 1 / PE`

simply because the API provides a PE ratio.

Candidate fields may be documented.

Definitions require a later research decision.

For every candidate field record, where possible:

- provider endpoint;
- provider field name;
- economic meaning;
- data type;
- frequency;
- historical availability;
- relevant timestamp fields;
- missingness;
- obvious data-quality issues;
- potential characteristic family.

---

## Missing data

Missing values MUST remain missing during the audit unless a
transformation is explicitly required for technical parsing.

Do NOT silently:

- replace missing values with zero;
- median-impute;
- mean-impute;
- forward-fill;
- backward-fill;
- drop companies;
- drop periods.

First measure and document missingness.

---

## Outliers and transformations

Do NOT silently:

- winsorize;
- clip;
- standardize;
- z-score;
- rank-transform;
- log-transform;
- normalize cross-sectionally.

These are research decisions that belong to later stages.

Raw values should remain auditable.

---

## Numerical implementation

When the project eventually reaches geometric decomposition, avoid
explicit matrix inversion when a numerically stable equivalent is
available.

However, Stage 0 does not implement the projection.

The theoretical formulation remains documented in
`docs/METHODOLOGY.md`.

---

## Research governance

Do not silently change frozen research decisions.

If evidence suggests the methodology should change:

1. document the finding;
2. explain why it matters;
3. propose the change;
4. wait for research approval before implementing it.

Distinguish clearly between:

- CONFIRMED;
- OBSERVED;
- ASSUMED;
- PENDING.

Do not convert an assumption into a confirmed fact.

---

## Research Notebook Governance

### Dual-layer workflow

For economically meaningful research transformations, preserve the following
auditable path:

```text
research question
    -> transparent notebook investigation
    -> methodological validation / frozen decision
    -> reusable Python implementation
    -> notebook <-> implementation consistency check
```

Notebooks and Python modules have distinct responsibilities. This workflow does
not replace reusable Python with notebooks.

### Notebook responsibility

Notebooks are the auditable research narrative. Where relevant, they SHOULD
make visible the research question, frozen assumptions, source datasets, small
dataframe samples, point-in-time eligibility, intermediate calculations,
missingness, distributions, diagnostic plots, economically interesting edge
cases, candidate comparisons, interpretation, and the research conclusion.

A researcher should be able to follow the path from raw information to a
research object without reverse-engineering a large Python module. Prefer a
research record using Markdown between computational sections:

```text
question -> data -> calculation -> diagnostic -> interpretation
```

Avoid a giant cell that contains an entire pipeline.

### Python responsibility

Reusable `.py` code remains the authoritative implementation layer for
acquisition, parsing, deterministic transformations, reusable feature
functions, validation utilities, and research/production pipelines. Do NOT
move large reusable implementations into notebooks.

### No hidden economic transformations

An economically material transformation must not exist only inside a large,
opaque Python function. Its logic must be explainable and inspectable in the
corresponding research notebook.

### Notebook/implementation consistency

After a research transformation is frozen and implemented in reusable Python,
the corresponding notebook SHOULD verify that a transparent/manual calculation
agrees with that implementation. Use deterministic numerical assertions when
feasible, for example `np.testing.assert_allclose(...)`.

The notebook is not a second independent production implementation.

### Research governance still applies

Notebooks are not unrestricted experimentation environments. In particular,
they MUST NOT:

- test many definitions and select one using future returns, Sharpe, or
  backtest performance;
- hide whether an item was exploratory or frozen before outcomes were seen;
- violate point-in-time discipline;
- silently impute, winsorize, clip, normalize, rank, z-score, or drop
  observations; or
- omit an explicitly approved transformation from the narrative.

Existing methodology, data-audit findings, and frozen research decisions remain
authoritative. Notebook scaffolds must mark any unapproved item as
`PENDING - requires research decision`.

---

## Code principles

Prefer:

- small functions;
- explicit schemas;
- deterministic transformations;
- type hints where useful;
- testable components;
- clear naming;
- minimal dependencies.

Avoid premature architecture.

Do not build abstractions for hypothetical future requirements.

Do not build a complete data platform during Stage 0.

The smallest implementation that can answer the research question is
preferred.

---

## Tests

Tests should validate behavior, not merely execute code.

For Stage 0, useful tests may include:

- API response parsing;
- schema expectations;
- timestamp parsing;
- preservation of missing values;
- prevention of accidental credential exposure;
- deterministic storage of raw samples.

Do not create tests for future portfolio functionality that does not
yet exist.

---

## Stage 0 deliverable

The primary research deliverable is:

`docs/DATA_AUDIT.md`

It should answer, for each relevant Alpha Vantage endpoint:

1. What data does it provide?
2. What raw fields are potentially useful?
3. What history is available?
4. What timestamps are provided?
5. What do those timestamps appear to represent?
6. What is missing?
7. What point-in-time limitations exist?
8. Which characteristic families could the data potentially support?
9. What must still be verified before defining characteristics?

The audit must distinguish observations from assumptions.

---

## Stop condition

Stage 0 is complete only when there is enough evidence to decide
whether the available data can support a defensible

\[
X_{\mathrm{char},t}.
\]

At that point, STOP.

Do not automatically proceed to Stage 1.

Stage 1 requires explicit research review and approval.
