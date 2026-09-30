# 03 — Experiment: From Characteristics to Geometry

**Status:** Research design — v0.1  
**Series:** Portfolio Geometry  
**Previous:**  
- 01 — Geometric interpretation of factor neutralization
- 02 — Numerical implementation of orthogonal projection

---

## 1. Motivation

Experiments 01 and 02 established the geometric interpretation of
factor neutralization.

Given a set of exposure directions collected in

\[
F_{\mathrm{risk}} \in \mathbb{R}^{N \times M},
\]

their column space defines

\[
S_{\mathrm{risk}}
=
\operatorname{Col}(F_{\mathrm{risk}}).
\]

A portfolio can then be decomposed as

\[
w_{\mathrm{raw}}
=
P_S w_{\mathrm{raw}}
+
(I-P_S)w_{\mathrm{raw}},
\]

where

\[
P_S w_{\mathrm{raw}}
\]

is the component aligned with the selected exposure space and

\[
(I-P_S)w_{\mathrm{raw}}
\]

is its orthogonal component.

Experiments 01–02 demonstrated this with synthetic exposures.

Experiment 03 moves to real equity data.

The objective is not merely to neutralize a portfolio.

The objective is to study the relationship between:

1. economically interpretable company characteristics;
2. the portfolio constructed from those characteristics;
3. selected known exposure directions;
4. the geometry induced by those directions.

---

## 2. Research question

The central question is:

> How much of the structure of a quantamental portfolio is independent
> of selected known risk exposures?

Equivalently, if a portfolio is constructed to favor characteristics
such as Value, Quality, Growth or Momentum, we want to determine how
much of that portfolio lies inside a selected risk-exposure subspace.

This leads to the decomposition

\[
w_{\mathrm{raw}}
=
w_{\parallel}
+
w_{\perp},
\]

with

\[
w_{\parallel}=P_S w_{\mathrm{raw}},
\]

and

\[
w_{\perp}=(I-P_S)w_{\mathrm{raw}}.
\]

The experiment then asks:

> Which characteristics of the original portfolio survive after the
> selected risk directions are removed?

---

## 3. Conceptual bridge

This experiment combines two ideas.

### Characteristic representation

Companies are represented using economically interpretable
characteristics rather than only historical return moments.

For a cross-section of \(N\) stocks at time \(t\),

\[
X_t \in \mathbb{R}^{N \times K}.
\]

Rows represent companies.

Columns represent characteristics.

Candidate characteristic families include:

\[
X_t =
[
\text{Value},
\text{Quality},
\text{Growth},
\text{Momentum},
\ldots
].
\]

The exact definitions of these characteristics are NOT yet frozen.

They will be determined only after the underlying data fields,
timestamps and availability have been audited.

### Geometric representation

A separate matrix

\[
F_{\mathrm{risk},t}
\in
\mathbb{R}^{N \times M}
\]

contains selected exposure directions that we want to study or control.

These directions generate

\[
S_{\mathrm{risk},t}
=
\operatorname{Col}(F_{\mathrm{risk},t}).
\]

The distinction between

\[
X_{\mathrm{char},t}
\]

and

\[
F_{\mathrm{risk},t}
\]

must be preserved throughout the experiment.

They represent different research objects.

---

## 4. Characteristic matrix

The target object has the form

\[
X_{\mathrm{char},t}
=
\begin{bmatrix}
x_{11} & \cdots & x_{1K}\\
x_{21} & \cdots & x_{2K}\\
\vdots &        & \vdots\\
x_{N1} & \cdots & x_{NK}
\end{bmatrix}.
\]

A conceptual example is:

| Stock | Value | Quality | Growth | Momentum |
|---|---:|---:|---:|---:|
| AAPL | ... | ... | ... | ... |
| MSFT | ... | ... | ... | ... |
| NVDA | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

Initial candidate families:

- Value
- Quality
- Growth
- Momentum

Possible additional characteristics remain outside the frozen
specification until justified.

### Important

The data provider does not define the economic characteristic.

For example, receiving a P/E ratio from an API does not automatically
define our Value characteristic.

For every final characteristic we must document:

1. economic interpretation;
2. mathematical definition;
3. raw fields required;
4. transformation;
5. cross-sectional normalization;
6. missing-value policy;
7. outlier policy;
8. observation date;
9. information-availability date.

---

## 5. Quantamental portfolio

The characteristic matrix will eventually generate a transparent
portfolio:

\[
w_{\mathrm{raw}}
=
g(X_{\mathrm{char},t}).
\]

The function \(g(\cdot)\) is currently UNDEFINED.

It must be specified before evaluating portfolio results.

Requirements:

- economically interpretable;
- deterministic;
- reproducible;
- defined before observing experimental outcomes;
- independent of mean-variance optimization.

The experiment must not silently modify \(g(\cdot)\) after observing
the geometric decomposition.

---

## 6. Risk-exposure matrix

Selected known exposures will be represented through

\[
F_{\mathrm{risk},t}
=
[
f_1,
f_2,
\ldots,
f_M
].
\]

Potential examples include:

- sector exposures;
- size;
- market exposure;
- rates sensitivity;
- commodity sensitivity.

These are candidates, not yet the final specification.

Every included exposure must have:

1. an economic interpretation;
2. a reproducible construction rule;
3. a documented data source;
4. a timestamp convention.

The experiment must not classify a direction as a risk exposure merely
because removing it improves subsequent portfolio performance.

---

## 7. Geometric experiment

Once \(F_{\mathrm{risk},t}\) is frozen, define

\[
S_{\mathrm{risk},t}
=
\operatorname{Col}(F_{\mathrm{risk},t}).
\]

For a full-column-rank exposure matrix, the theoretical orthogonal
projector is

\[
P_S
=
F_{\mathrm{risk}}
(F_{\mathrm{risk}}^\top F_{\mathrm{risk}})^{-1}
F_{\mathrm{risk}}^\top.
\]

Numerical implementation may use a more stable equivalent method.

We then calculate

\[
w_{\parallel}
=
P_S w_{\mathrm{raw}},
\]

and

\[
w_{\perp}
=
(I-P_S)w_{\mathrm{raw}}.
\]

Therefore,

\[
w_{\mathrm{raw}}
=
w_{\parallel}
+
w_{\perp}.
\]

The experiment studies both components.

The objective is NOT simply to obtain a neutral portfolio.

---

## 8. Primary diagnostics

### 8.1 Exposure removal

Compare

\[
F_{\mathrm{risk}}^\top w_{\mathrm{raw}}
\]

against

\[
F_{\mathrm{risk}}^\top w_{\perp}.
\]

The second quantity should be numerically close to zero for the
represented directions.

### 8.2 Portfolio decomposition

Measure the relative magnitude of

\[
w_{\parallel}
\]

and

\[
w_{\perp}.
\]

This tells us how strongly the original portfolio is geometrically
aligned with the selected risk subspace.

### 8.3 Characteristic preservation

Compare

\[
X_{\mathrm{char}}^\top w_{\mathrm{raw}}
\]

with

\[
X_{\mathrm{char}}^\top w_{\perp}.
\]

This is one of the central diagnostics of Experiment 03.

We want to determine which economic characteristics survive the
projection and which were strongly entangled with the selected
risk directions.

### 8.4 Structural change

Additional descriptive diagnostics may include:

- distance between portfolios;
- changes in individual weights;
- sign changes;
- concentration;
- gross exposure;
- net exposure;
- induced turnover.

These diagnostics describe structural consequences.

They are not, by themselves, evidence of alpha destruction.

---

## 9. Interpretation of alpha

Experiment 03 must distinguish between:

- characteristic exposure;
- risk exposure;
- portfolio structure;
- realized subsequent performance;
- alpha.

A large change in

\[
w_{\mathrm{raw}}
\rightarrow w_{\perp}
\]

does NOT imply that alpha was destroyed.

Likewise, preserving a characteristic does not establish that the
characteristic generates alpha.

Any claim about predictive performance requires a separately defined
out-of-sample test.

The first objective of Experiment 03 is geometric and structural.

---

## 10. Data source

Primary planned provider:

**Alpha Vantage Premium — 150 requests/minute plan.**

The current subscription gives sufficient capacity for the intended
research workflow, but API availability does not determine the
methodology.

Candidate required datasets:

- daily adjusted equity prices;
- company overview / classification;
- income statement;
- balance sheet;
- cash-flow statement;
- earnings;
- shares outstanding.

The exact endpoint-to-field mapping will be documented during the
data-feasibility phase.

Raw provider fields must remain distinguishable from derived research
features.

---

## 11. Point-in-time discipline

Fundamental information must not be treated as available before it
could reasonably have been known by the researcher.

Where applicable, preserve separately:

\[
t_{\mathrm{period}}
\]

the accounting period represented by the observation, and

\[
t_{\mathrm{available}}
\]

the date at which the information became available.

Portfolio characteristics at time \(t\) may only use information with

\[
t_{\mathrm{available}} \leq t.
\]

No forward filling across unknown publication dates may silently create
information that was unavailable at the time.

Exact Alpha Vantage timestamp behavior must be audited before this rule
is operationalized.

---

## 12. Experiment stages

### Stage 0 — Data feasibility

Do NOT begin with the complete stock universe.

Use a small set of approximately 3–5 heterogeneous companies.

Objectives:

- inspect API responses;
- identify available raw fields;
- understand timestamp semantics;
- inspect missingness;
- verify historical depth;
- determine whether point-in-time characteristics can be constructed;
- create a field dictionary.

Output:

`DATA_AUDIT.md`

No portfolio construction is required at this stage.

---

### Stage 1 — Freeze characteristic definitions

Using the feasibility audit, define the initial characteristic set.

For every characteristic freeze:

\[
\text{economic idea}
\rightarrow
\text{raw fields}
\rightarrow
\text{formula}
\rightarrow
\text{transformation}
\rightarrow
\text{cross-sectional score}.
\]

Output:

`CHARACTERISTICS.md`

---

### Stage 2 — Freeze universe

Define:

- inclusion criteria;
- exclusion criteria;
- liquidity requirements if applicable;
- treatment of missing fundamentals;
- sector coverage;
- universe date.

The target size is expected to be approximately 50–100 US equities,
subject to the feasibility results.

Output:

`UNIVERSE.md`

---

### Stage 3 — Construct \(X_{\mathrm{char}}\)

Build and validate the real characteristic matrix.

No geometric result should be interpreted before this matrix passes
data-quality and point-in-time checks.

---

### Stage 4 — Define \(g(X)\)

Freeze the deterministic mapping

\[
w_{\mathrm{raw}}=g(X_{\mathrm{char}}).
\]

Output:

`PORTFOLIO_RULE.md`

---

### Stage 5 — Construct \(F_{\mathrm{risk}}\)

Define and validate the selected exposure directions.

Output:

`RISK_EXPOSURES.md`

---

### Stage 6 — Geometric decomposition

Calculate

\[
w_{\parallel}=P_Sw_{\mathrm{raw}}
\]

and

\[
w_{\perp}=(I-P_S)w_{\mathrm{raw}}.
\]

Validate the relevant projection identities numerically.

---

### Stage 7 — Interpret results

Study:

\[
X^\top w_{\mathrm{raw}}
\quad\text{vs.}\quad
X^\top w_{\perp},
\]

together with the structural diagnostics.

Only after the frozen experiment is complete should extensions or
performance hypotheses be proposed.

---

## 13. Out of scope

Experiment 03 is NOT a general portfolio-optimization project.

Unless explicitly introduced in a future experiment, the following are
outside scope:

- Markowitz mean-variance optimization;
- Black-Litterman;
- efficient frontier construction;
- minimum-variance allocation;
- Sharpe-ratio optimization;
- covariance-based allocation as the primary portfolio mechanism;
- return-prediction ML;
- reinforcement learning;
- execution optimization;
- intraday trading signals;
- transaction-cost optimization;
- portfolio constraints as the primary research objective.

These methods are not rejected generally.

They simply do not answer the research question of Experiment 03.

---

## 14. Research governance

The following decisions must be frozen before their corresponding
results are observed:

- characteristic definitions;
- universe rules;
- portfolio construction rule \(g(\cdot)\);
- risk-exposure definitions;
- transformations;
- missing-value treatment;
- outlier treatment;
- normalization rules;
- primary diagnostics.

Changes made after observing results must be logged as new research
iterations rather than silently replacing the original specification.

---

## 15. Current state

### CONFIRMED

- Experiment 03 uses real equity data.
- The experiment is characteristic-based.
- \(X_{\mathrm{char}}\) and \(F_{\mathrm{risk}}\) are distinct objects.
- The geometric decomposition from Experiments 01–02 is preserved.
- Alpha Vantage is the planned primary data provider.
- Mean-variance / Black-Litterman optimization is outside scope.
- The initial objective is structural and geometric, not predictive.
- Stage 1C characteristic specification is **CLOSED** with seven raw
  components: EarningsYield; Profitability; CashRealization; RevenueCAGR_3Y;
  PositiveAnnualGrowthFraction_3Y; R_12M_AdjustedClose; and R_1M_AdjustedClose.

### PENDING

- exact stock universe;
- normalization method;
- function \(g(X)\);
- final \(F_{\mathrm{risk}}\);
- point-in-time feasibility of each fundamental field;
- missing-data policy;
- outlier policy.

---

## 16. Immediate next step

Stage 1C — Characteristic Specification is **CLOSED**. The next research
stage defined by this methodology is **Stage 2 — Freeze universe**.

Stage 2 has **not** started. It requires explicit research authorization and
must first define only the universe protocol. This status does not authorize
construction of \(X_{\mathrm{char}}\), normalization, missing-value or outlier
treatment, \(w_{\mathrm{raw}}\), \(g(X)\), \(F_{\mathrm{risk}}\), geometric
decomposition, or backtests.

The seven frozen raw characteristic components are inputs to that later work,
not an already-constructed characteristic matrix.
