# Intraday Leader-Follower Statistical Arbitrage Strategy

## Adrian Vazquez

---

# Overview

This project investigates whether leader-follower relationships between highly correlated Nasdaq-100 stocks can be identified statistically rather than assumed heuristically.

Traditional pair trading strategies frequently define the leader of a stock pair using market capitalization, implicitly assuming that larger companies always incorporate information faster than smaller companies.

Instead of relying on this assumption, this research proposes a statistical framework to identify empirical leader-follower relationships directly from historical intraday price dynamics.

The methodology combines:

- Same-sector pair construction
- One-minute cross-correlation analysis
- Custom Lead Score estimation
- Permutation-based statistical hypothesis testing
- Dynamic rolling correlation universe construction
- Event-driven intraday backtesting

The final objective is to determine whether statistically identified leaders produce a more informative trading universe than heuristic leader assignment.

Throughout the project, particular attention is given to avoiding look-ahead bias, survivorship bias and timestamp leakage.

---

# Research Motivation

The original trading specification assumes that, within every stock pair,

> the stock with the largest market capitalization is the information leader.

Although intuitive, this assumption is rarely verified statistically.

This project asks a different research question:

> **Can leader-follower relationships be estimated directly from historical intraday returns?**

If information consistently propagates from one stock to another, then leadership should emerge naturally from the return dynamics rather than from company size.

This transforms leader selection from a heuristic assumption into a statistical inference problem.

---

# Research Methodology

## 1. Candidate Pair Construction

The initial universe consists of Nasdaq-100 constituents.

Candidate pairs are generated only between companies belonging to the same economic sector.

Restricting pairs to the same sector reduces structural differences between companies and increases the likelihood that observed dependencies correspond to genuine information transmission rather than unrelated market movements.

---

## 2. Intraday Return Construction

For every stock,

1-minute log returns are computed as

$
r_t=\ln\left(\frac{P_t}{P_{t-1}}\right)
$

using synchronized intraday prices.

Only regular trading hours are considered.

---

## 3. Cross-Correlation Analysis

For every candidate pair $(A,B)$,

directional cross-correlations are computed at several time lags.

The primary analysis focuses on

$lag = 1\;minute $

because preliminary empirical analysis showed that most directional dependence disappears rapidly after the first minute.

Two asymmetric correlations are computed:

Forward direction

$ Corr(r_A(t),r_B(t+1)) $

Reverse direction

$ Corr(r_B(t),r_A(t+1)) $

Unlike ordinary correlation,

cross-correlation allows information propagation through time to be measured.

---

## 4. Lead Score

A custom directional metric called the **Lead Score** is defined as

$  LeadScore(A,B)= Corr(r_A(t),r_B(t+1)) - Corr(r_B(t),r_A(+1)) $

Interpretation

If

$ LeadScore>0 $

stock **A** tends to lead stock **B**.

If

$ LeadScore<0 $

stock **B** tends to lead stock **A**.

If

$ LeadScore\approx0 $

no meaningful directional relationship is detected.

This metric converts symmetric correlation into a directional measure of information leadership.

---

## 5. Permutation Test

A large Lead Score alone is insufficient evidence of a genuine leader-follower relationship.

Random correlations naturally arise in noisy financial time series.

To determine whether an observed Lead Score is statistically significant,

a non-parametric permutation test is performed.

For every candidate pair:

- One return series is randomly permuted 1,000 times.
- A Lead Score is recomputed after every permutation.
- This generates the empirical null distribution

$ LeadScore_{null} $

representing the Lead Scores expected under no directional dependence.

The empirical p-value is computed as

$ 
p=
P\left(
|LeadScore_{null}|
\ge
|LeadScore_{observed}|
\right)
$

Pairs satisfying

- p-value < 0.05
- minimum observation threshold

are retained.

All remaining pairs are discarded.

---

## 6. Empirical Leader Identification

For every statistically significant pair,

the empirical leader is defined as

$
Leader=
\begin{cases}
A,& LeadScore>0\\
B,& LeadScore<0
\end{cases}
$

The corresponding follower is assigned automatically.

The resulting empirical universe stores

- empirical leader
- empirical follower
- Lead Score
- absolute Lead Score
- p-value
- synchronized observations

forming a reproducible statistical universe.

---

## 7. Dynamic Daily Trading Universe

The validated empirical universe is then combined with a traditional rolling correlation framework.

For every trading day:

1. Compute rolling 60-day correlations.
2. Shift correlations forward one trading day to eliminate look-ahead bias.
3. Rank statistically validated pairs by rolling correlation.
4. Select the Top-50 pairs.
5. Preserve the empirical leader-follower assignments obtained from the statistical analysis.

Unlike traditional implementations,

leader assignment remains fixed according to the statistical validation while pair selection continues adapting dynamically to market conditions.

---

# Trading Strategy

The trading engine operates using the empirically validated daily universe.

For each selected pair:

- Compute 15-minute moving averages using the warm-up period.
- Apply asymmetric leader and follower edge factors.
- Generate long and short signals.
- Close positions when the leader reverts to its base SMA or at 15:55 ET.
- No overnight positions are allowed.
- Fixed notional sizing is used.
- Transaction costs are incorporated.

Importantly,

the trading rules themselves remain unchanged throughout the research.

The only modification introduced by this project is the methodology used to identify the leader and follower within each pair.

---

# Backtest Assumptions

- Universe: Nasdaq-100
- Same-sector pairs only
- Rolling correlation window: 60 trading days
- Top-50 pairs selected daily
- Fixed notional per trade: \$100,000
- Minimum stock price: \$10
- Transaction cost: \$0.0035/share
- Trading hours: 09:30–15:55 ET
- No overnight positions
- Buy-and-Hold QQQ benchmark

---

# Main Findings

## Statistical Results

The proposed Lead Score successfully identified statistically significant leader-follower relationships across the Nasdaq-100 universe.

Key findings include:

- Significant lead-lag effects were concentrated primarily at the one-minute horizon.
- Approximately 177 same-sector candidate pairs satisfied the minimum observation requirements.
- Statistical validation produced an empirical universe containing only significant leader-follower relationships.
- Roughly **60%** of empirical leaders coincided with the heuristic market-cap leader.
- Approximately **40%** of statistically significant pairs exhibited a different information leader than suggested by market capitalization.

These findings suggest that market capitalization captures leadership only partially and that intraday information transmission cannot be explained solely by firm size.

---

## Backtesting

The statistically validated universe was evaluated using the original event-driven intraday strategy.

The objective was to isolate the effect of replacing heuristic leader assignment with statistically inferred leader-follower relationships while leaving the execution logic unchanged.

Additional robustness analysis, walk-forward validation and out-of-sample evaluation remain part of the ongoing research.

---

# Repository Structure

```text
backtest/
├── 02_backtest_engine.ipynb
├── qq_test.ipynb

data/

documents/
├── research_papers/
├── strategy_description.pdf

notebooks/
├── 00_data_audit.ipynb
├── 00_universe_construction.ipynb
├── 01_hypothesis_validation.ipynb

results/

src/
├── data/
├── universe/
├── backtest/
```

---

# Future Research

Current work focuses on extending the statistical validation through

- Walk-forward validation
- Out-of-sample testing
- Parameter sensitivity analysis
- Dynamic lead-lag estimation
- Robustness across market regimes

The ultimate objective is to determine whether empirical leader identification consistently improves trading performance under realistic market conditions.

---

# Conclusion

Rather than assuming that larger companies always lead smaller ones, this project proposes a statistically grounded methodology for estimating leader-follower relationships directly from historical intraday returns.

By combining asymmetric cross-correlation, a custom Lead Score and permutation-based hypothesis testing, the project constructs an empirical trading universe in which every leader-follower assignment is supported by statistical evidence.

This transforms leader selection from a heuristic assumption into a reproducible quantitative inference procedure, providing a research framework that can be extended through robustness testing, walk-forward validation and future alpha research.