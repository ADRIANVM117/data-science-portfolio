# Optimal Execution under Market Impact

### Stochastic Execution, Market Microstructure, and Reinforcement Learning

**Adrián Vázquez**

---

## Overview

This project develops a quantitative execution research framework for studying optimal execution under market impact.

The framework combines:

- Classical execution schedules
- Stochastic execution simulation
- Real intraday market data
- Market microstructure modeling
- Reinforcement Learning execution environments
- Deep Q-Network adaptive execution agents

The project progressively evolves from analytical execution models toward adaptive execution systems capable of learning from market state dynamics.

---

## Problem

Executing large institutional orders introduces a fundamental trade-off between:

- **Market impact (execution cost)**
- **Price uncertainty (execution risk)**

Aggressive execution reduces exposure to stochastic price movements but increases market impact. Slower execution reduces impact but increases inventory risk.

Traditional execution schedules such as TWAP and VWAP provide heuristic solutions, while optimal control frameworks such as Almgren–Chriss explicitly model the cost-risk trade-off.

This project explores both classical and modern approaches to optimal execution under realistic market conditions.

---

# Core Components

## 1. Classical Execution Schedules

Implementation and comparison of:

- TWAP
- VWAP
- Almgren–Chriss

including inventory trajectories, execution prices, and implementation shortfall analysis.

---

## 2. Stochastic Execution Simulation

Execution dynamics under stochastic price evolution using:

- Arithmetic Brownian Motion
- Monte Carlo simulation
- Cost-risk decomposition
- Execution path distributions

The framework studies how volatility and market impact jointly affect execution quality.

---

## 3. Real Market Execution

Execution analysis using real intraday SPY market data sampled at 5-minute intervals.

Features include:

- Real volume dynamics
- Intraday liquidity structure
- Volume-aware execution
- Participation analysis
- Execution benchmarking

---

## 4. Reinforcement Learning Execution Environment

Development of a reinforcement learning environment for adaptive execution.

The environment includes:

- Inventory-aware state representations
- Market microstructure features
- Participation-based market impact
- Benchmark execution policies
- Sequential execution dynamics

This framework serves as the foundation for training Deep Q-Network execution agents.

---

## 5. Deep Reinforcement Learning Execution Agent

Development and training of a Deep Q-Network (DQN) execution agent capable of learning adaptive execution policies from market state observations.

The RL framework includes:

- Replay Buffer experience storage
- Epsilon-greedy exploration
- Target network stabilization
- State-dependent execution decisions
- Dynamic participation control
- Inventory-aware execution learning

The agent is evaluated against:

- Random execution
- TWAP
- VWAP-like execution
- Volume-Aware Almgren–Chriss

under realistic intraday market conditions.

---

# Key Findings

The Reinforcement Learning execution agent successfully learned a coherent execution policy under realistic intraday market conditions.

Main observations:

- The DQN agent converged toward a strongly front-loaded execution strategy.
- The learned policy aggressively reduced inventory during the early trading horizon.
- Participation intensity dynamically decreased as inventory exposure declined.
- The learned execution trajectory remained smooth and stable despite exploration-based training.
- The DQN agent achieved the lowest cumulative execution cost among benchmark policies during the evaluated trading session.
- The agent learned adaptive state-dependent execution behavior rather than a static deterministic schedule.

The results suggest that Reinforcement Learning can discover meaningful execution policies under market microstructure constraints.

---

# Project Structure

```text
Optimal_execution_under_market_impact/
│
├── notebooks/
│   ├── 01_execution_schedules.ipynb
│   ├── 02_stochastic_execution.ipynb
│   ├── 03_real_data_execution_analysis.ipynb
│   ├── 04_volume_aware_execution.ipynb
│   ├── 05_RL_Execution_agent.ipynb
│   └── 06_out_of_sample_generalization.ipynb
│
├── results/
│   ├── plots/
│   ├── tables/
│   ├── reports/
│   └── papers/
│
├── src/
│   ├── models/
│   ├── simulation/
│   ├── analytics/
│   └── rl/
│
├── requirements.txt
└── README.md
```

---

# Notebook Roadmap

| Notebook | Description |
|---|---|
| 01 | Classical execution schedules: TWAP, VWAP, Almgren–Chriss |
| 02 | Stochastic execution simulation under Brownian dynamics |
| 03 | Real intraday execution analysis using SPY 5-minute data |
| 04 | Volume-aware execution and market microstructure modeling |
| 05 | Deep Reinforcement Learning execution agent |
| 06 | Out-of-sample generalization and robustness analysis |

---

# Methodology

## Price Dynamics

Price evolution is modeled using Arithmetic Brownian Motion:

$$
dS_t = \sigma dW_t
$$

In discrete time:

$$
S_{k+1} = S_k + \sigma \sqrt{\tau} \epsilon_k,
\quad
\epsilon_k \sim \mathcal{N}(0,1)
$$

This captures stochastic price uncertainty during the execution horizon.

---

## Market Impact Model

Execution prices incorporate temporary market impact:

$$
P_t^{exec} =
P_t
+
\eta
\left(
\frac{q_t}{V_t}
\right)
$$

where:

- $P_t$ = observed market price
- $q_t$ = executed shares
- $V_t$ = market volume
- $\eta$ = impact intensity parameter

The framework studies how participation rate affects execution quality.

---

## Reinforcement Learning State Representation

The execution agent observes both execution state and market state variables:

$$
s_t =
[
x_t,
T-t,
\text{market features}
]
$$

where the state includes:

- remaining inventory,
- remaining execution horizon,
- relative volume,
- rolling volatility,
- momentum,
- participation proxies,
- and liquidity-related features.

---

## Deep Q-Learning

The DQN agent learns execution policies through sequential interaction with the execution environment.

At each timestep:

$$
(s_t, a_t, r_t, s_{t+1})
$$

transitions are stored in a Replay Buffer and used to optimize the Q-network.

The learned policy dynamically adapts execution intensity according to observed market conditions.

---

# Future Work

Future extensions include:

- Out-of-sample evaluation across multiple trading sessions
- Regime-aware execution analysis
- Reward engineering experiments
- Advanced market microstructure features
- Robustness testing under volatile market conditions
- Continuous-action RL execution frameworks
- Multi-asset execution environments

---

# Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Plotly
- Reinforcement Learning
- Alpha Vantage API

---

# Research Focus

This project focuses on the intersection of:

- Quantitative Finance
- Optimal Execution
- Market Microstructure
- Reinforcement Learning
- Algorithmic Trading
- Stochastic Processes

---

# References

- Almgren, R., & Chriss, N. (2001). *Optimal Execution of Portfolio Transactions.*

- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading.*

- Ning, B., Treichler, D., & Chen, S. (2021). *Double Deep Q-Learning for Optimal Execution.* Applied Mathematical Finance.

---

# Author

**Adrián Vázquez**

Actuary | Data Scientist | Quantitative Finance & Machine Learning

Focused on:
- Optimal execution
- Quantitative trading
- Reinforcement learning in finance
- Market microstructure
- Stochastic modeling