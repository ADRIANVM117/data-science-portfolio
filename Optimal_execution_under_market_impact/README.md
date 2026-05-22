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

# Project Structure

```text
Optimal_execution_under_market_impact/
│
├── notebooks/
│   ├── 01_execution_schedules.ipynb
│   ├── 02_stochastic_execution.ipynb
│   ├── 03_real_data_execution_analysis.ipynb
│   └── 04_Reiforcment_Learning_execution.ipynb
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
| 04 | Market state representation and RL execution environment |

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

## Implementation Shortfall

Execution performance is evaluated through Implementation Shortfall (IS):

$$
IS = \sum_{t=1}^{T} q_t P_t^{exec} - Q P_0
$$

where:

- $Q$ = total parent order
- $P_0$ = arrival price

---

## Reinforcement Learning State Representation

The RL environment constructs state vectors containing:

$$ s_t =
[
\text{inventory ratio},
\text{time remaining},
\text{relative volume},
\text{rolling volatility},
\text{spread proxy},
\text{momentum}
]
$$

This allows the execution agent to jointly observe:

- inventory pressure
- liquidity conditions
- volatility dynamics
- market microstructure behavior

---

# Key Results

- Liquidity-aware execution significantly improves execution stability
- VWAP-like execution produces smoother participation trajectories
- Execution quality depends strongly on intraday liquidity structure
- Benchmark simulations exhibit coherent market microstructure behavior
- Smaller execution participation rates generate more realistic execution dynamics
- The resulting environment provides a suitable foundation for reinforcement learning execution research

---

# Example Outputs

## Inventory Dynamics

The framework compares inventory liquidation trajectories across execution policies:

- Random execution
- TWAP
- VWAP-like execution

under realistic market conditions.

---

## Execution Cost Dynamics

The simulator tracks cumulative execution cost across time, allowing analysis of:

- execution aggressiveness
- participation dynamics
- liquidity sensitivity
- execution stability

---

## Intraday Volume Structure

The project reconstructs intraday liquidity profiles directly from market data, capturing the classical U-shaped equity market volume curve.

---

# Research Motivation

Optimal execution is fundamentally a stochastic control problem under market impact.

This project bridges:

- stochastic processes,
- market microstructure,
- optimal control,
- reinforcement learning,
- quantitative trading research.

The long-term objective is to develop adaptive execution agents capable of learning execution policies directly from market state dynamics.

---

# Future Work

- Deep Q-Network (DQN) execution agent
- Adaptive execution policies
- Double DQN execution
- Multi-asset execution
- Limit Order Book simulation
- Transaction Cost Analysis (TCA)
- Real-time execution systems
- Execution policy learning under stochastic liquidity

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