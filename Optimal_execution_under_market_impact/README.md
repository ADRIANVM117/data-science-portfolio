
# Optimal Execution under Market Impact  
### A Comparative Study of TWAP, VWAP, and Almgren–Chriss  
**Adrián Vázquez**

---

## Problem

Executing large orders in financial markets introduces a fundamental trade-off between:

- **Market impact** (execution cost)
- **Price uncertainty** (execution risk)

Naive execution strategies such as **TWAP** and **VWAP** do not explicitly optimize this trade-off, potentially leading to suboptimal outcomes under stochastic price dynamics.

---

## Objective

This project develops a **stochastic optimal execution framework** to:

- Compare TWAP, VWAP, and Almgren–Chriss strategies  
- Quantify execution performance under market impact  
- Analyze the **cost–risk trade-off** via Monte Carlo simulation  
- Identify regimes where optimal execution improves performance  

---

## Project Structure

```text
optimal-execution/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── models/          # TWAP, VWAP, Almgren–Chriss
│   ├── simulations/     # Brownian motion, execution engine
│   └── analytics/       # IS, cost, variance
│
├── notebooks/
│   ├── 01_execution_schedules.ipynb
│   └── 02_stochastic_execution.ipynb
│
└── results/
    ├── plots/
    └── tables/


```
=======

## Methodology

### Price Dynamics

We model price evolution using an Arithmetic Brownian Motion:

$$
dS_t = \sigma dW_t
$$

In discrete time:

$$
S_{k+1} = S_k + \sigma \sqrt{\tau} \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0,1)
$$

This captures stochastic price uncertainty during the execution horizon.

---

### Market Impact Model

Execution prices incorporate both permanent and temporary impact:

$$
P_k = S_k + \gamma \sum_{j<k} n_j + \eta \frac{n_k}{\tau}
$$

Where:

- $S_k$ = mid price  
- $P_k$ = execution price  
- $\gamma$ = permanent impact  
- $\eta$ = temporary impact  
- $n_k$ = shares executed at time $k$  
- $\tau$ = time step  

**Interpretation:**

- Permanent impact accumulates as trading progresses  
- Temporary impact penalizes aggressive execution  

---

### Execution Strategies

- **TWAP** — uniform execution over time  
- **VWAP** — proportional to expected volume  
- **Almgren–Chriss** — optimal control balancing cost and risk  

---

## Performance Metrics

### Implementation Shortfall (IS)

$$
\text{IS} = \sum_{k=1}^{N} n_k P_k - Q S_0
$$

Where:

- $Q$ = total shares  
- $S_0$ = initial price  

---

### Expected Cost

$$
\mathbb{E}[\text{Cost}] = \gamma \sum_{k=1}^{N} n_k X_{k-1} + \eta \sum_{k=1}^{N} \frac{n_k^2}{\tau}
$$

Where:

$$
X_{k-1} = \sum_{j<k} n_j
$$

---

### Execution Risk

$$
\text{Var}[\text{Cost}] = \sigma^2 \tau \sum_{k=1}^{N} x_k^2
$$

Where:

- $x_k$ = remaining inventory  

---

## Experiments

We simulate execution under stochastic price dynamics using Monte Carlo:

- Generate multiple price paths  
- Apply execution strategies  
- Compute distributions of outcomes  

**Key parameters:**

- Volatility $ \sigma $  
- Market impact $(\gamma, \eta)$  
- Risk aversion $ \lambda $  

---

## Results

### Key Findings

- Stochastic price dynamics introduce variability in execution cost  
- Market impact shifts execution prices above mid price  
- VWAP shows higher cost and heavier tail risk  
- Almgren–Chriss provides a tunable cost–risk trade-off  

---

### Cost-Risk Trade-off

By varying $\lambda$, we trace a cost-risk curve:

- Low $\lambda$ → low cost, high risk  
- High $\lambda$ → high cost, low risk  
- Intermediate $\lambda$ → optimal balance  

---

## Key Insight

Execution strategies must be evaluated across the full distribution of outcomes — not just expected cost.

A slightly higher expected cost may be justified if it significantly reduces execution risk.

---

## Conclusion

Optimal execution is fundamentally a stochastic control problem under market impact.

The Almgren–Chriss framework provides a systematic way to balance:

- Market impact (cost)  
- Price uncertainty (risk)  

making it a core tool in quantitative trading and execution research.

---

## Future Work

- Calibration with real market data  
- Dynamic volume profiles  
- Multi-asset execution  
- Reinforcement learning for adaptive execution strategies  