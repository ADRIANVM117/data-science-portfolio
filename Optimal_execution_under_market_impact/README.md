# Optimal Execution under Market Impact  
### A Comparative Study of TWAP, VWAP, and Almgren–Chriss
### Adrián Vázquez
---

##  Problem

Executing large orders in financial markets introduces a fundamental trade-off between:

- **Market impact** (execution cost)
- **Price uncertainty** (execution risk)

Naive execution strategies such as TWAP and VWAP do not explicitly account for this trade-off, potentially leading to suboptimal performance under varying market conditions.

---

##  Objective

This project develops a stochastic execution framework to:

- Compare TWAP, VWAP, and Almgren–Chriss strategies  
- Quantify execution performance under market impact  
- Identify regimes where optimal execution improves cost–risk efficiency  

---

##  Methodology

### Price Dynamics

\[
dS_t = \sigma dW_t
\]

---

### Market Impact Model

\[
P_k = S_k + \gamma \sum_{j<k} n_j + \eta \frac{n_k}{\tau}
\]

Where:
- \( \gamma \): permanent impact  
- \( \eta \): temporary impact  
- \( n_k \): executed shares  

---

### Execution Strategies

- **TWAP** — uniform execution  
- **VWAP** — volume-weighted execution  
- **Almgren–Chriss** — optimal control balancing cost and risk  

---

##  Performance Metrics

- Implementation Shortfall (IS)  
- Expected Execution Cost  
- Execution Risk (Variance / Std Dev)  

---

##  Experiments

We evaluate execution performance under different market regimes:

- Volatility (\( \sigma \))  
- Market impact (\( \gamma, \eta \))  
- Execution urgency (\( \lambda \))  

---

##  Results

*(Insert plots here)*

Example insights:

- Almgren–Chriss reduces execution risk under high volatility  
- TWAP converges to optimal execution when \( \lambda \to 0 \)  
- Higher execution urgency leads to faster but more expensive trades  

---

##  Key Insight

> Optimal execution is not universal — the best strategy depends on market conditions, liquidity, and execution urgency.

---

##  Project Structure
### Project Structure

```text
src/
├── models/         # Model logic (Almgren-Chriss implementation)     
├── simulation/    # Stochastic price dynamics (Brownian Motion)
├── analytics/ 
├── notebooks/
└── results/       # Sensitivity analysis plots and CSVs

```