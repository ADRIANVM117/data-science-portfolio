# V3 — Regime-Conditioned Alpha Discovery

## Research Question

Can Bayesian mispricing become informative once conditioned on market regime?

---
Results from V1 showed that Bayesian fair value estimation did not generate consistent alpha when applied across all prediction markets.

This motivated a new hypothesis:

Bayesian mispricing may not be universally predictive. Instead, its effectiveness may depend on how efficiently a market processes information.

Using the market regimes discovered in V2, this stage investigates whether the relationship between Bayesian mispricing and future market errors changes across different information-processing environments.

---

## Methodology

Two market regimes identified in V2 were used:

### Information Processing Markets

Characteristics:

* High reversals
* High drawdown
* Large probability range
* Low entropy
* Low forecast error

### Anchored / Noisy Markets

Characteristics:

* Low reversals
* Low drawdown
* Small probability range
* High entropy
* High forecast error

Bayesian mispricing was defined as:

 mispricing = bayesian_fair_probability - final_probability 

Future market error was defined as:


future_error = outcome - final_probability 

The objective was to determine whether the predictive relationship between mispricing and future error remains constant across regimes.

---

## Key Findings

### 1. Regime-Dependent Signal Direction

The direction of the Bayesian mispricing signal changes across regimes.

#### Anchored / Noisy Markets

Correlation:

$\rho = 0.554$

Positive mispricing tends to be associated with positive future errors.

This suggests that Bayesian fair value and realized outcomes move in the same direction.

#### Information Processing Markets

Correlation:

$\rho = -0.455 $

Positive mispricing tends to be associated with negative future errors.

This suggests that when Bayesian fair value disagrees with the market, the market is typically correct.

---

### 2. Bootstrap Stability Analysis

Bootstrap confidence intervals were used to evaluate the robustness of the observed correlations.

#### Anchored / Noisy

95% Confidence Interval:

$[-0.295,\ 0.924] $

The interval crosses zero, indicating insufficient evidence for a stable relationship.

#### Information Processing

95% Confidence Interval:

$ [-0.821,\ -0.164] $

The interval remains entirely negative, providing evidence that the inverse relationship is statistically stable within this regime.

---

### 3. Interaction Regression

The following interaction model was estimated:

$ future_{error} = \beta_0 + \beta_{1mispricing} + \beta_{2regime} + \beta_{3(mispricing \times regime)} $


Results:

| Variable            | Coefficient | p-value |
| ------------------- | ----------- | ------- |
| Mispricing          | 2.219       | 0.001   |
| Regime              | 0.045       | 0.533   |
| Mispricing × Regime | -2.914      | 0.001   |

The interaction term is highly significant.

This indicates that the predictive effect of Bayesian mispricing changes materially depending on the market regime.

---

## Conclusion

Bayesian mispricing is not a universal signal.

Its predictive behavior depends on the information-processing characteristics of the market.

In Information Processing markets, Bayesian mispricing exhibits a statistically stable contrarian relationship with future forecast errors.

These findings suggest that any potential alpha extraction strategy should be conditioned on market regime rather than applied uniformly across all prediction markets.

---

## Next Step

The next phase of the research will evaluate:

* Standard Bayesian Mispricing
* Inverted Bayesian Mispricing
* Regime-Conditioned Trading Signals

to determine whether regime-aware execution improves predictive performance and potential alpha generation.
