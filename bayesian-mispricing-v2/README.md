# V2 – Information Dynamics and Market Efficiency in Prediction Markets


The initial objective of V2 was to investigate whether prediction market calibration differed across market categories such as macroeconomic events and political events.

However, exploratory analysis revealed little evidence that market topic alone explained forecast accuracy.

This led to a different research question:

Do information incorporation dynamics explain prediction market accuracy better than market categories?

---

## Dataset

### Market-Level Dataset

43 prediction markets collected from Polymarket.

Variables include:

* Market outcome
* Final probability
* Market duration
* Liquidity
* Volume
* Market category labels

### Time-Series Dataset

100,642 probability observations across the 43 markets.

Variables include:

* `market_id`
* `timestamp`
* `implied_probability`

This dataset captures the full evolution of market beliefs over time.

---

## Research Workflow

###  Category Tagging

Markets were manually classified into:

* Monetary & Macro
* Politics & Geopolitics
* Other

Distribution:

| Category               | Markets |
| ---------------------- | ------- |
| Monetary & Macro       | 22      |
| Politics & Geopolitics | 13      |
| Other                  | 8       |

---

### Category EDA

The objective was to determine whether market topic explained forecast accuracy.

Metrics analyzed:

* Outcome rate
* Market duration
* Probability distributions
* Final probabilities
* Forecast error (`abs_surprise`)

Result:

No statistically meaningful differences were found between macroeconomic and political markets.

---

### Trajectory Feature Engineering

Using the complete probability trajectories, the following market-level features were constructed:

#### Realized Volatility

Measures cumulative probability movement.

#### Probability Range

[
\max(P_t)-\min(P_t)
]

Measures how far market beliefs traveled.

#### Trend

Linear slope of probability evolution.

#### Max Drawdown

Largest peak-to-trough decline in market probability.

#### Reversals

Number of probability direction changes.

#### Information-Theoretic Features

* Shannon Entropy
* Skewness
* Kurtosis
* Lag-1 Autocorrelation

---

## Forecast Error Definition

Forecast accuracy was measured using:

[
AbsSurprise = |Outcome - FinalProbability|
]

Interpretation:

* Small values indicate accurate forecasts.
* Large values indicate inaccurate forecasts.

---

## Main Results

### Correlation Analysis

Correlation with forecast error (`abs_surprise`):

| Feature           | Correlation |
| ----------------- | ----------: |
| Max Drawdown      |      -0.756 |
| Probability Range |      -0.587 |
| Reversals         |      -0.370 |
| Kurtosis          |      -0.306 |
| Shannon Entropy   |       0.234 |

The strongest relationship was observed for Max Drawdown.

---

### OLS Regression

Model:

[
AbsSurprise
===========

\beta_0
+
\beta_1(MaxDrawdown)
+
\beta_2(ProbabilityRange)
+
\varepsilon
]

Results:

* (R^2 = 0.614)
* Max Drawdown significant ((p < 0.001))
* Probability Range significant ((p = 0.042))

These two variables alone explained approximately 61% of the variation in forecast error.

---

### High vs Low Drawdown Markets

Markets were split at the median Max Drawdown.

#### Mean Forecast Error

| Group         | Mean Abs Surprise |
| ------------- | ----------------: |
| High Drawdown |            0.0188 |
| Low Drawdown  |            0.2109 |

#### Median Forecast Error

| Group         | Median Abs Surprise |
| ------------- | ------------------: |
| High Drawdown |              0.0050 |
| Low Drawdown  |              0.0575 |

High-drawdown markets exhibited approximately 11x lower forecast error.

---

### Mann-Whitney Test

Hypothesis:

[
H_0:
\text{High Drawdown Markets}
============================

\text{Low Drawdown Markets}
]

Result:

* p-value = 0.012

The null hypothesis was rejected.

Forecast accuracy differs significantly between the two groups.

---

## Key Finding

The evidence suggests that prediction market efficiency is not primarily driven by market topic.

Instead, efficiency appears to be strongly associated with the magnitude of belief revisions occurring throughout the market lifecycle.

Markets that aggressively revise probabilities in response to new information ultimately converge to substantially more accurate forecasts.

---

## Future Work

Potential extensions include:

* Hidden Markov Models (HMM)
* Regime detection
* Information flow metrics
* Advanced entropy measures
* Prediction of forecast accuracy from trajectory dynamics
* Bayesian state-space models

---

## Conclusion

This study provides evidence that information incorporation dynamics are more informative than market categories for explaining prediction market accuracy.

The strongest predictor identified was Max Drawdown, suggesting that markets capable of correcting large belief errors ultimately produce more accurate forecasts than markets that remain anchored to their initial expectations.
