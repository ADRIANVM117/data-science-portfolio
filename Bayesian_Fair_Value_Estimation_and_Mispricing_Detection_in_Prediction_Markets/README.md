# Bayesian Fair Value Estimation and Mispricing Detection in Prediction Markets

### Adrian Vazquez

---

## Research Question

Can Bayesian calibration identify systematic mispricing in prediction markets and generate exploitable trading signals?

---

## Motivation

Prediction markets continuously aggregate information from thousands of participants and express beliefs about future events through market prices.

Examples include:

* Federal Reserve decisions
* Inflation releases
* Macroeconomic events
* Political outcomes
* Cryptocurrency-related events

In binary prediction markets, contract prices can be interpreted as market-implied probabilities.

| Contract Price | Implied Probability |
| -------------- | ------------------- |
| 0.75           | 75%                 |

This project investigates whether Bayesian methods can improve these probability estimates and uncover systematic market mispricing.

---

## Research Workflow

1. Historical Data Acquisition
2. Dataset Construction
3. Exploratory Data Analysis
4. Market Calibration Analysis
5. Bayesian Fair Value Estimation
6. Signal Generation & Validation

---

## Project Structure

```text
Bayesian_Fair_Value_Estimation_and_Mispricing_Detection_in_Prediction_Markets/
│
├── notebooks/
│   ├── 01_Data_Acquisition.ipynb
│   ├── 02_Dataset_Construction.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Market_Calibration.ipynb
│   ├── 05_Bayesian_Fair_Value_Model.ipynb
│   └── 06_Signal_Generation.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
│
├── results/
│   ├── plots/
│   ├── tables/
│   └── reports/
│
├── src/
│   ├── analytics/
│   ├── models/
│   └── utils/
│
└── README.md
```

---

## Dataset

The project reconstructed a historical dataset from Polymarket using both the Gamma API and CLOB API.

Dataset summary:

* 43 resolved prediction markets
* 100,642 historical probability observations
* Binary event outcomes
* Historical probability trajectories
* Market metadata (volume, liquidity, expiration dates)

---

## Methodology

### Market Calibration

Prediction markets were evaluated using:

* Brier Score
* Reliability Diagrams
* Expected Calibration Error (ECE)

The objective was to determine whether market-implied probabilities accurately reflected realized event frequencies.

---

### Bayesian Fair Value Estimation

A Beta-Binomial framework was used to estimate fair probabilities.

Prior:

p ~ Beta(1,1)

Likelihood:

y ~ Binomial(n,p)

Posterior:

p | y ~ Beta(α+s, β+f)

where:

* s = historical successes
* f = historical failures

Posterior means were used as Bayesian fair probabilities.

---

### Mispricing Detection

Mispricing was defined as:

Mispricing = P(Bayes) − P(Market)

Positive values indicate potential BUY opportunities.

Negative values indicate potential SELL opportunities.

---

## Key Findings

### Market Calibration

Prediction markets appear reasonably well calibrated.

Metrics:

* Brier Score ≈ 0.053
* Expected Calibration Error (ECE) ≈ 0.056

Reliability analysis showed that extreme probabilities near 0% and 100% were generally consistent with realized outcomes.

---

### Bayesian Fair Value Results

The Bayesian model generated measurable deviations from market-implied probabilities.

However, most deviations remained relatively small and centered around zero.

Average absolute mispricing:

≈ 6.2%

---

### Signal Validation

Signal generation was evaluated across multiple mispricing thresholds.

Results indicate:

* Signal counts decrease rapidly as thresholds increase.
* Large mispricing events are relatively rare.
* Neither BUY-only nor BUY/SELL strategies produced accuracy materially above random chance.

The strongest BUY signals achieved only:

* 33% accuracy at τ = 0.05
* 50% accuracy at τ = 0.10
* 50% accuracy at τ = 0.15

These results provide no evidence that simple Bayesian calibration generates exploitable trading signals within the current sample.

---

## Conclusion

Within this dataset, a simple Beta-Binomial Bayesian calibration framework does not improve upon market-implied probabilities.

Although measurable mispricings exist, they do not appear to contain persistent predictive information.

The findings suggest that prediction markets already incorporate most publicly available information and are broadly efficient.

Importantly, this negative result establishes a rigorous quantitative baseline for future research.

---

## Future Research — Version 2

The current version applies a single calibration model across all prediction markets.

A natural extension is to investigate whether calibration and mispricing behavior differ across market categories.

Examples:

* Federal Reserve markets
* Inflation markets
* Macroeconomic events
* Political markets
* Sports markets
* Cryptocurrency markets

Potential improvements include:

* Market-specific Bayesian calibration
* Hierarchical Bayesian models
* Dynamic priors
* Time-to-expiration effects
* Liquidity-aware fair value estimation
* Category-level calibration analysis

---

## Future Research — Version 3

Inspired by modern Quant Research methodologies and the work of Marcos López de Prado.

Potential extensions include:

* Entropy-based features
* Information-theoretic signals
* Regime-dependent calibration
* Market state classification
* Hidden Markov Models (HMM)
* Bayesian regime switching
* Meta-labeling frameworks
* Cross-market information flow analysis

The objective is to move beyond simple probability calibration and investigate whether market inefficiencies emerge under specific information regimes.

---

## Main Takeaway

A simple Bayesian calibration framework is not sufficient to outperform prediction markets.

The next research question is not whether mispricing exists, but under what market conditions mispricing becomes predictive.
