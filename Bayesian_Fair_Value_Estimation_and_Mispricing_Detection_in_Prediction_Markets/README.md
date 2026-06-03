# Bayesian Fair Value Estimation and Mispricing Detection in Prediction Markets

## Adrian Vazquez

---

## Research Question

Can Bayesian models identify systematic mispricing in prediction markets and generate exploitable trading signals?

---

## Research Objective

The objective of this project is to investigate whether Bayesian methods can identify systematic mispricing in prediction markets by estimating a fair probability and comparing it against the market-implied probability.

The ultimate goal is to determine whether these discrepancies can generate exploitable trading signals.

---

## Why Prediction Markets?

Prediction markets aggregate information from thousands of participants and continuously update probabilities regarding future events.

Examples include:

* Federal Reserve decisions
* Interest rate changes
* Inflation releases
* Macroeconomic events
* Political outcomes

In binary prediction markets, contract prices can be interpreted as market-implied probabilities.

For example:

| Contract Price | Implied Probability |
| -------------- | ------------------- |
| 0.75           | 75%                 |

This project investigates whether these probabilities are systematically biased and whether Bayesian methods can improve probability estimation.

---

## Research Workflow

The project follows a professional Quant Research workflow:

1. Data Acquisition
2. Dataset Construction
3. Exploratory Analysis
4. Market Calibration Analysis
5. Bayesian Fair Value Modeling
6. Signal Generation
7. Backtesting
8. Performance Evaluation

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
│   ├── 06_Signal_Generation.ipynb
│   ├── 07_Backtesting.ipynb
│   └── 08_Performance_Evaluation.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
│
├── results/
│   ├── plots/
│   ├── tables/
│   ├── reports/
│   └── papers/
│
├── src/
│   ├── analytics/
│   ├── models/
│   ├── backtesting/
│   ├── simulation/
│   └── utils/
│
├── requirements.txt
└── README.md
```

---

## Notebook Roadmap

| Notebook | Description                                        |
| -------- | -------------------------------------------------- |
| 01       | Historical Market Data Acquisition from Polymarket |
| 02       | Research Dataset Construction                      |
| 03       | Exploratory Analysis of Prediction Markets         |
| 04       | Calibration Analysis and Brier Score Evaluation    |
| 05       | Bayesian Fair Probability Estimation               |
| 06       | Mispricing Detection and Signal Generation         |
| 07       | Strategy Backtesting                               |
| 08       | Performance Evaluation and Research Conclusions    |

---

## Current Status

### Phase 1 — Data Acquisition - DONE

* Historical market metadata successfully downloaded from the Polymarket Gamma API.
* Historical implied probabilities successfully recovered from the Polymarket CLOB API.
* Temporal chunking framework implemented to bypass API time-window limitations.
* Successfully collected historical trajectories for resolved prediction markets.

### Phase 2 — Dataset Construction - DONE 

* Historical probability trajectories merged with market metadata.
* Market-level feature engineering completed.
* Final probabilities extracted for each market.
* Market outcomes successfully reconstructed from resolved contract prices.
* Binary target variable generated for calibration and forecasting analysis.
* Research dataset validated and ready for quantitative analysis.

### Current Dataset

| Metric                              | Value                |
| ----------------------------------- | -------------------- |
| Markets                             | 43                   |
| Historical Probability Observations | 100,642              |
| Binary Outcomes Available           | 43                   |
| Time Span                           | Feb-2023 to Dec-2024 |

### Phase 3 — Exploratory Analysis  - DONE

The exploratory analysis successfully validated the research dataset and provided an initial characterization of prediction market behavior.

Completed analyses:

* Market outcome distribution
* Final probability distribution
* Probability vs realized outcome analysis
* Market-level descriptive statistics
* Historical probability trajectory validation
* Data quality verification
* Missing value assessment
* Initial evidence of market informativeness

Key findings:

* 43 resolved prediction markets were successfully collected.
* More than 100,000 historical probability observations were recovered.
* Final market probabilities show strong separation between realized successes and failures.
* Several markets exhibit apparent probability mispricing, motivating further calibration analysis.
* Historical probability trajectories were successfully reconstructed from Polymarket's CLOB API.

---

### Phase 4 — Market Calibration Analysis  IN PROGRESS

Next step:

* Reliability diagrams
* Calibration curves
* Probability binning analysis
* Brier Score evaluation
* Sharpness analysis
* Overconfidence vs underconfidence diagnostics
* Market efficiency assessment

---

## Expected Deliverables

* Historical prediction market dataset
* Exploratory market analysis
* Market calibration analysis
* Brier Score evaluation
* Reliability and calibration curves
* Bayesian fair probability estimator
* Mispricing detection framework
* Trading signal generation engine
* Backtesting framework
* Research report and conclusions

---

## Preliminary Results

The project successfully reconstructed historical probability trajectories for resolved prediction markets and generated a research dataset containing:

* 43 resolved markets
* 100,642 historical probability observations
* Binary event outcomes
* Market metadata (volume, liquidity, expiration dates)

Example:

| Timestamp | Implied Probability |
|------------|------------|
| 2024-03-21 17:00 | 0.16 |
| 2024-03-21 18:00 | 0.16 |
| 2024-03-21 19:00 | 0.17 |

Initial exploratory results suggest that prediction markets contain meaningful information regarding future outcomes, while also exhibiting potential cases of probability mispricing that justify further calibration and Bayesian analysis.

This validates the feasibility of constructing a quantitative research framework for:

* Market calibration studies
* Bayesian probability estimation
* Mispricing detection
* Trading signal generation
* Systematic prediction-market research

---

## Future Research Questions

* Are prediction markets well calibrated?
* Do market-implied probabilities systematically overestimate or underestimate event occurrence?
* Can Bayesian methods improve probability estimation?
* Can probability mispricing generate statistically significant trading signals?
* Do these signals survive realistic transaction costs and liquidity constraints?

---

# Future Research Extensions

The following extensions are intentionally excluded from Version 1 in order to first establish a robust baseline research framework.

---

## Version 2 — Information-Theoretic Features

Inspired by the work of Marcos López de Prado.

Potential features include:

* Shannon Entropy
* Probability Path Entropy
* Permutation Entropy
* Information Regime Classification
* Entropy-Based Market Segmentation

Potential research questions:

* Do high-entropy markets exhibit worse calibration?
* Does entropy predict future probability revisions?
* Are mispricing opportunities concentrated in information-rich markets?

---

## Version 3 — Regime Detection and Market States

Potential methodologies:

* Hidden Markov Models (HMM)
* Bayesian State Space Models
* Regime-Switching Probability Models

Potential market states:

* Consensus Regime
* Information Arrival Regime
* Panic Regime
* High-Uncertainty Regime

Potential research questions:

* Do calibration errors vary across market regimes?
* Does Bayesian fair value depend on market state?
* Can regime information improve trading signals?
* Can latent-state models identify periods of market inefficiency?

---

## Long-Term Vision

The long-term goal of this research is to bridge:

* Bayesian Statistics
* Prediction Markets
* Information Theory
* Market Microstructure
* Quantitative Trading

into a unified framework for identifying and exploiting probability mispricing in real-world prediction markets.
