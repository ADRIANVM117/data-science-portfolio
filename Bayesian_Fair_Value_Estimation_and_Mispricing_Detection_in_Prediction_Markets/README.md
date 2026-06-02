# Bayesian Fair Value Estimation and Mispricing Detection in Prediction Markets
## <b> Adrian Vazquez </b>
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

The project follows a  workflow:

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

### Completed

 -  Historical market metadata successfully downloaded from the Polymarket Gamma API.

 - Historical implied probabilities successfully recovered from the Polymarket CLOB API using temporal chunking.

 -  Market filtering framework for Federal Reserve, interest rate, inflation, and macroeconomic events.

 - Initial historical dataset construction pipeline completed.

### In Progress

$ \rightarrow  $  Research dataset construction.

$ \rightarrow  $ Market calibration analysis.

$ \rightarrow  $ Probability forecasting evaluation.

---

## Expected Deliverables

* Historical prediction market dataset
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

The project successfully recovered historical probability trajectories for resolved Polymarket markets.

Example:

| Timestamp        | Implied Probability |
| ---------------- | ------------------- |
| 2024-03-21 17:00 | 0.16                |
| 2024-03-21 18:00 | 0.16                |
| 2024-03-21 19:00 | 0.17                |

This validates the feasibility of constructing a prediction-market research dataset suitable for calibration studies, Bayesian probability estimation, and systematic mispricing detection.

---

## Future Research Questions

* Are prediction markets well calibrated?
* Do market-implied probabilities systematically overestimate or underestimate event occurrence?
* Can Bayesian methods improve probability estimation?
* Can probability mispricing generate statistically significant trading signals?
* Do these signals survive realistic transaction costs and liquidity constraints?
