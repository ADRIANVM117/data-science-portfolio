# Scenario Portfolio Lab

An educational quantitative-finance project exploring scenario-based portfolio construction through a progressive implementation inspired by Anton Vorobets.

The project uses real Alpha Vantage historical data and separates data validation, model construction, numerical checks, predictive evaluation, and portfolio decisions.

## Current status

The data pipeline and notebooks 02–04 are implemented and numerically checked:

- **Data:** 138 approved entry instruments and 125 data-eligible instruments for the common window from 2018-01-02 to 2026-10-09.
- **Historical distribution:** 2,204 joint daily return observations, with marginal, temporal, and dependence diagnostics.
- **Joint resampling:** one-session and five-session scenarios under uniform and exponential historical probabilities.
- **State conditioning:** entropy pooling using lagged VIX, followed by five-session simulations with state updates.

**Predictive validation and portfolio optimization remain pending.** Numerical checks establish consistency with the implemented model; they do not establish forecasting skill or investment usefulness.

## Data pipeline

| Stage | Function | Result |
|---|---|---:|
| Raw | Original responses and provenance | 138 instruments |
| Silver | Parsing and a uniform schema | 138 series |
| Gold | Series-quality controls | 138 series |
| Matrices | Aligned prices and daily returns | 2,205 price dates |
| Data-eligible universe | Full coverage and instrument requirements | 125 instruments |

Data eligibility is not portfolio selection. No economic ranking, portfolio weights, or allocation has been produced.

## Educational notebooks

| Notebook | Question |
|---|---|
| [02 — Returns and empirical distribution](notebooks/02_returns_and_empirical_distribution.ipynb) | What does the historical joint return sample contain? |
| [03 — Joint resampling](notebooks/03_joint_resampling.ipynb) | How do we construct scenarios and distinguish model effects from Monte Carlo noise? |
| [04 — State-conditioned resampling](notebooks/04_state_conditioned_resampling.ipynb) | How can historical probabilities incorporate market state and evolve across simulated sessions? |

Notebook 04 uses 2,203 return observations aligned with VIX. The first historical return remains in the audit but lacks the required antecedent within the published calendar.

Its three state groups use empirical 25th/75th percentile cuts. The exponential half-life is 1,102 sessions. These are illustrative configurations, not demonstrated optimal choices.

## Install and validate

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The tests use shareable synthetic fixtures and make no API requests. They are separate from the checks executed inside the notebooks.

See the [local-raw offline reproduction](docs/offline_run_2026-10-11.md) for the sealed historical data run. Executing the notebooks requires the corresponding local data artifacts, which are not published.

A new Alpha Vantage acquisition is a separate operation and may differ because historical data can be revised.

## Documentation

- [Data contract](docs/data-contract.md)
- [Full-window coverage audit](docs/full_window_coverage_audit_2026-10-11.md)
- [Methodology status](docs/methodology-status.md)

## Essential limitations

- The source catalog is contemporary, not a historical point-in-time universe.
- The historical sample has already been explored; it is not an untouched holdout.
- VIX conditioning uses a documented two-equity-session lag. Historical availability is assumed under that convention, not verified through point-in-time data versions.
- The original sealed manifest does not contain reference hashes for the two matrix Parquet files. Later hashes establish subsequent consistency, not original historical integrity.
- The state variable and historical support represent only part of market uncertainty.
- Numerical consistency does not demonstrate predictive accuracy, trading feasibility, or economic value.

Downloaded data, generated matrices, credentials, local working instructions, and the source PDF are excluded from the repository. Gold quality approval and data eligibility remain separate controls.

## Next steps

1. Define a temporal evaluation protocol before examining evaluation results.
2. Compare state-conditioned forecasts with uniform and exponential baselines.
3. Assess calibration, distributional forecast quality, and parameter sensitivity.
4. Investigate portfolio construction, constraints, costs, and execution in a separate stage.



