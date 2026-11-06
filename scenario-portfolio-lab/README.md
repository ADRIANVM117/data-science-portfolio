# Scenario Portfolio Lab

**A scenario-based portfolio engine: from market data to constrained allocations and cost-aware backtesting.**

Built for an independent Quant Desk, using entropy pooling, market-state conditioning and CVaR optimization to turn joint return scenarios into portfolio decisions. Inspired by Anton Vorobets’ portfolio-construction framework.

**Research implementation complete. Next milestone: operational integration with Interactive Brokers paper trading.**

## The engine

- **Data:** real Alpha Vantage history; 138 approved entry instruments, 125 data-eligible instruments and 2,204 joint daily return observations, covering 2018–2026-10-09.
- **Scenarios:** joint resampling, exponential time weighting and VIX-conditioned entropy pooling, including simulated state evolution.
- **Allocation:** minimum CVaR portfolios with explicit investment and concentration constraints.
- **Evaluation:** temporal forecast evaluation and monthly portfolio backtesting with drifting weights, turnover and self-financed transaction costs.

The implemented portfolio experiment uses eight USD ETFs, full investment, no shorting or leverage, and a 15% target-weight cap. Data eligibility and portfolio-universe selection are separate decisions.

## Portfolio results

January 31, 2020–October 9, 2026. USD 1M initial capital; 81 monthly allocation decisions. Net results assume **5 basis points per side**, including initial entry costs.

| Rule | Annualized return (CAGR) | Annualized volatility | Maximum drawdown |
|---|---:|---:|---:|
| Equal weight | 9.31% | 12.48% | −25.23% |
| Minimum CVaR · exponential | 8.20% | 11.36% | −22.35% |
| Minimum CVaR · VIX-conditioned | 8.32% | 11.33% | −22.24% |

CVaR portfolios traded lower growth for lower realized volatility and drawdown. The incremental VIX result was modest and sensitive to transaction costs; it does not establish predictive or investment superiority.

## Research workflow

| Notebook | Focus |
|---|---|
| [02](notebooks/02_returns_and_empirical_distribution.ipynb) | Historical joint distribution and dependence |
| [03](notebooks/03_joint_resampling.ipynb) | Joint scenarios and temporal weighting |
| [04](notebooks/04_state_conditioned_resampling.ipynb) | Entropy pooling and state-conditioned simulation |
| [05](notebooks/05_scenario_evaluation.ipynb) | Temporal forecast evaluation |
| [06](notebooks/06_portfolio_construction.ipynb) | Constrained CVaR portfolio construction |
| [07](notebooks/07_portfolio_backtest.ipynb) | Portfolio backtesting, costs and diagnostics |

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Tests use synthetic fixtures without API calls. Notebooks require local data artifacts, which are not distributed. See [offline reproduction](docs/offline_run_2026-10-11.md), [data contract](docs/data-contract.md) and [methodology status](docs/methodology-status.md).

## Scope and next milestone

This is retrospective research: the universe was selected after observing the history, the sample has been explored, and data may be revised. VIX timing follows a documented availability convention rather than historical point-in-time verification. Matrix hashes recorded later establish subsequent consistency, not original sealed provenance. Closing-price execution and transaction costs are modeled assumptions.

The next delivery is a repeatable paper-trading workflow: updated data, decision reports, position sizing, broker reconciliation and execution monitoring. **Broker integration and live execution are not yet implemented.**
