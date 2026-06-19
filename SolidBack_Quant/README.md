# Intraday Leader-Follower Statistical Arbitrage Strategy

## Adrian Vazquez 

## Overview

This project implements and evaluates an intraday statistical arbitrage strategy on Nasdaq-100 stocks.

The strategy exploits short-term leader-follower relationships between highly correlated stocks belonging to the same sector. Every trading day, stock pairs are selected using a rolling 60-day correlation ranking and traded intraday using moving-average based entry and exit signals.


The objective of this project was to build a robust and reusable backtesting framework while carefully avoiding common sources of backtest bias such as look-ahead bias, survivorship bias, and timestamp leakage.

---

## Strategy Summary

For each trading day:

1. Build all possible stock pairs within the same sector.
2. Compute rolling 60-day correlations using only historical information.
3. Select the Top-50 most correlated pairs.
4. Define:

   * Leader = stock with the highest market capitalization.
   * Follower = remaining stock.
5. Compute a 15-minute SMA using data available before the market open.
6. Generate long and short signals based on leader and follower deviations from their SMA-adjusted thresholds.
7. Close positions when:

   * The leader reverts to its base SMA.
   * The trading session approaches the close (15:55 ET).
8. No overnight positions are allowed.

---

## Backtest Assumptions

* Universe: Nasdaq-100 constituents provided in the dataset.
* Correlation lookback: 60 trading days.
* Fixed notional per trade: $100,000.
* Minimum stock price: $10.
* Transaction costs: $0.0035 per share.
* Trading window: 09:30–15:55 ET.
* Benchmark: Buy-and-Hold QQQ.

---

## Main Findings

### Full Long-Short Strategy

* Total Return: -51.1%
* Sharpe Ratio: -0.35
* Maximum Drawdown: -74.5%

The original long-short implementation significantly underperformed the benchmark.

### Long-Only Diagnostic Variant

A decomposition of trade performance revealed a strong asymmetry between long and short positions.

| Side  |  Net PnL |
| ----- | -------: |
| Long  |  +49,049 |
| Short | -100,328 |

After removing short positions:

* Total Return: +46.6%
* Sharpe Ratio: 1.58
* Sortino Ratio: 3.40
* Calmar Ratio: 4.91

The long-only implementation materially outperformed the benchmark over the backtest period.

---

## Repository Structure

```text

backtest/
├── 02_backtest_engine.ipynb.ipynb
├── qq_test.ipynb

data/

documents/
├── research_papers/
├── strategy_description.pdf/

notebooks/
├── 00_data_audit.ipynb
├── 00_universe_construction.ipynb
├── 01_hypothesis_validation.ipynb

results/
├── strategy_trades_full.csv
├── strategy_trades_long_only.csv
├── equity_curve_full_strategy_vs_qqq.csv
├── equity_curve_long_only_vs_qqq.csv
├── equity_curve_full_strategy_vs_qqq.png
├── equity_curve_long_only_strategy_vs_qqq.png
├── drawdown_full_strategy_vs_qqq.png
├── drawdown_long_only_strategy_vs_qqq.png

src/
├── data/
├── universe/
├── backtest/

```

---

## Conclusion

The original long-short specification was not profitable and does not appear tradable in its current form.

However, the analysis revealed a strong asymmetry between long and short signals. Long trades generated positive returns while short trades consistently destroyed performance.

This suggests that the underlying signal may contain predictive information, although additional research on position sizing, risk management, and out-of-sample validation would be required before deployment.
