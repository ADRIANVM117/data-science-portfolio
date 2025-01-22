# Portfolio Optimization Project Documentation

## Project Objective

The objective of this project is to create a diversified investment portfolio using various Exchange-Traded Funds (ETFs), optimize the portfolio to minimize risk (volatility) and maximize risk-adjusted return (Sharpe ratio), and perform a Monte Carlo simulation to project the future behavior of the portfolio.

## Steps Taken

### ETF Selection

Seven ETFs were selected to diversify the investment across different sectors and asset types:

- **AMZN (Amazon)**: 5% weight, exposure to the technology sector.
- **BND (Bonds)**: 10% weight, for portfolio stability.
- **CHDRAUIB.MX (Mexican Stock Market)**: 30% weight, exposure to the Mexican market.
- **VEA (International Markets)**: 10% weight, geographical diversification.
- **VOO (U.S. Market)**: 25% weight, exposure to large U.S. companies.
- **VYM (Dividends)**: 10% weight, aims for passive income through dividends.
- **XLRE (Real Estate)**: 10% weight, exposure to the real estate sector.

### Downloading Historical Data

Using the `yfinance` library, the adjusted close prices for each ETF were downloaded from January 1, 2018, to November 13, 2024.

### Calculating Daily Returns

The daily returns of each ETF were calculated using the percentage change of the adjusted prices.

### Calculating Performance and Risk Metrics

- **Average Return**: The average daily return for each ETF was calculated.
- **Covariance Matrix**: The covariance matrix of the returns was calculated to assess the relationship between the assets within the portfolio.

### Portfolio Optimization for Minimum Volatility

A function was defined to calculate the annualized return and risk (volatility) of a portfolio, and an optimizer (`scipy.optimize.minimize`) was used to find the asset weights that minimize the portfolio’s volatility, with the constraint that the weights sum to 1.

The resulting minimum volatility portfolio has the following weights:

- **Minimum Volatility Portfolio Weights**:
  - AMZN: 1.43%
  - BND: 0.00%
  - CHDRAUIB.MX: 91.45%
  - VEA: 1.61%
  - VOO: 0.00%
  - VYM: 6.94%
  - XLRE: 0.00%

### Efficient Frontier Simulation

10,000 portfolios with random weights were simulated, and the return and risk for each were calculated, representing the efficient frontier, which shows the risk-return relationship of an optimized portfolio.

### Maximizing the Sharpe Ratio

The portfolio with the highest Sharpe Ratio (i.e., the one offering the best return per unit of risk) was determined. The portfolio with the highest Sharpe Ratio has the following weights:

- **Sharpe Ratio Portfolio Weights**:
  - AMZN: 16.71%
  - BND: 0.02%
  - CHDRAUIB.MX: 59.02%
  - VEA: 4.25%
  - VOO: 14.20%
  - VYM: 0.18%
  - XLRE: 5.61%

### Monte Carlo Simulation

100,000 Monte Carlo simulations were performed to predict the portfolio's behavior over a 2-year horizon, using the weights from the portfolio with the highest Sharpe Ratio and the historical returns of the ETFs.

### Simulation Results


---
![image](https://github.com/user-attachments/assets/e3e73eba-3747-4e37-918c-5f9af1e8d601)

- **Average value at the end of 2 years**: $503,867.89
- **Median final value**: $492,655.62
- **5th Percentile (pessimistic scenario)**: $347,577.36
- **95th Percentile (optimistic scenario)**: $698,656.62

### Visualization of Results

The efficient frontier was plotted showing the simulated portfolios, with the minimum volatility portfolio and the Sharpe ratio portfolio highlighted.


---
![image](https://github.com/user-attachments/assets/e3e73eba-3747-4e37-918c-5f9af1e8d601)

## Conclusions

- The portfolio with the highest Sharpe Ratio offers a good balance between risk and return, with an expected annualized return of 27.68% and volatility of 21.24%.
- The Monte Carlo simulation shows that the portfolio’s final value can vary, with a median of approximately $492,655.


