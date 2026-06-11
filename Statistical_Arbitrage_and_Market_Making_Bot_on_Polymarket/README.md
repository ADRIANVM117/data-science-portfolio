
# Statistical Arbitrage and Market Making Infrastructure on Polymarket

A real-time prediction market microstructure research framework built on top of Polymarket order book data.

The system ingests live Level-2 market data, reconstructs order books, computes microstructure features, filters tradable markets, generates real-time signals, and tracks signal persistence for future statistical arbitrage and market making research.

## System Architecture

![Proyect - Adrian Vazquez](imagenes/general%20overview.png)


## Motivation

Prediction markets are an emerging asset class where traditional market microstructure research remains largely unexplored.

This project was designed to build the infrastructure required to study:

- Liquidity formation
- Order book dynamics
- Signal persistence
- Market making opportunities
- Statistical arbitrage opportunities

using real-time Polymarket order book data.

## <b> Current infrastructure </b>



The current version of the project focuses on building the real-time research infrastructure required to study prediction market microstructure and develop future statistical arbitrage and market making strategies.

### Market Discovery

The system automatically discovers and ranks active Polymarket markets using the Gamma API, allowing the framework to continuously focus on liquid and relevant prediction contracts.

**Current capabilities:**
- Retrieve active markets
- Rank markets by liquidity and activity
- Automatically build the live asset universe

---

### Real-Time Market Data Ingestion

A dedicated WebSocket client streams live Level-2 order book updates directly from Polymarket.

**Current capabilities:**
- Subscribe to multiple markets simultaneously
- Process live bid/ask updates
- Maintain low-latency data ingestion
- Asynchronous event-driven architecture

---

### Order Book Engine

The framework reconstructs and maintains in-memory order books for every tracked asset.

**Current capabilities:**
- Bid-side management
- Ask-side management
- Full book replacement from snapshots
- Real-time state synchronization

---

### Market Microstructure Layer

The system transforms raw order book data into quantitative market microstructure features.

**Current features:**
- Spread
- Mid Price
- Book Imbalance
- Microprice
- Top-of-Book Liquidity Depth

These features form the foundation for future alpha research and execution models.

---

### Tradable Universe Filter

Not all prediction markets are suitable for quantitative trading.

A dedicated filtering layer removes markets that are unlikely to provide meaningful execution opportunities.

**Current filters:**
- Minimum liquidity requirements
- Maximum spread thresholds
- Extreme probability exclusion
- Market quality validation

This creates a dynamic universe of tradable markets.

---

### Signal Generation Engine

The framework generates real-time market signals based on order book dynamics.

**Current signals:**
- BUY_PRESSURE
- SELL_PRESSURE
- NEUTRAL

Signals are generated using:
- Microprice deviation from mid-price
- Order book imbalance
- Liquidity conditions

---

### Signal Persistence Layer

Microstructure signals are often noisy.

To reduce false positives, the framework tracks signal persistence through time.

**Current capabilities:**
- Consecutive signal tracking
- Signal age monitoring
- State transitions
- Noise filtering

This allows the system to distinguish temporary fluctuations from persistent market pressure.

---

### Market State Management

A centralized state layer stores the latest market information and signals.

**Current capabilities:**
- Real-time feature storage
- Signal tracking
- Tradable universe management
- Monitoring integration

---

### Monitoring Dashboard

A live monitoring layer provides visibility into the current market state.

**Current outputs:**
- Top spreads
- Top liquidity
- Order book imbalance rankings
- Active signals
- Persistent signals

This serves as the primary research interface for observing live market dynamics.


## Next Steps

The current infrastructure establishes the foundation for advanced prediction market research. Future development will focus on validating whether microstructure signals contain predictive information and can generate actionable trading opportunities.

### Ground Truth Trade Classification

Current signals are generated exclusively from order book information.

Future versions will integrate Polygon blockchain data and `OrderFilled` events to determine the true aggressor side of every transaction.

**Goal:**
- Accurate trade classification
- Reduce WebSocket inference errors
- Build institutional-grade market microstructure datasets

---

### Alpha Validation Framework

The next research stage is to evaluate whether persistent microstructure signals predict future price movements.

**Research questions:**
- Does order book imbalance predict short-term returns?
- Does microprice contain predictive information?
- Does signal persistence improve forecasting accuracy?

**Expected outputs:**
- Future return datasets
- Information Coefficient (IC)
- Hit Rate analysis
- Alpha decay curves

---

### Historical Dataset Construction

Build a large-scale dataset of:

- Features
- Signals
- Signal persistence
- Future returns

for systematic backtesting and machine learning research.

---

### Statistical Arbitrage Research

Future research modules will explore:

- YES/NO parity violations
- Cross-market mispricing
- Event-linked market relationships
- Cross-exchange prediction market arbitrage

---

### Market Making Engine

Implement inventory-aware market making strategies inspired by modern electronic trading systems.

Potential approaches include:

- Avellaneda-Stoikov
- Inventory risk models
- Dynamic spread optimization
- Queue position modeling

---

### Backtesting and Simulation

Develop a realistic execution simulator capable of replaying historical order book data.

**Goals:**
- Evaluate signal profitability
- Measure execution quality
- Estimate slippage
- Validate trading strategies before deployment

---

### Machine Learning Research

Once a sufficiently large dataset is collected, machine learning models can be trained to predict future market movements.

Potential models include:

- Logistic Regression
- XGBoost
- LightGBM
- Reinforcement Learning
- Sequence Models

---

### Live Execution Layer

The final stage of the project is the deployment of a fully automated prediction market trading system capable of:

- Signal generation
- Opportunity ranking
- Risk management
- Order execution
- Portfolio monitoring

---


```text
polymarket-stat-arb-mm/
│
├── src/
│   └── polymarket_bot/
│       ├── config.py
│       ├── data/
│       │   ├── websocket_client.py
|       |   ├── config.py
|       |   ├── features.py
|       |   ├── filters.py
|       |   ├── get_active_token.py
|       |   ├── main.py
|       |   ├── market_discovery.py
│       │   ├── orderbook.py
|       |   ├── signal_engine.py
|       |   ├── signal_state.py
|       |   ├── state_manager.py
|       |   ├── test_ws.py
│       │   └── websocket_client.py
│       ├── backtest/
│       │   ├── simulator.py
│       │   └── execution.py
│       └── utils/
│           └── logging.py
│
├── notebooks/
│   └── 01_orderbook_streaming_demo.ipynb
│
├── tests/
│   └── test_orderbook.py
│ 
└── README.md

```