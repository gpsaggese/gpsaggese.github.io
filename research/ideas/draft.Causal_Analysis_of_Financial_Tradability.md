# Causal Analysis of Financial Tradability

## Description

- Analyze the causal relationship between trading horizon and predictability in
  financial markets
- Investigate the trade-off between returns (higher at longer horizons) and
  prediction accuracy (higher at shorter horizons)
- Determine the minimum hit rate (win probability) required to achieve a given
  probability of positive profit and loss (PnL)
- Use high-frequency cryptocurrency data to study market microstructure and price
  dynamics
- Develop a framework for identifying optimal trading horizons based on risk-return
  profiles
- Apply causal inference methods to isolate horizon effects from confounding market
  factors

## Project Objective

The goal is to understand the causal relationship between trading horizon and market
predictability, and to identify the optimal trading horizon that maximizes
risk-adjusted returns. Specifically, the project seeks to answer: _What is the
minimum hit rate (probability of correct predictions) needed at different trading
horizons to achieve a target probability of positive PnL?_ This involves analyzing
high-frequency cryptocurrency data to quantify the competing forces of reduced
returns at shorter horizons versus increased predictability

## Dataset Suggestions

- **Binance Spot Market Data**: Historical OHLCV (Open, High, Low, Close, Volume)
  data at multiple granularities (1m, 5m, 15m, 1h, etc.)
  - Source: Binance Public API
  - URL: `https://api.binance.com/api/v3/klines`
  - Data: Candlestick data for major crypto pairs (BTC/USDT, ETH/USDT)
  - Access: Free public API, no authentication required; rate limits apply
- **Kraken Historical Data**: High-frequency trading data with order book snapshots
  and trade history
  - Source: Kraken REST API and WebSocket feeds
  - URL: `https://api.kraken.com/0/public/Trades` and
    `https://api.kraken.com/0/public/Depth`
  - Data: Individual trades, order book depth, timestamps (millisecond precision)
  - Access: Free public API; WebSocket feed for real-time data requires no
    authentication
- **Kaggle Cryptocurrency Dataset**: Pre-aggregated Bitcoin and Ethereum minute-level
  data
  - Source: Kaggle Datasets
  - URL: `https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data`
  - Data: OHLCV data at 1-minute granularity from 2013-2021
  - Access: Free with Kaggle account; CSV download available
- **TickData-style Tick Dataset**: High-frequency tick data with microsecond
  timestamps from Bybit or similar exchanges
  - Source: Bybit Historical Data API
  - URL: `https://bybit-exchange.github.io/docs/linear/#t-publictradingrecords`
  - Data: Individual tick prices, sizes, and directions at sub-second granularity
  - Access: Free public API with rate limits; premium historical data available for
    purchase

## Tasks

- **Data Collection and Preprocessing**: Fetch cryptocurrency OHLCV data at multiple
  time horizons (1m, 5m, 15m, 1h) and clean for missing values, outliers, and
  synchronization across exchanges
- **Feature Engineering**: Create predictive features (technical indicators,
  volatility measures, order book imbalance) and target variables for different
  prediction horizons
- **Predictability Analysis**: Measure predictability (classification accuracy,
  AUC-ROC) as a function of trading horizon using baseline models (e.g., logistic
  regression, random forests)
- **Hit Rate and PnL Relationship**: Model the probability distribution of PnL given
  different hit rates and time horizons; calculate the minimum hit rate needed for
  positive expected PnL at each horizon
- **Causal Inference**: Apply causal inference techniques (e.g., causal forests,
  instrumental variables) to isolate the causal effect of horizon length on
  predictability from confounding factors
- **Optimal Horizon Identification**: Determine the trading horizon that maximizes a
  utility function balancing risk and return across different market regimes
- **Backtesting and Validation**: Implement a backtesting framework to validate model
  performance across different time periods and market conditions

## Bonus Ideas

- **Multi-Asset Analysis**: Extend analysis to other cryptocurrency pairs and compare
  horizon effects across different assets
- **Market Regime Detection**: Identify different market regimes (trending,
  mean-reverting, high volatility) and optimize trading horizons per regime
- **Transaction Cost Impact**: Incorporate realistic transaction costs and slippage
  to assess practical tradability
- **Ensemble Methods**: Build ensemble models combining multiple prediction
  approaches to improve hit rates
- **Reinforcement Learning Baseline**: Compare causal methods against RL-based
  horizon selection
- **Real-Time Implementation**: Develop a live trading strategy that dynamically
  adjusts horizons based on market conditions
- **Cross-Market Correlation**: Analyze how trading horizons and predictability
  change during synchronized vs. decoupled market movements

## Previous Research

- 2020, Easley & O'Hara, "Microstructure and Ambiguity", Journal of Finance
  - Studied how information asymmetry varies with trading frequency and order flow
  - Found that microstructure effects dominate at shorter horizons, suggesting
    predictability decays over time
- 2019, Arnott et al., "How Can 'Smart Beta' Go Horribly Wrong?", Research Affiliates
  - Analyzed factor performance across different rebalancing horizons and found
    regime-dependent optimal horizons
  - Showed that shorter-term strategies incur higher costs and often underperform
    after adjustment
- 2022, Krauss & Do, "Deep Learning in Finance", arXiv
  - Trained neural networks to predict cryptocurrency price movements at different
    horizons
  - Found that hit rates decrease significantly as prediction horizon increases,
    validating the speed-accuracy tradeoff
- GitHub: Optuna-based Hyperparameter Optimization for Trading,
  `https://github.com/gmarti/ml-monorepo`
  - Contains examples of optimizing trading strategies by tuning prediction horizons
    and model parameters
  - Includes backtesting utilities and risk metrics calculations
- 2021, Ritter et al., "Algorithmic Trading with Machine Learning", Medium
  - Tutorial on evaluating PnL probability distributions as a function of hit rate
    and position sizing
  - Provides formulas for relating minimum hit rates to profit factor targets