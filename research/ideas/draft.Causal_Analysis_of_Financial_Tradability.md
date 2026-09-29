# Causal Analysis of Financial Tradability

## Status

- **Status:**: draft
- **Complete Specs:**: 60%
- **Assignee:**: TBD

## Core Idea

- Understand the causal relationship between trading horizon and market
  predictability
  - Identify the optimal trading horizon that maximizes risk-adjusted returns
- Investigate the trade-off between returns and prediction accuracy:
  - Returns are higher at longer horizons
  - Prediction accuracy is higher at shorter horizons
  - Quantify these competing forces: reduced returns at shorter horizons versus
    increased predictability
- Determine the minimum hit rate (win probability) required to achieve a given
  probability of positive profit and loss (PnL)
- Use high-frequency cryptocurrency data to study market microstructure and price
  dynamics
- Develop a framework for identifying optimal trading horizons based on risk-return
  profiles
- Apply causal inference methods to isolate horizon effects from confounding
  market factors

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. What is the minimum hit rate (probability of correct predictions) needed at
   different trading horizons to achieve a target probability of positive PnL?
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- **Multi-asset analysis**: extend the analysis to other cryptocurrency pairs and
  compare horizon effects across different assets
- **Market regime detection**: identify different market regimes (trending,
  mean-reverting, high volatility) and optimize trading horizons per regime
- **Transaction cost impact**: incorporate realistic transaction costs and
  slippage to assess practical tradability
- **Ensemble methods**: build ensemble models combining multiple prediction
  approaches to improve hit rates
- **Reinforcement learning baseline**: compare causal methods against RL-based
  horizon selection
- **Real-time implementation**: develop a live trading strategy that dynamically
  adjusts horizons based on market conditions
- **Cross-market correlation**: analyze how trading horizons and predictability
  change during synchronized vs. decoupled market movements

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: data collection and preprocessing
  - Fetch cryptocurrency OHLCV data at multiple time horizons (1m, 5m, 15m, 1h)
  - Clean the data for:
    - Missing values
    - Outliers
    - Synchronization across exchanges
  - Datasets:
    - **Binance Spot Market Data**: historical OHLCV (Open, High, Low, Close,
      Volume) data at multiple granularities (1m, 5m, 15m, 1h, etc.)
      - Source: Binance Public API
      - URL: `https://api.binance.com/api/v3/klines`
      - Data: Candlestick data for major crypto pairs (BTC/USDT, ETH/USDT)
      - Access: Free public API, no authentication required; rate limits apply
    - **Kraken Historical Data**: high-frequency trading data with order book
      snapshots and trade history
      - Source: Kraken REST API and WebSocket feeds
      - URL: `https://api.kraken.com/0/public/Trades` and
        `https://api.kraken.com/0/public/Depth`
      - Data: Individual trades, order book depth, timestamps (millisecond
        precision)
      - Access: Free public API; WebSocket feed for real-time data requires no
        authentication
    - **Kaggle Cryptocurrency Dataset**: pre-aggregated Bitcoin and Ethereum
      minute-level data
      - Source: Kaggle Datasets
      - URL: `https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data`
      - Data: OHLCV data at 1-minute granularity from 2013-2021
      - Access: Free with Kaggle account; CSV download available
    - **TickData-style Tick Dataset**: high-frequency tick data with microsecond
      timestamps from Bybit or similar exchanges
      - Source: Bybit Historical Data API
      - URL: `https://bybit-exchange.github.io/docs/linear/#t-publictradingrecords`
      - Data: Individual tick prices, sizes, and directions at sub-second
        granularity
      - Access: Free public API with rate limits; premium historical data
        available for purchase

- Milestone 2: feature engineering
  - Create predictive features: technical indicators, volatility measures, order
    book imbalance
  - Create target variables for different prediction horizons

- Milestone 3: predictability analysis
  - Measure predictability (classification accuracy, AUC-ROC) as a function of
    trading horizon
  - Use baseline models (e.g., logistic regression, random forests)

- Milestone 4: hit rate and PnL relationship
  - Model the probability distribution of PnL given different hit rates and time
    horizons
  - Calculate the minimum hit rate needed for positive expected PnL at each
    horizon

- Milestone 5: causal inference
  - Apply causal inference techniques (e.g., causal forests, instrumental
    variables)
  - Isolate the causal effect of horizon length on predictability from
    confounding factors

- Milestone 6: optimal horizon identification
  - Determine the trading horizon that maximizes a utility function balancing risk
    and return
  - Evaluate across different market regimes

- Milestone 7: backtesting and validation
  - Implement a backtesting framework
  - Validate model performance across different time periods and market conditions

## References

- 2022, Krauss et al., "Deep Learning in Finance", arXiv
  - Trained neural networks to predict cryptocurrency price movements at different
    horizons
  - Found that hit rates decrease significantly as prediction horizon increases,
    validating the speed-accuracy tradeoff
- 2021, Ritter et al., "Algorithmic Trading with Machine Learning", Medium
  - Tutorial on evaluating PnL probability distributions as a function of hit rate
    and position sizing
  - Provides formulas for relating minimum hit rates to profit factor targets
- 2020, Easley et al., "Microstructure and Ambiguity", Journal of Finance
  - Studied how information asymmetry varies with trading frequency and order flow
  - Found that microstructure effects dominate at shorter horizons, suggesting
    predictability decays over time
- 2019, Arnott et al., "How Can 'Smart Beta' Go Horribly Wrong?", Research
  Affiliates
  - Analyzed factor performance across different rebalancing horizons and found
    regime-dependent optimal horizons
  - Showed that shorter-term strategies incur higher costs and often underperform
    after adjustment
- GitHub: Optuna-based Hyperparameter Optimization for Trading,
  `https://github.com/gmarti/ml-monorepo`
  - Contains examples of optimizing trading strategies by tuning prediction
    horizons and model parameters
  - Includes backtesting utilities and risk metrics calculations
