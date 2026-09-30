# Causal Analysis of Financial Tradability

## Status

- **Status:**: draft
- **Complete Specs:**: 60%

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

- Let $h$ be the trading horizon, $p_h$ the hit rate at horizon $h$, and $g_h$
  the average absolute move captured per trade
  - Expected return per trade: $\mathbb{E}[r_h] = (2 p_h - 1) \, g_h$
  - $g_h$ grows with $h$ (e.g., $g_h \propto \sigma \sqrt{h}$) while $p_h$
    falls with $h$
- Over $N$ independent trades, with $W \sim \mathrm{Bin}(N, p_h)$ wins:
  $\mathrm{PnL}_N = g_h (2W - N)$, so $P(\mathrm{PnL}_N > 0) = P(W > N/2)$
- Minimum hit rate for a target probability $q$ of positive PnL (normal
  approximation): $p^{*}_h \approx \frac{1}{2} + \frac{z_q}{2 \sqrt{N}}$
- Causal question: treat the horizon $H$ as the treatment and the net
  risk-adjusted return $Y$ as the outcome, adjusting for confounders $Z$
  (volatility, volume, regime): $\mathbb{E}[Y \mid do(H = h)]$
- Optimal horizon: $h^{*} = \arg\max_h \mathrm{Sharpe}(h)$ net of costs

## Key Examples

- **Minimum hit rate**: with $N = 1000$ one-minute trades and $q = 95\%$,
  $p^{*} \approx 0.5 + 1.645 / (2 \sqrt{1000}) \approx 0.526$, so a model
  needs a 52.6% hit rate before costs to be profitable 19 times out of 20
- **Horizon trade-off**: a 1-minute horizon may have a higher hit rate than a
  1-hour horizon, but the smaller move per trade may not cover fees and
  slippage; the optimal horizon depends on both effects
- **Regime confounding**: one horizon looks best only because the sample
  covers a trending, high-volatility week; adjusting for volatility removes
  the apparent horizon effect

## Questions

1. What is the minimum hit rate (probability of correct predictions) needed at
   different trading horizons to achieve a target probability of positive PnL?
2. What would refute the existence of a single optimal horizon? A
   horizon effect that changes sign across regimes, so no fixed $h^{*}$
   dominates, would be a counterexample.
3. If the required hit rate after costs exceeds what any model achieves at
   every horizon, does that mean the market is untradeable for this model
   class, and how would a null result be told apart from a weak model?

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

- Krauss et al., _Deep Learning in Finance_, arXiv. (2022)
  - Trained neural networks to predict cryptocurrency price movements at different
    horizons
  - Found that hit rates decrease significantly as prediction horizon increases,
    validating the speed-accuracy tradeoff
- Ritter et al., _Algorithmic Trading with Machine Learning_, Medium. (2021)
  - Tutorial on evaluating PnL probability distributions as a function of hit rate
    and position sizing
  - Provides formulas for relating minimum hit rates to profit factor targets
- Easley et al., _Microstructure and Ambiguity_, Journal of Finance. (2020)
  - Studied how information asymmetry varies with trading frequency and order flow
  - Found that microstructure effects dominate at shorter horizons, suggesting
    predictability decays over time
- Arnott et al., _How Can 'Smart Beta' Go Horribly Wrong?_, Research Affiliates.
  (2019)
  - Analyzed factor performance across different rebalancing horizons and found
    regime-dependent optimal horizons
  - Showed that shorter-term strategies incur higher costs and often underperform
    after adjustment
- GitHub: Optuna-based Hyperparameter Optimization for Trading,
  `https://github.com/gmarti/ml-monorepo`
  - Contains examples of optimizing trading strategies by tuning prediction
    horizons and model parameters
  - Includes backtesting utilities and risk metrics calculations
