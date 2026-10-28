# ADIA Lab Competitions

ADIA Lab runs yearly data science competitions in partnership with CrunchDAO.
This file collects the links to the four main competitions and one related
benchmark.

## Summary

| Year | Competition | Task |
|------|-------------|------|
| 2023 | Market Prediction | Rank investment vehicles from strongest to weakest at each date |
| 2024 | Causal Discovery | Recover the causal structure behind each dataset |
| 2025 | Structural Break | Detect abrupt shifts in univariate time series |
| 2026 | Structural Break, Real-Time Edition | Score structural breaks online, one observation at a time |

## 1. Market Prediction (2023)

- Task: rank a pool of investment vehicles at each date
- Scoring: Spearman rank correlation between the predicted and the true ranking
- Prize pool: $100,000 USD
- Links:
  - [CrunchDAO docs](https://docs.crunchdao.com/competitions/competitions/adia-lab-market-prediction-competition)
  - [ADIA Lab launch post](https://www.adialab.ae/adialab-news/adia-lab-crunchdao-competition-launch)
  - [Participation docs](https://docs.adialab.crunchdao.com)

## 2. Causal Discovery (2024)

- Task: recover the causal graph behind each dataset
- Data: example datasets with true graphs, to tune unsupervised methods or to
  train supervised models
- Results: 1,904 registered participants and 3,343 submissions. Supervised
  methods beat classical constraint-based discovery. The best entries reached
  about 77% balanced accuracy, against about 40% for the baselines.
- Links:
  - [CrunchDAO docs](https://docs.crunchdao.com/competitions/competitions/adia-lab-causal-discovery)
  - [Stanford GSB analysis of the results](https://www.gsb.stanford.edu/faculty-research/publications/evidence-adia-labs-causal-discovery-challenge)

## 3. Structural Break (2025)

- Task: detect abrupt shifts in univariate time series
- Prize pool: $100,000 USD
- Deadline: extended by 15 days to September 30, 2025
- Participation: by July 2025, over 45 teams and hundreds of individuals had
  entered, with more than 3,000 models submitted
- Links:
  - [CrunchDAO docs](https://docs.crunchdao.com/competitions/competitions/adia-lab-structural-break-challenge)

## 4. Structural Break, Real-Time Edition (2026)

- Task: the participant first gets a historical reference segment. The online
  segment then arrives one observation at a time. The participant gives a score
  after each observation, with no way to look ahead.
- Data: over 10,000 univariate time series per split
- Prize pool: $100K USDC
- Dates: May 4 to September 17, 2026
- Winners: announced at the ADIA Lab 2026 Symposium, October 26 to 28, 2026
- Links:
  - [CrunchDAO docs](https://docs.crunchdao.com/competitions/competitions/structural-break-real-time)
  - [Competition page](https://hub.crunchdao.com/competitions/structural-break-real-time)
  - [ADIA Lab page](https://www.adialab.ae/adia-lab-x-crunch-the-structural-break-challenge-2026)

## Related: Structural Break Open Benchmark

- Type: code competition. Submissions must produce deterministic output,
  otherwise they are ineligible for rewards.
- Ranking: private leaderboard
- Links:
  - [CrunchDAO docs](https://docs.crunchdao.com/competitions/competitions/structural-break-open-benchmark)

## Ideas for New Competitions for Quant Hedge Funds

Format for all ideas: out-of-sample scoring on hidden data, with a private
leaderboard, like the ADIA Lab and CrunchDAO competitions.

### 1. Covariance Forecasting

- Task: predict the next-period covariance matrix of 500 anonymized assets
- Metric: realized volatility of the minimum-variance portfolio built from the
  forecast
- Why funds care: risk and portfolio construction depend on it. Noise in the
  sample covariance is the main failure point.

### 2. Backtest Overfitting Detector

- Task: given the backtest PnL of a strategy and the number of trials tried,
  predict if the strategy survives live
- Data: thousands of simulated and real strategy series with a known
  out-of-sample outcome
- Metric: AUC on live-period Sharpe above zero
- Why funds care: every fund fights false discoveries. The deflated Sharpe
  ratio is a weak baseline to beat.

### 3. Market Impact and Execution Cost

- Task: predict the slippage of a parent order from order size, spread, depth,
  and volatility features
- Metric: error against the realized implementation shortfall
- Why funds care: execution cost decides if a signal is tradable. Public data on
  this is rare, so a synthetic or anonymized set has high value.

### 4. Factor Decay and Crowding Crash

- Task: given anonymized factor return series and crowding proxies, predict the
  factor drawdown over the next N weeks
- Metric: AUC for drawdown events, plus lead time
- Why funds care: quant crashes such as Aug 2007 and Mar 2020. An early warning
  protects capital.

### 5. Causal Feature Selection Under Regime Shift

- Task: pick the predictors with a causal link to returns from a set that
  includes spurious ones
- Data: synthetic markets with a known causal graph and hidden interventions.
  The test split changes the regime.
- Metric: out-of-sample information coefficient (IC) after the shift
- Why funds care: causal features survive a regime change. Links to ADIA Causal
  Discovery 2024 and to `research/Causal_Analysis_of_Financial_Tradability`.

### Suggested First Build

- Idea 2, the backtest overfitting detector
- Reason: it is cheap to simulate, has a clear label, and few public benchmarks
  exist

## More Ideas: Causality, Automated Research, and Agents

Same format as above: hidden data, private leaderboard.

### 6. Causal Event Study with Known Effect

- Task: estimate the causal effect of events (earnings surprise, index
  inclusion, buyback) on returns, despite confounding
- Data: real returns plus injected effects of known size. The truth is known,
  and the selection bias is built in.
- Metric: error between the estimated and the true effect, plus confidence
  interval coverage
- Why funds care: event-driven desks need effects that are causal, not
  correlation with size or momentum

### 7. Counterfactual Scenario Forecasting

- Task: predict asset returns under an intervention (rate shock, oil spike) in
  a macro-financial simulator
- Data: observational training data from the simulator. The test data comes from
  intervened runs.
- Metric: error against the simulator truth under intervention
- Why funds care: stress tests today are mostly historical replays. Causal
  models are needed for new scenarios.

### 8. Paper Replication Agent

- Task: an agent reads a finance paper (an anomaly or factor paper) and rebuilds
  the result from raw data
- Metric: match to the published tables, plus a score for correct handling of
  point-in-time data and sample choice
- Why funds care: replication is most of the junior research work. It also
  measures how much published alpha is real.

### 9. Lookahead Bias Auditor Agent

- Task: given a research pipeline (code and data), find injected leaks:
  lookahead, survivorship, point-in-time violations
- Metric: precision and recall on injected bugs, plus a false alarm penalty on
  clean pipelines
- Why funds care: leaks are the top cause of backtests that fail live. A good
  auditor works as research QA.

### 10. Alpha Research Agent Tournament

- Task: each agent gets a sandbox with data, a compute budget, and a trial
  budget. It proposes and tests signals, then submits a final signal.
- Metric: live forward-period IC, with a penalty for the number of trials used
- Why funds care: it tests whether agents can find real signal and control
  false discoveries, not only produce backtests

### Suggested First Build for Ideas 6 to 10

- Idea 9, the lookahead bias auditor agent
- Reason: labels are cheap to create by injecting bugs, and the score is clear
- Link to idea 2: the trial-count logic is shared, and it also feeds idea 10
- Related repo work:
  - Ideas 6 and 7: `research/Causal_Analysis_of_Financial_Tradability`
  - Ideas 8 to 10: `research/agentic_data_science`

## Notes

- The years for Market Prediction and Causal Discovery come from the sources.
  The year 2025 for Structural Break is inferred from a search result.
- No `hub.crunchdao.com` links were found for competitions 1, 2, 3 and the Open
  Benchmark. The CrunchDAO docs pages are used instead.
- The CrunchDAO docs index lists all competition pages:
  [llms.txt](https://docs.crunchdao.com/llms.txt)
- Stanford GSB describes Causal Discovery as classifying each variable's causal
  role (confounder, mediator, collider). The CrunchDAO page describes it as
  recovering the full graph. Check the official rules for the exact task.
