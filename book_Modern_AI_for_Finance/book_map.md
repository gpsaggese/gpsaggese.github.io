# Summary

// claude> Execute book_scripts/prompt.create_lesson_slides_from_chap_toc.md
// > gen_slides.py -i book_Modern_AI_for_Finance/10.01

## Title

- Modern AI for Finance

## Target Audience

- Quantitative researchers, ML engineers, and graduate students who apply deep
  learning to financial problems
- Assumes working knowledge of Python, probability, linear algebra, and supervised
  machine learning
- Assumes no deep finance background: market structure and data are introduced where
  needed

## Approach of the Book

- Focus on:
  - Finance-specific failure modes first: non-stationarity, noise, overfitting
  - Modern architectures (transformers, LLMs) applied to concrete financial tasks
  - Intuition over math, with derivations only when needed
  - Making theory operational through packages and notebooks
  - For each chapter: `Why AI` bullets inside each topic pair finance problems with
    how AI, ML, and coding automation help
- Provide resources to go one level deeper:
  - Related classes (`msml610`, `data605`)
  - Related books (`book_Agentic_AI`, `book_springer`, etc.)
  - Papers and books on quantitative finance and deep learning

## Short TOC

- Foundations
  - 01: Why Finance Is Different
    - Finance primer: instruments, returns, risk and Sharpe ratio, factor models,
      market microstructure basics, costs, capacity
    - Non-stationarity, heavy tails, volatility clustering, low signal-to-noise
  - 02: Strategy Backtesting
    - Backtest mechanics: simulation, fills, costs, look-ahead and survivorship bias
    - Backtest discipline: overfitting, multiple testing, purged CV, deflated Sharpe
    - Causal vs. predictive: spurious factors
  - 03: Deep Learning, Transformers, LLMs, and Agents
    - Refresher: deep learning, transformers, LLMs
    - Adapting LLMs: fine-tuning, RAG, domain-pretrained models
    - Agents
- Data Engineering
  - 04: Data Engineering
    - Point-in-time correctness (as-of joins)
    - Survivorship bias and delisting returns
    - Restatements and revisions
    - Corporate actions and adjusted prices
    - Identifiers and symbology (CUSIP, ISIN, PERMNO)
    - Timestamps, time zones, calendars
    - Data vendors (CRSP, Compustat, TAQ, EDGAR)
    - Alternative data: classification and practice
      - Satellite and web data
      - ESG and climate risk data
  - 05: News, Filings, Earnings Calls
    - LLMs for finance text
    - Conversational finance assistants
  - 06: Auto EDA
    - Levels of automation: profilers, AutoML, LLM agents
    - Automated profiling: schema, missing values, outliers, distributions
    - Finance-specific checks: look-ahead, survivorship, stale prices, corporate
      action breaks
    - Time-aware EDA: split by time first, structural breaks
    - Stylized facts: fat tails, volatility clustering, autocorrelation
    - EDA as snooping: every look is a trial
    - Agents generating hypotheses, features, and candidate signals
    - Reproducible EDA reports: generated notebooks with human review
- Prediction and Portfolio Construction
  - 07: Forecasting
    - Baselines first: linear, ARIMA, GBDT
    - Time series foundation models (Chronos, TimesFM, Lag-Lama)
    - Alpha research: signals, IC, turnover, decay
    - Factors: classic, ML-based, factor zoo, causal factors
    - Regime detection and macro nowcasting
    - Uncertainty quantification: probabilistic forecasts, calibration, conformal
      prediction
    - Volatility Forecasting
      - GARCH for volatility clustering
  - 08: Limit Order Book Modeling
    - Microstructure and LOB mechanics
    - Market impact and execution cost
  - 09: Portfolio Optimization and Reinforcement Learning
    - Portfolio optimization and constraints
    - Derivatives and fixed income
      - Neural option pricing and volatility surface
      - Yield curve and rates
      - Hedging (deep hedging)
    - RL where it works: execution, market making, hedging
    - RL where it is mostly hype: end-to-end trading
- Research Loop
  - 10: Auto DS
    - AutoML for finance: model and hyperparameter search under purged CV
    - Auto feature engineering: generated features, selection, leakage checks
    - Auto label generation and objective choice
    - Experiment tracking: trial counts feed the deflated Sharpe ratio
    - Human review gates and guardrails against automated overfitting
  - 11: Agents: Automating the Research Loop
    - Tool-using agents and architecture: planner, tools, sandbox, critic, memory
    - Research loop: agents per pipeline step, single vs. multi-agent
    - Data science agent landscape and evaluation: benchmarks, scorecards
    - Verification, security, and provenance
- Risk, Operations, and Governance
  - 12: Risk, Credit Risk, and Fraud
    - Market risk: VaR/ES, covariance, factor risk models
    - Credit risk and fraud detection
    - Uncertainty in credit: calibrated default probabilities, conformal prediction
  - 13: Explainability and Robustness
    - Uncertainty and trust: calibration, conformal prediction
  - 14: Regulation, Model Risk Management, and Deployment
- Frontiers
  - 15: Market Simulation
    - GANs / diffusion models for scenario generation
    - Agent-based market simulation
    - Stress testing
    - Synthetic data: augmenting scarce data
    - Fidelity: stylized facts (fat tails, volatility clustering)
  - 16: The Road Ahead
    - Multimodal models
    - Open problems

## All Lesson Materials

- Read the content of the file `/Users/saggese/src/notes1/finance_index.md` with
  pointers to more content
  - When referring to that content use ` //notes1/<file>`
- `data605/all_tocs.md`
- `data605/lectures_source/*.smd`
- `msml610/all_tocs.md`
- `msml610/lectures_source/*.smd`
- `book_Agentic_AI/all_tocs.md`
- `book_Agentic_AI/lectures_source/*.smd`
- `book_springer/all_tocs.md`
- `book_springer/lectures_source/*.smd`
- `book_cs_refreshers/lectures_source/*.smd`

# Roadmap

| Chap                                                  | Slides                                   | Slides % | Criticize | Tutorial                              | Book |
| :---------------------------------------------------- | :--------------------------------------- | :------- | :-------- | :------------------------------------ | :--- |
| **Foundations**                                       |                                          |          |           |                                       |      |
| 01: Why Finance Is Different                          | Lesson01.01_Why_Finance_Is_Different.smd |          |           | L01_01_why_finance_is_different.ipynb |      |
| 02: Strategy Backtesting                              | Lesson02.01_Strategy_Backtesting.smd     |          |           | L02_01_strategy_backtesting.ipynb     |      |
| 03: Deep Learning, Transformers, LLMs, and Agents     | N/A                                      |          |           |                                       |      |
| **Data Engineering**                                  |                                          |          |           |                                       |      |
| 04: Data Engineering                                  | N/A                                      |          |           |                                       |      |
| 05: News, Filings, Earnings Calls                     | N/A                                      |          |           |                                       |      |
| 06: Auto EDA                                          | Lesson06.01_Auto_EDA.smd                 |          |           | L06_01_auto_eda.ipynb                 |      |
| **Prediction and Portfolio Construction**             |                                          |          |           |                                       |      |
| 07: Forecasting                                       | N/A                                      |          |           |                                       |      |
| 08: Limit Order Book Modeling                         | N/A                                      |          |           |                                       |      |
| 09: Portfolio Optimization and Reinforcement Learning | N/A                                      |          |           |                                       |      |
| **Research Loop**                                     |                                          |          |           |                                       |      |
| 10: Auto DS                                           | Lesson10.01_Auto_DS.smd                  |          |           | L10_01_auto_ds.ipynb                  |      |
| 11: Agents: Automating the Research Loop              | Lesson11.01...                           |          |           |                                       |      |
| **Risk, Operations, and Governance**                  |                                          |          |           |                                       |      |
| 12: Risk, Credit Risk, and Fraud                      | N/A                                      |          |           |                                       |      |
| 13: Explainability and Robustness                     | N/A                                      |          |           |                                       |      |
| 14: Regulation, Model Risk Management, and Deployment | N/A                                      |          |           |                                       |      |
| **Frontiers**                                         |                                          |          |           |                                       |      |
| 15: Market Simulation                                 | N/A                                      |          |           |                                       |      |
| 16: The Road Ahead                                    | N/A                                      |          |           |                                       |      |

## `book_Modern_AI_for_Finance` Tutorials

> find book_Modern_AI_for_Finance/tutorials -name \*.ipynb

```
book_Modern_AI_for_Finance/tutorials/L01_why_finance_is_different/L01_01_why_finance_is_different.ipynb
book_Modern_AI_for_Finance/tutorials/L02_strategy_backtesting/L02_01_strategy_backtesting.ipynb
book_Modern_AI_for_Finance/tutorials/L06_auto_eda/L06_01_auto_eda.ipynb
book_Modern_AI_for_Finance/tutorials/L10_auto_ds/L10_01_auto_ds.ipynb
```

## TODOs

- Write one slide deck per chapter in `book_Modern_AI_for_Finance/lectures_source/`
  (30-35 slides each)
- Write the finance-specific content from scratch for chapters where the
- Add tutorial notebooks under `book_Modern_AI_for_Finance/tutorials/`
- Collect related books, packages, and papers for each chapter: the previous map had
  empty placeholders for them
- Generate `book_Modern_AI_for_Finance/all_tocs.md` once the slide decks exist

# Detailed TOC

# Part I: Foundations

## 01: Why Finance Is Different

### Goals

- Introduce the finance basics: instruments, returns, risk, factor models, and
  implementation costs
- Explain why non-stationarity, heavy tails, volatility clustering, and weak signals
  break the assumptions of standard ML
- Preview the failure modes that chapter 02 handles with backtest discipline

### Topics

- Finance Primer
  - Financial instruments: equities, bonds, futures, and options
  - Measuring returns: simple vs. log returns, compounding, and excess returns
  - Measuring risk and performance: volatility, drawdown, and the Sharpe ratio
  - Factor models: market beta, style factors (value, momentum, size, quality, low
    volatility), and alpha vs. beta
  - Market microstructure basics: order books, bid-ask spread, liquidity, and order
    types
  - Implementation realities: transaction costs, market impact, and strategy capacity
  - Why AI:
    - Hidden costs: transaction costs, market impact, and capacity erase paper
      profits
      - Automation: tested library functions encode one definition of returns, risk,
        and costs
- Statistical Properties of Financial Data
  - Non-stationarity: drifting distributions and regime shifts
  - Heavy tails: extreme events and the limits of the normal distribution
  - Dependence structure: volatility clustering and time-varying correlations
  - Weak signals: low signal-to-noise ratios and small effective sample sizes
  - Why AI:
    - Non-stationarity: return distributions and relationships drift, so past fits
      fail live
      - ML: regime-switching and change-point models detect drift and re-estimate
        parameters
    - Heavy tails: extreme events are far more frequent than a normal distribution
      implies
      - ML: heavy-tailed distributions and quantile losses model extremes explicitly
    - Weak signals: low signal-to-noise and few independent samples limit what can be
      learned
      - ML: strong regularization, shrinkage, and ensembles avoid fitting noise
    - Limits: ML cannot create signal that is not in the data

### Slides

- `book_Modern_AI_for_Finance/lectures_source/Lesson01.01_Why_Finance_Is_Different.smd`

### Lesson Materials

- `//notes1/notes/finance.all.txt`
  - [30%]: Market structure, trading hours, order types, quotes, short selling,
    arbitrage and no-arbitrage, simple, log, multi-period, and portfolio returns,
    annualization, derivatives and fixed-income basics, equity costs
- `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.smd`
  - [20%]: Strict and weak stationarity, ACF, AR(1) for financial series, ARCH/GARCH
    volatility, change-point detection, Markov-switching models
- `book_springer/lectures_source/Lesson08.1_Causal_Data_Pipelines.smd`
  - [10%]: Distribution shift, concept drift, non-stationarity, detecting shift in
    production
- `msml610/lectures_source/Lesson05.2-Overfitting.smd`
  - [5%]: Deterministic vs. stochastic noise, data size vs. noise
- `book_cs_refreshers/lectures_source/Lesson91.Refresher_probability.smd`
  - [5%]: Kurtosis and excess kurtosis as the definition of fat tails
- `//notes1/notes/IN_PROGRESS.finance.Active_portfolio_management.Grinold_Kahn.1999.txt`
  - [5%]: Alpha vs. beta, residual risk, ex-ante and ex-post information ratio,
    elementary factor risk models
- `//notes1/notes/IN_PROGRESS.finance.portfolio_theory.txt`
  - [5%]: Constant expected return model, single-index model, classical factor models
- `//notes1/notes/IN_PROGRESS.finance.Quantitative_trading.Chan.2009.txt`
  - [5%]: Performance measurement, transaction costs, leverage, why live performance
    diverges from backtests
- `//notes1/notes/IN_PROGRESS.finance.Derivatives.Hull.2010.txt`
  - [5%]: Forwards, futures, options, hedgers, speculators, and arbitrageurs
- `//notes1/notes/finance.time_series.txt`
  - [5%]: Strict and weak stationarity, ACF and PACF, trend and seasonality,
    transforming non-stationary series
- _Not covered_
  - [15%]: Extreme-value and tail-risk modeling, time-varying correlations, adaptive
    markets, strategy capacity

## 02: Strategy Backtesting

### Goals

- Simulate strategies realistically: signals, positions, fills, and costs
- Explain how overfitting, data snooping, and multiple testing create false
  discoveries
- Validate with purged cross-validation, deflated Sharpe, and causal reasoning

### Topics

- Backtest Mechanics
  - Simulation of signals, positions, fills, and transaction costs
  - Look-ahead and survivorship bias in backtests
  - Why AI:
    - Unrealistic simulation: ignoring fills, queue position, and costs inflates
      returns
      - Automation: event-driven backtesters with unit-tested fill and cost models
        make results reproducible
    - Information leakage: look-ahead and survivorship bias put future data into past
      decisions
      - Automation: as-of assertions and generated tests catch look-ahead bugs before
        a run
- Backtest Discipline
  - Backtest overfitting and data snooping
  - Multiple testing and false discoveries
  - Purged and embargoed cross-validation
  - Deflated Sharpe ratio
  - Why AI:
    - Backtest overfitting: many trials on the same data make luck look like skill
      - ML: purged and embargoed cross-validation and the deflated Sharpe ratio
        correct for the number of trials
      - Automation: experiment tracking logs every trial, so the trial count is known
- Causal vs. Predictive
  - Predictive vs. causal questions
  - Spurious factors and confounding
  - Why AI:
    - Spurious factors: predictive patterns without causal links fail out of sample
      - ML: causal discovery and causal ML separate confounded factors from causal
        ones

### Slides

- `book_Modern_AI_for_Finance/lectures_source/Lesson02.01_Strategy_Backtesting.smd`

### Lesson Materials

- `//notes1/notes/IN_PROGRESS.finance.Advances_in_Financial_ML.de_Prado.2018.md`
  - [35%]: Why financial ML fails, labeling, sample weights, purged and embargoed
    cross-validation, dangers of backtesting, backtesting through cross-validation,
    backtests on synthetic data, backtest statistics including the deflated Sharpe
    ratio, strategy risk
- `msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd`
  - [20%]: Data snooping, "burning the test set", sampling bias, out-of-sample
    discipline, multiple comparisons
- `//notes1/notes/IN_PROGRESS.finance.Algorithmic_trading.Chan.2013.txt`
  - [15%]: Importance of backtesting, common pitfalls, statistical significance of
    backtests, when not to backtest, whether a backtest predicts future returns
- `msml610/lectures_source/Lesson05.2-Overfitting.smd`
  - [10%]: Overfitting, bias-variance trade-off
- `book_cs_refreshers/lectures_source/Lesson91.Refresher_probability.smd`
  - [10%]: Multiple hypothesis testing, p-hacking, FWER and Bonferroni, FDR and
    Benjamini-Hochberg
- `msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd`
  - [10%]: Association vs. correlation vs. causation, what ML systems can and cannot
    tell you
- `book_springer/lectures_source/Lesson02.1_From_Data_Science_To_Decision_Science.smd`
  - [10%]: Causal vs. predictive questions, correlation that encodes confounding,
    selection bias, Simpson's paradox, collider bias
- `//notes1/notes/IN_PROGRESS.finance.Quantitative_trading.Chan.2009.txt`
  - [10%]: Backtesting platforms, pitfalls (look-ahead, survivorship, data- snooping
    bias), transaction costs, paper trading, gap between backtest and live
    performance
- `//notes1/notes/papers.finance.misc.txt`
  - [10%]: Multiple-testing crisis in financial research, evaluating trading
    strategies, FWER, Bonferroni, and FDR methods, event studies, backtesting
    systematic strategies
- `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.smd`
  - [5%]: Data leakage, walk-forward validation as time-aware cross-validation
- `msml610/lectures_source/Lesson05.3-Learn_Validation.smd`
  - [5%]: Cross-validation and repeated cross-validation as the basis for purged
    variants
- `//notes1/notes/IN_PROGRESS.finance.Machine_learning_for_asset_managers.de_Prado.2020.md`
  - [5%]: Two types of overfitting, testing set overfitting, false strategy theorem
- `//notes1/notes/papers.finance.data_mining_bias.txt`
  - [5%]: White's reality check, Diebold-Mariano test of predictive accuracy
- `//notes1/notes/papers.finance.causal_ai.txt`
  - [5%]: Causal factor investing: association vs. causation in factors
- `//notes1/notes/work.trading_ideas.txt`
  - [5%]: Own notes on false discovery rate and measuring overfitting
- _Not covered_
  - [10%]: Backtest simulator design and fill models, empirical case studies of
    spurious factors

## 03: Deep Learning, Transformers, LLMs, and Agents

### Goals

- Review the neural architectures and training methods used throughout the book
- Explain attention, transformers, and how LLMs are pretrained, scaled, evaluated
- Compare fine-tuning, instruction tuning, and RAG for adapting LLMs to finance

### Topics

- Neural Network Fundamentals
  - MLPs, backpropagation, and automatic differentiation
  - CNNs, RNNs, and LSTMs
- Training on Small and Noisy Datasets
  - Optimizers, learning-rate schedules, vanishing and exploding gradients
  - Weight decay, dropout, early stopping, and data augmentation
  - Why AI:
    - Small noisy data: deep models overfit when few independent samples exist
      - ML: weight decay, dropout, early stopping, and augmentation tuned to
        financial noise levels
- Attention and Transformers
  - Self-attention, multi-head attention, and positional encoding
  - Transformers vs. RNNs: long sequences, parallelism, and cost of attention
  - Variants for time series and multi-asset inputs
- LLM Pretraining and Scaling
  - Pretraining objectives and datasets
  - Scaling laws and emergent reasoning
- The LLM Landscape
  - Open vs. proprietary models
  - Evaluation, benchmarking, limitations, and failure modes
  - Why AI:
    - Model choice: many architectures, with little guidance on what fits tabular,
      sequence, or text data
      - Automation: benchmark harnesses compare baselines and deep models on
        identical splits
- Adapting LLMs to Finance
  - Fine-tuning and instruction tuning for financial tasks
  - RAG over filings, news, and research, and RAG vs. fine-tuning trade-offs
  - Domain-pretrained models such as BloombergGPT
  - Why AI:
    - Domain gap: general models lack finance vocabulary and current, sourced facts
      - LLM: fine-tuning, instruction tuning, and domain-pretrained models adapt
        general models to financial tasks
      - LLM: RAG grounds answers in filings, news, and research instead of model
        memory
    - Cost: large models are expensive to train, serve, and evaluate
      - ML: parameter-efficient fine-tuning and smaller models cut training and
        serving cost

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [35%]: Perceptron and MLP, activations, backpropagation, automatic
    differentiation, CNNs and ResNets, RNNs, vanishing gradients, LSTM/GRU, weight
    initialization, batch normalization, learning-rate schedules, early stopping,
    regularization, data augmentation, RNN limits, sequence-to-sequence attention,
    types of attention, transformer architecture, pretraining
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [30%]: Query-key-value attention, scaled dot-product, multi-head, self vs. cross
    vs. causal attention, positional encoding, transformer block, residual
    connections, LayerNorm, attention cost, masked vs. autoregressive objectives,
    pretraining pipeline, scaling laws, emergent abilities, decoding, context
    windows, expressivity limits
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [10%]: RAG loop, retrieval as a tool call, vector databases, approximate
    nearest-neighbor indexes, hybrid search, grounding in regulated domains, long
    context vs. retrieval, needle-in-a-haystack evaluation of long context
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [10%]: Chain-of-thought and variants, premise-order brittleness, limits of
    self-correction
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [10%]: Data curation, multi-stage training pipelines, case studies of open
    frontier models, training cost
- `//notes1/notes/papers.finance.deep_learning.txt`
  - [10%]: Paper reviews on deep learning in finance: DNN ensembles for index
    prediction, cross-sectional return forecasting, HFT with DNNs, NLP for stock
    prediction; distillation and best practices
- `msml610/lectures_source/Lesson05.2-Overfitting.smd`
  - [5%]: Bias-variance analysis, learning curves, high-bias vs. high-variance
    regimes as the basis for regularization
- `msml610/lectures_source/Lesson05.3-Learn_Validation.smd`
  - [5%]: Cross-validation and bootstrap for tuning on scarce data
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [5%]: Verifiable benchmarks, reward hacking, the verification gap
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [5%]: Preference data, reward models and PPO, DPO, iterative preference
    optimization
- `book_Agentic_AI/lectures_source/Lesson10.1_Open_training_recipes_for_reasoning.smd`
  - [5%]: Open post-training pipeline, DPO vs. PPO, preference feedback collection,
    RAG plus reasoning training for scientific synthesis
- `//notes1/notes/IN_PROGRESS.finance.Advances_in_Financial_ML.de_Prado.2018.md`
  - [5%]: Time-aware cross-validation and hyper-parameter tuning with cross-
    validation (ch 7 and 9)
- _Not covered_
  - [15%]: Regularization tuned to financial noise levels, time series transformers
    and multi-asset attention, survey of general-purpose benchmark suites,
    composition of pretraining datasets, parameter-efficient fine- tuning, supervised
    instruction tuning on financial tasks, domain-pretrained finance models

# Part II: Data Engineering

## 04: Data Engineering

### Goals

- Build point-in-time correct datasets free of look-ahead and survivorship bias
- Handle restatements, corporate actions, identifiers, and timestamps correctly
- Survey market, fundamental, filing, and alternative data vendors and their pitfalls

### Topics

- Point-in-Time Correctness
  - Look-ahead bias and as-of joins
  - Restatements and data revisions
  - Why AI:
    - Point-in-time errors: restated and revised values leak into past decisions
      - Automation: pipelines enforce as-of joins and fail loudly on future-dated
        records
- Survivorship Bias
  - Survivorship-bias-free universes
  - Delisting returns
  - Why AI:
    - Survivorship bias: datasets without delisted firms inflate returns
      - Automation: data-quality tests check that the universe includes delistings
        and delisting returns
- Reference Data
  - Corporate actions: splits, dividends, and adjusted prices
  - Identifiers and symbology: CUSIP, ISIN, PERMNO
  - Timestamps, time zones, and trading calendars
  - Why AI:
    - Messy reference data: splits, dividends, identifiers (CUSIP, ISIN, PERMNO),
      time zones, and calendars break joins
      - ML: entity resolution maps names and tickers across identifier systems
      - ML: anomaly detection flags bad ticks and unadjusted splits
- Data Vendors
  - Market data: CRSP and TAQ
  - Fundamental data: Compustat
  - Filings: EDGAR
  - Why AI:
    - Vendor formats: vendors differ in format, coverage, and quality
      - Automation: generated and tested vendor adapters share one schema
- Alternative Data
  - Classification of alternative data and practical pitfalls
  - Satellite and web data
  - ESG and climate risk data
  - Why AI:
    - Unstructured sources: web pages and documents have no fixed schema
      - LLM: extraction models turn filings, web pages, and PDFs into structured,
        timestamped fields

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

- `data605/lectures_source/Lesson02.3-Data_Pipelines.smd`
  - [15%]: Data ingestion, ETL/ELT paradigms, workflow orchestration, data lake
    vs. warehouse, data cleaning
- `data605/lectures_source/Lesson07.2-Data_Wrangling.smd`
  - [15%]: Tidy data, single-source and multi-source data problems, outlier detection
    including time series outliers
- `book_springer/lectures_source/Lesson08.1_Causal_Data_Pipelines.smd`
  - [15%]: Selection bias as the root of survivorship bias, missing-data mechanisms,
    distribution shift, measurement error, pre-flight data quality checks
- `//notes1/notes/IN_PROGRESS.finance.Advances_in_Financial_ML.de_Prado.2018.md`
  - [15%]: Fundamental, market, analytics, and alternative data, release date
    vs. report date, backfilling, reinstated values, point-in-time, time, tick,
    and volume bars
- `data605/lectures_source/Lesson10.2-Streaming_and_Real_Time_Analytics.smd`
  - [10%]: Event time vs. processing time, delivery semantics, Kafka, Flink, Spark
    Structured Streaming
- `//notes1/notes/IN_PROGRESS.finance.Compustat.2017.txt`
  - [10%]: Compustat Snapshot as a point-in-time database: originally reported
    vs. restated values, observation date vs. effective date
- `data605/lectures_source/Lesson07.3-Serialization_Formats.smd`
  - [5%]: CSV, Parquet, JSON, and Protocol Buffers for storing market data
- `//notes1/notes/finance.all.txt`
  - [5%]: Trading hours and time zones, adjusting equity prices, continuous
    adjustments for futures
- `//notes1/notes/IN_PROGRESS.finance.Quantitative_trading.Chan.2009.txt`
  - [5%]: Historical databases, survivorship bias, and look-ahead bias in market data
- `//notes1/notes/IN_PROGRESS.finance.XBRL.txt`
  - [5%]: XBRL markup for financial reporting
- `//notes1/notes/papers.finance.edgar.txt`
  - [5%]: EDGAR filings: Form 4 insider data, filing dissemination timing, changes in
    10-K text
- `//notes1/notes/finance.bloomberg.txt`
  - [5%]: Bloomberg terminal data access: identifiers such as CUSIP, functions,
    cheatsheets
- `//notes1/notes/papers.finance.misc.txt`
  - [5%]: Paper reviews on Compustat, market data, and alternative data
- _Not covered_
  - [30%]: Corporate actions in detail (splits, delistings), identifiers and
    symbology (ISIN, PERMNO), CRSP and TAQ dataset documentation, satellite, web, and
    ESG data

## 05: News, Filings, Earnings Calls

### Goals

- Extract signals from news, filings, and earnings-call transcripts
- Apply LLMs to sentiment, entity, and event extraction on financial text
- Link text events to market data over time without look-ahead bias

### Topics

- Financial Text Sources
  - News feeds, SEC filings (10-K, 10-Q, 8-K), earnings-call transcripts
  - Parsing filings and aligning text with timestamps
  - Why AI:
    - Unstructured text: filings and transcripts mix boilerplate with signal
      - Automation: parsers for filings and transcripts produce timestamped,
        sectioned text
- LLMs for Finance Text
  - Sentiment: lexicon-based vs. LLM-based
  - Named entity and event extraction
  - Financial text classification and summarization
  - Why AI:
    - Volume: thousands of documents per day exceed analyst reading capacity
      - LLM: sentiment, entity, and event extraction and summarization run at scale
- From Text to Signals
  - Event studies: linking text events to returns
  - Look-ahead bias from LLM pretraining data
  - Why AI:
    - Timing: text events must be aligned with market data without look-ahead
      - ML: event studies measure abnormal returns around aligned events
    - Pretraining leakage: LLMs may have seen the outcome during pretraining
      - Automation: evaluation splits use only data after the model training cutoff
- Assistant Use Cases
  - Advisory chatbots
  - Natural-language portfolio querying
  - Why AI:
    - Access: users cannot query portfolios and market data in natural language
      - LLM: natural-language requests map to portfolio queries and tool calls
- Reliability
  - Guardrails for factual accuracy
  - Handling ambiguity and uncertainty
  - Why AI:
    - Accuracy: a wrong answer has financial and legal cost
      - LLM: retrieval grounding, citations, and guardrails constrain answers to
        verified facts
    - Ambiguity: questions are underspecified and answers are uncertain
      - LLM: assistants ask clarifying questions and state uncertainty
- Human Factors
  - User experience and trust
  - Why AI:
    - Trust: users over-rely on or reject algorithmic advice
      - ML: calibrated uncertainty and escalation to humans set user expectations

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

// TODO(gp): Recompute coverage after merging Conversational Assistants
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [30%]: NLP tasks, word embeddings, language models, sequence-to-sequence models,
    transformers, masked language models, pretraining
- `//notes1/notes/IN_PROGRESS.finance.The_handbook_of_news_analytics.Mitra.2011.txt`
  - [20%]: News data, turning text into metrics and time series, news analytics
    framework, news and abnormal returns, portfolio simulation, news flow and asset
    returns
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [15%]: Tokenization, masked vs. autoregressive pretraining objectives
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [15%]: Embeddings, hybrid search, grounding on regulated-domain documents
  - [25%]: Grounding on enterprise knowledge, high-fidelity grounding for regulated
    domains, fact-checking, retrieving vs. stuffing context
- `//notes1/notes/papers.finance.misc.txt`
  - [15%]: News analytics, sentiment, and social sentiment paper reviews: event
    studies (MacKinlay), RavenPack and Macquarie news sentiment, which news moves
    prices, Twitter sentiment
- `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd`
  - [10%]: Interrupted time series, difference-in-differences, synthetic control as
    event-study designs
- `//notes1/notes/papers.finance.edgar.txt`
  - [10%]: Filings: Form 4, text changes in 10-K filings, SEC dissemination timing
- `//notes1/notes/IN_PROGRESS.finance.The_handbook_of_sentiment_analysis.Mitra.2016.txt`
  - [5%]: Sentiment analysis in finance
- `//notes1/notes/papers.finance.equities.news.txt`
  - [5%]: News vs. sentiment for predicting stock returns
- `//notes1/notes/papers.finance.equities.earnings.txt`
  - [5%]: Papers on earnings and equities
- `//notes1/notes/IN_PROGRESS.papers.finance.earning_announcements.txt`
  - [5%]: NLP on earnings call transcripts (stub)
- `//notes1/notes/IN_PROGRESS.finance.XBRL.txt`
  - [5%]: XBRL for parsing financial reports
- `//notes1/notes/IN_PROGRESS.finance.Handbook_of_equity_market_anomalies.Zacks.2011.txt`
  - [5%]: Post-earnings announcement drift and earnings surprise as event-study
    targets
- `//notes1/notes/papers.finance.deep_learning.txt`
  - [5%]: Predicting stock performance with NLP deep learning
- `//notes1/notes/finance.all.txt`
  - [5%]: News analytics under behavioral economics
- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [20%]: Agents vs. chatbots, spectrum of autonomy, human-in-the-loop vs. full
    autonomy
- `//notes1/notes/IN_PROGRESS.course.finance.Fintech_Wharton.txt`
  - [20%]: Robo-advisors (services, assets under management), clinical vs. actuarial
    decisions, trust in FinTech, algorithm aversion, choice architecture
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [15%]: Chain-of-Verification to reduce hallucination
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [15%]: Transparency, causal explanations, safety and trustworthy autonomy
- `msml610/lectures_source/Lesson11.1-Decision_Making_with_Causal_Models.smd`
  - [10%]: Aleatoric vs. epistemic uncertainty, communicating uncertainty to
    stakeholders
- _Not covered_
  - [20%]: LLM-based financial sentiment, named entity and event extraction with
    LLMs, SEC filing parsing pipelines, look-ahead bias from LLM pretraining data
  - [40%]: Advisory chatbot design, natural-language portfolio querying, trust
    measurement for LLM assistants

## 06: Auto EDA

### Goals

- Automate profiling of financial datasets: schema, missing values, outliers,
  distributions
- Catch finance-specific data errors: look-ahead, survivorship, stale prices,
  corporate actions
- Use agents to propose hypotheses and features, with logged looks and human review

### Topics

- Levels of Automation
  - Profilers and rules: fixed checks, no search
  - AutoML: search over a predefined space with a clear objective
  - LLM agents: open-ended search over analyses, with code generation
- Automated Profiling
  - Rule-based profilers: schema, missing values, outliers, and distributions
  - Panel structure: coverage and missingness by asset and date
- Finance-Specific Checks
  - Look-ahead and survivorship bias
  - Stale prices and corporate action breaks
  - Leakage detection: a known weak spot of agents
- Time-Aware EDA
  - Split by time first: explore the training period only
  - Stationarity, structural breaks, and regime-aware summaries
- Stylized Facts
  - Fat tails, volatility clustering, and autocorrelation
- EDA as Snooping
  - Every plot and hypothesis is a trial
  - More automation widens the search and multiplies trials
  - Logging the number of looks and holding out data
- Agent-Driven EDA
  - Hypotheses, features, and candidate signals
  - Chart selection: correct semantics vs. code that merely runs
  - Agent architecture and verification: see chapter 11
- Reproducible EDA Reports
  - Generated notebooks with human review

### TODO

- [ ] Add `Why AI` bullets to each topic

### Slides

- `book_Modern_AI_for_Finance/lectures_source/Lesson06.01_Auto_EDA.smd`

### Lesson Materials

// TODO(gp): Check the coverage percentages: estimated from the deck's sources
- `//notes1/notes/IN_PROGRESS.finance.Advances_in_Financial_ML.de_Prado.2018.md`
  - [30%]: Look-ahead and survivorship bias, point-in-time data, structural breaks
    and CUSUM tests, stationarity with maximum memory preservation, empirical
    observations about financial time series, multiple testing
- `research/agentic_data_science/sota_eda.md`
  - [25%]: Definitions and scope of agentic EDA, systems comparison, agent loop,
    failure modes in benchmarks and deployments, reproducibility and safety
    recommendations
- `research/agentic_data_science/agentic_data_science.report.md`
  - [15%]: Agents as search systems, current capabilities, where agents still fail
    (leakage, metric selection), the changing role of humans
- `data605/lectures_source/Lesson07.2-Data_Wrangling.smd`
  - [15%]: Data wrangling workflow, classification of data problems, single-source
    problems, univariate and time series outlier detection
- `book_Modern_AI_for_Finance/lectures_source/Lesson02.01_Strategy_Backtesting.smd`
  - [15%]: Subtle sources of leakage, survivorship bias, false strategy theorem, the
    backtest protocol that the Auto EDA protocol mirrors
- `msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd`
  - [10%]: Data cleaning, outliers, missing data and imputation, fit on train and
    apply to validation and test
- `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.smd`
  - [10%]: Stationarity, cross-validation for time series, change-point detection,
    Markov-switching models
- `msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd`
  - [10%]: Data snooping, "burning the test set", right and wrong approach to
    research
- `research/agentic_data_science/agentic_eda.notes.md`
  - [10%]: Goals of agentic EDA, analysis templates by data type, EDA plan proposal
- `research/agentic_data_science/AutoEDA_Plan.md`
  - [5%]: Out-of-sample validation system, user approval of the EDA plan
- `research/agentic_data_science/RL_for_auto_EDA_sota_review.md`
  - [5%]: Descriptive report generators, search-based pipeline cleaning and feature
    engineering
- `research/ideas/in_progress.Comparison_of_Data_Science_Agents.md`
  - [5%]: Key examples of data science agents, leakage failure case
- `research/ideas/draft.Benchmarking_Data_Science_Agents.md`
  - [5%]: Benchmark landscape for data science agents
- `book_Modern_AI_for_Finance/lectures_source/Lesson01.01_Why_Finance_Is_Different.smd`
  - [5%]: Running example (momentum strategy), stylized facts and ML failure modes
- `//notes1/notes/finance.time_series.txt`
  - [5%]: ADF test, Ljung-Box test
- `//notes1/notes/papers.finance.misc.txt`
  - [5%]: De Prado on the multiple-testing crisis in financial research
- `msml610/lectures_source/Lesson05.1-Learning_Theory.smd`
  - [5%]: Data snooping voids the VC guarantee
- `book_cs_refreshers/lectures_source/Lesson91.Refresher_probability.smd`
  - [5%]: P-hacking
- `msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd`
  - [5%]: EDA vs. inferential statistics
- `data605/lectures_source/Lesson01.3-Is_Data_Science_Just_Hype.smd`
  - [5%]: Where a data scientist spends most of the time
- `book_springer/lectures_source/Lesson02.1_From_Data_Science_To_Decision_Science.smd`
  - [5%]: Analytics maturity ladder
- `book_springer/lectures_source/Lesson15.1_Deployment_Monitoring_And_Adaptation.smd`
  - [5%]: Notebook-to-production gap
- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [5%]: Human-in-the-loop vs. full autonomy
- `msml610/lectures_source/Lesson12.2-Causal_Discovery.smd`
  - [5%]: Discovery as hypothesis generation
- _Not covered_
  - [5%]: Worked use of cleaning and feature-search tools (AlphaClean, featuretools,
    TPOT) that the deck only names, and finance-specific benchmark results for EDA
    agents, since the reported benchmarks are general data science tasks

# Part III: Prediction and Portfolio Construction

## 07: Forecasting

### Goals

- Establish strong baselines before reaching for deep models
- Apply time series foundation models and factor models to returns and volatility
- Evaluate signals by trading value, detect regimes, and quantify forecast
  uncertainty

### Topics

- Baselines First
  - Linear models and ARIMA for returns
  - GARCH for volatility clustering
  - Gradient-boosted decision trees (GBDT) on tabular features
  - Why AI:
    - Nonlinearity: linear baselines miss interactions between features
      - ML: GBDT and neural models capture nonlinear interactions that linear
        baselines miss
- Time Series Foundation Models
  - Chronos, TimesFM, and Lag-Llama
  - Zero-shot vs. fine-tuned forecasting against baselines
  - Why AI:
    - Short histories: many assets have too little data to fit models
      - ML: time series foundation models provide strong zero-shot baselines
- Alpha Research
  - Signals and the information coefficient (IC)
  - Turnover and signal decay
  - Why AI:
    - Weak, decaying alpha: signals have a low information coefficient and fade as
      others trade them
      - Automation: research pipelines compute IC, turnover, and decay for every
        signal in the same way
- Factors
  - Classic factors: market, size, value, momentum
  - ML-based factors
  - The factor zoo and false discoveries
  - Causal factors
  - Why AI:
    - Factor zoo: hundreds of published factors, many of them false discoveries
      - ML: multiple-testing correction and causal factor tests prune false
        discoveries
- Regimes and Macro
  - Regime detection
  - Macro nowcasting
  - Why AI:
    - Regime shifts: relationships change with the macro and volatility environment
      - ML: HMMs, change-point detection, and nowcasting models track the current
        regime
- Uncertainty Quantification
  - Probabilistic forecasts
  - Calibration
  - Conformal prediction
  - Why AI:
    - Point forecasts: a single number hides the uncertainty that drives sizing and
      risk
      - ML: probabilistic forecasts, calibration, and conformal prediction give valid
        intervals

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

- `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.smd`
  - [35%]: Stationarity, ACF, decomposition, AR/MA/ARIMA, ARCH/GARCH, VAR, state
    space models, ML and deep learning for time series, cross-validation for time
    series, probabilistic forecasting, change-point detection, Markov-switching
    models
- `//notes1/notes/IN_PROGRESS.finance.Machine_trading.Chan.2017.txt`
  - [15%]: Time-series and cross-sectional factors, AR, ARMA, VAR, state space
    models, regression trees, bagging, random forests, boosting, cross- validation
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [10%]: Aleatoric vs. epistemic uncertainty, calibration of neural networks,
    conformal prediction for deep models, transformer architecture
- `//notes1/notes/IN_PROGRESS.finance.Active_portfolio_management.Grinold_Kahn.1999.txt`
  - [10%]: Model of alpha, ex-ante and ex-post information ratio, APT, risk models
- `//notes1/notes/IN_PROGRESS.finance.Quantitative_equity_portfolio_management.Chincarini.2006.txt`
  - [10%]: Market efficiency, fundamental law of active management, statistical
    issues, fundamental, economic, and alternative factors, factor choice
- `//notes1/notes/IN_PROGRESS.finance.Algorithmic_trading.Chan.2013.txt`
  - [10%]: Mean reversion and momentum, cointegration, Kalman filter, cross-
    sectional strategies
- `//notes1/notes/papers.finance.alpha.txt`
  - [10%]: Statistical arbitrage in US equities: residual cointegration, factor
    decomposition of returns, market and factor neutrality
- `//notes1/notes/IN_PROGRESS.finance.Advances_in_Financial_ML.de_Prado.2018.md`
  - [10%]: Labeling, fractionally differentiated features, ensembles, structural
    breaks, entropy features
- `//notes1/notes/finance.time_series.txt`
  - [10%]: Stationarity, ACF and PACF, trend and seasonality, lag and difference
    operators
- `//notes1/notes/papers.finance.misc.txt`
  - [10%]: Return predictability (Hull, Haugen), time series and fundamental paper
    reviews, multiple testing of factors (Harvey), forecasting output and inflation
    with asset prices
- `msml610/lectures_source/Lesson09.5-Kalman_Filter.smd`
  - [5%]: Filtering and prediction in state space models as the basis for nowcasting
- `msml610/lectures_source/Lesson09.2-Hidden_Markov_Models.smd`
  - [5%]: HMMs and EM as regime-switching models
- `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd`
  - [5%]: Granger causality, time-varying confounders, feedback loops as tools for
    causal factors
- `book_springer/lectures_source/Lesson02.2_Integrating_Causality_And_Probability_in_ML.smd`
  - [5%]: Posteriors vs. point estimates, calibration and conformal prediction
- `msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd`
  - [5%]: MSE, RMSE, median-based metrics, train/validation/test splits, bagging and
    boosting ensembles
- `//notes1/notes/IN_PROGRESS.finance.Machine_learning_for_asset_managers.de_Prado.2020.md`
  - [5%]: Denoising and detoning, financial labels, feature importance
- `//notes1/notes/IN_PROGRESS.finance.Handbook_of_equity_market_anomalies.Zacks.2011.txt`
  - [5%]: Equity anomalies: accruals, analyst recommendations, post-earnings
    announcement drift, fundamental data anomalies
- `//notes1/notes/IN_PROGRESS.finance.151_Trading_strategies.Kakushadze.2018.txt`
  - [5%]: Catalog of 151 quantitative trading strategies
- `//notes1/notes/IN_PROGRESS.finance.Time_series_with_state_space_methods.Durbin.2012.txt`
  - [5%]: State space methods for time series (stub)
- `//notes1/notes/papers.finance.deep_learning.txt`
  - [5%]: Deep learning for cross-sectional return forecasting, forecasting company
    fundamentals
- `//notes1/notes/papers.finance.causal_ai.txt`
  - [5%]: Causal factor investing
- `//notes1/notes/IN_PROGRESS.finance.The_Secrets_of_Economic_Indicators.Baumohl.2012.txt`
  - [5%]: How economic indicators move markets, as inputs for macro nowcasting
- _Not covered_
  - [20%]: Time series foundation models (Chronos, TimesFM, Lag-Llama), probabilistic
    forecasts with deep models, conformal prediction on returns, signal decay of
    ML-based factors, macro nowcasting with modern data

## 08: Limit Order Book Modeling

### Goals

- Introduce market microstructure and the limit order book
- Show deep learning models on order flow for short-horizon signals
- Estimate market impact and execution cost, and the limits they put on
  high-frequency strategies

### Topics

- Microstructure and LOB Mechanics
  - Limit and market orders, queues, and matching
  - Order flow, spread, and depth
  - Deep learning on book snapshots and event sequences
  - Why AI:
    - Event streams: short-horizon signals hide in millions of order book events per
      day
      - ML: CNN, LSTM, and transformer models on book snapshots and event sequences
        predict short-horizon moves
- Market Impact and Execution Cost
  - Market impact estimation
  - Execution cost: spread, slippage, and impact
  - High-frequency signals and capacity constraints
  - Why AI:
    - Execution cost: spread, slippage, and own market impact erode returns
      - ML: impact and fill-probability models are estimated from own execution data
    - Capacity and latency: small high-frequency edges need low latency and cannot
      absorb size
      - Automation: matching-engine simulators and replay tools test models on event
        data before live use
    - Limits: gains from deep order flow models often vanish after costs and latency

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

- `//notes1/notes/finance.Security_trading.Hasbrouk.2017.txt`
  - [30%]: Limit order markets, multiple markets, auctions, dealers, dark markets,
    transaction cost analysis, statistical models of order-price dynamics, order
    splitting, Reg NMS, high frequency trading
- `//notes1/notes/papers.finance.microstructure.txt`
  - [15%]: Deep order flow imbalance for alpha from the limit order book, matching
    engine simulator, message and order book files, FIX and ITCH protocols
- `//notes1/notes/IN_PROGRESS.finance.Algorithmic_trading.Johnson.2010.txt`
  - [15%]: Execution methods, market structure and design, order types, transaction
    cost measurement and analysis, algorithmic trading and DMA
- `data605/lectures_source/Lesson10.2-Streaming_and_Real_Time_Analytics.smd`
  - [10%]: Event time vs. processing time, streaming infrastructure for low-latency
    feeds
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [10%]: CNN and RNN architectures usable on order flow sequences
- `//notes1/notes/finance.all.txt`
  - [10%]: Order books, market and limit orders, penny jumping, quotes, spread and
    depth metrics
- `//notes1/notes/papers.finance.misc.txt`
  - [10%]: High frequency trading overview, market impact models (Almgren, equity
    market impact)
- `//notes1/notes/IN_PROGRESS.papers.finance.auctions.txt`
  - [10%]: Market fragmentation and latency arbitrage
- `//notes1/notes/IN_PROGRESS.finance.Advances_in_Financial_ML.de_Prado.2018.md`
  - [5%]: Microstructural features: Kyle lambda, Amihud lambda
- `//notes1/notes/IN_PROGRESS.Algorithmic_trading_and_DMA.Johnson.2010.txt`
  - [5%]: Shorter notes on the same book
- `//notes1/notes/IN_PROGRESS.finance.Empirical_market_structure.Hasbrouck.2007.txt`
  - [5%]: Outline notes on the empirical market structure book (stub)
- _Not covered_
  - [35%]: Deep learning on order flow sequences beyond the deep order flow imbalance
    paper, queue and fill-probability models, market impact estimation on own data,
    high-frequency capacity and latency constraints

## 09: Portfolio Optimization and Reinforcement Learning

### Goals

- Solve portfolio optimization under risk, cost, and position constraints
- Apply neural networks to option pricing, volatility surfaces, yield curves, and
  hedging
- Separate RL applications that work (execution, market making, hedging) from hype
  (end-to-end trading)

### Topics

- Portfolio Optimization and Constraints
  - Mean-variance and CVaR objectives
  - Constraints: leverage, turnover, and transaction costs
  - Differentiable portfolio layers
  - Why AI:
    - Estimation error: optimizers amplify noise in expected returns and covariances
      - ML: shrinkage and robust estimators stabilize the inputs to the optimizer
    - Constraints: leverage, turnover, and costs make allocation a hard constrained
      problem
      - ML: differentiable portfolio layers optimize the final objective under
        constraints
- Derivatives and Fixed Income
  - Neural option pricing and volatility surface
  - Yield curve and rates
  - Deep hedging
  - Why AI:
    - Derivatives and rates: pricing and hedging models are slow to calibrate and
      misspecified under frictions
      - ML: neural networks fit volatility surfaces and yield curves, and deep
        hedging learns hedges under costs
- RL Where It Works
  - Optimal execution
  - Market making
  - Hedging as sequential decision making
  - Why AI:
    - Sequential decisions: execution, market making, and hedging are control
      problems under noisy simulators
      - ML: RL works where a faithful simulator exists, such as execution, market
        making, and hedging
- RL Where It Is Mostly Hype
  - End-to-end trading agents
  - Failure causes: non-stationarity, low signal-to-noise, simulator gaps
  - Why AI:
    - Limits: end-to-end RL trading agents mostly fail on low signal-to-noise and
      simulator gaps

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [30%]: MDPs, Bellman equation, value and policy iteration, model-based
    vs. model-free RL, temporal-difference and Q-learning, policy search, safe
    exploration, off-policy deconfounding
- `book_springer/lectures_source/Lesson12.1_Complex_Decisions.smd`
  - [20%]: Policy-gradient and actor-critic methods, planning with learned models,
    hierarchical decisions, offline and batch RL
- `book_springer/lectures_source/Lesson02.3_Integrating_Business_Objective_And_Real_World_Dynamics.smd`
  - [15%]: Decision-focused learning, utility functions, risk preferences, cost
    asymmetries, multi-objective Pareto trade-offs
- `//notes1/notes/finance.all.txt`
  - [15%]: Fixed income basics (bonds, interest rates, term structure lattice),
    binomial and Black-Scholes option pricing, futures hedging strategies
- `//notes1/notes/IN_PROGRESS.finance.portfolio_theory.txt`
  - [15%]: Constant expected return model, mean-variance optimization, risk
    budgeting, single-index and factor models
- `book_cs_refreshers/lectures_source/Lesson96.Refresher_stochastic_processes.smd`
  - [10%]: Martingales, Brownian motion, SDEs, Monte Carlo methods as the basis for
    option pricing and rate models
- `//notes1/causify/materials/portfolio_construction.md`
  - [10%]: Portfolio construction spec: MVO, multi-objective and factor-based
    optimization, Black-Litterman, turnover and diversification constraints, robust
    optimization
- `//notes1/notes/IN_PROGRESS.finance.Risk_and_asset_allocation.Meucci.2005.txt`
  - [10%]: Risk and asset allocation: invariants, projection to the investment
    horizon, estimators and shrinkage
- `//notes1/notes/IN_PROGRESS.finance.Active_portfolio_management.Grinold_Kahn.1999.txt`
  - [10%]: Objective of the active manager, benchmarks, mean-variance optimization,
    opportunity set
- `//notes1/notes/IN_PROGRESS.finance.Derivatives.Hull.2010.txt`
  - [10%]: Futures, options, and hedging strategies
- `book_cs_refreshers/lectures_source/Lesson97.Refresher_numerical_optimization.smd`
  - [5%]: Convex and constrained optimization, duality for portfolio problems
- `//notes1/notes/IN_PROGRESS.finance.Quantitative_equity_portfolio_management.Chincarini.2006.txt`
  - [5%]: Risk-adjusted return, performance attribution, factor-based portfolio
    management
- `//notes1/notes/IN_PROGRESS.finance.Quantitative_trading.Chan.2009.txt`
  - [5%]: Optimal capital allocation, leverage, Kelly criterion, risk management
- `//notes1/notes/IN_PROGRESS.finance.Algorithmic_trading.Johnson.2010.txt`
  - [5%]: Execution algorithms as baselines for optimal execution
- `//notes1/notes/IN_PROGRESS.book.cs.Foundations_of_Reinforcement_learning_in_finance.Rao.2022.txt`
  - [5%]: MDP framework and Markov processes (partial notes)
- _Not covered_
  - [25%]: CVaR portfolio optimization, differentiable portfolio layers, neural
    option pricing, volatility surface, deep hedging, execution and market- making
    RL, failure analysis of end-to-end trading RL

# Part IV: Research Loop

## 10: Auto DS

### Goals

- Automate model and hyperparameter search under purged cross-validation
- Generate and select labels and features automatically, with checks for leakage
- Track every trial and gate automated output with human review to limit overfitting

### Topics

- AutoML for Finance
  - Search over a predefined space of models and hyperparameters
  - Objective: purged and embargoed cross-validation score
  - AutoML tools: AutoGluon and PyCaret
- Auto Feature Engineering
  - Generated features and feature selection
  - Leakage checks on generated features
- Auto Label Generation
  - Labeling choices: horizon, threshold, triple barrier, and meta-labeling
  - Label choice as a hidden trial dimension
- Objective Choice
  - Economic objective net of costs vs. statistical loss
  - Agents may optimize the wrong metric
- Experiment Tracking
  - Trial counts as input to the deflated Sharpe ratio
  - Overfitting statistics: probability of backtest overfitting, effective trials
  - Run logs: code, data versions, and seeds
  - Search cost: compute and time per trial
- Guardrails
  - Final holdout touched once
  - Human review gates against automated overfitting

### TODO

- [ ] Add `Why AI` bullets to each topic
- [ ] Check labeling and overfitting statistics against
  `//notes1/notes/IN_PROGRESS.finance.Advances_in_Financial_ML.de_Prado.2018.md`

### Slides

- `book_Modern_AI_for_Finance/lectures_source/Lesson10.01_Auto_DS.smd`

### Lesson Materials

// TODO(gp): Check the coverage percentages: estimated from the deck's sources
- `//notes1/notes/IN_PROGRESS.finance.Advances_in_Financial_ML.de_Prado.2018.md`
  - [45%]: Grid and randomized search with cross-validation, financial observations
    are not IID, fractionally differentiated features, feature importance (MDI,
    MDA, single feature importance, substitution effects), labeling (fixed horizon,
    dynamic thresholds, triple barrier, meta-labeling), uniqueness and sequential
    bootstrap, log loss vs. accuracy, deflated Sharpe ratio, PBO, steps to reduce
    overfitting
- `book_Modern_AI_for_Finance/lectures_source/Lesson02.01_Strategy_Backtesting.smd`
  - [25%]: Purging and embargo, experiment tracking counts the trials, deflated
    Sharpe ratio, reality check and PBO
- `research/agentic_data_science/agentic_data_science.report.md`
  - [15%]: Agents as search systems, research architecture, current capabilities,
    where agents fail (leakage, metric selection, compute cost), the changing role
    of humans
- `research/ideas/draft.MDL_Extensions_with_Research_Process.md`
  - [10%]: Search cost per trial, hidden complexity of the research process, AutoML
    search trajectories
- `research/ideas/in_progress.Comparison_of_Data_Science_Agents.md`
  - [5%]: Data science agents as search systems, candidate agents
- `book_Modern_AI_for_Finance/lectures_source/Lesson01.01_Why_Finance_Is_Different.smd`
  - [5%]: Transaction costs for net objectives
- `//notes1/notes/work.trading_ideas.txt`
  - [5%]: Own notes on overfitting: the number of hypotheses tested, not their
    complexity
- `msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd`
  - [5%]: Data snooping, "burning the test set"
- `//notes1/notes/papers.finance.misc.txt`
  - [5%]: De Prado on the multiple-testing crisis in financial research
- _Not covered_
  - [5%]: Worked benchmarks of AutoGluon and PyCaret on a financial panel, and
    quantitative comparison of Bayesian and successive-halving search under purged
    cross-validation

## 11: Agents: Automating the Research Loop

### Goals

- Build agents that call tools and retrieve financial data
- Automate the research loop from hypothesis to backtest to report
- Evaluate agents and verify their outputs, guarding against automated data snooping

### Topics

- Tool-Using Agents
  - Function calling with finance tools: market data APIs, filings, backtesters
  - Retrieval pipelines for financial data
  - Architecture: planner, tool layer, sandbox, critic, and memory
  - Semantic layers and data catalogs fix metric definitions
  - Why AI:
    - Manual data access: market data, filings, and backtests are reached by hand
      - LLM: tool-using agents call market data APIs, filings, and backtesters
- Automating the Research Loop
  - Loop: hypothesis, data, code, backtest, report
  - Agents per pipeline step: cleaning, features, modeling, and backtest
  - Single-agent vs. multi-agent analyst workflows: AutoGen, CrewAI, LangGraph
  - Agent reasoning, memory, and planning
  - Strong steps: exploration, modeling, and iteration
  - Weak steps: problem framing, causal reasoning, and metric choice
  - Why AI:
    - Slow research loop: hypothesis, data, code, backtest, and report are manual
      steps
      - Automation: code generation turns hypotheses into runnable backtests and
        reports
      - LLM: multi-agent analyst workflows split data, modeling, and review roles
- Data Science Agent Landscape
  - Systems: LIDA, Data Interpreter, DS-STAR, RAGvis
  - Reusing prior solutions: skill libraries (AgentAda), case-based reasoning
    (DS-Agent)
  - Agents as search systems over analyses
- Evaluating Agents
  - Benchmarks for data science tasks: DSBench, MLE-Bench, DataSciBench
  - Benchmarks for analysis and insight: DA-Code, DABStep, InsightBench, BLADE
  - Scorecard: performance, code quality, runtime, and explainability
  - Cost: accuracy vs. tokens vs. latency
  - Programmatic metrics vs. LLM-as-judge
  - Benchmark limits: saturation and Kaggle-style tasks
- Verification
  - Fact-checking and verification of agent outputs
  - Critic loops and programmatic validators
  - Silent failures: leaked features and corrupted labels
  - Multiple testing in automated research: guarding against p-hacking
  - Division of labor: humans frame and decide, agents explore and build
  - Why AI:
    - Automated data snooping: faster search multiplies false discoveries
      - Automation: trial logging and multiple-testing corrections apply to every
        agent run
    - Unreliable outputs: agents can fabricate data or run incorrect code
      - Automation: fact-checking, tests, and human review verify agent outputs
- Security and Provenance
  - Sandboxing, least privilege, and prompt-injection defenses
  - Provenance: prompts, tool calls, and agent traces

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [30%]: Agents vs. chatbots, perceive-plan-act loop, tools and environments,
    grounding language in actions, single-agent vs. multi-agent taxonomy
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [30%]: Retrieval as a tool call, vector databases, enterprise grounding,
    fact-checking grounded answers
- `book_Agentic_AI/lectures_source/Lesson05.1-Reasoning_Memory_and_Planning.smd`
  - [25%]: Working vs. long-term memory, knowledge-graph retrieval, world models for
    planning
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [20%]: Chain-of-thought, self-consistency, search over reasoning, limits of
    self-correction
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [10%]: Chain-of-Verification to reduce hallucination
- `//notes1/notes/IN_PROGRESS.finance.IB_TWS_API.txt`
  - [10%]: TWS API setup, paper trading, historical data limits, connection settings
- `//notes1/btc_liquidation/BTCNow_liquidation.md`
  - [10%]: White paper on agentic execution for Bitcoin liquidation: data, benchmark,
    results
- `msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd`
  - [5%]: Research workflow, data snooping, multiple comparisons
- `//notes1/notes/IN_PROGRESS.finance.Machine_trading.Chan.2017.txt`
  - [5%]: Historical and live market data, backtesting and trading platforms, brokers
- `//notes1/notes/python.ib_insync.txt`
  - [5%]: `ib_insync` Python wrapper for the IB API
- `//notes1/notes/IN_PROGRESS.tools.Algorithmic_trading_with_IB.Scarpino.2020.txt`
  - [5%]: Algorithmic trading with IB in Python
- `//notes1/notes/papers.finance.misc.txt`
  - [5%]: Max Dama on automated trading
- _Not covered_
  - [25%]: Research-loop automation for finance, multi-agent analyst workflow design,
    LLM tool definitions for finance, multiple-testing controls for automated
    research

### Notes

- Sources used for the agent topics, also used for chapters 06 and 10:
  - `research/ideas/in_progress.Comparison_of_Data_Science_Agents.md`
  - `research/ideas/draft.Benchmarking_Data_Science_Agents.md`
  - `research/agentic_data_science/sota_eda.md`
  - `research/agentic_data_science/deep-research-report.md`
  - `research/agentic_data_science/agentic_data_science.report.md`

# Part V: Risk, Operations, and Governance

## 12: Risk, Credit Risk, and Fraud

### Goals

- Measure market risk with VaR, expected shortfall, covariance, and factor risk
  models
- Apply tabular deep learning to credit scoring and graph models to fraud detection
- Produce calibrated default probabilities with conformal guarantees

### Topics

- Market Risk
  - Value at Risk (VaR) and Expected Shortfall (ES)
  - Covariance estimation and shrinkage
  - Factor risk models
  - Why AI:
    - Market risk: VaR, ES, and covariance estimates are noisy in high dimensions and
      fail in crises
      - ML: shrinkage and factor risk models stabilize covariance estimates
- Credit Risk and Fraud Detection
  - Tabular deep learning vs. gradient-boosted trees for credit scoring
  - Graph neural networks for transaction networks
  - Fraud patterns and anomaly detection
  - Real-time scoring and regulatory constraints
  - Why AI:
    - Credit scoring: tabular data is imbalanced and decisions are regulated
      - ML: GBDT and tabular deep learning score imbalanced tabular data
    - Fraud: events are rare, adversaries adapt, and patterns span many accounts
      - ML: graph neural networks and anomaly detection find patterns across
        transaction networks
      - Automation: streaming pipelines score transactions in real time
- Uncertainty in Credit
  - Calibrated default probabilities
  - Conformal prediction for credit decisions
  - Why AI:
    - Calibration: regulated decisions need valid default probabilities
      - ML: calibration and conformal prediction give valid default probabilities

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

- `msml610/lectures_source/Lesson04.3-Models.smd`
  - [15%]: Anomaly detection with Gaussian models, evaluation of anomaly detectors,
    hacked-account example, k-means clustering
- `//notes1/notes/IN_PROGRESS.finance.Risk_and_asset_allocation.Meucci.2005.txt`
  - [15%]: Distributions and copulas, covariance estimation, shrinkage estimators,
    expected shortfall, factor models
- `msml610/lectures_source/Lesson04.1-Models.smd`
  - [10%]: Decision trees, random forests, feature selection with trees for tabular
    data
- `msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd`
  - [10%]: Confusion matrix, precision and recall, skewed classes, precision-recall
    curves
- `//notes1/notes/papers.finance.risk_models.txt`
  - [10%]: Axioma robust risk model handbook: multi-factor risk models, statistical
    approaches
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [5%]: Calibration of neural networks, conformal prediction for deep models
- `book_springer/lectures_source/Lesson02.2_Integrating_Causality_And_Probability_in_ML.smd`
  - [5%]: Calibration and conformal prediction, decision readiness under uncertainty
- `book_cs_refreshers/lectures_source/Lesson91.Refresher_probability.smd`
  - [5%]: Variance, covariance, and quantiles as the basis for VaR and ES
- `book_springer/lectures_source/Lesson15.1_Deployment_Monitoring_And_Adaptation.smd`
  - [5%]: High-asymmetry regulated decisions (credit denial), feedback loops from
    fraud flags
- `data605/lectures_source/Lesson12.3-Graph_Data_Processing.smd`
  - [5%]: Graph analysis tasks, subgraph matching, Pregel-style graph processing
- `data605/lectures_source/Lesson10.2-Streaming_and_Real_Time_Analytics.smd`
  - [5%]: Streaming infrastructure for real-time scoring
- `//notes1/notes/IN_PROGRESS.finance.Active_portfolio_management.Grinold_Kahn.1999.txt`
  - [5%]: Elementary risk models, choosing factors, uses of a risk model
- `//notes1/notes/IN_PROGRESS.finance.Quantitative_equity_portfolio_management.Chincarini.2006.txt`
  - [5%]: Risk and risk-adjusted return
- `//notes1/notes/IN_PROGRESS.finance.portfolio_theory.txt`
  - [5%]: Portfolio risk budgeting
- `//notes1/causify/proposals/sandbox_portfolio_risk.proposal.md`
  - [5%]: Causify proposal: causal risk models vs. correlation-based VaR and factor
    covariance
- `//notes1/notes/IN_PROGRESS.course.finance.Fintech_Wharton.txt`
  - [5%]: Marketplace lending
- _Not covered_
  - [35%]: VaR estimation and backtesting, tabular deep learning, graph neural
    networks, fraud typologies, fair-lending regulation, conformal credit scoring

## 13: Explainability and Robustness

### Goals

- Explain model predictions with Shapley values and attention
- Measure robustness to adversarial inputs and hallucination risk
- Build trust by explaining financial decisions and communicating uncertainty with
  calibration and conformal prediction

### Topics

- Attribution Methods
  - SHAP and Shapley values
  - Attention-based interpretability
  - Feature importance in finance
  - Why AI:
    - Black boxes: regulators and stakeholders need reasons for each decision
      - ML: SHAP, attention analysis, and feature importance attribute predictions to
        inputs
- Robustness
  - Adversarial robustness
  - Hallucination risk in LLM outputs
  - Why AI:
    - Fragility: small input changes or adversarial inputs flip model outputs
      - Automation: perturbation and adversarial test suites measure robustness
    - Hallucination: LLMs state false facts with confidence
      - LLM: retrieval and chain-of-verification reduce unsupported claims
- Decision Explanations
  - Explaining financial decisions
- Uncertainty and Trust
  - Calibration of model outputs
  - Conformal prediction: distribution-free prediction sets
  - Why AI:
    - Overconfidence: users cannot tell when a model should not be trusted
      - ML: calibration and conformal prediction attach valid uncertainty to outputs

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

- `msml610/lectures_source/Lesson13.1-Explainability.smd`
  - [55%]: SHAP, LIME, permutation feature importance, counterfactual explanations,
    uncertainty quantification as a form of explanation, faithfulness and stability
    of explanations, accuracy vs. interpretability trade-off
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [20%]: Saliency maps, SHAP, attention for interpretability, layer-wise relevance
    propagation, calibration of neural networks, conformal prediction for deep models
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [15%]: Chain-of-Verification to reduce hallucination
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [15%]: Causal explanations, robustness through causal constraints, adversarial
    robustness with causal models
- `//notes1/notes/IN_PROGRESS.finance.Advances_in_Financial_ML.de_Prado.2018.md`
  - [10%]: Feature importance for financial ML, substitution effects
- `book_springer/lectures_source/Lesson02.2_Integrating_Causality_And_Probability_in_ML.smd`
  - [5%]: Calibration and conformal prediction, decision readiness under uncertainty
- `//notes1/notes/IN_PROGRESS.finance.Machine_learning_for_asset_managers.de_Prado.2020.md`
  - [5%]: Feature importance analysis
- _Not covered_
  - [10%]: Adversarial attacks on market models

## 14: Regulation, Model Risk Management, and Deployment

### Goals

- Map AI systems to compliance frameworks such as SR 11-7 and the EU AI Act
- Validate and govern models across their lifecycle
- Operate models in production: latency, monitoring, and rollback

### Topics

- Compliance Frameworks
  - Compliance frameworks (SR 11-7, EU AI Act)
  - Why AI:
    - Compliance: SR 11-7 and the EU AI Act require documented controls
      - LLM: assistants map controls to regulatory text and draft documentation for
        human review
- Model Governance
  - Model validation and governance
  - Why AI:
    - Validation and governance: models need documented validation and audit trails
      - Automation: pipelines generate model documentation, validation reports, and
        audit trails from code and experiment logs
- Production Operations
  - Latency and production constraints
  - Monitoring and drift detection
  - Incident response and rollback
  - Why AI:
    - Model drift: deployed models decay as markets change
      - ML: drift detectors monitor inputs and performance and trigger retraining or
        rollback
    - Operations: latency limits, incidents, and rollbacks need engineering
      discipline
      - Automation: CI/CD, versioning, and canary deployments make releases and
        rollbacks routine

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

- `book_springer/lectures_source/Lesson15.1_Deployment_Monitoring_And_Adaptation.smd`
  - [55%]: Notebook-to-production, cost-aware deployment, monitoring causal
    assumptions, experimentation in production, versioning, error budgets, rollback,
    feedback loops, technical debt
- `book_springer/lectures_source/Lesson08.1_Causal_Data_Pipelines.smd`
  - [10%]: Detecting distribution shift and concept drift in production
- `data605/lectures_source/Lesson07.4-Big_Data_Architectures.smd`
  - [10%]: CI/CD, stages of deployment, semantic versioning, microservices
    vs. monolith
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [10%]: Causal monitoring and adaptation, safety constraints
- `//notes1/notes/IN_PROGRESS.course.finance.Fintech_Wharton.txt`
  - [10%]: Regulation: regulatory capture, regulation-innovation trade-off, post-2008
    regulation, shadow banking, FinTech regulation
- `//notes1/notes/finance.Security_trading.Hasbrouk.2017.txt`
  - [5%]: Reg NMS, securities class action law suits, insider trading law
- `//notes1/notes/IN_PROGRESS.finance.Quantitative_trading.Chan.2009.txt`
  - [5%]: Physical infrastructure, automated trading systems, paper trading, risk
    management
- `//notes1/notes/IN_PROGRESS.finance.IB_TWS_API.txt`
  - [5%]: Operating a broker API: paper trading, daily restarts, read-only API
- _Not covered_
  - [30%]: Compliance frameworks (SR 11-7, EU AI Act), model validation and
    governance process, latency budgets for financial systems

# Part VI: Frontiers

## 15: Market Simulation

### Goals

- Generate market scenarios with GANs, diffusion models, and agent-based simulators
- Use simulated data for stress testing and for augmenting scarce data
- Evaluate fidelity against stylized facts such as fat tails and volatility
  clustering

### Topics

- Generative Models for Scenario Generation
  - Generative Adversarial Networks (GANs) for time series
  - Diffusion models for time series
  - Why AI:
    - Scarce history: markets give one realized path and few rare events
      - ML: GANs and diffusion models generate synthetic paths that augment data
- Agent-Based Market Simulation
  - Heterogeneous trader agents and market mechanisms
  - Calibrating simulators to real markets
  - Why AI:
    - Stress testing: risk managers need plausible scenarios beyond history
      - ML: agent-based simulators calibrated to real markets produce scenarios with
        feedback effects
- Applications
  - Stress testing and scenario analysis
  - Synthetic data: augmenting scarce data
- Fidelity
  - Stylized facts: fat tails, volatility clustering, autocorrelation of returns
  - Risks of training and testing on synthetic data
  - Why AI:
    - Fidelity: synthetic data must reproduce stylized facts
      - Automation: test suites check fat tails, volatility clustering, and return
        autocorrelation
    - Limits: training and testing on the same synthetic generator gives circular
      evidence

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [15%]: Survey of VAEs, GANs, normalizing flows, diffusion models, score-based
    generative modeling
- `book_cs_refreshers/lectures_source/Lesson96.Refresher_stochastic_processes.smd`
  - [10%]: Brownian motion, random walks, SDEs, and Monte Carlo methods for
    simulating price paths
- `//notes1/notes/IN_PROGRESS.finance.Advances_in_Financial_ML.de_Prado.2018.md`
  - [10%]: Backtesting on synthetic data
- `book_cs_refreshers/lectures_source/Lesson95.Refresher_game_theory.smd`
  - [5%]: Equilibria, learning in games, multi-agent RL as the basis for agent-based
    simulation
- `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.smd`
  - [5%]: ARCH/GARCH volatility clustering as a fidelity benchmark
- `//notes1/notes/IN_PROGRESS.finance.Machine_learning_for_asset_managers.de_Prado.2020.md`
  - [5%]: Testing on synthetic data
- `//notes1/notes/papers.finance.microstructure.txt`
  - [5%]: JSE matching engine simulator
- `//notes1/notes/IN_PROGRESS.finance.Risk_and_asset_allocation.Meucci.2005.txt`
  - [5%]: Projection of market invariants to the investment horizon for scenario
    generation
- _Not covered_
  - [55%]: GAN and diffusion architectures for time series, agent-based market
    simulators, stress testing, fidelity tests based on stylized facts

## 16: The Road Ahead

### Goals

- Show how multimodal models fuse text, tabular, and time series data
- Survey emerging architectures and paradigms
- Frame open research problems and the future of AI in finance

### Topics

- Multimodal Finance
  - Multimodal models combining modalities
  - Text, tabular, and time-series fusion
  - Why AI:
    - Siloed modalities: text, tabular, and time series are modeled separately
      - ML: multimodal models fuse text, tabular, and time series data
- Emerging Directions
  - Emerging architectures and paradigms
  - Why AI:
    - Open problems: non-stationarity, scarce data, and crowding remain unsolved
      - ML: self-supervised pretraining and foundation models reduce the need for
        labels
    - Fast-moving tools: models and tooling change faster than validation practice
      - Automation: reproducible benchmarks and evaluation harnesses re-test new
        models quickly
- Outlook
  - Open research problems
  - Future of AI in finance
  - Why AI:
    - Limits: none of these methods removes non-stationarity or crowding

### Slides

- N/A: no dedicated deck yet

### Lesson Materials

- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [20%]: Multimodal deep learning, self-supervised learning, neural ODEs, modern
    research frontiers
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [15%]: Emerging training patterns, open problems, recommendations
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [10%]: Path forward for trustworthy reasoning agents
- _Not covered_
  - [65%]: Multimodal fusion of text, tabular, and time series data, open research
    problems specific to finance
