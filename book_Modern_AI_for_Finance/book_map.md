# Summary

## Title
- Modern AI for Finance

## Target Audience
- Quantitative researchers, ML engineers, and graduate students who apply deep
  learning to financial problems
- Assumes working knowledge of Python, probability, linear algebra, and
  supervised machine learning
- Assumes no deep finance background: market structure and data are introduced
  where needed

## Approach of the Book
- Focus on:
  - Finance-specific failure modes first: non-stationarity, noise, overfitting
  - Modern architectures (transformers, LLMs) applied to concrete financial tasks
  - Intuition over math, with derivations only when needed
  - Making theory operational through packages and notebooks
- Provide resources to go one level deeper:
  - Related classes (`msml610`, `data605`)
  - Related books (`book_Agentic_AI`, `book_springer`)
  - Papers and books on quantitative finance and deep learning

## Short TOC
- Foundations
  - 01, Why Finance Is Different
    - Finance primer: instruments, returns, Sharpe ratio, costs, capacity
    - Non-stationarity, fat tails, low signal-to-noise
    - Backtest discipline: overfitting, multiple testing, purged CV, deflated Sharpe
    - Causal vs. predictive: spurious factors
  - 02, Deep Learning, Transformers, and LLMs
    - Refresher: deep learning, transformers, LLMs
    - Adapting LLMs: fine-tuning, RAG, domain-pretrained models
  - 03, Data engineering
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
- Market Prediction with Deep Learning
  - 04, Price and Volatility Forecasting
    - Baselines first: linear, ARIMA/GARCH, GBDT
    - Time series foundation models (Chronos, TimesFM, Lag-Lama)
    - Alpha research: signals, IC, turnover, decay
    - Factors: classic, ML-based, factor zoo, causal factors
    - Regime detection and macro nowcasting
    - Uncertainty quantification: probabilistic forecasts, calibration,
      conformal prediction
  - 05, Limit Order Book Modeling
    - Microstructure and LOB mechanics
    - Market impact and execution cost
  - 06, Portfolio Optimization and Reinforcement Learning
    - Portfolio optimization and constraints
    - Derivatives and fixed income
      - Neural option pricing and volatility surface
      - Yield curve and rates
      - Hedging (deep hedging)
    - RL where it works: execution, market making, hedging
    - RL where it is mostly hype: end-to-end trading
- Language in Finance
  - 07, News, Filings, Earnings Calls
    - LLMs for finance text
  - 08, Agents: automating the research loop
  - 09, Conversational finance assistants
- Risk, Operations, and Governance
  - 10, Risk, Credit Risk, and Fraud
    - Market risk: VaR/ES, covariance, factor risk models
    - Credit risk and fraud detection
    - Uncertainty in credit: calibrated default probabilities, conformal
      prediction
  - 11, Explainability and Robustness
    - Uncertainty and trust: calibration, conformal prediction
  - 12, Regulation, Model Risk Management, and Deployment
- Frontiers
  - 13, Market simulation
    - GANs / diffusion models for scenario generation
    - Agent-based market simulation
    - Stress testing
    - Synthetic data: augmenting scarce data
    - Fidelity: stylized facts (fat tails, volatility clustering)
  - 14, The Road Ahead: multimodal models, open problems

## Resources
- /Users/saggese/src/notes1/finance_index.md

## All Lesson Materials
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

| Chap                                                    | Slides | Slides % | Criticize | Tutorial | Book |
| :------------------------------------------------------ | :----- | :------- | :-------- | :------- | :--- |
|                                                         |        |          |           |          |      |
| **Foundations**                                         |        |          |           |          |      |
| 01. Why Finance Is Different                            | N/A    |          |           |          |      |
| 02. Deep Learning, Transformers, and LLMs               | N/A    |          |           |          |      |
| 03. Data Engineering                                    | N/A    |          |           |          |      |
| **Market Prediction with Deep Learning**                |        |          |           |          |      |
| 04. Price and Volatility Forecasting                    | N/A    |          |           |          |      |
| 05. Limit Order Book Modeling                           | N/A    |          |           |          |      |
| 06. Portfolio Optimization and Reinforcement Learning   | N/A    |          |           |          |      |
| **Language in Finance**                                 |        |          |           |          |      |
| 07. News, Filings, Earnings Calls                       | N/A    |          |           |          |      |
| 08. Agents: Automating the Research Loop                | N/A    |          |           |          |      |
| 09. Conversational Finance Assistants                   | N/A    |          |           |          |      |
| **Risk, Operations, and Governance**                    |        |          |           |          |      |
| 10. Risk, Credit Risk, and Fraud                        | N/A    |          |           |          |      |
| 11. Explainability and Robustness                       | N/A    |          |           |          |      |
| 12. Regulation, Model Risk Management, and Deployment   | N/A    |          |           |          |      |
| **Frontiers**                                           |        |          |           |          |      |
| 13. Market Simulation                                   | N/A    |          |           |          |      |
| 14. The Road Ahead                                      | N/A    |          |           |          |      |

## `book_Modern_AI_for_Finance` Tutorials

> find book_Modern_AI_for_Finance/tutorials -name *.ipynb
```
```

## TODOs
- Write one slide deck per chapter in `book_Modern_AI_for_Finance/lectures_source/`
  (30-35 slides each)
- Write the finance-specific content from scratch for chapters where the
  `_Not covered_` share exceeds 50%: 03, 05, 06, 07, 10, 13, 14
- Add tutorial notebooks under `book_Modern_AI_for_Finance/tutorials/`
- Collect related books, packages, and papers for each chapter: the previous map
  had empty placeholders for them
- Generate `book_Modern_AI_for_Finance/all_tocs.md` once the slide decks exist

# Detailed TOC

# Part I: Foundations

## 01: Why Finance Is Different

### Goals
- Introduce the finance basics the book needs: instruments, returns, Sharpe
  ratio, costs, and capacity
- Explain why non-stationarity, fat tails, and low signal-to-noise break the
  assumptions of standard ML
- Teach backtest discipline and causal thinking to avoid false discoveries

### Topics
- Finance Primer
  - Instruments: equities, futures, options, bonds
  - Returns: simple vs. log returns, excess returns, compounding
  - Risk-adjusted performance: volatility, Sharpe ratio, drawdown
  - Transaction costs and strategy capacity
- Statistical Properties of Financial Data
  - Non-stationarity and regime shifts
  - Fat tails and extreme events
  - Low signal-to-noise ratios and small effective sample sizes
- Backtest Discipline
  - Backtest overfitting and data snooping
  - Multiple testing and false discoveries
  - Purged and embargoed cross-validation
  - Deflated Sharpe ratio
- Causal vs. Predictive Models
  - Spurious factors: correlation without causation
  - Markets that adapt to published signals

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.smd`
  - [25%]: Strict and weak stationarity, ACF, AR(1) for financial series,
    ARCH/GARCH volatility, change-point detection, Markov-switching models,
    cross-validation for time series
- `msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd`
  - [20%]: Data snooping, "burning the test set", sampling bias, out-of-sample
    discipline, multiple comparisons
- `msml610/lectures_source/Lesson05.2-Overfitting.smd`
  - [15%]: Overfitting, bias-variance trade-off, deterministic vs. stochastic
    noise, data size vs. noise
- `book_cs_refreshers/lectures_source/Lesson91.Refresher_probability.smd`
  - [15%]: Kurtosis and excess kurtosis as the definition of fat tails,
    multiple hypothesis testing, p-hacking, FWER and Bonferroni, FDR and
    Benjamini-Hochberg
- `msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd`
  - [10%]: Association vs. correlation vs. causation, what ML systems can and
    cannot tell you
- `book_springer/lectures_source/Lesson02.1_From_Data_Science_To_Decision_Science.smd`
  - [10%]: Causal vs. predictive questions, correlation that encodes
    confounding, selection bias, Simpson's paradox, collider bias
- `book_springer/lectures_source/Lesson08.1_Causal_Data_Pipelines.smd`
  - [10%]: Distribution shift, concept drift, non-stationarity, detecting shift
    in production
- _Not covered_
  - [40%]: Finance primer (instruments, returns, Sharpe ratio, costs, capacity),
    extreme-value and tail-risk modeling, purged and embargoed cross-validation,
    deflated Sharpe ratio, spurious factors in finance, adaptive markets

## 02: Deep Learning, Transformers, and LLMs

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
- Adapting LLMs to Finance
  - Fine-tuning and instruction tuning for financial tasks
  - RAG over filings, news, and research, and RAG vs. fine-tuning trade-offs
  - Domain-pretrained models such as BloombergGPT

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [35%]: Perceptron and MLP, activations, backpropagation, automatic
    differentiation, CNNs and ResNets, RNNs, vanishing gradients, LSTM/GRU,
    weight initialization, batch normalization, learning-rate schedules, early
    stopping, regularization, data augmentation, RNN limits, sequence-to-sequence
    attention, types of attention, transformer architecture, pretraining
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [30%]: Query-key-value attention, scaled dot-product, multi-head, self vs.
    cross vs. causal attention, positional encoding, transformer block, residual
    connections, LayerNorm, attention cost, masked vs. autoregressive objectives,
    pretraining pipeline, scaling laws, emergent abilities, decoding, context
    windows, expressivity limits
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [10%]: RAG loop, retrieval as a tool call, vector databases, approximate
    nearest-neighbor indexes, hybrid search, grounding in regulated domains,
    long context vs. retrieval, needle-in-a-haystack evaluation of long context
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [10%]: Chain-of-thought and variants, premise-order brittleness, limits of
    self-correction
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [10%]: Data curation, multi-stage training pipelines, case studies of open
    frontier models, training cost
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
  - [5%]: Open post-training pipeline, DPO vs. PPO, preference feedback
    collection, RAG plus reasoning training for scientific synthesis
- _Not covered_
  - [20%]: Regularization tuned to financial noise levels, time-aware validation
    for neural networks, time series transformers and multi-asset attention,
    survey of general-purpose benchmark suites, composition of pretraining
    datasets, parameter-efficient fine-tuning, supervised instruction tuning on
    financial tasks, domain-pretrained finance models

## 03: Data Engineering

### Goals
- Build point-in-time correct datasets free of look-ahead and survivorship bias
- Handle restatements, corporate actions, identifiers, and timestamps correctly
- Survey market, fundamental, filing, and alternative data vendors and their
  pitfalls

### Topics
- Point-in-Time Correctness
  - Look-ahead bias and as-of joins
  - Restatements and data revisions
- Survivorship Bias
  - Survivorship-bias-free universes
  - Delisting returns
- Reference Data
  - Corporate actions: splits, dividends, and adjusted prices
  - Identifiers and symbology: CUSIP, ISIN, PERMNO
  - Timestamps, time zones, and trading calendars
- Data Vendors
  - Market data: CRSP and TAQ
  - Fundamental data: Compustat
  - Filings: EDGAR
- Alternative Data
  - Classification of alternative data and practical pitfalls
  - Satellite and web data
  - ESG and climate risk data

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `data605/lectures_source/Lesson02.3-Data_Pipelines.smd`
  - [15%]: Data ingestion, ETL/ELT paradigms, workflow orchestration, data lake vs.
    warehouse, data cleaning
- `data605/lectures_source/Lesson07.2-Data_Wrangling.smd`
  - [15%]: Tidy data, single-source and multi-source data problems, outlier
    detection including time series outliers
- `book_springer/lectures_source/Lesson08.1_Causal_Data_Pipelines.smd`
  - [15%]: Selection bias as the root of survivorship bias, missing-data
    mechanisms, distribution shift, measurement error, pre-flight data quality
    checks
- `data605/lectures_source/Lesson10.2-Streaming_and_Real_Time_Analytics.smd`
  - [10%]: Event time vs. processing time, delivery semantics, Kafka, Flink,
    Spark Structured Streaming
- `data605/lectures_source/Lesson07.3-Serialization_Formats.smd`
  - [5%]: CSV, Parquet, JSON, and Protocol Buffers for storing market data
- _Not covered_
  - [65%]: Point-in-time (as-of) joins, restatements and revisions,
    survivorship-bias-free universes and delisting returns, corporate actions,
    identifiers and symbology, trading calendars, vendor datasets (CRSP,
    Compustat, TAQ, EDGAR), alternative data (satellite, web, ESG, climate)

# Part II: Market Prediction with Deep Learning

## 04: Price and Volatility Forecasting

### Goals
- Establish strong baselines before reaching for deep models
- Apply time series foundation models and factor models to returns and
  volatility
- Evaluate signals by trading value, detect regimes, and quantify forecast
  uncertainty

### Topics
- Baselines First
  - Linear models and ARIMA for returns
  - GARCH for volatility clustering
  - Gradient-boosted decision trees (GBDT) on tabular features
- Time Series Foundation Models
  - Chronos, TimesFM, and Lag-Llama
  - Zero-shot vs. fine-tuned forecasting against baselines
- Alpha Research
  - Signals and the information coefficient (IC)
  - Turnover and signal decay
- Factors
  - Classic factors: market, size, value, momentum
  - ML-based factors
  - The factor zoo and false discoveries
  - Causal factors
- Regimes and Macro
  - Regime detection
  - Macro nowcasting
- Uncertainty Quantification
  - Probabilistic forecasts
  - Calibration
  - Conformal prediction

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.smd`
  - [35%]: Stationarity, ACF, decomposition, AR/MA/ARIMA, ARCH/GARCH, VAR, state
    space models, ML and deep learning for time series, cross-validation for time
    series, probabilistic forecasting, change-point detection, Markov-switching
    models
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [10%]: Aleatoric vs. epistemic uncertainty, calibration of neural networks,
    conformal prediction for deep models, transformer architecture
- `msml610/lectures_source/Lesson09.5-Kalman_Filter.smd`
  - [5%]: Filtering and prediction in state space models as the basis for
    nowcasting
- `msml610/lectures_source/Lesson09.2-Hidden_Markov_Models.smd`
  - [5%]: HMMs and EM as regime-switching models
- `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd`
  - [5%]: Granger causality, time-varying confounders, feedback loops as tools
    for causal factors
- `book_springer/lectures_source/Lesson02.2_Integrating_Causality_And_Probability_in_ML.smd`
  - [5%]: Posteriors vs. point estimates, calibration and conformal prediction
- `msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd`
  - [5%]: MSE, RMSE, median-based metrics, train/validation/test splits,
    bagging and boosting ensembles
- _Not covered_
  - [50%]: GBDT for returns, time series foundation models (Chronos, TimesFM,
    Lag-Llama), alpha research metrics (IC, turnover, decay), classic and
    ML-based factors, factor zoo, macro nowcasting

## 05: Limit Order Book Modeling

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
- Market Impact and Execution Cost
  - Market impact estimation
  - Execution cost: spread, slippage, and impact
  - High-frequency signals and capacity constraints

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `data605/lectures_source/Lesson10.2-Streaming_and_Real_Time_Analytics.smd`
  - [10%]: Event time vs. processing time, streaming infrastructure for
    low-latency feeds
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [10%]: CNN and RNN architectures usable on order flow sequences
- _Not covered_
  - [85%]: Market microstructure, order book dynamics, market impact, high-frequency
    signals, latency and capacity constraints

## 06: Portfolio Optimization and Reinforcement Learning

### Goals
- Solve portfolio optimization under risk, cost, and position constraints
- Apply neural networks to option pricing, volatility surfaces, yield curves,
  and hedging
- Separate RL applications that work (execution, market making, hedging) from
  hype (end-to-end trading)

### Topics
- Portfolio Optimization and Constraints
  - Mean-variance and CVaR objectives
  - Constraints: leverage, turnover, and transaction costs
  - Differentiable portfolio layers
- Derivatives and Fixed Income
  - Neural option pricing and volatility surface
  - Yield curve and rates
  - Deep hedging
- RL Where It Works
  - Optimal execution
  - Market making
  - Hedging as sequential decision making
- RL Where It Is Mostly Hype
  - End-to-end trading agents
  - Failure causes: non-stationarity, low signal-to-noise, simulator gaps

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [30%]: MDPs, Bellman equation, value and policy iteration, model-based vs.
    model-free RL, temporal-difference and Q-learning, policy search, safe
    exploration, off-policy deconfounding
- `book_springer/lectures_source/Lesson12.1_Complex_Decisions.smd`
  - [20%]: Policy-gradient and actor-critic methods, planning with learned
    models, hierarchical decisions, offline and batch RL
- `book_springer/lectures_source/Lesson02.3_Integrating_Business_Objective_And_Real_World_Dynamics.smd`
  - [15%]: Decision-focused learning, utility functions, risk preferences, cost
    asymmetries, multi-objective Pareto trade-offs
- `book_cs_refreshers/lectures_source/Lesson96.Refresher_stochastic_processes.smd`
  - [10%]: Martingales, Brownian motion, SDEs, Monte Carlo methods as the basis
    for option pricing and rate models
- `book_cs_refreshers/lectures_source/Lesson97.Refresher_numerical_optimization.smd`
  - [5%]: Convex and constrained optimization, duality for portfolio problems
- _Not covered_
  - [55%]: Mean-variance and CVaR portfolio optimization, differentiable
    portfolio layers, neural option pricing, volatility surface, yield curve
    models, deep hedging, execution and market-making RL, failure analysis of
    end-to-end trading RL

# Part III: Language in Finance

## 07: News, Filings, Earnings Calls

### Goals
- Extract signals from news, filings, and earnings-call transcripts
- Apply LLMs to sentiment, entity, and event extraction on financial text
- Link text events to market data over time without look-ahead bias

### Topics
- Financial Text Sources
  - News feeds, SEC filings (10-K, 10-Q, 8-K), earnings-call transcripts
  - Parsing filings and aligning text with timestamps
- LLMs for Finance Text
  - Sentiment: lexicon-based vs. LLM-based
  - Named entity and event extraction
  - Financial text classification and summarization
- From Text to Signals
  - Event studies: linking text events to returns
  - Look-ahead bias from LLM pretraining data

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [30%]: NLP tasks, word embeddings, language models, sequence-to-sequence
    models, transformers, masked language models, pretraining
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [15%]: Tokenization, masked vs. autoregressive pretraining objectives
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [15%]: Embeddings, hybrid search, grounding on regulated-domain documents
- `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd`
  - [10%]: Interrupted time series, difference-in-differences, synthetic control
    as event-study designs
- _Not covered_
  - [60%]: Financial text sources, LLM-based financial sentiment, named entity
    and event extraction, SEC filing parsing, financial text classification,
    look-ahead bias from LLM pretraining data

## 08: Agents: Automating the Research Loop

### Goals
- Build agents that call tools and retrieve financial data
- Automate the research loop from hypothesis to backtest to report
- Verify agent outputs and guard against automated data snooping

### Topics
- Tool-Using Agents
  - Function calling with finance tools: market data APIs, filings, backtesters
  - Retrieval pipelines for financial data
- Automating the Research Loop
  - Loop: hypothesis, data, code, backtest, report
  - Multi-agent analyst workflows
  - Agent reasoning, memory, and planning
- Verification
  - Fact-checking and verification of agent outputs
  - Multiple testing in automated research: guarding against p-hacking

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
  - [25%]: Working vs. long-term memory, knowledge-graph retrieval, world models
    for planning
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [20%]: Chain-of-thought, self-consistency, search over reasoning, limits of
    self-correction
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [10%]: Chain-of-Verification to reduce hallucination
- `msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd`
  - [5%]: Research workflow, data snooping, multiple comparisons
- _Not covered_
  - [40%]: Research-loop automation for finance, multi-agent analyst workflow
    design, finance-specific tools (market data APIs, filings, backtesters),
    multiple-testing controls for automated research

## 09: Conversational Finance Assistants

### Goals
- Design advisory chatbots and natural-language portfolio queries
- Add guardrails that enforce factual accuracy
- Handle ambiguity, uncertainty, and user trust in conversations

### Topics
- Assistant Use Cases
  - Advisory chatbots
  - Natural-language portfolio querying
- Reliability
  - Guardrails for factual accuracy
  - Handling ambiguity and uncertainty
- Human Factors
  - User experience and trust

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [25%]: Grounding on enterprise knowledge, high-fidelity grounding for
    regulated domains, fact-checking, retrieving vs. stuffing context
- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [20%]: Agents vs. chatbots, spectrum of autonomy, human-in-the-loop vs. full
    autonomy
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [15%]: Chain-of-Verification to reduce hallucination
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [15%]: Transparency, causal explanations, safety and trustworthy autonomy
- `msml610/lectures_source/Lesson11.1-Decision_Making_with_Causal_Models.smd`
  - [10%]: Aleatoric vs. epistemic uncertainty, communicating uncertainty to
    stakeholders
- _Not covered_
  - [50%]: Advisory chatbot design, natural-language portfolio querying, user
    experience and trust measurement

# Part IV: Risk, Operations, and Governance

## 10: Risk, Credit Risk, and Fraud

### Goals
- Measure market risk with VaR, expected shortfall, covariance, and factor risk
  models
- Apply tabular deep learning to credit scoring and graph models to fraud
  detection
- Produce calibrated default probabilities with conformal guarantees

### Topics
- Market Risk
  - Value at Risk (VaR) and Expected Shortfall (ES)
  - Covariance estimation and shrinkage
  - Factor risk models
- Credit Risk and Fraud Detection
  - Tabular deep learning vs. gradient-boosted trees for credit scoring
  - Graph neural networks for transaction networks
  - Fraud patterns and anomaly detection
  - Real-time scoring and regulatory constraints
- Uncertainty in Credit
  - Calibrated default probabilities
  - Conformal prediction for credit decisions

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson04.3-Models.smd`
  - [15%]: Anomaly detection with Gaussian models, evaluation of anomaly
    detectors, hacked-account example, k-means clustering
- `msml610/lectures_source/Lesson04.1-Models.smd`
  - [10%]: Decision trees, random forests, feature selection with trees for
    tabular data
- `msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd`
  - [10%]: Confusion matrix, precision and recall, skewed classes,
    precision-recall curves
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [5%]: Calibration of neural networks, conformal prediction for deep models
- `book_springer/lectures_source/Lesson02.2_Integrating_Causality_And_Probability_in_ML.smd`
  - [5%]: Calibration and conformal prediction, decision readiness under
    uncertainty
- `book_cs_refreshers/lectures_source/Lesson91.Refresher_probability.smd`
  - [5%]: Variance, covariance, and quantiles as the basis for VaR and ES
- `book_springer/lectures_source/Lesson15.1_Deployment_Monitoring_And_Adaptation.smd`
  - [5%]: High-asymmetry regulated decisions (credit denial), feedback loops
    from fraud flags
- `data605/lectures_source/Lesson12.3-Graph_Data_Processing.smd`
  - [5%]: Graph analysis tasks, subgraph matching, Pregel-style graph processing
- `data605/lectures_source/Lesson10.2-Streaming_and_Real_Time_Analytics.smd`
  - [5%]: Streaming infrastructure for real-time scoring
- _Not covered_
  - [60%]: VaR and ES estimation and backtesting, covariance shrinkage, factor
    risk models, tabular deep learning, graph neural networks, fraud
    typologies, fair-lending regulation, conformal credit scoring

## 11: Explainability and Robustness

### Goals
- Explain model predictions with Shapley values and attention
- Measure robustness to adversarial inputs and hallucination risk
- Build trust by explaining financial decisions and communicating uncertainty
  with calibration and conformal prediction

### Topics
- Attribution Methods
  - SHAP and Shapley values
  - Attention-based interpretability
  - Feature importance in finance
- Robustness
  - Adversarial robustness
  - Hallucination risk in LLM outputs
- Decision Explanations
  - Explaining financial decisions
- Uncertainty and Trust
  - Calibration of model outputs
  - Conformal prediction: distribution-free prediction sets

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson13.1-Explainability.smd`
  - [55%]: SHAP, LIME, permutation feature importance, counterfactual
    explanations, uncertainty quantification as a form of explanation,
    faithfulness and stability of explanations, accuracy vs. interpretability
    trade-off
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [20%]: Saliency maps, SHAP, attention for interpretability, layer-wise
    relevance propagation, calibration of neural networks, conformal prediction
    for deep models
- `book_springer/lectures_source/Lesson02.2_Integrating_Causality_And_Probability_in_ML.smd`
  - [5%]: Calibration and conformal prediction, decision readiness under
    uncertainty
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [15%]: Chain-of-Verification to reduce hallucination
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [15%]: Causal explanations, robustness through causal constraints,
    adversarial robustness with causal models
- _Not covered_
  - [15%]: Finance-specific feature importance, adversarial attacks on market
    models

## 12: Regulation, Model Risk Management, and Deployment

### Goals
- Map AI systems to compliance frameworks such as SR 11-7 and the EU AI Act
- Validate and govern models across their lifecycle
- Operate models in production: latency, monitoring, and rollback

### Topics
- Compliance Frameworks
  - Compliance frameworks (SR 11-7, EU AI Act)
- Model Governance
  - Model validation and governance
- Production Operations
  - Latency and production constraints
  - Monitoring and drift detection
  - Incident response and rollback

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `book_springer/lectures_source/Lesson15.1_Deployment_Monitoring_And_Adaptation.smd`
  - [55%]: Notebook-to-production, cost-aware deployment, monitoring causal
    assumptions, experimentation in production, versioning, error budgets,
    rollback, feedback loops, technical debt
- `book_springer/lectures_source/Lesson08.1_Causal_Data_Pipelines.smd`
  - [10%]: Detecting distribution shift and concept drift in production
- `data605/lectures_source/Lesson07.4-Big_Data_Architectures.smd`
  - [10%]: CI/CD, stages of deployment, semantic versioning, microservices vs.
    monolith
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [10%]: Causal monitoring and adaptation, safety constraints
- _Not covered_
  - [40%]: Compliance frameworks (SR 11-7, EU AI Act), model validation and
    governance process, latency budgets for financial systems

# Part V: Frontiers

## 13: Market Simulation

### Goals
- Generate market scenarios with GANs, diffusion models, and agent-based
  simulators
- Use simulated data for stress testing and for augmenting scarce data
- Evaluate fidelity against stylized facts such as fat tails and volatility
  clustering

### Topics
- Generative Models for Scenario Generation
  - Generative Adversarial Networks (GANs) for time series
  - Diffusion models for time series
- Agent-Based Market Simulation
  - Heterogeneous trader agents and market mechanisms
  - Calibrating simulators to real markets
- Applications
  - Stress testing and scenario analysis
  - Synthetic data: augmenting scarce data
- Fidelity
  - Stylized facts: fat tails, volatility clustering, autocorrelation of returns
  - Risks of training and testing on synthetic data

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [15%]: Survey of VAEs, GANs, normalizing flows, diffusion models,
    score-based generative modeling
- `book_cs_refreshers/lectures_source/Lesson96.Refresher_stochastic_processes.smd`
  - [10%]: Brownian motion, random walks, SDEs, and Monte Carlo methods for
    simulating price paths
- `book_cs_refreshers/lectures_source/Lesson95.Refresher_game_theory.smd`
  - [5%]: Equilibria, learning in games, multi-agent RL as the basis for
    agent-based simulation
- `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.smd`
  - [5%]: ARCH/GARCH volatility clustering as a fidelity benchmark
- _Not covered_
  - [70%]: GAN and diffusion architectures for time series, agent-based market
    simulators, stress testing, synthetic data augmentation, fidelity tests
    based on stylized facts

## 14: The Road Ahead

### Goals
- Show how multimodal models fuse text, tabular, and time series data
- Survey emerging architectures and paradigms
- Frame open research problems and the future of AI in finance

### Topics
- Multimodal Finance
  - Multimodal models combining modalities
  - Text, tabular, and time-series fusion
- Emerging Directions
  - Emerging architectures and paradigms
- Outlook
  - Open research problems
  - Future of AI in finance

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [20%]: Multimodal deep learning, self-supervised learning, neural ODEs,
    modern research frontiers
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [15%]: Emerging training patterns, open problems, recommendations
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [10%]: Path forward for trustworthy reasoning agents
- _Not covered_
  - [65%]: Multimodal fusion of text, tabular, and time series data, open research
    problems specific to finance
