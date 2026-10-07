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
  - 02, Deep Learning Refresher
  - 03, Financial Data Engineering
- Transformers and LLMs
  - 04, Attention and the Transformer Architecture
  - 05, LLMs and Foundation Models
  - 06, Adapting LLMs to Finance
- Market Prediction with Deep Learning
  - 07, Price and Volatility Forecasting
  - 08, Limit Order Book Modeling
  - 09, Portfolio Optimization and Reinforcement Learning
- Language in Finance
  - 10, NLP for Filings, News, and Earnings Calls
  - 11, LLM Agents for Financial Research
  - 12, Conversational Finance Assistants
- Risk, Operations, and Governance
  - 13, Credit Risk and Fraud Detection
  - 14, Explainability and Robustness
  - 15, Regulation, Model Risk Management, and Deployment
- Frontiers
  - 16, Generative Models for Synthetic Markets
  - 17, The Road Ahead

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
| 02. Deep Learning Refresher                             | N/A    |          |           |          |      |
| 03. Financial Data Engineering                          | N/A    |          |           |          |      |
| **Transformers and LLMs**                               |        |          |           |          |      |
| 04. Attention and the Transformer Architecture          | N/A    |          |           |          |      |
| 05. LLMs and Foundation Models                          | N/A    |          |           |          |      |
| 06. Adapting LLMs to Finance                            | N/A    |          |           |          |      |
| **Market Prediction with Deep Learning**                |        |          |           |          |      |
| 07. Price and Volatility Forecasting                    | N/A    |          |           |          |      |
| 08. Limit Order Book Modeling                           | N/A    |          |           |          |      |
| 09. Portfolio Optimization and Reinforcement Learning   | N/A    |          |           |          |      |
| **Language in Finance**                                 |        |          |           |          |      |
| 10. NLP for Filings, News, and Earnings Calls           | N/A    |          |           |          |      |
| 11. LLM Agents for Financial Research                   | N/A    |          |           |          |      |
| 12. Conversational Finance Assistants                   | N/A    |          |           |          |      |
| **Risk, Operations, and Governance**                    |        |          |           |          |      |
| 13. Credit Risk and Fraud Detection                     | N/A    |          |           |          |      |
| 14. Explainability and Robustness                       | N/A    |          |           |          |      |
| 15. Regulation, Model Risk Management, and Deployment   | N/A    |          |           |          |      |
| **Frontiers**                                           |        |          |           |          |      |
| 16. Generative Models for Synthetic Markets             | N/A    |          |           |          |      |
| 17. The Road Ahead                                      | N/A    |          |           |          |      |

## `book_Modern_AI_for_Finance` Tutorials

> find book_Modern_AI_for_Finance/tutorials -name *.ipynb
```
```

## TODOs
- Write one slide deck per chapter in `book_Modern_AI_for_Finance/lectures_source/`
  (30-35 slides each)
- Write the finance-specific content from scratch for chapters where the
  `_Not covered_` share exceeds 50%: 03, 08, 10, 13, 16, 17
- Add tutorial notebooks under `book_Modern_AI_for_Finance/tutorials/`
- Collect related books, packages, and papers for each chapter: the previous map
  had empty placeholders for them
- Generate `book_Modern_AI_for_Finance/all_tocs.md` once the slide decks exist

# Detailed TOC

# Part I: Foundations

## 01: Why Finance Is Different

### Goals
- Explain why financial data breaks the assumptions of standard ML
- Show how noise, fat tails, and regime shifts degrade model performance
- Teach backtest discipline to avoid overfitting and false discoveries

### Topics
- Statistical Properties of Financial Data
  - Non-stationarity and regime shifts
  - Fat tails and extreme events
  - Low signal-to-noise ratios
- Evaluating Strategies
  - Backtest overfitting and its pitfalls
  - Data snooping and multiple testing
- Challenges Unique to Financial ML
  - Small effective sample sizes
  - Markets that adapt to published signals

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.smd`
  - [30%]: Strict and weak stationarity, ACF, AR(1) for financial series,
    ARCH/GARCH volatility, change-point detection, Markov-switching models,
    cross-validation for time series
- `msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd`
  - [25%]: Data snooping, "burning the test set", sampling bias, out-of-sample
    discipline
- `msml610/lectures_source/Lesson05.2-Overfitting.smd`
  - [20%]: Overfitting, bias-variance trade-off, deterministic vs. stochastic
    noise, data size vs. noise
- `book_springer/lectures_source/Lesson08.1_Causal_Data_Pipelines.smd`
  - [15%]: Distribution shift, concept drift, non-stationarity, detecting shift
    in production
- `book_cs_refreshers/lectures_source/Lesson91.Refresher_probability.smd`
  - [10%]: Kurtosis and excess kurtosis as the definition of fat tails
- _Not covered_
  - [35%]: Extreme-value and tail-risk modeling, multiple-testing corrections for
    backtests, adaptive markets, finance-specific regime detection

## 02: Deep Learning Refresher

### Goals
- Review the neural architectures used throughout the book
- Explain training dynamics: optimization, initialization, and normalization
- Show regularization methods for small and noisy financial datasets

### Topics
- Feedforward Networks
  - Multilayer Perceptrons (MLPs)
  - Backpropagation and automatic differentiation
- Architectures for Spatial and Sequential Data
  - Convolutional Neural Networks (CNNs)
  - Recurrent Neural Networks (RNNs) and LSTMs
- Optimization and Training Dynamics
  - Gradient-based optimizers and learning-rate schedules
  - Vanishing and exploding gradients
- Regularization for Small and Noisy Datasets
  - Weight decay, dropout, and early stopping
  - Data augmentation

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [90%]: Perceptron and MLP, activations, backpropagation, automatic
    differentiation, CNNs and ResNets, RNNs, vanishing gradients, LSTM/GRU,
    weight initialization, batch normalization, learning-rate schedules, early
    stopping, regularization, data augmentation
- `msml610/lectures_source/Lesson05.2-Overfitting.smd`
  - [20%]: Bias-variance analysis, learning curves, high-bias vs. high-variance
    regimes as the basis for regularization
- `msml610/lectures_source/Lesson05.3-Learn_Validation.smd`
  - [10%]: Cross-validation and bootstrap for tuning on scarce data
- _Not covered_
  - [10%]: Regularization tuned to financial noise levels, time-aware
    validation for neural networks

## 03: Financial Data Engineering

### Goals
- Describe the main market, fundamental, and alternative data sources
- Show how to build point-in-time correct datasets for backtesting
- Identify survivorship bias and data quality issues before modeling

### Topics
- Market Data
  - Tick data and market data streams
  - Order book structure and evolution
- Fundamental and Alternative Data
  - Fundamental data sources
  - Satellite, web scraping, and credit card data
- Data Integrity for Backtesting
  - Point-in-time correctness and backtesting
  - Survivorship bias
  - Data quality issues

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `data605/lectures_source/Lesson10.2-Streaming_and_Real_Time_Analytics.smd`
  - [25%]: Data streams, pub-sub systems, delivery semantics, event vs.
    processing time, Kafka, Flink, Spark Structured Streaming
- `data605/lectures_source/Lesson02.3-Data_Pipelines.smd`
  - [20%]: Data ingestion, ETL/ELT paradigms, workflow orchestration, data lake vs.
    warehouse, data cleaning
- `data605/lectures_source/Lesson07.2-Data_Wrangling.smd`
  - [20%]: Tidy data, single-source and multi-source data problems, outlier
    detection including time series outliers
- `book_springer/lectures_source/Lesson08.1_Causal_Data_Pipelines.smd`
  - [20%]: Selection bias, missing-data mechanisms, distribution shift,
    measurement error, pre-flight data quality checks
- `data605/lectures_source/Lesson07.3-Serialization_Formats.smd`
  - [10%]: CSV, Parquet, JSON, and Protocol Buffers for storing market data
- _Not covered_
  - [55%]: Order book structure, fundamental and alternative data sources,
    point-in-time (as-of) joins, survivorship-bias-free universes

# Part II: Transformers and LLMs

## 04: Attention and the Transformer Architecture

### Goals
- Explain self-attention and multi-head attention step by step
- Show how positional encoding injects order into attention
- Compare transformers with RNNs on long financial sequences

### Topics
- Attention Mechanisms
  - Self-attention mechanisms
  - Multi-head attention
- Sequence Order
  - Positional encoding
- Transformers vs. Recurrent Models
  - Why transformers beat RNNs on long sequences
  - Parallelism and the cost of attention
- Architectural Variants for Finance
  - Variants for time series and multi-asset inputs

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [85%]: Query-key-value attention, scaled dot-product, multi-head, self vs.
    cross vs. causal attention, positional encoding, transformer block,
    residual connections, LayerNorm, attention cost
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [40%]: RNN limits, sequence-to-sequence attention, types of attention,
    transformer architecture, pretraining
- _Not covered_
  - [10%]: Finance-specific variants: time series transformers, multi-asset
    attention

## 05: LLMs and Foundation Models

### Goals
- Describe how LLMs are pretrained and why they scale
- Survey the modern LLM landscape and how to evaluate it
- Identify LLM limitations and failure modes that matter in finance

### Topics
- Pretraining
  - Pretraining objectives and datasets
- Scaling
  - Scaling laws and emergent reasoning
- The Model Landscape
  - The modern LLM landscape
  - Open vs. proprietary models
- Evaluation and Limits
  - Evaluation and benchmarking
  - Limitations and failure modes

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [40%]: Masked vs. autoregressive objectives, pretraining pipeline, scaling
    laws, emergent abilities, decoding, context windows, expressivity limits
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [30%]: Chain-of-thought and variants, premise-order brittleness, limits of
    self-correction
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [25%]: Data curation, multi-stage training pipelines, case studies of open
    frontier models, training cost
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [20%]: Verifiable benchmarks, reward hacking, the verification gap
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [10%]: Needle-in-a-haystack evaluation of long context
- _Not covered_
  - [15%]: Survey of general-purpose benchmark suites, composition of pretraining
    datasets

## 06: Adapting LLMs to Finance

### Goals
- Compare fine-tuning, instruction tuning, and retrieval for finance
- Show how to build a RAG pipeline over financial documents
- Review domain-pretrained finance models and their trade-offs

### Topics
- Fine-Tuning
  - Fine-tuning strategies
  - Instruction tuning for financial tasks
- Retrieval-Augmented Generation (RAG)
  - Retrieval over filings, news, and research
  - RAG vs. fine-tuning trade-offs
- Domain-Pretrained Models
  - Domain-pretrained finance models
  - BloombergGPT and similar approaches

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [40%]: RAG loop, retrieval as a tool call, vector databases, approximate
    nearest-neighbor indexes, hybrid search, grounding in regulated domains,
    long context vs. retrieval
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [25%]: Preference data, reward models and PPO, DPO, iterative preference
    optimization
- `book_Agentic_AI/lectures_source/Lesson10.1_Open_training_recipes_for_reasoning.smd`
  - [25%]: Open post-training pipeline, DPO vs. PPO, preference feedback
    collection, RAG plus reasoning training for scientific synthesis
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [10%]: Data curation, multi-stage training pipelines, training cost
- _Not covered_
  - [40%]: Parameter-efficient fine-tuning, supervised instruction tuning on
    financial tasks, domain-pretrained finance models

# Part III: Market Prediction with Deep Learning

## 07: Price and Volatility Forecasting

### Goals
- Cover the time series fundamentals needed for financial forecasting
- Apply sequence models and temporal transformers to prices and volatility
- Evaluate forecasts with metrics that reflect trading value

### Topics
- Time Series Prediction Fundamentals
  - Stationarity, autocorrelation, and volatility clustering
  - Classical baselines: ARIMA and GARCH
- Deep Sequence Models
  - Sequence-to-sequence models
  - Temporal transformers for forecasting
- Multi-Asset Modeling
  - Attention over multi-asset panels
- Evaluation
  - Evaluation metrics for financial predictions
  - Time-aware validation

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.smd`
  - [55%]: Stationarity, ACF, decomposition, AR/MA/ARIMA, ARCH/GARCH, VAR, state
    space models, ML and deep learning for time series, cross-validation for time
    series, probabilistic forecasting
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [25%]: RNNs for language models, sequence-to-sequence attention, transformer
    architecture
- `msml610/lectures_source/Lesson09.5-Kalman_Filter.smd`
  - [10%]: Filtering and prediction in state space models
- `msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd`
  - [10%]: MSE, RMSE, median-based metrics, train/validation/test splits
- _Not covered_
  - [35%]: Temporal transformers for forecasting, multi-asset panel attention,
    trading-oriented metrics (directional accuracy, information coefficient)

## 08: Limit Order Book Modeling

### Goals
- Introduce market microstructure and the limit order book
- Show deep learning models on order flow for short-horizon signals
- Explain market impact and the limits of high-frequency strategies

### Topics
- Market Microstructure Fundamentals
  - Limit order book mechanics
  - Order flow, spread, and depth
- Deep Learning on Order Flow
  - Models for book snapshots and event sequences
- Market Impact
  - Market impact estimation
- High-Frequency Signals
  - High-frequency trading signals
  - Practical applications and constraints

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

## 09: Portfolio Optimization and Reinforcement Learning

### Goals
- Embed portfolio optimization inside differentiable models
- Frame execution and allocation as reinforcement learning problems
- Model transaction costs and risk constraints explicitly

### Topics
- Portfolio Optimization with Deep Learning
  - Differentiable portfolio layers
  - Risk constraints and objectives
- Reinforcement Learning in Trading
  - Reinforcement learning for execution
  - Reinforcement learning for asset allocation
- Trading Frictions
  - Transaction cost modeling
  - Costs inside the reward function

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [45%]: MDPs, Bellman equation, value and policy iteration, model-based vs.
    model-free RL, temporal-difference and Q-learning, policy search, safe
    exploration, off-policy deconfounding
- `book_springer/lectures_source/Lesson12.1_Complex_Decisions.smd`
  - [35%]: Policy-gradient and actor-critic methods, planning with learned
    models, hierarchical decisions, offline and batch RL
- `book_springer/lectures_source/Lesson02.3_Integrating_Business_Objective_And_Real_World_Dynamics.smd`
  - [25%]: Decision-focused learning, utility functions, risk preferences, cost
    asymmetries, multi-objective Pareto trade-offs
- _Not covered_
  - [40%]: Differentiable portfolio layers, execution-specific RL, transaction
    cost and market impact models, mean-variance and CVaR constraints

# Part IV: Language in Finance

## 10: NLP for Filings, News, and Earnings Calls

### Goals
- Extract signals from filings, news, and earnings-call transcripts
- Apply sentiment, entity, and event extraction to financial text
- Link text events to market data over time

### Topics
- Sentiment Extraction and Analysis
  - Lexicon-based and model-based sentiment
- Information Extraction
  - Named entity and event extraction
  - 10-K and 10-Q parsing
- Text Classification
  - Financial text classification
- Temporal Analysis
  - Temporal analysis of financial events

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
  - [60%]: Financial sentiment models, named entity and event extraction, SEC
    filing parsing, financial text classification

## 11: LLM Agents for Financial Research

### Goals
- Build agents that call tools and retrieve financial data
- Design multi-agent analyst workflows with planning and reasoning
- Verify agent outputs with fact-checking before use

### Topics
- Tool-Using Agents
  - Tool-using agents and function calling
  - Retrieval pipelines for financial data
- Agent Workflows
  - Multi-agent analyst workflows
  - Agent reasoning and planning
- Verification
  - Fact-checking and verification

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
- _Not covered_
  - [35%]: Multi-agent analyst workflow design, finance-specific tools (market data
    APIs, filings)

## 12: Conversational Finance Assistants

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

# Part V: Risk, Operations, and Governance

## 13: Credit Risk and Fraud Detection

### Goals
- Apply tabular deep learning to credit scoring
- Use graph neural networks to detect fraud in transaction networks
- Meet real-time and regulatory requirements in scoring systems

### Topics
- Credit Risk
  - Tabular deep learning for credit
- Fraud Detection
  - Graph neural networks for transactions
  - Fraud patterns and detection
- Production Scoring
  - Real-time scoring systems
  - Regulatory considerations

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson04.3-Models.smd`
  - [20%]: Anomaly detection with Gaussian models, evaluation of anomaly
    detectors, hacked-account example, k-means clustering
- `msml610/lectures_source/Lesson04.1-Models.smd`
  - [15%]: Decision trees, random forests, feature selection with trees for
    tabular data
- `msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd`
  - [15%]: Confusion matrix, precision and recall, skewed classes,
    precision-recall curves
- `book_springer/lectures_source/Lesson15.1_Deployment_Monitoring_And_Adaptation.smd`
  - [10%]: High-asymmetry regulated decisions (credit denial), feedback loops
    from fraud flags
- `data605/lectures_source/Lesson12.3-Graph_Data_Processing.smd`
  - [10%]: Graph analysis tasks, subgraph matching, Pregel-style graph processing
- `data605/lectures_source/Lesson10.2-Streaming_and_Real_Time_Analytics.smd`
  - [10%]: Streaming infrastructure for real-time scoring
- _Not covered_
  - [55%]: Tabular deep learning, graph neural networks, fraud typologies,
    fair-lending regulation

## 14: Explainability and Robustness

### Goals
- Explain model predictions with Shapley values and attention
- Measure feature importance and robustness to adversarial inputs
- Manage hallucination risk and explain financial decisions

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

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson13.1-Explainability.smd`
  - [60%]: SHAP, LIME, permutation feature importance, counterfactual
    explanations, faithfulness and stability of explanations, accuracy vs.
    interpretability trade-off
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [15%]: Saliency maps, SHAP, attention for interpretability, layer-wise
    relevance propagation
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [15%]: Chain-of-Verification to reduce hallucination
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [15%]: Causal explanations, robustness through causal constraints,
    adversarial robustness with causal models
- _Not covered_
  - [15%]: Finance-specific feature importance, adversarial attacks on market
    models

## 15: Regulation, Model Risk Management, and Deployment

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

# Part VI: Frontiers

## 16: Generative Models for Synthetic Markets

### Goals
- Explain GANs and diffusion models for financial time series
- Generate synthetic data and stress scenarios for testing and risk
- Evaluate the fidelity of synthetic data and the risks of using it

### Topics
- Generative Architectures
  - Generative Adversarial Networks (GANs)
  - Diffusion models for time series
- Applications
  - Synthetic financial data generation
  - Stress-scenario generation
- Evaluation and Risk
  - Model evaluation and risk

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [20%]: Survey of VAEs, GANs, normalizing flows, diffusion models,
    score-based generative modeling
- `book_cs_refreshers/lectures_source/Lesson96.Refresher_stochastic_processes.smd`
  - [10%]: Brownian motion as the base process for simulating price paths
- _Not covered_
  - [70%]: GAN and diffusion architectures for time series, synthetic financial data
    pipelines, stress-scenario generation, fidelity and risk evaluation

## 17: The Road Ahead

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
