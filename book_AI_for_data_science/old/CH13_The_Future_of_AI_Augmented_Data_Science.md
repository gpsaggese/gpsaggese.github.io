# Chapter 13: The Future of AI-Augmented Data Science

## 1. Foundation Models for Structured Data

### 1.1 Foundation Models and Tabular Data

- Foundation models = large models pretrained on massive data and adapted to
  downstream tasks via fine-tuning or prompting
  - Language foundation models (GPT, Claude, Llama): trained on text; can be
    applied to text columns, documentation, and code
  - Tabular foundation models: pretrained on diverse tabular datasets; can
    perform zero-shot or few-shot classification and regression
- Early results: tabular foundation models (TabPFN, TabFormer) are competitive
  with XGBoost on small datasets without any hyperparameter tuning

### 1.2 Transfer Learning for Structured Data

- Transfer learning workflow for tabular data:
  1. Pretrain on large, diverse tabular datasets
  2. Fine-tune on domain-specific data with a small number of labeled examples
  3. Evaluate zero-shot vs. few-shot vs. fine-tuned performance
- Use case: cold-start problems where labeled training data is scarce
- Current state: most effective when text columns are present; pure numeric
  tabular tasks still favor gradient boosting

### 1.3 Embeddings as Features

- Pretrained language models generate embeddings for text columns:
  - Sentence-transformers: 768-dimensional embeddings for free text
  - These embeddings become features for downstream ML models
- Advantage: capture semantic meaning without manual feature engineering
- Pattern: join embedding features with numeric features and train a gradient
  boosting model

- TUTORIAL: HuggingFace - Fine-tune a tabular foundation model on a
  domain-specific dataset; compare zero-shot and few-shot performance to a
  traditional XGBoost baseline; use AI to interpret the learned representations

- TUTORIAL: datasets - Use the Hugging Face datasets library to load and
  preprocess large structured datasets; demonstrate how foundation model
  pretraining data shapes performance on downstream tabular tasks

- TUTORIAL: peft - Apply parameter-efficient fine-tuning (LoRA) to a language
  model on a domain-specific tabular dataset with text columns; compare full
  fine-tuning to PEFT in terms of accuracy and compute cost

---

## 2. Multimodal AI in Data Pipelines

### 2.1 Multimodal Data in Practice

- Real-world data is rarely single-modal:
  - E-commerce: product images + text descriptions + sales history
  - Healthcare: clinical notes + lab values + imaging data
  - Finance: news articles + market data + company filings
- Multimodal models: combine representations from different modalities into a
  joint embedding space
- Engineering challenge: heterogeneous data types require different
  preprocessing and representation strategies

### 2.2 Feature Fusion Strategies

- Early fusion: concatenate raw features from different modalities before the
  model
  - Simple; may not capture cross-modal interactions well
- Late fusion: train separate models for each modality; combine predictions
  (average, stack, vote)
  - Flexible; can use best model for each modality
- Cross-attention fusion: use attention mechanisms to let each modality attend
  to the others
  - Captures complex interactions; requires more data and compute

### 2.3 Multimodal Pipelines with AI Assistance

- AI generates the feature fusion strategy from a description of the data:
  - "We have text reviews and numeric ratings. Suggest a fusion approach for
    a sentiment prediction task."
- AI selects preprocessing steps for each modality independently before fusion
- Speech-to-text as a data engineering step: extract structured information
  from audio recordings using Whisper

- TUTORIAL: Transformers - Build a multimodal pipeline that combines text and
  tabular features using a pretrained transformer; use AI to design the feature
  fusion strategy and evaluate its effectiveness

- TUTORIAL: Whisper Large V3 - Integrate speech-to-text transcription into a
  data pipeline; use Whisper to extract structured information from audio
  recordings as additional features for a prediction model

- TUTORIAL: sentence-transformers - Build a multimodal similarity search system
  that combines text embeddings and numeric features; use AI to design the
  similarity metric and evaluate retrieval quality

---

## 3. AI for Real-Time and Streaming ML

### 3.1 Streaming vs. Batch ML

- Batch ML: train offline, serve predictions on request or on a schedule
  - Works when latency of hours or days is acceptable
  - Easier to implement, test, and debug
- Streaming ML: process events as they arrive; update models continuously
  - Required when: real-time fraud detection, live recommendation systems,
    dynamic pricing, anomaly detection in industrial IoT
- Tradeoffs:
  - Streaming is harder to implement, debug, and reproduce
  - Batch retraining is simpler but may use stale models

### 3.2 Streaming Infrastructure

- Apache Kafka: distributed event streaming platform
  - Producer -> topic -> consumer pattern
  - Handles millions of events per second with durable storage
- Apache Flink / Apache Beam: stateful stream processing frameworks
  - Support windowing, joins, aggregations on unbounded streams
- River: online ML library; fits models one example at a time

### 3.3 AI in Streaming Pipelines

- AI generates the streaming pipeline code from a description:
  - "Write a Kafka consumer that reads transaction events, extracts features,
    and calls a fraud scoring model"
- AI diagnoses streaming failures:
  - "The consumer lag is growing. Given that processing time is stable, what
    could cause this?"
- AI designs the feature engineering strategy for streaming:
  - Stateful aggregations: rolling windows, session features
  - Feature consistency: ensure training and serving compute features
    identically

- TUTORIAL: Apache Kafka - Build a streaming data pipeline with Kafka; consume
  real-time events and apply an online model to each message; use AI to generate
  the producer and consumer code from a pipeline description

- TUTORIAL: River - Implement an online machine learning model with River that
  updates incrementally on a data stream; use AI to select the appropriate
  online algorithm based on the stream characteristics and concept drift profile

- TUTORIAL: Apache Beam - Build a unified batch and stream processing pipeline
  with Apache Beam; use AI to generate the transform code; run the same pipeline
  in batch mode for testing and streaming mode for production

---

## 4. Building an AI-Augmented Data Science Team

### 4.1 Role Evolution in AI-Augmented Teams

- Data science roles are evolving:
  - **Data scientist** -> problem framer, validator, and AI orchestrator rather
    than code author
  - **ML engineer** -> pipeline architect who builds the AI-augmented toolchain
  - **Data analyst** -> direct user of natural language AI interfaces for SQL
    and reporting
- New skills required:
  - Prompt engineering: communicate effectively with LLMs
  - AI output validation: verify correctness of generated code and analysis
  - Agent design: define bounded tasks and tool sets for agents

### 4.2 Team Workflow Patterns

- AI-augmented code review: every PR runs through an AI reviewer before human
  review; AI catches mechanical issues so humans can focus on design
- Shared prompt library: team maintains a repository of effective prompts for
  common tasks (EDA, model evaluation, pipeline scaffolding)
- AI pair programming: developer and AI agent work together in a shared coding
  environment; agent has access to the codebase and can run tests

### 4.3 Reproducibility and Knowledge Sharing at Team Scale

- Kedro as a team standard: enforces reproducible pipeline structure across
  all team members; reduces "it works on my machine" problems
- MLflow team server: shared experiment tracking; all team members contribute
  experiments to the same server; AI generates onboarding documentation from
  the experiment history
- Documentation culture: use AI to generate first-pass documentation; require
  human review before merging; AI keeps documentation in sync with code changes

- TUTORIAL: mlflow - Set up a team-wide MLflow tracking server; define
  conventions for experiment naming, tagging, and model registration; use AI to
  generate onboarding documentation from the server's experiment history

- TUTORIAL: Kedro - Adopt Kedro as a team standard for reproducible pipelines;
  use AI to migrate existing notebooks to Kedro nodes; demonstrate how the
  framework improves collaboration and code review

- TUTORIAL: Ray - Set up a Ray cluster for distributed team-wide ML workloads;
  use AI to generate the cluster configuration and job submission scripts; run
  a parallel hyperparameter search that multiple team members can observe
