# Chapter 5: AI-Driven Feature Engineering

## 1. LLM-Assisted Feature Hypothesis Generation

### 1.1 Feature Hypothesis Generation

- Feature engineering = creating new columns that help models predict the target
  more accurately
- Traditional approach: domain expert brainstorms features based on business
  knowledge
- AI-assisted approach: LLM generates feature hypotheses from schema + domain
  description
  - Input: column names, types, target variable, and domain context
  - Output: list of candidate features ranked by expected predictive value
- LLM advantage: broad knowledge of domain patterns across many industries
- LLM limitation: cannot estimate actual feature importance without seeing data;
  all suggestions require empirical validation

### 1.2 Prompting for Feature Ideas

- Effective feature hypothesis prompt:
  ```
  Dataset: customer transactions with columns [customer_id, amount, timestamp,
  merchant_category, is_international]
  Target: fraud (binary)
  Domain: retail banking fraud detection
  Suggest 10 feature engineering ideas ranked by expected signal strength.
  For each, explain why it should predict fraud.
  ```
- Common high-value feature types AI suggests:
  - Aggregations: rolling windows (sum, mean, std) over time
  - Ratios: amount / average_amount_for_category
  - Lag features: difference from previous transaction
  - Interaction terms: amount * is_international
  - Temporal features: hour of day, day of week, days since last transaction

### 1.3 Validating Feature Hypotheses

- Validate each hypothesis before adding to the model:
  - Compute the feature
  - Check mutual information or correlation with target
  - Add to a baseline model and compare performance
- Use AI to interpret the feature importance results and suggest refinements

- TUTORIAL: llm - Use an LLM to brainstorm domain-specific feature hypotheses
  given a dataset schema; evaluate each hypothesis against feature importance
  scores from a baseline model

- TUTORIAL: Featuretools - Generate candidate features automatically with deep
  feature synthesis; use AI to rank features by predicted predictive value
  before running the full feature matrix

- TUTORIAL: spaCy - Extract text features from unstructured columns using spaCy
  named entity recognition and dependency parsing; use AI to suggest which
  linguistic features are most predictive for the target task

---

## 2. Automated Feature Construction

### 2.1 Deep Feature Synthesis

- Deep feature synthesis (DFS): automatically generates features by traversing
  relationships in relational data
  - Applies mathematical operations (sum, count, mean, max, std) recursively
    across linked tables
  - Can generate hundreds of features from a few tables
- Challenge: many generated features are redundant or irrelevant; requires
  selection step

### 2.2 Polynomial and Interaction Features

- Polynomial features: $x_1^2$, $x_1 x_2$, $x_2^2$ — capture non-linear
  relationships
  - Risk: feature explosion; $n$ features produce $O(n^2)$ interaction terms
- Autofeat: automated polynomial feature generation with integrated selection
  to keep only statistically significant features

### 2.3 Time Series Feature Extraction

- Time series data requires domain-specific features:
  - Statistical: mean, variance, skewness, kurtosis over windows
  - Frequency domain: Fourier coefficients, dominant frequencies
  - Shape-based: number of peaks, zero-crossings, trend slope
- tsfresh: extracts 763 time series features from each time series column
  automatically with integrated relevance filtering

- TUTORIAL: tsfresh - Extract time series features automatically from a
  sequential dataset; use AI to filter the generated feature set based on
  statistical relevance and interpretability

- TUTORIAL: autofeat - Use Autofeat to generate polynomial and interaction
  features; compare AI-selected features to manual feature engineering on a
  regression task

- TUTORIAL: umap-learn - Apply UMAP dimensionality reduction to a high-
  dimensional feature set; use AI to interpret the 2D embedding structure and
  identify feature clusters that correspond to meaningful data segments

---

## 3. AI-Assisted Feature Selection

### 3.1 Why Feature Selection Matters

- Too many features cause:
  - Overfitting: model memorizes noise in training data
  - Curse of dimensionality: performance degrades in high-dimensional spaces
  - Increased training time and memory requirements
  - Reduced interpretability
- Feature selection methods:
  - Filter: statistical tests (chi-square, mutual information) independent of
    the model
  - Wrapper: recursive feature elimination using the model itself
  - Embedded: regularization (L1/Lasso) during training

### 3.2 Interpretation-Driven Selection

- Model-agnostic feature importance: SHAP values explain contribution of each
  feature to each prediction
  - Global importance: mean absolute SHAP value per feature
  - Local importance: per-prediction explanation
- AI interprets SHAP summary plots:
  - "Feature X has high magnitude but its direction flips based on feature Y;
    consider an interaction term between X and Y"
  - "Feature Z has near-zero importance; recommend dropping it"

### 3.3 Handling Correlated and Redundant Features

- Identify redundant features: high pairwise correlation (> 0.95)
  - Drop one of the pair; use AI to decide which one to keep based on
    interpretability and missingness rate
- Variance inflation factor (VIF): detects multicollinearity in regression
  - AI can generate the VIF computation and interpretation code

- TUTORIAL: SHAP - Use SHAP values to rank feature importance after model
  training; use AI to interpret the SHAP summary plot and recommend features to
  drop or transform

- TUTORIAL: imbalanced-learn - Apply feature selection in the context of
  imbalanced classification; use AI to diagnose whether low-performing features
  are caused by class imbalance or true irrelevance

- TUTORIAL: What-If Tool - Use the What-If Tool to interactively explore feature
  effects on model predictions; use AI to generate a written summary of the most
  influential features for a given decision boundary

---

## 4. Feature Store Integration

### 4.1 The Feature Store Concept

- Feature store = centralized registry for computing, storing, and serving
  features
  - Training store: historical features for model training
  - Online store: low-latency feature retrieval for real-time inference
- Benefits: eliminates redundant feature computation across teams; ensures
  training/serving consistency

### 4.2 Feature Definitions and Versioning

- Feature view: SQL or DataFrame logic that computes a feature from raw data
- Feature versioning: track changes to feature definitions over time
  - A model trained on feature version 1 must not be served with version 2
- AI can generate feature view definitions from natural language descriptions of
  the business logic

### 4.3 Weak Supervision for Feature Labels

- Weak supervision: generate noisy labels automatically using heuristic rules
  called labeling functions
- Snorkel: trains a label model from multiple noisy labeling functions
  - Useful when human annotation is expensive or slow
  - AI can write labeling functions from natural language descriptions of
    business rules

- TUTORIAL: Feast - Set up a local feature store with Feast; define feature
  views and retrieve features for training and serving; use AI to generate
  feature definitions from existing pandas transformations

- TUTORIAL: Snorkel - Build a weak supervision pipeline that generates
  programmatic labels as features; use AI to write labeling functions from
  natural language descriptions of business rules

- TUTORIAL: sktime - Define time-series feature transformers in sktime and
  register them in a reproducible feature pipeline; use AI to generate
  transformer configurations for a given forecasting problem
