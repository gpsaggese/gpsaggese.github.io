# Chapter 4: AI-Assisted Data Cleaning and Wrangling

## 1. Intelligent Missing Data Handling

### 1.1 Understanding Missing Data Patterns

- Three types of missing data (Rubin's taxonomy):
  - **MCAR** (Missing Completely at Random): missingness is independent of all
    variables; safe to impute with simple statistics
  - **MAR** (Missing at Random): missingness depends on observed variables;
    use model-based imputation
  - **MNAR** (Missing Not at Random): missingness depends on the missing value
    itself; requires domain knowledge
- AI role: infer the likely missingness mechanism from column descriptions and
  missingness correlations; recommend appropriate imputation strategy

### 1.2 Imputation Strategies

- Simple strategies: mean, median, mode — fast but ignore column relationships
- Model-based imputation: fit a model for each missing column using complete
  rows as training data
  - `IterativeImputer` (sklearn): iterates over all missing columns, training a
    regressor/classifier for each
- k-Nearest Neighbor imputation: impute using the average of k most similar rows
- Imputation in pipelines: always fit imputers on train set only, then transform
  both train and test

### 1.3 AI-Guided Imputation Selection

- Prompting pattern: share missingness summary + correlation with target ->
  ask for per-column imputation recommendation
- Common AI suggestions:
  - High missingness + low correlation with target -> drop the column
  - Numerical column + low missingness -> median imputation
  - Categorical column + correlated with other columns -> model-based imputation
  - Time series -> forward fill or interpolation

- TUTORIAL: Pyjanitor - Use AI-suggested imputation strategies implemented with
  pyjanitor's method-chaining API; compare mean, median, and model-based
  imputation approaches on a real dataset

- TUTORIAL: imbalanced-learn - Handle class imbalance caused by missing-data
  patterns; apply SMOTE oversampling guided by AI analysis of the imbalance root
  cause

- TUTORIAL: sktime - Apply time-series-aware imputation on sequential datasets
  using sktime's transformers; use AI to select the appropriate interpolation
  method based on series characteristics

---

## 2. Outlier Detection with AI Assistance

### 2.1 Types of Outliers

- Point outliers: single observations far from the bulk of the data
- Contextual outliers: values normal in isolation but anomalous in context (e.g.,
  temperature of 70°F in January in Alaska)
- Collective outliers: groups of observations that are anomalous together
- Detection strategy depends on type; AI can help classify which type is present

### 2.2 Statistical and ML Outlier Detection

- Univariate methods:
  - Z-score: flag values more than 3 standard deviations from the mean
  - IQR method: flag values outside 1.5 * IQR from the quartiles
- Multivariate methods:
  - Isolation Forest: isolates anomalies by random partitioning
  - DBSCAN: density-based clustering; outliers are points not assigned to any
    cluster
  - LOF (Local Outlier Factor): compares local density to neighbors
- Time series anomaly detection: ADTK, stumpy (matrix profile)

### 2.3 AI-Assisted Outlier Diagnosis

- After detection, use AI to distinguish between:
  - Data quality errors (typos, sensor malfunctions, ETL bugs)
  - Genuine rare events (fraud, equipment failure, extreme weather)
- Prompting pattern: "This transaction of $50,000 was flagged as an outlier.
  Given the customer's history, is this likely fraud or a legitimate large
  purchase?"

- TUTORIAL: pyod - Apply multiple outlier detection algorithms from PyOD to a
  dataset; use AI to select the most appropriate algorithm based on data
  characteristics and interpret the results

- TUTORIAL: adtk - Detect anomalies in time series data using ADTK; use AI to
  diagnose whether flagged anomalies are data quality issues or real events

- TUTORIAL: stumpy - Apply matrix profile analysis with stumpy to find recurring
  patterns and contextual anomalies in time series data; use AI to interpret
  discovered motifs and discords

---

## 3. Automated Data Transformation

### 3.1 Common Transformation Tasks

- Standardization and normalization: bring numeric features to comparable scales
  - Min-max scaling: [0, 1] range; sensitive to outliers
  - Z-score normalization: zero mean, unit variance; robust for Gaussian data
  - Robust scaling: uses median and IQR; robust to outliers
- Encoding categorical variables:
  - One-hot encoding: creates binary columns; increases dimensionality
  - Ordinal encoding: maps categories to integers; preserves order
  - Target encoding: replaces category with mean of target; risk of data leakage
- AI generates the transformation chain from a column description

### 3.2 Relational Data Wrangling

- Real-world data lives in multiple tables that must be joined
- AI can generate join logic from natural language descriptions:
  - "Join orders to customers on customer_id, keeping all orders even without
    a matching customer record"
- Deep feature synthesis: automatically traverse relationships to create
  features across multiple tables (Featuretools)

### 3.3 Scalable Data Transformation

- For datasets larger than memory, use distributed frameworks:
  - Apache Spark / PySpark: cluster-parallel transformations
  - Apache Beam: unified batch and stream transformation model
  - Modin: drop-in pandas replacement that parallelizes on Ray or Dask
- AI can generate PySpark or Beam code from pandas code:
  - "Convert this pandas transformation to PySpark"

- TUTORIAL: OpenRefine - Use OpenRefine's clustering and transformation
  functions to standardize messy categorical data; use AI to generate
  transformation recipes from natural language descriptions

- TUTORIAL: Featuretools - Automate relational data transformation with deep
  feature synthesis; use AI to interpret the generated features and select the
  most relevant ones

- TUTORIAL: Apache Beam - Write a scalable data cleaning pipeline with Apache
  Beam; use AI to generate the pipeline transforms from a description of the
  required cleaning steps; run on both local and distributed backends

---

## 4. Synthetic Data Generation

### 4.1 Why Synthetic Data

- Use cases for synthetic data:
  - Privacy-preserving development: test pipelines without exposing PII
  - Data augmentation: increase training set size for rare classes
  - Pipeline testing: verify transformations on controlled, known-good data
  - Sharing data across teams without regulatory risk
- Quality requirements:
  - Statistical fidelity: synthetic data should match real data distributions
  - Privacy: no individual real records should be recoverable from synthetic data

### 4.2 Generative Approaches

- **Statistical models**: fit marginal and joint distributions, sample from them
  - SDV (Synthetic Data Vault): GAN and CTGAN-based tabular synthesis
  - Gaussian copulas: model dependencies between columns
- **Rule-based generation**: Faker and PyDBGen generate realistic-looking data
  using domain-specific rules
  - Useful for test data that doesn't need to match a real distribution
- **Differential privacy**: add calibrated noise to protect individual records

### 4.3 Evaluating Synthetic Data Quality

- Metrics for synthetic data:
  - Column shape similarity: compare histograms of each column
  - Column pair trends: compare pairwise correlations
  - ML efficacy: train a model on synthetic, evaluate on real; compare to
    train-on-real baseline
- AI can interpret synthetic data quality reports and suggest adjustments to
  the synthesis parameters

- TUTORIAL: SDV (Synthetic Data Vault) - Train a synthetic data model on a
  sensitive dataset; generate privacy-preserving synthetic copies; evaluate
  statistical fidelity between real and synthetic distributions

- TUTORIAL: Faker - Build a configurable synthetic dataset generator for testing
  data pipelines; create realistic distributions for demographic, financial, and
  temporal columns

- TUTORIAL: Tonic.ai - Generate enterprise-grade synthetic data for a
  multi-table relational database; use AI to configure the synthesis parameters
  to preserve referential integrity and business logic
