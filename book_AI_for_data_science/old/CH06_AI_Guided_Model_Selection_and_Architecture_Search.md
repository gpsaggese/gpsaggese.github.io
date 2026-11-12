# Chapter 6: AI-Guided Model Selection and Architecture Search

## 1. LLM-Guided Model Selection

### 1.1 The Model Selection Problem

- Model selection = choosing which ML algorithm and configuration is appropriate
  for a given problem
  - No free lunch theorem: no single algorithm wins on all problems
  - Selection depends on: data size, feature types, label type, interpretability
    requirements, latency constraints
- Traditional approach: try several algorithms, compare cross-validated performance
- AI-assisted approach: LLM recommends a shortlist of algorithms based on
  problem characteristics before any code is run

### 1.2 Problem Characterization for Model Selection

- Factors the AI reasons over:
  - **Label type**: binary classification, multi-class, regression, ranking,
    clustering
  - **Feature types**: numeric, categorical, text, image, time series
  - **Dataset size**: small (<10K rows) vs. large (>1M rows)
  - **Interpretability requirement**: regulated industries require explainable
    models (logistic regression, decision tree) over black-box models
  - **Latency budget**: gradient boosting inference is fast; neural networks may
    require GPU
- AI prompt pattern: describe the problem with these factors and request a
  ranked algorithm shortlist with justification

### 1.3 Benchmarking Model Families

- Gradient boosting ensembles (XGBoost, LightGBM, CatBoost) often lead on
  structured/tabular data
  - XGBoost: efficient, widely supported, requires careful hyperparameter tuning
  - LightGBM: faster than XGBoost on large datasets, leaf-wise tree growth
  - CatBoost: handles categorical features natively without encoding
- Deep learning (MLPs, TabNet): competitive for very large datasets or
  embeddings of high-cardinality categoricals
- H2O AutoML and PyCaret: automated benchmarking frameworks that compare many
  algorithms automatically

- TUTORIAL: H2O.ai - Use H2O AutoML to benchmark multiple algorithms on a
  tabular dataset; use AI to interpret the leaderboard and recommend the best
  model for the business objective

- TUTORIAL: pycaret - Compare model families using PyCaret's compare_models();
  use AI to explain why certain algorithms outperform others on specific dataset
  characteristics

- TUTORIAL: LightGBM - Train a LightGBM model on a large tabular dataset; use
  AI to generate the hyperparameter search space and interpret the learning
  curves; compare to a logistic regression baseline

---

## 2. AutoML in Practice

### 2.1 What AutoML Automates

- AutoML pipeline search includes:
  - Preprocessing: imputation, scaling, encoding strategies
  - Algorithm selection: which model family to use
  - Hyperparameter tuning: optimal settings for the selected algorithm
  - Ensemble construction: combining multiple models
- AutoML does NOT automate:
  - Problem framing and target definition
  - Data collection and quality assessment
  - Feature engineering that requires domain knowledge
  - Deployment and monitoring

### 2.2 AutoML Tools Landscape

- `auto-sklearn`: based on Bayesian optimization over sklearn pipelines; uses
  meta-learning to warm-start the search from similar datasets
- `TPOT`: uses genetic programming to evolve the best pipeline; exports the
  result as runnable Python code
- `AutoGluon`: strong on multi-modal data; builds stacking ensembles
  automatically; state-of-the-art performance on many benchmarks
- `FLAML`: fast, resource-aware search; minimizes compute cost to reach a
  target performance threshold

### 2.3 AI + AutoML Workflow

- Use AI to configure AutoML:
  - Generate the time budget based on dataset size and business deadline
  - Constrain the search space based on interpretability requirements
  - Interpret the selected pipeline components
- Post-AutoML: use AI to review the generated code, explain it, and identify
  potential issues before production deployment

- TUTORIAL: auto-sklearn - Run auto-sklearn on a classification problem; inspect
  the selected pipeline components; use AI to explain the ensemble weighting and
  preprocessing choices

- TUTORIAL: TPOT - Use TPOT's genetic programming to evolve an ML pipeline;
  export the best pipeline as Python code; use AI to review and refine the
  generated code

- TUTORIAL: AutoGluon - Apply AutoGluon to a multimodal dataset combining
  tabular and text features; use AI to interpret the stacking ensemble and
  select a latency-appropriate deployment model

- TUTORIAL: CatBoost - Train a CatBoost model with native categorical feature
  support; use AI to compare encoding strategies and explain why CatBoost avoids
  the need for manual one-hot encoding

---

## 3. Neural Architecture Search

### 3.1 NAS Fundamentals

- Neural Architecture Search (NAS): automatically design neural network
  architectures instead of manually specifying layer types and dimensions
- Search space: defines the possible architectures to consider
  - Layer types: convolutional, recurrent, attention, dense
  - Number of layers and units per layer
  - Skip connections and activation functions
- Search strategy:
  - Reinforcement learning: controller generates architectures; evaluates them
  - Evolutionary search: mutate and select best-performing architectures
  - Differentiable NAS (DARTS): relax the discrete search space to continuous

### 3.2 NAS Tools for Practitioners

- AutoKeras: high-level NAS for classification and regression; searches image,
  text, and tabular architectures
- FLAML: compute-budget-aware NAS; useful when GPU time is limited
- Use case: NAS is most valuable for image and text tasks; for tabular data,
  gradient boosting usually outperforms neural architectures

### 3.3 AI Assistance in Architecture Design

- AI can reason about architecture choices without running NAS:
  - "For a dataset with 50K rows and 200 features, is a 3-layer MLP or a
    TabNet architecture more likely to outperform gradient boosting?"
  - AI recommends starting with gradient boosting and only investing in NAS
    if tabular deep learning is required (e.g., shared representation with text)

- TUTORIAL: AutoKeras - Run neural architecture search with AutoKeras on an
  image or text classification task; use AI to interpret the discovered
  architecture and estimate its computational cost

- TUTORIAL: FLAML - Use FLAML's low-cost AutoML to search neural architectures
  within a compute budget; compare the NAS result to a hand-designed baseline

---

## 4. Hyperparameter Tuning with AI Assistance

### 4.1 The Hyperparameter Tuning Problem

- Hyperparameters control the model training process but are not learned from
  data:
  - Learning rate, batch size, number of trees, max depth, regularization
    strength
- Tuning strategies:
  - Grid search: exhaustive but exponentially expensive
  - Random search: samples randomly from the search space; surprisingly
    effective
  - Bayesian optimization: builds a probabilistic model of the objective
    function and selects the most promising next configuration
  - Population-based training: evolves a population of configurations in
    parallel

### 4.2 Bayesian Optimization Tools

- Optuna: define an objective function; Optuna samples hyperparameters using
  Tree-structured Parzen Estimator (TPE); visualize the optimization history
- Ax: Adaptive Experimentation platform; supports both Bayesian optimization and
  bandit approaches; designed for large-scale experimentation
- BoTorch: Bayesian optimization in PyTorch; flexible for custom acquisition
  functions

### 4.3 AI-Assisted Hyperparameter Ranges

- AI suggests initial hyperparameter ranges from domain knowledge:
  - "For a gradient boosting model on 100K rows with 50 features, suggest
    reasonable ranges for learning rate, max depth, and n_estimators"
- AI interprets optimization history: "The parallel coordinate plot shows that
  low learning rates consistently perform better; narrow the search to [0.001,
  0.01]"
- Warm-starting: use AI suggestions as the initial trials to avoid wasting
  compute on clearly bad configurations

- TUTORIAL: optuna - Define an Optuna study to tune an XGBoost model; use AI
  to suggest initial hyperparameter ranges and interpret the optimization history

- TUTORIAL: Ax - Use Ax's Bayesian optimization to tune a deep learning model;
  demonstrate how AI suggestions can warm-start the search with informed priors

- TUTORIAL: nevergrad - Apply derivative-free optimization to a non-differentiable
  pipeline; use AI to select the most appropriate optimizer for the problem
  structure

- TUTORIAL: Hyperopt - Use Hyperopt with the Tree of Parzen Estimators to tune
  a LightGBM model; use AI to interpret the best trial and explain why those
  hyperparameters perform well
