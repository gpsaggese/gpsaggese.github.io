# Book Table of Contents

## 1. The Need for Probabilistic and Causal Machine Learning

### msml610/lectures_source/Lesson08.1-Causal_AI_intro.txt

- Introduction and Motivation (7)
  - Background (3)
  - What ML Systems Can and Cannot Tell You (4)
- Why Causal AI Matters (15)
  - Problems with Traditional AI (9)
  - Optimization vs. Inference vs. Decision Theory (2)
  - The Cost of Ignoring Causality (4)
- Causal AI Fundamentals (10)
  - The Ladder of Causation (4)
  - Correlation vs Causation Models (3)
  - Data Science vs. Decision Science (3)
- Causal AI in Business (18)
  - Business Context and Motivation (3)
  - The Causal AI Workflow (11)
  - Explainability and Interpretability (4)

## 2: Bayesian Networks

### msml610/lectures_source/Lesson06.1-Bayesian_Networks.txt

- Logic-Based AI Under Uncertainty (10)
  - Problem (3)
  - Solution (7)
- Probabilistic Reasoning (18)
  - Conditional Independence (5)
  - Bayesian Networks (13)

### msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.txt

- Semantics of Bayesian Networks (8)
- Constructing a Bayesian Network (34)
- Exact Inference in Bayesian Networks (4)
- Approximate Inference in Bayesian Networks (28)

## 3: Causal DAGs and Structural Models

### msml610/lectures_source/Lesson08.3-Do_Calculus.txt

  - Intervention and Counterfactuals (5)
  - Randomized Controlled Trial (3)
  - Backdoor Adjustment (6)
  - Frontdoor Adjustment (5)
  - Do-Calculus (3)

## 4: From Causal Models to Code

### msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.txt

- Concepts (12)
- Coin Example (18)
  - Analytical Approach (7)
  - Frequentist vs Bayesian (6)
  - Probabilistic Programming (5)

### msml610/lectures_source/Lesson07.2-Posterior_Based_Decisions.txt

- Posterior-Based Decisions (34)
  - Chemical Shift: Example (5)
  - Posterior Predictive Checks (10)
  - Groups Comparison (5)

### msml610/lectures_source/Lesson07.3-Hierarchical_Models.txt

- Hierarchical Models (22)

### msml610/lectures_source/Lesson07.4-Generalized_Linear_Models.txt

- Generalized Linear Models (29)
  - Simple Linear Model (15)
  - Logistic Regression (8)
  - Multiple Linear Regression (6)

### msml610/lectures_source/Lesson07.5-Bayesian_Model_Comparison.txt

- Bayesian Model Comparison (39)
  - The Balance Between Simplicity and Accuracy (3)
  - Measures of Predictive Accuracy (11)
  - Bayesian Model Selection and Ensemble (2)
  - Bayesian Hypothesis Testing (7)
  - Regularizing Priors (2)

## 5: Interventions, Experiments, and Adjustments

### msml610/lectures_source/Lesson08.3-Do_Calculus.txt

  - Intervention and Counterfactuals (5)
  - Randomized Controlled Trial (3)
  - Backdoor Adjustment (6)
  - Frontdoor Adjustment (5)
  - Do-Calculus (3)

## 6: Causal Identification and Estimation

### msml610/lectures_source/Lesson08.4.txt

- Introduction to Causal Inference (50)
- Randomized Experiments and Stats Review (8)
- Graphical Causal Models (34)
- The Unreasonable Effectiveness of Linear Regression (20)
  - Feature Selection Dilemma
- Propensity score (20)
- Effect heterogeneity (18)
- Metalearners (26)
- Difference-in-differences (5)
  - Canonical Difference-in-Differences (5)
  - Identification Assumption
- Synthetic control
- Geo and switchback experiments
- Non-compliance and instruments
- Next steps

## 8: Causal Inference for Time Series

### msml610/lectures_source/Lesson10-Timeseries_forecasting.txt

- Time Series (21)
  - Basic definition (9)
  - Time series operators (6)
  - Time series decomposition (6)
- Classical Methods (33)
  - Simple Models for Stochastic Process (3)
  - Autoregressive models (11)
  - Moving average models (5)
  - ARMA(p, q) process (6)
  - ARCH model (8)
- Modern Approaches (10)
- Special techniques for time series modeling (14)
  - Self-Supervised and Representation Learning for Time Series
  - Hierarchical Bayesian Forecasting
  - Reinforcement Learning for Time Series Decision Making
  - Transformers and Attention Mechanisms for Time Series
  - Energy-Based Models and Diffusion Models for Forecasting
  - Time Series Generative Models
  - Long-Horizon Forecasting Challenges
  - Uncertainty Quantification and Calibration

### msml610/lectures_source/Lesson10.1-Causal_Inference_for_Time_Series.txt

- Time Series vs. Cross-Sectional Causality (10)
  - Temporal Causal Structures (3)
  - Challenges Specific to Time Series (5)
  - When Temporal Structure Helps and When It Misleads (2)
- Granger Causality (7)
  - Definition and Intuition (3)
  - Assumptions and Limitations (2)
  - Practical Examples (2)
- Interrupted Time Series (ITS) (8)
  - Design and Estimation (4)
  - ITS and Regression Discontinuity (2)
  - Applications in Causal Inference (2)
- Difference-in-Differences (DiD) (10)
  - Parallel Trends Assumption (5)
  - Estimation and Robustness (2)
  - Extensions: Multiple Time Periods (3)
- Synthetic Control Methods (9)
  - Constructing a Counterfactual from Donor Series (2)
  - Weighted Combinations and Optimal Weights (3)
  - When Synthetic Control Succeeds and Fails (4)

## 9: A/B Testing and Experimentation

### msml610/lectures_source/Lesson09.3-Multi_Armed_Bandits.txt

- Introduction (24)
- Algorithms (30)
- Bayesian Approaches (40)

## 10: Causal Discovery

### msml610/lectures_source/Lesson10.2-Causal_Discovery.txt

- The Discovery Problem (11)
  - Inferring Causal Structure from Observational Data (4)
  - Identifiability and Causal Sufficiency (5)
  - Practical Limitations (2)
- When to Use Discovery vs Domain Knowledge (4)
  - Discovery as Hypothesis Generation (2)
  - Combining Discovery with Domain Knowledge (2)
- Discovery Algorithm Families (22)
- Challenges and Validation (8)
