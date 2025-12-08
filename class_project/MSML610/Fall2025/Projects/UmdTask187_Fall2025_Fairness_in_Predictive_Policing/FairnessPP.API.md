# FairnessPP API Documentation

## Overview
**FairnessPP** (Fairness in Predictive Policing) is a high-level Python API designed to democratize access to fair machine learning algorithms for high-stakes public sector applications.

It provides a streamlined interface around **Scikit-Learn** and **Fairlearn**, allowing users to switch between baseline models and fairness-constrained models without rewriting pipelines.

The core philosophy is **Abstraction of Complexity**: users interact with a single `FairnessPredictor` object, while reweighting, reduction, and metric calculations are handled internally.

---

## 1. Configuration Objects

### `ModelConfig`
Configuration container for model hyperparameters.

**Parameters:**
- `n_estimators` (int): Number of boosting stages. Default: `50`
- `random_state` (int): Random seed. Default: `42`
- `max_iter_mitigation` (int): Max iterations for `ExponentiatedGradient`. Default: `50`

**Usage:**
```python
from FairnessPP_utils import ModelConfig

config = ModelConfig(
    n_estimators=100,
    max_iter_mitigation=20
)
EvaluationResult
Structured return type for evaluation.

Attributes:

accuracy (float): Overall accuracy

balanced_accuracy (float): Mean recall across classes

fairness_disparity (float): Maximum difference in Equalized Odds between groups

group_metrics (pd.DataFrame): Accuracy and selection rates per demographic group

2. Core Wrapper Class
FairnessPredictor
Main entry point, encapsulating both standard and fairness-mitigated models.

__init__(config: ModelConfig = ModelConfig())
Initializes the predictor.

train(self, X, y, A=None, mitigate: bool = False)
Trains the internal model.

Parameters:

X (pd.DataFrame): Feature matrix

y (pd.Series): Target

A (pd.Series, optional): Sensitive attribute (required if mitigate=True)

mitigate (bool): If True, applies fairness mitigation

predict(self, X)
Generates predictions.

Returns: np.ndarray of labels

evaluate(self, X, y, A) -> EvaluationResult
Evaluates performance and fairness metrics and returns an EvaluationResult.

3. Utility Functions
load_chicago_data(local_cache_path)
Data loader with caching and feature engineering.

Functionality:

Checks for a local CSV cache

If missing, fetches 2020–2023 crime data via Chicago SODA API

Performs temporal splitting

Simulates demographics (Race + Income)

Returns:
(X, y, A, dates)

FairnessPP Example: Mitigating Bias in Crime Prediction

1. Problem Statement
Predictive policing algorithms may reinforce historical bias. Models trained on arrest data may over-police specific demographic groups because they are historically overrepresented.

This example demonstrates how to:

Diagnose bias

Mitigate using Equalized Odds

Evaluate fairness vs accuracy


2. Workflow Overview
Temporal validation simulates real deployment:

Training: 2020–2022

Testing: 2023

Models compared:

Baseline: Standard Gradient Boosting

Fair: Same model wrapped with mitigation

3. Usage Example
python
Copy code
from FairnessPP_utils import FairnessPredictor, load_chicago_data

# Load data
X, y, A, dates = load_chicago_data()

# Train baseline
baseline = FairnessPredictor()
baseline.train(X_train, y_train, mitigate=False)

# Train fair model
fair_model = FairnessPredictor()
fair_model.train(X_train, y_train, A=A_train, mitigate=True)

# Evaluate
res_base = baseline.evaluate(X_test, y_test, A_test)
res_fair = fair_model.evaluate(X_test, y_test, A_test)
4. Results & Analysis
Example results on 2023 test set:

Model	                       Accuracy	    Balanced Accuracy	Fairness Disparity
Baseline (Unmitigated)	        47.4%	       47.3%	          0.3939
Fair (Mitigated)	            46.6%	       46.6%	          0.2241

Key Findings
Reduction in Bias
Disparity dropped ~ 43% (0.39 → 0.22)

Error rates across demographic groups became more consistent

Minimal Performance Cost
Less than 1% loss in accuracy

Conclusion
FairnessPP shows that fairness constraints can be applied in policing contexts without catastrophic loss in predictive utility.

The tool simplifies:

Model training

Fairness evaluation

Deployment for public sector applications