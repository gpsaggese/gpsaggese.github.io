# FairnessPP API Documentation

## Overview
**FairnessPP** (Fairness in Predictive Policing) is a high-level Python API designed to democratize access to fair machine learning algorithms for high-stakes public sector applications. 

The core philosophy is **Abstraction of Complexity**: users interact with a single `FairnessPredictor` object, while reweighting, reduction, and metric calculations are handled internally.

---

## 1. Configuration Objects

### `class ModelConfig`
A stable configuration container for model hyperparameters.

* **Parameters:**
    * `n_estimators` (int): Number of boosting stages. Default: `50`.
    * `random_state` (int): Random seed. Default: `42`.
    * `max_iter_mitigation` (int): Maximum iterations for mitigation. Default: `50`.

### `class EvaluationResult`
A structured return type for model evaluation.

* **Attributes:**
    * `accuracy` (float): Overall model accuracy (0.0 to 1.0).
    * `balanced_accuracy` (float): Mean recall across classes.
    * `fairness_disparity` (float): The maximum difference in Equalized Odds between groups.
    * `group_metrics` (pd.DataFrame): A detailed table of accuracy and selection rates per demographic group.

---

## 2. Core Wrapper Class

### `class FairnessPredictor`
The main entry point for the library.

#### `train(self, X, y, A=None, mitigate: bool = False)`
Trains the internal model.
* **Parameters:**
    * `X`: Feature matrix.
    * `y`: Target vector.
    * `A`: Sensitive attribute (required if `mitigate=True`).
    * `mitigate`: If `True`, applies Fairlearn constraints.

#### `evaluate(self, X, y, A) -> EvaluationResult`
Evaluates the model and returns performance/fairness metrics.

---

## 3. Utility Functions

### `load_chicago_data(local_cache_path)`
Fetches 2020–2023 crime data via the Chicago API, handles caching, and performs temporal splitting.