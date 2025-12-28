````markdown
# Auto-Sklearn API Documentation

**Tool Tag:** `auto-sklearn`
**Type:** Automated Machine Learning (AutoML) for Regression
**Documentation:** [Official Auto-Sklearn Documentation](https://automl.github.io/auto-sklearn/master/)

---

## 1. Tool Overview

**Auto-Sklearn** is an automated machine learning toolkit that frees the user from algorithm selection and hyperparameter tuning. It leverages Bayesian optimization, meta-learning, and ensemble construction to automatically construct the best possible machine learning pipeline.

In this project, we utilize the **Regression** module (`AutoSklearnRegressor`) to predict continuous values (e.g., traffic volume).

---

## 2. Key Class: `AutoSklearnRegressor`

The primary class used in this project is `AutoSklearnRegressor`. It serves as a drop-in replacement for standard scikit-learn regressors.

### Initialization Parameters
| Parameter | Description | Recommended Setting (Project) |
| :--- | :--- | :--- |
| `time_left_for_this_task` | Total time limit (in seconds) for the search. | `900` (15 mins) |
| `per_run_time_limit` | Time limit (in seconds) for a single model call. | `30` |
| `ensemble_size` | Number of models added to the final ensemble. | `50` |
| `n_jobs` | Number of parallel jobs. | `1` (Required for stability on Mac M-series) |

---

## 3. Key Methods

The following methods are the core API endpoints used to train, evaluate, and inspect the models.

### `fit(X, y)`
Triggers the AutoML search process. It iteratively trains hundreds of models (SVM, Random Forest, GBM, etc.) and preprocessing pipelines within the `time_left_for_this_task` constraint.

### `predict(X)`
Generates predictions using the final **Ensemble** of the best models found during the search.
* **Input:** Feature matrix `X`.
* **Output:** Continuous values (Predicted Traffic Volume).

### `leaderboard()`
Returns a pandas DataFrame showing the top-performing models found during the search.
* **Usage:** Critical for verifying that the search converged and found valid models with high $R^2$ scores.

### `sprint_statistics()`
Returns a text summary of the search execution.
* **Usage:** Used to check how many models were trained, how many failed, and the best validation score achieved.

### `show_models()`
Returns the internal definition of the final ensemble.
* **Usage:** Allows "white-box" inspection to see exactly which algorithms (e.g., `random_forest`, `adaboost`) and hyperparameters were selected.

---

## 4. Usage Example (Generic Regression)

The following example demonstrates how to use `AutoSklearnRegressor` on a standard dataset (California Housing).

### Step 1: Import and Prepare Data
```python
import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

# Load a standard regression dataset
X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)
````

### Step 2: Initialize and Train

```python
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    n_jobs=1,
)

automl.fit(X_train, y_train)
```

### Step 3: Inspect and Predict

```python
print(automl.leaderboard())

predictions = automl.predict(X_test)
r2_score = sklearn.metrics.r2_score(y_test, predictions)

print(f"Final R2 Score: {r2_score:.4f}")
```

-----

## 5\. Installation Requirements

To ensure reproducibility on Apple Silicon (Mac M-series), specific version pinning is required in the `Dockerfile`.

```bash
# System Dependencies
apt-get install build-essential swig

# Python Dependencies
pip install auto-sklearn==0.15.0
pip install "numpy<2.0.0"  # Critical for compatibility
```

````
