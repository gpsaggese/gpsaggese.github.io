````markdown
# Auto-Sklearn: Automated Machine Learning Tool

**Tool Tag:** `auto-sklearn`
**Documentation:** [Official Auto-Sklearn Documentation](https://automl.github.io/auto-sklearn/master/)

---

## 1. Tool Overview

**Auto-Sklearn** is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator. It frees the user from algorithm selection and hyperparameter tuning. It leverages recent advantages in Bayesian optimization, meta-learning, and ensemble construction to automatically construct the best possible machine learning pipeline for a given dataset.

### Key Features
* **Automated Algorithm Selection:** Automatically selects the best classifier/regressor (e.g., Random Forest, SVM, GBM) for the data.
* **Hyperparameter Tuning:** Optimizes hyperparameters for both the model and the preprocessing steps.
* **Ensemble Building:** Instead of picking just one best model, it builds an ensemble of the top performing models to improve robustness.
* **Scikit-Learn Compatible:** It implements the standard `fit()`, `predict()`, and `score()` methods.

---

## 2. Installation & Requirements

Auto-Sklearn requires a Linux environment (or Docker on Mac/Windows) and Python >= 3.7. It relies heavily on `swig` for building dependencies.

### Standard Installation
```bash
pip install auto-sklearn
````

### System Dependencies (Ubuntu/Debian)

```bash
sudo apt-get install build-essential swig
```

*Note: On Apple Silicon (M1/M2/M3), it is highly recommended to run this inside a Docker container (standard `python:3.9-slim`) to avoid architecture conflicts with dependencies like `pyrfr`.*

-----

## 3\. Core API Components

The library provides two primary classes that mirror standard scikit-learn interfaces:

### `AutoSklearnClassifier`

Used for classification tasks (binary or multi-class).

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `time_left_for_this_task` | Total time limit (in seconds) for the search. | 3600 |
| `per_run_time_limit` | Time limit (in seconds) for a single model call. | 360 |
| `ensemble_size` | Number of models added to the final ensemble. | 50 |
| `n_jobs` | Number of parallel jobs. (Use `1` for stability on some systems). | 1 |

### `AutoSklearnRegressor`

Used for regression tasks (predicting continuous values). It accepts the same key parameters as the classifier.

-----

## 4\. Usage Example (Generic)

The following example demonstrates how to use `AutoSklearnClassifier` on a standard dataset. This code is generic and can be adapted to any tabular dataset.

### Step 1: Import and Prepare Data

```python
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

# Load a standard dataset (e.g., Digits)
X, y = sklearn.datasets.load_digits(return_X_y=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)
```

### Step 2: Initialize and Train

We constrain the search to 2 minutes (`120` seconds) for this example.

```python
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    n_jobs=1,  # Set to 1 for safety in Docker environments
)

# The .fit() method triggers the AutoML search process
automl.fit(X_train, y_train)
```

### Step 3: Inspect the Results

Auto-Sklearn provides a method to view the statistics of the search process.

```python
# Print the "Leaderboard" (Top models found)
print(automl.leaderboard())

# Print detailed statistics
print(automl.sprint_statistics())
```

### Step 4: Predict

The final object acts exactly like a standard scikit-learn model.

```python
predictions = automl.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_test, predictions)

print(f"Final Accuracy: {accuracy:.4f}")
```

-----

## 5\. Why use this tool?

1.  **Baseline Creation:** It creates a very strong baseline with minimal code, allowing data scientists to see "what is possible" on a dataset before manually tuning models.
2.  **Time Efficiency:** It automates the tedious trial-and-error phase of model selection.
3.  **Pipeline Optimization:** It doesn't just tune the model; it tunes the *preprocessing* (imputation, scaling, encoding) as part of the pipeline.

<!-- end list -->

```
```