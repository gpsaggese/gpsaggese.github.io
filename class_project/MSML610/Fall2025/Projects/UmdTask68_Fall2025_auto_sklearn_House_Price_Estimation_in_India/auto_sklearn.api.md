## Auto-sklearn API (Native interface)

This document explains the **native programming interface** used from auto-sklearn in `auto_sklearn.api.ipynb`.

> Setup/Docker instructions are intentionally not duplicated here; see `README.md`.

---

## What auto-sklearn is doing (conceptual model)

auto-sklearn is an AutoML library built around scikit-learn.

- **You provide**: feature matrix `X`, target `y`, and a time budget.
- **It performs**: automated model selection + hyperparameter optimization (via SMAC) + optional ensemble building.
- **It returns**: a fitted estimator that behaves like a scikit-learn model (`fit`, `predict`, `score`).

In practice, auto-sklearn explores many candidate pipelines (preprocessing + estimator + hyperparameters), evaluates them, and keeps the best models and (optionally) an ensemble of them.

---

## Native auto-sklearn interface used in this repo

### Core classes

- **`autosklearn.classification.AutoSklearnClassifier`**

  - For classification tasks.
  - Implements the familiar sklearn interface: `fit(X, y)`, `predict(X)`, `score(X, y)`.

- **`autosklearn.regression.AutoSklearnRegressor`**
  - For regression tasks.
  - Same sklearn-style API.

### Key configuration parameters (used throughout the notebook)

- **`time_left_for_this_task`**

  - Total wall-clock budget (seconds) for the AutoML search.

- **`per_run_time_limit`**

  - Budget (seconds) for each individual model evaluation.

- **`seed`**

  - Reproducibility (SMAC + internal randomness).

- **`metric`** (classification example)

  - Allows swapping the optimization objective from default accuracy to something else (e.g., macro F1).

- **`include={...}`**

  - Restricts which pipeline components are allowed.
  - In the notebook this is used to limit the search space (faster, more controlled runs).

- **`ensemble_kwargs={...}`**
  - Controls how the final ensemble is built (e.g., `ensemble_size`).

### Model inspection / reporting methods

- **`sprint_statistics()`**

  - Prints a compact training summary (runs, best score, etc.).

- **`leaderboard()`**
  - Shows a ranked table of models that contributed to the final solution / ensemble.

### Persistence

- auto-sklearn models can be saved and loaded with **`joblib.dump(...)`** / **`joblib.load(...)`**.

---

## Notebook walkthrough: `auto_sklearn.api.ipynb` (what each cell does)

This notebook is intentionally a set of **minimal native API usage patterns**, each centered on one concept.

### Cell 0 — Notebook introduction (Markdown)

- **Purpose**: describes the intent: quick snippets demonstrating common auto-sklearn patterns.

### Cells 1–2 — Basic classification with train/test split

- **Purpose**: show the smallest end-to-end classification run.
- **What happens**:
  - loads a toy dataset
  - splits into train/test
  - fits `AutoSklearnClassifier` under a small time budget
  - evaluates via `score(...)`
- **Key native APIs**: `AutoSklearnClassifier(...)`, `fit`, `score`.
- **Design note**: short budgets are used to keep runtime reasonable, but can lead to variability across runs.

### Cells 3–4 — Basic regression with cross-validation

- **Purpose**: demonstrate the regression counterpart and how it integrates with sklearn evaluation utilities.
- **What happens**:
  - loads a regression dataset
  - fits `AutoSklearnRegressor`
  - uses `cross_val_score(...)` to compute an average R² score
- **Key native APIs**: `AutoSklearnRegressor(...)`, `fit`, sklearn `cross_val_score`.
- **Common observation**: with short time budgets, you may see messages that the ensemble builder had too few successful runs.

### Cells 5–6 — Custom metric (macro F1)

- **Purpose**: show how to change the optimization target from default accuracy to macro F1.
- **What happens**:
  - trains `AutoSklearnClassifier(metric=f1_macro, ...)`
  - predicts labels on a held-out test set
  - computes macro F1 explicitly with sklearn
- **Key native APIs**: `autosklearn.metrics.f1_macro`, `metric=...` parameter.
- **Design note**: macro F1 is useful when you want balanced performance across classes.

### Cells 7–8 — Restrict estimators and preprocessors

- **Purpose**: demonstrate controlled searches by limiting what auto-sklearn is allowed to try.
- **What happens**:
  - configures `include={...}` to restrict classifier choices (e.g., random forest, gradient boosting)
  - restricts feature preprocessing
  - prints `sprint_statistics()`
- **Key native APIs**: `include`, `sprint_statistics()`.
- **Why this matters**:
  - smaller search spaces reduce runtime and noise
  - makes experiments more interpretable

### Cells 9–10 — Categorical features via `ColumnTransformer`

- **Purpose**: show how to handle mixed numeric/categorical features with sklearn preprocessing.
- **What happens**:
  - builds a toy DataFrame with a categorical `city` column
  - encodes it using `OneHotEncoder` inside a `ColumnTransformer`
  - trains `AutoSklearnClassifier` on the transformed matrix
- **Key APIs**: sklearn `ColumnTransformer`, `OneHotEncoder`, auto-sklearn `fit`/`score`.

### Cells 11–12 — Save and load a fitted model

- **Purpose**: demonstrate persistence of a trained auto-sklearn model.
- **What happens**:
  - trains a classifier
  - saves it with `joblib.dump`
  - reloads it with `joblib.load`
  - verifies it still scores correctly
- **Key APIs**: `joblib.dump`, `joblib.load`.

### Cells 13–14 — Inspect ensemble leaderboard

- **Purpose**: show how to inspect what models contributed to the final solution.
- **What happens**:
  - fits a classifier with a non-default `ensemble_kwargs` (larger ensemble)
  - prints `leaderboard()`
- **Key APIs**: `ensemble_kwargs`, `leaderboard()`.
