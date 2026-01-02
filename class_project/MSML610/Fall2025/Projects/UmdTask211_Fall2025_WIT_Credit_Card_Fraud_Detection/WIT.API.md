# Credit Card Fraud Detection API

- Native API for the Credit Card Fraud Detection project.
- Demonstrates how the reusable functions in `WIT_utils.py` load data, preprocess features, train anomaly and supervised models, and integrate WIT for inspection.
- Aligns with the executable notebook `WIT.API.ipynb`.

## Table of Contents

The markdown includes a Table of Contents for navigation.

<!-- toc -->

- [Credit Card Fraud Detection API](#credit-card-fraud-detection-api)
  * [Table of Contents](#table-of-contents)
  * [General Guidelines](#general-guidelines)
  * [Overview](#overview)
  * [API Functions](#api-functions)
  * [References](#references)
  * [Notes](#notes)

<!-- tocstop -->

## General Guidelines

- Use the restart-and-run-all notebook `WIT.API.ipynb` to see each function exercised in sequence.
- Heavy logic lives in `WIT_utils.py`; notebook cells stay minimal and illustrative.
- Keep the Load → Preprocess → Train → Evaluate → Visualize (WIT) flow intact.
- Dataset source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Overview

This API exposes reusable components for fraud detection, covering data preparation, anomaly detection, supervised learning, ensemble construction, evaluation, and WIT integration. It is designed to support production-grade workflows with leakage-safe preprocessing and clear evaluation.

## API Functions

| Function | Description | Input | Output |
| --- | --- | --- | --- |
| `load_raw_data(path, nrows=None)` | Load the credit card dataset from CSV. | `path`: str, `nrows`: optional int | `DataFrame` |
| `clean_data(df)` | Drop duplicates, fill numeric nulls, enforce `Class` int. | `df`: DataFrame | Cleaned `DataFrame` |
| `engineer_features(df)` | Add `Hour`, `Amount_log1p`, `Amount_per_hour`. | `df`: DataFrame | Feature-enhanced `DataFrame` |
| `split_features_target(df, target_col, test_size, random_state)` | Stratified train/test split. | `df`: DataFrame | `X_train`, `X_test`, `y_train`, `y_test` |
| `scale_features(X_train, X_test, scaler=None)` | Standardize features using train statistics only. | Train/Test DataFrames | Scaled DataFrames, `StandardScaler` |
| `balance_with_smote_tomek(X_train, y_train, random_state)` | Balance training data with SMOTE-Tomek. | Train DataFrame/Series | Resampled DataFrame/Series, sampler |
| `train_isolation_forest(X_train, contamination, n_estimators, random_state)` | Train Isolation Forest anomaly detector. | Scaled features | Trained model |
| `predict_isolation_forest(model, X)` | Predict labels and normalized anomaly scores. | Model, features | `(labels, scores)` |
| `train_autoencoder(X_train, y_train, ...)` | Train Keras autoencoder on normal class. | Arrays, labels | `(model, threshold, history)` |
| `predict_autoencoder(model, X, threshold)` | Anomaly labels and reconstruction errors. | Model, features, threshold | `(labels, errors)` |
| `train_supervised_models(X_train, y_train, random_state)` | Train LogReg, RandomForest, XGBoost, CatBoost with class weights. | Balanced features/labels | Dict of models |
| `build_soft_voting_ensemble(models)` | Soft-voting ensemble with weighted estimators. | Dict of models | `VotingClassifier` |
| `optimize_threshold(y_true, y_score, metric)` | Sweep thresholds to maximize F1 (default). | Labels, scores | `(best_threshold, stats)` |
| `evaluate_binary_classification(y_true, y_pred, y_score)` | Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix. | Labels, predictions, scores | Metrics dict |
| `collect_feature_importance(models, feature_names)` | Aggregate feature importances/coefficients. | Models, names | `DataFrame` |
| `build_predict_fn(model, feature_names)` | Wrap model into WIT-compatible predict function. | Model, names | Callable |
| `build_wit_widget(sample_df, feature_names, predict_fn, ...)` | Configure WIT for FP/FN and feature impact exploration. | Sampled DataFrame, predict fn | WIT widget |
| `save_processed(df, path)` | Persist processed dataset. | DataFrame, path | Saved CSV, `Path` |

## References

- Notebook: [`WIT.API.ipynb`](./WIT.API.ipynb)
- Utilities: [`WIT_utils.py`](./WIT_utils.py)
- Example walk-through: [`WIT.example.md`](./WIT.example.md) and [`WIT.example.ipynb`](./WIT.example.ipynb)
- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Notes

- Notebook sequence: Load → Preprocess → Train → Evaluate → Visualize (WIT).
- Comments and docstrings in `WIT_utils.py` explain non-obvious steps; notebooks keep code minimal.
- Threshold tuning is done on validation data; retrain/tune if class priors shift.
- WIT requires `witwidget`, `ipywidgets==7.*`, and TensorFlow; use the provided Docker image or `requirements.txt` for a ready environment.
