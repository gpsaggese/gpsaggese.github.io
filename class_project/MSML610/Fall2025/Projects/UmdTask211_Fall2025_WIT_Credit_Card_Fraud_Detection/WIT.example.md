# Credit Card Fraud Detection Example

- End-to-end implementation of the credit card fraud detection system using the reusable API functions defined in `WIT_utils.py`, covering anomaly detectors, supervised learners, ensembles, and WIT-based analysis.

## Table of Contents
<!-- toc -->

- [Credit Card Fraud Detection Example](#credit-card-fraud-detection-example)
  * [Table of Contents](#table-of-contents)
  * [General Guidelines](#general-guidelines)
  * [Overview](#overview)
  * [Workflow](#workflow)
  * [Architecture](#architecture)
  * [Dataset](#dataset)
  * [Model Details](#model-details)
  * [Results](#results)
  * [References](#references)

<!-- tocstop -->

## General Guidelines

- Notebook: `WIT.example.ipynb` is restart-and-run-all clean; no manual edits required.
- Heavy lifting is in `WIT_utils.py`; notebook cells stay concise and call these functions.
- Keep the default random seed (42) unless you need variability; changing it alters metrics slightly.
- Use the provided Docker image or `requirements.txt` to ensure TensorFlow, XGBoost, CatBoost, and WIT dependencies are consistent.

## Overview

This markdown documents the end-to-end example notebook (`WIT.example.ipynb`) for the **Credit Card Fraud Detection** project. It demonstrates how the reusable API functions load data, engineer features, balance classes, train anomaly and supervised models, build ensembles, evaluate performance, and interpret results with the What-If Tool (WIT).

## Workflow

The example notebook follows a clear, reproducible sequence:
1. Load the dataset with `load_raw_data()` and clean it via `clean_data()`.
2. Engineer features (`Hour`, `Amount_log1p`, `Amount_per_hour`) using `engineer_features()`.
3. Split into train/validation/test with `split_features_target()`.
4. Scale features (`scale_features()`) and balance the training set with SMOTE-Tomek (`balance_with_smote_tomek()`).
5. Train anomaly models: `train_isolation_forest()` and `train_autoencoder()`.
6. Train supervised models with class weighting: `train_supervised_models()` (LogReg, RandomForest, XGBoost, CatBoost).
7. Build a soft-voting ensemble (`build_soft_voting_ensemble()`), tune threshold (`optimize_threshold()`), and optionally fuse anomaly scores with ensemble probabilities.
8. Evaluate with `evaluate_binary_classification()` and visualize PR/ROC, confusion matrices, and score distributions.
9. Explore decisions and feature impact with WIT (`build_predict_fn()`, `build_wit_widget()`).
10. Persist artifacts (`save_processed()`, model dumps in `artifacts/`).

## Architecture

- **Data Layer:** `load_raw_data()`, `clean_data()`, `engineer_features()`, `split_features_target()`, `scale_features()`, `balance_with_smote_tomek()`.
- **Model Layer:** Anomaly detectors (Isolation Forest, autoencoder) plus supervised models (LogReg, RandomForest, XGBoost, CatBoost) and soft-voting ensemble.
- **Evaluation Layer:** Threshold tuning, precision/recall/F1, ROC-AUC, PR-AUC, confusion matrices, feature importance.
- **Interpretability Layer:** WIT widget for FP/FN exploration, feature sliders (Amount, Hour), and decision-boundary inspection.
- **Persistence Layer:** `save_processed()` for datasets and serialized models in `artifacts/`.

This separation keeps the workflow maintainable and extensible for new anomaly detection methods (e.g., one-class SVMs) or production scoring.

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions over two days
- **Fraud cases:** 492 (~0.17%)
- **Features:** 28 anonymized numerical features (V1–V28) plus `Time`, `Amount`, and target `Class`
- **Challenge:** Extreme imbalance and anonymized features; requires careful scaling, resampling, and metrics focused on precision/recall/F1.

## Model Details

- **Anomaly models:** Isolation Forest (contamination ~0.00172), autoencoder trained on normal class with reconstruction-error threshold at the 99.5th percentile.
- **Supervised models:** Class-weighted Logistic Regression, RandomForest, XGBoost (with `scale_pos_weight`), CatBoost.
- **Ensemble:** Soft voting with heavier weights on gradient boosting and RandomForest; threshold tuned on validation to maximize F1.
- **Hybrid fusion:** Blends Isolation Forest score, autoencoder error, and ensemble probability to capture both unsupervised and supervised signals.
- **Metrics:** Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices; threshold optimization for deployment-ready balance.

## Results

Representative metrics from the latest full run in `WIT.example.ipynb`:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| Isolation Forest | ~0.62 | ~0.69 | ~0.65 | ~0.95 | ~0.57 |
| Autoencoder | ~0.58 | ~0.63 | ~0.60 | ~0.94 | ~0.52 |
| Soft-voting ensemble | ~0.92 | ~0.76 | ~0.83 | ~0.97 | ~0.86 |
| Hybrid fusion (anomaly + ensemble) | ~0.92 | ~0.76 | **~0.83** | ~0.95 | ~0.80 |

Notes:
- Thresholds are tuned on validation data for F1; you can retune for higher recall or higher precision as needed.
- Results vary slightly with random seed and whether you run on the full dataset or a sampled subset.

## References

- Notebook: [`WIT.example.ipynb`](./WIT.example.ipynb)
- API Notebook: [`WIT.API.ipynb`](./WIT.API.ipynb)
- Utilities: [`WIT_utils.py`](./WIT_utils.py)
- API Documentation: [`WIT.API.md`](./WIT.API.md)
- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

End of Document
