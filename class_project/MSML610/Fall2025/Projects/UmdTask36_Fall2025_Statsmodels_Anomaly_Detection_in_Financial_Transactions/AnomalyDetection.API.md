---
# AnomalyDetection API Documentation
### Statsmodels GLM + Statistical Diagnostics + Anomaly Detection Utilities
---

## 1. Purpose

This document describes the internal Python API implemented in `AnomalyDetection_utils.py`.
Here, “API” refers to **reusable Python functions and their inputs/outputs**, not a web service.

The API is designed to support a statistically principled fraud and anomaly detection workflow,
combining supervised modeling, classical statistical diagnostics, and unsupervised anomaly detection.

The API supports:

- Loading and exploring the credit card fraud dataset
- Time-aware preprocessing with scaling and SMOTE (training data only)
- Fitting a Binomial Generalized Linear Model (GLM) using `statsmodels`
- Computing GLM residual and influence diagnostics for anomaly detection
- Evaluating supervised performance (precision, recall, F1, ROC, PR)
- Running an unsupervised baseline using Isolation Forest
- Performing a statistical test to assess whether anomaly flags are associated with fraud labels

For executable demonstrations of these utilities, see:
- `AnomalyDetection.API.ipynb`
- `AnomalyDetection.example.ipynb`

---

## 2. Module

**File:** `AnomalyDetection_utils.py`

This module contains all reusable components used across the project.
It is designed to separate modeling logic from notebook-specific analysis.

---

## 3. Dataset Assumptions

These utilities assume the **Credit Card Fraud Detection** dataset from Kaggle
(`creditcard.csv`).

Expected columns:

- `Class`: binary target  
  - `0` → legitimate transaction  
  - `1` → fraudulent transaction
- `Time`: seconds elapsed since the first transaction
- `Amount`: transaction amount
- `V1–V28`: PCA-transformed features used to preserve customer confidentiality

The dataset is extremely imbalanced (≈ 0.17% fraud), motivating the use of
anomaly detection techniques and precision–recall–focused evaluation metrics
instead of accuracy.

---

## 4. Exploratory Data Analysis Utilities

### 4.1 `load_data(path: str) -> pd.DataFrame`

Loads the credit card fraud dataset from disk.

**Inputs**
- Path to `creditcard.csv`

**Outputs**
- Pandas DataFrame containing all transactions

---

### 4.2 `basic_eda(df: pd.DataFrame) -> None`

Prints basic dataset diagnostics, including:

- dataset shape
- total missing value count
- summary statistics via `df.describe()`

This function is intended for quick sanity checks and exploratory inspection.

---

### 4.3 `class_distribution(df: pd.DataFrame) -> pd.Series`

Prints and plots the distribution of fraudulent and non-fraudulent transactions.

**Outputs**
- Pandas Series containing counts indexed by class label

This visualization highlights the severe class imbalance present in the data.

---

## 5. Data Preprocessing

### 5.1 `prepare_data(df: pd.DataFrame, scale_method: str = "standard")`

Implements a **time-aware preprocessing pipeline** suitable for fraud detection.

Processing steps:

1. Sorts transactions by `Time` to reduce temporal leakage
2. Splits data into training (80%) and test (20%) sets by chronological order
3. Scales features using `StandardScaler` or `RobustScaler`
4. Applies SMOTE **only to the training data**

**Inputs**
- DataFrame containing `Time` and `Class`
- Scaling method: `"standard"` or `"robust"`

**Outputs**
- SMOTE-resampled, scaled training features
- Scaled test features
- SMOTE-resampled training labels
- Original test labels
- Fitted scaler object
- Scaled training features before SMOTE
- Original training labels

**Note:**  
In `AnomalyDetection.API.ipynb`, this function is demonstrated primarily as an API reference.
The full end-to-end modeling and final evaluation using this pipeline are shown in
`AnomalyDetection.example.ipynb`.

---

## 6. Statistical Modeling (GLM)

### 6.1 `fit_glm_statsmodels(X_train, y_train)`

Fits a **Binomial Generalized Linear Model** with a logit link using `statsmodels`.

The GLM provides:

- predicted fraud probabilities
- access to classical statistical diagnostics
- a supervised baseline for comparison with anomaly detection methods

**Note:**  
In the API notebook, the GLM is fitted on a lightweight pipeline for demonstration purposes.
The final SMOTE-based training and evaluation workflow is presented in `AnomalyDetection.example.ipynb`.

---

### 6.2 `glm_predict_proba(glm_result, X)`

Generates predicted probabilities for the fraud class.

**Outputs**
- Array of probabilities in the range `[0, 1]`

---

### 6.3 `glm_predict_labels(glm_result, X, threshold: float = 0.5)`

Converts predicted probabilities into binary class labels using a specified threshold.

**Outputs**
- Binary predictions
- Corresponding probabilities

---

## 7. GLM Diagnostics for Statistical Anomaly Detection

### 7.1 `compute_glm_diagnostics(glm_result, X, y_true)`

Computes classical diagnostic measures used in GLMs to identify outliers
and influential observations, including:

- Deviance residuals
- Pearson residuals
- Standardized residuals
- Approximate leverage (hat values)
- Approximate Cook’s distance

These diagnostics provide **statistically interpretable anomaly signals**
beyond simple probability thresholds.

---

### 7.2 `flag_anomalies_from_diagnostics(...)`

Flags observations as anomalous using rule-based statistical thresholds on:

- standardized deviance residuals
- leverage
- Cook’s distance

An observation is flagged if **any** threshold is exceeded.

**Outputs**
- Boolean anomaly flags
- Dictionary of applied cutoff values

---

### 7.3 `evaluate_anomaly_flags(y_true, anomaly_flags)`

Evaluates anomaly flags as if they were fraud predictions.

**Outputs**
- True positives, false positives, false negatives, true negatives
- Precision and recall for the fraud class

**Important:**  
This function evaluates the **diagnostic rule itself**, not a calibrated classifier.
In the API notebook, this evaluation is shown as a functional demonstration; realistic
held-out evaluation is performed in `AnomalyDetection.example.ipynb`.

---

## 8. Supervised Evaluation

### 8.1 `evaluate_supervised(y_true, y_proba, threshold: float = 0.5)`

Computes supervised classification metrics and curve data:

- Precision, recall, and F1-score
- Confusion matrix
- ROC-AUC and PR-AUC
- Arrays for ROC and Precision–Recall curves

This function supports both threshold-based evaluation and ranking-based analysis.

---

### 8.2 Plotting Utilities

Matplotlib-based plotting helpers for visualizing model performance:

- Confusion matrix
- ROC curve
- Precision–Recall curve

---

## 9. Unsupervised Anomaly Detection (Isolation Forest)

### 9.1 `fit_isolation_forest(X, ...)`

Fits an Isolation Forest model on **training features without labels**.

This method isolates observations based on feature-space rarity rather than class labels.

---

### 9.2 `evaluate_isolation_forest(model, X, y_true)`

Evaluates Isolation Forest anomaly predictions against known fraud labels.

Outputs include:

- Precision, recall, and F1-score
- Confusion matrix
- ROC-AUC and PR-AUC
- Anomaly flags and raw anomaly scores

---

## 10. Statistical Test Utility

### 10.1 `two_proportion_ztest_flag_rate(y_true, flags, alternative="larger")`

Performs a **two-proportion z-test** to assess whether fraud transactions are flagged
at a higher rate than legitimate transactions.

**Null hypothesis:** Fraud and legitimate transactions are flagged at the same rate.  
**Alternative hypothesis:** Fraud transactions are flagged more frequently.

This test does not improve model performance; it provides a formal statistical check
that anomaly flagging behavior is associated with fraud labels rather than random chance.

---

## 11. Related Project Files

- `AnomalyDetection.API.ipynb` — executable API usage demonstrations
- `AnomalyDetection.example.ipynb` — full end-to-end workflow and final evaluation
- `AnomalyDetection.example.md` — narrative explanation of the workflow
- `README.md` — project setup, structure, and execution instructions
