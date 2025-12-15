# Anomaly Detection in Financial Transactions

### End-to-End Example Using Statistical Models and Machine Learning

---

## 1. Overview

This document provides an end-to-end explanation of the **fraud detection workflow**
implemented in `AnomalyDetection.example.ipynb`.

The purpose of this example is to show how **statistical modeling** and **anomaly detection**
can be combined to detect fraudulent financial transactions in a **highly imbalanced setting**.

This file is intended to be self-contained: a reader with no prior context should understand
what problem is being solved, what data is used, which methods are applied, and what the results imply.
The corresponding notebook (`AnomalyDetection.example.ipynb`) contains the full executable analysis.

---

## 2. Dataset Description

We use the **Credit Card Fraud Detection dataset** from Kaggle, which contains credit card
transactions made by European cardholders in September 2013.

Key characteristics:

- Total transactions: **284,807**
- Fraudulent transactions: **492 (≈ 0.17%)**
- Features:
  - `V1–V28`: anonymized principal components derived from the original transaction attributes
  - `Time`: seconds elapsed since the first recorded transaction
  - `Amount`: transaction amount
- Target:
  - `Class = 1` → fraudulent transaction
  - `Class = 0` → legitimate transaction

Because fraud cases are extremely rare, this dataset is not well suited for naïve accuracy-based
classification. Instead, it motivates **anomaly detection and ranking-based evaluation methods**.

---

## 3. Data Preparation

The data preparation pipeline mirrors a **realistic production workflow** used in fraud analytics:

1. Transactions are **sorted chronologically** using the `Time` variable to reduce information leakage.
2. The dataset is split into **training (80%)** and **testing (20%)** sets.
3. Feature scaling is applied using `StandardScaler`.
4. **SMOTE is applied only to the training data** to mitigate class imbalance during supervised model training.

The test set remains imbalanced to reflect real-world operating conditions, where fraud is rare
and false alarms are costly.

---

## 4. Statistical Modeling with GLM

A **Generalized Linear Model (GLM)** with a **Binomial family** and **logit link** is fitted using
the `statsmodels` library.

The GLM serves three purposes:

- Provides a **supervised probability baseline** for fraud prediction
- Enables access to **statistical diagnostics** not available in many machine-learning models
- Produces interpretable signals (residuals and influence measures) for anomaly detection

The model is trained on the **SMOTE-balanced training set** and evaluated on the original imbalanced test set.

---

## 5. Residual-Based Anomaly Detection

Rather than relying solely on predicted probabilities, this project uses **statistical diagnostics**
derived from the GLM to identify anomalous transactions.

Computed diagnostics include:

- **Standardized deviance residuals**
- **Leverage (hat values)**
- **Cook’s distance**

These quantities identify transactions that:

- Are poorly explained by the fitted model
- Exhibit unusual combinations of feature values
- Have disproportionate influence on model parameters

Such behavior is consistent with the notion of fraud as a **statistical outlier** rather than a typical
member of the transaction population.

---

## 6. Statistical Anomaly Flagging

Transactions are flagged as anomalous using **transparent, rule-based statistical thresholds**:

- Absolute standardized deviance residual > 3
- Leverage above the 99th percentile
- Cook’s distance above the 99th percentile

A transaction is flagged if it exceeds **any** of these criteria.
This approach mirrors classical diagnostic procedures used in statistical modeling,
quality control, and financial risk analysis.

Anomaly flags are evaluated against known fraud labels on the test set to assess detection behavior.

---

## 7. Supervised Model Evaluation

The GLM is also evaluated as a **supervised classifier** on the imbalanced test set.

Reported metrics include:

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve and ROC-AUC
- Precision–Recall curve and PR-AUC

Rather than optimizing a single metric, the analysis emphasizes how **threshold selection**
controls the trade-off between catching fraud (recall) and limiting false positives (precision).

**Note on thresholding in the notebook:** In addition to the default threshold (e.g., 0.5),
the notebook also demonstrates a **recall-constrained threshold selection** procedure to illustrate
how an operating point can be chosen under practical constraints (for example, ensuring recall is at
least a target value).

---

## 8. Unsupervised Anomaly Detection with Isolation Forest

To complement the statistical approach, an **Isolation Forest** model is applied as a fully
unsupervised anomaly detector.

Key characteristics:

- Trained **without fraud labels** on scaled training data
- Evaluated on the held-out test set
- Produces anomaly scores rather than calibrated probabilities

This method is particularly useful when labeled data is unavailable or delayed,
but it may flag a large number of legitimate transactions in highly imbalanced settings.

---

## 9. Statistical Test of Flagging Rates (Two-Proportion Z-Test)

To quantify whether anomaly flags are meaningfully associated with fraud labels, the notebook applies a
**two-proportion z-test** comparing:

- Flag rate among fraud transactions (`Class = 1`)
- Flag rate among legitimate transactions (`Class = 0`)

**Null hypothesis:** Fraud and legitimate transactions are flagged at the same rate.  
**Alternative (one-sided):** Fraud transactions are flagged at a higher rate.

This test is applied to both:
- Diagnostics-based flags from the GLM residual/influence measures
- Isolation Forest anomaly flags

Very small p-values provide evidence that flagged anomalies occur disproportionately in fraud transactions,
supporting that the anomaly detection signals are not random.

---

## 10. Comparative Interpretation

The results highlight complementary strengths:

- **GLM residual diagnostics (rule-based anomaly flags)**
  - High interpretability (residuals, leverage, influence)
  - In this run, diagnostics-based thresholding can produce **many false positives** on the imbalanced test set (low precision),
    while still identifying a subset of fraud cases (moderate recall).
  - Most useful as an explainable **triage signal** that can feed manual review or downstream decision logic.

- **Supervised GLM classification**
  - Strong ranking performance (ROC-AUC / PR-AUC)
  - Sensitive to decision threshold, and the notebook demonstrates both default evaluation and operational threshold selection.

- **Isolation Forest**
  - Higher recall in many settings
  - Lower precision, often flagging many legitimate transactions as anomalies in extremely imbalanced data
  - Effective at detecting unusual patterns without labels, but typically requires careful tuning and review workflows

These differences reflect the fundamental **precision–recall trade-off** inherent in fraud detection.

---

## 11. Conclusion

This example demonstrates that effective fraud detection benefits from
**multiple complementary modeling perspectives**:

- Statistical diagnostics provide transparent, explainable anomaly signals
- Supervised models enable probabilistic risk ranking
- Unsupervised models detect novel or unexpected behavior

Rather than optimizing for a single metric, the project emphasizes
**methodological correctness, interpretability, and realistic evaluation**—
key requirements in financial risk and fraud analytics.

---

## 12. Related Files

- **End-to-end analysis:** `AnomalyDetection.example.ipynb`
- **API reference:** `AnomalyDetection.API.ipynb`
- **Reusable utilities:** `AnomalyDetection_utils.py`
- **Project overview:** `README.md`
