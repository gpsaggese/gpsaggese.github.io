# Employee Attrition Prediction with LIME — End-to-End Example 

This document is the narrative to `lime_attrition.example.ipynb`. It presents a complete, end-to-end application of the project’s API layer (`lime_attrition_utils.py`) for:

- Exploring attrition trends (EDA),
- Training and evaluating gradient-boosting models,
- Generating **LIME** explanations for individual predictions,
- Aggregating LIME explanations into HR-friendly “global-ish” drivers,
- stress-testing explanation stability via feature subset experiments.

---

## 1) Dataset

**Dataset**: IBM HR Employee Attrition (commonly distributed via Kaggle). Each row represents an employee record containing:

- **Demographics** (e.g., `Age`, `Gender`, `MaritalStatus`)
- **Role & work context** (e.g., `JobRole`, `Department`, `OverTime`, `BusinessTravel`)
- **Compensation & tenure** (e.g., `MonthlyIncome`, `TotalWorkingYears`, `YearsAtCompany`)
- **Ratings** (e.g., `JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`)

**Target**: `Attrition` is the label indicating whether the employee left.

**How the dataset is used in this project**
- ID/constant columns (e.g., `EmployeeNumber`, `EmployeeCount`, `StandardHours`) are dropped via `AttritionDataConfig.id_columns`.
- Remaining columns are treated as features.
- Mixed-type preprocessing (numeric + categorical) is applied using a fitted pipeline (fit on train only).

---

## 2) API-layer usage

```python
from pathlib import Path
import lime_attrition_utils as u

# 1) Locate dataset CSV in ./data
csv_path = sorted(Path("data").glob("*.csv"))[0]

# 2) Load + clean
cfg = u.AttritionDataConfig()
df_raw = u.load_raw_attrition_data(str(csv_path))
df = u.clean_attrition_data(df_raw, cfg)

# 3) Split into X/y
X, y = u.split_features_target(df, cfg)

# 4) Train/test split
X_train, X_test, y_train, y_test = u.train_test_split_attrition(X, y, cfg)

# 5) Build preprocessing + train models
pre = u.build_preprocessor(X_train)
model_cfg = u.ModelConfig(use_xgboost=True, use_lightgbm=True, use_random_forest=True)
models = u.train_attrition_models(X_train, y_train, pre, model_cfg)

# 6) Evaluate and pick best
metrics = u.evaluate_models(models, X_test, y_test)
```

---

## 3) EDA: Attrition trends and Correlations

The notebook performs lightweight but representative EDA:

- overall attrition rate,
- attrition rate by key categorical features (overtime, job role, department),
- numeric distributions by attrition status,
- correlation checks among numeric variables.

Key utilities used:
- `u.compute_attrition_rate`
- `u.categorical_attrition_table`
- `u.numeric_summary_by_attrition`

The point of EDA is descriptive understanding and hypothesis generation not causal inference.

---

## 4) Feature engineering

The example adds a small set of engineered features (HR-style signals), such as:
- income normalized by tenure,
- tenure ratios,
- commute distance bucket features.

In the notebook these are implemented as a deterministic function prior to splitting. In a production setting, you would place the same logic in a dedicated transform step to ensure consistency.

---

## 5) Model training and evaluation

The project trains multiple non-linear models:

- XGBoost
- LightGBM
- Random Forest (strong non-linear baseline)

Evaluation focuses on:
- **PR AUC / Average Precision** (important for imbalanced positive class),
- ROC AUC,
- F1 and accuracy as secondary summaries.

The notebook also includes PR/ROC curves for the selected best model.

---

## 6) Error analysis

Beyond AUC-style metrics, HR workflows require a classification **threshold**. The example notebook:

- computes confusion matrices at threshold 0.50,
- computes a “best F1” illustrative threshold,
- inspects false positives and false negatives to understand common failure modes.

This provides concrete evidence that you understand and can defend model behavior in a review.

---

## 7) LIME explanations

### 7.1 Native LIME object model used via wrapper
LIME explanations are generated using:
- `LimeTabularExplainer(...)`
- `explain_instance(x, predict_fn=...)`

In this project, the wrapper layer enforces an important constraint: the LIME `predict_fn` must operate in the **same feature space** as the trained model including preprocessing.

### 7.2 Single-employee explanations
The notebook demonstrates explanations for:
- the highest-risk test employee,
- one predicted **LEAVE** vs one predicted **STAY** example,
- one false positive and one false negative (LIME explanations on mistakes).

Wrapper functions used:
- `u.build_lime_explainer(...)`
- `u.explain_single_employee(...)`

### 7.3 Batch explanations + aggregation
To move from “individual explanations” to HR-friendly insights, the project:
- generates LIME explanations for a batch of high-risk employees,
- aggregates LIME weights to identify consistently influential features.

Wrapper functions used:
- `u.batch_lime_explanations(...)`
- `u.aggregate_lime_features(...)`
- `u.plot_lime_aggregate_bar(...)` 

---

## 8) Bonus: Feature Subset Experiments

To test explanation stability, the notebook trains additional models on feature subsets of demographics-only, compensation/tenure-only and compares LIME top-features for the **same employee** across models.

This satisfies the project requirement to “test different feature subsets and evaluate how explanations change,” and demonstrates depth beyond a basic tutorial pipeline.

---

## 9) Limitations and practical considerations

- LIME explanations are local and can vary with sampling; increasing `num_samples` improves stability at higher computational cost.
- Correlations observed in EDA are not causal.
- Any deployment should consider data drift, fairness, and threshold policies aligned to HR constraints.

---

## 10) Artifacts

The notebook optionally saves LIME explanations as HTML under `./artifacts/` so they can be linked from reports or README.
