# Azua Tool API (`azua_utils.py`)

This document describes the **Azua internal Python API** implemented in `azua_utils.py`.  
It is written as **tool documentation** and is independent of any project-specific dataset.

---

## What the tool does

`azua_utils.py` provides a lightweight AutoML-style workflow for **tabular regression**:

- Automatic feature typing (numeric vs categorical)
- Automated preprocessing:
  - numeric: median imputation + standard scaling
  - categorical: most-frequent imputation + one-hot encoding
- Model selection across multiple regressors using **K-fold cross-validation**
- Evaluation with standard metrics: **RMSE** and **R²**
- Artifact export (`model.joblib`, `metrics.json`, `cv_results.csv`) for reuse and deployment
- Deployment-ready inference because preprocessing is embedded in the saved pipeline

The API is dataset-agnostic. Callers supply:
- a pandas DataFrame
- the target column name (default is `"Price"`)
- optional columns to drop from features

---

## Installation

Dependencies are managed via `requirements.txt`.

Typical setup:
- `pip install -r requirements.txt`

---

## Public API

### `load_csv(csv_path, target="Price", date_cols=None, derive_date_parts=False, drop_cols=None) -> pd.DataFrame`

Loads tabular data from a CSV file and optionally performs light feature derivation for date columns.

- Drops rows with missing target (if `target` exists in the CSV)
- If `derive_date_parts=True`, derives `<col>_Year` and `<col>_Month` for each column in `date_cols`
- Optionally drops columns via `drop_cols`

Use this when you want a standard CSV ingestion path.

---

### `split_features(df, target="Price", drop_cols=None) -> (num_cols, cat_cols)`

Automatically detects:
- numeric feature columns (based on pandas numeric dtypes)
- categorical feature columns (everything else)

It excludes:
- the `target` column
- any columns listed in `drop_cols`

---

### `build_preprocessor(num_cols, cat_cols) -> ColumnTransformer`

Builds a preprocessing transformer for mixed-type tabular data:

- numeric pipeline: `SimpleImputer(median)` → `StandardScaler`
- categorical pipeline: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`

This transformer is intended to be used inside a sklearn `Pipeline`.

---

### `candidate_models(random_state=42) -> Dict[str, estimator]`

Returns the default model search space used by the tool.

**Included models**
- `linreg`: Linear Regression (baseline)
- `elastic`: ElasticNet (regularized linear model)
- `rf`: Random Forest Regressor
- `xgb`: XGBoost Regressor

---

### `cv_scores(pipe, X, y, folds=5, random_state=42, fold_progress=False) -> (rmse, r2)`

Evaluates a full sklearn pipeline (preprocessing + model) using K-fold cross-validation and returns:

- mean RMSE
- mean R²

Use this when you want to evaluate a single pipeline without running the full model selection loop.

---

### `train_select_best(df, target="Price", folds=5, random_state=42, drop_cols=None, progress=True, fold_progress=True, return_folds=False, models=None) -> Dict[str, Any]`

Runs AutoML-style model selection:

1) Split `df` into features `X` and target `y`  
2) Detect numeric/categorical columns  
3) Build preprocessing transformer  
4) Train/evaluate candidate pipelines using K-fold CV  
5) Select best model by **lowest RMSE**  
6) Fit the best pipeline on all provided data  
7) Return a bundle containing results and the fitted pipeline

Returned bundle includes:
- `bundle["best"]`: `{name, rmse, r2, pipeline}`
- `bundle["all_results"]`: list of per-model CV results (includes std; optionally per-fold lists)
- `bundle["num_cols"]`, `bundle["cat_cols"]`, `bundle["drop_cols"]`, `bundle["folds"]`, `bundle["random_state"]`, `bundle["target"]`

Progress bars:
- `progress=True` shows progress across models
- `fold_progress=True` shows progress across folds (per model)

---

### `save_artifacts(best_bundle, outdir="artifacts") -> None`

Persists deployment-ready artifacts:

- `model.joblib`: fitted sklearn Pipeline (preprocessing + model)
- `metrics.json`: summary metadata (best model name, RMSE/R², columns, config)
- `cv_results.csv`: per-model CV results table (best-effort)

---

### `load_model(path="artifacts/model.joblib") -> Pipeline`

Loads the serialized pipeline for inference or deployment.

---

## Typical usage pattern (end-to-end)

```python
import pandas as pd
from azua_utils import train_select_best, save_artifacts, load_model

# df must contain features and a numeric target column (default name: "Price")
bundle = train_select_best(df, target="Price", folds=5, progress=True, fold_progress=True)

# Inspect model comparison
results = pd.DataFrame(bundle["all_results"]).sort_values("rmse")
print(results)

# Save artifacts
save_artifacts(bundle, outdir="artifacts_demo")

# Load for inference
pipe = load_model("artifacts_demo/model.joblib")
pred = pipe.predict(df.drop(columns=["Price"]).head(1))
print(pred)
````

