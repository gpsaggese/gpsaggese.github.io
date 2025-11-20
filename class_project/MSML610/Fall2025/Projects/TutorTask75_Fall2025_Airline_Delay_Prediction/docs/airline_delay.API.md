## Why this API exists

I trained three gradient-boosted tree models (XGBoost, LightGBM, CatBoost) using a unified schema.  
This API wraps the common tasks I need in notebooks and the Streamlit app:

- enforce a stable **schema** (columns & dtypes),
- **load** persisted models/metrics,
- **predict** P(delay) the same way across libraries,
- pick a consistent **operating threshold**,
- score a **row** or a **DataFrame**,
- assemble a **model leaderboard**.

All logic lives in `src/utils_model.py`. Notebooks and the app import only from this module.

---

## Schema

I standardized the training/serving schema to avoid “works on my laptop” issues:

- **Categorical**:  
  `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `ORIGIN_STATE`, `DEST_STATE`

- **Numeric (float32)**:  
  `DEPARTURE_DELAY`, `AIR_TIME`, `DISTANCE`,  
  `ORIGIN_LAT`, `ORIGIN_LON`, `DEST_LAT`, `DEST_LON`,  
  `temp`, `rhum`, `prcp`, `snow`, `wspd`, `pres`

> `is_delayed` may appear in datasets (0/1). It’s optional for scoring.

---

## Public API (functions)

### `coerce_schema(df: pd.DataFrame) -> pd.DataFrame`
Force the DataFrame to my serving dtypes (float32 for numeric, `category` for categorical).  
Missing or bad numerics become `NaN` which the tree libs can handle.

**Example**
```python
from src.utils_model import coerce_schema
X = coerce_schema(raw_df)
```

---

### `load_model(tag: str)`
Load a persisted model for a given tag (`"xgb_tuned"`, `"lgbm"`, `"cat"`).  
Returns the deserialized model object (xgboost Booster or sklearn-style estimator).

**Example**
```python
from src.utils_model import load_model
model = load_model("cat")
```

---

### `load_metrics(tag: str) -> dict`
Load metrics JSON for the given tag. Fields include:
- `roc_auc`, `pr_auc`, `f1_at_threshold`, `precision_at_threshold`, `recall_at_threshold`
- `log_loss` (or `valid_logloss` in baseline files)
- `threshold`, `best_iteration`, and snapshots for loss curves

**Example**
```python
from src.utils_model import load_metrics
m = load_metrics("lgbm")
print(m["roc_auc"], m["pr_auc"])
```

---

### `predict_proba(tag: str, model, X: pd.DataFrame) -> np.ndarray`
Library-aware prediction helper that returns **P(delay=1)**.  
- For XGBoost, builds a `DMatrix` with `enable_categorical=True` when available (falls back to codes).  
- For LightGBM/CatBoost sklearn wrappers, calls `predict_proba` and returns the positive class.

**Example**
```python
from src.utils_model import predict_proba, coerce_schema, load_model
model = load_model("cat")
p = predict_proba("cat", model, coerce_schema(X))
```

---

### `pick_threshold(metrics: dict, default=0.5) -> float`
Use the project’s chosen operating point if present (stored under `threshold`), else fall back to `default`.

**Example**
```python
from src.utils_model import pick_threshold, load_metrics
th = pick_threshold(load_metrics("xgb_tuned"), 0.5)
```

---

### `score_row(tag: str, model, row_dict: dict, threshold: float) -> dict`
Score one example and return a dict with:
- `proba_delay` (float), `pred_delay` (int), and the normalized row

**Example**
```python
from src.utils_model import score_row, load_model, pick_threshold, load_metrics
row = {"AIRLINE":"AA", "ORIGIN_AIRPORT":"JFK", "DESTINATION_AIRPORT":"LAX", ...}
model = load_model("cat")
th = pick_threshold(load_metrics("cat"))
out = score_row("cat", model, row, th)
```

---

### `score_dataframe(tag: str, model, df: pd.DataFrame, threshold: float) -> pd.DataFrame`
Batch scoring. Returns `df` with two new columns: `proba_delay` and `pred_delay`.

**Example**
```python
from src.utils_model import score_dataframe
scored = score_dataframe("lgbm", model, X, th)
```

---

### `load_all_metrics_table(tags: list[str]) -> pd.DataFrame`
Build the model “leaderboard” by reading all saved metrics.

**Example**
```python
from src.utils_model import load_all_metrics_table
tbl = load_all_metrics_table(["cat","lgbm","xgb_tuned"])
```

---

## Design Notes

- **Determinism & Dtypes**: everything flows through `coerce_schema` so I don’t get dtype drift across tools.
- **Interchangeability**: same feature columns work across XGBoost/LightGBM/CatBoost.
- **Operating point**: thresholds chosen on validation are stored with metrics; prediction uses that by default.
- **Artifacts**: I treat the `models/` folder as a contract (model.joblib, metrics.json, plots).
- **Failure modes**: missing files → empty dicts or `None` with clear messages instead of hard crashes.

That’s it. Consumers only need to import `src.utils_model` and they’re safe.
