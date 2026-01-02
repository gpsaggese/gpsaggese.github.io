# AzuaHousing.API

**Purpose:** Stable interface for this project’s code and the thin wrapper used by notebooks and the FastAPI service.

## Public Python API

```python
from azua_utils import (
    load_melbourne,       # (csv_path: str) -> pd.DataFrame
    train_select_best,    # (df: pd.DataFrame) -> dict bundle
    save_artifacts,       # (bundle: dict, outdir: str='artifacts') -> None
    load_model            # (path: str='artifacts/model.joblib') -> sklearn Pipeline
)
````

### `load_melbourne(csv_path: str) -> pd.DataFrame`

* Loads Melbourne dataset (works with `data/melb_data.csv`).
* Drops rows with missing `Price`.
* Parses `Date` (if present) and adds `Year`, `Month` numeric features.

### `train_select_best(df: pd.DataFrame) -> dict`

* Splits features into numeric vs categorical automatically.
* Preprocessing:

  * Numeric: `SimpleImputer(strategy="median")` → `StandardScaler`
  * Categorical: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`
* Models compared with 5-fold CV:

  * `"linreg"`: `LinearRegression()`
  * `"elastic"`: `ElasticNet(alpha=0.01, l1_ratio=0.1, random_state=42)`
  * `"rf"`: `RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)`
  * `"xgb"`: `XGBRegressor(n_estimators=600, max_depth=8, subsample=0.8, colsample_bytree=0.8, learning_rate=0.05, random_state=42, n_jobs=-1, objective="reg:squarederror")`
* Returns a bundle:

  ```python
  {
    "best": {"name": str, "rmse": float, "r2": float, "pipeline": sklearn.Pipeline},
    "all_results": [{"model": str, "rmse": float, "r2": float}, ...],
    "num_cols": List[str],
    "cat_cols": List[str],
  }
  ```

### `save_artifacts(bundle, outdir='artifacts')`

* Saves:

  * `artifacts/model.joblib` — fitted sklearn pipeline
  * `artifacts/metrics.json` — metadata with CV scores and schema (`num_cols`, `cat_cols`)

### `load_model(path='artifacts/model.joblib') -> Pipeline`

* Reloads the persisted pipeline for inference.

---

## REST API (FastAPI)

* `GET /` → health/usage
* `GET /schema` → returns training schema:

  ```json
  {"numeric": [...], "categorical": [...]}
  ```
* `POST /predict`
  Request body:

  ```json
  {"features": { "<feature>": <value>, ... }}
  ```

  Notes:

  * Missing expected columns are added as `NaN` and imputed by the pipeline.
  * If `Date` is provided, `Year` and `Month` are derived server-side to match training.

Response:

```json
{"predicted_price": 123456.78}
```

---

## Environment & Data

* Default dataset: `data/melb_data.csv` (Kaggle snapshot).

* Containerized workflow:

  * Build: `./docker_build.sh`
  * Train (detached):
    `docker run -d --name housing-train -v "$(pwd)":/app -e DATA_PATH=/app/data/melb_data.csv azua-housing:latest python train.py`
  * Serve (detached):
    `docker run -d --name housing-api -p 8000:8000 -v "$(pwd)":/app azua-housing:latest uvicorn serve:app --host 0.0.0.0 --port 8000`

