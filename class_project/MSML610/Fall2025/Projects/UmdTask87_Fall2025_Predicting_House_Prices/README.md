# Predicting House Prices — MSML610 (Fall 2025)

Melbourne housing price regression (Kaggle snapshot). This project includes:
- preprocessing (imputation + scaling + one-hot encoding)
- model comparison with cross-validation (RMSE, R²)
- artifact export (`artifacts/model.joblib`, `artifacts/metrics.json`)
- demo notebook for inference and evaluation

## Quickstart (Docker + Jupyter)

### Build
```bash
./docker_build.sh
````

### Start Jupyter

```bash
./docker_jupyter.sh
```

## Notebooks (Run in this order)

1. **Training + model selection + artifact export**
   `AzuaHousing.API.ipynb`

2. **End-to-end demo (loads artifacts, predicts, evaluates)**
   `AzuaHousing.example.ipynb`

## Data

Default dataset path: `data/melb_data.csv` (Melbourne Housing Snapshot).
You can override with environment variable `DATA_PATH` when launching the container.

````

---

## 7b) File: `AzuaHousing.API.md`
Path:
`class_project/MSML610/Fall2025/Projects/UmdTask87_Fall2025_Predicting_House_Prices/AzuaHousing.API.md`

Replace entire content with:

```md
# AzuaHousing.API

This document describes the internal Python API used by this project (implemented in `azua_utils.py`)
and how it is exercised in `AzuaHousing.API.ipynb`.

There is **no REST API / FastAPI deployment** in this submission.

## Public Python API

```python
from azua_utils import (
    load_melbourne,
    train_select_best,
    save_artifacts,
    load_model,
)
````

### `load_melbourne(csv_path: str) -> pd.DataFrame`

* Loads the Melbourne dataset (Kaggle snapshot or full CSV).
* Drops rows with missing `Price`.
* If `Date` exists, parses it and adds `Year` and `Month`.

### `train_select_best(df: pd.DataFrame) -> dict`

* Builds a preprocessing + model pipeline:

  * Numeric: median impute → standardize
  * Categorical: most-frequent impute → one-hot encode (`handle_unknown="ignore"`)
* Trains multiple regression models and compares them with k-fold CV.
* Selects the best model by RMSE (also tracks R²).
* Returns a bundle containing:

  * best model name + fitted pipeline
  * CV metrics table
  * feature schema (`num_cols`, `cat_cols`)

### `save_artifacts(bundle, outdir='artifacts')`

Saves:

* `artifacts/model.joblib` — fitted sklearn pipeline
* `artifacts/metrics.json` — CV metrics and schema (numeric/categorical columns)

### `load_model(path='artifacts/model.joblib')`

Loads the fitted pipeline for inference.

## Where training is demonstrated

* Training + evaluation + artifact export is in `AzuaHousing.API.ipynb`.

````

---

## 7c) File: `AzuaHousing.example.md`
Path:
`class_project/MSML610/Fall2025/Projects/UmdTask87_Fall2025_Predicting_House_Prices/AzuaHousing.example.md`

Replace entire content with:

```md
# AzuaHousing.example

This is an end-to-end usage demo implemented in `AzuaHousing.example.ipynb`.

## What it demonstrates
- Loading the dataset (`data/melb_data.csv`)
- Loading an exported model artifact (`artifacts/model.joblib`)
- Running predictions and evaluating RMSE / R²
- Showing a small table of predicted vs actual prices

If artifacts are missing, the notebook trains quickly to create them so the notebook is runnable
top-to-bottom via **Restart & Run All**.

## Run (Docker + Jupyter)
```bash
./docker_build.sh
./docker_jupyter.sh
````

Recommended order:

1. Run `AzuaHousing.API.ipynb` (full training + artifact export)
2. Run `AzuaHousing.example.ipynb` (demo inference + evaluation)

````

