# UmdTask87 — Tabular Regression AutoML Tool (Demo on House Prices)

This repository contains a lightweight **Azua tool** (`azua_utils.py`) plus a concrete **house-price prediction demo** using the Kaggle **Melbourne Housing Snapshot** dataset (`melb_data.csv`).

The project is notebook-driven:

* `AzuaHousing.API.ipynb` demonstrates the **tooling API** in `azua_utils.py`.
* `AzuaHousing.Example.ipynb` uses the tool to train/evaluate models on the Melbourne housing dataset and produces deployable artifacts.

---

## What’s included

* **`azua_utils.py`**: internal AutoML-style API for tabular regression

  * CSV loading (optional date-part derivation)
  * Mixed numeric/categorical preprocessing (impute + scale + one-hot)
  * K-fold cross-validation model comparison (RMSE, R²)
  * Train best model on full data
  * Save/load deployable sklearn pipelines + metadata artifacts

* **`AzuaHousing.API.ipynb`**: API/tool demonstration notebook

  * Shows how to call the functions/classes in `azua_utils.py`

* **`AzuaHousing.Example.ipynb`**: Melbourne housing end-to-end demo

  * Download dataset into `data/`
  * Load → compute stats → clean → compute stats
  * Cross-validation model comparison (RMSE, R²) with progress bars
  * Fit best model and evaluate on a held-out test split
  * Save artifacts to `artifacts/`
  * Reload artifacts and run inference + evaluation
  * Error analysis + visualization
  * Manual input prediction (edit a dict → predict)

* **`requirements.txt`**: dependencies (install once; no in-notebook installs)

---

## Repository structure

```text
.
├── AzuaHousing.API.ipynb
├── AzuaHousing.API.md
├── AzuaHousing.Example.ipynb
├── AzuaHousing.Example.md
├── azua_utils.py
├── requirements.txt
├── data/                 # created by API notebook
│   └── melb_data.csv
└── artifacts/            # created by API notebook
    ├── model.joblib
    ├── metrics.json
    ├── cv_results.csv
    ├── cv_rmse_by_model.png
    └── cv_r2_by_model.png
```

---

## Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the housing demo

Open and run all cells in:

* `AzuaHousing.Example.ipynb`

This will create:

* `data/melb_data.csv`
* `artifacts/` (trained pipeline + metadata + plots)

### 3) Run the API/tool demo notebook

Open and run all cells in:

* `AzuaHousing.API.ipynb`

This notebook is intended to demonstrate how to use `azua_utils.py` as a generic tabular regression tool.

---

## Dataset (Example notebook only)

The example notebook downloads the dataset via `kagglehub` and copies `melb_data.csv` into `./data`.

Default dataset path:

* `data/melb_data.csv`

Optional override:

```bash
export DATA_PATH=/path/to/melb_data.csv
```

---

## Internal tool API (`azua_utils.py`)

Key functions used across notebooks:

* `load_csv(csv_path, target="Price", date_cols=None, derive_date_parts=False, drop_cols=None) -> pd.DataFrame`
  Loads a CSV, optionally drops rows missing the target, and can derive date-part features.

  **Date-part naming convention** (matches the tool):
  For a date column named `Date`, the derived features are:

  * `Date_Year`
  * `Date_Month`

* `train_select_best(df, target="Price", folds=5, ...) -> dict`
  Runs K-fold CV across candidate regressors and selects the best by lowest CV RMSE. Fits the best pipeline on the full provided dataset.

* `save_artifacts(bundle, outdir="artifacts")`
  Saves:

  * `model.joblib` (sklearn `Pipeline`: preprocessing + model)
  * `metrics.json` (best model name + CV metrics + feature metadata)
  * `cv_results.csv` (all model CV summaries)

* `load_model("artifacts/model.joblib")`
  Loads the fitted sklearn pipeline for inference.

---

## Metrics and outputs

The demo tracks:

* **RMSE** (lower is better)
* **R²** (higher is better)
* fold variability: `rmse_std`, `r2_std`

Artifacts written to `artifacts/` include:

* `model.joblib`: deployable preprocessing + model pipeline
* `metrics.json`: selected model name, CV metrics, and feature schema (`num_cols`, `cat_cols`, `drop_cols`)
* `cv_results.csv`: comparison table across candidate models
* plots: `cv_rmse_by_model.png`, `cv_r2_by_model.png`, `cv_rmse_vs_r2.png`

---

## Manual input prediction (Example notebook)

`AzuaHousing.Example.ipynb` includes a manual-input workflow:

1. Edit a Python dict (`example_input`) with feature values.
2. Convert it to a one-row `DataFrame` using the feature schema stored in `artifacts/metrics.json`.
3. Call `model.predict(X_one)` and print the predicted price.

Missing fields can be set to `np.nan`; pipeline imputers handle them.

---

## REST API (Deployment)

The repo includes a lightweight REST API for real-time predictions using the trained artifacts.

### Prerequisite

Run `AzuaHousing.Example.ipynb` first (to generate `artifacts/model.joblib` and `artifacts/metrics.json`).

### Start the API server

```bash
python -m uvicorn serve:app --host 0.0.0.0 --port 8000
```

### Health check

```bash
curl -s http://127.0.0.1:8000/health
```

### Make a prediction

Provide a JSON body containing a `features` object. You may provide only a subset of features.

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Suburb": "Abbotsford",
      "Type": "h",
      "Method": "S",
      "Rooms": 2,
      "Distance": 2.5,
      "Postcode": 3067,
      "Landsize": 202,
      "Regionname": "Northern Metropolitan",
      "Date_Year": 2016,
      "Date_Month": 3
    }
  }'
```

The response returns:

* `predicted_price`
* `model_name` (from `metrics.json`)

---

## Reproducibility

* Train/test split uses `random_state=42`
* K-fold CV uses shuffle + `random_state=42`
* Stochastic models use the same seed where applicable

