# UmdTask87 — Predicting House Prices

This project trains and evaluates regression models to predict house prices using the Kaggle **Melbourne Housing Snapshot** dataset (`melb_data.csv`). The workflow is notebook-driven and uses a small internal Python API (`azua_utils.py`) for data loading, preprocessing, cross-validation model selection, and artifact saving.

## What’s included

- **`AzuaHousing.API.ipynb`**: end-to-end training + evaluation notebook
  - Download dataset into `data/`
  - Load → compute stats → clean → compute stats
  - Cross-validation model comparison (RMSE, R²) with progress bars
  - Fit best model and evaluate on held-out test set
  - Save artifacts to `artifacts/`
- **`AzuaHousing.Example.ipynb`**: consumer example notebook
  - Load saved artifacts (`model.joblib`, `metrics.json`)
  - Run inference + evaluation
  - Manual input prediction (edit a dict → predict)
- **`azua_utils.py`**: internal API used by both notebooks
- **`requirements.txt`**: dependencies (install once; no in-notebook installs)

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
    ├── cv_r2_by_model.png
    └── cv_rmse_vs_r2.png (optional)
````

---

## Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run training + evaluation

Open and run all cells in:

* `AzuaHousing.API.ipynb`

This will create:

* `data/melb_data.csv`
* `artifacts/` with the trained model and metrics

### 3) Run the consumer example

Open and run all cells in:

* `AzuaHousing.Example.ipynb`

This notebook assumes `artifacts/model.joblib` and `artifacts/metrics.json` already exist.

---

## Dataset

The API notebook downloads the dataset via `kagglehub` and copies `melb_data.csv` into `./data`.

Default dataset path used by notebooks:

* `data/melb_data.csv`

Optional override:

```bash
export DATA_PATH=/path/to/melb_data.csv
```

---

## Internal API (`azua_utils.py`)

Key functions used by notebooks:

* `load_melbourne(csv_path) -> pd.DataFrame`
  Loads and lightly processes the dataset (drops missing target, derives `Year` and `Month` if date exists).

* `train_select_best(df, folds=5, ...) -> dict`
  Runs K-fold CV across several models and selects the best by lowest CV RMSE. Fits the best pipeline on the full training split.

* `save_artifacts(bundle, outdir="artifacts")`
  Saves `model.joblib`, `metrics.json`, and `cv_results.csv`.

* `load_model("artifacts/model.joblib")`
  Loads the fitted sklearn pipeline for inference.

---

## Metrics and expected outputs

The notebooks report:

* **RMSE** (lower is better)
* **R²** (higher is better)
* Fold variability: `rmse_std`, `r2_std`

Artifacts saved:

* `artifacts/model.joblib`: preprocessing + model pipeline (ready for inference)
* `artifacts/metrics.json`: selected model + CV metrics + feature metadata
* `artifacts/cv_results.csv`: comparison table across candidate models

---

## Manual input prediction (Example notebook)

`AzuaHousing.Example.ipynb` includes a manual-input workflow:

1. Edit a Python dict (`example_input`) with feature values.
2. Convert it to a one-row `DataFrame`.
3. Call `model.predict(X_one)` and print the predicted price.

You can leave missing fields as `np.nan`; the pipeline imputers handle them.

---

## REST API (Deployment)

This project includes a lightweight REST API for real-time house price predictions using the trained model artifacts.

### Prerequisites
1) Run `AzuaHousing.API.ipynb` first to generate:
- `artifacts/model.joblib`
- `artifacts/metrics.json`

2) Install dependencies:
```bash
pip install -r requirements.txt
````

### Start the API server

Run the server from the project environment:

```bash
python -m uvicorn serve:app --host 0.0.0.0 --port 8000
```

### Health check

```bash
curl -s http://127.0.0.1:8000/health
```

Expected response includes `status: "ok"` and confirms the model is loaded.

### Make a prediction

Send a JSON payload containing a `features` object. You can provide only a subset of features; missing values are handled by the pipeline imputers.

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
      "Year": 2016,
      "Month": 3
    }
  }'
```

The response returns:

* `predicted_price`: predicted house price
* `model_name`: model selected during training (from `metrics.json`)

---

## Reproducibility

* Train/test split uses `random_state=42`
* K-fold CV uses shuffle + `random_state=42`
* Stochastic models use the same seed where applicable
