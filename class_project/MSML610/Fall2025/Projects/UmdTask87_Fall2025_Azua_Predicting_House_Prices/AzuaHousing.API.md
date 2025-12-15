# AzuaHousing.API

This notebook (`AzuaHousing.API.ipynb`) is the **training and evaluation driver** for the Azua Housing project. It demonstrates how to use the internal Python API (`azua_utils.py`) to:

- Download the Melbourne Housing Snapshot dataset
- Load and clean data
- Compute dataset statistics (before and after cleaning)
- Run cross-validation model comparison **with progress bars**
- Fit the best model on the full training split
- Evaluate on a held-out test split
- Save model artifacts to `artifacts/`

## Repository layout

Expected files:

- `azua_utils.py` — internal API (data loading, preprocessing, model selection, artifact IO)
- `AzuaHousing.API.ipynb` — training + evaluation notebook (this doc)
- `requirements.txt` — Python dependencies
- `data/` — dataset folder (created by the notebook)
- `artifacts/` — output folder (created by the notebook)

## Setup

Install dependencies once (outside the notebook):

```bash
pip install -r requirements.txt
````

## Dataset

Default dataset path:

* `data/melb_data.csv`

The notebook uses:

* `DATA_PATH` environment variable if set
* otherwise defaults to `data/melb_data.csv`

Example:

```bash
export DATA_PATH=data/melb_data.csv
```

The dataset is downloaded using `kagglehub` and copied into `data/`.

## Notebook flow

The notebook is structured to show a clear, repeatable pipeline:

### 1) Load data

* Reads `melb_data.csv`
* Drops rows where `Price` is missing
* If the dataset contains a sale date, derives `Year` and `Month`

### 2) Compute stats (raw)

Produces basic health checks, including:

* row/column counts
* duplicate rows
* missingness report (top missing columns)
* target distribution summary and histogram

### 3) Clean data

Applies conservative cleaning designed to be compatible with downstream imputation:

* drops exact duplicate rows
* coerces known numeric columns to numeric
* replaces invalid negative values with `NaN`
* sanity-checks `YearBuilt` and sets invalid values to `NaN`

### 4) Compute stats (cleaned)

Recomputes the same stats to verify that cleaning behaves as intended.

### 5) Do analysis

* Train/test split (held-out test set)
* Cross-validation model selection across several regressors:

  * Linear Regression
  * ElasticNet
  * Random Forest
  * XGBoost
* Metrics tracked:

  * RMSE (lower is better)
  * R² (higher is better)
  * per-fold variability (`rmse_std`, `r2_std`)

Progress bars appear during:

* model iteration
* fold iteration (per model)

### 6) Show results

* Evaluates the selected best model on the held-out test split
* Saves artifacts to disk

## Outputs

After a successful run, `artifacts/` contains:

* `model.joblib` — fitted sklearn `Pipeline` (preprocessing + model)
* `metrics.json` — metadata, including selected model name and CV metrics
* `cv_results.csv` — CV summary for all candidate models
* `cv_rmse_by_model.png` — bar chart of CV RMSE
* `cv_r2_by_model.png` — bar chart of CV R²
* 
## Recommended usage order

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training notebook:

* Open `AzuaHousing.API.ipynb`
* Run all cells
* Confirm `artifacts/` is created and populated

3. Then run the example notebook:

* `AzuaHousing.Example.ipynb`
