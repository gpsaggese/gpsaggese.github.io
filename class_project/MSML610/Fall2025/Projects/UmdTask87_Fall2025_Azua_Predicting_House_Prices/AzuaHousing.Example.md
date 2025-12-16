# AzuaHousing.Example

This notebook (`AzuaHousing.Example.ipynb`) is the **end-to-end demonstration notebook** for the Azua Housing project. It shows how to use the internal Python API (`azua_utils.py`) to:

* Download the Melbourne Housing Snapshot dataset
* Load and lightly clean data
* Run cross-validation model comparison **with progress bars**
* Fit the best model on the training split and evaluate on a held-out test split
* Save deployable model artifacts to `artifacts/`
* Reload saved artifacts and run inference
* Perform basic error analysis and visualization
* Run a single-row prediction example (manual input)

## Repository layout

Expected files:

* `azua_utils.py` — internal API (CSV loading, preprocessing, model selection, artifact I/O)
* `AzuaHousing.Example.ipynb` — demo notebook (this doc)
* `requirements.txt` — Python dependencies
* `data/` — dataset folder (created by the notebook)
* `artifacts/` — output folder (created by the notebook)

## Setup

Install dependencies once (outside the notebook):

```bash
pip install -r requirements.txt
```

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

The notebook is structured to show a clear, repeatable pipeline from training through deployment-style inference:

### 1) Download dataset

* Downloads the Kaggle dataset using `kagglehub`
* Copies `melb_data.csv` into `./data/`
* Reuses the local file if it already exists

### 2) Load dataset

* Loads `melb_data.csv`
* Drops rows where `Price` is missing
* If a sale date column is present, derives date-part features using the utils convention:

  * `Date_Year`
  * `Date_Month`

### 3) Compute stats (raw)

Computes basic dataset health checks, including:

* row/column counts
* duplicate rows
* missingness report (top missing columns)
* target distribution summary and histogram

### 4) Clean data

Applies conservative cleaning designed to remain compatible with downstream imputation:

* drops exact duplicate rows
* coerces known numeric columns to numeric
* replaces invalid negative values with `NaN`
* sanity-checks `YearBuilt` and sets invalid values to `NaN`
* retains missing values wherever possible (pipeline imputers handle them)

### 5) Compute stats (cleaned)

Recomputes the same statistics to verify that cleaning behaves as intended.

### 6) Train/test split

* Creates a held-out test set for final evaluation

### 7) Model comparison (cross-validation)

Runs K-fold CV across multiple regressors and selects the best model by lowest CV RMSE:

* Linear Regression
* ElasticNet
* Random Forest
* XGBoost

Metrics tracked:

* RMSE (lower is better)
* R² (higher is better)
* per-fold variability (`rmse_std`, `r2_std`)

Progress bars appear during:

* model iteration
* fold iteration (per model)

### 8) Plot CV results

Generates and saves plots under `artifacts/`:

* `cv_rmse_by_model.png`
* `cv_r2_by_model.png`
* `cv_rmse_vs_r2.png`

### 9) Evaluate best pipeline on held-out test set

* Evaluates the selected pipeline on the held-out test split
* Reports test RMSE and test R²

### 10) Save artifacts

Writes deployable artifacts:

* `artifacts/model.joblib` — fitted sklearn `Pipeline` (preprocessing + model)
* `artifacts/metrics.json` — metadata (best model name, CV scores, feature lists)
* `artifacts/cv_results.csv` — CV summary for all candidates

### 11) Load artifacts and run inference

Demonstrates “deployment-style” usage:

* loads the saved pipeline via `load_model()`
* loads `metrics.json` for metadata validation
* runs inference on a held-out split and reports RMSE/MAE/R²

### 12) Error analysis and visualization

* shows best/worst examples by absolute error
* plots:

  * predicted vs actual scatter
  * residual histogram
* exports `artifacts/predictions_sample.csv` for reporting/debugging

### 13) Single-row prediction examples

Two examples are provided:

* **Single-row from the test set** (quick sanity check)
* **Manual input dict → 1-row DataFrame → prediction** (interactive example)

## Outputs

After a successful run, `artifacts/` contains:

* `model.joblib`
* `metrics.json`
* `cv_results.csv`
* `cv_rmse_by_model.png`
* `cv_r2_by_model.png`
* `cv_rmse_vs_r2.png`
* `predictions_sample.csv`

## Recommended usage order

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the example notebook:

* Open `AzuaHousing.Example.ipynb`
* Run all cells top-to-bottom
* Confirm `data/` and `artifacts/` are created and populated
