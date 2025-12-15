## Auto-sklearn Example (End-to-End House Price Estimation)

This document explains the **complete example application** implemented in `auto_sklearn_example.ipynb`.

- **Goal**: train an AutoML regressor (auto-sklearn) to estimate `Price_in_Lakhs` from tabular features, compare against baselines, and generate interpretable evaluation outputs.
- **Design focus (per submission spec)**: notebooks stay _light_, while reusable logic lives in `auto_sklearn_utils.py` and the `utils_*.py` modules.

> Setup, environment, and Docker instructions are intentionally not repeated here; see `README.md` for those.

---

## How the project is structured (what talks to what)

### Key idea

The notebook is written as an “application script”: it **orchestrates** data loading → preprocessing → model training → evaluation. All “complex” steps are factored into utilities so they can be reused and tested.

### Files and responsibilities

- **`auto_sklearn_example.ipynb`**

  - The end-to-end example notebook.
  - Uses your **API layer** for preprocessing (via `auto_sklearn_utils.prepare_data`).
  - Trains auto-sklearn + baselines, compares MAE/RMSE, and plots residual diagnostics.

- **`auto_sklearn_utils.py`** (your “API layer facade”)

  - A single import surface for notebooks:
    - Re-exports `prepare_data`, `load_housing_data`, and helper utilities.
  - This reduces notebook boilerplate and keeps the interface stable even if you refactor internals.

- **Utility modules (implementation behind the API layer)**

  - **`utils_data_io.py`**: reads the CSV and normalizes empty/blank strings to `NaN`.
  - **`utils_feature_engineering.py`**:
    - defines column groups (numeric/binary/categorical/text)
    - encodes Yes/No binary flags to 1/0
  - **`utils_transformers.py`**: `AmenitiesEncoder` that expands the amenities text field into multi-hot features.
  - **`utils_preprocessing.py`**:
    - builds the `ColumnTransformer` preprocessor
    - implements `prepare_data(...)` which returns train/test matrices, target vectors, the fitted preprocessor, and feature names

- **Supporting docs**
  - **`data_exploration.md`**: human-readable EDA summary (included below and reformatted).

---

## Notebook walkthrough: `auto_sklearn_example.ipynb` (cell-by-cell)

The notebook currently contains **two major blocks**:

1. **EDA block** (cells 0–13)

- quick dataset sanity checks and visuals
- goal: justify why non-linear / ensemble models are a good fit

2. **Modeling block** (cells 14–41)

- uses the preprocessing API + trains AutoML and baselines
- goal: demonstrate an end-to-end application that runs top-to-bottom

Below, each row explains what the cell does and what output to expect.

### Cells 0–13: Data Exploration block

| Cell | Type     | What it does                                                                                      | Expected output (clean summary)                                                                            |
| ---: | -------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
|    0 | Markdown | Section header: “Data Exploration”.                                                               | No output.                                                                                                 |
|    1 | Code     | Imports plotting/data packages for EDA.                                                           | No output.                                                                                                 |
|    2 | Code     | Loads the raw CSV into a DataFrame and runs quick inspection (`head`, `info`, `describe`).        | A small preview + schema info + descriptive stats. (For final “clean outputs”, this should be kept short.) |
|    3 | Markdown | Explains the target distribution intent.                                                          | No output.                                                                                                 |
|    4 | Code     | Defines numeric columns and plots a histogram of `Price_in_Lakhs`.                                | One histogram plot.                                                                                        |
|    5 | Markdown | Interprets missingness checks.                                                                    | No output.                                                                                                 |
|    6 | Code     | Computes missing fraction and visualizes missingness for numeric columns.                         | Missingness table + heatmap (likely empty if no missing).                                                  |
|    7 | Markdown | Explains categorical frequency checks.                                                            | No output.                                                                                                 |
|    8 | Code     | Prints top category counts for each object column (excluding amenities + target).                 | Several small tables (top 10 each).                                                                        |
|    9 | Markdown | Interprets correlation + pairplot intent.                                                         | No output.                                                                                                 |
|   10 | Code     | Correlation heatmap + sampled pairplot.                                                           | Two plots (correlation heatmap + pairplot).                                                                |
|   11 | Markdown | Explains median price pivot intent.                                                               | No output.                                                                                                 |
|   12 | Code     | Creates a pivot table (median price by `Property_Type` × `Furnished_Status`) and plots a heatmap. | Table + heatmap.                                                                                           |
|   13 | Code     | Prints dataset scale stats (rows, distinct states/cities).                                        | 3 short printed lines.                                                                                     |

### Cells 14–22: Data Preparation (API-layer usage)

| Cell | Type     | What it does                                                                                                                              | Expected output                                                 |
| ---: | -------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
|   14 | Markdown | Section header: “Data Preparation”.                                                                                                       | No output.                                                      |
|   15 | Code     | Imports NumPy/Pandas and your wrapper function `prepare_data` from `auto_sklearn_utils`.                                                  | No output.                                                      |
|   16 | Code     | Defines `DATA_PATH` (relative) and calls `prepare_data(...)` to return processed train/test arrays + feature names + fitted preprocessor. | A single “loading…” message + “data preparation complete”.      |
|   17 | Code     | Prints the shapes of processed train/test arrays and target vectors; prints count of engineered features.                                 | Short shape summary.                                            |
|   18 | Code     | Prints a subset of feature names (first 20 + last 10).                                                                                    | A short, readable list.                                         |
|   19 | Code     | Prints target summary statistics for train and test sets.                                                                                 | A small `describe()` summary (count/mean/std/min/max).          |
|   20 | Code     | Prints raw processed feature rows, dtypes, and NaN counts.                                                                                | **Noisy** (big arrays) — should be minimized for clean outputs. |
|   21 | Code     | Builds a DataFrame for processed features and prints full `describe()` and global min/max/mean/std.                                       | **Very noisy** — should be reduced for clean outputs.           |
|   22 | Code     | Prints a text summary explaining what preprocessing did (imputation, scaling, one-hot, amenities).                                        | A short narrative block.                                        |

### Cells 23–41: Modeling + evaluation (end-to-end “application”)

| Cell | Type     | What it does                                                                                            | Expected output                                                    |
| ---: | -------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
|   23 | Markdown | Section header for modeling pipeline + what will be demonstrated.                                       | No output.                                                         |
|   24 | Code     | Imports modeling dependencies (sklearn, xgboost, autosklearn, joblib) and local utilities.              | No output.                                                         |
|   25 | Markdown | Step label: load + preprocess.                                                                          | No output.                                                         |
|   26 | Code     | Calls `prepare_data(...)` again (currently imported from `utils_preprocessing`).                        | Train/test shapes.                                                 |
|   27 | Markdown | Step label: train auto-sklearn model.                                                                   | No output.                                                         |
|   28 | Code     | Ensures feature matrix is float32 (auto-sklearn compatibility) using a small helper.                    | Prints data type info.                                             |
|   29 | Code     | Configures + trains `AutoSklearnRegressor` and prints training summary (`sprint_statistics`).           | Fit logs + statistics (can be long).                               |
|   30 | Markdown | Step label: model inspection.                                                                           | No output.                                                         |
|   31 | Code     | Prints leaderboard, shows ensemble models, and prints model weights/configs.                            | **Extremely long output** (not clean).                             |
|   32 | Code     | Analyzes `cv_results_`: prints status breakdown, score summary, top-5 configs, and plots distributions. | A concise table + 1–2 plots (can be kept clean if trimmed).        |
|   33 | Markdown | Step label: baselines.                                                                                  | No output.                                                         |
|   34 | Code     | Trains `RandomForestRegressor` and `XGBRegressor` on a subset.                                          | Two short “Training …” lines (avoid printing full estimator repr). |
|   35 | Markdown | Step label: evaluation.                                                                                 | No output.                                                         |
|   36 | Code     | Predicts with each model, computes MAE/RMSE, builds a comparison table, prints relative improvement.    | One compact results table + 1–2 summary lines.                     |
|   37 | Code     | Visualizes MAE/RMSE comparison.                                                                         | One figure (two bar charts).                                       |
|   38 | Code     | Prints additional autosklearn ensemble insights and repeats the improvement statement.                  | Text summary (can be shortened).                                   |
|   39 | Markdown | Step label: residual analysis.                                                                          | No output.                                                         |
|   40 | Code     | Picks best model by MAE and plots predicted-vs-actual + residual distribution.                          | One figure (2 subplots).                                           |
|   41 | Code     | Saves best model artifact to `models/` via joblib.                                                      | One “Saved model to …” line.                                       |
|   42 | Code     | Empty cell (safe to delete).                                                                            | No output.                                                         |

---

## Why this design (key decisions)

- **Preprocessing is a reusable API**

  - The model code should not need to know how amenities are parsed, how binary flags are encoded, or how the `ColumnTransformer` is assembled.
  - That’s why the notebook calls `prepare_data(...)` and receives already-processed matrices and feature names.

- **Wrapper/facade import (`auto_sklearn_utils.py`)**

  - Notebooks import from one place.
  - Internal modules (`utils_preprocessing.py`, `utils_transformers.py`, etc.) can evolve without rewriting notebooks.

- **Baselines included for credibility**
  - Random Forest and XGBoost provide a sanity check and help interpret whether AutoML is actually helping.

---

## Neatly formatted EDA summary (from `data_exploration.md`)

### Dataset overview

| Item         | Value                |
| ------------ | -------------------- |
| **Rows**     | 250,000              |
| **Features** | 23                   |
| **Coverage** | 20 states, 42 cities |
| **Target**   | `Price_in_Lakhs`     |

### Key findings

- **Data quality**

  - **No missing values** reported.
  - Dataset is ready for modeling without heavy cleaning.

- **Feature types**

  - **Numeric**: 9 columns (e.g., `BHK`, `Size_in_SqFt`, `Year_Built`, …)
  - **Categorical**: ~12 columns (e.g., `State`, `City`, `Furnished_Status`, …)
  - **Target**: `Price_in_Lakhs`

- **Target distribution**

  - Broad distribution with no extreme spikes.

- **Correlations**

  - **Weak linear correlation** with target across numeric features.
  - Suggests **non-linear models / ensembles** are appropriate.
  - Dataset may be synthetically generated (loosely coupled features).

- **Data balance**
  - Categories are reasonably balanced.
  - Broad geographic coverage helps avoid region-specific overfitting.

### Takeaway

This dataset is **clean** but appears **weakly linear**, making it a good candidate for **AutoML** and **ensemble** approaches like auto-sklearn.
