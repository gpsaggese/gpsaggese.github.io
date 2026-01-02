## Auto-sklearn Example (End-to-End House Price Estimation)

- **Goal**: train an AutoML regressor (autosklearn) to estimate `Price_in_Lakhs` from tabular features, compare against baselines, and generate interpretable evaluation outputs
- **Design focus (per submission spec)**: notebooks stay light while reusable logic lives in `auto_sklearn_utils.py` and the `utils_*.py` modules

>  See `README.md` for setup, environment, and Docker instructions

## Project Structure

### Key idea

The notebook is structures in the flow: data loading → preprocessing → model training → evaluation. All complex code is factored into utilities so they can be reused and tested

### Files and responsibilities

- **`auto_sklearn.example.ipynb`**

  - The end to end example notebook
  - Uses your **API layer** for preprocessing (via `auto_sklearn_utils.prepare_data`)
  - Trains autosklearn + baselines, compares MAE/RMSE, and plots residual diagnostics
  - Also contains data exploratory analysis of the dataset

- **`auto_sklearn_utils.py`** (your “API layer facade”)

  - A single import surface for notebooks:
    - Re exports `prepare_data`, `load_housing_data`, and helper utilities
  - This reduces notebook boilerplate and keeps the interface stable

- **Utility modules (implementation behind the API layer)**

  - **`utils_data_io.py`**: reads the CSV and normalizes empty/blank strings to `NaN`
  - **`utils_feature_engineering.py`**:
    - defines column groups (numeric/binary/categorical/text)
    - encodes Yes/No binary flags to 1/0
  - **`utils_transformers.py`**: `AmenitiesEncoder` that expands the amenities text field into multi hot features
  - **`utils_preprocessing.py`**:
    - builds the `ColumnTransformer` preprocessor
    - implements `prepare_data` which returns train/test matrices, target vectors, the fitted preprocessor, and feature names


## Notebook walkthrough: `auto_sklearn.example.ipynb` (section-by-section)

The notebook is organized into **two major blocks**:

1. **EDA block** 

- quick dataset sanity checks and visuals
- goal: justify why non-linear / ensemble models are a good fit

2. **Modeling block** 

- uses the preprocessing API + trains AutoML and baselines
- goal: demonstrate an end to end application that runs top to bottom

### Section A — Data Exploration (EDA)

**Purpose**: sanity check the dataset and justify model choice

**What it does**

- Loads the raw CSV into a DataFrame
- Confirms basic schema and summary statistics
- Visualizes the target distribution (`Price_in_Lakhs`)
- Verifies missingness (the dataset is effectively complete)
- Prints quick categorical frequency tables to understand category balance
- Checks linear correlations / pairwise relationships for numeric fields
- Builds a median-price pivot (by `Property_Type` × `Furnished_Status`) to show simple, interpretable structure
- Prints dataset scale context (rows, number of states/cities)

**Outputs**

- A small set of plots (target histogram, correlation heatmap, pivot heatmap)
- Small summary tables (missingness, top category counts)

**Why this matters**
The EDA supports the decision to use **non linear/ensemble models** (autosklearn + tree baselines) rather than assuming a strong linear signal

### Section B — Data Preparation (via your API layer)

**Purpose**: transform raw tabular inputs into model ready numeric matrices in a reusable way

**What it does**

- Calls `auto_sklearn_utils.prepare_data(...)` to:
  - load and clean the dataset (normalizing blanks to `NaN`)
  - drop `ID` and isolate the target (`Price_in_Lakhs`)
  - encode Yes/No flags to 1/0
  - split into train/test sets
  - build + fit a preprocessing pipeline (`ColumnTransformer`)
  - return processed `X_train`, `X_test`, `y_train`, `y_test`, the fitted preprocessor, and feature names

**Outputs**

- Train/test shapes and a feature count summary
- A readable list of sample engineered feature names

**Why this matters**
This is the **reusable API layer** your notebooks build on: it keeps preprocessing consistent across models and keeps the notebook from embedding complex pipeline code

### Section C — Modeling (AutoML + baselines)

**Purpose**: train models and compare performance 

**What it does**

- Trains an `AutoSklearnRegressor` on the processed feature matrix
- (Optional) Inspects the AutoML run:
  - shows a leaderboard
  - explores ensemble composition
  - analyzes `cv_results_` to understand what was tried and which configurations performed best
- Trains baseline regressors:
  - `RandomForestRegressor`
  - `XGBRegressor`

**Outputs**

- Autosklearn training statistics (best validation score, run counts, etc)
- A compact baseline vs AutoML comparison

**Why this matters**
Baselines ensure the AutoML result is meaningful and not just “a model that runs”

### Section D — Evaluation & residual diagnostics

**Purpose**: evaluate models with consistent metrics and visualize error patterns

**What it does**

- Computes MAE and RMSE for each model on the held-out test set
- Produces a comparison table and bar charts
- Selects the best model (by MAE) and plots:
  - predicted vs actual
  - residual distribution

**Outputs**

- A results table (`Model`, `MAE`, `RMSE`).
- Metric comparison plots
- Residual diagnostics plots

### Section E — Model persistence

**Purpose**: demonstrate how a trained model becomes a reusable artifact

**What it does**

- Saves the best model to `models/` using `joblib`

**Output**

- A single “Saved best model to …” confirmation line



## Key design decisions

- **Preprocessing is a reusable API**

  - The model code should not need to know how amenities are parsed, how binary flags are encoded, or how the `ColumnTransformer` is assembled
  - That’s why the notebook calls `prepare_data` and receives already processed matrices and feature names

- **Wrapper/facade import (`auto_sklearn_utils.py`)**

  - Notebooks import from one place
  - Internal modules (`utils_preprocessing.py`, `utils_transformers.py`, etc.) can evolve without rewriting notebooks

- **Baselines included for credibility**
  - Random Forest and XGBoost provide a sanity check and help interpret whether AutoML is actually helping


## EDA summary 

### Dataset overview

| Item         | Value                |
| ------------ | -------------------- |
| **Rows**     | 250,000              |
| **Features** | 23                   |
| **Coverage** | 20 states, 42 cities |
| **Target**   | `Price_in_Lakhs`     |

### Key findings

- **Data quality**

  - **No missing values** reported
  - Dataset is ready for modeling without heavy cleaning

- **Feature types**

  - **Numeric**: 9 columns (e.g., `BHK`, `Size_in_SqFt`, `Year_Built`, …)
  - **Categorical**: ~12 columns (e.g., `State`, `City`, `Furnished_Status`, …)
  - **Target**: `Price_in_Lakhs`

- **Target distribution**

  - Broad distribution with no extreme spikes

- **Correlations**

  - **Weak linear correlation** with target across numeric features
  - Suggests **non linear models / ensembles** are appropriate
  - Dataset may be synthetically generated (loosely coupled features)

- **Data balance**
  - Categories are reasonably balanced
  - Broad geographic coverage helps avoid region-specific overfitting

### Takeaway

This dataset is **clean** but appears **weakly linear**, making it a good candidate for **AutoML** and **ensemble** approaches like autosklearn
