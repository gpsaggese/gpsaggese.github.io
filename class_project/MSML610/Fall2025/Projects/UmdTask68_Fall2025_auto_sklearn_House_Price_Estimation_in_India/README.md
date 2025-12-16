# Auto-sklearn House Price Estimation (India)

Estimate residential prices across India using structured tabular data (location, size, amenities). The project leverages **auto-sklearn** to automatically select and tune regression models, comparing them against Random Forest and XGBoost baselines.

## Dataset

- Source: [India House Price Prediction (Kaggle)](https://www.kaggle.com/datasets/ankushpanday1/india-house-price-prediction)
- Download the CSV and place it at `data/raw/india_housing_prices.csv` (file not tracked via git because of size limitation)

## What’s Implemented

- **Data Pipeline**:
  - `utils_data_io.py`: Loading and cleaning
  - `utils_preprocessing.py`: End-to-end pipeline + train/test split, handling numeric imputation, one-hot encoding, and custom amenities parsing.
- **Modeling**:
  - `auto_sklearn.example.ipynb`: End-to-end notebook for preprocessing, training, and evaluation.
  - Implements `AutoSklearnRegressor`, `RandomForestRegressor`, and `XGBRegressor`.
- `auto_sklearn.api.ipynb`: Small native API snippets (classification/regression/metrics/leaderboard).
- **Infrastructure**:
  - `Dockerfile`: Linux-based environment for running auto-sklearn.
- **Analysis**:
  - `data_exploration.md`: EDA summary.
  - EDA visuals + evaluation metrics (MAE, RMSE) and residual analysis are demonstrated in `auto_sklearn.example.ipynb`.

## Usage

### 1. Using Docker (Recommended for Auto-sklearn)

Auto-sklearn requires a Linux environment. Use the provided Dockerfile:

```bash
# Build the image
docker build -t india-housing-prices .

# Run Jupyter Lab
docker run -p 8888:8888 -v $(pwd):/app india-housing-prices
```

Then open `auto_sklearn.example.ipynb` (end-to-end example) and `auto_sklearn.api.ipynb` (native API snippets) in the browser to run the experiments.

**Expected terminal output (Docker/Jupyter)**:

- You should see Jupyter start logs that include a URL with a token, similar to:

  - `http://127.0.0.1:8888/lab?token=...`

- The project directory is mounted into the container at **`/app`**, so any edits you make locally are visible inside the notebook environment.
- If port 8888 is already in use on your machine, stop the existing process or change the host port (e.g., `-p 8889:8888`).

### 2. Local Development

If you are on Linux or have a compatible environment:

```bash
pip install -r requirements.txt
jupyter lab
```

## Modeling Strategy

1.  **Preprocessing**: Features are standardized (numeric) or one-hot/binary encoded (categorical/text). Missing values are imputed.
2.  **AutoML**: `AutoSklearnRegressor` searches for the best ensemble of models (time limit configurable).
3.  **Baselines**: Standard Random Forest and XGBoost regressors are trained for comparison.
4.  **Evaluation**: Models are evaluated on a held-out test set (20%) using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
5.  **Interpretability**: Errors are analyzed by region (State) to identify geographic biases.

## Setup & Git

### Virtual env controls

```bash
source dev_scripts_umd_classes/thin_client/setenv.sh
deactivate
```

### Git controls

```bash
git add -A class_project/MSML610/Fall2025/Projects/UmdTask68_Fall2025_auto_sklearn_House_Price_Estimation_in_India
git commit -m "UmdTask68: Added auto-sklearn modeling and docker support (refs #68)"
git push origin UmdTask68_Fall2025_auto_sklearn_House_Price_Estimation_in_India
```
