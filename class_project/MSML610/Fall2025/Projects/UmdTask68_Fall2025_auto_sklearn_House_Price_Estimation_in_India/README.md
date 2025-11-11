# Auto-sklearn House Price Estimation (India)

Estimate residential prices across India using structured tabular data (location, size, amenities). The project currently covers data exploration and a reusable preprocessing pipeline in preparation for auto-sklearn modeling

## Dataset

- Source: [India House Price Prediction (Kaggle)](https://www.kaggle.com/datasets/ankushpanday1/india-house-price-prediction)
- Download the CSV and place it at `data/raw/india_housing_prices.csv` (file not tracked via git because of size limitation)

## What’s Implemented

- Exploratory analysis (`auto_sklearn_data_exploration.ipynb`)
- Modular preprocessing utilities:
  - `utils_data_io.py` – loading and cleaning
  - `utils_feature_engineering.py` – column grouping and binary mapping
  - `utils_transformers.py` – custom `AmenitiesEncoder`
  - `utils_preprocessing.py` – end-to-end pipeline + train/test split
  - `auto_sklearn_utils.py` – convenience re-exports
- Example notebook (`auto_sklearn_example.ipynb`) showing `prepare_data()` usage
- Supporting docs:
  - `data_exploration.md` – summary of EDA findings
  - This README – setup and progress tracker
  - Requirements, `.gitignore`

## Next Steps

1. Docker support
2. Train auto-sklearn regressors with `prepare_data()` output
3. Compare against RandomForest/XGBoost baselines
4. Evaluate with MAE/RMSE, visualize regional price trends

## Setup

### 1. Virtual env controls

source dev_scripts_umd_classes/thin_client/setenv.sh
deactivate

### 2. Start up process

1. start venv
2. open docker cli
3. now you can git push

### 3. Git controls

git config --list
git add -A class_project/MSML610/Fall2025/Projects/UmdTask68_Fall2025_auto_sklearn_House_Price_Estimation_in_India
git status
git commit -m "UmdTask68: commit message (refs #68)"
git push origin UmdTask68_Fall2025_auto_sklearn_House_Price_Estimation_in_India

### 4. Pip install

python -m pip install --upgrade pip
pip install -r /Users/ritvik/src/umd_classes1/class_project/MSML610/Fall2025/Projects/UmdTask68_Fall2025_auto_sklearn_House_Price_Estimation_in_India/requirements.txt
