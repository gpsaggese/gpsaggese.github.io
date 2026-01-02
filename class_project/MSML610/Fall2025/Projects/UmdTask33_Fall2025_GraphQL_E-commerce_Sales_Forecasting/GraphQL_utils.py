from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import os
import json
import joblib

# --------------------------------------------------------
# Helper: load all Kaggle CSVs from data/raw (+ calendar)
# --------------------------------------------------------
def load_raw_kaggle_data(
    base_path: str = "data/raw",
    calendar_path: str | None = "data/calendar_features.csv",
) -> Dict[str, pd.DataFrame]:
    """
    Load the Kaggle e-commerce files from data/raw and optionally
    a calendar_features.csv file for external features.

    Returns a dict with keys like:
        - 'sales_train', 'items', 'item_categories', 'shops', ...
        - 'calendar' (may be None if file not found)
    """
    files = {
        "sales_train": "sales_train.csv",
        "items": "items.csv",
        "item_categories": "item_categories.csv",
        "shops": "shops.csv",
        "test": "test.csv",
        "sample_submission": "sample_submission.csv",
    }

    data: Dict[str, pd.DataFrame] = {}
    for key, fname in files.items():
        path = os.path.join(base_path, fname)
        data[key] = pd.read_csv(path)

    # Optional calendar features
    if calendar_path is not None and os.path.exists(calendar_path):
        data["calendar"] = pd.read_csv(calendar_path)
    else:
        data["calendar"] = None

    return data


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def make_monthly_shop_item_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales to monthly shop-item level.

    Parameters
    sales_df : pandas.DataFrame
        Raw sales data as loaded from 'sales_train.csv'. Must contain at least
        the columns:
            - 'date_block_num'
            - 'shop_id'
            - 'item_id'
            - 'item_cnt_day'
            - 'item_price'

    Returns
    pandas.DataFrame
        DataFrame with one row per (date_block_num, shop_id, item_id) and
        the following columns:
            - 'date_block_num'
            - 'shop_id'
            - 'item_id'
            - 'item_cnt_month'   (sum of item_cnt_day within that month)
            - 'avg_item_price'   (mean item_price within that month)
    """
    required_cols = {"date_block_num", "shop_id", "item_id", "item_cnt_day", "item_price"}
    missing = required_cols - set(sales_df.columns)
    if missing:
        raise ValueError(f"sales_df is missing required columns: {missing}")

    group_cols = ["date_block_num", "shop_id", "item_id"]

    monthly = (
        sales_df
        .groupby(group_cols, as_index=False)
        .agg(
            item_cnt_month=("item_cnt_day", "sum"),
            avg_item_price=("item_price", "mean"),
        )
    )

    return monthly


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple time features derived from 'date_block_num'.

    In the Kaggle dataset, date_block_num is 0 for Jan 2013, 1 for Feb 2013,
    up to 33 for Oct 2015.

    We convert this into calendar-like features:
        - 'year'
        - 'month' (1-12)

    Parameters
    df : pandas.DataFrame
        Must contain a 'date_block_num' column.

    Returns
    pandas.DataFrame
        Copy of the input with additional 'year' and 'month' columns.
    """
    if "date_block_num" not in df.columns:
        raise ValueError("DataFrame must contain 'date_block_num' to add date features.")

    df = df.copy()
    df["year"] = 2013 + (df["date_block_num"] // 12)
    df["month"] = (df["date_block_num"] % 12) + 1
    return df


def build_base_training_table(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Create the base monthly training table from raw daily sales.

    Steps:
      1. Aggregate daily records into monthly shop-item sales.
      2. Add simple time features (year, month).
      3. Optionally join calendar / promo features on date_block_num.
    """
    monthly = make_monthly_shop_item_sales(sales_df)
    monthly = add_date_features(monthly)

    if calendar_df is not None:
        if "date_block_num" not in calendar_df.columns:
            raise ValueError("calendar_df must contain 'date_block_num'.")
        monthly = monthly.merge(calendar_df, on="date_block_num", how="left")
        # any missing calendar values => 0
        monthly.fillna(0.0, inplace=True)

    return monthly


# ---------------------------------------------------------------------------
# Lag features and model training
# ---------------------------------------------------------------------------

def make_lagged_features(
    monthly_df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """
    Add lag features of item_cnt_month for each (shop_id, item_id) pair.

    For example, with lags=(1, 2, 3), this will create columns:
        - 'lag_1'
        - 'lag_2'
        - 'lag_3'

    where 'lag_1' is last month's item_cnt_month for the same shop and item,
    'lag_2' is two months ago, etc.

    Missing values (at the start of the time series) are filled with 0.

    Parameters
    monthly_df : pandas.DataFrame
        Output of build_base_training_table, containing at least:
            - 'date_block_num'
            - 'shop_id'
            - 'item_id'
            - 'item_cnt_month'
    lags : tuple of int
        Lag sizes in months.

    Returns
    pandas.DataFrame
        Copy of the input with additional lag columns.
    """
    required_cols = {"date_block_num", "shop_id", "item_id", "item_cnt_month"}
    missing = required_cols - set(monthly_df.columns)
    if missing:
        raise ValueError(f"monthly_df is missing required columns: {missing}")

    df = monthly_df.copy()
    df = df.sort_values(["shop_id", "item_id", "date_block_num"])

    group_cols = ["shop_id", "item_id"]
    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(group_cols)["item_cnt_month"]
              .shift(lag)
        )

    # At the beginning of each time series we wont have full history
    # We fill those NaNs with 0 as a simple baseline
    df.fillna(0.0, inplace=True)

    return df


def make_train_val_sets(
    features_df: pd.DataFrame,
    val_block: int = 32,
    target_col: str = "item_cnt_month",
):
    """
    Create train/validation splits from a feature table.

    We use a simple time-based split:
        - training = all rows with date_block_num < val_block
        - validation = rows with date_block_num == val_block

    Parameters
    features_df : pandas.DataFrame
        Feature table produced by make_lagged_features.
    val_block : int
        The date_block_num to use as the validation month.
        For the Kaggle dataset, typical options are 32 or 33.
    target_col : str
        Name of the target column (default: 'item_cnt_month').

    Returns
    X_train, y_train, X_val, y_val, feature_cols
        - X_train : pandas.DataFrame
        - y_train : pandas.Series
        - X_val   : pandas.DataFrame
        - y_val   : pandas.Series
        - feature_cols : list of str (names of feature columns)
    """
    if "date_block_num" not in features_df.columns:
        raise ValueError("features_df must contain 'date_block_num'.")

    if target_col not in features_df.columns:
        raise ValueError(f"features_df must contain target column '{target_col}'.")

    # Feature columns = everything except the target and the time index.
    feature_cols = [
        c for c in features_df.columns
        if c not in (target_col, "date_block_num")
    ]

    train_mask = features_df["date_block_num"] < val_block
    val_mask = features_df["date_block_num"] == val_block

    train_df = features_df[train_mask]
    val_df = features_df[val_mask]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    return X_train, y_train, X_val, y_val, feature_cols


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Fit a simple RandomForestRegressor as a baseline model.

    Parameters
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training target.
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth of the trees (None => let the model decide).
    random_state : int
        Seed for reproducibility.

    Returns
    RandomForestRegressor
        Fitted model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(
    model: RandomForestRegressor,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """
    Evaluate a model on the validation set using RMSE.

    Parameters
    model : RandomForestRegressor
        Trained model.
    X_val : pandas.DataFrame
        Validation features.
    y_val : pandas.Series
        Validation target.

    Returns
    float
        Root Mean Squared Error (RMSE).
    """
    preds = model.predict(X_val)

    # Older versions of scikit-learn do not support `squared=` kwarg,
    # so we compute RMSE manually as sqrt(MSE).
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)

    return rmse

# ----------------------------------------------------------------------
# Inference helpers: load trained model, rebuild features, predict
# ----------------------------------------------------------------------
import json
import joblib



def load_trained_model_and_features(
    model_path: str = "models/rf_ecommerce_monthly.joblib",
    meta_path: str = "models/model_meta.json",
    raw_path: str = "data/raw",
) -> tuple[RandomForestRegressor, pd.DataFrame, list[str]]:
    """
    Convenience helper for notebook / API:

      1. Load trained RandomForest model + metadata.
      2. Reload raw Kaggle data (and calendar), rebuild monthly + lagged features.

    Returns
    model : RandomForestRegressor
    lagged : pandas.DataFrame
    feature_cols : list[str]
    """
    # 1) model + meta 
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    assert os.path.exists(meta_path), f"Meta file not found: {meta_path}"

    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols: list[str] = meta["feature_cols"]
    lags: list[int] = meta["lags"]

    # 2) rebuild monthly + lagged
    raw = load_raw_kaggle_data(base_path=raw_path)
    sales = raw["sales_train"]
    calendar = raw.get("calendar", None)

    monthly = build_base_training_table(sales_df=sales, calendar_df=calendar)
    lagged = make_lagged_features(monthly, lags=tuple(lags))

    return model, lagged, feature_cols

def build_feature_row(
    lagged: pd.DataFrame,
    feature_cols: list[str],
    shop_id: int,
    item_id: int,
    date_block_num: int,
) -> pd.DataFrame:
    """
    Find the row in `lagged` for the given (shop_id, item_id, date_block_num)
    and return a 1-row DataFrame with just the feature columns.
    """
    mask = (
        (lagged["shop_id"] == shop_id)
        & (lagged["item_id"] == item_id)
        & (lagged["date_block_num"] == date_block_num)
    )
    rows = lagged.loc[mask]

    if rows.empty:
        raise ValueError(
            f"No data for shop_id={shop_id}, item_id={item_id}, "
            f"date_block_num={date_block_num}"
        )

    # keep as a DataFrame with one row
    row = rows.iloc[[0]]
    X = row[feature_cols].copy()
    return X

def moving_average_baseline(
    row: pd.Series,
    lag_cols: tuple[str, ...] = ("lag_1", "lag_2", "lag_3"),
) -> float:
    """
    Simple statistical baseline: average of last-k-months sales
    using the lag_* columns.
    """
    vals = [row[c] for c in lag_cols if c in row.index]
    vals = [v for v in vals if pd.notnull(v)]
    if not vals:
        return 0.0
    return float(np.mean(vals))

def predict_sales(
    model: RandomForestRegressor,
    lagged: pd.DataFrame,
    feature_cols: list[str],
    shop_id: int,
    item_id: int,
    date_block_num: int,
) -> float:
    """
    Use the trained model to predict item_cnt_month for a given
    (shop_id, item_id, date_block_num).

    We combine:
      - RandomForest prediction
      - simple moving-average baseline from lag_1 ,...lag_3
    into a small ensemble.
    """
    # 1) build feature row
    X = build_feature_row(lagged, feature_cols, shop_id, item_id, date_block_num)

    # 2) ML prediction
    rf_pred = float(model.predict(X)[0])

    # 3) statistical baseline on same row
    ma_pred = moving_average_baseline(X.iloc[0])

    # 4) ensemble: tweak weights if you want
    final_pred = 0.7 * rf_pred + 0.3 * ma_pred
    return float(final_pred)
