"""
Utility functions for the Solar / Renewable Energy Forecasting project.

Includes:
- Data loading
- Time-based feature engineering (lags, rolling means, calendar features)
- Train/validation split for time-series
- Simple plotting helpers for diagnostics
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

TIME_COL = "timestamp"
TARGET_COL = "energy_mwh"


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_data(csv_path: str, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load the raw solar energy CSV file.

    Expected columns:
        - timestamp
        - energy_mwh
        - temp_c
        - cloud_cover
        - solar_radiation
        - wind_speed

    Returns
    -------
    df : pd.DataFrame
        DataFrame indexed by timestamp with the above columns (except timestamp),
        sorted by time.
    """
    df = pd.read_csv(csv_path)

    if parse_dates and TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df = df.sort_values(TIME_COL).set_index(TIME_COL)

    # Keep only the columns we care about (in case the CSV has extras)
    expected_cols = [
        TARGET_COL,
        "temp_c",
        "cloud_cover",
        "solar_radiation",
        "wind_speed",
    ]
    # Keep any that actually exist
    cols = [c for c in expected_cols if c in df.columns]
    df = df[cols]

    return df

def load_raw_solar(csv_path: str) -> pd.DataFrame:
    """
    Backwards-compatible helper for older code.

    scripts/make_features.py expects a function called `load_raw_solar`,
    but the main implementation here is `load_data`. This wrapper just
    forwards to `load_data` so both names work.
    """
    return load_data(csv_path)


# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------

def make_basic_time_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Create simple time-series features:

    - calendar features from the timestamp index:
        * hour of day
        * day of week
        * month
    - lagged values of energy_mwh
    - rolling mean of energy_mwh over a few windows

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by a DatetimeIndex and containing TARGET_COL.
    lags : list of int, optional
        Lag steps (in hours) for the target. Default: [1, 2, 24].
    rolling_windows : list of int, optional
        Window sizes (in hours) for rolling mean of the target.
        Default: [3, 24].

    Returns
    -------
    df_feats : pd.DataFrame
        DataFrame with original columns + new features, with rows
        at the beginning dropped to account for lag/rolling history.
    """
    if lags is None:
        lags = [1, 2, 24]
    if rolling_windows is None:
        rolling_windows = [3, 24]

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by a DatetimeIndex to build time features.")

    df_feats = df.copy()

    # Calendar features
    df_feats["hour"] = df_feats.index.hour
    df_feats["dayofweek"] = df_feats.index.dayofweek
    df_feats["month"] = df_feats.index.month

    # Lag features for the target
    for lag in lags:
        col_name = f"{TARGET_COL}_lag_{lag}"
        df_feats[col_name] = df_feats[TARGET_COL].shift(lag)

    # Rolling mean features for the target
    for window in rolling_windows:
        col_name = f"{TARGET_COL}_rollmean_{window}"
        df_feats[col_name] = df_feats[TARGET_COL].rolling(window=window, min_periods=1).mean().shift(1)

    # Drop initial rows that don't have full history
    max_history = max(max(lags), max(rolling_windows))
    df_feats = df_feats.iloc[max_history:]

    return df_feats


# -------------------------------------------------------------------
# Train / validation split
# -------------------------------------------------------------------

def train_val_split(
    df_feats: pd.DataFrame,
    test_size_days: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Split the feature DataFrame into train and validation sets,
    using the last `test_size_days` worth of data as validation.

    Parameters
    ----------
    df_feats : pd.DataFrame
        Feature DataFrame returned by make_basic_time_features.
        Must contain TARGET_COL.
    test_size_days : int, optional
        Number of days to hold out at the end of the series for validation.
        With hourly data, each day has 24 rows.

    Returns
    -------
    X_train : pd.DataFrame
    X_val   : pd.DataFrame
    y_train : pd.Series
    y_val   : pd.Series
    feature_cols : list of str
        Names of the feature columns used in X.
    """
    if TARGET_COL not in df_feats.columns:
        raise ValueError("Feature DataFrame must contain the target column '{}'".format(TARGET_COL))

    # Determine number of validation rows based on days
    rows_per_day = 24  # assumption: hourly data
    test_size = test_size_days * rows_per_day

    if test_size >= len(df_feats):
        raise ValueError("Validation set ({} rows) is larger than or equal to dataset size ({})."
                         .format(test_size, len(df_feats)))

    # Features are all columns except the target
    feature_cols = [c for c in df_feats.columns if c != TARGET_COL]

    X = df_feats[feature_cols]
    y = df_feats[TARGET_COL]

    # Time-based split: first part = train, last part = validation
    split_index = len(df_feats) - test_size

    X_train = X.iloc[:split_index]
    X_val = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_val = y.iloc[split_index:]

    return X_train, X_val, y_train, y_val, feature_cols


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

def plot_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot actual vs predicted target values on a common time index.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(y_true.values, label="Actual")
    plt.plot(y_pred, label="Predicted", alpha=0.8)
    plt.xlabel("Time index")
    plt.ylabel(TARGET_COL)
    plt.title("Actual vs Predicted {}".format(TARGET_COL))
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=120)
    plt.close()


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot feature importances for tree-based models (e.g., RandomForest, XGBoost).
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute.")

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError("Length of feature_importances_ does not match number of feature names.")

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    fi_top = fi.head(top_n)

    plt.figure(figsize=(8, 4))
    fi_top.plot(kind="bar")
    plt.ylabel("Importance")
    plt.title("Top {} feature importances".format(top_n))
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=120)
    plt.close()

