"""
Utility functions for the Renewable Energy Forecasting project.

This module keeps all shared logic in one place so that scripts like
`make_features.py` and `train.py` can import and reuse it.

Typical flow:
1. Load raw solar data from data/raw/solar_energy.csv
2. Create time-based features for modeling
3. Save the processed dataset to data/processed/train.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

# Project root is the folder that contains this file.
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_raw_solar(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load the raw solar energy CSV as a pandas DataFrame.

    Parameters
    ----------
    path : str or Path, optional
        Path to the raw CSV. If not provided, defaults to
        data/raw/solar_energy.csv under the project root.

    Returns
    -------
    df : pandas.DataFrame
        Raw data with (if possible) a parsed datetime column.
    """
    if path is None:
        path = RAW_DIR / "solar_energy.csv"

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    df = pd.read_csv(path)

    # Try to automatically find a datetime column and parse it.
    # We don't know the exact column name from the assignment,
    # so we try a few common options.
    datetime_candidates = ["timestamp", "datetime", "date", "time"]

    for col in datetime_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.sort_values(col)
            df = df.set_index(col)
            break

    return df


# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------

def add_basic_time_features(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """
    Add simple time-based features (month, day_of_week, hour, etc.)
    to the solar energy dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Index is expected to be datetime if possible.
    target_col : str
        Name of the target column (e.g., 'generation' or 'power').

    Returns
    -------
    features : pandas.DataFrame
        A new dataframe containing features and the target.
    """
    # If the index is not datetime but we have a datetime column,
    # you could adapt this function later. For now we assume the
    # index is datetime after load_raw_solar() runs.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "Expected df.index to be a DatetimeIndex. "
            "Make sure load_raw_solar() parsed the datetime column."
        )

    features = df.copy()

    # Calendar time features
    features["year"] = features.index.year
    features["month"] = features.index.month
    features["day"] = features.index.day
    features["day_of_week"] = features.index.dayofweek  # Monday=0, Sunday=6
    features["hour"] = features.index.hour
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

    # We keep the target column as-is; later scripts will choose
    # which columns to use as X (features) and y (target).
    if target_col not in features.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataframe columns: "
            f"{list(features.columns)}"
        )

    return features


# -------------------------------------------------------------------
# Saving processed data
# -------------------------------------------------------------------

def save_processed(
    df: pd.DataFrame,
    filename: str = "train.csv",
) -> Path:
    """
    Save a processed dataframe to the processed data folder.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to save.
    filename : str
        Name of the CSV file to create (default: 'train.csv').

    Returns
    -------
    path : pathlib.Path
        Path where the file was saved.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=True)
    return path
