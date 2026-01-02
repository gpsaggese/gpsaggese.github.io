"""
utils_data_io.py

Small helper module for loading the raw Kaggle "Predict Future Sales" data
for the GraphQL E-commerce Sales Forecasting project.

All the CSVs are expected to live in:

    <project_root>/data/raw/

where <project_root> is the folder that contains this file.
"""

from pathlib import Path
from typing import Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    We assume this file lives directly inside the project root, so we can
    take the parent directory of this file.
    """
    return Path(__file__).resolve().parent


def get_raw_data_dir() -> Path:
    """
    Return the absolute path to the 'data/raw' directory.

    Raises a clear error message if the directory does not exist.
    """
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Expected raw data directory at {raw_dir}, "
            "but it does not exist. Did you create data/raw and "
            "copy the Kaggle CSVs there?"
        )

    return raw_dir


def _read_raw_csv(filename: str, **read_csv_kwargs) -> pd.DataFrame:
    """
    Internal helper: read a CSV from the raw data directory.

    Parameters
    ----------
    filename : str
        Name of the CSV file, e.g. 'sales_train.csv'.
    read_csv_kwargs : dict
        Extra keyword arguments forwarded to pandas.read_csv.

    Returns
    -------
    pandas.DataFrame
    """
    raw_dir = get_raw_data_dir()
    path = raw_dir / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {filename} in {raw_dir}. "
            "Check that you copied the Kaggle files correctly."
        )

    return pd.read_csv(path, **read_csv_kwargs)


# ---------------------------------------------------------------------------
# Public loader functions
# ---------------------------------------------------------------------------

def load_sales_train() -> pd.DataFrame:
    """
    Load the main 'sales_train.csv' file.

    This contains daily sales, with columns such as:
    - date
    - date_block_num
    - shop_id
    - item_id
    - item_price
    - item_cnt_day

    We parse the 'date' column as a proper datetime.
    """
    df = _read_raw_csv("sales_train.csv", parse_dates=["date"], dayfirst=True)
    # "dayfirst=True" because dates in this dataset are in DD.MM.YYYY format.
    return df


def load_items() -> pd.DataFrame:
    """
    Load 'items.csv', which maps item_id to item_name and item_category_id.
    """
    return _read_raw_csv("items.csv")


def load_item_categories() -> pd.DataFrame:
    """
    Load 'item_categories.csv', which maps item_category_id to
    item_category_name.
    """
    return _read_raw_csv("item_categories.csv")


def load_shops() -> pd.DataFrame:
    """
    Load 'shops.csv', which maps shop_id to shop_name.
    """
    return _read_raw_csv("shops.csv")


def load_test() -> pd.DataFrame:
    """
    Load 'test.csv', which contains the shop_id and item_id pairs
    for which we will eventually predict next-month sales.
    """
    return _read_raw_csv("test.csv")


def load_sample_submission() -> pd.DataFrame:
    """
    Load 'sample_submission.csv' if present. This file is optional
    for our local experiments but useful as a reference for the
    expected submission format.
    """
    try:
        return _read_raw_csv("sample_submission.csv")
    except FileNotFoundError:
        # Make it non-fatal if sample_submission is missing.
        return pd.DataFrame()


def load_raw_kaggle_data() -> Dict[str, pd.DataFrame]:
    """
    Convenience function: load all available raw Kaggle tables into a dict.

    Returns
    -------
    dict
        Keys are table names (strings) and values are pandas DataFrames.
        For example:

            {
                "sales_train": <DataFrame>,
                "items": <DataFrame>,
                "item_categories": <DataFrame>,
                "shops": <DataFrame>,
                "test": <DataFrame>,
                "sample_submission": <DataFrame> or empty DataFrame,
            }
    """
    data = {
        "sales_train": load_sales_train(),
        "items": load_items(),
        "item_categories": load_item_categories(),
        "shops": load_shops(),
        "test": load_test(),
    }

    sample_sub = load_sample_submission()
    if not sample_sub.empty:
        data["sample_submission"] = sample_sub

    return data
