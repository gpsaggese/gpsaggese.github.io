"""
GluonTS Data Preparation Utilities.

Convert COVID-19 data for GluonTS models.

Import as:

import tutorials.tutorial_GluonTS_COVID19_Prediction.GluonTS_utils_gluonts as ttgcpgugl
"""

import logging
import warnings
from typing import List, Optional

import pandas as pd
from gluonts.dataset.common import ListDataset

warnings.filterwarnings("ignore")

_LOG = logging.getLogger(__name__)


def create_gluonts_dataset(
    df: pd.DataFrame,
    target_column: str,
    *,
    freq: str = "D",
    prediction_length: int = 14,
    past_feat_columns: Optional[List[str]] = None,
) -> ListDataset:
    """
    Convert pandas DataFrame to GluonTS ListDataset format.

    :param df: DataFrame with time series data
    :param target_column: Name of target column
    :param freq: Frequency of time series (default: daily)
    :param prediction_length: Forecast horizon
    :param past_feat_columns: Optional list of feature column names
    :return: GluonTS ListDataset
    """
    # Ensure we have a date column.
    if "Date" not in df.columns and "date" not in df.columns:
        raise ValueError("DataFrame must have a 'Date' or 'date' column")
    date_col = "Date" if "Date" in df.columns else "date"
    start_date = pd.to_datetime(df[date_col].iloc[0])
    target = df[target_column].values.tolist()
    # Create dataset entry.
    data_entry = {
        "start": start_date,
        "target": target,
    }
    # Add dynamic features if specified.
    if past_feat_columns:
        feat_dynamic_real = []
        for col in past_feat_columns:
            if col in df.columns:
                feat_dynamic_real.append(df[col].values.tolist())
            else:
                _LOG.warning("Column '%s' not found in DataFrame", col)
        if feat_dynamic_real:
            data_entry["feat_dynamic_real"] = feat_dynamic_real
    # Create ListDataset.
    dataset = ListDataset([data_entry], freq=freq)
    return dataset


def verify_dataset(
    dataset: ListDataset,
    *,
    name: str = "Dataset",
) -> None:
    """
    Verify and print information about a GluonTS dataset.

    :param dataset: GluonTS dataset to verify
    :param name: Name to display in output
    """
    try:
        data_list = list(dataset)
        _LOG.info("\n%s Dataset Info:", name)
        _LOG.info("=" * 50)
        _LOG.info("Valid GluonTS ListDataset")
        _LOG.info("  Number of time series: %s", len(data_list))
        if data_list:
            first_entry = data_list[0]
            _LOG.info("  Start date: %s", first_entry["start"])
            _LOG.info("  Target length: %s points", len(first_entry["target"]))
            if "feat_dynamic_real" in first_entry:
                n_features = len(first_entry["feat_dynamic_real"])
                _LOG.info("  Dynamic features: Yes (%s features)", n_features)
            else:
                _LOG.info("  Dynamic features: No")
        _LOG.info("=" * 50)
    except Exception as e:
        _LOG.error("Error verifying dataset: %s", e)


def prepare_train_test_split(
    full_df: pd.DataFrame,
    *,
    test_size: int = 14,
    target_column: str = "Daily_Cases_MA7",
) -> tuple:
    """
    Split DataFrame into train and test sets for time series.

    :param full_df: Complete DataFrame
    :param test_size: Number of days for test set
    :param target_column: Name of target column
    :return: Tuple of (train_df, test_df)
    """
    df_clean = full_df.dropna(subset=[target_column]).copy()
    split_idx = len(df_clean) - test_size
    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()
    _LOG.info("\nTrain/Test Split:")
    _LOG.info(
        "  Train: %s days (%s to %s)",
        len(train_df),
        train_df["Date"].min().date(),
        train_df["Date"].max().date(),
    )
    _LOG.info(
        "  Test:  %s days (%s to %s)",
        len(test_df),
        test_df["Date"].min().date(),
        test_df["Date"].max().date(),
    )
    return train_df, test_df


def get_feature_columns(
    df: pd.DataFrame,
    *,
    exclude_cols: List[str] = None,
) -> List[str]:
    """
    Get list of potential feature columns from DataFrame.

    :param df: DataFrame to extract features from
    :param exclude_cols: Columns to exclude
    :return: List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = ["Date", "date"]
    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols and df[col].dtype in ["int64", "float64"]
    ]
    return feature_cols


def summary_statistics(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Print summary statistics for the target variable.

    :param df: DataFrame with target column
    :param target_column: Name of target column
    """
    _LOG.info("\n%s Statistics:", target_column)
    _LOG.info("=" * 50)
    _LOG.info("  Count:  %s", df[target_column].count())
    _LOG.info("  Mean:   %.2f", df[target_column].mean())
    _LOG.info("  Median: %.2f", df[target_column].median())
    _LOG.info("  Std:    %.2f", df[target_column].std())
    _LOG.info("  Min:    %.2f", df[target_column].min())
    _LOG.info("  Max:    %.2f", df[target_column].max())
    _LOG.info("=" * 50)
