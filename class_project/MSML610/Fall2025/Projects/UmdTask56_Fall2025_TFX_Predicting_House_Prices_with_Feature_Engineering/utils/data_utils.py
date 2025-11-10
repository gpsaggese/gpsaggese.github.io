"""
Data utilities for the house price prediction pipeline

This module contains helper functions for:
- Loading and exploring data
- Data validation
- Data statistics and visualization
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

from . import config


def load_data(split: str = "train") -> pd.DataFrame:
    """
    Load train or test data.

    Args:
        split: Either "train" or "test"

    Returns:
        DataFrame containing the data
    """
    data_path = config.get_data_path(split)
    df = pd.read_csv(data_path)
    return df


def explore_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate basic statistics and information about the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary containing data exploration results
    """
    exploration = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "numeric_stats": df.describe().to_dict(),
    }
    return exploration


def get_feature_types(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Automatically detect numerical and categorical features.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (numerical_features, categorical_features)
    """
    numerical = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target if present
    if config.TARGET_COLUMN in numerical:
        numerical.remove(config.TARGET_COLUMN)

    return numerical, categorical


def test_data_ingestion():
    """Test function to verify data can be loaded."""
    print("Testing data ingestion...")

    # Load train data
    train_df = load_data("train")
    print(f"Train data loaded: {train_df.shape}")

    # Load test data
    test_df = load_data("test")
    print(f"Test data loaded: {test_df.shape}")

    # Explore train data
    exploration = explore_data(train_df)
    print(f"Train data has {exploration['shape'][1]} columns")
    print(f"Missing values in {len([k for k, v in exploration['missing_values'].items() if v > 0])} columns")

    print("Data ingestion test passed!")


if __name__ == "__main__":
    test_data_ingestion()
