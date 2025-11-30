"""
Data utilities for the house price prediction pipeline

This module contains helper functions for:
- Loading and exploring data
- Data validation
- Data statistics and visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List

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


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in detail.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with missing value analysis
    """
    missing_analysis = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_pct': (df.isnull().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values
    })

    # Sort by missing percentage (descending)
    missing_analysis = missing_analysis[missing_analysis['missing_count'] > 0].sort_values(
        'missing_pct', ascending=False
    ).reset_index(drop=True)

    return missing_analysis


def get_categorical_summary(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Dict]:
    """
    Get summary statistics for categorical features.

    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names

    Returns:
        Dictionary with categorical feature summaries
    """
    summary = {}

    for col in categorical_cols:
        if col in df.columns:
            value_counts = df[col].value_counts()
            summary[col] = {
                'unique_count': df[col].nunique(),
                'top_values': value_counts.head(5).to_dict(),
                'missing_count': df[col].isnull().sum(),
                'missing_pct': df[col].isnull().sum() / len(df) * 100
            }

    return summary


def get_numerical_summary(df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
    """
    Get summary statistics for numerical features.

    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names

    Returns:
        DataFrame with numerical feature summaries
    """
    summary_stats = []

    for col in numerical_cols:
        if col in df.columns:
            stats = {
                'column': col,
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing_count': df[col].isnull().sum(),
                'missing_pct': df[col].isnull().sum() / len(df) * 100
            }
            summary_stats.append(stats)

    return pd.DataFrame(summary_stats)


def analyze_target_variable(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """
    Analyze the target variable (SalePrice).

    Args:
        df: Input DataFrame
        target_col: Name of target column (defaults to config.TARGET_COLUMN)

    Returns:
        Dictionary with target variable analysis
    """
    if target_col is None:
        target_col = config.TARGET_COLUMN

    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in DataFrame"}

    target = df[target_col].dropna()

    analysis = {
        'count': len(target),
        'mean': target.mean(),
        'median': target.median(),
        'std': target.std(),
        'min': target.min(),
        'max': target.max(),
        'q25': target.quantile(0.25),
        'q75': target.quantile(0.75),
        'skewness': target.skew(),
        'kurtosis': target.kurtosis(),
        'missing_count': df[target_col].isnull().sum()
    }

    return analysis


def identify_ordinal_features(df: pd.DataFrame) -> List[str]:
    """
    Identify ordinal features based on config.

    Args:
        df: Input DataFrame

    Returns:
        List of ordinal feature names present in the DataFrame
    """
    ordinal_features = []
    for feature in config.ORDINAL_FEATURES.keys():
        if feature in df.columns:
            ordinal_features.append(feature)

    return ordinal_features


def validate_data_schema(df: pd.DataFrame, split: str = "train") -> Dict[str, Any]:
    """
    Validate data against expected schema.

    Args:
        df: Input DataFrame
        split: Either "train" or "test"

    Returns:
        Dictionary with validation results
    """
    validation = {
        'split': split,
        'row_count': len(df),
        'column_count': len(df.columns),
        'issues': []
    }

    # Check for expected number of features
    if split == "train":
        expected_cols = 81  # 80 features + SalePrice
        if len(df.columns) != expected_cols:
            validation['issues'].append(
                f"Expected {expected_cols} columns, found {len(df.columns)}"
            )
    else:  # test
        expected_cols = 80  # 80 features (no SalePrice)
        if len(df.columns) != expected_cols:
            validation['issues'].append(
                f"Expected {expected_cols} columns, found {len(df.columns)}"
            )

    # Check if target column exists for train data
    if split == "train" and config.TARGET_COLUMN not in df.columns:
        validation['issues'].append(
            f"Target column '{config.TARGET_COLUMN}' not found in train data"
        )

    # Check for completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        validation['issues'].append(
            f"Completely empty columns: {empty_cols}"
        )

    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        validation['issues'].append(
            f"Found {duplicates} duplicate rows"
        )

    validation['is_valid'] = len(validation['issues']) == 0

    return validation


def generate_data_report(split: str = "train") -> Dict[str, Any]:
    """
    Generate comprehensive data report for train or test split.

    Args:
        split: Either "train" or "test"

    Returns:
        Dictionary containing comprehensive data analysis
    """
    # Load data
    df = load_data(split)

    # Get feature types
    numerical, categorical = get_feature_types(df)

    # Generate report
    report = {
        'split': split,
        'basic_info': {
            'shape': df.shape,
            'row_count': len(df),
            'column_count': len(df.columns)
        },
        'feature_counts': {
            'numerical': len(numerical),
            'categorical': len(categorical),
            'ordinal': len(identify_ordinal_features(df))
        },
        'missing_values': analyze_missing_values(df).to_dict('records'),
        'validation': validate_data_schema(df, split)
    }

    # Add target analysis for train data
    if split == "train" and config.TARGET_COLUMN in df.columns:
        report['target_analysis'] = analyze_target_variable(df)

    return report


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

    # Analyze missing values
    missing_analysis = analyze_missing_values(train_df)
    print(f"\nTop features with missing values:")
    print(missing_analysis.head())

    # Analyze target
    if config.TARGET_COLUMN in train_df.columns:
        target_stats = analyze_target_variable(train_df)
        print(f"\nTarget variable ({config.TARGET_COLUMN}) statistics:")
        print(f"  Mean: ${target_stats['mean']:,.2f}")
        print(f"  Median: ${target_stats['median']:,.2f}")
        print(f"  Skewness: {target_stats['skewness']:.4f}")

    # Validate schema
    validation = validate_data_schema(train_df, "train")
    print(f"\nData validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if not validation['is_valid']:
        print(f"  Issues: {validation['issues']}")

    print("\nData ingestion test passed!")


if __name__ == "__main__":
    test_data_ingestion()
