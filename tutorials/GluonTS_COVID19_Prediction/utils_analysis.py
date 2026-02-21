"""
Analysis utilities for COVID-19 time series data.

Feature correlation, data quality checks, and exploratory analysis.

Import as:

import utils_analysis as analysis
"""

from typing import List, Optional

import pandas as pd


def analyze_feature_correlation(
    df: pd.DataFrame,
    target_col: str = "Daily_Cases_MA7",
    features: Optional[List[str]] = None,
    *,
    bar_char: str = "#",
    bar_width: int = 20,
) -> pd.Series:
    """
    Compute and print correlations between features and the target variable.

    :param df: DataFrame with features and target
    :param target_col: Target column name
    :param features: List of feature columns. If None, uses all numeric columns
        except target and Date.
    :param bar_char: Character for correlation bar (default: '#')
    :param bar_width: Scale factor for bar length (default: 20)
    :return: Series of correlations (feature -> correlation value)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    if features is None:
        exclude = {"Date", "date", target_col}
        features = [
            c
            for c in df.columns
            if c not in exclude and df[c].dtype in ("int64", "float64")
        ]

    available = [c for c in features + [target_col] if c in df.columns]
    if target_col not in available:
        available.append(target_col)
    feature_data = df[available].copy()
    corr_series = feature_data.corr()[target_col]
    if target_col in corr_series.index:
        corr_series = corr_series.drop(target_col)
    correlations = corr_series.sort_values(ascending=False)

    print(" Feature Correlations with Daily Cases:")
    print("=" * 70)
    for feat, corr in correlations.items():
        bar = bar_char * int(abs(corr) * bar_width)
        sign = "+" if corr > 0 else "-"
        print(f"  {feat:35s} [{sign}] {bar} {corr:+.3f}")

    print("\n Interpretation:")
    print(" • Positive correlation: Feature increases with cases")
    print(" • Negative correlation: Feature decreases when cases rise")
    print(" • Magnitude: Strength of relationship")

    return correlations


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in each column.

    :param df: DataFrame to check
    :return: DataFrame with column, missing_count, missing_pct
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values found.")
        return pd.DataFrame()

    result = pd.DataFrame(
        {
            "column": missing.index,
            "missing_count": missing.values,
            "missing_pct": (missing.values / len(df) * 100).round(2),
        }
    )
    print("Missing Values:")
    print("=" * 50)
    for _, row in result.iterrows():
        print(f"  {row['column']:30s} {row['missing_count']:6d} ({row['missing_pct']:.1f}%)")
    print("=" * 50)
    return result


def check_data_quality(
    df: pd.DataFrame,
    *,
    target_col: str = "Daily_Cases_MA7",
    date_col: str = "Date",
) -> None:
    """
    Run basic data quality checks: missing values, date range, target stats.

    :param df: DataFrame to check
    :param target_col: Target column for basic stats
    :param date_col: Date column name
    """
    print("\nData Quality Summary")
    print("=" * 70)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    if date_col in df.columns:
        df_date = pd.to_datetime(df[date_col])
        print(f"  Date range: {df_date.min().date()} to {df_date.max().date()}")

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  None")
    else:
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            print(f"  {col}: {cnt} ({pct:.1f}%)")

    if target_col in df.columns:
        t = df[target_col].dropna()
        print(f"\nTarget ({target_col}) stats:")
        print(f"  Count: {t.count():,}")
        print(f"  Mean:  {t.mean():.2f}")
        print(f"  Min:   {t.min():.2f}")
        print(f"  Max:   {t.max():.2f}")

    print("=" * 70)
