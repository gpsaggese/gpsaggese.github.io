"""
Data Preprocessing Utilities for COVID-19 Time Series.

Aggregate, clean, and merge COVID-19 data for time series forecasting with GluonTS.

Import as:

import tutorials.tutorial_GluonTS_COVID19_Prediction.utils_preprocessing as ttgcpgupr
"""

import logging

import pandas as pd

_LOG = logging.getLogger(__name__)


def aggregate_to_national(
    df: pd.DataFrame,
    *,
    data_type: str = "cases",
) -> pd.DataFrame:
    """
    Aggregate county-level data to national level.

    :param df: DataFrame with county-level data
    :param data_type: Type of data - 'cases' or 'deaths'
    :return: DataFrame with national-level aggregated data
    """
    # Deaths file has 12 metadata columns, cases has 11.
    skip_cols = 12 if data_type == "deaths" else 11
    date_columns = df.columns[skip_cols:]
    # Sum across all counties.
    national_cumulative = df[date_columns].sum()
    # Create DataFrame.
    prefix = "Cases" if data_type == "cases" else "Deaths"
    result_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(national_cumulative.index),
            f"Cumulative_{prefix}": national_cumulative.values,
        }
    )
    # Calculate daily values.
    result_df[f"Daily_{prefix}"] = (
        result_df[f"Cumulative_{prefix}"].diff().fillna(0)
    )
    # Calculate 7-day moving average.
    result_df[f"Daily_{prefix}_MA7"] = (
        result_df[f"Daily_{prefix}"].rolling(window=7).mean()
    )
    # Remove negative values (reporting corrections).
    result_df[f"Daily_{prefix}"] = result_df[f"Daily_{prefix}"].clip(lower=0)
    result_df[f"Daily_{prefix}_MA7"] = result_df[f"Daily_{prefix}_MA7"].clip(
        lower=0
    )
    return result_df


def extract_national_mobility(mobility_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract national-level mobility data from Google Mobility Reports.

    :param mobility_df: DataFrame with mobility data
    :return: DataFrame with national-level mobility data
    """
    # Filter for national level.
    national = mobility_df[
        (mobility_df["state"] == "Total") & (mobility_df["county"] == "Total")
    ].copy()
    national = national.sort_values("date").reset_index(drop=True)
    return national


def merge_all_data(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    mobility_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge cases, deaths, and mobility data into a single DataFrame.

    :param cases_df: DataFrame with cases data
    :param deaths_df: DataFrame with deaths data
    :param mobility_df: DataFrame with mobility data
    :return: Merged DataFrame with all data sources
    """
    # Ensure date columns are datetime.
    cases_df["Date"] = pd.to_datetime(cases_df["Date"])
    deaths_df["Date"] = pd.to_datetime(deaths_df["Date"])
    mobility_df["date"] = pd.to_datetime(mobility_df["date"])
    # Merge cases and deaths.
    merged = pd.merge(
        cases_df,
        deaths_df[
            ["Date", "Daily_Deaths", "Daily_Deaths_MA7", "Cumulative_Deaths"]
        ],
        on="Date",
        how="left",
    )
    # Merge with mobility.
    merged = pd.merge(
        merged,
        mobility_df,
        left_on="Date",
        right_on="date",
        how="left",
    )
    merged = merged.drop("date", axis=1)
    # Forward fill missing mobility values.
    mobility_cols = [
        "retail and recreation",
        "grocery and pharmacy",
        "parks",
        "transit stations",
        "workplaces",
        "residential",
    ]
    merged[mobility_cols] = merged[mobility_cols].fillna(method="ffill")
    # Fill missing deaths data.
    death_cols = ["Daily_Deaths", "Daily_Deaths_MA7", "Cumulative_Deaths"]
    merged[death_cols] = merged[death_cols].fillna(0)
    # Calculate case fatality ratio.
    merged["CFR"] = (
        merged["Cumulative_Deaths"] / merged["Cumulative_Cases"].replace(0, 1)
    ) * 100
    merged["CFR"] = merged["CFR"].fillna(0).clip(0, 100)
    return merged


def create_train_test_split(
    df: pd.DataFrame,
    *,
    test_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.

    :param df: DataFrame with time series data
    :param test_days: Number of days to use for test set
    :return: Tuple of (train_df, test_df)
    """
    df = df.dropna(subset=["Daily_Cases_MA7"])
    split_idx = len(df) - test_days
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def preprocess_pipeline(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    mobility_df: pd.DataFrame,
    *,
    test_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline from raw data to train/test split.

    :param cases_df: DataFrame with cases data
    :param deaths_df: DataFrame with deaths data
    :param mobility_df: DataFrame with mobility data
    :param test_days: Number of days to use for test set
    :return: Tuple of (merged_df, train_df, test_df)
    """
    _LOG.info("\n" + "=" * 70)
    _LOG.info("PREPROCESSING PIPELINE")
    _LOG.info("=" * 70)
    # Aggregate to national level.
    _LOG.info("\n[Step 1/4] Aggregating cases to national level...")
    national_cases = aggregate_to_national(cases_df, data_type="cases")
    _LOG.info("  National cases: %s days", len(national_cases))
    _LOG.info("[Step 2/4] Aggregating deaths to national level...")
    national_deaths = aggregate_to_national(deaths_df, data_type="deaths")
    _LOG.info("  National deaths: %s days", len(national_deaths))
    # Extract mobility.
    _LOG.info("[Step 3/4] Extracting national mobility data...")
    national_mobility = extract_national_mobility(mobility_df)
    _LOG.info("  National mobility: %s days", len(national_mobility))
    # Merge all data.
    _LOG.info("[Step 4/4] Merging all datasets...")
    merged_df = merge_all_data(
        national_cases, national_deaths, national_mobility
    )
    _LOG.info(
        "  Merged data: %s days, %s features",
        len(merged_df),
        len(merged_df.columns),
    )
    # Train/test split.
    _LOG.info("\nCreating train/test split (%s days for testing)...", test_days)
    train_df, test_df = create_train_test_split(merged_df, test_days=test_days)
    _LOG.info("  Train: %s days", len(train_df))
    _LOG.info("  Test: %s days", len(test_df))
    # Summary.
    _LOG.info("\n" + "=" * 70)
    _LOG.info("PREPROCESSING COMPLETE")
    _LOG.info("=" * 70)
    _LOG.info(
        "Date range: %s to %s",
        merged_df["Date"].min().date(),
        merged_df["Date"].max().date(),
    )
    _LOG.info(
        "Features: %s + mobility (6) + CFR",
        list(merged_df.columns[:7]),
    )
    _LOG.info("=" * 70 + "\n")
    return merged_df, train_df, test_df
