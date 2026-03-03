"""
Consolidated utilities for COVID-19 time series forecasting with GluonTS.

Sections:
- Analysis: feature correlation, data quality checks
- Data I/O: load cases, deaths, vaccines, mobility; DataLoader
- Data download: download from Google Drive
- GluonTS: create_gluonts_dataset, prepare_train_test_split
- Preprocessing: aggregate, merge, train/test split
- Notebook loader: load_covid_data_for_gluonts, quick_load_*
- Models: DeepAR, SimpleFeedForward, DeepNPTS, scenario analysis
- Evaluation: calculate_metrics, print_metrics, plot_forecast
- Synthetic: generators, prepare_synthetic_dataset
- Visualization: plotting for data, forecasts, comparisons

Import as:

    import GluonTS_utils
    from GluonTS_utils import load_covid_data_for_gluonts, train_deepar_covid
"""

import logging
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.torch.model.deep_npts import DeepNPTSEstimator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator

warnings.filterwarnings("ignore")

_LOG = logging.getLogger(__name__)

_DEFAULT_START = "2020-01-01"


# #############################################################################
# Analysis
# #############################################################################


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


# #############################################################################
# Data I/O
# #############################################################################


def load_jhu_cases(
    *,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load JHU CSSE COVID-19 cases data from CSV.

    :param data_dir: Directory containing data files
    :return: DataFrame with cases data
    """
    filepath = Path(data_dir) / "cases.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"cases.csv not found in {data_dir}")
    _LOG.info("Loading cases data from %s", filepath)
    df = pd.read_csv(filepath)
    _LOG.info("Loaded %s rows, %s columns", len(df), len(df.columns))
    return df


def load_jhu_deaths(
    *,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load JHU CSSE COVID-19 deaths data from CSV.

    :param data_dir: Directory containing data files
    :return: DataFrame with deaths data
    """
    filepath = Path(data_dir) / "deaths.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"deaths.csv not found in {data_dir}")
    _LOG.info("Loading deaths data from %s", filepath)
    df = pd.read_csv(filepath)
    _LOG.info("Loaded %s rows, %s columns", len(df), len(df.columns))
    return df


def load_jhu_vaccines(
    *,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load JHU CSSE COVID-19 vaccine data from CSV.

    :param data_dir: Directory containing data files
    :return: DataFrame with vaccine data
    """
    filepath = Path(data_dir) / "vaccine.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"vaccine.csv not found in {data_dir}")
    _LOG.info("Loading vaccine data from %s", filepath)
    df = pd.read_csv(filepath)
    _LOG.info("Loaded %s rows, %s columns", len(df), len(df.columns))
    return df


def load_google_mobility(
    *,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load Google COVID-19 Community Mobility Reports from CSV.

    :param data_dir: Directory containing data files
    :return: DataFrame with mobility data
    """
    filepath = Path(data_dir) / "mobility.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"mobility.csv not found in {data_dir}")
    _LOG.info("Loading mobility data from %s", filepath)
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    _LOG.info("Loaded %s rows, %s columns", len(df), len(df.columns))
    _LOG.info("  Date range: %s to %s", df["date"].min(), df["date"].max())
    return df


def load_all_data(
    *,
    data_dir: str = "data",
) -> Dict[str, pd.DataFrame]:
    """
    Load all COVID-19 datasets at once.

    :param data_dir: Directory containing data files
    :return: Dictionary with keys 'cases', 'deaths', 'vaccines', 'mobility'
    """
    _LOG.info("Loading all COVID-19 datasets...")
    _LOG.info("=" * 60)
    data = {}
    data["cases"] = load_jhu_cases(data_dir=data_dir)
    data["deaths"] = load_jhu_deaths(data_dir=data_dir)
    data["vaccines"] = load_jhu_vaccines(data_dir=data_dir)
    data["mobility"] = load_google_mobility(data_dir=data_dir)
    _LOG.info("=" * 60)
    _LOG.info("All datasets loaded successfully")
    return data


def verify_data_exists(
    *,
    data_dir: str = "data",
) -> bool:
    """
    Verify that all required data files exist.

    :param data_dir: Directory containing data files
    :return: True if all files exist, False otherwise
    """
    required_files = ["cases.csv", "deaths.csv", "vaccine.csv", "mobility.csv"]
    data_path = Path(data_dir)
    missing_files = []
    for filename in required_files:
        if not (data_path / filename).exists():
            missing_files.append(filename)
    if missing_files:
        _LOG.info("Missing files: %s", ", ".join(missing_files))
        _LOG.info("Expected location: %s", data_path.absolute())
        return False
    _LOG.info("All required data files present in %s", data_dir)
    return True


class DataLoader:
    """Convenience class for loading COVID-19 data."""

    def __init__(
        self,
        *,
        data_dir: str = "data",
    ):
        """
        Initialize DataLoader.

        :param data_dir: Directory containing data files
        """
        self.data_dir = data_dir

    def load_cases(self) -> pd.DataFrame:
        """Load cases data."""
        return load_jhu_cases(data_dir=self.data_dir)

    def load_deaths(self) -> pd.DataFrame:
        """Load deaths data."""
        return load_jhu_deaths(data_dir=self.data_dir)

    def load_vaccines(self) -> pd.DataFrame:
        """Load vaccines data."""
        return load_jhu_vaccines(data_dir=self.data_dir)

    def load_mobility(self) -> pd.DataFrame:
        """Load mobility data."""
        return load_google_mobility(data_dir=self.data_dir)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets."""
        return load_all_data(data_dir=self.data_dir)


# #############################################################################
# Data download
# #############################################################################


def _get_confirm_token(response):
    """
    Extract confirmation token from response headers.

    :param response: HTTP response object
    :return: Confirmation token or None
    """
    for key, value in response.headers.items():
        if key.lower() == "set-cookie":
            if "download_warning" in value:
                return value.split("download_warning=")[1].split(";")[0]
    return None


def download_file_from_google_drive(
    file_id: str,
    destination: Path,
) -> bool:
    """
    Download file from Google Drive.

    :param file_id: Google Drive file ID
    :param destination: Local path to save file
    :return: True if successful, False otherwise
    """
    _LOG.info("Downloading %s... ", destination.name)
    try:
        URL = "https://drive.google.com/uc?export=download"
        session = urllib.request.build_opener()
        response = session.open(f"{URL}&id={file_id}")
        token = _get_confirm_token(response)
        if token:
            response = session.open(f"{URL}&id={file_id}&confirm={token}")
        with open(destination, "wb") as f:
            f.write(response.read())
        _LOG.info("Done")
        return True
    except Exception as e:
        _LOG.error("Failed: %s", e)
        return False


def check_and_download_data(
    *,
    data_dir: str = "data",
) -> bool:
    """
    Check if data files exist, download if missing.

    :param data_dir: Directory containing data files
    :return: True if all files present or downloaded successfully
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    drive_files = {
        "cases.csv": "1ZfZtoV3PpZblZYES0A5LHCwp54cR8RJL",
        "deaths.csv": "1kYC9nrCnKbNpnoZKz8o6TDMM371gyxbl",
        "mobility.csv": "1TMqG8Z8vbxmQAv1rNKczYYPCzwT4ZS_q",
    }
    existing_files = []
    missing_files = []
    for filename in drive_files.keys():
        file_path = data_path / filename
        if file_path.exists():
            existing_files.append(filename)
            _LOG.info("Found: %s", filename)
        else:
            missing_files.append(filename)
    if not missing_files:
        _LOG.info("\nAll data files present!")
        return True
    _LOG.info("\nMissing files: %s", ", ".join(missing_files))
    _LOG.info("\nAttempting to download from Google Drive...")
    downloaded = []
    failed = []
    for filename in missing_files:
        file_id = drive_files[filename]
        file_path = data_path / filename
        if file_id:
            if download_file_from_google_drive(file_id, file_path):
                downloaded.append(filename)
            else:
                failed.append(filename)
        else:
            failed.append(filename)
    if downloaded:
        _LOG.info("\nSuccessfully downloaded: %s", ", ".join(downloaded))
    if failed:
        _LOG.info("\nSome files need manual download:")
        _LOG.info(
            "Visit: https://drive.google.com/drive/folders/1qMDGBstdY8H2hYpz8xSolhzNOsVxNHMA"
        )
        file_mapping = {
            "cases.csv": "time_series_covid19_confirmed_US.csv",
            "deaths.csv": "time_series_covid19_deaths_US.csv",
            "mobility.csv": "mobility_report_US.csv",
        }
        _LOG.info("\nDownload these files and save to 'data/' directory:")
        for local_name in failed:
            if local_name in file_mapping:
                drive_name = file_mapping[local_name]
                _LOG.info("  - %s → rename to '%s'", drive_name, local_name)
        return False
    return True


# #############################################################################
# GluonTS
# #############################################################################


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
    if "Date" not in df.columns and "date" not in df.columns:
        raise ValueError("DataFrame must have a 'Date' or 'date' column")
    date_col = "Date" if "Date" in df.columns else "date"
    start_date = pd.to_datetime(df[date_col].iloc[0])
    target = df[target_column].values.tolist()
    data_entry = {
        "start": start_date,
        "target": target,
    }
    if past_feat_columns:
        feat_dynamic_real = []
        for col in past_feat_columns:
            if col in df.columns:
                feat_dynamic_real.append(df[col].values.tolist())
            else:
                _LOG.warning("Column '%s' not found in DataFrame", col)
        if feat_dynamic_real:
            data_entry["feat_dynamic_real"] = feat_dynamic_real
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


# #############################################################################
# Preprocessing
# #############################################################################


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
    skip_cols = 12 if data_type == "deaths" else 11
    date_columns = df.columns[skip_cols:]
    national_cumulative = df[date_columns].sum()
    prefix = "Cases" if data_type == "cases" else "Deaths"
    result_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(national_cumulative.index),
            f"Cumulative_{prefix}": national_cumulative.values,
        }
    )
    result_df[f"Daily_{prefix}"] = (
        result_df[f"Cumulative_{prefix}"].diff().fillna(0)
    )
    result_df[f"Daily_{prefix}_MA7"] = (
        result_df[f"Daily_{prefix}"].rolling(window=7).mean()
    )
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
    cases_df["Date"] = pd.to_datetime(cases_df["Date"])
    deaths_df["Date"] = pd.to_datetime(deaths_df["Date"])
    mobility_df["date"] = pd.to_datetime(mobility_df["date"])
    merged = pd.merge(
        cases_df,
        deaths_df[
            ["Date", "Daily_Deaths", "Daily_Deaths_MA7", "Cumulative_Deaths"]
        ],
        on="Date",
        how="left",
    )
    merged = pd.merge(
        merged,
        mobility_df,
        left_on="Date",
        right_on="date",
        how="left",
    )
    merged = merged.drop("date", axis=1)
    mobility_cols = [
        "retail and recreation",
        "grocery and pharmacy",
        "parks",
        "transit stations",
        "workplaces",
        "residential",
    ]
    merged[mobility_cols] = merged[mobility_cols].ffill()
    death_cols = ["Daily_Deaths", "Daily_Deaths_MA7", "Cumulative_Deaths"]
    merged[death_cols] = merged[death_cols].fillna(0)
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


def train_test_split_covid(
    merged_df: pd.DataFrame,
    *,
    train_end_date: Optional[str] = None,
    pred_length: int = 14,
    test_length: int = 14,
    target_col: str = "Daily_Cases_MA7",
    past_feat_columns: Optional[List[str]] = None,
) -> tuple[Any, Any, pd.DataFrame, pd.DataFrame]:
    """
    Split merged COVID data into train/test and convert to GluonTS datasets.

    :param merged_df: Merged DataFrame (cases + deaths + mobility)
    :param train_end_date: Optional end date for training (YYYY-MM-DD).
    :param pred_length: Forecast horizon for GluonTS (default: 14)
    :param test_length: Number of days for test set (default: 14)
    :param target_col: Target column name (default: Daily_Cases_MA7)
    :param past_feat_columns: Optional feature columns for GluonTS.
    :return: Tuple of (train_ds, test_ds, train_df, test_df)
    """
    if past_feat_columns is None:
        past_feat_columns = ["Daily_Deaths_MA7", "Cumulative_Deaths", "CFR"]

    df_clean = merged_df.dropna(subset=[target_col]).copy()
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])

    if train_end_date is not None:
        end_dt = pd.to_datetime(train_end_date)
        train_df = df_clean[df_clean["Date"] <= end_dt].reset_index(drop=True)
        test_start = end_dt + pd.Timedelta(days=1)
        test_end = end_dt + pd.Timedelta(days=test_length)
        test_df = df_clean[
            (df_clean["Date"] >= test_start) & (df_clean["Date"] <= test_end)
        ].reset_index(drop=True)
    else:
        train_df, test_df = prepare_train_test_split(
            df_clean,
            test_size=test_length,
            target_column=target_col,
        )

    feat_cols = [c for c in past_feat_columns if c in merged_df.columns]
    full_df = merged_df.dropna(subset=[target_col])

    train_ds = create_gluonts_dataset(
        train_df,
        target_column=target_col,
        freq="D",
        prediction_length=pred_length,
        past_feat_columns=feat_cols if feat_cols else None,
    )
    test_ds = create_gluonts_dataset(
        full_df,
        target_column=target_col,
        freq="D",
        prediction_length=pred_length,
        past_feat_columns=feat_cols if feat_cols else None,
    )
    return train_ds, test_ds, train_df, test_df


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
    _LOG.info("\n[Step 1/4] Aggregating cases to national level...")
    national_cases = aggregate_to_national(cases_df, data_type="cases")
    _LOG.info("  National cases: %s days", len(national_cases))
    _LOG.info("[Step 2/4] Aggregating deaths to national level...")
    national_deaths = aggregate_to_national(deaths_df, data_type="deaths")
    _LOG.info("  National deaths: %s days", len(national_deaths))
    _LOG.info("[Step 3/4] Extracting national mobility data...")
    national_mobility = extract_national_mobility(mobility_df)
    _LOG.info("  National mobility: %s days", len(national_mobility))
    _LOG.info("[Step 4/4] Merging all datasets...")
    merged_df = merge_all_data(
        national_cases, national_deaths, national_mobility
    )
    _LOG.info(
        "  Merged data: %s days, %s features",
        len(merged_df),
        len(merged_df.columns),
    )
    _LOG.info("\nCreating train/test split (%s days for testing)...", test_days)
    train_df, test_df = create_train_test_split(merged_df, test_days=test_days)
    _LOG.info("  Train: %s days", len(train_df))
    _LOG.info("  Test: %s days", len(test_df))
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


# #############################################################################
# Notebook loader
# #############################################################################


def load_covid_data_for_gluonts(
    *,
    data_dir: str = "data",
    target_column: str = "Daily_Cases_MA7",
    test_size: int = 14,
    prediction_length: int = 14,
    use_features: bool = True,
    feature_subset: str = "minimal",
) -> Dict:
    """
    One-stop function to load US COVID-19 data and prepare for GluonTS.

    :param data_dir: Directory containing CSV files
    :param target_column: Column to forecast (default: 'Daily_Cases_MA7')
    :param test_size: Days for testing (default: 14)
    :param prediction_length: Forecast horizon (default: 14)
    :param use_features: Include exogenous features (default: True)
    :param feature_subset: "minimal", "moderate", or "full"
    :return: Dictionary with train_ds, test_ds, DataFrames, and metadata
    """
    if not check_and_download_data(data_dir=data_dir):
        raise FileNotFoundError(
            f"Required data files missing from '{data_dir}/' directory. "
            "Please download them manually as instructed above."
        )
    _LOG.info("=" * 70)
    _LOG.info("COVID-19 DATA LOADER")
    _LOG.info("=" * 70)
    _LOG.info("\nLoading raw data...")
    loader = DataLoader(data_dir=data_dir)
    try:
        cases_df = loader.load_cases()
        deaths_df = loader.load_deaths()
        mobility_df = loader.load_mobility()
        _LOG.info("Data files loaded (cases, deaths, mobility)")
    except Exception as e:
        _LOG.error("Error loading data: %s", e)
        _LOG.error("Make sure data files exist in '%s/' folder", data_dir)
        raise
    _LOG.info("\nPreprocessing...")
    national_cases = aggregate_to_national(cases_df, data_type="cases")
    national_deaths = aggregate_to_national(deaths_df, data_type="deaths")
    national_mobility = extract_national_mobility(mobility_df)
    _LOG.info("\nMerging data sources...")
    merged_df = merge_all_data(
        national_cases, national_deaths, national_mobility
    )
    _LOG.info("Merged data: %s days", len(merged_df))
    _LOG.info(
        "Date range: %s to %s",
        merged_df["Date"].min().date(),
        merged_df["Date"].max().date(),
    )
    _LOG.info("\nFeature selection: %s", feature_subset)
    if not use_features:
        feature_columns = None
        _LOG.info("Using target only (no exogenous features)")
    else:
        if feature_subset == "minimal":
            feature_columns = [
                "Daily_Deaths_MA7",
                "Cumulative_Deaths",
                "CFR",
            ]
        elif feature_subset == "moderate":
            feature_columns = [
                "Daily_Deaths_MA7",
                "CFR",
                "retail_and_recreation_percent_change_from_baseline",
                "grocery_and_pharmacy_percent_change_from_baseline",
                "workplaces_percent_change_from_baseline",
                "residential_percent_change_from_baseline",
            ]
        else:
            exclude = [
                "Date",
                target_column,
                "Daily_Cases",
                "Cumulative_Cases",
                "Daily_Deaths",
                "Cumulative_Deaths",
            ]
            feature_columns = [
                col
                for col in merged_df.columns
                if col not in exclude
                and merged_df[col].dtype in ["int64", "float64"]
            ]
        _LOG.info("Selected %s features:", len(feature_columns))
        for i, feat in enumerate(feature_columns[:5], 1):
            _LOG.info("  %s. %s", i, feat)
        if len(feature_columns) > 5:
            _LOG.info("  ... and %s more", len(feature_columns) - 5)
    _LOG.info("\nSplitting data (test size: %s days)...", test_size)
    train_df, test_df = prepare_train_test_split(
        merged_df,
        test_size=test_size,
        target_column=target_column,
    )
    _LOG.info("\nConverting to GluonTS format...")
    train_ds = create_gluonts_dataset(
        df=train_df,
        target_column=target_column,
        freq="D",
        prediction_length=prediction_length,
        past_feat_columns=feature_columns,
    )
    test_ds = create_gluonts_dataset(
        df=merged_df.dropna(subset=[target_column]),
        target_column=target_column,
        freq="D",
        prediction_length=prediction_length,
        past_feat_columns=feature_columns,
    )
    _LOG.info("GluonTS datasets created")
    _LOG.info(
        "Note: Test dataset contains full time series (train + test periods)"
    )
    info = {
        "total_days": len(merged_df),
        "train_days": len(train_df),
        "test_days": len(test_df),
        "date_range": f"{merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}",
        "target_column": target_column,
        "num_features": len(feature_columns) if feature_columns else 0,
        "feature_subset": feature_subset,
    }
    _LOG.info("\n" + "=" * 70)
    _LOG.info("DATA READY FOR TRAINING")
    _LOG.info("=" * 70)
    _LOG.info("\nSummary:")
    _LOG.info("  Target: %s", target_column)
    _LOG.info("  Features: %s (%s)", info["num_features"], feature_subset)
    _LOG.info("  Train: %s days", info["train_days"])
    _LOG.info("  Test: %s days", info["test_days"])
    _LOG.info("  Prediction length: %s days", prediction_length)
    _LOG.info("=" * 70)
    return {
        "train_ds": train_ds,
        "test_ds": test_ds,
        "train_df": train_df,
        "test_df": test_df,
        "merged_df": merged_df,
        "target": target_column,
        "features": feature_columns,
        "info": info,
    }


def quick_load_minimal() -> Dict:
    """Quickest load - minimal features, good for testing."""
    return load_covid_data_for_gluonts(feature_subset="minimal")


def quick_load_moderate() -> Dict:
    """Moderate features - balanced speed and accuracy."""
    return load_covid_data_for_gluonts(feature_subset="moderate")


def quick_load_full() -> Dict:
    """All features - maximum information."""
    return load_covid_data_for_gluonts(feature_subset="full")


# #############################################################################
# Models
# #############################################################################


@dataclass
class ModelResults:
    """
    Container for model training and evaluation results.
    """

    model_name: str
    predictor: object
    forecasts: List
    ground_truths: List
    metrics: Dict
    training_time: float = 0.0


def train_deepar_covid(
    train_ds,
    test_ds,
    *,
    prediction_length: int = 14,
    num_feat_dynamic_real: int = 0,
    epochs: int = 20,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    num_layers: int = 2,
    hidden_size: int = 40,
    dropout: float = 0.1,
    verbose: bool = True,
) -> ModelResults:
    """
    Train a DeepAR model on COVID-19 data.

    :param train_ds: training dataset
    :param test_ds: test dataset
    :param prediction_length: number of time steps to predict
    :param num_feat_dynamic_real: number of dynamic real features
    :param epochs: number of training epochs
    :param learning_rate: learning rate for optimizer
    :param context_length: number of time steps to use as context
    :param num_layers: number of RNN layers
    :param hidden_size: hidden layer size
    :param dropout: dropout rate
    :param verbose: whether to print progress information
    :return: ModelResults containing trained model and evaluation metrics
    """
    if verbose:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("TRAINING DeepAR MODEL")
        _LOG.info("=" * 70)
        _LOG.info("\nConfiguration:")
        _LOG.info("  Epochs: %s", epochs)
        _LOG.info(
            "  Context length: %s", context_length or prediction_length * 2
        )
        _LOG.info("  Features: %s", num_feat_dynamic_real)
        _LOG.info("  Hidden size: %s", hidden_size)
        _LOG.info("  Layers: %s", num_layers)
    start_time = time.time()
    estimator = DeepAREstimator(
        freq="D",
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout,
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={"max_epochs": epochs},
    )
    if verbose:
        _LOG.info("\nTraining in progress...")
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    if verbose:
        _LOG.info("\nTraining complete in %.1f seconds", training_time)
    if verbose:
        _LOG.info("\nGenerating probabilistic forecasts...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    if verbose:
        _LOG.info("\nDeepAR Performance:")
        _LOG.info("  MAPE: %.2f%%", agg_metrics.get("MAPE", 0))
        _LOG.info("  RMSE: %.2f", agg_metrics.get("RMSE", 0))
        _LOG.info("  MAE: %.2f", agg_metrics.get("MAE", 0))
        _LOG.info("=" * 70)
    return ModelResults(
        model_name="DeepAR",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time,
    )


def train_feedforward_covid(
    train_ds,
    test_ds,
    *,
    prediction_length: int = 14,
    epochs: int = 100,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    hidden_dimensions: Optional[List[int]] = None,
    verbose: bool = True,
) -> ModelResults:
    """
    Train a SimpleFeedForward model on COVID-19 data.

    :param train_ds: training dataset
    :param test_ds: test dataset
    :param prediction_length: number of time steps to predict
    :param epochs: number of training epochs
    :param learning_rate: learning rate for optimizer
    :param context_length: number of time steps to use as context
    :param hidden_dimensions: list of hidden layer dimensions
    :param verbose: whether to print progress information
    :return: ModelResults containing trained model and evaluation metrics
    """
    if verbose:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("TRAINING SimpleFeedForward MODEL")
        _LOG.info("=" * 70)
        _LOG.info("\nNote: This model doesn't use external features.")
        _LOG.info("\nConfiguration:")
        _LOG.info("  Epochs: %s", epochs)
        _LOG.info(
            "  Context length: %s", context_length or prediction_length * 2
        )
        _LOG.info("  Hidden layers: %s", hidden_dimensions or [40, 40])
    start_time = time.time()
    estimator = SimpleFeedForwardEstimator(
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        hidden_dimensions=hidden_dimensions or [40, 40],
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={"max_epochs": epochs},
    )
    if verbose:
        _LOG.info("\nTraining in progress...")
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    if verbose:
        _LOG.info("\nTraining complete in %.1f seconds", training_time)
    if verbose:
        _LOG.info("\nGenerating probabilistic forecasts...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    if verbose:
        _LOG.info("\nSimpleFeedForward Performance:")
        _LOG.info("  MAPE: %.2f%%", agg_metrics.get("MAPE", 0))
        _LOG.info("  RMSE: %.2f", agg_metrics.get("RMSE", 0))
        _LOG.info("  MAE: %.2f", agg_metrics.get("MAE", 0))
        _LOG.info("=" * 70)
    return ModelResults(
        model_name="SimpleFeedForward",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time,
    )


def train_deepnpts_covid(
    train_ds,
    test_ds,
    *,
    prediction_length: int = 14,
    num_feat_dynamic_real: int = 0,
    epochs: int = 30,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    num_hidden_nodes: Optional[List[int]] = None,
    dropout_rate: float = 0.1,
    verbose: bool = True,
) -> ModelResults:
    """
    Train a DeepNPTS model on COVID-19 data.

    :param train_ds: training dataset
    :param test_ds: test dataset
    :param prediction_length: number of time steps to predict
    :param num_feat_dynamic_real: number of dynamic real features
    :param epochs: number of training epochs
    :param learning_rate: learning rate for optimizer
    :param context_length: number of time steps to use as context
    :param num_hidden_nodes: list of hidden node sizes
    :param dropout_rate: dropout rate
    :param verbose: whether to print progress information
    :return: ModelResults containing trained model and evaluation metrics
    """
    if verbose:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("TRAINING DeepNPTS MODEL")
        _LOG.info("=" * 70)
        _LOG.info("\nConfiguration:")
        _LOG.info("  Epochs: %s", epochs)
        _LOG.info(
            "  Context length: %s", context_length or prediction_length * 2
        )
        _LOG.info("  Features: %s", num_feat_dynamic_real)
        _LOG.info("  Hidden nodes: %s", num_hidden_nodes or [40])
        _LOG.info("  Dropout: %s", dropout_rate)
    start_time = time.time()
    estimator = DeepNPTSEstimator(
        freq="D",
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_hidden_nodes=num_hidden_nodes or [40],
        dropout_rate=dropout_rate,
        epochs=epochs,
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50,
    )
    if verbose:
        _LOG.info("\nTraining in progress...")
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    if verbose:
        _LOG.info("\nTraining complete in %.1f seconds", training_time)
    if verbose:
        _LOG.info("\nGenerating probabilistic forecasts...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    if verbose:
        _LOG.info("\nDeepNPTS Performance:")
        _LOG.info("  MAPE: %.2f%%", agg_metrics.get("MAPE", 0))
        _LOG.info("  RMSE: %.2f", agg_metrics.get("RMSE", 0))
        _LOG.info("  MAE: %.2f", agg_metrics.get("MAE", 0))
        _LOG.info("=" * 70)
    return ModelResults(
        model_name="DeepNPTS",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time,
    )


def compare_models(results_list: List[ModelResults]) -> pd.DataFrame:
    """
    Create a comparison table of multiple trained models.

    :param results_list: list of ModelResults to compare
    :return: DataFrame with model comparison metrics
    """
    comparison_data = []
    for results in results_list:
        comparison_data.append(
            {
                "Model": results.model_name,
                "MAPE (%)": results.metrics.get("MAPE", np.nan),
                "RMSE": results.metrics.get("RMSE", np.nan),
                "MAE": results.metrics.get("MAE", np.nan),
                "Training Time (s)": results.training_time,
            }
        )
    df = pd.DataFrame(comparison_data)
    df = df.sort_values("MAPE (%)")
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Pretty print the model comparison table.

    :param comparison_df: DataFrame with model comparison metrics
    """
    _LOG.info("\n" + "=" * 80)
    _LOG.info("MODEL COMPARISON - COVID-19 FORECASTING")
    _LOG.info("=" * 80)
    _LOG.info("\nWhich model performed best?\n")
    _LOG.info(comparison_df.to_string(index=False))
    _LOG.info("\n" + "=" * 80)
    winner = comparison_df.iloc[0]
    _LOG.info(
        "\nWinner: %s with MAPE of %.2f%%", winner["Model"], winner["MAPE (%)"]
    )
    _LOG.info("=" * 80 + "\n")


def get_forecast_dataframe(
    forecast,
    ground_truth,
    start_date: pd.Timestamp,
    *,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Convert GluonTS forecast and ground truth to a convenient DataFrame.
    """
    forecast_length = len(forecast.mean)
    dates = pd.date_range(start=start_date, periods=forecast_length, freq=freq)
    return pd.DataFrame(
        {
            "Date": dates,
            "Prediction": forecast.mean,
            "Actual": ground_truth[-forecast_length:],
            "Lower_10": forecast.quantile(0.1),
            "Lower_25": forecast.quantile(0.25),
            "Median": forecast.quantile(0.5),
            "Upper_75": forecast.quantile(0.75),
            "Upper_90": forecast.quantile(0.9),
        }
    )


@dataclass
class ScenarioResult:
    """
    Container for scenario analysis results.
    """

    name: str
    description: str
    forecast: Any
    mean_daily_cases: float
    total_cases: float
    lower_bound: float
    upper_bound: float
    adjustments: Dict[str, float] = field(default_factory=dict)

    def cases_vs_baseline(self, baseline_total: float) -> Tuple[float, float]:
        """Calculate difference from baseline scenario."""
        diff = self.total_cases - baseline_total
        pct_diff = (diff / baseline_total) * 100 if baseline_total != 0 else 0
        return diff, pct_diff


def create_scenario_dataset(
    merged_df: pd.DataFrame,
    feature_columns: List[str],
    *,
    target_column: str = "Daily_Cases_MA7",
    mobility_adjustment: float = 1.0,
    cfr_adjustment: float = 1.0,
    deaths_adjustment: float = 1.0,
    prediction_length: int = 14,
    freq: str = "D",
) -> ListDataset:
    """
    Create a modified GluonTS dataset for scenario analysis.
    """
    df = merged_df.copy()
    mobility_cols = [
        "retail and recreation",
        "grocery and pharmacy",
        "parks",
        "transit stations",
        "workplaces",
        "residential",
    ]
    cfr_cols = ["CFR"]
    deaths_cols = ["Daily_Deaths_MA7", "Cumulative_Deaths", "Daily_Deaths"]
    forecast_start_idx = len(df) - prediction_length
    for col in df.columns:
        if col in mobility_cols and mobility_adjustment != 1.0:
            df.loc[df.index[forecast_start_idx:], col] = (
                df.loc[df.index[forecast_start_idx:], col] * mobility_adjustment
            )
        elif col in cfr_cols and cfr_adjustment != 1.0:
            df.loc[df.index[forecast_start_idx:], col] = (
                df.loc[df.index[forecast_start_idx:], col] * cfr_adjustment
            )
        elif col in deaths_cols and deaths_adjustment != 1.0:
            df.loc[df.index[forecast_start_idx:], col] = (
                df.loc[df.index[forecast_start_idx:], col] * deaths_adjustment
            )
    df_clean = df.dropna(subset=[target_column]).copy()
    date_col = "Date" if "Date" in df_clean.columns else "date"
    start_date = pd.to_datetime(df_clean[date_col].iloc[0])
    target = df_clean[target_column].values.tolist()
    data_entry = {"start": start_date, "target": target}
    if feature_columns:
        feat_dynamic_real = []
        for col in feature_columns:
            if col in df_clean.columns:
                feat_dynamic_real.append(df_clean[col].values.tolist())
        if feat_dynamic_real:
            data_entry["feat_dynamic_real"] = feat_dynamic_real
    return ListDataset([data_entry], freq=freq)


def run_scenario_forecast(
    predictor,
    scenario_dataset: ListDataset,
    scenario_name: str,
    scenario_description: str,
    adjustments: Dict[str, float],
    *,
    num_samples: int = 100,
) -> ScenarioResult:
    """Run a forecast for a specific scenario using a trained predictor."""
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=scenario_dataset, predictor=predictor, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    forecast = forecasts[0]
    mean_daily = float(forecast.mean.mean())
    total_cases = float(forecast.mean.sum())
    lower = float(forecast.quantile(0.1).mean())
    upper = float(forecast.quantile(0.9).mean())
    return ScenarioResult(
        name=scenario_name,
        description=scenario_description,
        forecast=forecast,
        mean_daily_cases=mean_daily,
        total_cases=total_cases,
        lower_bound=lower,
        upper_bound=upper,
        adjustments=adjustments,
    )


def run_all_scenarios(
    predictor,
    merged_df: pd.DataFrame,
    feature_columns: List[str],
    *,
    target_column: str = "Daily_Cases_MA7",
    prediction_length: int = 14,
    verbose: bool = True,
) -> List[ScenarioResult]:
    """Run all predefined scenarios and return results."""
    scenarios_config = [
        {
            "name": "Baseline",
            "description": "No intervention - current trends continue",
            "mobility": 1.0,
            "cfr": 1.0,
            "deaths": 1.0,
        },
        {
            "name": "Moderate Intervention",
            "description": "15% mobility reduction (masks, capacity limits)",
            "mobility": 0.85,
            "cfr": 1.0,
            "deaths": 1.0,
        },
        {
            "name": "Strong Intervention",
            "description": "30% mobility reduction (lockdowns, closures)",
            "mobility": 0.70,
            "cfr": 1.0,
            "deaths": 1.0,
        },
        {
            "name": "Relaxation",
            "description": "20% mobility increase (reopening, holidays)",
            "mobility": 1.20,
            "cfr": 1.0,
            "deaths": 1.0,
        },
        {
            "name": "Healthcare Strain",
            "description": "15% higher CFR (hospital capacity stressed)",
            "mobility": 1.0,
            "cfr": 1.15,
            "deaths": 1.10,
        },
    ]
    results = []
    if verbose:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("RUNNING SCENARIO ANALYSIS")
        _LOG.info("=" * 70)
    for i, config in enumerate(scenarios_config, 1):
        if verbose:
            _LOG.info(
                "\n[%s/5] %s: %s", i, config["name"], config["description"]
            )
        scenario_ds = create_scenario_dataset(
            merged_df=merged_df,
            feature_columns=feature_columns,
            target_column=target_column,
            mobility_adjustment=config["mobility"],
            cfr_adjustment=config["cfr"],
            deaths_adjustment=config["deaths"],
            prediction_length=prediction_length,
        )
        adjustments = {
            "mobility": config["mobility"],
            "cfr": config["cfr"],
            "deaths": config["deaths"],
        }
        result = run_scenario_forecast(
            predictor=predictor,
            scenario_dataset=scenario_ds,
            scenario_name=config["name"],
            scenario_description=config["description"],
            adjustments=adjustments,
        )
        results.append(result)
        if verbose:
            _LOG.info(
                "   Avg daily: %,.0f | Total: %,.0f",
                result.mean_daily_cases,
                result.total_cases,
            )
    if verbose:
        _LOG.info("\nScenario analysis complete.")
    return results


def print_scenario_summary(results: List[ScenarioResult]) -> pd.DataFrame:
    """Print a formatted summary table of all scenario results."""
    baseline = next((r for r in results if r.name == "Baseline"), results[0])
    baseline_total = baseline.total_cases
    summary_data = []
    for result in results:
        diff, pct = result.cases_vs_baseline(baseline_total)
        summary_data.append(
            {
                "Scenario": result.name,
                "Avg Daily Cases": result.mean_daily_cases,
                "Total Cases (14d)": result.total_cases,
                "Range (10%-90%)": f"{result.lower_bound:,.0f} - {result.upper_bound:,.0f}",
                "vs Baseline": f"{pct:+.1f}%"
                if result.name != "Baseline"
                else "--",
                "Cases Δ": f"{diff:+,.0f}"
                if result.name != "Baseline"
                else "--",
            }
        )
    df = pd.DataFrame(summary_data)
    _LOG.info("\n" + "=" * 90)
    _LOG.info("SCENARIO COMPARISON SUMMARY")
    _LOG.info("=" * 90)
    _LOG.info("\nForecast horizon: 14 days")
    _LOG.info("Baseline total cases: %,.0f", baseline_total)
    _LOG.info("")
    _LOG.info(
        "%s %s %s %s %s",
        "Scenario".ljust(25),
        "Avg Daily".ljust(12),
        "Total Cases".ljust(14),
        "vs Baseline".ljust(12),
        "Cases Δ".ljust(15),
    )
    _LOG.info("-" * 90)
    for _, row in df.iterrows():
        _LOG.info(
            "%s %10,.0f %12,.0f %10s %14s",
            row["Scenario"].ljust(25),
            row["Avg Daily Cases"],
            row["Total Cases (14d)"],
            row["vs Baseline"].rjust(10),
            row["Cases Δ"].rjust(14),
        )
    _LOG.info("=" * 90)
    return df


def print_policy_insights(results: List[ScenarioResult]) -> None:
    """Print policy insights comparing intervention impact vs baseline/relaxation."""
    baseline = next((r for r in results if r.name == "Baseline"), results[0])
    strong_intervention = next(
        (r for r in results if r.name == "Strong Intervention"), None
    )
    relaxation = next(
        (r for r in results if r.name == "Relaxation"), None
    )
    cases_prevented = (
        baseline.total_cases - strong_intervention.total_cases
        if strong_intervention
        else 0
    )
    additional_cases = (
        relaxation.total_cases - baseline.total_cases if relaxation else 0
    )
    print("POLICY INSIGHTS")
    print("=" * 70)
    print()
    print("Intervention Impact:")
    if strong_intervention:
        pct = (cases_prevented / baseline.total_cases) * 100
        print(
            f"  Strong intervention (30% mobility reduction) could prevent"
        )
        print(f"  ~{cases_prevented:,.0f} cases over 14 days ({pct:.1f}% reduction)")
    else:
        print("  Strong intervention scenario not found.")
    print()
    print("Relaxation Risk:")
    if relaxation:
        pct = (additional_cases / baseline.total_cases) * 100
        print(
            f"  Lifting restrictions (20% mobility increase) could add"
        )
        print(f"  ~{additional_cases:,.0f} cases over 14 days ({pct:.1f}% increase)")
    else:
        print("  Relaxation scenario not found.")
    print()
    print("Caveats:")
    print("  - These are model projections, not guarantees")
    print("  - Correlation does not imply causation")
    print("  - Use to inform discussion, not dictate policy")
    print("=" * 70)


def plot_scenario_comparison(
    results: list,
    *,
    prediction_length: int = 14,
    save_path: Optional[str] = None,
) -> None:
    """
    Create visualizations comparing all scenarios.

    Expects objects with .name, .forecast (mean, quantile), .total_cases,
    .cases_vs_baseline() - e.g. ScenarioResult from models.

    :param results: list of scenario result objects
    :param prediction_length: forecast horizon for x-axis
    :param save_path: optional path to save the figure
    """
    colors = {
        "Baseline": "#6B7280",
        "Moderate Intervention": "#3B82F6",
        "Strong Intervention": "#10B981",
        "Relaxation": "#F59E0B",
        "Healthcare Strain": "#EF4444",
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1 = axes[0]
    days = list(range(1, prediction_length + 1))
    for result in results:
        color = colors.get(result.name, "#6B7280")
        forecast = result.forecast
        ax1.plot(
            days, forecast.mean, label=result.name, color=color, linewidth=2.5
        )
        ax1.fill_between(
            days,
            forecast.quantile(0.1),
            forecast.quantile(0.9),
            alpha=0.15,
            color=color,
        )
    ax1.set_xlabel("Days Ahead", fontsize=12)
    ax1.set_ylabel("Daily Cases", fontsize=12)
    ax1.set_title(
        "Forecast Trajectories by Scenario", fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, prediction_length + 1, 2))

    ax2 = axes[1]
    names = [r.name for r in results]
    totals = [r.total_cases for r in results]
    bar_colors = [colors.get(n, "#6B7280") for n in names]
    bars = ax2.barh(names, totals, color=bar_colors, alpha=0.8)
    baseline_total = next(
        (r.total_cases for r in results if r.name == "Baseline"), totals[0]
    )
    for bar, result in zip(bars, results):
        width = bar.get_width()
        diff, pct = result.cases_vs_baseline(baseline_total)
        ax2.text(
            width + baseline_total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:,.0f}",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )
        if result.name != "Baseline":
            pct_text = f"({pct:+.1f}%)"
            ax2.text(
                width + baseline_total * 0.08,
                bar.get_y() + bar.get_height() / 2,
                pct_text,
                ha="left",
                va="center",
                fontsize=9,
                color="green" if pct < 0 else "red",
            )
    ax2.set_xlabel("Total Cases (14 days)", fontsize=12)
    ax2.set_title("Total Cases by Scenario", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.axvline(
        x=baseline_total,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Baseline",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _LOG.info("Saved scenario comparison plot to: %s", save_path)
    plt.show()


# #############################################################################
# Evaluation
# #############################################################################


def calculate_metrics(
    forecast_values: Union[np.ndarray, pd.Series, list],
    actual_values: Union[np.ndarray, pd.Series, list],
) -> Dict[str, float]:
    """
    Calculate comprehensive forecasting metrics.

    :param forecast_values: Forecasted values
    :param actual_values: Actual observed values
    :return: Dictionary with MAE, RMSE, MAPE, ME, and max_error
    """
    forecast_values = np.asarray(forecast_values).flatten()
    actual_values = np.asarray(actual_values).flatten()
    if len(forecast_values) != len(actual_values):
        min_len = min(len(forecast_values), len(actual_values))
        forecast_values = forecast_values[:min_len]
        actual_values = actual_values[:min_len]
    errors = forecast_values - actual_values
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / actual_values)) * 100
    me = np.mean(errors)
    max_error = np.max(np.abs(errors))
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "me": me,
        "max_error": max_error,
    }


def print_metrics(
    metrics: Dict[str, float],
    *,
    model_name: str = "Model",
) -> None:
    """Print metrics in a formatted way."""
    _LOG.info("\n%s Performance:", model_name)
    _LOG.info("=" * 60)
    _LOG.info("MAE (Mean Absolute Error):      %10,.2f", metrics["mae"])
    _LOG.info("RMSE (Root Mean Squared Error): %10,.2f", metrics["rmse"])
    _LOG.info("MAPE (Mean Abs. %% Error):       %10.2f %%", metrics["mape"])
    _LOG.info("ME (Mean Error / Bias):         %10,.2f", metrics["me"])
    _LOG.info("Maximum Error:                   %10,.2f", metrics["max_error"])
    _LOG.info("=" * 60)
    if metrics["mape"] < 10:
        _LOG.info("\nExcellent performance, error less than 10%%")
    elif metrics["mape"] < 20:
        _LOG.info("\nGood performance, error less than 20%%")
    else:
        _LOG.info("\nModerate performance (COVID data is highly variable)")
    if abs(metrics["me"]) < metrics["mae"] / 2:
        _LOG.info("Low bias (not systematically over or under-predicting)")
    else:
        bias_direction = "over" if metrics["me"] > 0 else "under"
        _LOG.info("Model tends to %s-predict", bias_direction)


def plot_forecast(
    train_df: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    actual_values: np.ndarray,
    forecast_quantiles: Dict[float, np.ndarray],
    target_column: str,
    model_name: str,
    *,
    save_path: str = None,
    context_days: int = 60,
) -> None:
    """Create a comprehensive forecast visualization."""
    plt.figure(figsize=(16, 6))
    train_context = train_df.tail(context_days)
    plt.plot(
        train_context["Date"],
        train_context[target_column],
        label="Historical Data",
        color="steelblue",
        linewidth=2,
        alpha=0.8,
    )
    plt.plot(
        forecast_dates,
        actual_values,
        label="Actual",
        color="orange",
        linewidth=3,
        marker="o",
        markersize=8,
        zorder=5,
    )
    plt.plot(
        forecast_dates,
        forecast_values,
        label=f"{model_name} Forecast",
        color="red",
        linewidth=3,
        marker="s",
        markersize=7,
        linestyle="--",
        zorder=4,
    )
    if 0.05 in forecast_quantiles and 0.95 in forecast_quantiles:
        plt.fill_between(
            forecast_dates,
            forecast_quantiles[0.05],
            forecast_quantiles[0.95],
            alpha=0.15,
            color="red",
            label="90% Confidence",
        )
    if 0.25 in forecast_quantiles and 0.75 in forecast_quantiles:
        plt.fill_between(
            forecast_dates,
            forecast_quantiles[0.25],
            forecast_quantiles[0.75],
            alpha=0.25,
            color="red",
            label="50% Confidence",
        )
    plt.title(
        f"{model_name} Forecast Visualization",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Date", fontsize=13)
    plt.ylabel(target_column.replace("_", " "), fontsize=13)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _LOG.info("Plot saved as '%s'", save_path)
    plt.show()


def plot_error_analysis(
    forecast_values: np.ndarray,
    actual_values: np.ndarray,
    forecast_quantiles: Dict[float, np.ndarray],
    model_name: str,
    *,
    save_path: str = None,
) -> None:
    """Create detailed error analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    forecast_period = len(forecast_values)
    errors = forecast_values - actual_values
    axes[0, 0].plot(
        range(1, forecast_period + 1),
        actual_values,
        "o-",
        label="Actual",
        color="orange",
        linewidth=2,
        markersize=8,
    )
    axes[0, 0].plot(
        range(1, forecast_period + 1),
        forecast_values,
        "s--",
        label="Forecast",
        color="red",
        linewidth=2,
        markersize=7,
    )
    if 0.1 in forecast_quantiles and 0.9 in forecast_quantiles:
        axes[0, 0].fill_between(
            range(1, forecast_period + 1),
            forecast_quantiles[0.1],
            forecast_quantiles[0.9],
            alpha=0.2,
            color="red",
        )
    axes[0, 0].set_title("Forecast vs Actual", fontweight="bold")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    colors = ["red" if e > 0 else "green" for e in errors]
    axes[0, 1].bar(range(1, forecast_period + 1), errors, color=colors)
    axes[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Daily Forecast Errors", fontweight="bold")
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].set_ylabel("Error (Forecast - Actual)")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    ape = np.abs(errors / actual_values) * 100
    mape = np.mean(ape)
    axes[1, 0].bar(
        range(1, forecast_period + 1),
        ape,
        color="steelblue",
        alpha=0.7,
    )
    axes[1, 0].axhline(
        y=mape,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean APE: {mape:.1f}%",
    )
    axes[1, 0].set_title("Absolute Percentage Error by Day", fontweight="bold")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Absolute % Error")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    if 0.1 in forecast_quantiles and 0.9 in forecast_quantiles:
        ci_width = forecast_quantiles[0.9] - forecast_quantiles[0.1]
        axes[1, 1].plot(
            range(1, forecast_period + 1),
            ci_width,
            "o-",
            color="purple",
            linewidth=2,
            markersize=8,
        )
        axes[1, 1].set_title(
            "Forecast Uncertainty (80% CI Width)",
            fontweight="bold",
        )
        axes[1, 1].set_xlabel("Day")
        axes[1, 1].set_ylabel("CI Width")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Quantiles not available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Uncertainty Analysis", fontweight="bold")
    plt.suptitle(
        f"{model_name} Error Analysis",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _LOG.info("Error analysis saved as '%s'", save_path)
    plt.show()


def compare_models_metrics(
    results: Dict[str, Dict[str, float]],
    *,
    save_path: str = None,
) -> None:
    """
    Compare multiple models side by side (bar chart + print table).

    :param results: Dictionary mapping model names to their metrics
    :param save_path: Optional path to save plot
    """
    metrics_to_plot = ["mae", "rmse", "mape"]
    model_names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, metric in enumerate(metrics_to_plot):
        values = [results[model][metric] for model in model_names]
        axes[idx].bar(
            model_names,
            values,
            color=["steelblue", "green", "purple"][: len(model_names)],
        )
        axes[idx].set_title(metric.upper(), fontweight="bold")
        axes[idx].set_ylabel(metric.upper())
        axes[idx].grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(values):
            axes[idx].text(i, v, f"{v:.1f}", ha="center", va="bottom")
    plt.suptitle("Model Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _LOG.info("Comparison saved as '%s'", save_path)
    plt.show()
    _LOG.info("\nModel Comparison Table:")
    _LOG.info("=" * 70)
    _LOG.info("%-20s %12s %12s %12s", "Model", "MAE", "RMSE", "MAPE")
    _LOG.info("-" * 70)
    for model, metrics in results.items():
        _LOG.info(
            "%-20s %12,.2f %12,.2f %11.2f%%",
            model,
            metrics["mae"],
            metrics["rmse"],
            metrics["mape"],
        )
    _LOG.info("=" * 70)
    best_model = min(results.items(), key=lambda x: x[1]["mape"])
    _LOG.info(
        "\nBest Model (by MAPE): %s (%.2f%%)",
        best_model[0],
        best_model[1]["mape"],
    )


# #############################################################################
# Synthetic
# #############################################################################


def generate_sinusoid(
    n_points: int = 365,
    *,
    period: int = 30,
    amplitude: float = 10.0,
    baseline: float = 50.0,
    noise_std: float = 1.0,
    seed: int = 42,
    start_date: str = _DEFAULT_START,
) -> pd.DataFrame:
    """Pure sine wave with additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    signal = baseline + amplitude * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0, noise_std, n_points)
    dates = pd.date_range(start=start_date, periods=n_points, freq="D")
    return pd.DataFrame({"Date": dates, "value": signal + noise})


def generate_multi_frequency(
    n_points: int = 365,
    *,
    trend_slope: float = 0.02,
    seasonal_period: int = 30,
    seasonal_amplitude: float = 8.0,
    weekly_amplitude: float = 3.0,
    baseline: float = 50.0,
    noise_std: float = 1.5,
    seed: int = 42,
    start_date: str = _DEFAULT_START,
) -> pd.DataFrame:
    """Combination of trend, seasonal, weekly cycle, and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    trend = baseline + trend_slope * t
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
    weekly = weekly_amplitude * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, noise_std, n_points)
    dates = pd.date_range(start=start_date, periods=n_points, freq="D")
    return pd.DataFrame({"Date": dates, "value": trend + seasonal + weekly + noise})


def generate_regime_change(
    n_points: int = 365,
    *,
    changepoint_frac: float = 0.5,
    baseline_before: float = 50.0,
    amplitude_before: float = 5.0,
    period_before: int = 30,
    baseline_after: float = 80.0,
    amplitude_after: float = 12.0,
    period_after: int = 15,
    noise_std: float = 1.5,
    seed: int = 42,
    start_date: str = _DEFAULT_START,
) -> pd.DataFrame:
    """Time series that changes behavior at a configurable changepoint."""
    rng = np.random.default_rng(seed)
    cp = int(n_points * changepoint_frac)
    t_before = np.arange(cp)
    t_after = np.arange(n_points - cp)
    before = baseline_before + amplitude_before * np.sin(
        2 * np.pi * t_before / period_before
    )
    after = baseline_after + amplitude_after * np.sin(
        2 * np.pi * t_after / period_after
    )
    signal = np.concatenate([before, after])
    noise = rng.normal(0, noise_std, n_points)
    dates = pd.date_range(start=start_date, periods=n_points, freq="D")
    return pd.DataFrame({"Date": dates, "value": signal + noise})


def prepare_synthetic_dataset(
    df: pd.DataFrame,
    *,
    target_col: str = "value",
    prediction_length: int = 30,
    freq: str = "D",
) -> Dict:
    """
    Split a synthetic DataFrame into train/test and convert to GluonTS format.

    :param df: DataFrame with Date and target columns
    :param target_col: name of the target column
    :param prediction_length: forecast horizon (also used as test size)
    :param freq: time series frequency
    :return: dict with train_ds, test_ds, train_df, test_df, and metadata
    """
    date_col = "Date" if "Date" in df.columns else "date"
    df = df.copy().sort_values(date_col).reset_index(drop=True)
    split_idx = len(df) - prediction_length
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    start = pd.to_datetime(df[date_col].iloc[0])
    train_ds = ListDataset(
        [{"start": start, "target": train_df[target_col].values}],
        freq=freq,
    )
    test_ds = ListDataset(
        [{"start": start, "target": df[target_col].values}],
        freq=freq,
    )
    info = {
        "n_points": len(df),
        "train_points": len(train_df),
        "test_points": len(test_df),
        "prediction_length": prediction_length,
        "start_date": str(start.date()),
        "train_end": str(train_df[date_col].iloc[-1].date()),
        "test_start": str(test_df[date_col].iloc[0].date()),
        "test_end": str(test_df[date_col].iloc[-1].date()),
    }
    _LOG.info("Prepared synthetic dataset:")
    _LOG.info("  Train: %s points (%s to %s)",
              info["train_points"], info["start_date"], info["train_end"])
    _LOG.info("  Test:  %s points (%s to %s)",
              info["test_points"], info["test_start"], info["test_end"])
    return {
        "train_ds": train_ds,
        "test_ds": test_ds,
        "train_df": train_df,
        "test_df": test_df,
        "target": target_col,
        "prediction_length": prediction_length,
        "info": info,
    }


def plot_synthetic_series(
    df: pd.DataFrame,
    *,
    target_col: str = "value",
    title: str = "Synthetic Time Series",
    figsize: tuple = (14, 4),
) -> None:
    """Quick visualization of a synthetic series."""
    date_col = "Date" if "Date" in df.columns else "date"
    plt.figure(figsize=figsize)
    plt.plot(df[date_col], df[target_col], linewidth=1.2, color="steelblue")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_train_test_split(
    data: Dict,
    *,
    title: str = "Train / Test Split",
    figsize: tuple = (14, 4),
) -> None:
    """Visualize the train/test split from prepare_synthetic_dataset."""
    target_col = data["target"]
    date_col = "Date" if "Date" in data["train_df"].columns else "date"
    plt.figure(figsize=figsize)
    plt.plot(
        data["train_df"][date_col],
        data["train_df"][target_col],
        label="Train",
        color="steelblue",
        linewidth=1.2,
    )
    plt.plot(
        data["test_df"][date_col],
        data["test_df"][target_col],
        label="Test",
        color="orange",
        linewidth=1.2,
    )
    plt.axvline(
        x=data["test_df"][date_col].iloc[0],
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Split point",
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forecast_result(
    data: Dict,
    forecast_entry,
    *,
    model_name: str = "Model",
    context_points: int = 60,
    figsize: tuple = (14, 5),
) -> None:
    """Plot forecast against actuals with confidence intervals."""
    target_col = data["target"]
    date_col = "Date" if "Date" in data["train_df"].columns else "date"
    train_tail = data["train_df"].tail(context_points)
    test_dates = data["test_df"][date_col].values
    actuals = data["test_df"][target_col].values
    pred_mean = forecast_entry.mean
    plt.figure(figsize=figsize)
    plt.plot(
        train_tail[date_col], train_tail[target_col],
        label="History", color="steelblue", linewidth=1.2,
    )
    plt.plot(
        test_dates, actuals,
        label="Actual", color="orange", linewidth=2, marker="o", markersize=4,
    )
    plt.plot(
        test_dates[:len(pred_mean)], pred_mean,
        label=f"{model_name} forecast", color="red",
        linewidth=2, linestyle="--", marker="s", markersize=4,
    )
    q_low = forecast_entry.quantile(0.1)
    q_high = forecast_entry.quantile(0.9)
    plt.fill_between(
        test_dates[:len(q_low)], q_low, q_high,
        alpha=0.15, color="red", label="80% interval",
    )
    plt.title(f"{model_name} Forecast", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# #############################################################################
# Visualization
# #############################################################################


def plot_data_overview(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    *,
    date_col: str = "Date",
    title: str = "US COVID-19 Cases: 7-Day Moving Average",
    ylabel: str = "Daily Cases (7-day avg)",
) -> None:
    """Plot training and test data with forecast boundary line."""
    plt.figure(figsize=(14, 5))
    plt.plot(
        train_df[date_col],
        train_df[target_col],
        label="Training Data",
        color="steelblue",
        linewidth=1.5,
    )
    plt.plot(
        test_df[date_col],
        test_df[target_col],
        label="Test Data (Future)",
        color="coral",
        linewidth=1.5,
    )
    plt.axvline(
        x=train_df[date_col].iloc[-1],
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Today (Forecast Start)",
    )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("\n Note that there are multiple peaks and troughs in the case data.")
    print(
        "\n This makes sense as there were multiple rounds of vaccinations and Covid variants."
    )


def plot_forecast_with_confidence_intervals(
    train_df: pd.DataFrame,
    data: Dict[str, Any],
    actual: np.ndarray,
    forecast: Any,
    model_name: str,
    *,
    color: str = "forestgreen",
    context_days: int = 90,
    prediction_length: int = 14,
    ylabel: str = "Daily Cases (7-day avg)",
) -> tuple:
    """Plot forecast with historical context and confidence intervals."""
    target_col = data["target"]
    train_dates = train_df["Date"].values[-context_days:]
    train_values = train_df[target_col].values[-context_days:]
    last_train_date = pd.Timestamp(train_dates[-1])
    forecast_dates = pd.date_range(
        start=last_train_date + pd.Timedelta(days=1),
        periods=prediction_length,
        freq="D",
    )
    actual_values = actual[-prediction_length:]
    plt.figure(figsize=(14, 6))
    plt.plot(
        train_dates, train_values, label="Historical", color="steelblue", linewidth=2
    )
    plt.plot(
        forecast_dates,
        actual_values,
        label="Actual Future",
        color="coral",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        forecast_dates,
        forecast.mean,
        label=f"{model_name} Forecast",
        color=color,
        linewidth=2.5,
        marker="s",
        markersize=5,
        linestyle="--",
    )
    plt.fill_between(
        forecast_dates,
        forecast.quantile(0.1),
        forecast.quantile(0.9),
        alpha=0.3,
        color=color,
        label="80% Confidence",
    )
    plt.fill_between(
        forecast_dates,
        forecast.quantile(0.05),
        forecast.quantile(0.95),
        alpha=0.2,
        color=color,
        label="90% Confidence",
    )
    plt.axvline(
        x=last_train_date, color="red", linestyle="--", linewidth=1.5, alpha=0.7
    )
    plt.title(f"{model_name}: COVID-19 Case Forecasting", fontsize=16, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    if "DeepAR" in model_name:
        print("\n Observe that DeepAR captures the trend and provides uncertainty bounds.")
    elif "SimpleFeedForward" in model_name:
        print("\n SimpleFeedForward gives a smooth baseline forecast!")
    elif "DeepNPTS" in model_name:
        print("\n DeepNPTS adapts to the data's natural distribution!")
    return train_dates, train_values, forecast_dates, actual_values


def plot_data_exploration(merged_df: pd.DataFrame) -> None:
    """Plot cases, deaths, and mobility in 3-panel layout."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    axes[0].plot(
        merged_df["Date"],
        merged_df["Daily_Cases_MA7"],
        linewidth=2,
        color="#2E86AB",
    )
    axes[0].set_title(
        " COVID-19 Daily Cases (7-Day Moving Average)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].set_ylabel("Cases", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        merged_df["Date"],
        merged_df["Daily_Deaths_MA7"],
        linewidth=2,
        color="#A23B72",
    )
    axes[1].set_title(
        " COVID-19 Daily Deaths (7-Day Moving Average)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_ylabel("Deaths", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(
        merged_df["Date"],
        merged_df["workplaces"],
        linewidth=2,
        color="#F18F01",
    )
    axes[2].set_title(
        " Workplace Mobility (% change from baseline)",
        fontsize=14,
        fontweight="bold",
    )
    axes[2].set_ylabel("% Change", fontsize=12)
    axes[2].set_xlabel("Date", fontsize=12)
    axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("\n Key Observations:")
    print(" • Multiple distinct waves of cases visible")
    print(" • Deaths follow cases with a lag")
    print(" • Mobility patterns shifted dramatically during lockdowns")
    print(" • These patterns provide valuable signals for forecasting!")


def plot_model_comparison_3panel(
    deepar_results: Any,
    feedforward_results: Any,
    deepnpts_results: Any,
) -> None:
    """Plot 3-panel forecast comparison for DeepAR, SimpleFeedForward, DeepNPTS."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    models = [
        (deepar_results, "DeepAR", "#2E86AB"),
        (feedforward_results, "SimpleFeedForward", "#A23B72"),
        (deepnpts_results, "DeepNPTS", "#F18F01"),
    ]
    for idx, (results, name, color) in enumerate(models):
        ax = axes[idx]
        forecast = results.forecasts[0]
        actual = results.ground_truths[0]
        history_len = len(actual) - len(forecast.mean)
        ax.plot(
            range(history_len),
            actual[:history_len],
            label="Historical",
            color="gray",
            alpha=0.6,
            linewidth=2,
        )
        ax.plot(
            range(history_len, len(actual)),
            actual[history_len:],
            label="Actual",
            color="black",
            linewidth=2,
        )
        forecast_range = range(history_len, history_len + len(forecast.mean))
        ax.plot(
            forecast_range,
            forecast.mean,
            label="Forecast",
            color=color,
            linewidth=2,
            linestyle="--",
        )
        ax.fill_between(
            forecast_range,
            forecast.quantile(0.1),
            forecast.quantile(0.9),
            alpha=0.3,
            color=color,
            label="80% CI",
        )
        ax.set_title(f"{name} Forecast", fontsize=14, fontweight="bold")
        ax.set_ylabel("Daily Cases", fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel("Days", fontsize=12)
    plt.tight_layout()
    plt.show()
    print("\n Visual Insights:")
    print(" • All models capture the general trend")
    print(" • Confidence intervals show forecast uncertainty")
    print(" • Compare forecast accuracy against actual values (black line)")


def print_model_comparison_from_metrics(
    deepar_metrics: Dict[str, float],
    ff_metrics: Dict[str, float],
    npts_metrics: Dict[str, float],
) -> None:
    """Print model comparison table and winner from three metrics dicts."""
    comparison = pd.DataFrame(
        [
            {
                "Model": "DeepAR",
                "MAPE (%)": deepar_metrics["mape"],
                "MAE": deepar_metrics["mae"],
                "RMSE": deepar_metrics["rmse"],
            },
            {
                "Model": "SimpleFeedForward",
                "MAPE (%)": ff_metrics["mape"],
                "MAE": ff_metrics["mae"],
                "RMSE": ff_metrics["rmse"],
            },
            {
                "Model": "DeepNPTS",
                "MAPE (%)": npts_metrics["mape"],
                "MAE": npts_metrics["mae"],
                "RMSE": npts_metrics["rmse"],
            },
        ]
    )
    comparison = comparison.sort_values("MAPE (%)")
    comparison.insert(0, "Rank", [1, 2, 3])
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON")
    print("=" * 70)
    print(comparison.to_string(index=False))
    print("=" * 70)
    winner = comparison.iloc[0]["Model"]
    print(f"\n Winner: {winner}!")
    print("\nNote: These results are from a quick training demo.")
    print("Ideally, you should train with more epochs and tune hyperparameters!")


def plot_metrics_comparison_barplot(
    results_dict: Dict[str, Dict[str, float]],
    *,
    metrics: Optional[list] = None,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing metrics across models."""
    if metrics is None:
        metrics = ["mae", "rmse", "mape"]
    model_names = list(results_dict.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    colors = ["steelblue", "green", "purple", "coral", "darkblue"][: len(model_names)]
    for idx, metric in enumerate(metrics):
        values = [results_dict[model].get(metric, 0) for model in model_names]
        axes[idx].bar(model_names, values, color=colors)
        axes[idx].set_title(metric.upper(), fontweight="bold")
        axes[idx].set_ylabel(metric.upper())
        axes[idx].grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(values):
            axes[idx].text(i, v, f"{v:.1f}", ha="center", va="bottom")
    plt.suptitle("Model Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# #############################################################################
# CLI for data download
# #############################################################################


def _main() -> None:
    """CLI entry point for data download."""
    _LOG.info("=" * 60)
    _LOG.info("COVID-19 Data Setup")
    _LOG.info("=" * 60)
    _LOG.info("")
    success = check_and_download_data()
    _LOG.info("")
    _LOG.info("=" * 60)
    if success:
        _LOG.info("Ready to run notebooks!")
    else:
        _LOG.info("Please download the data files as instructed above")
    _LOG.info("=" * 60)


if __name__ == "__main__":
    _main()
