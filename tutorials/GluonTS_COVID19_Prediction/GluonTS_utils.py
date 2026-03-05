"""
Consolidated utilities for COVID-19 time series forecasting with GluonTS.

Sections:
- Data Management: loading, downloading, preprocessing
- Synthetic Data: generators and utilities
- GluonTS Core: dataset creation and validation
- Model Training: individual model trainers and utilities
- Evaluation: metrics calculation and model comparison
- Scenario Analysis: what-if analysis functions
- Visualization: plotting functions

Import as:

    import GluonTS_utils
    from GluonTS_utils import load_covid_data_for_gluonts, train_deepar_covid
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import logging
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# GluonTS imports
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.torch.model.deep_npts import DeepNPTSEstimator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator

# =============================================================================
# CONSTANTS
# =============================================================================

# Default settings
_DEFAULT_START = "2020-01-01"
_DEFAULT_PREDICTION_LENGTH = 14
_DEFAULT_CONTEXT_LENGTH = 60

# Data directories and files
REQUIRED_DATA_FILES = ["cases.csv", "deaths.csv", "mobility.csv"]

# Model training defaults
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-3

# Visualization defaults
DEFAULT_FIGSIZE = (12, 6)
DEFAULT_DPI = 150

# Logging setup -- configure so messages show up in notebooks by default
_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(_handler)
    _LOG.setLevel(logging.INFO)
warnings.filterwarnings("ignore")


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

    print("Feature Correlations with Daily Cases:")
    print("=" * 70)
    for feat, corr in correlations.items():
        bar = bar_char * int(abs(corr) * bar_width)
        sign = "+" if corr > 0 else "-"
        print(f"  {feat:35s} [{sign}] {bar} {corr:+.3f}")

    print("\nInterpretation:")
    print("  Positive correlation: Feature increases with cases")
    print("  Negative correlation: Feature decreases when cases rise")
    print("  Magnitude: Strength of relationship")

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
# Data Management
# #############################################################################


def load_csv_data(
    filename: str,
    *,
    data_dir: str = "data",
    date_columns: Optional[List[str]] = None,
    required: bool = True,
) -> pd.DataFrame:
    """
    Generic CSV data loader with error handling.

    :param filename: Name of the CSV file to load
    :param data_dir: Directory containing the file
    :param date_columns: Columns to parse as dates
    :param required: Whether file is required (raises error if missing)
    :return: Loaded DataFrame
    :raises FileNotFoundError: If required file is missing
    """
    filepath = Path(data_dir) / filename

    if not filepath.exists():
        if required:
            raise FileNotFoundError(f"Required file '{filename}' not found in {data_dir}")
        _LOG.warning("Optional file '%s' not found in %s", filename, data_dir)
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

        _LOG.info("Loaded %s: %d rows, %d columns", filename, len(df), len(df.columns))
        return df
    except Exception as e:
        if required:
            raise RuntimeError(f"Failed to load {filename}: {e}")
        _LOG.warning("Failed to load optional file %s: %s", filename, e)
        return pd.DataFrame()


def load_jhu_cases(*, data_dir: str = "data") -> pd.DataFrame:
    """
    Load JHU CSSE COVID-19 cases data.

    :param data_dir: Directory containing data files
    :return: DataFrame with cases data
    """
    return load_csv_data("cases.csv", data_dir=data_dir)


def load_jhu_deaths(*, data_dir: str = "data") -> pd.DataFrame:
    """
    Load JHU CSSE COVID-19 deaths data.

    :param data_dir: Directory containing data files
    :return: DataFrame with deaths data
    """
    return load_csv_data("deaths.csv", data_dir=data_dir)




def load_google_mobility(*, data_dir: str = "data") -> pd.DataFrame:
    """
    Load Google COVID-19 Community Mobility Reports.

    :param data_dir: Directory containing data files
    :return: DataFrame with mobility data (date column parsed)
    """
    df = load_csv_data("mobility.csv", data_dir=data_dir, date_columns=["date"])
    if not df.empty and "date" in df.columns:
        _LOG.info("  Date range: %s to %s", df["date"].min().date(), df["date"].max().date())
    return df


def load_all_data(*, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load all COVID-19 datasets at once.

    :param data_dir: Directory containing data files
    :return: Dictionary with keys 'cases', 'deaths', 'mobility'
    """
    _LOG.info("Loading all COVID-19 datasets...")
    _LOG.info("=" * 60)

    data = {}
    data["cases"] = load_jhu_cases(data_dir=data_dir)
    data["deaths"] = load_jhu_deaths(data_dir=data_dir)
    data["mobility"] = load_google_mobility(data_dir=data_dir)

    _LOG.info("=" * 60)
    _LOG.info("All datasets loaded successfully")
    return data


def verify_data_exists(*, data_dir: str = "data") -> bool:
    """
    Verify that all required data files exist.

    :param data_dir: Directory containing data files
    :return: True if all files exist, False otherwise
    """
    data_path = Path(data_dir)
    missing_files = []

    for filename in REQUIRED_DATA_FILES:
        if not (data_path / filename).exists():
            missing_files.append(filename)

    if missing_files:
        _LOG.info("Missing files: %s", ", ".join(missing_files))
        _LOG.info("Expected location: %s", data_path.absolute())
        return False

    _LOG.info("All required data files present in %s", data_dir)
    return True


class DataLoader:
    """
    Convenience class for loading COVID-19 data with consistent interface.
    """

    def __init__(self, *, data_dir: str = "data"):
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
    Check if data files exist, download missing ones from Google Drive.

    :param data_dir: Directory containing data files
    :return: True if all files present or downloaded successfully
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    # Google Drive file IDs for each dataset
    drive_files = {
        "cases.csv": "1ZfZtoV3PpZblZYES0A5LHCwp54cR8RJL",
        "deaths.csv": "1kYC9nrCnKbNpnoZKz8o6TDMM371gyxbl",
        "mobility.csv": "1TMqG8Z8vbxmQAv1rNKczYYPCzwT4ZS_q",
    }

    existing_files = []
    missing_files = []

    for filename in REQUIRED_DATA_FILES:
        file_path = data_path / filename
        if file_path.exists():
            existing_files.append(filename)
            _LOG.info("  Found: %s", filename)
        else:
            missing_files.append(filename)

    if not missing_files:
        _LOG.info("\nAll data files present.")
        return True

    _LOG.info("\nMissing files: %s", ", ".join(missing_files))
    _LOG.info("\nAttempting to download from Google Drive...")

    downloaded = []
    failed = []

    for filename in missing_files:
        file_id = drive_files.get(filename)
        if not file_id:
            _LOG.warning("No download link available for %s", filename)
            failed.append(filename)
            continue

        file_path = data_path / filename
        if download_file_from_google_drive(file_id, file_path):
            downloaded.append(filename)
        else:
            failed.append(filename)

    if failed:
        _LOG.error("Failed to download: %s", ", ".join(failed))
        _LOG.info("Please download manually from: https://drive.google.com/drive/folders/1qMDGBstdY8H2hYpz8xSolhzNOsVxNHMA")
        file_mapping = {
            "cases.csv": "time_series_covid19_confirmed_US.csv",
            "deaths.csv": "time_series_covid19_deaths_US.csv",
            "mobility.csv": "mobility_report_US.csv",
        }
        _LOG.info("\nDownload these files and save to 'data/' directory:")
        for local_name in failed:
            if local_name in file_mapping:
                drive_name = file_mapping[local_name]
                _LOG.info("  - %s -> rename to '%s'", drive_name, local_name)
        return False

    _LOG.info("Successfully downloaded: %s", ", ".join(downloaded))
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


def compute_custom_metrics(forecasts, ground_truths) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and MAPE from GluonTS forecast/ground-truth pairs.

    GluonTS's ``make_evaluation_predictions`` returns each ground truth as the
    *full* time series (train + test).  The forecast only covers the last
    ``prediction_length`` steps.  We therefore align by taking the *tail* of
    the ground truth that matches the forecast length.

    :param forecasts: List of forecast objects (each has a ``.mean`` attribute)
    :param ground_truths: List of ground truth time-series objects returned by
        ``make_evaluation_predictions``
    :return: Dictionary with ``MAE``, ``RMSE``, ``MAPE`` keys
    """
    mae_values = []
    rmse_values = []
    mape_values = []

    for forecast, ground_truth in zip(forecasts, ground_truths):
        forecast_mean = np.array(forecast.mean)
        forecast_len = len(forecast_mean)

        if hasattr(ground_truth, "target"):
            target = np.array(ground_truth.target)
        elif isinstance(ground_truth, pd.DataFrame):
            if "target" in ground_truth.columns:
                target = ground_truth["target"].values
            else:
                target = ground_truth.iloc[:, 0].values
        else:
            target = np.array(ground_truth)

        # Align: the forecast corresponds to the LAST forecast_len values
        # of the full ground-truth series.
        target = target[-forecast_len:]

        mae = np.mean(np.abs(forecast_mean - target))
        rmse = np.sqrt(np.mean((forecast_mean - target) ** 2))

        denom = np.maximum(np.abs(target), 1.0)
        mape_vals = np.abs((target - forecast_mean) / denom)
        mape = np.mean(mape_vals) if mape_vals.size > 0 else np.nan

        mae_values.append(mae)
        rmse_values.append(rmse)
        mape_values.append(mape)

    return {
        "MAE": float(np.nanmean(mae_values)),
        "RMSE": float(np.nanmean(rmse_values)),
        "MAPE": float(np.nanmean(mape_values)) * 100,
    }


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
    prediction_length: int = _DEFAULT_PREDICTION_LENGTH,
    num_feat_dynamic_real: int = 0,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    context_length: Optional[int] = None,
    num_layers: int = 2,
    hidden_size: int = 40,
    dropout_rate: float = 0.1,
    verbose: bool = True,
) -> ModelResults:
    """
    Train a DeepAR model for COVID-19 forecasting.

    DeepAR is great for complex patterns with multiple seasonalities and long-term
    dependencies. It uses recurrent neural networks to "remember" past patterns.

    :param train_ds: Training dataset in GluonTS format
    :param test_ds: Test dataset for evaluation
    :param prediction_length: How many days to forecast ahead
    :param num_feat_dynamic_real: Number of external features (like mobility data)
    :param epochs: Training iterations (more = better but slower)
    :param learning_rate: How aggressively to update model weights
    :param context_length: How far back to look (default: 2x prediction_length)
    :param num_layers: RNN layers (more = complex but slower)
    :param hidden_size: Size of hidden layers
    :param dropout_rate: Regularization to prevent overfitting
    :param verbose: Show training progress and results
    :return: ModelResults with trained model, forecasts, and metrics
    """
    if verbose:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("TRAINING DeepAR MODEL")
        _LOG.info("=" * 70)
        _LOG.info("\nConfiguration:")
        _LOG.info("  Epochs: %s", epochs)
        _LOG.info("  Context length: %s days", context_length or prediction_length * 2)
        _LOG.info("  External features: %s", num_feat_dynamic_real)
        _LOG.info("  Hidden size: %s", hidden_size)
        _LOG.info("  RNN layers: %s", num_layers)

    start_time = time.time()
    estimator = DeepAREstimator(
        freq="D",
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
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
        _LOG.info("Training complete in %.1f seconds", training_time)
        _LOG.info("\nGenerating probabilistic forecasts...")

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))

    custom_metrics = compute_custom_metrics(forecasts, ground_truths)
    agg_metrics.update(custom_metrics)

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
    prediction_length: int = _DEFAULT_PREDICTION_LENGTH,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    context_length: Optional[int] = None,
    hidden_dimensions: Optional[List[int]] = None,
    verbose: bool = True,
) -> ModelResults:
    """
    Train a SimpleFeedForward model for COVID-19 forecasting.

    This is the fastest model - just maps recent history directly to future predictions.
    Good for quick baselines and stable trends, but doesn't handle complex patterns well.

    :param train_ds: Training dataset in GluonTS format
    :param test_ds: Test dataset for evaluation
    :param prediction_length: How many days to forecast ahead
    :param epochs: Training iterations (more = better but slower)
    :param learning_rate: How aggressively to update model weights
    :param context_length: How far back to look (default: 2x prediction_length)
    :param hidden_dimensions: Size of hidden layers (default: [40, 40])
    :param verbose: Show training progress and results
    :return: ModelResults with trained model, forecasts, and metrics
    """
    if verbose:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("TRAINING SimpleFeedForward MODEL")
        _LOG.info("=" * 70)
        _LOG.info("\nNote: This model doesn't use external features")
        _LOG.info("\nConfiguration:")
        _LOG.info("  Epochs: %s", epochs)
        _LOG.info("  Context length: %s days", context_length or prediction_length * 2)
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
        _LOG.info("Training complete in %.1f seconds", training_time)
        _LOG.info("\nGenerating probabilistic forecasts...")

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))

    custom_metrics = compute_custom_metrics(forecasts, ground_truths)
    agg_metrics.update(custom_metrics)

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
    prediction_length: int = _DEFAULT_PREDICTION_LENGTH,
    num_feat_dynamic_real: int = 0,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    context_length: Optional[int] = None,
    num_hidden_nodes: Optional[List[int]] = None,
    dropout_rate: float = 0.1,
    verbose: bool = True,
) -> ModelResults:
    """
    Train a DeepNPTS model for COVID-19 forecasting.

    DeepNPTS adapts well to changing patterns and regime shifts. It's great when
    the data behavior changes over time (like new virus variants).

    :param train_ds: Training dataset in GluonTS format
    :param test_ds: Test dataset for evaluation
    :param prediction_length: How many days to forecast ahead
    :param num_feat_dynamic_real: Number of external features (like mobility data)
    :param epochs: Training iterations (more = better but slower)
    :param learning_rate: How aggressively to update model weights
    :param context_length: How far back to look (default: 2x prediction_length)
    :param num_hidden_nodes: Size of hidden layers (default: [40])
    :param dropout_rate: Regularization to prevent overfitting
    :param verbose: Show training progress and results
    :return: ModelResults with trained model, forecasts, and metrics
    """
    if verbose:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("TRAINING DeepNPTS MODEL")
        _LOG.info("=" * 70)
        _LOG.info("\nConfiguration:")
        _LOG.info("  Epochs: %s", epochs)
        _LOG.info("  Context length: %s days", context_length or prediction_length * 2)
        _LOG.info("  External features: %s", num_feat_dynamic_real)
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
        _LOG.info("Training complete in %.1f seconds", training_time)
        _LOG.info("\nGenerating probabilistic forecasts...")

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))

    custom_metrics = compute_custom_metrics(forecasts, ground_truths)
    agg_metrics.update(custom_metrics)

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
    # interpretation guidance
    _LOG.info("\nMetric guidelines: MAPE <10%% highly accurate, 10-20%% good, 21-50%% reasonable, >50%% inaccurate.")
    _LOG.info("Lower RMSE and MAE are always better; compare them to the scale or baseline of the target series.")
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
    A handy container that holds all the results from running a scenario forecast.

    This bundles together the forecast data, summary statistics, and scenario details
    so you can easily compare different "what-if" situations.
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
        """Calculate how this scenario compares to the baseline in terms of total cases."""
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
    Create a modified version of your data to test different "what-if" scenarios.

    This is useful for exploring questions like:
    - What if mobility decreased by 20% (stricter lockdowns)?
    - What if the case fatality rate increased (healthcare strain)?
    - What if restrictions were relaxed?

    The function keeps all historical data unchanged and extends the external
    features for the forecast period with scenario adjustments applied.

    Args:
        merged_df: Your main COVID-19 dataset
        feature_columns: Which columns to include as features
        target_column: The column you're trying to predict
        mobility_adjustment: Multiply mobility values by this factor (0.8 = 20% reduction)
        cfr_adjustment: Multiply case fatality rate by this factor
        deaths_adjustment: Multiply death-related values by this factor
        prediction_length: How many days to forecast
        freq: Frequency of the data ('D' for daily)

    Returns:
        A GluonTS dataset ready for scenario forecasting
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

    df_clean = df.dropna(subset=[target_column]).copy()
    date_col = "Date" if "Date" in df_clean.columns else "date"
    start_date = pd.to_datetime(df_clean[date_col].iloc[0])
    
    # Use all historical data for the target
    target = df_clean[target_column].values.tolist()

    data_entry = {"start": start_date, "target": target}

    if feature_columns:
        feat_dynamic_real = []
        
        # Get the last row as a template for future feature values
        last_row_idx = len(df_clean) - 1
        
        for col in feature_columns:
            if col in df_clean.columns:
                # Get historical values
                historical_values = df_clean[col].values.tolist()
                
                # Get the last value and extend it for the forecast period
                last_value = df_clean[col].iloc[last_row_idx]
                
                # Apply scenario adjustments to the extended forecast period
                if col in mobility_cols:
                    extended_value = last_value * mobility_adjustment
                elif col in cfr_cols:
                    extended_value = last_value * cfr_adjustment
                elif col in deaths_cols:
                    extended_value = last_value * deaths_adjustment
                else:
                    extended_value = last_value
                
                # Create extended feature values: historical + forecast period with adjustments
                extended_values = historical_values + [extended_value] * prediction_length
                feat_dynamic_real.append(extended_values)
        
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
    """
    Run a forecast for one specific scenario and package up all the results.

    This takes your trained model and a modified dataset (representing a scenario)
    and generates the forecast along with summary statistics.

    Args:
        predictor: Your trained GluonTS model
        scenario_dataset: The modified dataset for this scenario
        scenario_name: Short name for the scenario (e.g., "Strong Intervention")
        scenario_description: Longer description of what this scenario represents
        adjustments: Dictionary of what was changed (e.g., {"mobility": 0.8})
        num_samples: Number of forecast samples to generate (more = better uncertainty estimates)

    Returns:
        ScenarioResult object with forecast, summary stats, and scenario details
    """
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
    """
    Run forecasts for all the predefined scenarios to explore different possibilities.

    This function tests 5 different scenarios:
    1. Baseline - No changes, current trends continue
    2. Moderate Intervention - 15% mobility reduction
    3. Strong Intervention - 30% mobility reduction
    4. Relaxation - 20% mobility increase
    5. Healthcare Strain - 15% higher case fatality rate

    Args:
        predictor: Your trained GluonTS model
        merged_df: The main COVID-19 dataset
        feature_columns: Which columns to use as features
        target_column: Which column you're predicting
        prediction_length: How many days to forecast
        verbose: Whether to print progress updates

    Returns:
        List of ScenarioResult objects, one for each scenario
    """
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
        print("\n" + "=" * 70)
        print("EXPLORING DIFFERENT SCENARIOS")
        print("=" * 70)

    for i, config in enumerate(scenarios_config, 1):
        if verbose:
            print(f"\n[{i}/5] {config['name']}: {config['description']}")

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
            print(f"   Avg daily: {result.mean_daily_cases:,.0f} | Total: {result.total_cases:,.0f}")
    if verbose:
        print("\nScenario exploration complete!")
    return results


def print_scenario_summary(results: List[ScenarioResult]) -> pd.DataFrame:
    """
    Print a clear comparison table showing how all scenarios differ from each other.

    This creates a nice table that lets you quickly see:
    - Average daily cases for each scenario
    - Total cases over the forecast period
    - How much each scenario differs from the baseline

    Args:
        results: List of ScenarioResult objects from run_all_scenarios()

    Returns:
        DataFrame with the summary data (useful for further analysis)
    """
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
                "vs Baseline": f"{pct:+.1f}%"
                if result.name != "Baseline"
                else "--",
                "Cases Delta": diff
                if result.name != "Baseline"
                else 0,
            }
        )

    df = pd.DataFrame(summary_data)

    print("\n" + "=" * 90)
    print("SCENARIO FORECAST COMPARISON")
    print("=" * 90)
    print(f"\nForecast horizon: 14 days")
    print(f"Baseline total cases: {baseline_total:,.0f}")
    print("")
    print("Scenario".ljust(25), "Avg Daily".ljust(12), "Total Cases".ljust(14), "vs Baseline".ljust(12), "Cases Delta".ljust(15))
    print("-" * 90)

    for _, row in df.iterrows():
        scenario_name = str(row["Scenario"]).ljust(25)
        avg_daily = f"{row['Avg Daily Cases']:,.0f}".ljust(12)
        total_cases = f"{row['Total Cases (14d)']:,.0f}".ljust(14)
        vs_baseline = str(row["vs Baseline"]).ljust(12)
        cases_delta_val = "--"
        if row["Scenario"] != "Baseline":
            cases_delta_val = f"{int(row['Cases Delta']):+,.0f}"
        cases_delta = cases_delta_val.ljust(15)
        print(scenario_name + avg_daily + total_cases + vs_baseline + cases_delta)
    print("=" * 90)
    return df


def print_policy_insights(results: List[ScenarioResult]) -> None:
    """
    Print key insights about the potential impact of different policy decisions.

    This translates the scenario results into practical insights like:
    - How many cases could stricter interventions prevent?
    - What risks come with relaxing restrictions?
    - What happens if healthcare gets overwhelmed?

    Args:
        results: List of ScenarioResult objects from run_all_scenarios()
    """
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

    print("\nPOLICY INSIGHTS FROM SCENARIO ANALYSIS")
    print("=" * 70)
    print()

    print("Intervention Impact:")
    if strong_intervention:
        pct = (cases_prevented / baseline.total_cases) * 100
        print("  Strong intervention (30% mobility reduction) could prevent")
        print(f"  ~{cases_prevented:,.0f} cases over 14 days ({pct:.1f}% reduction)")
    else:
        print("  Strong intervention scenario not found.")
    print()

    print("Relaxation Risk:")
    if relaxation:
        pct = (additional_cases / baseline.total_cases) * 100
        print("  Lifting restrictions (20% mobility increase) could add")
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
    figsize: tuple = (16, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Create a comprehensive visualization comparing all scenarios side-by-side.

    This plot shows two views:
    1. Forecast trajectories over time for each scenario
    2. Total cases comparison as a bar chart

    Perfect for presentations or reports to show the range of possible outcomes.

    Args:
        results: List of scenario result objects (from run_all_scenarios)
        prediction_length: Number of days being forecasted
        figsize: Size of the plot (width, height) in inches
        save_path: If provided, save the plot to this file path
    """
    colors = {
        "Baseline": "#6B7280",
        "Moderate Intervention": "#3B82F6",
        "Strong Intervention": "#10B981",
        "Relaxation": "#F59E0B",
        "Healthcare Strain": "#EF4444",
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Forecast trajectories over time
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
    ax1.set_title("Forecast Trajectories by Scenario", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, prediction_length + 1, 2))

    # Right plot: Total cases comparison
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

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scenario comparison plot to: {save_path}")

    plt.show()

    print("\nScenario comparison insights:")
    print("  - Left plot shows how each scenario evolves over the 14-day forecast")
    print("  - Right plot compares total case burden for each scenario")
    print("  - Shaded areas show forecast uncertainty (80% confidence intervals)")
    print("  - Percentages show change relative to baseline scenario")


# #############################################################################
# Evaluation
# #############################################################################


def calculate_metrics(
    forecast_values: Union[np.ndarray, pd.Series, list],
    actual_values: Union[np.ndarray, pd.Series, list],
) -> Dict[str, float]:
    """
    Calculate standard forecasting accuracy metrics.

    This gives you the key numbers to understand how well your forecast performed:
    - MAE: Average absolute error (easy to understand, in same units as data)
    - RMSE: Penalizes big errors more (good for detecting outliers)
    - MAPE: Percentage error (good for comparing across different scales)
    - ME: Average bias (positive = over-forecasting, negative = under-forecasting)

    :param forecast_values: What your model predicted
    :param actual_values: What actually happened
    :return: Dictionary with all the metrics
    """
    forecast_values = np.asarray(forecast_values).flatten()
    actual_values = np.asarray(actual_values).flatten()

    # Handle mismatched lengths (take the shorter one)
    if len(forecast_values) != len(actual_values):
        min_len = min(len(forecast_values), len(actual_values))
        forecast_values = forecast_values[:min_len]
        actual_values = actual_values[:min_len]

    errors = forecast_values - actual_values

    return {
        "mae": np.mean(np.abs(errors)),  # Mean Absolute Error
        "rmse": np.sqrt(np.mean(errors**2)),  # Root Mean Square Error
        "mape": np.mean(np.abs(errors / actual_values)) * 100,  # Mean Absolute Percentage Error
        "me": np.mean(errors),  # Mean Error (bias)
        "max_error": np.max(np.abs(errors)),  # Worst single prediction
    }


def print_metrics(
    metrics: Dict[str, float],
    *,
    model_name: str = "Model",
) -> None:
    """
    Print forecasting metrics in a clear, easy-to-read format with helpful interpretation.

    This takes the raw numbers from calculate_metrics() and presents them nicely,
    plus adds some guidance on what the numbers mean for your model's performance.

    Args:
        metrics: Dictionary returned by calculate_metrics()
        model_name: Name of your model (just for the header)
    """
    print(f"\n{model_name} Performance Metrics:")
    print("=" * 60)
    print(f"MAE (Mean Absolute Error):      {metrics['mae']:10,.2f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:10,.2f}")
    print(f"MAPE (Mean Abs. %% Error):       {metrics['mape']:10.2f} %%")
    print(f"ME (Mean Error / Bias):         {metrics['me']:10,.2f}")
    print(f"Maximum Error:                   {metrics['max_error']:10,.2f}")
    print("=" * 60)

    if metrics["mape"] < 10:
        print("Excellent -- error less than 10%")
    elif metrics["mape"] < 20:
        print("Good performance -- error less than 20%")
    else:
        print("Moderate performance (COVID data is highly variable)")

    if abs(metrics["me"]) < metrics["mae"] / 2:
        print("Low bias -- not systematically over/under-predicting")
    else:
        bias_direction = "over" if metrics["me"] > 0 else "under"
        print(f"Model tends to {bias_direction}-predict")


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
    """
    Create a comprehensive forecast visualization that shows how well your model did.

    This plot combines everything you need to evaluate your forecast:
    - Recent historical data (to see the patterns your model learned)
    - What actually happened (the ground truth)
    - Your model's predictions
    - Uncertainty bands (how confident the model is)

    Args:
        train_df: Your training data DataFrame
        forecast_dates: Dates for the forecast period
        forecast_values: The model's point predictions (usually the mean)
        actual_values: What actually happened during the forecast period
        forecast_quantiles: Uncertainty intervals like {0.1: lower, 0.9: upper}
        target_column: Name of the column you're forecasting
        model_name: Name of your model (for the plot title)
        save_path: If provided, save the plot to this file path
        context_days: How many days of historical data to show before the forecast
    """
    plt.figure(figsize=DEFAULT_FIGSIZE)

    # Show recent historical context
    train_context = train_df.tail(context_days)
    plt.plot(
        train_context["Date"],
        train_context[target_column],
        label="Historical Data",
        color="steelblue",
        linewidth=2,
        alpha=0.8,
    )

    # Plot actual values (what really happened)
    plt.plot(
        forecast_dates,
        actual_values,
        label="Actual Values",
        color="orange",
        linewidth=3,
        marker="o",
        markersize=8,
        zorder=5,
    )

    # Plot model predictions
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

    # Add uncertainty intervals if available
    if 0.05 in forecast_quantiles and 0.95 in forecast_quantiles:
        plt.fill_between(
            forecast_dates,
            forecast_quantiles[0.05],
            forecast_quantiles[0.95],
            alpha=0.15,
            color="red",
            label="90% Confidence Interval",
        )

    if 0.25 in forecast_quantiles and 0.75 in forecast_quantiles:
        plt.fill_between(
            forecast_dates,
            forecast_quantiles[0.25],
            forecast_quantiles[0.75],
            alpha=0.25,
            color="red",
            label="50% Confidence Interval",
        )

    plt.title(
        f"{model_name} Forecast: {len(forecast_dates)}-Day Prediction",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Date", fontsize=13)
    plt.ylabel(target_column.replace("_", " ").title(), fontsize=13)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        print(f"Forecast plot saved to: {save_path}")

    plt.show()


def plot_error_analysis(
    forecast_values: np.ndarray,
    actual_values: np.ndarray,
    forecast_quantiles: Dict[float, np.ndarray],
    model_name: str,
    *,
    save_path: str = None,
) -> None:
    """
    Create detailed error analysis plots to understand where your forecast went wrong.

    This shows 4 different views of your model's performance:
    1. Forecast vs Actual over time - See the overall pattern
    2. Daily prediction errors - When did it over/under-predict?
    3. Absolute percentage error - Which days had the biggest relative errors?
    4. Forecast uncertainty - How confident was the model each day?

    Args:
        forecast_values: What your model predicted
        actual_values: What actually happened
        forecast_quantiles: The uncertainty intervals from your model
        model_name: Name of your model (for the plot titles)
        save_path: If provided, save the plot to this file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    forecast_period = len(forecast_values)
    errors = forecast_values - actual_values

    # Panel 1: Forecast vs Actual
    axes[0, 0].plot(
        range(1, forecast_period + 1),
        actual_values,
        "o-",
        label="Actual Values",
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

    # Add uncertainty band if available
    if 0.1 in forecast_quantiles and 0.9 in forecast_quantiles:
        axes[0, 0].fill_between(
            range(1, forecast_period + 1),
            forecast_quantiles[0.1],
            forecast_quantiles[0.9],
            alpha=0.2,
            color="red",
            label="80% Confidence",
        )

    axes[0, 0].set_title("Forecast vs Actual Values", fontweight="bold")
    axes[0, 0].set_xlabel("Day in Forecast Period")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Daily Errors
    colors = ["red" if e > 0 else "green" for e in errors]
    axes[0, 1].bar(range(1, forecast_period + 1), errors, color=colors)
    axes[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Daily Forecast Errors", fontweight="bold")
    axes[0, 1].set_xlabel("Day in Forecast Period")
    axes[0, 1].set_ylabel("Error (Forecast - Actual)")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Panel 3: Absolute Percentage Error
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
        label=f"Average: {mape:.1f}%",
    )
    axes[1, 0].set_title("Absolute Percentage Error by Day", fontweight="bold")
    axes[1, 0].set_xlabel("Day in Forecast Period")
    axes[1, 0].set_ylabel("Absolute % Error")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Panel 4: Uncertainty Analysis
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
        axes[1, 1].set_title("Forecast Uncertainty (80% CI Width)", fontweight="bold")
        axes[1, 1].set_xlabel("Day in Forecast Period")
        axes[1, 1].set_ylabel("Confidence Interval Width")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Quantiles not available\nfor uncertainty analysis",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=12,
        )
        axes[1, 1].set_title("Uncertainty Analysis", fontweight="bold")

    plt.suptitle(
        f"{model_name} - Detailed Error Analysis",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        _LOG.info("Error analysis plot saved to: %s", save_path)

    plt.show()


def compare_models_metrics(
    results: Dict[str, Dict[str, float]],
    *,
    save_path: str = None,
) -> None:
    """
    Compare how different models performed using bar charts and a summary table.

    This creates an easy-to-read comparison showing which model did best
    on each accuracy metric, helping you choose the right model for your needs.

    Args:
        results: Dictionary where keys are model names and values are metric dictionaries
        save_path: If provided, save the comparison plot to this file path
    """
    metrics_to_plot = ["mae", "rmse", "mape"]
    model_names = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(metrics_to_plot):
        values = [results[model][metric] for model in model_names]
        colors = ["steelblue", "green", "purple", "orange", "red"][:len(model_names)]

        bars = axes[idx].bar(model_names, values, color=colors, alpha=0.8)
        axes[idx].set_title(metric.upper(), fontweight="bold", fontsize=12)
        axes[idx].set_ylabel(metric.upper(), fontsize=11)
        axes[idx].grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, v in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        print(f"Model comparison plot saved to: {save_path}")

    plt.show()

    # Print summary table
    print("\nModel Comparison Summary:")
    print("=" * 70)
    print("%-20s %12s %12s %12s" % ("Model", "MAE", "RMSE", "MAPE"))
    print("-" * 70)

    for model, metrics in results.items():
        print("%-20s %12,.2f %12,.2f %11.2f%%" % (
            model,
            metrics["mae"],
            metrics["rmse"],
            metrics["mape"],
        ))

    print("=" * 70)

    # Find and highlight best model
    best_model = min(results.items(), key=lambda x: x[1]["mape"])
    _LOG.info(
        "\nBest Model (by MAPE): %s (%.2f%% error)",
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
    """
    Create a simple sine wave pattern with some noise - the "hello world" of time series.

    This generates a repeating up-and-down pattern that's easy for models to learn.
    Think of it like the tides or daily temperature variations - predictable cycles
    with a bit of randomness thrown in.

    Args:
        n_points: How many days of data you want to generate
        period: How many days for one complete cycle (up and down)
        amplitude: How much the signal varies from the baseline
        baseline: The center value around which it oscillates
        noise_std: How much random noise to add (makes it realistic)
        seed: Random seed for reproducible results
        start_date: When your time series starts

    Returns:
        DataFrame with Date and value columns, ready for forecasting
    """
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
    """
    Create a complex, realistic time series with multiple patterns happening at once.

    This combines several different cycles and trends, just like real-world data:
    - A long-term upward trend (like growing sales or population)
    - Seasonal cycles (like yearly weather patterns)
    - Weekly cycles (like weekend vs weekday behavior)
    - Random noise (because life is unpredictable)

    Much more challenging than a simple sine wave - great for testing how robust your models are!

    Args:
        n_points: How many days of data to generate
        trend_slope: How much the baseline increases each day (growth rate)
        seasonal_period: Length of the big seasonal cycle in days
        seasonal_amplitude: How much the seasonal pattern varies
        weekly_amplitude: How much the weekly pattern varies
        baseline: Starting value for the series
        noise_std: How much random noise to add
        seed: Random seed for reproducible results
        start_date: When your time series starts

    Returns:
        DataFrame with Date and value columns, ready for forecasting
    """
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
    """
    Create a time series that suddenly changes behavior - like when everything changes overnight.

    This simulates real-world disruptions like:
    - A new COVID variant emerging
    - Government policies changing
    - Market crashes or booms
    - Seasonal weather shifts

    The pattern is completely different before and after the changepoint,
    making this a tough test for forecasting models.

    Args:
        n_points: How many days of data to generate
        changepoint_frac: Where the change happens (0.5 = halfway through)
        baseline_before: Average value before the change
        amplitude_before: How much the signal varies before the change
        period_before: Length of cycles before the change
        baseline_after: Average value after the change
        amplitude_after: How much the signal varies after the change
        period_after: Length of cycles after the change
        noise_std: How much random noise to add
        seed: Random seed for reproducible results
        start_date: When your time series starts

    Returns:
        DataFrame with Date and value columns, ready for forecasting
    """
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
    Split your synthetic time series into training and test sets for forecasting experiments.

    This takes the data you generated and divides it so your model can learn from
    the past and then try to predict the future. The model sees everything up to
    a certain point, then has to forecast what happens next.

    Args:
        df: DataFrame from one of the generate_* functions above
        target_col: Which column contains the values you want to forecast
        prediction_length: How many days into the future to forecast
        freq: How often the data is sampled ('D' for daily)

    Returns:
        Dictionary containing:
        - train_ds, test_ds: GluonTS datasets ready for training/testing
        - train_df, test_df: The original DataFrames split up
        - target: The column name being forecasted
        - prediction_length: How far ahead you're forecasting
        - info: Summary statistics about the split
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
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Create a simple, clean plot of a synthetic time series.

    Perfect for quickly checking what your generated data looks like
    before using it for training models.

    Args:
        df: DataFrame containing the time series data
        target_col: Name of the column with the values to plot
        title: Title for the plot
        figsize: Size of the plot (width, height) in inches
        save_path: If provided, save the plot to this file path
    """
    date_col = "Date" if "Date" in df.columns else "date"
    plt.figure(figsize=figsize)
    plt.plot(df[date_col], df[target_col], linewidth=2, color="#2E86AB", alpha=0.8)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_train_test_split(
    data: Dict,
    *,
    title: str = "Train/Test Data Split",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize how your data is split between training and testing periods.

    This helps you understand what portion of your data the model learns from
    versus what it's tested on.

    Args:
        data: Dictionary returned by prepare_synthetic_dataset()
        title: Title for the plot
        figsize: Size of the plot (width, height) in inches
        save_path: If provided, save the plot to this file path
    """
    target_col = data["target"]
    date_col = "Date" if "Date" in data["train_df"].columns else "date"

    plt.figure(figsize=figsize)
    plt.plot(
        data["train_df"][date_col],
        data["train_df"][target_col],
        label="Training Data",
        color="#2E86AB",
        linewidth=2,
        alpha=0.8,
    )
    plt.plot(
        data["test_df"][date_col],
        data["test_df"][target_col],
        label="Test Data",
        color="#A23B72",
        linewidth=2,
        alpha=0.8,
    )
    plt.axvline(
        x=data["test_df"][date_col].iloc[0],
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Split Point",
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    print("\nData split information:")
    print(f"  - Training data: {len(data['train_df'])} points")
    print(f"  - Test data: {len(data['test_df'])} points")
    print("  - The red line shows where training ends and testing begins")


def plot_forecast_result(
    data: Dict,
    forecast_entry,
    *,
    model_name: str = "Model",
    context_points: int = 60,
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a forecast result with some historical context.

    Shows the recent historical data along with the model's prediction
    and uncertainty bounds for easy interpretation.

    Args:
        data: Dictionary returned by prepare_synthetic_dataset()
        forecast_entry: Forecast object from a GluonTS model
        model_name: Name of the model for the plot title
        context_points: Number of historical points to show before forecast
        figsize: Size of the plot (width, height) in inches
        save_path: If provided, save the plot to this file path
    """
    target_col = data["target"]
    date_col = "Date" if "Date" in data["train_df"].columns else "date"
    train_tail = data["train_df"].tail(context_points)
    test_dates = data["test_df"][date_col].values
    actuals = data["test_df"][target_col].values
    pred_mean = forecast_entry.mean

    plt.figure(figsize=figsize)
    plt.plot(
        train_tail[date_col], train_tail[target_col],
        label="Historical", color="#2E86AB", linewidth=2, alpha=0.8,
    )
    plt.plot(
        test_dates, actuals,
        label="Actual", color="#A23B72", linewidth=2, marker="o", markersize=4, alpha=0.8,
    )
    plt.plot(
        test_dates[:len(pred_mean)], pred_mean,
        label=f"{model_name} Forecast", color="#F18F01",
        linewidth=2.5, linestyle="--", marker="s", markersize=4,
    )

    # Add confidence interval
    q_low = forecast_entry.quantile(0.1)
    q_high = forecast_entry.quantile(0.9)
    plt.fill_between(
        test_dates[:len(q_low)], q_low, q_high,
        alpha=0.2, color="#F18F01", label="80% Confidence",
    )

    plt.title(f"{model_name} Forecast Results", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    print(f"\n{model_name} forecast summary:")
    print("  - Blue line shows recent historical data")
    print("  - Red line shows what actually happened")
    print("  - Orange dashed line is the model's prediction")
    print("  - Shaded area shows the model's uncertainty (80% confidence)")


# #############################################################################
# Visualization
# #############################################################################


def plot_data_overview(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    *,
    date_col: str = "Date",
    title: str = "COVID-19 Cases: Training and Test Data",
    ylabel: str = "Daily Cases (7-day avg)",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Show the complete dataset split between training and testing periods.

    This gives you the full picture of your data, with a clear marker
    showing where the model stops learning and starts being tested.

    Args:
        train_df: Training data DataFrame
        test_df: Test data DataFrame
        target_col: Name of the column with the target values
        date_col: Name of the date column
        title: Title for the plot
        ylabel: Label for the y-axis
        figsize: Size of the plot (width, height) in inches
        save_path: If provided, save the plot to this file path
    """
    plt.figure(figsize=figsize)
    plt.plot(
        train_df[date_col],
        train_df[target_col],
        label="Training Data",
        color="#2E86AB",
        linewidth=2,
        alpha=0.8,
    )
    plt.plot(
        test_df[date_col],
        test_df[target_col],
        label="Test Data",
        color="#A23B72",
        linewidth=2,
        alpha=0.8,
    )
    plt.axvline(
        x=train_df[date_col].iloc[-1],
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Forecast Start",
    )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    print("\nData overview:")
    print("  - Blue line shows data used for training the model")
    print("  - Red line shows future data for testing predictions")
    print("  - The vertical dashed line marks the forecast start point")
    print("  - Notice the multiple peaks -- this makes forecasting challenging!")


def plot_forecast_with_confidence_intervals(
    forecast_result: Any,
    model_name: str,
    *,
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None,
) -> tuple:
    """
    Plot forecast results with confidence intervals in a clean, readable way.

    This creates a nice visualization showing your model's predictions alongside
    the actual data, with uncertainty bands to show how confident the model is.

    Args:
        forecast_result: The forecast object from a trained model
        model_name: Name of the model (for the title)
        figsize: Size of the plot (width, height) in inches
        save_path: If provided, save the plot to this file path

    Returns:
        Tuple of (train_dates, train_values, forecast_dates, actual_values)
        for any additional analysis you might want to do
    """
    # Extract the data we need
    forecast = forecast_result.forecasts[0]
    actual = forecast_result.ground_truths[0]

    # Split into training history and future actuals
    history_len = len(actual) - len(forecast.mean)
    train_values = actual[:history_len]
    actual_values = actual[history_len:]
    forecast_mean = forecast.mean
    forecast_lower = forecast.quantile(0.05)  # 90% confidence interval
    forecast_upper = forecast.quantile(0.95)

    # Create date ranges (assuming daily data)
    last_train_date = len(train_values) - 1
    forecast_dates = range(last_train_date + 1, last_train_date + 1 + len(forecast_mean))

    # Set up the plot with a nice style
    plt.figure(figsize=figsize)
    plt.style.use('default')  # Clean style

    # Plot training data (what the model learned from)
    plt.plot(range(len(train_values)), train_values,
             color='#2E86AB', linewidth=2, label='Training Data', alpha=0.8)

    # Plot actual future values (ground truth)
    plt.plot(forecast_dates, actual_values,
             color='#A23B72', linewidth=2, label='Actual Future', alpha=0.8)

    # Plot forecast with confidence interval
    plt.plot(forecast_dates, forecast_mean,
             color='#F18F01', linewidth=2.5, linestyle='--', label='Forecast', alpha=0.9)

    # Add confidence interval shading
    plt.fill_between(forecast_dates, forecast_lower, forecast_upper,
                     alpha=0.2, color='#F18F01', label='90% Confidence Interval')

    # Add a vertical line to show where training ended
    plt.axvline(x=last_train_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label='Training End')

    # Make it look nice
    plt.title(f'{model_name}: COVID-19 Forecasting Results', fontsize=16, fontweight='bold')
    plt.xlabel('Days from Start', fontsize=12)
    plt.ylabel('Daily Cases', fontsize=12)
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    # Print some helpful observations
    print(f"\n{model_name} Results:")
    print("  - The blue line shows the historical data the model learned from")
    print("  - The red line shows what actually happened after training")
    print("  - The orange dashed line is the model's forecast")
    print("  - The shaded area shows the model's uncertainty (90% confidence)")

    return range(len(train_values)), train_values, forecast_dates, actual_values


def plot_data_exploration(
    merged_df: pd.DataFrame,
    *,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Create a comprehensive 3-panel view of COVID-19 data trends.

    This shows cases, deaths, and mobility patterns together so you can see
    how they relate to each other over time.

    Args:
        merged_df: DataFrame with Date, Daily_Cases_MA7, Daily_Deaths_MA7, workplaces columns
        figsize: Size of the plot (width, height) in inches
        save_path: If provided, save the plot to this file path
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Top panel: Cases
    axes[0].plot(
        merged_df["Date"],
        merged_df["Daily_Cases_MA7"],
        linewidth=2,
        color="#2E86AB",
    )
    axes[0].set_title(
        "COVID-19 Daily Cases (7-Day Moving Average)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].set_ylabel("Cases", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Middle panel: Deaths
    axes[1].plot(
        merged_df["Date"],
        merged_df["Daily_Deaths_MA7"],
        linewidth=2,
        color="#A23B72",
    )
    axes[1].set_title(
        "COVID-19 Daily Deaths (7-Day Moving Average)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_ylabel("Deaths", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Bottom panel: Mobility
    axes[2].plot(
        merged_df["Date"],
        merged_df["workplaces"],
        linewidth=2,
        color="#F18F01",
    )
    axes[2].set_title(
        "Workplace Mobility (% change from baseline)",
        fontsize=14,
        fontweight="bold",
    )
    axes[2].set_ylabel("% Change", fontsize=12)
    axes[2].set_xlabel("Date", fontsize=12)
    axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    print("\nKey observations from the data:")
    print("  - Multiple distinct waves of cases are visible")
    print("  - Deaths follow cases with a lag (as expected)")
    print("  - Mobility patterns shifted dramatically during lockdowns")
    print("  - These patterns provide valuable signals for forecasting")


def plot_model_comparison_3panel(
    deepar_results: Any,
    feedforward_results: Any,
    deepnpts_results: Any,
    *,
    figsize: tuple = (15, 12),
    save_path: Optional[str] = None,
) -> None:
    """
    Compare three forecasting models side-by-side in a 3-panel layout.

    This helps you see how different models perform on the same data,
    making it easier to choose the best approach for your needs.

    Args:
        deepar_results: Forecast results from DeepAR model
        feedforward_results: Forecast results from SimpleFeedForward model
        deepnpts_results: Forecast results from DeepNPTS model
        figsize: Size of the plot (width, height) in inches
        save_path: If provided, save the plot to this file path
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Model configurations with consistent colors
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

        # Plot historical data (training)
        ax.plot(
            range(history_len),
            actual[:history_len],
            label="Historical",
            color="gray",
            alpha=0.6,
            linewidth=2,
        )

        # Plot actual future values
        ax.plot(
            range(history_len, len(actual)),
            actual[history_len:],
            label="Actual",
            color="black",
            linewidth=2,
        )

        # Plot forecast
        forecast_range = range(history_len, history_len + len(forecast.mean))
        ax.plot(
            forecast_range,
            forecast.mean,
            label="Forecast",
            color=color,
            linewidth=2,
            linestyle="--",
        )

        # Add confidence interval
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

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    print("\nModel comparison insights:")
    print("  - All models capture the general trend in the data")
    print("  - Confidence intervals show each model's uncertainty")
    print("  - Compare forecast accuracy against the black 'Actual' line")
    print("  - Look for models that balance accuracy with reasonable uncertainty bounds")


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
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Create bar charts comparing model performance across different metrics.

    This gives you a quick visual way to see which model performs best
    on different accuracy measures.

    Args:
        results_dict: Dictionary with model names as keys and metric dictionaries as values
        metrics: List of metrics to plot (defaults to ['mae', 'rmse', 'mape'])
        figsize: Size of the plot (width, height) in inches
        save_path: If provided, save the plot to this file path
    """
    if metrics is None:
        metrics = ["mae", "rmse", "mape"]

    model_names = list(results_dict.keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    # Consistent color palette
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#FCCA46", "#6B2737"][: len(model_names)]

    for idx, metric in enumerate(metrics):
        values = [results_dict[model].get(metric, 0) for model in model_names]
        bars = axes[idx].bar(model_names, values, color=colors, alpha=0.8)

        axes[idx].set_title(metric.upper(), fontsize=14, fontweight="bold")
        axes[idx].set_ylabel(metric.upper(), fontsize=12)
        axes[idx].grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + max(values) * 0.02, f"{v:.2f}",
                          ha="center", va="bottom", fontsize=10)

    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    # Print some insights
    print("\nPerformance comparison notes:")
    print("  - Lower values are better for all metrics shown")
    print("  - MAE = Mean Absolute Error (average prediction error)")
    print("  - RMSE = Root Mean Square Error (emphasizes larger errors)")
    print("  - MAPE = Mean Absolute Percentage Error (relative error)")


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
