"""
Data Loader for GluonTS Notebooks.

Simple one-function loader to get COVID-19 data ready for GluonTS models.
Loads US COVID-19 cases, deaths, and Google mobility data.

Import as:

import tutorials.tutorial_GluonTS_COVID19_Prediction.GluonTS_utils_notebook_loader as ttgcpgunl
"""

import logging
from pathlib import Path
from typing import Dict


from GluonTS_utils_data_io import DataLoader
from GluonTS_utils_gluonts import (
    create_gluonts_dataset,
    prepare_train_test_split,
)
from GluonTS_utils_preprocessing import (
    aggregate_to_national,
    extract_national_mobility,
    merge_all_data,
)

_LOG = logging.getLogger(__name__)


def check_and_download_data(
    *,
    data_dir: str = "data",
) -> bool:
    """
    Check if required data files exist, download if missing.

    :param data_dir: Directory where data files should be
    :return: True if all files present or successfully downloaded
    """
    required_files = ["cases.csv", "deaths.csv", "mobility.csv"]
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    # Check which files are missing.
    missing = []
    for filename in required_files:
        if not (data_path / filename).exists():
            missing.append(filename)
    if not missing:
        return True
    # Files are missing - try to download.
    _LOG.info("\n" + "=" * 70)
    _LOG.info("DATA FILES MISSING - ATTEMPTING DOWNLOAD")
    _LOG.info("=" * 70)
    _LOG.info("\nMissing files: %s", ", ".join(missing))
    _LOG.info("Attempting to download from Google Drive...\n")
    # Import download function.
    try:
        from GluonTS_utils_data_download import (
            check_and_download_data as download_data,
        )

        # Try to download.
        success = download_data(data_dir=data_dir)
        if success:
            _LOG.info("\n" + "=" * 70)
            _LOG.info("DATA DOWNLOAD SUCCESSFUL")
            _LOG.info("=" * 70 + "\n")
            return True
        else:
            # Download failed - show manual instructions.
            _LOG.info("\n" + "=" * 70)
            _LOG.info("AUTOMATIC DOWNLOAD FAILED")
            _LOG.info("=" * 70)
            _LOG.info("\nPlease download the data files manually:")
            _LOG.info(
                "1. Visit: https://drive.google.com/drive/folders/1qMDGBstdY8H2hYpz8xSolhzNOsVxNHMA"
            )
            _LOG.info("2. Download these files and rename them:")
            _LOG.info("   - time_series_covid19_confirmed_US.csv -> cases.csv")
            _LOG.info("   - time_series_covid19_deaths_US.csv -> deaths.csv")
            _LOG.info("   - mobility_report_US.csv -> mobility.csv")
            _LOG.info("3. Place them in the '%s/' directory", data_dir)
            _LOG.info("\nOr run manually: python GluonTS_utils_data_download.py")
            _LOG.info("=" * 70 + "\n")
            return False
    except Exception as e:
        # Download module not available or error occurred.
        _LOG.warning("\nAutomatic download not available: %s", e)
        _LOG.info("\nPlease download the data files manually:")
        _LOG.info(
            "1. Visit: https://drive.google.com/drive/folders/1qMDGBstdY8H2hYpz8xSolhzNOsVxNHMA"
        )
        _LOG.info("2. Download and rename the files as shown above")
        _LOG.info("3. Place them in the '%s/' directory", data_dir)
        _LOG.info("=" * 70 + "\n")
        return False


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

    This function:
    1. Checks if data files exist (provides download instructions if missing)
    2. Loads raw COVID data (cases, deaths, mobility)
    3. Preprocesses and aggregates to national level
    4. Merges all sources
    5. Splits into train/test
    6. Converts to GluonTS format
    7. Returns everything ready to use

    :param data_dir: Directory containing CSV files
    :param target_column: Column to forecast (default: 'Daily_Cases_MA7')
    :param test_size: Days for testing (default: 14)
    :param prediction_length: Forecast horizon (default: 14)
    :param use_features: Include exogenous features (default: True)
    :param feature_subset: Which features to use:
        - "minimal": Just deaths (3 features)
        - "moderate": Deaths + key mobility (6 features)
        - "full": All available features (10+ features)
    :return: Dictionary with train_ds, test_ds, DataFrames, and metadata
    """
    # Check if data files exist, download if missing.
    if not check_and_download_data(data_dir=data_dir):
        raise FileNotFoundError(
            f"Required data files missing from '{data_dir}/' directory. "
            "Please download them manually as instructed above."
        )
    _LOG.info("=" * 70)
    _LOG.info("COVID-19 DATA LOADER")
    _LOG.info("=" * 70)
    # Load raw data.
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
    # Preprocess.
    _LOG.info("\nPreprocessing...")
    national_cases = aggregate_to_national(cases_df, data_type="cases")
    national_deaths = aggregate_to_national(deaths_df, data_type="deaths")
    national_mobility = extract_national_mobility(mobility_df)
    # Merge.
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
    # Select features.
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
        else:  # full
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
    # Split train/test.
    _LOG.info("\nSplitting data (test size: %s days)...", test_size)
    train_df, test_df = prepare_train_test_split(
        merged_df,
        test_size=test_size,
        target_column=target_column,
    )
    # Convert to GluonTS format.
    _LOG.info("\nConverting to GluonTS format...")
    # Train dataset: only training period.
    train_ds = create_gluonts_dataset(
        df=train_df,
        target_column=target_column,
        freq="D",
        prediction_length=prediction_length,
        past_feat_columns=feature_columns,
    )
    # Test dataset: full data (train + test) - GluonTS needs full history.
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
    # Prepare return info.
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
    """
    Quickest load - minimal features, good for testing.

    :return: Data dictionary ready for GluonTS
    """
    return load_covid_data_for_gluonts(feature_subset="minimal")


def quick_load_moderate() -> Dict:
    """
    Moderate features - balanced speed and accuracy.

    :return: Data dictionary ready for GluonTS
    """
    return load_covid_data_for_gluonts(feature_subset="moderate")


def quick_load_full() -> Dict:
    """
    All features - maximum information.

    :return: Data dictionary ready for GluonTS
    """
    return load_covid_data_for_gluonts(feature_subset="full")
