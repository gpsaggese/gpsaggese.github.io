"""
Data Loader for GluonTS Notebooks

Simple one-function loader to get COVID-19 data ready for GluonTS models.
Loads US COVID-19 cases, deaths, and Google mobility data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
import sys

from GluonTS_utils_data_io import DataLoader
from GluonTS_utils_preprocessing import (
    aggregate_to_national,
    extract_national_mobility,
    merge_all_data
)
from GluonTS_utils_gluonts import (
    create_gluonts_dataset,
    prepare_train_test_split
)


def check_and_download_data(data_dir: str = "data") -> bool:
    """
    Check if required data files exist, download if missing.
    
    Args:
        data_dir: Directory where data files should be
    
    Returns:
        bool: True if all files present or successfully downloaded
    """
    required_files = ['cases.csv', 'deaths.csv', 'mobility.csv']
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Check which files are missing
    missing = []
    for filename in required_files:
        if not (data_path / filename).exists():
            missing.append(filename)
    
    if not missing:
        return True
    
    # Files are missing - try to download
    print("\n" + "="*70)
    print("DATA FILES MISSING - ATTEMPTING DOWNLOAD")
    print("="*70)
    print(f"\nMissing files: {', '.join(missing)}")
    print("Attempting to download from Google Drive...\n")
    
    # Import download function
    try:
        from GluonTS_utils_data_download import check_and_download_data as download_data
        
        # Try to download
        success = download_data(data_dir)
        
        if success:
            print("\n" + "="*70)
            print("DATA DOWNLOAD SUCCESSFUL")
            print("="*70 + "\n")
            return True
        else:
            # Download failed - show manual instructions
            print("\n" + "="*70)
            print("AUTOMATIC DOWNLOAD FAILED")
            print("="*70)
            print("\nPlease download the data files manually:")
            print("1. Visit: https://drive.google.com/drive/folders/1qMDGBstdY8H2hYpz8xSolhzNOsVxNHMA")
            print("2. Download these files and rename them:")
            print("   - time_series_covid19_confirmed_US.csv -> cases.csv")
            print("   - time_series_covid19_deaths_US.csv -> deaths.csv")
            print("   - mobility_report_US.csv -> mobility.csv")
            print(f"3. Place them in the '{data_dir}/' directory")
            print("\nOr run manually: python GluonTS_utils_data_download.py")
            print("="*70 + "\n")
            return False
            
    except Exception as e:
        # Download module not available or error occurred
        print(f"\nAutomatic download not available: {e}")
        print("\nPlease download the data files manually:")
        print("1. Visit: https://drive.google.com/drive/folders/1qMDGBstdY8H2hYpz8xSolhzNOsVxNHMA")
        print("2. Download and rename the files as shown above")
        print(f"3. Place them in the '{data_dir}/' directory")
        print("="*70 + "\n")
        return False


def load_covid_data_for_gluonts(
    data_dir: str = "data",
    target_column: str = "Daily_Cases_MA7",
    test_size: int = 14,
    prediction_length: int = 14,
    use_features: bool = True,
    feature_subset: str = "minimal"
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
    
    Args:
        data_dir: Directory containing CSV files
        target_column: Column to forecast (default: 'Daily_Cases_MA7')
        test_size: Days for testing (default: 14)
        prediction_length: Forecast horizon (default: 14)
        use_features: Include exogenous features (default: True)
        feature_subset: Which features to use:
            - "minimal": Just deaths (3 features)
            - "moderate": Deaths + key mobility (6 features)
            - "full": All available features (10+ features)
    
    Returns:
        Dictionary with train_ds, test_ds, DataFrames, and metadata
    """
    # Check if data files exist, download if missing
    if not check_and_download_data(data_dir):
        raise FileNotFoundError(
            f"Required data files missing from '{data_dir}/' directory. "
            "Please download them manually as instructed above."
        )
    
    print("=" * 70)
    print("COVID-19 DATA LOADER")
    print("=" * 70)
    
    # Load raw data
    print("\nLoading raw data...")
    loader = DataLoader(data_dir=data_dir)
    
    try:
        cases_df = loader.load_cases()
        deaths_df = loader.load_deaths()
        mobility_df = loader.load_mobility()
        print("Data files loaded (cases, deaths, mobility)")
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Make sure data files exist in '{data_dir}/' folder")
        raise
    
    # Preprocess
    print("\nPreprocessing...")
    national_cases = aggregate_to_national(cases_df, data_type='cases')
    national_deaths = aggregate_to_national(deaths_df, data_type='deaths')
    national_mobility = extract_national_mobility(mobility_df)
    
    # Merge
    print("\nMerging data sources...")
    merged_df = merge_all_data(national_cases, national_deaths, national_mobility)
    
    print(f"Merged data: {len(merged_df)} days")
    print(f"Date range: {merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}")
    
    # Select features
    print(f"\nFeature selection: {feature_subset}")
    
    if not use_features:
        feature_columns = None
        print("Using target only (no exogenous features)")
    else:
        if feature_subset == "minimal":
            feature_columns = [
                'Daily_Deaths_MA7',
                'Cumulative_Deaths',
                'CFR'
            ]
        elif feature_subset == "moderate":
            feature_columns = [
                'Daily_Deaths_MA7',
                'CFR',
                'retail_and_recreation_percent_change_from_baseline',
                'grocery_and_pharmacy_percent_change_from_baseline',
                'workplaces_percent_change_from_baseline',
                'residential_percent_change_from_baseline'
            ]
        else:  # full
            exclude = ['Date', target_column, 'Daily_Cases', 'Cumulative_Cases', 
                      'Daily_Deaths', 'Cumulative_Deaths']
            feature_columns = [col for col in merged_df.columns 
                              if col not in exclude and merged_df[col].dtype in ['int64', 'float64']]
        
        print(f"Selected {len(feature_columns)} features:")
        for i, feat in enumerate(feature_columns[:5], 1):
            print(f"  {i}. {feat}")
        if len(feature_columns) > 5:
            print(f"  ... and {len(feature_columns) - 5} more")
    
    # Split train/test
    print(f"\nSplitting data (test size: {test_size} days)...")
    train_df, test_df = prepare_train_test_split(
        merged_df,
        test_size=test_size,
        target_column=target_column
    )
    
    # Convert to GluonTS format
    print("\nConverting to GluonTS format...")
    
    # Train dataset: only training period
    train_ds = create_gluonts_dataset(
        df=train_df,
        target_column=target_column,
        freq='D',
        prediction_length=prediction_length,
        past_feat_columns=feature_columns
    )
    
    # Test dataset: full data (train + test) - GluonTS needs full history
    test_ds = create_gluonts_dataset(
        df=merged_df.dropna(subset=[target_column]),
        target_column=target_column,
        freq='D',
        prediction_length=prediction_length,
        past_feat_columns=feature_columns
    )
    
    print("GluonTS datasets created")
    print("Note: Test dataset contains full time series (train + test periods)")
    
    # Prepare return info
    info = {
        'total_days': len(merged_df),
        'train_days': len(train_df),
        'test_days': len(test_df),
        'date_range': f"{merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}",
        'target_column': target_column,
        'num_features': len(feature_columns) if feature_columns else 0,
        'feature_subset': feature_subset
    }
    
    print("\n" + "=" * 70)
    print("DATA READY FOR TRAINING")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Target: {target_column}")
    print(f"  Features: {info['num_features']} ({feature_subset})")
    print(f"  Train: {info['train_days']} days")
    print(f"  Test: {info['test_days']} days")
    print(f"  Prediction length: {prediction_length} days")
    print("=" * 70)
    
    return {
        'train_ds': train_ds,
        'test_ds': test_ds,
        'train_df': train_df,
        'test_df': test_df,
        'merged_df': merged_df,
        'target': target_column,
        'features': feature_columns,
        'info': info
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
