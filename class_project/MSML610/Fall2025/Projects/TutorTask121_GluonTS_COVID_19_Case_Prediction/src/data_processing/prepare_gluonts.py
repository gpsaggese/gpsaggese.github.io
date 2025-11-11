"""
Prepare data in GluonTS format for time series forecasting
"""

import pandas as pd
import json
from pathlib import Path
from gluonts.dataset.common import ListDataset
import importlib
from typing import Optional


def create_gluonts_dataset(
    df,
    target_column='Daily_MA7',
    start_date=None,
    freq='D',
    prediction_length=14,
    multivariate=False,
    mobility_columns=None
):
    """
    Convert pandas DataFrame to GluonTS ListDataset
    
    Args:
        df: DataFrame with time series data
        target_column: Column to use as target variable
        start_date: Start date (if None, uses first date in df)
        freq: Frequency string ('D' for daily)
        prediction_length: Forecast horizon
        multivariate: If True, create multivariate dataset (for DeepVAR)
        mobility_columns: List of mobility column names for multivariate
        
    Returns:
        GluonTS ListDataset
    """
    if start_date is None:
        start_date = pd.Timestamp(df['Date'].iloc[0])
    
    # Default mobility columns
    if mobility_columns is None:
        mobility_columns = [
            'retail and recreation', 'grocery and pharmacy',
            'parks', 'transit stations', 'workplaces', 'residential'
        ]
    
    if multivariate:
        # Create multivariate target (cases + mobility features)
        # Stack target with mobility data
        target_cols = [target_column] + [col for col in mobility_columns if col in df.columns]
        
        # Fill NaN values
        target_data = df[target_cols].fillna(method='ffill').fillna(0)
        
        # Convert to 2D array (num_features x time_steps)
        target = target_data.values.T.tolist()
        
        # Create dataset
        data = [{
            "start": start_date,
            "target": target
        }]
    else:
        # Univariate (original behavior)
        target = df[target_column].fillna(0).values
        
        # Create dataset
        data = [{
            "start": start_date,
            "target": target.tolist()
        }]
    
    return ListDataset(data, freq=freq)


def create_train_test_split(df, test_days=60):
    """
    Split data into train and test sets
    
    Args:
        df: DataFrame with time series data
        test_days: Number of days to reserve for testing
        
    Returns:
        train_df, test_df
    """
    split_idx = len(df) - test_days
    train_df = df.iloc[:split_idx].copy()
    test_df = df.copy()  # Test includes all data for evaluation
    
    return train_df, test_df


def save_metadata(output_dir, train_df, test_df, prediction_length, freq):
    """Save dataset metadata as JSON"""
    metadata = {
        "prediction_length": prediction_length,
        "freq": freq,
        "train_start": str(train_df['Date'].iloc[0]),
        "train_end": str(train_df['Date'].iloc[-1]),
        "train_days": len(train_df),
        "test_start": str(test_df['Date'].iloc[0]),
        "test_end": str(test_df['Date'].iloc[-1]),
        "test_days": len(test_df)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


if __name__ == "__main__":
    # Configuration
    PREDICTION_LENGTH = 14  # Forecast 14 days ahead
    TEST_DAYS = 60  # Reserve last 60 days for testing
    FREQ = 'D'  # Daily frequency
    
    # Load processed data (auto-create if missing)
    data_path = Path('data/processed/national_data.csv')
    if not data_path.exists():
        print("Processed data not found. Attempting to build from raw CSVs using local loader/preprocess helpers...")

        # Try to import the project's DataLoader and preprocessing helpers.
        DataLoader = None
        aggregate_to_national = None
        extract_national_mobility = None
        merge_cases_and_mobility = None

        # Prefer package-style imports but fall back to local imports when run as a script
        try:
            # e.g. when running from project root with src on PYTHONPATH
            from data_processing.load_data import DataLoader
        except Exception:
            try:
                from load_data import DataLoader  # local import
            except Exception:
                DataLoader = None

        try:
            from data_processing.preprocess_data import (
                aggregate_to_national,
                extract_national_mobility,
                merge_cases_and_mobility,
            )
        except Exception:
            try:
                from preprocess_data import (
                    aggregate_to_national,
                    extract_national_mobility,
                    merge_cases_and_mobility,
                )
            except Exception:
                aggregate_to_national = None
                extract_national_mobility = None
                merge_cases_and_mobility = None

        # If we have the pieces, attempt to create processed data
        if DataLoader is not None and aggregate_to_national is not None:
            loader = DataLoader()

            # Attempt to load the raw JHU and mobility CSVs
            try:
                cases_df = loader.load_cases()
            except Exception as e:
                print(f"Could not load cases CSV: {e}")
                cases_df = None

            try:
                mobility_df = loader.load_mobility()
            except Exception as e:
                print(f"Could not load mobility CSV: {e}")
                mobility_df = None

            if cases_df is None:
                print("No raw cases CSV available. Please download raw data into data/ or run the project's download script.")
                exit(1)

            print("Aggregating national cases and merging mobility (if available)...")
            national_cases = aggregate_to_national(cases_df)

            if mobility_df is not None and extract_national_mobility is not None:
                national_mobility = extract_national_mobility(mobility_df)
                merged = merge_cases_and_mobility(national_cases, national_mobility)
            else:
                merged = national_cases

            output_dir = Path('data/processed')
            output_dir.mkdir(exist_ok=True, parents=True)
            merged.to_csv(output_dir / 'national_data.csv', index=False)
            print(f"Saved processed data to {output_dir / 'national_data.csv'}")
        else:
            print("Missing preprocessing utilities (DataLoader/preprocess functions). Please run preprocessing in the data project or ensure PYTHONPATH includes src/.")
            exit(1)

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create train/test split
    print(f"\nSplitting data (test_days={TEST_DAYS})")
    train_df, test_df = create_train_test_split(df, test_days=TEST_DAYS)
    
    print(f"Train: {train_df['Date'].iloc[0]} to {train_df['Date'].iloc[-1]} ({len(train_df)} days)")
    print(f"Test:  {test_df['Date'].iloc[0]} to {test_df['Date'].iloc[-1]} ({len(test_df)} days)")
    
    # Create GluonTS datasets
    print("\nCreating GluonTS datasets...")
    train_ds = create_gluonts_dataset(
        train_df,
        target_column='Daily_MA7',
        freq=FREQ,
        prediction_length=PREDICTION_LENGTH
    )
    
    test_ds = create_gluonts_dataset(
        test_df,
        target_column='Daily_MA7',
        freq=FREQ,
        prediction_length=PREDICTION_LENGTH
    )
    
    # Save datasets
    output_dir = Path('data/gluonts')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metadata
    metadata = save_metadata(output_dir, train_df, test_df, PREDICTION_LENGTH, FREQ)
    
    print(f"\n✓ GluonTS datasets ready!")
    print(f"  Output directory: {output_dir}")
    print(f"  Prediction length: {PREDICTION_LENGTH} days")
    print(f"  Frequency: {FREQ}")
    
    print("\nNext steps: Train a model")

