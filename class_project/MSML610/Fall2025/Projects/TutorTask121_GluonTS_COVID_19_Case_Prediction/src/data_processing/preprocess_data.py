"""
Preprocess COVID-19 data for time series forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path

def aggregate_to_national(cases_df):
    """
    Aggregate county-level cases to national level
    
    Args:
        cases_df: DataFrame with time series columns
        
    Returns:
        DataFrame with Date and Daily_Cases columns
    """
    # Get date columns (skip first 11 metadata columns)
    date_columns = cases_df.columns[11:]
    
    # Sum across all counties for each date
    national_cumulative = cases_df[date_columns].sum()
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': pd.to_datetime(national_cumulative.index),
        'Cumulative': national_cumulative.values
    })
    
    # Calculate daily new cases
    df['Daily'] = df['Cumulative'].diff().fillna(0)
    
    # Calculate 7-day moving average
    df['Daily_MA7'] = df['Daily'].rolling(window=7).mean()
    
    # Remove negative values (reporting corrections)
    df['Daily'] = df['Daily'].clip(lower=0)
    
    return df


def extract_national_mobility(mobility_df):
    """
    Extract national-level mobility data
    
    Args:
        mobility_df: DataFrame with mobility data
        
    Returns:
        DataFrame with national mobility trends
    """
    # Filter for national level (state='Total', county='Total')
    national = mobility_df[
        (mobility_df['state'] == 'Total') & 
        (mobility_df['county'] == 'Total')
    ].copy()
    
    # Sort by date
    national = national.sort_values('date').reset_index(drop=True)
    
    return national


def merge_cases_and_mobility(cases_df, mobility_df):
    """
    Merge cases and mobility data on date
    
    Args:
        cases_df: DataFrame with Date and Daily columns
        mobility_df: DataFrame with date and mobility columns
        
    Returns:
        Merged DataFrame
    """
    # Ensure date columns are datetime
    cases_df['Date'] = pd.to_datetime(cases_df['Date'])
    mobility_df['date'] = pd.to_datetime(mobility_df['date'])
    
    # Merge on date
    merged = pd.merge(
        cases_df,
        mobility_df,
        left_on='Date',
        right_on='date',
        how='left'
    )
    
    # Drop duplicate date column
    merged = merged.drop('date', axis=1)
    
    # Forward fill missing mobility values
    mobility_cols = ['retail and recreation', 'grocery and pharmacy', 
                     'parks', 'transit stations', 'workplaces', 'residential']
    merged[mobility_cols] = merged[mobility_cols].fillna(method='ffill')
    
    return merged


if __name__ == "__main__":
    # Import DataLoader so this script can be run either from the
    # project root (with `src` on PYTHONPATH) or directly from this folder.
    try:
        from data_processing.load_data import DataLoader
    except Exception:
        try:
            from load_data import DataLoader
        except Exception as e:
            raise ImportError("Could not import DataLoader from data_processing.load_data or load_data") from e

    # Load data (explicitly set data_dir to the expected 'data' folder)
    loader = DataLoader(data_dir="data")
    cases_df = loader.load_cases()
    mobility_df = loader.load_mobility()
    
    # Preprocess
    print("\nProcessing cases data...")
    national_cases = aggregate_to_national(cases_df)
    print(f"National cases: {len(national_cases)} days")
    
    print("\nProcessing mobility data...")
    national_mobility = extract_national_mobility(mobility_df)
    print(f"National mobility: {len(national_mobility)} days")
    
    print("\nMerging datasets...")
    merged = merge_cases_and_mobility(national_cases, national_mobility)
    print(f"Merged data: {len(merged)} days")
    
    # Save processed data
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    merged.to_csv(output_dir / 'national_data.csv', index=False)
    print(f"\nSaved to {output_dir / 'national_data.csv'}")

