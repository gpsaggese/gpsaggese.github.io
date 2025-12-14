"""
GluonTS Data Preparation Utilities

Convert COVID-19 data for GluonTS models.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from gluonts.dataset.common import ListDataset


def create_gluonts_dataset(
    df: pd.DataFrame,
    target_column: str,
    freq: str = 'D',
    prediction_length: int = 14,
    past_feat_columns: Optional[List[str]] = None
) -> ListDataset:
    """Convert pandas DataFrame to GluonTS ListDataset format."""
    # Ensure we have a date column
    if 'Date' not in df.columns and 'date' not in df.columns:
        raise ValueError("DataFrame must have a 'Date' or 'date' column")
    
    date_col = 'Date' if 'Date' in df.columns else 'date'
    
    start_date = pd.to_datetime(df[date_col].iloc[0])
    target = df[target_column].values.tolist()
    
    # Create dataset entry
    data_entry = {
        "start": start_date,
        "target": target
    }
    
    # Add dynamic features if specified
    if past_feat_columns:
        feat_dynamic_real = []
        for col in past_feat_columns:
            if col in df.columns:
                feat_dynamic_real.append(df[col].values.tolist())
            else:
                print(f"Warning: Column '{col}' not found in DataFrame")
        
        if feat_dynamic_real:
            data_entry["feat_dynamic_real"] = feat_dynamic_real
    
    # Create ListDataset
    dataset = ListDataset([data_entry], freq=freq)
    return dataset


def verify_dataset(dataset: ListDataset, name: str = "Dataset") -> None:
    """Verify and print information about a GluonTS dataset."""
    try:
        data_list = list(dataset)
        
        print(f"\n{name} Dataset Info:")
        print("=" * 50)
        print("Valid GluonTS ListDataset")
        print(f"  Number of time series: {len(data_list)}")
        
        if data_list:
            first_entry = data_list[0]
            print(f"  Start date: {first_entry['start']}")
            print(f"  Target length: {len(first_entry['target'])} points")
            
            if 'feat_dynamic_real' in first_entry:
                n_features = len(first_entry['feat_dynamic_real'])
                print(f"  Dynamic features: Yes ({n_features} features)")
            else:
                print("  Dynamic features: No")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"Error verifying dataset: {e}")


def prepare_train_test_split(
    full_df: pd.DataFrame,
    test_size: int = 14,
    target_column: str = 'Daily_Cases_MA7'
) -> tuple:
    """Split DataFrame into train and test sets for time series."""
    df_clean = full_df.dropna(subset=[target_column]).copy()
    
    split_idx = len(df_clean) - test_size
    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()
    
    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(train_df)} days ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"  Test:  {len(test_df)} days ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """Get list of potential feature columns from DataFrame."""
    if exclude_cols is None:
        exclude_cols = ['Date', 'date']
    
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and 
                   df[col].dtype in ['int64', 'float64']]
    
    return feature_cols


def summary_statistics(df: pd.DataFrame, target_column: str) -> None:
    """Print summary statistics for the target variable."""
    print(f"\n{target_column} Statistics:")
    print("=" * 50)
    print(f"  Count:  {df[target_column].count()}")
    print(f"  Mean:   {df[target_column].mean():.2f}")
    print(f"  Median: {df[target_column].median():.2f}")
    print(f"  Std:    {df[target_column].std():.2f}")
    print(f"  Min:    {df[target_column].min():.2f}")
    print(f"  Max:    {df[target_column].max():.2f}")
    print("=" * 50)
