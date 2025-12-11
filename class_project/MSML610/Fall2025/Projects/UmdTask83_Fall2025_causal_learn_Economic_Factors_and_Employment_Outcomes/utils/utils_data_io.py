"""
Data I/O utilities for loading and preprocessing economic data.

This module provides functions for:
- Loading economic data from CSV files
- Time-aligning economic indicators and employment outcomes
- Creating derived features (wage growth, inflation-adjusted metrics)
- Handling missing values and outliers
- Preparing data for causal discovery
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_economic_data(file_path: str = None) -> pd.DataFrame:
    """
    Load economic data from CSV file.
    
    Args:
        file_path: Path to the CSV file (default: data/economic_data.csv)
        
    Returns:
        DataFrame containing the loaded data
        
    Example:
        >>> df = load_economic_data()
        >>> print(df.shape)
    """
    if file_path is None:
        # Default to project data directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, 'data', 'economic_data.csv')
    
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        logger.info("Run: python3 data/download_data.py to download data from FRED")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def time_align_data(
    df: pd.DataFrame,
    time_column: str = 'date',
    frequency: str = 'MS'
) -> pd.DataFrame:
    """
    Time-align economic indicators to a regular frequency.
    
    Args:
        df: Input DataFrame
        time_column: Name of the time/date column
        frequency: Pandas frequency string ('MS'=month start, 'Q'=quarterly, 'A'=annual)
        
    Returns:
        DataFrame with time-aligned data
        
    Example:
        >>> aligned_df = time_align_data(df, frequency='MS')
    """
    logger.info(f"Time-aligning data to frequency: {frequency}")
    
    df = df.copy()
    
    # Ensure datetime type
    if df[time_column].dtype != 'datetime64[ns]':
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    
    # Sort by time
    df = df.sort_values(time_column).reset_index(drop=True)
    
    # Set index and resample
    df = df.set_index(time_column)
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Resample to specified frequency
    df = df[numeric_cols].resample(frequency).mean()
    
    df = df.reset_index()
    
    logger.info(f"Time-aligned dataset shape: {df.shape}")
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw economic data.
    
    Creates:
    - Real wage growth (nominal - inflation)
    - Employment rate from unemployment
    - Lag features for causal analysis
    
    Args:
        df: Input DataFrame with economic data
        
    Returns:
        DataFrame with additional derived features
    """
    logger.info("Creating derived features")
    
    df = df.copy()
    
    # Real wage growth (inflation-adjusted)
    if 'wage_growth' in df.columns and 'inflation_rate' in df.columns:
        df['real_wage_growth'] = df['wage_growth'] - df['inflation_rate']
    
    # Employment rate from unemployment
    if 'unemployment_rate' in df.columns:
        df['employment_rate'] = 100 - df['unemployment_rate']
    
    # Create lag features for temporal analysis
    lag_columns = ['unemployment_rate', 'inflation_rate', 'gdp_growth']
    for col in lag_columns:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag3'] = df[col].shift(3)
    
    # Create rate of change features
    change_columns = ['unemployment_rate', 'federal_funds_rate']
    for col in change_columns:
        if col in df.columns:
            df[f'{col}_change'] = df[col].diff()
    
    logger.info(f"Created derived features. New shape: {df.shape}")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'interpolate',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        method: Method for handling missing values 
                ('interpolate', 'forward_fill', 'backward_fill', 'drop')
        columns: Specific columns to process (None for all numeric columns)
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info(f"Handling missing values using method: {method}")
    
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns:
            if method == 'interpolate':
                df[col] = df[col].interpolate(method='linear')
            elif method == 'forward_fill':
                df[col] = df[col].ffill()
            elif method == 'backward_fill':
                df[col] = df[col].bfill()
            elif method == 'drop':
                df = df.dropna(subset=[col])
    
    logger.info(f"Missing values handled. Remaining shape: {df.shape}")
    return df


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'iqr',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Remove outliers from specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to process
        method: Method for outlier detection ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    logger.info(f"Removing outliers from columns: {columns}")
    
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
    
    logger.info(f"Outliers removed. New shape: {df.shape}")
    return df


def prepare_features_for_causal_discovery(
    df: pd.DataFrame,
    variables: List[str]
) -> pd.DataFrame:
    """
    Prepare features for causal discovery algorithms.
    
    Args:
        df: Input DataFrame
        variables: List of variable names to include
        
    Returns:
        DataFrame with selected variables, ready for causal discovery
        
    Example:
        >>> causal_df = prepare_features_for_causal_discovery(df, 
        ...     variables=['inflation_rate', 'unemployment_rate', 'wage_growth'])
    """
    logger.info(f"Preparing features for causal discovery: {variables}")
    
    # Select only specified variables
    available_vars = [v for v in variables if v in df.columns]
    missing_vars = [v for v in variables if v not in df.columns]
    
    if missing_vars:
        logger.warning(f"Missing variables: {missing_vars}")
    
    if not available_vars:
        raise ValueError(f"None of the requested variables found. Available: {list(df.columns)}")
    
    causal_df = df[available_vars].copy()
    
    # Remove any remaining missing values
    causal_df = causal_df.dropna()
    
    logger.info(f"Prepared {len(causal_df)} observations with {len(available_vars)} variables")
    return causal_df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
        'missing_values': df.isnull().sum().to_dict(),
        'date_range': None,
    }
    
    # Add date range if date column exists
    if 'date' in df.columns:
        summary['date_range'] = {
            'start': str(df['date'].min()),
            'end': str(df['date'].max()),
            'count': len(df)
        }
    
    # Add numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    return summary


def get_economic_data(force_download: bool = False) -> pd.DataFrame:
    """
    Get economic data - load from cache or download from FRED.
    
    Args:
        force_download: Force re-download even if cached data exists
        
    Returns:
        DataFrame with economic data
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'economic_data.csv')
    
    if os.path.exists(data_path) and not force_download:
        return load_economic_data(data_path)
    
    # Try to download
    try:
        import sys
        sys.path.insert(0, os.path.join(project_root, 'data'))
        from download_data import get_data
        return get_data(force_download=force_download)
    except ImportError:
        raise ImportError(
            "Data not found. Run: python3 data/download_data.py\n"
            "Requires: export FRED_API_KEY=your_key"
        )
