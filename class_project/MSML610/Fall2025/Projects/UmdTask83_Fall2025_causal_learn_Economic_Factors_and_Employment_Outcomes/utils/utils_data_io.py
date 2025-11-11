"""
Data I/O utilities for loading and preprocessing US Labor Statistics data.

This module provides functions for:
- Loading labor statistics data from CSV files
- Merging multiple data sources
- Time-aligning economic indicators and employment outcomes
- Creating derived features (wage growth, inflation-adjusted metrics)
- Handling missing values and outliers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_labor_data(file_path: str) -> pd.DataFrame:
    """
    Load US Labor Statistics data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
        
    Example:
        >>> df = load_labor_data('Data/all.data.combined.csv')
        >>> print(df.shape)
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def merge_labor_datasets(
    main_df: pd.DataFrame,
    series_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    merge_key: str = 'series_id'
) -> pd.DataFrame:
    """
    Merge main labor statistics data with supporting metadata.
    
    Args:
        main_df: Main dataset with labor statistics
        series_df: Series definitions and metadata
        industry_df: Industry classifications
        merge_key: Key column for merging
        
    Returns:
        Merged DataFrame with additional metadata columns
        
    Example:
        >>> combined_df = merge_labor_datasets(main_df, series_df, industry_df)
    """
    logger.info("Merging labor statistics datasets")
    
    # Merge series information
    merged_df = main_df.merge(series_df, on=merge_key, how='left', suffixes=('', '_series'))
    
    # Merge industry information
    merged_df = merged_df.merge(industry_df, on='industry_id', how='left', suffixes=('', '_industry'))
    
    logger.info(f"Merged dataset shape: {merged_df.shape}")
    return merged_df


def time_align_data(
    df: pd.DataFrame,
    time_column: str = 'period',
    frequency: str = 'monthly'
) -> pd.DataFrame:
    """
    Time-align economic indicators and employment outcomes.
    
    Args:
        df: Input DataFrame
        time_column: Name of the time/date column
        frequency: Data frequency ('monthly', 'quarterly', 'annual')
        
    Returns:
        DataFrame with time-aligned data
        
    Example:
        >>> aligned_df = time_align_data(combined_df, time_column='period', frequency='monthly')
    """
    logger.info(f"Time-aligning data with frequency: {frequency}")
    
    # Convert time column to datetime if needed
    if df[time_column].dtype == 'object':
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    
    # Sort by time
    df = df.sort_values(time_column).reset_index(drop=True)
    
    # Resample to specified frequency if needed
    if frequency == 'monthly':
        df = df.set_index(time_column).resample('M').mean().reset_index()
    elif frequency == 'quarterly':
        df = df.set_index(time_column).resample('Q').mean().reset_index()
    elif frequency == 'annual':
        df = df.set_index(time_column).resample('Y').mean().reset_index()
    
    logger.info(f"Time-aligned dataset shape: {df.shape}")
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw labor statistics data.
    
    Creates:
    - Wage growth rate (percentage change)
    - Inflation-adjusted wages (real wages)
    - Employment rate (employment-to-population ratio)
    - Year-over-year changes for key indicators
    
    Args:
        df: Input DataFrame with raw labor statistics
        
    Returns:
        DataFrame with additional derived features
        
    Example:
        >>> processed_df = create_derived_features(aligned_df)
    """
    logger.info("Creating derived features")
    
    df = df.copy()
    
    # Calculate wage growth rate (if avg_hourly_earnings exists)
    if 'avg_hourly_earnings' in df.columns:
        df['wage_growth'] = df.groupby('series_id')['avg_hourly_earnings'].pct_change() * 100
    
    # Calculate inflation-adjusted wages (if both exist)
    if 'avg_hourly_earnings' in df.columns and 'inflation_rate' in df.columns:
        df['real_wages'] = df['avg_hourly_earnings'] / (1 + df['inflation_rate'] / 100)
    
    # Calculate employment rate (if components exist)
    if 'employed' in df.columns and 'labor_force' in df.columns:
        df['employment_rate'] = (df['employed'] / df['labor_force']) * 100
    
    # Calculate year-over-year changes
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in ['inflation_rate', 'unemployment_rate', 'gdp_growth']:
        if col in df.columns:
            df[f'{col}_yoy'] = df.groupby('series_id')[col].pct_change(periods=12) * 100
    
    logger.info(f"Created derived features. New shape: {df.shape}")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'forward_fill',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        method: Method for handling missing values ('forward_fill', 'backward_fill', 'interpolate', 'drop')
        columns: Specific columns to process (None for all columns)
        
    Returns:
        DataFrame with missing values handled
        
    Example:
        >>> df_clean = handle_missing_values(df, method='interpolate')
    """
    logger.info(f"Handling missing values using method: {method}")
    
    df = df.copy()
    cols_to_process = columns if columns else df.columns
    
    for col in cols_to_process:
        if col in df.columns:
            if method == 'forward_fill':
                df[col] = df[col].fillna(method='ffill')
            elif method == 'backward_fill':
                df[col] = df[col].fillna(method='bfill')
            elif method == 'interpolate':
                df[col] = df[col].interpolate(method='linear')
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
        
    Example:
        >>> df_clean = remove_outliers(df, columns=['wage_growth', 'inflation_rate'], method='iqr')
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
        
    Example:
        >>> summary = get_data_summary(df)
        >>> print(summary['shape'])
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    }
    
    return summary

