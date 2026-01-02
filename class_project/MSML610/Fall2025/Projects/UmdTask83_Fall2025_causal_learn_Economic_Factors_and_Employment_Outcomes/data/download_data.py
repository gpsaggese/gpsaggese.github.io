"""
Download economic data from FRED (Federal Reserve Economic Data).

FRED provides clean, ready-to-use macroeconomic time series including:
- Unemployment rate
- Inflation (CPI)
- GDP growth
- Average hourly earnings (wages)
- Federal funds rate

Usage:
    python3 Data/download_data.py
    
Requires:
    pip3 install fredapi pandas

FRED API Key (free):
    1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
    2. Create account and get API key
    3. Set: export FRED_API_KEY=your_key_here
"""

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def download_fred_data(api_key=None, start_date='2000-01-01'):
    """
    Download economic data from FRED API.
    
    Series downloaded:
    - UNRATE: Unemployment Rate
    - CPIAUCSL: Consumer Price Index (for inflation rate)
    - FEDFUNDS: Federal Funds Rate
    - CES0500000003: Average Hourly Earnings
    - PAYEMS: Total Nonfarm Payrolls
    - A191RL1Q225SBEA: Real GDP Growth (quarterly)
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "fredapi not installed. Run: pip3 install fredapi"
        )
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get('FRED_API_KEY')
    
    if api_key is None:
        raise ValueError(
            "FRED API key required.\n"
            "1. Get free key: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "2. Set: export FRED_API_KEY=your_key_here"
        )
    
    logger.info("Connecting to FRED API...")
    fred = Fred(api_key=api_key)
    
    # Series to download
    series_map = {
        'UNRATE': 'unemployment_rate',
        'CPIAUCSL': 'cpi',
        'FEDFUNDS': 'federal_funds_rate',
        'CES0500000003': 'avg_hourly_earnings',
        'PAYEMS': 'total_employment',
        'A191RL1Q225SBEA': 'gdp_growth_quarterly',
    }
    
    data_dict = {}
    
    for series_id, col_name in series_map.items():
        try:
            logger.info(f"Downloading {series_id} -> {col_name}")
            series = fred.get_series(series_id, observation_start=start_date)
            data_dict[col_name] = series
            logger.info(f"  Got {len(series)} observations")
        except Exception as e:
            logger.warning(f"Failed to download {series_id}: {e}")
    
    if len(data_dict) == 0:
        raise ValueError("No data downloaded from FRED")
    
    # Combine into DataFrame
    df = pd.DataFrame(data_dict)
    df.index.name = 'date'
    df = df.reset_index()
    
    # Resample to monthly
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').resample('MS').mean().reset_index()
    
    # Calculate derived variables
    logger.info("Calculating derived variables...")
    
    # Inflation rate (YoY % change in CPI)
    if 'cpi' in df.columns:
        df['inflation_rate'] = df['cpi'].pct_change(12) * 100
    
    # Wage growth (YoY % change)
    if 'avg_hourly_earnings' in df.columns:
        df['wage_growth'] = df['avg_hourly_earnings'].pct_change(12) * 100
    
    # Employment growth
    if 'total_employment' in df.columns:
        df['employment_growth'] = df['total_employment'].pct_change(12) * 100
    
    # Real wage growth
    if 'wage_growth' in df.columns and 'inflation_rate' in df.columns:
        df['real_wage_growth'] = df['wage_growth'] - df['inflation_rate']
    
    # Interpolate quarterly GDP to monthly
    if 'gdp_growth_quarterly' in df.columns:
        df['gdp_growth'] = df['gdp_growth_quarterly'].interpolate(method='linear')
    
    # Drop rows with too many NaN
    df = df.dropna(thresh=4).reset_index(drop=True)
    
    logger.info(f"Final data shape: {df.shape}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def load_data(filename='economic_data.csv'):
    """Load previously downloaded data."""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=['date'])
        logger.info(f"Loaded {len(df)} rows from {filename}")
        return df
    else:
        raise FileNotFoundError(f"Data file not found: {filepath}")


def save_data(df, filename='economic_data.csv'):
    """Save data to CSV."""
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved to: {filepath}")


def get_data(api_key=None, force_download=False):
    """
    Get economic data - download if needed, otherwise load from cache.
    """
    cache_path = os.path.join(DATA_DIR, 'economic_data.csv')
    
    if os.path.exists(cache_path) and not force_download:
        logger.info("Loading cached data...")
        return load_data()
    
    logger.info("Downloading from FRED...")
    df = download_fred_data(api_key)
    save_data(df)
    return df


def main():
    """Download and save FRED data."""
    print("=" * 60)
    print("FRED Economic Data Downloader")
    print("=" * 60)
    
    try:
        df = download_fred_data()
        save_data(df)
        
        print("\n" + "=" * 60)
        print("SUCCESS - Data downloaded and saved")
        print("=" * 60)
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print("\nSummary:")
        print(df.describe().round(2))
        
    except ValueError as e:
        print(f"\nERROR: {e}")
        return 1
    except Exception as e:
        print(f"\nFAILED: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
