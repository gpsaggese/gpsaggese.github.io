import pandas as pd
import numpy as np

def create_lagged_features(weekly, lag_weeks=3):
    """Create lagged features for causal analysis"""
    weekly = weekly.sort_values(['country_code','week_start']).reset_index(drop=True)
    
    weekly['vac_pct_lag'] = weekly.groupby('country_code')['vac_pct'].shift(lag_weeks)
    weekly['cases_per_100k_lag'] = weekly.groupby('country_code')['cases_per_100k'].shift(lag_weeks)
    weekly['deaths_per_100k_lag'] = weekly.groupby('country_code')['deaths_per_100k'].shift(lag_weeks)
    
    print(f"Created {lag_weeks}-week lagged features")
    
    # Show first few rows
    feature_cols = ['country_code','week_start','vac_pct','vac_pct_lag','cases_per_100k','cases_per_100k_lag']
    available_cols = [col for col in feature_cols if col in weekly.columns]
    print("Lagged features sample:")
    print(weekly[available_cols].head(10).to_string())
    
    return weekly

def create_rolling_features(weekly, window=3):
    """Create rolling average features"""
    weekly['cases_per_100k_roll3'] = weekly.groupby('country_code')['cases_per_100k'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    weekly['deaths_per_100k_roll3'] = weekly.groupby('country_code')['deaths_per_100k'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    weekly['vac_pct_roll3'] = weekly.groupby('country_code')['vac_pct'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    
    print(f"Created {window}-week rolling averages")
    return weekly