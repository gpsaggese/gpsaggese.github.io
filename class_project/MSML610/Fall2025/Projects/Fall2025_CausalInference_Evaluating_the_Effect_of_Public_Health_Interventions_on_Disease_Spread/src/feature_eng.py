import pandas as pd
import numpy as np

def create_lagged_features(weekly, lag_weeks=3):
    """Create lagged features for causal analysis"""
    weekly = weekly.sort_values(['country_code', 'week_start']).reset_index(drop=True)

    weekly['vac_pct_lag'] = weekly.groupby('country_code')['vac_pct'].shift(lag_weeks)
    weekly['cases_per_100k_lag'] = weekly.groupby('country_code')['cases_per_100k'].shift(lag_weeks)
    weekly['deaths_per_100k_lag'] = weekly.groupby('country_code')['deaths_per_100k'].shift(lag_weeks)

    return weekly


def create_rolling_features(weekly, window=3):
    """Create rolling average features"""
    weekly['cases_per_100k_roll3'] = (
        weekly.groupby('country_code')['cases_per_100k']
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    weekly['deaths_per_100k_roll3'] = (
        weekly.groupby('country_code')['deaths_per_100k']
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    weekly['vac_pct_roll3'] = (
        weekly.groupby('country_code')['vac_pct']
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    return weekly


def add_features(weekly):
    """Unified feature engineering wrapper"""
    weekly = create_lagged_features(weekly, lag_weeks=3)
    weekly = create_rolling_features(weekly, window=3)
    return weekly
