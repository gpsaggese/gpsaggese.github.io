"""
Preprocessing utilities for OWID compact COVID dataset
"""

import os
import pandas as pd
import numpy as np

def clean_data_minimal(df):
    """Minimal cleaning before weekly aggregation (keep raw structure)."""

    # --- Standardize ISO3 country code column ---
    if 'code' in df.columns:
        df = df.rename(columns={'code': 'country_code'})
    elif 'iso_code' in df.columns:
        df = df.rename(columns={'iso_code': 'country_code'})
    else:
        raise ValueError("No ISO3 country code found (expected `code` or `iso_code`).")

    # Drop non-valid ISO codes (strings of length 3 only)
    df = df[df['country_code'].str.len() == 3].copy()

    # Essential columns required throughout analysis
    essential_cols = [
        'country_code', 'country', 'date', 'continent',
        'people_vaccinated_per_hundred',
        'new_cases_per_million',
        'new_deaths_per_million',
        'population_density',
        'median_age',
        'hospital_beds_per_thousand',
        'gdp_per_capita'
    ]

    # Keep only columns that exist
    df = df[[c for c in essential_cols if c in df.columns]].copy()

    # Rename to match notebook variables
    rename_map = {
        'people_vaccinated_per_hundred': 'vac_pct',
        'new_cases_per_million': 'cases_per_100k',
        'new_deaths_per_million': 'deaths_per_100k',
    }
    df = df.rename(columns=rename_map)

    return df



def build_weekly_panel(df):
    """Convert country-day data into country-week aggregated panel dataset."""

    # Floor date to weekly frequency (week start Monday)
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='d')

    # Aggregate per week per country-code
    weekly = df.groupby(['country_code', 'week_start'], as_index=False).agg({
        'vac_pct': 'mean',
        'cases_per_100k': 'sum',
        'deaths_per_100k': 'sum',
        'continent': 'first',
        'population_density': 'first',
        'median_age': 'first',
        'hospital_beds_per_thousand': 'first',
        'gdp_per_capita': 'first'
    })

    # Remove missing key data
    weekly = weekly.dropna(subset=['vac_pct', 'cases_per_100k', 'deaths_per_100k'])

    print(weekly.head(5))

    return weekly


def final_clean(df):
    """Final clean before causal analysis & feature engineering."""

    # Remove obvious errors / below-zero values
    df = df[df['cases_per_100k'] >= 0]
    df = df[df['deaths_per_100k'] >= 0]
    df = df[df['vac_pct'] >= 0]

    # Remove rows where vaccination is missing
    df = df.dropna(subset=['vac_pct', 'cases_per_100k', 'deaths_per_100k'])

    # Drop any duplicate rows
    df = df.drop_duplicates(subset=['country_code', 'week_start'])

    # Ensure dataset is sorted properly
    df = df.sort_values(['country_code', 'week_start']).reset_index(drop=True)

    print(f"Final clean done: {df.shape[0]:,} rows remain after validation checks")
    return df
