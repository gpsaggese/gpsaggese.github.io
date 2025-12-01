import pandas as pd
import numpy as np
from pathlib import Path

def filter_vaccine_era(df, start_date="2021-01-01"):
    """Filter data to vaccine era period"""
    df = df[df['date'] >= pd.Timestamp(start_date)].copy()
    print(f"After time filter: {len(df):,} rows")
    return df

def remove_aggregates(df):
    """Remove OWID aggregate regions"""
    invalid_codes = ['OWID_WRL','OWID_AFR','OWID_ASI','OWID_EUR','OWID_EUN','OWID_INT','OWID_KOS','OWID_OWID']
    if 'code' in df.columns:
        before = len(df)
        df = df[~df['code'].isin(invalid_codes)].copy()
        print(f"Removed {before - len(df):,} aggregate rows; remaining: {len(df):,}")
    
    countries = df['code'].nunique()
    print(f"Unique countries: {countries}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df

def create_weekly_panel(df):
    """Create weekly aggregated panel data"""
    df['week_start'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
    
    agg_dict = {
        'new_cases': 'sum',
        'new_deaths': 'sum',
        'new_vaccinations': 'sum',
        'new_cases_smoothed': 'mean',
        'new_vaccinations_smoothed': 'mean',
        'stringency_index': 'mean',
        'total_cases': 'max',
        'total_deaths': 'max',
        'total_vaccinations': 'max',
        'people_vaccinated': 'max',
        'people_fully_vaccinated': 'max',
        'total_vaccinations_per_hundred': 'max',
        'people_vaccinated_per_hundred': 'max',
        'people_fully_vaccinated_per_hundred': 'max',
        'population': 'last',
        'population_density': 'last',
        'median_age': 'last',
        'hospital_beds_per_thousand': 'last',
        'gdp_per_capita': 'last',
        'continent': 'last',
        'country': 'last'
    }
    
    # Keep only columns present in df
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    weekly = df.groupby(['code', 'week_start'], as_index=False).agg(agg_dict)
    weekly.rename(columns={'code': 'country_code', 'country': 'country'}, inplace=True)
    
    print(f"Weekly panel created: {weekly.shape}")
    return weekly

def compute_metrics(weekly):
    """Compute per-100k metrics and vaccination percentage"""
    weekly['cases_per_100k'] = weekly['new_cases'] / weekly['population'] * 100000
    weekly['deaths_per_100k'] = weekly['new_deaths'] / weekly['population'] * 100000
    
    if 'people_vaccinated_per_hundred' in weekly.columns:
        weekly['vac_pct'] = weekly['people_vaccinated_per_hundred']
    elif 'people_vaccinated' in weekly.columns:
        weekly['vac_pct'] = weekly['people_vaccinated'] / weekly['population'] * 100
    else:
        weekly['vac_pct'] = np.nan
    
    print("Metrics computed: cases_per_100k, deaths_per_100k, vac_pct")
    return weekly

def clean_data(weekly):
    """Clean and impute missing values"""
    weekly = weekly.dropna(subset=['country_code', 'week_start', 'vac_pct', 
                                  'cases_per_100k', 'deaths_per_100k'])
    
    covars = ['population_density', 'median_age', 'hospital_beds_per_thousand', 'gdp_per_capita']
    for c in covars:
        if c in weekly.columns:
            median_val = weekly[c].median()
            weekly[c] = weekly[c].fillna(median_val)
    
    print(f"After cleaning: {len(weekly):,} rows")
    print("Missing values after cleaning:")
    key_cols = ['vac_pct', 'cases_per_100k', 'deaths_per_100k'] + covars
    key_cols_present = [c for c in key_cols if c in weekly.columns]
    print(weekly[key_cols_present].isna().sum().to_string())
    
    return weekly