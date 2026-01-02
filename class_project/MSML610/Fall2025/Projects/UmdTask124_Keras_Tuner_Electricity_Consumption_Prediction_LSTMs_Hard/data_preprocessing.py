"""
data_preprocessing.py

Functions to load and preprocess PJM load data.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

def load_and_engineer(path, processed_dir=Path("data/processed")):
    df = pd.read_csv(path, parse_dates=['Datetime'], index_col='Datetime')
    df = df.sort_index().asfreq('h')
    df['PJM_Load_MW'] = df['PJM_Load_MW'].ffill()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['rolling_24h_mean'] = df['PJM_Load_MW'].rolling(24, min_periods=1).mean()
    df['rolling_7d_mean'] = df['PJM_Load_MW'].rolling(24*7, min_periods=1).mean()
    scaler = MinMaxScaler()
    df['load_scaled'] = scaler.fit_transform(df[['PJM_Load_MW']])
    df['hour_norm'] = df['hour'] / 23.0
    df['dayofweek_norm'] = df['dayofweek'] / 6.0
    df['month_norm'] = (df['month'] - 1) / 11.0
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "pjm_processed_features.csv")
    joblib.dump(scaler, processed_dir / "scaler.pkl")
    return df, scaler
