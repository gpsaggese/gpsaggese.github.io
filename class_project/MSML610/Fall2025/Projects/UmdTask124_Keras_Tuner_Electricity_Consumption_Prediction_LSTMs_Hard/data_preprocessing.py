import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_engineer(file_path: str):
    df = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')
    df = df.asfreq('h')
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
    return df, scaler
