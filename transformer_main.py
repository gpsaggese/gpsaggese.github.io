# transformer_utils.py
"""
Data acquisition and feature engineering utilities for
Time-Series Transformer forecasting (Xformers-ready).
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Ensure folders
os.makedirs("data", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

def download_stock_data(ticker='AAPL', start='2020-01-01', end='2025-01-01', save_path='data/raw_data.csv'):
    """
    Download stock data and create basic engineered features.
    Saves processed CSV to save_path (creates parent dir if needed).
    """
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Feature engineering
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Return'].rolling(window=10).std()

    df = df.dropna()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=True)
    print(f"📂 Data saved to {save_path} with {len(df)} rows.")
    return df

def create_features_df(df):
    """Return DataFrame with selected engineered columns."""
    cols = ['Close', 'Return', 'MA10', 'Volatility']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered columns: {missing}. Run download_stock_data() first.")
    return df[cols].copy()

def get_scaler_and_scale(df_features):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_features.values)
    return scaler, scaled

def create_sequences_from_df(df_features, seq_len=30):
    data = df_features.values
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def make_dataloaders(X, y, batch_size=64, val_split=0.2, shuffle=False):
    n = len(X)
    split = int((1 - val_split) * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, (X_val_t, y_val_t)

def plot_predictions(y_true, y_pred, scaler=None, save_path="results/plots/pred_vs_true.png", title="Actual vs Predicted"):
    if scaler is not None:
        pad = np.zeros((len(y_true), scaler.n_features_in_ - 1))
        y_true_inv = scaler.inverse_transform(np.concatenate([y_true.reshape(-1,1), pad], axis=1))[:,0]
        y_pred_inv = scaler.inverse_transform(np.concatenate([y_pred.reshape(-1,1), pad], axis=1))[:,0]
    else:
        y_true_inv, y_pred_inv = y_true, y_pred

    plt.figure(figsize=(12,5))
    plt.plot(y_true_inv, label='Actual')
    plt.plot(y_pred_inv, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Close')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📈 Plot saved to {save_path}")
    plt.show()

def save_metrics(metrics_dict, csv_path='results/metrics.csv'):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([metrics_dict])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    print(f"📊 Metrics appended to {csv_path}")
