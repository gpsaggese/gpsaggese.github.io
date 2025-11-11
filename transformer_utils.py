"""
transformer_utils.py
--------------------
Data acquisition and feature engineering utilities
for Time-Series Transformer forecasting.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np

def download_stock_data(ticker='AAPL', start='2020-01-01', end='2025-01-01'):
    """
    Download stock data and create engineered features.
    """
    os.makedirs('data', exist_ok=True)

    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Feature engineering
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Return'].rolling(window=10).std()

    # Drop NaN values
    df.dropna(inplace=True)

    # Save processed data
    df.to_csv('data/raw_data.csv', index=True)
    print(f"📂 Data saved to data/raw_data.csv with {len(df)} rows.")
    return df
