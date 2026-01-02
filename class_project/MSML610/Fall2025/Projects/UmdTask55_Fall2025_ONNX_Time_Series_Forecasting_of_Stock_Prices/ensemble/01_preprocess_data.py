import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# CUDA fixes for WSL
wsl_cuda_path = '/usr/lib/wsl/lib'
if os.path.exists(wsl_cuda_path):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if wsl_cuda_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{wsl_cuda_path}:{current_ld_path}"
        print(f"Added WSL CUDA library path: {wsl_cuda_path}")

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from onnx_forecasting_utils import (
    load_and_stack_mag7_stocks,
    prepare_mag7_features,
    split_data_chronological,
    create_rolling_windows
)

sns.set_style('whitegrid')
print("All libraries imported successfully!")

MAG7_STOCKS = {
    'GOOG': 'data/Stocks/goog.us.txt',
    'AAPL': 'data/Stocks/aapl.us.txt',
    'AMZN': 'data/Stocks/amzn.us.txt',
    'META': 'data/Stocks/fb.us.txt',
    'NVDA': 'data/Stocks/nvda.us.txt',
    'TSLA': 'data/Stocks/tsla.us.txt',
    'MSFT': 'data/Stocks/msft.us.txt'
}

print("Loading MAG 7 stocks...")
stacked_df = load_and_stack_mag7_stocks(MAG7_STOCKS, apply_features=True)

print(f"\n{'='*60}")
print(f"Stacked Dataset Summary:")
print(f"{'='*60}")
print(f"Total rows: {len(stacked_df)}")
print(f"Date range: {stacked_df['Date'].min()} to {stacked_df['Date'].max()}")
print(f"Stocks: {stacked_df['stock'].unique()}")
print(f"\nStock distribution:")
print(stacked_df['stock'].value_counts().sort_index())

fig, axes = plt.subplots(4, 2, figsize=(20, 16))
axes = axes.flatten()

stocks = sorted(stacked_df['stock'].unique())

for idx, stock in enumerate(stocks):
    stock_data = stacked_df[stacked_df['stock'] == stock].copy()

    ax = axes[idx]
    ax.plot(stock_data['Date'], stock_data['Close'], linewidth=1.5, label=stock)
    ax.set_title(f'{stock} Stock Price History', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

if len(stocks) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
os.makedirs('models', exist_ok=True)
plt.savefig('models/mag7_price_history.png', dpi=300)
print("\nPrice history visualization saved to models/mag7_price_history.png")

feature_df, feature_cols = prepare_mag7_features(stacked_df)

print(f"\nFeature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

print(f"\nFeature matrix shape: {feature_df.shape}")

nan_counts = feature_df.isnull().sum()
if nan_counts.sum() > 0:
    print(f"\nWarning: NaN values found:")
    print(nan_counts[nan_counts > 0])
else:
    print("\nNo NaN values found. Data is clean!")

train_df, val_df, test_df = split_data_chronological(stacked_df, train_ratio=0.7, val_ratio=0.15)

print(f"\nDataset splits:")
print(f"  Train: {len(train_df)} samples ({len(train_df)/len(stacked_df)*100:.1f}%)")
print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(stacked_df)*100:.1f}%)")
print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(stacked_df)*100:.1f}%)")

train_stock_labels = train_df['stock'].values
val_stock_labels = val_df['stock'].values
test_stock_labels = test_df['stock'].values

scaler = MinMaxScaler()

# Fit on training data only
train_features = train_df[feature_cols].values
val_features = val_df[feature_cols].values
test_features = test_df[feature_cols].values

train_normalized = scaler.fit_transform(train_features)
val_normalized = scaler.transform(val_features)
test_normalized = scaler.transform(test_features)

print(f"\nNormalized data shapes:")
print(f"  Train: {train_normalized.shape}")
print(f"  Val:   {val_normalized.shape}")
print(f"  Test:  {test_normalized.shape}")

with open('models/mag7_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("\nScaler saved to models/mag7_scaler.pkl")

# Verify normalization
print(f"\nNormalized data range:")
print(f"  Min: {train_normalized.min():.4f}")
print(f"  Max: {train_normalized.max():.4f}")


SEQUENCE_LENGTH = 15

X_train, y_train = create_rolling_windows(train_normalized, window_size=SEQUENCE_LENGTH, step_size=1)
X_val, y_val = create_rolling_windows(val_normalized, window_size=SEQUENCE_LENGTH, step_size=1)
X_test, y_test = create_rolling_windows(test_normalized, window_size=SEQUENCE_LENGTH, step_size=1)

y_train = y_train[:, 0:1]  # Shape: (n_samples, 1)
y_val = y_val[:, 0:1]
y_test = y_test[:, 0:1]

print(f"\nSequence data shapes:")
print(f"  X_train: {X_train.shape} (samples, timesteps, features)")
print(f"  y_train: {y_train.shape} (samples, output_dim)")
print(f"  X_val:   {X_val.shape}")
print(f"  y_val:   {y_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_test:  {y_test.shape}")

# Adjust stock labels to match sequence indices
train_stock_labels_seq = train_stock_labels[SEQUENCE_LENGTH:]
val_stock_labels_seq = val_stock_labels[SEQUENCE_LENGTH:]
test_stock_labels_seq = test_stock_labels[SEQUENCE_LENGTH:]

print(f"\nAdjusted stock labels shapes:")
print(f"  Train: {train_stock_labels_seq.shape}")
print(f"  Val:   {val_stock_labels_seq.shape}")
print(f"  Test:  {test_stock_labels_seq.shape}")


processed_data = {
    # Sequences for LSTM, TCN, XGBoost
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test,

    # Normalized data for TCN (DARTS needs full arrays)
    'train_normalized': train_normalized,
    'val_normalized': val_normalized,
    'test_normalized': test_normalized,

    # Stock labels (adjusted for sequences)
    'train_stock_labels': train_stock_labels_seq,
    'val_stock_labels': val_stock_labels_seq,
    'test_stock_labels': test_stock_labels_seq,

    # Metadata
    'feature_cols': feature_cols,
    'stocks': stocks,
    'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
    'scaler': scaler
}

output_path = 'ensemble/processed_data.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(processed_data, f)

print(f"\n{'='*60}")
print(f"DATA PREPROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Processed data saved to: {output_path}")
print(f"Total size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
print("\nReady for training scripts!")
