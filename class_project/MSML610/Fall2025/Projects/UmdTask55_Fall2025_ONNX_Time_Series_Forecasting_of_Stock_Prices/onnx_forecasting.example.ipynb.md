---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

# Stock Price Forecasting with ONNX

This notebook demonstrates end-to-end stock price forecasting using LSTM and ONNX deployment.

---

## Cell 1: Import Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from onnx_forecasting_utils import (
    load_stock_data, parse_and_sort_dates, handle_missing_values,
    detect_and_handle_outliers, split_data_chronological, normalize_data,
    calculate_moving_averages, calculate_bollinger_bands, calculate_atr,
    calculate_rsi, calculate_macd, calculate_volume_indicators,
    apply_all_features, create_rolling_windows,
    LSTMConfig, build_lstm_model, compile_model, create_and_train_lstm,
    convert_to_onnx, verify_onnx, ONNXInferenceSession,
    compare_frameworks_inference,
    evaluate_forecasts, plot_predictions_vs_actual, plot_residuals,
    create_forecast_report, compare_models
)

sns.set_style('whitegrid')
print("All libraries imported successfully!")
```

---

## Cell 2: Load and Explore Stock Data

```python
data_path = 'data/Stocks/goog.us.txt'
df = load_stock_data(data_path, date_column='Date')

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset info:")
print(df.info())

print(f"\nMissing values:")
print(df.isnull().sum())
```

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].plot(df['Date'], df['Close'], linewidth=1)
axes[0, 0].set_title('GOOG Stock Price (Close)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].bar(df['Date'], df['Volume'], width=1, alpha=0.7)
axes[0, 1].set_title('GOOG Trading Volume', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Volume')
axes[0, 1].grid(True, alpha=0.3)

price_changes = df['Close'].pct_change()
axes[1, 0].hist(price_changes.dropna(), bins=100, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Return (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

recent_df = df.tail(252)
axes[1, 1].plot(recent_df['Date'], recent_df['Close'], label='Close', linewidth=2)
axes[1, 1].plot(recent_df['Date'], recent_df['Open'], label='Open', linewidth=1, alpha=0.7)
axes[1, 1].set_title('GOOG Recent Price (Last Year)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nBasic Statistics:")
print(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
```

---

## Cell 4: Data Preprocessing

```python
df = parse_and_sort_dates(df, date_column='Date')
df = handle_missing_values(df, method='forward_fill')

price_cols = ['Open', 'High', 'Low', 'Close']
df = detect_and_handle_outliers(df, columns=price_cols, method='iqr', threshold=1.5)

print("Data preprocessing complete")
print(f"Cleaned dataset shape: {df.shape}")
print(f"Missing values after preprocessing: {df.isnull().sum().sum()}")
```

---

## Cell 5: Feature Engineering - Technical Indicators

```python
print("Calculating technical indicators...")

df = calculate_moving_averages(df, price_column='Close', windows=[5, 10, 20, 50])
df = calculate_bollinger_bands(df, price_column='Close', window=20, num_std=2.0)
df = calculate_atr(df, high_col='High', low_col='Low', close_col='Close', window=14)
df = calculate_rsi(df, price_column='Close', window=14)
df = calculate_macd(df, price_column='Close', fast_period=12, slow_period=26, signal_period=9)
df = calculate_volume_indicators(df, volume_column='Volume', price_column='Close')

# Calculate percentage change (target variable)
# Model will output 0.1 for 10% change, -0.05 for -5% change, etc.
df['Price_Change_Pct'] = df['Close'].pct_change()

df = df.dropna()

print(f"Technical indicators calculated")
print(f"Dataset shape after feature engineering: {df.shape}")
print(f"\nAvailable features:")
print(df.columns.tolist())
print(f"\nTarget variable statistics (Price_Change_Pct):")
print(f"  Mean: {df['Price_Change_Pct'].mean():.6f} ({df['Price_Change_Pct'].mean()*100:.4f}%)")
print(f"  Std:  {df['Price_Change_Pct'].std():.6f} ({df['Price_Change_Pct'].std()*100:.4f}%)")
print(f"  Min:  {df['Price_Change_Pct'].min():.6f} ({df['Price_Change_Pct'].min()*100:.4f}%)")
print(f"  Max:  {df['Price_Change_Pct'].max():.6f} ({df['Price_Change_Pct'].max()*100:.4f}%)")
```

---

## Cell 6: Visualize Technical Indicators

```python
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

recent_df = df.tail(252)
ax1 = axes[0]
ax1.plot(recent_df.index, recent_df['Close'], label='Close Price', linewidth=2, color='black')
ax1.plot(recent_df.index, recent_df['SMA_20'], label='SMA 20', linewidth=1.5, alpha=0.7)
ax1.plot(recent_df.index, recent_df['EMA_20'], label='EMA 20', linewidth=1.5, alpha=0.7)
ax1.plot(recent_df.index, recent_df['BB_Upper'], label='BB Upper', linewidth=1, linestyle='--', alpha=0.6)
ax1.plot(recent_df.index, recent_df['BB_Lower'], label='BB Lower', linewidth=1, linestyle='--', alpha=0.6)
ax1.fill_between(recent_df.index, recent_df['BB_Upper'], recent_df['BB_Lower'], alpha=0.1)
ax1.set_title('Price and Moving Averages', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price ($)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(recent_df.index, recent_df['RSI'], label='RSI', linewidth=2, color='purple')
ax2.axhline(y=70, color='r', linestyle='--', linewidth=1, label='Overbought')
ax2.axhline(y=30, color='g', linestyle='--', linewidth=1, label='Oversold')
ax2.fill_between(recent_df.index, 30, 70, alpha=0.1, color='gray')
ax2.set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
ax2.set_ylabel('RSI')
ax2.set_ylim(0, 100)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

ax3 = axes[2]
ax3.plot(recent_df.index, recent_df['MACD'], label='MACD', linewidth=2)
ax3.plot(recent_df.index, recent_df['MACD_Signal'], label='Signal', linewidth=2)
ax3.bar(recent_df.index, recent_df['MACD_Histogram'], label='Histogram', alpha=0.5)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_title('MACD Indicator', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('MACD')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Technical indicators visualization complete")
```

---

## Cell 7: Select Features for Training

```python
feature_cols = [
    'Close', 'Open', 'High', 'Low', 'Volume',
    'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal',
    'BB_Width', 'ATR', 'Volume_Ratio'
]

target_col = 'Price_Change_Pct'

print(f"Selected features ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

print(f"\nTarget variable: {target_col}")
print(f"Note: Model predicts percentage change (e.g., 0.1 = 10% increase)")
```

---

## Cell 8: Split Data Chronologically

```python
train_df, val_df, test_df = split_data_chronological(df, train_ratio=0.7, val_ratio=0.15)

print("=" * 60)
print("Data Split Summary")
print("=" * 60)
print(f"Training set:   {len(train_df):5d} samples ({train_df.index.min()} to {train_df.index.max()})")
print(f"Validation set: {len(val_df):5d} samples ({val_df.index.min()} to {val_df.index.max()})")
print(f"Test set:       {len(test_df):5d} samples ({test_df.index.min()} to {test_df.index.max()})")
print("=" * 60)

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(train_df.index, train_df['Close'], label='Train', linewidth=1, alpha=0.8)
ax.plot(val_df.index, val_df['Close'], label='Validation', linewidth=1, alpha=0.8)
ax.plot(test_df.index, test_df['Close'], label='Test', linewidth=1, alpha=0.8)
ax.set_title('Train/Validation/Test Split', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Cell 9: Normalize Features

```python
# Include target column in normalization
all_cols = feature_cols + [target_col]

train_scaled, scaler = normalize_data(train_df, columns=all_cols, scaler_type='minmax')

val_scaled = val_df.copy()
val_scaled[all_cols] = scaler.transform(val_df[all_cols])

test_scaled = test_df.copy()
test_scaled[all_cols] = scaler.transform(test_df[all_cols])

print("Feature normalization complete")
print(f"\nScaler range: {scaler.feature_range}")
print(f"\nSample scaled values (first 5 rows, first 5 features):")
print(train_scaled[feature_cols].head())
print(f"\nTarget variable range in training set:")
print(f"  Original: [{train_df[target_col].min():.6f}, {train_df[target_col].max():.6f}]")
print(f"  Scaled:   [{train_scaled[target_col].min():.6f}, {train_scaled[target_col].max():.6f}]")
```

---

## Cell 10: Create Sequences for LSTM

```python
sequence_length = 60

# Create sequences with features only (X) and target separately (y)
X_train, _ = create_rolling_windows(
    train_scaled[feature_cols].values,
    window_size=sequence_length,
    step_size=1
)

X_val, _ = create_rolling_windows(
    val_scaled[feature_cols].values,
    window_size=sequence_length,
    step_size=1
)

X_test, _ = create_rolling_windows(
    test_scaled[feature_cols].values,
    window_size=sequence_length,
    step_size=1
)

# Extract target (percentage change) separately
# Target is the percentage change at the next time step
y_train = train_scaled[target_col].values[sequence_length:]
y_val = val_scaled[target_col].values[sequence_length:]
y_test = test_scaled[target_col].values[sequence_length:]

# Reshape to (n_samples, 1) for single output
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print("=" * 60)
print("Sequence Creation Summary")
print("=" * 60)
print(f"Sequence length: {sequence_length} days")
print(f"Number of features: {len(feature_cols)}")
print(f"Target: {target_col} (percentage change)")
print(f"\nTraining sequences:   X={X_train.shape}, y={y_train.shape}")
print(f"Validation sequences: X={X_val.shape}, y={y_val.shape}")
print(f"Test sequences:       X={X_test.shape}, y={y_test.shape}")
print("=" * 60)
```

---

## Cell 11: Build and Train LSTM Model

```python
config = LSTMConfig(
    sequence_length=sequence_length,
    n_features=len(feature_cols),
    lstm_units_1=512,
    lstm_units_2=256,
    dropout_rate=0.2,
    dense_units=128,
    output_dim=1,  # Single output: percentage change prediction
    learning_rate=0.001,
    batch_size=32,
    epochs=50,
    validation_split=0.0
)

print("Building and training LSTM model...")
print("=" * 60)
print("Loss function: MAE (Mean Absolute Error)")
print("Optimizer: Adam")
print("=" * 60)

import os
os.makedirs('models', exist_ok=True)

model, history, file_paths = create_and_train_lstm(
    X_train, y_train,
    X_val, y_val,
    config=config,
    model_dir='models',
    verbose=1
)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
```

---

## Cell 13: Convert Model to ONNX

```python
print("Converting Keras model to ONNX format...\n")

onnx_model_path = 'models/stock_forecast_lstm.onnx'

onnx_path = convert_to_onnx(
    model_path='models/lstm_model.keras',
    onnx_path=onnx_model_path,
)

print(f"Model converted successfully!")
print(f"ONNX model saved to: {onnx_path}")

import os
keras_size = os.path.getsize('models/lstm_model.keras') / 1024
onnx_size = os.path.getsize(onnx_path) / 1024

print(f"\nModel Size Comparison:")
print(f"  Keras (.keras):  {keras_size:.2f} KB")
print(f"  ONNX (.onnx): {onnx_size:.2f} KB")
print(f"  Compression:  {keras_size/onnx_size:.2f}x")
```

---

## Cell 14: Verify ONNX Model

```python
verification = verify_onnx(onnx_model_path)

print("=" * 60)
print("ONNX Model Verification")
print("=" * 60)
print(f"Valid:              {verification['is_valid']}")
print(f"Error:              {verification['error']}")
print(f"Opset Version:      {verification['opset_version']}")
print(f"Number of Nodes:    {verification['num_nodes']}")
print("=" * 60)

if verification['is_valid']:
    print("\nONNX model is valid and ready for deployment!")
else:
    print(f"\nVerification failed: {verification['error']}")
    raise Exception
```

---

## Cell 15: Compare TensorFlow vs ONNX Inference

```python
print("Comparing TensorFlow vs ONNX Runtime performance...\n")

comparison = compare_frameworks_inference(
    keras_model_path='models/lstm_model.keras',
    onnx_model_path=onnx_model_path,
    test_input=X_test[:100]
)

print("=" * 70)
print("Framework Comparison Results")
print("=" * 70)
print(f"TensorFlow inference time:  {comparison['tensorflow_time']:.6f} seconds")
print(f"ONNX Runtime inference time: {comparison['onnx_time']:.6f} seconds")
print(f"Speedup:                     {comparison['speedup']:.2f}x")
print("-" * 70)
print(f"Max numerical difference:    {comparison['max_difference']:.2e}")
print(f"Mean numerical difference:   {comparison['mean_difference']:.2e}")
print(f"Numerically equivalent:      {comparison['numerically_close']}")
print("=" * 70)

if comparison['speedup'] > 1:
    print(f"\nONNX Runtime is {comparison['speedup']:.2f}x faster!")
else:
    print(f"\nTensorFlow is {1/comparison['speedup']:.2f}x faster")

if comparison['numerically_close']:
    print("Models produce numerically equivalent results")
```

---

## Cell 16: Run Inference with ONNX Runtime

```python
print("Running inference with ONNX Runtime...\n")

onnx_session = ONNXInferenceSession(onnx_model_path)

print(f"Input shape: {onnx_session.get_input_shape()}")
print(f"Output shape: {onnx_session.get_output_shape()}")

# Get predictions (scaled percentage changes)
onnx_predictions_scaled = onnx_session.predict(X_test)

# Denormalize percentage change predictions
# We need to create a dummy array with all features to use the scaler
target_col_idx = all_cols.index(target_col)
dummy_array = np.zeros((len(onnx_predictions_scaled), len(all_cols)))
dummy_array[:, target_col_idx] = onnx_predictions_scaled.flatten()
onnx_predictions_pct = scaler.inverse_transform(dummy_array)[:, target_col_idx]

# Denormalize actual percentage changes
dummy_array_actual = np.zeros((len(y_test), len(all_cols)))
dummy_array_actual[:, target_col_idx] = y_test.flatten()
y_test_pct = scaler.inverse_transform(dummy_array_actual)[:, target_col_idx]

# Convert percentage changes to actual prices for visualization
# We need the previous close price for each prediction
test_close_prices = test_df['Close'].values[sequence_length-1:sequence_length-1+len(y_test_pct)]

# Predicted price = Previous close * (1 + predicted percentage change)
onnx_predictions_unscaled = test_close_prices * (1 + onnx_predictions_pct)

# Actual price = Previous close * (1 + actual percentage change)
y_test_unscaled = test_close_prices * (1 + y_test_pct)

print(f"\nInference complete!")
print(f"Number of predictions: {len(onnx_predictions_scaled)}")
print(f"\nSample predictions (percentage change):")
for i in range(min(5, len(onnx_predictions_pct))):
    print(f"  Predicted: {onnx_predictions_pct[i]:+.4f} ({onnx_predictions_pct[i]*100:+.2f}%), " +
          f"Actual: {y_test_pct[i]:+.4f} ({y_test_pct[i]*100:+.2f}%)")
```

---

## Cell 17: Evaluate Forecast Performance

```python
# Evaluate on percentage change predictions (primary metric)
pct_mae = np.mean(np.abs(y_test_pct - onnx_predictions_pct))
pct_rmse = np.sqrt(np.mean((y_test_pct - onnx_predictions_pct) ** 2))
pct_r2 = 1 - (np.sum((y_test_pct - onnx_predictions_pct) ** 2) / np.sum((y_test_pct - y_test_pct.mean()) ** 2))

# Directional accuracy (did we predict the right direction?)
correct_direction = np.sum((y_test_pct > 0) == (onnx_predictions_pct > 0))
directional_accuracy_pct = (correct_direction / len(y_test_pct)) * 100

print("=" * 70)
print("Percentage Change Prediction Performance (Primary Metric)")
print("=" * 70)
print(f"Mean Absolute Error (MAE):       {pct_mae:.6f} ({pct_mae*100:.4f}%)")
print(f"Root Mean Squared Error (RMSE):  {pct_rmse:.6f} ({pct_rmse*100:.4f}%)")
print(f"R² Score:                        {pct_r2:.4f}")
print(f"Directional Accuracy:            {directional_accuracy_pct:.2f}%")
print("=" * 70)

# Also evaluate on reconstructed prices (secondary metric for comparison)
metrics = evaluate_forecasts(y_test_unscaled, onnx_predictions_unscaled)

print("\n" + "=" * 70)
print("Reconstructed Price Forecasting Performance (Secondary Metric)")
print("=" * 70)
print(f"Mean Absolute Error (MAE):       ${metrics['MAE']:.4f}")
print(f"Root Mean Squared Error (RMSE):  ${metrics['RMSE']:.4f}")
print(f"Mean Absolute % Error (MAPE):    {metrics['MAPE']:.2f}%")
print(f"R² Score:                        {metrics['R2']:.4f}")
print(f"Directional Accuracy:            {metrics['Directional_Accuracy']:.2f}%")
print("=" * 70)

avg_price = y_test_unscaled.mean()
print(f"\nAverage stock price: ${avg_price:.2f}")
print(f"MAE as % of avg price: {(metrics['MAE'] / avg_price) * 100:.2f}%")

if directional_accuracy_pct > 55:
    print(f"\nGood directional accuracy: {directional_accuracy_pct:.1f}% > 55%")
else:
    print(f"\nNote: Directional accuracy {directional_accuracy_pct:.1f}% (random = 50%)")
```

---

## Cell 18: Visualize Predictions vs Actual

```python
test_dates = test_df.index[sequence_length:sequence_length + len(y_test_unscaled)]

plot_predictions_vs_actual(
    y_true=y_test_unscaled,
    y_pred=onnx_predictions_unscaled,
    dates=test_dates,
    title='GOOG Stock Price Forecasting: Predictions vs Actual (ONNX)',
    figsize=(18, 7)
)

recent_window = 252
if len(y_test_unscaled) > recent_window:
    plot_predictions_vs_actual(
        y_true=y_test_unscaled[-recent_window:],
        y_pred=onnx_predictions_unscaled[-recent_window:],
        dates=test_dates[-recent_window:],
        title='GOOG Stock Price Forecasting: Last Year (ONNX)',
        figsize=(18, 7)
    )
```

---

## Cell 20: Sample Predictions Table

```python
sample_size = 20
sample_indices = np.linspace(0, len(y_test_unscaled) - 1, sample_size, dtype=int)

results_df = pd.DataFrame({
    'Date': test_dates[sample_indices],
    'Actual % Chg': y_test_pct[sample_indices] * 100,
    'Pred % Chg': onnx_predictions_pct[sample_indices] * 100,
    'Actual Price': y_test_unscaled[sample_indices],
    'Pred Price': onnx_predictions_unscaled[sample_indices],
    'Price Error': y_test_unscaled[sample_indices] - onnx_predictions_unscaled[sample_indices]
})

# Format for display
results_df['Actual % Chg'] = results_df['Actual % Chg'].apply(lambda x: f"{x:+.2f}%")
results_df['Pred % Chg'] = results_df['Pred % Chg'].apply(lambda x: f"{x:+.2f}%")
results_df['Actual Price'] = results_df['Actual Price'].apply(lambda x: f"${x:.2f}")
results_df['Pred Price'] = results_df['Pred Price'].apply(lambda x: f"${x:.2f}")
results_df['Price Error'] = results_df['Price Error'].apply(lambda x: f"${x:.2f}")

print("Sample Predictions (Percentage Change & Reconstructed Prices):")
print(results_df.to_string(index=False))
```

---

## Cell 21: Feature Importance Analysis

```python
print("Analyzing which features most impact percentage change predictions...\n")

feature_importance = []
baseline_pred = onnx_session.predict(X_test[:100])

for i, feature in enumerate(feature_cols):
    X_test_perturbed = X_test[:100].copy()
    X_test_perturbed[:, :, i] = np.random.permutation(X_test_perturbed[:, :, i])
    perturbed_pred = onnx_session.predict(X_test_perturbed)

    importance = np.mean(np.abs(baseline_pred - perturbed_pred))
    feature_importance.append((feature, importance))

feature_importance.sort(key=lambda x: x[1], reverse=True)

print("Feature Importance for Percentage Change Prediction:")
print("-" * 60)
for rank, (feature, importance) in enumerate(feature_importance, 1):
    print(f"{rank:2d}. {feature:20s}: {importance:.6f}")

fig, ax = plt.subplots(figsize=(12, 6))
features, importances = zip(*feature_importance)
ax.barh(features, importances, alpha=0.7, edgecolor='black')
ax.set_xlabel('Importance Score (Impact on % Change Prediction)')
ax.set_title('Feature Importance for Percentage Change Forecasting', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

---

## Cell 22: Forecast Report

```python
report_df = create_forecast_report(
    y_true=y_test_unscaled,
    y_pred=onnx_predictions_unscaled,
    model_name='LSTM + ONNX'
)

print("\n" + "=" * 60)
print("Forecast Performance Report")
print("=" * 60)
print(report_df.to_string(index=False))
print("=" * 60)
```
