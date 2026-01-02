import numpy as np
import pandas as pd
import warnings
import os
import pickle
import gc

warnings.filterwarnings('ignore')

# CUDA fixes for WSL
wsl_cuda_path = '/usr/lib/wsl/lib'
if os.path.exists(wsl_cuda_path):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if wsl_cuda_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{wsl_cuda_path}:{current_ld_path}"
        
import torch
from darts import TimeSeries

from onnx_forecasting_utils import (
    TCNConfig, train_tcn_model,
    convert_tcn_to_onnx, verify_onnx, ONNXInferenceSession,
    evaluate_forecasts
)

# Configure PyTorch for Tensor Cores
torch.set_float32_matmul_precision('medium')
print("PyTorch configured for Tensor Core usage with medium precision")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available in PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

print("\n" + "="*60)
print("Loading preprocessed data...")
print("="*60)

with open('ensemble/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
train_normalized = data['train_normalized']
val_normalized = data['val_normalized']
test_normalized = data['test_normalized']
test_stock_labels = data['test_stock_labels']
stocks = data['stocks']
SEQUENCE_LENGTH = data['SEQUENCE_LENGTH']

print(f"Data loaded successfully!")
print(f"  train_normalized: {train_normalized.shape}")
print(f"  val_normalized: {val_normalized.shape}")
print(f"  test_normalized: {test_normalized.shape}")


print("\n" + "="*60)
print("Preparing DARTS TimeSeries...")
print("="*60)

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU memory cleared before TCN training")

train_target = train_normalized[:, 0]  # Close price (first feature)
train_covariates = train_normalized  # All 13 features

val_target = val_normalized[:, 0]
val_covariates = val_normalized

test_target = test_normalized[:, 0]
test_covariates = test_normalized

print(f"\nTCN training data prepared:")
print(f"  Train target: {train_target.shape}")
print(f"  Train covariates: {train_covariates.shape}")
print(f"  Val target: {val_target.shape}")
print(f"  Val covariates: {val_covariates.shape}")

# Create TimeSeries objects (using integer index for multi-stock data)
train_target_ts = TimeSeries.from_values(train_target)
train_cov_ts = TimeSeries.from_values(train_covariates)

val_target_ts = TimeSeries.from_values(val_target)
val_cov_ts = TimeSeries.from_values(val_covariates)

print("TimeSeries objects created successfully")

tcn_config = TCNConfig(
    input_chunk_length=15,
    output_chunk_length=1,
    kernel_size=3,
    num_filters=64,
    dropout=0.2,
    n_epochs=30,
    dilation_base=2,
    weight_norm=False,
    batch_size=32,
    model_name='tcn_mag7'
)

print("\n" + "="*60)
print("TCN Configuration:")
print("="*60)
print(f"  Input Chunk Length: {tcn_config.input_chunk_length}")
print(f"  Output Chunk Length: {tcn_config.output_chunk_length}")
print(f"  Kernel Size: {tcn_config.kernel_size}")
print(f"  Num Filters: {tcn_config.num_filters}")
print(f"  Dropout: {tcn_config.dropout}")
print(f"  Epochs: {tcn_config.n_epochs}")

print("\n" + "="*60)
print("Training TCN model on MAG 7 stacked data...")
print("="*60)

tcn_model = train_tcn_model(
    target_series=train_target_ts,
    past_covariates=train_cov_ts,
    val_series=val_target_ts,
    val_covariates=val_cov_ts,
    config=tcn_config,
    model_dir='models',
    verbose=True
)

print("\nTCN training completed!")
tcn_onnx_path = 'models/tcn_mag7.onnx'

print("\n" + "="*60)
print("Converting TCN model to ONNX...")
print("="*60)

tcn_onnx_result = convert_tcn_to_onnx(
    tcn_model,
    input_shape=(15, 13),  # (sequence_length, n_features)
    output_path=tcn_onnx_path
)

if tcn_onnx_result:
    print("\nTCN successfully converted to ONNX!")

    # Verify ONNX model
    verification = verify_onnx(tcn_onnx_path)
    print(f"\nONNX Verification:")
    print(f"  Valid: {verification['is_valid']}")
    print(f"  Opset Version: {verification['opset_version']}")
    print(f"  Number of Nodes: {verification['num_nodes']}")
else:
    print("\nTCN ONNX conversion failed. Will use native DARTS inference.")


print("\n" + "="*60)
print("Making predictions with DARTS TCN...")
print("="*60)

# Create test TimeSeries
test_cov_ts = TimeSeries.from_values(test_covariates)

# Use historical_forecasts for efficient batched prediction
test_target_ts = TimeSeries.from_values(test_target)

tcn_darts_predictions = tcn_model.historical_forecasts(
    series=test_target_ts,
    past_covariates=test_cov_ts,
    forecast_horizon=1,
    stride=1,
    retrain=False,
    verbose=False,
)

tcn_darts_pred = tcn_darts_predictions.values()
y_test_tcn = y_test[:len(tcn_darts_pred)]
test_stock_labels_tcn = test_stock_labels[:len(tcn_darts_pred)]

print(f"\nTCN DARTS Predictions shape: {tcn_darts_pred.shape}")

if tcn_onnx_result:
    print("Making predictions with TCN ONNX...")
    # Note: TCN ONNX model uses wrapper that handles transpose internally
    tcn_onnx_session = ONNXInferenceSession(tcn_onnx_path)

    X_test_tcn = X_test[:len(tcn_darts_pred)]
    tcn_onnx_pred = tcn_onnx_session.predict(X_test_tcn)

    print(f"TCN ONNX Predictions shape: {tcn_onnx_pred.shape}")

    # Compare
    max_diff = np.max(np.abs(tcn_darts_pred - tcn_onnx_pred))
    print(f"\nMax difference between DARTS and ONNX: {max_diff:.6f}")
else:
    tcn_onnx_pred = tcn_darts_pred
    print("Using DARTS predictions (ONNX conversion not available)")

tcn_metrics_overall = evaluate_forecasts(y_test_tcn, tcn_onnx_pred)

print("\n" + "="*60)
print("TCN Overall Performance:")
print("="*60)
print(f"  MAE: {tcn_metrics_overall['MAE']:.4f}")
print(f"  RMSE: {tcn_metrics_overall['RMSE']:.4f}")
print(f"  MAPE: {tcn_metrics_overall['MAPE']:.2f}%")
print(f"  R²: {tcn_metrics_overall['R2']:.4f}")
print(f"  Directional Accuracy: {tcn_metrics_overall['Directional_Accuracy']:.2f}%")

print("\n" + "="*60)
print("Per-Stock TCN Performance:")
print("="*60)

tcn_per_stock = []

for stock in stocks:
    mask = test_stock_labels_tcn == stock
    if mask.sum() > 0:
        metrics = evaluate_forecasts(y_test_tcn[mask], tcn_onnx_pred[mask])
        metrics['Stock'] = stock
        tcn_per_stock.append(metrics)
        print(f"\n{stock}:")
        print(f"  MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")

tcn_per_stock_df = pd.DataFrame(tcn_per_stock)
print("\n")
print(tcn_per_stock_df.to_string(index=False))

tcn_per_stock_df.to_csv('models/tcn_per_stock_metrics.csv', index=False)

predictions_data = {
    'tcn_predictions': tcn_onnx_pred,
    'y_test_tcn': y_test_tcn,
    'test_stock_labels_tcn': test_stock_labels_tcn,
    'stocks': stocks,
    'tcn_metrics': tcn_metrics_overall,
    'tcn_per_stock': tcn_per_stock_df,
    'tcn_onnx_result': tcn_onnx_result
}

with open('ensemble/tcn_predictions.pkl', 'wb') as f:
    pickle.dump(predictions_data, f)

print("\n" + "="*60)
print("TCN TRAINING COMPLETE")
print("="*60)
print("Predictions saved to: ensemble/tcn_predictions.pkl")
if tcn_onnx_result:
    print("ONNX model saved to: models/tcn_mag7.onnx")
