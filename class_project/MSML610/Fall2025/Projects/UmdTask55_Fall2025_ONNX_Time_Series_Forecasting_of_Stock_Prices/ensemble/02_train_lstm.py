import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import pickle
import time

warnings.filterwarnings('ignore')

# CUDA fixes for WSL
wsl_cuda_path = '/usr/lib/wsl/lib'
if os.path.exists(wsl_cuda_path):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if wsl_cuda_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{wsl_cuda_path}:{current_ld_path}"

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras

from onnx_forecasting_utils import (
    LSTMConfig, build_lstm_model, compile_model, train_lstm_model,
    convert_to_onnx, verify_onnx, ONNXInferenceSession,
    compare_frameworks_inference, evaluate_forecasts
)

from model import plot_training_history

# Configure TensorFlow GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow configured for GPU memory growth on {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth configuration error: {e}")

print(f"TensorFlow version: {tf.__version__}")


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
test_stock_labels = data['test_stock_labels']
stocks = data['stocks']

print(f"Data loaded successfully!")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

lstm_config = LSTMConfig(
    sequence_length=15,
    n_features=13,
    lstm_units_1=512,
    lstm_units_2=256,
    dropout_rate=0.2,
    dense_units=128,
    output_dim=1,
    learning_rate=0.001,
    batch_size=32,
    epochs=50,
    validation_split=0.0
)

print("\n" + "="*60)
print("LSTM Configuration:")
print("="*60)
print(f"  Sequence Length: {lstm_config.sequence_length}")
print(f"  Input Features: {lstm_config.n_features}")
print(f"  LSTM Units: {lstm_config.lstm_units_1}, {lstm_config.lstm_units_2}")
print(f"  Dense Units: {lstm_config.dense_units}")
print(f"  Output Dimension: {lstm_config.output_dim}")
print(f"  Learning Rate: {lstm_config.learning_rate}")
print(f"  Batch Size: {lstm_config.batch_size}")
print(f"  Epochs: {lstm_config.epochs}")

lstm_model = build_lstm_model(lstm_config)
lstm_model = compile_model(lstm_model, learning_rate=lstm_config.learning_rate)

print("\nLSTM Model Architecture:")
lstm_model.summary()

print("\n" + "="*60)
print("Training LSTM model on MAG 7 stacked data...")
print("="*60)

lstm_model, lstm_history = train_lstm_model(
    lstm_model,
    X_train, y_train,
    X_val, y_val,
    config=lstm_config,
    model_path='models/lstm_mag7_best.keras',
    verbose=1
)

print("\nLSTM training completed!")

plot_training_history(lstm_history, save_path='models/lstm_mag7_training.png')

lstm_onnx_path = 'models/lstm_mag7.onnx'

print("\n" + "="*60)
print("Converting LSTM model to ONNX...")
print("="*60)

onnx_path = convert_to_onnx('models/lstm_mag7_best.keras', lstm_onnx_path)

if onnx_path:
    verification = verify_onnx(lstm_onnx_path)
    print(f"\nONNX Verification:")
    print(f"  Valid: {verification['is_valid']}")
    print(f"  Opset Version: {verification['opset_version']}")
    print(f"  Number of Nodes: {verification['num_nodes']}")
    if verification['error']:
        print(f"  Error: {verification['error']}")


print("\n" + "="*60)
print("Making predictions...")
print("="*60)

# TensorFlow predictions
print("Making predictions with TensorFlow LSTM...")
lstm_tf_pred = lstm_model.predict(X_test, verbose=0)

# ONNX Runtime predictions
print("Making predictions with ONNX Runtime...")
lstm_onnx_session = ONNXInferenceSession(lstm_onnx_path)
lstm_onnx_pred = lstm_onnx_session.predict(X_test)

print(f"\nPrediction shapes:")
print(f"  TensorFlow: {lstm_tf_pred.shape}")
print(f"  ONNX: {lstm_onnx_pred.shape}")

# Check numerical agreement
max_diff = np.max(np.abs(lstm_tf_pred - lstm_onnx_pred))
mean_diff = np.mean(np.abs(lstm_tf_pred - lstm_onnx_pred))

print(f"\nNumerical Comparison:")
print(f"  Max Difference: {max_diff:.6f}")
print(f"  Mean Difference: {mean_diff:.6f}")
print(f"  Predictions Match: {np.allclose(lstm_tf_pred, lstm_onnx_pred, rtol=1e-4)}")

print("\n" + "="*60)
print("LSTM Framework Speed Comparison")
print("="*60)

comparison = compare_frameworks_inference(
    'models/lstm_mag7_best.keras',
    lstm_onnx_path,
    X_test
)

print(f"  TensorFlow Time: {comparison['tensorflow_time']:.4f}s")
print(f"  ONNX Runtime Time: {comparison['onnx_time']:.4f}s")
print(f"  Speedup: {comparison['speedup']:.2f}x")
print(f"  Max Difference: {comparison['max_difference']:.6f}")
print(f"  Mean Difference: {comparison['mean_difference']:.6f}")
print(f"  Numerically Close: {comparison['numerically_close']}")

lstm_metrics_overall = evaluate_forecasts(y_test, lstm_onnx_pred)

print("\n" + "="*60)
print("LSTM Overall Performance:")
print("="*60)
print(f"  MAE: {lstm_metrics_overall['MAE']:.4f}")
print(f"  RMSE: {lstm_metrics_overall['RMSE']:.4f}")
print(f"  MAPE: {lstm_metrics_overall['MAPE']:.2f}%")
print(f"  R²: {lstm_metrics_overall['R2']:.4f}")
print(f"  Directional Accuracy: {lstm_metrics_overall['Directional_Accuracy']:.2f}%")

print("\n" + "="*60)
print("Per-Stock LSTM Performance:")
print("="*60)

lstm_per_stock = []

for stock in stocks:
    mask = test_stock_labels == stock
    metrics = evaluate_forecasts(y_test[mask], lstm_onnx_pred[mask])
    metrics['Stock'] = stock
    lstm_per_stock.append(metrics)
    print(f"\n{stock}:")
    print(f"  MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")

lstm_per_stock_df = pd.DataFrame(lstm_per_stock)
print("\n")
print(lstm_per_stock_df.to_string(index=False))

lstm_per_stock_df.to_csv('models/lstm_per_stock_metrics.csv', index=False)

predictions_data = {
    'lstm_predictions': lstm_onnx_pred,
    'y_test': y_test,
    'test_stock_labels': test_stock_labels,
    'stocks': stocks,
    'lstm_metrics': lstm_metrics_overall,
    'lstm_per_stock': lstm_per_stock_df
}

with open('ensemble/lstm_predictions.pkl', 'wb') as f:
    pickle.dump(predictions_data, f)

print("\n" + "="*60)
print("LSTM TRAINING COMPLETE")
print("="*60)
print("Predictions saved to: ensemble/lstm_predictions.pkl")
print("Model saved to: models/lstm_mag7_best.keras")
print("ONNX model saved to: models/lstm_mag7.onnx")
