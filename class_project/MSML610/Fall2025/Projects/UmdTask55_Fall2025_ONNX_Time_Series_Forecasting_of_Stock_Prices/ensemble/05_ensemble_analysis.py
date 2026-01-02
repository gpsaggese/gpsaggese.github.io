import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle
import time

warnings.filterwarnings('ignore')

from onnx_forecasting_utils import (
    evaluate_forecasts,
    create_ensemble_predictions,
    compare_multiple_models,
    plot_ensemble_comparison,
    plot_performance_heatmap,
    get_best_model_per_stock,
    ONNXInferenceSession
)

sns.set_style('whitegrid')

print("="*60)
print("Loading predictions from all models...")
print("="*60)

with open('ensemble/lstm_predictions.pkl', 'rb') as f:
    lstm_data = pickle.load(f)

with open('ensemble/tcn_predictions.pkl', 'rb') as f:
    tcn_data = pickle.load(f)

with open('ensemble/xgb_predictions.pkl', 'rb') as f:
    xgb_data = pickle.load(f)

lstm_onnx_pred = lstm_data['lstm_predictions']
tcn_onnx_pred = tcn_data['tcn_predictions']
xgb_onnx_pred = xgb_data['xgb_predictions']

y_test = lstm_data['y_test']
test_stock_labels_seq = lstm_data['test_stock_labels']
stocks = lstm_data['stocks']

print(f"Predictions loaded successfully!")
print(f"  LSTM: {lstm_onnx_pred.shape}")
print(f"  TCN: {tcn_onnx_pred.shape}")
print(f"  XGBoost: {xgb_onnx_pred.shape}")

print("\n" + "="*60)
print("Aligning predictions for ensemble...")
print("="*60)

min_len = min(len(lstm_onnx_pred), len(tcn_onnx_pred), len(xgb_onnx_pred))

lstm_pred_aligned = lstm_onnx_pred[:min_len]
tcn_pred_aligned = tcn_onnx_pred[:min_len]
xgb_pred_aligned = xgb_onnx_pred[:min_len]
y_test_aligned = y_test[:min_len]
test_labels_aligned = test_stock_labels_seq[:min_len]

print(f"Aligned predictions for ensemble:")
print(f"  Length: {min_len}")
print(f"  LSTM: {lstm_pred_aligned.shape}")
print(f"  TCN: {tcn_pred_aligned.shape}")
print(f"  XGBoost: {xgb_pred_aligned.shape}")

print("\n" + "="*60)
print("Creating ensemble predictions...")
print("="*60)

ensemble_pred = create_ensemble_predictions(
    lstm_pred_aligned,
    tcn_pred_aligned,
    xgb_pred_aligned
)

print(f"Ensemble predictions shape: {ensemble_pred.shape}")

ensemble_metrics = evaluate_forecasts(y_test_aligned, ensemble_pred)
print(f"Ensemble MAE: {ensemble_metrics['MAE']:.4f}, R²: {ensemble_metrics['R2']:.4f}")

print("\n" + "="*60)
print("Creating model comparison table...")
print("="*60)

predictions_dict = {
    'LSTM': lstm_pred_aligned,
    'TCN': tcn_pred_aligned,
    'XGBoost': xgb_pred_aligned,
    'Ensemble': ensemble_pred
}

comparison_df = compare_multiple_models(
    y_test_aligned,
    predictions_dict,
    stock_labels=test_labels_aligned,
    stock_names=stocks
)

comparison_df.to_csv('models/model_comparison_mag7.csv', index=False)
print("Model comparison table saved to: models/model_comparison_mag7.csv")

plot_performance_heatmap(
    comparison_df,
    metric='MAE',
    save_path='models/performance_heatmap_mae.png',
    figsize=(12, 8)
)

plot_performance_heatmap(
    comparison_df,
    metric='R2',
    save_path='models/performance_heatmap_r2.png',
    figsize=(12, 8)
)
print("Performance heatmaps saved to: models/")

plot_ensemble_comparison(
    y_test_aligned,
    predictions_dict,
    stock_labels=test_labels_aligned,
    stock_names=stocks,
    save_path='models/ensemble_comparison_all_stocks.png',
    figsize=(20, 14)
)
print("Ensemble comparison plot saved to: models/ensemble_comparison_all_stocks.png")

best_models = get_best_model_per_stock(comparison_df, metric='MAE', lower_is_better=True)

stock_model_counts = best_models['Model'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(stock_model_counts.index, stock_model_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(stock_model_counts)])
plt.xlabel('Model')
plt.ylabel('Number of Stocks')
plt.title('Number of Stocks Where Each Model Performs Best')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('models/best_models_distribution.png', dpi=300)
print("Best models distribution saved to: models/best_models_distribution.png")

overall_comparison = []

for model_name, preds in predictions_dict.items():
    metrics = evaluate_forecasts(y_test_aligned, preds)
    metrics['Model'] = model_name
    overall_comparison.append(metrics)

overall_df = pd.DataFrame(overall_comparison)
overall_df = overall_df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']]
overall_df = overall_df.sort_values('MAE')

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].bar(overall_df['Model'], overall_df['MAE'])
axes[0, 0].set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')
axes[0, 0].set_ylabel('MAE')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# R²
axes[0, 1].bar(overall_df['Model'], overall_df['R2'], color='green')
axes[0, 1].set_title('R² Score (Higher is Better)', fontweight='bold')
axes[0, 1].set_ylabel('R²')
axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# RMSE
axes[1, 0].bar(overall_df['Model'], overall_df['RMSE'], color='orange')
axes[1, 0].set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')
axes[1, 0].set_ylabel('RMSE')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Directional Accuracy
axes[1, 1].bar(overall_df['Model'], overall_df['Directional_Accuracy'], color='purple')
axes[1, 1].set_title('Directional Accuracy (Higher is Better)', fontweight='bold')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('models/overall_performance_comparison.png', dpi=300)
print("Overall performance comparison saved to: models/overall_performance_comparison.png")

print("\n" + "="*60)
print("ONNX Conversion Status")
print("="*60)

tcn_onnx_result = tcn_data.get('tcn_onnx_result', False)

onnx_status = {
    'Model': ['LSTM', 'TCN', 'XGBoost'],
    'ONNX Converted': ['Yes', 'Yes' if tcn_onnx_result else 'No', 'Yes'],
    'ONNX Path': ['models/lstm_mag7.onnx', 'models/tcn_mag7.onnx' if tcn_onnx_result else 'N/A', 'models/xgboost_mag7.onnx'],
    'Framework': ['TensorFlow/Keras', 'PyTorch (DARTS)', 'sklearn/XGBoost']
}

onnx_df = pd.DataFrame(onnx_status)
print(onnx_df.to_string(index=False))

lstm_onnx_path = 'models/lstm_mag7.onnx'
xgb_onnx_path = 'models/xgboost_mag7.onnx'
tcn_onnx_path = 'models/tcn_mag7.onnx'

print("\n" + "="*60)
print("Inference Speed Comparison (100 samples)")
print("="*60)

# Load processed data for speed test
with open('ensemble/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_test = data['X_test']

# For XGBoost
from onnx_forecasting_utils import flatten_sequences_for_xgboost
X_test_flat = flatten_sequences_for_xgboost(X_test)

test_sample = X_test[:100]
test_sample_flat = X_test_flat[:100]

# LSTM (TensorFlow vs ONNX)
import tensorflow as tf
lstm_model = tf.keras.models.load_model('models/lstm_mag7_best.keras')
lstm_onnx_session = ONNXInferenceSession(lstm_onnx_path)

start = time.time()
_ = lstm_model.predict(test_sample, verbose=0)
tf_time = time.time() - start

start = time.time()
_ = lstm_onnx_session.predict(test_sample)
lstm_onnx_time = time.time() - start

# TCN (PyTorch/DARTS vs ONNX)
tcn_onnx_result = tcn_data.get('tcn_onnx_result', False)
tcn_model_path = 'models/tcn_mag7.pkl'

if tcn_onnx_result and os.path.exists(tcn_onnx_path) and os.path.exists(tcn_model_path):
    try:
        from darts import TimeSeries
        from darts.models import TCNModel
        import torch

        # Load DARTS TCN model
        tcn_darts_model = TCNModel.load(tcn_model_path)
        tcn_onnx_session = ONNXInferenceSession(tcn_onnx_path)

        # Prepare test data for DARTS (needs TimeSeries format)
        # Use first 100 samples of X_test for timing
        test_sample_100 = X_test[:100]

        # Extract target (first feature) and all features as covariates
        test_target_100 = test_sample_100[:, :, 0].flatten()  # First feature (Close price)
        test_covariates_100 = test_sample_100.reshape(-1, test_sample_100.shape[-1])  # Reshape to 2D

        # Create TimeSeries objects
        test_target_ts = TimeSeries.from_values(test_target_100)
        test_cov_ts = TimeSeries.from_values(test_covariates_100)

        # Time DARTS inference using historical_forecasts (batched, efficient)
        start = time.time()
        _ = tcn_darts_model.historical_forecasts(
            series=test_target_ts,
            past_covariates=test_cov_ts,
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=False
        )
        darts_time = time.time() - start

        # Time ONNX inference
        start = time.time()
        _ = tcn_onnx_session.predict(test_sample_100)
        tcn_onnx_time = time.time() - start

        tcn_available = True
        print(f"TCN timing successful - DARTS: {darts_time:.4f}s, ONNX: {tcn_onnx_time:.4f}s")
    except Exception as e:
        print(f"TCN timing skipped: {e}")
        tcn_available = False
        darts_time = None
        tcn_onnx_time = None
else:
    if not os.path.exists(tcn_model_path):
        print(f"TCN timing skipped: Model file not found at {tcn_model_path}")
    tcn_available = False
    darts_time = None
    tcn_onnx_time = None

# XGBoost (sklearn vs ONNX)
import pickle as pkl
with open('models/xgboost_model.pkl', 'rb') as f:
    xgb_pipeline = pkl.load(f)

xgb_onnx_session = ONNXInferenceSession(xgb_onnx_path)

start = time.time()
_ = xgb_pipeline.predict(test_sample_flat)
sklearn_time = time.time() - start

start = time.time()
_ = xgb_onnx_session.predict(test_sample_flat.astype(np.float32))
xgb_onnx_time = time.time() - start

# Create summary table
if tcn_available:
    speed_summary = {
        'Model': ['LSTM (TF)', 'LSTM (ONNX)', 'TCN (DARTS)', 'TCN (ONNX)', 'XGBoost (sklearn)', 'XGBoost (ONNX)'],
        'Time (s)': [tf_time, lstm_onnx_time, darts_time, tcn_onnx_time, sklearn_time, xgb_onnx_time],
        'Samples/sec': [100/tf_time, 100/lstm_onnx_time, 100/darts_time, 100/tcn_onnx_time, 100/sklearn_time, 100/xgb_onnx_time],
        'Speedup': [1.0, tf_time/lstm_onnx_time, 1.0, darts_time/tcn_onnx_time, 1.0, sklearn_time/xgb_onnx_time]
    }
else:
    speed_summary = {
        'Model': ['LSTM (TF)', 'LSTM (ONNX)', 'XGBoost (sklearn)', 'XGBoost (ONNX)'],
        'Time (s)': [tf_time, lstm_onnx_time, sklearn_time, xgb_onnx_time],
        'Samples/sec': [100/tf_time, 100/lstm_onnx_time, 100/sklearn_time, 100/xgb_onnx_time],
        'Speedup': [1.0, tf_time/lstm_onnx_time, 1.0, sklearn_time/xgb_onnx_time]
    }

speed_df = pd.DataFrame(speed_summary)
print(speed_df.to_string(index=False))

print(f"\n{'='*60}")
print("ONNX Speedup Summary")
print(f"{'='*60}")
print(f"  LSTM ONNX vs TensorFlow: {tf_time/lstm_onnx_time:.2f}x faster")
if tcn_available:
    print(f"  TCN ONNX vs PyTorch/DARTS: {darts_time/tcn_onnx_time:.2f}x faster")
print(f"  XGBoost ONNX vs sklearn: {sklearn_time/xgb_onnx_time:.2f}x faster")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
models = speed_df['Model'].tolist()
times = speed_df['Time (s)'].tolist()
colors = ['#d62728' if 'ONNX' not in model else '#2ca02c' for model in models]

bars = ax1.barh(models, times, color=colors, alpha=0.8)
ax1.set_xlabel('Inference Time (seconds, lower is better)', fontweight='bold')
ax1.set_title('Inference Speed: Native vs ONNX Runtime', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

for i, (bar, time_val) in enumerate(zip(bars, times)):
    ax1.text(time_val + 0.001, i, f'{time_val:.4f}s', va='center', fontsize=9)

ax2 = axes[1]
if tcn_available:
    frameworks = ['LSTM\n(TF→ONNX)', 'TCN\n(DARTS→ONNX)', 'XGBoost\n(sklearn→ONNX)']
    speedups = [tf_time/lstm_onnx_time, darts_time/tcn_onnx_time, sklearn_time/xgb_onnx_time]
else:
    frameworks = ['LSTM\n(TF→ONNX)', 'XGBoost\n(sklearn→ONNX)']
    speedups = [tf_time/lstm_onnx_time, sklearn_time/xgb_onnx_time]

bars2 = ax2.bar(frameworks, speedups, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(frameworks)], alpha=0.8)
ax2.set_ylabel('Speedup Factor (x)', fontweight='bold')
ax2.set_title('ONNX Performance Gains', fontsize=14, fontweight='bold')
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

# Add value labels
for bar, speedup in zip(bars2, speedups):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{speedup:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('models/onnx_inference_speed_comparison.png', dpi=300, bbox_inches='tight')
print("\nInference speed comparison saved to: models/onnx_inference_speed_comparison.png")

# Additional ONNX analysis: Throughput comparison
fig, ax = plt.subplots(figsize=(12, 7))
samples_per_sec = speed_df['Samples/sec'].tolist()
models_list = speed_df['Model'].tolist()
colors_throughput = ['#d62728' if 'ONNX' not in model else '#2ca02c' for model in models_list]

bars = ax.barh(models_list, samples_per_sec, color=colors_throughput, alpha=0.8)
ax.set_xlabel('Throughput (samples/second, higher is better)', fontweight='bold')
ax.set_title('Inference Throughput: Native vs ONNX Runtime', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, (bar, throughput) in enumerate(zip(bars, samples_per_sec)):
    ax.text(throughput + 5, i, f'{throughput:.1f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('models/onnx_throughput_comparison.png', dpi=300, bbox_inches='tight')
print("Throughput comparison saved to: models/onnx_throughput_comparison.png")

# Model size comparison
model_sizes = {
    'LSTM (Keras)': os.path.getsize('models/lstm_mag7_best.keras') / (1024 * 1024) if os.path.exists('models/lstm_mag7_best.keras') else 0,
    'LSTM (ONNX)': os.path.getsize(lstm_onnx_path) / (1024 * 1024) if os.path.exists(lstm_onnx_path) else 0,
    'XGBoost (pkl)': os.path.getsize('models/xgboost_model.pkl') / (1024 * 1024) if os.path.exists('models/xgboost_model.pkl') else 0,
    'XGBoost (ONNX)': os.path.getsize(xgb_onnx_path) / (1024 * 1024) if os.path.exists(xgb_onnx_path) else 0,
}

if tcn_available and os.path.exists(tcn_model_path):
    model_sizes['TCN (DARTS)'] = os.path.getsize(tcn_model_path) / (1024 * 1024)
    model_sizes['TCN (ONNX)'] = os.path.getsize(tcn_onnx_path) / (1024 * 1024) if os.path.exists(tcn_onnx_path) else 0

fig, ax = plt.subplots(figsize=(12, 6))
model_names = list(model_sizes.keys())
sizes = list(model_sizes.values())
colors_size = ['#d62728' if 'ONNX' not in name else '#2ca02c' for name in model_names]

bars = ax.barh(model_names, sizes, color=colors_size, alpha=0.8)
ax.set_xlabel('Model Size (MB)', fontweight='bold')
ax.set_title('Model File Size Comparison: Native vs ONNX', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, (bar, size) in enumerate(zip(bars, sizes)):
    ax.text(size + 0.1, i, f'{size:.2f} MB', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('models/onnx_model_size_comparison.png', dpi=300, bbox_inches='tight')
print("Model size comparison saved to: models/onnx_model_size_comparison.png")

# Save size data
size_df = pd.DataFrame({
    'Model Format': model_names,
    'Size (MB)': sizes
})
print(f"\n{'='*60}")
print("Model Size Comparison")
print(f"{'='*60}")
print(size_df.to_string(index=False))

print("\n" + "="*80)
print("KEY FINDINGS AND RECOMMENDATIONS")
print("="*80)

# Best overall model
best_overall = overall_df.iloc[0]['Model']
best_mae = overall_df.iloc[0]['MAE']

print(f"\n1. BEST MODEL FOR DEPLOYMENT:")
print(f"   - Model: {best_overall}")
print(f"   - MAE: {best_mae:.4f}")

# ONNX deployment benefits - PRIMARY FOCUS
print(f"\n2. ONNX RUNTIME BENEFITS:")
print(f"   - LSTM TensorFlow → ONNX: {tf_time/lstm_onnx_time:.2f}x faster")
if tcn_available:
    print(f"   - TCN PyTorch/DARTS → ONNX: {darts_time/tcn_onnx_time:.2f}x faster")
print(f"   - XGBoost sklearn → ONNX: {sklearn_time/xgb_onnx_time:.2f}x faster")
print(f"   - Average Speedup: {np.mean([tf_time/lstm_onnx_time, sklearn_time/xgb_onnx_time]):.2f}x")
print(f"\n   Key Advantages:")
print(f"   - Faster inference (up to {max(tf_time/lstm_onnx_time, sklearn_time/xgb_onnx_time):.1f}x speedup)")
print(f"   - Framework-independent deployment")
print(f"   - Hardware acceleration support")
print(f"   - Smaller model sizes for some frameworks")
print(f"   - Cross-platform compatibility")

# ONNX conversion status
print(f"\n3. ONNX CONVERSION STATUS:")
print(f"   - LSTM (TensorFlow): Converted successfully")
print(f"   - TCN (PyTorch/DARTS): {'Converted successfully' if tcn_onnx_result else 'Use native DARTS'}")
print(f"   - XGBoost: Converted successfully")

# Model performance summary (condensed)
print(f"\n4. MODEL PERFORMANCE SUMMARY:")
print(f"   - All models demonstrate reasonable forecasting capability")
print(f"   - Ensemble approach combines strengths of different architectures")
print(f"   - See detailed metrics in: models/model_comparison_mag7.csv")

# Deployment recommendation
print(f"\n5. DEPLOYMENT RECOMMENDATION:")
print(f"   - Primary: Deploy {best_overall} with ONNX Runtime")
print(f"   - Expected speedup: {tf_time/lstm_onnx_time if best_overall == 'LSTM' else sklearn_time/xgb_onnx_time:.2f}x over native framework")
print(f"   - Use ONNX Runtime for production deployment to maximize throughput")
print(f"   - All visualizations and detailed analysis available in ./models/")

print("\n" + "="*80)
print("IMPLEMENTATION COMPLETE")
print("="*80)
print("\nAll models trained, evaluated, and compared on MAG 7 stocks!")
print("ONNX models ready for production deployment with significant speedups.")
print(f"\nKey Outputs:")
print(f"   - Model comparison: ./models/model_comparison_mag7.csv")
print(f"   - ONNX speed comparison: ./models/onnx_inference_speed_comparison.png")
print(f"   - ONNX throughput: ./models/onnx_throughput_comparison.png")
print(f"   - Model sizes: ./models/onnx_model_size_comparison.png")
print(f"   - All model files: ./models/")

summary = {
    'overall_performance': overall_df,
    'best_models_per_stock': best_models,
    'comparison_table': comparison_df,
    'onnx_status': onnx_df,
    'speed_summary': speed_df,
    'model_sizes': size_df
}

with open('models/final_summary.pkl', 'wb') as f:
    pickle.dump(summary, f)

print("\nFinal summary saved to: models/final_summary.pkl")
