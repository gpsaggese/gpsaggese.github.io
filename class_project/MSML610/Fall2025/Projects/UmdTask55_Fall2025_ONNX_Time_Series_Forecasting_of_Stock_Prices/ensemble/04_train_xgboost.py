import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

from onnx_forecasting_utils import (
    XGBoostConfig, build_xgboost_model, train_xgboost_model,
    flatten_sequences_for_xgboost,
    convert_xgboost_to_onnx, verify_onnx, ONNXInferenceSession,
    evaluate_forecasts
)

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


print("\n" + "="*60)
print("Flattening sequences for XGBoost...")
print("="*60)

X_train_flat = flatten_sequences_for_xgboost(X_train)
X_val_flat = flatten_sequences_for_xgboost(X_val)
X_test_flat = flatten_sequences_for_xgboost(X_test)

y_train_flat = y_train.flatten()
y_val_flat = y_val.flatten()
y_test_flat = y_test.flatten()

print(f"Flattened data shapes for XGBoost:")
print(f"  X_train: {X_train_flat.shape} (samples, flattened_features)")
print(f"  y_train: {y_train_flat.shape}")
print(f"  X_val:   {X_val_flat.shape}")
print(f"  y_val:   {y_val_flat.shape}")
print(f"  X_test:  {X_test_flat.shape}")
print(f"  y_test:  {y_test_flat.shape}")

print(f"\nFeatures per sample: {X_train_flat.shape[1]} (15 timesteps × 13 features)")


xgb_config = XGBoostConfig(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("\n" + "="*60)
print("XGBoost Configuration:")
print("="*60)
print(f"  N Estimators: {xgb_config.n_estimators}")
print(f"  Max Depth: {xgb_config.max_depth}")
print(f"  Learning Rate: {xgb_config.learning_rate}")
print(f"  Subsample: {xgb_config.subsample}")
print(f"  Colsample by Tree: {xgb_config.colsample_bytree}")

print("\n" + "="*60)
print("Training XGBoost model on MAG 7 stacked data...")
print("="*60)

xgb_pipeline = train_xgboost_model(
    X_train_flat, y_train_flat,
    X_val_flat, y_val_flat,
    config=xgb_config,
    model_dir='models',
    verbose=True
)

print("\nXGBoost training completed!")

xgb_model = xgb_pipeline.named_steps['xgb']
feature_importance = xgb_model.feature_importances_

top_n = 20
top_indices = np.argsort(feature_importance)[-top_n:]

plt.figure(figsize=(10, 8))
plt.barh(range(top_n), feature_importance[top_indices])
plt.yticks(range(top_n), [f"Feature {i}" for i in top_indices])
plt.xlabel('Feature Importance')
plt.title(f'Top {top_n} XGBoost Feature Importances')
plt.tight_layout()
plt.savefig('models/xgboost_feature_importance.png', dpi=300)
print(f"\nFeature importance plot saved to models/xgboost_feature_importance.png")

xgb_onnx_path = 'models/xgboost_mag7.onnx'

print("\n" + "="*60)
print("Converting XGBoost model to ONNX...")
print("="*60)

xgb_onnx_result = convert_xgboost_to_onnx(
    xgb_pipeline,
    n_features=X_train_flat.shape[1],  # 195 features
    output_path=xgb_onnx_path
)

if xgb_onnx_result:
    print("\nXGBoost successfully converted to ONNX!")
    verification = verify_onnx(xgb_onnx_path)
    print(f"\nONNX Verification:")
    print(f"  Valid: {verification['is_valid']}")
    print(f"  Opset Version: {verification['opset_version']}")
    print(f"  Number of Nodes: {verification['num_nodes']}")


print("\n" + "="*60)
print("Making predictions...")
print("="*60)

# Sklearn predictions
print("Making predictions with sklearn XGBoost...")
xgb_sklearn_pred = xgb_pipeline.predict(X_test_flat).reshape(-1, 1)

# ONNX Runtime predictions
print("Making predictions with ONNX Runtime...")
xgb_onnx_session = ONNXInferenceSession(xgb_onnx_path)
xgb_onnx_pred = xgb_onnx_session.predict(X_test_flat.astype(np.float32))

# Ensure correct shape
if xgb_onnx_pred.ndim == 1:
    xgb_onnx_pred = xgb_onnx_pred.reshape(-1, 1)

print(f"\nPrediction shapes:")
print(f"  sklearn: {xgb_sklearn_pred.shape}")
print(f"  ONNX: {xgb_onnx_pred.shape}")

max_diff = np.max(np.abs(xgb_sklearn_pred - xgb_onnx_pred))
mean_diff = np.mean(np.abs(xgb_sklearn_pred - xgb_onnx_pred))

print(f"\nNumerical Comparison:")
print(f"  Max Difference: {max_diff:.6f}")
print(f"  Mean Difference: {mean_diff:.6f}")
print(f"  Predictions Match: {np.allclose(xgb_sklearn_pred, xgb_onnx_pred, rtol=1e-3)}")

xgb_metrics_overall = evaluate_forecasts(y_test, xgb_onnx_pred)

print("\n" + "="*60)
print("XGBoost Overall Performance:")
print("="*60)
print(f"  MAE: {xgb_metrics_overall['MAE']:.4f}")
print(f"  RMSE: {xgb_metrics_overall['RMSE']:.4f}")
print(f"  MAPE: {xgb_metrics_overall['MAPE']:.2f}%")
print(f"  R²: {xgb_metrics_overall['R2']:.4f}")
print(f"  Directional Accuracy: {xgb_metrics_overall['Directional_Accuracy']:.2f}%")

print("\n" + "="*60)
print("Per-Stock XGBoost Performance:")
print("="*60)

xgb_per_stock = []

for stock in stocks:
    mask = test_stock_labels == stock
    metrics = evaluate_forecasts(y_test[mask], xgb_onnx_pred[mask])
    metrics['Stock'] = stock
    xgb_per_stock.append(metrics)
    print(f"\n{stock}:")
    print(f"  MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")

xgb_per_stock_df = pd.DataFrame(xgb_per_stock)
print("\n")
print(xgb_per_stock_df.to_string(index=False))

xgb_per_stock_df.to_csv('models/xgb_per_stock_metrics.csv', index=False)

predictions_data = {
    'xgb_predictions': xgb_onnx_pred,
    'y_test': y_test,
    'test_stock_labels': test_stock_labels,
    'stocks': stocks,
    'xgb_metrics': xgb_metrics_overall,
    'xgb_per_stock': xgb_per_stock_df
}

with open('ensemble/xgb_predictions.pkl', 'wb') as f:
    pickle.dump(predictions_data, f)

print("\n" + "="*60)
print("XGBOOST TRAINING COMPLETE")
print("="*60)
print("Predictions saved to: ensemble/xgb_predictions.pkl")
print("Model saved to: models/xgboost_mag7.pkl")
print("ONNX model saved to: models/xgboost_mag7.onnx")
