# IoT Anomaly Detection - Complete Example

This document provides a complete end-to-end example of using the IoT Anomaly Detection system, from data loading to model deployment.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Example 1: Quick Anomaly Detection](#example-1-quick-anomaly-detection)
- [Example 2: Complete Training Pipeline](#example-2-complete-training-pipeline)
- [Example 3: Using Pre-Trained Models](#example-3-using-pre-trained-models)
- [Example 4: Custom Model Training](#example-4-custom-model-training)
- [Example 5: Hybrid Deep Learning Model](#example-5-hybrid-deep-learning-model)
- [Example 6: Batch Predictions on New Data](#example-6-batch-predictions-on-new-data)
- [Results Interpretation](#results-interpretation)

## Introduction

This guide walks through practical examples of using the IoT anomaly detection system. Each example builds on the previous one, culminating in a production-ready deployment workflow.

## Prerequisites

```bash
# Install dependencies
pip install -r docker/requirements.txt

# Verify data is available
ls data/smart_manufacturing_data.csv

# Verify scripts are available
ls scripts/
```

## Example 1: Quick Anomaly Detection

**Goal**: Detect anomalies in IoT sensor data using a pre-trained model

**Time**: ~2 minutes

### Step 1: Train a Simple Model

```bash
python scripts/train_models_kfold.py \
    --input data/smart_manufacturing_data.csv \
    --output_dir results \
    --sample 5000 \
    --n_folds 3
```

**What this does**:
- Loads 5,000 samples from the dataset
- Engineers 100+ features from raw sensors
- Trains 12 models with 3-fold cross-validation
- Saves the best model for each task
- Outputs results to `results/` directory

**Expected output**:
```
================================================================================
IoT Anomaly Detection - Model Training Pipeline
================================================================================
Input: data/smart_manufacturing_data.csv
Output: results
K-Folds: 3
Sample size: 5000
================================================================================

Loading dataset...
Dataset loaded. Shape: (500000, 13)
Sampled to 5000 rows

Computing engineered features...
Features computed. Shape: (4998, 108)

Standardizing features...
Scaler saved to models/scaler.pkl

================================================================================
TASK 1: Anomaly Detection (Binary Classification)
================================================================================
Class distribution: [3847 1151]
  Training 1_RandomForest...
    F1: 0.9234 (+/- 0.0123)
  Training 2_XGBoost...
    F1: 0.9456 (+/- 0.0089)
  ...

Top 3 models by F1 score:
                 model  f1_mean  precision_mean  recall_mean
2            2_XGBoost   0.9456          0.9512       0.9401
1      1_RandomForest   0.9234          0.9289       0.9180
3           3_LightGBM   0.9401          0.9445       0.9358

Best model: 2_XGBoost with f1_mean=0.9456
Saved best model to models/best_anomaly_model.pkl
```

### Step 2: Use the Model

```python
from iot_anomaly_utils import IoTAnomalyDetector

# Load detector
detector = IoTAnomalyDetector()
detector.load_data("data/smart_manufacturing_data.csv", sample_size=1000)
detector.prepare_features("anomaly_flag")
X_train, X_test, y_train, y_test = detector.prepare_train_test_split()

# Load pre-trained model
detector.load_model("models/best_anomaly_model.pkl")

# Predict
predictions = detector.predict(X_test)

# Evaluate
metrics = detector.evaluate_classification(y_test, predictions)

# Output:
# Classification Metrics:
#   Accuracy:  0.9450
#   Precision: 0.9512
#   Recall:    0.9401
#   F1 Score:  0.9456
```

---

## Example 2: Complete Training Pipeline

**Goal**: Train all models on all tasks and generate comprehensive visualizations

**Time**: ~15-30 minutes (depending on sample size)

### Full Training Workflow

```bash
#!/bin/bash
# complete_training.sh

echo "Step 1: Training all classical ML models..."
python scripts/train_models_kfold.py \
    --input data/smart_manufacturing_data.csv \
    --output_dir results \
    --sample 10000 \
    --n_folds 5

echo "Step 2: Training hybrid LSTM-XGBoost model..."
python scripts/train_lstm_xgb_ensemble.py \
    --input data/smart_manufacturing_data.csv \
    --output_dir results \
    --sample 20000 \
    --epochs 30

echo "Step 3: Generating visualizations..."
python scripts/create_visualizations.py \
    --results_dir results \
    --output_dir charts

echo "Complete! Check 'results/' and 'charts/' directories"
```

**Run it**:
```bash
chmod +x complete_training.sh
./complete_training.sh
```

### Results Analysis

```python
from iot_anomaly_utils import load_results, plot_model_comparison
import matplotlib.pyplot as plt

# Load all results
results = load_results("results")

# View summary for each task
for task, df in results.items():
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}")
    print(df[['model', 'f1_mean', 'precision_mean', 'recall_mean']].head())

# Plot comparison
for task, df in results.items():
    if 'f1_mean' in df.columns:
        fig = plot_model_comparison(df, metric='f1_mean', title=f'{task} - F1 Score')
        plt.savefig(f'charts/{task}_comparison.png')
        plt.show()
```

---

## Example 3: Using Pre-Trained Models

**Goal**: Load saved models and make predictions on new data

```python
from iot_anomaly_utils import IoTAnomalyDetector
import pandas as pd

# Initialize detector
detector = IoTAnomalyDetector()

# Load new data
detector.load_data("data/smart_manufacturing_data.csv", sample_size=2000)

# Prepare features (must match training)
detector.prepare_features("anomaly_flag")

# Split data
X_train, X_test, y_train, y_test = detector.prepare_train_test_split(
    target_col="anomaly_flag",
    test_size=0.3  # Use more for testing
)

# Load best anomaly detection model
detector.load_model("models/best_anomaly_model.pkl")

# Make predictions
anomaly_predictions = detector.predict(X_test)

# Evaluate
metrics = detector.evaluate_classification(y_test, anomaly_predictions)

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': anomaly_predictions,
    'correct': y_test == anomaly_predictions
})

# Analyze results
print(f"\nAccuracy: {predictions_df['correct'].mean():.2%}")
print(f"Total samples: {len(predictions_df)}")
print(f"Correct predictions: {predictions_df['correct'].sum()}")
print(f"Incorrect predictions: {(~predictions_df['correct']).sum()}")

# Save predictions
predictions_df.to_csv("predictions_anomaly.csv", index=False)
```

---

## Example 4: Custom Model Training

**Goal**: Train your own custom model configuration

```python
from iot_anomaly_utils import IoTAnomalyDetector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Prepare data
detector = IoTAnomalyDetector()
detector.load_data("data/smart_manufacturing_data.csv", sample_size=8000)
detector.prepare_features("maintenance_required")
X_train, X_test, y_train, y_test = detector.prepare_train_test_split(
    target_col="maintenance_required"
)

# Define custom model with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Train
print("Training custom model with grid search...")
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Assign to detector
detector.model = best_model

# Evaluate on test set
predictions = detector.predict(X_test)
metrics = detector.evaluate_classification(y_test, predictions)

# Save custom model
detector.save_model("models/custom_maintenance_model.pkl")

# Plot feature importance
fig = detector.plot_feature_importance(top_n=20)
plt.title("Custom Model - Top 20 Features")
plt.tight_layout()
plt.savefig("charts/custom_feature_importance.png")
plt.show()
```

---

## Example 5: Hybrid Deep Learning Model

**Goal**: Train and use the LSTM-XGBoost hybrid ensemble

### Training

```bash
python scripts/train_lstm_xgb_ensemble.py \
    --input data/smart_manufacturing_data.csv \
    --output_dir results \
    --sample 15000 \
    --seq_length 10 \
    --epochs 30 \
    --batch_size 64 \
    --lstm_hidden 128 \
    --lstm_layers 2 \
    --dropout 0.3
```

### Using the Hybrid Model

```python
import joblib
import numpy as np
from scripts.feature_engineering import compute_features
from sklearn.preprocessing import StandardScaler

# Load hybrid model
hybrid_model = joblib.load("models/hybrid_lstm_xgb_anomaly.pkl")

# Prepare new data
import pandas as pd
df_new = pd.read_csv("data/smart_manufacturing_data.csv", nrows=1000)

# Compute features
df_features = compute_features(df_new)

# Extract features
feature_cols = [c for c in df_features.columns
                if c not in ['machine_id', 'anomaly_flag', 'maintenance_required',
                            'downtime_risk', 'predicted_remaining_life', 'failure_type']]
X_new = df_features[feature_cols].values

# Make predictions (hybrid model handles scaling internally)
predictions = hybrid_model.predict(X_new)

print(f"Predictions: {predictions}")
print(f"Anomalies detected: {predictions.sum()}")
```

---

## Example 6: Batch Predictions on New Data

**Goal**: Process large datasets in batches for production deployment

```python
from iot_anomaly_utils import IoTAnomalyDetector
import pandas as pd
import numpy as np

def batch_predict(data_path, model_path, batch_size=1000):
    """Process data in batches for memory efficiency."""

    detector = IoTAnomalyDetector()

    # Load model once
    detector.load_model(model_path)

    # Read data in chunks
    all_predictions = []

    for chunk in pd.read_csv(data_path, chunksize=batch_size):
        # Prepare features for chunk
        detector.df_raw = chunk
        detector.prepare_features("anomaly_flag")

        # Get features
        feature_cols = [c for c in detector.df_features.columns
                       if c not in ['machine_id', 'anomaly_flag',
                                   'maintenance_required', 'downtime_risk',
                                   'predicted_remaining_life', 'failure_type']]

        X_chunk = detector.df_features[feature_cols].values

        # Scale if needed
        if detector.scaler is None:
            # Load scaler
            import joblib
            detector.scaler = joblib.load("models/scaler.pkl")

        X_chunk_scaled = detector.scaler.transform(X_chunk)

        # Predict
        predictions = detector.model.predict(X_chunk_scaled)

        all_predictions.append(predictions)

    # Combine all predictions
    return np.concatenate(all_predictions)

# Use it
predictions = batch_predict(
    "data/smart_manufacturing_data.csv",
    "models/best_anomaly_model.pkl",
    batch_size=5000
)

print(f"Total predictions: {len(predictions)}")
print(f"Anomalies detected: {predictions.sum()}")
print(f"Anomaly rate: {predictions.mean():.2%}")
```

---

## Results Interpretation

### Classification Results

**F1 Score**: Harmonic mean of precision and recall
- `>0.9`: Excellent
- `0.8-0.9`: Good
- `0.7-0.8`: Fair
- `<0.7`: Needs improvement

**Precision**: Of all predicted anomalies, how many were actual anomalies?
- High precision = Few false alarms

**Recall**: Of all actual anomalies, how many did we detect?
- High recall = Few missed anomalies

**Example interpretation**:
```
F1 Score: 0.9456
Precision: 0.9512
Recall: 0.9401
```
This means:
- Overall performance is excellent (F1 > 0.9)
- 95.12% of predicted anomalies are real (good precision)
- We detect 94.01% of all anomalies (good recall)

### Regression Results

**MAE (Mean Absolute Error)**: Average prediction error
- Lower is better
- Same units as target (e.g., hours for RUL)

**RMSE (Root Mean Squared Error)**: Penalizes large errors more
- Lower is better
- Same units as target

**R² Score**: Proportion of variance explained
- `1.0`: Perfect predictions
- `>0.9`: Excellent
- `0.7-0.9`: Good
- `<0.7`: Needs improvement

**Example interpretation**:
```
MAE: 15.23 hours
RMSE: 22.45 hours
R²: 0.8956
```
This means:
- Average error is ~15 hours
- Model explains 89.56% of variance (good)
- Useful for maintenance scheduling

---

## Best Practices

1. **Always use cross-validation** for model selection
2. **Monitor for data drift** in production
3. **Retrain periodically** as new data arrives
4. **Version your models** (e.g., `model_v1_20250109.pkl`)
5. **Log predictions** for audit trails
6. **Set appropriate thresholds** based on business costs
7. **Validate on held-out data** before deployment

---

## Production Deployment Example

```python
import joblib
import pandas as pd
from scripts.feature_engineering import compute_features

class AnomalyDetectionService:
    """Production service for anomaly detection."""

    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_single(self, sensor_data: dict) -> dict:
        """Predict for a single sensor reading."""

        # Convert to DataFrame
        df = pd.DataFrame([sensor_data])

        # Engineer features
        df_features = compute_features(df)

        # Extract features
        feature_cols = [c for c in df_features.columns
                       if c not in ['machine_id', 'anomaly_flag',
                                   'maintenance_required', 'downtime_risk',
                                   'predicted_remaining_life', 'failure_type']]

        X = df_features[feature_cols].values

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X_scaled)[0]

        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(X_scaled)[0]
        else:
            probability = None

        return {
            'prediction': int(prediction),
            'is_anomaly': bool(prediction == 1),
            'probability': probability.tolist() if probability is not None else None
        }

# Use in production
service = AnomalyDetectionService(
    "models/best_anomaly_model.pkl",
    "models/scaler.pkl"
)

# Example sensor reading
reading = {
    'timestamp': '2025-01-09 10:30:00',
    'machine_id': 15,
    'temperature': 95.2,
    'vibration': 62.5,
    'humidity': 45.3,
    'pressure': 3.8,
    'energy_consumption': 4.2,
    'machine_status': 1,
    'anomaly_flag': 0,
    'predicted_remaining_life': 50,
    'failure_type': 'Normal',
    'downtime_risk': 0.0,
    'maintenance_required': 0
}

result = service.predict_single(reading)
print(result)
# Output: {'prediction': 1, 'is_anomaly': True, 'probability': [0.12, 0.88]}
```

---

## Summary

This guide demonstrated:
1.  Quick anomaly detection with pre-trained models
2.  Complete training pipeline for all models
3.  Using saved models for predictions
4.  Custom model training with hyperparameter tuning
5.  Hybrid deep learning ensemble
6.  Batch processing for production
7.  Results interpretation
8.  Production deployment pattern

**For API reference, see [iot_anomaly.API.md](iot_anomaly.API.md)**

**For Jupyter notebook version, see [iot_anomaly.example.ipynb](iot_anomaly.example.ipynb)**
