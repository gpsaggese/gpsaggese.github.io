# IoT Anomaly Detection - Complete Examples

This document provides end-to-end examples of using the IoT Anomaly Detection system with the **hybrid LSTM-XGBoost ensemble**.

## Table of Contents

- [Quick Start](#quick-start)
- [Example 1: Anomaly Detection with XGBoost](#example-1-anomaly-detection-with-xgboost)
- [Example 2: Full Hybrid Ensemble Training](#example-2-full-hybrid-ensemble-training)
- [Example 3: Production Deployment](#example-3-production-deployment)
- [Results Interpretation](#results-interpretation)

## Quick Start

```bash
# Setup environment
pip install -r requirements.txt

# Or use Docker
./docker/docker_build.sh
./docker/docker_bash.sh
```

## Example 1: Anomaly Detection with XGBoost

**Goal**: Use the pre-trained XGBoost model for anomaly detection

```python
from iot_anomaly_utils import IoTAnomalyDetector
import joblib
import numpy as np

# Initialize detector
detector = IoTAnomalyDetector()

# Load data
detector.load_data("data/smart_manufacturing_data.csv", sample_size=5000)
detector.prepare_features("anomaly_flag")
X_train, X_test, y_train, y_test = detector.prepare_train_test_split()

# Load pre-trained model and scaler
scaler = joblib.load("models/anomaly_scaler.pkl")
model = joblib.load("models/anomaly_xgb.pkl")

# Scale features and predict
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)

# Evaluate
metrics = detector.evaluate_classification(y_test, predictions)

print("="*50)
print("ANOMALY DETECTION RESULTS")
print("="*50)
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1 Score:  {metrics['f1']:.4f}")
```

**Expected Output**:
```
ANOMALY DETECTION RESULTS
==================================================
Accuracy:  0.9930
Precision: 0.9922
Recall:    0.9938
F1 Score:  0.9930
```

## Example 2: Full Hybrid Ensemble Training

**Goal**: Train the complete LSTM + XGBoost + LightGBM + CatBoost ensemble

This is done in `lstm.ipynb` on Google Colab Pro. Key components:

### 2.1 Configuration

```python
@dataclass
class Config:
    sample_size: int = 100000      # Full dataset
    n_folds: int = 5               # K-Fold CV
    seq_length: int = 10           # LSTM sequence length
    lstm_hidden: int = 256         # LSTM hidden units
    lstm_layers: int = 3           # LSTM layers
    dropout: float = 0.4           # Dropout rate
    epochs: int = 100              # Max epochs
    batch_size: int = 256          # Batch size
    learning_rate: float = 0.001   # Learning rate
    patience: int = 15             # Early stopping patience
```

### 2.2 AttentionLSTM Architecture

```python
class AttentionLSTM(nn.Module):
    """Bidirectional LSTM with Self-Attention."""

    def __init__(self, input_size, hidden_size=256, num_layers=3,
                 output_size=1, dropout=0.4, task='classification'):
        super().__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Self-attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Dense layers with BatchNorm
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
```

### 2.3 Hybrid Ensemble

```python
class ProductionEnsemble:
    """LSTM + XGBoost + LightGBM + CatBoost ensemble."""

    def __init__(self, input_size, task='classification', output_size=1):
        # LSTM
        self.lstm = AttentionLSTM(input_size, ...)

        # Gradient Boosting Models
        self.xgb = XGBClassifier(n_estimators=300, learning_rate=0.05)
        self.lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05)
        self.catboost = CatBoostClassifier(iterations=300, task_type='GPU')

    def predict(self, X, weights=None):
        if weights is None:
            weights = {'lstm': 0.4, 'xgb': 0.25, 'lgbm': 0.2, 'catboost': 0.15}

        # Combine predictions with weights
        ensemble = (lstm_pred * 0.4 + xgb_pred * 0.25 +
                   lgbm_pred * 0.2 + catboost_pred * 0.15)
        return (ensemble > 0.5).astype(int)
```

### 2.4 K-Fold Training

```python
def train_with_kfold(X, y, task_name, task_type, n_classes=1):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        # Compute class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

        # Train ensemble
        model = ProductionEnsemble(X.shape[1], task, output_size)
        model.fit(X_train, y_train, X_val, y_val, class_weights)

        # Evaluate
        predictions = model.predict(X_val)
        f1 = f1_score(y_val_aligned, predictions)
```

## Example 3: Production Deployment

### 3.1 Batch Prediction Service

```python
import joblib
import pandas as pd
import numpy as np
from scripts.feature_engineering import compute_features

class AnomalyDetectionService:
    """Production service for anomaly detection."""

    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomalies for a batch of sensor readings."""
        # Compute features
        df_features = compute_features(df)

        # Extract feature columns
        exclude = ['machine_id', 'anomaly_flag', 'maintenance_required',
                  'downtime_risk', 'predicted_remaining_life', 'failure_type']
        feature_cols = [c for c in df_features.columns if c not in exclude]

        X = df_features[feature_cols].values
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

# Usage
service = AnomalyDetectionService(
    "models/anomaly_xgb.pkl",
    "models/anomaly_scaler.pkl"
)

# Batch prediction
df_new = pd.read_csv("data/smart_manufacturing_data.csv", nrows=1000)
predictions = service.predict_batch(df_new)

print(f"Total predictions: {len(predictions)}")
print(f"Anomalies detected: {predictions.sum()}")
print(f"Anomaly rate: {predictions.mean():.2%}")
```

### 3.2 Single Sample Prediction

```python
def predict_single(self, sensor_data: dict) -> dict:
    """Predict for a single sensor reading."""
    df = pd.DataFrame([sensor_data])
    df_features = compute_features(df)

    feature_cols = [c for c in df_features.columns
                   if c not in exclude_cols]
    X = df_features[feature_cols].values
    X_scaled = self.scaler.transform(X)

    prediction = self.model.predict(X_scaled)[0]
    probability = self.model.predict_proba(X_scaled)[0]

    return {
        'prediction': int(prediction),
        'is_anomaly': bool(prediction == 1),
        'probability': probability.tolist(),
        'confidence': float(max(probability))
    }

# Test
reading = {
    'timestamp': '2025-01-09 10:30:00',
    'machine_id': 15,
    'temperature': 125.8,  # High - anomalous
    'vibration': 85.2,     # High - anomalous
    'humidity': 25.1,
    'pressure': 5.2,
    'energy_consumption': 7.5,
    'machine_status': 1,
    'anomaly_flag': 0,     # Unknown
    'predicted_remaining_life': 50,
    'failure_type': 'Normal',
    'downtime_risk': 0.0,
    'maintenance_required': 0
}

result = service.predict_single(reading)
print(f"Prediction: {'ANOMALY' if result['is_anomaly'] else 'NORMAL'}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Results Interpretation

### Classification Metrics

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **F1 Score** | Harmonic mean of precision & recall | > 0.90 |
| **Precision** | % of predicted anomalies that are real | High = few false alarms |
| **Recall** | % of real anomalies detected | High = few missed |

### Our Results (5-Fold CV, 100K samples)

| Task | F1 | Precision | Recall | Assessment |
|------|-----|-----------|--------|------------|
| Anomaly Detection | 99.30% | 99.22% | 99.38% | Excellent |
| Downtime Risk | 99.33% | 99.27% | 99.39% | Excellent |
| Maintenance | 61.65% | 97.10% | 45.17% | Conservative |
| Failure Type | 19.16% | - | - | Class imbalance |
| RUL (R²) | 17.81% | - | MAE=113.6h | Limited signal |

### Interpretation

**Anomaly Detection (F1 = 99.30%)**
- Near-perfect detection
- Very few false positives or missed anomalies
- Production ready

**Maintenance (F1 = 61.65%)**
- High precision (97%) = very few false alarms
- Low recall (45%) = misses many maintenance needs
- **Recommendation**: Lower threshold from 0.5 to ~0.3 if recall is important

**Failure Type (F1_macro = 19.16%)**
- 92% of samples are "Normal" class
- Model predicts "Normal" for everything
- **Recommendation**: Train only on anomalous samples

**RUL (R² = 17.81%)**
- Model explains only 18% of variance
- MAE of 113.6 hours with mean RUL of 234.3 hours
- **Conclusion**: Sensor features don't predict remaining life well

## Known Limitations

1. **Correlated Targets**: `anomaly_flag` ≈ `downtime_risk` (same underlying signal)
2. **Class Imbalance**: Failure type has 92% "Normal" class
3. **Temporal Validation**: Used stratified k-fold, not time-series split
4. **Feature Engineering**: Per-machine statistics computed on full dataset

## Best Practices

1. **Always use the scaler** from `models/anomaly_scaler.pkl`
2. **For maintenance**: Consider `predict_proba` with custom threshold
3. **For failure type**: Filter to anomalous samples first
4. **Monitor**: Watch for data drift in production
5. **Validate**: Test on truly held-out temporal data before deployment

## Next Steps

- **API Reference**: See [iot_anomaly.API.md](iot_anomaly.API.md)
- **Training Notebook**: See [lstm.ipynb](lstm.ipynb) for Colab training
- **Project Overview**: See [README.md](README.md)
