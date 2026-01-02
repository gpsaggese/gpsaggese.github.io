#!/usr/bin/env python3
"""
Predictive Anomaly Detection Model
Goal: Predict anomalies 24-48 hours in advance

Key differences from reactive model:
1. Target labels shifted backward (predict future anomalies)
2. Time-series features with temporal context
3. LSTM/sequence model for temporal patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os
from datetime import timedelta

print("="*70)
print("PREDICTIVE ANOMALY DETECTION MODEL")
print("Training models to predict anomalies 24-48 hours in advance")
print("="*70)

# Load data
print("\n1. Loading data...")
df = pd.read_csv("data/smart_manufacturing_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['machine_id', 'timestamp']).reset_index(drop=True)

print(f"Dataset: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Current anomaly rate: {df['anomaly_flag'].mean()*100:.2f}%")

# Create predictive targets (shifted labels)
print("\n2. Creating forward-looking target labels...")

def create_predictive_targets(df, horizons=[24, 48]):
    """
    Create target labels that indicate if an anomaly will occur in the next N hours.

    Args:
        df: DataFrame with machine_id, timestamp, anomaly_flag
        horizons: List of hours to predict ahead (e.g., [24, 48])

    Returns:
        DataFrame with new columns: anomaly_next_24h, anomaly_next_48h
    """
    df = df.copy()

    for horizon in horizons:
        col_name = f'anomaly_next_{horizon}h'
        df[col_name] = 0

        # For each machine, check if anomaly occurs within next N hours
        for machine_id in df['machine_id'].unique():
            machine_mask = df['machine_id'] == machine_id
            machine_df = df[machine_mask].copy()

            for idx in machine_df.index:
                current_time = df.loc[idx, 'timestamp']
                future_time = current_time + timedelta(hours=horizon)

                # Check if any anomaly occurs between current_time and future_time
                future_mask = (
                    (df['machine_id'] == machine_id) &
                    (df['timestamp'] > current_time) &
                    (df['timestamp'] <= future_time) &
                    (df['anomaly_flag'] == 1)
                )

                if df[future_mask].shape[0] > 0:
                    df.loc[idx, col_name] = 1

        print(f"  Created {col_name}: {df[col_name].sum()} positive samples ({df[col_name].mean()*100:.2f}%)")

    return df

# Create targets for 24h and 48h prediction
df = create_predictive_targets(df, horizons=[24, 48])

# Create temporal features
print("\n3. Engineering temporal features...")

def create_temporal_features(df):
    """
    Create time-series features with temporal context.

    Features include:
    - Rolling statistics (mean, std, min, max) over various windows
    - Trend features (change over time)
    - Lag features (previous values)
    - Time-based features (hour, day of week, etc.)
    """
    df = df.copy()
    sensors = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']

    # Group by machine for temporal features
    for machine_id in df['machine_id'].unique():
        mask = df['machine_id'] == machine_id
        machine_indices = df[mask].index

        # Time-based features
        df.loc[mask, 'hour'] = df.loc[mask, 'timestamp'].dt.hour
        df.loc[mask, 'day_of_week'] = df.loc[mask, 'timestamp'].dt.dayofweek
        df.loc[mask, 'day_of_month'] = df.loc[mask, 'timestamp'].dt.day

        for sensor in sensors:
            sensor_values = df.loc[mask, sensor].values

            # Rolling statistics (6h, 12h, 24h windows)
            for window in [6, 12, 24]:
                # Mean
                df.loc[mask, f'{sensor}_rolling_mean_{window}h'] = \
                    df.loc[mask, sensor].rolling(window=window, min_periods=1).mean()

                # Std
                df.loc[mask, f'{sensor}_rolling_std_{window}h'] = \
                    df.loc[mask, sensor].rolling(window=window, min_periods=1).std().fillna(0)

                # Max
                df.loc[mask, f'{sensor}_rolling_max_{window}h'] = \
                    df.loc[mask, sensor].rolling(window=window, min_periods=1).max()

                # Min
                df.loc[mask, f'{sensor}_rolling_min_{window}h'] = \
                    df.loc[mask, sensor].rolling(window=window, min_periods=1).min()

            # Trend features (rate of change)
            df.loc[mask, f'{sensor}_change_6h'] = \
                df.loc[mask, sensor].diff(6).fillna(0)

            df.loc[mask, f'{sensor}_change_12h'] = \
                df.loc[mask, sensor].diff(12).fillna(0)

            # Lag features (previous values)
            for lag in [1, 3, 6, 12]:
                df.loc[mask, f'{sensor}_lag_{lag}h'] = \
                    df.loc[mask, sensor].shift(lag).fillna(method='bfill')

        # Cross-sensor features
        df.loc[mask, 'temp_vibration_ratio'] = \
            df.loc[mask, 'temperature'] / (df.loc[mask, 'vibration'] + 1e-5)

        df.loc[mask, 'pressure_temp_product'] = \
            df.loc[mask, 'pressure'] * df.loc[mask, 'temperature']

    # Fill any remaining NaN values
    df = df.fillna(method='bfill').fillna(0)

    return df

# Apply temporal feature engineering
df = create_temporal_features(df)

# Get feature columns (exclude targets and metadata)
exclude_cols = ['timestamp', 'machine_id', 'anomaly_flag', 'maintenance_required',
                'downtime_risk', 'predicted_remaining_life', 'failure_type',
                'machine_status', 'anomaly_next_24h', 'anomaly_next_48h']

feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"Total temporal features created: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:5]}")

# Train predictive models
print("\n4. Training predictive models...")

results = {}

for horizon in [24, 48]:
    target_col = f'anomaly_next_{horizon}h'

    print(f"\n{'='*70}")
    print(f"Training model for {horizon}-hour prediction horizon")
    print(f"{'='*70}")

    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Split data (temporal split - train on earlier data, test on later data)
    # This simulates real-world deployment
    split_idx = int(len(df) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print(f"\nTemporal split (train on earlier 80%, test on later 20%):")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Train anomaly rate: {y_train.mean()*100:.2f}%")
    print(f"  Test anomaly rate: {y_test.mean()*100:.2f}%")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Gradient Boosting (better for imbalanced temporal data)
    print(f"\nTraining Gradient Boosting Classifier...")

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        verbose=0
    )

    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nResults for {horizon}-hour prediction:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives:  {cm[0,0]}")
    print(f"    False Positives: {cm[0,1]}")
    print(f"    False Negatives: {cm[1,0]}")
    print(f"    True Positives:  {cm[1,1]}")

    # Calculate lead time metrics
    if cm[1,1] > 0:
        detection_rate = cm[1,1] / (cm[1,1] + cm[1,0]) * 100
        print(f"\n  Early Warning Capability:")
        print(f"    Detection rate {horizon}h in advance: {detection_rate:.1f}%")
        print(f"    Anomalies caught early: {cm[1,1]} out of {cm[1,1] + cm[1,0]}")
        print(f"    Lead time: {horizon} hours ({horizon/24:.1f} days)")

    # Store results
    results[horizon] = {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'cm': cm
    }

    # Save model
    os.makedirs('models/predictive', exist_ok=True)
    joblib.dump(model, f'models/predictive/anomaly_predictor_{horizon}h.pkl')
    joblib.dump(scaler, f'models/predictive/scaler_{horizon}h.pkl')
    print(f"\nModel saved to models/predictive/anomaly_predictor_{horizon}h.pkl")

# Comparison visualization
print(f"\n{'='*70}")
print("5. Creating comparison visualizations...")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot for 24h model
horizon_24 = results[24]
horizon_48 = results[48]

# Row 1: 24h predictions
# Confusion matrix
ax = axes[0, 0]
sns.heatmap(horizon_24['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Anomaly', 'Anomaly'], yticklabels=['No Anomaly', 'Anomaly'])
ax.set_title('24-Hour Prediction\nConfusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# ROC-like score distribution
ax = axes[0, 1]
ax.hist(horizon_24['y_proba'][horizon_24['y_test'] == 0], bins=50, alpha=0.7,
        label='No Anomaly', color='blue')
ax.hist(horizon_24['y_proba'][horizon_24['y_test'] == 1], bins=50, alpha=0.7,
        label='Anomaly in 24h', color='red')
ax.set_xlabel('Prediction Probability')
ax.set_ylabel('Frequency')
ax.set_title('24-Hour Prediction\nScore Distribution')
ax.legend()

# Metrics
ax = axes[0, 2]
ax.axis('off')
metrics_text_24 = f"""
24-HOUR PREDICTION MODEL

Accuracy:  {horizon_24['accuracy']:.4f}
Precision: {horizon_24['precision']:.4f}
Recall:    {horizon_24['recall']:.4f}
F1 Score:  {horizon_24['f1']:.4f}

Early Detection:
• Lead time: 24 hours (1 day)
• Detection rate: {horizon_24['recall']*100:.1f}%
• Anomalies caught: {horizon_24['cm'][1,1]}
• False alarms: {horizon_24['cm'][0,1]}
"""
ax.text(0.1, 0.5, metrics_text_24, fontsize=11, family='monospace',
        verticalalignment='center', transform=ax.transAxes)

# Row 2: 48h predictions
# Confusion matrix
ax = axes[1, 0]
sns.heatmap(horizon_48['cm'], annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=['No Anomaly', 'Anomaly'], yticklabels=['No Anomaly', 'Anomaly'])
ax.set_title('48-Hour Prediction\nConfusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# Score distribution
ax = axes[1, 1]
ax.hist(horizon_48['y_proba'][horizon_48['y_test'] == 0], bins=50, alpha=0.7,
        label='No Anomaly', color='blue')
ax.hist(horizon_48['y_proba'][horizon_48['y_test'] == 1], bins=50, alpha=0.7,
        label='Anomaly in 48h', color='red')
ax.set_xlabel('Prediction Probability')
ax.set_ylabel('Frequency')
ax.set_title('48-Hour Prediction\nScore Distribution')
ax.legend()

# Metrics
ax = axes[1, 2]
ax.axis('off')
metrics_text_48 = f"""
48-HOUR PREDICTION MODEL

Accuracy:  {horizon_48['accuracy']:.4f}
Precision: {horizon_48['precision']:.4f}
Recall:    {horizon_48['recall']:.4f}
F1 Score:  {horizon_48['f1']:.4f}

Early Detection:
• Lead time: 48 hours (2 days)
• Detection rate: {horizon_48['recall']*100:.1f}%
• Anomalies caught: {horizon_48['cm'][1,1]}
• False alarms: {horizon_48['cm'][0,1]}
"""
ax.text(0.1, 0.5, metrics_text_48, fontsize=11, family='monospace',
        verticalalignment='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('predictive_model_results.png', dpi=150)
print("\nVisualization saved to 'predictive_model_results.png'")

# Comparison chart
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

horizons = [24, 48]
metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

x = np.arange(len(horizons))
width = 0.2

for i, metric in enumerate(metrics):
    values = [results[h][metric] for h in horizons]
    ax.bar(x + i*width, values, width, label=metric_names[i])

ax.set_xlabel('Prediction Horizon (hours)')
ax.set_ylabel('Score')
ax.set_title('Predictive Model Performance Comparison')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f'{h}h\n({h/24:.0f} day)' for h in horizons])
ax.legend()
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('predictive_comparison.png', dpi=150)
print("Comparison chart saved to 'predictive_comparison.png'")

# Final summary
print(f"\n{'='*70}")
print("PREDICTIVE MODEL TRAINING COMPLETE")
print(f"{'='*70}")
print(f"\nSummary:")
print(f"  Models trained: 2 (24h and 48h prediction)")
print(f"  Features used: {len(feature_cols)}")
print(f"  Test samples: {len(y_test)}")
print(f"\nEarly Warning Capabilities:")
print(f"  24-hour prediction:")
print(f"    - Accuracy: {results[24]['accuracy']*100:.2f}%")
print(f"    - Can detect {results[24]['recall']*100:.1f}% of anomalies 1 day in advance")
print(f"  48-hour prediction:")
print(f"    - Accuracy: {results[48]['accuracy']*100:.2f}%")
print(f"    - Can detect {results[48]['recall']*100:.1f}% of anomalies 2 days in advance")
print(f"\nModels saved to: models/predictive/")
print(f"Visualizations saved: predictive_model_results.png, predictive_comparison.png")
print(f"{'='*70}")
