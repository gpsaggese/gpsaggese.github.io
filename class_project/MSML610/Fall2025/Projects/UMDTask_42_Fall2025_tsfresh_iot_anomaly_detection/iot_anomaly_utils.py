"""
IoT Anomaly Detection Utilities

This module provides reusable utility functions and wrappers for:
- Feature engineering (using tsfresh library)
- Model training and evaluation
- Data loading and preprocessing
- Visualization helpers
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import joblib
from pathlib import Path


# ============================================================================
# Data Loading
# ============================================================================

def load_iot_data(filepath: str) -> pd.DataFrame:
    """
    Load IoT sensor data from CSV file.

    Args:
        filepath: Path to CSV file containing sensor data

    Returns:
        DataFrame with timestamp column converted to datetime
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract feature column names excluding metadata and target columns.

    Args:
        df: DataFrame containing features and targets

    Returns:
        List of feature column names
    """
    exclude_cols = [
        'timestamp', 'machine_id', 'anomaly_flag', 'maintenance_required',
        'downtime_risk', 'predicted_remaining_life', 'failure_type', 'machine_status'
    ]
    return [c for c in df.columns if c not in exclude_cols]


# ============================================================================
# Feature Engineering
# ============================================================================

def compute_basic_features(df: pd.DataFrame, sensors: List[str]) -> pd.DataFrame:
    """
    Compute basic feature transformations.

    Args:
        df: DataFrame with sensor columns
        sensors: List of sensor column names

    Returns:
        DataFrame with additional basic features
    """
    data = df.copy()

    for sensor in sensors:
        # Squared
        data[f'{sensor}_squared'] = data[sensor] ** 2

        # Square root
        data[f'{sensor}_sqrt'] = np.sqrt(np.abs(data[sensor]))

        # Log (with offset to handle negatives/zeros)
        data[f'{sensor}_log'] = np.log1p(np.abs(data[sensor]))

    return data


def compute_rolling_features(df: pd.DataFrame, sensors: List[str],
                            windows: List[int] = [6, 12, 24]) -> pd.DataFrame:
    """
    Compute rolling window statistics per machine.

    Args:
        df: DataFrame with sensor data sorted by machine_id and timestamp
        sensors: List of sensor column names
        windows: Window sizes in hours

    Returns:
        DataFrame with rolling statistics features
    """
    data = df.copy()

    for machine_id in df['machine_id'].unique():
        mask = df['machine_id'] == machine_id

        for sensor in sensors:
            for window in windows:
                # Rolling mean
                data.loc[mask, f'{sensor}_rolling_mean_{window}h'] = \
                    data.loc[mask, sensor].rolling(window=window, min_periods=1).mean()

                # Rolling std
                data.loc[mask, f'{sensor}_rolling_std_{window}h'] = \
                    data.loc[mask, sensor].rolling(window=window, min_periods=1).std().fillna(0)

                # Rolling max
                data.loc[mask, f'{sensor}_rolling_max_{window}h'] = \
                    data.loc[mask, sensor].rolling(window=window, min_periods=1).max()

                # Rolling min
                data.loc[mask, f'{sensor}_rolling_min_{window}h'] = \
                    data.loc[mask, sensor].rolling(window=window, min_periods=1).min()

    return data


# ============================================================================
# Model Training
# ============================================================================

def train_anomaly_detector(X_train: np.ndarray, y_train: np.ndarray,
                          **model_params) -> object:
    """
    Train Random Forest anomaly detector with SMOTE balancing.

    Args:
        X_train: Training features
        y_train: Training labels
        **model_params: Additional parameters for RandomForestClassifier

    Returns:
        Trained model
    """
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Balance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # Train model
    default_params = {
        'n_estimators': 300,
        'max_depth': 20,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    default_params.update(model_params)

    model = RandomForestClassifier(**default_params)
    model.fit(X_train_balanced, y_train_balanced)

    return model, scaler


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance with standard metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                                recall_score, f1_score, confusion_matrix)

    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': cm
    }


# ============================================================================
# Model Persistence
# ============================================================================

def save_model(model: object, filepath: str) -> None:
    """
    Save trained model to disk.

    Args:
        model: Trained model object
        filepath: Path to save the model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> object:
    """
    Load trained model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model object
    """
    return joblib.load(filepath)


# ============================================================================
# Visualization
# ============================================================================

def plot_confusion_matrix(cm: np.ndarray, labels: List[str],
                         title: str = 'Confusion Matrix') -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        labels: Class labels
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model: object, feature_names: List[str],
                           top_n: int = 15) -> None:
    """
    Plot top N feature importances.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Get feature importances
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importances['feature'], importances['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# ============================================================================
# Predictive Models
# ============================================================================

def create_forward_looking_labels(df: pd.DataFrame,
                                 horizon_hours: int = 24) -> pd.Series:
    """
    Create forward-looking labels for predictive models.

    Args:
        df: DataFrame with machine_id, timestamp, and anomaly_flag
        horizon_hours: Prediction horizon in hours

    Returns:
        Series with forward-looking labels
    """
    from datetime import timedelta

    df = df.copy()
    label_col = f'anomaly_next_{horizon_hours}h'
    df[label_col] = 0

    for machine_id in df['machine_id'].unique():
        machine_mask = df['machine_id'] == machine_id
        machine_df = df[machine_mask].copy()

        for idx in machine_df.index:
            current_time = df.loc[idx, 'timestamp']
            future_time = current_time + timedelta(hours=horizon_hours)

            # Check if anomaly occurs in next N hours
            future_mask = (
                (df['machine_id'] == machine_id) &
                (df['timestamp'] > current_time) &
                (df['timestamp'] <= future_time) &
                (df['anomaly_flag'] == 1)
            )

            if df[future_mask].shape[0] > 0:
                df.loc[idx, label_col] = 1

    return df[label_col]


# ============================================================================
# Data Validation
# ============================================================================

def validate_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return statistics.

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    return {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'date_range': {
            'min': df['timestamp'].min() if 'timestamp' in df.columns else None,
            'max': df['timestamp'].max() if 'timestamp' in df.columns else None
        },
        'num_machines': df['machine_id'].nunique() if 'machine_id' in df.columns else None
    }
