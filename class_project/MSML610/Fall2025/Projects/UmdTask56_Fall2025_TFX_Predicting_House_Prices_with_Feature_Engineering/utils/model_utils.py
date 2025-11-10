"""
Model training utilities for the house price prediction pipeline

This module contains:
- XGBoost model training functions
- TensorFlow DNN model training functions
- Model persistence and loading
"""

import tensorflow as tf
import xgboost as xgb
from typing import Dict, Any

from . import config


def create_xgboost_model(params: Dict[str, Any] = None):
    """
    Create and configure XGBoost model.

    Args:
        params: XGBoost hyperparameters (optional)

    Returns:
        Configured XGBoost model

    TODO: Implement in Phase 4
    """
    if params is None:
        params = config.XGBOOST_PARAMS

    model = xgb.XGBRegressor(**params)
    return model


def create_tensorflow_model(input_shape: int, params: Dict[str, Any] = None):
    """
    Create and configure TensorFlow DNN model.

    Args:
        input_shape: Number of input features
        params: Model hyperparameters (optional)

    Returns:
        Compiled Keras model

    TODO: Implement in Phase 4
    """
    if params is None:
        params = config.TF_DNN_PARAMS

    # Placeholder model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def train_xgboost_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    Train XGBoost model.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        params: Model parameters (optional)

    Returns:
        Trained model

    TODO: Implement in Phase 4
    """
    model = create_xgboost_model(params)

    if X_val is not None and y_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
    else:
        model.fit(X_train, y_train)

    return model


def train_tensorflow_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    Train TensorFlow DNN model.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        params: Model parameters (optional)

    Returns:
        Trained model

    TODO: Implement in Phase 4
    """
    if params is None:
        params = config.TF_DNN_PARAMS

    model = create_tensorflow_model(X_train.shape[1], params)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params.get('early_stopping_patience', 10),
            restore_best_weights=True
        )
    ]

    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

    return model, history


def test_model_training():
    """Test function to verify model utilities work."""
    print("Model training placeholder - will be implemented in Phase 4")


if __name__ == "__main__":
    test_model_training()
