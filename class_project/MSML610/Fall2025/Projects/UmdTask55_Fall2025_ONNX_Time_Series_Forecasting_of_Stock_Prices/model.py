import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import json
import os


@dataclass
class LSTMConfig:
    """Configuration for LSTM model hyperparameters."""
    sequence_length: int = 60
    n_features: int = 1
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dropout_rate: float = 0.2
    dense_units: int = 32
    output_dim: int = 1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2


def build_lstm_model(config: LSTMConfig) -> keras.Model:
    """
    Build LSTM model architecture.

    Args:
        config: LSTMConfig with model hyperparameters

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=(config.sequence_length, config.n_features)),

        layers.LSTM(config.lstm_units_1, return_sequences=True),
        layers.Dropout(config.dropout_rate),

        layers.LSTM(config.lstm_units_2, return_sequences=False),
        layers.Dropout(config.dropout_rate),

        layers.Dense(config.dense_units, activation='relu'),
        layers.Dropout(config.dropout_rate),

        layers.Dense(config.output_dim)
    ])

    return model


def compile_model(model: keras.Model, learning_rate: float = 0.001) -> keras.Model:
    """
    Compile LSTM model with optimizer and loss function.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mae',  # Mean Absolute Error - less sensitive to outliers than MSE
        metrics=['mae', 'mape']
    )
    return model


def create_callbacks(model_path: str, patience: int = 10) -> list:
    """
    Create training callbacks.

    Args:
        model_path: Path to save best model
        patience: Patience for early stopping

    Returns:
        List of callbacks
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    return callback_list


def train_lstm_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[LSTMConfig] = None,
    model_path: str = 'models/lstm_model.keras',
    verbose: int = 1
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Train LSTM model with validation.

    Args:
        model: Compiled Keras model
        X_train: Training sequences (n_samples, sequence_length, n_features)
        y_train: Training targets (n_samples, output_dim)
        X_val: Validation sequences
        y_val: Validation targets
        config: LSTMConfig with training parameters
        model_path: Path to save best model
        verbose: Verbosity level

    Returns:
        Tuple of (trained_model, history)
    """
    if config is None:
        config = LSTMConfig()

    if X_val is None or y_val is None:
        validation_data = None
        validation_split = config.validation_split
    else:
        validation_data = (X_val, y_val)
        validation_split = 0.0

    callback_list = create_callbacks(model_path, patience=10)

    history = model.fit(
        X_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=validation_split,
        validation_data=validation_data,
        callbacks=callback_list,
        verbose=verbose
    )

    return model, history


def save_model_and_history(
    model: keras.Model,
    history: keras.callbacks.History,
    model_dir: str = 'models'
) -> Dict[str, str]:
    """
    Save trained model and training history.

    Args:
        model: Trained Keras model
        history: Training history
        model_dir: Directory to save model files

    Returns:
        Dictionary with file paths
    """
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'lstm_model.keras')
    model.save(model_path)

    history_path = os.path.join(model_dir, 'training_history.json')
    history_dict = {key: [float(val) for val in values]
                   for key, values in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    return {
        'keras_model': model_path,
        'history': history_path
    }


def load_training_history(history_path: str) -> Dict[str, list]:
    """
    Load training history from JSON file.

    Args:
        history_path: Path to history JSON file

    Returns:
        Dictionary with training history
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def plot_training_history(
    history: keras.callbacks.History,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Visualize training and validation curves.

    Args:
        history: Training history object or dict
        save_path: Path to save plot (optional)
        figsize: Figure size
    """
    if isinstance(history, keras.callbacks.History):
        history_dict = history.history
    else:
        history_dict = history

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(history_dict['loss'], label='Train Loss')
    if 'val_loss' in history_dict:
        axes[0].plot(history_dict['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history_dict['mae'], label='Train MAE')
    if 'val_mae' in history_dict:
        axes[1].plot(history_dict['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history_dict['mape'], label='Train MAPE')
    if 'val_mape' in history_dict:
        axes[2].plot(history_dict['val_mape'], label='Val MAPE')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].set_title('Mean Absolute Percentage Error')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def get_model_summary(model: keras.Model) -> str:
    """
    Get model architecture summary.

    Args:
        model: Keras model

    Returns:
        String with model summary
    """
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return '\n'.join(summary_list)


def create_and_train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[LSTMConfig] = None,
    model_dir: str = 'models',
    verbose: int = 1
) -> Tuple[keras.Model, keras.callbacks.History, Dict[str, str]]:
    """
    End-to-end function to create, train, and save LSTM model.

    Args:
        X_train: Training sequences
        y_train: Training targets
        X_val: Validation sequences
        y_val: Validation targets
        config: LSTMConfig with hyperparameters
        model_dir: Directory to save model files
        verbose: Verbosity level

    Returns:
        Tuple of (model, history, file_paths)
    """
    if config is None:
        config = LSTMConfig()
        config.n_features = X_train.shape[2]
        config.sequence_length = X_train.shape[1]

    print(f"Building LSTM model with config: {config}")
    model = build_lstm_model(config)

    print("\nModel Architecture:")
    print(get_model_summary(model))

    model = compile_model(model, learning_rate=config.learning_rate)

    print("\nTraining model...")
    model_path = os.path.join(model_dir, 'lstm_model_best.keras')
    model, history = train_lstm_model(
        model, X_train, y_train, X_val, y_val,
        config=config, model_path=model_path, verbose=verbose
    )

    print("\nSaving model and history...")
    file_paths = save_model_and_history(model, history, model_dir)

    print("\nPlotting training history...")
    plot_path = os.path.join(model_dir, 'training_curves.png')
    plot_training_history(history, save_path=plot_path)
    file_paths['plot'] = plot_path

    print(f"\nTraining complete. Files saved:")
    for key, path in file_paths.items():
        print(f"  {key}: {path}")

    return model, history, file_paths
