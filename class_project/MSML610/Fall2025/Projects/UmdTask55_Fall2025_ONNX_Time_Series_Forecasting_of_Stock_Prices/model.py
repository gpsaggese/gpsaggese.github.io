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
from darts import TimeSeries
from darts.models import TCNModel
from darts.dataprocessing.transformers import Scaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import pickle


@dataclass
class LSTMConfig:
    """Configuration for LSTM model hyperparameters."""
    sequence_length: int = 15
    n_features: int = 13
    lstm_units_1: int = 512
    lstm_units_2: int = 256
    dropout_rate: float = 0.2
    dense_units: int = 128
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
        loss='mae', 
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


@dataclass
class TCNConfig:
    """Configuration for TCN model hyperparameters."""
    input_chunk_length: int = 15
    output_chunk_length: int = 1
    kernel_size: int = 3
    num_filters: int = 64
    dropout: float = 0.2
    n_epochs: int = 50
    dilation_base: int = 2
    weight_norm: bool = False  # Disabled due to PyTorch 2.0 compatibility issue with Darts 0.39.0
    batch_size: int = 32
    learning_rate: float = 0.001
    model_name: str = 'tcn_model'


def prepare_darts_timeseries(
    df: pd.DataFrame,
    target_col: str = 'Close',
    feature_cols: Optional[list] = None,
    date_col: str = 'Date'
) -> Tuple[TimeSeries, Optional[TimeSeries]]:
    """
    Prepare DARTS TimeSeries from DataFrame.

    Args:
        df: Input DataFrame with time series data
        target_col: Target column name
        feature_cols: List of feature columns for covariates
        date_col: Date column name

    Returns:
        Tuple of (target_series, covariate_series)
    """
    target = TimeSeries.from_dataframe(
        df,
        time_col=date_col,
        value_cols=[target_col],
        fill_missing_dates=True,
        freq=None
    )

    covariates = None
    if feature_cols:
        covariates = TimeSeries.from_dataframe(
            df,
            time_col=date_col,
            value_cols=feature_cols,
            fill_missing_dates=True,
            freq=None
        )

    return target, covariates


def build_tcn_model(config: TCNConfig) -> TCNModel:
    """
    Build TCN model.

    Args:
        config: TCNConfig with model hyperparameters

    Returns:
        TCNModel instance
    """
    pl_trainer_kwargs = {
        "accelerator": "auto",
        "devices": 1,
        "num_sanity_val_steps": 0, 
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }

    model = TCNModel(
        input_chunk_length=config.input_chunk_length,
        output_chunk_length=config.output_chunk_length,
        kernel_size=config.kernel_size,
        num_filters=config.num_filters,
        dropout=config.dropout,
        n_epochs=config.n_epochs,
        dilation_base=config.dilation_base,
        weight_norm=config.weight_norm,
        batch_size=config.batch_size,
        model_name=config.model_name,
        force_reset=True,
        save_checkpoints=True,
        random_state=42,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return model


def train_tcn_model(
    target_series: TimeSeries,
    past_covariates: Optional[TimeSeries] = None,
    val_series: Optional[TimeSeries] = None,
    val_covariates: Optional[TimeSeries] = None,
    config: Optional[TCNConfig] = None,
    model_dir: str = 'models',
    verbose: bool = True
) -> TCNModel:
    """
    Train TCN model.

    Args:
        target_series: Target TimeSeries
        past_covariates: Past covariates TimeSeries
        val_series: Validation target TimeSeries
        val_covariates: Validation covariates TimeSeries
        config: TCNConfig with training parameters
        model_dir: Directory to save model
        verbose: Verbosity flag

    Returns:
        Trained TCNModel
    """
    if config is None:
        config = TCNConfig()

    os.makedirs(model_dir, exist_ok=True)

    model = build_tcn_model(config)

    if verbose:
        print(f"Training TCN model with config: {config}")

    def validate_timeseries(ts, name):
        if ts is not None:
            values = ts.values()
            if np.any(np.isnan(values)):
                raise ValueError(f"{name} contains NaN values. Please clean your data.")
            if np.any(np.isinf(values)):
                raise ValueError(f"{name} contains Inf values. Please clean your data.")

    if verbose:
        print("Validating data...")
    validate_timeseries(target_series, "target_series")
    validate_timeseries(past_covariates, "past_covariates")
    validate_timeseries(val_series, "val_series")
    validate_timeseries(val_covariates, "val_covariates")
    if verbose:
        print("Data validation passed.")

    dataloader_kwargs = {
        'num_workers': 0 
    }

    try:
        model.fit(
            series=target_series,
            past_covariates=past_covariates,
            val_series=val_series,
            val_past_covariates=val_covariates,
            dataloader_kwargs=dataloader_kwargs
        )
    except Exception as e:
        print(f"\nERROR: Training failed with error: {type(e).__name__}: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check data shapes and ensure no NaN/Inf values")
        print("2. Try reducing batch_size in TCNConfig")
        print("3. Check GPU memory with nvidia-smi")
        print("4. Consider using CPU by setting accelerator='cpu' in pl_trainer_kwargs")
        raise
    model = TCNModel.load_from_checkpoint(model_name=config.model_name, best=True)
    model_path = os.path.join(model_dir, f'{config.model_name}.pkl')
    model.save(model_path)

    if verbose:
        print(f"TCN model saved to: {model_path}")

    return model

@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model hyperparameters."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


def build_xgboost_model(config: XGBoostConfig) -> Pipeline:
    """
    Build XGBoost model with preprocessing pipeline.

    Args:
        config: XGBoostConfig with model hyperparameters

    Returns:
        Sklearn Pipeline with StandardScaler and XGBRegressor
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            min_child_weight=config.min_child_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            random_state=config.random_state,
            objective='reg:squarederror'
        ))
    ])

    return pipeline


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[XGBoostConfig] = None,
    model_dir: str = 'models',
    verbose: bool = True
) -> Pipeline:
    """
    Train XGBoost model.

    Args:
        X_train: Training features (flattened sequences)
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: XGBoostConfig with training parameters
        model_dir: Directory to save model
        verbose: Verbosity flag

    Returns:
        Trained Pipeline
    """
    if config is None:
        config = XGBoostConfig()

    os.makedirs(model_dir, exist_ok=True)

    if verbose:
        print(f"Building XGBoost model with config: {config}")
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    pipeline = build_xgboost_model(config)

    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]

    if verbose:
        print("Training XGBoost model...")

    pipeline.fit(X_train, y_train)

    model_path = os.path.join(model_dir, 'xgboost_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    if verbose:
        print(f"XGBoost model saved to: {model_path}")

    return pipeline


def flatten_sequences_for_xgboost(X_sequences: np.ndarray) -> np.ndarray:
    """
    Flatten 3D sequences to 2D for XGBoost.

    Args:
        X_sequences: Input sequences (n_samples, sequence_length, n_features)

    Returns:
        Flattened array (n_samples, sequence_length * n_features)
    """
    n_samples = X_sequences.shape[0]
    return X_sequences.reshape(n_samples, -1)
