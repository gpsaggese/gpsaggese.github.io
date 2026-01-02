"""
LSTM model for temporal economic prediction.

This model provides:
- Temporal dependency modeling
- Sequence-to-value prediction
- Deep learning baseline for causal analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging

from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """
    LSTM (Long Short-Term Memory) model for time series prediction.
    
    Captures temporal dependencies in economic data for predicting
    outcomes like wage growth based on historical indicators.
    
    Example:
        >>> model = LSTMModel(sequence_length=12, lstm_units=64)
        >>> model.fit(X_train, y_train, epochs=50)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        sequence_length: int = 12,
        lstm_units: int = 64,
        lstm_layers: int = 2,
        dropout_rate: float = 0.2,
        dense_units: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42,
        name: str = "LSTM"
    ):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps in input sequence
            lstm_units: Number of units in LSTM layers
            lstm_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            dense_units: Units in dense layer before output
            learning_rate: Learning rate for Adam optimizer
            random_state: Random seed
            name: Model name
        """
        super().__init__(name=name)
        
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.history = None
        self.n_features = None
        self.scaler_X = None
        self.scaler_y = None
        
    def _build_model(self, n_features: int) -> Any:
        """
        Build the LSTM architecture.
        
        Args:
            n_features: Number of input features
            
        Returns:
            Compiled Keras model
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            import tensorflow as tf
            
            # Set random seed
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)
            
        except ImportError:
            raise ImportError(
                "TensorFlow not installed. Run: pip install tensorflow"
            )
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.lstm_units,
            return_sequences=(self.lstm_layers > 1),
            input_shape=(self.sequence_length, n_features)
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, self.lstm_layers):
            return_seq = (i < self.lstm_layers - 1)
            model.add(LSTM(self.lstm_units // (2 ** i), return_sequences=return_seq))
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        if self.dense_units > 0:
            model.add(Dense(self.dense_units, activation='relu'))
            model.add(Dropout(self.dropout_rate / 2))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        features: List[str],
        target: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for LSTM training.
        
        Args:
            data: DataFrame with time series data
            features: List of feature column names
            target: Target column name
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        feature_data = data[features].values
        target_data = data[target].values
        
        # Remove missing values
        mask = ~(np.isnan(feature_data).any(axis=1) | np.isnan(target_data))
        feature_data = feature_data[mask]
        target_data = target_data[mask]
        
        if len(feature_data) < self.sequence_length + 1:
            raise ValueError(
                f"Not enough data ({len(feature_data)}) for sequence length {self.sequence_length}"
            )
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(feature_data)):
            X_seq.append(feature_data[i - self.sequence_length:i])
            y_seq.append(target_data[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1,
        **kwargs
    ) -> 'LSTMModel':
        """
        Train the LSTM model.
        
        Args:
            X: Training sequences (n_samples, sequence_length, n_features)
            y: Training targets (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
            verbose: Verbosity level (0, 1, or 2)
            **kwargs: Additional parameters for model.fit()
            
        Returns:
            self: Fitted model
        """
        logger.info(f"Training {self.name} with {len(X)} sequences")
        
        # Validate input shape
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input (samples, timesteps, features), got shape {X.shape}"
            )
        
        self.n_features = X.shape[2]
        
        # Build model
        self.model = self._build_model(self.n_features)
        
        # Train
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            **kwargs
        )
        
        self.is_fitted = True
        logger.info(f"{self.name} training complete")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained LSTM.
        
        Args:
            X: Input sequences (n_samples, sequence_length, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns:
            Dictionary with loss and metrics over epochs
        """
        if self.history is None:
            return {}
        
        return {
            'loss': self.history.history.get('loss', []),
            'val_loss': self.history.history.get('val_loss', []),
            'mae': self.history.history.get('mae', []),
            'val_mae': self.history.history.get('val_mae', [])
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'lstm_layers': self.lstm_layers,
            'dropout_rate': self.dropout_rate,
            'dense_units': self.dense_units,
            'learning_rate': self.learning_rate
        }
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'LSTMModel':
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: Model with loaded weights
        """
        from tensorflow.keras.models import load_model
        
        self.model = load_model(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
        
        return self
    
    def summary(self) -> str:
        """Return model architecture summary."""
        if self.model is None:
            return f"LSTM Model (not built)\nParams: {self.get_params()}"
        
        # Capture model summary
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
