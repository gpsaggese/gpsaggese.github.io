"""Hybrid LSTM-XGBoost Ensemble Model for IoT Anomaly Detection.

This script implements a sophisticated hybrid deep learning approach combining:
1. LSTM (Long Short-Term Memory) neural network for temporal pattern learning
2. XGBoost for gradient boosting on engineered features
3. Ensemble layer combining both models for superior performance

The hybrid approach leverages:
- LSTM's ability to capture long-term temporal dependencies in time-series data
- XGBoost's power in handling tabular features and non-linear relationships
- Ensemble method to combine strengths of both architectures

Architecture:
    Input Features
         |
    ┌────┴────┐
    |         |
  LSTM    XGBoost
    |         |
    └────┬────┘
         |
   Meta-Learner
         |
    Predictions

Example usage:
    python train_lstm_xgb_ensemble.py --input data/smart_manufacturing_data.csv \
        --output_dir results --epochs 50 --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score,
                             classification_report)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import compute_features, SENSOR_COLUMNS

warnings.filterwarnings('ignore')


class LSTMModel(nn.Module):
    """LSTM neural network for time-series anomaly detection.

    Architecture:
        - 2 LSTM layers with dropout
        - Fully connected layers
        - Output layer for classification/regression
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 1,
                 dropout: float = 0.3, task: str = 'classification'):
        """
        Parameters
        ----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of LSTM hidden units
        num_layers : int
            Number of LSTM layers
        output_size : int
            Number of output classes (for classification) or 1 (for regression)
        dropout : float
            Dropout probability
        task : str
            'classification' or 'regression'
        """
        super(LSTMModel, self).__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task = task

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        if task == 'classification' and output_size > 1:
            self.output_activation = nn.Softmax(dim=1)
        elif task == 'classification' and output_size == 1:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        last_hidden = h_n[-1]  # shape: (batch_size, hidden_size)

        # Fully connected layers
        out = self.dropout(last_hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.output_activation(out)

        return out


class HybridLSTMXGBoost:
    """Hybrid model combining LSTM and XGBoost predictions.

    This ensemble uses a meta-learning approach where predictions from both
    LSTM and XGBoost are combined using a weighted average or a simple
    logistic regression meta-learner.
    """

    def __init__(self, input_size: int, task: str = 'classification',
                 output_size: int = 1, lstm_hidden: int = 128,
                 lstm_layers: int = 2, dropout: float = 0.3,
                 device: str = 'cpu'):
        """
        Parameters
        ----------
        input_size : int
            Number of input features
        task : str
            'classification' or 'regression'
        output_size : int
            Number of output classes or 1 for regression
        lstm_hidden : int
            LSTM hidden size
        lstm_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate
        device : str
            'cpu' or 'cuda'
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        self.task = task
        self.output_size = output_size
        self.device = device

        # Initialize LSTM model
        self.lstm_model = LSTMModel(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            output_size=output_size,
            dropout=dropout,
            task=task
        ).to(device)

        # Initialize XGBoost model
        if task == 'classification':
            if output_size == 1:
                self.xgb_model = XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
            else:
                self.xgb_model = XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='mlogloss',
                    use_label_encoder=False
                )
        else:
            self.xgb_model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )

        self.scaler = StandardScaler()
        self.is_fitted = False

    def create_sequences(self, X: np.ndarray, seq_length: int = 10) -> np.ndarray:
        """Create sequences from feature data for LSTM input.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        seq_length : int
            Length of sequences

        Returns
        -------
        np.ndarray
            Sequences (n_samples - seq_length + 1, seq_length, n_features)
        """
        sequences = []
        for i in range(len(X) - seq_length + 1):
            sequences.append(X[i:i + seq_length])
        return np.array(sequences)

    def train_lstm(self, X_seq: np.ndarray, y: np.ndarray,
                   epochs: int = 50, batch_size: int = 64,
                   learning_rate: float = 0.001, validation_split: float = 0.2):
        """Train the LSTM component.

        Parameters
        ----------
        X_seq : np.ndarray
            Sequence data (n_samples, seq_length, n_features)
        y : np.ndarray
            Target labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        validation_split : float
            Fraction of data for validation
        """
        # Split data
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        if self.task == 'classification' and self.output_size == 1:
            y_train_t = y_train_t.unsqueeze(1)
            y_val_t = y_val_t.unsqueeze(1)

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        if self.task == 'classification':
            if self.output_size == 1:
                criterion = nn.BCELoss()
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.Adam(self.lstm_model.parameters(), lr=learning_rate)

        # Training loop
        print("  Training LSTM...")
        self.lstm_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)

                if self.task == 'classification' and self.output_size > 1:
                    batch_y = batch_y.long()

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"    Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    def fit(self, X: np.ndarray, y: np.ndarray, seq_length: int = 10,
            epochs: int = 50, batch_size: int = 64):
        """Fit the hybrid model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        seq_length : int
            Sequence length for LSTM
        epochs : int
            Number of epochs for LSTM training
        batch_size : int
            Batch size for LSTM training
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train XGBoost on full features
        print("  Training XGBoost...")
        self.xgb_model.fit(X_scaled, y)

        # Create sequences for LSTM
        X_seq = self.create_sequences(X_scaled, seq_length)
        # Adjust y to match sequence length
        y_seq = y[seq_length - 1:]

        # Train LSTM
        self.train_lstm(X_seq, y_seq, epochs, batch_size)

        self.seq_length = seq_length
        self.is_fitted = True

    def predict_lstm(self, X: np.ndarray) -> np.ndarray:
        """Get LSTM predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            LSTM predictions
        """
        X_scaled = self.scaler.transform(X)
        X_seq = self.create_sequences(X_scaled, self.seq_length)

        self.lstm_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            predictions = self.lstm_model(X_tensor).cpu().numpy()

        if self.task == 'classification' and self.output_size == 1:
            predictions = (predictions > 0.5).astype(int).flatten()
        elif self.task == 'classification':
            predictions = np.argmax(predictions, axis=1)

        return predictions

    def predict_xgb(self, X: np.ndarray) -> np.ndarray:
        """Get XGBoost predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            XGBoost predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.xgb_model.predict(X_scaled)

    def predict(self, X: np.ndarray, ensemble_weight: float = 0.5) -> np.ndarray:
        """Make ensemble predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        ensemble_weight : float
            Weight for LSTM (0 to 1), XGBoost gets (1 - weight)

        Returns
        -------
        np.ndarray
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Get predictions from both models
        lstm_preds = self.predict_lstm(X)
        xgb_preds = self.predict_xgb(X[self.seq_length - 1:])

        # Ensemble predictions (simple weighted average)
        if self.task == 'classification':
            # Use voting for classification
            ensemble_preds = np.zeros(len(lstm_preds))
            for i in range(len(lstm_preds)):
                if ensemble_weight * lstm_preds[i] + (1 - ensemble_weight) * xgb_preds[i] > 0.5:
                    ensemble_preds[i] = 1
                else:
                    ensemble_preds[i] = 0
        else:
            # Weighted average for regression
            ensemble_preds = ensemble_weight * lstm_preds + (1 - ensemble_weight) * xgb_preds

        return ensemble_preds.astype(int if self.task == 'classification' else float)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features and return a cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data

    Returns
    -------
    pd.DataFrame
        DataFrame with features and labels
    """
    print("Computing engineered features...")
    features_df = compute_features(df)

    original_cols = ["machine_id", "anomaly_flag", "maintenance_required",
                     "downtime_risk", "predicted_remaining_life", "failure_type"]
    feature_only_df = features_df.drop(
        columns=[c for c in original_cols if c in features_df.columns],
        errors='ignore'
    )

    merged = pd.concat([df[original_cols], feature_only_df], axis=1)
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna()
    print(f"Features computed. Shape: {merged.shape}")
    return merged


def main(args: argparse.Namespace) -> None:
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not installed. Please install with: pip install torch")
        return
    if not XGBOOST_AVAILABLE:
        print("ERROR: XGBoost is not installed. Please install with: pip install xgboost")
        return

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")

    print("="*80)
    print("Hybrid LSTM-XGBoost Ensemble Model Training")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print("="*80)

    # Load data
    print("\nLoading dataset...")
    df_raw = pd.read_csv(input_path)
    print(f"Dataset loaded. Shape: {df_raw.shape}")

    if args.sample > 0 and len(df_raw) > args.sample:
        df_raw = df_raw.sample(n=args.sample, random_state=42)
        print(f"Sampled to {args.sample} rows")

    # Prepare data
    df_feat = prepare_data(df_raw)

    # Separate features and labels
    target_cols = ["anomaly_flag", "maintenance_required", "downtime_risk",
                  "predicted_remaining_life", "failure_type"]
    feature_cols = [c for c in df_feat.columns if c not in target_cols + ["machine_id"]]
    X_all = df_feat[feature_cols].to_numpy(dtype=float)

    results = {}

    # Task 1: Anomaly Detection (Binary Classification)
    print("\n" + "="*80)
    print("TASK 1: Anomaly Detection (Binary Classification)")
    print("="*80)
    y_anomaly = df_feat["anomaly_flag"].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_anomaly, test_size=0.2, random_state=42, stratify=y_anomaly
    )

    model_anomaly = HybridLSTMXGBoost(
        input_size=X_train.shape[1],
        task='classification',
        output_size=1,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        device=device
    )

    model_anomaly.fit(X_train, y_train, seq_length=args.seq_length,
                     epochs=args.epochs, batch_size=args.batch_size)

    # Predictions
    y_pred = model_anomaly.predict(X_test)
    y_test_aligned = y_test[args.seq_length - 1:]

    # Metrics
    f1 = f1_score(y_test_aligned, y_pred)
    precision = precision_score(y_test_aligned, y_pred, zero_division=0)
    recall = recall_score(y_test_aligned, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test_aligned, y_pred)

    results['anomaly_detection'] = {
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy)
    }

    print(f"\nResults:")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(model_anomaly, models_dir / "hybrid_lstm_xgb_anomaly.pkl")
    print(f"  Model saved to {models_dir / 'hybrid_lstm_xgb_anomaly.pkl'}")

    # Task 2: Remaining Useful Life Prediction (Regression)
    print("\n" + "="*80)
    print("TASK 2: Remaining Useful Life Prediction (Regression)")
    print("="*80)
    y_rul = df_feat["predicted_remaining_life"].astype(float).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_rul, test_size=0.2, random_state=42
    )

    model_rul = HybridLSTMXGBoost(
        input_size=X_train.shape[1],
        task='regression',
        output_size=1,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        device=device
    )

    model_rul.fit(X_train, y_train, seq_length=args.seq_length,
                 epochs=args.epochs, batch_size=args.batch_size)

    # Predictions
    y_pred = model_rul.predict(X_test)
    y_test_aligned = y_test[args.seq_length - 1:]

    # Metrics
    mae = mean_absolute_error(y_test_aligned, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_aligned, y_pred))
    r2 = r2_score(y_test_aligned, y_pred)

    results['rul_prediction'] = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2)
    }

    print(f"\nResults:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}")

    # Save model
    joblib.dump(model_rul, models_dir / "hybrid_lstm_xgb_rul.pkl")
    print(f"  Model saved to {models_dir / 'hybrid_lstm_xgb_rul.pkl'}")

    # Save results
    results_path = output_dir / "hybrid_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Results saved to: {results_path}")
    print(f"Models saved to: {models_dir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train hybrid LSTM-XGBoost ensemble model for IoT anomaly detection"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--sample", type=int, default=20000,
                       help="Number of samples (0 for all)")
    parser.add_argument("--seq_length", type=int, default=10,
                       help="Sequence length for LSTM")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--lstm_hidden", type=int, default=128,
                       help="LSTM hidden size")
    parser.add_argument("--lstm_layers", type=int, default=2,
                       help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--cpu", action='store_true',
                       help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    main(args)
