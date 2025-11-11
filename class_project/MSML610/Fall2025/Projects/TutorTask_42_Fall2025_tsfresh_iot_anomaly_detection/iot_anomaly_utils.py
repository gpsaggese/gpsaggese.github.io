"""IoT Anomaly Detection - Utilities Module

This module provides a clean API for the IoT anomaly detection system.
It wraps all the functionality needed for:
- Data loading and preprocessing
- Feature engineering
- Model training and evaluation
- Predictions
- Visualization

This is the main interface that notebooks and external code should use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from feature_engineering import compute_features, SENSOR_COLUMNS

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class IoTAnomalyDetector:
    """Main class for IoT anomaly detection pipeline.

    This class provides a high-level API for:
    - Loading and preprocessing IoT sensor data
    - Engineering features
    - Training models
    - Making predictions
    - Evaluating performance

    Example:
        >>> detector = IoTAnomalyDetector()
        >>> detector.load_data("data/smart_manufacturing_data.csv")
        >>> detector.prepare_features()
        >>> detector.load_model("models/best_anomaly_model.pkl")
        >>> predictions = detector.predict(detector.X_test)
    """

    def __init__(self):
        """Initialize the IoT anomaly detector."""
        self.df_raw = None
        self.df_features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoder = None
        self.model = None
        self.feature_names = None

    def load_data(self, filepath: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load raw IoT sensor data from CSV.

        Parameters
        ----------
        filepath : str
            Path to CSV file
        sample_size : int, optional
            Number of rows to sample (for faster prototyping)

        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        print(f"Loading data from {filepath}...")
        self.df_raw = pd.read_csv(filepath)

        if sample_size and len(self.df_raw) > sample_size:
            self.df_raw = self.df_raw.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} rows")

        print(f"Data loaded. Shape: {self.df_raw.shape}")
        return self.df_raw

    def prepare_features(self, target_col: str = "anomaly_flag") -> pd.DataFrame:
        """Compute engineered features from raw data.

        Parameters
        ----------
        target_col : str
            Name of target column

        Returns
        -------
        pd.DataFrame
            DataFrame with engineered features
        """
        if self.df_raw is None:
            raise ValueError("Load data first using load_data()")

        print("Computing features...")
        self.df_features = compute_features(self.df_raw)

        # Merge with original targets
        original_cols = ["machine_id", "anomaly_flag", "maintenance_required",
                        "downtime_risk", "predicted_remaining_life", "failure_type"]

        feature_only_df = self.df_features.drop(
            columns=[c for c in original_cols if c in self.df_features.columns],
            errors='ignore'
        )

        self.df_features = pd.concat([
            self.df_raw[original_cols],
            feature_only_df
        ], axis=1)

        # Clean data
        self.df_features = self.df_features.replace([np.inf, -np.inf], np.nan)
        self.df_features = self.df_features.dropna()

        print(f"Features computed. Shape: {self.df_features.shape}")
        print(f"Number of features: {len([c for c in self.df_features.columns if c not in original_cols])}")

        return self.df_features

    def prepare_train_test_split(self, target_col: str = "anomaly_flag",
                                 test_size: float = 0.2,
                                 scale: bool = True,
                                 stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/test sets.

        Parameters
        ----------
        target_col : str
            Name of target column
        test_size : float
            Fraction of data for testing
        scale : bool
            Whether to standardize features
        stratify : bool
            Whether to stratify split (for classification)

        Returns
        -------
        Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.df_features is None:
            raise ValueError("Prepare features first using prepare_features()")

        # Get target
        target_cols = ["anomaly_flag", "maintenance_required", "downtime_risk",
                      "predicted_remaining_life", "failure_type"]

        if target_col == "failure_type":
            # Encode categorical target
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(self.df_features[target_col].astype(str))
        elif target_col == "downtime_risk":
            y = (self.df_features[target_col] > 0).astype(int).to_numpy()
        else:
            y = self.df_features[target_col].to_numpy()

        # Get features
        self.feature_names = [c for c in self.df_features.columns
                             if c not in target_cols + ["machine_id"]]
        X = self.df_features[self.feature_names].to_numpy(dtype=float)

        # Split
        stratify_array = y if stratify and target_col != "predicted_remaining_life" else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_array
        )

        # Scale if requested
        if scale:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

        print(f"Train/test split: {len(self.X_train)} / {len(self.X_test)}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def load_model(self, model_path: str):
        """Load a trained model from disk.

        Parameters
        ----------
        model_path : str
            Path to saved model (.pkl file)
        """
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print("Model loaded successfully")

    def save_model(self, model_path: str):
        """Save the current model to disk.

        Parameters
        ----------
        model_path : str
            Path to save model (.pkl file)
        """
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the loaded model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Predictions
        """
        if self.model is None:
            raise ValueError("Load a model first using load_model()")
        return self.model.predict(X)

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                               labels: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate classification performance.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        labels : list, optional
            Class labels for display

        Returns
        -------
        dict
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        print("Classification Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

        return metrics

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression performance.

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        dict
            Dictionary of metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }

        print("Regression Metrics:")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")

        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             labels: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """Plot confusion matrix.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        labels : list, optional
            Class labels
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        return fig

    def plot_feature_importance(self, top_n: int = 20,
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot feature importance from tree-based model.

        Parameters
        ----------
        top_n : int
            Number of top features to show
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        if self.model is None:
            raise ValueError("Load a model first")

        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(top_n), importances[indices])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.invert_yaxis()

        return fig


def load_results(results_dir: str = "results") -> Dict[str, pd.DataFrame]:
    """Load all result CSV files from training.

    Parameters
    ----------
    results_dir : str
        Directory containing result CSVs

    Returns
    -------
    dict
        Dictionary mapping task name to results DataFrame
    """
    results_dir = Path(results_dir)
    results = {}

    result_files = {
        'anomaly': 'results_anomaly.csv',
        'maintenance': 'results_maintenance.csv',
        'downtime': 'results_downtime.csv',
        'failure_type': 'results_failure_type.csv',
        'remaining_life': 'results_remaining_life.csv'
    }

    for task, filename in result_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            results[task] = pd.read_csv(filepath)
            print(f"Loaded {task} results: {len(results[task])} models")
        else:
            print(f"Warning: {filepath} not found")

    return results


def plot_model_comparison(results: pd.DataFrame, metric: str = 'f1_mean',
                          title: str = 'Model Comparison',
                          figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """Plot comparison of model performances.

    Parameters
    ----------
    results : pd.DataFrame
        Results DataFrame from training
    metric : str
        Metric to plot
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by metric
    results_sorted = results.sort_values(metric, ascending=True)

    # Plot horizontal bar chart
    ax.barh(results_sorted['model'], results_sorted[metric])
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(results_sorted.iterrows()):
        ax.text(row[metric], i, f' {row[metric]:.4f}',
               va='center', fontsize=9)

    plt.tight_layout()
    return fig


def create_predictions_dataframe(detector: IoTAnomalyDetector,
                                 task_name: str = "anomaly") -> pd.DataFrame:
    """Create a DataFrame with predictions and metadata.

    Parameters
    ----------
    detector : IoTAnomalyDetector
        Fitted detector instance
    task_name : str
        Name of the prediction task

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions
    """
    if detector.model is None or detector.X_test is None:
        raise ValueError("Model and test data must be available")

    predictions = detector.predict(detector.X_test)

    df_predictions = pd.DataFrame({
        'true_label': detector.y_test,
        'predicted_label': predictions,
        'task': task_name
    })

    # Add correctness flag for classification
    if task_name != "remaining_life":
        df_predictions['correct'] = (df_predictions['true_label'] ==
                                     df_predictions['predicted_label'])

    return df_predictions


def generate_summary_report(results_dir: str = "results",
                           output_file: str = "summary_report.txt"):
    """Generate a text summary report of all results.

    Parameters
    ----------
    results_dir : str
        Directory containing results
    output_file : str
        Output file path
    """
    results = load_results(results_dir)

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("IoT ANOMALY DETECTION - MODEL PERFORMANCE SUMMARY\n")
        f.write("="*80 + "\n\n")

        for task, df in results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Task: {task.upper()}\n")
            f.write(f"{'='*80}\n\n")

            # Find best model
            if 'f1_mean' in df.columns:
                best_idx = df['f1_mean'].idxmax()
                metric_col = 'f1_mean'
            elif 'r2_mean' in df.columns:
                best_idx = df['r2_mean'].idxmax()
                metric_col = 'r2_mean'
            else:
                continue

            best_model = df.loc[best_idx]
            f.write(f"Best Model: {best_model['model']}\n")
            f.write(f"Best {metric_col}: {best_model[metric_col]:.4f}\n\n")

            f.write("All Models:\n")
            f.write("-" * 80 + "\n")

            # Sort and display
            df_sorted = df.sort_values(metric_col, ascending=False)
            for idx, row in df_sorted.iterrows():
                f.write(f"  {row['model']:30s} {metric_col}: {row[metric_col]:.4f}\n")

            f.write("\n")

    print(f"Summary report saved to {output_file}")


# Convenience function for quick predictions
def quick_predict(data_path: str, model_path: str,
                 target_col: str = "anomaly_flag") -> np.ndarray:
    """Quick prediction on new data.

    Parameters
    ----------
    data_path : str
        Path to CSV data
    model_path : str
        Path to saved model
    target_col : str
        Target column name (for feature preparation)

    Returns
    -------
    np.ndarray
        Predictions
    """
    detector = IoTAnomalyDetector()
    detector.load_data(data_path)
    detector.prepare_features(target_col)
    detector.prepare_train_test_split(target_col, test_size=0.2, scale=True)
    detector.load_model(model_path)

    predictions = detector.predict(detector.X_test)
    return predictions


if __name__ == "__main__":
    print("IoT Anomaly Detection Utils Module")
    print("===================================")
    print("\nThis module provides utilities for IoT anomaly detection.")
    print("\nMain classes:")
    print("  - IoTAnomalyDetector: Main pipeline class")
    print("\nMain functions:")
    print("  - load_results(): Load training results")
    print("  - plot_model_comparison(): Compare model performances")
    print("  - generate_summary_report(): Create text report")
    print("  - quick_predict(): Make predictions on new data")
