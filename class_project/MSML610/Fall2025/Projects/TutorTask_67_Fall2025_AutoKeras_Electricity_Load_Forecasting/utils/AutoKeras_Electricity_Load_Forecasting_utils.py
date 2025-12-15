"""
AutoKeras Electricity Load Forecasting Utility Module

This module provides a clean wrapper around AutoKeras StructuredDataRegressor
for time series forecasting tasks, specifically designed for electricity load forecasting.

Author: Course Project
Date: December 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# AutoKeras imports
import autokeras as ak
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class ElectricityDataPreprocessor:
    """
    Handles all data preprocessing tasks for electricity load forecasting.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = None
        
    def load_and_prepare_data(self, filepath: str, datetime_col: str = 'Datetime', 
                              target_col: str = 'Load') -> pd.DataFrame:
        """
        Load and prepare the electricity consumption dataset.
        
        Args:
            filepath: Path to the CSV file
            datetime_col: Name of the datetime column
            target_col: Name of the target variable column
            
        Returns:
            DataFrame with datetime index and cleaned data
        """
        df = pd.read_csv(filepath)
        
        # Convert datetime column to datetime type
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
        df = df.sort_index()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        self.target_column = target_col
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using forward fill and interpolation.
        """
        # First forward fill
        df = df.fillna(method='ffill')
        # Then backward fill for any remaining NaNs
        df = df.fillna(method='bfill')
        # Finally interpolate any remaining gaps
        df = df.interpolate(method='linear')
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from the datetime index.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                           lags: List[int] = [1, 2, 3, 24, 48, 168]) -> pd.DataFrame:
        """
        Create lagged features for time series forecasting.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str,
                               windows: List[int] = [3, 6, 12, 24, 168]) -> pd.DataFrame:
        """
        Create rolling window statistics features.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            windows: List of window sizes for rolling statistics
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str,
                        create_lags: bool = True, create_rolling: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            create_lags: Whether to create lag features
            create_rolling: Whether to create rolling features
            
        Returns:
            DataFrame with all engineered features
        """
        # Create time features
        df = self.create_time_features(df)
        
        # Create lag features
        if create_lags:
            df = self.create_lag_features(df, target_col)
        
        # Create rolling features
        if create_rolling:
            df = self.create_rolling_features(df, target_col)
        
        # Drop rows with NaN values created by lag/rolling operations
        df = df.dropna()
        
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str, 
                   test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """
        Split data into train, validation, and test sets.
        Maintains temporal order for time series.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Calculate split indices
        n = len(df)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Split maintaining temporal order
        X_train = X.iloc[:val_idx]
        X_val = X.iloc[val_idx:test_idx]
        X_test = X.iloc[test_idx:]
        
        y_train = y.iloc[:val_idx]
        y_val = y.iloc[val_idx:test_idx]
        y_test = y.iloc[test_idx:]
        
        self.feature_columns = X.columns.tolist()
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                          X_test: pd.DataFrame) -> Tuple:
        """
        Normalize features using StandardScaler fitted on training data.
        
        Args:
            X_train, X_val, X_test: Feature DataFrames
            
        Returns:
            Tuple of normalized (X_train, X_val, X_test)
        """
        # Fit scaler on training data
        self.scaler.fit(X_train)
        
        # Transform all sets
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_val_scaled, X_test_scaled


class AutoKerasForecaster:
    """
    Wrapper around AutoKeras StructuredDataRegressor for electricity load forecasting.
    """
    
    def __init__(self, max_trials: int = 10, epochs: int = 100, 
                 objective: str = 'val_loss', seed: int = 42):
        """
        Initialize the AutoKeras forecaster.
        
        Args:
            max_trials: Maximum number of different models to try
            epochs: Number of training epochs per trial
            objective: Metric to optimize
            seed: Random seed for reproducibility
        """
        self.max_trials = max_trials
        self.epochs = epochs
        self.objective = objective
        self.seed = seed
        self.model = None
        self.best_model = None
        
    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Build and configure the AutoKeras StructuredDataRegressor.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        self.model = ak.StructuredDataRegressor(
            max_trials=self.max_trials,
            objective=self.objective,
            overwrite=True,
            seed=self.seed
        )
        
        print(f"AutoKeras model initialized with {self.max_trials} max trials")
        print(f"Input shape: {X_train.shape}")
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              verbose: int = 1):
        """
        Train the AutoKeras model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            verbose: Verbosity level
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        print("\n" + "="*50)
        print("Starting AutoKeras Model Training")
        print("="*50)
        print(f"Training samples: {len(X_train)}")
        if validation_data:
            print(f"Validation samples: {len(X_val)}")
        print(f"Max trials: {self.max_trials}")
        print(f"Epochs per trial: {self.epochs}")
        print("="*50 + "\n")
        
        self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            verbose=verbose
        )
        
        # Export the best model
        self.best_model = self.model.export_model()
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        predictions = self.model.predict(X)
        return predictions.flatten()
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on given data.
        
        Args:
            X: Features
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mape = mean_absolute_percentage_error(y, predictions) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        return metrics, predictions


class BaselineModels:
    """
    Simple baseline models for comparison with AutoKeras.
    """
    
    @staticmethod
    def naive_forecast(y_train: pd.Series, n_steps: int) -> np.ndarray:
        """
        Naive forecast: repeat the last known value.
        """
        last_value = y_train.iloc[-1]
        return np.array([last_value] * n_steps)
    
    @staticmethod
    def seasonal_naive_forecast(y_train: pd.Series, n_steps: int, 
                               season_length: int = 24) -> np.ndarray:
        """
        Seasonal naive forecast: repeat the last season.
        """
        last_season = y_train.iloc[-season_length:].values
        forecasts = []
        
        for i in range(n_steps):
            forecasts.append(last_season[i % season_length])
        
        return np.array(forecasts)
    
    @staticmethod
    def moving_average_forecast(y_train: pd.Series, n_steps: int, 
                               window: int = 24) -> np.ndarray:
        """
        Moving average forecast.
        """
        last_values = y_train.iloc[-window:].values
        mean_value = np.mean(last_values)
        return np.array([mean_value] * n_steps)


class ForecastVisualizer:
    """
    Visualization utilities for forecasting results.
    """
    
    @staticmethod
    def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, 
                        title: str = "Actual vs Predicted", 
                        figsize: Tuple[int, int] = (15, 6)):
        """
        Plot actual vs predicted values.
        """
        plt.figure(figsize=figsize)
        plt.plot(y_true.index, y_true.values, label='Actual', alpha=0.7, linewidth=2)
        plt.plot(y_true.index, y_pred, label='Predicted', alpha=0.7, linewidth=2)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Load', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_error_distribution(y_true: pd.Series, y_pred: np.ndarray,
                               figsize: Tuple[int, int] = (12, 5)):
        """
        Plot error distribution and residuals.
        """
        errors = y_true.values - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Error distribution histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Prediction Error', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        axes[1].scatter(y_pred, errors, alpha=0.5, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Values', fontsize=11)
        axes[1].set_ylabel('Residuals', fontsize=11)
        axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                               figsize: Tuple[int, int] = (12, 6)):
        """
        Plot comparison of metrics across different models.
        """
        models = list(metrics_dict.keys())
        metrics = ['MAE', 'RMSE', 'MAPE']
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for idx, metric in enumerate(metrics):
            values = [metrics_dict[model][metric] for model in models]
            axes[idx].bar(models, values, alpha=0.7, edgecolor='black')
            axes[idx].set_ylabel(metric, fontsize=11)
            axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance(model, feature_names: List[str], 
                               top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance (if available from the model).
        Note: AutoKeras models may not always provide feature importance directly.
        """
        try:
            # This is a placeholder - actual implementation depends on model structure
            plt.figure(figsize=figsize)
            plt.text(0.5, 0.5, 'Feature importance not directly available from AutoKeras\nConsider using SHAP values for interpretation', 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
            return plt.gcf()
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
            return None


def print_evaluation_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print evaluation metrics in a formatted way.
    """
    print("\n" + "="*50)
    print(f"{model_name} Performance Metrics")
    print("="*50)
    for metric_name, value in metrics.items():
        if metric_name == 'MAPE':
            print(f"{metric_name:15s}: {value:.2f}%")
        else:
            print(f"{metric_name:15s}: {value:.2f}")
    print("="*50 + "\n")
