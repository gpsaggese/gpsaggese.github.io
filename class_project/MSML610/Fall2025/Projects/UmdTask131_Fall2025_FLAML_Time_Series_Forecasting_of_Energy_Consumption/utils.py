"""
Utility Functions for Energy Consumption Forecasting with FLAML

This module provides reusable functions for:
- Data loading and preprocessing
- Feature engineering (temporal, lag, rolling, EMA)
- Model training and evaluation
- Visualization
- Results saving

Author: Anisha Katiyar
Course: MSML610 - Advanced Machine Learning
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    r2_score
)
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_energy_data(filepath: str, sep: str = ';') -> pd.DataFrame:
    """
    Load the UCI Household Electric Power Consumption dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the data file
    sep : str, default ';'
        Delimiter used in the file
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with datetime index
    """
    df = pd.read_csv(
        filepath,
        sep=sep,
        parse_dates={'datetime': ['Date', 'Time']},
        dayfirst=True,
        low_memory=False,
        na_values=['?']
    )
    return df


def clean_and_resample(
    df: pd.DataFrame, 
    target_col: str = 'Global_active_power',
    freq: str = 'D',
    agg_func: str = 'mean'
) -> pd.DataFrame:
    """
    Clean data and resample to specified frequency.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index
    target_col : str, default 'Global_active_power'
        Target column to keep
    freq : str, default 'D'
        Resampling frequency ('D' for daily, 'W' for weekly, 'H' for hourly)
    agg_func : str, default 'mean'
        Aggregation function for resampling
        
    Returns
    -------
    pd.DataFrame
        Cleaned and resampled DataFrame
    """
    df_clean = df.set_index('datetime')[[target_col]]
    df_resampled = df_clean.resample(freq).agg(agg_func)
    
    # Handle missing values with hybrid imputation
    df_resampled = df_resampled.ffill().bfill().interpolate(method='linear')
    
    return df_resampled


def remove_outliers(
    df: pd.DataFrame, 
    column: str, 
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column to check for outliers
    method : str, default 'iqr'
        Method for outlier detection ('iqr' or 'zscore')
    threshold : float, default 1.5
        Threshold for outlier detection (1.5 for IQR, 3 for z-score)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with outliers removed
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        mask = z_scores < threshold
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df[mask].copy()


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features from datetime index.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Basic temporal features
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['week_of_year'] = df.index.isocalendar().week.values
    df['day_of_year'] = df.index.dayofyear
    
    # Cyclical encoding for periodic features
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Calendar features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    
    # Season feature
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    
    df['season'] = df['month'].apply(get_season)
    
    return df


def add_lag_features(
    df: pd.DataFrame, 
    column: str, 
    lags: List[int]
) -> pd.DataFrame:
    """
    Add lag features for a column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column to create lags for
    lags : List[int]
        List of lag periods
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added lag features
    """
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[column].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, 
    column: str, 
    windows: List[int]
) -> pd.DataFrame:
    """
    Add rolling statistics features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column to compute rolling stats for
    windows : List[int]
        List of rolling window sizes
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added rolling features
    """
    df = df.copy()
    for window in windows:
        # Shift by 1 to prevent data leakage
        shifted = df[column].shift(1)
        df[f'rolling_mean_{window}'] = shifted.rolling(window=window).mean()
        df[f'rolling_std_{window}'] = shifted.rolling(window=window).std()
        df[f'rolling_min_{window}'] = shifted.rolling(window=window).min()
        df[f'rolling_max_{window}'] = shifted.rolling(window=window).max()
    return df


def add_ema_features(
    df: pd.DataFrame, 
    column: str, 
    spans: List[int]
) -> pd.DataFrame:
    """
    Add exponential moving average features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column to compute EMA for
    spans : List[int]
        List of EMA spans
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added EMA features
    """
    df = df.copy()
    for span in spans:
        # Shift by 1 to prevent data leakage
        df[f'ema_{span}'] = df[column].shift(1).ewm(span=span, adjust=False).mean()
    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'Global_active_power',
    lag_periods: List[int] = [1, 2, 3, 7, 14, 30],
    rolling_windows: List[int] = [7, 14, 30],
    ema_spans: List[int] = [7, 30]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete feature engineering pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index and target column
    target_col : str, default 'Global_active_power'
        Target column name
    lag_periods : List[int]
        Lag periods for lag features
    rolling_windows : List[int]
        Window sizes for rolling features
    ema_spans : List[int]
        Spans for EMA features
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Processed DataFrame and list of feature column names
    """
    # Apply all feature engineering
    df = add_temporal_features(df)
    df = add_lag_features(df, target_col, lag_periods)
    df = add_rolling_features(df, target_col, rolling_windows)
    df = add_ema_features(df, target_col, ema_spans)
    
    # Add difference features
    df['diff_1'] = df[target_col].diff(1)
    df['diff_7'] = df[target_col].diff(7)
    
    # Drop NaN rows
    df = df.dropna()
    
    # Get feature columns (everything except target and datetime)
    feature_cols = [col for col in df.columns if col != target_col]
    
    return df, feature_cols


# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

def temporal_train_test_split(
    df: pd.DataFrame,
    target_col: str = 'y',
    test_size: float = 0.2,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Perform temporal train-test split for time-series data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str, default 'y'
        Target column name
    test_size : float, default 0.2
        Proportion of data for test set
    feature_cols : Optional[List[str]]
        List of feature columns (if None, auto-detect)
        
    Returns
    -------
    Tuple containing:
        - train_data: Full training DataFrame
        - test_data: Full test DataFrame
        - X_train: Training features
        - X_test: Test features
        - y_train: Training target
        - y_test: Test target
        - feature_cols: List of feature column names
    """
    split_idx = int(len(df) * (1 - test_size))
    
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()
    
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ['ds', target_col, 'datetime']]
    
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    y_train = train_data[target_col]
    y_test = test_data[target_col]
    
    return train_data, test_data, X_train, X_test, y_train, y_test, feature_cols


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_flaml_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    time_budget: int = 600,
    metric: str = 'rmse',
    estimator_list: Optional[List[str]] = None,
    seed: int = 42,
    verbose: int = 0
):
    """
    Train a FLAML AutoML model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    time_budget : int, default 600
        Time budget in seconds
    metric : str, default 'rmse'
        Optimization metric
    estimator_list : Optional[List[str]]
        List of estimators to try
    seed : int, default 42
        Random seed
    verbose : int, default 0
        Verbosity level
        
    Returns
    -------
    AutoML
        Trained FLAML model
    """
    from flaml import AutoML
    
    if estimator_list is None:
        estimator_list = ['lgbm', 'xgboost', 'rf', 'extra_tree']
    
    automl = AutoML()
    automl.fit(
        X_train, y_train,
        task='regression',
        metric=metric,
        time_budget=time_budget,
        estimator_list=estimator_list,
        seed=seed,
        verbose=verbose
    )
    
    return automl


def train_prophet_model(
    train_data: pd.DataFrame,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    seasonality_mode: str = 'multiplicative'
):
    """
    Train a Facebook Prophet model.
    
    Parameters
    ----------
    train_data : pd.DataFrame
        Training data with 'ds' and 'y' columns
    yearly_seasonality : bool, default True
        Include yearly seasonality
    weekly_seasonality : bool, default True
        Include weekly seasonality
    daily_seasonality : bool, default False
        Include daily seasonality
    seasonality_mode : str, default 'multiplicative'
        Seasonality mode ('additive' or 'multiplicative')
        
    Returns
    -------
    Prophet
        Trained Prophet model
    """
    from prophet import Prophet
    
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95
    )
    
    model.fit(train_data[['ds', 'y']])
    
    return model


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    Dict[str, float]
        Dictionary with RMSE, MAE, MAPE, R2 metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'R2': r2_score(y_true, y_pred)
    }


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str = 'flaml'
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a trained model on train and test sets.
    
    Parameters
    ----------
    model : trained model
        FLAML AutoML, Prophet, or similar model
    X_train, X_test : pd.DataFrame
        Train and test features
    y_train, y_test : pd.Series
        Train and test targets
    model_type : str, default 'flaml'
        Type of model ('flaml' or 'prophet')
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with train and test metrics
    """
    if model_type == 'flaml':
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
    elif model_type == 'prophet':
        train_forecast = model.predict(X_train[['ds']])
        test_forecast = model.predict(X_test[['ds']])
        train_pred = train_forecast['yhat'].values
        test_pred = test_forecast['yhat'].values
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return {
        'train': calculate_metrics(y_train, train_pred),
        'test': calculate_metrics(y_test, test_pred),
        'train_pred': train_pred,
        'test_pred': test_pred
    }


# =============================================================================
# ENSEMBLE METHODS
# =============================================================================

def create_ensemble_predictions(
    pred1: np.ndarray,
    pred2: np.ndarray,
    weight1: float = 0.5
) -> np.ndarray:
    """
    Create weighted ensemble predictions from two models.
    
    Parameters
    ----------
    pred1 : np.ndarray
        Predictions from first model
    pred2 : np.ndarray
        Predictions from second model
    weight1 : float, default 0.5
        Weight for first model (second model gets 1 - weight1)
        
    Returns
    -------
    np.ndarray
        Ensemble predictions
    """
    weight2 = 1 - weight1
    return weight1 * np.array(pred1) + weight2 * np.array(pred2)


def rolling_forecast_evaluation(
    test_data: pd.DataFrame,
    models: Dict,
    feature_cols: List[str],
    window_size: int = 30,
    step_size: int = 7
) -> Dict[str, Dict[str, List[float]]]:
    """
    Perform rolling forecast evaluation.
    
    Parameters
    ----------
    test_data : pd.DataFrame
        Test data with features and target
    models : Dict
        Dictionary of trained models {'name': model}
    feature_cols : List[str]
        List of feature column names
    window_size : int, default 30
        Size of evaluation window
    step_size : int, default 7
        Step size between windows
        
    Returns
    -------
    Dict[str, Dict[str, List[float]]]
        Results for each model with RMSE and MAPE lists
    """
    n_windows = (len(test_data) - window_size) // step_size + 1
    
    results = {name: {'rmse': [], 'mape': []} for name in models.keys()}
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        if end_idx > len(test_data):
            break
        
        window_data = test_data.iloc[start_idx:end_idx]
        y_window = window_data['y'].values
        X_window = window_data[feature_cols]
        
        for name, model in models.items():
            if hasattr(model, 'predict'):
                if hasattr(model, 'best_estimator'):  # FLAML
                    pred = model.predict(X_window)
                else:  # Prophet
                    forecast = model.predict(window_data[['ds']])
                    pred = forecast['yhat'].values
                
                rmse = np.sqrt(mean_squared_error(y_window, pred))
                mape = mean_absolute_percentage_error(y_window, pred) * 100
                
                results[name]['rmse'].append(rmse)
                results[name]['mape'].append(mape)
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_time_series(
    df: pd.DataFrame,
    date_col: str = 'ds',
    value_col: str = 'y',
    title: str = 'Time Series',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot a time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to plot
    date_col : str, default 'ds'
        Date column name
    value_col : str, default 'y'
        Value column name
    title : str, default 'Time Series'
        Plot title
    figsize : Tuple[int, int], default (14, 6)
        Figure size
    save_path : Optional[str]
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    plt.plot(df[date_col], df[value_col], linewidth=1, alpha=0.8)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions_comparison(
    dates: pd.Series,
    actual: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = 'Predictions Comparison',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot actual vs multiple model predictions.
    
    Parameters
    ----------
    dates : pd.Series
        Date values for x-axis
    actual : np.ndarray
        Actual values
    predictions : Dict[str, np.ndarray]
        Dictionary of predictions {'model_name': predictions}
    title : str, default 'Predictions Comparison'
        Plot title
    figsize : Tuple[int, int], default (14, 6)
        Figure size
    save_path : Optional[str]
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    plt.plot(dates, actual, 'o-', label='Actual', linewidth=2, color='black', alpha=0.8)
    
    colors = ['#2ECC71', '#F39C12', '#3498DB', '#E74C3C', '#9B59B6']
    for i, (name, pred) in enumerate(predictions.items()):
        plt.plot(dates, pred, '-', label=name, linewidth=2, 
                color=colors[i % len(colors)], alpha=0.7)
    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance as a horizontal bar chart.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int, default 15
        Number of top features to show
    title : str, default 'Feature Importance'
        Plot title
    figsize : Tuple[int, int], default (10, 8)
        Figure size
    save_path : Optional[str]
        Path to save the figure
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_features['importance'], 
             color='steelblue', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(title, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'Test RMSE',
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot model comparison bar chart.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with 'Model' and metric columns
    metric : str, default 'Test RMSE'
        Metric column to plot
    title : str, default 'Model Comparison'
        Plot title
    figsize : Tuple[int, int], default (10, 6)
        Figure size
    save_path : Optional[str]
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    models = comparison_df['Model'].tolist()
    values = comparison_df[metric].tolist()
    
    colors = ['#2ECC71' if v == min(values) else '#95A5A6' for v in values]
    
    bars = plt.bar(models, values, color=colors, edgecolor='black')
    plt.ylabel(metric)
    plt.title(title, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')
    
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# RESULTS SAVING
# =============================================================================

def save_results(
    predictions_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    summary: Dict,
    output_dir: str = 'outputs'
) -> None:
    """
    Save all results to files.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions
    comparison_df : pd.DataFrame
        Model comparison DataFrame
    summary : Dict
        Summary dictionary
    output_dir : str, default 'outputs'
        Output directory
    """
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    predictions_df.to_csv(f"{output_dir}/predictions.csv", index=False)
    
    # Save comparison
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    # Save summary
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"✅ Results saved to {output_dir}/")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Utils module loaded successfully!")
    print("\nAvailable functions:")
    print("  - load_energy_data()")
    print("  - clean_and_resample()")
    print("  - add_temporal_features()")
    print("  - add_lag_features()")
    print("  - add_rolling_features()")
    print("  - add_ema_features()")
    print("  - prepare_features()")
    print("  - temporal_train_test_split()")
    print("  - train_flaml_model()")
    print("  - train_prophet_model()")
    print("  - calculate_metrics()")
    print("  - evaluate_model()")
    print("  - create_ensemble_predictions()")
    print("  - rolling_forecast_evaluation()")
    print("  - plot_time_series()")
    print("  - plot_predictions_comparison()")
    print("  - plot_feature_importance()")
    print("  - plot_model_comparison()")
    print("  - save_results()")
