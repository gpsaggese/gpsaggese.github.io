"""
ONNX Time Series Forecasting Utilities

This module provides a unified interface for stock price forecasting using ONNX,
combining data preprocessing, model training, ONNX conversion, and evaluation.
"""

from preprocessing import *
from model import *
from utils import *
from evaluation import *

__all__ = [
    # Preprocessing
    'load_stock_data',
    'parse_and_sort_dates',
    'handle_missing_values',
    'detect_and_handle_outliers',
    'split_data_chronological',
    'normalize_data',
    'calculate_moving_averages',
    'calculate_bollinger_bands',
    'calculate_atr',
    'calculate_rsi',
    'calculate_macd',
    'calculate_volume_indicators',
    'calculate_returns',
    'create_lagged_features',
    'create_rolling_windows',
    'apply_all_features',
    'load_and_stack_mag7_stocks',
    'prepare_mag7_features',

    # LSTM Model
    'LSTMConfig',
    'build_lstm_model',
    'compile_model',
    'create_callbacks',
    'train_lstm_model',
    'save_model_and_history',
    'load_training_history',
    'plot_training_history',
    'get_model_summary',
    'create_and_train_lstm',

    # TCN Model
    'TCNConfig',
    'prepare_darts_timeseries',
    'build_tcn_model',
    'train_tcn_model',

    # XGBoost Model
    'XGBoostConfig',
    'build_xgboost_model',
    'train_xgboost_model',
    'flatten_sequences_for_xgboost',

    # ONNX Conversion
    'convert_to_onnx',
    'convert_tcn_to_onnx',
    'convert_xgboost_to_onnx',
    'verify_onnx',
    'ONNXInferenceSession',
    'compare_frameworks_inference',

    # Evaluation
    'calculate_mae',
    'calculate_rmse',
    'calculate_mape',
    'calculate_r2',
    'calculate_directional_accuracy',
    'evaluate_forecasts',
    'plot_predictions_vs_actual',
    'plot_residuals',
    'create_forecast_report',
    'compare_models',
    'plot_forecast_comparison',

    # Ensemble & Multi-Model Comparison
    'create_ensemble_predictions',
    'compare_multiple_models',
    'plot_ensemble_comparison',
    'plot_performance_heatmap',
    'get_best_model_per_stock',
]
