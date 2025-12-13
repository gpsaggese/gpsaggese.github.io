"""
Prophet Utilities Module
========================
Reusable wrapper functions for Facebook Prophet time series forecasting.
Includes data preparation, model fitting, evaluation metrics, and visualization helpers.

KEY FEATURE: All predictions are guaranteed non-negative (essential for count data).

Author: Ibrahim Ahmed Mohammed
Course: DATA610
Project: COVID-19 Case Prediction

MODELS INCLUDED:
- Prophet (with wrapper class)
- ARIMA (statistical baseline)
- SARIMA (seasonal ARIMA)
- LSTM (deep learning)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Prophet import
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Statistical models for comparison
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Deep Learning - LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')


# =============================================================================
# DATA PREPARATION UTILITIES
# =============================================================================

def load_covid_data(filepath: str, date_col: str = 'Date') -> pd.DataFrame:
    """
    Load COVID-19 dataset and perform initial preprocessing.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    date_col : str
        Name of the date column
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with parsed dates
    """
    logger.info(f"Loading COVID data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        logger.info(f"Successfully loaded {len(df)} records from {df[date_col].min().date()} to {df[date_col].max().date()}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {str(e)}")
        raise


def load_jhu_timeseries(filepath: str, country: str = 'US') -> pd.DataFrame:
    """
    Load and transform Johns Hopkins COVID-19 time series data.
    
    JHU format has rows as countries/provinces and columns as dates.
    Data available from Jan 2020 to March 2023.
    
    Parameters
    ----------
    filepath : str
        Path to JHU time series CSV file
    country : str
        Country name to filter (default 'US')
        
    Returns
    -------
    pd.DataFrame
        Prophet-formatted dataframe with 'ds' and 'y' columns (daily new cases)
    """
    logger.info(f"Loading JHU time series data for {country}")
    try:
        df = pd.read_csv(filepath)
        
        # Filter to country
        country_df = df[df['Country/Region'] == country]
        
        if len(country_df) == 0:
            logger.error(f"Country '{country}' not found in dataset")
            raise ValueError(f"Country '{country}' not found in dataset")
        
        # Get date columns - must start with a digit (e.g., "1/22/20")
        date_cols = [c for c in df.columns if c[0].isdigit()]
        logger.debug(f"Found {len(date_cols)} date columns")
        
        # Sum across all provinces/states
        totals = country_df[date_cols].sum()
        
        # Convert to Prophet format
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(totals.index, format='%m/%d/%y'),
            'y': totals.values
        })
        
        # Convert cumulative to daily new cases
        prophet_df['y'] = prophet_df['y'].diff().fillna(0).clip(lower=0)
        
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        logger.info(f"Successfully loaded {len(prophet_df)} days of data for {country}: {prophet_df['ds'].min().date()} to {prophet_df['ds'].max().date()}")
        
        return prophet_df
    except Exception as e:
        logger.error(f"Failed to load JHU data: {str(e)}")
        raise


def get_available_countries(filepath: str) -> List[str]:
    """
    Get list of available countries in JHU dataset.
    
    Parameters
    ----------
    filepath : str
        Path to JHU time series CSV file
        
    Returns
    -------
    list
        List of country names
    """
    logger.info(f"Getting available countries from {filepath}")
    df = pd.read_csv(filepath)
    countries = sorted(df['Country/Region'].unique().tolist())
    logger.info(f"Found {len(countries)} countries")
    return countries


def filter_region(df: pd.DataFrame, 
                  country: str, 
                  country_col: str = 'Country/Region',
                  province_col: Optional[str] = 'Province/State') -> pd.DataFrame:
    """
    Filter dataset to a specific country or region.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full COVID dataset
    country : str
        Country name to filter
    country_col : str
        Column name containing country information
    province_col : str, optional
        Column name for province/state (if aggregation needed)
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe for the specified region
    """
    logger.info(f"Filtering data for country: {country}")
    filtered = df[df[country_col] == country].copy()
    
    # If province column exists, aggregate to country level
    if province_col and province_col in filtered.columns:
        date_col = 'Date' if 'Date' in filtered.columns else filtered.columns[0]
        numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
        filtered = filtered.groupby(date_col)[numeric_cols].sum().reset_index()
    
    logger.info(f"Filtered to {len(filtered)} records for {country}")
    return filtered


def prepare_prophet_data(df: pd.DataFrame,
                         date_col: str = 'Date',
                         target_col: str = 'Confirmed',
                         compute_daily: bool = True) -> pd.DataFrame:
    """
    Transform data into Prophet-required format (ds, y columns).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_col : str
        Name of date column
    target_col : str
        Name of target variable column (cumulative cases)
    compute_daily : bool
        If True, compute daily new cases from cumulative
        
    Returns
    -------
    pd.DataFrame
        Prophet-formatted dataframe with 'ds' and 'y' columns
    """
    logger.info(f"Preparing Prophet data from {date_col} and {target_col}")
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = pd.to_datetime(df[date_col])
    
    if compute_daily:
        # Compute daily new cases from cumulative
        prophet_df['y'] = df[target_col].diff().fillna(0)
        # Handle negative values (data corrections)
        prophet_df['y'] = prophet_df['y'].clip(lower=0)
        logger.info("Computed daily new cases from cumulative data")
    else:
        prophet_df['y'] = df[target_col]
    
    # Remove any NaN values
    prophet_df = prophet_df.dropna()
    logger.info(f"Prepared {len(prophet_df)} rows of Prophet-formatted data")
    
    return prophet_df


def create_intervention_dataframe(interventions: Dict[str, str]) -> pd.DataFrame:
    """
    Create a holidays/interventions dataframe for Prophet.
    
    Parameters
    ----------
    interventions : dict
        Dictionary mapping intervention names to dates
        Example: {'lockdown_start': '2020-03-15', 'vaccine_rollout': '2020-12-14'}
        
    Returns
    -------
    pd.DataFrame
        Prophet-compatible holidays dataframe
    """
    logger.info(f"Creating intervention dataframe with {len(interventions)} events")
    holidays = pd.DataFrame({
        'holiday': list(interventions.keys()),
        'ds': pd.to_datetime(list(interventions.values())),
        'lower_window': 0,
        'upper_window': 14  # Effect lasts ~2 weeks after intervention
    })
    logger.debug(f"Interventions: {list(interventions.keys())}")
    return holidays


# =============================================================================
# HELPER FUNCTION FOR NON-NEGATIVE PREDICTIONS
# =============================================================================

def _clip_forecast_to_floor(forecast: pd.DataFrame, floor: float = 0.0) -> pd.DataFrame:
    """
    Clip forecast values to ensure non-negative predictions.
    
    This is essential for count data like COVID cases where negative
    predictions are not meaningful.
    
    Parameters
    ----------
    forecast : pd.DataFrame
        Prophet forecast dataframe
    floor : float
        Minimum allowed value (default 0.0)
        
    Returns
    -------
    pd.DataFrame
        Forecast with clipped values
    """
    forecast = forecast.copy()
    
    # Clip main prediction
    if 'yhat' in forecast.columns:
        negative_count = (forecast['yhat'] < floor).sum()
        if negative_count > 0:
            logger.warning(f"Clipping {negative_count} negative predictions to floor={floor}")
        forecast['yhat'] = forecast['yhat'].clip(lower=floor)
    
    # Clip confidence intervals
    if 'yhat_lower' in forecast.columns:
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=floor)
    if 'yhat_upper' in forecast.columns:
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=floor)
    
    return forecast


# =============================================================================
# PROPHET MODEL WRAPPER CLASS
# =============================================================================

class ProphetWrapper:
    """
    Wrapper class for Facebook Prophet with COVID-19 specific configurations.
    
    Provides simplified interface for:
    - Model configuration with sensible defaults
    - Adding interventions/holidays
    - Adding external regressors
    - Forecasting with confidence intervals (guaranteed non-negative)
    - Model diagnostics and evaluation
    
    Key Feature: All predictions are automatically clipped to be non-negative,
    which is essential for count data like COVID-19 cases.
    """
    
    def __init__(self,
                 growth: str = 'linear',
                 weekly_seasonality: bool = True,
                 yearly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 interval_width: float = 0.95,
                 floor: float = 0.0,
                 cap: Optional[float] = None):
        """
        Initialize Prophet wrapper with configurable parameters.
        
        Parameters
        ----------
        growth : str
            'linear' or 'logistic' growth model
        weekly_seasonality : bool
            Include weekly seasonal component (captures reporting cycles)
        yearly_seasonality : bool
            Include yearly seasonal component
        daily_seasonality : bool
            Include daily seasonal component (usually False for daily data)
        changepoint_prior_scale : float
            Flexibility of trend changes (higher = more flexible)
        seasonality_prior_scale : float
            Strength of seasonality (higher = stronger)
        holidays_prior_scale : float
            Strength of holiday effects
        interval_width : float
            Width of uncertainty intervals (default 95%)
        floor : float
            Minimum value for predictions (default 0.0 for COVID cases)
            All predictions will be clipped to this value.
        cap : float, optional
            Maximum value for logistic growth (auto-calculated if None)
        """
        self.growth = growth
        self.floor = floor
        self.cap = cap
        
        self.config = {
            'growth': growth,
            'weekly_seasonality': weekly_seasonality,
            'yearly_seasonality': yearly_seasonality,
            'daily_seasonality': daily_seasonality,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'holidays_prior_scale': holidays_prior_scale,
            'interval_width': interval_width
        }
        self.model = None
        self.forecast = None
        self.train_data = None
        self.holidays = None
        self.regressors = []
        
        logger.info(f"ProphetWrapper initialized with config: {self.config}")
        logger.info(f"Floor constraint: {floor} (predictions will be non-negative)")
        
    def set_holidays(self, holidays_df: pd.DataFrame) -> 'ProphetWrapper':
        """
        Set holidays/interventions dataframe.
        
        Parameters
        ----------
        holidays_df : pd.DataFrame
            DataFrame with 'holiday', 'ds', 'lower_window', 'upper_window' columns
            
        Returns
        -------
        self : ProphetWrapper
            Returns self for method chaining
        """
        self.holidays = holidays_df
        logger.info(f"Set {len(holidays_df)} holidays/interventions")
        return self
    
    def add_regressor(self, name: str, 
                      prior_scale: float = 10.0,
                      mode: str = 'additive') -> 'ProphetWrapper':
        """
        Register an external regressor to be added to the model.
        
        Parameters
        ----------
        name : str
            Column name of the regressor in the training data
        prior_scale : float
            Regularization strength for this regressor
        mode : str
            'additive' or 'multiplicative'
            
        Returns
        -------
        self : ProphetWrapper
            Returns self for method chaining
        """
        self.regressors.append({
            'name': name,
            'prior_scale': prior_scale,
            'mode': mode
        })
        logger.info(f"Added regressor: {name} (prior_scale={prior_scale}, mode={mode})")
        return self
    
    def fit(self, df: pd.DataFrame) -> 'ProphetWrapper':
        """
        Fit the Prophet model to training data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with 'ds', 'y' columns (and regressor columns if any)
            
        Returns
        -------
        self : ProphetWrapper
            Returns self for method chaining
        """
        logger.info(f"Fitting Prophet model on {len(df)} observations")
        self.train_data = df.copy()
        
        # Add floor constraint to training data
        self.train_data['floor'] = self.floor
        
        # Add cap for logistic growth
        if self.growth == 'logistic':
            if self.cap is None:
                # Auto-calculate cap as 2x max value
                self.cap = self.train_data['y'].max() * 2.0
                logger.info(f"Auto-calculated cap for logistic growth: {self.cap:,.0f}")
            self.train_data['cap'] = self.cap
        
        # Initialize Prophet with configuration
        self.model = Prophet(
            growth=self.growth,
            weekly_seasonality=self.config['weekly_seasonality'],
            yearly_seasonality=self.config['yearly_seasonality'],
            daily_seasonality=self.config['daily_seasonality'],
            changepoint_prior_scale=self.config['changepoint_prior_scale'],
            seasonality_prior_scale=self.config['seasonality_prior_scale'],
            holidays_prior_scale=self.config['holidays_prior_scale'],
            holidays=self.holidays,
            interval_width=self.config['interval_width']
        )
        
        # Add regressors
        for reg in self.regressors:
            self.model.add_regressor(
                reg['name'],
                prior_scale=reg['prior_scale'],
                mode=reg['mode']
            )
            logger.debug(f"Added regressor to model: {reg['name']}")
        
        # Fit model
        self.model.fit(self.train_data)
        logger.info("Prophet model fitted successfully")
        
        return self
    
    def predict(self, 
                periods: int = 28,
                freq: str = 'D',
                include_history: bool = True,
                future_regressors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate forecasts for future periods.
        
        All predictions are automatically clipped to be >= floor (default 0).
        This ensures no negative case counts in the forecast.
        
        Parameters
        ----------
        periods : int
            Number of periods to forecast (default 28 days = 4 weeks)
        freq : str
            Frequency of predictions ('D' for daily)
        include_history : bool
            Include historical dates in forecast
        future_regressors : pd.DataFrame, optional
            Values of regressors for future periods
            
        Returns
        -------
        pd.DataFrame
            Forecast dataframe with non-negative predictions and intervals
        """
        if self.model is None:
            logger.error("Model not fitted. Call fit() first.")
            raise ValueError("Model must be fit before predicting")
        
        logger.info(f"Generating {periods}-period forecast")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Add floor constraint
        future['floor'] = self.floor
        
        # Add cap for logistic growth
        if self.growth == 'logistic':
            future['cap'] = self.cap
        
        # Add regressor values if needed
        if self.regressors and future_regressors is not None:
            for reg in self.regressors:
                if reg['name'] in future_regressors.columns:
                    future = future.merge(
                        future_regressors[['ds', reg['name']]],
                        on='ds',
                        how='left'
                    )
                    logger.debug(f"Added regressor {reg['name']} to future dataframe")
        
        # Generate forecast
        self.forecast = self.model.predict(future)
        
        # CRITICAL: Enforce floor constraint on predictions to prevent negative values
        self.forecast = _clip_forecast_to_floor(self.forecast, self.floor)
        
        if not include_history:
            cutoff = self.train_data['ds'].max()
            self.forecast = self.forecast[self.forecast['ds'] > cutoff]
        
        logger.info(f"Forecast generated: {len(self.forecast)} rows (min yhat: {self.forecast['yhat'].min():.2f}, max yhat: {self.forecast['yhat'].max():.2f})")
        return self.forecast
    
    def cross_validate(self,
                       initial: str = '365 days',
                       period: str = '30 days',
                       horizon: str = '28 days') -> pd.DataFrame:
        """
        Perform time series cross-validation.
        
        Note: Cross-validation results may contain negative predictions
        from individual folds. Use get_performance_metrics() which handles this.
        
        Parameters
        ----------
        initial : str
            Size of initial training period
        period : str
            Spacing between cutoff dates
        horizon : str
            Forecast horizon
            
        Returns
        -------
        pd.DataFrame
            Cross-validation results with clipped predictions
        """
        if self.model is None:
            logger.error("Model not fitted. Call fit() first.")
            raise ValueError("Model must be fit before cross-validation")
        
        logger.info(f"Running cross-validation: initial={initial}, period={period}, horizon={horizon}")
        cv_results = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        # Clip CV predictions to floor
        if 'yhat' in cv_results.columns:
            negative_count = (cv_results['yhat'] < self.floor).sum()
            if negative_count > 0:
                logger.warning(f"Clipping {negative_count} negative CV predictions to floor={self.floor}")
            cv_results['yhat'] = cv_results['yhat'].clip(lower=self.floor)
        
        logger.info(f"Cross-validation complete: {len(cv_results)} predictions")
        
        return cv_results
    
    def get_performance_metrics(self, cv_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics from cross-validation results.
        
        Parameters
        ----------
        cv_results : pd.DataFrame
            Output from cross_validate()
            
        Returns
        -------
        pd.DataFrame
            Performance metrics (MAE, RMSE, MAPE, etc.)
        """
        logger.info("Calculating performance metrics from cross-validation")
        metrics = performance_metrics(cv_results)
        logger.info(f"Metrics calculated for {len(metrics)} horizons")
        return metrics
    
    def get_components(self) -> Dict[str, pd.DataFrame]:
        """
        Extract trend and seasonal components from fitted model.
        
        Returns
        -------
        dict
            Dictionary with 'trend', 'weekly', 'yearly' components
        """
        if self.forecast is None:
            logger.error("No forecast available. Call predict() first.")
            raise ValueError("Must generate forecast before extracting components")
        
        logger.info("Extracting forecast components")
        components = {
            'trend': self.forecast[['ds', 'trend']],
            'weekly': self.forecast[['ds', 'weekly']] if 'weekly' in self.forecast.columns else None,
            'yearly': self.forecast[['ds', 'yearly']] if 'yearly' in self.forecast.columns else None
        }
        
        # Add holiday effects if present
        holiday_cols = [c for c in self.forecast.columns if c.startswith('holidays') or 
                       (self.holidays is not None and c in self.holidays['holiday'].values)]
        if holiday_cols:
            components['holidays'] = self.forecast[['ds'] + holiday_cols]
            logger.debug(f"Extracted holiday components: {holiday_cols}")
        
        return components


# =============================================================================
# COMPARISON MODEL UTILITIES - ARIMA/SARIMA
# =============================================================================

def fit_arima(df: pd.DataFrame, 
              order: Tuple[int, int, int] = (5, 1, 0),
              enforce_non_negative: bool = True) -> Tuple[object, pd.Series]:
    """
    Fit ARIMA model for baseline comparison.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prophet-formatted data with 'ds' and 'y' columns
    order : tuple
        ARIMA (p, d, q) order
    enforce_non_negative : bool
        If True, clip fitted values to be >= 0
        
    Returns
    -------
    tuple
        (fitted_model, fitted_values)
    """
    logger.info(f"Fitting ARIMA{order} model")
    try:
        model = ARIMA(df['y'].values, order=order)
        fitted = model.fit()
        fitted_values = pd.Series(fitted.fittedvalues, index=df.index)
        
        if enforce_non_negative:
            negative_count = (fitted_values < 0).sum()
            if negative_count > 0:
                logger.warning(f"Clipping {negative_count} negative ARIMA fitted values to 0")
            fitted_values = fitted_values.clip(lower=0)
        
        logger.info(f"ARIMA model fitted. AIC: {fitted.aic:.2f}, BIC: {fitted.bic:.2f}")
        return fitted, fitted_values
    except Exception as e:
        logger.error(f"Failed to fit ARIMA model: {str(e)}")
        raise


def fit_sarima(df: pd.DataFrame,
               order: Tuple[int, int, int] = (1, 1, 1),
               seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
               enforce_non_negative: bool = True) -> Tuple[object, pd.Series]:
    """
    Fit SARIMA model for baseline comparison with weekly seasonality.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prophet-formatted data with 'ds' and 'y' columns
    order : tuple
        ARIMA (p, d, q) order
    seasonal_order : tuple
        Seasonal (P, D, Q, s) order - default s=7 for weekly
    enforce_non_negative : bool
        If True, clip fitted values to be >= 0
        
    Returns
    -------
    tuple
        (fitted_model, fitted_values)
    """
    logger.info(f"Fitting SARIMA{order}x{seasonal_order} model")
    try:
        model = SARIMAX(df['y'].values, order=order, seasonal_order=seasonal_order)
        fitted = model.fit(disp=False)
        fitted_values = pd.Series(fitted.fittedvalues, index=df.index)
        
        if enforce_non_negative:
            negative_count = (fitted_values < 0).sum()
            if negative_count > 0:
                logger.warning(f"Clipping {negative_count} negative SARIMA fitted values to 0")
            fitted_values = fitted_values.clip(lower=0)
        
        logger.info(f"SARIMA model fitted. AIC: {fitted.aic:.2f}, BIC: {fitted.bic:.2f}")
        return fitted, fitted_values
    except Exception as e:
        logger.error(f"Failed to fit SARIMA model: {str(e)}")
        raise


def forecast_arima(model, periods: int = 28, enforce_non_negative: bool = True) -> np.ndarray:
    """
    Generate ARIMA forecasts.
    
    Parameters
    ----------
    model : ARIMA fitted model
        Fitted ARIMA model
    periods : int
        Number of periods to forecast
    enforce_non_negative : bool
        If True, clip forecast values to be >= 0
        
    Returns
    -------
    np.ndarray
        Forecasted values (non-negative if enforce_non_negative=True)
    """
    logger.info(f"Generating {periods}-period ARIMA forecast")
    forecast = model.forecast(steps=periods)
    
    if enforce_non_negative:
        forecast_array = np.array(forecast)
        negative_count = (forecast_array < 0).sum()
        if negative_count > 0:
            logger.warning(f"Clipping {negative_count} negative ARIMA forecast values to 0")
        forecast = np.clip(forecast_array, 0, None)
    
    logger.info(f"ARIMA forecast generated: {len(forecast)} values")
    return forecast


def forecast_sarima(model, periods: int = 28, enforce_non_negative: bool = True) -> np.ndarray:
    """
    Generate SARIMA forecasts.
    
    Parameters
    ----------
    model : SARIMAX fitted model
        Fitted SARIMA model
    periods : int
        Number of periods to forecast
    enforce_non_negative : bool
        If True, clip forecast values to be >= 0
        
    Returns
    -------
    np.ndarray
        Forecasted values (non-negative if enforce_non_negative=True)
    """
    logger.info(f"Generating {periods}-period SARIMA forecast")
    forecast = model.forecast(steps=periods)
    
    if enforce_non_negative:
        forecast_array = np.array(forecast)
        negative_count = (forecast_array < 0).sum()
        if negative_count > 0:
            logger.warning(f"Clipping {negative_count} negative SARIMA forecast values to 0")
        forecast = np.clip(forecast_array, 0, None)
    
    logger.info(f"SARIMA forecast generated: {len(forecast)} values")
    return forecast


# =============================================================================
# LSTM MODEL UTILITIES
# =============================================================================

class LSTMForecaster:
    """
    LSTM Neural Network for time series forecasting.
    
    Designed for COVID-19 case prediction with:
    - Configurable sequence length (lookback window)
    - Multiple LSTM layers with dropout
    - Automatic data scaling
    - Non-negative predictions
    
    Example
    -------
    >>> lstm = LSTMForecaster(sequence_length=14, n_features=1)
    >>> lstm.fit(train_df, epochs=50)
    >>> predictions = lstm.predict(test_df)
    """
    
    def __init__(self,
                 sequence_length: int = 14,
                 n_features: int = 1,
                 lstm_units: List[int] = [64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM forecaster.
        
        Parameters
        ----------
        sequence_length : int
            Number of time steps to look back (default 14 days)
        n_features : int
            Number of input features (default 1 for univariate)
        lstm_units : list
            Number of units in each LSTM layer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for Adam optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.is_fitted = False
        
        logger.info(f"LSTMForecaster initialized: seq_len={sequence_length}, units={lstm_units}")
    
    def _build_model(self) -> Sequential:
        """Build the LSTM neural network architecture."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=(self.sequence_length, self.n_features)
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_seq = i < len(self.lstm_units) - 2
            model.add(LSTM(units=units, return_sequences=return_seq))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"LSTM model built: {model.count_params()} parameters")
        return model
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Parameters
        ----------
        data : np.ndarray
            Scaled input data
            
        Returns
        -------
        tuple
            (X, y) sequences for training
        """
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i, 0])
            y.append(data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        
        return X, y
    
    def fit(self, 
            df: pd.DataFrame,
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.1,
            early_stopping_patience: int = 10,
            verbose: int = 0) -> 'LSTMForecaster':
        """
        Fit the LSTM model to training data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with 'ds' and 'y' columns
        epochs : int
            Maximum training epochs
        batch_size : int
            Training batch size
        validation_split : float
            Fraction of data for validation
        early_stopping_patience : int
            Epochs to wait before early stopping
        verbose : int
            Verbosity level (0=silent, 1=progress, 2=detailed)
            
        Returns
        -------
        self : LSTMForecaster
            Returns self for method chaining
        """
        logger.info(f"Fitting LSTM model on {len(df)} observations")
        
        # Extract and scale data
        values = df['y'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        logger.info(f"Created {len(X)} training sequences")
        
        # Build model
        self.model = self._build_model()
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        # Train
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        self.is_fitted = True
        final_loss = self.history.history['loss'][-1]
        logger.info(f"LSTM model fitted. Final loss: {final_loss:.6f}")
        
        return self
    
    def predict(self, 
                df: pd.DataFrame,
                enforce_non_negative: bool = True) -> np.ndarray:
        """
        Generate predictions for the given data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with 'ds' and 'y' columns
        enforce_non_negative : bool
            If True, clip predictions to be >= 0
            
        Returns
        -------
        np.ndarray
            Predictions (same length as input minus sequence_length)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before predicting")
        
        logger.info(f"Generating LSTM predictions for {len(df)} observations")
        
        # Scale data
        values = df['y'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(values)
        
        # Create sequences
        X, _ = self._create_sequences(scaled_data)
        
        # Predict
        scaled_predictions = self.model.predict(X, verbose=0)
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(scaled_predictions).flatten()
        
        if enforce_non_negative:
            negative_count = (predictions < 0).sum()
            if negative_count > 0:
                logger.warning(f"Clipping {negative_count} negative LSTM predictions to 0")
            predictions = np.clip(predictions, 0, None)
        
        logger.info(f"LSTM predictions generated: {len(predictions)} values")
        return predictions
    
    def forecast(self,
                 df: pd.DataFrame,
                 periods: int = 28,
                 enforce_non_negative: bool = True) -> np.ndarray:
        """
        Generate multi-step ahead forecasts.
        
        Uses recursive forecasting: each prediction is fed back
        as input for the next prediction.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical data with 'ds' and 'y' columns
        periods : int
            Number of periods to forecast
        enforce_non_negative : bool
            If True, clip forecasts to be >= 0
            
        Returns
        -------
        np.ndarray
            Forecasted values for future periods
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before forecasting")
        
        logger.info(f"Generating {periods}-period LSTM forecast")
        
        # Get the last sequence from training data
        values = df['y'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(values)
        
        # Start with the last sequence_length values
        current_sequence = scaled_data[-self.sequence_length:].flatten()
        
        forecasts = []
        for _ in range(periods):
            # Reshape for prediction
            X = current_sequence.reshape(1, self.sequence_length, self.n_features)
            
            # Predict next value
            next_pred = self.model.predict(X, verbose=0)[0, 0]
            forecasts.append(next_pred)
            
            # Update sequence (slide window)
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = self.scaler.inverse_transform(forecasts).flatten()
        
        if enforce_non_negative:
            negative_count = (forecasts < 0).sum()
            if negative_count > 0:
                logger.warning(f"Clipping {negative_count} negative LSTM forecast values to 0")
            forecasts = np.clip(forecasts, 0, None)
        
        logger.info(f"LSTM forecast generated: {len(forecasts)} values")
        return forecasts
    
    def get_training_history(self) -> Dict:
        """Get training history (loss, metrics per epoch)."""
        if self.history is None:
            return {}
        return self.history.history
    
    def summary(self) -> None:
        """Print model architecture summary."""
        if self.model is not None:
            self.model.summary()


def prepare_lstm_data(df: pd.DataFrame, 
                      sequence_length: int = 14,
                      train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare data for LSTM training with train/test split.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prophet-formatted data with 'ds' and 'y' columns
    sequence_length : int
        Number of time steps to look back
    train_size : float
        Fraction of data for training
        
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test, scaler)
    """
    logger.info(f"Preparing LSTM data with sequence_length={sequence_length}")
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = df['y'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(values)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Train/test split
    train_len = int(len(X) * train_size)
    X_train, X_test = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]
    
    logger.info(f"Data prepared: {len(X_train)} train, {len(X_test)} test sequences")
    
    return X_train, y_train, X_test, y_test, scaler


def build_lstm_model(sequence_length: int = 14,
                     n_features: int = 1,
                     lstm_units: List[int] = [64, 32],
                     dropout_rate: float = 0.2,
                     learning_rate: float = 0.001) -> Sequential:
    """
    Build a standalone LSTM model (functional approach).
    
    Parameters
    ----------
    sequence_length : int
        Number of time steps
    n_features : int
        Number of features
    lstm_units : list
        Units per LSTM layer
    dropout_rate : float
        Dropout rate
    learning_rate : float
        Learning rate
        
    Returns
    -------
    Sequential
        Compiled Keras model
    """
    model = Sequential()
    
    model.add(LSTM(
        units=lstm_units[0],
        return_sequences=len(lstm_units) > 1,
        input_shape=(sequence_length, n_features)
    ))
    model.add(Dropout(dropout_rate))
    
    for i, units in enumerate(lstm_units[1:]):
        return_seq = i < len(lstm_units) - 2
        model.add(LSTM(units=units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"LSTM model built: {model.count_params()} parameters")
    return model


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    logger.debug(f"RMSE: {rmse:.4f}")
    return rmse


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    mae = mean_absolute_error(actual, predicted)
    logger.debug(f"MAE: {mae:.4f}")
    return mae


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    SMAPE = (100/n) * Σ(|F - A| / ((|A| + |F|) / 2))
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1, denominator)
    
    smape = np.mean(np.abs(predicted - actual) / denominator) * 100
    logger.debug(f"SMAPE: {smape:.4f}%")
    return smape


def evaluate_forecast(actual: np.ndarray, 
                      predicted: np.ndarray,
                      model_name: str = 'Model') -> Dict[str, float]:
    """
    Calculate all evaluation metrics for a forecast.
    
    Parameters
    ----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    model_name : str
        Name of the model for reporting
        
    Returns
    -------
    dict
        Dictionary with RMSE, MAE, and SMAPE
    """
    logger.info(f"Evaluating {model_name} forecast")
    metrics = {
        'model': model_name,
        'rmse': calculate_rmse(actual, predicted),
        'mae': calculate_mae(actual, predicted),
        'smape': calculate_smape(actual, predicted)
    }
    logger.info(f"{model_name} metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, SMAPE={metrics['smape']:.2f}%")
    return metrics


def compare_models(actual: np.ndarray,
                   predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compare multiple model forecasts.
    
    Parameters
    ----------
    actual : array-like
        Actual values
    predictions : dict
        Dictionary mapping model names to their predictions
        
    Returns
    -------
    pd.DataFrame
        Comparison table of metrics across models
    """
    logger.info(f"Comparing {len(predictions)} models")
    results = []
    for model_name, pred in predictions.items():
        metrics = evaluate_forecast(actual, pred, model_name)
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results).set_index('model')
    logger.info("Model comparison complete")
    return comparison_df


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_forecast(df: pd.DataFrame,
                  forecast: pd.DataFrame,
                  title: str = 'COVID-19 Cases Forecast',
                  ylabel: str = 'Daily New Cases',
                  figsize: Tuple[int, int] = (14, 7),
                  show_intervals: bool = True) -> plt.Figure:
    """
    Plot actual vs forecasted values with confidence intervals.
    
    Parameters
    ----------
    df : pd.DataFrame
        Actual data with 'ds' and 'y' columns
    forecast : pd.DataFrame
        Prophet forecast output
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    show_intervals : bool
        Whether to show confidence intervals
        
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    logger.info("Creating forecast plot")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values
    ax.plot(df['ds'], df['y'], 'k.', alpha=0.5, label='Actual', markersize=3)
    
    # Plot forecast
    ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast', linewidth=1.5)
    
    # Plot confidence intervals
    if show_intervals and 'yhat_lower' in forecast.columns:
        ax.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            alpha=0.2,
            color='blue',
            label='95% CI'
        )
    
    # Ensure y-axis starts at 0
    ax.set_ylim(bottom=0)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    logger.info("Forecast plot created")
    return fig


def plot_components(forecast: pd.DataFrame,
                    figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Plot Prophet forecast components (trend, seasonality).
    
    Parameters
    ----------
    forecast : pd.DataFrame
        Prophet forecast output
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    logger.info("Creating components plot")
    
    components = ['trend']
    if 'weekly' in forecast.columns:
        components.append('weekly')
    if 'yearly' in forecast.columns:
        components.append('yearly')
    
    n_components = len(components)
    fig, axes = plt.subplots(n_components, 1, figsize=figsize)
    
    if n_components == 1:
        axes = [axes]
    
    for ax, comp in zip(axes, components):
        ax.plot(forecast['ds'], forecast[comp], linewidth=1.5)
        ax.set_ylabel(comp.capitalize())
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    axes[-1].set_xlabel('Date')
    plt.suptitle('Forecast Components', fontsize=14)
    plt.tight_layout()
    
    logger.info("Components plot created")
    return fig


def plot_intervention_effects(forecast: pd.DataFrame,
                              interventions: Dict[str, str],
                              figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Visualize intervention effects on forecast.
    
    Parameters
    ----------
    forecast : pd.DataFrame
        Prophet forecast with intervention components
    interventions : dict
        Dictionary of intervention names and dates
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    logger.info("Creating intervention effects plot")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot trend
    ax.plot(forecast['ds'], forecast['trend'], label='Trend', linewidth=2)
    
    # Mark interventions
    colors = plt.cm.Set2(np.linspace(0, 1, len(interventions)))
    for (name, date), color in zip(interventions.items(), colors):
        date = pd.to_datetime(date)
        ax.axvline(x=date, color=color, linestyle='--', alpha=0.7, label=name)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Trend Component')
    ax.set_title('Intervention Effects on COVID-19 Trend')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    logger.info("Intervention effects plot created")
    return fig


def plot_model_comparison(actual_dates: pd.Series,
                          actual_values: np.ndarray,
                          predictions: Dict[str, np.ndarray],
                          title: str = 'Model Comparison',
                          figsize: Tuple[int, int] = (14, 7)) -> plt.Figure:
    """
    Plot actual vs multiple model predictions for comparison.
    
    Parameters
    ----------
    actual_dates : pd.Series
        Dates for x-axis
    actual_values : np.ndarray
        Actual values
    predictions : dict
        Dictionary mapping model names to predictions
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    logger.info(f"Creating model comparison plot for {len(predictions)} models")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual
    ax.plot(actual_dates, actual_values, 'k-', label='Actual', linewidth=1.5, alpha=0.7)
    
    # Plot predictions with different colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for (model_name, pred), color in zip(predictions.items(), colors):
        ax.plot(actual_dates, pred, '--', 
                label=model_name, linewidth=1.5, color=color)
    
    # Ensure y-axis starts at 0
    ax.set_ylim(bottom=0)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily New Cases')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    logger.info("Model comparison plot created")
    return fig


def plot_training_history(history: Dict, 
                          figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """
    Plot LSTM training history (loss curves).
    
    Parameters
    ----------
    history : dict
        Training history from Keras model.fit()
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    logger.info("Creating training history plot")
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    axes[0].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    if 'mae' in history:
        axes[1].plot(history['mae'], label='Training MAE')
        if 'val_mae' in history:
            axes[1].plot(history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training & Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    logger.info("Training history plot created")
    return fig


def plot_forecast_comparison(df: pd.DataFrame,
                             forecast_periods: int,
                             prophet_forecast: np.ndarray,
                             arima_forecast: np.ndarray,
                             sarima_forecast: np.ndarray,
                             lstm_forecast: np.ndarray,
                             title: str = 'Multi-Model Forecast Comparison',
                             figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Plot multi-step forecasts from all models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Historical data with 'ds' and 'y' columns
    forecast_periods : int
        Number of periods forecasted
    prophet_forecast : np.ndarray
        Prophet predictions
    arima_forecast : np.ndarray
        ARIMA predictions
    sarima_forecast : np.ndarray
        SARIMA predictions
    lstm_forecast : np.ndarray
        LSTM predictions
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    logger.info("Creating multi-model forecast comparison plot")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Historical data (last 60 days for context)
    history_days = min(60, len(df))
    hist_df = df.tail(history_days)
    
    # Future dates
    last_date = df['ds'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods)
    
    # Plot historical
    ax.plot(hist_df['ds'], hist_df['y'], 'k-', label='Historical', linewidth=1.5)
    
    # Plot forecasts
    ax.plot(future_dates, prophet_forecast, 'b--', label='Prophet', linewidth=2)
    ax.plot(future_dates, arima_forecast, 'r--', label='ARIMA', linewidth=2)
    ax.plot(future_dates, sarima_forecast, 'g--', label='SARIMA', linewidth=2)
    ax.plot(future_dates, lstm_forecast, 'm--', label='LSTM', linewidth=2)
    
    # Add vertical line at forecast start
    ax.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
    
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily New Cases')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    logger.info("Multi-model forecast comparison plot created")
    return fig


# =============================================================================
# SCENARIO ANALYSIS UTILITIES
# =============================================================================

def create_scenario_regressors(base_df: pd.DataFrame,
                               scenario: str = 'baseline',
                               restriction_level: float = 0.5) -> pd.DataFrame:
    """
    Create regressor values for scenario analysis.
    
    Parameters
    ----------
    base_df : pd.DataFrame
        Base future dataframe with 'ds' column
    scenario : str
        'baseline', 'strict', or 'relaxed'
    restriction_level : float
        Restriction intensity (0 = none, 1 = full lockdown)
        
    Returns
    -------
    pd.DataFrame
        Dataframe with scenario regressors
    """
    logger.info(f"Creating scenario regressors for '{scenario}' scenario")
    df = base_df.copy()
    
    if scenario == 'strict':
        df['restriction_index'] = restriction_level * 1.5
    elif scenario == 'relaxed':
        df['restriction_index'] = restriction_level * 0.5
    else:  # baseline
        df['restriction_index'] = restriction_level
    
    logger.debug(f"Restriction index set to {df['restriction_index'].iloc[0]:.2f}")
    return df


def run_scenario_analysis(wrapper: ProphetWrapper,
                          periods: int = 28,
                          base_restriction: float = 0.5) -> Dict[str, pd.DataFrame]:
    """
    Run multiple scenario forecasts.
    
    Parameters
    ----------
    wrapper : ProphetWrapper
        Fitted Prophet wrapper with restriction_index regressor
    periods : int
        Forecast periods
    base_restriction : float
        Baseline restriction level
        
    Returns
    -------
    dict
        Dictionary of scenario forecasts (all with non-negative values)
    """
    logger.info(f"Running scenario analysis for {periods} periods")
    scenarios = {}
    
    for scenario in ['baseline', 'strict', 'relaxed']:
        logger.info(f"Generating '{scenario}' scenario forecast")
        future = wrapper.model.make_future_dataframe(periods=periods)
        future_reg = create_scenario_regressors(
            future, scenario, base_restriction
        )
        
        # Merge regressor
        future = future.merge(
            future_reg[['ds', 'restriction_index']],
            on='ds',
            how='left'
        )
        future['restriction_index'] = future['restriction_index'].fillna(base_restriction)
        
        # Add floor constraint
        future['floor'] = wrapper.floor
        if wrapper.growth == 'logistic':
            future['cap'] = wrapper.cap
        
        forecast = wrapper.model.predict(future)
        
        # Clip to floor
        forecast = _clip_forecast_to_floor(forecast, wrapper.floor)
        
        scenarios[scenario] = forecast
    
    logger.info("Scenario analysis complete")
    return scenarios


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_us_covid_interventions() -> Dict[str, str]:
    """
    Return dictionary of major US COVID-19 intervention dates.
    
    Returns
    -------
    dict
        Intervention names mapped to dates
    """
    logger.debug("Getting US COVID-19 interventions")
    return {
        'national_emergency': '2020-03-13',
        'lockdowns_begin': '2020-03-19',
        'reopening_phase1': '2020-05-01',
        'summer_surge': '2020-07-01',
        'fall_surge': '2020-10-15',
        'vaccine_auth': '2020-12-11',
        'vaccine_rollout': '2021-01-15',
        'delta_surge': '2021-07-01',
        'omicron_surge': '2021-12-15'
    }


def get_country_interventions(country: str) -> Dict[str, str]:
    """
    Get intervention dates for specific countries.
    
    Parameters
    ----------
    country : str
        Country name
        
    Returns
    -------
    dict
        Intervention dates for the country
    """
    logger.debug(f"Getting interventions for {country}")
    interventions = {
        'US': get_us_covid_interventions(),
        'Germany': {
            'first_lockdown': '2020-03-22',
            'reopening': '2020-05-06',
            'second_lockdown': '2020-11-02',
            'vaccine_start': '2020-12-27',
            'third_wave': '2021-03-01'
        },
        'Brazil': {
            'first_case': '2020-02-26',
            'state_lockdowns': '2020-03-24',
            'peak_first_wave': '2020-07-29',
            'vaccine_start': '2021-01-17',
            'gamma_surge': '2021-03-01'
        },
        'India': {
            'lockdown_start': '2020-03-25',
            'unlock_1': '2020-06-01',
            'second_wave': '2021-03-01',
            'vaccine_rollout': '2021-01-16',
            'delta_peak': '2021-05-01'
        }
    }
    
    return interventions.get(country, {})


def summarize_data(df: pd.DataFrame, 
                   date_col: str = 'ds',
                   value_col: str = 'y') -> Dict:
    """
    Generate summary statistics for time series data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prophet-formatted dataframe
    date_col : str
        Date column name
    value_col : str
        Value column name
        
    Returns
    -------
    dict
        Summary statistics
    """
    logger.info("Generating data summary statistics")
    summary = {
        'start_date': df[date_col].min(),
        'end_date': df[date_col].max(),
        'n_observations': len(df),
        'mean': df[value_col].mean(),
        'std': df[value_col].std(),
        'min': df[value_col].min(),
        'max': df[value_col].max(),
        'missing_values': df[value_col].isna().sum()
    }
    logger.info(f"Data summary: {summary['n_observations']} observations from {summary['start_date'].date()} to {summary['end_date'].date()}")
    return summary


def train_test_split_temporal(df: pd.DataFrame, 
                               test_size: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.
    
    Uses temporal split (last N days for test) to prevent data leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'ds' and 'y' columns
    test_size : int
        Number of days for test set
        
    Returns
    -------
    tuple
        (train_df, test_df)
    """
    logger.info(f"Splitting data: {len(df)-test_size} train, {test_size} test")
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    return train_df, test_df


if __name__ == "__main__":
    # Quick test of utilities
    logger.info("Prophet Utilities Module - Self Test")
    print("=" * 60)
    print("Prophet Utilities Module (with LSTM)")
    print("=" * 60)
    print("\nKEY FEATURE: All predictions are guaranteed non-negative!")
    print("            (Essential for count data like COVID-19 cases)")
    print("\nAvailable classes:")
    print("  - ProphetWrapper: Enhanced Prophet interface")
    print("  - LSTMForecaster: LSTM neural network for time series")
    print("\nAvailable functions:")
    print("  Data Loading:")
    print("    - load_covid_data, load_jhu_timeseries, get_available_countries")
    print("    - filter_region, prepare_prophet_data, create_intervention_dataframe")
    print("  Statistical Models:")
    print("    - fit_arima, fit_sarima, forecast_arima, forecast_sarima")
    print("  Deep Learning:")
    print("    - LSTMForecaster class, prepare_lstm_data, build_lstm_model")
    print("  Metrics:")
    print("    - calculate_rmse, calculate_mae, calculate_smape")
    print("    - evaluate_forecast, compare_models")
    print("  Visualization:")
    print("    - plot_forecast, plot_components")
    print("    - plot_intervention_effects, plot_model_comparison")
    print("    - plot_training_history, plot_forecast_comparison")
    print("  Scenarios:")
    print("    - create_scenario_regressors, run_scenario_analysis")
    print("  Helpers:")
    print("    - get_us_covid_interventions, get_country_interventions")
    print("    - summarize_data, train_test_split_temporal")
    print("")
    print("Recommended data source: Johns Hopkins University (JHU)")
    print("  Download: https://github.com/CSSEGISandData/COVID-19")
    print("  Use: load_jhu_timeseries('jhu_confirmed_global.csv', country='US')")
    print("=" * 60)
    logger.info("Self test complete")
