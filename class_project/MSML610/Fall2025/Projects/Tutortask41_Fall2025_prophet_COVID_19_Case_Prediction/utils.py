"""
Prophet Utilities Module
========================
Reusable wrapper functions for Facebook Prophet time series forecasting.
Includes data preparation, model fitting, evaluation metrics, and visualization helpers.

Author: Ibrahim Ahmed Mohammed
Course: DATA607-PCS2
Project: COVID-19 Case Prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

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
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


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
    df = pd.read_csv(filepath)
    
    # Filter to country
    country_df = df[df['Country/Region'] == country]
    
    # Get date columns - must start with a digit (e.g., "1/22/20")
    date_cols = [c for c in df.columns if c[0].isdigit()]
    
    # Sum across all provinces/states
    totals = country_df[date_cols].sum()
    
    # Convert to Prophet format
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(totals.index, format='%m/%d/%y'),
        'y': totals.values
    })
    
    # Convert cumulative to daily new cases
    prophet_df['y'] = prophet_df['y'].diff().fillna(0).clip(lower=0)
    
    return prophet_df.sort_values('ds').reset_index(drop=True)


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
    df = pd.read_csv(filepath)
    return sorted(df['Country/Region'].unique().tolist())


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
    filtered = df[df[country_col] == country].copy()
    
    # If province column exists, aggregate to country level
    if province_col and province_col in filtered.columns:
        date_col = 'Date' if 'Date' in filtered.columns else filtered.columns[0]
        numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
        filtered = filtered.groupby(date_col)[numeric_cols].sum().reset_index()
    
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
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = pd.to_datetime(df[date_col])
    
    if compute_daily:
        # Compute daily new cases from cumulative
        prophet_df['y'] = df[target_col].diff().fillna(0)
        # Handle negative values (data corrections)
        prophet_df['y'] = prophet_df['y'].clip(lower=0)
    else:
        prophet_df['y'] = df[target_col]
    
    # Remove any NaN values
    prophet_df = prophet_df.dropna()
    
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
    holidays = pd.DataFrame({
        'holiday': list(interventions.keys()),
        'ds': pd.to_datetime(list(interventions.values())),
        'lower_window': 0,
        'upper_window': 14  # Effect lasts ~2 weeks after intervention
    })
    return holidays


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
    - Forecasting with confidence intervals
    - Model diagnostics and evaluation
    """
    
    def __init__(self,
                 weekly_seasonality: bool = True,
                 yearly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 interval_width: float = 0.95):
        """
        Initialize Prophet wrapper with configurable parameters.
        
        Parameters
        ----------
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
        """
        self.config = {
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
        self.train_data = df.copy()
        
        # Initialize Prophet with configuration
        self.model = Prophet(
            weekly_seasonality=self.config['weekly_seasonality'],
            yearly_seasonality=self.config['yearly_seasonality'],
            daily_seasonality=self.config['daily_seasonality'],
            changepoint_prior_scale=self.config['changepoint_prior_scale'],
            seasonality_prior_scale=self.config['seasonality_prior_scale'],
            holidays_prior_scale=self.config['holidays_prior_scale'],
            interval_width=self.config['interval_width'],
            holidays=self.holidays
        )
        
        # Add any registered regressors
        for reg in self.regressors:
            self.model.add_regressor(
                reg['name'],
                prior_scale=reg['prior_scale'],
                mode=reg['mode']
            )
        
        # Fit the model
        self.model.fit(df)
        
        return self
    
    def predict(self, periods: int = 28, 
                freq: str = 'D',
                include_history: bool = True,
                future_regressors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate forecasts for future periods.
        
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
            Forecast dataframe with predictions and intervals
        """
        if self.model is None:
            raise ValueError("Model must be fit before predicting")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Add regressor values if needed
        if self.regressors and future_regressors is not None:
            for reg in self.regressors:
                if reg['name'] in future_regressors.columns:
                    future = future.merge(
                        future_regressors[['ds', reg['name']]],
                        on='ds',
                        how='left'
                    )
        
        # Generate forecast
        self.forecast = self.model.predict(future)
        
        if not include_history:
            cutoff = self.train_data['ds'].max()
            self.forecast = self.forecast[self.forecast['ds'] > cutoff]
        
        return self.forecast
    
    def cross_validate(self,
                       initial: str = '180 days',
                       period: str = '30 days',
                       horizon: str = '28 days') -> pd.DataFrame:
        """
        Perform time series cross-validation.
        
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
            Cross-validation results
        """
        if self.model is None:
            raise ValueError("Model must be fit before cross-validation")
        
        cv_results = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
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
        metrics = performance_metrics(cv_results)
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
            raise ValueError("Must generate forecast before extracting components")
        
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
        
        return components


# =============================================================================
# COMPARISON MODEL UTILITIES
# =============================================================================

def fit_arima(df: pd.DataFrame, 
              order: Tuple[int, int, int] = (5, 1, 0)) -> Tuple[object, pd.Series]:
    """
    Fit ARIMA model for baseline comparison.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prophet-formatted data with 'ds' and 'y' columns
    order : tuple
        ARIMA (p, d, q) order
        
    Returns
    -------
    tuple
        (fitted_model, fitted_values)
    """
    model = ARIMA(df['y'].values, order=order)
    fitted = model.fit()
    return fitted, pd.Series(fitted.fittedvalues, index=df.index)


def fit_sarima(df: pd.DataFrame,
               order: Tuple[int, int, int] = (1, 1, 1),
               seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)) -> Tuple[object, pd.Series]:
    """
    Fit SARIMA model for baseline comparison with weekly seasonality.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prophet-formatted data with 'ds' and 'y' columns
    order : tuple
        ARIMA (p, d, q) order
    seasonal_order : tuple
        Seasonal (P, D, Q, s) order
        
    Returns
    -------
    tuple
        (fitted_model, fitted_values)
    """
    model = SARIMAX(df['y'].values, order=order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)
    return fitted, pd.Series(fitted.fittedvalues, index=df.index)


def forecast_arima(model, periods: int = 28) -> np.ndarray:
    """
    Generate ARIMA forecasts.
    
    Parameters
    ----------
    model : ARIMA fitted model
        Fitted ARIMA model
    periods : int
        Number of periods to forecast
        
    Returns
    -------
    np.ndarray
        Forecasted values
    """
    forecast = model.forecast(steps=periods)
    return forecast


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(actual, predicted))


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(actual, predicted)


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
    metrics = {
        'model': model_name,
        'rmse': calculate_rmse(actual, predicted),
        'mae': calculate_mae(actual, predicted),
        'smape': calculate_smape(actual, predicted)
    }
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
    results = []
    for model_name, pred in predictions.items():
        metrics = evaluate_forecast(actual, pred, model_name)
        results.append(metrics)
    
    return pd.DataFrame(results).set_index('model')


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
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values
    ax.plot(df['ds'], df['y'], 'k.', alpha=0.5, label='Actual', markersize=3)
    
    # Plot forecast
    ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast', linewidth=1.5)
    
    # Plot confidence intervals
    if show_intervals:
        ax.fill_between(forecast['ds'], 
                        forecast['yhat_lower'], 
                        forecast['yhat_upper'],
                        color='blue', alpha=0.2, label='95% CI')
    
    # Mark the forecast period
    if len(df) > 0:
        last_actual = df['ds'].max()
        ax.axvline(x=last_actual, color='red', linestyle='--', 
                   alpha=0.7, label='Forecast Start')
    
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def plot_components(model: Prophet,
                    forecast: pd.DataFrame,
                    figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Plot Prophet model components (trend, seasonality, holidays).
    
    Parameters
    ----------
    model : Prophet
        Fitted Prophet model
    forecast : pd.DataFrame
        Prophet forecast output
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    fig = model.plot_components(forecast, figsize=figsize)
    plt.tight_layout()
    return fig


def plot_intervention_effects(forecast: pd.DataFrame,
                              interventions: Dict[str, str],
                              figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Visualize the effects of interventions on the forecast.
    
    Parameters
    ----------
    forecast : pd.DataFrame
        Prophet forecast with holiday effects
    interventions : dict
        Dictionary of intervention names and dates
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot trend
    ax.plot(forecast['ds'], forecast['trend'], 'b-', 
            label='Trend', linewidth=2)
    
    # Mark interventions
    colors = plt.cm.Set1(np.linspace(0, 1, len(interventions)))
    for (name, date), color in zip(interventions.items(), colors):
        ax.axvline(x=pd.to_datetime(date), color=color, 
                   linestyle='--', linewidth=2, label=name)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Trend Component')
    ax.set_title('Intervention Effects on COVID-19 Trend')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_model_comparison(actual_dates: pd.Series,
                          actual_values: np.ndarray,
                          predictions: Dict[str, np.ndarray],
                          title: str = 'Model Comparison',
                          figsize: Tuple[int, int] = (14, 7)) -> plt.Figure:
    """
    Plot comparison of multiple model forecasts.
    
    Parameters
    ----------
    actual_dates : pd.Series
        Dates for actual values
    actual_values : array-like
        Actual values
    predictions : dict
        Dictionary mapping model names to their predictions
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual
    ax.plot(actual_dates, actual_values, 'k-', 
            label='Actual', linewidth=2, alpha=0.8)
    
    # Plot each model's predictions
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    for (model_name, pred), color in zip(predictions.items(), colors):
        ax.plot(actual_dates, pred, '--', 
                label=model_name, linewidth=1.5, color=color)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily New Cases')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
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
    df = base_df.copy()
    
    if scenario == 'strict':
        df['restriction_index'] = restriction_level * 1.5
    elif scenario == 'relaxed':
        df['restriction_index'] = restriction_level * 0.5
    else:  # baseline
        df['restriction_index'] = restriction_level
    
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
        Dictionary of scenario forecasts
    """
    scenarios = {}
    
    for scenario in ['baseline', 'strict', 'relaxed']:
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
        
        forecast = wrapper.model.predict(future)
        scenarios[scenario] = forecast
    
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
    return {
        'start_date': df[date_col].min(),
        'end_date': df[date_col].max(),
        'n_observations': len(df),
        'mean': df[value_col].mean(),
        'std': df[value_col].std(),
        'min': df[value_col].min(),
        'max': df[value_col].max(),
        'missing_values': df[value_col].isna().sum()
    }


if __name__ == "__main__":
    # Quick test of utilities
    print("Prophet Utilities Module")
    print("=" * 40)
    print("Available classes: ProphetWrapper")
    print("Available functions:")
    print("  - Data: load_covid_data, load_jhu_timeseries, get_available_countries")
    print("  - Data: filter_region, prepare_prophet_data")
    print("  - Models: fit_arima, fit_sarima, forecast_arima")
    print("  - Metrics: calculate_rmse, calculate_mae, calculate_smape")
    print("  - Visualization: plot_forecast, plot_components, plot_model_comparison")
    print("  - Scenarios: create_scenario_regressors, run_scenario_analysis")
    print("")
    print("Recommended data source: Johns Hopkins University (JHU)")
    print("  Download: https://github.com/CSSEGISandData/COVID-19")
    print("  Use: load_jhu_timeseries('jhu_confirmed_global.csv', country='US')")