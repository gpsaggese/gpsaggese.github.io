# covid_utils.py
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import pandas as pd
import numpy as np

@dataclass
class RegionSpec:
    country: str
    state: Optional[str] = None   # for US state-level or province-level

# ---------- Data I/O ----------
def fetch_jhu_timeseries(confirmed_url: str=None) -> pd.DataFrame:
    """
    Fetches JHU confirmed cases time-series CSV (global) and returns tidy DataFrame:
    columns: ['date', 'country', 'state', 'cases'] (daily cumulative)
    """
    # default urls will point to GitHub raw files; override for Kaggle mirrors
    ...

def extract_region_daily(df: pd.DataFrame, region: RegionSpec) -> pd.DataFrame:
    """
    Return daily new cases (not cumulative) with date as datetime and columns ['ds','y']
    (Prophet-friendly), plus 'cases_cum' if needed.
    """
    ...

# ---------- Intervention/regressor construction ----------
def build_intervention_regressors(dates: List[pd.Timestamp], interventions: Dict[str,List[str]]) -> pd.DataFrame:
    """
    interventions: {"lockdown": ["2020-03-15", ...], "vax_start": [...]}
    returns DataFrame indexed by date with binary or intensity columns.
    """
    ...

# ---------- Evaluation ----------
def rmse(y_true, y_pred): ...
def mae(y_true, y_pred): ...
def smape(y_true, y_pred): ...

# ---------- Prophet ----------
def fit_prophet(df_ds_y: pd.DataFrame, regressors: Optional[pd.DataFrame]=None, weekly=True, changepoint_prior_scale=0.05):
    """
    Fit Prophet model; returns fitted model and forecast dataframe.
    """
    ...

def forecast_prophet(model, periods=28, regressors_future: Optional[pd.DataFrame]=None):
    ...

# ---------- ARIMA/SARIMA (pmdarima / statsmodels wrappers) ----------
def fit_arima(series: pd.Series, seasonal: bool=True, m:int=7):
    ...
def forecast_arima(model, steps:int):
    ...

# ---------- LSTM (Keras) ----------
def prepare_sequences(series: pd.Series, seq_len=28, scale=True):
    """
    Returns X_train, y_train, scaler
    """
    ...
def build_lstm_model(input_shape, units=64, dropout=0.2):
    ...
def train_lstm(model, X, y, callbacks=None, epochs=50, batch_size=32):
    ...
def forecast_lstm(model, X_input, steps):
    ...

# ---------- Plotting ----------
def plot_actual_vs_pred(ds, y_true, y_pred, ci=None, interventions:Optional[Dict]=None):
    ...
