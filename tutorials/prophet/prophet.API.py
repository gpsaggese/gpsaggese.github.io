# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Prophet API Overview
#
# [Prophet](https://facebook.github.io/prophet/) is an open-source time series
# forecasting library developed by Facebook (Meta). It is designed to handle
# common challenges in business time series:
# - **Trend**: linear or logistic growth with automatic changepoint detection
# - **Seasonality**: yearly, weekly, and daily patterns via Fourier series
# - **Holidays**: custom holiday effects with configurable windows
# - **Robustness**: handles missing data and outliers gracefully
#
# This notebook walks through the core Prophet API components with minimal,
# self-contained examples.

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prophet
import prophet.diagnostics
import prophet.plot
import prophet.utilities

import helpers.hdbg as hdbg

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## Basic Prophet Usage
#
# Prophet expects a DataFrame with exactly two columns:
# - `ds`: datestamps (datetime or parseable string)
# - `y`: the numeric metric to forecast
#
# Key workflow:
# - `Prophet(**kwargs)` — create the model with desired configuration
# - `.fit(df_train)` — train the model on historical data
# - `.make_future_dataframe(periods=N)` — build a forecast horizon DataFrame
# - `.predict(future)` — generate the forecast

# %%
# Create 2 years of synthetic daily data: linear trend + weekly seasonality.
np.random.seed(42)
dates = pd.date_range(start="2022-01-01", periods=365 * 2, freq="D")
t = np.arange(len(dates))
y = (
    10
    + 0.01 * t
    + 5 * np.sin(2 * np.pi * t / 7)
    + np.random.normal(0, 1, len(t))
)
df_basic = pd.DataFrame({"ds": dates, "y": y})
df_basic.head(5)

# %%
# Fit a basic Prophet model with weekly seasonality only.
m_basic = prophet.Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
)
m_basic.fit(df_basic)

# %%
# Forecast the next 30 days beyond the training period.
future = m_basic.make_future_dataframe(periods=30)
forecast = m_basic.predict(future)
# Key forecast columns: ds (date), yhat (point estimate),
# yhat_lower / yhat_upper (uncertainty interval).
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10)

# %%
# Plot the full forecast: historical data (dots) + prediction (line +
# shaded uncertainty band).
fig = m_basic.plot(forecast, include_legend=True)
plt.title("Prophet Basic Forecast")
plt.tight_layout()

# %% [markdown]
# ## Trend Types
#
# Prophet supports three growth modes:
# - **`linear`** (default): constant growth rate; changepoints allow slope
#   shifts
# - **`logistic`**: saturating S-curve growth toward a capacity ceiling
#   (`cap`); useful for market penetration or user growth
# - **`flat`**: no trend component; suitable for stationary series
#
# Changepoints are dates where the trend slope can change. Prophet
# auto-detects up to `n_changepoints` candidates in the first 80 % of
# training data.

# %%
# Linear trend with 5 changepoints.
m_linear = prophet.Prophet(
    growth="linear",
    n_changepoints=5,
    weekly_seasonality=False,
    yearly_seasonality=False,
)
m_linear.fit(df_basic)
forecast_linear = m_linear.predict(m_linear.make_future_dataframe(periods=14))
fig1 = m_linear.plot(forecast_linear)
plt.title("Linear Trend (5 changepoints)")
plt.tight_layout()

# %%
# Logistic growth requires a saturation cap column in the DataFrame.
df_logistic = df_basic.copy()
# Set capacity ceiling above the current max value.
df_logistic["cap"] = df_logistic["y"].max() * 1.5
m_logistic = prophet.Prophet(
    growth="logistic",
    weekly_seasonality=False,
    yearly_seasonality=False,
)
m_logistic.fit(df_logistic)
future_logistic = m_logistic.make_future_dataframe(periods=30)
# The cap must be set in the future DataFrame too.
future_logistic["cap"] = df_logistic["cap"].iloc[0]
forecast_logistic = m_logistic.predict(future_logistic)
fig2 = m_logistic.plot(forecast_logistic)
plt.title("Logistic Growth (saturating)")
plt.tight_layout()

# %%
# Flat trend: only seasonality and holiday effects, no growth.
m_flat = prophet.Prophet(
    growth="flat",
    weekly_seasonality=True,
    yearly_seasonality=False,
)
m_flat.fit(df_basic)
forecast_flat = m_flat.predict(m_flat.make_future_dataframe(periods=14))
fig3 = m_flat.plot(forecast_flat)
plt.title("Flat Trend")
plt.tight_layout()

# %% [markdown]
# ## Seasonality
#
# Prophet models seasonality using **Fourier series** — sums of sine and
# cosine terms. Key parameters:
# - `yearly_seasonality`: annual 365.25-day pattern (default `auto`)
# - `weekly_seasonality`: 7-day pattern (default `auto`)
# - `daily_seasonality`: intra-day pattern (default `auto`)
# - `fourier_order`: number of Fourier term pairs; higher = more flexible
#   curve but may overfit
#
# Custom seasonality for any period can be added via `add_seasonality()`.

# %%
# 3 years of data with both weekly and yearly seasonality.
np.random.seed(42)
dates_3y = pd.date_range(start="2020-01-01", periods=365 * 3, freq="D")
t3 = np.arange(len(dates_3y))
y_3y = (
    10
    + 0.005 * t3
    + 5 * np.sin(2 * np.pi * t3 / 7)  # weekly
    + 8 * np.sin(2 * np.pi * t3 / 365)  # yearly
    + np.random.normal(0, 1, len(t3))
)
df_seas = pd.DataFrame({"ds": dates_3y, "y": y_3y})
# Fit with both seasonalities active.
m_seas = prophet.Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)
m_seas.fit(df_seas)
forecast_seas = m_seas.predict(m_seas.make_future_dataframe(periods=30))
# plot_components() decomposes the forecast into trend and each seasonality.
fig_seas = m_seas.plot_components(forecast_seas)
plt.suptitle("Decomposed Seasonality Components", y=1.02)

# %%
# Custom monthly seasonality: period=30.5 days, 3 Fourier term pairs.
m_custom = prophet.Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
m_custom.add_seasonality(name="monthly", period=30.5, fourier_order=3)
m_custom.fit(df_basic)
forecast_custom = m_custom.predict(m_custom.make_future_dataframe(periods=30))
fig_custom = m_custom.plot_components(forecast_custom)
plt.suptitle("Custom Monthly Seasonality", y=1.02)

# %% [markdown]
# ## Holidays
#
# Holidays are modeled as additive indicator variables that adjust the
# forecast around specific dates.
#
# Holiday DataFrame schema (required columns):
# - `ds`: the holiday date
# - `holiday`: a unique string name for each event
# - `lower_window`: days **before** the holiday date that are also affected
#   (negative or zero)
# - `upper_window`: days **after** the holiday date that are also affected
#   (positive or zero)
#
# The strength of each holiday's effect is controlled by
# `holidays_prior_scale` (default `10`). Lower values enforce stronger
# regularization.

# %%
# Christmas holiday with Christmas Eve included (lower_window=-1).
holidays = pd.DataFrame(
    {
        "holiday": ["christmas"] * 3,
        "ds": pd.to_datetime(["2022-12-25", "2023-12-25", "2024-12-25"]),
        "lower_window": -1,  # Include Christmas Eve.
        "upper_window": 0,
    }
)
m_hols = prophet.Prophet(
    holidays=holidays,
    holidays_prior_scale=5.0,  # Moderate regularization.
    weekly_seasonality=True,
    yearly_seasonality=False,
)
m_hols.fit(df_basic)
forecast_hols = m_hols.predict(m_hols.make_future_dataframe(periods=30))
fig_hols = m_hols.plot_components(forecast_hols)
plt.suptitle("Holiday Effects in Components", y=1.02)

# %% [markdown]
# ## Component Plots
#
# `Prophet.plot_components()` decomposes the forecast into interpretable
# parts:
# - **Trend**: the underlying growth direction
# - **Holidays**: additive effects around special dates
# - **Seasonalities**: weekly, yearly, or custom periodic patterns
#
# `Prophet.plot()` shows the raw forecast with uncertainty bands.

# %%
# Fit a model with all components for the 3-year dataset.
m_all = prophet.Prophet(
    holidays=holidays,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)
m_all.fit(df_seas)
forecast_all = m_all.predict(m_all.make_future_dataframe(periods=60))
fig_all = m_all.plot(forecast_all, include_legend=True)
plt.title("Full Forecast with All Components")
plt.tight_layout()

# %%
# Decomposed view of all fitted components.
fig_comp = m_all.plot_components(forecast_all)
plt.suptitle("All Decomposed Components", y=1.02)

# %% [markdown]
# ## External Regressors
#
# Prophet supports adding **external variables** (covariates) that help
# explain the target. Use `model.add_regressor(name)` before fitting.
#
# Rules for external regressors:
# - The regressor column must be present in the **training** DataFrame
# - It must also be provided in the **prediction** DataFrame with known
#   values (in-sample / test period where the regressor is observable)
# - Regressors are treated as additive adjustments to the baseline model

# %%
# Use a 1-step lag of y as an external regressor (AR(1) via Prophet).
df_reg = df_basic.copy()
df_reg["y_lag1"] = df_reg["y"].shift(1)
df_reg = df_reg.dropna().reset_index(drop=True)
n_train = int(0.8 * len(df_reg))
df_train_reg = df_reg.iloc[:n_train]
df_test_reg = df_reg.iloc[n_train:].reset_index(drop=True)
# Register the regressor before fitting.
m_reg = prophet.Prophet(weekly_seasonality=True, yearly_seasonality=False)
m_reg.add_regressor("y_lag1")
m_reg.fit(df_train_reg)
# Predict on the test period where the lagged values are known.
forecast_reg = m_reg.predict(df_test_reg)
_LOG.info("Predicted %d rows with external regressor", len(forecast_reg))
forecast_reg[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(5)

# %%
# Inspect regressor coefficients.
reg_coefs = prophet.utilities.regressor_coefficients(m_reg)
_LOG.info("Regressor coefficients:\n%s", reg_coefs.to_string())

# %% [markdown]
# ## Cross-Validation
#
# Prophet's `cross_validation()` performs **sliding-window** backtesting:
# - `initial`: the minimum training window length
# - `period`: how far to shift the cutoff date between folds
# - `horizon`: the forecast horizon evaluated at each cutoff
#
# `performance_metrics()` aggregates MAE, RMSE, MAPE, and coverage across
# all folds grouped by forecast horizon distance.

# %%
# Run cross-validation on the basic model (no MCMC for speed).
m_cv = prophet.Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
)
m_cv.fit(df_basic)
df_cv = prophet.diagnostics.cross_validation(
    model=m_cv,
    initial="180 days",
    period="90 days",
    horizon="30 days",
)
df_perf = prophet.diagnostics.performance_metrics(df_cv)
_LOG.info("Cross-validation performance:\n%s", df_perf.to_string())

# %%
# Plot MAPE over the forecast horizon.
fig_cv = prophet.plot.plot_cross_validation_metric(df_cv, metric="mape")
plt.title("MAPE over Forecast Horizon (Cross-Validation)")
plt.tight_layout()
