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
# # Prophet: Time Series Forecasting with Synthetic Data
#
# This notebook demonstrates an end-to-end Prophet forecasting workflow on a
# synthetic daily time series that is designed to contain controlled
# components:
# - **Linear trend**: slow upward drift
# - **Weekly seasonality**: a 7-day Fourier pattern
# - **Holiday effects**: additive Christmas spikes
# - **AR(1) autoregression**: the value at time `t` depends on `t-1`
#
# The notebook is divided into four parts:
# 1. **Data Generation** — create and visualise the synthetic series
# 2. **Model Training** — configure Prophet and inspect fitted parameters
# 3. **Forecasting & Analysis** — predictions, static and interactive plots,
#    component decomposition
# 4. **Performance Evaluation** — out-of-sample metrics and cross-validation
#
# References:
# - White paper: https://peerj.com/preprints/3190.pdf
# - Official docs: https://facebook.github.io/prophet/docs/quick_start.html

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import prophet.diagnostics
import prophet.plot
import prophet.utilities
import scipy.stats

import helpers.hdbg as hdbg
import helpers.hpandas as hpandas
import helpers.hprint as hprint

import tutorials.tutorial_prophet.prophet_utils as tprpru

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)
hprint.config_notebook()

# %% [markdown]
# ## Config
#
# All tunable parameters live in a single dict so the entire notebook can be
# re-run by changing one place.

# %%
config = {
    # Date range for training and testing.
    "train_start_date": "2020-01-01",
    "train_end_date": "2023-12-31",
    "test_start_date": "2024-01-01",
    "test_end_date": "2024-12-31",
    # Data generation parameters.
    "data": {
        # Linear trend.
        "slope": 0.005,
        "intercept": 15,
        # Weekly seasonality (Fourier amplitudes).
        "weekly_amp_sin": 5.0,
        "weekly_amp_cos": 2.5,
        # Holiday dates and their additive impact.
        "holidays_dates": [
            "2020-12-25",
            "2021-12-25",
            "2022-12-25",
            "2023-12-25",
            "2024-12-25",
        ],
        "holidays_impact": 2.5,
        # AR(1) coefficient.
        "ar_order": 1,
        "phi": 0.7,
        # Noise.
        "seed": 42,
        "noise_sigma": 2.0,
    },
    # Prophet model configuration (passed directly to prophet.Prophet).
    "model": {
        "growth": "linear",
        "yearly_seasonality": False,
        # Integer Fourier order (1 = one sine + one cosine term).
        "weekly_seasonality": 1,
        "daily_seasonality": False,
        "n_changepoints": 0,
        # Internal target scaling method.
        "scaling": "minmax",
        # 95 % credible interval.
        "interval_width": 0.95,
        # MCMC samples; 0 = MAP point estimate only (faster).
        "mcmc_samples": 300,
        # Tight prior for holiday effect regularization.
        "holidays_prior_scale": 0.1,
    },
}
_LOG.info("config=%s", config)

# %% [markdown]
# ## Part 1: Data Generation
#
# **Objective**: Build a controlled synthetic time series that mirrors
# real-world properties (trend + seasonality + holidays + autocorrelation).
#
# Using `tprpru.generate_synthetic_time_series()` keeps all generation logic
# in the utility module so the notebook stays readable.

# %%
df = tprpru.generate_synthetic_time_series(config)
_LOG.info(hpandas.df_to_str(df, log_level=logging.INFO))

# %%
# Visualise the raw series over the entire date range.
df.set_index("ds")["y"].plot(
    title="Synthetic Time Series (2020–2024)",
    ylabel="Target variable",
    xlabel="Date",
    figsize=(12, 4),
)
plt.tight_layout()

# %% [markdown]
# ## Part 2: Model Training
#
# **Objective**: Fit a Prophet model to the training split (2020–2023) with
# custom holiday configuration and inspect the inferred parameters.
#
# Key design decisions:
# - `mcmc_samples=300`: full Bayesian inference for uncertainty quantification
# - `n_changepoints=0`: no changepoints because the synthetic trend is linear
# - `holidays_prior_scale=0.1`: strong regularization to prevent overfitting
#   the small holiday spikes

# %%
# Isolate the training split.
df_train = df[
    (df["ds"] >= config["train_start_date"])
    & (df["ds"] <= config["train_end_date"])
]
_LOG.info("Training set shape: %s", df_train.shape)

# %%
# Build the holidays DataFrame using the utility helper.
# Christmas holiday: no window extension (single-day effect).
holiday_names = [f"Christmas {y}" for y in range(2020, 2025)]
holidays_df = tprpru.build_holidays_df(
    dates=config["data"]["holidays_dates"],
    names=holiday_names,
    lower_window=0,
    upper_window=0,
)
_LOG.info(hpandas.df_to_str(holidays_df, log_level=logging.INFO))

# %%
# Fit the model.
forecaster = tprpru.ProphetForecastModel(config["model"], holidays=holidays_df)
forecaster.fit(df_train)
model = forecaster.get_model()
_LOG.info("Model fitted successfully")

# %%
# Inspect internal scaling applied by Prophet.
_LOG.info("y_scale=%s, y_min=%s", model.y_scale, model.y_min)

# %%
# Distribution of posterior intercept samples.
estimated_intercept = model.params["m"]
pd.Series(estimated_intercept).plot(kind="hist", title="Posterior: intercept")
_LOG.info(
    "intercept shape=%s, mean=%.4f",
    estimated_intercept.shape,
    estimated_intercept.mean(),
)

# %%
# Distribution of posterior slope samples.
estimated_slope = model.params["k"]
pd.Series(estimated_slope).plot(kind="hist", title="Posterior: slope")
_LOG.info(
    "slope shape=%s, mean=%.4f",
    estimated_slope.shape,
    estimated_slope.mean(),
)

# %% [markdown]
# The estimated AR(1) coefficient should be close to the ground-truth `phi=0.7`.

# %%
# Extract external regressor coefficients (already de-scaled by Prophet).
reg_coefs = prophet.utilities.regressor_coefficients(model)
_LOG.info(hpandas.df_to_str(reg_coefs, log_level=logging.INFO))
_LOG.info("True AR(1) coefficient phi = %s", config["data"]["phi"])

# %%
# Inspect the full beta coefficient matrix (all Fourier + holiday terms).
col_names = model.make_all_seasonality_features(df_train)[0].columns
coefficients_df = pd.DataFrame(model.params["beta"], columns=col_names)
_LOG.info(hpandas.df_to_str(coefficients_df, log_level=logging.INFO))

# %% [markdown]
# ## Part 3: Forecasting & Analysis
#
# **Objective**: Generate in-sample and out-of-sample predictions and
# visualise the forecast alongside the ground truth.
#
# Because the model uses a lagged regressor (`y.lag1`), we cannot forecast
# genuinely future dates (the lagged values would be unknown). Instead, we
# predict on the full historical dataset where the lagged values exist, then
# inspect both the in-sample (train) and out-of-sample (test) windows.

# %%
# Predict on the full dataset (train + test, all lagged values are known).
forecast = forecaster.predict(df)
# Merge actuals back so we can compute residuals.
forecast = forecast.merge(df, how="inner", on=["ds"])
forecast["residual"] = forecast["y"] - forecast["yhat"]
_LOG.info(hpandas.df_to_str(forecast.head(), log_level=logging.INFO))

# %%
# Split forecast into train and test windows for separate analysis.
ins_forecast = forecast[forecast["ds"] <= config["train_end_date"]]
oos_forecast = forecast[forecast["ds"] >= config["test_start_date"]]
_LOG.info("In-sample rows: %d, Out-of-sample rows: %d",
          len(ins_forecast), len(oos_forecast))

# %%
# --- In-sample residual diagnostics ---
ins_forecast["residual"].plot(
    title="In-sample residuals (train set)", figsize=(12, 3)
)
plt.axhline(0, color="red", linestyle="--")
plt.tight_layout()

# %%
# Q-Q plot: residuals should be approximately normal if the model is
# well-specified.
scipy.stats.probplot(ins_forecast["residual"], dist="norm", plot=plt)
plt.title("Q-Q Plot: in-sample residuals")
plt.tight_layout()

# %%
# Plot observed vs predicted for the training period.
plt.figure(figsize=(12, 4))
plt.plot(ins_forecast["ds"], ins_forecast["y"],
         label="Observed", color="blue")
plt.plot(ins_forecast["ds"], ins_forecast["yhat"],
         label="Point Estimate", ls="--", color="#0072B2")
plt.fill_between(
    ins_forecast["ds"],
    ins_forecast["yhat_lower"],
    ins_forecast["yhat_upper"],
    color="blue", alpha=0.2,
    label="95% Credible Interval",
)
plt.title("In-sample: Observed vs Predicted (train set)")
plt.legend()
plt.tight_layout()

# %%
# Prophet's built-in plot for the in-sample window.
model.plot(ins_forecast, include_legend=True)
plt.title("Prophet.plot() — Train Set")
plt.tight_layout()

# %%
# Decompose components for the in-sample period.
fig_comp_train = model.plot_components(ins_forecast)
plt.suptitle("Component Decomposition — Train Set", y=1.02)

# %%
# Prophet's built-in plot for the out-of-sample window.
model.plot(oos_forecast, include_legend=True)
plt.title("Prophet.plot() — Test Set")
plt.tight_layout()

# %%
# Decompose components for the test period.
fig_comp_test = model.plot_components(oos_forecast)
plt.suptitle("Component Decomposition — Test Set", y=1.02)

# %%
# Interactive Plotly forecast with ground truth overlay.
fig_plotly = prophet.plot.plot_plotly(model, forecast)
fig_plotly.add_trace(
    go.Scatter(
        x=oos_forecast["ds"],
        y=oos_forecast["y"],
        mode="markers",
        name="Ground Truth (test)",
        marker=dict(color="red", size=5),
    )
)
fig_plotly.update_layout(
    title="Interactive Forecast vs Ground Truth",
    xaxis_title="Date",
    yaxis_title="Value",
    legend=dict(orientation="h", y=-0.2),
)
fig_plotly.show()

# %% [markdown]
# ## Part 4: Performance Evaluation
#
# **Objective**: Quantify forecast accuracy with out-of-sample metrics and
# validate robustness via cross-validation.
#
# Metrics used:
# - **MAE** (Mean Absolute Error): average absolute deviation
# - **RMSE** (Root Mean Squared Error): penalises large errors more heavily
# - **MAPE** (Mean Absolute Percentage Error): scale-free relative error
#
# Prophet's `cross_validation()` performs sliding-window backtesting on the
# training data without refitting the Bayesian model each fold (it
# re-uses MCMC samples), making it computationally feasible.

# %%
# Out-of-sample accuracy on the 2024 test set.
metrics = forecaster.evaluate(oos_forecast)
_LOG.info(
    "Test set metrics — MAE=%.4f, RMSE=%.4f, MAPE=%.4f%%",
    metrics["mae"], metrics["rmse"], metrics["mape"],
)

# %%
# Cross-validation with a 2-year initial window, 180-day period,
# and 365-day horizon.
df_cv = prophet.diagnostics.cross_validation(
    model=model,
    initial="730 days",
    period="180 days",
    horizon="365 days",
)

# %%
# Aggregate metrics across all cross-validation folds.
df_perf = prophet.diagnostics.performance_metrics(df_cv)
_LOG.info(hpandas.df_to_str(df_perf, log_level=logging.INFO))

# %%
# Inspect first rows of the cross-validation results.
df_cv.head()

# %%
# Plot MAPE as a function of forecast horizon distance.
fig_mape = prophet.plot.plot_cross_validation_metric(df_cv, metric="mape")
plt.title("MAPE over Forecast Horizon (Cross-Validation)")
plt.tight_layout()
