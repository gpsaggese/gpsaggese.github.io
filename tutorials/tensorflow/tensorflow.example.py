# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TensorFlow: Structural Time Series Forecasting
#
# This notebook demonstrates a complete end-to-end time series forecasting
# pipeline using **TensorFlow Probability's Structural Time Series (STS)**
# module.
#
# **Workflow:**
# 1. Generate a synthetic daily time series with trend, seasonality, holidays,
#    and autoregressive noise
# 2. Build an STS model combining multiple interpretable components
# 3. Fit the model via Variational Inference (VI)
# 4. Forecast future values and evaluate performance

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics
import tensorflow_probability as tfp
import tf_keras

import helpers.hdbg as hdbg
import helpers.hpandas as hpandas
import tutorials.tensorflow.tensorflow_utils as tteteuti

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## Config
#
# All model and data parameters are centralised here so the notebook can be
# reproduced by changing a single dictionary.

# %%
config = {
    # Train / test split.
    "train_start_date": "2020-01-01",
    "train_end_date": "2023-12-31",
    "test_start_date": "2024-01-01",
    "test_end_date": "2024-12-31",
    "data": {
        # Linear trend.
        "slope": 0.005,
        "intercept": 15,
        # Weekly seasonal drift.
        "seasonal_drift_scale": 0.5,
        # Holiday dates and their additive impact.
        "holidays_dates": [
            "2020-12-25",
            "2021-12-25",
            "2022-12-25",
            "2023-12-25",
            "2024-12-25",
        ],
        "holidays_impact": 0.25,
        # AR(1) coefficient.
        "ar_order": 1,
        "phi": 0.7,
        # Observation noise.
        "seed": 42,
        "observational_noise_sigma": 2.0,
    },
    "model": {
        # Weekly seasonality: 7 seasons, 1 step each.
        "num_seasons": 7,
        "num_steps_per_season": 1,
        "learning_rate": 0.1,
        # Number of VI optimisation steps.
        "num_steps": 200,
    },
}
print(config)

# %% [markdown]
# ## Part 1: Data Generation
#
# We generate a realistic synthetic daily time series that combines:
# - A **linear trend** (slow upward drift)
# - **Weekly seasonality** with stochastic drift
# - **Holiday effects** (additive spikes on Christmas each year)
# - **AR(1) autoregression** to capture temporal dependence
# - **Gaussian observation noise**
#
# Using synthetic data lets us later verify that the model recovers the known
# ground-truth parameters.

# %%
df = tteteuti.generate_time_series_data(config)
_LOG.info(hpandas.df_to_str(df, log_level=logging.INFO))

# %%
df.set_index("ds")["y"].plot(
    title="Synthetic Daily Time Series",
    ylabel="Target variable",
    xlabel="Date",
    figsize=(12, 4),
)

# %%
# Split into train and test sets.
train_mask = (df["ds"] >= config["train_start_date"]) & (
    df["ds"] <= config["train_end_date"]
)
test_mask = (df["ds"] >= config["test_start_date"]) & (
    df["ds"] <= config["test_end_date"]
)
df_train = df[train_mask].reset_index(drop=True)
df_test = df[test_mask].reset_index(drop=True)
_LOG.info("Train rows=%d, Test rows=%d", len(df_train), len(df_test))

# %% [markdown]
# ## Part 2: Model Building
#
# The STS model decomposes the observed series into four interpretable
# components:
#
# | Component | Purpose |
# |---|---|
# | `LocalLinearTrend` | Captures slow-moving level and slope |
# | `Seasonal` (7 seasons) | Weekly day-of-week effects |
# | `Autoregressive` (AR-1) | Short-term temporal dependence |
# | `LinearRegression` | Additive holiday spikes |
#
# The components are summed into a `tfp.sts.Sum` model.

# %%
# Build one-hot holiday indicator matrix for the full date range.
holiday_indicators = tteteuti.build_holiday_indicators(config)
_LOG.info("holiday_indicators shape=%s", holiday_indicators.shape)

# %%
model = tteteuti.build_sts_model(
    df_train["y"].to_numpy(), holiday_indicators, config
)

# %% [markdown]
# ### Variational Inference
#
# We approximate the posterior over model parameters using **Variational
# Inference (VI)**:
# 1. Define a factored surrogate posterior `q(θ)` (one Normal per parameter)
# 2. Maximise the ELBO: `ELBO = E_q[log p(y, θ)] - KL(q || prior)`
# 3. Use the Adam optimiser for gradient-based optimisation

# %%
# Build the variational surrogate posteriors.
variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

# %%
elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=model.joint_distribution(
        observed_time_series=df_train["y"].to_numpy()
    ).log_prob,
    surrogate_posterior=variational_posteriors,
    optimizer=tf_keras.optimizers.Adam(
        learning_rate=config["model"]["learning_rate"]
    ),
    num_steps=config["model"]["num_steps"],
    jit_compile=True,
)
plt.figure(figsize=(8, 4))
plt.plot(elbo_loss_curve, label="ELBO Loss")
plt.xlabel("Optimisation step")
plt.ylabel("ELBO Loss")
plt.title("Variational Optimisation Progress")
plt.legend()
plt.tight_layout()

# %%
# Draw posterior samples.
q_samples_ = variational_posteriors.sample(50)

# %%
# Report inferred parameter values.
_LOG.info("Inferred parameters:")
for param in model.parameters:
    _LOG.info(
        "  %s: %.4f ± %.4f",
        param.name,
        np.mean(q_samples_[param.name], axis=0),
        np.std(q_samples_[param.name], axis=0),
    )

# %% [markdown]
# ## Part 3: Forecasting and Evaluation
#
# With the fitted posterior we:
# 1. Decompose the training signal into its constituent components
# 2. Forecast `num_steps_forecast` steps into the future
# 3. Evaluate using MAE and MSE

# %%
# Decompose training data into components.
component_dists = tfp.sts.decompose_by_component(
    model,
    observed_time_series=df_train["y"].to_numpy(),
    parameter_samples=q_samples_,
)
component_means_ = {k.name: c.mean() for k, c in component_dists.items()}
component_stddevs_ = {k.name: c.stddev() for k, c in component_dists.items()}

# %%
_ = tteteuti.plot_components(
    df_train["ds"], component_means_, component_stddevs_
)

# %%
# Forecast the test period.
forecast_dist = tfp.sts.forecast(
    model=model,
    observed_time_series=df_train["y"].to_numpy(),
    parameter_samples=q_samples_,
    num_steps_forecast=len(df_test),
)
num_samples = 10
forecast_mean = forecast_dist.mean().numpy()[..., 0]
forecast_scale = forecast_dist.stddev().numpy()[..., 0]
forecast_samples = forecast_dist.sample(num_samples).numpy()[..., 0]

# %%
fig, ax = tteteuti.plot_forecast(
    df_test["ds"],
    df_test["y"].to_numpy(),
    forecast_mean,
    forecast_scale,
    forecast_samples,
    title="STS Forecast vs Ground Truth",
)
fig.tight_layout()

# %%
# Residual analysis.
residuals = df_test["y"].to_numpy() - forecast_mean
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(df_test["ds"], residuals, color="steelblue")
axes[0].axhline(0, color="red", ls="--")
axes[0].set_title("Residuals Over Time")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Residual")
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot of Residuals")
plt.tight_layout()

# %%
# Performance metrics.
mae = metrics.mean_absolute_error(df_test["y"].to_numpy(), forecast_mean)
mse = metrics.mean_squared_error(df_test["y"].to_numpy(), forecast_mean)
_LOG.info("MAE=%.4f", mae)
_LOG.info("MSE=%.4f", mse)
