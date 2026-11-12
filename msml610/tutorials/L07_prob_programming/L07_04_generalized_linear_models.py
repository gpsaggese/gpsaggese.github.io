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
# # Generalized linear models
#
# - This notebook fits generalized linear models with `pymc`, changing the
#   likelihood and the link to match the type of the data
# - The pedagogical arc:
#   - Linear regression on synthetic and on bike rental data
#   - Counting: a Negative Binomial model and its posterior predictive checks
#   - Robust regression on Anscombe's outlier dataset
#   - Logistic regression on the iris data, and its decision boundary
#   - Variable variance: a model whose variance depends on age
#   - Multiple linear regression, and a Negative Binomial model with two
#     predictors for the rented bikes

# %%
# !pip install -q dataframe_image==0.2.7 graphviz==0.21

import dataframe_image

print("dataframe_image version: ", dataframe_image.__version__)

import graphviz

print("graphviz version: ", graphviz.__version__)

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats
import seaborn as sns
import xarray as xr

# %%
import helpers.hnotebook as hnotebook
import helpers.htutorial as ut

import L07_04_generalized_linear_models_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %%
dir_name = "./L07_data"
# !ls $dir_name

# %% [markdown]
# # Part 1: Linear Regression

# %% [markdown]
# ## Cell 1.1: Synthetic example
#
# **Goal**
# - Fit a Bayesian linear regression on synthetic data with a known
#   `alpha`/`beta`/noise, so the recovered posterior can be checked against
#   the ground truth
#
# **Implementation**
# - `alpha ~ Normal(0, 10)`, `beta ~ Normal(0, 1)`, `sigma ~ HalfCauchy(5)`,
#   `y ~ Normal(alpha + beta*x, sigma)`

# %%
np.random.seed(1)

# Number of samples.
N = 100

# Ground-truth parameters.
alpha_real = 2.5
beta_real = 0.9
sigma_eps_real = 0.5

# Generate data.
x = np.random.normal(10, 1, N)
y_real = alpha_real + beta_real * x

# Add noise.
eps_real = np.random.normal(0, sigma_eps_real, size=N)
y = y_real + eps_real

# %%
sns.scatterplot(x=x, y=y_real, label="noiseless")
sns.scatterplot(x=x, y=y, label="observed")

# %%
df = pd.DataFrame({"X": x, "Y": y})
sns.regplot(
    x="X",
    y="Y",
    data=df,
    scatter_kws={"color": "blue"},
    line_kws={"color": "red"},
)

# %%
with pm.Model() as model_g:
    # Priors, wide relative to the ground-truth values above.
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=1)
    sigma = pm.HalfCauchy("sigma", 5)
    # Linear predictor and Normal likelihood.
    mu = pm.Deterministic("mu", alpha + beta * x)
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=y)
    idata_g = pm.sample(2000, tune=1000)

# %%
pm.model_to_graphviz(model_g)

# %%
az.plot_trace(idata_g, var_names=["alpha", "beta", "sigma"])

# %%
display(az.summary(idata_g, var_names="alpha beta sigma".split(), kind="stats"))

# %% [markdown]
# ## Cell 1.2: Bike rental example
#
# **Goal**
# - Fit the same linear-regression pattern to real data: predicting bike
#   rentals from temperature
#
# **Implementation**
# - Same `alpha`/`beta`/`sigma` priors as `model_g`, refit on
#   `bikes.temperature`/`bikes.rented`
# - `utils.plot_data_and_model()` overlays the fitted mean and 50%/94%
#   posterior-predictive bands on the raw data

# %%
bikes = pd.read_csv(dir_name + "/bikes.csv")
bikes.plot(x="temperature", y="rented", figsize=(12, 3), kind="scatter")

# %%
display(bikes.head())

# %%
with pm.Model() as model_lb:
    alpha = pm.Normal("alpha", mu=0, sigma=100)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfCauchy("sigma", 10)
    # Linear predictor and Normal likelihood.
    mu = pm.Deterministic("mu", alpha + beta * bikes.temperature)
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=bikes.rented)
    idata_lb = pm.sample()

# %%
# Plot all the vars, excluding mu.
az.plot_posterior(idata_lb, var_names=["~mu"])

# %%
# Sample from the posterior.
posterior = az.extract(idata_lb, num_samples=50)

# Create 50 equally-spaced points from min to max temperature.
x_plot = xr.DataArray(
    np.linspace(bikes.temperature.min(), bikes.temperature.max(), 50),
    dims="plot_id",
)
# Compute the expected value of the model for these points.
mean_line = posterior["alpha"].mean() + posterior["beta"].mean() * x_plot
# Compute 50 lines using the posterior.
lines = posterior["alpha"] + posterior["beta"] * x_plot
print("x_plot.shape=", x_plot.shape)
print("mean_line.shape=", mean_line.shape)

# %%
idata_lb_pp = pm.sample_posterior_predictive(idata_lb, model=model_lb)
mean_line = idata_lb.posterior["mu"].mean(("chain", "draw"))

# %%
# Plot the data.
# zorder is to plot behind the line.
plt.plot(bikes.temperature, bikes.rented, "C2.", zorder=-3)

# Plot the 50 models from the posterior.
# lines.T.values are the 50 lines.
lines_ = plt.plot(x_plot, lines.T.values, c="C1", alpha=0.2, label="lines")
# Remove the label for all the lines but the first one.
plt.setp(lines_[1:], label="_")

plt.xlabel("temp")
plt.ylabel("rented bikes")
_ = plt.legend()

# %%
utils.plot_data_and_model(bikes, idata_lb_pp, mean_line)

# %% [markdown]
# # Part 2: Counting

# %% [markdown]
# ## Cell 2.1: Fitting a Negative Binomial model
#
# **Goal**
# - Refit the bike-rental data with a Negative Binomial likelihood, better
#   suited to non-negative counts than the Gaussian model in Part 1
#
# **Implementation**
# - `mu = exp(alpha + beta*temperature)` keeps the mean positive;
#   `y ~ NegativeBinomial(mu, sigma)`, where `sigma` controls the variance

# %%
np.random.seed(42)
with pm.Model() as model_neg:
    alpha = pm.Normal("alpha", mu=0, sigma=100)
    beta = pm.Normal("beta", mu=0, sigma=10)
    # We use exp to keep the mean positive.
    mu = pm.Deterministic("mu", pm.math.exp(alpha + beta * bikes.temperature))
    # NegativeBinomial has an extra param, sigma, to control the variance.
    sigma = pm.HalfNormal("sigma", 10)
    y_pred = pm.NegativeBinomial(
        "y_pred", mu=mu, alpha=sigma, observed=bikes.rented
    )
    #
    idata_neg = pm.sample()
    idata_neg = pm.sample_posterior_predictive(
        idata_neg, extend_inferencedata=True
    )

# %%
pm.model_to_graphviz(model_neg)

# %%
az.plot_trace(idata_neg, var_names=["~mu"])

# %% [markdown]
# ## Cell 2.2: Comparing posterior predictive checks
#
# **Goal**
# - Compare the Negative Binomial fit against the Gaussian fit from Part 1,
#   on both the fitted curve and the posterior-predictive distribution

# %%
utils.plot_data_and_model(bikes, idata_neg, mean_line)

# %%
az.plot_ppc(idata_lb_pp, num_pp_samples=200, alpha=0.1, mean=False)

# %%
az.plot_ppc(idata_neg, num_pp_samples=200, alpha=0.1, mean=False)

# %% [markdown]
# # Part 3: Robust Regression

# %% [markdown]
# ## Cell 3.1: Anscombe's outlier dataset
#
# **Goal**
# - Look at a small dataset with one clear outlier, used next to contrast
#   a non-robust and a robust regression

# %%
ans = pd.read_csv(dir_name + "/anscombe_3.csv")
display(ans.head())

# %%
ans.plot("x", "y", kind="scatter")

# %% [markdown]
# ## Cell 3.2: Non-robust vs robust fit
#
# **Goal**
# - Fit an ordinary least-squares line (sensitive to the outlier) and a
#   Bayesian Student-t regression (robust to it), and compare both fits
#
# **Implementation**
# - OLS via `scipy.stats.linregress`
# - Student-t model: `nu ~ Exponential(1/29) + 1` (shifted so `nu >= 1`),
#   `y ~ StudentT(alpha + beta*x, sigma, nu)`

# %%
import scipy

beta_c, alpha_c, *_ = scipy.stats.linregress(ans.x, ans.y)

_, ax = plt.subplots()
ax.plot(ans.x, (alpha_c + beta_c * ans.x), "C0:", label="non-robust")
ax.plot(ans.x, ans.y, "C0o")
ut.save_ax(ax, "Lesson07_Non_robust_regression1.png")

# %%
with pm.Model() as model_t:
    # Alpha is Normal, centered around the mean of the y data.
    alpha = pm.Normal("alpha", mu=ans.y.mean(), sigma=1)
    # Beta is standard Normal(0, 1).
    beta = pm.Normal("beta", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", 5)
    # The Exponential puts too much weight close to 0, so shift by 1.
    nu_ = pm.Exponential("nu_", 1 / 29)
    nu = pm.Deterministic("nu", nu_ + 1)

    # Linear predictor and Student-t likelihood.
    mu = pm.Deterministic("mu", alpha + beta * ans.x)
    y_pred = pm.StudentT("y_pred", mu=mu, sigma=sigma, nu=nu, observed=ans.y)
    idata_t = pm.sample(2000, tune=2000)
    idata_t = pm.sample_posterior_predictive(idata_t, extend_inferencedata=True)

# %%
ut.save_dot(model_t, "Lesson07_Robust_regression_model")

# %%
var_names = "alpha beta sigma nu".split()
az.plot_trace(idata_t, var_names=var_names)
display(az.summary(idata_t, var_names=var_names, round_to=2, kind="stats"))

# %%
_, ax = plt.subplots()

# Non-robust.
ax.plot(ans.x, (alpha_c + beta_c * ans.x), "C0:", label="non-robust")
ax.plot(ans.x, ans.y, "C0o")

# Robust.
alpha_m = idata_t.posterior["alpha"].mean(("chain", "draw"))
beta_m = idata_t.posterior["beta"].mean(("chain", "draw"))

x_plot = xr.DataArray(np.linspace(ans.x.min(), ans.x.max(), 50), dims="plot_id")
ax.plot(x_plot, alpha_m + beta_m * x_plot, c="C0", label="robust")
az.plot_hdi(ans.x, az.hdi(idata_t.posterior["mu"])["mu"].T, ax=ax)
ax.set_xlabel("x")
ax.set_ylabel("y", rotation=0)
ax.legend(loc=2)
ut.save_ax(ax, "Lesson07_Non_robust_regression2")

# %% [markdown]
# ## Cell 3.3: Posterior predictive check
#
# **Goal**
# - Check the robust model's posterior-predictive fit against the data

# %%
# Posterior predictive check.
ppc = pm.sample_posterior_predictive(
    idata_t,
    model=model_t,
    random_seed=2,
)
az.plot_ppc(idata_t, mean=True, num_pp_samples=100)
plt.xlim(0, 20)

# %% [markdown]
# # Part 4: Logistic Regression

# %% [markdown]
# ## Cell 4.1: Iris data for two species
#
# **Goal**
# - Prepare a 2-class subset of the iris dataset and one feature, sepal
#   length, to classify `setosa` vs `versicolor`

# %%
iris = pd.read_csv(dir_name + "/iris.csv")
display(iris.head())
ut.save_df(iris.head(), "Lesson07_Logistic_regression_df.png")

# %%
# Filter the dataframe keeping only 2 values for species.
df = iris.query("species == ('setosa', 'versicolor')")
display(df.head())

# %%
# Get the predicted variable.
y_0 = pd.Categorical(df["species"]).codes
print("y_0=", y_0)

# %%
# Get the sepal length as feature, centered.
x_n = "sepal_length"
x_0 = df[x_n].values
x_c = x_0 - x_0.mean()

# %% [markdown]
# ## Cell 4.2: Fitting a logistic regression
#
# **Goal**
# - Fit a Bayesian logistic regression, and derive the decision boundary
#   `bd` where the predicted probability crosses 0.5
#
# **Implementation**
# - `theta = sigmoid(alpha + beta*x_c)`, `y ~ Bernoulli(theta)`;
#   `bd = -alpha / beta` is the boundary where `theta = 0.5`

# %%
with pm.Model() as model_lrs:
    # Linear part.
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=5)
    mu = alpha + x_c * beta
    # Sigmoid link.
    theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
    # Likelihood.
    yl = pm.Bernoulli("yl", p=theta, observed=y_0)
    # Decision boundary: theta = 0.5.
    bd = pm.Deterministic("bd", -alpha / beta)
    #
    idata_lrs = pm.sample(random_seed=123)

# %%
ut.save_dot(model_lrs, "Lesson07_Logistic_regression_model.png")

# %%
var_names = ["~bd", "~theta"]
display(az.summary(idata_lrs, var_names=var_names, round_to=2, kind="stats"))

# %%
ax = az.plot_trace(idata_lrs, var_names=var_names)
ut.save_fig(ax, "Lesson07_Logistic_regression_result.png")

# %% [markdown]
# ## Cell 4.3: Visualizing the decision boundary
#
# **Goal**
# - Plot the fitted sigmoid, the decision boundary and its HDI, and the
#   raw data together

# %%
posterior = idata_lrs.posterior
theta = posterior["theta"].mean(("chain", "draw"))
idx = np.argsort(x_c)

# Plot the model.
_, ax = plt.subplots()
ax.plot(x_c[idx], theta[idx], color="C0", lw=2)

# Plot the decision boundary and its HDI.
ax.vlines(posterior["bd"].mean(("chain", "draw")), 0, 1, color="C2", zorder=0)
bd_hdi = az.hdi(posterior["bd"])
ax.fill_betweenx(
    [0, 1], bd_hdi["bd"][0], bd_hdi["bd"][1], color="C2", alpha=0.6, lw=0
)

# Plot the data.
ax.scatter(x_c, np.random.normal(y_0, 0.02), marker=".")
az.plot_hdi(x_c, posterior["theta"], color="C0", ax=ax)
ut.save_ax(ax, "Lesson07_Logistic_regression_result2.png")

# %% [markdown]
# # Part 5: Variable Variance

# %% [markdown]
# ## Cell 5.1: Babies growth data
#
# **Goal**
# - Look at babies' length by age in months: the spread grows with age, so
#   a constant-variance model would be a poor fit

# %%
data = pd.read_csv(dir_name + "/babies.csv")
data.columns = ["month", "length"]
data.plot.scatter("month", "length")
display(data.head())

# %%
ax = data.plot.scatter("month", "length")
ut.save_ax(ax, "Lesson07_Variable_variance_data.png")

# %% [markdown]
# ## Cell 5.2: Fitting a model with variance as a function of age
#
# **Goal**
# - Model both the mean and the standard deviation as functions of `month`,
#   instead of assuming constant variance
#
# **Implementation**
# - `mu = alpha + beta*sqrt(month)`, `sigma = gamma + delta*month`,
#   `y ~ Normal(mu, sigma)`

# %%
with pm.Model() as model_vv:
    # Create a shared variable so the data can change after the model is built.
    x_shared = pm.Data("x_shared", data.month.values.astype(float))
    # Linear model for the mean is a function of sqrt(month).
    alpha = pm.Normal("alpha", sigma=10)
    beta = pm.Normal("beta", sigma=10)
    mu = pm.Deterministic("mu", alpha + beta * x_shared**0.5)
    # Linear model for the std dev.
    gamma = pm.HalfNormal("gamma", sigma=10)
    delta = pm.HalfNormal("delta", sigma=10)
    sigma = pm.Deterministic("sigma", gamma + delta * x_shared)
    # Fit.
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=data.length)
    #
    idata_vv = pm.sample(random_seed=123)

# %%
pm.model_to_graphviz(model_vv)
ut.save_dot(model_vv, "Lesson07_Variable_variance_model.png")

# %% [markdown]
# ## Cell 5.3: Visualizing the fitted mean and variance bands
#
# **Goal**
# - Plot the fitted mean with 1 and 2 standard-deviation bands against the
#   raw data, to see the variance growing with age

# %%
# Plot the data.
plt.plot(data.month, data.length, "C0.", alpha=0.1)

# Compute the posterior mean and sigma.
posterior = az.extract(idata_vv)
mu_m = posterior["mu"].mean("sample").values
sigma_m = posterior["sigma"].mean("sample").values

# Plot 1 and 2 std dev of the model.
plt.plot(data.month, mu_m, c="k")
plt.fill_between(
    data.month, mu_m + 1 * sigma_m, mu_m - 1 * sigma_m, alpha=0.6, color="C1"
)
plt.fill_between(
    data.month, mu_m + 2 * sigma_m, mu_m - 2 * sigma_m, alpha=0.4, color="C1"
)
ut.save_plt("Lesson07_Variable_variance_result.png")

# %% [markdown]
# # Part 6: Multiple Linear Regression

# %% [markdown]
# ## Cell 6.1: Synthetic multi-feature data
#
# **Goal**
# - Generate synthetic data with 2 independent features, to fit a multiple
#   linear regression against a known ground truth
#
# **Implementation** `utils.scatter_plot(x, y)`
# - Plots `y` against each feature, plus the two features against each
#   other, in a 1xN layout

# %%
np.random.seed(314)

N = 100
alpha_real = 2.5
beta_real = [0.9, 1.5]
eps_stddev_real = 0.5
eps_real = np.random.normal(0, eps_stddev_real, size=N)

# Independent variables, means [10, 2] and std devs [1, 1.5].
X = np.array([np.random.normal(i, j, N) for i, j in zip([10, 2], [1, 1.5])]).T
X_mean = X.mean(axis=0, keepdims=True)
X_centered = X - X_mean

# Create samples.
y = alpha_real + np.dot(X, beta_real) + eps_real

# %%
utils.scatter_plot(X_centered, y)
ut.save_plt("Lesson07_Multiple_linear_regression3.png")

# %% [markdown]
# ## Cell 6.2: Fitting the multiple regression model
#
# **Goal**
# - Fit the multiple linear regression, centering `X` for a more stable
#   sampler geometry, then recovering the original-scale intercept
#
# **Implementation**
# - `alpha_tmp ~ Normal(0, 10)`, `beta ~ Normal(0, 1)` (length 2), fit on
#   `X_centered`; `alpha = alpha_tmp - X_mean . beta` undoes the centering

# %%
with pm.Model() as model_mlr:
    alpha_tmp = pm.Normal("alpha_tmp", mu=0, sigma=10)
    # Beta is a length-2 vector, one weight per feature.
    beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
    eps = pm.HalfCauchy("eps", 5)
    # Linear predictor on the centered features.
    mu = alpha_tmp + pm.math.dot(X_centered, beta)
    # Recover the original-scale intercept.
    alpha = pm.Deterministic("alpha", alpha_tmp - pm.math.dot(X_mean, beta))

    # Likelihood.
    y_pred = pm.Normal("y_pred", mu=mu, sigma=eps, observed=y)
    idata_mlr = pm.sample(2000)

# %%
ut.save_dot(model_mlr, "Lesson07_Multiple_linear_regression_model.png")
pm.model_to_graphviz(model_mlr)

# %% [markdown]
# ## Cell 6.3: Inspecting the fit
#
# **Goal**
# - Check the sampling trace and numerical summary against the known
#   ground-truth `alpha_real`/`beta_real`

# %%
var_names = ["alpha", "beta", "eps"]
az.plot_trace(idata_mlr, var_names=var_names)
ut.save_plt("Lesson07_Multiple_linear_regression_results1.png")

# %%
mlr_summary = az.summary(
    idata_mlr, var_names=var_names, round_to=2, kind="stats"
)
ut.save_df(mlr_summary, "Lesson07_Multiple_linear_regression_results2.png")
display(mlr_summary)

# %% [markdown]
# # Part 7: Rented Bikes With Two Predictors

# %% [markdown]
# ## Cell 7.1: Fitting a Negative Binomial model with two predictors
#
# **Goal**
# - Extend Part 2's single-predictor Negative Binomial model with a second
#   predictor, `hour`, to see if it improves the fit
#
# **Implementation**
# - `mu = exp(alpha + beta0*temperature + beta1*hour)`,
#   `y ~ NegativeBinomial(mu, sigma)`

# %%
with pm.Model() as model_mlb:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta0 = pm.Normal("beta0", mu=0, sigma=10)
    beta1 = pm.Normal("beta1", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", 10)
    mu = pm.Deterministic(
        "mu", pm.math.exp(alpha + beta0 * bikes.temperature + beta1 * bikes.hour)
    )
    _ = pm.NegativeBinomial("y_pred", mu=mu, alpha=sigma, observed=bikes.rented)
    #
    idata_mlb = pm.sample()

# %%
pm.model_to_graphviz(model_mlb)
ut.save_dot(
    model_mlb, "Lesson07_Multiple_linear_regression_model_RentedBikes_model.png"
)

# %% [markdown]
# ## Cell 7.2: Inspecting the fit
#
# **Goal**
# - Check the sampling trace and numerical summary for the two-predictor
#   model

# %%
var_names = ["alpha", "beta0", "beta1", "sigma"]
az.plot_trace(idata_mlb, var_names=var_names)
ut.save_plt(
    "Lesson07_Multiple_linear_regression_model_RentedBikes_model_trace.png"
)

# %%
mlb_summary = az.summary(
    idata_mlb, var_names=var_names, round_to=2, kind="stats"
)
display(mlb_summary)
