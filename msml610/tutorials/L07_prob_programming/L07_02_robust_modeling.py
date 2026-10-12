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
# # Robust modeling
#
# - This notebook models chemical-shift data that looks Gaussian but has a
#   couple of outliers, to show how a Student-t likelihood makes the fit more
#   robust than a Gaussian one
# - The pedagogical arc:
#   - Loading and visualizing the data
#   - Fitting a Gaussian model, inspecting it, and checking it with a
#     posterior predictive check
#   - The tails of the Student-t distribution
#   - Fitting a Student-t model and checking it the same way

# %%
# !pip install -q graphviz==0.21

import graphviz

print("graphviz version: ", graphviz.__version__)

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import arviz as az
import numpy as np
import pymc as pm

# %%
import helpers.hnotebook as hnotebook
import helpers.htutorial as ut

import L07_02_robust_modeling_utils as utils

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
# # Part 1: Chemical Shift

# %% [markdown]
# ## Cell 1.1: Loading the data
#
# **Goal**
# - Load the chemical-shift measurements used throughout this notebook

# %%
data = np.loadtxt(f"{dir_name}/chemical_shifts.csv")
print("len(data)=", len(data))
print("data=", data)

# %% [markdown]
# ## Cell 1.2: Visualizing the raw distribution
#
# **Goal**
# - Look at the data's shape before fitting anything: it looks Gaussian,
#   but with a couple of outliers

# %%
az.plot_kde(data, rug=True)
ut.process_figure("Chemical shift")

# %% [markdown]
# ## Cell 1.3: Fitting a Gaussian model
#
# **Goal**
# - Fit the data with the simplest reasonable model, a single Gaussian,
#   as the baseline the Student-t model in Cell 1.7 will be compared
#   against
#
# **Implementation**
# - `mu ~ Uniform(40, 70)` (wider than the data), `sigma ~ HalfNormal(10)`,
#   `y ~ Normal(mu, sigma)` observing `data`

# %%
with pm.Model() as model_g:
    # The mean is Uniform in [40, 70] (wider than the data).
    mu = pm.Uniform("mu", lower=40, upper=70)
    # The std dev is half normal with a wide scale relative to the data.
    sigma = pm.HalfNormal("sigma", sigma=10)
    # The model is N(mu, sigma).
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
    # Sample.
    idata_g = pm.sample(1000)

# %%
pm.model_to_graphviz(model_g)

# %% [markdown]
# ## Cell 1.4: Inspecting the Gaussian fit
#
# **Goal**
# - Check the sampling trace, the joint posterior of `mu`/`sigma`, and the
#   numerical summary for the Gaussian fit
#
# **Implementation** `az.plot_trace()`, `az.plot_pair()`, `az.summary()`

# %%
# There are 4 traces for 2 variables.
az.plot_trace(idata_g)

# %%
# The posterior distribution of the params is bi-dimensional: mu and sigma.
az.plot_pair(idata_g, kind="kde", marginals=True)

# %%
# Report a summary of the inference.
display(az.summary(idata_g, kind="stats").round(2))

# %% [markdown]
# ## Cell 1.5: Posterior predictive check for the Gaussian model
#
# **Goal**
# - Compare the observed data's density against the Gaussian model's
#   posterior-predictive samples
#
# **Implementation** `pm.sample_posterior_predictive()`, `az.plot_ppc()`
# - Black: KDE of the observed data
# - Blue: KDEs of the posterior predictive samples
# - Orange: KDE of the posterior predictive mean

# %%
# Compute 100 posterior predictive samples.
y_pred_g = pm.sample_posterior_predictive(idata_g, model=model_g)

# %%
az.plot_ppc(y_pred_g, mean=True, num_pp_samples=100)

# %% [markdown]
# ## Cell 1.6: The Student-t distribution's tails
#
# **Goal**
# - Show that the Student-t distribution approaches a Gaussian as its
#   degrees of freedom `nu` grow, but keeps much heavier tails for small
#   `nu`, which is what makes it robust to outliers
#
# **Implementation** `plot_student_t_sweep(dfs, figsize=None)`
# - Plots the Student-t PDF for `nu` in `{0.1, 0.5, 1, 2, 5, 10, 30}` next
#   to the Gaussian (the `nu -> infinity` limit)

# %%
utils.plot_student_t_sweep()
ut.process_figure("Chap7: Student-t")

# %% [markdown]
# ## Cell 1.7: Fitting a Student-t model
#
# **Goal**
# - Refit the same data with a Student-t likelihood instead of a Gaussian,
#   letting the model learn how heavy-tailed the noise should be
#
# **Implementation**
# - Same `mu`/`sigma` priors as `model_g`, plus `nu ~ Exponential(1/30)`
#   (a `nu` around 30 is close to Gaussian); `y ~ StudentT(mu, sigma, nu)`

# %%
# Use a Student-t model.
with pm.Model() as model_t:
    mu = pm.Uniform("mu", 40, 75)
    sigma = pm.HalfNormal("sigma", sigma=10)
    # A Student-t with nu = 30 is close to a Gaussian.
    nu = pm.Exponential("nu", 1 / 30)
    #
    y = pm.StudentT("y", mu=mu, sigma=sigma, nu=nu, observed=data)
    idata_t = pm.sample(1_000)

# %% [markdown]
# ## Cell 1.8: Inspecting the Student-t fit
#
# **Goal**
# - Check the sampling trace and numerical summary for the Student-t fit,
#   the same diagnostics run on the Gaussian fit in Cell 1.4

# %%
az.plot_trace(idata_t)

# %%
display(az.summary(idata_t, kind="stats").round(2))

# %% [markdown]
# ## Cell 1.9: Posterior predictive check for the Student-t model
#
# **Goal**
# - Compare the Student-t model's posterior-predictive fit against the
#   data, and against the Gaussian model's fit from Cell 1.5
#
# **Implementation** `pm.sample_posterior_predictive()`, `az.plot_ppc()`

# %%
# Compute 100 posterior predictive samples.
y_ppc_t = pm.sample_posterior_predictive(idata_t, model_t)

# %%
ax = az.plot_ppc(y_ppc_t, num_pp_samples=100, mean=True)
ax.set_xlim(40, 70)
