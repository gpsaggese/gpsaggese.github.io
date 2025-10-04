# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md:myst
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Imports

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Install packages

# %%
# !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet jupyterlab-vim)"
# !jupyter labextension enable

# %%
# !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet graphviz)"

# %%
# !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet dataframe_image)"

# %%
# !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet jupyterlab-hide-code)"

# %% [markdown]
# ### Import modules

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import arviz as az
import pandas as pd
import xarray as xr
import pymc as pm
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import preliz as pz

import ipywidgets as widgets
from IPython.display import display

# %%
import msml610_utils as ut

ut.config_notebook()

# %% [markdown] heading_collapsed=true
# # Probability distributions

# %% [markdown]
# ## Bernoulli

# %% [markdown]
# - A **Bernoulli variable** is a random variable that takes only two possible values.
#   - Typically, these values are $1$ (success) and $0$ (failure).
#
# - **Definition:**
#   - $X \sim \text{Bernoulli}(p)$ means $P(X = 1) = p$ and $P(X = 0) = 1 - p$
#   - The parameter $p$ represents the probability of success, where $0 \leq p \leq 1$.
#
# - **Intuition:**
#   - Represents a single trial of an experiment that can result in one of two outcomes.
#   - Examples:
#     - Coin flip: $X = 1$ if heads, $X = 0$ if tails.
#     - Answer correctness: $X = 1$ if correct, $X = 0$ if incorrect.

# %%
np.random.seed(42)

n = 4
p = 0.35

data = stats.bernoulli.rvs(p=p, size=n)
print(data)

# %%
# Set random seed for reproducibility
np.random.seed(42)

# Define interactive function
def sample_bernoulli(n=4, p=0.35):
    data = stats.bernoulli.rvs(p=p, size=n)
    print(f"Bernoulli(p={p}) - {n} realizations:")
    print(data)

# Create interactive sliders
widgets.interact(
    sample_bernoulli,
    n=widgets.IntSlider(value=4, min=1, max=50, step=1, description='n (samples)'),
    p=widgets.FloatSlider(value=0.35, min=0.0, max=1.0, step=0.01, description='p (success prob)')
);

# %% [markdown]
# ## Binomial

# %% [markdown]
# A **binomial random variable** represents the number of successes in a fixed number of independent trials, where each trial has two possible outcomes: success or failure.
#
# - **Parameters:**
#   - $n$: number of trials  
#   - $p$: probability of success in each trial
#
# - **Probability formula:**
#   $$
#   P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}
#   $$
#   where $k = 0, 1, 2, \dots, n$
#
# - **Example:**
#   - If you flip a fair coin 10 times, the number of heads follows a `Binomial(10, 0.5)` distribution

# %% editable=true slideshow={"slide_type": ""}
# #?stats.binom

# %%
np.random.seed(42)

# Create a Binomial.
n = 8
p = 0.01
X = stats.binom(n, p)

# Print k realizations.
k = 4
x = X.rvs(k)
print(x)

# %%
# Set random seed for reproducibility.
np.random.seed(42)

# Define interactive function.
def sample_binomial(n: int, p: float, k: int) -> None:
    X = stats.binom(n, p)
    x = X.rvs(k)
    print(f"Binomial(n={n}, p={p}) - {k} realizations:")
    print(x)
    

# Create interactive controls.
widgets.interact(
    sample_binomial,
    n=widgets.IntSlider(value=8, min=1, max=100, step=1, description='n (trials)'),
    p=widgets.FloatSlider(value=0.01, min=0.0, max=1.0, step=0.01, description='p (success prob)'),
    k=widgets.IntSlider(value=4, min=1, max=50, step=1, description='k (samples)')
);

# %%
ut.plot_binomial()

# %%
params = {
    #"kind": "cdf",
    "kind": "pdf",
    "pointinterval": False,
    "interval": "hdi",   # Highest density interval.
    #"interval": "eti",  # Equal tailed interval.
    "xy_lim": "auto"
}

#help(pz.Binomial.plot_interactive)

# Probability of k successes on N trial flipping a coin with p success
pz.Binomial(p=0.5, n=5).plot_interactive(**params)

# %% [markdown]
# ## Beta
#
# - Continuous prob distribution defined in [0, 1]
# - It is useful to model probability or proportion
#     - E.g., the probability of success in a Bernoulli trial
#
# - alpha represents "success" parameter
# - beta represents "failure" parameter
#     - When alpha is larger than beta the distribution skews toward 1, indicating a higher probability of success
#     - When alpha = beta the distribution is symmetric and centered around 0.5

# %%
np.random.seed(123)

trials = 4
theta = 0.35

# Generate some values.
data = stats.bernoulli.rvs(p=theta_real, size=trials)
print(data)

# %%
ut.plot_beta()

# %%
params = {
    #"kind": "cdf",
    "kind": "pdf",
    "pointinterval": False,
    "interval": "hdi",   # Highest density interval.
    #"interval": "eti",  # Equal tailed interval.
    "xy_lim": "auto"
}

alpha = 3.0
beta = 1.0

pz.Beta(alpha=alpha, beta=beta).plot_interactive(**params)

# %% [markdown]
# # Coin problem: analytical solution

# %%
ut.update_prior()

# %% [markdown]
# ## Coin problem: PyMC solution

# %%
np.random.seed(123)
n = 4
# Unknown value.
theta_real = 0.35

# Generate some observational data.
data1 = stats.bernoulli.rvs(p=theta_real, size=n)
data1

# %%
with pm.Model() as model1:
    # Prior.
    theta = pm.Beta('theta', alpha=1., beta=1.)
    # Likelihood.
    y = pm.Bernoulli('y', p=theta, observed=data1)
    # (Numerical) Inference to estimate the posterior distribution through samples.
    idata1 = pm.sample(1000, random_seed=123)

# %%
az.plot_trace(idata1);

# %%
# #?az.summary

# %%
az.summary(idata1, kind="stats")

# %%
az.plot_trace(idata1, kind="rank_bars", combined=True);

# %%
az.plot_posterior(idata1);

# %% [markdown]
# ## More data

# %%
np.random.seed(123)
n = 20
# Unknown value.
theta_real = 0.35

# Generate some observational data.
data2 = stats.bernoulli.rvs(p=theta_real, size=n)
data2

# %%
with pm.Model() as model2:
    # Prior.
    theta = pm.Beta('theta', alpha=1., beta=1.)
    # Likelihood.
    y = pm.Bernoulli('y', p=theta, observed=data2)
    # (Numerical) Inference to estimate the posterior distribution through samples.
    idata2 = pm.sample(1000, random_seed=123)

# %%
az.summary(idata2, kind="stats")

# %%
az.plot_posterior(idata2);

# %% [markdown]
# ## Even more data

# %%
np.random.seed(123)
n = 100
# Unknown value.
theta_real = 0.35

# Generate some observational data.
data3 = stats.bernoulli.rvs(p=theta_real, size=n)
data3

# %%
with pm.Model() as model3:
    # Prior.
    theta = pm.Beta('theta', alpha=1., beta=1.)
    # Likelihood.
    y = pm.Bernoulli('y', p=theta, observed=data3)
    # (Numerical) Inference to estimate the posterior distribution through samples.
    idata3 = pm.sample(1000, random_seed=123)

# %%
az.summary(idata3, kind="stats")

# %%
az.plot_posterior(idata3);

# %% [markdown]
# ## Savage-Dickey ratio

# %%
for idata in [idata1, idata2, idata3]:
    az.plot_bf(idata, var_name="theta", prior=np.random.uniform(0, 1, 10000), ref_val=0.5);
    plt.xlim(0, 1);

# %% [markdown]
# ## ROPE

# %%
for idata in [idata1, idata2, idata3]:
    az.plot_posterior(idata, rope=[0.45, .55], ref_val=0.5)
    plt.xlim(0, 1);
