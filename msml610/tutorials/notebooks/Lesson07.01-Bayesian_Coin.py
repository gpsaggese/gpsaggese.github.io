# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [markdown]
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
# Set random seed for reproducibility.
np.random.seed(42)

# Define an interactive function.
def sample_bernoulli(n: int =4, p: float=0.35) -> None:
    data = stats.bernoulli.rvs(p=p, size=n)
    print(f"Bernoulli(p={p}) - {n} realizations:")
    print(data)

# Create interactive sliders.
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

# %%
# Set random seed for reproducibility
np.random.seed(42)

# Define interactive function with type hints
def sample_binomial(n: int = 4, p: float = 0.35, trials: int = 10) -> None:
    """
    Sample n values from a Binomial(trials, p) distribution and print them.
    """
    data: np.ndarray = stats.binom.rvs(n=trials, p=p, size=n)
    print(f"Binomial(n={trials}, p={p}) - {n} realizations:")
    print(data)

# Create interactive sliders
widgets.interact(
    sample_binomial,
    n=widgets.IntSlider(value=4, min=1, max=50, step=1, description='n (samples)'),
    trials=widgets.IntSlider(value=10, min=1, max=100, step=1, description='trials per sample'),
    p=widgets.FloatSlider(value=0.35, min=0.0, max=1.0, step=0.01, description='p (success prob)')
);

# %%
params = {
    #"kind": "cdf",
    "kind": "pdf",
    "pointinterval": False,
    "interval": "hdi",   # Highest density interval.
    #"interval": "eti",  # Equally tailed interval.
    "xy_lim": "auto"
}

#help(pz.Binomial.plot_interactive)

# Probability of k successes on N trials flipping a coin with p success
pz.Binomial(p=0.5, n=5).plot_interactive(**params)

# %%
ut.plot_binomial()

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
# Set random seed for reproducibility.
np.random.seed(42)

# Define an interactive function.
def sample_beta(n: int, a: float, b: float) -> None:
    """
    Sample n values from a Beta(a, b) distribution and print them.
    """
    data: np.ndarray = stats.beta.rvs(a=a, b=b, size=n)
    print(f"Beta(a={a}, b={b}) - {n} realizations:")
    print(data)

# Create interactive sliders.
widgets.interact(
    sample_beta,
    n=widgets.IntSlider(value=4, min=1, max=50, step=1, description='n (samples)'),
    a=widgets.FloatSlider(value=2.0, min=0.1, max=10.0, step=0.1, description='α (shape1)'),
    b=widgets.FloatSlider(value=5.0, min=0.1, max=10.0, step=0.1, description='β (shape2)')
);

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

# %%
ut.plot_beta()

# %% [markdown]
# # Coin Example: Analytical Solution

# %%
# prior=(1, 1) -> uniform
# (20, 20) -> "Gaussian" centered around 0.5
# (1, 4) -> "Exponential" centered around 0

# theta = 0.35, 1.00

# %%
ut.beta_prior_interactive()

# %%
ut.update_prior()

# %% [markdown]
# # Coin Example: Numerical Solution

# %% [markdown]
# - It's a synthetic example!
#   - Assume you know the true value of $\theta$ (not true in general)
#
# - **Workflow**
#   - Model the prior $\theta$ and the likelihood $Y | \theta$
#     \begin{equation*}
#       \begin{cases}
#       \theta \sim \text{Beta}(\alpha = 1, \beta = 1) \\
#       Y \sim \text{Binomial}(n = 1, p = \theta) \\
#       \end{cases}
#     \end{equation*}
#   - Observe samples of the variable $Y$
#   - Run inference
#   - Generate samples of the posterior
#   - Summarize posterior
#      - E.g., Highest-Posterior Density (HPD)
#   - ...

# %%
# Generate data from ground truth model.

np.random.seed(123)
n = 4
# Unknown value.
theta_real = 0.35

# Generate some observational data.
data1 = stats.bernoulli.rvs(p=theta_real, size=n)
data1

# %%
# Build PyMC model matching mathematical model.

with pm.Model() as model1:
    # Prior.
    theta = pm.Beta('theta', alpha=1., beta=1.)
    # Likelihood.
    y = pm.Bernoulli('y', p=theta, observed=data1)
    # (Numerical) Inference to estimate the posterior distribution through samples.
    idata1 = pm.sample(1000, random_seed=123)

# %%
az.plot_trace(idata1);

# %% [markdown]
# - PyMC uses NUTS sampler, computes 4 chains
# - No trace diverges
# - Kernel density estimation (KDE) for posterior (should be Beta)
#

# %%
# #?az.summary

# %%
az.summary(idata1, kind="stats")

# %% [markdown]
# - Traces appear "noisy" and non-diverging (good)
# - Numerical summary of posterior: mean, std dev, HDI
# - $E[\hat{\theta}] \approx 0.324$
# - $\Pr(\hat{\theta} \in [0.031, 0.653]) = 0.94$

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

# %% [markdown]
# # Decision with loss function

# %%
#loss_func = lambda x: ut.squared_loss(x, theta_real)
#loss_func = lambda x: ut.abs_loss(x, theta_real)
#loss_func = lambda x: ut.asymmetric_loss(x, theta_real)
loss_func = lambda x: ut.sin_loss(x, theta_real)

grid = np.linspace(-2.0, 2.0, 50)
ut.plot_loss(grid, loss_func)

# %%
idata1.to_dataframe()[("posterior", "theta")]

# %%
plt.plot(idata1.to_dataframe()[("posterior", "theta")]);

# %%
df = idata1.to_dataframe()[("posterior", "theta")]
df.plot(kind="kde")

# %%
ut.pick_best_theta(idata1)

# %%
ut.pick_best_theta(idata2)
