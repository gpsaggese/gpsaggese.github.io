# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bayesian Coin

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import arviz as az
import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import preliz as pz
from IPython.display import display

# %%
import helpers.hintrospection as hintros
import helpers.htutorial as ut
import L07_01_bayesian_coin_utils as coin_ut

ut.config_notebook()

# %% [markdown]
# # Part 1: Probability Distributions

# %% [markdown]
# ## Cell 1.1: Bernoulli
#
# **Goal**:
# - Build intuition for the Bernoulli distribution, a single trial with two
#   outcomes
# - $X \sim \text{Bernoulli}(p)$ means $P(X = 1) = p$ and
#   $P(X = 0) = 1 - p$, $0 \leq p \leq 1$
#
# **Implementation**: `cell1_1_sample_bernoulli_widget()`
# - Draws `n` samples from `scipy.stats.bernoulli(p)` and prints them

# %%
hintros.print_obj_info(coin_ut.cell1_1_sample_bernoulli_widget)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`n`**: number of samples drawn, 1-50
#   - **`p`**: success probability, 0-1
#
# - Panels
#   - **`data`**: printed array of `n` realizations, each 0 or 1

# %%
coin_ut.cell1_1_sample_bernoulli_widget()

# %% [markdown]
# **Guided usage**
# - Raise `p` from 0 toward 1, leaving `n` fixed
#   - Observe the printed `data` shift from mostly 0s to mostly 1s: a coin
#     flip is $X=1$ if heads, $X=0$ if tails, and `p` is the heads
#     probability

# %% [markdown]
# ## Cell 1.2: Binomial
#
# **Goal**:
# - Build intuition for the binomial distribution, the count of successes
#   over `trials` independent Bernoulli draws
# - $P(X = k) = \binom{\text{trials}}{k} p^k (1 - p)^{\text{trials} - k}$
#
# **Implementation**: `cell1_2_sample_binomial_widget()`
# - Draws `n` samples from `scipy.stats.binom(trials, p)` and prints them

# %%
hintros.print_obj_info(coin_ut.cell1_2_sample_binomial_widget)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`n`**: number of samples drawn, 1-50
#   - **`trials`**: trials per sample, 1-100
#   - **`p`**: success probability, 0-1
#
# - Panels
#   - **`data`**: printed array of `n` realizations, each a count in
#     `[0, trials]`

# %%
coin_ut.cell1_2_sample_binomial_widget()

# %% [markdown]
# **Guided usage**
# - Raise `trials` from 1 toward 100, leaving `p` fixed
#   - Observe the printed counts spread over a wider range: e.g. flipping
#     a fair coin 10 times, the number of heads follows `Binomial(10, 0.5)`

# %% [markdown]
# ## Cell 1.3: Interactive binomial PMF
#
# **Goal**:
# - Explore the binomial PMF's shape directly through `preliz`'s own
#   interactive widget
#
# **Implementation**: `pz.Binomial(p, n).plot_interactive(**params)`
# - `preliz` builds and displays its own sliders for `p` and `n`; `kind` and
#   `interval` control how the distribution is summarized

# %%
# Probability of k successes on N trials flipping a coin with p success.
params = {
    # "kind": "cdf",
    "kind": "pdf",
    "pointinterval": False,
    "interval": "hdi",  # Highest density interval.
    # "interval": "eti",  # Equally tailed interval.
    "xy_lim": "auto",
}
pz.Binomial(p=0.5, n=5).plot_interactive(**params)

# %% [markdown]
# ## Cell 1.4: Binomial distribution grid
#
# **Goal**:
# - Compare the binomial PMF's shape across a grid of `n` and `p` values at
#   once
#
# **Implementation**: `plot_binomial()`
# - Plots `stats.binom(n, p).pmf(x)` for every combination of 3 `n` values
#   and 5 `p` values, one subplot each

# %%
coin_ut.plot_binomial()

# %% [markdown]
# ## Cell 1.5: Beta
#
# **Goal**:
# - Build intuition for the Beta distribution, a continuous distribution on
#   $[0, 1]$ used to model a probability or proportion
# - `a` is the "success" shape parameter, `b` is the "failure" shape
#   parameter: `a` > `b` skews the density toward 1, `a` = `b` centers it
#   on 0.5
#
# **Implementation**: `cell1_5_sample_beta_widget()`
# - Draws `n` samples from `scipy.stats.beta(a, b)` and prints them

# %%
hintros.print_obj_info(coin_ut.cell1_5_sample_beta_widget)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`n`**: number of samples drawn, 1-50
#   - **`a`**: alpha shape parameter, 0.1-10
#   - **`b`**: beta shape parameter, 0.1-10
#
# - Panels
#   - **`data`**: printed array of `n` realizations in `[0, 1]`

# %%
coin_ut.cell1_5_sample_beta_widget()

# %% [markdown]
# **Guided usage**
# - Set `a` well above `b`, then well below it
#   - Observe the printed samples cluster near 1 in the first case and
#     near 0 in the second: `a` > `b` skews toward 1, `a` < `b` skews
#     toward 0

# %% [markdown]
# ## Cell 1.6: Interactive beta PMF
#
# **Goal**:
# - Explore the Beta density's shape directly through `preliz`'s own
#   interactive widget
#
# **Implementation**: `pz.Beta(alpha, beta).plot_interactive(**params)`
# - `preliz` builds and displays its own sliders for `alpha` and `beta`

# %%
params = {
    # "kind": "cdf",
    "kind": "pdf",
    "pointinterval": False,
    "interval": "hdi",  # Highest density interval.
    # "interval": "eti",  # Equal tailed interval.
    "xy_lim": "auto",
}
alpha = 3.0
beta = 1.0
pz.Beta(alpha=alpha, beta=beta).plot_interactive(**params)

# %% [markdown]
# ## Cell 1.7: Beta distribution grid
#
# **Goal**:
# - Compare the Beta density's shape across a grid of `alpha` and `beta`
#   values at once
#
# **Implementation**: `plot_beta()`
# - Plots `stats.beta(a, b).pdf(x)` for every combination of 4 `alpha`
#   values and 4 `beta` values, one subplot each

# %%
coin_ut.plot_beta()

# %% [markdown]
# # Part 2: Coin Example, Analytical Solution

# %% [markdown]
# ## Cell 2.1: Beta prior updated by streaming data
#
# **Goal**:
# - Watch a single Beta prior turn into a posterior as more coin-flip data
#   arrives, one trial count at a time
# - `prior=(1, 1)` is uniform, `(20, 20)` looks Gaussian centered on 0.5,
#   `(1, 4)` looks exponential centered near 0
#
# **Implementation**: `beta_prior_interactive()`
# - Generates binomial counts $y \sim \text{Binomial}(N, \theta)$ at each
#   `n_trials` entry with `_generate_data()`
# - Plots the Beta(`a`+y, `b`+N-y) posterior density for the trial count at
#   `Index`, steppable via the play button

# %%
hintros.print_obj_info(coin_ut.beta_prior_interactive)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`seed`**: random seed for the generated data
#   - **`theta (true)`**: the true, otherwise-unknown success probability
#   - **`n_trials`**: comma-separated trial counts to step through
#   - **`a`, `b`**: the prior's Beta shape parameters
#   - **`Index`**/play button: which `n_trials` entry is shown
#
# - Panels
#   - **`Posterior after N=... trials, y=... heads`**: the posterior
#     density, with `theta (true)` marked
#   - **`Comments`**: current parameters and the resulting posterior
#     distribution

# %%
coin_ut.beta_prior_interactive()

# %% [markdown]
# **Guided usage**
# - Set `a`/`b` to `1, 1` (uniform prior), then step `Index` forward
#   - Observe the posterior widen out then narrow around `theta (true)` as
#     `N` grows
# - Compare `a`/`b` = `1, 1` against `20, 20` at the same `Index`
#   - Observe the strong `(20, 20)` prior barely moves at low `N`, while
#     the uniform prior is dominated by the data almost immediately

# %% [markdown]
# ## Cell 2.2: Updating three priors at once
#
# **Goal**:
# - Compare three different Beta priors, uniform, Gaussian-like, and
#   exponential-like, updated by the same growing dataset side by side
#
# **Implementation**: `update_prior()`
# - Fixes `theta_real = 0.35` and a shared sequence of `n_trials`/observed
#   head counts
# - Plots all three priors' posteriors at each trial count in one grid, so
#   every prior can be compared at the same amount of data

# %%
coin_ut.update_prior()

# %% [markdown]
# - All three posteriors converge toward the same spike at `theta_real` as
#   `N` grows, regardless of how different the priors started
# - The Bayesian update replaces "guess and check" with an explicit rule
#   for how a prior belief and new data combine into a posterior belief

# %% [markdown]
# # Part 3: Coin Example, Numerical Solution
#
# - It's a synthetic example!
#   - Assume you know the true value of $\theta$ (not true in general)
#
# - **Workflow**
#   - Model the prior $\theta$ and the likelihood $Y \mid \theta$
#     \begin{equation*}
#       \begin{cases}
#       \theta \sim \text{Beta}(\alpha = 1, \beta = 1) \\
#       Y \sim \text{Binomial}(n = 1, p = \theta) \\
#       \end{cases}
#     \end{equation*}
#   - Observe samples of the variable $Y$
#   - Run inference
#   - Generate samples of the posterior
#   - Summarize the posterior
#     - E.g., Highest-Posterior Density (HPD)

# %% [markdown]
# ## Cell 3.1: Fitting the model with N=4
#
# **Goal**:
# - Fit the model above to `N=4` synthetic observations, and read the
#   fitted posterior with `arviz`'s trace, summary, rank-bar, and
#   posterior plots
#
# **Implementation**:
# - Builds `model1` with a `Beta(1, 1)` prior and a `Bernoulli` likelihood
#   observing `data1`, then draws posterior samples with `pm.sample()`
#   (PyMC's NUTS sampler, 4 chains)
# - `az.plot_trace()`/`az.summary()`/`az.plot_posterior()` summarize the
#   resulting `idata1`

# %%
# Generate data from ground truth model.
np.random.seed(123)
n = 4
# Unknown value.
theta_real = 0.35
# Generate some observational data.
data1 = stats.bernoulli.rvs(p=theta_real, size=n)
print("data1=", data1)

# %%
# Build PyMC model matching the mathematical model.
with pm.Model() as model1:
    # Prior.
    theta = pm.Beta("theta", alpha=1.0, beta=1.0)
    # Likelihood.
    y = pm.Bernoulli("y", p=theta, observed=data1)
    # (Numerical) Inference to estimate the posterior distribution through samples.
    idata1 = pm.sample(1000, random_seed=123)

# %%
# Trace plot: one row per parameter, KDE (left) and chains (right).
az.plot_trace(idata1)

# %% [markdown]
# - No trace diverges, and the 4 chains mix well
# - The KDE for `theta` should resemble a Beta density

# %%
# Numerical summary: mean, std dev, and HDI.
display(az.summary(idata1, kind="stats"))

# %% [markdown]
# - $E[\hat{\theta}] \approx 0.324$
# - $\Pr(\hat{\theta} \in [0.031, 0.653]) = 0.94$

# %%
# Rank-bar diagnostic: flat bars indicate good mixing across chains.
az.plot_trace(idata1, kind="rank_bars", combined=True)

# %%
# Posterior density with the mean and HDI annotated.
az.plot_posterior(idata1)

# %% [markdown]
# ## Cell 3.2: More data (N=20)
#
# **Goal**:
# - Repeat Cell 3.1 with `N=20` observations, and see the posterior
#   sharpen around `theta_real` as more data arrives
#
# **Implementation**:
# - Same `Beta(1, 1)`/`Bernoulli` model as `model1`, refit on 20
#   observations as `model2`

# %%
np.random.seed(123)
n = 20
# Unknown value.
theta_real = 0.35
# Generate some observational data.
data2 = stats.bernoulli.rvs(p=theta_real, size=n)
print("data2=", data2)

# %%
with pm.Model() as model2:
    # Prior.
    theta = pm.Beta("theta", alpha=1.0, beta=1.0)
    # Likelihood.
    y = pm.Bernoulli("y", p=theta, observed=data2)
    # (Numerical) Inference to estimate the posterior distribution through samples.
    idata2 = pm.sample(1000, random_seed=123)

# %%
display(az.summary(idata2, kind="stats"))

# %%
az.plot_posterior(idata2)

# %% [markdown]
# ## Cell 3.3: Even more data (N=100)
#
# **Goal**:
# - Repeat Cell 3.1 with `N=100` observations, to see the posterior
#   sharpen even further
#
# **Implementation**:
# - Same `Beta(1, 1)`/`Bernoulli` model as `model1`, refit on 100
#   observations as `model3`

# %%
np.random.seed(123)
n = 100
# Unknown value.
theta_real = 0.35
# Generate some observational data.
data3 = stats.bernoulli.rvs(p=theta_real, size=n)
print("data3=", data3)

# %%
with pm.Model() as model3:
    # Prior.
    theta = pm.Beta("theta", alpha=1.0, beta=1.0)
    # Likelihood.
    y = pm.Bernoulli("y", p=theta, observed=data3)
    # (Numerical) Inference to estimate the posterior distribution through samples.
    idata3 = pm.sample(1000, random_seed=123)

# %%
display(az.summary(idata3, kind="stats"))

# %%
az.plot_posterior(idata3)

# %% [markdown]
# ## Cell 3.4: Savage-Dickey ratio
#
# **Goal**:
# - Test the point hypothesis $\theta = 0.5$ against each of the three
#   fitted posteriors via the Savage-Dickey density ratio
#
# **Implementation**:
# - `az.plot_bf()` compares the posterior density at `ref_val=0.5` against
#   a uniform prior, for `idata1`, `idata2`, and `idata3` in turn

# %%
for idata in [idata1, idata2, idata3]:
    az.plot_bf(
        idata,
        var_name="theta",
        prior=np.random.uniform(0, 1, 10000),
        ref_val=0.5,
    )
    plt.xlim(0, 1)

# %% [markdown]
# ## Cell 3.5: ROPE
#
# **Goal**:
# - Test whether $\theta$ plausibly equals 0.5 using a Region Of Practical
#   Equivalence (ROPE) instead of a single point
#
# **Implementation**:
# - `az.plot_posterior()` shades the `[0.45, 0.55]` ROPE and reports the
#   fraction of the posterior it contains, for each fitted `idata`

# %%
for idata in [idata1, idata2, idata3]:
    az.plot_posterior(idata, rope=[0.45, 0.55], ref_val=0.5)
    plt.xlim(0, 1)

# %% [markdown]
# # Part 4: Decision With a Loss Function

# %% [markdown]
# ## Cell 4.1: Minimizing a loss function
#
# **Goal**:
# - Turn a posterior into a point estimate by minimizing expected loss,
#   instead of just reporting the posterior mean
#
# **Implementation**: `plot_loss(grid, loss_func)`
# - Plots the chosen `loss_func` (squared, absolute, asymmetric, or sine)
#   against a candidate-estimate grid

# %%
# loss_func = lambda x: coin_ut.squared_loss(x, theta_real)
# loss_func = lambda x: coin_ut.abs_loss(x, theta_real)
# loss_func = lambda x: coin_ut.asymmetric_loss(x, theta_real)
loss_func = lambda x: coin_ut.sin_loss(x, theta_real)

grid = np.linspace(-2.0, 2.0, 50)
coin_ut.plot_loss(grid, loss_func)

# %% [markdown]
# ## Cell 4.2: Posterior samples as a distribution
#
# **Goal**:
# - Look at the raw posterior draws for `theta` directly: as a sequence,
#   as a trace, and as a density
#
# **Implementation**:
# - Extracts the `theta` column from `idata1.to_dataframe()` and plots it
#   as a trace and a KDE

# %%
theta_draws = idata1.to_dataframe()[("posterior", "theta")]
display(theta_draws)

# %%
plt.plot(theta_draws)

# %%
theta_draws.plot(kind="kde")

# %% [markdown]
# ## Cell 4.3: Picking the best theta
#
# **Goal**:
# - Combine the loss function from Cell 4.1 with the posterior samples to
#   pick the loss-minimizing point estimate, for both the `N=4` and
#   `N=20` posteriors
#
# **Implementation**: `pick_best_theta(idata)`
# - Searches over candidate `theta` values and reports the one minimizing
#   the expected loss under the posterior in `idata`

# %%
coin_ut.pick_best_theta(idata1)

# %%
coin_ut.pick_best_theta(idata2)
