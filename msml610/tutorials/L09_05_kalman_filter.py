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

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import msml610_utils as ut

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %%
import helpers.hio as hio
import L09_05_kalman_filter_utils as time_ut

dst_dir = "figures"
hio.create_dir(dst_dir, incremental=True)
# # cp msml610/tutorials/figures/*.png msml610/lectures_source/figures

# %% [markdown]
# # Cell 1: Sum and Product of Gaussians

# %%
from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'N(mu={s[0]:.3f}, sigma^2={s[1]:.3f})'

x = gaussian(3.4, 10.1)
print(x)
print("x.mean=", x.mean)
print("x.var=", x.var)


# %% [markdown]
# ## Cell 1.1: Sum of Gaussians
# - Given two Gaussians $X$ and $Y$
#   - $X \sim Normal(\mu_1, \sigma_1^2)$
#   - $Y \sim Normal(\mu_2, \sigma_2^2)$
# - For correlated Gaussians with correlation coefficient $\rho$, the sum $Z = X + Y$ is a Gaussian $Normal(\mu, \sigma^2)$ with:
#   - $\mu = \mu_1 + \mu_2$
#   - $\sigma^2 = \sigma_1^2 + \sigma_2^2 + 2\rho\sigma_1\sigma_2$
# - **Interpretation:**
#   - The mean is the sum of the means (by linearity)
#   - For independent Gaussians ($\rho = 0$), the variance is the sum of variances
#   - Positive correlation increases variance, negative correlation decreases it

# %%
def gaussian_sum(g1, g2):
    return gaussian(g1.mean + g2.mean, g1.var + g2.var)


# %%
x = gaussian(10, 0.2 ** 2)
y = gaussian(15, 0.7 ** 2)
print(gaussian_sum(x, y))

# %%

# %%
# Interactive exploration of sum of Gaussians with correlation.
time_ut.cell1_1_plot_gaussian_sum()


# %% [markdown]
# ## Cell 1.2: Product of Gaussians
# - Given two Gaussians $X$ and $Y$
#   - $X \sim Normal(\mu_1, \sigma_1^2)$
#   - $Y \sim Normal(\mu_2, \sigma_2^2)$
# - The product $Z = X \cdot Y$ (PDF multiplication) is a Gaussian $Normal(\mu, \sigma^2)$ with:
#   - $\mu = \frac{\mu_1 \sigma_2^2 + \mu_2 \sigma_1^2}{\sigma_1^2 + \sigma_2^2}$
#   - $\sigma^2 = \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}$
# - **Interpretation:**
#   - Reduces variance by incorporating more information
#   - If one Gaussian $X$ is narrower (more accurate), result leans towards $X$
#   - If two Gaussians are similar (measures corroborate), result becomes more certain

# %%
def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)


# %%
z = gaussian(10.1)

# %%
# Interactive exploration of product of Gaussians.
time_ut.cell1_2_plot_gaussian_product()

# %% [markdown]
# # Cell 2

# %% [markdown]
# - The intuition is the same as the discrete case
#
# - There is a cycle of prediction and updates
#     1) Predict: prior = x_est using system model
#     2) Update: posterior = likelihood * prior
#
# - Create prior (using current estimate and system model)
#   - `prior = predict(x, process_model)`
# - Create likelihood (using measurement)
#   - `likelihood = gaussian(z, sensor_var)`
# - Update belief using prior and likelihood
#   - `x = update(prior, likelihood)`

# %% [markdown]
# - Let's assume that the dog moves in the hallway, back and forth
#   - It's not circular
# - We have a sensor that measures the distance of the dog from one extreme

# %% [markdown]
# We can use Newton's equation of motion to compute the position of the dog, based on current position and velocity
#
# $$\overline{x}_k = x_{k-1} + v_k \Delta_t$$
#
# - $x_{k-1}$ has uncertainty quantified by a Gaussian
# - $v_k$ has also uncertainty quantified by a Gaussian
#
# We can compute the sum of two Gaussians in terms of mean and uncertainty
# - It makes sense since we know that uncertainy becomes larger 

# %%
- The likelihood $z | x$ is the probability of measures given the current state



# %%
