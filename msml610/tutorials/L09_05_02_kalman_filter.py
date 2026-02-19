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
x = time_ut.Gaussian(3.4, 10.1)
print(x)
print("x.mean=", x.mean)
print("x.var=", x.var)


# %% [markdown]
# ## Cell 1.1: Sum of Gaussians
# - Given two Gaussians $X$ and $Y$
#   $$X \sim Normal(\mu_1, \sigma_1^2)$$
#   $$Y \sim Normal(\mu_2, \sigma_2^2)$$
# - For correlated Gaussians with correlation coefficient $\rho$, the sum $Z = X + Y$ is a Gaussian $Normal(\mu, \sigma^2)$ with:
#   $$\mu = \mu_1 + \mu_2$$
#   $$\sigma^2 = \sigma_1^2 + \sigma_2^2 + 2\rho\sigma_1\sigma_2$$
# - **Interpretation:**
#   - The mean is the sum of the means (by linearity)
#   - For independent Gaussians ($\rho = 0$), the variance is the sum of variances (uncertainty increases)
#   - Positive correlation increases variance, negative correlation decreases it

# %%
def gaussian_sum(g1, g2):
    return gaussian(g1.mean + g2.mean, g1.var + g2.var)


# %%
# Sum two Gaussians.
x = time_ut.Gaussian(10, 0.2 ** 2)
y = time_ut.Gaussian(15, 0.7 ** 2)

z = gaussian_sum(x, y)
print(z)

# %%
ax = time_ut.plot_gaussian(x, label="x")
time_ut.plot_gaussian(y, ax=ax, label="y")
time_ut.plot_gaussian(z, ax=ax, label="z");

# %%

# %%
# Interactive exploration of sum of Gaussians with correlation.
time_ut.cell1_1_plot_gaussian_sum()


# %% [markdown]
# ## Cell 1.2: Product of Gaussians
# - Given two Gaussians $X$ and $Y$
#   $$X \sim Normal(\mu_X, \sigma_X^2)$$
#   $$Y \sim Normal(\mu_Y, \sigma_Y^2)$$
# - The product $Z = X \cdot Y$ (PDF multiplication) is a Gaussian $Normal(\mu_Z, \sigma_Z^2)$ with:
#   $$\mu_Z = \frac{\mu_X \sigma_Y^2 + \mu_Y \sigma_X^2}{\sigma_X^2 + \sigma_Y^2}$$
#   $$\sigma_Z^2 = \frac{\sigma_X^2 \sigma_Y^2}{\sigma_X^2 + \sigma_Y^2}$$
# - **Interpretation:**
#   - Reduces variance by incorporating more information
#   - If one Gaussian $X$ is narrower (more accurate), result leans towards $X$
#   - If two Gaussians are similar (measures corroborate), result becomes more certain

# %% [markdown]
# **Gaussian products in terms of precision**
#
# - The precision of a Gaussian is
#   $$\tau = \frac{1}{\sigma^2}$$
# - The precision of the product is the sum of the precisions
#   $$\tau_Z = \tau_X + \tau_Y$$
#   $$\sigma_Z^2 = \frac{1}{\frac{1}{\sigma_X^2} + \frac{1}{\sigma_Y^2}}$$
# - The mean is the average of the means weighted by the precisions
#   $$\mu_Z = \sigma_Z^2 (\frac{\mu_X}{\sigma_X^2} + \frac{\mu_Y}{\sigma_Y^2})$$
#
# - The mean is averaged towards the more certain Gaussian
# - The variance is smaller than both

# %%
def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)


# %%
# Product of two equal Gaussians.
x = time_ut.Gaussian(10, 1.)

z = gaussian_multiply(x, x)
print(z)

# The result is more certain than both.

# %%
ax = time_ut.plot_gaussian(x, label="x")
time_ut.plot_gaussian(x, ax=ax, label="x")
time_ut.plot_gaussian(z, ax=ax, label="z");

# %%
# Product of two different Gaussians.
x = time_ut.Gaussian(10, 0.2 ** 2)
y = time_ut.Gaussian(15, 0.7 ** 2)

z = gaussian_multiply(x, y)
print(z)

ax = time_ut.plot_gaussian(x, label="x")
time_ut.plot_gaussian(y, ax=ax, label="y")
time_ut.plot_gaussian(z, ax=ax, label="z");

# %%
x = time_ut.Gaussian(10.2, 1)
y = time_ut.Gaussian(9.7, 1)

z = gaussian_multiply(x, y)
print(z)

ax = time_ut.plot_gaussian(x, label="x")
time_ut.plot_gaussian(y, ax=ax, label="y")
time_ut.plot_gaussian(z, ax=ax, label="z");

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
# - Sum adds uncertainty
# - Multiplication reduces uncertainty

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
import numpy as np
np.random.seed(13)

# Variance in the dog's movement.
process_var = 1.0
# Variance in the sensor.
sensor_var = 2.0

# dog's position, N(0, 20**2)
x = gaussian(0., 20.**2)
velocity = 1
# time step in seconds
dt = 1. 
process_model = gaussian(velocity*dt, process_var) # displacement to add to x
  
# simulate dog and get measurements
dog = time_ut.DogSimulation(
    x0=x.mean, 
    velocity=process_model.mean, 
    measurement_var=sensor_var, 
    process_var=process_model.var)

# create list of measurements
zs = [dog.move_and_sense() for _ in range(10)]

# %%
