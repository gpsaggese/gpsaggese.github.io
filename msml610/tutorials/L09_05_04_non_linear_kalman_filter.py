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
# # Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

from numpy.random import multivariate_normal, normal
import numpy as np
import matplotlib.pyplot as plt

import msml610_utils as ut

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %%
import helpers.hio as hio
import L09_05_04_non_linear_kalman_filter_utils as time_ut

dst_dir = "figures"
hio.create_dir(dst_dir, incremental=True)
# # cp msml610/tutorials/figures/*.png msml610/lectures_source/figures

# %%
# !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet filterpy)"

import filterpy


# %%
def f(x):
    return (np.cos(4 * (x / 2 + 0.7))) - 1.3 * x

    
time_ut.plot_function(f)

# %%
# Create 500,000 samples with mean 0, std 1.
gaussian = (0.0, 1.0)
data = normal(loc=gaussian[0], scale=gaussian[1], size=500000)

time_ut.plot_nonlinear_func(data, f)

# %%
# Plot N points 
N = 30000
plt.subplot(121)
plt.scatter(data[:N], range(N), alpha=0.2, s=1)
plt.title("Input")
plt.subplot(122)
plt.title("Output")
plt.scatter(f(data[:N]), range(N), alpha=0.2, s=1)

# %%
def f_nonlinear_xy(x, y):
    return np.array([x + y, 0.1 * x**2 + y * y])


# %%
time_ut.plot_nonlinear_xy()

# %% [markdown]
# # 

# %% [markdown]
# # 

# %%
# Create a Gaussian.
N = 10000
mean = (0.0, 0.0)
p = np.array([[32.0, 15.0], [15.0, 40.0]])
xs, ys = multivariate_normal(mean=mean, cov=p, size=N).T

# Compute linearized mean.
mean1 = f_nonlinear_xy(np.mean(xs), np.mean(ys))
print("f(mean)=", mean1)
mean2 = np.mean([f_nonlinear_xy(xs_tmp, ys_tmp) for xs_tmp, ys_tmp in zip(xs, ys)], axis=0)
print("mean(f)=", mean2)

# Plot both.
time_ut.plot_monte_carlo_mean(xs, ys, f_nonlinear_xy, mean_fx, "Linearized Mean")

# %%
np.array([f_nonlinear_xy(xs_tmp, ys_tmp) for xs_tmp, ys_tmp in zip(xs, ys)])

# %%
xs

# %%
