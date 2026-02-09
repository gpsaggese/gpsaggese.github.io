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

import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import msml610_utils as ut
import L05_02_overfitting_utils as utils

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## Cell 1: True Target Function and Data Sampling
#
# This cell visualizes the true target function and how we sample data from it.
#
# **Purpose**: Understand that in real-world machine learning, we don't have
# access to the complete target function - we only observe sampled points. This
# cell shows the relationship between the true function, in-sample (training)
# data, and out-of-sample (test) data.
#
# **Parameters**:
# - `Function`: Select the true target function (Slow Sinusoid, Fast Sinusoid,
#   Parabola, Constant, or Linear)
# - `epsilon`: Standard deviation of noise added to observations
# - `N (total samples)`: Number of data points to sample from the function
#
# **Four plots**:
# 1. **True Target Function**: The complete unknown function we want to learn (shown with and without noise)
# 2. **In-Sample Data (80%)**: Green points used for training the model
# 3. **Out-of-Sample Data (20%)**: Red points used for testing the model
# 4. **Comments**: Summary of parameters and observations
#
# **Key observations**:
# - The complete curve represents the unknown target function f(x)
# - In practice, we only have access to a few noisy samples from this function
# - We split data into training (green) and testing (red) sets
# - The goal is to learn from training data and generalize to test data

# %%
# Display the true target function with interactive controls.
utils.cell1_plot_true_target_function()

# %% [markdown]
# ## Cell 2: Model Comparison - Constant vs Linear
#
# This cell demonstrates learning with either a constant hypothesis h(x) = b
# or a linear hypothesis h(x) = a*x + b.
#
# **Purpose**: Compare two model types to understand the bias-variance tradeoff.
# The constant model has high bias and low variance, while the linear model
# has lower bias but higher variance. Use the selector to switch between models
# and observe the differences.
#
# **Setup**: This cell uses the same configuration as Cell 1. All parameters
# (function type, epsilon, N, seed) are synchronized with Cell 1. To change
# the setup, adjust the parameters in Cell 1.
#
# **Models**:
# - **Constant**: h(x) = b, where b is the mean of training y-values
# - **Linear**: h(x) = a*x + b, where a and b are fit using least squares
#
# **Four plots**:
# 1. **In-Sample Data**: Green training points with fitted model and E_in
# 2. **Out-of-Sample Data**: Red test points with fitted model and E_out
# 3. **True Function vs Model**: Blue true function, fitted model, orange shaded approximation error
# 4. **Comments**: Learned parameters, errors, and key observations
#
# **Key observations**:
# - **Constant model**: HIGH BIAS (poor approximation), LOW VARIANCE (stable)
# - **Linear model**: LOWER BIAS (better approximation), HIGHER VARIANCE (more sensitive)
# - Click "Resample and Relearn" to see how each model changes with different training data
# - The orange shaded area shows the approximation error
# - Compare E_in and E_out for both models to see the bias-variance tradeoff

# %%
# Display model learning with interactive controls.
utils.cell2_plot_model()

# %%
