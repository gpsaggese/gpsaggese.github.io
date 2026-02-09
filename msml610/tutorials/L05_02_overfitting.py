# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
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
# ## Cell 2: Constant Model (H_0)
#
# This cell demonstrates learning with a constant hypothesis h(x) = b.
#
# **Purpose**: Show how the simplest model (a constant) fits the data. The
# constant model finds the best horizontal line by computing the mean of
# training points. This model has high bias (poor approximation of complex
# functions) but low variance (very stable across different training sets).
#
# **Setup**: This cell uses the same configuration as Cell 1. All parameters
# (function type, epsilon, N, seed) are synchronized with Cell 1. To change
# the setup, adjust the parameters in Cell 1.
#
# **Model**: h(x) = b, where b is the mean of training y-values.
#
# **Three plots**:
# 1. **In-Sample Data**: Green training points with fitted constant model and E_in
# 2. **Out-of-Sample Data**: Red test points with fitted constant model and E_out
# 3. **True Function vs Model**: Blue true function, green constant model, orange shaded approximation error
# 4. **Comments**: Learned parameter b, errors, and key observations
#
# **Key observations**:
# - The constant model (horizontal line) is the simplest possible hypothesis
# - It has HIGH BIAS: cannot approximate complex target functions well
# - It has LOW VARIANCE: the fitted line is very stable across different training sets
# - Click "Resample and Relearn" to see how the model changes with different training data
# - The orange shaded area shows the approximation error between the true function and the constant

# %%
# Display constant model learning with interactive controls.
utils.cell2_plot_constant_model()
