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
# ## Cell 1: True Target Function
#
# This cell visualizes the true target function that we want to learn.
#
# **Purpose**: Understand that in real-world machine learning, we don't have
# access to the complete target function - we only observe sampled points.
#
# **Parameters**:
# - `Function`: Select the true target function (Slow Sinusoid, Fast Sinusoid,
#   Parabola, Constant, or Linear)
# - `epsilon`: Standard deviation of noise added to observations
#
# **Key observation**:
# - The complete curve represents the unknown target function f(x)
# - In practice, we only have access to a few noisy samples from this function
# - The goal of learning is to approximate this function from limited data

# %%
# Display the true target function with interactive controls.
utils.cell1_plot_true_target_function()

# %% [markdown]
# ## Cell 2: Sampled Data - In-Sample vs Out-of-Sample
#
# This cell visualizes how we sample data from the true target function and split it into training and test sets.
#
# **Purpose**: Understand the concept of in-sample (training) and out-of-sample (test) data.
#
# **Parameters**:
# - `Random Seed`: Controls the random sampling of data points
# - `Function`: Select the true target function to sample from
# - `epsilon`: Standard deviation of noise added to observations
# - `N (total samples)`: Total number of data points to sample
#
# **Key observations**:
# - **Green points (In-Sample)**: 80% of data used for training the model
# - **Red points (Out-of-Sample)**: 20% of data used for testing the model
# - The model should learn from green points but generalize to red points
# - This split helps us evaluate how well our model generalizes to unseen data

# %%
# Display the sampled data with in-sample/out-of-sample split.
utils.cell2_plot_sampled_data_interactive()

# %%
