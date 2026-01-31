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
# # Lesson 05.1.2: Bin Analogy of ML
#
# **Course**: MSML610: Advanced Machine Learning
#
# **Instructor**: Dr. GP Saggese

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import seaborn as sns

import utils_Lesson05_Learning_Theory_Bin_Analogy_ML as utils

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import msml610_utils as ut

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## Cell 1: Visual Bin - Population of Marbles
#
# This visualization shows a bin filled with red and green marbles representing
# the unknown population. The parameter $\mu$ represents the true proportion of
# red marbles in the bin.

# %%
utils.draw_bin_with_marbles_interactive()

# %% [markdown]
# ## Cell 5: Single Experiment - Is nu Close to mu?
#
# This cell demonstrates a single sampling experiment from the bin. We draw N
# marbles randomly (with replacement) and compute the sample proportion $\nu$.
# The key question is: How close is $\nu$ to the true $\mu$?

# %%
utils.plot_single_experiment_interactive()

# %%

# %%
