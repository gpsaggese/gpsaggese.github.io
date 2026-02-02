# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
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
# ## Cell 1: Visual Bin: Population of Marbles.
#
# This visualization shows a bin filled with red and green marbles representing
# the unknown population. The parameter $\mu$ represents the true proportion of
# red marbles in the bin.

# %%
utils.cell1_draw_bin_with_marbles_interactive()

# %% [markdown]
# ## Cell 2: Single Experiment: Is nu Close to mu?
#
# - This cell demonstrates a single sampling experiment from the bin.
# - We draw N marbles randomly (with replacement) and compute the sample proportion $\nu$.
# - The key question is: How close is $\nu$ to the true $\mu$?

# %%
utils.cell2_plot_single_experiment_interactive()

# %% [markdown]
# - Single experiments don't tell the full story
#   - We need to know P(|nu - mu| > eps)
# - Questions to consider:
#   - What if we got unlucky in our sample?
#   - How confident can we be that nu ≈ mu?
#   - Does sample size matter? How much?
#   - "Let's repeat this many times..."

# %% [markdown]
# ## Cell 3: Monte Carlo Simulation: Distribution of nu
#
# - Run many sampling experiments (n_experiments times)
# - Each experiment: draw N marbles, compute $\nu$
# - Collect all $\nu$ values and visualize their distribution
# - Key questions:
#   - How are the $\nu$ values distributed?
#   - What fraction of experiments have $|\nu - \mu| > \epsilon$?
#   - Does the distribution concentrate around $\mu$?

# %%
utils.cell3_monte_carlo_simulation_interactive()

# %% [markdown]
# - The distribution of $\nu$ values clusters around the true $\mu$
# - As N increases, the distribution becomes tighter (smaller variance)
# - The empirical probability P(|nu - mu| > eps) decreases with larger N
# - This demonstrates the Law of Large Numbers in action
