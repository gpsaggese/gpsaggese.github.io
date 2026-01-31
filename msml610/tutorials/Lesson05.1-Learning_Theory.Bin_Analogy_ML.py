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

import utils_Lesson05_Learning_Theory as utils

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
import ipywidgets

# TODO(ai_gp): Move this call to the corresponding utils file as a function.
# Create slider for mu.
mu_slider, mu_box = ut.build_widget_control(
    name="mu",
    description="true proportion of red marbles",
    min_val=0.0,
    max_val=1.0,
    step=0.01,
    initial_value=0.5,
    is_float=True,
)

# Create interactive output.
output = ipywidgets.interactive_output(
    utils._draw_bin_with_marbles, {"mu": mu_slider}
)

# Display widgets.
display(mu_box)
display(output)

# %% [markdown]
# ## Cell 5: Single Experiment - Is nu Close to mu?
#
# This cell demonstrates a single sampling experiment from the bin. We draw N
# marbles randomly (with replacement) and compute the sample proportion $\nu$.
# The key question is: How close is $\nu$ to the true $\mu$?

# %%
# TODO(ai_gp): Move this call to the corresponding utils file as a function.
# Create interactive widgets.
mu_slider_5, mu_box_5 = ut.build_widget_control(
    name="mu",
    description="true proportion of red marbles",
    min_val=0.0,
    max_val=1.0,
    step=0.05,
    initial_value=0.6,
    is_float=True,
)

N_slider_5, N_box_5 = ut.build_widget_control(
    name="N",
    description="number of samples",
    min_val=10,
    max_val=1000,
    step=10,
    initial_value=100,
    is_float=False,
)

seed_slider_5, seed_box_5 = ut.build_widget_control(
    name="seed",
    description="random seed",
    min_val=0,
    max_val=1000,
    step=1,
    initial_value=42,
    is_float=False,
)

# Create interactive output.
output_5 = ipywidgets.interactive_output(
    utils._plot_single_experiment,
    {"mu": mu_slider_5, "N": N_slider_5, "seed": seed_slider_5},
)

# Display widgets.
display(ipywidgets.VBox([mu_box_5, N_box_5, seed_box_5, output_5]))

# %%
