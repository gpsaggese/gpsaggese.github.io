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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import msml610_utils as ut
ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %%
import os

import helpers.hio as hio
import L09_05_kalman_filter_utils as time_ut

dst_dir = "figures"
hio.create_dir(dst_dir, incremental=True)
# # cp msml610/tutorials/figures/*.png msml610/lectures_source/figures

# %% [markdown]
# # Cell 1: Sum and Product of Gaussians

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
# Interactive exploration of product of Gaussians.
time_ut.cell1_2_plot_gaussian_product()

# %%
