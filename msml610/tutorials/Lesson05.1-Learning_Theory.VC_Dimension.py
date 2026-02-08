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
import utils_Lesson05_1_Learning_Theory_VC_Dimension as utils

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## Cell 1: Dichotomy Explorer - 2D Perceptron with 3 Points

# %%
# Explore how a 2D perceptron can classify 3 points in different ways.
# Adjust the angle and offset of the separating line to discover all possible dichotomies.
utils.cell1_dichotomy_explorer_3points()
# Try different angles and offsets to discover all 8 possible classifications of 3 points.
