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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import msml610_utils as ut
import utils_Lesson09_3_Multi_Armed_Bandits as utils

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# # Cell 1: Introduction - Casino Slot Machines
#
# Interactive casino slot machine visualization.
# - There are 3 slot machines
# - You have 10 coins
# - Each gives you a payout in [-1, 1] with an unknown mean $\mu_i$
# - Choose which machine to play
# - Track total winnings and coin budget
# - How do you maximize your winnings?

# %%
utils.cell1_casino_slot_machines()

# %%
