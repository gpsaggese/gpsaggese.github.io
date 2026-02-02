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

# %% [markdown]
# # Cell 2: Exploration vs Exploitation Dilemma
#
# Demonstrate the fundamental tradeoff between exploration and exploitation.
# - Pure exploration learns but earns little
# - Pure exploitation gets stuck on suboptimal choices
# - Balance is key!

# %%
utils.cell2_exploration_vs_exploitation()
# Pure exploration learns but earns little. Pure exploitation gets stuck on suboptimal choices. Balance is key.

# %% [markdown]
# <!-- # Cell A: Strategy Comparison with Epsilon Sweep
#
# Interactive comparison of strategies with epsilon sweep analysis.
# - Run multiple trials with different epsilon values
# - Compare exploration, exploitation, and balanced strategies
# - Visualize mean performance with error bars
# - Find optimal epsilon value -->

# %%
# utils.cell3_strategy_comparison()

# %% [markdown]
# <!-- # Cell 4: Ensemble Comparison Across Random Mu Configurations
#
# Compare strategies averaged over multiple random mu configurations.
# - Test strategy robustness across different machine configurations
# - Average results over many random mu values
# - Compare mean performance with confidence intervals
# - Understand which strategy is most reliable -->

# %%
# utils.cell4_ensemble_comparison()

# %%
