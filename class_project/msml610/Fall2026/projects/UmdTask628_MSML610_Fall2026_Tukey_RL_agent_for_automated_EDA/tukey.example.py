# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tukey: learning to explore data with verifiable rewards
#
# - This notebook shows the project's results: synthetic causal worlds with a
#   known answer and a statistical toolbox with a scripted baseline (this
#   version); the trained policy and its generalization tests come with the next
#   PRs
# - The pedagogical arc:
#   - Part 1: the synthetic environment and what its data leak (Milestone 1)
#   - Part 2: the toolbox, the scripted agent and the baseline table
#     (Milestone 2)
# - This version covers Milestones 1-2; the trained policy (Milestone 3) and
#   generalization and real data (Milestone 4) arrive with the next PRs

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
import helpers.hnotebook as hnotebook
import tukey_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Part 1: The synthetic environment

# %% [markdown]
# ## Cell 1.1: Pinned scientific stack
#
# **Goal**
# - Confirm the container runs the pinned versions the results were produced
#   with

# %%
# Report the version of every package the project depends on.
versions = utils.cell1_1_get_package_versions()
display(versions)
# Outcome: one row per package, matching `requirements.txt`.

# %% [markdown]
# ## Cell 1.2: A gallery of hidden graphs
#
# **Goal**
# - See the graph shapes the generator draws and the target each one sets
#
# **Description**
# - One panel per topology (Erdos-Renyi, scale-free hubs, star, layered), each
#   with its identifiable target class (DAG or CPDAG, by the noise drawn); the
#   generator also draws chains

# %%
# Draw one hidden graph per topology.
utils.example_cell1_2_plot_gallery(seed=0)

# %% [markdown]
# **Implementation** `example_cell1_2_plot_gallery()`
# - Samples one instance per topology and plots its graph with
#   `plot_causal_dag()`

# %% [markdown]
# ## Cell 1.3: What simulated data leak about the causal order
#
# **Goal**
# - Measure how far simple scores (variance, R^2, number of relatives) reveal
#   the causal order, before any agent is trained
#
# **Description**
# - `var_raw`, `var_std`: sortability by variance on raw and standardized data
# - `r2`, `rel_oracle`, `rel_empirical`: sortability by R^2 and by the number
#   of relatives (true graph, or significant correlations)
# - `var_sort_minus_random_shd`: SHD of var-sort-regress minus random-order
#   regress on standardized data, with its Holm-adjusted p-value

# %%
# Load the committed report (100 validation graphs per family).
sortability = utils.example_cell1_3_load_sortability()
display(sortability)

# %%
# Plot variance and relatives sortability per family.
utils.example_cell1_3_plot_sortability(sortability)

# %% [markdown]
# **Guided usage**
# - Compare the blue and orange bars: standardizing removes the variance leak
#   (0.5 everywhere) while the prototype's raw data stay at 0.94
# - Look at the green bars: the number of relatives stays above 0.5 in every
#   random-graph family, including iSCM data, because it depends on the graph,
#   not on the weights
#
# **Implementation** `example_cell1_3_load_sortability()`
# - Reads `results/m1/sortability.json` (run `wp2_m1_sortability_b`); the edge
#   definition (Reisach et al. 2026, Eq. 4) counts each edge once, ties 1/2

# %% [markdown]
# # Part 2: The toolbox, the scripted agent and the baselines

# %% [markdown]
# ## Cell 2.1: The baseline table on the main task
#
# **Goal**
# - Rank every non-learned method on the main task distribution (setting `D`:
#   6-8 variables, one noise family per graph, standardized)
#
# **Description**
# - `score`: mean combined score over 200 benchmark graphs and its bootstrap
#   95% interval; `shd_target`: distance to the identifiable target;
#   `exact_recovery`: share of graphs recovered exactly; `calls`: tool calls
#   (scripted agents) or distinct CI tests (PC; corrected after the audit, the
#   stored counts include 3 cache entries, `results/post_audit/`)

# %%
# Methods tuned on the validation twin of the setting, scored once here.
display(utils.example_cell2_1_load_m2_table("D"))
# Outcome: GES, DirectLiNGAM and sort-by-relatives lead; the prototype's rule
# (`proto_scripted`) is below the classical methods; this is the baseline the
# learned policy must beat.

# %% [markdown]
# ## Cell 2.2: Every method in every setting
#
# **Goal**
# - See where each method's assumptions hold
#
# **Description**
# - One column per setting (`A0` prototype data, `A1` exponential, `B` Gaussian,
#   `Braw` equal-variance raw, `C` uniform, `D` main task, `E` nonlinear, `F`
#   observed confounder, `G` pgmpy, `H` 12 variables); one row per method

# %%
# Mean combined score, 200 graphs per cell.
utils.example_cell2_2_plot_m2_heatmap()

# %% [markdown]
# **Guided usage**
# - Read the `direct_lingam` row: near 0 on non-Gaussian data (`A0`, `A1`, `C`),
#   poor on Gaussian data (`B`, `Braw`)
# - Read the `var_sort_regress` row: strong only where variance is a real or
#   leaked signal (`A0` raw prototype data, `Braw` equal variances)
# - Read the `ges` and `pc` rows: best on Gaussian data, where the target is the
#   equivalence class
#
# **Implementation** `example_cell2_2_plot_m2_heatmap()`
# - Reads `results/m2/table_summary.json` (runs `wp3_m2_table`, `wp3_m2_split`)
