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
# # Difference in difference
#
# - This notebook estimates the effect of an offline marketing campaign on
#   downloads from a (`date`, `city`) panel, using the
#   difference-in-differences estimator
# - The pedagogical arc:
#   - Loading the marketing panel data
#   - Pre/post intervention windows and group-time average outcomes
#   - The average treatment effect on the treated (ATT), as the treated
#     group's post-period outcome minus its parallel-trends counterfactual
#   - Comparing the estimate against the known ground-truth effect `tau`

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import pandas as pd

# %%
import helpers.hnotebook as hnotebook

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
hnotebook.init_loggers(_LOG, set_all_loggers_to_print=True)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Part 1: Loading Data

# %% [markdown]
# ## Cell 1.1: Loading the marketing panel data
#
# **Goal**
# - Load a marketing panel dataset used to estimate the effect of an
#   offline campaign via difference-in-differences
#
# The data is in a panel format:
# - Each row is a (`date`, `city`) pair
# - **downloads**: the outcome to predict
# - **treated**: indicator of whether the city received the intervention
# - **tau**: the (known, ground-truth) treatment effect

# %%
dir_name = "L08_data"
# !ls $dir_name

out_dir_name = "figures/"

# %%
mkt_data = pd.read_csv(f"{dir_name}/short_offline_mkt_south.csv").astype(
    {"date": "datetime64[ns]"}
)
print("mkt_data.shape=", mkt_data.shape)
display(mkt_data.head())

# %% [markdown]
# # Part 2: Canonical Difference-in-Differences

# %% [markdown]
# ## Cell 2.1: Checking the pre/post intervention windows
#
# **Goal**
# - Check the date range covered by each of the 4 (treated, post) groups,
#   to confirm the panel spans both a pre- and a post-intervention period

# %%
# Compute pre- and post-intervention period.
pre_post_windows = (
    mkt_data.assign(w=lambda d: d["treated"] * d["post"])
    .groupby(["w"])
    .agg({"date": ["min", "max"]})
)
display(pre_post_windows)

# %% [markdown]
# ## Cell 2.2: Computing group-time average outcomes
#
# **Goal**
# - Compute the average outcome for each of the 4 (treated, post) groups,
#   the building block of the difference-in-differences estimator

# %%
did_data = mkt_data.groupby(["treated", "post"]).agg(
    {"downloads": "mean", "date": "min"}
)
display(did_data)

# %% [markdown]
# ## Cell 2.3: Computing the average treatment effect on the treated
#
# **Goal**
# - Estimate the ATT as the treated group's actual post-period outcome
#   minus its counterfactual (parallel-trends) outcome
#
# **Implementation**
# - `y0_est`: the treated group's pre-period outcome, plus the control
#   group's pre-to-post change
# - `att = <treated, post-period outcome> - y0_est`

# %%
y0_est = (
    did_data.loc[1].loc[0, "downloads"]  # Treated baseline.
    + did_data.loc[0].diff().loc[1, "downloads"]  # Control evolution.
)

att = did_data.loc[1].loc[1, "downloads"] - y0_est
print("att=", att)

# %% [markdown]
# ## Cell 2.4: Comparing against the known ground-truth effect
#
# **Goal**
# - Compare the estimated ATT against the dataset's known, ground-truth
#   `tau`, to check how close the difference-in-differences estimator got

# %%
print("mean tau=", mkt_data.query("post==1").query("treated==1")["tau"].mean())
