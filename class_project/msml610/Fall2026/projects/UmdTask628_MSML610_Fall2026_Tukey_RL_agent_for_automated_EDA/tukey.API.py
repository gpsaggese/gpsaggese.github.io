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
# # Tukey API
#
# - This notebook tours the Tukey API: a Gym-style environment that hides a
#   random causal graph, statistical tools an agent can call on the training
#   data, a verifiable score, and the baselines a learned policy must beat
# - The pedagogical arc:
#   - Part 1: check the environment and the pinned scientific stack
#   - Part 2: sample a hidden graph and score candidate graphs against it
#   - Part 3: call the statistical tools, run the scripted agent and the
#     baselines
# - This version covers Milestones 1-2; the learned policy (Milestone 3) and
#   the Milestone 4 additions arrive with the next PRs

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import numpy as np
import pandas as pd

# %%
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook
import tukey_utils as utils
import tukey_env as tenv
import tukey_scripted as tscrip
import tukey_tools as ttools

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
# ## Library Overview
#
# - **What problem it solves**: learning which statistical test to run next on
#   an unknown dataset, rewarded by how well the emitted graph matches the
#   hidden graph that generated the data
# - **Key abstraction**: an episode = sample a hidden graph and data, call
#   tools on the training rows, emit a graph, receive a verifiable score
# - **Mental model**:
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `GraphConfig` | Generator settings | Size, family, noise, topology, scaling |
# | `sample_instance(cfg, seed)` | Hidden graph, data, split | Reproducible |
# | `TukeyEnv.reset(seed=...)` | Starts an episode | Returns `(Evidence, info)` |
# | `TukeyEnv.step(action)` | Runs one tool call or STOP | Reward = minus the call cost |
# | `TukeyEnv.submit_graph(edges)` | Scores an emitted graph | Ends the episode |
# | `score_graph(B, truth)` | Verifiable score | Structure + prediction |
#
# - **Key classes**:
#   - `TukeyEnv`: the environment
#   - `Evidence`: everything an agent sees (correlations and its own calls)
#   - `ScriptedExpert`: the prototype's rule as an agent

# %% [markdown]
# # Part 1: Environment check

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
# Outcome: one row per core package, each at the version pinned in
# `requirements.txt`.

# %% [markdown]
# **Implementation** `cell1_1_get_package_versions()`
# - Reads each version from the installed distribution metadata

# %%
hintros.print_obj_info(utils.cell1_1_get_package_versions)

# %% [markdown]
# # Part 2: A hidden graph and its score

# %% [markdown]
# ## Cell 2.1: Configure and sample an instance
#
# **Goal**
# - Sample one hidden causal world and see what is stored about it

# %%
# Six variables, seven edges, Gaussian noise with unequal variances.
cfg = tenv.GraphConfig(n_nodes=6, n_edges=7, family="linear_gauss")
inst = tenv.sample_instance(cfg, seed=3)
display(utils.cell2_1_describe_instance(inst))
# Outcome: the target class is the CPDAG: with Gaussian noise and unequal
# variances the data identify only the Markov equivalence class.

# %%
# The training rows, standardized with their own statistics.
display(inst.d_train.head())

# %%
hintros.print_obj_info(tenv.GraphConfig)

# %% [markdown]
# ## Cell 2.2: Plot the hidden graph
#
# **Goal**
# - See the answer an agent must recover (agents never see it)
#
# **Description**
# - `Hidden graph G*`: one node per column of the data; an arrow x_i -> x_j
#   means x_i is a direct cause of x_j

# %%
# Draw G*.
utils.cell2_2_plot_graph(inst.truth.B, "Hidden graph G*")

# %% [markdown]
# **Implementation** `cell2_2_plot_graph()`
# - Builds a `networkx` graph from the adjacency and draws it with the course's
#   `plot_causal_dag()` style

# %%
hintros.print_obj_info(utils.cell2_2_plot_graph)

# %% [markdown]
# ## Cell 2.3: Score candidate graphs
#
# **Goal**
# - Read the score on graphs whose quality is known in advance
#
# **Description**
# - `score`: `-0.5 * shd_target / |E*| - 0.5 * (1 - pred_frac)`; G* scores 0,
#   the empty graph -1
# - `shd_target`: structural distance to the identifiable target (here the
#   CPDAG); `shd_dag`, `shd_cpdag`: the same against the DAG and its class
# - `pred_frac`: held-out likelihood gain over the empty graph, relative to G*

# %%
# Score G*, the empty graph and four edits of G*.
display(utils.cell2_3_score_candidates(inst))
# Outcome: G* scores 0 and the empty graph -1; reversing a covered edge stays in
# the equivalence class, so it scores exactly like G* against the CPDAG target
# while `shd_dag` still counts it; reversing an uncovered edge changes the class
# and costs structure and prediction; extra edges (complete order) leave
# `pred_frac` near 1, so only the structural term penalizes them.

# %% [markdown]
# **Implementation** `cell2_3_score_candidates()`
# - Edits G* (a covered and an uncovered reversal, all reversals, the complete
#   graph along one order) and calls `score_graph()` on each

# %%
hintros.print_obj_info(utils.cell2_3_score_candidates)

# %% [markdown]
# ## Cell 2.4: Run an episode
#
# **Goal**
# - Walk through `reset()`, `step()` and `submit_graph()`

# %%
# Start an episode with a budget of 5 tool calls.
env = tenv.TukeyEnv(cfg, budget=5)
evidence, info = env.reset(seed=3)
display(info)
# Outcome: `info` holds only sizes and costs, never the hidden graph.

# %%
# Run one conditional-independence test: x0 independent of x1 given x2?
action = ttools.ci_test_action(0, 1, (2,))
evidence, reward, terminated, truncated, _ = env.step(action)
print("result:", evidence.results[action])
print("reward:", reward, "terminated:", terminated, "truncated:", truncated)
# Outcome: the partial correlation, its Fisher z and p-value; the reward is
# the call's training cost.

# %%
# Emit G* itself as an edge list and score it.
edges = [(int(i), int(j)) for i, j in np.argwhere(inst.truth.B == 1)]
scores = env.submit_graph(edges)
print("score:", scores["score"], "shd_target:", scores["shd_target"])
print("calls used:", scores["calls_used"])
# Outcome: the hidden graph scores 0 with one call used.

# %%
hintros.print_obj_info(tenv.TukeyEnv)

# %% [markdown]
# # Part 3: Tools, the scripted agent and the baselines

# %% [markdown]
# ## Cell 3.1: Call each tool once
#
# **Goal**
# - See what each statistical tool returns

# %%
# A fresh episode, then one call of each tool on x0, x1 (given x2).
evidence, _ = env.reset(seed=3)
# Show each tool's full result (pandas would cut the long rows).
with pd.option_context("display.max_colwidth", None):
    display(utils.cell3_1_call_each_tool(evidence, 0, 1, 2))
# Outcome: correlations and regressions are scale-free; only `variance` sees
# the raw scale (0 here, since the data are standardized).

# %% [markdown]
# **Implementation** `cell3_1_call_each_tool()`
# - Records each call in the evidence, which caches results per dataset

# %%
hintros.print_obj_info(ttools.ToolCache)

# %% [markdown]
# ## Cell 3.2: Run the prototype's rule as an agent
#
# **Goal**
# - Play the prototype's scripted rule through the tool API and print the
#   prototype's own edges beside its score (`test_tukey_scripted.py` checks
#   that the two graphs are equal on seeds 0-19)

# %%
# The prototype's own setting (exponential noise, raw scale), unlimited budget.
proto_env = tenv.TukeyEnv(tenv.PROTO_CONFIG, budget=1000)
scores, history = tscrip.run_episode(proto_env, tscrip.ScriptedExpert(), seed=0)
print("calls:", len(history), "shd:", scores["shd"], "f1:", scores["f1"])
reference = tscrip.proto_faithful_edges(proto_env.d_train.to_numpy())
print("prototype's edges:", reference)
# Outcome: SHD 3 and F1 0.667, the prototype's numbers on seed 0.

# %%
hintros.print_obj_info(tscrip.ScriptedExpert)

# %% [markdown]
# ## Cell 3.3: Compare the baselines on one instance
#
# **Goal**
# - Run the classical and sort-and-regress baselines on the instance of Part 2

# %%
# Every baseline sees only the training rows.
display(utils.cell3_3_compare_baselines(inst))
# Outcome: on Gaussian data the CPDAG methods (PC, GES) can match the target;
# DirectLiNGAM assumes non-Gaussian noise and orients at random here; sorting by
# the number of relatives can recover the class (the sortability of Milestone 1);
# the empty graph scores -1.

# %% [markdown]
# **Implementation** `cell3_3_compare_baselines()`
# - Calls `run_baseline()` and `score_output()` for each name in `BASELINES`

# %%
hintros.print_obj_info(utils.cell3_3_compare_baselines)
# %% [markdown]
# ## Summary: The Mental Model
#
# - An instance hides a graph; an agent sees only the training rows through
#   `Evidence`, and the score compares its graph with what the data identify
# - The score is 0 for the hidden graph and -1 for the empty graph, and graphs
#   in the same equivalence class tie when only the class is identifiable
# - Tools are pure functions of the training rows; the scripted agent and the
#   baselines use the same interface that the learned policy of Milestone 3
#   will use
