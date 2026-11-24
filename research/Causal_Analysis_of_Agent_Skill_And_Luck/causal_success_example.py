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
# # Causal analysis of success: talent, luck, and capital
#
# - This notebook runs one baseline experiment of the talent and luck simulation
#   (100 agents, 80 periods, seed 42) and analyzes it end to end
#   - The building blocks of the simulation are in `causal_success_API.ipynb`
#   - The set-up and the notation are in `README.md`
# - The pedagogical arc:
#   - The baseline: one config, one run, the talents and the final capital
#   - Associations: how talent and realized events line up with log-capital
#   - Causal estimates: the effect of one realized beneficial event, with the
#     uncertainty of the estimator
#   - Policies: what allocating resources before the run changes
#   - The paper: how the baseline compares with the reported numbers

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import pandas as pd

# %%
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

import causal_success_utils as utils
import causal_success_analysis_utils as analysis

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
# # Part 1: The baseline experiment

# %% [markdown]
# ## Cell 1.1: The config of the baseline
#
# - `get_default_config()` returns every parameter of one experiment
#   - Each value is the default of the simulation function that uses it
# - Every result in this notebook comes from this config and from one call to
#   `run_experiment()`

# %%
# Get the config of the baseline experiment and list its parameters.
config = utils.get_default_config()
display(analysis.cell1_1_config_to_df(config))
# Outcome: 22 parameters, e.g., 100 agents, 80 periods, seed 42, and no policy.

# %%
# Describe the set-up in one line, to show it on every plot.
setup_txt = analysis.cell1_1_get_setup_txt(config)
print("setup_txt=", setup_txt)
# Outcome: "100 agents, 80 periods, seed 42".

# %% [markdown]
# ## Cell 1.2: Running the baseline

# %%
# How one run works, with the values of the default config:
#
# - Each agent has 3 talents drawn from N(0.5, 0.15) and clipped to [0, 1], and
#   starts with capital 1.0
# - Each period has 5 lucky and 5 unlucky events
#   - Each event hits one agent, picked with probability proportional to a
#     sigmoid of its intensity
# - A lucky event multiplies the capital by (1 + impact) only if the agent
#   passes an IQ check, i.e., with probability `iq`
#   - The impact is drawn from N(0.25, 0.08) and clipped to [0.05, 0.5]
# - After every lucky event, with probability 0.1, a spillover can reach another
#   agent
#   - The check runs whether or not the hit agent capitalized on the event
#   - The recipient is drawn from all agents in proportion to networking: there
#     is no network graph
#   - If the draw picks the hit agent, there is no spillover
#   - The recipient must pass its own IQ check
#   - The spillover impact is half the impact of the lucky event
# - An unlucky event always applies: it multiplies the capital by
#   (1 - impact), with the impact drawn from N(0.15, 0.05) and clipped to
#   [0.05, 0.3]
# - `lucky_events` counts the realized beneficial events of an agent
#   - The direct events it capitalized on, plus the spillovers it received
#   - Not the lucky events it met without capitalizing on them
print("One run applies random events to the agents and compounds their capital")

# %%
# Run the baseline experiment once: every later cell uses `results`.
results = utils.run_experiment(config)
df = results["df"]
display(df.head())
# Outcome: one row per agent with its talents, final capital, and event counts.

# %%
# Show the summary statistics of the baseline.
display(pd.Series(results["stats"], name="baseline").to_frame())
# Outcome: mean capital 0.888, median 0.816, Gini 0.217, 2.32 realized
# beneficial events and 4.00 unlucky events per agent on average.

# %% [markdown]
# - The median agent ends below its starting capital of 1.0
#   - On average an agent realizes 2.32 beneficial events and suffers 4.00
#     unlucky ones
#   - A lucky event is realized only with probability `iq`, about 0.5 on
#     average (Cell 1.3), while every unlucky event applies
# - The richest agent holds 7.79 times the capital of the poorest one

# %% [markdown]
# ## Cell 1.3: The talents of the agents
#
# **Goal**
# - Check that the talents are roughly symmetric around 0.5 before the run

# %% [markdown]
# **Description**
# - Inputs
#   - `df`: the results table of the baseline
#   - `setup_txt`: the set-up shown in the title
#
# - Panels
#   - `Intensity`: histogram of the intensity, with its mean and std
#   - `IQ`: histogram of the IQ, with its mean and std
#   - `Networking`: histogram of the networking, with its mean and std

# %%
# Plot the distribution of each talent on the same axes.
analysis.cell1_3_plot_talent_distributions(df, setup_txt, figsize=(10.5, 3))

# %% [markdown]
# **Guided usage**
# - Look at the 3 panels
#   - Observe means near 0.5 and sample standard deviations around 0.14
#   - These differ from the configured 0.15 because this is a finite sample
#   - No talent draws were clipped in this baseline run
# - Compare the ranges of the panels
#   - Observe that most agents have talents between about 0.3 and 0.7, with a
#     few agents in the tails

# %% [markdown]
# **Implementation** `cell1_3_plot_talent_distributions(df, setup_txt, ...)`
# - Draws one `seaborn` histogram per talent, with the same bins on [0, 1]
# - Marks the mean with a dashed line and writes the std in the title

# %%
hintros.print_obj_info(analysis.cell1_3_plot_talent_distributions)

# %% [markdown]
# # Part 2: Inequality of the final capital

# %% [markdown]
# ## Cell 2.1: The distribution of the final capital
#
# **Goal**
# - See how unequal the capital becomes when every agent starts from 1.0

# %% [markdown]
# **Description**
# - Inputs
#   - `df`: the results table of the baseline
#   - `results["stats"]`: the Gini and the shares of the same run
#   - `setup_txt`: the set-up shown in the title
#
# - Panels
#   - `Capital`: histogram of the final capital, with its mean and median
#   - `log(capital)`: histogram of the log of the final capital, the outcome of
#     Part 4
#   - `Lorenz`: share of capital held by the poorest share of agents

# %%
# Plot the final capital, its log, and its Lorenz curve.
analysis.cell2_1_plot_capital_distribution(
    df, results["stats"], setup_txt, figsize=(10.5, 3)
)

# %% [markdown]
# **Guided usage**
# - Look at the `Capital` panel
#   - Observe a right skew: the mean is above the median
# - Look at the `log(capital)` panel
#   - Observe a much smaller skew: the multiplicative events act additively on
#     the log
# - Look at the `Lorenz` panel
#   - Observe a curve close to the diagonal: with a Gini of 0.217, the top 10%
#     hold 18.4% of the capital and the bottom 50% hold 34.5%

# %% [markdown]
# **Implementation** `cell2_1_plot_capital_distribution(df, stats, setup_txt, ...)`
# - Draws the histograms of the capital and of its log with `seaborn`
# - Builds the Lorenz curve from the sorted capital
# - Takes the Gini and the shares from `generate_summary_statistics()`, via
#   `stats`

# %%
hintros.print_obj_info(analysis.cell2_1_plot_capital_distribution)

# %% [markdown]
# - This baseline is moderately unequal, not heavy-tailed
#   - A single run of 100 agents cannot show a power law
# - Part 6 shows that seed 42 has the lowest Gini of 50 seeds

# %% [markdown]
# # Part 3: Talent, realized events, and log-capital

# %% [markdown]
# ## Cell 3.1: What the event counts measure

# %%
# Event counts as predictors:
#
# - `lucky_events` is not pure luck: the talents change it
#   - Intensity raises the chance of being hit by any event
#   - IQ decides whether a lucky event is realized
#   - Networking and IQ decide who receives the spillovers
# - `unlucky_events` depends only on intensity and on chance
# - `net_events` is `lucky_events - unlucky_events`
# - The correlations below are associations: Part 4 adjusts for the talents
print("Realized events depend on talents, so their correlations are not causal")

# %% [markdown]
# ## Cell 3.2: Correlations with log-capital

# %%
# Compute the correlation of each talent and event count with log-capital.
corr_df = analysis.cell3_2_compute_log_capital_correlations(df)
display(corr_df)
# Outcome: r = -0.045 for the talent norm, 0.540 for the realized beneficial
# events, and 0.920 for the net events.

# %% [markdown]
# - No talent has a correlation with log-capital that is distinguishable from 0
#   in this run
#   - The 95% interval of every talent contains 0, e.g., [-0.239, 0.153] for
#     the talent norm
# - The event counts are strongly associated with log-capital
#   - Realized beneficial events: $r = 0.540$, 95% CI [0.384, 0.666]
#   - Unlucky events: $r = -0.596$
#   - Net events: $r = 0.920$, since each event adds or removes a similar
#     amount of log-capital
# - The ratio of the two correlations is not meaningful here: the talent
#   correlation is negative and its interval contains 0

# %% [markdown]
# ## Cell 3.3: Talent and realized events against log-capital
#
# **Goal**
# - See the weak talent association and the strong event association agent by
#   agent

# %% [markdown]
# **Description**
# - Inputs
#   - `df`: the results table of the baseline
#   - `corr_df`: the correlations of Cell 3.2, shown in the titles
#   - `setup_txt`: the set-up shown in the title
#
# - Panels
#   - `Talent norm`: log-capital against the talent norm, with a linear fit
#   - `Realized beneficial events`: log-capital against `lucky_events`, with a
#     linear fit

# %%
# Plot log-capital against the talent norm and the realized events.
analysis.cell3_3_plot_talent_and_luck_vs_log_capital(
    df, corr_df, setup_txt, figsize=(8.5, 3)
)

# %% [markdown]
# **Guided usage**
# - Look at the `Talent norm` panel
#   - Observe a flat fit: the talent norm does not sort the agents by capital
# - Look at the `Realized beneficial events` panel
#   - Observe the rising fit, and the vertical spread at each count: the
#     unlucky events and the impact sizes also matter

# %% [markdown]
# **Implementation** `cell3_3_plot_talent_and_luck_vs_log_capital(df, corr_df, ...)`
# - Draws a `seaborn` scatter with a linear fit per predictor, on a shared
#   y-axis
# - Writes the correlation and its interval from `corr_df` in each title

# %%
hintros.print_obj_info(analysis.cell3_3_plot_talent_and_luck_vs_log_capital)

# %% [markdown]
# ## Cell 3.4: The richest agents
#
# - `talent_rank` 1 is the agent with the highest talent norm
# - `capital_rank` 1 is the agent with the highest final capital
# - The two are separate rankings, usually of different agents
# - A lower rank is better: the average rank of 100 agents is 50.5

# %%
# Rank the agents by talent and by capital.
df_ranked = analysis.cell3_4_add_ranks(df)
top_cols = [
    "id",
    "capital",
    "capital_rank",
    "talent_norm",
    "talent_rank",
    "lucky_events",
    "unlucky_events",
]
display(df_ranked.nsmallest(10, "capital_rank")[top_cols])
# Outcome: the 10 richest agents, with talent ranks from 17 to 94.

# %% [markdown]
# ## Cell 3.5: The richest agents against the others

# %%
# Compare the 10 richest agents with the other 90 and with all the agents.
n_top = 10
display(analysis.cell3_5_summarize_top_performers(df_ranked, n_top))
# Outcome: mean talent rank 42.6 for the top 10 against 51.4 for the others,
# and 4.0 against 2.1 realized beneficial events.

# %% [markdown]
# - The 10 richest agents are slightly more talented than average in this run
#   - Mean talent rank 42.6 and median 36.0, against 50.5 for the population
#   - Lower is better, so they are not below-average talent
#   - Their ranks still go from 17 to 94: the richest agents are not the most
#     talented ones
# - They realized more beneficial events and suffered fewer unlucky ones
#   - 4.0 against 2.1 realized beneficial events
#   - 1.8 against 4.2 unlucky events
#   - Part of this is mechanical: selecting the richest agents selects the agents
#     with the most realized gains
# - 10 agents from one run are too few to separate talent from chance: Part 4
#   estimates the effect of the events while adjusting for the talents

# %% [markdown]
# # Part 4: Causal effect of one realized beneficial event

# %% [markdown]
# ## Cell 4.1: The causal set-up

# %%
# Causal set-up of Part 4:
#
# - Outcome Y_i: `log(capital)` of agent i at the end of the run
#   - The capital floor keeps the capital positive, so no `+ 1` shift is needed
# - Treatment D_i: `lucky_events`, the count of realized beneficial events,
#   including spillovers
# - Controls X_i: intensity, IQ, and networking
#   - They change both D_i (exposure, IQ check, spillovers) and Y_i
# - Estimand: the average change of Y_i for one more realized beneficial event,
#   holding the talents fixed
# - Assumptions that make the estimate causal:
#   - Given the talents, the rest of the variation of D_i comes from the random
#     draws of the simulator, independent of the unlucky events and of the
#     impact sizes
#   - Each event adds log(1 + impact) to Y_i, so the effect is additive in the
#     count
#   - Direct events and spillovers (half the impact) are pooled in one count, so
#     the estimand is an average over that mix
# - A confidence interval measures the sampling noise of the estimator under
#   these assumptions: it does not show that the assumptions hold
print("Y_i = log(capital), D_i = realized beneficial events, X_i = talents")

# %% [markdown]
# ## Cell 4.2: Observed log-capital change of each event
#
# - In a simulation the log-capital change of each realized event is observed
#   - Each event adds one entry to the capital history of the agent
#   - Its change is the log of the ratio of 2 consecutive entries
# - The kind of an event is read from the sign of its change
#   - `unchanged` counts the events with no change, e.g., an unlucky event on
#     an agent already at the capital floor
#   - The floor can also make a detrimental change smaller than its impact
#   - A kind with no event has undefined statistics (NaN)

# %%
# Measure the log-capital change of every realized event of the baseline.
display(analysis.cell4_2_compute_log_change_per_event(results["agents"]))
# Outcome: 232 beneficial events with a mean change of 0.199, 400 detrimental
# events with a mean change of -0.164, and no unchanged event.

# %% [markdown]
# - The observed average event-level log gain is 0.199 in this run
#   - The changes go from 0.034 to 0.373, since the impacts are random and the
#     spillovers have half the impact
# - The 232 events are the sum of `lucky_events` over the agents: 2.32 per agent
# - No event is unchanged: no agent reaches the capital floor in the baseline

# %% [markdown]
# ## Cell 4.3: Double machine learning

# %%
# Estimate the average effect of one realized beneficial event with DML.
dml_df = analysis.cell4_3_estimate_dml_effect(df, config["seed"])
display(dml_df)
# Outcome: effect 0.157, standard error 0.026, 95% CI [0.106, 0.208].

# %% [markdown]
# - `LinearDML` estimates a constant effect of 0.157 on log-capital
#   - Standard error 0.026 and 95% CI [0.106, 0.208], from the inference of the
#     estimator
#   - In capital, one more realized beneficial event is a gain of about 17.0%
# - The observed average event-level log gain of Cell 4.2, 0.199, lies inside
#   the interval, near its upper end
#   - The two quantities differ: DML estimates an effect conditional on the
#     talents, while 0.199 averages the realized events
#   - So this is a consistency check: it does not validate the estimate or the
#     assumptions of Cell 4.1
# - This is one run of 100 agents: another seed gives another estimate

# %%
hintros.print_obj_info(analysis.cell4_3_estimate_dml_effect)

# %% [markdown]
# ## Cell 4.4: Causal forest

# %%
# Estimate the effect of one realized beneficial event for each agent.
cate_df, cate_summary_df = analysis.cell4_4_estimate_cate(df, config["seed"])
display(cate_summary_df)
# Outcome: average effect 0.159, 95% CI [0.094, 0.224], std of the per-agent
# effects 0.013 against a mean interval width of 0.126.

# %% [markdown]
# - The causal forest gives an average effect of 0.159, 95% CI [0.094, 0.224],
#   close to the DML estimate
# - The fitted per-agent effects vary little relative to their uncertainty
#   - Their std is 0.013, while their intervals have a mean width of 0.126
#   - No agent has an interval that excludes the average effect
#   - Overlapping intervals are a description, not a formal test of
#     heterogeneity
# - The mechanism does not imply a constant effect
#   - The talents change the mix of direct events and spillovers, which have
#     half the impact
#   - So the average gain of one realized event can differ across agents

# %%
hintros.print_obj_info(analysis.cell4_4_estimate_cate)

# %% [markdown]
# ## Cell 4.5: Per-agent effects
#
# **Goal**
# - See how wide the per-agent intervals are compared with the spread of the
#   per-agent effects

# %% [markdown]
# **Description**
# - Inputs
#   - `cate_df`: the per-agent effects and intervals of Cell 4.4
#   - `dml_df`: the DML estimate of Cell 4.3, drawn as a gray band
#   - `setup_txt`: the set-up shown in the title
#
# - Panels
#   - `Per-agent effects with their intervals`: agents sorted by effect
#   - `Per-agent effects against IQ`: the same effects against the IQ

# %%
# Plot the per-agent effects with their intervals.
analysis.cell4_5_plot_cate(cate_df, dml_df, setup_txt, figsize=(8.5, 3))

# %% [markdown]
# **Guided usage**
# - Look at the sorted panel
#   - Observe that every interval overlaps the gray DML band
# - Look at the IQ panel
#   - Observe no clear trend with IQ, given the width of the intervals

# %% [markdown]
# **Implementation** `cell4_5_plot_cate(cate_df, dml_df, setup_txt, ...)`
# - Draws the precomputed intervals with `matplotlib` `errorbar()`
# - Draws the DML effect as a dashed line and its interval as a gray band

# %%
hintros.print_obj_info(analysis.cell4_5_plot_cate)

# %% [markdown]
# # Part 5: Resource allocation policies (exploratory)

# %% [markdown]
# ## Cell 5.1: How the policies run

# %%
# Policies of `run_policy_simulation()`, run with `run_experiment()`:
#
# - Each policy adds a budget of 100 to the starting capital, then runs the
#   same simulation with the same seed
#   - `egalitarian`: the same amount to every agent
#   - `meritocratic`: in proportion to the talent norm
#   - `performance`: in proportion to the current capital
#     - Every agent starts with 1.0, so it equals `egalitarian`
#   - `random`: the whole budget to one agent drawn at random
#   - `cate_optimal`: in proportion to the per-agent effects of Cell 4.4
#     - A heuristic, not a proven optimal rule: the effects are of a realized
#       event, not of money
#     - The effects come from the same run that the policy is evaluated on, so
#       the evaluation is in-sample
# - Who is hit by an event does not depend on capital
#   - So every policy run has the same events as the baseline
#   - If the capital floor never binds, agent i ends with C_i0 * M_i, where M_i
#     is the product of the multipliers of its events
#   - A bound floor changes the multiplier of an event with the capital, so
#     this rescaling no longer holds
# - The `sqrt(allocation)` model of the earlier notebook is gone: it was a
#   different experiment, outside the simulator
print("A policy changes the starting capital, not the events of the run")

# %% [markdown]
# ## Cell 5.2: One run per policy

# %%
# Run the baseline config once per policy.
cate_values = cate_df["cate"].to_numpy()
policy_df, policy_results = analysis.cell5_2_run_policy_comparison(
    config, cate_values
)
display(policy_df)
# Outcome: `egalitarian` and `performance` are identical, and `random` has the
# highest total capital (229.4) and Gini (0.690).

# %% [markdown]
# - The run without allocation (`none`) starts from a total of 100: 100 agents
#   with 1.0
# - Each funded policy starts from a total of 200: the same 100, plus the budget
#   of 100
# - The totals are the actual starting capitals: the simulator raises an
#   `initial_capital` below the floor to the floor
# - `egalitarian` and `performance` give the same row
#   - Every agent starts with 2.0 and no agent reaches the floor (Cell 5.3), so
#     the Gini stays at 0.217 and the capital keeps 0.888 of its start, as
#     without a policy
# - `meritocratic` and `cate_optimal` are close to `egalitarian`
#   - Their weights vary little across the agents (Cell 5.3)
# - `random` gives the whole budget to one agent
#   - Its total of 229.4 and Gini of 0.690 depend on how lucky that one agent
#     was in this run
# - These are single runs with one seed: they do not rank the policies

# %%
hintros.print_obj_info(analysis.cell5_2_run_policy_comparison)

# %% [markdown]
# ## Cell 5.3: Policies rescale each agent's luck while the floor never binds

# %%
# Infer the starting capital that each policy run implies for each agent.
display(analysis.cell5_3_compute_implied_initial_capital(policy_results))
# Outcome: every run has the same event counts as the baseline and no agent at
# the floor, and the implied starting capital sums to the actual total: 100
# without allocation and 200 for each funded policy.

# %% [markdown]
# - Every policy run keeps the event counts of the baseline
# - No agent reaches the capital floor in any run, so `rescaling_holds` is True
# - $M_i$ is the final capital over the actual starting capital of the
#   baseline
#   - The actual one, since an `initial_capital` below the floor starts at the
#     floor
# - Dividing the final capital by $M_i$ gives back the starting capital of each
#   run
#   - 1.0 for every agent without allocation (`none`), a total of 100
#   - With a funded policy, 1.0 plus the allocation, a total of 200
#   - 2.0 for every agent with `egalitarian` and `performance`
#   - 1.0 for 99 agents and 101.0 for the winner with `random`
# - So in this baseline a policy can only move capital toward agents whose luck
#   is already decided by the seed
# - With a higher floor, or more or larger unlucky events, the floor can bind
#   - Then the helper reports the agents at the floor and does not infer the
#     starting capital

# %%
hintros.print_obj_info(analysis.cell5_3_compute_implied_initial_capital)

# %% [markdown]
# # Part 6: Comparison with the paper

# %% [markdown]
# ## Cell 6.1: The baseline over 50 seeds
#
# - The paper reports a Gini averaged over runs, so one seed is not enough to
#   compare with it
# - This runs the same config with seeds 0 to 49, changing only the seed

# %%
# Run the baseline config once per seed.
seeds = list(range(50))
sweep_df = analysis.cell6_1_run_seed_sweep(config, seeds)
display(sweep_df.describe().loc[["mean", "std", "min", "max"]].T)
# Outcome: Gini 0.276 on average, from 0.217 to 0.334, and 2.17 realized
# beneficial events per agent on average.

# %% [markdown]
# - Seed 42, the baseline, has the lowest Gini of the 50 seeds: 0.217
# - The talent correlation varies a lot with the seed
#   - 0.125 on average, with a std of 0.094, from -0.097 to 0.260

# %% [markdown]
# ## Cell 6.2: The paper's values

# %%
# Put the paper's values next to the baseline and the seed average.
comparison_df = analysis.cell6_2_compare_with_paper(
    results,
    sweep_df,
    dml_df["effect"].iloc[0],
    cate_summary_df["std_cate"].iloc[0],
)
display(comparison_df)
# Outcome: Gini 0.38 in the paper, 0.217 for the baseline, and 0.276 over seeds.

# %% [markdown]
# - `baseline` is the single run with seed 42, and `seeds_mean` is the average
#   over seeds 0 to 49
# - The paper does not say how most of its values were computed: only the Gini
#   is stated as an average over runs
# - Inequality is lower than in the paper, also on average over seeds
#   - Gini 0.276 against 0.38, and no seed above 0.334
#   - Top 10% share 22.4% against 28%
# - The agents realize fewer beneficial events than in the paper
#   - 2.17 per agent on average against 4.8
#   - 4.71 in the top decile against 8.3
# - The talent correlation is close on average: 0.125 against 0.12
# - The event correlations differ in both directions
#   - Realized beneficial events: 0.652 against 0.78
#   - Net events: 0.942 against 0.82
# - The DML effect of the baseline is 0.157 against 0.124, and the CATE std is
#   0.013 against 0.03: both are single-seed estimates here
# - The parameters were not tuned to match the paper

# %% [markdown]
# ## Cell 6.3: Seed variability
#
# **Goal**
# - See where the baseline and the paper fall in the spread over seeds

# %% [markdown]
# **Description**
# - Inputs
#   - `sweep_df`: the per-seed metrics of Cell 6.1
#   - `comparison_df`: the paper and baseline values of Cell 6.2
#   - `setup_txt`: the set-up shown in the title
#
# - Panels
#   - `Gini of final capital`: histogram over seeds
#   - `r(lucky events, log(capital))`: histogram over seeds

# %%
# Plot the spread over seeds with the baseline and the paper.
analysis.cell6_3_plot_seed_sweep(
    sweep_df, comparison_df, setup_txt, figsize=(8.5, 3)
)

# %% [markdown]
# **Guided usage**
# - Look at the Gini panel
#   - Observe the baseline at the left edge and the paper beyond the right edge
# - Look at the correlation panel
#   - Observe the paper's 0.78 beyond every seed

# %% [markdown]
# **Implementation** `cell6_3_plot_seed_sweep(sweep_df, comparison_df, ...)`
# - Draws one `seaborn` histogram per metric over the seeds
# - Draws the seed mean (dashed), the baseline (solid), and the paper (dotted)

# %%
hintros.print_obj_info(analysis.cell6_3_plot_seed_sweep)

# %% [markdown]
# ## Cell 6.4: Model differences with the paper
#
# - Spillovers
#   - The paper: only a successful lucky event creates a spillover
#   - The code: the spillover check runs after every lucky event
#   - The notebook describes the code and does not change it
# - Talent spread: the paper is not consistent with itself
#   - Its simulation set-up draws each talent from $N(0.5, 0.15)$ clipped to
#     $[0, 1]$, as the config does
#   - Its results report a talent std of 0.11, without saying which measure
#   - The simulation gives about 0.148 per talent after clipping (Cell 6.1)
# - Performance policy
#   - The paper reports different results for `performance` and `egalitarian`
#   - The code gives the same result when all agents start with the same
#     capital (Cell 5.2)
# - Policy totals
#   - The paper: total capital 743 without allocation
#   - The baseline: 88.8 (Cell 5.2)

# %% [markdown]
# # Part 7: Conclusions

# %% [markdown]
# ## Cell 7.1: What the baseline shows
#
# - With 100 agents and 80 periods, inequality emerges from equal starts
#   - Gini 0.217 for seed 42 and 0.276 on average over 50 seeds
#   - The median agent ends below its starting capital
# - The seed-42 run does not show a statistically distinguishable linear
#   association between talent norm and log-capital, while the event counts
#   are strongly associated with it
#   - Over seeds the talent correlation is 0.125 on average, and varies a lot
# - Adjusting for the talents, one more realized beneficial event raises
#   log-capital by 0.157 (95% CI [0.106, 0.208])
#   - The observed average event-level log gain, 0.199, lies inside the
#     interval: consistent with the estimate, but not a validation of it
#   - The per-agent effects of the causal forest vary little relative to their
#     uncertainty
# - While the capital floor never binds, as in the baseline, a policy that
#   allocates capital before the run only rescales each agent's luck
#   - None of the single runs ranks the policies

# %% [markdown]
# ## Cell 7.2: Limitations
#
# - Every estimate except the seed sweep comes from one run of 100 agents
# - The causal estimates assume that, given the talents, the realized events are
#   as good as random
# - The CATE-based allocation is a heuristic evaluated in-sample
# - A longer run (e.g., 200 periods) is a separate experiment, not part of this
#   baseline
