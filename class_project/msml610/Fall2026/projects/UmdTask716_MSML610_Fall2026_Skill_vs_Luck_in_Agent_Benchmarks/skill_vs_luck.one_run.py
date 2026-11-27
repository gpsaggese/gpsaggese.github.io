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
# # Can run luck be estimated from 1 run?
#
# - Leaderboards publish 1 run per agent, so the share of the variance of a
#   score that is run-to-run luck (the run share) is never measured
#   - With 1 run, luck cannot be told apart from an agent being good at a
#     specific task without a model of the tasks
# - This notebook builds the estimator of Study B in `preregistration_2.md`
#   and tunes it on the development set, the public EvalPlus samples
#   - Our own runs are the test set, used once after the tuned estimator is
#     committed, so this notebook never loads them
# - The estimator:
#   1. Fit a Rasch or 2PL model to an agents x tasks matrix with 1 run per
#      agent
#   2. Estimate $\sigma^2_{run}$ of an agent as the average over the tasks of
#      p (1 - p), with p the fitted chance of a pass, times a calibration
#      factor
#   3. Divide it by score x (1 - score) to get the run share
# - The pedagogical arc:
#   - The development set and its true run shares
#   - The estimator before tuning
#   - Tuning the model, the penalty, the extra agents, and the calibration
#   - How many tasks need a second run

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
import pandas as pd

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebook

import skill_vs_luck_runs_utils as rutils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
rutils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Part 1: The development set

# %% [markdown]
# ## Cell 1.1: Load the public samples
#
# - The public EvalPlus samples on HumanEval+ come from 8 models sampled at
#   temperatures 0.2 and 0.8, scored on the first 50 of the 200 released
#   samples per task
#   - Each sample of a task plays the role of 1 run
# - 31 public models also have a greedy sample, the possible extra agents
# - Only the `evalplus_humaneval_*` files are loaded: the `local_*` files are
#   the test set

# %%
# Load the public HumanEval+ scores, and none of our runs.
public = rutils.load_scores(
    "data/coding_runs", pattern="evalplus_humaneval_*.csv.gz"
)
hdbg.dassert_eq(public["source"].unique().tolist(), ["evalplus"])
# Keep the models sampled at both temperatures.
temperatures = public.groupby("model")["temperature"].agg(set)
models = sorted(
    model for model, temps in temperatures.items() if {0.2, 0.8} <= temps
)
samples = {
    temperature: {
        model: rutils.get_outcome_matrix(
            public,
            source="evalplus",
            benchmark="humaneval",
            model=model,
            scaffold="one_shot",
            temperature=temperature,
        )
        for model in models
    }
    for temperature in (0.2, 0.8)
}
extras = rutils.get_public_greedy_agents(public, "humaneval")

print("models=", models)
print("samples=", samples[0.2][models[0]].shape, "extras=", extras.shape)
# Outcome: 8 models with 164 tasks x 50 samples at each temperature, and 31
# public greedy agents.

# %% [markdown]
# ## Cell 1.2: The true run shares
#
# - The truth of a case (1 model at 1 temperature) is the run share measured
#   from all its 50 samples, as in phase 1
# - These 16 cases are the development set

# %%
# Measure the run share of each case from all its samples.
truth = pd.DataFrame(
    {
        temperature: {
            model: rutils.compute_variance_components(matrix)["run_share"]
            for model, matrix in by_model.items()
        }
        for temperature, by_model in samples.items()
    }
)

display(truth.round(3))
# Outcome: the run share is 15% to 30% at 0.2 and 42% to 55% at 0.8, the same
# numbers as in Part 6 of the runs notebook.

# %% [markdown]
# # Part 2: The estimator before tuning

# %% [markdown]
# ## Cell 2.1: Run a first grid of settings
#
# - Each case gets 30 draws: in each draw, the case and some other sampled
#   models get 1 sample drawn at random per task as their 1 run
#   - The matrix holds 3 sampled models at 0.2 and 6 at 0.8, like the test
#     matrices, which hold our 3 old models at 0.2 and all 6 models at 0.8
#   - With `extras`, the matrix also holds the public greedy agents of the
#     models that are not in it
#   - Every setting sees the same draws
# - The setting before any tuning is `rasch 0.01 alone`: Rasch, the penalty
#   0.01, and no extra agents
# - This cell takes about 6 minutes

# %%
# Run the first grid of settings on both temperatures.
N_OTHERS = {0.2: 2, 0.8: 5}
N_DRAWS = 30
UNTUNED = "rasch 0.01 alone"
settings = rutils.get_one_run_settings(
    ["rasch", "2pl"], [0.001, 0.01, 0.1, 1.0], [False, True]
)
untuned = {**settings[UNTUNED], "calibration": 1.0}
hdbg.dassert_eq(untuned, rutils.UNTUNED_ONE_RUN_SETTING)
results = pd.concat(
    [
        rutils.evaluate_one_run_estimator(
            samples[temperature], extras, settings, n_others, n_draws=N_DRAWS
        ).assign(temperature=temperature)
        for temperature, n_others in N_OTHERS.items()
    ],
    ignore_index=True,
)

print("settings=", len(settings), "rows=", len(results))
# Outcome: 16 settings x 16 cases x 30 draws.

# %% [markdown]
# ## Cell 2.2: Check the estimator before tuning
#
# - `above` is the share of the draws where the estimate is above the truth
# - B2 predicts that the estimate before tuning is above the truth in more
#   than half of the 18 phase-2 test cases, since it counts agent-task
#   strengths as luck

# %%
# Compare the estimate before tuning with the truth, at each temperature.
cases = rutils.summarize_one_run_cases(results, UNTUNED, 1.0)
before = results[results["setting"] == UNTUNED]
by_temperature = cases.groupby(level="temperature")[
    ["truth", "estimate", "within"]
].mean()
by_temperature["above"] = (
    (before["estimate"] > before["truth"]).groupby(before["temperature"]).mean()
)

display(by_temperature.round(3))
# Outcome: before tuning, the estimate is above the truth in 98% of the draws
# at 0.2 (32.5% against 22.2% on average) and in 21% at 0.8 (47.5% against
# 50.3%), so it is within 10 points in 95% of the draws at 0.8 but only 46% at
# 0.2.

# %% [markdown]
# ## Cell 2.3: Compare the first grid
#
# - The factor of each setting is fit on all 16 cases
# - `cv_error` and `cv_within` fit the factor of each model on the other 7
#   models, so they estimate the error on a new model
# - `mean_0.2` and `mean_0.8` are the mean estimates before calibration, while
#   the truth averages 0.222 and 0.503
# - Sorted by `cv_error`, the rule for picking the setting

# %%
# Summarize the errors of each setting.
summary = rutils.summarize_one_run_errors(results)
# Add the mean estimate before calibration at each temperature.
means = results.groupby(["setting", "temperature"])["estimate"].mean().unstack()
means.columns = [f"mean_{temperature:g}" for temperature in means.columns]

display(summary.join(means).sort_values("cv_error").round(3))
# Outcome: without extra agents, 2PL with the penalty 0.1 and Rasch with 0.001
# are the best, with 5.6 and 5.7 points of error on a new model. With the extra
# agents every setting is worse, within 10 points in at most 45% of the draws:
# their estimate moves by only 1 to 4 points from 0.2 to 0.8, against 15 to 19
# points for the best settings without them and 28 points for the truth, since
# the greedy agents have no run luck and set the task difficulties.

# %% [markdown]
# # Part 3: Tuning

# %% [markdown]
# ## Cell 3.1: Run a finer grid without extra agents
#
# - The extra agents made every setting worse, so the finer grid tries more
#   penalties without them, around the best ones
# - Same draws as the first grid

# %%
# Run the finer grid and summarize all the settings.
finer = {
    **rutils.get_one_run_settings(["rasch"], [0.0001, 0.0003, 0.003], [False]),
    **rutils.get_one_run_settings(["2pl"], [0.03, 0.3], [False]),
}
finer_results = pd.concat(
    [
        rutils.evaluate_one_run_estimator(
            samples[temperature], extras, finer, n_others, n_draws=N_DRAWS
        ).assign(temperature=temperature)
        for temperature, n_others in N_OTHERS.items()
    ],
    ignore_index=True,
)
all_settings = {**settings, **finer}
all_results = pd.concat([results, finer_results], ignore_index=True)
all_summary = rutils.summarize_one_run_errors(all_results).sort_values(
    "cv_error"
)

print("settings=", len(all_settings))
display(all_summary.round(4))
# Outcome: the best 3 settings are within 0.1 points of each other: 2PL with
# the penalty 0.1 (5.56 points), Rasch with 0.0001 (5.59), and Rasch with
# 0.0003 (5.62). Rasch stays between 5.6 and 6.1 points for penalties up to
# 0.003, while 2PL breaks down at 0.3 (9.35 points).

# %% [markdown]
# ## Cell 3.2: Pick the tuned estimator
#
# - The tuned setting is the one with the smallest `cv_error`, with its factor
#   fit on all 16 cases
# - It is saved as `TUNED_ONE_RUN_SETTING` in `skill_vs_luck_runs_utils.py`,
#   and this cell checks that the 2 agree
# - Its error here is optimistic, since it is the best of 21 settings on the
#   same 16 cases: the test set gives the honest error

# %%
# Pick the setting with the smallest error on new models.
best = all_summary.index[0]
tuned = {
    **all_settings[best],
    "calibration": round(float(all_summary.loc[best, "factor"]), 3),
}
hdbg.dassert_eq(tuned, rutils.TUNED_ONE_RUN_SETTING)

print("best=", best)
print("tuned=", tuned)
# Outcome: the tuned estimator is 2PL with the penalty 0.1, no extra agents,
# and the factor 1.344, the same as `TUNED_ONE_RUN_SETTING`.

# %% [markdown]
# ## Cell 3.3: Errors at each temperature
#
# - `within` is the share of the draws within 10 points of the truth, the
#   rule of B1

# %%
# Compare the 2 settings at each temperature.
factors = {UNTUNED: 1.0, best: tuned["calibration"]}
by_setting = pd.concat(
    {
        setting: rutils.summarize_one_run_cases(all_results, setting, factor)
        .groupby(level="temperature")[["truth", "estimate", "within"]]
        .mean()
        for setting, factor in factors.items()
    }
)

display(by_setting.round(3))
# Outcome: after tuning, the estimate is within 10 points in 93% of the draws
# at 0.2 and 82% at 0.8, against 46% and 95% before tuning; on average it is 3
# points above the truth at 0.2 and 3 points below at 0.8.

# %% [markdown]
# ## Cell 3.4: Plot the estimates against the truth
#
# - Each point is 1 model at 1 temperature, at the mean of its 30 draws, with
#   a bar from the 10th to the 90th percentile of the draws
# - Points in the gray band are within 10 points of the truth

# %%
# Plot both settings.
rutils.plot_one_run_estimates(all_results, factors, figsize=(10, 5))
# Outcome: after tuning, both temperatures sit in the band, but within a
# temperature the estimate does not follow the truth: at 0.2 all the models
# are estimated near 25% while the truth goes from 15% to 30%. The estimator
# finds how much run luck there is at a temperature, not which model has more.

# %% [markdown]
# # Part 4: How many tasks need a second run

# %% [markdown]
# ## Cell 4.1: Second runs on some tasks
#
# - Instead of a model of the tasks, a second run of some tasks measures the
#   run share directly: on those tasks, half the squared difference of the 2
#   runs estimates p (1 - p) without bias
# - For each case, the first 2 samples play the 2 runs, and the number shown
#   is the smallest number of tasks with a second run such that 95% of the
#   random picks of those tasks are within 10 points of the truth

# %%
# Find how many tasks need a second run in each case.
second_runs = pd.DataFrame(
    {
        temperature: {
            model: rutils.compute_second_run_tasks(matrix)
            for model, matrix in by_model.items()
        }
        for temperature, by_model in samples.items()
    }
)

display(second_runs)
# Outcome: to be within 10 points 95% of the time, 73 to 162 of the 164 tasks
# need a second run, and for CodeGen 16B at 0.8 even a second run of every task
# is not enough with these 2 samples: measuring run luck directly costs close
# to a full second run.

# %% [markdown]
# # Part 5: Takeaways
#
# - From 1 run, a 2PL fit of agents that all ran at the same temperature
#   estimates the run share within 10 points in 86% of the draws on the public
#   samples, after a calibration factor fit on the other models
# - The estimate sees the temperature only through the other agents in the
#   matrix: the public greedy agents, which have no run luck, flattened it,
#   and every setting with them got worse
# - Before tuning, the estimate is too high at 0.2 and close at 0.8; 12 of the
#   18 phase-2 test cases are at 0.8, so B2 (above the truth in more than half
#   of them) may not hold
# - The estimator finds the level of run luck at each temperature, but not
#   which model has more
# - Measuring run luck directly needs a second run on 45% to 99% of the tasks
#   to be as reliable
# - The tuned setting is the best of 21 on the same 16 cases, so its error here
#   is optimistic; it is frozen in `TUNED_ONE_RUN_SETTING`, and the test on our
#   30 cases comes after it is committed
