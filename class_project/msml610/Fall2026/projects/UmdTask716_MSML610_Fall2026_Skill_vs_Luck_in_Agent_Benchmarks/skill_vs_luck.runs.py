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
# # How much of a coding-agent score is run-to-run luck?
#
# - This notebook splits the score of 6 local coding agents (3 models x 2
#   scaffolds) into the luck of the tasks and the luck of the runs, and checks
#   which agents can be told apart
#   - For now it reads the pilot scores: 20 HumanEval+ tasks, 2 runs at
#     temperature 0.8
#   - The full scores (HumanEval+ and MBPP+, 5 runs) replace them when the runs
#     finish
# - The pedagogical arc:
#   - The agents and their scores
#   - How much of the variance of 1 run is run-to-run luck
#   - Which pairs of agents differ, over the tasks and the runs
#   - How often a leaderboard from 1 run puts the agents in the wrong order
#   - Whether more runs or more tasks help more

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
import helpers.hnotebook as hnotebook

import skill_vs_luck_runs_utils as utils

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
# # Part 1: The agents and their scores

# %% [markdown]
# ## Cell 1.1: Load the scores
#
# - Each row is 1 solution: 1 model, 1 scaffold, 1 task, and 1 run
# - The scaffolds share the first attempt, so `one_shot` and `repair` are paired

# %%
# Read the scores and the settings of the analysis.
SCORES_DIR = "data/coding_runs_pilot"
BENCHMARK = "humaneval"
TEMPERATURE = 0.8
scores = utils.load_scores(SCORES_DIR)

print("scores.shape=", scores.shape)
display(utils.compute_score_table(scores).round(3))
# Outcome: on these 20 easy tasks, the pilot pass rates at temperature 0.8 go
# from 82.5% (CodeQwen, one-shot) to 97.5% (OpenCoder).

# %% [markdown]
# ## Cell 1.2: Greedy scores against the published ones
#
# - The greedy run (temperature 0) of each model checks the whole harness (the
#   prompt, Ollama, and the scoring) against the published EvalPlus scores
# - The published scores use full-precision weights, while these models run
#   with 4-bit weights, so a few points of difference are expected

# %%
# Compare the greedy one-shot scores with the published ones.
validation = utils.compute_validation_table(scores)

display(validation.round(3))
# Outcome: the greedy scores are 11 to 25 points above the published ones, since
# the 20 pilot tasks are easier than the rest (the public CodeLlama-7B samples
# pass 60% of them and 34% of all 164): this check needs all the tasks.

# %% [markdown]
# ## Cell 1.3: Failure modes
#
# - Each failure is labeled by running the first failing input again
# - A task flips for an agent when some of its runs pass and some fail
#   - Those failures are run-to-run luck
#   - A task that fails in every run is beyond the agent

# %%
# Count the failure modes of the sampled runs, by kind of task.
failures = utils.compute_failure_modes(
    scores, source="local", benchmark=BENCHMARK, temperature=TEMPERATURE
)

display(failures)
# Outcome: the tasks that always fail end in wrong answers only, while all the
# load and runtime errors come from tasks that flip; the repair scaffold removes
# the 2 load errors.

# %% [markdown]
# ## Cell 1.4: Flaky tests
#
# - Every solution is scored twice: a solution that passes once and fails once
#   has a flaky test, which is luck of the environment, not of the model

# %%
# Compute the share of solutions whose 2 scorings disagree.
flaky_share = utils.compute_flaky_share(scores)

print("flaky_share=", flaky_share)
# Outcome: 0.0, no solution changed its result when scored again.

# %% [markdown]
# # Part 2: How much of the variance is run-to-run luck?

# %% [markdown]
# ## Cell 2.1: Task and run parts of the variance
#
# - For each agent, the variance of a 1-run outcome splits into:
#   - $\sigma^2_{task}$: how much the tasks differ in difficulty
#   - $\sigma^2_{run}$: how much the outcome of 1 task changes between runs
# - `run_share` is the part of a 1-run score's noise that more runs would remove

# %%
# Split the variance of every agent.
variance = utils.compute_variance_table(
    scores, source="local", benchmark=BENCHMARK, temperature=TEMPERATURE
)

display(variance.round(4))
# Outcome: with 2 runs on 20 tasks the split is rough, e.g., the task part of
# 4 agents is estimated at 0.

# %% [markdown]
# ## Cell 2.2: Plot of the 2 parts

# %%
# Plot the task part and the run part of every agent.
utils.plot_variance_parts(
    variance,
    title=f"Task and run parts of the variance ({BENCHMARK})",
    figsize=(10, 5),
)

# %% [markdown]
# # Part 3: Which agents differ?

# %% [markdown]
# ## Cell 3.1: Paired comparisons over tasks and runs
#
# - The pre-registered family of comparisons:
#   - The close pair (CodeQwen vs OpenCoder), for each scaffold
#   - The weaker model (DeepSeek 1.3B) against each model of the pair
#   - Repair against one-shot, for each model
# - The Holm adjustment covers the whole family

# %%
# Compare the pre-registered pairs.
MODELS = {
    "deepseek": "deepseek-coder:1.3b-instruct-q4_K_M",
    "codeqwen": "codeqwen:7b-chat-v1.5-q4_K_M",
    "opencoder": "opencoder:8b",
}
pairs = []
for scaffold in ("one_shot", "repair"):
    pairs.append(
        ((MODELS["codeqwen"], scaffold), (MODELS["opencoder"], scaffold))
    )
    for strong in ("codeqwen", "opencoder"):
        pairs.append(
            ((MODELS[strong], scaffold), (MODELS["deepseek"], scaffold))
        )
for model in MODELS.values():
    pairs.append(((model, "repair"), (model, "one_shot")))
comparisons = utils.compare_agent_pairs(
    scores,
    pairs,
    source="local",
    benchmarks=[BENCHMARK],
    temperature=TEMPERATURE,
)

display(comparisons.round(4))
# Outcome: no comparison survives Holm on 20 tasks; the smallest raw p-value is
# 0.021 (OpenCoder vs DeepSeek, one-shot).

# %% [markdown]
# # Part 4: How stable is a leaderboard from 1 run?

# %% [markdown]
# ## Cell 4.1: Order of the agents in random 1-run leaderboards
#
# - A leaderboard usually runs each agent once
# - Picking 1 run per agent at random shows how often that order matches the
#   order over all the runs

# %%
# Draw random 1-run leaderboards.
matrices = utils.get_agent_matrices(
    scores, source="local", benchmark=BENCHMARK, temperature=TEMPERATURE
)
summary, flips = utils.compute_ranking_stability(matrices)

print("summary=", summary.round(3).to_dict())
display(flips.round(3))
# Outcome: only about half of the 1-run leaderboards (50.3%) keep the order of
# every pair whose means differ.

# %% [markdown]
# ## Cell 4.2: Plot of the flip rate of each pair

# %%
# Plot how often each pair swaps places, against its gap.
utils.plot_flip_rates(
    flips, title=f"Pairs swapped by 1 run ({BENCHMARK})", figsize=(10, 5)
)

# %% [markdown]
# # Part 5: More runs or more tasks?

# %% [markdown]
# ## Cell 5.1: Tasks needed for a gap, by runs per task
#
# - More runs only shrink the run part of the variance, so for the same number
#   of evaluations, more tasks always help more than more runs
# - The curves use the variance parts of the close pair with the one-shot
#   scaffold

# %%
# Split the variance of the gap of the close pair.
parts = utils.compute_pair_variance_components(
    matrices["codeqwen / one_shot"], matrices["opencoder / one_shot"]
)

print("parts=", {key: round(value, 4) for key, value in parts.items()})
# Outcome: the run part of the gap variance (0.10) is larger than the task part
# (0.058) for this pair.

# %%
# Plot the tasks needed against the gap, for a few runs per task.
utils.plot_min_tasks_vs_runs(
    parts["sigma2_task_diff"],
    parts["sigma2_run_diff"],
    n_tasks=164,
    title="Tasks needed to separate CodeQwen and OpenCoder (HumanEval+)",
    figsize=(10, 5),
)
