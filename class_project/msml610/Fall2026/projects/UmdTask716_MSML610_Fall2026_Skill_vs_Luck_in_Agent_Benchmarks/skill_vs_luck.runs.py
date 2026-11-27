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
#   - It reads the full scores: HumanEval+ (164 tasks) and MBPP+ (378 tasks),
#     1 greedy run and 5 runs at temperature 0.8 per task
#   - The analyses and the predictions were committed in `preregistration.md`
#     before any of these scores existed
# - The pedagogical arc:
#   - The agents and their scores
#   - How much of the variance of 1 run is run-to-run luck
#   - Which pairs of agents differ, over the tasks and the runs
#   - How often a leaderboard from 1 run puts the agents in the wrong order
#   - Whether more runs or more tasks help more
#   - How temperature changes run-to-run luck, in the public EvalPlus samples
#   - Which predictions held

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
#   - `source` is `local` for our runs and `evalplus` for the public samples,
#     which only Part 6 uses
# - The scaffolds share the first attempt, so `one_shot` and `repair` are paired

# %%
# Read the scores and the settings of the analysis.
SCORES_DIR = "data/coding_runs"
BENCHMARKS = ["humaneval", "mbpp"]
TEMPERATURE = 0.8
scores = utils.load_scores(SCORES_DIR)

print("scores.shape=", scores.shape)
display(utils.compute_score_table(scores[scores["source"] == "local"]).round(3))
# Outcome: at temperature 0.8 the one-shot plus scores are 74.1% (CodeQwen),
# 72.7% (OpenCoder), and 55.5% (DeepSeek) on HumanEval+, and 66.0%, 68.8%, and
# 48.7% on MBPP+; sampling costs 2 to 7 points against the greedy run.

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
# Outcome: all 6 greedy scores are within 2 points of the published ones, from
# -1.4 (DeepSeek, MBPP+) to +1.9 (OpenCoder, HumanEval+), and CodeQwen gets the
# published 78.7 on HumanEval+ exactly, so the harness and the 4-bit weights
# hold up.

# %% [markdown]
# ## Cell 1.3: Failure modes
#
# - Each failure is labeled by running the first failing input again
# - A task flips for an agent when some of its runs pass and some fail
#   - Those failures are run-to-run luck
#   - A task that fails in every run is beyond the agent

# %%
# Count the failure modes of the sampled runs, by kind of task.
for benchmark in BENCHMARKS:
    failures = utils.compute_failure_modes(
        scores, source="local", benchmark=benchmark, temperature=TEMPERATURE
    )
    print("benchmark=", benchmark)
    display(failures)
# Outcome: wrong answers are 71% to 84% of the one-shot failures on both kinds of
# tasks; timeouts come mostly from tasks that always fail (21 of 29 on
# HumanEval+), and repair cuts their missing functions on MBPP+ from 58 to 42.

# %% [markdown]
# ## Cell 1.4: Flaky tests
#
# - Every solution is scored twice: a solution that passes once and fails once
#   has a flaky test, which is luck of the environment, not of the model

# %%
# Compute the share of solutions whose 2 scorings disagree.
flaky_share = utils.compute_flaky_share(scores)

print("flaky_share=", flaky_share)
# Outcome: 0.0, none of the 19,512 answers changed its result when scored again.

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
# Split the variance of every agent, on each benchmark.
variance = {
    benchmark: utils.compute_variance_table(
        scores, source="local", benchmark=benchmark, temperature=TEMPERATURE
    )
    for benchmark in BENCHMARKS
}

for benchmark, table in variance.items():
    print("benchmark=", benchmark)
    display(table.round(4))
# Outcome: run luck is 28% to 52% of the variance of a 1-run outcome, the most
# for CodeQwen on HumanEval+ (52% one-shot) and the least for OpenCoder on MBPP+
# (28% with repair).

# %% [markdown]
# ## Cell 2.2: Plot of the 2 parts

# %%
# Plot the task part and the run part of every agent.
for benchmark, table in variance.items():
    utils.plot_variance_parts(
        table,
        title=f"Task and run parts of the variance ({benchmark})",
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
    benchmarks=BENCHMARKS,
    temperature=TEMPERATURE,
)

display(comparisons.round(4))
# Outcome: the close pair stays indistinguishable (Holm-adjusted p 0.26 to 1),
# even with OpenCoder 2.8 points ahead on MBPP+ at raw p 0.043 and 0.048; the
# weaker model trails by 16 to 20 points (adjusted p below 0.0001); repair beats
# one-shot in 4 of the 6 model and benchmark pairs.

# %% [markdown]
# # Part 4: How stable is a leaderboard from 1 run?

# %% [markdown]
# ## Cell 4.1: Order of the agents in random 1-run leaderboards
#
# - A leaderboard usually runs each agent once
# - Picking 1 run per agent at random shows how often that order matches the
#   order over all the runs

# %%
# Draw random 1-run leaderboards, on each benchmark.
matrices = {
    benchmark: utils.get_agent_matrices(
        scores, source="local", benchmark=benchmark, temperature=TEMPERATURE
    )
    for benchmark in BENCHMARKS
}
stability = {
    benchmark: utils.compute_ranking_stability(agent_matrices)
    for benchmark, agent_matrices in matrices.items()
}

for benchmark, (summary, flips) in stability.items():
    print("benchmark=", benchmark)
    print("summary=", summary.round(3).to_dict())
    display(flips.round(3))
# Outcome: only 18.5% (HumanEval+) and 25.3% (MBPP+) of the 1-run leaderboards
# keep the order of every pair; pairs up to 3 points apart swap 8% to 44% of the
# time, while the 16 to 21 point gaps to DeepSeek never swap.

# %% [markdown]
# ## Cell 4.2: Plot of the flip rate of each pair

# %%
# Plot how often each pair swaps places, against its gap.
for benchmark, (_, flips) in stability.items():
    utils.plot_flip_rates(
        flips, title=f"Pairs swapped by 1 run ({benchmark})", figsize=(10, 5)
    )

# %% [markdown]
# ## Cell 4.3: The 3 models alone
#
# - Not in the pre-registration: an extra look at Cell 4.1
# - A model with and without repair is almost the same agent, so those pairs
#   swap often; keeping 1 scaffold at a time leaves the 3 distinct models

# %%
# Draw random 1-run leaderboards of the 3 models, for each scaffold.
for benchmark, agent_matrices in matrices.items():
    for scaffold in ("one_shot", "repair"):
        models_only = {
            label: matrix
            for label, matrix in agent_matrices.items()
            if label.endswith(scaffold)
        }
        summary, _ = utils.compute_ranking_stability(models_only)
        print("benchmark=", benchmark, "scaffold=", scaffold)
        print("summary=", summary.round(3).to_dict())
# Outcome: with the 3 models alone, 76% to 96% of the 1-run leaderboards keep
# the order; the swaps come from the close pair, and DeepSeek is never misplaced.

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
# Split the variance of the gap of the close pair, on each benchmark.
parts = {
    benchmark: utils.compute_pair_variance_components(
        agent_matrices["codeqwen 7b / one_shot"],
        agent_matrices["opencoder 8b / one_shot"],
    )
    for benchmark, agent_matrices in matrices.items()
}

for benchmark, values in parts.items():
    print("benchmark=", benchmark)
    print("parts=", {key: round(value, 4) for key, value in values.items()})
# Outcome: for the close pair, the run part of the gap variance is about 3 times
# the task part (0.174 vs 0.053 on HumanEval+, 0.147 vs 0.046 on MBPP+), so more
# runs help, but the HumanEval+ gap of 1.5 points still needs about 3,200 tasks
# with 5 runs, and about 2,000 even with unlimited runs.

# %%
# Plot the tasks needed against the gap, for a few runs per task.
BENCHMARK_SIZES = {"humaneval": 164, "mbpp": 378}
for benchmark, values in parts.items():
    utils.plot_min_tasks_vs_runs(
        values["sigma2_task_diff"],
        values["sigma2_run_diff"],
        n_tasks=BENCHMARK_SIZES[benchmark],
        title=f"Tasks needed to separate CodeQwen and OpenCoder ({benchmark})",
        figsize=(10, 5),
    )

# %% [markdown]
# # Part 6: Does temperature change run luck?

# %% [markdown]
# ## Cell 6.1: Run variance of the public samples at 2 temperatures
#
# - EvalPlus published 200 samples per HumanEval+ task for 8 models at
#   temperatures 0.2 and 0.8, and the first 50 per task are scored here
# - Each sample of a task plays the role of 1 run

# %%
# Compare the run variance of every public model at the 2 temperatures.
temperature_table = utils.compute_temperature_table(
    scores, benchmark="humaneval"
)

display(temperature_table.round(4))
# Outcome: from temperature 0.2 to 0.8 the run variance grows 1.8 to 3.0 times,
# and the run share goes from 15% to 30% at 0.2 to 42% to 55% at 0.8; only 4 of
# the 8 models reach the factor of 2 that P7 predicted.

# %% [markdown]
# ## Cell 6.2: Plot of the run variance at the 2 temperatures

# %%
# Plot the run variance at 0.8 against the one at 0.2.
utils.plot_temperature_effect(
    temperature_table,
    title="Run variance of the public samples (humaneval)",
    figsize=(10, 5),
)

# %% [markdown]
# ## Cell 6.3: 1-run leaderboards of the public models
#
# - Not in the pre-registration: an extra look
# - The same random 1-run leaderboards as in Part 4, for the 8 public models
#   sampled at both temperatures

# %%
# Draw random 1-run leaderboards of the public models, at each temperature.
public_stability = {}
for temperature in (0.2, 0.8):
    public_matrices = utils.get_agent_matrices(
        scores, source="evalplus", benchmark="humaneval", temperature=temperature
    )
    # Keep the models of Cell 6.1, so both temperatures rank the same models.
    public_matrices = {
        label: matrix
        for label, matrix in public_matrices.items()
        if label.split(" / ")[0] in temperature_table.index
    }
    public_stability[temperature] = utils.compute_ranking_stability(
        public_matrices
    )

for temperature, (summary, _) in public_stability.items():
    print("temperature=", temperature)
    print("summary=", summary.round(3).to_dict())
# Outcome: with the same 8 models, 23.2% of the 1-run leaderboards keep the
# order of every pair at temperature 0.2, and only 13.3% at 0.8.

# %% [markdown]
# # Part 7: Were the predictions right?

# %% [markdown]
# ## Cell 7.1: Predictions against results
#
# - Each prediction of `preregistration.md` is checked with the rule committed
#   before any scoring, in `check_predictions()`

# %%
# Check every pre-registered prediction.
predictions = utils.check_predictions(
    validation=validation,
    score_table=utils.compute_score_table(scores),
    variance_tables=variance,
    comparisons=comparisons,
    flaky_share=flaky_share,
    temperature_table=temperature_table,
    weak_model=MODELS["deepseek"],
    pair_models=(MODELS["codeqwen"], MODELS["opencoder"]),
)

display(predictions)
# Outcome: 4 of the 7 predictions hold (P1, P2, P4, P6); P3 fails because run
# luck is above 40% for 3 agents on HumanEval+ (up to 52%), P5 because CodeQwen
# gains more from repair on HumanEval+, and P7 because only 4 of the 8 public
# models double their run variance.

# %% [markdown]
# # Part 8: Takeaways and next steps

# %% [markdown]
# ## Takeaways
#
# - The harness reproduces the published EvalPlus scores: the 6 greedy scores
#   are within 2 points, even with 4-bit weights, and no test is flaky
# - At temperature 0.8, run-to-run luck is a large part of the noise
#   - It is 28% to 52% of the variance of a 1-run outcome for our 6 agents
#   - In the public samples it is 42% to 55% at 0.8, against 15% to 30% at 0.2
# - A 1-run leaderboard often has the order wrong
#   - Only 18.5% (HumanEval+) and 25.3% (MBPP+) of them keep the order of all 6
#     agents, mostly because a model with and without repair is almost the same
#     agent
#   - With the 3 models alone, 76% to 96% keep the order, and the close pair is
#     the one that swaps
#   - On HumanEval+, the gap of the close pair in the 5 runs goes from -2.4 to
#     +6.1 points, so 1 run can show either model ahead
#   - In the public samples, the share drops from 23.2% at temperature 0.2 to
#     13.3% at 0.8
# - The close pair cannot be told apart, even with 5 runs per task, while the
#   weaker model is 16 to 20 points behind and always distinguishable
#   - The HumanEval+ gap of the pair, 1.5 points, would need about 3,200 tasks
#     with 5 runs, and about 2,000 even with unlimited runs
# - Repair helps a little: +0.1 to +1.6 points, significant in 4 of 6 cases
# - 4 of the 7 pre-registered predictions held; P3 (run share), P5 (repair), and
#   P7 (temperature) did not

# %% [markdown]
# ## Next steps
#
# - Combine the run share measured here with the SWE-bench Verified intervals,
#   to see how much of that leaderboard a rerun would reorder
# - Milestone 3: split-half and cross-benchmark rank correlations, and a 2PL IRT
#   model of task difficulty
# - Milestone 4: the winner's curse of picking the best of several attempts
