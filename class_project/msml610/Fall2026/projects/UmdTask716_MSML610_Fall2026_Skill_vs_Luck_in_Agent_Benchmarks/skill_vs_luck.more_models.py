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
# # Does a bigger leaderboard hold up to run luck?
#
# - This notebook adds 3 models to the 3 of phase 1 and checks whether a
#   leaderboard of 6 models can be read from 1 run
#   - It is Study D of `preregistration_2.md`, whose design and predictions
#     were committed before any phase-2 answer was scored
#   - The new models are DeepSeek-Coder 6.7B, Llama 3.1 8B, and CodeGemma 7B,
#     with 1 greedy run and 3 runs at 0.8 on both benchmarks
#   - On MBPP+, Llama 3.1 8B, CodeGemma 7B, and the DeepSeek-Coder 1.3B of
#     phase 1 have published scores within 2.1 points of each other
# - The pedagogical arc:
#   - The greedy scores of the new models against the published ones
#   - How much of their variance is run-to-run luck
#   - Whether the 3 close models on MBPP+ can be told apart
#   - How often a 1-run leaderboard of the 6 models keeps their order
#   - Which predictions held

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
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
# # Part 1: The new models

# %% [markdown]
# ## Cell 1.1: Load the scores of both phases
#
# - The 6-model leaderboard uses the one-shot scaffold at temperature 0.8 with
#   the first 3 runs of every model, so all the models have the same number of
#   runs

# %%
# Read the local scores of both phases and the settings of the analysis.
PHASE_1_DIR = "data/coding_runs"
PHASE_2_DIR = "data/coding_runs_phase2"
BENCHMARKS = ["humaneval", "mbpp"]
TEMPERATURE = 0.8
N_RUNS = 3
NEW_MODELS = list(rutils.PHASE_2_MODELS)
ALL_MODELS = list(rutils.PHASE_1_MODELS) + NEW_MODELS
old_scores = rutils.load_scores(PHASE_1_DIR, pattern="local_*.csv.gz")
new_scores = rutils.load_scores(PHASE_2_DIR)
scores = rutils.combine_scores(old_scores, new_scores)
new_models = scores[scores["model"].isin(NEW_MODELS)]

print("scores.shape=", scores.shape)
display(rutils.compute_score_table(new_models).round(3))
# Outcome: the greedy one-shot plus scores of the new models are 71.3 on
# HumanEval+ and 63.8 on MBPP+ for DeepSeek-Coder 6.7B, 64.0 and 63.2 for Llama
# 3.1 8B, and 51.2 and 56.3 for CodeGemma 7B. At 0.8 every new agent scores 1.1
# to 6.7 points below its greedy score.

# %% [markdown]
# ## Cell 1.2: Greedy scores against the published ones
#
# - As in phase 1, the greedy one-shot run checks the harness against the
#   published EvalPlus scores, which use full-precision weights

# %%
# Compare the greedy one-shot scores of the new models with the published ones.
validation = rutils.compute_validation_table(new_models)

display(validation.round(3))
# Outcome: 5 of the 6 greedy scores are within 2 points of the published ones,
# DeepSeek-Coder 6.7B on HumanEval+ exactly, but Llama 3.1 8B gets 63.2 on
# MBPP+ against a published 55.6 (+7.6 points), while it matches on HumanEval+
# (+1.2).

# %% [markdown]
# # Part 2: How much of the variance is run luck?

# %% [markdown]
# ## Cell 2.1: Task and run parts of the variance of the new agents
#
# - Not a prediction of Study D: the same split as in phase 1, from the 3 runs
#   at 0.8 of each new agent
#   - These run shares are also the truth of the phase-2 cases of Study B

# %%
# Split the variance of every new agent, on each benchmark.
variance = {
    benchmark: rutils.compute_variance_table(
        new_models,
        source="local",
        benchmark=benchmark,
        temperature=TEMPERATURE,
    )
    for benchmark in BENCHMARKS
}

for benchmark, table in variance.items():
    print("benchmark=", benchmark)
    display(table.round(4))
# Outcome: runs are 35% to 53% of the variance of 1 run on HumanEval+ and 28%
# to 35% on MBPP+, the same range as the old models at 0.8 (28% to 52%).

# %% [markdown]
# # Part 3: Can the 3 close models on MBPP+ be told apart?

# %% [markdown]
# ## Cell 3.1: Paired comparisons of the cluster
#
# - The 3 pairs of Llama 3.1 8B, CodeGemma 7B, and DeepSeek-Coder 1.3B, with
#   the one-shot scaffold, over the tasks and the first 3 runs at 0.8
# - The Holm adjustment covers the 3 tests

# %%
# Compare the 3 pairs of the MBPP+ cluster.
LLAMA = "llama3.1:8b-instruct-q4_K_M"
CODEGEMMA = "codegemma:7b-instruct-q4_K_M"
DEEPSEEK_SMALL = "deepseek-coder:1.3b-instruct-q4_K_M"
pairs = [
    ((LLAMA, "one_shot"), (CODEGEMMA, "one_shot")),
    ((LLAMA, "one_shot"), (DEEPSEEK_SMALL, "one_shot")),
    ((CODEGEMMA, "one_shot"), (DEEPSEEK_SMALL, "one_shot")),
]
first_runs = scores[scores["run"] < N_RUNS]
comparisons = rutils.compare_agent_pairs(
    first_runs,
    pairs,
    source="local",
    benchmarks=["mbpp"],
    temperature=TEMPERATURE,
)

display(comparisons.round(4))
# Outcome: in our runs the cluster is not close: Llama 3.1 8B is 5.0 points
# above CodeGemma 7B and 12.0 above DeepSeek-Coder 1.3B, and CodeGemma is 7.0
# above DeepSeek-Coder 1.3B, so all 3 pairs are distinguishable (Holm p of
# 0.0044 or less).

# %% [markdown]
# # Part 4: How stable is a 1-run leaderboard of 6 models?

# %% [markdown]
# ## Cell 4.1: Order of the 6 models in random 1-run leaderboards on MBPP+
#
# - A 1-run leaderboard picks 1 of the first 3 runs of each model, with the
#   one-shot scaffold
# - Every combination of runs is taken once (3^6 = 729), so the shares are
#   exact instead of coming from random draws

# %%
# Draw random 1-run leaderboards of the 6 models on MBPP+.
matrices = rutils.get_first_runs_matrices(
    scores,
    models=ALL_MODELS,
    scaffold="one_shot",
    benchmark="mbpp",
    temperature=TEMPERATURE,
    n_runs=N_RUNS,
)
stability, flips = rutils.compute_ranking_stability(matrices, n_draws=None)

print("stability=", stability.round(3).to_dict())
display(flips.round(3))
# Outcome: only 44.4% of the 729 leaderboards keep the order of the 6 models.
# Every break comes from DeepSeek-Coder 6.7B and Llama 3.1 8B, 1.1 points
# apart, which swap in 5 of their 9 pairs of runs; OpenCoder 8B and CodeQwen
# 7B, 2.0 points apart, tie in 1 of 9 (a tie counts as half a swap), and no
# other pair ever swaps.

# %% [markdown]
# ## Cell 4.2: Plot of the flip rate of each pair

# %%
# Plot how often each pair swaps places, against its gap.
rutils.plot_flip_rates(
    flips, title="Pairs swapped by 1 run (mbpp, 6 models)", figsize=(10, 5)
)
# Outcome: only the 2 closest pairs ever swap, and DeepSeek-Coder 6.7B and
# Llama 3.1 8B swap in 56% of the leaderboards, above the coin-flip line.

# %% [markdown]
# ## Cell 4.3: The same leaderboard on HumanEval+
#
# - Not in the pre-registration: an extra look at Cell 4.1 on the other
#   benchmark

# %%
# Draw random 1-run leaderboards of the 6 models on HumanEval+.
humaneval_matrices = rutils.get_first_runs_matrices(
    scores,
    models=ALL_MODELS,
    scaffold="one_shot",
    benchmark="humaneval",
    temperature=TEMPERATURE,
    n_runs=N_RUNS,
)
humaneval_stability, _ = rutils.compute_ranking_stability(
    humaneval_matrices, n_draws=None
)

print("humaneval_stability=", humaneval_stability.round(3).to_dict())
# Outcome: on HumanEval+, 88.9% of the leaderboards keep the order of the 6
# models, twice the share on MBPP+.

# %% [markdown]
# # Part 5: Which predictions held?

# %% [markdown]
# ## Cell 5.1: Predictions against the results
#
# - The rules follow `preregistration_2.md`, which was committed before any
#   phase-2 answer was scored

# %%
# Check each prediction of Study D.
predictions = rutils.check_new_model_predictions(
    validation=validation, comparisons=comparisons, stability=stability
)

display(predictions)
# Outcome: D3 holds and D1 and D2 do not: D1 misses on 1 of its 6 values,
# Llama 3.1 8B on MBPP+, and D2 misses because the 3 models that the published
# scores put within 2.1 points of each other are 5 to 12 points apart in our
# runs.

# %% [markdown]
# # Part 6: Takeaways
#
# - With 6 models on MBPP+, fewer than half of the 1-run leaderboards (44.4%)
#   keep the order, and a single pair 1.1 points apart causes every break
# - The published scores did not tell which models would be close in our
#   setup:
#   - The 3 models within 2.1 points of each other in the published MBPP+
#     scores are 5 to 12 points apart in our runs at 0.8
#   - DeepSeek-Coder 6.7B and Llama 3.1 8B, 10 points apart in the published
#     MBPP+ scores, are 1.1 points apart in our runs
# - The harness matches 5 of the 6 published greedy scores within 2 points;
#   Llama 3.1 8B on MBPP+ is 7.6 points above, and we have not found why
# - The new agents have about as much run luck at 0.8 as the old ones: 28% to
#   53% of the variance of 1 run
