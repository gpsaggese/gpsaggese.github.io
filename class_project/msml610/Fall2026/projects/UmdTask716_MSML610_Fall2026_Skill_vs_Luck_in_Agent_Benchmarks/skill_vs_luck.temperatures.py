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
# # How much run luck is left at temperature 0.2 and at temperature 0?
#
# - This notebook measures the run-to-run luck of the 3 phase-1 models on
#   HumanEval+ at lower temperatures, with both scaffolds
#   - It is Study C of `preregistration_2.md`, whose design and predictions
#     were committed before any phase-2 answer was scored
#   - Each model got 3 runs at temperature 0.2 and a second greedy run
#     (temperature 0), next to its 5 runs at 0.8 from phase 1
# - The pedagogical arc:
#   - The scores at the 3 temperatures
#   - How much of the variance is run-to-run luck at 0.2
#   - How often 2 greedy runs disagree
#   - Whether the close pair can be told apart at 0.2
#   - How stable a 1-run leaderboard is at 0.2 and at 0.8
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
# # Part 1: The scores at 3 temperatures

# %% [markdown]
# ## Cell 1.1: Load the scores of both phases
#
# - Phase 1 has 1 greedy run and 5 runs at 0.8; phase 2 adds 3 runs at 0.2 and
#   a second greedy run
# - The greedy files got their second run in phase 2 and were scored again, so
#   their phase-2 scores replace the phase-1 ones

# %%
# Read the local scores of both phases and the settings of the analysis.
PHASE_1_DIR = "data/coding_runs"
PHASE_2_DIR = "data/coding_runs_phase2"
BENCHMARK = "humaneval"
LOW, HIGH = 0.2, 0.8
MODELS = list(rutils.PHASE_1_MODELS)
CLOSE_PAIR = ("codeqwen:7b-chat-v1.5-q4_K_M", "opencoder:8b")
old_scores = rutils.load_scores(PHASE_1_DIR, pattern="local_*.csv.gz")
new_scores = rutils.load_scores(PHASE_2_DIR)
scores = rutils.combine_scores(old_scores, new_scores)
study = scores[(scores["benchmark"] == BENCHMARK) & scores["model"].isin(MODELS)]

print("study.shape=", study.shape)
display(rutils.compute_score_table(study).round(3))
# Outcome: 9,840 rows, the 3 models x 2 scaffolds x 164 tasks with 2 greedy
# runs, 3 runs at 0.2, and 5 runs at 0.8. The one-shot plus scores, greedy /
# 0.2 / 0.8, are 78.7 / 80.1 / 74.1 for CodeQwen 7B, 79.3 / 78.0 / 72.7 for
# OpenCoder 8B, and 59.1 / 56.9 / 55.5 for DeepSeek-Coder 1.3B, so only
# CodeQwen does better at 0.2 than greedy.

# %% [markdown]
# ## Cell 1.2: Checks of the scoring
#
# - Every phase-2 answer is scored twice, so a flaky test shows up as an answer
#   that passes once and fails once
# - Not in the pre-registration: the run-0 greedy answers were scored in both
#   phases, so they should get the same results

# %%
# Check the scoring of phase 2.
flaky_share = rutils.compute_flaky_share(new_scores)
agreement = rutils.compute_rescored_agreement(old_scores, new_scores)

print("flaky_share=", flaky_share)
print("agreement=", agreement)
# Outcome: 2 of the 17,928 phase-2 rows are flaky (0.011%), both in the greedy
# MBPP+ scores of DeepSeek-Coder 6.7B, and the 984 greedy answers of run 0,
# scored in both phases, got the same result both times.

# %% [markdown]
# # Part 2: How much of the variance is run luck at 0.2?

# %% [markdown]
# ## Cell 2.1: Task and run parts of the variance at 0.2 and at 0.8
#
# - The run share at 0.2 comes from its 3 runs, and the one at 0.8 from the 5
#   runs of phase 1
#   - Both estimates are unbiased, whatever the number of runs

# %%
# Split the variance of every agent at the 2 temperatures.
variance = {
    temperature: rutils.compute_variance_table(
        study, source="local", benchmark=BENCHMARK, temperature=temperature
    )
    for temperature in (LOW, HIGH)
}

for temperature, table in variance.items():
    print("temperature=", temperature)
    display(table.round(4))
# Outcome: at 0.2 runs are 15% to 21% of the variance of 1 run, against 37% to
# 52% at 0.8, and the run variance is 1.9 to 3.1 times smaller. Part of the
# variance moves from the runs to the tasks, so the standard error of a 1-run
# score is 3.1 to 3.9 points at 0.2, against 3.4 to 3.9 at 0.8.

# %% [markdown]
# ## Cell 2.2: Plot of the 2 parts at 0.2

# %%
# Plot the task part and the run part of every agent at 0.2.
rutils.plot_variance_parts(
    variance[LOW],
    title=f"Task and run parts of the variance ({BENCHMARK}, T={LOW:g})",
    figsize=(10, 5),
)
# Outcome: the run part is a thin slice of every bar at 0.2, and DeepSeek-Coder
# 1.3B has the largest total because its score is the closest to 50%.

# %% [markdown]
# # Part 3: Do 2 greedy runs agree?

# %% [markdown]
# ## Cell 3.1: Tasks that change between the 2 greedy runs
#
# - Greedy decoding has no randomness, so a task can only change through
#   Ollama itself, e.g., its prompt cache

# %%
# Count the tasks whose one-shot result changes between the greedy runs.
greedy_changes = rutils.compute_greedy_changes(
    study, models=MODELS, benchmark=BENCHMARK
)

display(greedy_changes.round(4))
# Outcome: none of the 164 one-shot results changes between the 2 greedy runs,
# for any of the 3 models, so a greedy rerun gives the same score.

# %% [markdown]
# # Part 4: Can the close pair be told apart at 0.2?

# %% [markdown]
# ## Cell 4.1: Paired comparisons of the close pair
#
# - CodeQwen 7B against OpenCoder 8B, 1 test per scaffold, over the tasks and
#   the 3 runs at 0.2
# - The Holm adjustment covers the 2 tests

# %%
# Compare the close pair at 0.2, for each scaffold.
pairs = [
    ((CLOSE_PAIR[0], scaffold), (CLOSE_PAIR[1], scaffold))
    for scaffold in ("one_shot", "repair")
]
comparisons = rutils.compare_agent_pairs(
    study, pairs, source="local", benchmarks=[BENCHMARK], temperature=LOW
)

display(comparisons.round(4))
# Outcome: at 0.2 CodeQwen 7B is 2.0 points above OpenCoder 8B with one-shot
# and 2.4 points with repair, with 95% intervals of -3.9 to +7.5 and -3.5 to
# +7.9 points and a Holm p of 0.79 for both, so the pair still cannot be told
# apart.

# %% [markdown]
# # Part 5: How stable is a 1-run leaderboard at 0.2?

# %% [markdown]
# ## Cell 5.1: Order of the 3 models in random 1-run leaderboards
#
# - For each scaffold, a 1-run leaderboard picks 1 run per model
#   - Every combination of runs is taken once (3^3 = 27), so the shares are
#     exact instead of coming from random draws, which could split 2 equal
#     shares by chance
# - At 0.8 only the first 3 runs are used, as at 0.2: with fewer runs, the drawn
#   run is a bigger part of the mean it is checked against

# %%
# Draw random 1-run leaderboards of the 3 models from their first 3 runs.
stability = rutils.compute_stability_table(
    study,
    models=MODELS,
    benchmark=BENCHMARK,
    temperatures=[LOW, HIGH],
    n_runs=3,
)

display(stability.round(3))
# Outcome: with 3 runs per model there are only 27 leaderboards per scaffold.
# With one-shot, 24 of the 27 keep the order at both temperatures; with repair,
# 24 at 0.2 and 18 at 0.8. The mean Kendall tau is higher at 0.2 with both
# scaffolds (0.91 against 0.87 and 0.78).

# %% [markdown]
# # Part 6: Which predictions held?

# %% [markdown]
# ## Cell 6.1: Predictions against the results
#
# - The rules follow `preregistration_2.md`, which was committed before any
#   phase-2 answer was scored

# %%
# Check each prediction of Study C.
predictions = rutils.check_temperature_predictions(
    variance_low=variance[LOW],
    variance_high=variance[HIGH],
    greedy_changes=greedy_changes,
    comparisons=comparisons,
    stability=stability,
    low=LOW,
    high=HIGH,
)

display(predictions)
# Outcome: C1 to C4 hold and C5 does not: with one-shot, 1-run leaderboards
# keep the order exactly as often at 0.2 as at 0.8 (88.9%), and only with
# repair are they more stable at 0.2 (88.9% against 66.7%).

# %% [markdown]
# # Part 7: Takeaways
#
# - Going from temperature 0.8 to 0.2 cuts run luck from 37% to 52% of the
#   variance of 1 run down to 15% to 21%, and 2 greedy runs give the same
#   result on every task
# - A 1-run score is only a little more precise at 0.2: part of the variance
#   moves from the runs to the tasks, so its standard error is 3.1 to 3.9
#   points against 3.4 to 3.9 at 0.8
# - The close pair stays too close to call at 0.2, and its order depends on the
#   temperature: OpenCoder 8B is ahead in the greedy one-shot runs (79.3
#   against 78.7) and CodeQwen 7B at 0.2 and 0.8
# - Prediction C5 missed with the one-shot scaffold, by a tie: 24 of the 27
#   1-run leaderboards keep the order at both temperatures
