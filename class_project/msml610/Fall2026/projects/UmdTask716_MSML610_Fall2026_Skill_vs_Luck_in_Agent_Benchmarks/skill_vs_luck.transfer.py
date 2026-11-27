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
# # Do HumanEval+ and MBPP+ rank models the same way?
#
# - This notebook checks whether coding skill transfers between 2 benchmarks,
#   and how many of their tasks carry information (Milestone 3)
#   - It is Study E of `preregistration_3.md`, whose design and predictions
#     were committed before any of it was run
#   - The agents are 18 models with a greedy answer to every task of both
#     benchmarks: 12 public EvalPlus models of 2023 and our 6 local models
# - The pedagogical arc:
#   - The scores of the 18 models on both benchmarks
#   - How reliable each benchmark is, i.e., how much of the spread of the
#     scores is the models and how much is the luck of the tasks
#   - How well the 2 rankings agree, before and after removing task luck
#   - Which models do better on one benchmark than a typical model does
#   - How many equally informative tasks each benchmark is worth
#   - The same agreement on the 89 models of the EvalPlus leaderboard
#   - Which predictions held

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
import pandas as pd
import scipy.stats

import helpers.hnotebook as hnotebook

import skill_vs_luck_runs_utils as rutils
import skill_vs_luck_utils as utils

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
# # Part 1: The 18 models on both benchmarks

# %% [markdown]
# ## Cell 1.1: Load the greedy answers
#
# - Public models: the EvalPlus greedy samples, scored by
#   `score_coding_runs.py`
# - Local models: the first greedy run, one-shot, of phase 1 (DeepSeek Coder
#   1.3B, CodeQwen 7B, OpenCoder 8B) and of Study D (DeepSeek Coder 6.7B,
#   Llama 3.1 8B, CodeGemma 7B)

# %%
# Settings of the analysis.
config = {
    "phase_1_dir": "data/coding_runs",
    "phase_2_dir": "data/coding_runs_phase2",
    "leaderboard_file": "data/evalplus_leaderboard.csv",
    "benchmarks": ["humaneval", "mbpp"],
    "n_splits": 1000,
    "n_boot": 2000,
    # The 2PL fit by SVI.
    "n_svi_steps": 5000,
    "n_svi_samples": 1000,
    "seed": 0,
    "figsize": (10, 5),
}
# Greedy scores of the public models and of the local models of phase 1, and
# of the 3 models of Study D.
phase_1_scores = rutils.load_scores(config["phase_1_dir"], pattern="*_t0.csv.gz")
phase_2_scores = rutils.load_scores(
    config["phase_2_dir"], pattern="local_*_t0.csv.gz"
)
phase_2_scores = phase_2_scores[
    phase_2_scores["model"].isin(rutils.PHASE_2_MODELS)
]
scores = pd.concat([phase_1_scores, phase_2_scores], ignore_index=True)
agents = [("evalplus", model) for model in rutils.TRANSFER_PUBLIC_MODELS] + [
    ("local", model) for model in rutils.PHASE_1_MODELS + rutils.PHASE_2_MODELS
]
outcomes = {}
for benchmark in config["benchmarks"]:
    matrix = rutils.get_greedy_matrix(scores, benchmark, agents)
    matrix.index = [rutils.get_model_label(model) for model in matrix.index]
    outcomes[benchmark] = matrix
    print(f"{benchmark}: {matrix.shape[0]} models x {matrix.shape[1]} tasks")

# %% [markdown]
# ## Cell 1.2: Scores of the 18 models

# %%
# Show the share of tasks each model solves, in %.
model_scores = pd.DataFrame(
    {
        "humaneval_plus": 100 * outcomes["humaneval"].mean(axis=1),
        "mbpp_plus": 100 * outcomes["mbpp"].mean(axis=1),
    }
).sort_values("humaneval_plus", ascending=False)
display(model_scores.round(1))

# %% [markdown]
# # Part 2: How reliable is each benchmark?

# %% [markdown]
# ## Cell 2.1: KR-20 and split-half reliability
#
# - KR-20 is the share of the spread of the scores that comes from the models,
#   not from the luck of which tasks were drawn
# - The split-half reliability correlates the scores on 2 random halves of the
#   tasks and steps it up to the whole benchmark; it should be close to KR-20

# %%
# Compute both reliabilities of each benchmark.
reliability = pd.DataFrame(
    {
        benchmark: {
            "n_tasks": matrix.shape[1],
            "kr20": utils.compute_kr20(matrix),
            "split_half": utils.compute_split_half_reliability(
                matrix, n_splits=config["n_splits"], seed=config["seed"]
            ),
        }
        for benchmark, matrix in outcomes.items()
    }
).T
display(reliability.round(4))

# %% [markdown]
# # Part 3: Do the 2 benchmarks agree?

# %% [markdown]
# ## Cell 3.1: Correlation of the 2 rankings, before and after task luck
#
# - The corrected correlation divides the Pearson correlation by the square
#   root of the product of the 2 KR-20 values
#   - 1 means the 2 benchmarks measure the same skill, and any disagreement is
#     task luck
# - The 95% intervals resample the models and the tasks of each benchmark

# %%
# Compare the rankings of the 18 models.
transfer = utils.compute_transfer(
    outcomes["humaneval"],
    outcomes["mbpp"],
    n_boot=config["n_boot"],
    seed=config["seed"],
)
display(transfer.round(4).to_frame("HumanEval+ vs MBPP+"))

# %% [markdown]
# ## Cell 3.2: Models that do better on one benchmark
#
# - The residual of a model is how much more it gains on HumanEval+, on the
#   logit scale, than the median model does
#   - Positive: better on HumanEval+ than its MBPP+ score suggests; negative:
#     better on MBPP+
#   - `gap_points` is the same gap in HumanEval+ points
# - A model is benchmark-specific when the 95% interval of its residual,
#   which resamples the tasks, excludes 0

# %%
# Find the benchmark-specific models.
residuals = utils.compute_benchmark_residuals(
    outcomes["humaneval"],
    outcomes["mbpp"],
    n_boot=config["n_boot"],
    seed=config["seed"],
)
display(residuals.sort_values("residual", ascending=False).round(3))

# %%
# Plot the 2 scores of each model and the residuals.
utils.plot_benchmark_scores(
    residuals, name_a="HumanEval+", name_b="MBPP+", figsize=config["figsize"]
)
utils.plot_benchmark_residuals(
    residuals, name_a="HumanEval+", figsize=config["figsize"]
)

# %% [markdown]
# ## Cell 3.3: Chat and base models with the same MBPP+ score (extra look)
#
# - Not in `preregistration_3.md`: added after Cell 3.2 showed that the 7 chat
#   models (GPT-4 Turbo and the 6 local ones) all lie above the typical model
# - A steeper HumanEval+ scale would also put the best models above the curve,
#   so this cell compares models with the same MBPP+ score, where no
#   extrapolation is needed
# - For each chat model, the base model of 2023 with the closest MBPP+ score,
#   kept when the 2 MBPP+ scores are within 3 points

# %%
# Pair each chat model with the closest base model on MBPP+.
chat_models = ["gpt-4-1106-preview"] + [
    rutils.get_model_label(model)
    for model in rutils.PHASE_1_MODELS + rutils.PHASE_2_MODELS
]
base = model_scores.drop(index=chat_models)
pairs = []
for chat_model in chat_models:
    mbpp_gaps = (
        base["mbpp_plus"] - model_scores.loc[chat_model, "mbpp_plus"]
    ).abs()
    base_model = mbpp_gaps.idxmin()
    pairs.append(
        {
            "chat_model": chat_model,
            "base_model": base_model,
            "mbpp_gap": mbpp_gaps.min(),
            "humaneval_gain": model_scores.loc[chat_model, "humaneval_plus"]
            - model_scores.loc[base_model, "humaneval_plus"],
        }
    )
pairs = pd.DataFrame(pairs).set_index("chat_model")
display(pairs[pairs["mbpp_gap"] <= 3].round(1))

# %% [markdown]
# # Part 4: How many tasks carry information?

# %% [markdown]
# ## Cell 4.1: Effective size of each benchmark under a 2PL fit
#
# - The 2PL model of Milestone 3, fit with SVI as for SWE-bench Verified
# - The information of each task is averaged over the 18 fitted abilities,
#   and the effective size is the number of equally informative tasks the
#   benchmark is worth (decision D-3)
# - Tasks with mixed outcomes are those that some models solve and some fail

# %%
# Fit each benchmark and measure its effective size.
fits = {}
effective_rows = {}
for benchmark, matrix in outcomes.items():
    fit = utils.fit_irt_2pl(
        matrix,
        method="svi",
        n_steps=config["n_svi_steps"],
        n_samples=config["n_svi_samples"],
        seed=config["seed"],
    )
    information = utils.compute_task_information(
        fit["agents"]["ability"], fit["tasks"]
    )
    fits[benchmark] = fit
    effective_rows[benchmark] = {
        "n_tasks": matrix.shape[1],
        "n_mixed_tasks": utils.count_informative_tasks(matrix),
        "effective_size": utils.compute_effective_size(information),
    }
effective = pd.DataFrame(effective_rows).T
effective["effective_share"] = effective["effective_size"] / effective["n_tasks"]
display(effective.round(3))

# %% [markdown]
# ## Cell 4.2: Do the fitted abilities agree across the 2 benchmarks?

# %%
# Correlate the fitted abilities, which weigh the tasks by information.
abilities = pd.DataFrame(
    {benchmark: fit["agents"]["ability"] for benchmark, fit in fits.items()}
)
ability_spearman = scipy.stats.spearmanr(
    abilities["humaneval"], abilities["mbpp"]
).statistic
print(f"Spearman correlation of the fitted abilities: {ability_spearman:.3f}")

# %% [markdown]
# # Part 5: Does the agreement hold on a bigger leaderboard?

# %% [markdown]
# ## Cell 5.1: The 89 models of the EvalPlus leaderboard
#
# - The snapshot of `data/evalplus_leaderboard.csv`, made by
#   `fetch_evalplus_data.py` on 2026-11-26
# - Only the scores are public, so this check has no task luck correction

# %%
# Correlate the published HumanEval+ and MBPP+ scores.
leaderboard = pd.read_csv(config["leaderboard_file"])
leaderboard = leaderboard.dropna(subset=["humaneval_plus", "mbpp_plus"])
leaderboard_spearman = scipy.stats.spearmanr(
    leaderboard["humaneval_plus"], leaderboard["mbpp_plus"]
).statistic
print(
    f"{len(leaderboard)} models, Spearman correlation {leaderboard_spearman:.3f}"
)
rutils.plot_leaderboard_agreement(leaderboard, figsize=config["figsize"])

# %% [markdown]
# # Part 6: Which predictions held?

# %% [markdown]
# ## Cell 6.1: Predictions against the results

# %%
# Check the predictions E1 to E6 of `preregistration_3.md`.
predictions = rutils.check_transfer_predictions(
    transfer=transfer,
    residuals=residuals,
    effective=effective,
    leaderboard_spearman=leaderboard_spearman,
)
display(predictions)

# %% [markdown]
# # Part 7: Takeaways
#
# - Both benchmarks rank the 18 models reliably (KR-20 0.986 on HumanEval+ and
#   0.980 on MBPP+), and they rank them almost the same way
#   - The Spearman correlation is 0.98, and without task luck the correlation
#     is 0.99, with a 95% interval that reaches 1.00 (E3 did not hold): the
#     order of the models transfers
# - The scale does not transfer: 6 of the 7 chat models score 14 to 21
#   HumanEval+ points more than a typical model with the same MBPP+ score, and
#   2 base models (StarCoder and CodeT5+ 6B) about 4 to 5 points less, so 8 of
#   the 18 models are benchmark-specific (E4 held)
#   - CodeGemma 7B, the 7th chat model, is 8 points above, within task luck
#   - At the same MBPP+ score, DeepSeek Coder 1.3B beats Code Llama 13B by 22.0
#     HumanEval+ points, and CodeGemma 7B beats Code Llama 34B by 7.3 (Cell
#     3.3, an extra look)
#   - Either the chat models were tuned toward HumanEval-style tasks, or their
#     chat prompt suits HumanEval+ better than the completion prompt of the
#     public samples; this data cannot tell the 2 apart
#   - So a gap between 2 models of different families on 1 benchmark says
#     little about their gap on the other, even though the order holds
# - HumanEval+ is worth about 119 equally informative tasks for these models
#   (72% of its 164) and MBPP+ about 214 (57% of its 378); 29 and 79 tasks have
#   the same outcome for all 18 models (E5 did not hold, since HumanEval+ is
#   above 70%)
# - On the 89 models of the EvalPlus leaderboard the 2 benchmarks also agree
#   (Spearman 0.94, E6 held)
# - 4 of the 6 predictions held (E1, E2, E4, and E6), and E3 and E5 did not
