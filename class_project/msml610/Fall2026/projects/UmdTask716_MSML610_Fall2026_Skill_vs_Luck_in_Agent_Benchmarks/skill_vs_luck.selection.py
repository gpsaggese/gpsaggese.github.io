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
# # How much does picking the best of K inflate a score, and how to fix it?
#
# - This notebook measures the winner's curse on real runs and real prompt
#   variants, and picks a correction to report with a score (Milestone 4)
#   - It is Studies F, G, and H of `preregistration_3.md`, whose design and
#     predictions were committed before any of it was run
#   - Milestone 4 so far only simulated the curse; here the variants are real
# - The pedagogical arc:
#   - Reporting the best of K runs, where every difference is luck
#   - Reporting the best of K prompt variants, scored on tasks the selection
#     never saw
#   - Which of 5 corrections recovers the winner's true score, and a fix for
#     the one that misses the most (an extra look)
#   - The reporting protocol of the project, on the SWE-bench Verified
#     leaderboard
#   - Which predictions held, and the takeaways

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
import numpy as np
import pandas as pd

import helpers.hdbg as hdbg
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

# %%
# Settings of the analysis.
config = {
    "phase_1_dir": "data/coding_runs",
    "phase_2_dir": "data/coding_runs_phase2",
    "variants_dir": "data/coding_runs_variants",
    "swebench_dir": "data",
    "k_public": [2, 5, 10, 20],
    "k_variants": [2, 5, 10, 20],
    "n_draws": 500,
    "n_trials": 1000,
    "seed": 0,
    "figsize": (10, 5),
}
CORRECTIONS = list(utils.WINNER_CORRECTIONS)

# %% [markdown]
# # Part 1: Reporting the best of K runs (Study F)

# %% [markdown]
# ## Cell 1.1: The cells with several runs on the same tasks
#
# - Public cells: the EvalPlus samples on HumanEval+, 50 scored runs per task
# - Local cells: our runs, 5 per task in phase 1 and 3 in Studies C and D, each
#   scaffold

# %%
# Collect the tasks x runs matrix of each cell.
cells = {}
public_scores = rutils.load_scores(
    config["phase_1_dir"], pattern="evalplus_humaneval_*_t0.[28].csv.gz"
)
for (model, temperature), _ in public_scores.groupby(["model", "temperature"]):
    cells[("public", model, temperature, "humaneval", "one_shot")] = (
        rutils.get_outcome_matrix(
            public_scores,
            source="evalplus",
            benchmark="humaneval",
            model=model,
            scaffold="one_shot",
            temperature=temperature,
        )
    )
for scores_dir in (config["phase_1_dir"], config["phase_2_dir"]):
    local_scores = rutils.load_scores(
        scores_dir, pattern="local_*_t0.[28].csv.gz"
    )
    groups = local_scores.groupby(
        ["model", "temperature", "benchmark", "scaffold"]
    )
    for (model, temperature, benchmark, scaffold), _ in groups:
        key = ("local", model, temperature, benchmark, scaffold)
        # Phases 1 and 2 must not have the same cell.
        hdbg.dassert_not_in(key, cells)
        cells[key] = rutils.get_outcome_matrix(
            local_scores,
            source="local",
            benchmark=benchmark,
            model=model,
            scaffold=scaffold,
            temperature=temperature,
        )
cell_table = pd.DataFrame(
    [(*key, matrix.shape[0], matrix.shape[1]) for key, matrix in cells.items()],
    columns=[
        "group",
        "model",
        "temperature",
        "benchmark",
        "scaffold",
        "n_tasks",
        "n_runs",
    ],
)
display(cell_table.groupby(["group", "n_runs"]).size().rename("n_cells"))

# %% [markdown]
# ## Cell 1.2: The best of K runs against the runs left out
#
# - For each cell and `K`, 500 draws of `K` runs: the inflation is the best of
#   the `K` runs minus the mean of the runs left out
# - The predicted inflation is `sigma_run x E[max of K]`, from the run noise of
#   the cell
# - Each correction is applied to the best run from the `K` runs only, and its
#   error is measured against the runs left out

# %%
# Draw the best of K runs in every cell, and keep the draws for Cell 3.2.
rows = []
runs_draws = {}
for key, matrix in cells.items():
    group = key[0]
    n_runs = matrix.shape[1]
    # The public cells use every K; the local cells K = 2, and also K = 3
    # for the 5-run cells of phase 1, so that 2 runs are left out.
    if group == "public":
        k_values = config["k_public"]
    else:
        k_values = [2, 3] if n_runs == 5 else [2]
    for k in k_values:
        draws = utils.simulate_best_of_k_runs(
            matrix, k, n_draws=config["n_draws"], seed=config["seed"]
        )
        runs_draws[(key, k)] = draws
        summary = utils.summarize_corrections(draws, "truth")
        row = dict(zip(cell_table.columns[:5], key))
        row.update(
            {
                "k": k,
                "n_runs": n_runs,
                "inflation": 100 * draws["inflation"].mean(),
                "predicted": 100 * draws["predicted"].mean(),
            }
        )
        for correction in CORRECTIONS:
            row[f"error_{correction}"] = summary.loc[correction, "mean_error"]
            row[f"rmse_{correction}"] = summary.loc[correction, "rmse"]
        rows.append(row)
runs_cells = pd.DataFrame(rows)
display(
    runs_cells[runs_cells["k"] == 5][
        ["model", "temperature", "inflation", "predicted"]
    ].round(2)
)

# %% [markdown]
# ## Cell 1.3: The public cells and the local cells, pooled

# %%
# Pool the cells of each group, with the RMSE pooled over all the draws.
error_columns = [f"error_{c}" for c in CORRECTIONS]
rmse_columns = [f"rmse_{c}" for c in CORRECTIONS]
grouped = runs_cells.groupby(["group", "k"])
runs_pooled = grouped[["inflation", "predicted"] + error_columns].mean()
runs_pooled[rmse_columns] = np.sqrt(
    (runs_cells[rmse_columns] ** 2)
    .groupby([runs_cells["group"], runs_cells["k"]])
    .mean()
)
runs_pooled["ratio"] = runs_pooled["inflation"] / runs_pooled["predicted"]
runs_pooled["n_cells"] = grouped.size()
display(runs_pooled.round(2))

# %% [markdown]
# ## Cell 1.4: The predicted inflation with few runs (extra look)
#
# - Not in `preregistration_3.md`: added after Cell 1.3 showed the local cells
#   above their prediction
# - The prediction estimates the run noise from the pass rate `p_t` of each
#   task over the `R` runs of the cell, and `p_t (1 - p_t)` from `R` runs is too
#   low by a factor `(R - 1) / R`: 2 / 3 with 3 runs, while the public cells have
#   50 runs
# - Multiplying the predicted inflation by `sqrt(R / (R - 1))` removes that bias

# %%
# Correct the prediction for the number of runs of each cell.
runs_cells["predicted_unbiased"] = runs_cells["predicted"] * np.sqrt(
    runs_cells["n_runs"] / (runs_cells["n_runs"] - 1)
)
unbiased = runs_cells.groupby(["group", "k"])[
    ["inflation", "predicted", "predicted_unbiased"]
].mean()
unbiased["ratio_unbiased"] = (
    unbiased["inflation"] / unbiased["predicted_unbiased"]
)
display(unbiased.round(2))

# %%
# Plot the measured and the predicted inflation.
rutils.plot_best_of_k(runs_pooled, figsize=config["figsize"])

# %% [markdown]
# # Part 2: Reporting the best of K prompt variants (Study G)

# %% [markdown]
# ## Cell 2.1: The 20 prompt variants of DeepSeek Coder 1.3B
#
# - Greedy one-shot answers to every task, with each variant of
#   `PROMPT_VARIANTS`: 2 system prompts x 2 instructions x 5 lines of guidance
# - Made by `run_prompt_variants.sh` on 2026-11-26 and 27, with Ollama 0.40.16
#   and the same model file as phase 1, which used Ollama 0.40.8
# - 33 MBPP+ answers were first made by 2 extra streams running at the same
#   time; Ollama can batch parallel requests, which can change greedy answers,
#   so those 33 were deleted and made again one at a time
# - 1 MBPP+ answer is empty: Ollama stopped it for repeating a token, and it
#   counts as a failure
# - The table gives the score of each variant in percent, and the number of
#   answers cut at the limit of 768 tokens (`truncated`)

# %%
# Load the scores of the variants.
variant_scores = rutils.load_scores(config["variants_dir"])
variant_matrices = {
    benchmark: rutils.get_variant_matrix(variant_scores, benchmark)
    for benchmark in rutils.BENCHMARKS
}
variant_table = pd.DataFrame(
    {
        benchmark: 100 * matrix.mean(axis=1)
        for benchmark, matrix in variant_matrices.items()
    }
)
# Count the answers cut at the token limit, which the guidance can change.
one_shot = variant_scores[variant_scores["scaffold"] == "one_shot"]
truncated = one_shot.pivot_table(
    index="variant", columns="benchmark", values="truncated", aggfunc="sum"
)
variant_table = variant_table.join(truncated.add_suffix("_truncated"))
display(variant_table.sort_values("humaneval", ascending=False).round(1))

# %% [markdown]
# ## Cell 2.2: The baseline variant repeats phase 1
#
# - The baseline variant sends the prompt of phase 1, so its outcomes should
#   be those of the greedy one-shot run of phase 1
# - The 2 runs used different versions of Ollama (0.40.8 and 0.40.16), which
#   can change a few greedy answers

# %%
# Compare the baseline variant with the greedy one-shot run of phase 1.
phase_1_greedy = rutils.load_scores(
    config["phase_1_dir"],
    pattern="local_*_deepseek-coder_1.3b-instruct-q4_K_M_t0.csv.gz",
)
agreement = {}
for benchmark, matrix in variant_matrices.items():
    phase_1 = rutils.get_outcome_matrix(
        phase_1_greedy,
        source="local",
        benchmark=benchmark,
        model=rutils.PHASE_1_MODELS[0],
        scaffold="one_shot",
        temperature=0,
    )[0]
    baseline = matrix.loc[rutils.BASELINE_VARIANT]
    agreement[benchmark] = float((baseline == phase_1[baseline.index]).mean())
agreement = pd.Series(agreement)
display(agreement.round(4).to_frame("same_outcome"))

# %% [markdown]
# ## Cell 2.3: Are the variants different?
#
# - Cochran's Q tests whether the 20 variants solve the same share of the tasks,
#   using only the tasks where they disagree

# %%
# Test the variants of each benchmark.
cochran = pd.DataFrame(
    {
        benchmark: dict(
            zip(("q", "p_value"), utils.compute_cochran_q(matrix)),
            low=100 * matrix.mean(axis=1).min(),
            high=100 * matrix.mean(axis=1).max(),
        )
        for benchmark, matrix in variant_matrices.items()
    }
).T
display(cochran.round(4))

# %% [markdown]
# ## Cell 2.4: The best of K variants on held-out tasks
#
# - Each trial draws `K` variants and splits the tasks in 2 halves at random;
#   the winner is picked on the first half and scored on the second
# - Each correction uses only the first half; its error is the corrected score
#   minus the score on the second half

# %%
# Run the trials for each benchmark and K, and keep them for Cell 3.2.
rows = []
variant_trials = {}
for benchmark, matrix in variant_matrices.items():
    for k in config["k_variants"]:
        trials = utils.simulate_variant_selection(
            matrix, k, n_trials=config["n_trials"], seed=config["seed"]
        )
        variant_trials[(benchmark, k)] = trials
        summary = utils.summarize_corrections(trials, "evaluation_score")
        row = {
            "benchmark": benchmark,
            "k": k,
            "drop": 100 * trials["drop"].mean(),
        }
        for correction in CORRECTIONS:
            row[f"error_{correction}"] = summary.loc[correction, "mean_error"]
            row[f"rmse_{correction}"] = summary.loc[correction, "rmse"]
        rows.append(row)
variants = pd.DataFrame(rows).set_index(["benchmark", "k"])
display(variants.round(2))

# %%
# Plot the mean error of each correction.
for benchmark in rutils.BENCHMARKS:
    rutils.plot_correction_errors(variants, benchmark, figsize=config["figsize"])

# %% [markdown]
# # Part 3: Which correction to recommend? (Study H)

# %% [markdown]
# ## Cell 3.1: The rule fixed in `preregistration_3.md`
#
# - Keep the corrections (paired, empirical Bayes, conditional) whose mean error
#   is within 1 point of 0 in every setting of Studies F and G
# - Recommend the one with the lowest RMSE averaged over the settings

# %%
# Gather the settings and apply the rule.
summaries = {}
for (group, k), row in runs_pooled.iterrows():
    summaries[f"runs {group} K={k}"] = pd.DataFrame(
        {
            "mean_error": [row[f"error_{c}"] for c in CORRECTIONS],
            "rmse": [row[f"rmse_{c}"] for c in CORRECTIONS],
        },
        index=CORRECTIONS,
    )
for (benchmark, k), row in variants.iterrows():
    summaries[f"variants {benchmark} K={k}"] = pd.DataFrame(
        {
            "mean_error": [row[f"error_{c}"] for c in CORRECTIONS],
            "rmse": [row[f"rmse_{c}"] for c in CORRECTIONS],
        },
        index=CORRECTIONS,
    )
rule_table, recommended = utils.choose_shrinkage_rule(summaries)
display(rule_table.round(3))
print(f"Recommended correction: {recommended}")

# %% [markdown]
# ## Cell 3.2: Why the conditional estimate misses, and the hybrid fix (extra look)
#
# - Not in `preregistration_3.md`: added after Study F showed the conditional
#   estimate with a mean error of several points below 0
# - The conditional estimate is right on the median, but when the winner only
#   just won, it falls far below the score, at times below 0
# - The hybrid estimate of the same paper also assumes that the winner's mean
#   is within an interval that covers all the means at once with a 99.5%
#   chance, so it stays within about 3 standard deviations of the score
# - The draws and trials are those of Parts 1 and 2; the rule of Cell 3.1 is
#   then applied again with the hybrid estimate as a fourth candidate

# %%
# Collect the errors of the conditional and hybrid estimates, in points.
frames = []
for (key, k), draws in runs_draws.items():
    frames.append(
        pd.DataFrame(
            {
                "setting": f"runs {key[0]} K={k}",
                "conditional": 100 * (draws["conditional"] - draws["truth"]),
                "hybrid": 100 * (draws["hybrid"] - draws["truth"]),
                "below_zero": draws["conditional"] < 0,
            }
        )
    )
for (benchmark, k), trials in variant_trials.items():
    truth = trials["evaluation_score"]
    frames.append(
        pd.DataFrame(
            {
                "setting": f"variants {benchmark} K={k}",
                "conditional": 100 * (trials["conditional"] - truth),
                "hybrid": 100 * (trials["hybrid"] - truth),
                "below_zero": trials["conditional"] < 0,
            }
        )
    )
errors = pd.concat(frames, ignore_index=True)
by_setting = errors.groupby("setting", sort=False)
tails = pd.DataFrame(
    {
        "conditional_median": by_setting["conditional"].median(),
        "conditional_5%": by_setting["conditional"].quantile(0.05),
        "conditional_below_0": 100 * by_setting["below_zero"].mean(),
        "hybrid_mean": by_setting["hybrid"].mean(),
        "hybrid_5%": by_setting["hybrid"].quantile(0.05),
        "hybrid_rmse": np.sqrt(
            by_setting["hybrid"].apply(lambda e: (e**2).mean())
        ),
    }
)
display(tails.round(2))

# %%
# Apply the rule of Cell 3.1 with the hybrid estimate as a candidate.
hybrid_summaries = {}
for setting, summary in summaries.items():
    # Every cell has the same number of draws, so pooling the draws gives the
    # mean errors of Cell 3.1.
    conditional = errors.loc[errors["setting"] == setting, "conditional"]
    hdbg.dassert_lt(
        abs(conditional.mean() - summary.loc["conditional", "mean_error"]),
        1e-9,
    )
    hybrid = errors.loc[errors["setting"] == setting, "hybrid"]
    hybrid_row = pd.DataFrame(
        {"mean_error": [hybrid.mean()], "rmse": [np.sqrt((hybrid**2).mean())]},
        index=["hybrid"],
    )
    hybrid_summaries[setting] = pd.concat([summary, hybrid_row])
hybrid_table, hybrid_recommended = utils.choose_shrinkage_rule(
    hybrid_summaries,
    candidates=("paired", "empirical_bayes", "conditional", "hybrid"),
)
display(hybrid_table.round(3))
print(
    f"Recommended with the hybrid estimate as a candidate: {hybrid_recommended}"
)

# %% [markdown]
# # Part 4: The protocol on the SWE-bench Verified leaderboard

# %% [markdown]
# ## Cell 4.1: The report of the top agents
#
# - `report_agent_scores()` gives, for each agent, its score with a 95%
#   interval, its rank interval, the paired test against the next agent
#   (Holm-adjusted), and the tasks that would resolve that gap
# - The leader is also reported with its score corrected for having been
#   picked first among the 173 submissions, with the correction recommended
#   in Cell 3.1 (the conditional estimate if none passed)
#   - 2 agents tie for first with 396 of the 500 tasks; the one listed first
#     gets the correction
# - Cell 4.2 shows the leader under every correction, since the 173 agents
#   are not equal variants, which some of the corrections assume

# %%
# Report the whole leaderboard and show the top 15.
outcomes, submissions, tasks = utils.load_swebench_dataset(
    config["swebench_dir"]
)
leader_correction = recommended or "conditional"
report = utils.report_agent_scores(
    outcomes, correction=leader_correction, seed=config["seed"]
)
# Show the scores in points.
shown = report.head(15).copy()
points = ["score", "ci_low", "ci_high", "gap_next", "corrected"]
shown[points] = 100 * shown[points]
display(shown.round(2))
leader = report.index[0]
print(
    f"Leader: {100 * report.loc[leader, 'score']:.1f}%, corrected for being "
    f"picked first: {100 * report.loc[leader, 'corrected']:.1f}%"
)

# %% [markdown]
# ## Cell 4.2: The leader under every correction (extra look)
#
# - Not in `preregistration_3.md`
# - The binomial and paired rules assume that all the agents are equal, which
#   the 173 submissions are not, so they correct too much here; empirical Bayes
#   shrinks toward the mean of all the agents; the conditional estimate only
#   looks at the agents that could have beaten the leader
#   - A check from Milestone 3: in `skill_vs_luck.example.ipynb` (Cell 6.2),
#     the leader of one half of the tasks scores 2.0 points less on the other
#     half, with 250 tasks per half; with all 500 tasks the noise is smaller,
#     so the leader's real inflation is likely 2 points or less
# - The hybrid estimate of Cell 3.2 is also shown

# %%
# Correct the leader's score with each correction.
ordered = outcomes.loc[report.index].to_numpy(dtype=float)
leader_estimates = utils.correct_winner_score(ordered, 0, seed=config["seed"])
display((100 * pd.Series(leader_estimates)).round(2).to_frame("leader_score"))

# %% [markdown]
# # Part 5: Which predictions held?

# %% [markdown]
# ## Cell 5.1: Predictions F1 to F3 and G0 to G5 against the results

# %%
# Check the predictions of Studies F and G.
predictions = rutils.check_selection_predictions(
    runs_pooled=runs_pooled,
    runs_cells=runs_cells,
    variants=variants,
    agreement=agreement,
    cochran=cochran,
)
display(predictions)

# %% [markdown]
# # Part 6: Takeaways
#
# - On the public samples, the best of K runs is above the runs left out by
#   just what `sigma_run x E[max of K]` predicts: 1.2 points for K = 2 up to
#   4.0 for K = 20, with ratios of 1.00 to 1.04
#   - On our runs, with only 3 or 5 runs per cell, the inflation is 1.41 times
#     the prediction at K = 2 (F1 did not hold), since `p_t (1 - p_t)` from few
#     runs is too low; correcting for that brings the ratios to 1.20 and 1.06
#     (Cell 1.4, an extra look)
#   - The best of 5 runs at temperature 0.8 is 2.3 to 3.2 points above the runs
#     left out, for each of the 9 public cells (F2 held)
# - The 20 prompt variants differ for real: 50.0% to 60.4% on HumanEval+ and
#   41.8% to 54.0% on MBPP+, with Cochran's Q p of 0.002 and about 1e-33 (G1
#   held)
#   - The line asking to handle edge cases costs 7 to 9 points on HumanEval+
#     and 10 to 12 on MBPP+, and more of its answers hit the token limit
#   - The baseline variant repeats the outcome of phase 1 on every HumanEval+
#     task and on 377 of the 378 MBPP+ tasks (G0 held)
# - Picked on half of the tasks, the best of K variants scores less on the
#   other half: 1.9 points for K = 2 up to 5.1 for K = 20 on HumanEval+, and
#   0.6 up to 2.5 on MBPP+ (G2 held)
# - How the corrections did:
#   - Binomial: corrects too much, more as K grows: -4.9 and -4.3 points at
#     K = 20 (G3 held), and -1.7 for the best of 5 runs (F3 held)
#   - Empirical Bayes: corrects too little as K grows, +2.0 on HumanEval+ and
#     +1.6 on MBPP+ at K = 20, since the spread of the K scores also holds the
#     real gaps between the variants
#   - Conditional: right on the median, but its mean error is -0.9 to -7.7
#     points (G4 did not hold) and its root mean squared error up to 31
#     points: when the winner only just won, the estimate falls far below the
#     score, under 0 in up to 5.3% of the draws
#   - Paired: within 1 point in every setting of Studies F and G (worst
#     -0.97), and the lowest root mean squared error at K = 20 on both
#     benchmarks (G5 did not hold, since it named empirical Bayes or
#     conditional)
# - Rule H recommends the paired correction: it is the only candidate within
#   1 point everywhere, with a mean root mean squared error of 4.3 points
#   - The hybrid estimate cuts the root mean squared error of the conditional
#     one by 1.5 to 3.8 times, but it still misses by up to 1.5 points, with a
#     mean root mean squared error of 6.5, so the paired correction stays the
#     recommendation (Cell 3.2, an extra look)
# - On SWE-bench Verified, the paired correction takes the leader from 79.2%
#   to 75.1%, and the 5 corrections range from 72.7% (conditional) to 79.0%
#   (empirical Bayes)
#   - The 173 agents are far from equal, so the paired rule likely corrects too
#     much here: the leader of one half of the tasks loses only 2.0 points on
#     the other half (Cell 4.2)
# - 6 of the 9 predictions held (F2, F3, and G0 to G3), and F1, G4, and G5 did
#   not; with Study E, 10 of the 15 predictions of `preregistration_3.md` held
