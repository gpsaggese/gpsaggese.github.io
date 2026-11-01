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
# # How much of the SWE-bench Verified leaderboard is luck?
#
# - This notebook uses the public per-task results of every SWE-bench Verified
#   submission to check which gaps on the leaderboard are real
#   - The data is pinned to commit `40f164d` of `SWE-bench/experiments`
#   - The numbers in the text are for that snapshot
# - The pedagogical arc:
#   - The benchmark, the submissions, and a check of the data
#   - How uncertain each score is
#   - Whether the leaderboard can tell neighbors apart
#   - How many tasks it would take to resolve the gaps it shows

# %%
# %load_ext autoreload
# %autoreload 2

import logging
import os

import numpy as np
import pandas as pd
import scipy.stats

# %%
import helpers.hnotebook as hnotebook

import skill_vs_luck_utils as utils

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
# # Part 1: The benchmark and the submissions

# %% [markdown]
# ## Cell 1.1: Load the data
#
# - SWE-bench Verified has 500 real GitHub issues from 12 Python repos
#   - A fix counts only if it passes the tests of the issue
# - Every submission publishes which tasks it solved
#   - So the whole leaderboard becomes a 0/1 matrix of agents x tasks

# %%
# Load the outcome matrix and the metadata saved by `fetch_swebench_verified.py`.
outcomes, submissions, tasks = utils.load_swebench_dataset("data")

print("outcomes.shape=", outcomes.shape)
print("len(submissions)=", len(submissions))
# Outcome: 173 usable submissions out of 182, all on the same 500 tasks.

# %%
# Show the submissions left out of the analysis, and why.
excluded = submissions.loc[
    submissions["excluded_reason"] != "", ["name", "excluded_reason"]
]

display(excluded)
# Outcome: 7 submissions have no per-task results, and 2 have per-task results
# that do not match their own reported score.

# %% [markdown]
# ## Cell 1.2: Check the data against the reported scores
#
# - If the parsing is right, the score computed from the per-task results should
#   match the score each submission reports

# %%
# Compare the recomputed scores with the reported ones, in points.
kept = submissions.loc[outcomes.index]
mismatch = (kept["score"] - kept["reported_score"]).abs()

print("mismatch.count()=", mismatch.count())
print("mismatch.max()=", round(mismatch.max(), 2), "points")
# Outcome: all 39 submissions that report a score are within 0.13 points of it.

# %% [markdown]
# ## Cell 1.3: The leaderboard

# %%
# Rank the agents by the share of tasks solved, like the official site.
leaderboard = utils.compute_leaderboard(outcomes, submissions)

display(
    leaderboard.head(10)[["rank", "label", "score", "n_resolved", "attempts"]]
)
# Outcome: the top 10 are within 3.4 points of each other.

# %% [markdown]
# ## Cell 1.4: Attempts allowed per task
#
# - Some submissions try several times per task and submit the best attempt
#   - This is a selection effect, which Milestone 4 looks at

# %%
# Count the submissions by the number of attempts they allow per task.
attempts = leaderboard["attempts"].replace("", "not reported").value_counts()

display(attempts)
# Outcome: about half of the submissions allow more than 1 attempt.

# %%
# Compare the 2 EntroPO submissions: the same system with 1 and with several
# attempts.
entropo = leaderboard[leaderboard["name"].str.startswith("EntroPO")]

display(entropo[["label", "score", "attempts"]])
# Outcome: allowing several attempts adds about 8 points.

# %% [markdown]
# # Part 2: The uncertainty of each score

# %% [markdown]
# ## Cell 2.1: Interval on every score
#
# - The 500 tasks are 1 sample from a bigger pool of similar issues
# - A bootstrap over tasks shows how much each score would move with a different
#   sample of 500

# %%
# Compute a 95% interval for every score.
cis = utils.compute_score_cis(outcomes)
width = 100 * (cis["ci_high"] - cis["ci_low"])

print("width.median()=", round(width.median(), 1), "points")
# Outcome: a typical score is only known to within about +/- 4 points.

# %% [markdown]
# ## Cell 2.2: The top of the leaderboard with its intervals

# %%
# Plot the top 25 agents, shading the interval of the leader.
utils.plot_leaderboard_cis(leaderboard, cis, top_k=25, figsize=(14, 9))

# %% [markdown]
# # Part 3: Can the leaderboard tell neighbors apart?

# %% [markdown]
# ## Cell 3.1: Test every pair of neighbors
#
# - Each agent is compared with the next one down, task by task
#   - A paired bootstrap gives an interval for the gap
#   - The exact McNemar test gives a p-value
# - With 172 pairs, the Holm adjustment keeps the chance of any false positive
#   at 5%

# %%
# Compare every agent with the next one on the leaderboard.
adjacent = utils.compare_adjacent_ranks(outcomes, leaderboard)
# Count the pairs that differ before and after the Holm adjustment.
median_gap = 100 * adjacent["gap"].abs().median()
num_raw_significant = int((adjacent["p_value"] < 0.05).sum())
num_distinguishable = int(adjacent["distinguishable"].sum())

print("len(adjacent)=", len(adjacent))
print("median_gap=", round(median_gap, 1), "points")
print("num_raw_significant=", num_raw_significant)
print("num_distinguishable=", num_distinguishable)
# Outcome: the typical gap between neighbors is 0.2 points, and no pair can be
# told apart after Holm.

# %%
# Plot the gap between each pair of neighbors, with its paired interval.
utils.plot_adjacent_gaps(adjacent, figsize=(14, 5))

# %% [markdown]
# ## Cell 3.2: Who is tied with the leader?
#
# - Which agents can the data not separate from the agent ranked first?
# - The decision uses the Holm-adjusted p-value, since there are 172
#   comparisons with the leader
#   - The intervals in the table are not adjusted, so 10 of the 18 exclude 0 even
#     though those agents are tied after the adjustment
# - The leader was picked for having the top observed score, which makes its
#   lead look bigger than it is (winner's curse)
#   - So, if anything, this undercounts the ties

# %%
# Compare every agent with the leader.
versus_leader = utils.compare_to_leader(outcomes, leaderboard)
tied = versus_leader[versus_leader["tied_with_leader"]]
leader = leaderboard["label"].iloc[0]

print("leader=", leader)
print("len(tied)=", len(tied))
# Show the agents tied with the leader.
columns = ["rank", "label", "gap", "ci_low", "ci_high", "p_holm"]
display(tied[columns].round(4).reset_index(drop=True))
# Outcome: 18 agents, down to rank 19, are statistically tied for first place.

# %% [markdown]
# # Part 4: How many tasks would it take?

# %% [markdown]
# ## Cell 4.1: Tasks needed for a given gap
#
# - $\sigma_d$, the std of the per-task difference between 2 neighbors, sets how
#   fast the paired gap estimate gets sharper with more tasks
# - With the median $\sigma_d$ of the neighbor pairs, we get the number of tasks
#   needed to resolve a gap at 80% power

# %%
# Compute the tasks needed for a few gap sizes, with the median `sd_diff`.
sd_diff = adjacent["sd_diff"].median()
gaps = [0.01, 0.02, 0.03, 0.05]
min_tasks = pd.Series(
    [utils.compute_min_tasks(sd_diff, gap) for gap in gaps],
    index=[f"{100 * gap:.0f} points" for gap in gaps],
    name="tasks needed",
)

print("sd_diff=", round(sd_diff, 3))
display(min_tasks.round(0))
# Outcome: a 1-point gap needs about 14,700 tasks, almost 30 times the size of
# the benchmark.

# %% [markdown]
# ## Cell 4.2: The smallest gap 500 tasks can resolve
#
# - Solving the same formula for the gap gives the smallest gap that 500 tasks
#   detect with 80% power

# %%
# Compute the smallest resolvable gap for the real benchmark size.
min_gap = utils.compute_min_detectable_gap(sd_diff, outcomes.shape[1])

print("min_gap=", round(100 * min_gap, 1), "points")
# Outcome: 500 tasks resolve gaps of about 5.4 points or more, while the
# typical gap between neighbors is 0.2 points.

# %%
# Plot the tasks needed against the gap size.
utils.plot_min_tasks_curve(adjacent["sd_diff"], figsize=(12, 5))

# %% [markdown]
# # Part 5: Ranking under uncertainty
#
# - Part 3 compared agents 2 at a time
# - Here the whole leaderboard is bootstrapped at once
#   - Resample the 500 tasks with replacement 2,000 times
#   - Recompute all 173 scores and rank the agents in each sample
#   - The samples are the same ones used for the score intervals in Cell 2.1
# - Ties are split
#   - Tied agents share the average of their ranks
#   - An `n`-way tie for first gives each agent `1 / n` of the first place
#   - This matters because the top 2 agents both solve exactly 396 tasks

# %% [markdown]
# ## Cell 5.1: Rank interval of every agent

# %%
# Bootstrap the ranks of all the agents.
rank_intervals = utils.compute_rank_intervals(outcomes)
# Add the labels and keep the leaderboard order.
ranks = leaderboard[["label"]].join(rank_intervals)
rank_width = ranks["rank_high"] - ranks["rank_low"]

columns = ["label", "score", "rank", "median_rank", "rank_low", "rank_high"]
display(ranks[columns].head(10).round(2))
print("rank_width.head(25).median()=", rank_width.head(25).median())
# Outcome: the leader's 95% rank interval goes from 1 to 6, and a typical
# top-25 agent could land anywhere in a span of about 18 ranks.

# %%
# Plot the rank intervals of the top 25 agents.
utils.plot_rank_intervals(rank_intervals, leaderboard, top_k=25, figsize=(14, 9))

# %% [markdown]
# ## Cell 5.2: Who would come out first?

# %%
# Count the agents with a real chance of coming out first.
p_first = ranks["p_first"]

display(ranks[["label", "score", "p_first"]].head(5).round(3))
print("(p_first > 0).sum()=", int((p_first > 0).sum()))
print("p_first.head(3).sum()=", round(p_first.head(3).sum(), 3))
# Outcome: 14 agents come out first in at least 1 sample, and the top 3 share
# about 94% of the first place, with no agent above 34%.

# %%
# Plot the chance of coming out first for the top 10.
utils.plot_first_place_chance(
    rank_intervals, leaderboard, top_k=10, figsize=(14, 5)
)

# %% [markdown]
# ## Cell 5.3: Does the observed ranking find the true one?
#
# - On the real data the true ranking is unknown
# - So we simulate benchmarks with the Rasch model, where it is known
#   - 10 agents with evenly spaced abilities, centered at 0
#   - 4 gap sizes between neighbors and 4 benchmark sizes
#   - 200 simulated benchmarks for each setting
# - For each setting we measure:
#   - How often the truly best agent comes out first
#   - The Spearman and Kendall correlation between the observed and true
#     rankings

# %%
# Simulate every combination of gap and benchmark size.
logit_gaps = [0.02, 0.05, 0.1, 0.25]
n_tasks_list = [100, 500, 2000, 10000]
rows = []
for logit_gap in logit_gaps:
    abilities = logit_gap * (np.arange(10) - 4.5)
    # Express the gap as the difference in true score between the top 2.
    true_scores = utils.compute_expected_scores(abilities)
    gap_points = 100 * (true_scores.iloc[-1] - true_scores.iloc[-2])
    for n_tasks in n_tasks_list:
        recovery = utils.check_ranking_recovery(abilities, n_tasks)
        rows.append({"gap_points": gap_points, "n_tasks": n_tasks, **recovery})
recovery = pd.DataFrame(rows)

display(recovery.round(3))
# Outcome: with 500 tasks and a 1-point gap, the truly best agent comes out
# first only about half the time, and it takes 10,000 tasks to get it right
# more than 9 times out of 10.

# %%
# Plot the recovery against the number of tasks.
utils.plot_ranking_recovery(recovery, figsize=(14, 5))

# %% [markdown]
# # Part 6: Does the ranking hold on other tasks?
#
# - If the leaderboard measured the agents, the ranking on one half of the
#   tasks would match the ranking on the other half
# - Each split puts 250 random tasks in each half, and we repeat it 1,000 times

# %% [markdown]
# ## Cell 6.1: Split-half rank correlation
#
# - We correlate the 2 rankings for all 173 agents, the top 50, and the top 20
#   - The top groups are picked on all 500 tasks, which, if anything, makes
#     them look more stable than they are

# %%
# Correlate the rankings on the 2 halves, for 3 groups of agents.
groups = {
    "all 173 agents": outcomes,
    "top 50": outcomes.loc[leaderboard.index[:50]],
    "top 20": outcomes.loc[leaderboard.index[:20]],
}
split_half = {
    name: utils.compute_split_half_correlation(group)
    for name, group in groups.items()
}
summary = pd.DataFrame(
    {name: corr.median() for name, corr in split_half.items()}
).T

display(summary.round(3))
print(
    "(split_half['top 20']['kendall'] < 0).mean()=",
    (split_half["top 20"]["kendall"] < 0).mean(),
)
# Outcome: across all agents the ranking holds (Kendall 0.90), but among the
# top 20 it mostly does not (Kendall 0.25), and some splits even reverse it.

# %%
# Plot the Kendall tau of the 3 groups.
utils.plot_split_half_correlations(split_half, figsize=(12, 5))

# %% [markdown]
# ## Cell 6.2: The leader of one half on the other half
#
# - Pick the leader on one half of the tasks, then score it on the other half
#   - This is the held-out check of Milestone 4, on the real leaderboard
#   - A tie for first is split among the tied agents
# - The other half was not used to pick the leader, so its score there is a
#   fair estimate of its true score
#   - The drop is the winner's curse of picking on 250 tasks, about $\sqrt{2}$
#     times the one of picking on all 500 (Cell 7.2)

# %%
# Pick the leader on half of the tasks and score it on the other half.
held_out = utils.compute_held_out_drop(outcomes)

display(held_out.mean().round(4))
print("(held_out['drop'] > 0).mean()=", (held_out["drop"] > 0).mean())
# Outcome: the leader of one half scores 2.0 points less on the other half,
# and is the leader there too only about 2% of the time.

# %% [markdown]
# # Part 7: The winner's curse of trying many variants
#
# - A lab tries $K$ scaffolds or prompts and reports the best
#   - If they are all equally good, any gap between them is luck
#   - So the best observed score overstates the true score
# - The spec corrects for it with $\sigma \sqrt{2 \ln K}$
#   - $\sigma$ is the standard error of 1 score
# - The exact mean of the largest of $K$ normal draws is smaller
#   - E.g., 0.56 instead of 1.18 for $K = 2$

# %% [markdown]
# ## Cell 7.1: Simulate the best of K equal variants
#
# - $K$ variants with the same true ability (true score 50%) on the same tasks
# - $K$ = 1, 2, 5, 10, 20, 50 and 100, 500, 2,000 tasks, with 500 benchmarks
#   each
# - Each benchmark also picks the best variant on half of the tasks and scores
#   it on the other half

# %%
# Simulate every combination of K and benchmark size.
rows = []
for n_tasks in [100, 500, 2000]:
    for n_variants in [1, 2, 5, 10, 20, 50]:
        curse = utils.simulate_winners_curse(n_variants, n_tasks)
        rows.append({"n_tasks": n_tasks, "n_variants": n_variants, **curse})
curse = pd.DataFrame(rows)
# Compare each correction with the real inflation.
curse["correction_ratio"] = curse["correction"] / curse["inflation"]
curse["exact_ratio"] = curse["correction_exact"] / curse["inflation"]
with_selection = curse[curse["n_variants"] > 1]

columns = [
    "n_tasks",
    "n_variants",
    "inflation",
    "correction",
    "correction_exact",
    "held_out_drop",
    "correction_ratio",
    "exact_ratio",
]
display(with_selection[columns].round(4))
print(
    "correction_ratio range=",
    with_selection["correction_ratio"].agg(["min", "max"]).round(2).tolist(),
)
print(
    "exact_ratio range=",
    with_selection["exact_ratio"].agg(["min", "max"]).round(2).tolist(),
)
# Outcome: with 500 tasks, the best of 20 equal variants looks 3.7 points better
# than it is; the spec's correction (5.5 points) is 1.4 to 2.5 times the real
# inflation, while the exact one (4.2 points) is only 6% to 22% too high.

# %%
# Plot the inflation against K, with both corrections.
utils.plot_winners_curse(curse, figsize=(14, 6))

# %% [markdown]
# ## Cell 7.2: The held-out drop
#
# - The drop on the held-out half is what a real prompt search can measure,
#   since the true score is unknown
# - It picks on half of the tasks, so it carries the inflation of a smaller
#   benchmark: about $\sqrt{2}$ times the inflation on all the tasks

# %%
# Compare the held-out drop with the inflation on all the tasks.
ratio = (curse["held_out_drop"] / curse["inflation"])[curse["n_variants"] > 1]

print("ratio.median()=", round(ratio.median(), 2))
# Outcome: the held-out drop is about 1.5 times the inflation on all the tasks,
# close to sqrt(2), so a measured drop can be scaled back to the full benchmark.

# %% [markdown]
# # Part 8: Item response theory
#
# - The leaderboard scores every task the same, but tasks differ
#   - Some are solved by every agent, some by none
# - A Bayesian 2PL item response theory (IRT) model fits:
#   - An ability $\theta_a$ for each agent
#   - A difficulty $b_i$ and a discrimination $\alpha_i$ for each task
#   - Agent $a$ solves task $i$ with probability
#     $\sigma(\alpha_i (\theta_a - b_i))$
# - We fit it in NumPyro with variational inference (SVI)

# %% [markdown]
# ## Cell 8.1: Fit the model

# %%
# Fit the 2PL model on the whole matrix.
irt = utils.fit_irt_2pl(outcomes)

display(irt["tasks"].describe().round(2))
# Outcome: the difficulties span about 12 logits, and the discriminations sit
# around 1.3.

# %%
# Check SVI against NUTS (MCMC) on a subset of the matrix, where NUTS is fast.
subset = outcomes.loc[leaderboard.index[:40], outcomes.columns[:150]]
irt_svi = utils.fit_irt_2pl(subset)
irt_nuts = utils.fit_irt_2pl(subset, method="nuts", n_samples=500)
ability_corr = np.corrcoef(
    irt_svi["agents"]["ability"], irt_nuts["agents"]["ability"]
)[0, 1]
sd_ratio = (
    irt_svi["agents"]["ability_sd"] / irt_nuts["agents"]["ability_sd"]
).median()

print("ability_corr=", round(ability_corr, 4))
print("sd_ratio=", round(sd_ratio, 2))
# Outcome: SVI and NUTS agree on the abilities (correlation 0.997) and on their
# uncertainty (ratio of stds 0.98), so SVI is enough for the full matrix.

# %% [markdown]
# ## Cell 8.2: Ranking by ability against the leaderboard
#
# - The ability weighs a hard task more than an easy one

# %%
# Compare the ranking by ability with the leaderboard.
agents = leaderboard[["label", "score", "rank"]].join(irt["agents"])
agents["irt_rank"] = (
    agents["ability"].rank(ascending=False, method="min").astype(int)
)
top_20 = agents.head(20)

print(
    "kendall_all=",
    round(
        scipy.stats.kendalltau(agents["ability"], agents["score"]).statistic, 3
    ),
)
print(
    "kendall_top_20=",
    round(
        scipy.stats.kendalltau(top_20["ability"], top_20["score"]).statistic, 3
    ),
)
display(top_20[["label", "score", "rank", "ability", "irt_rank"]].round(3))
# Outcome: the 2 rankings agree overall (Kendall 0.97) but less in the top 20
# (Kendall 0.81), where ACoder drops from rank 9 to 19 since it solves easier
# tasks.

# %% [markdown]
# ## Cell 8.3: Fitted difficulty against the human labels
#
# - Each task has a human estimate of the time an engineer needs to fix it

# %%
# Compare the fitted difficulties with the human labels.
label_order = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]
label_code = tasks["difficulty"].map(
    {label: code for code, label in enumerate(label_order)}
)
label_corr = scipy.stats.spearmanr(
    label_code.loc[irt["tasks"].index], irt["tasks"]["difficulty"]
).statistic
by_label = (
    irt["tasks"].join(tasks["difficulty"].rename("label")).groupby("label")
)

display(
    by_label[["difficulty", "discrimination"]].median().loc[label_order].round(2)
)
print("label_corr=", round(label_corr, 2))
# Outcome: the median difficulty rises with the human label (-1.6, 0.0, 2.9,
# 5.0 logits), but the labels explain only part of it (Spearman 0.45).

# %%
# Plot the difficulties by human label.
utils.plot_difficulty_by_label(
    irt["tasks"], tasks["difficulty"], label_order=label_order, figsize=(12, 5)
)

# %% [markdown]
# ## Cell 8.4: The effective size of the benchmark
#
# - A task tells agents apart only if some solve it and some fail it
# - The Fisher information of a task, averaged over a group of agents,
#   measures how much it tells about them
# - The effective size $(\sum_i I_i)^2 / \sum_i I_i^2$ is the number of equally
#   informative tasks the benchmark is worth

# %%
# Count the tasks that tell the top 20 agents apart.
top_20_outcomes = outcomes.loc[top_20.index]
n_solved_top_20 = top_20_outcomes.sum(axis=0)

print("(n_solved_top_20 == 20).sum()=", int((n_solved_top_20 == 20).sum()))
print("(n_solved_top_20 == 0).sum()=", int((n_solved_top_20 == 0).sum()))
print(
    "utils.count_informative_tasks(top_20_outcomes)=",
    utils.count_informative_tasks(top_20_outcomes),
)
# Outcome: the top 20 all solve 254 tasks and all fail 36, so only 210 of the 500
# tasks can tell them apart.

# %%
# Compute the information of every task for 2 groups of agents.
information = {
    "all 173 agents": utils.compute_task_information(
        agents["ability"], irt["tasks"]
    ),
    "top 20": utils.compute_task_information(top_20["ability"], irt["tasks"]),
}
effective_size = pd.Series(
    {
        name: utils.compute_effective_size(info)
        for name, info in information.items()
    }
)

# Count the most informative tasks that hold half of the information.
top_20_share = (
    information["top 20"].sort_values(ascending=False).cumsum()
    / information["top 20"].sum()
)
n_tasks_half = int((top_20_share < 0.5).sum()) + 1

display(effective_size.round(0))
print("n_tasks_half=", n_tasks_half)
# Outcome: for the top 20 the benchmark is worth about 107 equally informative
# tasks, and 39 tasks hold half of the information.

# %%
# Plot the share of the information held by the most informative tasks.
utils.plot_task_information(information, figsize=(12, 5))

# %% [markdown]
# ## Cell 8.5: Save the fit
#
# - The abilities and difficulties are saved for the analysis of run luck,
#   which needs task difficulties

# %%
# Save the fitted abilities and task parameters.
irt["agents"].to_csv(os.path.join("data", "swebench_verified_irt_agents.csv"))
irt["tasks"].to_csv(os.path.join("data", "swebench_verified_irt_tasks.csv"))

# %% [markdown]
# # Part 9: Takeaways and next steps

# %% [markdown]
# ## Takeaways
#
# - A single SWE-bench Verified score is only known to about $\pm 4$ points
#   - The median 95% interval is 8.2 points wide
# - The leaderboard cannot tell neighbors apart
#   - The median gap between neighbors is 0.2 points
#   - None of the 172 neighbor pairs differ after the Holm correction
#   - Only 2 differ before it
# - The top is a statistical tie
#   - 18 agents, down to rank 19 at 74.4%, cannot be separated from the leader
#     at 79.2%
# - The ranking itself is uncertain
#   - The top 3 agents share about 94% of the first place, and none is above
#     34%
#   - A typical top-25 agent has a 95% rank interval about 18 ranks wide
#   - In simulation, 500 tasks pick the truly best of 10 agents only about half
#     the time when neighbors are 1 point apart
# - The top of the leaderboard does not persist
#   - Over all 173 agents, the rankings on 2 halves of the tasks agree (Kendall
#     0.90), but among the top 20 they barely do (Kendall 0.25)
#   - The leader of one half scores 2.0 points less on the other half, and
#     leads it only about 2% of the time
# - Reporting the best of many variants inflates the score
#   - The best of 20 equal variants on 500 tasks looks 3.7 points better than
#     it is
#   - The spec's $\sigma \sqrt{2 \ln K}$ is 1.4 to 2.5 times the real
#     inflation for $K$ up to 50
#   - $\sigma$ times the mean of the largest of $K$ normals is only 6% to 22%
#     too high, since the variants share the same tasks
# - Only a fraction of the benchmark tells the top agents apart
#   - The top 20 all solve 254 tasks and all fail 36, leaving 210
#   - Under a 2PL IRT fit, the 500 tasks are worth about 107 equally
#     informative tasks for the top 20, and 39 tasks hold half the information
#   - Ranking by IRT ability mostly matches the leaderboard (Kendall 0.97), but
#     moves agents by up to 10 ranks in the top 20
#   - The fitted difficulty rises with the human estimate of the time to fix
#     (Spearman 0.45)
# - 500 tasks are too few for the gaps on the leaderboard
#   - Resolving a 1-point gap with 80% power needs about 14,700 tasks
#   - With 500 tasks, the smallest resolvable gap is about 5.4 points
# - These intervals only cover the luck of which tasks are in the benchmark
#   - Each submission reports 1 run per task, so run-to-run luck is not included
#   - The real uncertainty is larger

# %% [markdown]
# ## Next steps
#
# - Milestone 2: measure run-to-run luck
#   - Run small open models 5 times per task on HumanEval+ and MBPP+
#   - Everything runs locally with Ollama
# - Milestone 3: run the same analysis on HumanEval+ and MBPP+, and compare the
#   rankings across benchmarks
# - Milestone 4: compare the simulated held-out drop with the drop measured on
#   real prompt variants
#   - 82 of the 173 submissions allow 2 or more attempts per task (Cell 1.4)
#   - The same EntroPO system scores 52.2% with 1 attempt and 60.4% with several
