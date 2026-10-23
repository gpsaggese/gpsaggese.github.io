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
# # Skill vs luck API: measuring uncertainty in agent benchmarks
#
# - This notebook goes through the functions in `skill_vs_luck_utils.py`, which
#   check how much of an agent leaderboard is skill and how much is luck
#   - Each function runs on the real SWE-bench Verified data or on a small
#     example that can be checked by hand
# - The pedagogical arc:
#   - The data: a 0/1 matrix of agents x tasks
#   - How uncertain 1 score is: a bootstrap over tasks
#   - Comparing 2 agents: paired tests on the same tasks
#   - Benchmark size: how many tasks a given gap needs
#   - Checking the tools on simulated benchmarks where the truth is known

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import pandas as pd

# %%
import helpers.hintrospection as hintros
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
# ## Library Overview
#
# - **What problem it solves**: small gaps on a leaderboard get read as real
#   differences
#   - Each agent gets 1 number, from 1 run over a few hundred tasks
#   - These functions put error bars on the numbers and test whether 2 agents
#     really differ
# - **Key abstraction**: the outcome matrix $s_{a,i} \in \{0, 1\}$
#   - 1 row per agent $a$ and 1 column per task $i$
#   - A score is a row mean
#   - The tasks are a sample from a bigger pool of similar tasks
# - **Mental model**:
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `load_swebench_dataset(data_dir)` | Outcome matrix, submissions, tasks | 173 x 500 matrix of 0/1 |
# | `compute_leaderboard(outcomes, submissions)` | Agents ranked by score | Ties share a rank |
# | `compute_score_cis(outcomes)` | Interval on each score | Bootstrap over tasks |
# | `compare_pair(a, b)` | Paired test of 2 agents | Gap, interval, McNemar p-value |
# | `holm_adjust(p_values)` | Correction for many tests | Controls the chance of any false positive |
# | `compute_min_tasks(sd_diff, gap)` | Tasks needed for a gap | Normal approximation |
# | `compute_min_detectable_gap(sd_diff, n_tasks)` | Smallest gap a benchmark can detect | Inverse of the above |
# | `simulate_outcomes(abilities, n_tasks)` | Benchmark where the truth is known | Rasch model |
#
# - **Key functions**:
#   - `compute_score_cis()`: how uncertain 1 score is
#   - `compare_pair()`: whether 2 agents differ
#   - `compute_min_tasks()`: how many tasks a benchmark needs

# %% [markdown]
# # Part 1: The data

# %% [markdown]
# ## Cell 1.1: Load the outcome matrix
#
# - `fetch_swebench_verified.py` builds the files in `data/` once
#   - It uses the public per-task results of every SWE-bench Verified submission
# - The notebooks only read these files, so they run offline

# %%
# Load the outcome matrix, the submission metadata, and the task metadata.
outcomes, submissions, tasks = utils.load_swebench_dataset("data")

print("outcomes.shape=", outcomes.shape)
print("submissions.shape=", submissions.shape)
print("tasks.shape=", tasks.shape)
# Outcome: 173 of the 182 submission folders are kept, each with all 500 tasks.

# %%
# Show a corner of the outcome matrix: 1 means the agent solved the task.
display(outcomes.iloc[:5, :4])
# Outcome: each row is a submission folder and each column a task id.

# %% [markdown]
# ## Cell 1.2: Parse 1 submission's results
#
# - Submissions save their results in 2 formats
#   - Both become a row of the outcome matrix
# - `parse_results_json()` reads the older one: a list of solved task ids

# %%
# Parse a small made-up `results.json` with 3 tasks.
results = {"resolved": ["task_b"], "no_generation": ["task_c"]}
row = utils.parse_results_json(results, ["task_a", "task_b", "task_c"])

print("row=", row.to_dict())
# Outcome: only `task_b` is solved. A task with no patch counts as a failure,
# like on the official leaderboard.

# %% [markdown]
# ## Cell 1.3: Build the leaderboard

# %%
# Rank the agents by the share of tasks they solved.
leaderboard = utils.compute_leaderboard(outcomes, submissions)

display(leaderboard.head(5))
# Outcome: the top 2 agents solved the same number of tasks, so both are rank 1.

# %% [markdown]
# # Part 2: The uncertainty of 1 score

# %% [markdown]
# ## Cell 2.1: Bootstrap over tasks
#
# - A score is an average over 500 tasks
#   - A different set of 500 similar tasks would give a different average
# - `compute_score_cis()` resamples the tasks with replacement and recomputes
#   every score each time
#   - All agents use the same resamples, so comparisons between agents stay
#     paired

# %%
# Show the signature and the docstring summary of `compute_score_cis()`.
hintros.print_obj_info(utils.compute_score_cis)

# %%
# Compute a 95% interval for every score.
cis = utils.compute_score_cis(outcomes)

display(cis.loc[leaderboard.index[:5]])
# Outcome: each interval is about 7 points wide, so 79.2% really means
# somewhere between about 75.6% and 82.6%.

# %% [markdown]
# ## Cell 2.2: Compare the bootstrap with the binomial formula
#
# - For 0/1 outcomes the standard error has a closed form: $\sqrt{p (1 - p) / N}$
# - The bootstrap should give about the same value, which checks the code

# %%
# Compare the 2 standard errors across all agents, in points.
gap_se = 100 * (cis["se_boot"] - cis["se_binomial"]).abs()

print("gap_se.max()=", round(gap_se.max(), 3), "points")
# Outcome: the 2 agree to about a tenth of a point.

# %% [markdown]
# # Part 3: The comparison of 2 agents

# %% [markdown]
# ## Cell 3.1: Paired test of the 1st and 3rd agents
#
# - A lot of the noise in a score comes from which tasks happen to be easy
#   - Both agents get the same tasks, so comparing them task by task cancels it
# - Only the tasks where exactly 1 of the 2 agents succeeds say anything about
#   the gap
#   - The exact McNemar test checks whether these tasks split 50/50
# - The top 2 agents have the same score, so the example uses the 1st and the 3rd

# %%
# Compare the agents ranked 1st and 3rd.
first, third = leaderboard.index[0], leaderboard.index[2]
comparison = utils.compare_pair(outcomes.loc[first], outcomes.loc[third])

display(pd.Series(comparison).round(4))
# Outcome: a 0.4-point gap with an interval that includes 0 and a p-value of
# 0.9, so the data cannot tell these 2 agents apart.

# %% [markdown]
# ## Cell 3.2: Correct for testing many pairs
#
# - Testing 172 neighbor pairs at the 5% level gives about 9 false positives by
#   chance, even if no pair really differs
# - `holm_adjust()` raises the p-values so that the chance of even 1 false
#   positive stays at 5%

# %%
# Adjust 3 made-up p-values.
adjusted = utils.holm_adjust([0.01, 0.04, 0.03])

print("adjusted=", adjusted.round(4))
# Outcome: [0.03, 0.06, 0.06], so only the first one stays below 0.05.

# %% [markdown]
# # Part 4: The size of a benchmark

# %% [markdown]
# ## Cell 4.1: Tasks needed to resolve a gap
#
# - The paired gap has standard error $\sigma_d / \sqrt{N}$
#   - $\sigma_d$ is the standard deviation of the per-task difference
# - To detect a gap $\Delta$ at level $\alpha$ with power $1 - \beta$:
#   $$N \geq \frac{(z_{1 - \alpha / 2} + z_{1 - \beta})^2 \sigma_d^2}{\Delta^2}$$

# %%
# Use the `sd_diff` of the 1st and 3rd agents from Cell 3.1.
gaps = [0.01, 0.02, 0.05, 0.10]
min_tasks = pd.Series(
    [utils.compute_min_tasks(comparison["sd_diff"], gap) for gap in gaps],
    index=[f"{100 * gap:.0f} points" for gap in gaps],
    name="tasks needed",
)

display(min_tasks.round(0))
# Outcome: resolving a 1-point gap needs over 10,000 tasks, while SWE-bench
# Verified has 500.

# %% [markdown]
# ## Cell 4.2: The smallest gap 500 tasks can resolve
#
# - `compute_min_detectable_gap()` solves the same formula for $\Delta$
#   - Given $N$, it returns the smallest gap the test detects with 80% power

# %%
# Use the same `sd_diff` as in Cell 4.1, with the real size of 500 tasks.
min_gap = utils.compute_min_detectable_gap(comparison["sd_diff"], 500)

print("min_gap=", round(100 * min_gap, 1), "points")
# Outcome: about 4.5 points, more than 10 times the 0.4-point gap between the
# 1st and the 3rd agent.

# %% [markdown]
# # Part 5: Checking the tools on simulated benchmarks

# %% [markdown]
# ## Cell 5.1: Simulate a benchmark with known abilities
#
# - `simulate_outcomes()` draws a benchmark from a Rasch model
#   - Agent $a$ solves task $i$ with probability $\sigma(\theta_a - b_i)$
#   - $\theta_a$ is the ability of the agent
#   - $b_i$ is the difficulty of the task
# - The true abilities are known here, so we can check if the tools get the
#   right answer

# %%
# Simulate 3 agents on 500 tasks.
abilities = [-1.5, 0.0, 1.5]
simulated = utils.simulate_outcomes(abilities, 500, difficulty_sd=1.0, seed=0)
true_scores = utils.compute_expected_scores(abilities, difficulty_sd=1.0)

display(
    pd.DataFrame({"true_score": true_scores, "observed": simulated.mean(axis=1)})
)
# Outcome: the observed scores are within 2 points of the true ones.

# %% [markdown]
# ## Cell 5.2: Check the coverage of the intervals
#
# - A correct 95% interval should contain the true score in about 95% of
#   repeated benchmarks

# %%
# Simulate 200 benchmarks and count how often each interval has the true score.
coverage = utils.check_ci_coverage(abilities, 500, n_trials=200, seed=0)

display(coverage.round(3))
# Outcome: coverage is close to 0.95 for all 3 agents.

# %% [markdown]
# ## Cell 5.3: Paired test vs overlapping intervals
#
# - A common shortcut calls 2 agents different only when their intervals do not
#   overlap
# - The simulation runs the shortcut and the paired test on the same benchmarks
#   - With 2 equal agents, a good rule finds a difference about 5% of the time
#   - With 2 different agents, a good rule finds it as often as possible

# %%
# Equal abilities: how often does each rule raise a false alarm?
false_alarms = utils.compare_test_power(0.0, 0.0, 500, difficulty_sd=2.0, seed=0)

print("false_alarms=", false_alarms.to_dict())
# Outcome: the paired test is close to 5%, and the overlap rule never fires
# here.

# %%
# Abilities 0.3 apart on the logit scale: how often does each rule notice?
power = utils.compare_test_power(0.3, 0.0, 500, difficulty_sd=2.0, seed=0)

print("power=", power.to_dict())
# Outcome: the paired test finds the gap far more often than the overlap rule.

# %% [markdown]
# ## Summary: The Mental Model
#
# - A benchmark score is an estimate
#   - With 500 tasks, its 95% interval is several points wide
#   - A bootstrap over tasks measures how wide
# - 2 agents on the same tasks should be compared task by task
#   - The paired test keeps false alarms near 5% and finds real gaps much more
#     often than checking whether intervals overlap
# - Many comparisons need a correction like Holm's, or chance alone produces
#   "significant" gaps
# - The tasks needed to resolve a gap grow like $1 / \Delta^2$
#   - Halving the gap needs 4 times the tasks
