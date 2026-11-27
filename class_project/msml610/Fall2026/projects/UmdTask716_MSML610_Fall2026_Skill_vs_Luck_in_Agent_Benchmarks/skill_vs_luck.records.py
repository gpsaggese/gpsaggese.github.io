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
# # Were the SWE-bench Verified records real?
#
# - This notebook tests every record of the SWE-bench Verified leaderboard
#   against the record it broke, on the same 500 tasks
#   - It is Study A of `preregistration_2.md`, whose design and predictions
#     were committed before this analysis was run
#   - A submission sets a record when it resolves more tasks than every
#     submission dated before it
#   - Each record gets the exact McNemar test of Milestone 2, with Holm over
#     all the records
# - The pedagogical arc:
#   - The chain of records
#   - Each record against the one it broke
#   - How much of the progress came in steps the tasks can resolve
#   - Which predictions held
#   - 2 extra looks, not pre-registered: the tests without Holm, and how many
#     levels the 500 tasks can tell apart

# %%
# %load_ext autoreload
# %autoreload 2

import logging

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
# # Part 1: The chain of records

# %% [markdown]
# ## Cell 1.1: Find the records
#
# - The data is the pinned snapshot of Milestones 1 and 2: 173 submissions on
#   the same 500 tasks, each dated by its folder name
#   - The 9 submissions without usable per-task results cannot be in the chain
# - Of the submissions with the same date, only the best can set the record
# - The earliest submission starts the chain, so every later record has a
#   record it broke

# %%
# Load the data and find the records.
outcomes, submissions, tasks = utils.load_swebench_dataset("data")
leaderboard = utils.compute_leaderboard(outcomes, submissions)
records = utils.find_records(leaderboard)

print("records=", len(records))
display(records[["date", "label", "score", "n_resolved", "attempts"]])
# Outcome: 31 records, from RAG + Claude 2 at 4.4% on 2023-10-10 to Sonar
# Foundation Agent + Claude 4.5 Opus at 79.2% on 2025-12-05.

# %% [markdown]
# # Part 2: Each record against the one it broke

# %% [markdown]
# ## Cell 2.1: Test each record
#
# - `gap` is the score of the new record minus the record it broke
# - `only_a` and `only_b` count the tasks that only the new record or only the
#   old one solves, and only those tasks say anything about the gap
# - `ci_low` and `ci_high` are the 95% paired bootstrap interval of the gap,
#   for each record on its own, so they are not adjusted for the 30 tests

# %%
# Compare each record with the record it broke.
audit = utils.audit_records(outcomes, records)

print("median_gap=", round(audit["gap"].median(), 4))
display(audit.drop(columns=["previous"]))
# Outcome: only 2 of the 30 records are distinguishable after Holm: SWE-agent +
# GPT 4 (1106), 18.0 points above RAG + Claude 2, and MASAI + GPT 4o, 7.0
# points above the record before it. The median record is 1.5 points above the
# one it broke.

# %% [markdown]
# ## Cell 2.2: Plot the records
#
# - The gray staircase is the record over time
# - Green records are distinguishable from the record they broke, after Holm

# %%
# Plot the records over time.
utils.plot_records(records, audit, figsize=(10, 5))
# Outcome: after MASAI + GPT 4o in June 2024 no record is distinguishable from
# the one it broke, though the next 27 records took the top score from 32.6% to
# 79.2%.

# %% [markdown]
# # Part 3: How much of the progress was real

# %% [markdown]
# ## Cell 3.1: Summarize the audit
#
# - `total_gap` is the score of the last record minus the first, which is the
#   sum of all the record gaps
# - `p_first_last` tests the last record against the first one
# - `distinguishable_share` is the share of `total_gap` that came in the
#   records distinguishable after Holm

# %%
# Summarize the record audit.
summary = utils.summarize_records(outcomes, records, audit)

display(summary)
# Outcome: the 74.8 points from the first record to the last are clearly real
# (p = 5.2e-113), but only 33% of them came in the 2 distinguishable steps, and
# 25 of the 30 records come from submissions that allow more than 1 attempt.

# %% [markdown]
# # Part 4: Which predictions held

# %% [markdown]
# ## Cell 4.1: Check A1 to A3
#
# - The rules come from `preregistration_2.md`, committed before this analysis
#   was run

# %%
# Check the pre-registered predictions.
predictions = utils.check_record_predictions(summary)

display(predictions)
# Outcome: A1 and A2 hold and A3 does not: only 2 of the 30 records stand out,
# and 67% of the progress came in steps too small to tell apart one at a time,
# even though the total is clearly real.

# %% [markdown]
# # Part 5: Extra looks, not pre-registered

# %% [markdown]
# ## Cell 5.1: The same counts without Holm
#
# - Not pre-registered: this checks how much A1 and A3 depend on the Holm
#   adjustment

# %%
# Count the records with a raw p-value below 0.05.
raw = audit["p_value"] < 0.05
raw_share = audit.loc[raw, "gap"].sum() / audit["gap"].sum()

print("raw_significant=", int(raw.sum()), "raw_share=", round(raw_share, 3))
# Outcome: without Holm, 6 of the 30 records have p below 0.05 and they carry
# 56% of the points, so A1 holds either way, while A3 only holds without the
# adjustment.

# %% [markdown]
# ## Cell 5.2: Levels the tasks can tell apart
#
# - Not pre-registered: going through the records in order, a record starts a
#   new level when it is distinguishable from the record that started the
#   current level (raw p below 0.05)
# - The tests are not adjusted, so the count of levels is rough

# %%
# Group the records into levels.
levels = utils.find_distinct_levels(outcomes, records)

print("levels=", len(levels))
display(levels)
# Outcome: the 31 records fall into 12 levels, and each new level is 4.0 to
# 18.0 points above the one before, so on 500 tasks a jump of about 4 points is
# the smallest that stands out.

# %% [markdown]
# # Part 6: Takeaways
#
# - The SWE-bench Verified record was broken 30 times in this snapshot, but only
#   2 of those records are distinguishable from the record they broke
# - The progress itself is real: 74.8 points from the first record to the last
#   - Since June 2024, 27 records in a row were each within the noise of the
#     record before, and together they still added 46.6 points
#   - A single new record on 500 tasks says little, while the trend over many
#     records says a lot
# - 25 of the 30 records come from submissions that allow more than 1 attempt
#   per task, so most records are not 1-run scores
# - Prediction A3 missed: we expected most of the progress to come in steps
#   that stand out, and only 33% did
