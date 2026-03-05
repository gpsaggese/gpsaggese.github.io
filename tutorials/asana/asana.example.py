# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Asana: User Activity Analytics
#
# This end-to-end example shows how to pull data from Asana and compute
# user-activity statistics for a set of projects over a configurable time
# window.
#
# **Workflow:**
# 1. Authenticate with an Asana Personal Access Token (PAT).
# 2. Fetch all tasks created within a date range.
# 3. Fetch comments (stories) for those tasks.
# 4. Aggregate per-user statistics: tasks opened, tasks closed, comments made.
#
# **Prerequisites:**
# - Set `ASANA_ACCESS_TOKEN` to a valid Asana PAT.
# - Set `ASANA_PROJECT_GID` to a comma-separated list of project GIDs
#   (e.g. `"123456789,987654321"`).

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import datetime
import logging
import os

import helpers.hdbg as hdbg
import helpers.hpandas as hpandas
import helpers.hprint as hprint
import tutorials.tutorial_asana.asana_utils as ttuaasuti

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)
hprint.config_notebook()

# %% [markdown]
# ## Part 1: Configuration
#
# All run-time parameters live in a single `config` dictionary.  Update the
# values here to analyse a different project or time window.

# %%
today = datetime.datetime.now(tz=datetime.timezone.utc)
one_month_ago = today - datetime.timedelta(days=30)

# Read project GIDs from the environment; fall back to an empty list.
_project_gids_raw = os.environ.get("ASANA_PROJECT_GID", "")
project_ids = [g.strip() for g in _project_gids_raw.split(",") if g.strip()]

config = {
    # List of Asana project GIDs to analyse.
    "project_ids": project_ids,
    # ISO-8601 date range for task creation filter.
    "start_date": one_month_ago.isoformat(),
    "end_date": today.isoformat(),
    # PAT loaded from environment variable.
    "access_token": os.environ.get("ASANA_ACCESS_TOKEN", ""),
}

_LOG.info("Project IDs : %s", config["project_ids"])
_LOG.info("Date range  : %s → %s", config["start_date"], config["end_date"])

# %% [markdown]
# ## Part 2: Authentication
#
# We create a single `AsanaClient` instance that is reused throughout the
# notebook.  Credentials are read from the environment rather than hard-coded.

# %%
hdbg.dassert(
    config["access_token"],
    "Set the ASANA_ACCESS_TOKEN environment variable before running.",
)
hdbg.dassert(
    config["project_ids"],
    "Set ASANA_PROJECT_GID to a comma-separated list of project GIDs.",
)

client = ttuaasuti.AsanaClient(access_token=config["access_token"])
_LOG.info("Asana client initialised.")

# %% [markdown]
# ## Part 3: Fetch Task Data
#
# `fetch_tasks` retrieves all tasks created within the configured date window
# and returns a normalised DataFrame with project and task status columns.

# %%
tasks_df = ttuaasuti.fetch_tasks(
    client,
    project_ids=config["project_ids"],
    start_date=config["start_date"],
    end_date=config["end_date"],
)
_LOG.info(
    "tasks_df=\n%s", hpandas.df_to_str(tasks_df, log_level=logging.INFO)
)

# %% [markdown]
# ## Part 4: Fetch Comments
#
# `fetch_comments` collects all `comment_added` stories for the retrieved
# tasks and returns them as a single DataFrame.

# %%
task_ids = tasks_df["task_id"].tolist() if not tasks_df.empty else []
comments_df = ttuaasuti.fetch_comments(client, task_ids=task_ids)
_LOG.info(
    "comments_df=\n%s", hpandas.df_to_str(comments_df, log_level=logging.INFO)
)

# %% [markdown]
# ## Part 5: Summary Statistics
#
# High-level counts give a quick snapshot of activity in the period.

# %%
_LOG.info("Tasks created  : %d", len(tasks_df))
_LOG.info("Tasks completed: %d", tasks_df["task_status"].eq("Completed").sum())
_LOG.info("Comments total : %d", len(comments_df))

# %% [markdown]
# ## Part 6: Per-User Analytics
#
# ### 6.1 Tasks per User

# %%
if not tasks_df.empty and "assignee" in tasks_df.columns:
    tasks_by_user = (
        tasks_df.groupby("assignee")["task_id"]
        .count()
        .reset_index()
        .rename(columns={"task_id": "tasks_created"})
        .sort_values("tasks_created", ascending=False)
    )
    print("Tasks created per user:")
    print(tasks_by_user.to_string(index=False))
else:
    print("No task data available.")

# %% [markdown]
# ### 6.2 Completed Tasks per User

# %%
completed_df = (
    tasks_df[tasks_df["task_status"] == "Completed"]
    if not tasks_df.empty
    else tasks_df
)
if not completed_df.empty and "assignee" in completed_df.columns:
    tasks_completed_by_user = (
        completed_df.groupby("assignee")["task_id"]
        .count()
        .reset_index()
        .rename(columns={"task_id": "tasks_completed"})
        .sort_values("tasks_completed", ascending=False)
    )
    print("Tasks completed per user:")
    print(tasks_completed_by_user.to_string(index=False))
else:
    print("No completed tasks in the selected period.")

# %% [markdown]
# ### 6.3 Comments per User

# %%
if not comments_df.empty and "author" in comments_df.columns:
    comments_by_user = (
        comments_df.groupby("author")["task_id"]
        .count()
        .reset_index()
        .rename(columns={"task_id": "comments_count"})
        .sort_values("comments_count", ascending=False)
    )
    print("Comments per user:")
    print(comments_by_user.to_string(index=False))
else:
    print("No comments found in the selected period.")

# %% [markdown]
# ### 6.4 Combined User Activity

# %%
if not tasks_df.empty:
    activity_df = ttuaasuti.get_user_activity_stats(tasks_df)
    print("User activity summary (tasks opened / closed):")
    print(activity_df.to_string(index=False))
