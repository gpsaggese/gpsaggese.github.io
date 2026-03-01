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
# # Asana API Overview
#
# This notebook walks through the key concepts of the **Asana REST API** and
# the thin Python wrapper built on top of the official
# [`python-asana`](https://github.com/Asana/python-asana) client.
#
# **What you will learn:**
# - How to authenticate with a Personal Access Token (PAT).
# - The core Asana resource hierarchy: Workspaces → Projects → Tasks → Stories.
# - How to call each API tier with the `AsanaClient` wrapper.
# - How retry logic protects against transient failures.
#
# **Prerequisites:**
# - Set the environment variable `ASANA_ACCESS_TOKEN` to a valid PAT before
#   running this notebook.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import logging
import os

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import tutorials.tutorial_asana.asana_utils as ttuaasuti

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)
hprint.config_notebook()

# %% [markdown]
# ## 1. Authentication
#
# The Asana API uses **Personal Access Tokens (PAT)** or OAuth 2.0.  Our
# wrapper hides the SDK boilerplate behind a single `AsanaClient` object:
#
# - Instantiate once with `AsanaClient(access_token)`.
# - The client exposes three pre-configured API handles:
#   - `.tasks_api` — `asana.TasksApi`
#   - `.projects_api` — `asana.ProjectsApi`
#   - `.comments_api` — `asana.StoriesApi`

# %%
# Load the PAT from an environment variable (never hard-code tokens).
ACCESS_TOKEN = os.environ.get("ASANA_ACCESS_TOKEN", "")
hdbg.dassert(ACCESS_TOKEN, "Set the ASANA_ACCESS_TOKEN environment variable.")

# Create the client.
client = ttuaasuti.AsanaClient(access_token=ACCESS_TOKEN)
_LOG.info("AsanaClient created: tasks_api=%s", type(client.tasks_api).__name__)

# %% [markdown]
# ## 2. Retry Helpers
#
# The `fetch_with_retries` helper wraps any callable and automatically handles:
#
# | HTTP status | Action |
# |---|---|
# | `429` (rate limit) | Sleep for the value of `Retry-After`, then retry |
# | `5xx` (server error) | Sleep for *delay* seconds, then retry |
# | Other errors | Raise immediately |
#
# Usage pattern:
#
# ```python
# result = ttuaasuti.fetch_with_retries(
#     client.tasks_api.some_method,
#     arg1, arg2,
#     retries=3,
#     delay=2,
# )
# ```

# %%
# Demonstration: call a benign endpoint through fetch_with_retries.
# (Replace PROJECT_GID with a real project GID from your workspace.)
PROJECT_GID = os.environ.get("ASANA_PROJECT_GID", "")
if PROJECT_GID:
    project_info = ttuaasuti.fetch_with_retries(
        client.projects_api.get_project,
        PROJECT_GID,
        {"opt_fields": "name,archived"},
    )
    _LOG.info("Project name: %s  archived: %s", project_info.get("name"), project_info.get("archived"))
else:
    _LOG.warning("Set ASANA_PROJECT_GID to see a live demo.")

# %% [markdown]
# ## 3. Projects API
#
# **Key concepts:**
#
# - A **Workspace** is the top-level container (your organisation).
# - A **Project** groups related tasks; it can be *active* or *archived*.
# - Projects are identified by an opaque string **GID** (global identifier).
#
# **Relevant endpoint:**
#
# ```
# GET /projects/{project_gid}
#     opt_fields: name, archived, created_at, ...
# ```

# %%
if PROJECT_GID:
    project = ttuaasuti.fetch_with_retries(
        client.projects_api.get_project,
        PROJECT_GID,
        {"opt_fields": "name,archived,created_at"},
    )
    print(f"Project : {project.get('name')}")
    print(f"Archived: {project.get('archived')}")
    print(f"Created : {project.get('created_at')}")
else:
    print("(Set ASANA_PROJECT_GID to see live output.)")

# %% [markdown]
# ## 4. Tasks API
#
# **Key concepts:**
#
# - A **Task** is the basic unit of work in Asana.
# - Tasks have an *assignee*, a *completion status*, and timestamps.
# - The `fetch_tasks` helper:
#   1. Calls `GET /projects/{project_gid}/tasks` for each project.
#   2. Parses and normalises the response into a Pandas DataFrame.
#   3. Filters tasks by creation-date range.
#
# **Returned columns:**
# `task_id`, `name`, `assignee`, `created_at`, `completed_at`,
# `completed`, `project_id`, `project_status`, `task_status`

# %%
import datetime

today = datetime.datetime.now(tz=datetime.timezone.utc)
thirty_days_ago = today - datetime.timedelta(days=30)

if PROJECT_GID:
    tasks_df = ttuaasuti.fetch_tasks(
        client,
        project_ids=[PROJECT_GID],
        start_date=thirty_days_ago.isoformat(),
        end_date=today.isoformat(),
    )
    _LOG.info("Fetched %d tasks.", len(tasks_df))
    print(tasks_df[["task_id", "name", "assignee", "task_status"]].head())
else:
    print("(Set ASANA_PROJECT_GID to see live output.)")
    import pandas as pd
    # Show the expected schema as a placeholder.
    tasks_df = pd.DataFrame(
        columns=["task_id", "name", "assignee", "task_status"]
    )
    print(tasks_df)

# %% [markdown]
# ## 5. Stories (Comments) API
#
# **Key concepts:**
#
# - Asana calls comments **stories** (resource type `comment_added`).
# - Each story records the author, creation time, and text.
# - The `fetch_comments` helper:
#   1. Calls `GET /tasks/{task_gid}/stories` for each task ID.
#   2. Filters to `resource_subtype == "comment_added"`.
#   3. Returns a consolidated DataFrame.
#
# **Returned columns:** `task_id`, `text`, `author`, `created_at`

# %%
if PROJECT_GID and not tasks_df.empty:
    sample_ids = tasks_df["task_id"].head(3).tolist()
    comments_df = ttuaasuti.fetch_comments(client, task_ids=sample_ids)
    _LOG.info("Fetched %d comments for %d tasks.", len(comments_df), len(sample_ids))
    print(comments_df[["task_id", "author", "text"]].head())
else:
    print("(Set ASANA_PROJECT_GID to see live output.)")

# %% [markdown]
# ## 6. Analytics Helpers
#
# Two high-level functions aggregate raw data into user-activity statistics.
#
# ### `get_user_activity_stats(tasks_df)`
#
# Groups by `assignee` and counts:
# - `tasks_opened` — total tasks in the DataFrame
# - `tasks_closed` — tasks with a non-null `completed_at`
#
# ### `get_comments_stats(client, tasks_df)`
#
# Fetches comments for all tasks and counts comments per `author`.

# %%
if PROJECT_GID and not tasks_df.empty:
    activity = ttuaasuti.get_user_activity_stats(tasks_df)
    print("User activity stats:")
    print(activity.to_string(index=False))
else:
    print("(Set ASANA_PROJECT_GID to see live output.)")

# %%
if PROJECT_GID and not tasks_df.empty:
    comment_stats = ttuaasuti.get_comments_stats(client, tasks_df)
    print("Comment stats:")
    print(comment_stats.to_string(index=False))
else:
    print("(Set ASANA_PROJECT_GID to see live output.)")

# %% [markdown]
# ## 7. Summary
#
# | Layer | Function | Description |
# |---|---|---|
# | Authentication | `AsanaClient(token)` | Configure SDK once |
# | Retry | `fetch_with_retries(fn, *args)` | Auto-retry on 429/5xx |
# | Tasks | `fetch_tasks(client, ids, start, end)` | Fetch + filter tasks |
# | Comments | `fetch_comments(client, task_ids)` | Fetch all comments |
# | Analytics | `get_user_activity_stats(df)` | Tasks opened/closed per user |
# | Analytics | `get_comments_stats(client, df)` | Comments per author |
