"""
Utility functions for Asana-based workflows.

Provides a thin wrapper around the Asana Python client to simplify
authentication, task retrieval, comment fetching, and user-activity
statistics.

Import as:

import tutorials.tutorial_asana.asana_utils as ttuaasuti
"""

import logging
import time
from typing import Any, Callable, List

import asana
import pandas as pd

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)


# #############################################################################
# Client
# #############################################################################


class AsanaClient:
    def __init__(self, access_token: str) -> None:
        """
        Initialize the Asana client using a personal access token.

        :param access_token: a valid Asana personal access token
        """
        hdbg.dassert_isinstance(access_token, str)
        configuration = asana.Configuration()
        configuration.access_token = access_token
        self.api_client = asana.ApiClient(configuration)
        self.tasks_api = asana.TasksApi(self.api_client)
        self.projects_api = asana.ProjectsApi(self.api_client)
        self.comments_api = asana.StoriesApi(self.api_client)


# #############################################################################
# Retry helpers
# #############################################################################


def fetch_with_retries(
    func: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    delay: int = 2,
    **kwargs: Any,
) -> Any:
    """
    Retry an API call on transient errors (rate limits or server errors).

    :param func: callable to invoke
    :param args: positional arguments forwarded to *func*
    :param retries: maximum number of attempts
    :param delay: base delay in seconds between retries
    :param kwargs: keyword arguments forwarded to *func*
    :return: return value of *func*
    """
    hdbg.dassert_lte(1, retries)
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except asana.rest.ApiException as e:
            if e.status == 429:
                # Rate-limit: honour the Retry-After header when present.
                retry_after = (
                    int(e.body.get("retry_after", delay)) if e.body else delay
                )
                _LOG.warning(
                    "Rate limit exceeded (attempt %d). Retrying in %ds...",
                    attempt + 1,
                    retry_after,
                )
                time.sleep(retry_after)
            elif e.status >= 500:
                _LOG.warning(
                    "Server error %s (attempt %d). Retrying...",
                    e.reason,
                    attempt + 1,
                )
                time.sleep(delay)
            else:
                _LOG.error(
                    "API error (status=%d): %s  details=%s",
                    e.status,
                    e.reason,
                    e.body,
                )
                raise
    raise RuntimeError(f"Max retries reached for {func.__name__}")


# #############################################################################
# Task fetching
# #############################################################################


def fetch_tasks(
    client: AsanaClient,
    project_ids: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch tasks from one or more projects and filter by creation date range.

    :param client: initialised AsanaClient instance
    :param project_ids: list of Asana project GIDs
    :param start_date: inclusive start date in ISO-8601 format (``YYYY-MM-DD``)
    :param end_date: exclusive end date in ISO-8601 format (``YYYY-MM-DD``)
    :return: consolidated DataFrame with columns
        ``task_id``, ``name``, ``assignee``, ``created_at``, ``completed_at``,
        ``completed``, ``project_id``, ``project_status``, ``task_status``
    """
    hdbg.dassert_isinstance(project_ids, list)
    hdbg.dassert_lte(1, len(project_ids))
    all_tasks = []
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)
    for project_id in project_ids:
        try:
            # Determine whether the project is archived (active / completed).
            project_info = fetch_with_retries(
                client.projects_api.get_project,
                project_id,
                {"opt_fields": "archived"},
            )
            project_status = (
                "Completed" if project_info.get("archived") else "Active"
            )
            tasks_response = fetch_with_retries(
                client.tasks_api.get_tasks_for_project,
                project_id,
                {
                    "opt_fields": (
                        "gid,name,assignee.name,completed,completed_at,created_at"
                    )
                },
            )
            df = pd.DataFrame(tasks_response["data"])
            if df.empty:
                continue
            df.rename(columns={"gid": "task_id"}, inplace=True)
            df["created_at"] = pd.to_datetime(
                df["created_at"], errors="coerce", utc=True
            )
            df["completed_at"] = pd.to_datetime(
                df["completed_at"], errors="coerce", utc=True
            )
            df["assignee"] = df["assignee"].apply(
                lambda x: x["name"] if isinstance(x, dict) else None
            )
            df["project_id"] = project_id
            df["project_status"] = project_status
            df["task_status"] = df["completed"].apply(
                lambda x: "Completed" if x else "Incomplete"
            )
            # Filter by creation date range.
            mask = (df["created_at"] >= start_dt) & (df["created_at"] < end_dt)
            df = df[mask]
            all_tasks.append(df)
        except asana.rest.ApiException as e:
            _LOG.warning(
                "Failed to fetch tasks for project %s: %s (status=%d)",
                project_id,
                e.reason,
                e.status,
            )
    if all_tasks:
        return pd.concat(all_tasks, ignore_index=True)
    return pd.DataFrame(
        columns=[
            "task_id",
            "name",
            "assignee",
            "created_at",
            "completed_at",
            "completed",
            "project_id",
            "project_status",
            "task_status",
        ]
    )


# #############################################################################
# Comment fetching
# #############################################################################


def fetch_comments(
    client: AsanaClient,
    task_ids: List[str],
) -> pd.DataFrame:
    """
    Fetch all comments (stories of type ``comment_added``) for a list of tasks.

    :param client: initialised AsanaClient instance
    :param task_ids: list of task GIDs
    :return: DataFrame with columns
        ``task_id``, ``text``, ``author``, ``created_at``
    """
    hdbg.dassert_isinstance(task_ids, list)
    dfs = []
    for task_id in task_ids:
        try:
            stories = fetch_with_retries(
                client.comments_api.get_stories_for_task,
                task_id,
                {
                    "opt_fields": (
                        "text,created_at,created_by.name,resource_subtype"
                    )
                },
            )
            comments = [
                {
                    "task_id": task_id,
                    "text": s.get("text", ""),
                    "author": s.get("created_by", {}).get("name"),
                    "created_at": s.get("created_at"),
                }
                for s in stories
                if s.get("resource_subtype") == "comment_added"
            ]
            if comments:
                dfs.append(pd.DataFrame(comments))
        except asana.rest.ApiException as e:
            _LOG.warning(
                "Failed to fetch comments for task %s: %s (status=%d)",
                task_id,
                e.reason,
                e.status,
            )
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["task_id", "text", "author", "created_at"])


# #############################################################################
# Analytics
# #############################################################################


def get_user_activity_stats(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate task-open and task-close counts per assignee.

    :param tasks_df: DataFrame returned by :func:`fetch_tasks`
    :return: DataFrame with columns
        ``assignee``, ``tasks_opened``, ``tasks_closed``
    """
    hdbg.dassert_isinstance(tasks_df, pd.DataFrame)
    return (
        tasks_df.groupby("assignee")
        .agg(
            tasks_opened=("task_id", "count"),
            tasks_closed=("completed_at", lambda x: x.notnull().sum()),
        )
        .reset_index()
    )


def get_comments_stats(
    client: AsanaClient,
    tasks_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate comment counts per author across all tasks in *tasks_df*.

    :param client: initialised AsanaClient instance
    :param tasks_df: DataFrame returned by :func:`fetch_tasks`
    :return: DataFrame with columns ``author``, ``comments_count``
    """
    hdbg.dassert_isinstance(tasks_df, pd.DataFrame)
    task_ids = tasks_df["task_id"].tolist()
    all_comments = fetch_comments(client, task_ids)
    if all_comments.empty:
        return pd.DataFrame(columns=["author", "comments_count"])
    return (
        all_comments.groupby("author").size().reset_index(name="comments_count")
    )
