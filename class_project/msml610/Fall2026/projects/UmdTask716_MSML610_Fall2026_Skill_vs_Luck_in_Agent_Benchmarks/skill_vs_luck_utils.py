"""
Helper functions for the Skill vs Luck project.

Used by the 2 notebooks and by `fetch_swebench_verified.py`:
- Download and parse the SWE-bench Verified per-task results
- Load the saved data
- Bootstrap intervals for each score
- Bootstrap intervals for each rank, and the chance of coming out first
- Split-half rank correlation and a 2PL item response theory model
- Paired tests between 2 agents
- How many tasks are needed to resolve a gap
- Simulated benchmarks to check the tools

Import as:

import skill_vs_luck_utils as utils
"""

import concurrent.futures
import datetime
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer
import numpyro.infer.autoguide
import numpyro.optim
import pandas as pd
import requests
import scipy.special
import scipy.stats
import tqdm
import yaml

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)

# Run JAX on the CPU, without probing for accelerators that are not there.
jax.config.update("jax_platforms", "cpu")

# #############################################################################
# Constants
# #############################################################################


# Commit of `SWE-bench/experiments` the data was built from, pinned so that a
# rebuild gives the same files.
SWEBENCH_EXPERIMENTS_SHA = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
# Repo with the per-task results of every submission.
_EXPERIMENTS_REPO = "SWE-bench/experiments"
# Hugging Face dataset with the 500 SWE-bench Verified tasks.
_HF_DATASET = "SWE-bench/SWE-bench_Verified"
# Files written by `save_swebench_dataset()`.
OUTCOMES_FILE = "swebench_verified_outcomes.csv"
SUBMISSIONS_FILE = "swebench_verified_submissions.csv"
TASKS_FILE = "swebench_verified_tasks.csv"
COSTS_FILE = "swebench_verified_costs.csv"
API_CALLS_FILE = "swebench_verified_api_calls.csv"
PROVENANCE_FILE = "provenance.json"


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Connect the notebook logger to this module's logger.

    :param notebook_log: logger of the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Download helpers
# #############################################################################


def _download(url: str) -> Tuple[int, bytes]:
    """
    Download a URL.

    A 404 is returned instead of raised, since some submissions keep their
    results in a different file.

    :param url: URL to download
    :return: status code (200 or 404) and content (empty for a 404)
    """
    _LOG.debug(hprint.to_str("url"))
    response = requests.get(url, timeout=60)
    if response.status_code != 404:
        # Any other error is a real problem.
        response.raise_for_status()
    content = response.content if response.status_code == 200 else b""
    return response.status_code, content


def _get_raw_url(sha: str, folder: str, file_path: str) -> str:
    """
    Get the URL of a file in a submission folder.

    :param sha: commit of `SWE-bench/experiments`
    :param folder: submission folder, e.g., `20260217_mini-v2.0.0_gpt-5-mini`
    :param file_path: file inside the folder, e.g., `results/results.json`
    :return: URL on `raw.githubusercontent.com`
    """
    url = (
        f"https://raw.githubusercontent.com/{_EXPERIMENTS_REPO}/{sha}/"
        f"evaluation/verified/{folder}/{file_path}"
    )
    return url


def _fetch_task_table() -> pd.DataFrame:
    """
    Download the list of SWE-bench Verified tasks.

    :return: one row per task, indexed by `instance_id`, with columns `repo`,
        `difficulty`, `created_at`, and `version`
    """
    # The API returns at most 100 rows per call, so read it page by page.
    rows: List[Dict[str, Any]] = []
    num_rows_total = 1
    offset = 0
    while offset < num_rows_total:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={_HF_DATASET}&config=default&split=test"
            f"&offset={offset}&length=100"
        )
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        page = response.json()
        num_rows_total = page["num_rows_total"]
        rows.extend(row["row"] for row in page["rows"])
        offset += 100
    # Keep the columns we use.
    tasks = pd.DataFrame(rows)
    tasks = tasks[["instance_id", "repo", "difficulty", "created_at", "version"]]
    tasks["created_at"] = tasks["created_at"].str[:10]
    hdbg.dassert_eq(
        len(tasks), num_rows_total, "Some pages of the task list are missing"
    )
    hdbg.dassert_no_duplicates(
        tasks["instance_id"], "Task ids must identify a single task"
    )
    tasks = tasks.sort_values("instance_id").set_index("instance_id")
    return tasks


def _list_submission_folders(sha: str) -> List[str]:
    """
    List the SWE-bench Verified submission folders.

    :param sha: commit of `SWE-bench/experiments`
    :return: sorted folder names
    """
    url = (
        f"https://api.github.com/repos/{_EXPERIMENTS_REPO}/contents/"
        f"evaluation/verified?ref={sha}"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    folders = sorted(
        entry["name"] for entry in response.json() if entry["type"] == "dir"
    )
    _LOG.debug("Found '%s' submission folders", len(folders))
    return folders


def _fetch_submission(folder: str, sha: str) -> Dict[str, Any]:
    """
    Download the results and the metadata of 1 submission.

    Older submissions use `results/results.json`, the mini-SWE-agent ones use
    `per_instance_details.json`.

    :param folder: submission folder
    :param sha: commit of `SWE-bench/experiments`
    :return: dict with `results`, `per_instance`, and `metadata`, each `{}`
        when the file is missing
    """
    submission: Dict[str, Any] = {"results": {}, "per_instance": {}}
    # Try the older format first.
    status, content = _download(
        _get_raw_url(sha, folder, "results/results.json")
    )
    if status == 200:
        submission["results"] = json.loads(content)
    else:
        status, content = _download(
            _get_raw_url(sha, folder, "per_instance_details.json")
        )
        if status == 200:
            submission["per_instance"] = json.loads(content)
    # A few early submissions have no metadata.
    status, content = _download(_get_raw_url(sha, folder, "metadata.yaml"))
    submission["metadata"] = yaml.safe_load(content) if status == 200 else {}
    return submission


# #############################################################################
# Parsing helpers
# #############################################################################


def _get_date_from_folder(folder: str) -> str:
    """
    Get the submission date from the folder name.

    :param folder: e.g., `20250807_mini-v1.7.0_gpt-5-mini`
    :return: e.g., `2025-08-07`
    """
    # Folder names start with `YYYYMMDD_`.
    match = re.match(r"^(\d{4})(\d{2})(\d{2})_", folder)
    match = hdbg.dassert_re_match(
        match, "Folder '%s' has no date prefix", folder
    )
    date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return date


def parse_results_json(
    results: Dict[str, Any], task_ids: List[str]
) -> pd.Series:
    """
    Turn a `results.json` into a row of 0/1 outcomes.

    :param results: parsed `results.json`, e.g., `{"resolved": ["task_b"]}`
    :param task_ids: ids of all the tasks
    :return: 1 for solved tasks and 0 otherwise, indexed by `task_ids`
    """
    resolved = results.get("resolved", [])
    hdbg.dassert_is_subset(
        resolved, task_ids, "Solved tasks must belong to the benchmark"
    )
    outcomes = pd.Series(0, index=task_ids, dtype="int8")
    outcomes.loc[list(resolved)] = 1
    return outcomes


def parse_per_instance_details(
    details: Dict[str, Dict[str, Any]], task_ids: List[str]
) -> pd.DataFrame:
    """
    Turn a `per_instance_details.json` into per-task records.

    :param details: parsed file, e.g.,
        `{"task_a": {"cost": 0.0157, "api_calls": 12, "resolved": True}}`
    :param task_ids: ids of all the tasks
    :return: one row per task with `resolved` (0/1), `cost`, and `api_calls`
    """
    hdbg.dassert_set_eq(
        details.keys(), task_ids, "The file must cover every benchmark task"
    )
    records = pd.DataFrame.from_dict(details, orient="index")
    records = records.reindex(task_ids)
    records["resolved"] = records["resolved"].astype(bool).astype("int8")
    records = records[["resolved", "cost", "api_calls"]]
    return records


def _to_text(value: Any) -> str:
    """
    Convert a metadata value to a string.

    Some fields are a list in one submission and a single value in another.

    :param value: e.g., `None`, `"OpenAI"`, `["gpt-5-mini"]`, or `1`
    :return: e.g., `""`, `"OpenAI"`, `"gpt-5-mini"`, or `"1"`
    """
    if value is None:
        text = ""
    elif isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in value)
    else:
        text = str(value)
    return text


def parse_metadata(metadata: Dict[str, Any], folder: str) -> Dict[str, Any]:
    """
    Get the fields we use from a `metadata.yaml`.

    :param metadata: parsed `metadata.yaml`, or `{}` if missing
    :param folder: submission folder, also used as the name when missing
    :return: dict with `submission`, `name`, `date`, `agent`, `model`, `org`,
        `attempts`, and `reported_score` (in %, NaN if missing)
    """
    info = metadata.get("info") or {}
    tags = metadata.get("tags") or {}
    system = tags.get("system") or {}
    reported_score = info.get("resolved")
    fields = {
        "submission": folder,
        "name": _to_text(info.get("name")) or folder,
        "date": _get_date_from_folder(folder),
        "agent": _to_text(tags.get("agent")),
        "model": _to_text(tags.get("model")),
        "org": _to_text(tags.get("org")),
        "attempts": _to_text(system.get("attempts")),
        "reported_score": (
            np.nan if reported_score is None else float(reported_score)
        ),
    }
    return fields


# #############################################################################
# Dataset construction
# #############################################################################


def build_swebench_dataset(
    sha: str, *, num_threads: int = 8
) -> Dict[str, pd.DataFrame]:
    """
    Download all the SWE-bench Verified submissions and build the dataset.

    A submission is dropped, with the reason saved, if it has no per-task
    results or if its per-task score is more than 1 point off its reported
    score.

    :param sha: commit of `SWE-bench/experiments`
    :param num_threads: number of parallel downloads
    :return: dict with:
        - `outcomes`: agents x tasks matrix of 0/1
        - `submissions`: metadata of every folder, with `excluded_reason`
        - `tasks`: task metadata
        - `costs`, `api_calls`: agents x tasks matrices, for the submissions
          that report them
    """
    _LOG.debug(hprint.to_str("sha num_threads"))
    tasks = _fetch_task_table()
    task_ids = tasks.index.tolist()
    folders = _list_submission_folders(sha)
    # Download the submissions in parallel.
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        downloads = list(
            tqdm.tqdm(
                executor.map(
                    lambda folder: _fetch_submission(folder, sha), folders
                ),
                total=len(folders),
                desc="Downloading submissions",
                ascii=True,
            )
        )
    # Parse each submission.
    outcome_rows: Dict[str, pd.Series] = {}
    cost_rows: Dict[str, pd.Series] = {}
    api_call_rows: Dict[str, pd.Series] = {}
    submission_rows: List[Dict[str, Any]] = []
    for folder, submission in zip(folders, downloads):
        row = parse_metadata(submission["metadata"], folder)
        row["source"] = ""
        row["excluded_reason"] = ""
        if submission["results"]:
            row["source"] = "results.json"
            outcomes = parse_results_json(submission["results"], task_ids)
        elif submission["per_instance"]:
            row["source"] = "per_instance_details.json"
            records = parse_per_instance_details(
                submission["per_instance"], task_ids
            )
            outcomes = records["resolved"]
            cost_rows[folder] = records["cost"]
            api_call_rows[folder] = records["api_calls"]
        else:
            row["excluded_reason"] = "no per-task results"
            submission_rows.append(row)
            continue
        # Check the per-task score against the reported one.
        row["score"] = 100 * outcomes.mean()
        gap = abs(row["score"] - row["reported_score"])
        if gap > 1:
            row["excluded_reason"] = (
                f"per-task score {row['score']:.1f}% disagrees with the "
                f"reported {row['reported_score']:.1f}%"
            )
        else:
            outcome_rows[folder] = outcomes
        submission_rows.append(row)
    # Build the tables.
    outcomes = pd.DataFrame(outcome_rows).T
    outcomes.index.name = "submission"
    submissions = pd.DataFrame(submission_rows).set_index("submission")
    # Costs only for the kept submissions, rounded to keep the file small.
    kept = [folder for folder in cost_rows if folder in outcome_rows]
    costs = pd.DataFrame({folder: cost_rows[folder] for folder in kept}).T
    costs = costs.round(4)
    costs.index.name = "submission"
    api_calls = pd.DataFrame(
        {folder: api_call_rows[folder] for folder in kept}
    ).T
    api_calls = api_calls.astype(int)
    api_calls.index.name = "submission"
    dataset = {
        "outcomes": outcomes,
        "submissions": submissions,
        "tasks": tasks,
        "costs": costs,
        "api_calls": api_calls,
    }
    _LOG.info("Kept '%s' of '%s' submissions", len(outcomes), len(submissions))
    return dataset


def save_swebench_dataset(
    dataset: Dict[str, pd.DataFrame], dst_dir: str, sha: str
) -> None:
    """
    Save the dataset as CSV files, plus `provenance.json` with the sources.

    :param dataset: output of `build_swebench_dataset()`
    :param dst_dir: output directory
    :param sha: commit of `SWE-bench/experiments` used
    """
    hio.create_dir(dst_dir, incremental=True)
    dataset["outcomes"].to_csv(os.path.join(dst_dir, OUTCOMES_FILE))
    dataset["submissions"].to_csv(os.path.join(dst_dir, SUBMISSIONS_FILE))
    dataset["tasks"].to_csv(os.path.join(dst_dir, TASKS_FILE))
    dataset["costs"].to_csv(os.path.join(dst_dir, COSTS_FILE))
    dataset["api_calls"].to_csv(os.path.join(dst_dir, API_CALLS_FILE))
    # Save where the data came from.
    submissions = dataset["submissions"]
    excluded = submissions[submissions["excluded_reason"] != ""]
    provenance = {
        "experiments_repo": f"https://github.com/{_EXPERIMENTS_REPO}",
        "experiments_sha": sha,
        "tasks_dataset": f"https://huggingface.co/datasets/{_HF_DATASET}",
        "built_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "num_tasks": len(dataset["tasks"]),
        "num_submission_folders": len(submissions),
        "num_kept_submissions": len(dataset["outcomes"]),
        "excluded_submissions": excluded["excluded_reason"].to_dict(),
    }
    hio.to_json(os.path.join(dst_dir, PROVENANCE_FILE), provenance)
    _LOG.info("Saved the dataset in '%s'", dst_dir)


def load_swebench_dataset(
    data_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset saved by `save_swebench_dataset()`.

    :param data_dir: directory with the CSV files
    :return: outcome matrix, submission metadata, and task metadata
    """
    hdbg.dassert_dir_exists(data_dir, "Run `fetch_swebench_verified.py` first")
    outcomes = pd.read_csv(os.path.join(data_dir, OUTCOMES_FILE), index_col=0)
    outcomes = outcomes.astype("int8")
    submissions = pd.read_csv(
        os.path.join(data_dir, SUBMISSIONS_FILE),
        index_col=0,
        dtype={"attempts": str, "date": str},
    )
    # Empty text cells come back as NaN, so turn them back into "".
    text_columns = ["name", "agent", "model", "org", "attempts", "source"]
    text_columns += ["excluded_reason"]
    submissions[text_columns] = submissions[text_columns].fillna("")
    tasks = pd.read_csv(os.path.join(data_dir, TASKS_FILE), index_col=0)
    # The 3 tables must match.
    hdbg.dassert_set_eq(
        outcomes.columns,
        tasks.index,
        "Outcomes and tasks must cover the same tasks",
    )
    hdbg.dassert_is_subset(
        outcomes.index,
        submissions.index,
        "Every kept submission must have metadata",
    )
    return outcomes, submissions, tasks


# #############################################################################
# Leaderboard
# #############################################################################


def compute_leaderboard(
    outcomes: pd.DataFrame, submissions: pd.DataFrame
) -> pd.DataFrame:
    """
    Rank the agents by score, like the public leaderboard.

    :param outcomes: agents x tasks matrix of 0/1
    :param submissions: submission metadata
    :return: one row per agent, best first, with `rank` (ties share a rank),
        `label` (a unique display name), `score`, `n_resolved`, `name`,
        `date`, `agent`, and `attempts`
    """
    leaderboard = submissions.loc[
        outcomes.index, ["name", "date", "agent", "attempts"]
    ]
    leaderboard = leaderboard.copy()
    leaderboard["n_resolved"] = outcomes.sum(axis=1).astype(int)
    leaderboard["score"] = outcomes.mean(axis=1)
    # Add the date when the same name shows up twice.
    labels = leaderboard["name"].copy()
    duplicated = labels.duplicated(keep=False)
    labels[duplicated] = (
        labels[duplicated] + " (" + leaderboard.loc[duplicated, "date"] + ")"
    )
    # A few also share the date (e.g., the same system with 1 and with several
    # attempts), so use the folder name for those.
    duplicated = labels.duplicated(keep=False)
    labels[duplicated] = (
        leaderboard.loc[duplicated, "name"]
        + " ("
        + labels.index[duplicated]
        + ")"
    )
    leaderboard["label"] = labels
    hdbg.dassert_no_duplicates(
        leaderboard["label"], "Labels must tell the agents apart in plots"
    )
    leaderboard = leaderboard.sort_values(
        ["score", "date"], ascending=[False, True]
    )
    leaderboard["rank"] = (
        leaderboard["score"].rank(ascending=False, method="min").astype(int)
    )
    leaderboard = leaderboard[
        [
            "rank",
            "label",
            "score",
            "n_resolved",
            "name",
            "date",
            "agent",
            "attempts",
        ]
    ]
    return leaderboard


# #############################################################################
# Uncertainty of a single score
# #############################################################################


def _get_bootstrap_weights(n_tasks: int, n_boot: int, seed: int) -> np.ndarray:
    """
    Draw how many times each task shows up in each bootstrap sample.

    Resampling tasks with replacement is the same as a multinomial count per
    task, so every agent can use the same bootstrap samples.

    :param n_tasks: number of tasks
    :param n_boot: number of bootstrap samples
    :param seed: random seed
    :return: `n_boot` x `n_tasks` matrix of counts
    """
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n_tasks, np.full(n_tasks, 1 / n_tasks), size=n_boot
    )
    return weights


def compute_score_cis(
    outcomes: pd.DataFrame,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    r"""
    Compute a bootstrap interval for each agent's score.

    The interval shows how much the score could change with a different
    sample of tasks.

    :param outcomes: agents x tasks matrix of 0/1
    :param n_boot: number of bootstrap samples
    :param alpha: the interval covers `1 - alpha`
    :param seed: random seed
    :return: one row per agent with `score`, `se_binomial`
        ($\sqrt{p (1 - p) / N}$), `se_boot`, `ci_low`, and `ci_high`
    """
    _LOG.debug(hprint.to_str("n_boot alpha seed"))
    n_tasks = outcomes.shape[1]
    weights = _get_bootstrap_weights(n_tasks, n_boot, seed)
    # Each row of `boot_scores` has the scores of all agents in 1 sample.
    boot_scores = weights @ outcomes.to_numpy(dtype=float).T / n_tasks
    scores = outcomes.mean(axis=1)
    cis = pd.DataFrame(
        {
            "score": scores,
            "se_binomial": np.sqrt(scores * (1 - scores) / n_tasks),
            "se_boot": boot_scores.std(axis=0, ddof=1),
            "ci_low": np.quantile(boot_scores, alpha / 2, axis=0),
            "ci_high": np.quantile(boot_scores, 1 - alpha / 2, axis=0),
        },
        index=outcomes.index,
    )
    return cis


# #############################################################################
# Comparing 2 agents
# #############################################################################


def _compare_pair_with_weights(
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    """
    Compare 2 agents using bootstrap weights computed beforehand.

    :param outcomes_a: 0/1 outcomes of agent A
    :param outcomes_b: 0/1 outcomes of agent B on the same tasks
    :param weights: output of `_get_bootstrap_weights()`
    :param alpha: significance level
    :return: same as `compare_pair()`
    """
    n_tasks = len(outcomes_a)
    diff = outcomes_a.astype(float) - outcomes_b.astype(float)
    # Only tasks where the 2 agents disagree say anything about the gap.
    only_a = int(np.sum((outcomes_a == 1) & (outcomes_b == 0)))
    only_b = int(np.sum((outcomes_a == 0) & (outcomes_b == 1)))
    n_discordant = only_a + only_b
    # Exact McNemar test: do the disagreements split 50/50?
    p_value = 1.0
    if n_discordant > 0:
        p_value = scipy.stats.binomtest(only_a, n_discordant, 0.5).pvalue
    # Paired bootstrap: resample tasks, keeping both outcomes of a task.
    boot_gaps = weights @ diff / n_tasks
    comparison = {
        "gap": float(diff.mean()),
        "ci_low": float(np.quantile(boot_gaps, alpha / 2)),
        "ci_high": float(np.quantile(boot_gaps, 1 - alpha / 2)),
        "only_a": only_a,
        "only_b": only_b,
        "sd_diff": float(diff.std(ddof=1)),
        "p_value": float(p_value),
    }
    return comparison


def compare_pair(
    outcomes_a: pd.Series,
    outcomes_b: pd.Series,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Compare 2 agents task by task.

    This is more powerful than checking if 2 intervals overlap, since a lot
    of the noise comes from which tasks are easy and that cancels out.

    :param outcomes_a: 0/1 outcomes of agent A, indexed by task
    :param outcomes_b: 0/1 outcomes of agent B, on the same tasks
    :param n_boot: number of bootstrap samples
    :param alpha: significance level
    :param seed: random seed
    :return: dict with:
        - `gap`: score of A minus score of B
        - `ci_low`, `ci_high`: paired bootstrap interval for `gap`
        - `only_a`, `only_b`: tasks solved by only 1 of the 2
        - `sd_diff`: std of the per-task difference
        - `p_value`: exact McNemar test
    """
    hdbg.dassert(
        outcomes_a.index.equals(outcomes_b.index),
        "The 2 agents must be evaluated on the same tasks",
    )
    weights = _get_bootstrap_weights(len(outcomes_a), n_boot, seed)
    comparison = _compare_pair_with_weights(
        outcomes_a.to_numpy(), outcomes_b.to_numpy(), weights, alpha
    )
    return comparison


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """
    Adjust p-values for testing many pairs (Holm method).

    :param p_values: e.g., `[0.01, 0.04, 0.03]`
    :return: adjusted p-values in the same order, e.g., `[0.03, 0.06, 0.06]`
    """
    p_values = np.asarray(p_values, dtype=float)
    n_tests = len(p_values)
    order = np.argsort(p_values)
    # Multiply the k-th smallest by (n - k), then make the result
    # non-decreasing.
    scaled = p_values[order] * (n_tests - np.arange(n_tests))
    scaled = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(n_tests)
    adjusted[order] = scaled
    return adjusted


def compare_adjacent_ranks(
    outcomes: pd.DataFrame,
    leaderboard: pd.DataFrame,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compare each agent with the next one on the leaderboard.

    :param outcomes: agents x tasks matrix of 0/1
    :param leaderboard: output of `compute_leaderboard()`
    :param n_boot: number of bootstrap samples
    :param alpha: significance level, after the Holm adjustment
    :param seed: random seed
    :return: one row per pair (A is the better agent) with the
        `compare_pair()` fields, `rank_upper`, `upper`, `lower`, `p_holm`, and
        `distinguishable`
    """
    _LOG.debug(hprint.to_str("n_boot alpha seed"))
    weights = _get_bootstrap_weights(outcomes.shape[1], n_boot, seed)
    rows = []
    for upper, lower in zip(leaderboard.index[:-1], leaderboard.index[1:]):
        row = _compare_pair_with_weights(
            outcomes.loc[upper].to_numpy(),
            outcomes.loc[lower].to_numpy(),
            weights,
            alpha,
        )
        row["rank_upper"] = leaderboard.loc[upper, "rank"]
        row["upper"] = leaderboard.loc[upper, "label"]
        row["lower"] = leaderboard.loc[lower, "label"]
        rows.append(row)
    adjacent = pd.DataFrame(rows)
    adjacent["p_holm"] = holm_adjust(adjacent["p_value"])
    adjacent["distinguishable"] = adjacent["p_holm"] < alpha
    adjacent = adjacent[
        [
            "rank_upper",
            "upper",
            "lower",
            "gap",
            "ci_low",
            "ci_high",
            "only_a",
            "only_b",
            "sd_diff",
            "p_value",
            "p_holm",
            "distinguishable",
        ]
    ]
    return adjacent


def compare_to_leader(
    outcomes: pd.DataFrame,
    leaderboard: pd.DataFrame,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compare every agent with the top agent.

    :param outcomes: agents x tasks matrix of 0/1
    :param leaderboard: output of `compute_leaderboard()`
    :param n_boot: number of bootstrap samples
    :param alpha: significance level, after the Holm adjustment
    :param seed: random seed
    :return: one row per agent (A is the leader) with the `compare_pair()`
        fields, `rank`, `label`, `p_holm`, and `tied_with_leader`
    """
    weights = _get_bootstrap_weights(outcomes.shape[1], n_boot, seed)
    leader = leaderboard.index[0]
    rows = []
    for other in leaderboard.index[1:]:
        row = _compare_pair_with_weights(
            outcomes.loc[leader].to_numpy(),
            outcomes.loc[other].to_numpy(),
            weights,
            alpha,
        )
        row["rank"] = leaderboard.loc[other, "rank"]
        row["label"] = leaderboard.loc[other, "label"]
        rows.append(row)
    versus_leader = pd.DataFrame(rows)
    versus_leader["p_holm"] = holm_adjust(versus_leader["p_value"])
    versus_leader["tied_with_leader"] = versus_leader["p_holm"] >= alpha
    versus_leader = versus_leader[
        [
            "rank",
            "label",
            "gap",
            "ci_low",
            "ci_high",
            "only_a",
            "only_b",
            "p_value",
            "p_holm",
            "tied_with_leader",
        ]
    ]
    return versus_leader


# #############################################################################
# Benchmark size
# #############################################################################


def compute_min_tasks(
    sd_diff: float,
    gap: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    r"""
    Compute how many tasks are needed to detect a score gap.

    Uses $N = (z_{1 - \alpha / 2} + z_{power})^2 \sigma_d^2 / \Delta^2$. The
    spec's version, $2 z^2 \sigma^2_{pair} / \Delta^2$, is the same with
    `power=0.5` for 2 independent agents; using the observed $\sigma_d$ also
    accounts for the agents being correlated.

    :param sd_diff: std of the per-task difference, e.g., from `compare_pair()`
    :param gap: gap to detect, e.g., `0.02` for 2 points
    :param alpha: significance level
    :param power: chance of detecting a real gap
    :return: number of tasks
    """
    hdbg.dassert_lt(0, gap, "The gap must be positive")
    z_alpha = scipy.stats.norm.ppf(1 - alpha / 2)
    z_power = scipy.stats.norm.ppf(power)
    n_tasks = (z_alpha + z_power) ** 2 * sd_diff**2 / gap**2
    return float(n_tasks)


def compute_min_detectable_gap(
    sd_diff: float,
    n_tasks: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    r"""
    Compute the smallest gap a benchmark of a given size can detect.

    Inverse of `compute_min_tasks()`:
    $\Delta = (z_{1 - \alpha / 2} + z_{power}) \sigma_d / \sqrt{N}$.

    :param sd_diff: std of the per-task difference
    :param n_tasks: number of tasks
    :param alpha: significance level
    :param power: chance of detecting a real gap
    :return: smallest gap, e.g., `0.054` for 5.4 points
    """
    hdbg.dassert_lt(0, n_tasks, "The benchmark must have tasks")
    z_alpha = scipy.stats.norm.ppf(1 - alpha / 2)
    z_power = scipy.stats.norm.ppf(power)
    gap = (z_alpha + z_power) * sd_diff / np.sqrt(n_tasks)
    return float(gap)


# #############################################################################
# Ranking under uncertainty
# #############################################################################


def _rank_scores(scores: np.ndarray) -> np.ndarray:
    """
    Rank the agents in each row of scores, best first.

    Tied agents share the average of the ranks they span, e.g., 2 agents tied
    for first both get rank 1.5.

    :param scores: samples x agents matrix of scores
    :return: samples x agents matrix of ranks, starting at 1
    """
    ranks = scipy.stats.rankdata(-scores, method="average", axis=1)
    return ranks


def _get_first_place_shares(scores: np.ndarray) -> np.ndarray:
    """
    Split the first place of each row among the agents tied for it.

    E.g., if 2 agents share the top score, each gets 0.5 of the first place.

    :param scores: samples x agents matrix of scores
    :return: samples x agents matrix of shares, each row sums to 1
    """
    is_top = scores == scores.max(axis=1, keepdims=True)
    shares = is_top / is_top.sum(axis=1, keepdims=True)
    return shares


def compute_rank_intervals(
    outcomes: pd.DataFrame,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compute how much the rank of each agent could change with other tasks.

    Each bootstrap sample resamples the tasks with replacement, recomputes all
    the scores, and ranks the agents. The samples are the same as in
    `compute_score_cis()` for the same `seed`. Ties are split: tied agents
    share the average of their ranks, and an `n`-way tie for first gives each
    agent `1 / n` of the first place.

    :param outcomes: agents x tasks matrix of 0/1
    :param n_boot: number of bootstrap samples
    :param alpha: the interval covers `1 - alpha`
    :param seed: random seed
    :return: one row per agent, best observed score first, with `score`,
        `rank` (observed, ties share the best rank like the leaderboard),
        `median_rank`, `rank_low`, `rank_high`, and `p_first` (chance of
        coming out first)
    """
    _LOG.debug(hprint.to_str("n_boot alpha seed"))
    hdbg.dassert_lt(1, outcomes.shape[0], "Ranking needs at least 2 agents")
    n_tasks = outcomes.shape[1]
    weights = _get_bootstrap_weights(n_tasks, n_boot, seed)
    # Each row of `boot_scores` has the scores of all agents in 1 sample.
    boot_scores = weights @ outcomes.to_numpy(dtype=float).T / n_tasks
    boot_ranks = _rank_scores(boot_scores)
    # Summarize the ranks of each agent across the samples.
    scores = outcomes.mean(axis=1)
    rank_intervals = pd.DataFrame(
        {
            "score": scores,
            "rank": scores.rank(ascending=False, method="min").astype(int),
            "median_rank": np.median(boot_ranks, axis=0),
            "rank_low": np.quantile(boot_ranks, alpha / 2, axis=0),
            "rank_high": np.quantile(boot_ranks, 1 - alpha / 2, axis=0),
            "p_first": _get_first_place_shares(boot_scores).mean(axis=0),
        },
        index=outcomes.index,
    )
    rank_intervals = rank_intervals.sort_values("rank", kind="stable")
    return rank_intervals


# #############################################################################
# Persistence across halves of the tasks
# #############################################################################


def _get_random_halves(n_tasks: int, n_splits: int, seed: int) -> np.ndarray:
    """
    Split the tasks into 2 random halves, several times.

    :param n_tasks: number of tasks, at least 2
    :param n_splits: number of random splits
    :param seed: random seed
    :return: `n_splits` x `n_tasks` boolean matrix, True for the tasks in the
        first half; each row has `n_tasks // 2` True values
    """
    hdbg.dassert_lte(2, n_tasks, "Splitting needs at least 2 tasks")
    rng = np.random.default_rng(seed)
    in_first = np.zeros((n_splits, n_tasks), dtype=bool)
    for split in range(n_splits):
        first = rng.permutation(n_tasks)[: n_tasks // 2]
        in_first[split, first] = True
    return in_first


def _get_half_scores(
    outcomes: np.ndarray, in_first: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score every agent on each half of every split.

    :param outcomes: agents x tasks matrix of 0/1
    :param in_first: output of `_get_random_halves()`
    :return: 2 `n_splits` x agents matrices, with the scores on the first and
        on the second half
    """
    first = in_first.astype(float)
    second = 1 - first
    scores_first = first @ outcomes.T / first.sum(axis=1, keepdims=True)
    scores_second = second @ outcomes.T / second.sum(axis=1, keepdims=True)
    return scores_first, scores_second


def compute_split_half_correlation(
    outcomes: pd.DataFrame,
    *,
    n_splits: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Check whether the ranking on half of the tasks holds on the other half.

    Each split puts half of the tasks, at random, in each half, ranks the
    agents on both halves, and correlates the 2 rankings. Low correlations
    mean the ranking reflects which tasks were drawn more than the agents.

    :param outcomes: agents x tasks matrix of 0/1
    :param n_splits: number of random splits
    :param seed: random seed
    :return: one row per split with `spearman` and `kendall`
    """
    _LOG.debug(hprint.to_str("n_splits seed"))
    hdbg.dassert_lt(2, outcomes.shape[0], "Correlations need at least 3 agents")
    in_first = _get_random_halves(outcomes.shape[1], n_splits, seed)
    scores_first, scores_second = _get_half_scores(
        outcomes.to_numpy(dtype=float), in_first
    )
    # Correlate the 2 rankings of each split.
    correlations = pd.DataFrame(
        {
            "spearman": [
                scipy.stats.spearmanr(a, b).statistic
                for a, b in zip(scores_first, scores_second)
            ],
            "kendall": [
                scipy.stats.kendalltau(a, b).statistic
                for a, b in zip(scores_first, scores_second)
            ],
        }
    )
    return correlations


def compute_held_out_drop(
    outcomes: pd.DataFrame,
    *,
    n_splits: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Pick the leader on half of the tasks and score it on the other half.

    The leader of the first half was picked for its high score there, so its
    score on the second half is a fair estimate of its true score, and the
    difference is the winner's curse. A tie for first is split: the tied
    agents are averaged.

    :param outcomes: agents x tasks matrix of 0/1
    :param n_splits: number of random splits
    :param seed: random seed
    :return: one row per split with `score_selected` (the leader's score on
        the first half), `score_held_out` (its score on the second half),
        `drop` (the difference), and `same_leader` (share of the first-half
        leaders that also lead the second half)
    """
    _LOG.debug(hprint.to_str("n_splits seed"))
    in_first = _get_random_halves(outcomes.shape[1], n_splits, seed)
    scores_first, scores_second = _get_half_scores(
        outcomes.to_numpy(dtype=float), in_first
    )
    # Weight each leader of the first half, splitting ties.
    leader_shares = _get_first_place_shares(scores_first)
    score_selected = scores_first.max(axis=1)
    score_held_out = (leader_shares * scores_second).sum(axis=1)
    # Check whether the leaders of the first half also lead the second.
    is_second_leader = scores_second == scores_second.max(axis=1, keepdims=True)
    same_leader = (leader_shares * is_second_leader).sum(axis=1)
    held_out = pd.DataFrame(
        {
            "score_selected": score_selected,
            "score_held_out": score_held_out,
            "drop": score_selected - score_held_out,
            "same_leader": same_leader,
        }
    )
    return held_out


# #############################################################################
# Item response theory
# #############################################################################


def _irt_2pl_model(outcomes: jnp.ndarray) -> None:
    r"""
    Define the Bayesian 2PL model for an agents x tasks matrix of 0/1.

    Agent $a$ solves task $i$ with probability
    $\sigma(\alpha_i (\theta_a - b_i))$, with ability
    $\theta_a \sim N(0, 1)$, difficulty $b_i \sim N(0, 3^2)$, and
    discrimination $\log \alpha_i \sim N(0, 0.5^2)$. The prior on the
    abilities fixes the scale.

    :param outcomes: agents x tasks matrix of 0/1
    """
    n_agents, n_tasks = outcomes.shape
    ability = numpyro.sample("ability", dist.Normal(0, 1).expand([n_agents]))
    difficulty = numpyro.sample(
        "difficulty", dist.Normal(0, 3).expand([n_tasks])
    )
    log_discrimination = numpyro.sample(
        "log_discrimination", dist.Normal(0, 0.5).expand([n_tasks])
    )
    logits = jnp.exp(log_discrimination)[None, :] * (
        ability[:, None] - difficulty[None, :]
    )
    numpyro.sample("solved", dist.Bernoulli(logits=logits), obs=outcomes)


def fit_irt_2pl(
    outcomes: pd.DataFrame,
    *,
    method: str = "svi",
    n_steps: int = 5000,
    n_samples: int = 1000,
    seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Fit the Bayesian 2PL model to an outcome matrix.

    :param outcomes: agents x tasks matrix of 0/1
    :param method: "svi" (variational, fast) or "nuts" (MCMC with 1 chain,
        slow but exact)
    :param n_steps: optimization steps, for "svi"
    :param n_samples: posterior samples (and warmup steps for "nuts")
    :param seed: random seed
    :return: "agents" with `ability` and `ability_sd`, one row per agent, and
        "tasks" with `difficulty`, `difficulty_sd`, `discrimination`, and
        `discrimination_sd`, one row per task (posterior means and stds)
    """
    _LOG.debug(hprint.to_str("method n_steps n_samples seed"))
    hdbg.dassert_in(method, ["svi", "nuts"])
    data = jnp.asarray(outcomes.to_numpy(dtype=np.float32))
    rng_fit, rng_sample = jax.random.split(jax.random.PRNGKey(seed))
    # Draw posterior samples of every parameter.
    if method == "svi":
        guide = numpyro.infer.autoguide.AutoNormal(_irt_2pl_model)
        svi = numpyro.infer.SVI(
            _irt_2pl_model,
            guide,
            numpyro.optim.Adam(0.01),
            numpyro.infer.Trace_ELBO(),
        )
        result = svi.run(rng_fit, n_steps, data, progress_bar=False)
        samples = guide.sample_posterior(
            rng_sample, result.params, sample_shape=(n_samples,)
        )
    else:
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(_irt_2pl_model),
            num_warmup=n_samples,
            num_samples=n_samples,
            num_chains=1,
            progress_bar=False,
        )
        mcmc.run(rng_fit, data)
        samples = mcmc.get_samples()
    # Summarize each parameter by its posterior mean and std.
    ability = np.asarray(samples["ability"])
    difficulty = np.asarray(samples["difficulty"])
    discrimination = np.exp(np.asarray(samples["log_discrimination"]))
    agents = pd.DataFrame(
        {"ability": ability.mean(axis=0), "ability_sd": ability.std(axis=0)},
        index=outcomes.index,
    )
    tasks = pd.DataFrame(
        {
            "difficulty": difficulty.mean(axis=0),
            "difficulty_sd": difficulty.std(axis=0),
            "discrimination": discrimination.mean(axis=0),
            "discrimination_sd": discrimination.std(axis=0),
        },
        index=outcomes.columns,
    )
    fit = {"agents": agents, "tasks": tasks}
    return fit


def compute_task_information(
    abilities: Sequence[float], tasks: pd.DataFrame
) -> pd.Series:
    r"""
    Compute how much each task tells about agents of the given abilities.

    The Fisher information of task $i$ at ability $\theta$ is
    $\alpha_i^2 p (1 - p)$, with $p = \sigma(\alpha_i (\theta - b_i))$. It is
    largest for a task that the agents solve half of the time, and near 0 for
    a task they all solve or all fail.

    :param abilities: abilities of the agents to tell apart
    :param tasks: one row per task with `difficulty` and `discrimination`,
        e.g., from `fit_irt_2pl()`
    :return: information of each task, averaged over the abilities
    """
    abilities = np.asarray(abilities, dtype=float)
    discrimination = tasks["discrimination"].to_numpy()
    probabilities = scipy.special.expit(
        discrimination[None, :]
        * (abilities[:, None] - tasks["difficulty"].to_numpy()[None, :])
    )
    information = (
        discrimination[None, :] ** 2 * probabilities * (1 - probabilities)
    )
    information = pd.Series(information.mean(axis=0), index=tasks.index)
    return information


def compute_effective_size(information: pd.Series) -> float:
    r"""
    Compute how many equally informative tasks a benchmark is worth.

    This is $(\sum_i I_i)^2 / \sum_i I_i^2$: $N$ when all $N$ tasks carry the
    same information, and 1 when a single task carries all of it.

    :param information: information of each task, e.g., from
        `compute_task_information()`
    :return: effective number of tasks
    """
    hdbg.dassert_lt(0, information.sum(), "Some task must carry information")
    effective_size = information.sum() ** 2 / (information**2).sum()
    return float(effective_size)


def count_informative_tasks(outcomes: pd.DataFrame) -> int:
    """
    Count the tasks that not all agents solve or all agents fail.

    A task with the same outcome for every agent cannot tell them apart.

    :param outcomes: agents x tasks matrix of 0/1
    :return: number of tasks with at least 1 success and 1 failure
    """
    n_solved = outcomes.sum(axis=0)
    n_informative = int(((n_solved > 0) & (n_solved < len(outcomes))).sum())
    return n_informative


def compute_selection_effect(
    fit: Dict[str, pd.DataFrame],
    *,
    top_k: int = 20,
    n_trials: int = 300,
    n_splits: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Check how picking the top agents on their score changes their split-half
    correlation.

    The 2 half scores of an agent add up to its total, so among agents picked
    for a high total, a good half means a weak other half. Each trial simulates
    a benchmark from the fit, where the true abilities are known, and compares
    the top agents picked on the simulated score with the top agents picked on
    the true ability. The fitted abilities stand in for the true ones.

    :param fit: output of `fit_irt_2pl()`
    :param top_k: number of top agents
    :param n_trials: number of simulated benchmarks
    :param n_splits: random splits of each benchmark
    :param seed: random seed
    :return: one row per way of picking, `score` and `ability`, with `kendall`
        (median Kendall tau between the rankings on the 2 halves) and
        `negative` (share of the splits with a negative tau)
    """
    _LOG.debug(hprint.to_str("top_k n_trials n_splits seed"))
    abilities = fit["agents"]["ability"]
    tasks = fit["tasks"]
    hdbg.dassert_lt(2, top_k, "Correlations need at least 3 agents")
    probabilities = scipy.special.expit(
        tasks["discrimination"].to_numpy()[None, :]
        * (
            abilities.to_numpy()[:, None]
            - tasks["difficulty"].to_numpy()[None, :]
        )
    )
    top_on_ability = abilities.sort_values(ascending=False).index[:top_k]
    rng = np.random.default_rng(seed)
    kendalls = {"score": [], "ability": []}
    for trial in range(n_trials):
        solved = rng.random(probabilities.shape) < probabilities
        outcomes = pd.DataFrame(
            solved.astype("int8"), index=abilities.index, columns=tasks.index
        )
        scores = outcomes.mean(axis=1).sort_values(
            ascending=False, kind="stable"
        )
        groups = {"score": scores.index[:top_k], "ability": top_on_ability}
        for name, group in groups.items():
            correlations = compute_split_half_correlation(
                outcomes.loc[group], n_splits=n_splits, seed=seed + trial
            )
            kendalls[name].append(correlations["kendall"])
    # Summarize the splits of all the benchmarks for each way of picking.
    all_kendalls = {name: pd.concat(values) for name, values in kendalls.items()}
    selection_effect = pd.DataFrame(
        {
            "kendall": {k: v.median() for k, v in all_kendalls.items()},
            "negative": {k: (v < 0).mean() for k, v in all_kendalls.items()},
        }
    )
    return selection_effect


# #############################################################################
# Simulated benchmarks with known abilities
# #############################################################################


def simulate_outcomes(
    abilities: Sequence[float],
    n_tasks: int,
    *,
    difficulty_sd: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    r"""
    Simulate a benchmark where the true abilities are known.

    Rasch model: agent $a$ solves task $i$ with probability
    $\sigma(\theta_a - b_i)$, with difficulties $b_i \sim N(0, s^2)$. All
    agents get the same tasks, like on a real benchmark.

    :param abilities: ability of each agent (logit scale)
    :param n_tasks: number of tasks
    :param difficulty_sd: std $s$ of the task difficulties
    :param seed: random seed
    :return: agents x tasks matrix of 0/1, with rows `agent_0`, ... and
        columns `task_0`, ...
    """
    outcomes, _ = _simulate_outcomes_with_probabilities(
        abilities, n_tasks, difficulty_sd=difficulty_sd, seed=seed
    )
    return outcomes


def _simulate_outcomes_with_probabilities(
    abilities: Sequence[float],
    n_tasks: int,
    *,
    difficulty_sd: float = 1.0,
    seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Simulate a Rasch benchmark and return the probabilities behind it.

    :param abilities: ability of each agent (logit scale)
    :param n_tasks: number of tasks
    :param difficulty_sd: std of the task difficulties
    :param seed: random seed
    :return: the outcomes of `simulate_outcomes()`, and the agents x tasks
        matrix of the solve probabilities they were drawn from
    """
    # Draw the task difficulties, then the 0/1 outcomes, from the same seed.
    rng = np.random.default_rng(seed)
    difficulties = rng.normal(0, difficulty_sd, size=n_tasks)
    probabilities = scipy.special.expit(
        np.asarray(abilities, dtype=float)[:, None] - difficulties[None, :]
    )
    solved = rng.random(probabilities.shape) < probabilities
    outcomes = pd.DataFrame(
        solved.astype("int8"),
        index=[f"agent_{i}" for i in range(len(abilities))],
        columns=[f"task_{i}" for i in range(n_tasks)],
    )
    return outcomes, probabilities


def compute_expected_scores(
    abilities: Sequence[float], *, difficulty_sd: float = 1.0
) -> pd.Series:
    r"""
    Compute the true score of each agent over all possible tasks.

    This is $E_b[\sigma(\theta - b)]$ with $b \sim N(0, s^2)$, computed with
    Gauss-Hermite quadrature.

    :param abilities: ability of each agent (logit scale)
    :param difficulty_sd: std $s$ of the task difficulties
    :return: expected score of each agent
    """
    # Hermite nodes integrate against a standard normal, up to $\sqrt{2 \pi}$.
    nodes, node_weights = np.polynomial.hermite_e.hermegauss(80)
    node_weights = node_weights / np.sqrt(2 * np.pi)
    abilities = np.asarray(abilities, dtype=float)
    probabilities = scipy.special.expit(
        abilities[:, None] - difficulty_sd * nodes[None, :]
    )
    expected = pd.Series(
        probabilities @ node_weights,
        index=[f"agent_{i}" for i in range(len(abilities))],
    )
    return expected


def check_ci_coverage(
    abilities: Sequence[float],
    n_tasks: int,
    *,
    n_trials: int = 200,
    n_boot: int = 500,
    alpha: float = 0.05,
    difficulty_sd: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Check how often the bootstrap interval contains the true score.

    A 95% interval should contain it in about 95% of the simulated
    benchmarks.

    :param abilities: ability of each agent
    :param n_tasks: tasks per benchmark
    :param n_trials: number of simulated benchmarks
    :param n_boot: bootstrap samples per benchmark
    :param alpha: the intervals target `1 - alpha`
    :param difficulty_sd: std of the task difficulties
    :param seed: seed of the first benchmark
    :return: one row per agent with `true_score`, `coverage`, and `mean_width`
    """
    true_scores = compute_expected_scores(abilities, difficulty_sd=difficulty_sd)
    covered = []
    widths = []
    # Update the progress bar rarely, to keep the notebook output short.
    for trial in tqdm.tqdm(
        range(n_trials),
        desc="Simulating benchmarks",
        mininterval=10,
        ascii=True,
    ):
        outcomes = simulate_outcomes(
            abilities, n_tasks, difficulty_sd=difficulty_sd, seed=seed + trial
        )
        cis = compute_score_cis(
            outcomes, n_boot=n_boot, alpha=alpha, seed=seed + trial
        )
        covered.append(
            (cis["ci_low"] <= true_scores) & (true_scores <= cis["ci_high"])
        )
        widths.append(cis["ci_high"] - cis["ci_low"])
    coverage = pd.DataFrame(
        {
            "true_score": true_scores,
            "coverage": pd.concat(covered, axis=1).mean(axis=1),
            "mean_width": pd.concat(widths, axis=1).mean(axis=1),
        }
    )
    return coverage


def compare_test_power(
    ability_a: float,
    ability_b: float,
    n_tasks: int,
    *,
    n_trials: int = 500,
    alpha: float = 0.05,
    difficulty_sd: float = 1.0,
    seed: int = 0,
) -> pd.Series:
    """
    Compare the paired test with the "intervals don't overlap" rule.

    :param ability_a: true ability of agent A
    :param ability_b: true ability of agent B
    :param n_tasks: tasks per benchmark
    :param n_trials: number of simulated benchmarks
    :param alpha: significance level
    :param difficulty_sd: std of the task difficulties
    :param seed: seed of the first benchmark
    :return: how often each rule finds a difference, for `paired_test` (the
        McNemar test from `compare_pair()`) and `ci_overlap`
    """
    z_alpha = scipy.stats.norm.ppf(1 - alpha / 2)
    paired_rejections = 0
    overlap_rejections = 0
    for trial in range(n_trials):
        outcomes = simulate_outcomes(
            [ability_a, ability_b],
            n_tasks,
            difficulty_sd=difficulty_sd,
            seed=seed + trial,
        )
        a = outcomes.iloc[0].to_numpy()
        b = outcomes.iloc[1].to_numpy()
        # Paired test on the tasks where the 2 disagree.
        only_a = int(np.sum((a == 1) & (b == 0)))
        only_b = int(np.sum((a == 0) & (b == 1)))
        if only_a + only_b > 0:
            p_value = scipy.stats.binomtest(only_a, only_a + only_b, 0.5).pvalue
            paired_rejections += p_value < alpha
        # Overlap rule with 2 separate binomial intervals.
        score_a, score_b = a.mean(), b.mean()
        half_a = z_alpha * np.sqrt(score_a * (1 - score_a) / n_tasks)
        half_b = z_alpha * np.sqrt(score_b * (1 - score_b) / n_tasks)
        overlap_rejections += abs(score_a - score_b) > half_a + half_b
    rejection_rates = pd.Series(
        {
            "paired_test": paired_rejections / n_trials,
            "ci_overlap": overlap_rejections / n_trials,
        }
    )
    return rejection_rates


def check_ranking_recovery(
    abilities: Sequence[float],
    n_tasks: int,
    *,
    n_trials: int = 200,
    difficulty_sd: float = 1.0,
    seed: int = 0,
) -> pd.Series:
    """
    Check how well the observed ranking matches the true one.

    Each trial simulates a benchmark with `simulate_outcomes()`, ranks the
    agents by score, and compares the ranking with the order of the true
    abilities.

    :param abilities: ability of each agent, all different
    :param n_tasks: tasks per benchmark
    :param n_trials: number of simulated benchmarks
    :param difficulty_sd: std of the task difficulties
    :param seed: seed of the first benchmark
    :return: averages over the trials of `p_best_first` (share of first place
        won by the truly best agent, split if tied), `spearman`, and `kendall`
        (rank correlations between the scores and the true abilities)
    """
    abilities = np.asarray(abilities, dtype=float)
    hdbg.dassert_lt(1, len(abilities), "Ranking needs at least 2 agents")
    hdbg.dassert_no_duplicates(
        abilities.tolist(), "The true ranking must have no ties"
    )
    best = int(np.argmax(abilities))
    p_best_first = []
    spearman = []
    kendall = []
    for trial in range(n_trials):
        outcomes = simulate_outcomes(
            abilities, n_tasks, difficulty_sd=difficulty_sd, seed=seed + trial
        )
        scores = outcomes.mean(axis=1).to_numpy()
        # Share of the first place that goes to the truly best agent.
        shares = _get_first_place_shares(scores[None, :])
        p_best_first.append(shares[0, best])
        # Rank correlations between the observed and the true order.
        spearman.append(scipy.stats.spearmanr(scores, abilities).statistic)
        kendall.append(scipy.stats.kendalltau(scores, abilities).statistic)
    recovery = pd.Series(
        {
            "p_best_first": np.mean(p_best_first),
            "spearman": np.mean(spearman),
            "kendall": np.mean(kendall),
        }
    )
    return recovery


def _get_expected_max_normal(n_draws: int) -> float:
    r"""
    Compute the mean of the largest of `n_draws` standard normal draws.

    This is $\int x \, K \phi(x) \Phi(x)^{K - 1} dx$, computed with
    Gauss-Hermite quadrature. It is 0 for 1 draw and about 0.56 for 2, while
    $\sqrt{2 \ln K}$ gives 0 and 1.18.

    :param n_draws: number of draws $K$
    :return: expected maximum
    """
    hdbg.dassert_lte(1, n_draws, "There must be at least 1 draw")
    # Hermite nodes integrate against a standard normal, up to $\sqrt{2 \pi}$.
    nodes, node_weights = np.polynomial.hermite_e.hermegauss(120)
    node_weights = node_weights / np.sqrt(2 * np.pi)
    density_of_max = n_draws * scipy.stats.norm.cdf(nodes) ** (n_draws - 1)
    expected_max = float(np.sum(node_weights * nodes * density_of_max))
    return expected_max


def simulate_winners_curse(
    n_variants: int,
    n_tasks: int,
    *,
    ability: float = 0.0,
    n_trials: int = 500,
    difficulty_sd: float = 1.0,
    seed: int = 0,
) -> pd.Series:
    r"""
    Measure how much picking the best of equal variants inflates its score.

    Each trial simulates `n_variants` scaffolds with the same true ability on
    the same tasks, like a lab trying several prompts. Any gap between them is
    luck, so the best observed score overstates the true one. The spec
    corrects for it with $\sigma \sqrt{2 \ln K}$, where $\sigma$ is the
    standard error of 1 score.

    Each trial also picks the best variant on a random half of the tasks and
    scores it on the other half, which is what a held-out check measures.

    :param n_variants: number of variants $K$ tried
    :param n_tasks: tasks per benchmark
    :param ability: true ability of every variant
    :param n_trials: number of simulated benchmarks
    :param difficulty_sd: std of the task difficulties
    :param seed: seed of the first benchmark
    :return: `true_score`, `inflation` (mean best score minus the true
        score), `sigma` (binomial standard error of 1 score on all tasks),
        `correction` ($\sigma \sqrt{2 \ln K}$), `correction_exact` ($\sigma$
        times the mean of the largest of $K$ standard normals),
        `sigma_given_tasks` (the noise of 1 score given the drawn tasks,
        $\sqrt{\overline{p_i (1 - p_i)} / N}$ with $p_i$ the true solve
        probabilities), `correction_given_tasks` ($\sigma_{\text{tasks}}$
        times the mean of the largest of $K$ standard normals), and
        `held_out_drop` (mean score of the best variant on the selection half
        minus its score on the other half)
    """
    hdbg.dassert_lte(1, n_variants, "There must be at least 1 variant")
    true_score = float(
        compute_expected_scores([ability], difficulty_sd=difficulty_sd).iloc[0]
    )
    best_scores = []
    held_out_drops = []
    task_sds = []
    for trial in range(n_trials):
        outcomes, probabilities = _simulate_outcomes_with_probabilities(
            [ability] * n_variants,
            n_tasks,
            difficulty_sd=difficulty_sd,
            seed=seed + trial,
        )
        outcomes = outcomes.to_numpy(dtype=float)
        best_scores.append(outcomes.mean(axis=1).max())
        # Noise of 1 score given these tasks: only the run-to-run draws vary.
        task_sds.append(
            np.sqrt(np.mean(probabilities[0] * (1 - probabilities[0])) / n_tasks)
        )
        # Pick the best variant on half of the tasks, score it on the other.
        in_first = _get_random_halves(n_tasks, 1, seed + trial)
        scores_first, scores_second = _get_half_scores(outcomes, in_first)
        shares = _get_first_place_shares(scores_first)
        held_out_drops.append(
            scores_first.max() - (shares * scores_second).sum()
        )
    sigma = np.sqrt(true_score * (1 - true_score) / n_tasks)
    sigma_given_tasks = np.mean(task_sds)
    curse = pd.Series(
        {
            "true_score": true_score,
            "inflation": np.mean(best_scores) - true_score,
            "sigma": sigma,
            "correction": sigma * np.sqrt(2 * np.log(n_variants)),
            "correction_exact": sigma * _get_expected_max_normal(n_variants),
            "sigma_given_tasks": sigma_given_tasks,
            "correction_given_tasks": sigma_given_tasks
            * _get_expected_max_normal(n_variants),
            "held_out_drop": np.mean(held_out_drops),
        }
    )
    return curse


# #############################################################################
# Plots
# #############################################################################


def plot_leaderboard_cis(
    leaderboard: pd.DataFrame,
    cis: pd.DataFrame,
    *,
    top_k: int = 25,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the top of the leaderboard with the interval of each score.

    :param leaderboard: output of `compute_leaderboard()`
    :param cis: output of `compute_score_cis()`
    :param top_k: number of agents to show
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    data = leaderboard.head(top_k).join(cis[["ci_low", "ci_high"]])
    # Put the best agent at the top.
    data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        100 * data["score"],
        data["label"],
        xerr=[
            100 * (data["score"] - data["ci_low"]),
            100 * (data["ci_high"] - data["score"]),
        ],
        fmt="o",
        color="tab:blue",
        ecolor="tab:gray",
        capsize=3,
    )
    # Shade the leader's interval to show who overlaps it.
    leader = data.iloc[-1]
    ax.axvspan(
        100 * leader["ci_low"],
        100 * leader["ci_high"],
        color="tab:orange",
        alpha=0.15,
        label="95% interval of the leader",
    )
    ax.set_xlabel("Tasks solved (%), with 95% bootstrap interval")
    ax.set_title(
        f"Top {len(data)} of the {len(cis)} agents on SWE-bench Verified"
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    plt.show()


def plot_adjacent_gaps(
    adjacent: pd.DataFrame,
    *,
    alpha: float = 0.05,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the gap between each pair of neighbors on the leaderboard.

    :param adjacent: output of `compare_adjacent_ranks()`
    :param alpha: significance level used for the colors
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=figsize)
    # 3 groups: significant after Holm, only before Holm, and not at all.
    raw_significant = adjacent["p_value"] < alpha
    groups = [
        (adjacent["distinguishable"], "tab:green", "distinguishable after Holm"),
        (
            raw_significant & ~adjacent["distinguishable"],
            "tab:orange",
            f"raw p < {alpha} only",
        ),
        (~raw_significant, "tab:gray", f"raw p >= {alpha}"),
    ]
    for mask, color, name in groups:
        subset = adjacent[mask]
        ax.errorbar(
            subset["rank_upper"],
            100 * subset["gap"],
            yerr=[
                100 * (subset["gap"] - subset["ci_low"]),
                100 * (subset["ci_high"] - subset["gap"]),
            ],
            fmt="o",
            color=color,
            ecolor=color,
            alpha=0.7,
            capsize=2,
            label=f"{name} ({len(subset)} pairs)",
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Rank of the better agent in the pair")
    ax.set_ylabel("Gap to the next agent (points)")
    ax.set_title(
        "Score gap between leaderboard neighbors, with 95% paired interval"
    )
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_min_tasks_curve(
    sd_diffs: pd.Series,
    *,
    n_benchmark_tasks: int = 500,
    alpha: float = 0.05,
    power: float = 0.8,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the tasks needed to detect a gap, against the gap size.

    Shows the median `sd_diff` of the neighbor pairs, the 25th to 75th
    percentile as a band, the benchmark size, and the smallest gap it can
    detect.

    :param sd_diffs: `sd_diff` of the neighbor pairs
    :param n_benchmark_tasks: tasks in the real benchmark
    :param alpha: significance level
    :param power: chance of detecting a real gap
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    gaps = np.linspace(0.005, 0.10, 100)
    # 1 curve for each quantile of `sd_diff`.
    curves = {}
    for quantile in (0.25, 0.5, 0.75):
        sd_diff = sd_diffs.quantile(quantile)
        curves[quantile] = [
            compute_min_tasks(sd_diff, gap, alpha=alpha, power=power)
            for gap in gaps
        ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(
        100 * gaps,
        curves[0.25],
        curves[0.75],
        color="tab:blue",
        alpha=0.2,
        label="25th to 75th percentile of the pairs",
    )
    ax.plot(
        100 * gaps, curves[0.5], color="tab:blue", label="median neighbor pair"
    )
    ax.axhline(
        n_benchmark_tasks,
        color="tab:red",
        linestyle="--",
        label=f"SWE-bench Verified ({n_benchmark_tasks} tasks)",
    )
    # Mark the smallest gap the real benchmark can detect.
    min_gap = compute_min_detectable_gap(
        sd_diffs.median(), n_benchmark_tasks, alpha=alpha, power=power
    )
    ax.axvline(100 * min_gap, color="tab:red", linestyle=":")
    ax.text(
        100 * min_gap,
        n_benchmark_tasks * 3,
        f"  smallest resolvable gap: {100 * min_gap:.1f} points",
        color="tab:red",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Score gap to resolve (points)")
    ax.set_ylabel("Tasks needed")
    ax.set_title(
        f"Tasks needed to detect a gap (alpha = {alpha}, power = {power})"
    )
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_rank_intervals(
    rank_intervals: pd.DataFrame,
    leaderboard: pd.DataFrame,
    *,
    top_k: int = 25,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the 95% bootstrap interval of the rank of the top agents.

    :param rank_intervals: output of `compute_rank_intervals()`
    :param leaderboard: output of `compute_leaderboard()`, for the labels
    :param top_k: number of agents to show
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    data = leaderboard[["label"]].head(top_k).join(rank_intervals)
    # Put the best agent at the top.
    data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        data["median_rank"],
        data["label"],
        xerr=[
            data["median_rank"] - data["rank_low"],
            data["rank_high"] - data["median_rank"],
        ],
        fmt="o",
        color="tab:blue",
        ecolor="tab:gray",
        capsize=3,
        label="median rank, with 95% interval",
    )
    ax.scatter(
        data["rank"],
        data["label"],
        marker="x",
        color="tab:red",
        zorder=3,
        label="rank on the leaderboard",
    )
    ax.set_xlabel("Rank (1 = best), over bootstrap samples of the tasks")
    ax.set_title(f"Rank intervals of the top {len(data)} agents")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def plot_first_place_chance(
    rank_intervals: pd.DataFrame,
    leaderboard: pd.DataFrame,
    *,
    top_k: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the chance of each top agent coming out first.

    :param rank_intervals: output of `compute_rank_intervals()`
    :param leaderboard: output of `compute_leaderboard()`, for the labels
    :param top_k: number of agents to show
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    data = leaderboard[["label"]].head(top_k).join(rank_intervals)
    # Put the best agent at the top.
    data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(data["label"], 100 * data["p_first"], color="tab:blue")
    # Write the chance next to each bar.
    for label, p_first in zip(data["label"], data["p_first"]):
        ax.text(100 * p_first, label, f" {100 * p_first:.1f}%", va="center")
    ax.set_xlabel("Chance of coming out first (%), ties split")
    ax.set_title(
        f"Who would win with another sample of tasks? (top {len(data)})"
    )
    fig.tight_layout()
    plt.show()


def plot_ranking_recovery(
    recovery: pd.DataFrame,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot how well the observed ranking matches the true one, by benchmark size.

    :param recovery: one row per simulation with `n_tasks`, `gap_points`,
        `p_best_first`, and `kendall`
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = [
        ("p_best_first", "Chance the truly best agent comes out first"),
        ("kendall", "Kendall tau, observed vs true ranking"),
    ]
    # 1 line per gap size in each panel.
    for ax, (column, title) in zip(axes, panels):
        for gap_points, group in recovery.groupby("gap_points"):
            ax.plot(
                group["n_tasks"],
                group[column],
                marker="o",
                label=f"{gap_points:.1f} points between neighbors",
            )
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Tasks in the benchmark")
        ax.set_title(title)
    axes[0].legend()
    fig.tight_layout()
    plt.show()


def plot_split_half_correlations(
    correlations: Dict[str, pd.DataFrame],
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the Kendall tau between the rankings on 2 halves of the tasks.

    :param correlations: outputs of `compute_split_half_correlation()`, by the
        name of the group of agents
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=figsize)
    bins = np.linspace(-0.2, 1, 61)
    # 1 histogram per group of agents.
    for name, group in correlations.items():
        median = group["kendall"].median()
        ax.hist(
            group["kendall"],
            bins=bins,
            alpha=0.6,
            label=f"{name} (median {median:.2f})",
        )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Kendall tau between the rankings on the 2 halves")
    ax.set_ylabel("Random splits")
    ax.set_title("Does the ranking on half of the tasks hold on the other half?")
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def plot_winners_curse(
    curse: pd.DataFrame,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the inflation of the best of K equal variants against K.

    :param curse: one row per simulation with `n_tasks`, `n_variants`, and the
        outputs of `simulate_winners_curse()`
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    # 1 line style per correction, in the left panel.
    corrections = {
        "correction": ("--", "sigma sqrt(2 ln K)"),
        "correction_exact": (":", "sigma E[max of K normals]"),
        "correction_given_tasks": ("-.", "sigma_tasks E[max of K normals]"),
    }
    # 1 color per benchmark size, in both panels.
    for (n_tasks, group), color in zip(
        curse.groupby("n_tasks"),
        plt.rcParams["axes.prop_cycle"].by_key()["color"],
    ):
        axes[0].plot(
            group["n_variants"],
            100 * group["inflation"],
            marker="o",
            color=color,
        )
        for column, (linestyle, _) in corrections.items():
            axes[0].plot(
                group["n_variants"],
                100 * group[column],
                linestyle=linestyle,
                color=color,
            )
        axes[1].plot(
            group["n_variants"],
            100 * group["held_out_drop"],
            marker="o",
            color=color,
            label=f"{n_tasks} tasks",
        )
    # The left legend explains the line styles and the right one the colors.
    axes[0].plot([], [], marker="o", color="gray", label="simulated")
    for linestyle, label in corrections.values():
        axes[0].plot([], [], linestyle=linestyle, color="gray", label=label)
    n_variants = sorted(curse["n_variants"].unique())
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(n_variants, labels=[str(k) for k in n_variants])
        ax.minorticks_off()
        ax.set_xlabel("Variants tried (K)")
    axes[0].set_ylabel("Points")
    axes[0].set_title("Best score minus true score")
    axes[1].set_title("Drop of the best variant on the held-out half")
    axes[0].legend(fontsize="small", loc="upper left")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def plot_difficulty_by_label(
    tasks: pd.DataFrame,
    labels: pd.Series,
    *,
    label_order: List[str],
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the fitted difficulty of the tasks against a human label.

    :param tasks: output "tasks" of `fit_irt_2pl()`
    :param labels: human label of each task, e.g., the estimated fix time
    :param label_order: labels from easiest to hardest
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    data = tasks[["difficulty"]].join(labels.rename("label"))
    groups = [
        data.loc[data["label"] == label, "difficulty"] for label in label_order
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(groups)
    # Show every task on top of the boxes.
    for position, group in enumerate(groups, start=1):
        jitter = np.random.default_rng(position).uniform(-0.15, 0.15, len(group))
        ax.scatter(position + jitter, group, s=6, alpha=0.3, color="tab:blue")
    counts = [
        f"{label} ({len(group)})" for label, group in zip(label_order, groups)
    ]
    ax.set_xticks(range(1, len(label_order) + 1), counts)
    ax.set_xlabel("Human estimate of the time to fix (tasks)")
    ax.set_ylabel("Fitted difficulty (logits)")
    ax.set_title("Fitted task difficulty against the human label")
    fig.tight_layout()
    plt.show()


def plot_task_information(
    information: Dict[str, pd.Series],
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the share of the information held by the most informative tasks.

    :param information: outputs of `compute_task_information()`, by the name
        of the group of agents
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=figsize)
    # 1 curve per group of agents, tasks from most to least informative.
    for name, info in information.items():
        share = info.sort_values(ascending=False).cumsum() / info.sum()
        effective_size = compute_effective_size(info)
        ax.plot(
            np.arange(1, len(share) + 1),
            100 * share.to_numpy(),
            label=f"{name} (effective size {effective_size:.0f} tasks)",
        )
    ax.axhline(50, color="tab:gray", linestyle=":")
    ax.set_xlabel("Most informative tasks kept")
    ax.set_ylabel("Share of the information (%)")
    ax.set_title("How many tasks carry the information about the agents?")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()
