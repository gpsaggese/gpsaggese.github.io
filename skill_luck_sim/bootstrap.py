"""
Bootstrap confidence intervals on config scores and their coverage of the
true Ybar.

Two resampling schemes:
- `task`: resample tasks with replacement, keeping all seeds of each task.
  Tasks are the cluster, so seed outcomes on the same task stay together
- `task_seed`: resample tasks as above, then resample seeds with replacement
  inside each drawn task (two-stage bootstrap)

Import as:

import skill_luck_sim.bootstrap as slsiboot
"""

import logging

import numpy as np
import pandas as pd

import skill_luck_sim.generate as slsigene

_LOG = logging.getLogger(__name__)

METHODS = ["task", "task_seed"]


# #############################################################################
# Bootstrap
# #############################################################################


def bootstrap_scores(
    y: np.ndarray, n_boot: int, method: str, rng: np.random.Generator
) -> np.ndarray:
    """
    Compute bootstrap replicates of one config's score.

    :param y: outcomes of one config, shape (N, S)
    :param n_boot: number of bootstrap replicates
    :param method: `task` or `task_seed`
    :param rng: random generator
    :return: array of shape (n_boot,)
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}', expected one of {METHODS}")
    n_tasks, n_seeds = y.shape
    task_idx = rng.integers(0, n_tasks, size=(n_boot, n_tasks))
    if method == "task":
        # Resampling whole tasks only needs the per-task means.
        task_means = y.mean(axis=1)
        replicates = task_means[task_idx].mean(axis=1)
    else:
        # Draw seeds inside each resampled task, independently per draw.
        seed_idx = rng.integers(0, n_seeds, size=(n_boot, n_tasks, n_seeds))
        replicates = y[task_idx[:, :, None], seed_idx].mean(axis=(1, 2))
    return replicates


def compute_percentile_ci(
    y: np.ndarray,
    n_boot: int,
    method: str,
    rng: np.random.Generator,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Compute a percentile bootstrap confidence interval for one config's
    score.

    :param y: outcomes of one config, shape (N, S)
    :param n_boot: number of bootstrap replicates
    :param method: `task` or `task_seed`
    :param rng: random generator
    :param alpha: one minus the confidence level
    :return: lower and upper bound
    """
    replicates = bootstrap_scores(y, n_boot, method, rng)
    lo, hi = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def compute_cis(
    y: np.ndarray,
    n_boot: int,
    method: str,
    rng: np.random.Generator,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute a confidence interval for every config.

    :param y: outcomes, shape (K, N, S)
    :param n_boot: number of bootstrap replicates
    :param method: `task` or `task_seed`
    :param rng: random generator
    :param alpha: one minus the confidence level
    :return: one row per config with columns `config`, `score`, `lo`, `hi`
    """
    rows = []
    for k in range(y.shape[0]):
        lo, hi = compute_percentile_ci(y[k], n_boot, method, rng, alpha=alpha)
        rows.append({"config": k, "score": y[k].mean(), "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    return df


# #############################################################################
# Coverage
# #############################################################################


def run_coverage(
    params: slsigene.SimParams,
    n_reps: int,
    n_boot: int,
    seed: int,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Measure how often the bootstrap intervals contain the true Ybar.

    Each replication draws a new world and builds one interval per config
    and method. Coverage is pooled over configs and replications.

    :param params: simulation parameters
    :param n_reps: number of replications
    :param n_boot: number of bootstrap replicates per interval
    :param seed: seed of the whole batch
    :param alpha: one minus the confidence level
    :return: one row per method with columns `method`, `coverage`,
        `coverage_se`, `mean_width`, `n_intervals`
    """
    children = np.random.SeedSequence(seed).spawn(n_reps)
    covered: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    widths: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    for child in children:
        rng = np.random.default_rng(child)
        world = slsigene.draw_world(params, rng)
        for method in METHODS:
            cis = compute_cis(world.y, n_boot, method, rng, alpha=alpha)
            inside = (cis["lo"] <= world.ybar) & (world.ybar <= cis["hi"])
            covered[method].append(inside.to_numpy())
            widths[method].append((cis["hi"] - cis["lo"]).to_numpy())
    # Pool the intervals of all configs and replications.
    rows = []
    for method in METHODS:
        hits = np.concatenate(covered[method])
        coverage = hits.mean()
        rows.append(
            {
                "method": method,
                "coverage": coverage,
                "coverage_se": np.sqrt(coverage * (1 - coverage) / len(hits)),
                "mean_width": np.concatenate(widths[method]).mean(),
                "n_intervals": len(hits),
            }
        )
    df = pd.DataFrame(rows)
    return df
