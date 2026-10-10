"""
Score configs, pick the winner, and measure the winner's curse and the
re-evaluation fixes over many replications.

Import as:

import skill_luck_sim.estimate as slsiesti
"""

import logging

import numpy as np
import pandas as pd

import skill_luck_sim.generate as slsigene

_LOG = logging.getLogger(__name__)


# #############################################################################
# Scores and ranks
# #############################################################################


def compute_scores(y: np.ndarray) -> np.ndarray:
    """
    Compute each config's observed score, i.e. its mean over tasks and seeds.

    :param y: outcomes of shape (K, N, S)
    :return: array of shape (K,)
    """
    scores = y.mean(axis=(1, 2))
    return scores


def select_winner(scores: np.ndarray) -> int:
    """
    Pick the config with the highest observed score.

    Ties go to the lowest index. Configs are drawn in random order, so this
    does not favor any skill level.

    :param scores: observed scores, shape (K,)
    :return: index of the winner
    """
    winner = int(np.argmax(scores))
    return winner


def compute_ranks(values: np.ndarray) -> np.ndarray:
    """
    Rank values from best (1) to worst (K), averaging ranks on ties.

    E.g., [0.2, 0.5, 0.5] -> [3.0, 1.5, 1.5].

    :param values: array of shape (K,)
    :return: ranks of shape (K,)
    """
    ranks = pd.Series(values).rank(ascending=False, method="average").to_numpy()
    return ranks


# #############################################################################
# Re-evaluation of the winner
# #############################################################################


def evaluate_on_fresh_seeds(
    world: slsigene.World,
    config: int,
    n_seeds: int,
    rng: np.random.Generator,
) -> float:
    """
    Rerun one config on the same tasks with new seeds and return its score.

    This removes the seed luck that helped the config win, but not the luck
    of which tasks were in the benchmark.

    :param world: the world the config was selected in
    :param config: index of the config
    :param n_seeds: number of new seeds per task
    :param rng: random generator
    :return: mean outcome over the same tasks and the new seeds
    """
    p = slsigene.sigmoid(world.theta[config] - world.b + world.u[config])
    score = float(slsigene.draw_outcomes(p, n_seeds, rng).mean())
    return score


def evaluate_on_fresh_tasks(
    theta: float,
    params: slsigene.SimParams,
    rng: np.random.Generator,
) -> float:
    """
    Run one config on `params.n_tasks` new tasks with `params.n_seeds` seeds
    and return its score.

    New tasks come with new interactions, so this estimate does not depend on
    anything used to select the config.

    :param theta: skill of the config
    :param params: simulation parameters
    :param rng: random generator
    :return: mean outcome over the new tasks and seeds
    """
    b = slsigene.draw_difficulties(params.n_tasks, params, rng)
    u = slsigene.draw_interactions(1, params.n_tasks, params, rng)[0]
    p = slsigene.sigmoid(theta - b + u)
    score = float(slsigene.draw_outcomes(p, params.n_seeds, rng).mean())
    return score


# #############################################################################
# Replications
# #############################################################################


def run_replication(
    params: slsigene.SimParams, rng: np.random.Generator
) -> dict[str, float]:
    """
    Draw one benchmark, select the winner, and measure what went wrong.

    :param params: simulation parameters
    :param rng: random generator
    :return: metrics of this replication:
        - `curse`: observed score of the winner minus its true Ybar
        - `is_truly_best`: 1 if the winner has the highest Ybar, else 0. NaN
          when `sigma_theta == 0`, since then no config is truly best
        - `regret`: best Ybar minus the winner's Ybar
        - `rank_error`: mean absolute difference between observed and true
          rank over all configs
        - `fresh_seed_error`: fresh-seed score of the winner minus its Ybar
        - `fresh_task_error`: fresh-task score of the winner minus its Ybar
    """
    world = slsigene.draw_world(params, rng)
    scores = compute_scores(world.y)
    winner = select_winner(scores)
    # True ranking is by Ybar, which is increasing in theta.
    if params.sigma_theta == 0:
        is_truly_best = np.nan
    else:
        is_truly_best = float(winner == int(np.argmax(world.ybar)))
    rank_error = np.abs(compute_ranks(scores) - compute_ranks(world.ybar)).mean()
    # Re-evaluate the winner with the same budget as one config in the
    # original run.
    fresh_seed_score = evaluate_on_fresh_seeds(
        world, winner, params.n_seeds, rng
    )
    fresh_task_score = evaluate_on_fresh_tasks(world.theta[winner], params, rng)
    winner_ybar = world.ybar[winner]
    metrics = {
        "curse": scores[winner] - winner_ybar,
        "is_truly_best": is_truly_best,
        "regret": world.ybar.max() - winner_ybar,
        "rank_error": rank_error,
        "fresh_seed_error": fresh_seed_score - winner_ybar,
        "fresh_task_error": fresh_task_score - winner_ybar,
    }
    return metrics


def run_replications(
    params: slsigene.SimParams, n_reps: int, seed: int
) -> pd.DataFrame:
    """
    Run `n_reps` independent replications.

    Each replication gets its own generator spawned from `seed`, so a
    replication's result does not depend on the others.

    :param params: simulation parameters
    :param n_reps: number of replications
    :param seed: seed of the whole batch
    :return: one row per replication with the columns of `run_replication()`
    """
    children = np.random.SeedSequence(seed).spawn(n_reps)
    rows = [run_replication(params, np.random.default_rng(c)) for c in children]
    df = pd.DataFrame(rows)
    return df


def summarize_replications(
    df: pd.DataFrame, params: slsigene.SimParams
) -> dict[str, float]:
    """
    Average the metrics over replications and add Monte Carlo standard
    errors and the cost of re-evaluation.

    :param df: output of `run_replications()`
    :param params: parameters used to produce `df`
    :return: for each metric `m`, the keys `m` (mean) and `m_se` (Monte Carlo
        standard error), plus:
        - `extra_runs`: runs needed to re-evaluate the winner, N * S
        - `extra_cost_frac`: `extra_runs` over the original K * N * S runs
    """
    summary: dict[str, float] = {}
    for col in df.columns:
        values = df[col].dropna()
        summary[col] = values.mean() if len(values) else np.nan
        summary[f"{col}_se"] = (
            values.std(ddof=1) / np.sqrt(len(values))
            if len(values) > 1
            else np.nan
        )
    extra_runs = params.n_tasks * params.n_seeds
    summary["extra_runs"] = float(extra_runs)
    summary["extra_cost_frac"] = extra_runs / (
        params.n_configs * params.n_tasks * params.n_seeds
    )
    return summary
