"""
Utility functions to generate the figures of Lesson 11.01 "Agents: Automating
the Research Loop".

Import as:

import L11_01_agents_automating_the_research_loop_utils as l1aartlu
"""

import logging
from typing import Any, List, Optional, Tuple

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sopt
import scipy.special as sspec
import seaborn as sns

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Plot helpers
# #############################################################################


# Colors of the figures of Lessons 01.01, 02.01, 06.01, and 10.01, so that the
# figures of the lessons look alike.
_BLUE = "#3C6FB0"
_CORAL = "#C0455B"
_ORANGE = "#D98E2B"
_GREEN = "#4F9A5C"
_GRAY = "#7B8794"


def _new_figure(
    figsize: Optional[Tuple[float, float]],
    *,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> Tuple[mfigure.Figure, np.ndarray]:
    """
    Create a figure with a clean style and no top / right spines.

    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :param nrows: number of rows of subplots
    :param ncols: number of columns of subplots
    :param kwargs: additional arguments passed to `plt.subplots()`
    :return: figure and array of axes with shape (nrows, ncols)
    """
    if figsize is None:
        figsize = tuple(plt.rcParams["figure.figsize"])
    # Use a white background without the default seaborn grid, so that the
    # figure blends into the slide.
    with sns.axes_style("ticks"):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            **kwargs,
        )
    sns.despine(fig=fig)
    return fig, axes


# #############################################################################
# Cell 1.1: Agents are stochastic
# #############################################################################


def _simulate_task_success(
    rng: np.random.Generator,
    *,
    n_tasks: int,
    n_runs: int,
    mean_success: float,
    concentration: float,
) -> np.ndarray:
    """
    Simulate repeated runs of an agent on a set of tasks.

    The success probability of each task is drawn from a Beta distribution with
    mean `mean_success`, so that some tasks are easy and some are hard for the
    agent. Each run of a task succeeds independently with that probability.

    :param rng: random number generator
    :param n_tasks: number of tasks of the benchmark
    :param n_runs: number of independent runs of each task
    :param mean_success: mean success probability over the tasks
    :param concentration: sum of the two Beta parameters, where a small value
        spreads the tasks between always solved and never solved
    :return: boolean array of shape (n_tasks, n_runs)
    """
    hdbg.dassert_lt(0.0, mean_success, "The mean success must be positive")
    hdbg.dassert_lt(mean_success, 1.0, "The mean success must be below 1")
    hdbg.dassert_lt(0.0, concentration, "The concentration must be positive")
    alpha = mean_success * concentration
    beta = (1.0 - mean_success) * concentration
    task_prob = rng.beta(alpha, beta, size=n_tasks)
    success = rng.random((n_tasks, n_runs)) < task_prob[:, None]
    return success


def _pass_at_k_curves(
    success: np.ndarray, max_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the pass@k and the pass^k of an agent from its runs.

    - pass@k: share of the tasks where at least one of `k` runs succeeds
    - pass^k: share of the tasks where all the `k` runs succeed

    Both use the unbiased estimator over all the subsets of `k` of the `n` runs
    of a task, i.e., `1 - C(n - c, k) / C(n, k)` and `C(c, k) / C(n, k)` with
    `c` successful runs.

    :param success: boolean array of shape (n_tasks, n_runs)
    :param max_k: largest number of runs `k`
    :return: pass@k and pass^k for `k = 1, ..., max_k`
    """
    n_runs = success.shape[1]
    hdbg.dassert_lte(max_k, n_runs, "Need at least max_k runs per task")
    n_success = success.sum(axis=1)
    pass_at_k = np.zeros(max_k)
    pass_hat_k = np.zeros(max_k)
    for k in range(1, max_k + 1):
        total = sspec.comb(n_runs, k)
        pass_at_k[k - 1] = np.mean(
            1.0 - sspec.comb(n_runs - n_success, k) / total
        )
        pass_hat_k[k - 1] = np.mean(sspec.comb(n_success, k) / total)
    return pass_at_k, pass_hat_k


def cell1_1_plot_pass_at_k(
    *,
    n_tasks: int,
    n_runs: int,
    max_k: int,
    mean_success: float,
    concentration: float,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the pass@k and the pass^k of an agent against the number of runs.

    :param n_tasks: number of tasks of the benchmark
    :param n_runs: number of independent runs of each task
    :param max_k: largest number of runs `k`
    :param mean_success: mean success probability over the tasks, i.e., the
        pass@1
    :param concentration: sum of the two Beta parameters of the task success
        probability
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the pass@k and the pass^k against `k`
    """
    rng = np.random.default_rng(seed)
    success = _simulate_task_success(
        rng,
        n_tasks=n_tasks,
        n_runs=n_runs,
        mean_success=mean_success,
        concentration=concentration,
    )
    pass_at_k, pass_hat_k = _pass_at_k_curves(success, max_k)
    _LOG.info(
        "pass@1=%.2f pass@%d=%.2f pass^%d=%.2f",
        pass_at_k[0],
        max_k,
        pass_at_k[-1],
        max_k,
        pass_hat_k[-1],
    )
    # Plot the two curves against the number of runs.
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    ks = np.arange(1, max_k + 1)
    ax.plot(
        ks,
        pass_at_k,
        color=_BLUE,
        linewidth=2.5,
        marker="o",
        label="pass@k: at least one run succeeds",
    )
    ax.plot(
        ks,
        pass_hat_k,
        color=_CORAL,
        linewidth=2.5,
        marker="o",
        label="pass^k: every run succeeds",
    )
    ax.axhline(pass_at_k[0], color=_GRAY, linewidth=1.0, linestyle="--")
    ax.text(
        max_k,
        pass_at_k[0] + 0.02,
        "single run",
        color=_GRAY,
        fontsize=9,
        ha="right",
    )
    ax.set_xlabel("number of runs k")
    ax.set_ylabel("share of tasks")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(ks)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    return fig


# #############################################################################
# Cell 1.2: Benchmark rank noise
# #############################################################################


def _abilities_for_accuracies(
    accuracies: List[float], difficulty_std: float
) -> np.ndarray:
    """
    Find the ability of each agent that gives a target mean accuracy.

    The success probability of agent `i` on task `j` is
    `sigmoid(ability_i - difficulty_j)` with `difficulty_j ~ N(0,
    difficulty_std^2)`. The mean accuracy is the expectation over the tasks,
    computed with Gauss-Hermite quadrature.

    :param accuracies: target mean accuracy of each agent
    :param difficulty_std: standard deviation of the task difficulty
    :return: ability of each agent
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(60)
    weights = weights / weights.sum()

    def _mean_accuracy(ability: float) -> float:
        prob = sspec.expit(ability - difficulty_std * nodes)
        return float(np.sum(weights * prob))

    abilities = []
    for accuracy in accuracies:
        hdbg.dassert_lt(0.0, accuracy, "The accuracy must be positive")
        hdbg.dassert_lt(accuracy, 1.0, "The accuracy must be below 1")
        ability = sopt.brentq(
            lambda a, acc=accuracy: _mean_accuracy(a) - acc, -20.0, 20.0
        )
        abilities.append(ability)
    return np.array(abilities)


def _prob_true_best_wins(
    rng: np.random.Generator,
    *,
    accuracies: List[float],
    n_tasks: int,
    difficulty_std: float,
    n_sims: int,
) -> float:
    """
    Estimate the probability that the true best agent tops the leaderboard.

    All the agents run on the same tasks, so the shared task difficulty makes
    their outcomes positively correlated. Ties at the top are broken at random.

    :param rng: random number generator
    :param accuracies: true mean accuracy of each agent
    :param n_tasks: number of tasks of the benchmark
    :param difficulty_std: standard deviation of the task difficulty
    :param n_sims: number of simulated benchmarks
    :return: share of the benchmarks where the true best agent ranks first
    """
    abilities = _abilities_for_accuracies(accuracies, difficulty_std)
    best = int(np.argmax(accuracies))
    n_wins = 0.0
    for _ in range(n_sims):
        difficulty = rng.normal(0.0, difficulty_std, size=n_tasks)
        prob = sspec.expit(abilities[:, None] - difficulty[None, :])
        scores = (rng.random(prob.shape) < prob).sum(axis=1)
        top = np.flatnonzero(scores == scores.max())
        n_wins += float(best in top) / len(top)
    return n_wins / n_sims


def cell1_2_plot_benchmark_rank_noise(
    *,
    n_agents: int,
    base_accuracy: float,
    gaps: List[float],
    task_counts: List[int],
    difficulty_std: float,
    n_sims: int,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the probability that a benchmark ranks the true best agent first.

    The agents have true accuracies `base_accuracy, base_accuracy + gap, ...`,
    so neighboring agents differ by `gap`.

    :param n_agents: number of agents on the leaderboard
    :param base_accuracy: true accuracy of the weakest agent
    :param gaps: accuracy gaps between neighboring agents, one curve each
    :param task_counts: numbers of benchmark tasks on the x axis
    :param difficulty_std: standard deviation of the task difficulty
    :param n_sims: number of simulated benchmarks per point
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the probability of a correct winner against the
        number of tasks
    """
    rng = np.random.default_rng(seed)
    colors = [_CORAL, _ORANGE, _BLUE, _GREEN]
    hdbg.dassert_lte(len(gaps), len(colors), "Too many gaps to plot")
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    for gap, color in zip(gaps, colors):
        accuracies = [base_accuracy + i * gap for i in range(n_agents)]
        probs = [
            _prob_true_best_wins(
                rng,
                accuracies=accuracies,
                n_tasks=n_tasks,
                difficulty_std=difficulty_std,
                n_sims=n_sims,
            )
            for n_tasks in task_counts
        ]
        _LOG.info(
            "gap=%.3f: %s",
            gap,
            ", ".join(f"{n}: {p:.2f}" for n, p in zip(task_counts, probs)),
        )
        ax.plot(
            task_counts,
            probs,
            color=color,
            linewidth=2.5,
            marker="o",
            label=f"{100 * gap:g}-point gap between agents",
        )
    ax.axhline(1.0 / n_agents, color=_GRAY, linewidth=1.0, linestyle="--")
    ax.text(
        task_counts[-1],
        1.0 / n_agents + 0.02,
        "random pick",
        color=_GRAY,
        fontsize=9,
        ha="right",
    )
    ax.set_xscale("log")
    ax.set_xticks(task_counts)
    ax.set_xticklabels([str(n) for n in task_counts])
    ax.minorticks_off()
    ax.set_xlabel("number of benchmark tasks")
    ax.set_ylabel("Pr(true best agent ranks first)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    return fig
