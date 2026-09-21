"""
Utility functions for the probabilistic programming tutorial (L07_02).

Import as:

import msml610.tutorials.L07_prob_programming.L07_02_probabilistic_programming_utils as mtlppl0ppu
"""

import logging
from typing import Any, Callable, List, Tuple

import numpy as np
import preliz as pz

import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Cell 1.5: Bayesian p-value
# #############################################################################


def iqr(x: np.ndarray, a: int = -1) -> np.ndarray:
    """
    Compute the interquartile range of `x` along axis `a`.

    :param x: data to summarize
    :param a: axis to reduce over
    :return: interquartile range (75th percentile minus 25th)
    """
    return np.subtract(*np.percentile(x, [75, 25], axis=a))


# #############################################################################
# Cell 2.2: Fitting polynomial models of increasing order
# #############################################################################


def plot_models(
    ax: Any,
    x_n: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    order: List[int],
    ps: List[np.polynomial.Polynomial],
) -> None:
    """
    Plot fitted polynomials of several orders and report each one's R^2.

    :param ax: axes to plot into
    :param x_n: dense x grid used to draw each fitted curve
    :param x0: x values the R^2 is computed against
    :param y0: y values the R^2 is computed against
    :param order: polynomial order of each fitted model in `ps`
    :param ps: fitted `Polynomial` objects, one per entry in `order`
    """
    for i, p in zip(order, ps):
        # Evaluate on the raw data.
        yhat = p(x0)
        # Estimate the error between the estimates and the true values.
        ss_regression = np.sum((yhat - y0) ** 2)
        # Compute R^2.
        ybar = np.mean(y0)
        ss_total = np.sum((ybar - y0) ** 2)
        r2 = 1 - ss_regression / ss_total
        ax.plot(x_n, p(x_n), label=f"order {i}, $R^2$= {r2:.3f}", lw=3)
    ax.legend(loc=2)


# #############################################################################
# Cell 5.1: Grid approximation
# #############################################################################


def posterior_grid(
    grid_points: int, heads: int, tails: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate a coin-flip posterior by grid approximation.

    :param grid_points: number of points in the `theta` grid
    :param heads: observed heads
    :param tails: observed tails
    :return: tuple of (grid, prior, likelihood, posterior)
    """
    # The interval for the parameter is [0, 1].
    grid = np.linspace(0, 1, grid_points)
    # The prior is uniform.
    prior = np.repeat(1 / grid_points, grid_points)
    # Likelihood is Binomial with known params.
    likelihood = pz.Binomial(n=heads + tails, p=grid).pdf(heads)
    # Compute the integral of the PDF.
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, prior, likelihood, posterior


# #############################################################################
# Cell 5.3: Metropolis sampler
# #############################################################################


def metropolis(func: Callable, draws: int = 10000) -> np.ndarray:
    """
    Run a simple Metropolis sampler targeting `func`'s density.

    :param func: frozen `scipy.stats` distribution to sample from, via its
        `pdf()` method
    :param draws: number of samples to draw
    :return: trace of accepted (or repeated) samples, one per draw
    """
    # Initialize an array to store sampled values.
    trace = np.zeros(draws)
    # Start at an initial value for the chain and compute its probability.
    old_x = 0.5
    old_prob = func.pdf(old_x)
    # Generate proposal deltas from a normal distribution.
    delta = np.random.normal(0, 0.5, draws)
    # Loop through the desired number of samples.
    for i in range(draws):
        # Propose a new sample by adding the delta to the current state.
        new_x = old_x + delta[i]
        # Compute the probability of the proposed sample.
        new_prob = func.pdf(new_x)
        # Calculate acceptance ratio between proposed and current probabilities.
        acceptance = new_prob / old_prob
        # Accept or reject the new sample based on the acceptance ratio.
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace
