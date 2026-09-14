"""
Utility functions for the evaluating models tutorial (L07_05).

Import as:

import msml610.tutorials.L07_prob_programming.L07_05_evaluating_models_utils as mtlpel0emu
"""

import logging
from typing import Any, List

import numpy as np

_LOG = logging.getLogger(__name__)


# #############################################################################
# Cell 3.2/3.3: Overfitting
# #############################################################################


def iqr(x: np.ndarray, a: int = -1) -> float:
    """
    Compute the interquartile range of `x`.

    :param x: input data
    :param a: axis to reduce over
    :return: 75th minus 25th percentile
    """
    return np.subtract(*np.percentile(x, [75, 25], axis=a))


def plot_models(x0: np.ndarray, y0: np.ndarray, ps: List[Any], ax: Any) -> None:
    """
    Plot each fitted polynomial in `ps` against the data, with its R^2.

    :param x0: x-coordinates the polynomials are evaluated against for R^2
    :param y0: y-coordinates the polynomials are evaluated against for R^2
    :param ps: fitted `np.polynomial.Polynomial` objects, one per order in
        `order = [0, 1, 5]`
    :param ax: axis to plot on
    """
    order = [0, 1, 5]
    x_n = np.linspace(x0.min(), x0.max(), 100)
    for i, p in zip(order, ps):
        # Evaluate on the raw data.
        yhat = p(x0)
        # Estimate the error between the estimates and the true values.
        ss_regression = np.sum((yhat - y0) ** 2)
        # Compute R^2.
        ybar = np.mean(y0)
        ss_total = np.sum((ybar - y0) ** 2)
        r2 = 1 - ss_regression / ss_total
        #
        ax.plot(x_n, p(x_n), label=f"order {i}, $R^2$= {r2:.3f}", lw=3)
    ax.legend(loc=2)
