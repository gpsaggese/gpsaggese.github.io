"""
Utility functions for the generalized linear models tutorial (L07_04).

Import as:

import msml610.tutorials.L07_prob_programming.L07_04_generalized_linear_models_utils as mtlppl0glmu
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Cell 1.2/2.2: Bike rental example
# #############################################################################


def plot_data_and_model(bikes: pd.DataFrame, idata: Any, mean_line: Any) -> None:
    """
    Plot the bike-rental data against a fitted model's mean and HDI bands.

    :param bikes: bike-rental data with `temperature` and `rented` columns
    :param idata: inference data with a `posterior_predictive["y_pred"]`
        group
    :param mean_line: the model's mean prediction, aligned with
        `bikes.temperature`
    """
    # Generate a vector with the temperatures and a bit of jitter.
    temperatures = np.random.normal(bikes.temperature.values, 0.01)
    # Sort in increasing order.
    idx = np.argsort(temperatures)
    # Sample the temperature intervals.
    x = np.linspace(temperatures.min(), temperatures.max(), 15)
    # Compute the quantiles, flattening over chain and draw.
    y_pred_q = idata.posterior_predictive["y_pred"].quantile(
        [0.03, 0.97, 0.25, 0.75], dim=["chain", "draw"]
    )

    from scipy.interpolate import PchipInterpolator

    y_hat_bounds = [
        PchipInterpolator(temperatures[idx], y_pred_q[i][idx])(x)
        for i in range(4)
    ]
    # Plot the data set.
    plt.plot(bikes.temperature, bikes.rented, "C2.", zorder=-3)
    # Plot the mean.
    plt.plot(bikes.temperature[idx], mean_line[idx], c="C0")
    # Plot the quantiles.
    lb, ub = y_hat_bounds[0], y_hat_bounds[1]
    plt.fill_between(x, lb, ub, color="C1", alpha=0.2)
    lb, ub = y_hat_bounds[2], y_hat_bounds[3]
    plt.fill_between(x, lb, ub, color="C1", alpha=0.2)


# #############################################################################
# Cell 6.1: Synthetic multi-feature data
# #############################################################################


def scatter_plot(x: np.ndarray, y: np.ndarray) -> None:
    """
    Plot `y` against each column of `x`, plus the columns against each other.

    Uses a 1xN horizontal layout, one panel per relationship, instead of a 2x2
    grid.

    :param x: independent variables, one column per feature
    :param y: dependent variable
    """
    n_features = x.shape[1]
    # One panel per feature (y vs x_i), plus one panel for x_1 vs x_2.
    # `squeeze=False` guarantees a 2D array of Axes regardless of
    # `n_features`, so a single row can always be indexed.
    _, axes_grid = plt.subplots(
        1, n_features + 1, squeeze=False, figsize=(5 * (n_features + 1), 4)
    )
    axes = axes_grid[0]
    for idx, x_i in enumerate(x.T):
        axes[idx].scatter(x_i, y)
        axes[idx].set_xlabel(f"x_{idx + 1}")
        axes[idx].set_ylabel("y", rotation=0)
    # Plot x_2 vs x_1 in the last panel.
    axes[-1].scatter(x[:, 0], x[:, 1])
    axes[-1].set_xlabel("x_1")
    axes[-1].set_ylabel("x_2", rotation=0)
