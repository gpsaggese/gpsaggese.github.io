"""
Utility functions for non-linear Kalman filter tutorials.

Import as:

import msml610.tutorials.L09_05_04_non_linear_kalman_filter_utils as l090504u
"""

import logging
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

_LOG = logging.getLogger(__name__)


# #############################################################################
# Plotting functions.
# #############################################################################


def plot_nonlinear_func(
    data: np.ndarray,
    f: Callable,
    *,
    out_lim: Optional[List[float]] = None,
    num_bins: int = 300,
) -> None:
    """
    Plot the input, output, and transfer function for a nonlinear transformation.

    :param data: input data samples
    :param f: the nonlinear function to apply
    :param out_lim: limits for the output axis [min, max]
    :param num_bins: number of histogram bins
    """
    ys = f(data)
    x0 = np.mean(data)
    in_std = np.std(data)
    y = f(x0)
    std = np.std(ys)
    in_lims = [x0 - in_std * 3, x0 + in_std * 3]
    if out_lim is None:
        out_lim = [y - std * 3, y + std * 3]
    # Plot output.
    h = np.histogram(ys, num_bins, density=False)
    plt.subplot(221)
    plt.plot(h[1][1:], h[0], lw=2, alpha=0.8)
    if out_lim is not None:
        plt.xlim(out_lim[0], out_lim[1])
    plt.gca().yaxis.set_ticklabels([])
    plt.title("Output")
    plt.axvline(np.mean(ys), ls="--", lw=2)
    plt.axvline(f(x0), lw=1)
    # Plot transfer function.
    plt.subplot(2, 2, 3)
    x = np.arange(in_lims[0], in_lims[1], 0.1)
    y = f(x)
    plt.plot(x, y, "k")
    isct = f(x0)
    plt.plot([x0, x0, in_lims[1]], [out_lim[1], isct, isct], color="r", lw=1)
    plt.xlim(in_lims)
    plt.ylim(out_lim)
    plt.title("f(x)")
    # Plot input.
    h = np.histogram(data, num_bins, density=True)
    plt.subplot(2, 2, 4)
    plt.plot(h[0], h[1][1:], lw=2)
    plt.gca().xaxis.set_ticklabels([])
    plt.title("Input")
    plt.tight_layout()
    plt.show()


def plot_bivariate_colormap(xs: np.ndarray, ys: np.ndarray) -> None:
    """
    Plot a bivariate colormap using kernel density estimation.

    :param xs: x-axis data samples
    :param ys: y-axis data samples
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xmin = xs.min()
    xmax = xs.max()
    ymin = ys.min()
    ymax = ys.max()
    values = np.vstack([xs, ys])
    kernel = scipy.stats.gaussian_kde(values)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel.evaluate(positions).T, X.shape)
    plt.gca().imshow(
        np.rot90(Z),
        cmap=plt.cm.Greys,
        extent=[xmin, xmax, ymin, ymax],
    )


def plot_monte_carlo_mean(
    xs: np.ndarray,
    ys: np.ndarray,
    f: Callable,
    mean_fx: List[float],
    label: str,
    *,
    plot_colormap: bool = True,
) -> None:
    """
    Plot the Monte Carlo mean estimate versus the true mean of a nonlinear function.

    :param xs: x-axis input samples
    :param ys: y-axis input samples
    :param f: nonlinear function to apply to inputs
    :param mean_fx: the true mean [x, y] to compare against
    :param label: label for the true mean in the legend
    :param plot_colormap: whether to overlay a bivariate colormap
    """
    fxs, fys = f(xs, ys)
    computed_mean_x = np.average(fxs)
    computed_mean_y = np.average(fys)
    ax = plt.subplot(121)
    ax.grid(False)
    plot_bivariate_colormap(xs, ys)
    plt.scatter(xs, ys, marker=".", alpha=0.02, color="k")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax = plt.subplot(122)
    ax.grid(False)
    plt.scatter(fxs, fys, marker=".", alpha=0.02, color="k")
    plt.scatter(mean_fx[0], mean_fx[1], marker="v", s=300, c="r", label=label)
    plt.scatter(
        computed_mean_x,
        computed_mean_y,
        marker="*",
        s=120,
        c="b",
        label="Computed Mean",
    )
    plot_bivariate_colormap(fxs, fys)
    ax.set_xlim([-100, 100])
    ax.set_ylim([-10, 200])
    plt.legend(loc="best", scatterpoints=1)
    _LOG.info(
        "Difference in mean x=%.3f, y=%.3f",
        computed_mean_x - mean_fx[0],
        computed_mean_y - mean_fx[1],
    )
