"""
Utility functions for non-linear Kalman filter tutorials.

Import as:

import msml610.tutorials.L09_05_04_non_linear_kalman_filter_utils as l090504u
"""

import logging
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.stats

_LOG = logging.getLogger(__name__)


# #############################################################################
# Plotting functions.
# #############################################################################


def f_nonlinear_xy(x: float, y: float):
    """
    Apply a nonlinear function mapping (x, y) to (x+y, 0.1*x^2 + y^2).

    :param x: x input value
    :param y: y input value
    :return: tuple of transformed values (fx, fy)
    """
    return x + y, 0.1 * x**2 + y * y


def plot_nonlinear_xy() -> None:
    """
    Plot the nonlinear transformation f_nonlinear_xy as 3D surfaces.
    """
    # Grid in original plane.
    x = np.linspace(-2, 2, 120)
    y = np.linspace(-2, 2, 120)
    X, Y = np.meshgrid(x, y)
    # Transform inputs.
    FX, FY = f_nonlinear_xy(X, Y)
    # 3D embedding: (x, y) -> (fx, fy).
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Plane z = x + y.
    ax.plot_surface(
        X, Y, FX,
        color="#4C72B0",
        alpha=0.55,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    # Paraboloid z = 0.1*x^2 + y^2.
    ax.plot_surface(
        X, Y, FY,
        color="#DD8452",
        alpha=0.55,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    # Viewing angle similar to textbook plots.
    ax.view_init(elev=25, azim=-60)
    ax.set_title("Plane mapped into (f_x, f_y) space")
    ax.set_xlabel("f_x(x,y)")
    ax.set_ylabel("f_y(x,y)")
    ax.set_zlabel("original y (acts as parameter)")
    plt.show()


def plot_function(
    func: Callable,
    *,
    xmin: float = -4,
    xmax: float = 4,
    num: int = 800,
    title: str = "Function plot",
) -> None:
    """
    Plot a 1D function over a specified range.

    :param func: the function to plot
    :param xmin: minimum x value
    :param xmax: maximum x value
    :param num: number of sample points
    :param title: plot title
    """
    x = np.linspace(xmin, xmax, num)
    y = func(x)
    plt.figure(figsize=(7, 5))
    # Function curve.
    plt.plot(x, y, color="#4C72B0", linewidth=2)
    # Axis lines.
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1, alpha=0.4)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


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
    fig = plt.figure(figsize=(8, 6))
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
    plt.scatter(xs, ys, marker="o", alpha=0.05, color="k", edgecolors="none")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax = plt.subplot(122)
    ax.grid(False)
    plt.scatter(fxs, fys, marker="o", alpha=0.05, color="k", edgecolors="none")
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
