"""
Utility functions for multivariate Kalman filter tutorial (L09_05_03).

Import as:

import msml610.tutorials.L09_05_03_multivariate_kalman_filter_utils as mtl090503mkfuti
"""

import logging
from typing import Optional

import filterpy.stats as stats
from filterpy.stats import plot_covariance_ellipse
import ipywidgets
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.random import multivariate_normal
from IPython.display import display

import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)


def plot_correlated_data(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    equal: bool = True,
) -> None:
    """
    Plot correlation between x and y by performing linear regression between X
    and Y.

    :param X: x data
    :param Y: y data
    :param xlabel: optional label for x axis
    :param ylabel: optional label for y axis
    :param equal: use equal scale for x and y axis
    """
    plt.scatter(X, Y)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    # Fit line through data.
    m, b = np.polyfit(X, Y, 1)
    plt.plot(X, np.asarray(X) * m + b, color="k")
    if equal:
        plt.gca().set_aspect("equal")
    plt.show()


def plot_3d_sampled_covariance(
    mean: np.ndarray,
    cov: np.ndarray,
) -> None:
    """
    Plot a 2x2 covariance matrix positioned at mean. Mean will be plotted in x
    and y, and the probability in the z axis.

    :param mean: 2x1 mean for x and y coordinates, e.g., (2.3, 7.5)
    :param cov: 2x2 covariance matrix
    """
    # Compute width and height of covariance ellipse to choose appropriate
    # ranges for x and y.
    o, w, h = stats.covariance_ellipse(cov, 3)
    # Rotate width and height to x,y axis.
    wx = abs(w * np.cos(o) + h * np.sin(o)) * 1.2
    wy = abs(h * np.cos(o) - w * np.sin(o)) * 1.2
    # Ensure axis are of the same size so everything is plotted with the same
    # scale.
    if wx > wy:
        w = wx
    else:
        w = wy
    minx = mean[0] - w
    maxx = mean[0] + w
    miny = mean[1] - w
    maxy = mean[1] + w
    count = 1000
    x, y = multivariate_normal(mean=mean, cov=cov, size=count).T
    xs = np.arange(minx, maxx, (maxx - minx) / 40.0)
    ys = np.arange(miny, maxy, (maxy - miny) / 40.0)
    xv, yv = np.meshgrid(xs, ys)
    zs = np.array(
        [
            100.0 * stats.multivariate_gaussian(np.array([xx, yy]), mean, cov)
            for xx, yy in zip(np.ravel(xv), np.ravel(yv))
        ]
    )
    zv = zs.reshape(xv.shape)
    ax = plt.gcf().add_subplot(111, projection="3d")
    ax.scatter(x, y, [0] * count, marker=".")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    x = mean[0]
    zs = np.array(
        [
            100.0 * stats.multivariate_gaussian(np.array([x, y]), mean, cov)
            for _, y in zip(np.ravel(xv), np.ravel(yv))
        ]
    )
    zv = zs.reshape(xv.shape)
    ax.contour(xv, yv, zv, zdir="x", offset=minx - 1, cmap=cm.binary)
    y = mean[1]
    zs = np.array(
        [
            100.0 * stats.multivariate_gaussian(np.array([x, y]), mean, cov)
            for x, _ in zip(np.ravel(xv), np.ravel(yv))
        ]
    )
    zv = zs.reshape(xv.shape)
    ax.contour(xv, yv, zv, zdir="y", offset=maxy, cmap=cm.binary)


# #############################################################################
# Cell 1: Interactive 2D Covariance Matrix
# #############################################################################


def _plot_covariance_matrix(
    var_x: float,
    var_y: float,
    cov_xy: float,
    n_samples: int,
    seed: int,
) -> None:
    """
    Plot 2D Gaussian samples with covariance ellipse.

    :param var_x: variance of x dimension
    :param var_y: variance of y dimension
    :param cov_xy: covariance between x and y
    :param n_samples: number of samples to draw
    :param seed: random seed for reproducibility
    """
    mean = np.array([0.0, 0.0])
    cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])
    np.random.seed(seed)
    # Sample from the multivariate Gaussian.
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    plt.figure(figsize=(6, 6))
    # Plot samples.
    plt.scatter(
        samples[:, 0], samples[:, 1], alpha=0.3, s=10, label="Samples"
    )
    # Plot covariance ellipse at 1, 2, 3 standard deviations.
    plot_covariance_ellipse(
        mean,
        cov,
        std=[1, 2, 3],
        axis_equal=True,
        show_semiaxis=True,
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    cov_str = (
        f"[[{var_x:.2f}, {cov_xy:.2f}], [{cov_xy:.2f}, {var_y:.2f}]]"
    )
    plt.title(f"2D Covariance: {cov_str}\nN={n_samples}, seed={seed}")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def cell_1_1_plot_covariance_matrix() -> None:
    """
    Create interactive widget for exploring 2D covariance matrix.

    Allows user to adjust var_x, var_y, cov_xy, number of samples, and seed
    to visualize the covariance ellipse and sampled data points.
    """
    fig_cov = None

    def _plot(
        var_x: float,
        var_y: float,
        cov_xy: float,
        n_samples: int,
        seed: int,
    ) -> None:
        """
        Plot covariance matrix with given parameters.

        :param var_x: variance of x dimension
        :param var_y: variance of y dimension
        :param cov_xy: covariance between x and y
        :param n_samples: number of samples to draw
        :param seed: random seed for reproducibility
        """
        nonlocal fig_cov
        if fig_cov is not None:
            plt.close(fig_cov)
        _plot_covariance_matrix(var_x, var_y, cov_xy, n_samples, seed)
        fig_cov = plt.gcf()

    # Create var_x widget.
    var_x_slider, var_x_box = mtumsuti.build_widget_control(
        name="var_x",
        description="var_x",
        min_val=0.1,
        max_val=20.0,
        step=0.1,
        initial_value=5.0,
        is_float=True,
    )
    # Create var_y widget.
    var_y_slider, var_y_box = mtumsuti.build_widget_control(
        name="var_y",
        description="var_y",
        min_val=0.1,
        max_val=20.0,
        step=0.1,
        initial_value=5.0,
        is_float=True,
    )
    # Create cov_xy widget.
    cov_xy_slider, cov_xy_box = mtumsuti.build_widget_control(
        name="cov_xy",
        description="cov_xy",
        min_val=-10.0,
        max_val=10.0,
        step=0.1,
        initial_value=1.5,
        is_float=True,
    )
    # Create n_samples widget.
    n_samples_slider, n_samples_box = mtumsuti.build_widget_control(
        name="n_samples",
        description="n_samples",
        min_val=100,
        max_val=5000,
        step=100,
        initial_value=1000,
        is_float=False,
    )
    # Create seed widget (always last by convention).
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot,
        {
            "var_x": var_x_slider,
            "var_y": var_y_slider,
            "cov_xy": cov_xy_slider,
            "n_samples": n_samples_slider,
            "seed": seed_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [var_x_box, var_y_box, cov_xy_box, n_samples_box, seed_box, output]
        )
    )