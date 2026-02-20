"""
Utility functions for multivariate Kalman filter tutorial (L09_05_03).

Import as:

import msml610.tutorials.L09_05_03_multivariate_kalman_filter_utils as mtl090503mkfuti
"""

import logging
from typing import Optional

import filterpy.stats as stats
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.random import multivariate_normal

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


# TODO(ai_gp): Add interactive plot of 2D covariance matrix called cell_1_1_plot_covariance_matrix
# The widgets are the covariance matrix elements, var_x, var_y, and cov_xy.
# There is seed
# There is N number of samples
# The covariance is shown in the plot
# Add it to the notebook L09_05_03_multivariate_kalman_filter.ipynb

from filterpy.stats import plot_covariance_ellipse

fig = None
def plot_covariance(var_x, var_y, cov_xy):
    global fig
    if fig: plt.close(fig)
    fig = plt.figure(figsize=(4,4))
    P1 = [[var_x, cov_xy], [cov_xy, var_y]]

    plot_covariance_ellipse((10, 10), P1, 
    std=[1, 2, 3],
    axis_equal=False,
                            show_semiaxis=True)

    plt.xlim(4, 16)
    plt.gca().set_aspect('equal')
    plt.ylim(4, 16)

    
with figsize(y=6):
    interact (plot_covariance,           
          var_x=FloatSlider(5, min=0, max=20), 
          var_y=FloatSlider(5, min=0, max=20), 
          cov_xy=FloatSlider(1.5, min=0, max=50, step=.2));