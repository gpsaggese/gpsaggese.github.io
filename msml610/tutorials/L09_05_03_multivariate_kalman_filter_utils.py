"""
Utility functions for multivariate Kalman filter tutorial (L09_05_03).

Import as:

import msml610.tutorials.L09_05_03_multivariate_kalman_filter_utils as mtl090503mkfuti
"""

import logging
import math
from typing import Optional, Tuple

import filterpy.stats as stats
import ipywidgets
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
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
    x, y = np.random.multivariate_normal(mean=mean, cov=cov, size=count).T
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
    ax.contour(xv, yv, zv, zdir="x", offset=minx - 1, cmap=matplotlib.cm.binary)
    y = mean[1]
    zs = np.array(
        [
            100.0 * stats.multivariate_gaussian(np.array([x, y]), mean, cov)
            for x, _ in zip(np.ravel(xv), np.ravel(yv))
        ]
    )
    zv = zs.reshape(xv.shape)
    ax.contour(xv, yv, zv, zdir="y", offset=maxy, cmap=matplotlib.cm.binary)


# #############################################################################
# Cell 1.2 / 1.3: Helpers
# #############################################################################


def _sample_valid_cov_params(
    var_min: float = 0.1,
    var_max: float = 10.0,
    *,
    cov_fraction: float = 0.9,
) -> tuple:
    """
    Sample random valid 2D covariance parameters (positive definite matrix).

    Draws var_x and var_y uniformly in [var_min, var_max], then draws
    cov_xy in (-cov_fraction * sqrt(var_x * var_y),
               +cov_fraction * sqrt(var_x * var_y)) to guarantee positive
    definiteness.

    :param var_min: minimum value for var_x and var_y
    :param var_max: maximum value for var_x and var_y
    :param cov_fraction: fraction of the geometric mean used as cov_xy limit
    :return: tuple (var_x, var_y, cov_xy) rounded to one decimal place
    """
    var_x = round(np.random.uniform(var_min, var_max), 1)
    var_y = round(np.random.uniform(var_min, var_max), 1)
    max_cov = cov_fraction * np.sqrt(var_x * var_y)
    cov_xy = round(np.random.uniform(-max_cov, max_cov), 1)
    return var_x, var_y, cov_xy


# #############################################################################
# Cell 1.2: Sum of Two 2D Gaussians
# #############################################################################


def _plot_sum_of_gaussians(
    var_x1: float,
    var_y1: float,
    cov_xy1: float,
    var_x2: float,
    var_y2: float,
    cov_xy2: float,
) -> None:
    """
    Plot two 2D Gaussians and their sum as covariance ellipses.

    If X ~ N(0, Sigma1) and Y ~ N(0, Sigma2) are independent, then
    X + Y ~ N(0, Sigma1 + Sigma2): the covariances add.

    :param var_x1: variance of x dimension for Gaussian 1
    :param var_y1: variance of y dimension for Gaussian 1
    :param cov_xy1: covariance between x and y for Gaussian 1
    :param var_x2: variance of x dimension for Gaussian 2
    :param var_y2: variance of y dimension for Gaussian 2
    :param cov_xy2: covariance between x and y for Gaussian 2
    """
    mean = np.array([0.0, 0.0])
    cov1 = np.array([[var_x1, cov_xy1], [cov_xy1, var_y1]])
    cov2 = np.array([[var_x2, cov_xy2], [cov_xy2, var_y2]])
    # Validate positive definiteness.
    for cov, name in [(cov1, "G1"), (cov2, "G2")]:
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            _LOG.warning(
                "%s covariance not positive definite;"
                " need var_x * var_y > cov_xy^2",
                name,
            )
            return
    # Compute sum covariance: covariances add for independent variables.
    cov_sum = cov1 + cov2
    plt.figure(figsize=(8, 8))
    # Plot ellipses at 1 and 2 standard deviations.
    stats.plot_covariance_ellipse(mean, cov1, fc="y", alpha=0.4)
    stats.plot_covariance_ellipse(mean, cov2, fc="g", alpha=0.4)
    stats.plot_covariance_ellipse(
        mean, cov_sum, fc="b", alpha=0.4
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    cov1_str = (
        f"[[{var_x1:.1f}, {cov_xy1:.1f}], [{cov_xy1:.1f}, {var_y1:.1f}]]"
    )
    cov2_str = (
        f"[[{var_x2:.1f}, {cov_xy2:.1f}], [{cov_xy2:.1f}, {var_y2:.1f}]]"
    )
    plt.title(f"Sum of Gaussians\nG1={cov1_str}\nG2={cov2_str}")
    legend_elements = [
        matplotlib.patches.Patch(fc="y", alpha=0.4, label="G1"),
        matplotlib.patches.Patch(fc="g", alpha=0.4, label="G2"),
        matplotlib.patches.Patch(fc="b", alpha=0.4, label="G1 + G2 (sum)"),
    ]
    plt.legend(handles=legend_elements)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# #############################################################################
# Cell 1.3: Product of Two 2D Gaussians
# #############################################################################


def _plot_product_of_gaussians(
    var_x1: float,
    var_y1: float,
    cov_xy1: float,
    var_x2: float,
    var_y2: float,
    cov_xy2: float,
) -> None:
    """
    Plot two 2D Gaussians and their product as covariance ellipses.

    The product of G1 ~ N(0, Sigma1) and G2 ~ N(0, Sigma2) is proportional
    to N(0, Sigma) where Sigma^{-1} = Sigma1^{-1} + Sigma2^{-1}. The
    product is always more certain (smaller ellipse) than either factor.

    :param var_x1: variance of x dimension for Gaussian 1
    :param var_y1: variance of y dimension for Gaussian 1
    :param cov_xy1: covariance between x and y for Gaussian 1
    :param var_x2: variance of x dimension for Gaussian 2
    :param var_y2: variance of y dimension for Gaussian 2
    :param cov_xy2: covariance between x and y for Gaussian 2
    """
    mean = np.array([0.0, 0.0])
    cov1 = np.array([[var_x1, cov_xy1], [cov_xy1, var_y1]])
    cov2 = np.array([[var_x2, cov_xy2], [cov_xy2, var_y2]])
    # Validate positive definiteness.
    for cov, name in [(cov1, "G1"), (cov2, "G2")]:
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            _LOG.warning(
                "%s covariance not positive definite;"
                " need var_x * var_y > cov_xy^2",
                name,
            )
            return
    # Compute product covariance: Sigma^{-1} = Sigma1^{-1} + Sigma2^{-1}.
    cov_prod = np.linalg.inv(np.linalg.inv(cov1) + np.linalg.inv(cov2))
    plt.figure(figsize=(8, 8))
    # Plot ellipses at 1 and 2 standard deviations.
    stats.plot_covariance_ellipse(mean, cov1, fc="y", alpha=0.4)
    stats.plot_covariance_ellipse(mean, cov2, fc="g", alpha=0.4)
    stats.plot_covariance_ellipse(
        mean, cov_prod, fc="b", alpha=0.6
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    cov1_str = (
        f"[[{var_x1:.1f}, {cov_xy1:.1f}], [{cov_xy1:.1f}, {var_y1:.1f}]]"
    )
    cov2_str = (
        f"[[{var_x2:.1f}, {cov_xy2:.1f}], [{cov_xy2:.1f}, {var_y2:.1f}]]"
    )
    plt.title(f"Product of Gaussians\nG1={cov1_str}\nG2={cov2_str}")
    legend_elements = [
        matplotlib.patches.Patch(fc="y", alpha=0.4, label="G1"),
        matplotlib.patches.Patch(fc="g", alpha=0.4, label="G2"),
        matplotlib.patches.Patch(
            fc="b", alpha=0.6, label="G1 * G2 (product)"
        ),
    ]
    plt.legend(handles=legend_elements)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# #############################################################################
# Cell 1.1: Interactive 2D Covariance Matrix
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
    # Use axis_equal=False to avoid conflict with fixed xlim/ylim; set equal
    # aspect manually with adjustable='box' so the axes box is resized instead
    # of the data limits being overridden.
    stats.plot_covariance_ellipse(
        mean,
        cov,
        axis_equal=False,
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
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def cell_1_2_plot_sum_of_gaussians() -> None:
    """
    Create interactive widget for exploring the sum of two 2D Gaussians.

    Shows G1 (yellow), G2 (green), and G1 + G2 (blue) as covariance
    ellipses. The sum covariance equals Sigma1 + Sigma2. A "Random" button
    assigns random valid covariance parameters to both Gaussians.
    """
    fig_sum = None

    def _plot(
        var_x1: float,
        var_y1: float,
        cov_xy1: float,
        var_x2: float,
        var_y2: float,
        cov_xy2: float,
    ) -> None:
        """
        Update plot for sum of Gaussians.

        :param var_x1: variance of x for Gaussian 1
        :param var_y1: variance of y for Gaussian 1
        :param cov_xy1: covariance xy for Gaussian 1
        :param var_x2: variance of x for Gaussian 2
        :param var_y2: variance of y for Gaussian 2
        :param cov_xy2: covariance xy for Gaussian 2
        """
        nonlocal fig_sum
        if fig_sum is not None:
            plt.close(fig_sum)
        _plot_sum_of_gaussians(
            var_x1, var_y1, cov_xy1, var_x2, var_y2, cov_xy2
        )
        fig_sum = plt.gcf()

    # Create widgets for Gaussian 1.
    var_x1_slider, var_x1_box = mtumsuti.build_widget_control(
        name="var_x1",
        description="var_x1 (G1)",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=2.0,
        is_float=True,
    )
    var_y1_slider, var_y1_box = mtumsuti.build_widget_control(
        name="var_y1",
        description="var_y1 (G1)",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=2.0,
        is_float=True,
    )
    cov_xy1_slider, cov_xy1_box = mtumsuti.build_widget_control(
        name="cov_xy1",
        description="cov_xy1 (G1)",
        min_val=-5.0,
        max_val=5.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    # Create widgets for Gaussian 2.
    var_x2_slider, var_x2_box = mtumsuti.build_widget_control(
        name="var_x2",
        description="var_x2 (G2)",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=3.0,
        is_float=True,
    )
    var_y2_slider, var_y2_box = mtumsuti.build_widget_control(
        name="var_y2",
        description="var_y2 (G2)",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=1.0,
        is_float=True,
    )
    cov_xy2_slider, cov_xy2_box = mtumsuti.build_widget_control(
        name="cov_xy2",
        description="cov_xy2 (G2)",
        min_val=-5.0,
        max_val=5.0,
        step=0.1,
        initial_value=-0.5,
        is_float=True,
    )
    # Create Random button.
    random_button = ipywidgets.Button(
        description="Random",
        button_style="info",
        layout={"width": "100px"},
    )

    def _on_random(b: ipywidgets.Button) -> None:
        """
        Assign random valid covariance parameters to all six sliders.

        :param b: button widget (unused)
        """
        var_x1, var_y1, cov_xy1 = _sample_valid_cov_params()
        var_x2, var_y2, cov_xy2 = _sample_valid_cov_params()
        var_x1_slider.value = var_x1
        var_y1_slider.value = var_y1
        cov_xy1_slider.value = cov_xy1
        var_x2_slider.value = var_x2
        var_y2_slider.value = var_y2
        cov_xy2_slider.value = cov_xy2

    random_button.on_click(_on_random)
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot,
        {
            "var_x1": var_x1_slider,
            "var_y1": var_y1_slider,
            "cov_xy1": cov_xy1_slider,
            "var_x2": var_x2_slider,
            "var_y2": var_y2_slider,
            "cov_xy2": cov_xy2_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                random_button,
                var_x1_box,
                var_y1_box,
                cov_xy1_box,
                var_x2_box,
                var_y2_box,
                cov_xy2_box,
                output,
            ]
        )
    )


def cell_1_3_plot_product_of_gaussians() -> None:
    """
    Create interactive widget for exploring the product of two 2D Gaussians.

    Shows G1 (yellow), G2 (green), and G1 * G2 (blue) as covariance
    ellipses. The product is always more certain (smaller) than either
    factor. A "Random" button assigns random valid covariance parameters to
    both Gaussians.
    """
    fig_prod = None

    def _plot(
        var_x1: float,
        var_y1: float,
        cov_xy1: float,
        var_x2: float,
        var_y2: float,
        cov_xy2: float,
    ) -> None:
        """
        Update plot for product of Gaussians.

        :param var_x1: variance of x for Gaussian 1
        :param var_y1: variance of y for Gaussian 1
        :param cov_xy1: covariance xy for Gaussian 1
        :param var_x2: variance of x for Gaussian 2
        :param var_y2: variance of y for Gaussian 2
        :param cov_xy2: covariance xy for Gaussian 2
        """
        nonlocal fig_prod
        if fig_prod is not None:
            plt.close(fig_prod)
        _plot_product_of_gaussians(
            var_x1, var_y1, cov_xy1, var_x2, var_y2, cov_xy2
        )
        fig_prod = plt.gcf()

    # Create widgets for Gaussian 1.
    var_x1_slider, var_x1_box = mtumsuti.build_widget_control(
        name="var_x1",
        description="var_x1 (G1)",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=4.0,
        is_float=True,
    )
    var_y1_slider, var_y1_box = mtumsuti.build_widget_control(
        name="var_y1",
        description="var_y1 (G1)",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=2.0,
        is_float=True,
    )
    cov_xy1_slider, cov_xy1_box = mtumsuti.build_widget_control(
        name="cov_xy1",
        description="cov_xy1 (G1)",
        min_val=-5.0,
        max_val=5.0,
        step=0.1,
        initial_value=1.0,
        is_float=True,
    )
    # Create widgets for Gaussian 2.
    var_x2_slider, var_x2_box = mtumsuti.build_widget_control(
        name="var_x2",
        description="var_x2 (G2)",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=2.0,
        is_float=True,
    )
    var_y2_slider, var_y2_box = mtumsuti.build_widget_control(
        name="var_y2",
        description="var_y2 (G2)",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=4.0,
        is_float=True,
    )
    cov_xy2_slider, cov_xy2_box = mtumsuti.build_widget_control(
        name="cov_xy2",
        description="cov_xy2 (G2)",
        min_val=-5.0,
        max_val=5.0,
        step=0.1,
        initial_value=-1.0,
        is_float=True,
    )
    # Create Random button.
    random_button = ipywidgets.Button(
        description="Random",
        button_style="info",
        layout={"width": "100px"},
    )

    def _on_random(b: ipywidgets.Button) -> None:
        """
        Assign random valid covariance parameters to all six sliders.

        :param b: button widget (unused)
        """
        var_x1, var_y1, cov_xy1 = _sample_valid_cov_params()
        var_x2, var_y2, cov_xy2 = _sample_valid_cov_params()
        var_x1_slider.value = var_x1
        var_y1_slider.value = var_y1
        cov_xy1_slider.value = cov_xy1
        var_x2_slider.value = var_x2
        var_y2_slider.value = var_y2
        cov_xy2_slider.value = cov_xy2

    random_button.on_click(_on_random)
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot,
        {
            "var_x1": var_x1_slider,
            "var_y1": var_y1_slider,
            "cov_xy1": cov_xy1_slider,
            "var_x2": var_x2_slider,
            "var_y2": var_y2_slider,
            "cov_xy2": cov_xy2_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                random_button,
                var_x1_box,
                var_y1_box,
                cov_xy1_box,
                var_x2_box,
                var_y2_box,
                cov_xy2_box,
                output,
            ]
        )
    )


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


# #############################################################################
# Dog tracking simulation
# #############################################################################


def compute_dog_data(
    z_var: float,
    process_var: float,
    *,
    count: int = 50,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a dog moving on a 1-d track and return positions and measurements.

    The simulation runs for `count` steps, moving the dog forward approximately
    1 meter per step. At each step the velocity varies according to the process
    variance `process_var`. After updating the position, a noisy measurement is
    computed with an assumed sensor variance of `z_var`.

    :param z_var: variance of the measurement noise (sensor variance)
    :param process_var: variance of the process noise (velocity variance)
    :param count: number of steps to simulate
    :param dt: time step duration
    :return: (xs, zs) tuple of 1D NumPy arrays where `xs` contains the true
        positions and `zs` contains the noisy measurements
    """
    x = 0.0
    vel = 1.0
    z_std = math.sqrt(z_var)
    p_std = math.sqrt(process_var)
    xs, zs = [], []
    for _ in range(count):
        v = vel + (np.random.randn() * p_std)
        x += v * dt
        xs.append(x)
        zs.append(x + np.random.randn() * z_std)
    return np.array(xs), np.array(zs)


# #############################################################################
# Kalman Filter for Dog Tracking
# #############################################################################

# TODO(ai_gp): Use import filterpy.kalman as kf
from filterpy.kalman import KalmanFilter


def run_dog_kalman_filter(
    zs: np.ndarray,
    z_var: float,
    process_var: float,
    *,
    dt: float = 1.0,
    initial_pos: float = 0.0,
    initial_vel: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a multivariate Kalman filter to track a dog's 1-D position.

    The state vector is [position, velocity]. Only position is measured;
    velocity is a hidden variable inferred by the filter.

    Matrices used:
    - F = [[1, dt], [0, 1]]: constant velocity motion model
    - H = [[1, 0]]: observe position only
    - P = diag([500, 49]): large initial uncertainty
    - Q = diag([0, process_var]): velocity can change randomly
    - R = [[z_var]]: measurement noise

    :param zs: noisy position measurements, shape (count,)
    :param z_var: measurement noise variance (sensor variance)
    :param process_var: process noise variance (velocity variance)
    :param dt: time step duration
    :param initial_pos: initial estimated position
    :param initial_vel: initial estimated velocity
    :return: (means, variances) arrays of estimated position means and
        position variances at each step
    """
    # State transition: position += velocity * dt, velocity unchanged.
    F = np.array([[1.0, dt], [0.0, 1.0]])
    # Measurement: observe only position.
    H = np.array([[1.0, 0.0]])
    # Initial covariance: large uncertainty in both position and velocity.
    # Top dog speed ~21 m/s => 3*sigma_vel = 21 => sigma_vel^2 = 49.
    P = np.diag([500.0, 49.0])
    # Process noise: only velocity is perturbed by the environment.
    Q = np.array([[0.0, 0.0], [0.0, process_var]])
    # Measurement noise.
    R = np.array([[z_var]])
    # Build and initialize filter.
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = F
    kf.H = H
    kf.P = P.copy()
    kf.Q = Q
    kf.R = R
    kf.x = np.array([[initial_pos], [initial_vel]])
    # Run filter over all measurements.
    means = []
    variances = []
    for z in zs:
        kf.predict()
        kf.update(np.array([[z]]))
        means.append(float(kf.x[0]))
        variances.append(float(kf.P[0, 0]))
    return np.array(means), np.array(variances)


def plot_dog_tracking(
    xs: np.ndarray,
    zs: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    *,
    title: str = "Dog Tracking with Multivariate Kalman Filter",
) -> None:
    """
    Plot true positions, noisy measurements, and Kalman filter estimates.

    :param xs: true positions, shape (count,)
    :param zs: noisy measurements, shape (count,)
    :param means: Kalman filter estimated positions, shape (count,)
    :param variances: Kalman filter position variances, shape (count,)
    :param title: plot title
    """
    steps = np.arange(len(xs))
    std = np.sqrt(np.array(variances))
    plt.figure(figsize=(10, 5))
    plt.plot(steps, xs, label="True position", color="k", lw=2)
    plt.scatter(
        steps,
        zs,
        label="Measurements",
        color="r",
        s=20,
        alpha=0.7,
        zorder=5,
    )
    plt.plot(steps, means, label="KF estimate", color="b", lw=2)
    plt.fill_between(
        steps,
        means - std,
        means + std,
        color="b",
        alpha=0.2,
        label="KF +/- 1 std",
    )
    plt.xlabel("Time step")
    plt.ylabel("Position (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# #############################################################################
# Exercise: Show Effect of Hidden Variables
# #############################################################################


def run_dog_kalman_filter_1d(
    zs: np.ndarray,
    z_var: float,
    process_var: float,
    *,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a 1D Kalman filter that tracks only position (no hidden velocity).

    The state is a scalar: position only. The filter has no motion model
    for velocity so it treats the dog as approximately stationary between
    measurements (F = [[1]], constant position).  This is the "no hidden
    variable" baseline to compare against the 2D filter.

    Matrices used:
    - F = [[1]]: no velocity, position is constant between steps
    - H = [[1]]: observe position
    - P = [[500]]: large initial position uncertainty
    - Q = [[process_var]]: position diffuses randomly each step
    - R = [[z_var]]: measurement noise

    :param zs: noisy position measurements, shape (count,)
    :param z_var: measurement noise variance (sensor variance)
    :param process_var: process noise variance applied to position
    :param dt: time step duration (unused, kept for API symmetry)
    :return: (means, variances) 1D arrays of estimated position and variance
    """
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.F = np.array([[1.0]])
    kf.H = np.array([[1.0]])
    kf.P = np.array([[500.0]])
    kf.Q = np.array([[process_var]])
    kf.R = np.array([[z_var]])
    kf.x = np.array([[0.0]])
    means = []
    variances = []
    for z in zs:
        kf.predict()
        kf.update(np.array([[z]]))
        means.append(float(kf.x[0]))
        variances.append(float(kf.P[0, 0]))
    return np.array(means), np.array(variances)


def plot_hidden_variable_comparison(
    xs: np.ndarray,
    zs: np.ndarray,
    means_1d: np.ndarray,
    variances_1d: np.ndarray,
    means_2d: np.ndarray,
    variances_2d: np.ndarray,
    *,
    title: str = "Effect of Hidden Variable: 1D vs 2D Kalman Filter",
) -> None:
    """
    Plot side-by-side comparison of 1D KF vs 2D KF tracking.

    The 1D filter tracks position only (no hidden variable).
    The 2D filter tracks position and velocity (velocity is the hidden variable).

    :param xs: true positions, shape (count,)
    :param zs: noisy measurements, shape (count,)
    :param means_1d: 1D KF estimated positions, shape (count,)
    :param variances_1d: 1D KF position variances, shape (count,)
    :param means_2d: 2D KF estimated positions, shape (count,)
    :param variances_2d: 2D KF position variances, shape (count,)
    :param title: overall figure title
    """
    steps = np.arange(len(xs))
    std_1d = np.sqrt(np.array(variances_1d))
    std_2d = np.sqrt(np.array(variances_2d))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, means, stds, label, color in [
        (axes[0], means_1d, std_1d, "1D KF (position only)", "orange"),
        (axes[1], means_2d, std_2d, "2D KF (pos + vel hidden)", "b"),
    ]:
        ax.plot(steps, xs, color="k", lw=2, label="True position")
        ax.scatter(
            steps,
            zs,
            color="r",
            s=20,
            alpha=0.7,
            zorder=5,
            label="Measurements",
        )
        ax.plot(steps, means, color=color, lw=2, label=label)
        ax.fill_between(
            steps,
            means - stds,
            means + stds,
            color=color,
            alpha=0.2,
            label=f"{label} +/- 1 std",
        )
        ax.set_xlabel("Time step")
        ax.set_ylabel("Position (m)")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    mse_1d = float(np.mean((means_1d - xs) ** 2))
    mse_2d = float(np.mean((means_2d - xs) ** 2))
    fig.suptitle(
        f"{title}\nMSE: 1D={mse_1d:.3f}  2D={mse_2d:.3f}", fontsize=12
    )
    plt.tight_layout()


def cell_hidden_variable_comparison_interactive() -> None:
    """
    Create interactive widget comparing 1D KF vs 2D KF dog tracking.

    Controls:
    - seed: random seed for reproducibility
    - z_var: measurement noise variance
    - process_var: process noise variance
    - count: number of simulation steps

    The experiment shows that the 2D filter (with velocity as a hidden
    variable) consistently outperforms the 1D filter (position only),
    especially when the dog's velocity is non-trivial or changing.
    The Mean Squared Error (MSE) displayed in the title quantifies the
    improvement.
    """
    fig_cmp = None

    def _plot(
        seed: int,
        z_var: float,
        process_var: float,
        count: int,
    ) -> None:
        """
        Recompute and display the 1D vs 2D comparison plot.

        :param seed: random seed for reproducibility
        :param z_var: measurement noise variance
        :param process_var: process noise variance
        :param count: number of simulation steps
        """
        nonlocal fig_cmp
        if fig_cmp is not None:
            plt.close(fig_cmp)
        np.random.seed(seed)
        xs_i, zs_i = compute_dog_data(z_var, process_var, count=count)
        means_1d, var_1d = run_dog_kalman_filter_1d(
            zs_i, z_var, process_var
        )
        means_2d, var_2d = run_dog_kalman_filter(zs_i, z_var, process_var)
        plot_hidden_variable_comparison(
            xs_i, zs_i, means_1d, var_1d, means_2d, var_2d
        )
        fig_cmp = plt.gcf()

    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    z_var_slider, z_var_box = mtumsuti.build_widget_control(
        name="z_var",
        description="z_var",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=1.0,
        is_float=True,
    )
    process_var_slider, process_var_box = mtumsuti.build_widget_control(
        name="process_var",
        description="process_var",
        min_val=0.01,
        max_val=2.0,
        step=0.01,
        initial_value=0.1,
        is_float=True,
    )
    count_slider, count_box = mtumsuti.build_widget_control(
        name="count",
        description="count",
        min_val=10,
        max_val=200,
        step=10,
        initial_value=50,
        is_float=False,
    )
    output = ipywidgets.interactive_output(
        _plot,
        {
            "seed": seed_slider,
            "z_var": z_var_slider,
            "process_var": process_var_slider,
            "count": count_slider,
        },
    )
    display(
        ipywidgets.VBox(
            [seed_box, z_var_box, process_var_box, count_box, output]
        )
    )


def cell_dog_tracking_interactive() -> None:
    """
    Create interactive widget for dog tracking with multivariate Kalman filter.

    Allows adjustment of z_var (measurement noise), process_var (process
    noise), count (number of steps), and seed to visualize how the Kalman
    filter adapts to different noise conditions.

    - Increasing z_var makes the sensor noisier: the KF smooths more.
    - Increasing process_var makes the dog more unpredictable: the KF follows
      measurements more closely.
    """
    fig_dog = None

    def _plot(
        seed: int,
        z_var: float,
        process_var: float,
        count: int,
    ) -> None:
        """
        Recompute and display the dog tracking plot.

        :param seed: random seed for reproducibility
        :param z_var: measurement noise variance
        :param process_var: process noise variance
        :param count: number of simulation steps
        """
        nonlocal fig_dog
        if fig_dog is not None:
            plt.close(fig_dog)
        np.random.seed(seed)
        xs_i, zs_i = compute_dog_data(z_var, process_var, count=count)
        means_i, variances_i = run_dog_kalman_filter(zs_i, z_var, process_var)
        plot_dog_tracking(xs_i, zs_i, means_i, variances_i)
        fig_dog = plt.gcf()

    # Seed widget is always first.
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    z_var_slider, z_var_box = mtumsuti.build_widget_control(
        name="z_var",
        description="z_var",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=1.0,
        is_float=True,
    )
    process_var_slider, process_var_box = mtumsuti.build_widget_control(
        name="process_var",
        description="process_var",
        min_val=0.01,
        max_val=2.0,
        step=0.01,
        initial_value=0.1,
        is_float=True,
    )
    count_slider, count_box = mtumsuti.build_widget_control(
        name="count",
        description="count",
        min_val=10,
        max_val=200,
        step=10,
        initial_value=50,
        is_float=False,
    )
    output = ipywidgets.interactive_output(
        _plot,
        {
            "seed": seed_slider,
            "z_var": z_var_slider,
            "process_var": process_var_slider,
            "count": count_slider,
        },
    )
    display(
        ipywidgets.VBox(
            [seed_box, z_var_box, process_var_box, count_box, output]
        )
    )