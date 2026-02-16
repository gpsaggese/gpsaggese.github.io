"""
Utility functions for g-h filter tutorial (L09_04).

Import as:

import msml610.tutorials.L09_05_kalman_filter_utils as mtl0kfiut
"""

import logging
from typing import Tuple

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from IPython.display import display

import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)


# #############################################################################
# Cell 3: Sum and Product of Gaussians
# #############################################################################


def gaussian_sum(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> Tuple[float, float]:
    """
    Compute sum of two Gaussians.

    Given X ~ N(mu1, sigma1^2) and Y ~ N(mu2, sigma2^2),
    compute Z = X + Y ~ N(mu, sigma^2).

    :param mu1: Mean of first Gaussian
    :param sigma1: Standard deviation of first Gaussian
    :param mu2: Mean of second Gaussian
    :param sigma2: Standard deviation of second Gaussian
    :return: Tuple of (mu, sigma) for the sum distribution
    """
    mu = mu1 + mu2
    sigma = np.sqrt(sigma1**2 + sigma2**2)
    return mu, sigma


def gaussian_product(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> Tuple[float, float]:
    """
    Compute product of two Gaussians.

    Given X ~ N(mu1, sigma1^2) and Y ~ N(mu2, sigma2^2),
    compute the product (point-wise multiplication of PDFs).

    :param mu1: Mean of first Gaussian
    :param sigma1: Standard deviation of first Gaussian
    :param mu2: Mean of second Gaussian
    :param sigma2: Standard deviation of second Gaussian
    :return: Tuple of (mu, sigma) for the product distribution
    """
    sigma1_sq = sigma1**2
    sigma2_sq = sigma2**2
    # Product of Gaussians formula.
    mu = (mu1 * sigma2_sq + mu2 * sigma1_sq) / (sigma1_sq + sigma2_sq)
    sigma = np.sqrt((sigma1_sq * sigma2_sq) / (sigma1_sq + sigma2_sq))
    return mu, sigma


def _plot_gaussian_sum_with_correlation(
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
    rho: float,
    *,
    n_samples: int = 10000,
) -> None:
    """
    Plot the sum of two Gaussians with correlation.

    :param mu1: Mean of first Gaussian
    :param sigma1: Standard deviation of first Gaussian
    :param mu2: Mean of second Gaussian
    :param sigma2: Standard deviation of second Gaussian
    :param rho: Correlation coefficient between X and Y (-1 to 1)
    :param n_samples: Number of samples for numerical computation
    """
    # Compute sum analytically with correlation.
    # For Z = X + Y with correlated X and Y:
    # mu_Z = mu_X + mu_Y
    # sigma_Z^2 = sigma_X^2 + sigma_Y^2 + 2*rho*sigma_X*sigma_Y
    mu_sum = mu1 + mu2
    sigma_sum = np.sqrt(sigma1**2 + sigma2**2 + 2 * rho * sigma1 * sigma2)
    Z_analytical = stats.norm(mu_sum, sigma_sum)
    # Compute sum numerically by sampling from bivariate normal.
    np.random.seed(42)
    # Create covariance matrix for bivariate normal.
    cov_matrix = np.array(
        [
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2],
        ]
    )
    mean_vector = np.array([mu1, mu2])
    # Sample from bivariate normal.
    samples = np.random.multivariate_normal(mean_vector, cov_matrix, n_samples)
    X_samples = samples[:, 0]
    Y_samples = samples[:, 1]
    Z_samples = X_samples + Y_samples
    # Plot.
    x_range = np.linspace(
        min(mu1 - 4 * sigma1, mu2 - 4 * sigma2, mu_sum - 4 * sigma_sum),
        max(mu1 + 4 * sigma1, mu2 + 4 * sigma2, mu_sum + 4 * sigma_sum),
        1000,
    )
    plt.figure()
    X_marginal = stats.norm(mu1, sigma1)
    Y_marginal = stats.norm(mu2, sigma2)
    # Plot input Gaussians with filled areas.
    X_pdf = X_marginal.pdf(x_range)
    Y_pdf = Y_marginal.pdf(x_range)
    plt.fill_between(
        x_range,
        X_pdf,
        alpha=0.3,
        color="blue",
        label=f"X ~ N({mu1:.1f}, {sigma1:.1f}^2)",
    )
    plt.plot(
        x_range,
        X_pdf,
        linewidth=2,
        color="blue",
    )
    plt.fill_between(
        x_range,
        Y_pdf,
        alpha=0.3,
        color="yellow",
        label=f"Y ~ N({mu2:.1f}, {sigma2:.1f}^2)",
    )
    plt.plot(
        x_range,
        Y_pdf,
        linewidth=2,
        color="orange",
    )
    # Plot sum (analytical).
    plt.plot(
        x_range,
        Z_analytical.pdf(x_range),
        label=f"Sum Analytical: N({mu_sum:.2f}, {sigma_sum:.2f}^2)",
        linewidth=2,
        color="red",
    )
    # Plot sum (numerical histogram).
    plt.hist(
        Z_samples,
        bins=50,
        density=True,
        alpha=0.5,
        label=f"Sum Numerical ({n_samples} samples)",
        color="lightcoral",
    )
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title(f"Sum of Gaussians: Z = X + Y (rho={rho:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([-5, 5])
    plt.ylim([0, 1])
    plt.tight_layout()


def cell1_1_plot_gaussian_sum() -> None:
    """
    Create interactive widget for exploring sum of Gaussians with correlation.

    Allows user to adjust parameters of two Gaussians (means, standard
    deviations, and correlation) and see their sum both analytically and
    numerically.
    """
    fig_sum = None

    def _plot_sum(
        mu1: float,
        sigma1: float,
        mu2: float,
        sigma2: float,
        rho: float,
    ) -> None:
        """
        Plot sum of Gaussians with given parameters.

        :param mu1: Mean of first Gaussian
        :param sigma1: Standard deviation of first Gaussian
        :param mu2: Mean of second Gaussian
        :param sigma2: Standard deviation of second Gaussian
        :param rho: Correlation coefficient
        """
        nonlocal fig_sum
        if fig_sum is not None:
            plt.close(fig_sum)
        _plot_gaussian_sum_with_correlation(mu1, sigma1, mu2, sigma2, rho)
        fig_sum = plt.gcf()

    # Create widgets for first Gaussian.
    mu1_slider, mu1_box = mtumsuti.build_widget_control(
        name="mu1",
        description="Mean 1",
        min_val=-10.0,
        max_val=10.0,
        step=0.5,
        initial_value=0.0,
        is_float=True,
    )
    sigma1_slider, sigma1_box = mtumsuti.build_widget_control(
        name="sigma1",
        description="Std Dev 1",
        min_val=0.1,
        max_val=5.0,
        step=0.1,
        initial_value=1.0,
        is_float=True,
    )
    # Create widgets for second Gaussian.
    mu2_slider, mu2_box = mtumsuti.build_widget_control(
        name="mu2",
        description="Mean 2",
        min_val=-10.0,
        max_val=10.0,
        step=0.5,
        initial_value=3.0,
        is_float=True,
    )
    sigma2_slider, sigma2_box = mtumsuti.build_widget_control(
        name="sigma2",
        description="Std Dev 2",
        min_val=0.1,
        max_val=5.0,
        step=0.1,
        initial_value=1.5,
        is_float=True,
    )
    # Create widget for correlation.
    rho_slider, rho_box = mtumsuti.build_widget_control(
        name="rho",
        description="Correlation",
        min_val=-0.99,
        max_val=0.99,
        step=0.05,
        initial_value=0.0,
        is_float=True,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot_sum,
        {
            "mu1": mu1_slider,
            "sigma1": sigma1_slider,
            "mu2": mu2_slider,
            "sigma2": sigma2_slider,
            "rho": rho_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [mu1_box, sigma1_box, mu2_box, sigma2_box, rho_box, output]
        )
    )


def _plot_gaussian_product_helper(
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
    *,
    n_samples: int = 10000,
) -> None:
    """
    Plot the product of two Gaussians.

    :param mu1: Mean of first Gaussian
    :param sigma1: Standard deviation of first Gaussian
    :param mu2: Mean of second Gaussian
    :param sigma2: Standard deviation of second Gaussian
    :param n_samples: Number of samples for numerical computation
    """
    # Create scipy Gaussian distributions.
    X = stats.norm(mu1, sigma1)
    Y = stats.norm(mu2, sigma2)
    # Compute product analytically.
    mu_prod, sigma_prod = gaussian_product(mu1, sigma1, mu2, sigma2)
    Z_analytical = stats.norm(mu_prod, sigma_prod)
    # Compute product numerically using importance sampling.
    np.random.seed(42)
    # Sample from first Gaussian (prior).
    X_samples = X.rvs(size=n_samples)
    # Compute weights based on second Gaussian (likelihood).
    weights = Y.pdf(X_samples)
    # Normalize weights.
    weights = weights / np.sum(weights)
    # Resample using weights to get samples from product distribution.
    Z_samples = np.random.choice(X_samples, size=n_samples, p=weights)
    # Plot.
    x_range = np.linspace(
        min(mu1 - 4 * sigma1, mu2 - 4 * sigma2, mu_prod - 4 * sigma_prod),
        max(mu1 + 4 * sigma1, mu2 + 4 * sigma2, mu_prod + 4 * sigma_prod),
        1000,
    )
    plt.figure()
    # Plot input Gaussians with filled areas.
    X_pdf = X.pdf(x_range)
    Y_pdf = Y.pdf(x_range)
    plt.fill_between(
        x_range,
        X_pdf,
        alpha=0.3,
        color="blue",
        label=f"X ~ N({mu1:.1f}, {sigma1:.1f}^2)",
    )
    plt.plot(
        x_range,
        X_pdf,
        linewidth=2,
        color="blue",
    )
    plt.fill_between(
        x_range,
        Y_pdf,
        alpha=0.3,
        color="yellow",
        label=f"Y ~ N({mu2:.1f}, {sigma2:.1f}^2)",
    )
    plt.plot(
        x_range,
        Y_pdf,
        linewidth=2,
        color="orange",
    )
    # Plot product (analytical).
    plt.plot(
        x_range,
        Z_analytical.pdf(x_range),
        label=f"Product Analytical: N({mu_prod:.2f}, {sigma_prod:.2f}^2)",
        linewidth=2,
        color="red",
    )
    # Plot product (numerical histogram).
    plt.hist(
        Z_samples,
        bins=50,
        density=True,
        alpha=0.5,
        label=f"Product Numerical ({n_samples} samples)",
        color="lightcoral",
    )
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title("Product of Gaussians: Z = X * Y (PDF multiplication)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([-5, 5])
    plt.ylim([0, 1])
    plt.tight_layout()


def cell1_2_plot_gaussian_product() -> None:
    """
    Create interactive widget for exploring product of Gaussians.

    Allows user to adjust parameters of two Gaussians and see their
    product both analytically and numerically.
    """
    fig_prod = None

    def _plot_product(
        mu1: float, sigma1: float, mu2: float, sigma2: float
    ) -> None:
        """
        Plot product of Gaussians with given parameters.

        :param mu1: Mean of first Gaussian
        :param sigma1: Standard deviation of first Gaussian
        :param mu2: Mean of second Gaussian
        :param sigma2: Standard deviation of second Gaussian
        """
        nonlocal fig_prod
        if fig_prod is not None:
            plt.close(fig_prod)
        _plot_gaussian_product_helper(mu1, sigma1, mu2, sigma2)
        fig_prod = plt.gcf()

    # Create widgets for first Gaussian.
    mu1_slider, mu1_box = mtumsuti.build_widget_control(
        name="mu1",
        description="Mean 1",
        min_val=-10.0,
        max_val=10.0,
        step=0.5,
        initial_value=0.0,
        is_float=True,
    )
    sigma1_slider, sigma1_box = mtumsuti.build_widget_control(
        name="sigma1",
        description="Std Dev 1",
        min_val=0.1,
        max_val=5.0,
        step=0.1,
        initial_value=2.0,
        is_float=True,
    )
    # Create widgets for second Gaussian.
    mu2_slider, mu2_box = mtumsuti.build_widget_control(
        name="mu2",
        description="Mean 2",
        min_val=-10.0,
        max_val=10.0,
        step=0.5,
        initial_value=3.0,
        is_float=True,
    )
    sigma2_slider, sigma2_box = mtumsuti.build_widget_control(
        name="sigma2",
        description="Std Dev 2",
        min_val=0.1,
        max_val=5.0,
        step=0.1,
        initial_value=1.0,
        is_float=True,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot_product,
        {
            "mu1": mu1_slider,
            "sigma1": sigma1_slider,
            "mu2": mu2_slider,
            "sigma2": sigma2_slider,
        },
    )
    # Display widgets.
    display(ipywidgets.VBox([mu1_box, sigma1_box, mu2_box, sigma2_box, output]))
