"""
Utility functions for Kalman filter tutorial (L09_05_02).

Import as:

import msml610.tutorials.L09_05_02_univariate_kalman_filter_utils as mtl090502ukfuti
"""

import logging
import math
from collections import namedtuple
from typing import List, Optional, Tuple

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython.display import display
from numpy.random import randn

import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)


# #############################################################################
# Gaussian utility functions
# #############################################################################


Gaussian = namedtuple("Gaussian", ["mean", "var"])
Gaussian.__repr__ = lambda s: f"N(mu={s[0]:.3f}, sigma^2={s[1]:.3f})"


def gaussian_sum(g1, g2):
    """
    Compute sum of two Gaussians.

    :param g1: first Gaussian
    :param g2: second Gaussian
    :return: sum Gaussian
    """
    return Gaussian(g1.mean + g2.mean, g1.var + g2.var)


def gaussian_multiply(g1, g2):
    """
    Compute product of two Gaussians (PDF multiplication).

    :param g1: first Gaussian
    :param g2: second Gaussian
    :return: product Gaussian
    """
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return Gaussian(mean, variance)


def plot_gaussian(
    gaussian: Gaussian,
    *,
    ax: Optional[plt.Axes] = None,
    label: str = "",
    color: str = "",
    style: str = "-",
) -> plt.Axes:
    """
    Plot a Gaussian distribution as its PDF.

    :param gaussian: Gaussian named tuple with mean and var fields
    :param ax: Matplotlib axes to plot on; if None, create a new figure using
        default rcParams dimensions
    :param color: Line color; empty string uses default matplotlib color cycle
    :param style: Line style (e.g., "-", "--", ":"), defaults to solid line
    """
    if ax is None:
        _, ax = plt.subplots()
    mu = gaussian.mean
    sigma = np.sqrt(gaussian.var)
    # Generate x values covering +/- 4 standard deviations.
    xs = np.arange(mu - 4 * sigma, mu + 4 * sigma, sigma / 100)
    ys = [stats.norm.pdf(x, mu, sigma) for x in xs]
    if label:
        label += " ~ "
    label += f"N({mu:.3f}, {gaussian.var:.3f})"
    kwargs: dict = {"label": label, "linestyle": style}
    if color:
        kwargs["color"] = color
    ax.plot(xs, ys, **kwargs)
    ax.legend()
    return ax


# #############################################################################
# Cell 1: Sum and Product of Gaussians
# #############################################################################


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
        description="mu1",
        min_val=-10.0,
        max_val=10.0,
        step=0.5,
        initial_value=0.0,
        is_float=True,
    )
    sigma1_slider, sigma1_box = mtumsuti.build_widget_control(
        name="sigma1",
        description="sigma1",
        min_val=0.1,
        max_val=5.0,
        step=0.1,
        initial_value=1.0,
        is_float=True,
    )
    # Create widgets for second Gaussian.
    mu2_slider, mu2_box = mtumsuti.build_widget_control(
        name="mu2",
        description="mu2",
        min_val=-10.0,
        max_val=10.0,
        step=0.5,
        initial_value=3.0,
        is_float=True,
    )
    sigma2_slider, sigma2_box = mtumsuti.build_widget_control(
        name="sigma2",
        description="sigma2",
        min_val=0.1,
        max_val=5.0,
        step=0.1,
        initial_value=1.5,
        is_float=True,
    )
    # Create widget for correlation.
    rho_slider, rho_box = mtumsuti.build_widget_control(
        name="rho",
        description="rho",
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
    # Compute product analytically using gaussian_multiply().
    g1 = Gaussian(mu1, sigma1**2)
    g2 = Gaussian(mu2, sigma2**2)
    g_prod = gaussian_multiply(g1, g2)
    mu_prod = g_prod.mean
    sigma_prod = math.sqrt(g_prod.var)
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
        description="mu1",
        min_val=-10.0,
        max_val=10.0,
        step=0.5,
        initial_value=0.0,
        is_float=True,
    )
    sigma1_slider, sigma1_box = mtumsuti.build_widget_control(
        name="sigma1",
        description="sigma1",
        min_val=0.1,
        max_val=5.0,
        step=0.1,
        initial_value=2.0,
        is_float=True,
    )
    # Create widgets for second Gaussian.
    mu2_slider, mu2_box = mtumsuti.build_widget_control(
        name="mu2",
        description="mu2",
        min_val=-10.0,
        max_val=10.0,
        step=0.5,
        initial_value=3.0,
        is_float=True,
    )
    sigma2_slider, sigma2_box = mtumsuti.build_widget_control(
        name="sigma2",
        description="sigma2",
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


# #############################################################################
# Dog Simulation
# #############################################################################


class DogSimulation:
    """
    Simulate a dog moving in 1D with process and measurement noise.
    """

    def __init__(
        self,
        *,
        x0: float = 0,
        velocity: float = 1,
        measurement_var: float = 0.0,
        process_var: float = 0.0,
        acceleration: float = 0.0,
    ) -> None:
        """
        Initialize dog simulation with position, velocity, and noise.

        :param x0: initial position
        :param velocity: velocity (+=right, -=left)
        :param measurement_var: variance in measurement m^2
        :param process_var: variance in process (m/s)^2
        :param acceleration: acceleration (m/s^2), how fast the velocity of the
            dog changes
        """
        self.x = x0
        self.velocity = velocity
        self.acceleration = acceleration
        #
        self.measurement_std = math.sqrt(measurement_var)
        self.process_std = math.sqrt(process_var)

    def move(self, *, dt: float = 1.0) -> None:
        """
        Compute new position of the dog in dt seconds.

        :param dt: time step in seconds
        """
        # Compute the change in position due to movement and noise.
        dx = self.velocity + randn() * self.process_std
        # Update the position.
        self.x += dx * dt
        # Update the velocity.
        self.velocity += self.acceleration * dt

    def sense_position(self) -> float:
        """
        Return measurement of new position in meters.

        :return: noisy measurement of current position
        """
        measurement = self.x + randn() * self.measurement_std
        return measurement

    def move_and_sense(self) -> Tuple[float, float]:
        """
        Move dog and return measurement and actual position.

        :return: tuple of (measurement, actual_pos) where measurement is the
            noisy sensor reading and actual_pos is the true dog position
        """
        self.move()
        measurement = self.sense_position()
        return measurement, self.x


def update(prior: Gaussian, likelihood: Gaussian) -> Gaussian:
    """
    Compute posterior given prior and likelihood.

    :param prior: prior Gaussian
    :param likelihood: likelihood Gaussian
    :return: posterior Gaussian
    """
    posterior = gaussian_multiply(prior, likelihood)
    return posterior


def predict(pos: Gaussian, movement: Gaussian) -> Gaussian:
    """
    Predict next position given current position and movement.

    :param pos: current position Gaussian
    :param movement: movement Gaussian
    :return: predicted position Gaussian
    """
    # Sum the position and movement Gaussians to get the predicted position.
    # Adding Gaussians: new mean = pos.mean + movement.mean,
    # new variance = pos.var + movement.var.
    return gaussian_sum(pos, movement)


# Prior, Measurement, Actual Position, Posterior.
KfInfo = namedtuple("KfInfo", ["prior", "measurement", "actual_pos", "posterior"])


def kf_info_to_df(info: List[KfInfo]) -> pd.DataFrame:
    """
    Convert a list of KfInfo tuples to a DataFrame.

    The output format is:

    ```
    PREDICT          UPDATE
    x      var       z       actual_pos    x      var
    1.000  401.000   1.354   1.234         1.352  1.990
    2.352    2.990   1.882   2.345         2.070  1.198
    ```

    :param info: list of KfInfo named tuples with prior, measurement,
        actual_pos, and posterior fields
    :return: DataFrame with columns predict_x, predict_var, z, actual_pos,
        update_x, update_var
    """
    rows = []
    for kf in info:
        rows.append(
            {
                "predict_x": kf.prior.mean,
                "predict_var": kf.prior.var,
                "z": kf.measurement,
                "actual_pos": kf.actual_pos,
                "update_x": kf.posterior.mean,
                "update_var": kf.posterior.var,
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "predict_x",
            "predict_var",
            "z",
            "actual_pos",
            "update_x",
            "update_var",
        ],
    )
    return df


def plot_kf_info(
    info: List[KfInfo],
    *,
    ylim: Optional[Tuple[float, float]] = None,
    show_actual_pos: str = "none",
    show_prior: str = "scatter",
    show_measurements: str = "scatter",
    show_posterior: str = "scatter",
) -> None:
    """
    Plot Kalman filter info with measurements, prior, and posterior over time.

    For each time step, the prior (predict) is plotted as a red up-triangle,
    the measurement as a black circle, and the posterior (update) as a green
    down-triangle (or line if show_posterior="line"). Posterior uncertainty
    is shown as shaded bands for 1, 2, and 3 standard deviations.

    :param info: list of KfInfo named tuples with prior, measurement,
        actual_pos, and posterior fields
    :param ylim: y-axis limits as (ymin, ymax); if None, matplotlib auto-scales
    :param show_prior: how to plot the prior — "scatter", "line", or "none"
    :param show_actual_pos: how to plot actual dog position — "scatter",
        "line", or "none"
    :param show_posterior: how to plot the posterior — "scatter", "line",
        or "none"
    """
    times = list(range(len(info)))
    predict_xs = [kf.prior.mean for kf in info]
    update_xs = [kf.posterior.mean for kf in info]
    update_stds = [np.sqrt(kf.posterior.var) for kf in info]
    zs = [kf.measurement for kf in info]
    actual_positions = [kf.actual_pos for kf in info]
    _, ax = plt.subplots(figsize=(12, 6))
    # Optionally plot prior (predict) as red up-triangle or line.
    if show_prior == "scatter":
        ax.scatter(
            times,
            predict_xs,
            marker="^",
            color="red",
            zorder=5,
            label="Prior (predict)",
            s=100,
        )
    elif show_prior == "line":
        ax.plot(
            times,
            predict_xs,
            color="red",
            label="Prior (predict)",
        )
    elif show_prior == "none":
        pass
    else:
        raise ValueError(f"Invalid show_prior='{show_prior}'")
    # Plot measurement as black circle.
    if show_measurements == "scatter":
        ax.scatter(
            times,
            zs,
            marker=".",
            color="black",
            zorder=5,
            label="Measurement (z)",
            s=100,
        )
    elif show_measurements == "none":
        pass
    else:
        raise ValueError(f"Invalid show_measurements='{show_measurements}'")
    # Optionally plot actual dog position as blue star or line.
    if show_actual_pos == "scatter":
        ax.scatter(
            times,
            actual_positions,
            marker="*",
            color="blue",
            zorder=5,
            label="Actual position",
            s=100,
        )
    elif show_actual_pos == "line":
        ax.plot(
            times,
            actual_positions,
            color="blue",
            label="Actual position",
        )
    elif show_actual_pos == "none":
        pass
    else:
        raise ValueError(f"Invalid show_actual_pos='{show_actual_pos}'")
    # Plot posterior (update) as green down-triangle or line.
    if show_posterior == "scatter":
        ax.scatter(
            times,
            update_xs,
            marker="v",
            color="green",
            zorder=5,
            label="Posterior (update)",
            s=100,
        )
    elif show_posterior == "line":
        ax.plot(
            times,
            update_xs,
            color="green",
            label="Posterior (update)",
        )
    elif show_posterior == "none":
        pass
    else:
        raise ValueError(f"Invalid show_posterior='{show_posterior}'")
    # Plot posterior uncertainty as shaded bands for 1, 2, and 3 std devs.
    for n_std, alpha in [(3, 0.10), (2, 0.15), (1, 0.20)]:
        lower = [x - n_std * s for x, s in zip(update_xs, update_stds)]
        upper = [x + n_std * s for x, s in zip(update_xs, update_stds)]
        ax.fill_between(
            times,
            lower,
            upper,
            alpha=alpha,
            color="green",
            label=f"Posterior ±{n_std}σ",
        )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position")
    ax.set_title("Kalman Filter: Predictions, Updates, and Measurements")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()


# #############################################################################
# Cell 2: Interactive Dog Simulation
# #############################################################################


def _run_dog_simulation(
    seed: int,
    process_var: float,
    sensor_var: float,
    initial_position: float,
    *,
    actual_initial_pos: Optional[float] = None,
    initial_pos_var: float = 400.0,
    acceleration: float = 0.0,
    n_steps: int = 10,
) -> List[KfInfo]:
    """
    Run dog simulation and Kalman filter with given parameters.

    :param seed: random seed for reproducibility
    :param process_var: variance in the dog's movement
    :param sensor_var: variance in the sensor
    :param initial_position: dog's initial position
    :param initial_pos_var: initial position variance (default 400.0 = 20^2)
    :param acceleration: dog's acceleration (default 0.0)
    :param n_steps: number of simulation steps
    :return: list of KfInfo named tuples with prior, measurement, actual_pos,
        and posterior fields
    """
    np.random.seed(seed)
    x = Gaussian(initial_position, initial_pos_var)
    velocity = 1.0
    dt = 1.0
    process_model = Gaussian(velocity * dt, process_var)
    if actual_initial_pos is None:
        actual_initial_pos = x.mean
    dog = DogSimulation(
        x0=actual_initial_pos,
        velocity=process_model.mean,
        measurement_var=sensor_var,
        process_var=process_model.var,
        acceleration=acceleration,
    )
    info = []
    for _ in range(n_steps):
        measurement, actual_pos = dog.move_and_sense()
        prior = predict(x, process_model)
        likelihood = Gaussian(measurement, sensor_var)
        x = update(prior, likelihood)
        info.append(
            KfInfo(
                prior=prior,
                measurement=measurement,
                actual_pos=actual_pos,
                posterior=x,
            )
        )
    return info


def cell2_interactive_dog_simulation() -> None:
    """
    Create interactive widget for exploring Dog simulation with Kalman filter.

    Allows user to adjust seed, process_var, sensor_var, initial_position,
    actual_initial_pos, initial_pos_var, and acceleration and see the Kalman
    filter results plotted over time.
    """
    fig_dog = None

    def _plot_simulation(
        seed: int,
        process_var: float,
        sensor_var: float,
        initial_position: float,
        actual_initial_pos: float,
        initial_pos_var: float,
        acceleration: float,
    ) -> None:
        """
        Run and plot dog simulation with given parameters.

        :param seed: random seed for reproducibility
        :param process_var: variance in the dog's movement
        :param sensor_var: variance in the sensor
        :param initial_position: belief about dog's initial position
        :param actual_initial_pos: actual initial position of the dog
        :param initial_pos_var: variance of the initial position belief
        :param acceleration: dog's acceleration (m/s^2)
        """
        nonlocal fig_dog
        if fig_dog is not None:
            plt.close(fig_dog)
        num_steps = 25
        info = _run_dog_simulation(
            seed,
            process_var,
            sensor_var,
            initial_position,
            actual_initial_pos=actual_initial_pos,
            initial_pos_var=initial_pos_var,
            acceleration=acceleration,
            n_steps=num_steps,
        )
        plot_kf_info(info, show_actual_pos="line")
        fig_dog = plt.gcf()

    # Create process_var widget.
    process_var_slider, process_var_box = mtumsuti.build_widget_control(
        name="process_var",
        description="process_var",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=1.0,
        is_float=True,
    )
    # Create sensor_var widget.
    sensor_var_slider, sensor_var_box = mtumsuti.build_widget_control(
        name="sensor_var",
        description="sensor_var",
        min_val=0.1,
        max_val=10.0,
        step=0.1,
        initial_value=2.0,
        is_float=True,
    )
    # Create initial_position widget (belief about where the dog starts).
    initial_position_slider, initial_position_box = mtumsuti.build_widget_control(
        name="initial_position",
        description="initial_position",
        min_val=-50.0,
        max_val=500.0,
        step=1.0,
        initial_value=0.0,
        is_float=True,
    )
    # Create actual_initial_pos widget (true starting position of the dog).
    actual_initial_pos_slider, actual_initial_pos_box = mtumsuti.build_widget_control(
        name="actual_initial_pos",
        description="actual_initial_pos",
        min_val=-50.0,
        max_val=500.0,
        step=1.0,
        initial_value=0.0,
        is_float=True,
    )
    # Create initial_pos_var widget (uncertainty in initial position belief).
    initial_pos_var_slider, initial_pos_var_box = mtumsuti.build_widget_control(
        name="initial_pos_var",
        description="initial_pos_var",
        min_val=1.0,
        max_val=500.0,
        step=1.0,
        initial_value=400.0,
        is_float=True,
    )
    # Create acceleration widget (how fast the dog's velocity changes).
    acceleration_slider, acceleration_box = mtumsuti.build_widget_control(
        name="acceleration",
        description="acceleration",
        min_val=0.0,
        max_val=0.2,
        step=0.01,
        initial_value=0.0,
        is_float=True,
    )
    # Create seed widget last (per convention, seed is always the last widget).
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=13,
        is_float=False,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot_simulation,
        {
            "seed": seed_slider,
            "process_var": process_var_slider,
            "sensor_var": sensor_var_slider,
            "initial_position": initial_position_slider,
            "actual_initial_pos": actual_initial_pos_slider,
            "initial_pos_var": initial_pos_var_slider,
            "acceleration": acceleration_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                process_var_box,
                sensor_var_box,
                initial_position_box,
                actual_initial_pos_box,
                initial_pos_var_box,
                acceleration_box,
                seed_box,
                output,
            ]
        )
    )