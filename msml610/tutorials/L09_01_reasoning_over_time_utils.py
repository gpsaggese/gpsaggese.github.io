"""
Utility functions for reasoning over time tutorial (L09_01).

Import as:

import msml610.tutorials.L09_01_reasoning_over_time_utils as mturetium
"""

import logging
import os
from typing import List, Optional, Tuple, Union

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython.display import display

import helpers.hdbg as hdbg
import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)


# #############################################################################


def plot_gh_filter_results_with_params(
    measurements: np.ndarray,
    preds: List[float],
    ests: List[float],
    ground_truth: List[float],
    params: dict,
    *,
    tag_measurements: str = "measurements",
) -> None:
    """
    Plot prediction results with parameter information in the legend.

    :param measurements: Actual measurements
    :param preds: Predicted values
    :param ests: Estimated values
    :param ground_truth: True values
    :param params: Dictionary of parameters to display (e.g., initial_weight,
        gain_rate, scale_factor)
    :param tag_measurements: Label for measurements in plot
    """
    # Create dataframe with measurements and results.
    idx = pd.date_range("2011-01-01", periods=len(measurements))
    df = pd.DataFrame(measurements.T, index=idx, columns=[tag_measurements])
    linewidth = 2
    if preds is not None:
        df["pred"] = preds
    df["ests"] = ests
    df["ground_truth"] = ground_truth
    # Measurements as points.
    df["measurements"].plot(marker=".", markersize=10, linestyle="None")
    # Ground truth line.
    df["ground_truth"].plot(color="k", linewidth=linewidth)
    # Predictions as dashed line.
    if preds is not None:
        df["pred"].plot(color="b", linewidth=linewidth, linestyle="--")
    # Estimates as solid line.
    df["ests"].plot(color="r", linewidth=linewidth)
    plt.legend(loc="upper left")
    # Add parameter information to the title.
    param_str = ", ".join(
        [f"{key}: {value:.4g}" for key, value in params.items()]
    )
    plt.title(f"Parameters: {param_str}", fontsize=10)


# #############################################################################
# Cell 1
# #############################################################################


def predict_using_gain_guess(
    initial_weight: float,
    measures: List[float],
    gain_rate: float,
    scale_factor: float,
    time_step: float,
) -> Tuple[List[float], List[float]]:
    """
    Predict weight using gain guess model.

    :param estimated_weight: Initial estimated weight
    :param measures: List of weight measurements
    :param gain_rate: Rate of weight gain
    :param scale_factor: Scale factor for blending prediction and
        measurement
    :param time_step: Time step between predictions
    :return: Tuple of (estimated weights, predicted weights)
    """
    ests = []
    preds = []
    for z in measures:
        # Predict using the internal model.
        predicted_weight = initial_weight + gain_rate * time_step
        # Update by blending prediction and measurement.
        initial_weight = predicted_weight + scale_factor * (
            z - predicted_weight
        )
        # Log values.
        ests.append(initial_weight)
        preds.append(predicted_weight)
        _LOG.debug(
            "z=%.2f pred=%.2f est=%.2f", z, predicted_weight, initial_weight
        )
    return ests, preds


def cell1_2_plot_gh_filter_with_known_gain_rate(
    measured_weights: np.ndarray,
    ground_truth: np.ndarray,
    params: dict,
    dst_dir: str,
    dst_filename: str,
) -> None:
    """
    Plot gain rate prediction with known gain rate.

    :param measured_weights: Array of weight measurements
    :param ground_truth: Array of true weight values
    :param params: Dictionary of parameters to display
    :param dst_dir: Directory to save output figure
    :param dst_filename: Filename for the output figure
    """
    ests, preds = predict_using_gain_guess(
        params["initial_weight"],
        measured_weights,
        params["gain_rate"],
        params["weight_scale"],
        params["time_step"],
    )
    plot_gh_filter_results_with_params(
        measured_weights, preds, ests, ground_truth, params
    )
    plt.savefig(os.path.join(dst_dir, dst_filename))


def cell1_3_plot_gh_filter_with_known_gain_rate(
    measured_weights: np.ndarray,
    ground_truth: np.ndarray,
    params: dict,
    dst_dir: str,
    dst_filename: str,
) -> None:
    """
    Plot gain rate prediction with wrong gain rate guess.

    Wrapper for cell1_2_plot_gh_filter_with_known_gain_rate.

    :param measured_weights: Array of weight measurements
    :param ground_truth: Array of true weight values
    :param params: Dictionary of parameters to display
    :param dst_dir: Directory to save output figure
    :param dst_filename: Filename for the output figure
    """
    cell1_2_plot_gh_filter_with_known_gain_rate(
        measured_weights, ground_truth, params, dst_dir, dst_filename
    )


# def cell_1_3_wrong_guess_gain_rate(
#     measured_weights: np.ndarray,
#     ground_truth: np.ndarray,
#     dst_dir: str,
# ) -> None:
#     """
#     Plot gain rate prediction with wrong gain rate guess.

#     :param measured_weights: Array of weight measurements
#     :param ground_truth: Array of true weight values
#     :param dst_dir: Directory to save output figure
#     """

#     #
#     ests, preds = predict_using_gain_guess(
#         params["initial_weight"],
#         measured_weights,
#         params["gain_rate"],
#         params["weight_scale"],
#         params["time_step"],
#     )
#     plot_gh_filter_results_with_params(
#         measured_weights, preds, ests, ground_truth, params
#     )
#     plt.savefig(os.path.join(dst_dir, "L09_04_wrong_gain_rate.png"))


def cell1_4_create_interactive_gain_rate_widget(
    measured_weights: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    """
    Create interactive widget for exploring gain rate prediction parameters.

    :param measured_weights: Array of weight measurements
    :param ground_truth: Array of true weight values
    """
    fig_gain = None

    def _plot_gain_rate_prediction(
        weight: float, weight_scale: float, gain_rate: float
    ) -> None:
        """
        Plot gain rate prediction with given parameters.

        :param weight: Initial weight estimate
        :param weight_scale: Scale factor for blending prediction and
            measurement
        :param gain_rate: Rate of weight gain per time step
        """
        nonlocal fig_gain
        if fig_gain is not None:
            plt.close(fig_gain)
        fig_gain = plt.figure(figsize=plt.rcParams["figure.figsize"])
        time_step = 1
        ests, preds = predict_using_gain_guess(
            weight, measured_weights, gain_rate, weight_scale, time_step
        )
        params = {
            "initial_weight": weight,
            "weight_scale": weight_scale,
            "gain_rate": gain_rate,
        }
        plot_gh_filter_results_with_params(
            measured_weights, preds, ests, ground_truth, params
        )

    # Create slider for initial weight.
    weight_slider, weight_box = mtumsuti.build_widget_control(
        name="weight",
        description="",
        min_val=100.0,
        max_val=200.0,
        step=1.0,
        initial_value=160.0,
        is_float=True,
    )
    # Create slider for weight scale.
    weight_scale_slider, weight_scale_box = mtumsuti.build_widget_control(
        name="weight_scale",
        description="",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.4,
        is_float=True,
    )
    # Create slider for gain rate.
    gain_rate_slider, gain_rate_box = mtumsuti.build_widget_control(
        name="gain_rate",
        description="",
        min_val=-20.0,
        max_val=20.0,
        step=0.5,
        initial_value=1.0,
        is_float=True,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot_gain_rate_prediction,
        {
            "weight": weight_slider,
            "weight_scale": weight_scale_slider,
            "gain_rate": gain_rate_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox([weight_box, weight_scale_box, gain_rate_box, output])
    )


def cell1_5_plot_gh_filter_with_learning_gain_rate(
    measured_weights: np.ndarray,
    ground_truth: np.ndarray,
    params: dict,
    dst_dir: str,
    dst_filename: str,
) -> None:
    """
    Plot gain rate prediction while learning the gain rate.

    :param measured_weights: Array of weight measurements
    :param ground_truth: Array of true weight values
    :param params: Dictionary of parameters to display
    :param dst_dir: Directory to save output figure
    :param dst_filename: Filename for the output figure
    """
    ests, preds = predict_learning_gain_rate(
        params["initial_weight"],
        measured_weights,
        params["gain_rate"],
        params["weight_scale"],
        params["gain_scale"],
        params["time_step"],
    )
    plot_gh_filter_results_with_params(
        measured_weights, preds, ests, ground_truth, params
    )
    plt.savefig(os.path.join(dst_dir, dst_filename))


def predict_learning_gain_rate(
    weight: float,
    measures: np.ndarray,
    gain_rate: float,
    weight_factor: float,
    gain_scale: float,
    time_step: float,
) -> Tuple[List[float], List[float]]:
    """
    Predict learning gain rate using Kalman filter approach.

    :param weight: Initial weight
    :param measures: Array of weight measurements
    :param gain_rate: Initial gain rate
    :param weight_factor: Weight update factor
    :param gain_scale: Gain scale factor
    :param time_step: Time step between measurements
    :return: Tuple of (estimated weights, predicted weights)
    """
    ests = []
    preds = []
    for z in measures:
        # Predict step.
        weight = weight + gain_scale * time_step
        preds.append(weight)
        # Update step.
        residual = z - weight
        gain_rate = gain_rate + gain_scale * residual / time_step
        weight = weight + weight_factor * residual
        ests.append(weight)
        _LOG.debug("z=%.2f pred=%.2f weight=%.2f", z, weight, weight)
    return ests, preds


def gh_filter(
    data: np.ndarray,
    x0: float,
    dx: float,
    g: float,
    h: float,
    *,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Perform g-h filter on 1 state variable with a fixed g and h.

    :param data: Measurements to filter
    :param x0: Initial value for state variable
    :param dx: Initial change rate for state variable
    :param g: Scale factor to blend prediction and measurement
    :param h: Scale factor to update change rate
    :param dt: Time step between measurements
    :return: Array of filtered estimates
    """
    x_est = x0
    results = []
    for z in data:
        # Predict step.
        x_pred = x_est + (dx * dt)
        # Update step.
        residual = z - x_pred
        dx = dx + h * (residual / dt)
        x_est = x_pred + g * residual
        results.append(x_est)
        _LOG.debug("z=%.2f pred=%.2f est=%.2f", z, x_pred, x_est)
    return np.array(results)


# #############################################################################
# Cell 2
# #############################################################################


def gen_linear_noisy_data(
    x0: float,
    dx: float,
    count: int,
    noise_factor: float,
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, List[float]]:
    """
    Generate random data with additive noise N(0, noise_factor).

    Starting from x0, with slope dx.

    :param x0: Initial value
    :param dx: Slope
    :param count: Number of points to generate
    :param noise_factor: Standard deviation of Gaussian noise
    :param seed: Random seed for reproducibility
    :return: Tuple of (noisy data array, ground truth values)
    """
    np.random.seed(seed)
    vals = [
        x0 + (dx * i) + np.random.randn() * noise_factor for i in range(count)
    ]
    ground_truth = [x0 + dx * i for i in range(count)]
    return np.array(vals), ground_truth


def gen_non_linear_noisy_data(
    x0: float,
    dx: float,
    count: int,
    noise_factor: float,
    accel: float,
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random data with acceleration and additive noise.

    Starting from x0, with initial slope dx, affected by random noise
    N(0, noise_factor).

    :param x0: Initial value
    :param dx: Initial slope
    :param count: Number of points to generate
    :param noise_factor: Standard deviation of Gaussian noise
    :param accel: Acceleration factor
    :param seed: Random seed for reproducibility
    :return: Tuple of (noisy data array, ground truth values)
    """
    np.random.seed(seed)
    ground_truth = [x0 + (dx + i * accel) * i for i in range(count)]
    vals = [
        ground_truth[i] + np.random.randn() * noise_factor for i in range(count)
    ]
    return np.array(vals), np.array(ground_truth)


def cell2_2_plot_gh_filter_with_params(params: dict) -> None:
    """
    Demonstrate g-h filter with correct initial guesses.

    Shows how the filter performs when starting values match the true
    system parameters.

    :param params: Dictionary of filter parameters (x0, dx, dt, g, h)
    """
    vals, ground_truth = gen_linear_noisy_data(
        x0=0, dx=1, count=100, noise_factor=10
    )
    ests = gh_filter(
        data=vals,
        x0=params["x0"],
        dx=params["dx"],
        dt=params["dt"],
        g=params["g"],
        h=params["h"],
    )
    preds = None
    plot_gh_filter_results_with_params(vals, preds, ests, ground_truth, params)


def cell2_3_plot_gh_filter_with_params(params: dict) -> None:
    """
    Demonstrate g-h filter with wrong initial guesses.

    Wrapper for cell2_2_plot_gh_filter_with_params.

    :param params: Dictionary of filter parameters (x0, dx, dt, g, h)
    """
    cell2_2_plot_gh_filter_with_params(params)


def cell2_4_extreme_noise() -> None:
    """
    Demonstrate g-h filter performance with extreme noise.

    Shows how the filter handles measurements with very high noise levels.
    """
    vals, ground_truth = gen_linear_noisy_data(
        x0=0, dx=1, count=100, noise_factor=100
    )
    params = {
        "x0": 100,
        "dx": 1,
        "dt": 1,
        "g": 0.1,
        "h": 0.02,
    }
    ests = gh_filter(
        data=vals,
        x0=params["x0"],
        dx=params["dx"],
        dt=params["dt"],
        g=params["g"],
        h=params["h"],
    )
    preds = None
    plot_gh_filter_results_with_params(vals, preds, ests, ground_truth, params)


def cell2_6_non_linear_gh_filter() -> None:
    """
    Demonstrate g-h filter on non-linear data.

    Shows filter performance when the underlying system has acceleration,
    violating the constant velocity assumption.
    """
    vals, ground_truth = gen_non_linear_noisy_data(
        x0=0, dx=1, count=20, noise_factor=100, accel=5
    )
    params = {
        "x0": 100,
        "dx": 1,
        "dt": 1,
        "g": 0.1,
        "h": 0.02,
    }
    ests = gh_filter(
        data=vals,
        x0=params["x0"],
        dx=params["dx"],
        dt=params["dt"],
        g=params["g"],
        h=params["h"],
    )
    preds = None
    plot_gh_filter_results_with_params(vals, preds, ests, ground_truth, params)


def cell2_1_create_interactive_linear_noisy_data_widget() -> None:
    """
    Create interactive widget for visualizing linear noisy data generation.

    Allows user to interactively adjust parameters to see how they affect
    the generated data and ground truth.
    """
    fig_noisy = None

    def _plot_linear_noisy_data(
        seed: int,
        count: int,
        noise_factor: float,
    ) -> None:
        """
        Plot linear noisy data with ground truth.

        :param seed: Random seed for reproducibility
        :param count: Number of points to generate
        :param noise_factor: Standard deviation of Gaussian noise
        """
        nonlocal fig_noisy
        if fig_noisy is not None:
            plt.close(fig_noisy)
        fig_noisy = plt.figure(figsize=plt.rcParams["figure.figsize"])
        # Use fixed values for x0 and dx.
        vals, ground_truth = gen_linear_noisy_data(
            x0=0, dx=1, count=count, noise_factor=noise_factor, seed=seed
        )
        df = pd.DataFrame({
            "measurements": vals,
            "ground_truth": ground_truth,
        })
        df["measurements"].plot(marker=".", markersize=10, linestyle="None")
        df["ground_truth"].plot(color="k", linewidth=2)
        plt.xlim(-10, 110)
        plt.ylim(-10, 110)
        plt.legend(loc="upper left")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title("Linear Noisy Data Generation")
        plt.grid(True, alpha=0.3)

    # Create seed widget (first widget per convention).
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="Random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    # Create count widget.
    count_slider, count_box = mtumsuti.build_widget_control(
        name="count",
        description="Number of points",
        min_val=10,
        max_val=200,
        step=10,
        initial_value=100,
        is_float=False,
    )
    # Create noise_factor widget.
    noise_factor_slider, noise_factor_box = mtumsuti.build_widget_control(
        name="noise_factor",
        description="Noise std dev",
        min_val=0.0,
        max_val=20.0,
        step=0.5,
        initial_value=5.0,
        is_float=True,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot_linear_noisy_data,
        {
            "seed": seed_slider,
            "count": count_slider,
            "noise_factor": noise_factor_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [seed_box, count_box, noise_factor_box, output]
        )
    )


def cell2_9_create_interactive_gh_filter_widget() -> None:
    """
    Create interactive widget for exploring g-h filter parameters.

    Allows user to interactively adjust initial state (x, dx), filter
    gains (g, h), and noise level to see their effect on filtering noisy
    linear data.
    """
    fig_gh = None

    def _plot_gh_filter(
        x: float,
        dx: float,
        g: float,
        h: float,
        noise_factor: float,
    ) -> None:
        """
        Plot g-h filter results with given parameters.

        :param x: Initial state estimate
        :param dx: Initial rate of change estimate
        :param g: Scale factor to blend prediction and measurement
        :param h: Scale factor to update rate of change
        :param noise_factor: Standard deviation of Gaussian noise
        """
        nonlocal fig_gh
        if fig_gh is not None:
            plt.close(fig_gh)
        fig_gh = plt.figure(figsize=plt.rcParams["figure.figsize"])
        # Generate test data with current noise level.
        zs, ground_truth = gen_linear_noisy_data(
            x0=5, dx=5, count=100, noise_factor=noise_factor
        )
        # Apply g-h filter.
        data = gh_filter(data=zs, x0=x, dx=dx, g=g, h=h)
        # Plot ground truth as black line.
        plt.plot(ground_truth, color="k", linewidth=2, label="Ground truth")
        # Plot measurements as scatter.
        plt.scatter(list(range(len(zs))), zs, marker=".", lw=1, label="Measurements")
        # Plot filtered estimates as line.
        plt.plot(data, color="b", label="Filtered estimates")
        plt.legend(loc="upper left")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title("g-h Filter Interactive Example")
        plt.grid(True, alpha=0.3)

    # Create x widget.
    x_slider, x_box = mtumsuti.build_widget_control(
        name="x",
        description="Initial state",
        min_val=-200.0,
        max_val=2000.0,
        step=10.0,
        initial_value=0.0,
        is_float=True,
    )
    # Create dx widget.
    dx_slider, dx_box = mtumsuti.build_widget_control(
        name="dx",
        description="Initial rate",
        min_val=-50.0,
        max_val=50.0,
        step=0.5,
        initial_value=5.0,
        is_float=True,
    )
    # Create g widget.
    g_slider, g_box = mtumsuti.build_widget_control(
        name="g",
        description="State gain",
        min_val=0.01,
        max_val=2.0,
        step=0.02,
        initial_value=0.1,
        is_float=True,
    )
    # Create h widget.
    h_slider, h_box = mtumsuti.build_widget_control(
        name="h",
        description="Rate gain",
        min_val=0.0,
        max_val=0.5,
        step=0.01,
        initial_value=0.02,
        is_float=True,
    )
    # Create noise_factor widget.
    noise_factor_slider, noise_factor_box = mtumsuti.build_widget_control(
        name="noise_factor",
        description="Noise std dev",
        min_val=0.0,
        max_val=100.0,
        step=5.0,
        initial_value=50.0,
        is_float=True,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot_gh_filter,
        {
            "x": x_slider,
            "dx": dx_slider,
            "g": g_slider,
            "h": h_slider,
            "noise_factor": noise_factor_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox([x_box, dx_box, g_box, h_box, noise_factor_box, output])
    )


def cell2_5_create_interactive_non_linear_noisy_data_widget() -> None:
    """
    Create interactive widget for visualizing non-linear noisy data.

    Allows user to interactively adjust parameters including acceleration
    to see how they affect the generated data and ground truth.
    """
    fig_non_linear = None

    def _plot_non_linear_noisy_data(
        seed: int,
        count: int,
        noise_factor: float,
        accel: float,
    ) -> None:
        """
        Plot non-linear noisy data with ground truth.

        :param seed: Random seed for reproducibility
        :param count: Number of points to generate
        :param noise_factor: Standard deviation of Gaussian noise
        :param accel: Acceleration factor
        """
        nonlocal fig_non_linear
        if fig_non_linear is not None:
            plt.close(fig_non_linear)
        fig_non_linear = plt.figure(figsize=plt.rcParams["figure.figsize"])
        # Use fixed values for x0 and dx.
        vals, ground_truth = gen_non_linear_noisy_data(
            x0=0,
            dx=1,
            count=count,
            noise_factor=noise_factor,
            accel=accel,
            seed=seed,
        )
        # Plot ground truth as line.
        pd.Series(ground_truth).plot(color="k", linewidth=2, label="Ground truth")
        # Plot measurements as scatter points.
        pd.Series(vals).plot(
            marker="o",
            markersize=6,
            linestyle="None",
            color="b",
            label="Measurements",
        )
        plt.legend(loc="upper left")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title("Non-Linear Noisy Data Generation")
        plt.grid(True, alpha=0.3)

    # Create seed widget (first widget per convention).
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="Random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    # Create count widget.
    count_slider, count_box = mtumsuti.build_widget_control(
        name="count",
        description="Number of points",
        min_val=10,
        max_val=200,
        step=10,
        initial_value=100,
        is_float=False,
    )
    # Create noise_factor widget.
    noise_factor_slider, noise_factor_box = mtumsuti.build_widget_control(
        name="noise_factor",
        description="Noise std dev",
        min_val=0.0,
        max_val=2000.0,
        step=50.0,
        initial_value=1000.0,
        is_float=True,
    )
    # Create accel widget.
    accel_slider, accel_box = mtumsuti.build_widget_control(
        name="accel",
        description="Acceleration",
        min_val=-10.0,
        max_val=10.0,
        step=0.5,
        initial_value=5.0,
        is_float=True,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot_non_linear_noisy_data,
        {
            "seed": seed_slider,
            "count": count_slider,
            "noise_factor": noise_factor_slider,
            "accel": accel_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                seed_box,
                count_box,
                noise_factor_box,
                accel_box,
                output,
            ]
        )
    )


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
    cov_matrix = np.array([
        [sigma1**2, rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2**2],
    ])
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
    plt.figure(figsize=(12, 5))
    # Left plot: Individual Gaussians.
    plt.subplot(1, 2, 1)
    X_marginal = stats.norm(mu1, sigma1)
    Y_marginal = stats.norm(mu2, sigma2)
    plt.plot(
        x_range,
        X_marginal.pdf(x_range),
        label=f"X ~ N({mu1:.1f}, {sigma1:.1f}^2)",
        linewidth=2,
    )
    plt.plot(
        x_range,
        Y_marginal.pdf(x_range),
        label=f"Y ~ N({mu2:.1f}, {sigma2:.1f}^2)",
        linewidth=2,
    )
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title(f"Input Gaussians (rho={rho:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Right plot: Sum (analytical vs numerical).
    plt.subplot(1, 2, 2)
    plt.plot(
        x_range,
        Z_analytical.pdf(x_range),
        label=f"Analytical: N({mu_sum:.2f}, {sigma_sum:.2f}^2)",
        linewidth=2,
        color="red",
    )
    plt.hist(
        Z_samples,
        bins=50,
        density=True,
        alpha=0.5,
        label=f"Numerical ({n_samples} samples)",
        color="blue",
    )
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title("Sum: Z = X + Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def cell3_1_plot_gaussian_sum() -> None:
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
) -> None:
    """
    Plot the product of two Gaussians.

    :param mu1: Mean of first Gaussian
    :param sigma1: Standard deviation of first Gaussian
    :param mu2: Mean of second Gaussian
    :param sigma2: Standard deviation of second Gaussian
    """
    # Create scipy Gaussian distributions.
    X = stats.norm(mu1, sigma1)
    Y = stats.norm(mu2, sigma2)
    # Compute product analytically.
    mu_prod, sigma_prod = gaussian_product(mu1, sigma1, mu2, sigma2)
    Z_analytical = stats.norm(mu_prod, sigma_prod)
    # Compute product numerically (point-wise multiplication of PDFs).
    x_range = np.linspace(
        min(mu1 - 4 * sigma1, mu2 - 4 * sigma2, mu_prod - 4 * sigma_prod),
        max(mu1 + 4 * sigma1, mu2 + 4 * sigma2, mu_prod + 4 * sigma_prod),
        1000,
    )
    # Numerical product of PDFs (unnormalized).
    pdf_product_unnormalized = X.pdf(x_range) * Y.pdf(x_range)
    # Normalize the product.
    normalization = np.trapz(pdf_product_unnormalized, x_range)
    pdf_product_normalized = pdf_product_unnormalized / normalization
    # Plot.
    plt.figure(figsize=(12, 5))
    # Left plot: Individual Gaussians.
    plt.subplot(1, 2, 1)
    plt.plot(
        x_range,
        X.pdf(x_range),
        label=f"X ~ N({mu1:.1f}, {sigma1:.1f}^2)",
        linewidth=2,
    )
    plt.plot(
        x_range,
        Y.pdf(x_range),
        label=f"Y ~ N({mu2:.1f}, {sigma2:.1f}^2)",
        linewidth=2,
    )
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title("Input Gaussians")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Right plot: Product (analytical vs numerical).
    plt.subplot(1, 2, 2)
    plt.plot(
        x_range,
        Z_analytical.pdf(x_range),
        label=f"Analytical: N({mu_prod:.2f}, {sigma_prod:.2f}^2)",
        linewidth=2,
        color="red",
    )
    plt.plot(
        x_range,
        pdf_product_normalized,
        label="Numerical (normalized product)",
        linewidth=2,
        linestyle="--",
        color="blue",
    )
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title("Product: Z = X * Y (PDF multiplication)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def cell3_2_plot_gaussian_product() -> None:
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
    display(
        ipywidgets.VBox([mu1_box, sigma1_box, mu2_box, sigma2_box, output])
    )


