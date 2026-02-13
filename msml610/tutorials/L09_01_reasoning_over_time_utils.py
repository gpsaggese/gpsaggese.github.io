"""
Utility functions for reasoning over time tutorial (L09_01).

Import as:

import msml610.tutorials.L09_01_reasoning_over_time_utils as mturetium
"""

import logging
from typing import List, Optional, Tuple, Union

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

import helpers.hdbg as hdbg
import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)


# #############################################################################


def plot_gh_filter_results(
    measurements: np.ndarray,
    preds: List[float],
    ests: List[float],
    ground_truth: List[float],
    *,
    tag_measurements: str = "measurements",
) -> None:
    """
    Plot g-h filter results with measurements, predictions, and estimates.

    :param measurements: Actual measurements
    :param preds: Predicted values
    :param ests: Estimated values
    :param ground_truth: True values
    :param tag_measurements: Label for measurements in plot
    """
    idx = pd.date_range("2011-01-01", periods=len(measurements))
    df = pd.DataFrame(measurements.T, index=idx, columns=[tag_measurements])
    linewidth = 2
    if preds is not None:
        df["pred"] = preds
    df["ests"] = ests
    df["ground_truth"] = ground_truth
    # Measurements as points.
    df["measurements"].plot(marker="o", markersize=10, linestyle="None")
    # Ground truth line.
    df["ground_truth"].plot(color="k", linewidth=linewidth)
    # Predictions as dashed line.
    if preds is not None:
        df["pred"].plot(color="r", linewidth=linewidth, linestyle="--")
    # Estimates as solid line.
    df["ests"].plot(color="b", linewidth=linewidth)
    plt.legend(loc="upper left")


# TODO(ai_gp): -> plot_gh_filter_results_with_params
def plot_prediction_with_params(
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
    # Call the base plotting function.
    plot_gh_filter_results(
        measurements,
        preds,
        ests,
        ground_truth,
        tag_measurements=tag_measurements,
    )
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


def create_interactive_gain_rate_widget(
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
        fig_gain = plt.figure(figsize=(10, 5))
        time_step = 1
        ests, preds = predict_using_gain_guess(
            weight, measured_weights, gain_rate, weight_scale, time_step
        )
        params = {
            "initial_weight": weight,
            "weight_scale": weight_scale,
            "gain_rate": gain_rate,
        }
        plot_prediction_with_params(
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


# #############################################################################
# Discrete Bayes Filter.
# #############################################################################


def plot_dog_in_office_pdf(
    probs: Union[List[float], np.ndarray],
    *,
    hallway: Optional[np.ndarray] = None,
    title: str = "Probability Histogram",
) -> None:
    """
    Plot histogram-like bar chart of class probabilities with door markers.

    :param probs: List or array of class probabilities
    :param hallway: Binary array marking door positions
    :param title: Title for the plot
    """
    hdbg.dassert_isinstance(probs, (list, np.ndarray))
    hdbg.dassert_lte(0.0, np.min(probs))
    hdbg.dassert_lte(np.max(probs), 1.0)
    # Check that the sum of probabilities is 1.0.
    hdbg.dassert_lte(0.99, np.sum(probs))
    hdbg.dassert_lte(np.sum(probs), 1.01)
    #
    indices = np.arange(len(probs))
    if hallway is None:
        hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    hdbg.dassert_eq(len(probs), len(hallway))
    # Create plot.
    plt.bar(indices, probs, color="deepskyblue")
    # Add markers for hallway positions with value 1.
    for i, val in enumerate(hallway):
        if val == 1:
            label = "Door" if i == 0 else ""
            plt.plot(i, 0.0, "^r", markersize=20, label=label)
    plt.ylim(0, 1)
    plt.xlabel("Class Index")
    plt.ylabel("Probability")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()
