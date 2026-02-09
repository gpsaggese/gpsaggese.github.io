"""
Utility functions for L05_02_bias_variance notebook.

Import as:

import msml610.tutorials.L05_02_bias_variance_utils as mtl0bvaut
"""

import logging
from typing import Tuple

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)


# #############################################################################
# Target Function
# #############################################################################


def target_function(x: np.ndarray) -> np.ndarray:
    """
    Target function: f(x) = sin(pi * x).

    :param x: Input array
    :return: Output array
    """
    return np.sin(np.pi * x)


# #############################################################################
# Model Fitting Functions
# #############################################################################


def fit_constant_model(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fit a constant model: g_0(x) = b (horizontal line).

    The best constant is the mean of y values.

    :param x: Input x values
    :param y: Target y values
    :return: Constant value b
    """
    return np.mean(y)


def fit_linear_model(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit a linear model: g_1(x) = a*x + b.

    Uses least squares to find the best fit line.

    :param x: Input x values
    :param y: Target y values
    :return: Tuple (a, b) where a is slope and b is intercept
    """
    # Use numpy polyfit for linear regression (degree 1).
    coeffs = np.polyfit(x, y, deg=1)
    a = coeffs[0]  # Slope
    b = coeffs[1]  # Intercept
    return a, b


def compute_approximation_error(
    x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Compute mean squared error between true function and model.

    :param x: Input x values
    :param y_true: True function values
    :param y_pred: Predicted function values
    :return: Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)


# #############################################################################
# Helper Functions
# #############################################################################


def generate_training_data(
    n_samples: int,
    noise_std: float = 0.0,
    x_range: Tuple[float, float] = (-1, 1),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random training data from the target function with optional noise.

    :param n_samples: Number of training points to generate
    :param noise_std: Standard deviation of Gaussian noise (default 0.0)
    :param x_range: Range for uniform sampling of x values
    :return: Tuple of (x_train, y_train) arrays
    """
    x_train = np.random.uniform(x_range[0], x_range[1], n_samples)
    x_train = np.sort(x_train)
    y_train = target_function(x_train)
    if noise_std > 0:
        y_train = y_train + np.random.normal(0, noise_std, n_samples)
    return x_train, y_train


def fit_models_and_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_dense: np.ndarray
) -> Tuple[float, np.ndarray, Tuple[float, float], np.ndarray]:
    """
    Fit constant and linear models, generate predictions on dense grid.

    :param x_train: Training x values
    :param y_train: Training y values
    :param x_dense: Dense x values for prediction
    :return: Tuple of (b, y_const_dense, (a, b_linear), y_linear_dense)
    """
    # Fit constant model.
    b = fit_constant_model(x_train, y_train)
    y_const_dense = np.full_like(x_dense, b)

    # Fit linear model.
    a, b_linear = fit_linear_model(x_train, y_train)
    y_linear_dense = a * x_dense + b_linear

    return b, y_const_dense, (a, b_linear), y_linear_dense


def compute_all_errors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dense: np.ndarray,
    y_true: np.ndarray,
    y_const_train: np.ndarray,
    y_const_dense: np.ndarray,
    y_linear_train: np.ndarray,
    y_linear_dense: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute all error metrics (E_in and E_out for both models).

    :param x_train: Training x values
    :param y_train: Training y values
    :param x_dense: Dense x values
    :param y_true: True function values on dense grid
    :param y_const_train: Constant model predictions on training data
    :param y_const_dense: Constant model predictions on dense grid
    :param y_linear_train: Linear model predictions on training data
    :param y_linear_dense: Linear model predictions on dense grid
    :return: Tuple of (e_in_const, e_in_linear, e_out_const, e_out_linear)
    """
    # Compute in-sample error E_in (on training data).
    e_in_const = compute_approximation_error(x_train, y_train, y_const_train)
    e_in_linear = compute_approximation_error(x_train, y_train, y_linear_train)

    # Compute out-of-sample error E_out (on full dense grid).
    e_out_const = compute_approximation_error(x_dense, y_true, y_const_dense)
    e_out_linear = compute_approximation_error(x_dense, y_true, y_linear_dense)

    return e_in_const, e_in_linear, e_out_const, e_out_linear


def setup_model_comparison_axis(
    ax: plt.Axes,
    title: str,
    x_label: str = "x",
    y_label: str = "f(x)",
    y_lim: Tuple[float, float] = (-1.5, 1.5),
    add_origin_lines: bool = True,
) -> None:
    """
    Setup standard axis formatting for model comparison plots.

    :param ax: Matplotlib axis to configure
    :param title: Plot title
    :param x_label: Label for x-axis
    :param y_label: Label for y-axis
    :param y_lim: Y-axis limits
    :param add_origin_lines: Whether to add horizontal/vertical lines at origin
    """
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(y_lim)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    if add_origin_lines:
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)


def plot_training_points(
    ax: plt.Axes,
    x_train: np.ndarray,
    y_train: np.ndarray,
    color: str = "red",
    size: int = 100,
    label: str = "Training points",
) -> None:
    """
    Plot training points as scatter plot.

    :param ax: Matplotlib axis
    :param x_train: Training x values
    :param y_train: Training y values
    :param color: Point color
    :param size: Point size
    :param label: Legend label
    """
    ax.scatter(
        x_train,
        y_train,
        color=color,
        s=size,
        zorder=5,
        label=label,
        edgecolors="black",
    )


def compute_bias_variance(
    predictions: list, y_true: np.ndarray
) -> Tuple[float, float]:
    """
    Compute bias squared and variance for a set of model predictions.

    :param predictions: List of prediction arrays from different experiments
    :param y_true: True function values
    :return: Tuple of (bias_squared, variance)
    """
    # Average model predictions across experiments.
    avg_predictions = np.mean(predictions, axis=0)
    # Bias: squared error between average model and true function.
    bias_squared = np.mean((avg_predictions - y_true) ** 2)
    # Variance: expected squared deviation from average model.
    variance_vals = [
        np.mean((pred - avg_predictions) ** 2) for pred in predictions
    ]
    variance = np.mean(variance_vals)
    return bias_squared, variance


def plot_error_metrics(
    ax: plt.Axes,
    n_samples_range: range,
    e_in_avg: list,
    e_out_avg: list,
    bias: list,
    variance: list,
    title: str,
    y_max: float = 1.75,
) -> None:
    """
    Plot error metrics (E_in, E_out, bias, variance) over N_samples.

    :param ax: Matplotlib axis
    :param n_samples_range: Range of N_samples values
    :param e_in_avg: Average in-sample errors
    :param e_out_avg: Average out-of-sample errors
    :param bias: Bias squared values
    :param variance: Variance values
    :param title: Plot title
    :param y_max: Maximum y-axis value
    """
    ax.plot(
        n_samples_range,
        e_in_avg,
        "o-",
        linewidth=2,
        markersize=6,
        label="E_in (In-Sample Error)",
        color="blue",
    )
    ax.plot(
        n_samples_range,
        e_out_avg,
        "s-",
        linewidth=2,
        markersize=6,
        label="E_out (Out-of-Sample Error)",
        color="red",
    )
    ax.plot(
        n_samples_range,
        bias,
        "^-",
        linewidth=2,
        markersize=6,
        label="Bias²",
        color="green",
    )
    ax.plot(
        n_samples_range,
        variance,
        "d-",
        linewidth=2,
        markersize=6,
        label="Variance",
        color="orange",
    )
    ax.set_xlabel("N_samples", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_ylim(0, y_max)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


# #############################################################################
# Cell 1: Approximation
# #############################################################################


def cell1_approximation() -> None:
    """
    Visualize approximation error for constant and linear models.

    Shows how well a constant model (horizontal line) and a linear model
    (diagonal line) can approximate the true sinusoidal target function.
    Displays three plots:
    1. True function vs constant model with approximation error
    2. True function vs linear model with approximation error
    3. Comments box with error values and observations
    """
    # Create dense x values for plotting the true function.
    x_dense = np.linspace(-1, 1, 200)
    y_true = target_function(x_dense)

    # Fit constant model (finds best horizontal line).
    b = fit_constant_model(x_dense, y_true)
    y_const = np.full_like(x_dense, b)
    error_const = compute_approximation_error(x_dense, y_true, y_const)

    # Fit linear model (finds best diagonal line).
    a, b_linear = fit_linear_model(x_dense, y_true)
    y_linear = a * x_dense + b_linear
    error_linear = compute_approximation_error(x_dense, y_true, y_linear)

    # Create figure with 3 subplots.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: True function vs Constant model.
    ax1 = axes[0]
    ax1.plot(x_dense, y_true, "b-", linewidth=2, label="True f(x)")
    ax1.plot(
        x_dense, y_const, "g-", linewidth=2, label=f"Constant g_0(x) = {b:.3f}"
    )
    # Shade approximation error.
    ax1.fill_between(
        x_dense,
        y_true,
        y_const,
        alpha=0.3,
        color="orange",
        label="Approximation Error",
    )
    setup_model_comparison_axis(ax1, "True Function vs Constant Model")

    # Plot 2: True function vs Linear model.
    ax2 = axes[1]
    ax2.plot(x_dense, y_true, "b-", linewidth=2, label="True f(x)")
    ax2.plot(
        x_dense,
        y_linear,
        "m-",
        linewidth=2,
        label=f"Linear g_1(x) = {a:.3f}*x + {b_linear:.3f}",
    )
    # Shade approximation error.
    ax2.fill_between(
        x_dense,
        y_true,
        y_linear,
        alpha=0.3,
        color="orange",
        label="Approximation Error",
    )
    setup_model_comparison_axis(ax2, "True Function vs Linear Model")

    # Plot 3: Comments.
    ax3 = axes[2]
    ax3.axis("off")
    comment_text = f"""
APPROXIMATION ERRORS

Target Function:
  f(x) = sin(pi*x)

Constant Model (g_0):
  g_0(x) = {b:.3f}
  Error: {error_const:.4f}

Linear Model (g_1):
  g_1(x) = {a:.3f}*x + {b_linear:.3f}
  Error: {error_linear:.4f}

OBSERVATION:
The linear model has LOWER
approximation error than the
constant model.

Error_linear = {error_linear:.4f}
Error_const = {error_const:.4f}

The linear model fits the
sinusoid better in the range
[-1, 1], even though neither
can capture the curvature
perfectly.
"""
    ax3.text(
        0.1,
        0.5,
        comment_text,
        fontsize=11,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


# #############################################################################
# Cell 2: Learning Once
# #############################################################################


def cell2_learning_once() -> None:
    """
    Show learning from N random samples is different from approximation.

    Uses interactive widgets to control:
    - seed: Random seed for reproducibility
    - N_samples: Number of training points to sample

    Shows how fitted models perform on both in-sample (E_in) and
    out-of-sample (E_out) data. Demonstrates that learning from limited
    data differs from approximation with full knowledge of the function.
    """
    # Create output widget for displaying plots.
    output = ipywidgets.Output()

    # Create widgets - seed must be first as per conventions.
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="Random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    n_samples_slider, n_samples_box = mtumsuti.build_widget_control(
        name="N_samples",
        description="Number of training samples",
        min_val=2,
        max_val=20,
        step=1,
        initial_value=2,
        is_float=False,
    )

    def update_plot(seed: int, n_samples: int) -> None:
        """Update the visualization based on widget values."""
        with output:
            clear_output(wait=True)

            # Set random seed for reproducibility.
            np.random.seed(seed)

            # Generate training data by sampling random points.
            x_train, y_train = generate_training_data(n_samples)

            # Create dense x values for plotting the true function and computing E_out.
            x_dense = np.linspace(-1, 1, 200)
            y_true = target_function(x_dense)

            # Fit models to training data and generate predictions.
            b, y_const_dense, (a, b_linear), y_linear_dense = (
                fit_models_and_predict(x_train, y_train, x_dense)
            )

            # Generate predictions on training data for E_in computation.
            y_const_train = np.full_like(x_train, b)
            y_linear_train = a * x_train + b_linear

            # Compute all error metrics.
            e_in_const, e_in_linear, e_out_const, e_out_linear = (
                compute_all_errors(
                    x_train,
                    y_train,
                    x_dense,
                    y_true,
                    y_const_train,
                    y_const_dense,
                    y_linear_train,
                    y_linear_dense,
                )
            )

            # Create figure with 3 subplots.
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Plot 1: True function vs Constant model.
            ax1 = axes[0]
            ax1.plot(x_dense, y_true, "b-", linewidth=2, label="True f(x)")
            ax1.plot(
                x_dense,
                y_const_dense,
                "g-",
                linewidth=2,
                label="Constant g_0(x)",
            )
            # Show training points.
            plot_training_points(ax1, x_train, y_train)
            setup_model_comparison_axis(
                ax1,
                f"Constant Model: E_in={e_in_const:.4f}, E_out={e_out_const:.4f}",
            )

            # Plot 2: True function vs Linear model.
            ax2 = axes[1]
            ax2.plot(x_dense, y_true, "b-", linewidth=2, label="True f(x)")
            ax2.plot(
                x_dense, y_linear_dense, "m-", linewidth=2, label="Linear g_1(x)"
            )
            # Show training points.
            plot_training_points(ax2, x_train, y_train)
            setup_model_comparison_axis(
                ax2,
                f"Linear Model: E_in={e_in_linear:.4f}, E_out={e_out_linear:.4f}",
            )

            # Plot 3: Comments.
            ax3 = axes[2]
            ax3.axis("off")
            comment_text = f"""
LEARNING vs APPROXIMATION

Training Set: {n_samples} random points
Seed: {seed}

IN-SAMPLE ERROR (E_in):
  Constant: {e_in_const:.4f}
  Linear:   {e_in_linear:.4f}

OUT-OF-SAMPLE ERROR (E_out):
  Constant: {e_out_const:.4f}
  Linear:   {e_out_linear:.4f}

OBSERVATION:
With only {n_samples} points, the models
fit the TRAINING data (E_in) but
may not generalize well (E_out).

Learning ≠ Approximation!

The constant model has E_in=0
when N=1 (perfect fit!) but
E_out is still high.

Try different seeds to see how
training set selection affects
both E_in and E_out.
"""
            mtumsuti.add_fitted_text_box(ax3, comment_text)

            plt.tight_layout()
            plt.show()

    # Link widgets to update function.
    ipywidgets.interactive_output(
        update_plot,
        {"seed": seed_slider, "n_samples": n_samples_slider},
    )

    # Display widgets and output.
    display(seed_box, n_samples_box, output)

    # Initial plot.
    update_plot(seed_slider.value, n_samples_slider.value)


# #############################################################################
# Cell 3: Learning (Bias-Variance)
# #############################################################################


def cell3_learning_bias_variance() -> None:
    """
    Visualize bias-variance decomposition over multiple experiments.

    Uses interactive widgets to control:
    - seed: Random seed for reproducibility
    - N_samples: Number of training points per experiment
    - N_experiments: Number of different training sets to generate

    Shows how models trained on different random samples vary around
    the true function, demonstrating:
    - Bias: How far the average model is from the true function
    - Variance: How much models vary across different training sets
    """
    # Create output widget for displaying plots.
    output = ipywidgets.Output()

    # Create widgets - seed must be first as per conventions.
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="Random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    n_samples_slider, n_samples_box = mtumsuti.build_widget_control(
        name="N_samples",
        description="Number of training samples per experiment",
        min_val=2,
        max_val=20,
        step=1,
        initial_value=2,
        is_float=False,
    )
    n_experiments_slider, n_experiments_box = mtumsuti.build_widget_control(
        name="N_experiments",
        description="Number of experiments",
        min_val=5,
        max_val=100,
        step=5,
        initial_value=100,
        is_float=False,
    )

    def update_plot(seed: int, n_samples: int, n_experiments: int) -> None:
        """Update the visualization based on widget values."""
        with output:
            clear_output(wait=True)

            # Set random seed for reproducibility.
            np.random.seed(seed)

            # Create dense x values for plotting the true function.
            x_dense = np.linspace(-1, 1, 200)
            y_true = target_function(x_dense)

            # Storage for fitted models across experiments.
            const_models = []  # List of constant values (b)
            linear_models = []  # List of (a, b) tuples

            # Storage for errors.
            e_in_const_list = []
            e_in_linear_list = []
            e_out_const_list = []
            e_out_linear_list = []

            # Run N_experiments with different random training sets.
            for _ in range(n_experiments):
                # Generate training data by sampling random points.
                x_train, y_train = generate_training_data(n_samples)

                # Fit models and generate predictions.
                b, y_const_dense, (a, b_linear), y_linear_dense = (
                    fit_models_and_predict(x_train, y_train, x_dense)
                )
                const_models.append(b)
                linear_models.append((a, b_linear))

                # Generate predictions on training data for E_in computation.
                y_const_train = np.full_like(x_train, b)
                y_linear_train = a * x_train + b_linear

                # Compute errors.
                e_in_const, e_in_linear, e_out_const, e_out_linear = (
                    compute_all_errors(
                        x_train,
                        y_train,
                        x_dense,
                        y_true,
                        y_const_train,
                        y_const_dense,
                        y_linear_train,
                        y_linear_dense,
                    )
                )
                e_in_const_list.append(e_in_const)
                e_in_linear_list.append(e_in_linear)
                e_out_const_list.append(e_out_const)
                e_out_linear_list.append(e_out_linear)

            # Compute average errors.
            avg_e_in_const = np.mean(e_in_const_list)
            avg_e_in_linear = np.mean(e_in_linear_list)
            avg_e_out_const = np.mean(e_out_const_list)
            avg_e_out_linear = np.mean(e_out_linear_list)

            # Create figure with 3 subplots.
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Plot 1: True function vs all Constant models.
            ax1 = axes[0]
            ax1.plot(
                x_dense, y_true, "b-", linewidth=3, label="True f(x)", zorder=10
            )
            # Plot all constant models with transparency.
            for b in const_models:
                ax1.axhline(y=b, color="green", alpha=0.3, linewidth=1)
            # Plot average constant model.
            avg_b = np.mean(const_models)
            ax1.axhline(
                y=avg_b,
                color="darkgreen",
                linewidth=3,
                linestyle="--",
                label=f"Avg g_0 = {avg_b:.3f}",
                zorder=9,
            )
            setup_model_comparison_axis(
                ax1, f"Constant Models ({n_experiments} experiments)"
            )

            # Plot 2: True function vs all Linear models.
            ax2 = axes[1]
            ax2.plot(
                x_dense, y_true, "b-", linewidth=3, label="True f(x)", zorder=10
            )
            # Plot all linear models with transparency.
            for a, b_linear in linear_models:
                y_linear = a * x_dense + b_linear
                ax2.plot(x_dense, y_linear, "m-", alpha=0.3, linewidth=1)
            # Plot average linear model.
            avg_a = np.mean([a for a, _ in linear_models])
            avg_b_linear = np.mean([b for _, b in linear_models])
            y_avg_linear = avg_a * x_dense + avg_b_linear
            ax2.plot(
                x_dense,
                y_avg_linear,
                color="darkmagenta",
                linewidth=3,
                linestyle="--",
                label="Avg g_1",
                zorder=9,
            )
            setup_model_comparison_axis(
                ax2, f"Linear Models ({n_experiments} experiments)"
            )

            # Plot 3: Comments.
            ax3 = axes[2]
            ax3.axis("off")
            comment_text = f"""
BIAS-VARIANCE DECOMPOSITION

Setup: {n_experiments} experiments
       {n_samples} samples per experiment
       Seed: {seed}

AVERAGE IN-SAMPLE ERROR:
  Constant: {avg_e_in_const:.4f}
  Linear:   {avg_e_in_linear:.4f}

AVERAGE OUT-OF-SAMPLE ERROR:
  Constant: {avg_e_out_const:.4f}
  Linear:   {avg_e_out_linear:.4f}

OBSERVATION:
Constant model (g_0):
  - LOW variance (all lines similar)
  - HIGH bias (far from true f(x))

Linear model (g_1):
  - HIGHER variance (lines spread)
  - LOWER bias (closer to f(x))

This is the BIAS-VARIANCE
TRADEOFF!

The transparent lines show
individual models. The dashed
line shows the average model.
"""
            mtumsuti.add_fitted_text_box(ax3, comment_text)

            plt.tight_layout()
            plt.show()

    # Link widgets to update function.
    ipywidgets.interactive_output(
        update_plot,
        {
            "seed": seed_slider,
            "n_samples": n_samples_slider,
            "n_experiments": n_experiments_slider,
        },
    )

    # Display widgets and output.
    display(seed_box, n_samples_box, n_experiments_box, output)

    # Initial plot.
    update_plot(
        seed_slider.value, n_samples_slider.value, n_experiments_slider.value
    )


# #############################################################################
# Cell 4: Learning Plots (Bias-Variance as Function of N_samples)
# #############################################################################


def cell4_learning_plots() -> None:
    """
    Compute and visualize bias-variance decomposition as a function of N_samples.

    Uses interactive widgets to control:
    - seed: Random seed for reproducibility (fixed for consistency)
    - N_experiments: Number of experiments to average over
    - max_N_samples: Maximum number of samples to test

    Shows how E_in, E_out, bias, and variance change as the number of
    training samples increases. Demonstrates:
    - How variance decreases with more data
    - How bias remains relatively constant
    - The total out-of-sample error decomposition
    """
    # Create output widget for displaying plots.
    output = ipywidgets.Output()

    # Create widgets - seed must be first as per conventions.
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="Random seed (fixed)",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    n_experiments_slider, n_experiments_box = mtumsuti.build_widget_control(
        name="N_experiments",
        description="Number of experiments",
        min_val=20,
        max_val=200,
        step=20,
        initial_value=100,
        is_float=False,
    )
    max_n_samples_slider, max_n_samples_box = mtumsuti.build_widget_control(
        name="max_N_samples",
        description="Maximum N_samples",
        min_val=5,
        max_val=30,
        step=5,
        initial_value=20,
        is_float=False,
    )

    def update_plot(seed: int, n_experiments: int, max_n_samples: int) -> None:
        """Update the visualization based on widget values."""
        with output:
            clear_output(wait=True)

            # Set random seed for reproducibility.
            np.random.seed(seed)

            # Create dense x values for plotting the true function and computing metrics.
            x_dense = np.linspace(-1, 1, 200)
            y_true = target_function(x_dense)

            # Storage for metrics across different N_samples.
            n_samples_range = range(2, max_n_samples + 1)
            e_in_const_avg = []
            e_out_const_avg = []
            bias_const = []
            variance_const = []

            e_in_linear_avg = []
            e_out_linear_avg = []
            bias_linear = []
            variance_linear = []

            # For each N_samples value, run multiple experiments.
            for n_samples in n_samples_range:
                # Storage for this N_samples across experiments.
                const_predictions = []  # Store predictions on x_dense for each experiment
                linear_predictions = []
                e_in_const_list = []
                e_out_const_list = []
                e_in_linear_list = []
                e_out_linear_list = []

                # Run N_experiments with different random training sets.
                for _ in range(n_experiments):
                    # Generate training data.
                    x_train, y_train = generate_training_data(n_samples)

                    # Fit models and generate predictions.
                    b, y_const_dense, (a, b_linear), y_linear_dense = (
                        fit_models_and_predict(x_train, y_train, x_dense)
                    )
                    const_predictions.append(y_const_dense)
                    linear_predictions.append(y_linear_dense)

                    # Generate predictions on training data for E_in computation.
                    y_const_train = np.full_like(x_train, b)
                    y_linear_train = a * x_train + b_linear

                    # Compute errors.
                    e_in_const, e_in_linear, e_out_const, e_out_linear = (
                        compute_all_errors(
                            x_train,
                            y_train,
                            x_dense,
                            y_true,
                            y_const_train,
                            y_const_dense,
                            y_linear_train,
                            y_linear_dense,
                        )
                    )
                    e_in_const_list.append(e_in_const)
                    e_out_const_list.append(e_out_const)
                    e_in_linear_list.append(e_in_linear)
                    e_out_linear_list.append(e_out_linear)

                # Compute average errors.
                e_in_const_avg.append(np.mean(e_in_const_list))
                e_out_const_avg.append(np.mean(e_out_const_list))
                e_in_linear_avg.append(np.mean(e_in_linear_list))
                e_out_linear_avg.append(np.mean(e_out_linear_list))

                # Compute bias and variance for constant model.
                bias_squared_const, variance_const_val = compute_bias_variance(
                    const_predictions, y_true
                )
                bias_const.append(bias_squared_const)
                variance_const.append(variance_const_val)

                # Compute bias and variance for linear model.
                bias_squared_linear, variance_linear_val = compute_bias_variance(
                    linear_predictions, y_true
                )
                bias_linear.append(bias_squared_linear)
                variance_linear.append(variance_linear_val)

            # Create figure with 3 subplots.
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Fixed y-axis limit for both plots.
            y_max = 1.75

            # Plot 1: Metrics for Constant Model.
            ax1 = axes[0]
            plot_error_metrics(
                ax1,
                n_samples_range,
                e_in_const_avg,
                e_out_const_avg,
                bias_const,
                variance_const,
                "Constant Model (g_0) - Bias-Variance Analysis",
                y_max,
            )

            # Plot 2: Metrics for Linear Model.
            ax2 = axes[1]
            plot_error_metrics(
                ax2,
                n_samples_range,
                e_in_linear_avg,
                e_out_linear_avg,
                bias_linear,
                variance_linear,
                "Linear Model (g_1) - Bias-Variance Analysis",
                y_max,
            )

            # Plot 3: Comments.
            ax3 = axes[2]
            ax3.axis("off")

            # Get final values for comments.
            final_e_out_const = e_out_const_avg[-1]
            final_bias_const = bias_const[-1]
            final_var_const = variance_const[-1]
            final_e_out_linear = e_out_linear_avg[-1]
            final_bias_linear = bias_linear[-1]
            final_var_linear = variance_linear[-1]

            comment_text = f"""
BIAS-VARIANCE DECOMPOSITION
AS FUNCTION OF N_samples

Setup: {n_experiments} experiments per N
       Seed: {seed} (fixed)
       N_samples: 2 to {max_n_samples}

DECOMPOSITION FORMULA:
E_out = Bias² + Variance + Noise

(Noise = 0 for our deterministic
 target function)

CONSTANT MODEL (g_0):
At N={max_n_samples}:
  E_out:    {final_e_out_const:.4f}
  Bias²:    {final_bias_const:.4f}
  Variance: {final_var_const:.4f}

LINEAR MODEL (g_1):
At N={max_n_samples}:
  E_out:    {final_e_out_linear:.4f}
  Bias²:    {final_bias_linear:.4f}
  Variance: {final_var_linear:.4f}

KEY OBSERVATIONS:
• Constant model: VERY LOW variance
  (insensitive to training data)
  but HIGH bias (can't fit f(x))

• Linear model: HIGHER variance
  (sensitive to training samples)
  but LOWER bias (better fit)

• As N increases, variance ↓
  for both models

• E_out ≈ Bias² + Variance
"""
            mtumsuti.add_fitted_text_box(ax3, comment_text)

            plt.tight_layout()
            plt.show()

    # Link widgets to update function.
    ipywidgets.interactive_output(
        update_plot,
        {
            "seed": seed_slider,
            "n_experiments": n_experiments_slider,
            "max_n_samples": max_n_samples_slider,
        },
    )

    # Display widgets and output.
    display(seed_box, n_experiments_box, max_n_samples_box, output)

    # Initial plot.
    update_plot(
        seed_slider.value, n_experiments_slider.value, max_n_samples_slider.value
    )


# #############################################################################
# Cell 5: Learning with Noise (Bias-Variance Decomposition)
# #############################################################################


def cell5_learning_with_noise() -> None:
    """
    Visualize bias-variance decomposition with noise over multiple experiments.

    Uses interactive widgets to control:
    - seed: Random seed for reproducibility
    - N_samples: Number of training points per experiment
    - N_experiments: Number of different training sets to generate
    - noise_std: Standard deviation of Gaussian noise added to training data

    Similar to cell3_learning_bias_variance but includes a noise widget.
    Shows how noise affects the learned models and error decomposition.
    """
    # Create output widget for displaying plots.
    output = ipywidgets.Output()

    # Create widgets - seed must be first as per conventions.
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="Random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    n_samples_slider, n_samples_box = mtumsuti.build_widget_control(
        name="N_samples",
        description="Number of training samples per experiment",
        min_val=2,
        max_val=20,
        step=1,
        initial_value=2,
        is_float=False,
    )
    n_experiments_slider, n_experiments_box = mtumsuti.build_widget_control(
        name="N_experiments",
        description="Number of experiments",
        min_val=5,
        max_val=100,
        step=5,
        initial_value=100,
        is_float=False,
    )
    noise_slider, noise_box = mtumsuti.build_widget_control(
        name="noise_std",
        description="Noise standard deviation",
        min_val=0.0,
        max_val=0.5,
        step=0.05,
        initial_value=0.0,
        is_float=True,
    )

    def update_plot(
        seed: int, n_samples: int, n_experiments: int, noise_std: float
    ) -> None:
        """Update the visualization based on widget values."""
        with output:
            clear_output(wait=True)

            # Set random seed for reproducibility.
            np.random.seed(seed)

            # Create dense x values for plotting the true function.
            x_dense = np.linspace(-1, 1, 200)
            y_true = target_function(x_dense)

            # Storage for fitted models across experiments.
            const_models = []  # List of constant values (b)
            linear_models = []  # List of (a, b) tuples

            # Storage for errors.
            e_in_const_list = []
            e_in_linear_list = []
            e_out_const_list = []
            e_out_linear_list = []

            # Run N_experiments with different random training sets.
            for _ in range(n_experiments):
                # Generate training data by sampling random points.
                x_train, y_train = generate_training_data(n_samples, noise_std)

                # Fit models and generate predictions.
                b, y_const_dense, (a, b_linear), y_linear_dense = (
                    fit_models_and_predict(x_train, y_train, x_dense)
                )
                const_models.append(b)
                linear_models.append((a, b_linear))

                # Generate predictions on training data for E_in computation.
                y_const_train = np.full_like(x_train, b)
                y_linear_train = a * x_train + b_linear

                # Compute errors.
                e_in_const, e_in_linear, e_out_const, e_out_linear = (
                    compute_all_errors(
                        x_train,
                        y_train,
                        x_dense,
                        y_true,
                        y_const_train,
                        y_const_dense,
                        y_linear_train,
                        y_linear_dense,
                    )
                )
                e_in_const_list.append(e_in_const)
                e_in_linear_list.append(e_in_linear)
                e_out_const_list.append(e_out_const)
                e_out_linear_list.append(e_out_linear)

            # Compute average errors.
            avg_e_in_const = np.mean(e_in_const_list)
            avg_e_in_linear = np.mean(e_in_linear_list)
            avg_e_out_const = np.mean(e_out_const_list)
            avg_e_out_linear = np.mean(e_out_linear_list)

            # Create figure with 3 subplots.
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Plot 1: True function vs all Constant models.
            ax1 = axes[0]
            ax1.plot(
                x_dense, y_true, "b-", linewidth=3, label="True f(x)", zorder=10
            )
            # Plot noisy versions of the function as continuous curves.
            if noise_std > 0:
                # Generate a few noisy realizations of the full function.
                n_noisy_curves = 5
                for i in range(n_noisy_curves):
                    y_noisy = y_true + np.random.normal(
                        0, noise_std, len(y_true)
                    )
                    ax1.plot(
                        x_dense,
                        y_noisy,
                        color="cyan",
                        alpha=0.4,
                        linewidth=1.5,
                        zorder=5,
                        label="f(x) + noise" if i == 0 else "",
                    )
            # Plot all constant models with transparency.
            for b in const_models:
                ax1.axhline(y=b, color="green", alpha=0.3, linewidth=1)
            # Plot average constant model.
            avg_b = np.mean(const_models)
            ax1.axhline(
                y=avg_b,
                color="darkgreen",
                linewidth=3,
                linestyle="--",
                label=f"Avg g_0 = {avg_b:.3f}",
                zorder=9,
            )
            setup_model_comparison_axis(
                ax1, f"Constant Models ({n_experiments} experiments)"
            )

            # Plot 2: True function vs all Linear models.
            ax2 = axes[1]
            ax2.plot(
                x_dense, y_true, "b-", linewidth=3, label="True f(x)", zorder=10
            )
            # Plot noisy versions of the function as continuous curves.
            if noise_std > 0:
                # Generate a few noisy realizations of the full function.
                n_noisy_curves = 5
                for i in range(n_noisy_curves):
                    y_noisy = y_true + np.random.normal(
                        0, noise_std, len(y_true)
                    )
                    ax2.plot(
                        x_dense,
                        y_noisy,
                        color="cyan",
                        alpha=0.4,
                        linewidth=1.5,
                        zorder=5,
                        label="f(x) + noise" if i == 0 else "",
                    )
            # Plot all linear models with transparency.
            for a, b_linear in linear_models:
                y_linear = a * x_dense + b_linear
                ax2.plot(x_dense, y_linear, "m-", alpha=0.3, linewidth=1)
            # Plot average linear model.
            avg_a = np.mean([a for a, _ in linear_models])
            avg_b_linear = np.mean([b for _, b in linear_models])
            y_avg_linear = avg_a * x_dense + avg_b_linear
            ax2.plot(
                x_dense,
                y_avg_linear,
                color="darkmagenta",
                linewidth=3,
                linestyle="--",
                label="Avg g_1",
                zorder=9,
            )
            setup_model_comparison_axis(
                ax2, f"Linear Models ({n_experiments} experiments)"
            )

            # Plot 3: Comments.
            ax3 = axes[2]
            ax3.axis("off")
            comment_text = f"""
BIAS-VARIANCE WITH NOISE

Setup: {n_experiments} experiments
       {n_samples} samples per experiment
       Noise std: {noise_std:.2f}
       Seed: {seed}

AVERAGE IN-SAMPLE ERROR:
  Constant: {avg_e_in_const:.4f}
  Linear:   {avg_e_in_linear:.4f}

AVERAGE OUT-OF-SAMPLE ERROR:
  Constant: {avg_e_out_const:.4f}
  Linear:   {avg_e_out_linear:.4f}

OBSERVATION:
Constant model (g_0):
  - LOW variance (all lines similar)
  - HIGH bias (far from true f(x))

Linear model (g_1):
  - HIGHER variance (lines spread)
  - LOWER bias (closer to f(x))

NOISE EFFECT:
With noise_std > 0, the training
data is corrupted by Gaussian noise.
This increases variance for both
models and E_out increases.

E_out = bias² + variance + noise²
"""
            mtumsuti.add_fitted_text_box(ax3, comment_text)

            plt.tight_layout()
            plt.show()

    # Link widgets to update function.
    ipywidgets.interactive_output(
        update_plot,
        {
            "seed": seed_slider,
            "n_samples": n_samples_slider,
            "n_experiments": n_experiments_slider,
            "noise_std": noise_slider,
        },
    )

    # Display widgets and output.
    display(seed_box, n_samples_box, n_experiments_box, noise_box, output)

    # Initial plot.
    update_plot(
        seed_slider.value,
        n_samples_slider.value,
        n_experiments_slider.value,
        noise_slider.value,
    )


# #############################################################################
# Cell 6: Learning Plots with Noise (Bias-Variance as Function of N_samples)
# #############################################################################


def cell6_learning_plots_with_noise() -> None:
    """
    Compute and visualize bias-variance decomposition with noise as a function of N_samples.

    Uses interactive widgets to control:
    - seed: Random seed for reproducibility (fixed for consistency)
    - N_experiments: Number of experiments to average over
    - max_N_samples: Maximum number of samples to test
    - noise_std: Standard deviation of Gaussian noise added to training data

    Similar to cell4_learning_plots but includes a noise widget.
    Shows how E_in, E_out, bias, and variance change with noise and N_samples.
    """
    # Create output widget for displaying plots.
    output = ipywidgets.Output()

    # Create widgets - seed must be first as per conventions.
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="Random seed (fixed)",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    n_experiments_slider, n_experiments_box = mtumsuti.build_widget_control(
        name="N_experiments",
        description="Number of experiments",
        min_val=20,
        max_val=200,
        step=20,
        initial_value=100,
        is_float=False,
    )
    max_n_samples_slider, max_n_samples_box = mtumsuti.build_widget_control(
        name="max_N_samples",
        description="Maximum N_samples",
        min_val=5,
        max_val=30,
        step=5,
        initial_value=20,
        is_float=False,
    )
    noise_slider, noise_box = mtumsuti.build_widget_control(
        name="noise_std",
        description="Noise standard deviation",
        min_val=0.0,
        max_val=0.5,
        step=0.05,
        initial_value=0.0,
        is_float=True,
    )

    def update_plot(
        seed: int, n_experiments: int, max_n_samples: int, noise_std: float
    ) -> None:
        """Update the visualization based on widget values."""
        with output:
            clear_output(wait=True)

            # Set random seed for reproducibility.
            np.random.seed(seed)

            # Create dense x values for plotting the true function and computing metrics.
            x_dense = np.linspace(-1, 1, 200)
            y_true = target_function(x_dense)

            # Storage for metrics across different N_samples.
            n_samples_range = range(2, max_n_samples + 1)
            e_in_const_avg = []
            e_out_const_avg = []
            bias_const = []
            variance_const = []

            e_in_linear_avg = []
            e_out_linear_avg = []
            bias_linear = []
            variance_linear = []

            # For each N_samples value, run multiple experiments.
            for n_samples in n_samples_range:
                # Storage for this N_samples across experiments.
                const_predictions = []  # Store predictions on x_dense for each experiment
                linear_predictions = []
                e_in_const_list = []
                e_out_const_list = []
                e_in_linear_list = []
                e_out_linear_list = []

                # Run N_experiments with different random training sets.
                for _ in range(n_experiments):
                    # Generate training data.
                    x_train, y_train = generate_training_data(
                        n_samples, noise_std
                    )

                    # Fit models and generate predictions.
                    b, y_const_dense, (a, b_linear), y_linear_dense = (
                        fit_models_and_predict(x_train, y_train, x_dense)
                    )
                    const_predictions.append(y_const_dense)
                    linear_predictions.append(y_linear_dense)

                    # Generate predictions on training data for E_in computation.
                    y_const_train = np.full_like(x_train, b)
                    y_linear_train = a * x_train + b_linear

                    # Compute errors.
                    e_in_const, e_in_linear, e_out_const, e_out_linear = (
                        compute_all_errors(
                            x_train,
                            y_train,
                            x_dense,
                            y_true,
                            y_const_train,
                            y_const_dense,
                            y_linear_train,
                            y_linear_dense,
                        )
                    )
                    e_in_const_list.append(e_in_const)
                    e_out_const_list.append(e_out_const)
                    e_in_linear_list.append(e_in_linear)
                    e_out_linear_list.append(e_out_linear)

                # Compute average errors.
                e_in_const_avg.append(np.mean(e_in_const_list))
                e_out_const_avg.append(np.mean(e_out_const_list))
                e_in_linear_avg.append(np.mean(e_in_linear_list))
                e_out_linear_avg.append(np.mean(e_out_linear_list))

                # Compute bias and variance for constant model.
                bias_squared_const, variance_const_val = compute_bias_variance(
                    const_predictions, y_true
                )
                bias_const.append(bias_squared_const)
                variance_const.append(variance_const_val)

                # Compute bias and variance for linear model.
                bias_squared_linear, variance_linear_val = compute_bias_variance(
                    linear_predictions, y_true
                )
                bias_linear.append(bias_squared_linear)
                variance_linear.append(variance_linear_val)

            # Create figure with 3 subplots.
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Fixed y-axis limit for both plots.
            y_max = 1.75

            # Plot 1: Metrics for Constant Model.
            ax1 = axes[0]
            plot_error_metrics(
                ax1,
                n_samples_range,
                e_in_const_avg,
                e_out_const_avg,
                bias_const,
                variance_const,
                "Constant Model (g_0) - Bias-Variance Analysis",
                y_max,
            )

            # Plot 2: Metrics for Linear Model.
            ax2 = axes[1]
            plot_error_metrics(
                ax2,
                n_samples_range,
                e_in_linear_avg,
                e_out_linear_avg,
                bias_linear,
                variance_linear,
                "Linear Model (g_1) - Bias-Variance Analysis",
                y_max,
            )

            # Plot 3: Comments.
            ax3 = axes[2]
            ax3.axis("off")

            # Get final values for comments.
            final_e_out_const = e_out_const_avg[-1]
            final_bias_const = bias_const[-1]
            final_var_const = variance_const[-1]
            final_e_out_linear = e_out_linear_avg[-1]
            final_bias_linear = bias_linear[-1]
            final_var_linear = variance_linear[-1]

            comment_text = f"""
BIAS-VARIANCE WITH NOISE
AS FUNCTION OF N_samples

Setup: {n_experiments} experiments per N
       Seed: {seed} (fixed)
       N_samples: 2 to {max_n_samples}
       Noise std: {noise_std:.2f}

DECOMPOSITION FORMULA:
E_out = Bias² + Variance + Noise²

CONSTANT MODEL (g_0):
At N={max_n_samples}:
  E_out:    {final_e_out_const:.4f}
  Bias²:    {final_bias_const:.4f}
  Variance: {final_var_const:.4f}

LINEAR MODEL (g_1):
At N={max_n_samples}:
  E_out:    {final_e_out_linear:.4f}
  Bias²:    {final_bias_linear:.4f}
  Variance: {final_var_linear:.4f}

KEY OBSERVATIONS:
• With noise > 0, E_out increases
  for both models

• Variance increases with noise
  (models fit noisy data)

• As N increases, variance ↓
  (more data averages out noise)

• Bias remains constant
  (determined by model capacity)

• E_out ≈ Bias² + Variance + σ²
"""
            mtumsuti.add_fitted_text_box(ax3, comment_text)

            plt.tight_layout()
            plt.show()

    # Link widgets to update function.
    ipywidgets.interactive_output(
        update_plot,
        {
            "seed": seed_slider,
            "n_experiments": n_experiments_slider,
            "max_n_samples": max_n_samples_slider,
            "noise_std": noise_slider,
        },
    )

    # Display widgets and output.
    display(seed_box, n_experiments_box, max_n_samples_box, noise_box, output)

    # Initial plot.
    update_plot(
        seed_slider.value,
        n_experiments_slider.value,
        max_n_samples_slider.value,
        noise_slider.value,
    )
