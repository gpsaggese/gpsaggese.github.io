"""
Utility functions for L05_02_overfitting notebook.

Import as:

import L05_02_overfitting_utils as utils
"""

import logging
from typing import Callable, Tuple

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)


# #############################################################################
# Target Functions
# #############################################################################


def slow_sinusoid(x: np.ndarray) -> np.ndarray:
    """
    Slow sinusoid function: f(x) = sin(0.5 * pi * x).

    :param x: Input array
    :return: Output array
    """
    return np.sin(0.5 * np.pi * x)


def fast_sinusoid(x: np.ndarray) -> np.ndarray:
    """
    Fast sinusoid function: f(x) = sin(2 * pi * x).

    :param x: Input array
    :return: Output array
    """
    return np.sin(2 * np.pi * x)


def parabola(x: np.ndarray) -> np.ndarray:
    """
    Parabola function: f(x) = 2*x^2 - 1 (scaled to [-1, 1] range).

    :param x: Input array
    :return: Output array
    """
    # Scale parabola to fit in [-1, 1] range
    # x^2 gives [0, 1] for x in [-1, 1], so 2*x^2 - 1 gives [-1, 1]
    return 2 * (x ** 2) - 1


def constant(x: np.ndarray, c: float = 0.0) -> np.ndarray:
    """
    Constant function: f(x) = c.

    :param x: Input array
    :param c: Constant value
    :return: Output array filled with constant value
    """
    return np.full_like(x, c)


def linear(x: np.ndarray) -> np.ndarray:
    """
    Linear function: f(x) = x.

    :param x: Input array
    :return: Output array
    """
    return x


# Target function dictionary.
TARGET_FUNCTIONS = {
    "Slow Sinusoid": slow_sinusoid,
    "Fast Sinusoid": fast_sinusoid,
    "Parabola": parabola,
    "Constant": lambda x: constant(x, c=0.0),
    "Linear": linear,
}


# #############################################################################
# Cell 1: True Target Function - Sinusoid
# #############################################################################


def cell1_plot_true_target_function() -> None:
    """
    Interactive widget to visualize true target functions with noise.

    Shows the unknown target function that we want to learn. Allows selection
    of different target functions and noise levels.
    """
    # Create seed widget with slider and +/- buttons.
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="Random Seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    # Create widgets.
    function_dropdown = ipywidgets.Dropdown(
        options=list(TARGET_FUNCTIONS.keys()),
        value="Slow Sinusoid",
        description="Function:",
        style={"description_width": "initial"},
    )
    # Create epsilon widget with slider and +/- buttons.
    epsilon_slider, epsilon_box = mtumsuti.build_widget_control(
        name="epsilon",
        description="epsilon (noise std dev)",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.0,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: dict = None) -> None:
        """
        Update the plot when widget values change.

        :param change: Dictionary with change information (unused)
        """
        with output:
            clear_output(wait=True)
            # Get current values.
            seed = seed_slider.value
            func_name = function_dropdown.value
            epsilon = epsilon_slider.value
            # Generate x values.
            x = np.linspace(-1, 1, 200)
            # Get target function.
            target_func = TARGET_FUNCTIONS[func_name]
            y_true = target_func(x)
            # Ensure y_true is clipped to [-1, 1] range.
            y_true = np.clip(y_true, -1.0, 1.0)
            # Add noise for visualization if epsilon > 0.
            if epsilon > 0:
                np.random.seed(seed)
                y_noisy = y_true + np.random.normal(0, epsilon, len(x))
            else:
                y_noisy = y_true
            # Create plot.
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot the true function.
            if epsilon > 0:
                ax.plot(x, y_true, "b-", linewidth=2, label="True Function (noiseless)")
                ax.plot(x, y_noisy, "b-", linewidth=1, alpha=0.5, label="With Noise")
            else:
                ax.plot(x, y_true, "b-", linewidth=2, label="True Function")
            # Format plot.
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("f(x)", fontsize=12)
            ax.set_title("True Target Function", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="k", linewidth=0.5)
            ax.axvline(x=0, color="k", linewidth=0.5)
            ax.legend(fontsize=10)
            ax.set_xlim([-1, 1])
            # Set y-axis limits to show full range with room for noise.
            ax.set_ylim([-1.5, 1.5])
            # Add comment box.
            comment = (
                "This is the unknown target function we want to learn.\n"
                "In real-world problems, we don't have access to this\n"
                "complete curve - we only see a few sampled points."
            )
            ax.text(
                0.02,
                0.98,
                comment,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            plt.tight_layout()
            plt.show()

    # Attach observers.
    seed_slider.observe(update_plot, names="value")
    function_dropdown.observe(update_plot, names="value")
    epsilon_slider.observe(update_plot, names="value")
    # Initial plot.
    update_plot()
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label("Select the true target function and noise level:"),
                seed_box,
                function_dropdown,
                epsilon_box,
                output,
            ]
        )
    )


# #############################################################################
# Cell 2: Sampled Data - In-Sample and Out-of-Sample
# #############################################################################


def _plot_sampled_data(
    seed: int, func_name: str, epsilon: float, N: int
) -> None:
    """
    Plot sampled data with in-sample and out-of-sample split.

    :param seed: Random seed for reproducibility
    :param func_name: Name of target function
    :param epsilon: Noise standard deviation
    :param N: Total number of samples
    """
    # Set random seed.
    np.random.seed(seed)
    # Calculate split sizes (80-20).
    N_in = int(0.8 * N)
    N_out = N - N_in
    # Get target function.
    target_func = TARGET_FUNCTIONS[func_name]
    # Generate continuous x for true function.
    x_true = np.linspace(-1, 1, 200)
    y_true = target_func(x_true)
    y_true = np.clip(y_true, -1.0, 1.0)
    # Generate random x samples for in-sample and out-of-sample.
    x_samples = np.random.uniform(-1, 1, N)
    # Sort samples to split them properly.
    x_samples_sorted_idx = np.argsort(x_samples)
    x_samples_sorted = x_samples[x_samples_sorted_idx]
    # Split into in-sample (first 80%) and out-of-sample (last 20%).
    x_in = x_samples_sorted[:N_in]
    x_out = x_samples_sorted[N_in:]
    # Generate y values with noise.
    y_in = target_func(x_in) + np.random.normal(0, epsilon, N_in)
    y_out = target_func(x_out) + np.random.normal(0, epsilon, N_out)
    # Create visualization with 2 subplots.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    # Plot 1: Function and sampled data.
    # Plot true function.
    ax1.plot(x_true, y_true, "b-", linewidth=2, label="True Function", alpha=0.7)
    # Plot in-sample points.
    ax1.scatter(
        x_in,
        y_in,
        c="green",
        s=50,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        label=f"In-Sample (N={N_in})",
        zorder=5,
    )
    # Plot out-of-sample points.
    ax1.scatter(
        x_out,
        y_out,
        c="red",
        s=50,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        label=f"Out-of-Sample (N={N_out})",
        zorder=5,
    )
    # Format plot.
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("f(x)", fontsize=12)
    ax1.set_title(
        f"Sampled Data: In-Sample vs Out-of-Sample\n{func_name}, epsilon={epsilon:.2f}, seed={seed}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linewidth=0.5)
    ax1.axvline(x=0, color="k", linewidth=0.5)
    ax1.legend(fontsize=10, loc="best")
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1.5, 1.5])
    # Plot 2: Interpretation box.
    ax2.axis("off")
    ax2.set_title("Interpretation", fontsize=14, fontweight="bold", pad=20)
    # Generate interpretation text.
    text_content = (
        f"Parameters:\n"
        f"  Function: {func_name}\n"
        f"  N (total): {N}\n"
        f"  N (in-sample): {N_in} (80%)\n"
        f"  N (out-of-sample): {N_out} (20%)\n"
        f"  epsilon: {epsilon:.2f}\n"
        f"  seed: {seed}\n\n"
        f"Key Observations:\n"
        f"• Green points: Training data\n"
        f"  (used to fit the model)\n"
        f"• Red points: Test data\n"
        f"  (used to evaluate the model)\n\n"
        f"Learning Goal:\n"
        f"We want to fit a model to the\n"
        f"in-sample (green) points that\n"
        f"generalizes well to out-of-sample\n"
        f"(red) points.\n\n"
        f"The challenge: With limited data,\n"
        f"we must balance fitting the\n"
        f"training data vs. generalizing\n"
        f"to unseen data."
    )
    ax2.text(
        0.05,
        0.95,
        text_content,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )
    plt.tight_layout()
    plt.show()


def cell2_plot_sampled_data_interactive() -> None:
    """
    Interactive widget to visualize sampled data with in-sample/out-of-sample split.

    Shows how data is split into training (in-sample) and test (out-of-sample) sets.
    """
    # Create interactive widgets.
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="Random Seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    function_dropdown = ipywidgets.Dropdown(
        options=list(TARGET_FUNCTIONS.keys()),
        value="Slow Sinusoid",
        description="Function:",
        style={"description_width": "initial"},
    )
    epsilon_slider, epsilon_box = mtumsuti.build_widget_control(
        name="epsilon",
        description="epsilon (noise std dev)",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.1,
        is_float=True,
    )
    N_slider, N_box = mtumsuti.build_widget_control(
        name="N",
        description="N (total samples)",
        min_val=5,
        max_val=100,
        step=5,
        initial_value=20,
        is_float=False,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot_sampled_data,
        {
            "seed": seed_slider,
            "func_name": function_dropdown,
            "epsilon": epsilon_slider,
            "N": N_slider,
        },
    )
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label("Configure sampling parameters:"),
                seed_box,
                function_dropdown,
                epsilon_box,
                N_box,
                output,
            ]
        )
    )
