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


def plot_true_target_function() -> None:
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
