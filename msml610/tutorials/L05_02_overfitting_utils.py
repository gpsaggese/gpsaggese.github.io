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
    of different target functions and noise levels. Displays four plots:
    1. True target function
    2. In-sample data (80% of N samples)
    3. Out-of-sample data (20% of N samples)
    4. Key insights and comments
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
    # Create N widget with slider and +/- buttons.
    N_slider, N_box = mtumsuti.build_widget_control(
        name="N",
        description="N (total samples)",
        min_val=5,
        max_val=100,
        step=5,
        initial_value=20,
        is_float=False,
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
            N = N_slider.value
            # Generate x values for true function (dense).
            x_dense = np.linspace(-1, 1, 200)
            # Get target function.
            target_func = TARGET_FUNCTIONS[func_name]
            y_true_dense = target_func(x_dense)
            # Ensure y_true is clipped to [-1, 1] range.
            y_true_dense = np.clip(y_true_dense, -1.0, 1.0)
            # Generate sampled data points.
            np.random.seed(seed)
            # Sample N points uniformly from [-1, 1].
            x_samples = np.random.uniform(-1, 1, N)
            x_samples = np.sort(x_samples)
            y_samples = target_func(x_samples)
            y_samples = np.clip(y_samples, -1.0, 1.0)
            # Add noise to samples.
            if epsilon > 0:
                y_samples_noisy = y_samples + np.random.normal(0, epsilon, N)
            else:
                y_samples_noisy = y_samples
            # Split into in-sample (80%) and out-of-sample (20%).
            np.random.seed(seed)
            indices = np.arange(N)
            np.random.shuffle(indices)
            n_train = int(0.8 * N)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
            # In-sample data.
            x_train = x_samples[train_indices]
            y_train = y_samples_noisy[train_indices]
            # Out-of-sample data.
            x_test = x_samples[test_indices]
            y_test = y_samples_noisy[test_indices]
            # Create 2x2 subplot layout.
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                2, 2, figsize=(14, 10)
            )
            # Plot 1: True target function.
            # Plot the true function (noiseless).
            ax1.plot(
                x_dense,
                y_true_dense,
                "b-",
                linewidth=2,
                label="True Function (noiseless)",
            )
            # Add noisy function if epsilon > 0.
            if epsilon > 0:
                y_noisy_dense = y_true_dense + np.random.normal(0, epsilon, len(x_dense))
                ax1.plot(
                    x_dense,
                    y_noisy_dense,
                    "b-",
                    linewidth=1,
                    alpha=0.5,
                    label="With Noise",
                )
            ax1.set_xlabel("x", fontsize=12)
            ax1.set_ylabel("f(x)", fontsize=12)
            ax1.set_title(
                "True Target Function", fontsize=14, fontweight="bold"
            )
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color="k", linewidth=0.5)
            ax1.axvline(x=0, color="k", linewidth=0.5)
            ax1.legend(fontsize=10)
            ax1.set_xlim([-1, 1])
            ax1.set_ylim([-1.5, 1.5])
            # Plot 2: In-sample data.
            ax2.scatter(
                x_train,
                y_train,
                color="green",
                s=50,
                alpha=0.7,
                label=f"In-Sample (n={len(x_train)})",
                zorder=5,
            )
            ax2.set_xlabel("x", fontsize=12)
            ax2.set_ylabel("f(x)", fontsize=12)
            ax2.set_title(
                "In-Sample Data (80%)", fontsize=14, fontweight="bold"
            )
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color="k", linewidth=0.5)
            ax2.axvline(x=0, color="k", linewidth=0.5)
            ax2.legend(fontsize=10)
            ax2.set_xlim([-1, 1])
            ax2.set_ylim([-1.5, 1.5])
            # Plot 3: Out-of-sample data.
            ax3.scatter(
                x_test,
                y_test,
                color="red",
                s=50,
                alpha=0.7,
                label=f"Out-of-Sample (n={len(x_test)})",
                zorder=5,
            )
            ax3.set_xlabel("x", fontsize=12)
            ax3.set_ylabel("f(x)", fontsize=12)
            ax3.set_title(
                "Out-of-Sample Data (20%)", fontsize=14, fontweight="bold"
            )
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color="k", linewidth=0.5)
            ax3.axvline(x=0, color="k", linewidth=0.5)
            ax3.legend(fontsize=10)
            ax3.set_xlim([-1, 1])
            ax3.set_ylim([-1.5, 1.5])
            # Plot 4: Comments.
            ax4.axis("off")
            ax4.set_title(
                "Comments", fontsize=16, fontweight="bold", pad=20
            )
            # Generate comment text.
            text_content = (
                f"Parameters:\n"
                f"  Function: {func_name}\n"
                f"  epsilon (noise): {epsilon:.2f}\n"
                f"  N (total samples): {N}\n"
                f"  seed: {seed}\n\n"
                f"Data Split:\n"
                f"  In-sample: {len(x_train)} points (80%)\n"
                f"  Out-of-sample: {len(x_test)} points (20%)\n\n"
                f"Key Observations:\n"
                f"- The true function (blue curve) is\n"
                f"  the unknown target we want to learn\n"
                f"- In practice, we only observe a few\n"
                f"  noisy samples from this function\n"
                f"- Green points are used for training\n"
                f"- Red points are held out for testing\n"
                f"- The goal is to learn from green points\n"
                f"  and generalize to red points\n\n"
                f"Try varying:\n"
                f"- N: more samples → better learning\n"
                f"- epsilon: more noise → harder learning\n"
                f"- seed: different random samples"
            )
            mtumsuti.add_fitted_text_box(ax4, text_content, max_fontsize=14, min_fontsize=10)
            plt.tight_layout()
            plt.show()

    # Attach observers.
    seed_slider.observe(update_plot, names="value")
    function_dropdown.observe(update_plot, names="value")
    epsilon_slider.observe(update_plot, names="value")
    N_slider.observe(update_plot, names="value")
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
                N_box,
                output,
            ]
        )
    )