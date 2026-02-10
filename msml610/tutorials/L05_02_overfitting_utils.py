"""
Utility functions for L05_02_overfitting notebook.

Import as:

import msml610.tutorials.L05_02_overfitting_utils as mtl0ovut
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
# Global Variables for Synchronized State
# #############################################################################

# Store global state for synchronization across cells.
_GLOBAL_STATE = {
    "seed": 42,
    "function_name": "Slow Sinusoid",
    "epsilon": 0.0,
    "N": 16,  # Default: 2^4 = 16 (logarithmic control)
    "x_train": None,
    "y_train": None,
    "x_test": None,
    "y_test": None,
    "x_dense": None,
    "y_true_dense": None,
}


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
    return 2 * (x**2) - 1


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
    # Create N widget with logarithmic slider and +/- buttons.
    # Uses exponents 2-10 for base 2: gives values 4, 8, 16, 32, 64, 128, 256, 512, 1024
    # Initial exponent 4 gives initial value of 16
    N_exp_slider, N_box = mtumsuti.build_log_widget_control(
        name="log(N)",
        description="N (total samples)",
        min_exp=2,
        max_exp=10,
        initial_exp=4,
        base=2,
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
            # N_exp_slider contains the exponent; compute actual N value.
            N = 2 ** N_exp_slider.value
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
            # Update global state for synchronization with other cells.
            _GLOBAL_STATE["seed"] = seed
            _GLOBAL_STATE["function_name"] = func_name
            _GLOBAL_STATE["epsilon"] = epsilon
            _GLOBAL_STATE["N"] = N
            _GLOBAL_STATE["x_train"] = x_train
            _GLOBAL_STATE["y_train"] = y_train
            _GLOBAL_STATE["x_test"] = x_test
            _GLOBAL_STATE["y_test"] = y_test
            _GLOBAL_STATE["x_dense"] = x_dense
            _GLOBAL_STATE["y_true_dense"] = y_true_dense
            # Create 1x4 subplot layout.
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
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
                y_noisy_dense = y_true_dense + np.random.normal(
                    0, epsilon, len(x_dense)
                )
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
            ax1.set_title("True Target Function", fontsize=14, fontweight="bold")
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
            ax2.set_title("In-Sample Data (80%)", fontsize=14, fontweight="bold")
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
            ax4.set_title("Comments", fontsize=16, fontweight="bold", pad=20)
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
                f"Try varying:\n"
                f"- N: more samples → better learning\n"
                f"- epsilon: more noise → harder learning\n"
                f"- seed: different random samples"
            )
            mtumsuti.add_fitted_text_box(
                ax4, text_content, max_fontsize=14, min_fontsize=10
            )
            plt.tight_layout()
            plt.show()

    # Attach observers.
    seed_slider.observe(update_plot, names="value")
    function_dropdown.observe(update_plot, names="value")
    epsilon_slider.observe(update_plot, names="value")
    N_exp_slider.observe(update_plot, names="value")
    # Initial plot.
    update_plot()
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label(
                    "Select the true target function and noise level:"
                ),
                seed_box,
                function_dropdown,
                epsilon_box,
                N_box,
                output,
            ]
        )
    )


# #############################################################################
# Cell 2: Constant Model (H_0)
# #############################################################################


def fit_constant_model(x_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Fit a constant model h(x) = b by finding the mean of y_train.

    :param x_train: Training input data
    :param y_train: Training output data
    :return: Learned parameter b
    """
    return np.mean(y_train)


def fit_linear_model(
    x_train: np.ndarray, y_train: np.ndarray
) -> Tuple[float, float]:
    """
    Fit a linear model h(x) = a*x + b using least squares.

    :param x_train: Training input data
    :param y_train: Training output data
    :return: Tuple of (a, b) parameters
    """
    # Use least squares to fit y = a*x + b.
    # Create design matrix [x, 1].
    A = np.vstack([x_train, np.ones(len(x_train))]).T
    # Solve least squares: A @ [a, b]^T = y.
    params = np.linalg.lstsq(A, y_train, rcond=None)[0]
    a, b = params[0], params[1]
    return a, b


def compute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean squared error.

    :param y_true: True values
    :param y_pred: Predicted values
    :return: Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)


def cell2_plot_model() -> None:
    """
    Interactive widget to visualize constant or linear model learning.

    Shows how a constant hypothesis h(x) = b or linear hypothesis h(x) = a*x + b
    fits the data. Allows comparison between the two model types. Displays
    three plots:
    1. In-sample data with fitted model and E_in
    2. Out-of-sample data with fitted model and E_out
    3. True function with fitted model
    4. Comments and learned parameters
    """
    # Create model selector dropdown.
    model_selector = ipywidgets.Dropdown(
        options=["Constant", "Linear"],
        value="Constant",
        description="Model Type:",
        style={"description_width": "initial"},
    )
    # Create button to resample and relearn.
    resample_button = ipywidgets.Button(
        description="Resample and Relearn",
        button_style="primary",
        tooltip="Generate new training points and refit the model",
    )
    # Create output widget.
    output = ipywidgets.Output()

    def update_plot(change: dict = None) -> None:
        """
        Update the plot with current global state.

        :param change: Dictionary with change information (unused)
        """
        with output:
            clear_output(wait=True)
            # Check if global state is initialized.
            if _GLOBAL_STATE["x_train"] is None:
                print("Please run Cell 1 first to initialize the data.")
                return
            # Get data from global state.
            x_train = _GLOBAL_STATE["x_train"]
            y_train = _GLOBAL_STATE["y_train"]
            x_test = _GLOBAL_STATE["x_test"]
            y_test = _GLOBAL_STATE["y_test"]
            x_dense = _GLOBAL_STATE["x_dense"]
            y_true_dense = _GLOBAL_STATE["y_true_dense"]
            func_name = _GLOBAL_STATE["function_name"]
            epsilon = _GLOBAL_STATE["epsilon"]
            # Get selected model type.
            model_type = model_selector.value
            # Fit model based on selection.
            if model_type == "Constant":
                b = fit_constant_model(x_train, y_train)
                a = 0.0
                # Make predictions.
                y_pred_train = np.full_like(y_train, b)
                y_pred_test = np.full_like(y_test, b)
                y_pred_dense = np.full_like(x_dense, b)
                model_eq = f"h(x) = {b:.3f}"
                params_text = f"b = {b:.4f}"
            else:  # Linear
                a, b = fit_linear_model(x_train, y_train)
                # Make predictions.
                y_pred_train = a * x_train + b
                y_pred_test = a * x_test + b
                y_pred_dense = a * x_dense + b
                model_eq = f"h(x) = {a:.3f}*x + {b:.3f}"
                params_text = f"a = {a:.4f}, b = {b:.4f}"
            # Compute errors.
            E_in = compute_error(y_train, y_pred_train)
            E_out = compute_error(y_test, y_pred_test)
            # Create 1x4 subplot layout.
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
            # Plot 1: In-sample data with fitted model.
            ax1.scatter(
                x_train,
                y_train,
                color="green",
                s=50,
                alpha=0.7,
                label=f"In-Sample (n={len(x_train)})",
                zorder=5,
            )
            if model_type == "Constant":
                ax1.axhline(
                    y=b,
                    color="darkgreen",
                    linewidth=2,
                    label=model_eq,
                    linestyle="--",
                )
            else:  # Linear
                ax1.plot(
                    x_dense,
                    y_pred_dense,
                    color="orange",
                    linewidth=2,
                    label=model_eq,
                    linestyle="--",
                )
            ax1.set_xlabel("x", fontsize=12)
            ax1.set_ylabel("y", fontsize=12)
            ax1.set_title(
                f"In-Sample Data and Model (E_in = {E_in:.4f})",
                fontsize=14,
                fontweight="bold",
            )
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color="k", linewidth=0.5)
            ax1.axvline(x=0, color="k", linewidth=0.5)
            ax1.legend(fontsize=10)
            ax1.set_xlim([-1, 1])
            ax1.set_ylim([-1.5, 1.5])
            # Plot 2: Out-of-sample data with fitted model.
            ax2.scatter(
                x_test,
                y_test,
                color="red",
                s=50,
                alpha=0.7,
                label=f"Out-of-Sample (n={len(x_test)})",
                zorder=5,
            )
            if model_type == "Constant":
                ax2.axhline(
                    y=b,
                    color="darkgreen",
                    linewidth=2,
                    label=model_eq,
                    linestyle="--",
                )
            else:
                # Linear.
                ax2.plot(
                    x_dense,
                    y_pred_dense,
                    color="orange",
                    linewidth=2,
                    label=model_eq,
                    linestyle="--",
                )
            ax2.set_xlabel("x", fontsize=12)
            ax2.set_ylabel("y", fontsize=12)
            ax2.set_title(
                f"Out-of-Sample Data and Model (E_out = {E_out:.4f})",
                fontsize=14,
                fontweight="bold",
            )
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color="k", linewidth=0.5)
            ax2.axvline(x=0, color="k", linewidth=0.5)
            ax2.legend(fontsize=10)
            ax2.set_xlim([-1, 1])
            ax2.set_ylim([-1.5, 1.5])
            # Plot 3: True function with fitted model.
            ax3.plot(
                x_dense,
                y_true_dense,
                "b-",
                linewidth=2,
                label="True Function",
                alpha=0.7,
            )
            if model_type == "Constant":
                ax3.axhline(
                    y=b,
                    color="darkgreen",
                    linewidth=2,
                    label=model_eq,
                    linestyle="--",
                )
                # Shade the area between constant and true function.
                ax3.fill_between(
                    x_dense,
                    y_true_dense,
                    b,
                    alpha=0.3,
                    color="orange",
                    label="Approximation Error",
                )
            else: 
                # Linear.
                ax3.plot(
                    x_dense,
                    y_pred_dense,
                    color="orange",
                    linewidth=2,
                    label=model_eq,
                    linestyle="--",
                )
                # Shade the area between linear model and true function.
                ax3.fill_between(
                    x_dense,
                    y_true_dense,
                    y_pred_dense,
                    alpha=0.3,
                    color="orange",
                    label="Approximation Error",
                )
            ax3.set_xlabel("x", fontsize=12)
            ax3.set_ylabel("y", fontsize=12)
            ax3.set_title(
                "True Function vs Constant Model",
                fontsize=14,
                fontweight="bold",
            )
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color="k", linewidth=0.5)
            ax3.axvline(x=0, color="k", linewidth=0.5)
            ax3.legend(fontsize=10)
            ax3.set_xlim([-1, 1])
            ax3.set_ylim([-1.5, 1.5])
            # Plot 4: Comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=16, fontweight="bold", pad=20)
            # Generate comment text based on model type.
            text_content = (
                f"Model Type: {model_type}\n"
                f"Model: {model_eq}\n"
                f"Learned parameters: {params_text}\n\n"
                f"Current Setup:\n"
                f"  Function: {func_name}\n"
                f"  epsilon (noise): {epsilon:.2f}\n"
                f"  N (total): {_GLOBAL_STATE['N']}\n"
                f"  n_train: {len(x_train)}\n"
                f"  n_test: {len(x_test)}\n\n"
                f"Error:\n"
                f"  E_in = {E_in:.4f}\n"
                f"  E_out = {E_out:.4f}\n\n"
                f"\n"
                f"Click 'Resample and Relearn' to see\n"
                f"how the model changes with different\n"
                f"training data."
            )
            mtumsuti.add_fitted_text_box(
                ax4, text_content, max_fontsize=14, min_fontsize=10
            )
            plt.tight_layout()
            plt.show()

    def on_resample_clicked(b: ipywidgets.Button) -> None:
        """
        Handle resample button click.

        Generates new training data by incrementing seed and updates global state.

        :param b: Button widget (unused)
        """
        # Increment seed to get new samples.
        new_seed = _GLOBAL_STATE["seed"] + 1
        _GLOBAL_STATE["seed"] = new_seed
        # Get current parameters.
        func_name = _GLOBAL_STATE["function_name"]
        epsilon = _GLOBAL_STATE["epsilon"]
        N = _GLOBAL_STATE["N"]
        # Get target function.
        target_func = TARGET_FUNCTIONS[func_name]
        # Generate x values for true function (dense).
        x_dense = np.linspace(-1, 1, 200)
        y_true_dense = target_func(x_dense)
        y_true_dense = np.clip(y_true_dense, -1.0, 1.0)
        # Generate new sampled data points.
        np.random.seed(new_seed)
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
        np.random.seed(new_seed)
        indices = np.arange(N)
        np.random.shuffle(indices)
        n_train = int(0.8 * N)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        # Update global state.
        _GLOBAL_STATE["x_train"] = x_samples[train_indices]
        _GLOBAL_STATE["y_train"] = y_samples_noisy[train_indices]
        _GLOBAL_STATE["x_test"] = x_samples[test_indices]
        _GLOBAL_STATE["y_test"] = y_samples_noisy[test_indices]
        _GLOBAL_STATE["x_dense"] = x_dense
        _GLOBAL_STATE["y_true_dense"] = y_true_dense
        # Update plot.
        update_plot()

    # Attach observers.
    model_selector.observe(update_plot, names="value")
    resample_button.on_click(on_resample_clicked)
    # Initial plot.
    update_plot()
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label("Model Comparison: Constant vs Linear"),
                ipywidgets.Label(
                    "This cell uses the same setup as Cell 1. "
                    "Adjust parameters in Cell 1 to change the setup."
                ),
                model_selector,
                resample_button,
                output,
            ]
        )
    )
