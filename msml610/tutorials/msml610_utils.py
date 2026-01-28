"""
Utility functions for MSML610 course tutorials.

Import as:

import msml610.tutorials.msml610_utils as mtumsuti
"""

import copy
import logging
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as stats
import ipywidgets
from ipywidgets import Button, FloatSlider, FloatText, HBox, IntSlider, IntText
from IPython.display import clear_output, display
from PIL import Image

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)


# #############################################################################
# Widget Builder Utilities
# #############################################################################


def _create_slider_widget(
    *,
    name: str,
    description: str,
    min_val: float,
    max_val: float,
    step: float,
    initial_value: float,
    is_float: bool = True,
) -> Tuple:
    """
    Create a slider widget with text field and +/- buttons.

    Creates a complete widget control with slider, text input field, and
    increment/decrement buttons following the notebook conventions.

    :param name: Variable name (e.g., "mu", "N", "seed")
    :param description: Human-readable description (e.g., "prob of success")
    :param min_val: Minimum value for the slider
    :param max_val: Maximum value for the slider
    :param step: Step size for slider and buttons
    :param initial_value: Initial value
    :param is_float: If True, create FloatSlider/FloatText, else IntSlider/IntText
    :return: Tuple of (slider, text, minus_button, plus_button)
    """
    # Create widgets based on type.
    if is_float:
        slider = FloatSlider(
            min=min_val,
            max=max_val,
            step=step,
            value=initial_value,
            description=f"{name} = {description}",
            continuous_update=False,
            style={"description_width": "150px"},
            layout={"width": "500px"},
        )
        text = FloatText(
            value=initial_value,
            step=step,
            description="",
            layout={"width": "80px"},
        )
    else:
        slider = IntSlider(
            min=int(min_val),
            max=int(max_val),
            step=int(step),
            value=int(initial_value),
            description=f"{name} = {description}",
            continuous_update=False,
            style={"description_width": "150px"},
            layout={"width": "500px"},
        )
        text = IntText(
            value=int(initial_value),
            step=int(step),
            description="",
            layout={"width": "80px"},
        )
    # Create buttons.
    minus_button = Button(description="-", layout={"width": "40px"})
    plus_button = Button(description="+", layout={"width": "40px"})
    return slider, text, minus_button, plus_button


# TODO(ai_gp): Add type hints.
def _link_slider_widgets(slider, text, minus_button, plus_button) -> None:
    """
    Link slider, text field, and buttons together.

    Sets up bidirectional sync between slider and text field, and connects
    buttons to increment/decrement the slider value.

    :param slider: The slider widget (FloatSlider or IntSlider)
    :param text: The text input widget (FloatText or IntText)
    :param minus_button: The decrement button
    :param plus_button: The increment button
    """

    def slider_changed(change):
        text.value = change["new"]

    def text_changed(change):
        if slider.min <= change["new"] <= slider.max:
            slider.value = change["new"]

    def minus_clicked(b):
        slider.value = max(slider.min, slider.value - slider.step)

    def plus_clicked(b):
        slider.value = min(slider.max, slider.value + slider.step)

    # Connect observers.
    slider.observe(slider_changed, names="value")
    text.observe(text_changed, names="value")
    minus_button.on_click(minus_clicked)
    plus_button.on_click(plus_clicked)


# TODO(ai_gp): Add type hints.
def _create_widget_box(slider, minus_button, text, plus_button) -> HBox:
    """
    Create horizontal box layout for widget controls.

    :param slider: The slider widget
    :param minus_button: The decrement button
    :param text: The text input widget
    :param plus_button: The increment button
    :return: HBox containing all widgets in proper order
    """
    return HBox([slider, minus_button, text, plus_button])


def build_widget_control(
    *,
    name: str,
    description: str,
    min_val: float,
    max_val: float,
    step: float,
    initial_value: float,
    is_float: bool = True,
) -> Tuple[Union[FloatSlider, IntSlider], HBox]:
    """
    Build a complete widget control with slider, text field, and +/- buttons.

    Convenience function that creates, links, and lays out all widget
    components in a single call.

    :param name: Variable name (e.g., "mu", "N", "seed")
    :param description: Human-readable description (e.g., "prob of success")
    :param min_val: Minimum value for the slider
    :param max_val: Maximum value for the slider
    :param step: Step size for slider and buttons
    :param initial_value: Initial value
    :param is_float: If True, create FloatSlider/FloatText, else IntSlider/IntText
    :return: Tuple of (slider, box) where slider is the control widget and box
        is the HBox layout containing all components
    """
    # Create widgets with sliders, text fields, and +/- buttons.
    slider, text, minus_button, plus_button = _create_slider_widget(
        name=name,
        description=description,
        min_val=min_val,
        max_val=max_val,
        step=step,
        initial_value=initial_value,
        is_float=is_float,
    )
    # Link sliders and text fields.
    _link_slider_widgets(slider, text, minus_button, plus_button)
    # Create layout.
    box = _create_widget_box(slider, minus_button, text, plus_button)
    return slider, box


# #############################################################################
# Notebook configuration.
# #############################################################################


def set_notebook_style() -> None:
    """
    Set default matplotlib style for notebooks.
    """
    _LOG.info("Setting notebook style")
    plt.rcParams["figure.figsize"] = [8, 3]


def notebook_signature() -> None:
    """
    Display Python environment information including version and module versions.
    """
    _LOG.info("Notebook signature")
    cmd = "python --version"
    os.system(cmd)
    cmd = "uname -a"
    os.system(cmd)
    modules = ["numpy", "pymc", "matplotlib", "arviz", "preliz"]
    for module in modules:
        cmd = f"import {module}"
        exec(cmd)
        version = eval(f"{module}.__version__")
        _LOG.info("%s version=%s", module, version)


def config_notebook() -> None:
    """
    Configure notebook with default style and display environment signature.
    """
    if os.environ["CSFY_HOST_USER_NAME"] == "saggese":
        cmd = 'sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet jupyterlab-vim)"'
        hsystem.system(cmd)
        cmd = "jupyter labextension enable"
        hsystem.system(cmd)
        _LOG.warning("vim support installed: restart the notebook, if needed")
    set_notebook_style()
    notebook_signature()


def obj_to_str(var_name: str, val: Any, *, top_n: int = 3) -> str:
    """
    Convert object to string representation showing name, type, and preview.

    :param var_name: Name of the variable
    :param val: Value to convert
    :param top_n: Number of elements to show from start and end for arrays
    :return: String representation of the object
    """
    txt = []
    txt_tmp = "var_name=%s (type=%s)" % (var_name, str(type(val)))
    txt.append(txt_tmp)
    if isinstance(val, np.ndarray):
        txt.append("shape=%s" % val.shape)
        if len(val.shape) == 1:
            txt_tmp = "%s ... %s" % (val[:top_n], val[-top_n:])
            txt_tmp = txt_tmp.replace("[", "")
            txt_tmp = txt_tmp.replace("]", "")
            txt_tmp = f"[{txt_tmp}]"
            txt.append(txt_tmp)
    return "\n".join(txt)


def print_obj(*args: Any, **kwargs: Any) -> None:
    """
    Print object information using obj_to_str.
    """
    _LOG.info(obj_to_str(*args, **kwargs))


# Lesson 7, notebook 1


def convert_to_filename(string: str) -> str:
    """
    Convert string to sanitized filename path in figures directory.

    :param string: Input string to convert
    :return: Full path to PNG file
    """
    dst_dir = os.path.join(
        os.environ["CSFY_GIT_ROOT_PATH"], "lectures_source/figures"
    )
    dst_dir = os.path.normpath(dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    file_name = string
    file_name = file_name.replace(":", "")
    file_name = file_name.replace(" ", "_")
    file_name = file_name.replace(".", "_")
    file_name = os.path.join(dst_dir, file_name)
    file_name += ".png"
    return file_name


def print_figure(file_name: str) -> None:
    """
    Print markdown image reference with fixed width.

    :param file_name: Path to image file
    """
    txt = f"![]({file_name})" + "{ width=100px }"
    _LOG.info(txt)


def process_figure(title: str) -> None:
    """
    Save current figure with title-based filename.

    :param title: Title used to generate filename
    """
    file_name = convert_to_filename(title)
    plt.savefig(file_name, dpi=300)


def plot_binomial() -> None:
    """
    Plot binomial distribution for various n and p parameter combinations.
    """
    n_params = [2, 4, 8]
    p_params = [0, 0.25, 0.5, 0.75, 1]
    max_n = max(n_params) + 1
    # Create a plot.
    _, ax = plt.subplots(
        len(n_params),
        len(p_params),
        sharex=True,
        sharey=True,
        figsize=(9, 7),
        constrained_layout=True,
    )
    _LOG.debug("ax.shape=%s", ax.shape)
    for i in range(len(n_params)):
        for j in range(len(p_params)):
            x = list(range(0, max_n))
            n = n_params[i]
            p = p_params[j]
            # Evaluate the PDF in several points.
            y = stats.binom(n=n, p=p).pmf(x)
            y = [y[k] if k <= n else np.nan for k in range(max_n)]
            # Plot the PDF.
            ax[i, j].bar(x, y)
            # Add the legend.
            label = "n={:3.2f}\np={:3.2f}".format(n, p)
            ax[i, j].plot([], label=label, alpha=0)
            ax[i, j].legend(loc="best")
    ax[2, 1].set_xlabel("x")
    ax[1, 0].set_ylabel("p(x)", rotation=0, labelpad=20)
    ax[1, 0].set_xticks(range(0, max_n))
    #
    title = "Binomial distribution"
    process_figure(title)


def plot_beta() -> None:
    """
    Plot beta distribution for various alpha and beta parameter combinations.
    """
    # Alpha and beta values to plot.
    a_params = [0.8, 1.0, 2.0, 4.0]
    b_params = [0.8, 1.0, 2.0, 4.0]
    x = np.linspace(0, 1, 200)
    # Create a plot.
    _, ax = plt.subplots(
        len(a_params),
        len(b_params),
        sharex=True,
        sharey=True,
        figsize=(9, 7),
        constrained_layout=True,
    )
    for i in range(len(a_params)):
        for j in range(len(b_params)):
            alpha = a_params[i]
            beta = b_params[j]
            # Evaluate the PDF in several points.
            y = stats.beta(a=alpha, b=beta).pdf(x)
            # Plot the PDF.
            ax[i, j].plot(x, y)
            # Add the legend.
            label = "a={:3.2f}\nb={:3.2f}".format(alpha, beta)
            ax[i, j].plot([], label=label, alpha=0)
            ax[i, j].legend(loc=1)
    ax[2, 1].set_xlabel("x")
    ax[1, 0].set_ylabel("p(x)", rotation=0, labelpad=20)
    #
    title = "Beta distribution"
    process_figure(title)


# #############################################################################
# Interactive Beta Prior updater.
# #############################################################################


def _parse_trials(text: str) -> List[int]:
    """
    Parse comma-separated trial counts.

    Non-negative, unique, keep order.
    """
    try:
        vals: List[int] = [int(x.strip()) for x in text.split(",") if x.strip()]
        vals = [v for i, v in enumerate(vals) if v >= 0 and v not in vals[:i]]
        return vals if vals else [0]
    except Exception:
        return [0]


def _generate_data(
    theta_real: float, n_trials: List[int], seed: int
) -> List[int]:
    """
    Generate binomial counts y ~ Binomial(N, theta_real).

    Deterministically per index via seed.
    """
    y_vals: List[int] = []
    for idx, N in enumerate(n_trials):
        rng = np.random.default_rng(seed + idx * 1009)
        y_vals.append(rng.binomial(N, theta_real))
    return y_vals


def _validate_ab(
    a: Union[float, str],
    b: Union[float, str],
    fallback: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Ensure α, β are valid positive floats.

    Return fallback if invalid.
    """
    try:
        a_f, b_f = float(a), float(b)
        if a_f > 0 and b_f > 0:
            return a_f, b_f
        return fallback
    except Exception:
        return fallback


def beta_prior_interactive() -> None:
    """
    Create an interactive ipywidgets visualization with a single Beta prior.
    """
    # Widgets.
    theta_slider: ipywidgets.FloatSlider = ipywidgets.FloatSlider(
        value=0.35,
        min=0.0,
        max=1.0,
        step=0.01,
        description="θ (true)",
        readout_format=".2f",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=ipywidgets.Layout(width="350px"),
    )
    trials_text: ipywidgets.Text = ipywidgets.Text(
        value="0,1,2,3,4,8,16,32,64,96,128,160",
        description="n_trials",
        style={"description_width": "90px"},
        layout=ipywidgets.Layout(width="420px"),
    )
    seed_int: ipywidgets.IntText = ipywidgets.IntText(
        value=42,
        description="seed",
        style={"description_width": "90px"},
        layout=ipywidgets.Layout(width="200px"),
    )
    # Single prior parameter widgets.
    a1: ipywidgets.FloatText = ipywidgets.FloatText(
        value=1.0, description="α", layout=ipywidgets.Layout(width="150px")
    )
    b1: ipywidgets.FloatText = ipywidgets.FloatText(
        value=1.0, description="β", layout=ipywidgets.Layout(width="150px")
    )
    index_slider: ipywidgets.IntSlider = ipywidgets.IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="Index",
        style={"description_width": "90px"},
        layout=ipywidgets.Layout(width="420px"),
        continuous_update=False,
    )
    play: ipywidgets.Play = ipywidgets.Play(
        interval=600, value=0, min=0, max=0, step=1
    )
    ipywidgets.jslink((play, "value"), (index_slider, "value"))
    out: ipywidgets.Output = ipywidgets.Output()

    # Core update function
    def refresh_plot(*args) -> None:
        with out:
            clear_output(wait=True)
            theta_real: float = theta_slider.value
            n_trials: List[int] = _parse_trials(trials_text.value)
            index_slider.max = max(0, len(n_trials) - 1)
            play.max = index_slider.max
            seed: int = seed_int.value
            y_vals: List[int] = _generate_data(theta_real, n_trials, seed)
            idx: int = max(0, min(index_slider.value, len(n_trials) - 1))
            N: int = n_trials[idx]
            y: int = y_vals[idx]
            alpha_beta: Tuple[float, float] = _validate_ab(
                a1.value, b1.value, (1.0, 1.0)
            )
            alpha, beta = alpha_beta
            x: np.ndarray = np.linspace(0, 1, 400)
            post: np.ndarray = stats.beta.pdf(x, alpha + y, beta + N - y)
            ymax: float = (
                float(np.max(post))
                if np.isfinite(np.max(post)) and np.max(post) > 0
                else 1.0
            )
            ymax *= 1.1
            plt.figure(figsize=(8, 5))
            label = f"Posterior: α={alpha:g}, β={beta:g}"
            plt.fill_between(x, 0, post, alpha=0.5, label=label)
            plt.axvline(theta_real, ymax=0.3, linestyle="--")
            plt.xlabel("θ")
            plt.ylabel("density")
            plt.xlim(0, 1)
            plt.ylim(0, max(10, ymax))
            plt.legend(loc="upper left", frameon=False)
            title = f"Posterior after N={N} trials, y={y} heads"
            plt.title(title)
            plt.show()

    # Bind observers
    for w in [theta_slider, trials_text, seed_int, a1, b1, index_slider]:
        w.observe(refresh_plot, names="value")

    # Initial draw
    refresh_plot()

    # Layout.
    prior_box = ipywidgets.HBox([a1, b1])
    top_box = ipywidgets.HBox([theta_slider, seed_int])
    trials_box = ipywidgets.HBox([trials_text])
    index_box = ipywidgets.HBox([play, index_slider])
    #
    ui = ipywidgets.VBox([top_box, trials_box, prior_box, index_box, out])
    display(ui)


# #############################################################################


def update_prior() -> None:
    """
    Visualize how different Beta priors are updated with observed data.
    """
    plt.figure(figsize=(10, 8))
    theta_real = 0.35
    # 3 different Beta priors.
    prior_params = [(1, 1), (20, 20), (1, 4)]
    prior_colors = ["r", "g", "b"]
    # Observed data.
    n_trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
    data = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48]
    #
    x = np.linspace(0, 1, 200)
    #
    for idx, N in enumerate(n_trials):
        if idx == 0:
            # Plot the prior.
            plt.subplot(4, 3, 2)
            plt.xlabel("θ")
        else:
            plt.subplot(4, 3, idx + 3)
            plt.xticks([])
        y = data[idx]
        # Plot the result of applying the observed data to the 3 priors.
        for i, (a_prior, b_prior) in enumerate(prior_params):
            p_theta_given_y = stats.beta.pdf(x, a_prior + y, b_prior + N - y)
            plt.fill_between(
                x, 0, p_theta_given_y, alpha=0.7, color=prior_colors[i % 3]
            )
        # Plot the ground truth.
        plt.axvline(theta_real, ymax=0.3, color="k")
        # Legend.
        plt.plot(0, 0, label=f"{N:4d} trials\n{y:4d} heads", alpha=0)
        plt.xlim(0, 1)
        plt.ylim(0, 12)
        plt.legend()
        plt.yticks([])
    plt.tight_layout()
    #
    title = "Updating the prior"
    process_figure(title)


# #############################################################################
# Loss functions.
# #############################################################################

LossValue = Union[float, np.ndarray]

# We rely on broadcasted operations to handle both scalar and array inputs:
# y_hat     y_true    loss      comment
# -----------------------------------------------------------------------------
# scalar    scalar    scalar    Point-wise loss
# scalar    array     array     The true value is a distribution (e.g., posterior)
# array     scalar    array     Apply loss element-wise
# array     array     array     Error


def squared_loss(y_hat: LossValue, y_true: LossValue) -> LossValue:
    """
    Compute squared loss (L2).

    :param y_hat: Predicted value(s)
    :param y_true: True value(s)
    :return: Mean squared error
    """
    return np.mean((y_true - y_hat) ** 2)


def abs_loss(y_hat: LossValue, y_true: LossValue) -> LossValue:
    """
    Compute absolute loss (L1).

    :param y_hat: Predicted value(s)
    :param y_true: True value(s)
    :return: Mean absolute error
    """
    return np.mean(np.abs(y_true - y_hat))


def sin_loss(y_hat: LossValue, y_true: LossValue) -> LossValue:
    """
    Compute sinusoidal loss function.

    :param y_hat: Predicted value(s)
    :param y_true: True value(s)
    :return: Sinusoidal loss
    """
    return y_true + np.sin(2 * np.pi * y_hat) + 0.5 * y_hat


def asymmetric_loss(y_hat: LossValue, y_true: LossValue) -> LossValue:
    """
    Compute asymmetric loss function with different penalties for over/under prediction.

    :param y_hat: Predicted value(s)
    :param y_true: True value(s)
    :return: Asymmetric loss
    """
    y_hat /= 10
    if y_hat < 0.0:
        val = -np.abs(y_true - y_hat)
    else:
        val = -y_hat / (y_true - y_hat)
    return val


def plot_loss(grid: np.ndarray, loss_func: Callable) -> None:
    """
    Plot the loss function on a grid of values.

    :param grid: Array of x values
    :param loss_func: Loss function to plot
    """
    loss_values = [loss_func(i) for i in grid]
    plt.plot(grid, loss_values, label=loss_func.__name__)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Loss")
    plt.title("Loss function")


def pick_best_theta(idata: az.InferenceData) -> None:
    """
    Plot absolute and squared loss functions for theta and mark minimum points.

    :param idata: InferenceData object containing posterior distribution of theta
    """
    grid = np.linspace(0, 1, 200)
    theta_posterior = idata.to_dataframe()[("posterior", "theta")]
    # We don't have a single value for y_true, but a distribution of values.
    # For each point in the grid, compute the loss of that point vs all the
    # points in the posterior.
    # Absolute loss.
    lossf_abs = [np.mean(abs(i - theta_posterior)) for i in grid]
    # Squared loss.
    lossf_sqr = [np.mean((i - theta_posterior) ** 2) for i in grid]
    for lossf, c, tag in zip(
        [lossf_abs, lossf_sqr], ["C0", "C1"], "Absolute Squared".split()
    ):
        # Plot loss.
        plt.plot(grid, lossf, label=f"{tag} loss function")
        # Find and plot the minimum value.
        min_x = np.argmin(lossf)
        plt.plot(grid[min_x], lossf[min_x], "o", color=c)
        plt.annotate(
            "{:.2f}".format(grid[min_x]),
            (grid[min_x], lossf[min_x] + 0.03),
            color=c,
        )
    #
    plt.legend()
    plt.yticks([])
    plt.xlabel(r"$\hat{\theta}$")


# #############################################################################
# Kalman filtering and g-h filters.
# #############################################################################


def predict_using_gain_guess(
    estimated_weight: float,
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
        predicted_weight = estimated_weight + gain_rate * time_step
        # Update by blending prediction and measurement.
        estimated_weight = predicted_weight + scale_factor * (
            z - predicted_weight
        )
        # Log values.
        ests.append(estimated_weight)
        preds.append(predicted_weight)
        _LOG.debug(
            "z=%.2f pred=%.2f est=%.2f", z, predicted_weight, estimated_weight
        )
    return ests, preds


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
    if preds is not None:
        df["pred"] = preds
    df["ests"] = ests
    df["ground_truth"] = ground_truth
    # Plot measurements as points.
    df["measurements"].plot(marker="o", markersize=10, linestyle="None")
    # Plot ground truth line.
    df["ground_truth"].plot(color="k")
    # Plot predictions as dashed line.
    if preds is not None:
        df["pred"].plot(color="r", linewidth=3, linestyle="--")
    # Plot estimates as solid line.
    df["ests"].plot(color="b", linewidth=4)
    plt.legend()


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


# #############################################################################
# Animation generation utilities.
# #############################################################################


def generate_animation_values(
    mode: str,
    sweep_variable: str,
    const_variable: Optional[str] = None,
    const_value: Optional[Any] = None,
    *,
    n_steps: int = 11,
    sweep_min: float = 0.0,
    sweep_max: float = 1.0,
    **extra_constants: Any,
) -> List[dict]:
    """
    Generate a list of values for a given mode, sweep variable, and constant variable(s).

    :param mode: Mode of the sweep variable.
    :param sweep_variable: Name of the sweep variable.
    :param const_variable: Name of the constant variable (optional).
    :param const_value: Value of the constant variable (optional).
    :param n_steps: Number of steps in the sweep.
    :param sweep_min: Minimum value for the sweep variable.
    :param sweep_max: Maximum value for the sweep variable.
    :param extra_constants: Additional constant variables as keyword arguments.
    :return: List of values.
    """
    if mode == "linear":
        sweep_values = np.linspace(sweep_min, sweep_max, n_steps)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    values = []
    for val in sweep_values:
        entry = {sweep_variable: val}
        if const_variable is not None:
            entry[const_variable] = const_value
        entry.update(extra_constants)
        values.append(entry)
    return values


def generate_animation(
    functor: Callable,
    values: List[dict],
    dst_dir: str,
    *,
    incremental: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 150,
    convert_to_movie: bool = False,
) -> None:
    """
    Generate animation frames by calling a functor with different parameter values.

    The function creates a directory for frames, then iterates through the
    provided values, calling the functor with each set of kwargs. Each frame
    is saved as a PNG file with the naming pattern frame_000.png,
    frame_001.png, etc.

    :param functor: Function to call for each frame (should use plt.show())
    :param values: List of dictionaries containing kwargs to pass to functor
    :param dst_dir: Directory path where frames will be saved
    :param incremental: If False, directory is recreated from scratch; if
        True, existing directory is reused
    :param figsize: Optional figure size as (width, height) to pass to
        functor (if functor accepts figsize parameter)
    :param dpi: Resolution for saved frames in dots per inch
    :param convert_to_movie: If True, convert PNG frames to movie using
        convert_png_dir_to_movie.py script
    """
    # Create directory for frames.
    hio.create_dir(dst_dir, incremental=incremental)
    n_steps = len(values)
    _LOG.info("Generating %s frames...", n_steps)
    # Generate frames by calling the function with different parameter values.
    for i, kwargs in enumerate(values):
        _LOG.debug("Frame %s/%s: %s", i + 1, n_steps, kwargs)
        # Add figsize to kwargs if provided and not already present.
        if figsize is not None and "figsize" not in kwargs:
            kwargs = {**kwargs, "figsize": figsize}
        # Save the original plt.show.
        original_show = plt.show

        # Create a custom show function that saves the figure.
        def save_figure():
            frame_path = os.path.join(dst_dir, f"frame_{i:03d}.png")
            plt.savefig(
                frame_path,
                dpi=dpi,
                bbox_inches=None,
                facecolor="white",
            )
            plt.close()

        # Replace plt.show temporarily.
        plt.show = save_figure
        try:
            # Call the visualization function.
            functor(**kwargs)
        finally:
            # Restore original plt.show.
            plt.show = original_show
    # Report completion.
    frame_files = sorted([f for f in os.listdir(dst_dir) if f.endswith(".png")])
    _LOG.info("Frames saved to %s/", dst_dir)
    _LOG.debug("Total frames generated: %s", len(frame_files))
    # Validate that all frames have the same dimensions.
    if frame_files:
        dimensions = []
        for frame_file in frame_files:
            frame_path = os.path.join(dst_dir, frame_file)
            with Image.open(frame_path) as img:
                dimensions.append((frame_file, img.size))
        # Check if all dimensions are the same.
        unique_dimensions = set(dim[1] for dim in dimensions)
        if len(unique_dimensions) == 1:
            width, height = dimensions[0][1]
            _LOG.info(
                "All frames have consistent dimensions: %sx%s pixels",
                width,
                height,
            )
            # Convert frames to movie if requested.
            if convert_to_movie:
                _LOG.info("Converting frames to movie...")
                cmd = f"convert_png_dir_to_movie.py --input_dir {dst_dir}"
                hsystem.system(cmd)
        else:
            _LOG.warning("Inconsistent frame dimensions detected:")
            for frame_file, size in dimensions:
                _LOG.warning("  %s: %sx%s pixels", frame_file, size[0], size[1])
            hdbg.dfatal(
                "Frame dimensions are inconsistent. Expected all frames to have the same size."
            )


# #############################################################################
# Figure saving utilities.
# #############################################################################

FIG_DIR = "/app/lectures_source/figures"


def save_ax(ax: Any, file_name: str) -> None:
    """
    Save matplotlib axes figure to file and print markdown reference.

    :param ax: Matplotlib axes object
    :param file_name: Output filename
    """
    file_name = os.path.join(FIG_DIR, file_name)
    ax.figure.savefig(file_name, dpi=300, bbox_inches="tight")
    #
    file_name = file_name.replace("/app/", "")
    cmd = f"![]({file_name})"
    _LOG.info(cmd)


def save_fig(axes: Any, file_name: str) -> None:
    """
    Save matplotlib figure from axes array to file and print markdown reference.

    :param axes: Array of matplotlib axes
    :param file_name: Output filename
    """
    file_name = os.path.join(FIG_DIR, file_name)
    fig = axes[0, 0].figure
    fig.savefig(file_name, dpi=300, bbox_inches="tight")
    #
    file_name = file_name.replace("/app/", "")
    cmd = f"![]({file_name})"
    _LOG.info(cmd)


def save_dot(model: Any, file_name: str) -> None:
    """
    Save PyMC model graph to PNG file and print markdown reference.

    :param model: PyMC model object
    :param file_name: Output filename
    """
    dot = pm.model_to_graphviz(model)
    dot2 = copy.deepcopy(dot)
    file_name = file_name.replace(".png", "")
    file_name = os.path.join(FIG_DIR, file_name)
    # 300 is print quality; try 600 for very sharp images.
    dot2.graph_attr["dpi"] = "300"
    dot2.render(file_name, format="png", cleanup=True)
    #
    file_name = file_name.replace("/app/", "")
    cmd = f"![]({file_name})"
    _LOG.info(cmd)


def save_df(df: pd.DataFrame, file_name: str) -> None:
    """
    Save DataFrame as image file and print markdown reference.

    :param df: DataFrame to save
    :param file_name: Output filename
    """
    import dataframe_image as dfi

    file_name = os.path.join(FIG_DIR, file_name)
    dfi.export(df, file_name, table_conversion="matplotlib", dpi=300)
    #
    file_name = file_name.replace("/app/", "")
    cmd = f"![]({file_name})"
    _LOG.info(cmd)


def save_plt(file_name: str) -> None:
    """
    Save current matplotlib figure to file and print markdown reference.

    :param file_name: Output filename
    """
    file_name = os.path.join(FIG_DIR, file_name)
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    #
    file_name = file_name.replace("/app/", "")
    cmd = f"![]({file_name})"
    _LOG.info(cmd)
