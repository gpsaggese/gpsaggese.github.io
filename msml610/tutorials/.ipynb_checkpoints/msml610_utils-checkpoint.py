import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as stats
import ipywidgets as W
from IPython.display import clear_output, display

import helpers_root.helpers.hdbg as hdbg


_LOG = logging.getLogger(__name__)


def set_notebook_style() -> None:
    print("# Setting notebook style")
    plt.rcParams["figure.figsize"] = [8, 3]


def notebook_signature() -> None:
    print("# Notebook signature")
    cmd = "python --version"
    os.system(cmd)
    cmd = "uname -a"
    os.system(cmd)
    modules = ["numpy", "pymc", "matplotlib", "arviz", "preliz"]
    for module in modules:
        cmd = f"import {module}"
        exec(cmd)
        version = eval(f"{module}.__version__")
        print(f"{module} version={version}")


def config_notebook() -> None:
    set_notebook_style()
    notebook_signature()


def obj_to_str(var_name: str, val: any, top_n: int = 3) -> str:
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


def print_obj(*args: any, **kwargs: any) -> None:
    print(obj_to_str(*args, **kwargs))


# Lesson 7, notebook 1


def convert_to_filename(string: str) -> str:
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
    txt = f"![]({file_name})" + "{ width=100px }"
    print(txt)


def process_figure(title: str) -> None:
    file_name = convert_to_filename(title)
    plt.savefig(file_name, dpi=300)
    # print_figure(file_name)


def plot_binomial() -> None:
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
        # Fit plots into the figure cleanly.
        constrained_layout=True,
    )
    # constrained_layout=False)
    print(ax.shape)
    for i in range(len(n_params)):
        for j in range(len(p_params)):
            x = list(range(0, max_n))
            n = n_params[i]
            p = p_params[j]
            # Evaluate the PDF in several points.
            y = stats.binom(n=n, p=p).pmf(x)
            y = [y[k] if k <= n else np.nan for k in range(max_n)]
            # print(n, p, x, y)
            # Plot the PDF.
            # ax[i, j].plot(x, y, marker='o', linestyle='--')
            ax[i, j].bar(x, y)  # vertical bars
            # Add the legend.
            ax[i, j].plot([], label="n={:3.2f}\np={:3.2f}".format(n, p), alpha=0)
            ax[i, j].legend(loc="best")
    ax[2, 1].set_xlabel("x")
    ax[1, 0].set_ylabel("p(x)", rotation=0, labelpad=20)
    # ax[1, 0].set_yticks([])
    ax[1, 0].set_xticks(range(0, max_n))
    #
    title = "Binomial distribution"
    process_figure(title)


def plot_beta() -> None:
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
        # Fit plots into the figure cleanly.
        constrained_layout=True,
    )
    # constrained_layout=False)
    for i in range(len(a_params)):
        for j in range(len(b_params)):
            alpha = a_params[i]
            beta = b_params[j]
            # Evaluate the PDF in several points.
            y = stats.beta(a=alpha, b=beta).pdf(x)
            # Plot the PDF.
            ax[i, j].plot(x, y)
            # Add the legend.
            ax[i, j].plot(
                [], label="a={:3.2f}\nb={:3.2f}".format(alpha, beta), alpha=0
            )
            ax[i, j].legend(loc=1)
    ax[2, 1].set_xlabel("x")
    ax[1, 0].set_ylabel("p(x)", rotation=0, labelpad=20)
    #
    title = "Beta distribution"
    process_figure(title)


# #############################################################################

# Interactive Beta Prior updater


def _parse_trials(text: str) -> List[int]:
    """
    Parse comma-separated trial counts (non-negative, unique, keep order).
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
    Generate binomial counts y ~ Binomial(N, theta_real) deterministically per
    index via seed.
    """
    y_vals: List[int] = []
    for idx, N in enumerate(n_trials):
        rng = np.random.default_rng(seed + idx * 1009)
        y_vals.append(rng.binomial(N, theta_real))
    return y_vals


def _validate_ab(
    a: float | str, b: float | str, fallback: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Ensure α, β are valid positive floats; otherwise return fallback.
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
    # Widgets
    theta_slider: W.FloatSlider = W.FloatSlider(
        value=0.35,
        min=0.0,
        max=1.0,
        step=0.01,
        description="θ (true)",
        readout_format=".2f",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=W.Layout(width="350px"),
    )
    trials_text: W.Text = W.Text(
        value="0,1,2,3,4,8,16,32,64,96,128,160",
        description="n_trials",
        style={"description_width": "90px"},
        layout=W.Layout(width="420px"),
    )
    seed_int: W.IntText = W.IntText(
        value=42,
        description="seed",
        style={"description_width": "90px"},
        layout=W.Layout(width="200px"),
    )
    # Single prior parameter widgets
    a1: W.FloatText = W.FloatText(
        value=1.0, description="α", layout=W.Layout(width="150px")
    )
    b1: W.FloatText = W.FloatText(
        value=1.0, description="β", layout=W.Layout(width="150px")
    )
    index_slider: W.IntSlider = W.IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="Index",
        style={"description_width": "90px"},
        layout=W.Layout(width="420px"),
        continuous_update=False,
    )
    play: W.Play = W.Play(interval=600, value=0, min=0, max=0, step=1)
    W.jslink((play, "value"), (index_slider, "value"))
    out: W.Output = W.Output()

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
            plt.fill_between(
                x, 0, post, alpha=0.5, label=f"Posterior: α={alpha:g}, β={beta:g}"
            )
            plt.axvline(theta_real, ymax=0.3, linestyle="--")
            plt.xlabel("θ")
            plt.ylabel("density")
            plt.xlim(0, 1)
            plt.ylim(0, max(10, ymax))
            plt.legend(loc="upper left", frameon=False)
            plt.title(f"Posterior after N={N} trials, y={y} heads")
            plt.show()

    # Bind observers
    for w in [theta_slider, trials_text, seed_int, a1, b1, index_slider]:
        w.observe(refresh_plot, names="value")

    # Initial draw
    refresh_plot()

    # Layout
    prior_box = W.HBox([a1, b1])
    top_box = W.HBox([theta_slider, seed_int])
    trials_box = W.HBox([trials_text])
    index_box = W.HBox([play, index_slider])

    ui = W.VBox([top_box, trials_box, prior_box, index_box, out])
    display(ui)


# #############################################################################


def update_prior() -> None:
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


#
LossValue = Union[float, np.array]


# We rely on brodcasted operations to handle both scalar and array inputs, e.g.,
# y_hat     y_true    loss      comment
# -----------------------------------------------------------------------------
# scalar    scalar    scalar    Point-wise loss
# scalar    array     array     The true value is a distribution (e.g., posterior)
# array     scalar    array     Apply loss element-wise
# array     array     array     Error


def squared_loss(y_hat: LossValue, y_true: LossValue) -> LossValue:
    return np.mean((y_true - y_hat) ** 2)


def abs_loss(y_hat: LossValue, y_true: LossValue) -> LossValue:
    return np.mean(np.abs(y_true - y_hat))


def sin_loss(y_hat: LossValue, y_true: LossValue) -> LossValue:
    return y_true + np.sin(2 * np.pi * y_hat) + 0.5 * y_hat


def asymmetric_loss(y_hat: LossValue, y_true: LossValue) -> LossValue:
    y_hat /= 10
    if y_hat < 0.0:
        val = -np.abs(y_true - y_hat)
    else:
        val = -y_hat / (y_true - y_hat)
    return val


def plot_loss(grid: np.array, loss_func: Callable) -> None:
    """
    Plot the loss function on a grid of values.
    """
    loss_values = [loss_func(i) for i in grid]
    plt.plot(grid, loss_values, label=loss_func.__name__)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Loss")
    plt.title("Loss function")


def pick_best_theta(idata: az.InferenceData) -> None:
    """
    Plot the absolute and squared loss functions for a range of theta values
    and mark the minimum loss points on the plot.

    :param idata: InferenceData object containing the posterior
        distribution of theta.
    """
    grid = np.linspace(0, 1, 200)
    theta_posterior = idata.to_dataframe()[("posterior", "theta")]
    # E.g.,
    # 0       0.145339
    # 1       0.146737
    # 2       0.040329
    # 3       0.109264
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


# Gaussian inference

# #############################################################################
# Lesson 8
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
    measurements: np.array,
    preds: List[float],
    ests: List[float],
    ground_truth: List[float],
    *,
    tag_measurements: str = "measurements",
) -> None:
    """
    Plot weight gain data including measurements, ground truth, predictions and
    estimates.

    :param df: DataFrame containing weight data with columns:
        - measurements: actual weight measurements
        - ground_truth: true weight values
        - pred: predicted weights
        - ests: estimated weights
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
    measures: np.array,
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
    data: np.array,
    x0: float,
    dx: float,
    g: float,
    h: float,
    *,
    dt: float = 1.0,
) -> np.array:
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
    Generate random data starting from x0, with slope dx, affected by additive
    random noise N(0, noise_factor).

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
    Generate random data with acceleration, starting from x0, with initial
    slope dx, affected by additive random noise N(0, noise_factor).

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


# Discrete Bayes Filter

import matplotlib.pyplot as plt
import numpy as np


def plot_dog_in_office_pdf(
    probs: Union[List[float], np.ndarray],
    *,
    hallway: Optional[np.ndarray] = None,
    title: str = "Probability Histogram",
) -> None:
    """
    Plot a histogram-like bar chart of class probabilities.

    :param probabilities: List or numpy array of class probabilities
    :param title: Title for the plot
    """
    hdbg.dassert_isinstance(probs, (list, np.ndarray))
    hdbg.dassert_lte(0.0, np.min(probs))
    hdbg.dassert_lte(np.max(probs), 1.0)
    # Check that the sum of probabilities is 1.0.
    hdbg.dassert_lte(0.99, np.sum(probs))
    hdbg.dassert_lte(np.sum(probs), 1.01)
    indices = np.arange(len(probs))
    if hallway is None:
        hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    hdbg.dassert_eq(len(probs), len(hallway))
    # Create plot.
    # plt.figure(figsize=(8, 4))
    plt.bar(indices, probs, color="deepskyblue")
    # Add markers for hallway positions with value 1
    for i, val in enumerate(hallway):
        if val == 1:
            plt.plot(i, 0.0, "^r", markersize=20, label="Door" if i == 0 else "")
    plt.ylim(0, 1)
    plt.xlabel("Class Index")
    plt.ylabel("Probability")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()


# #############################################################################
# Save figures.
# #############################################################################


fig_dir = "/app/lectures_source/figures"
import copy
import os

#!sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet dataframe_image)"

import dataframe_image as dfi


def save_ax(ax, file_name):
    file_name = os.path.join(fig_dir, file_name)
    ax.figure.savefig(file_name, dpi=300, bbox_inches="tight")
    #
    file_name = file_name.replace("/app/", "")
    cmd = f"![]({file_name})"
    print(cmd)


def save_fig(axes, file_name):
    file_name = os.path.join(fig_dir, file_name)
    fig = axes[0, 0].figure
    fig.savefig(file_name, dpi=300, bbox_inches="tight")
    #
    file_name = file_name.replace("/app/", "")
    cmd = f"![]({file_name})"
    print(cmd)


def save_dot(model, file_name):
    dot = pm.model_to_graphviz(model)
    dot2 = copy.deepcopy(dot)
    file_name = file_name.replace(".png", "")
    file_name = os.path.join(fig_dir, file_name)
    dot2.graph_attr["dpi"] = (
        "300"  # 300 is print quality; try 600 for very sharp images
    )
    dot2.render(file_name, format="png", cleanup=True)
    # dot.graph_attr['dpi'] = '96'  # 300 is print quality; try 600 for very sharp images
    #
    file_name = file_name.replace("/app/", "")
    cmd = f"![]({file_name})"
    print(cmd)


def save_df(df, file_name):
    file_name = os.path.join(fig_dir, file_name)
    dfi.export(df, file_name, table_conversion="matplotlib", dpi=300)
    #
    file_name = file_name.replace("/app/", "")
    cmd = f"![]({file_name})"
    print(cmd)


def save_plt(file_name):
    file_name = os.path.join(fig_dir, file_name)
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    #
    file_name = file_name.replace("/app/", "")
    cmd = f"![]({file_name})"
    print(cmd)
