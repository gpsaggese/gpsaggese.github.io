"""
Utility functions for Bayesian coin flip tutorial (L07_01).

Import as:

import msml610.tutorials.L07_01_bayesian_coin_utils as mtubacoum
"""

import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import arviz as az
import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from IPython.display import clear_output

import msml610.tutorials.msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)


# #############################################################################
# Probability distributions.
# #############################################################################


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
    mtumsuti.process_figure(title)


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
    mtumsuti.process_figure(title)


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
    ipywidgets.display(ui)


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
    mtumsuti.process_figure(title)


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
