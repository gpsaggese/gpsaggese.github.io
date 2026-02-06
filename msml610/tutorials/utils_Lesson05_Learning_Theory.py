"""
Utility functions for Learning Theory lesson.

Import as:

import msml610.tutorials.utils_Lesson05_Learning_Theory as mtulleth
"""

import logging
import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import Output, VBox
from scipy.stats import norm

import helpers.hdbg as hdbg
import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)

# Suppress FutureWarnings from seaborn and other libraries.
warnings.filterwarnings("ignore", category=FutureWarning)


# #############################################################################
# Interactive Widget Setup Functions
# #############################################################################


# TODO(ai_gp): sample_bernoulli1
def sample_bernoulli_with_pdf() -> None:
    """
    Create interactive Bernoulli sampling visualization with PDF comparison.

    Sets up an interactive widget that allows exploration of Bernoulli sampling
    behavior. The widget displays three synchronized plots showing: (1) temporal
    sequence of sampled outcomes, (2) comparison of empirical vs theoretical
    probability distributions, and (3) detailed statistics and interpretation.
    Users can adjust the success probability (mu), sample size (N), and random
    seed to observe how empirical results converge to theoretical predictions.
    """
    mu_init = 0.6
    N_init = 100
    seed_init = 42
    # Create widgets.
    mu_slider, mu_box = mtumsuti.build_widget_control(
        name="mu",
        description="prob of success",
        min_val=0.1,
        max_val=0.9,
        step=0.05,
        initial_value=mu_init,
        is_float=True,
    )
    N_slider, N_box = mtumsuti.build_widget_control(
        name="N",
        description="number of samples",
        min_val=10,
        max_val=500,
        step=10,
        initial_value=N_init,
        is_float=False,
    )
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=1000,
        step=1,
        initial_value=seed_init,
        is_float=False,
    )
    # Create output widget.
    output = Output()

    def update_plot(change=None):
        """
        Update plot when slider values change.
        """
        with output:
            output.clear_output(wait=True)
            plot_bernoulli_samples_and_pdf(
                mu=mu_slider.value, N=N_slider.value, seed=seed_slider.value
            )

    # Observe slider changes.
    mu_slider.observe(update_plot, names="value")
    N_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    # Display widgets and initial plot.
    display(VBox([mu_box, N_box, seed_box, output]))
    update_plot()


# TODO(ai_gp): Rename sample_bernoulli2
def create_interactive_widget_cell3(
    plot_func: Callable,
    *,
    mu_init: float = 0.6,
    N_init: int = 100,
    seed_init: int = 42,
) -> None:
    """
    Create interactive widget for Cell 3 (PDF and Statistics).

    Sets up complete interactive widget with sliders for mu, N, and seed
    parameters, connecting them to the provided plotting function.

    :param plot_func: Plotting function to call (e.g., plot_bernoulli_pdf_cell2)
    :param mu_init: Initial value for mu (probability of success)
    :param N_init: Initial value for N (number of samples)
    :param seed_init: Initial value for seed
    """
    # Create widgets.
    mu_slider, mu_box = mtumsuti.build_widget_control(
        name="mu",
        description="prob of success",
        min_val=0.1,
        max_val=0.9,
        step=0.05,
        initial_value=mu_init,
        is_float=True,
    )
    N_slider, N_box = mtumsuti.build_widget_control(
        name="N",
        description="number of samples",
        min_val=10,
        max_val=500,
        step=10,
        initial_value=N_init,
        is_float=False,
    )
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=1000,
        step=1,
        initial_value=seed_init,
        is_float=False,
    )
    # Create output widget.
    output = Output()

    def update_plot(change=None):
        """Update plot when slider values change."""
        with output:
            output.clear_output(wait=True)
            plot_func(
                mu=mu_slider.value, N=N_slider.value, seed=seed_slider.value
            )

    # Observe slider changes.
    mu_slider.observe(update_plot, names="value")
    N_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    # Display widgets and initial plot.
    display(VBox([mu_box, N_box, seed_box, output]))
    update_plot()


def create_interactive_widget_cell4(
    *,
    mu_init: float = 0.6,
    N_init: int = 100,
    n_samples_init: int = 1000,
    seed_init: int = 42,
) -> None:
    """
    Create interactive widget for Cell 4 (Distribution of Empirical Mean).

    Sets up complete interactive widget with sliders for mu, N, n_samples,
    and seed parameters, connecting them to the provided plotting function.

    :param plot_func: Plotting function to call (e.g., plot_empirical_mean_distribution_cell3)
    :param mu_init: Initial value for mu (probability of success)
    :param N_init: Initial value for N (samples per trial)
    :param n_samples_init: Initial value for n_samples (number of trials)
    :param seed_init: Initial value for seed
    """
    # Create widgets.
    mu_slider, mu_box = mtumsuti.build_widget_control(
        name="mu",
        description="prob of success",
        min_val=0.1,
        max_val=0.9,
        step=0.05,
        initial_value=mu_init,
        is_float=True,
    )
    N_slider, N_box = mtumsuti.build_widget_control(
        name="N",
        description="samples per trial",
        min_val=10,
        max_val=500,
        step=10,
        initial_value=N_init,
        is_float=False,
    )
    n_samples_slider, n_samples_box = mtumsuti.build_widget_control(
        name="n_samples",
        description="number of trials",
        min_val=100,
        max_val=5000,
        step=100,
        initial_value=n_samples_init,
        is_float=False,
    )
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=1000,
        step=1,
        initial_value=seed_init,
        is_float=False,
    )
    # Create output widget.
    output = Output()

    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)
            plot_bernoulli_pdf_cell2(
                mu=mu_slider.value,
                N=N_slider.value,
                n_samples=n_samples_slider.value,
                seed=seed_slider.value,
            )

    # Observe slider changes.
    mu_slider.observe(update_plot, names="value")
    N_slider.observe(update_plot, names="value")
    n_samples_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    # Display widgets and initial plot.
    display(VBox([mu_box, N_box, n_samples_box, seed_box, output]))
    update_plot()


## #############################################################################
## Hoeffding Inequality: Step 1 Cell 2 - Samples and PDF
## #############################################################################
#
#
#def plot_bernoulli_samples_and_pdf(
#    *,
#    mu: float = 0.6,
#    N: int = 100,
#    seed: int = 42,
#) -> None:
#    """
#    Display N samples from Bernoulli distribution over time and empirical PDF.
#
#    Shows the temporal sequence of samples, empirical PDF, and explanatory
#    comments in a single row of 3 plots following interactive widget
#    conventions.
#
#    :param mu: True probability of success (0 < mu < 1)
#    :param N: Number of samples to draw
#    :param seed: Random seed for reproducibility
#    """
#    # Validate parameters.
#    hdbg.dassert_lte(0, mu, "mu must be positive:", mu)
#    hdbg.dassert_lte(mu, 1, "mu must be at most 1:", mu)
#    hdbg.dassert_lte(1, N, "N must be at least 1:", N)
#    # Set random seed for reproducibility.
#    np.random.seed(seed)
#    # Generate N Bernoulli samples.
#    samples = np.random.binomial(1, mu, size=N)
#    # Compute statistics.
#    n_successes = np.sum(samples)
#    n_failures = N - n_successes
#    empirical_prob = n_successes / N
#    # Create visualization with 3 subplots in a single row.
#    fig, (ax1, ax2, ax3) = plt.subplots(
#        1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 1, 1.5]}
#    )
#    # Plot 1: Samples over time.
#    time_indices = np.arange(N)
#    colors = ["red" if s == 0 else "green" for s in samples]
#    ax1.scatter(time_indices, samples, c=colors, alpha=0.6, s=30)
#    ax1.set_xlabel("Sample Index (Time)", fontsize=12)
#    ax1.set_ylabel("Outcome", fontsize=12)
#    ax1.set_title(
#        "Bernoulli Samples Over Time",
#        fontsize=14,
#        fontweight="bold",
#    )
#    ax1.set_yticks([0, 1])
#    ax1.set_yticklabels(["Failure (0)", "Success (1)"])
#    ax1.grid(True, alpha=0.3, axis="x")
#    # Add horizontal line for expected probability.
#    ax1.axhline(
#        y=mu,
#        color="blue",
#        linestyle="--",
#        linewidth=2,
#        alpha=0.7,
#        label=f"mu={mu:.2f}",
#    )
#    ax1.legend(fontsize=10, loc="upper right")
#    # Plot 2: Empirical PDF with side-by-side bars for comparison.
#    sample_counts = pd.Series(samples).value_counts().sort_index()
#    # Normalize to get probabilities.
#    sample_probs = sample_counts / N
#    # Prepare data for both outcomes (0 and 1).
#    outcomes = np.array([0, 1])
#    empirical_probs = np.array(
#        [
#            sample_probs.get(0, 0),
#            sample_probs.get(1, 0),
#        ]
#    )
#    theoretical_probs = np.array([1 - mu, mu])
#    # Set bar width and positions for side-by-side display.
#    bar_width = 0.35
#    x_positions = outcomes
#    # Plot empirical probabilities with darker solid bars.
#    ax2.bar(
#        x_positions - bar_width / 2,
#        empirical_probs,
#        width=bar_width,
#        color=["darkred", "darkgreen"],
#        alpha=0.85,
#        edgecolor="black",
#        linewidth=1.5,
#        label="Empirical",
#    )
#    # Plot theoretical probabilities with lighter bars.
#    ax2.bar(
#        x_positions + bar_width / 2,
#        theoretical_probs,
#        width=bar_width,
#        color="steelblue",
#        alpha=0.5,
#        edgecolor="black",
#        linewidth=1.5,
#        label="Theoretical",
#    )
#    ax2.set_ylabel("Probability", fontsize=12)
#    ax2.set_xlabel("Outcome", fontsize=12)
#    ax2.set_title("Empirical PDF", fontsize=14, fontweight="bold")
#    ax2.set_ylim([0, 1])
#    ax2.legend(fontsize=10)
#    ax2.grid(True, alpha=0.3, axis="y")
#    # Plot 3: Comments and explanation.
#    ax3.axis("off")
#    ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
#    # Generate interpretation text.
#    text_content = (
#        f"Parameters:\n"
#        f"  mu = {mu:.4f}\n"
#        f"    (true probability)\n"
#        f"  N = {N}\n"
#        f"    (number of samples)\n"
#        f"  seed = {seed}\n\n"
#        f"Sample Statistics:\n"
#        f"  Successes (1): {n_successes}\n"
#        f"  Failures (0): {n_failures}\n"
#        f"  Empirical prob: {empirical_prob:.4f}\n\n"
#        f"Interpretation:\n"
#        f"  Each sample is an\n"
#        f"  independent Bernoulli trial\n"
#        f"  with success probability mu.\n\n"
#        f"  The left plot shows samples\n"
#        f"  as they occur over time.\n\n"
#        f"  The center plot compares the\n"
#        f"  empirical PDF (bars) with\n"
#        f"  the theoretical probabilities\n"
#        f"  (blue line)."
#    )
#    ax3.text(
#        0.05,
#        0.95,
#        text_content,
#        transform=ax3.transAxes,
#        fontsize=11,
#        ha="left",
#        va="top",
#        family="monospace",
#        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
#        wrap=True,
#    )
#    # Use subplots_adjust for consistent spacing.
#    plt.subplots_adjust(
#        left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.30
#    )
#    plt.show()
#
#
## #############################################################################
## Hoeffding Inequality: Step 1 Cell 3 - PDF and Statistics
## #############################################################################
#
#
#def plot_bernoulli_pdf_cell2(
#    *,
#    mu: float = 0.6,
#    N: int = 100,
#    seed: int = 42,
#) -> None:
#    """
#    Display PDF, empirical mean, and statistics of Bernoulli samples.
#
#    Shows the probability distribution of N samples, the empirical mean nu,
#    and compares it with the true mean and variance in a single row of 3-4
#    plots following interactive widget conventions.
#
#    :param mu: True probability of success (0 < mu < 1)
#    :param N: Number of samples to draw
#    :param seed: Random seed for reproducibility
#    """
#    # Validate parameters.
#    hdbg.dassert_lte(0, mu, "mu must be positive:", mu)
#    hdbg.dassert_lte(mu, 1, "mu must be at most 1:", mu)
#    hdbg.dassert_lte(1, N, "N must be at least 1:", N)
#    # Set random seed for reproducibility.
#    np.random.seed(seed)
#    # Generate N Bernoulli samples.
#    samples = np.random.binomial(1, mu, size=N)
#    # Compute the sample mean nu.
#    nu = np.mean(samples)
#    # Compute theoretical mean and variance of Bernoulli distribution.
#    theoretical_mean = mu
#    theoretical_variance = mu * (1 - mu)
#    # Compute sample variance.
#    sample_variance = np.var(samples, ddof=1)
#    # Create visualization with 3 subplots in a single row.
#    fig, (ax1, ax2, ax3) = plt.subplots(
#        1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 1, 1.2]}
#    )
#    # Plot 1: PDF of samples.
#    sample_counts = pd.Series(samples).value_counts().sort_index()
#    # Normalize to get probabilities.
#    sample_probs = sample_counts / N
#    # Ensure Outcome is numeric to avoid matplotlib warning.
#    outcomes = [0, 1]
#    probabilities = [
#        sample_probs.get(0, 0),
#        sample_probs.get(1, 0),
#    ]
#    # Plot empirical data with darker solid bars.
#    ax1.bar(
#        outcomes,
#        probabilities,
#        color=["darkred", "darkgreen"],
#        alpha=0.85,
#        edgecolor="black",
#        linewidth=1.5,
#        label="Empirical",
#        width=0.4,
#    )
#    # Add theoretical probabilities with lighter, transparent, dotted line.
#    ax1.plot(
#        [0, 1],
#        [1 - mu, mu],
#        "o--",
#        color="steelblue",
#        markersize=10,
#        linewidth=2,
#        label="Theoretical",
#        alpha=0.5,
#    )
#    ax1.set_ylabel("Probability", fontsize=12)
#    ax1.set_xlabel("Outcome", fontsize=12)
#    ax1.set_title("PDF of Samples", fontsize=14, fontweight="bold")
#    ax1.set_ylim([0, 1])
#    ax1.legend(fontsize=10)
#    ax1.grid(True, alpha=0.3, axis="y")
#    # Plot 2: Comparison of empirical vs theoretical statistics.
#    metrics = ["Mean", "Variance"]
#    empirical = [nu, sample_variance]
#    theoretical = [theoretical_mean, theoretical_variance]
#    x_pos = np.arange(len(metrics))
#    width = 0.35
#    ax2.bar(
#        x_pos - width / 2,
#        empirical,
#        width,
#        label="Empirical",
#        color="steelblue",
#        alpha=0.7,
#        edgecolor="black",
#    )
#    ax2.bar(
#        x_pos + width / 2,
#        theoretical,
#        width,
#        label="Theoretical",
#        color="coral",
#        alpha=0.7,
#        edgecolor="black",
#    )
#    ax2.set_ylabel("Value", fontsize=12)
#    ax2.set_xlabel("Statistic", fontsize=12)
#    ax2.set_title("Statistics Comparison", fontsize=14, fontweight="bold")
#    ax2.set_xticks(x_pos)
#    ax2.set_xticklabels(metrics)
#    ax2.legend(fontsize=10)
#    ax2.grid(True, alpha=0.3, axis="y")
#    # Add value labels on bars.
#    for i, (emp, theo) in enumerate(zip(empirical, theoretical)):
#        ax2.text(
#            i - width / 2,
#            emp + 0.02,
#            f"{emp:.4f}",
#            ha="center",
#            va="bottom",
#            fontsize=9,
#        )
#        ax2.text(
#            i + width / 2,
#            theo + 0.02,
#            f"{theo:.4f}",
#            ha="center",
#            va="bottom",
#            fontsize=9,
#        )
#    # Plot 3: Comments and explanation.
#    ax3.axis("off")
#    ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
#    # Generate interpretation text.
#    error = abs(nu - mu)
#    text_content = (
#        f"Parameters:\n"
#        f"  mu = {mu:.4f}\n"
#        f"    (true probability)\n"
#        f"  N = {N}\n"
#        f"    (number of samples)\n"
#        f"  seed = {seed}\n\n"
#        f"Empirical Statistics:\n"
#        f"  nu (sample mean) = {nu:.4f}\n"
#        f"  Sample variance = {sample_variance:.4f}\n\n"
#        f"Theoretical Statistics:\n"
#        f"  Mean = {theoretical_mean:.4f}\n"
#        f"  Variance = {theoretical_variance:.4f}\n"
#        f"    (= mu * (1-mu))\n\n"
#        f"Error:\n"
#        f"  |nu - mu| = {error:.4f}\n\n"
#        f"Interpretation:\n"
#        f"  The empirical mean nu\n"
#        f"  estimates the true mean mu.\n"
#        f"  Change seed to see new\n"
#        f"  realizations with different\n"
#        f"  empirical values."
#    )
#    ax3.text(
#        0.05,
#        0.95,
#        text_content,
#        transform=ax3.transAxes,
#        fontsize=10,
#        ha="left",
#        va="top",
#        family="monospace",
#        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
#        wrap=True,
#    )
#    # Use subplots_adjust for consistent spacing.
#    plt.subplots_adjust(
#        left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.30
#    )
#    plt.show()
#
#
## #############################################################################
## Hoeffding Inequality: Step 1 Cell 4 - Distribution of Empirical Mean
## #############################################################################
#
#
#def plot_empirical_mean_distribution_cell3(
#    *,
#    mu: float = 0.6,
#    N: int = 100,
#    n_samples: int = 1000,
#    seed: int = 42,
#) -> None:
#    """
#    Display distribution of empirical mean nu from repeated sampling.
#
#    Shows the empirical distribution of the sample mean nu when n_samples
#    independent samples are generated, and compares it with the expected
#    distribution predicted by the Central Limit Theorem, in a single row
#    of plots following interactive widget conventions.
#
#    :param mu: True probability of success (0 < mu < 1)
#    :param N: Number of samples in each trial
#    :param n_samples: Number of trials to generate empirical distribution
#    :param seed: Random seed for reproducibility
#    """
#    # Validate parameters.
#    hdbg.dassert_lte(0, mu, "mu must be positive:", mu)
#    hdbg.dassert_lte(mu, 1, "mu must be at most 1:", mu)
#    hdbg.dassert_lte(1, N, "N must be at least 1:", N)
#    hdbg.dassert_lte(1, n_samples, "n_samples must be at least 1:", n_samples)
#    # Set random seed for reproducibility.
#    np.random.seed(seed)
#    # Generate empirical distribution by repeated sampling.
#    empirical_nus = []
#    for _ in range(n_samples):
#        trial_samples = np.random.binomial(1, mu, size=N)
#        trial_nu = np.mean(trial_samples)
#        empirical_nus.append(trial_nu)
#    empirical_nus = np.array(empirical_nus)
#    # Compute expected distribution parameters using Central Limit Theorem.
#    # For large N, the sample mean follows a normal distribution:
#    # nu ~ N(mu, sqrt(mu * (1-mu) / N))
#    expected_mean = mu
#    expected_std = np.sqrt(mu * (1 - mu) / N)
#    # Compute empirical statistics.
#    empirical_mean = np.mean(empirical_nus)
#    empirical_std = np.std(empirical_nus)
#    # Create visualization with 2 subplots in a single row.
#    fig, (ax1, ax2) = plt.subplots(
#        1, 2, figsize=(18, 5), gridspec_kw={"width_ratios": [1.5, 1]}
#    )
#    # Plot 1: Distribution of empirical mean nu.
#    # Plot empirical data with darker color and solid bars.
#    ax1.hist(
#        empirical_nus,
#        bins=30,
#        density=True,
#        alpha=0.85,
#        color="darkblue",
#        edgecolor="black",
#        linewidth=1.5,
#        label=f"Empirical (n={n_samples})",
#    )
#    # Plot expected normal distribution with lighter, transparent, dotted line.
#    x_range = np.linspace(
#        expected_mean - 4 * expected_std,
#        expected_mean + 4 * expected_std,
#        200,
#    )
#    y_expected = norm.pdf(x_range, expected_mean, expected_std)
#    ax1.plot(
#        x_range,
#        y_expected,
#        "--",
#        color="coral",
#        linewidth=2.5,
#        label="Theoretical (CLT)",
#        alpha=0.5,
#    )
#    # Mark the true mean.
#    ax1.axvline(
#        x=mu,
#        color="green",
#        linestyle="--",
#        linewidth=2,
#        label=f"True mu={mu:.4f}",
#    )
#    ax1.set_xlabel("Sample Mean (nu)", fontsize=12)
#    ax1.set_ylabel("Density", fontsize=12)
#    ax1.set_title(
#        "Distribution of Empirical Mean nu",
#        fontsize=14,
#        fontweight="bold",
#    )
#    ax1.legend(fontsize=10)
#    ax1.grid(True, alpha=0.3)
#    # Plot 2: Comments and explanation.
#    ax2.axis("off")
#    ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
#    # Generate interpretation text.
#    if N < 30:
#        sample_size_note = (
#            "Sample size is small.\nCLT approximation may\nnot be accurate."
#        )
#    elif N < 100:
#        sample_size_note = (
#            "Sample size is moderate.\nCLT approximation is\nreasonable."
#        )
#    else:
#        sample_size_note = (
#            "Sample size is large.\nCLT approximation is\nvery accurate."
#        )
#    text_content = (
#        f"Parameters:\n"
#        f"  mu = {mu:.4f}\n"
#        f"    (true probability)\n"
#        f"  N = {N}\n"
#        f"    (samples per trial)\n"
#        f"  n_samples = {n_samples}\n"
#        f"    (number of trials)\n"
#        f"  seed = {seed}\n\n"
#        f"Empirical Statistics:\n"
#        f"  Mean = {empirical_mean:.4f}\n"
#        f"  Std Dev = {empirical_std:.4f}\n\n"
#        f"Expected by CLT:\n"
#        f"  Mean = {expected_mean:.4f}\n"
#        f"  Std Dev = {expected_std:.4f}\n"
#        f"    (= sqrt(mu(1-mu)/N))\n\n"
#        f"Interpretation:\n"
#        f"  {sample_size_note}\n\n"
#        f"  By Central Limit Theorem,\n"
#        f"  nu ~ N(mu, sqrt(mu(1-mu)/N))\n\n"
#        f"  As N increases, the\n"
#        f"  distribution becomes more\n"
#        f"  concentrated around mu."
#    )
#    ax2.text(
#        0.05,
#        0.95,
#        text_content,
#        transform=ax2.transAxes,
#        fontsize=10,
#        ha="left",
#        va="top",
#        family="monospace",
#        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
#        wrap=True,
#    )
#    # Use subplots_adjust for consistent spacing.
#    plt.subplots_adjust(
#        left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.25
#    )
#    plt.show()
