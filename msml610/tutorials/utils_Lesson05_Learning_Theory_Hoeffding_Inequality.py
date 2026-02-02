"""
Utility functions for Learning Theory lesson - Hoeffding Inequality.

Import as:

import msml610.tutorials.utils_Lesson05_Learning_Theory_Hoeffding_Inequality as mtullthin
"""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets
import scipy.stats
from IPython.display import display

import helpers.hdbg as hdbg
import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)

# Suppress FutureWarnings from seaborn and other libraries.
warnings.filterwarnings("ignore", category=FutureWarning)


# #############################################################################
# Helper functions
# #############################################################################


def _validate_bernoulli_params(mu: float, N: int, n_samples: int = None) -> None:
    """
    Validate parameters for Bernoulli sampling functions.

    :param mu: Probability of success (must be in [0, 1])
    :param N: Number of samples (must be >= 1)
    :param n_samples: Optional number of trials (must be >= 1 if provided)
    """
    hdbg.dassert_lte(0, mu, "mu must be positive:", mu)
    hdbg.dassert_lte(mu, 1, "mu must be at most 1:", mu)
    hdbg.dassert_lte(1, N, "N must be at least 1:", N)
    if n_samples is not None:
        hdbg.dassert_lte(
            1, n_samples, "n_samples must be at least 1:", n_samples
        )


def _generate_bernoulli_samples(mu: float, N: int, seed: int) -> np.ndarray:
    """
    Generate N Bernoulli samples with a fixed random seed.

    :param mu: Probability of success
    :param N: Number of samples
    :param seed: Random seed for reproducibility
    :return: Array of Bernoulli samples (0s and 1s)
    """
    np.random.seed(seed)
    return np.random.binomial(1, mu, size=N)


def _plot_bernoulli_pdf_bars(
    ax,
    samples: np.ndarray,
    mu: float,
    N: int,
    title: str = "Empirical PDF",
) -> None:
    """
    Plot side-by-side bars comparing empirical and theoretical Bernoulli PDF.

    :param ax: Matplotlib axes to plot on
    :param samples: Array of Bernoulli samples (0s and 1s)
    :param mu: True probability parameter
    :param N: Number of samples
    :param title: Title for the plot
    """
    sample_counts = pd.Series(samples).value_counts().sort_index()
    # Normalize to get probabilities.
    sample_probs = sample_counts / N
    # Prepare data for both outcomes (0 and 1).
    outcomes = np.array([0, 1])
    empirical_probs = np.array(
        [
            sample_probs.get(0, 0),
            sample_probs.get(1, 0),
        ]
    )
    theoretical_probs = np.array([1 - mu, mu])
    # Set bar width and positions for side-by-side display.
    bar_width = 0.35
    x_positions = outcomes
    # Plot empirical probabilities with darker solid bars.
    ax.bar(
        x_positions - bar_width / 2,
        empirical_probs,
        width=bar_width,
        color=["darkred", "darkgreen"],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.5,
        label="Empirical",
    )
    # Plot theoretical probabilities with lighter bars.
    ax.bar(
        x_positions + bar_width / 2,
        theoretical_probs,
        width=bar_width,
        color="steelblue",
        alpha=0.5,
        edgecolor="black",
        linewidth=1.5,
        label="Theoretical",
    )
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_xlabel("Outcome", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")


def _create_basic_widget_controls(
    mu_init: float = 0.6,
    N_init: int = 100,
    seed_init: int = 42,
) -> tuple:
    """
    Create standard mu, N, and seed slider widgets with boxes.

    :param mu_init: Initial mu value
    :param N_init: Initial N value
    :param seed_init: Initial seed value
    :return: Tuple of (mu_slider, mu_box, N_slider, N_box, seed_slider, seed_box)
    """
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
    return mu_slider, mu_box, N_slider, N_box, seed_slider, seed_box


# #############################################################################
# Cell 1: Basic Bernoulli Sampling Code
# #############################################################################


def cell1_basic_bernoulli_sampling(
    *,
    mu: float = 0.6,
    N: int = 10,
    seed: int = 42,
) -> None:
    """
    Demonstrate basic Bernoulli sampling with code display.

    Shows the fundamental process of:
    1. Generating Bernoulli samples
    2. Computing the empirical mean
    3. Comparing with the true mean

    :param mu: True probability of success (0 < mu < 1)
    :param N: Number of samples to draw
    :param seed: Random seed for reproducibility
    """
    # Validate parameters.
    _validate_bernoulli_params(mu, N)
    # Execute the sampling.
    samples = _generate_bernoulli_samples(mu, N, seed)
    empirical_mean = np.mean(samples)
    # Display results.
    print("Parameters:")
    print(f"  True probability (mu): {mu}")
    print(f"  Number of samples (N): {N}")
    print(f"  Random seed: {seed}")
    print()
    print("Generated samples:")
    print(f"  {samples}")
    print()
    print("Statistics:")
    print(f"  Number of successes (1s): {np.sum(samples)}")
    print(f"  Number of failures (0s): {N - np.sum(samples)}")
    print(f"  Empirical mean (nu): {empirical_mean:.4f}")
    print(f"  True mean (mu): {mu:.4f}")
    print(f"  Error |nu - mu|: {abs(empirical_mean - mu):.4f}")


# #############################################################################
# Cell 2: Helper - Plot Bernoulli Sample with PDF
# #############################################################################


def _plot_bernoulli_sample2(
    *,
    mu: float = 0.6,
    N: int = 100,
    seed: int = 42,
) -> None:
    """
    Display N samples from Bernoulli distribution over time and empirical PDF.

    Shows the temporal sequence of samples, empirical PDF, and explanatory
    comments in a single row of 3 plots following interactive widget
    conventions.

    :param mu: True probability of success (0 < mu < 1)
    :param N: Number of samples to draw
    :param seed: Random seed for reproducibility
    """
    # Validate parameters.
    _validate_bernoulli_params(mu, N)
    # Generate N Bernoulli samples.
    samples = _generate_bernoulli_samples(mu, N, seed)
    # Compute statistics.
    n_successes = np.sum(samples)
    n_failures = N - n_successes
    empirical_prob = n_successes / N
    # Create visualization with 3 subplots in a single row.
    _, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 1, 1.5]}
    )
    # Plot 1: Samples over time.
    time_indices = np.arange(N)
    colors = ["red" if s == 0 else "green" for s in samples]
    ax1.scatter(time_indices, samples, c=colors, alpha=0.6, s=30)
    ax1.set_xlabel("Sample Index (Time)", fontsize=12)
    ax1.set_ylabel("Outcome", fontsize=12)
    ax1.set_title(
        "Bernoulli Samples Over Time",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Failure (0)", "Success (1)"])
    ax1.grid(True, alpha=0.3, axis="x")
    # Add horizontal line for expected probability.
    ax1.axhline(
        y=mu,
        color="blue",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"mu={mu:.2f}",
    )
    ax1.legend(fontsize=10, loc="upper right")
    # Plot 2: Empirical PDF with side-by-side bars for comparison.
    _plot_bernoulli_pdf_bars(ax2, samples, mu, N)
    # Plot 3: Comments and explanation.
    ax3.axis("off")
    ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Generate interpretation text.
    text_content = (
        f"Parameters:\n"
        f"  mu = {mu:.4f} (true probability)\n"
        f"  N = {N} (number of samples)\n"
        f"  seed = {seed}\n\n"
        f"Sample Statistics:\n"
        f"  Successes (1): {n_successes}\n"
        f"  Failures (0): {n_failures}\n"
        f"  Empirical prob: {empirical_prob:.4f}\n\n"
        f"Interpretation:\n"
        f"- Each sample is an independent Bernoulli trial with success\n"
        "   probability mu.\n\n"
        f"- The left plot shows samples as they occur over time.\n\n"
        f"- The center plot compares the empirical PDF (bars) with\n"
        f"  the theoretical probabilities (blue line)."
    )
    mtumsuti.add_fitted_text_box(ax3, text_content)
    # Use subplots_adjust for consistent spacing.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.30
    )
    plt.show()


# #############################################################################
# Cell 2: Samples Over Time and Empirical PDF
# #############################################################################


def cell2_samples_over_time_and_pdf() -> None:
    """
    Create interactive Bernoulli sampling visualization with PDF comparison.

    Sets up an interactive widget that allows exploration of Bernoulli sampling
    behavior. The widget displays three synchronized plots showing:
    (1) temporal sequence of sampled outcomes,
    (2) comparison of empirical vs theoretical probability distributions
    (3) detailed statistics and interpretation.
    """
    mu_init = 0.6
    N_init = 100
    seed_init = 42
    # Create widgets.
    mu_slider, mu_box, N_slider, N_box, seed_slider, seed_box = (
        _create_basic_widget_controls(mu_init, N_init, seed_init)
    )
    # Create output widget.
    output = ipywidgets.Output()

    def update_plot(change=None):
        """
        Update plot when slider values change.
        """
        with output:
            output.clear_output(wait=True)
            _plot_bernoulli_sample2(
                mu=mu_slider.value, N=N_slider.value, seed=seed_slider.value
            )

    # Observe slider changes.
    mu_slider.observe(update_plot, names="value")
    N_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    # Display widgets and initial plot.
    display(ipywidgets.VBox([mu_box, N_box, seed_box, output]))
    update_plot()


# #############################################################################
# Cell 3: Helper - Plot Bernoulli Sample with Statistics
# #############################################################################


def _plot_bernoulli_sample3(
    *,
    mu: float = 0.6,
    N: int = 100,
    seed: int = 42,
) -> None:
    """
    Display PDF, empirical mean, and statistics of Bernoulli samples.

    Shows the probability distribution of N samples, the empirical mean nu,
    and compares it with the true mean and variance in a single row of 3-4
    plots following interactive widget conventions.

    :param mu: True probability of success (0 < mu < 1)
    :param N: Number of samples to draw
    :param seed: Random seed for reproducibility
    """
    # Validate parameters.
    _validate_bernoulli_params(mu, N)
    # Generate N Bernoulli samples.
    samples = _generate_bernoulli_samples(mu, N, seed)
    # Compute the sample mean nu.
    nu = np.mean(samples)
    # Compute theoretical mean and variance of Bernoulli distribution.
    theoretical_mean = mu
    theoretical_variance = mu * (1 - mu)
    # Compute sample variance.
    sample_variance = np.var(samples, ddof=1)
    # Create visualization with 3 subplots in a single row.
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 1, 1.2]}
    )
    # Plot 1: PDF of samples.
    _plot_bernoulli_pdf_bars(ax1, samples, mu, N, "PDF of Samples")
    # Plot 2: Comparison of empirical vs theoretical statistics.
    metrics = ["Mean", "Variance"]
    empirical = [nu, sample_variance]
    theoretical = [theoretical_mean, theoretical_variance]
    x_pos = np.arange(len(metrics))
    width = 0.35
    ax2.bar(
        x_pos - width / 2,
        empirical,
        width,
        label="Empirical",
        color="darkgreen",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.5,
    )
    ax2.bar(
        x_pos + width / 2,
        theoretical,
        width,
        label="Theoretical",
        color="steelblue",
        alpha=0.5,
        edgecolor="black",
        linewidth=1.5,
    )
    ax2.set_ylabel("Value", fontsize=12)
    ax2.set_xlabel("Statistic", fontsize=12)
    ax2.set_title("Statistics Comparison", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics)
    ax2.set_ylim([0, 1.0])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    # Add value labels on bars.
    for i, (emp, theo) in enumerate(zip(empirical, theoretical)):
        ax2.text(
            i - width / 2,
            emp + 0.02,
            f"{emp:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax2.text(
            i + width / 2,
            theo + 0.02,
            f"{theo:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    # Plot 3: Comments and explanation.
    ax3.axis("off")
    ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Generate interpretation text.
    error = abs(nu - mu)
    text_content = (
        f"Parameters:\n"
        f"  mu = {mu:.4f} (true probability)\n"
        f"  N = {N} (number of samples)\n"
        f"  seed = {seed}\n\n"
        f"Empirical Statistics:\n"
        f"  nu (sample mean) = {nu:.4f}\n"
        f"  Sample variance = {sample_variance:.4f}\n\n"
        f"Theoretical Statistics:\n"
        f"  Mean = {theoretical_mean:.4f}\n"
        f"  Variance = {theoretical_variance:.4f} (= mu * (1-mu))\n\n"
        f"Error:\n"
        f"  |nu - mu| = {error:.4f}\n\n"
        f"Interpretation:\n"
        f"- The empirical mean nu estimates the true mean mu.\n\n"
        f"- Change seed to see new realizations with different\n"
        f"  empirical values."
    )
    mtumsuti.add_fitted_text_box(ax3, text_content)
    # Use subplots_adjust for consistent spacing.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.30
    )
    plt.show()


# #############################################################################
# Cell 3: PDF, Empirical Mean, and Statistics
# #############################################################################


def cell3_pdf_empirical_mean_stats(
    *,
    mu_init: float = 0.6,
    N_init: int = 100,
    seed_init: int = 42,
) -> None:
    """
    Sets up complete interactive widget with sliders for mu, N, and seed
    parameters.

    Connects sliders to _plot_bernoulli_sample3() for interactive visualization.

    :param mu_init: Initial value for mu (probability of success)
    :param N_init: Initial value for N (number of samples)
    :param seed_init: Initial value for seed
    """
    # Create widgets.
    mu_slider, mu_box, N_slider, N_box, seed_slider, seed_box = (
        _create_basic_widget_controls(mu_init, N_init, seed_init)
    )
    # Create output widget.
    output = ipywidgets.Output()

    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)
            _plot_bernoulli_sample3(
                mu=mu_slider.value, N=N_slider.value, seed=seed_slider.value
            )

    # Observe slider changes.
    mu_slider.observe(update_plot, names="value")
    N_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    # Display widgets and initial plot.
    display(ipywidgets.VBox([mu_box, N_box, seed_box, output]))
    update_plot()


# #############################################################################
# Cell 4: Helper - Plot Distribution of Empirical Mean
# #############################################################################


def _plot_bernoulli_sample4(
    *,
    mu: float = 0.6,
    N: int = 100,
    n_samples: int = 1000,
    seed: int = 42,
) -> None:
    """
    Display distribution of empirical mean nu from repeated sampling.

    Shows the empirical distribution of the sample mean nu when n_samples
    independent samples are generated, and compares it with the expected
    distribution predicted by the Central Limit Theorem, in a single row
    of plots following interactive widget conventions.

    :param mu: True probability of success (0 < mu < 1)
    :param N: Number of samples in each trial
    :param n_samples: Number of trials to generate empirical distribution
    :param seed: Random seed for reproducibility
    """
    # Validate parameters.
    _validate_bernoulli_params(mu, N, n_samples)
    # Set random seed for reproducibility.
    np.random.seed(seed)
    # Generate empirical distribution by repeated sampling.
    empirical_nus = []
    for _ in range(n_samples):
        trial_samples = np.random.binomial(1, mu, size=N)
        trial_nu = np.mean(trial_samples)
        empirical_nus.append(trial_nu)
    empirical_nus = np.array(empirical_nus)
    # Compute expected distribution parameters using Central Limit Theorem.
    # For large N, the sample mean follows a normal distribution:
    # nu ~ N(mu, sqrt(mu * (1-mu) / N))
    expected_mean = mu
    expected_std = np.sqrt(mu * (1 - mu) / N)
    # Compute empirical statistics.
    empirical_mean = np.mean(empirical_nus)
    empirical_std = np.std(empirical_nus)
    # Create visualization with 2 subplots in a single row.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, 5), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    # Plot 1: Distribution of empirical mean nu.
    # Plot empirical data with light blue color and solid bars.
    ax1.hist(
        empirical_nus,
        bins=30,
        density=True,
        alpha=0.85,
        color="lightblue",
        edgecolor="black",
        linewidth=1.5,
        label=f"Empirical (n={n_samples})",
    )
    # Plot expected normal distribution with lighter, transparent, dotted line.
    x_range = np.linspace(
        expected_mean - 4 * expected_std,
        expected_mean + 4 * expected_std,
        200,
    )
    y_expected = scipy.stats.norm.pdf(x_range, expected_mean, expected_std)
    ax1.plot(
        x_range,
        y_expected,
        "--",
        color="coral",
        linewidth=2.5,
        label="Theoretical (CLT)",
        alpha=0.5,
    )
    # Mark the true mean.
    ax1.axvline(
        x=mu,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"True mu={mu:.4f}",
    )
    ax1.set_xlabel("Sample Mean (nu)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title(
        "Distribution of Empirical Mean nu",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlim([0, 1.0])
    ax1.set_ylim([0, 25])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Plot 2: Comments and explanation.
    ax2.axis("off")
    ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Generate interpretation text.
    if N < 30:
        sample_size_note = (
            "Sample size is small. CLT approximation may not be accurate."
        )
    elif N < 100:
        sample_size_note = (
            "Sample size is moderate. CLT approximation is reasonable."
        )
    else:
        sample_size_note = (
            "Sample size is large. CLT approximation is very accurate."
        )
    text_content = (
        f"Parameters:\n"
        f"  mu = {mu:.4f} (true probability)\n"
        f"  N = {N} (samples per trial)\n"
        f"  n_samples = {n_samples} (number of trials)\n"
        f"  seed = {seed}\n\n"
        f"Empirical Statistics:\n"
        f"  Mean = {empirical_mean:.4f}\n"
        f"  Std Dev = {empirical_std:.4f}\n\n"
        f"Expected by CLT:\n"
        f"  Mean = {expected_mean:.4f}\n"
        f"  Std Dev = {expected_std:.4f} (= sqrt(mu(1-mu)/N))\n\n"
        f"Interpretation:\n"
        f"- {sample_size_note}\n\n"
        f"- By Central Limit Theorem, nu ~ N(mu, sqrt(mu(1-mu)/N)).\n\n"
        f"- As N increases, the distribution becomes more concentrated\n"
        f"  around mu."
    )
    mtumsuti.add_fitted_text_box(ax2, text_content)
    # Use subplots_adjust for consistent spacing.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.25
    )
    plt.show()


# #############################################################################
# Cell 4: Distribution of Empirical Mean
# #############################################################################


def cell4_distribution_empirical_mean() -> None:
    """
    Create interactive widget for Cell 4 (Distribution of Empirical Mean).

    Sets up complete interactive widget with sliders for mu, N, n_samples,
    and seed parameters.

    Connects sliders to _plot_bernoulli_sample4() for interactive visualization.
    """
    mu_init = 0.6
    N_init = 100
    n_samples_init = 1000
    seed_init = 42
    # Create widgets.
    mu_slider, mu_box, N_slider, N_box, seed_slider, seed_box = (
        _create_basic_widget_controls(mu_init, N_init, seed_init)
    )
    # Update N slider description for this specific use case.
    N_box.children[0].description = "samples per trial"
    # Create logarithmic n_samples slider.
    # The slider operates in log10 space: 10^slider_value = n_samples
    # log10(100) = 2.0, log10(10000) = 4.0
    log_n_samples_init = np.log10(n_samples_init)
    n_samples_slider, n_samples_box = mtumsuti.build_widget_control(
        name="log10(n_samples)",
        description="number of trials (log scale)",
        min_val=2.0,
        max_val=4.0,
        step=0.1,
        initial_value=log_n_samples_init,
        is_float=True,
    )
    # Create output widget.
    output = ipywidgets.Output()

    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)
            # Convert log slider value to actual n_samples.
            n_samples = int(10**n_samples_slider.value)
            _plot_bernoulli_sample4(
                mu=mu_slider.value,
                N=N_slider.value,
                n_samples=n_samples,
                seed=seed_slider.value,
            )

    # Observe slider changes.
    mu_slider.observe(update_plot, names="value")
    N_slider.observe(update_plot, names="value")
    n_samples_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    # Display widgets and initial plot.
    display(ipywidgets.VBox([mu_box, N_box, n_samples_box, seed_box, output]))
    update_plot()


# #############################################################################
# Cell 5: Helper - Plot Hoeffding Inequality Demo
# #############################################################################


def _generate_samples_from_distribution(
    distribution: str,
    N: int,
    mu: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate N samples from specified distribution in [0, 1].

    :param distribution: Distribution type (bernoulli, uniform, binomial,
        truncated_gaussian, truncated_exponential)
    :param N: Number of samples to generate
    :param mu: Parameter for the distribution (interpretation depends on type)
    :param seed: Random seed for reproducibility
    :return: Array of N samples in [0, 1]
    """
    np.random.seed(seed)
    if distribution == "bernoulli":
        # Bernoulli(mu): samples are 0 or 1.
        return np.random.binomial(1, mu, size=N)
    elif distribution == "uniform":
        # Uniform[0, 1]: mu parameter is ignored.
        return np.random.uniform(0, 1, size=N)
    elif distribution == "binomial":
        # Binomial(10, mu) scaled to [0, 1]: samples are k/10 where k~Binomial(10, mu).
        n_trials = 10
        return np.random.binomial(n_trials, mu, size=N) / n_trials
    elif distribution == "truncated_gaussian":
        # Truncated Gaussian: mean=mu, std=0.2, truncated to [0, 1].
        samples = np.random.normal(mu, 0.2, size=N)
        return np.clip(samples, 0, 1)
    elif distribution == "truncated_exponential":
        # Truncated Exponential: rate parameter chosen to have mean near mu.
        # Exponential(lambda) has mean 1/lambda.
        # We want mean around mu, so lambda = 1/mu.
        # Then truncate to [0, 1].
        if mu <= 0 or mu > 1:
            mu = 0.5
        lambda_param = 1 / mu
        samples = np.random.exponential(1 / lambda_param, size=N)
        return np.clip(samples, 0, 1)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def _plot_hoeffding_inequality_demo(
    *,
    distribution: str = "bernoulli",
    mu: float = 0.6,
    N: int = 100,
    epsilon: float = 0.1,
    n_trials: int = 10000,
    seed: int = 42,
) -> None:
    """
    Visualize Hoeffding inequality with different distributions and tail areas.

    Shows:
    1. Distribution of sample mean with shaded tail regions representing
       deviations greater than epsilon
    2. Comparison of theoretical Hoeffding bound vs empirical probability
    3. Comments explaining the results

    :param distribution: Distribution type (bernoulli, uniform, binomial,
        truncated_gaussian, truncated_exponential)
    :param mu: Distribution parameter (interpretation depends on type)
    :param N: Number of samples per trial
    :param epsilon: Deviation threshold for Hoeffding bound
    :param n_trials: Number of trials for empirical probability estimation
    :param seed: Random seed for reproducibility
    """
    # Validate parameters.
    hdbg.dassert_lte(1, N, "N must be at least 1:", N)
    hdbg.dassert_lte(0, epsilon, "epsilon must be positive:", epsilon)
    hdbg.dassert_lte(epsilon, 1, "epsilon must be at most 1:", epsilon)
    # Set random seed for reproducibility.
    np.random.seed(seed)
    # Generate empirical distribution by repeated sampling.
    empirical_nus = []
    empirical_means_per_trial = []
    for trial_idx in range(n_trials):
        trial_samples = _generate_samples_from_distribution(
            distribution, N, mu, seed + trial_idx
        )
        trial_nu = np.mean(trial_samples)
        empirical_nus.append(trial_nu)
        empirical_means_per_trial.append(trial_samples)
    empirical_nus = np.array(empirical_nus)
    # Compute true mean of the distribution.
    if distribution == "bernoulli":
        true_mean = mu
    elif distribution == "uniform":
        true_mean = 0.5
    elif distribution == "binomial":
        true_mean = mu
    elif distribution == "truncated_gaussian":
        # Approximate mean (exact would require computing truncated normal mean).
        true_mean = mu
    elif distribution == "truncated_exponential":
        # Approximate mean (exact would require computing truncated exponential mean).
        true_mean = mu
    else:
        true_mean = 0.5
    # Compute theoretical Hoeffding bound (capped at 1.0).
    # P(|nu - true_mean| >= epsilon) <= 2 * exp(-2 * N * epsilon^2)
    hoeffding_bound = min(1.0, 2 * np.exp(-2 * N * epsilon**2))
    # Compute empirical probability of deviation >= epsilon.
    empirical_prob = np.mean(np.abs(empirical_nus - true_mean) >= epsilon)
    # Create histogram of empirical sample means for visualization.
    hist, bin_edges = np.histogram(empirical_nus, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    # Convert density to probability (multiply by bin width).
    sample_probs = hist * bin_width
    # Identify tail regions: |bin_center - true_mean| >= epsilon.
    tail_mask = np.abs(bin_centers - true_mean) >= epsilon
    # Create visualization with 4 subplots.
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(
        1, 4, figsize=(20, 5), gridspec_kw={"width_ratios": [1, 1.5, 0.8, 1]}
    )
    # Plot 0: Show the underlying distribution PDF/PMF.
    x_values = np.linspace(0, 1, 200)
    if distribution == "bernoulli":
        # Show PMF as bars for discrete distribution.
        outcomes = [0, 1]
        probs = [1 - mu, mu]
        ax0.bar(
            outcomes,
            probs,
            width=0.1,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax0.set_xlabel("X", fontsize=11)
        ax0.set_ylabel("P(X)", fontsize=11)
        ax0.set_xlim([-0.2, 1.2])
        ax0.set_ylim([0, 1.0])
    elif distribution == "uniform":
        # Uniform PDF.
        pdf_values = np.ones_like(x_values)
        ax0.fill_between(
            x_values,
            0,
            pdf_values,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax0.set_xlabel("X", fontsize=11)
        ax0.set_ylabel("PDF(X)", fontsize=11)
        ax0.set_xlim([0, 1.0])
        ax0.set_ylim([0, 1.5])
    elif distribution == "binomial":
        # Binomial(10, mu) scaled to [0, 1].
        n_trials = 10
        k_values = np.arange(0, n_trials + 1)
        probs = scipy.stats.binom.pmf(k_values, n_trials, mu)
        x_scaled = k_values / n_trials
        ax0.bar(
            x_scaled,
            probs,
            width=0.08,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax0.set_xlabel("X", fontsize=11)
        ax0.set_ylabel("P(X)", fontsize=11)
        ax0.set_xlim([0, 1.0])
    elif distribution == "truncated_gaussian":
        # Truncated Gaussian PDF.
        pdf_values = scipy.stats.norm.pdf(x_values, mu, 0.2)
        # Normalize to account for truncation.
        cdf_0 = scipy.stats.norm.cdf(0, mu, 0.2)
        cdf_1 = scipy.stats.norm.cdf(1, mu, 0.2)
        normalization = cdf_1 - cdf_0
        pdf_values = pdf_values / normalization
        ax0.fill_between(
            x_values,
            0,
            pdf_values,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax0.set_xlabel("X", fontsize=11)
        ax0.set_ylabel("PDF(X)", fontsize=11)
        ax0.set_xlim([0, 1.0])
    elif distribution == "truncated_exponential":
        # Truncated Exponential PDF.
        if mu <= 0 or mu > 1:
            mu_eff = 0.5
        else:
            mu_eff = mu
        lambda_param = 1 / mu_eff
        pdf_values = lambda_param * np.exp(-lambda_param * x_values)
        # Normalize to account for truncation.
        cdf_0 = 1 - np.exp(-lambda_param * 0)
        cdf_1 = 1 - np.exp(-lambda_param * 1)
        normalization = cdf_1 - cdf_0
        pdf_values = pdf_values / normalization
        ax0.fill_between(
            x_values,
            0,
            pdf_values,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax0.set_xlabel("X", fontsize=11)
        ax0.set_ylabel("PDF(X)", fontsize=11)
        ax0.set_xlim([0, 1.0])
    dist_name = distribution.replace("_", " ").title()
    ax0.set_title(
        f"{dist_name}\nDistribution",
        fontsize=13,
        fontweight="bold",
    )
    ax0.grid(True, alpha=0.3, axis="y")
    # Plot 1: Distribution of sample mean with shaded tail areas.
    ax1.bar(
        bin_centers[~tail_mask],
        sample_probs[~tail_mask],
        width=bin_width,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label="Within epsilon",
    )
    ax1.bar(
        bin_centers[tail_mask],
        sample_probs[tail_mask],
        width=bin_width,
        color="red",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label=f"|nu - mean| >= {epsilon}",
    )
    # Mark true mean and bounds.
    ax1.axvline(
        x=true_mean,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"True mean={true_mean:.2f}",
    )
    ax1.axvline(
        x=true_mean - epsilon,
        color="red",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
    )
    ax1.axvline(
        x=true_mean + epsilon,
        color="red",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
    )
    ax1.set_xlabel("Sample Mean (nu)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.set_title(
        "Distribution of\nSample Mean",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_xlim([0, 1.0])
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    # Plot 2: Comparison of bounds.
    labels = ["Hoeffding\nBound", "Empirical\nProbability"]
    values = [hoeffding_bound, empirical_prob]
    colors = ["coral", "steelblue"]
    bars = ax2.bar(
        labels,
        values,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=1.5,
    )
    # Add value labels on bars.
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{val:.6f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax2.set_ylabel("Probability", fontsize=11)
    ax2.set_title(
        "Bound vs\nEmpirical",
        fontsize=13,
        fontweight="bold",
    )
    ax2.set_ylim([0, min(1.0, max(values) * 1.3)])
    ax2.grid(True, alpha=0.3, axis="y")
    # Plot 3: Comments and explanation.
    ax3.axis("off")
    ax3.set_title("Comments", fontsize=13, fontweight="bold", pad=20)
    # Check if bound is tight.
    bound_ratio = (
        hoeffding_bound / empirical_prob if empirical_prob > 0 else float("inf")
    )
    if bound_ratio < 2:
        tightness_note = "The Hoeffding bound is quite tight."
    elif bound_ratio < 10:
        tightness_note = "The Hoeffding bound is reasonably tight."
    else:
        tightness_note = "The Hoeffding bound is conservative (loose)."
    dist_name = distribution.replace("_", " ").title()
    text_content = (
        f"Parameters:\n"
        f"  distribution = {dist_name}\n"
        f"  mu = {mu:.4f} (dist. parameter)\n"
        f"  true mean = {true_mean:.4f}\n"
        f"  N = {N} (samples per trial)\n"
        f"  epsilon = {epsilon:.4f}\n"
        f"  n_trials = {n_trials}\n"
        f"  seed = {seed}\n\n"
        f"Hoeffding Inequality:\n"
        f"  P(|nu - mean| >= epsilon) <= 2*exp(-2*N*epsilon^2)\n"
        f"  Bound = {hoeffding_bound:.6f} (capped at 1.0)\n\n"
        f"Empirical Result:\n"
        f"  P(|nu - mean| >= {epsilon}) = {empirical_prob:.6f}\n"
        f"  (from {n_trials} trials)\n\n"
        f"Interpretation:\n"
        f"- {tightness_note}\n\n"
        f"- Hoeffding bound applies to any distribution in [0, 1].\n\n"
        f"- Red bars show tail areas where |nu - mean| >= epsilon.\n\n"
        f"- As N increases, the bound becomes tighter."
    )
    mtumsuti.add_fitted_text_box(ax3, text_content)
    # Use subplots_adjust for consistent spacing.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.30
    )
    plt.show()


# #############################################################################
# Cell 5: Interactive Hoeffding Inequality Demonstration
# #############################################################################


def cell5_hoeffding_inequality_demo() -> None:
    """
    Create interactive widget demonstrating the Hoeffding inequality.

    Shows:
    - Distribution of sample mean with shaded tail areas
    - Comparison of theoretical Hoeffding bound vs empirical probability
    - Interactive controls for distribution type, parameters, N, epsilon, seed

    The Hoeffding inequality states:
    P(|nu - mean| >= epsilon) <= 2 * exp(-2 * N * epsilon^2)

    where nu is the sample mean of N samples from any distribution in [0, 1].

    Supports multiple distributions:
    - Bernoulli: Binary outcomes (0 or 1)
    - Uniform: Continuous uniform in [0, 1]
    - Binomial: Discrete binomial scaled to [0, 1]
    - Truncated Gaussian: Normal distribution truncated to [0, 1]
    - Truncated Exponential: Exponential distribution truncated to [0, 1]
    """
    mu_init = 0.6
    N_init = 100
    epsilon_init = 0.1
    seed_init = 42
    # Create distribution selector.
    distribution_dropdown = ipywidgets.Dropdown(
        options=[
            ("Bernoulli", "bernoulli"),
            ("Uniform [0, 1]", "uniform"),
            ("Binomial (scaled)", "binomial"),
            ("Truncated Gaussian", "truncated_gaussian"),
            ("Truncated Exponential", "truncated_exponential"),
        ],
        value="bernoulli",
        description="Distribution:",
        style={"description_width": "150px"},
        layout=ipywidgets.Layout(width="500px"),
    )
    # Create widgets.
    mu_slider, mu_box, N_slider, N_box, seed_slider, seed_box = (
        _create_basic_widget_controls(mu_init, N_init, seed_init)
    )
    # Update descriptions.
    mu_box.children[0].description = "mu = distribution parameter"
    N_box.children[0].description = "samples per trial"
    # Update N slider range for better exploration.
    N_slider.min = 10
    N_slider.max = 500
    epsilon_slider, epsilon_box = mtumsuti.build_widget_control(
        name="epsilon",
        description="deviation threshold",
        min_val=0.01,
        max_val=0.5,
        step=0.01,
        initial_value=epsilon_init,
        is_float=True,
    )
    # Create output widget.
    output = ipywidgets.Output()

    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)
            _plot_hoeffding_inequality_demo(
                distribution=distribution_dropdown.value,
                mu=mu_slider.value,
                N=N_slider.value,
                epsilon=epsilon_slider.value,
                seed=seed_slider.value,
            )

    # Observe slider changes.
    distribution_dropdown.observe(update_plot, names="value")
    mu_slider.observe(update_plot, names="value")
    N_slider.observe(update_plot, names="value")
    epsilon_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    # Display widgets and initial plot.
    display(
        ipywidgets.VBox(
            [distribution_dropdown, mu_box, N_box, epsilon_box, seed_box, output]
        )
    )
    update_plot()


# #############################################################################
# Cell 6: Helper - Plot Empirical vs Bound
# #############################################################################


def _plot_hoeffding_inequality_demo2(
    *,
    distribution: str = "bernoulli",
    mu: float = 0.6,
    scan_variable: str = "N",
    fixed_N: int = 100,
    fixed_epsilon: float = 0.1,
    n_trials: int = 1000,
    seed: int = 42,
) -> None:
    """
    Plot empirical probability vs Hoeffding bound as function of N or epsilon.

    :param distribution: Distribution type
    :param mu: Distribution parameter
    :param scan_variable: Which variable to scan ("N" or "epsilon")
    :param fixed_N: Fixed N value when scanning epsilon
    :param fixed_epsilon: Fixed epsilon value when scanning N
    :param n_trials: Number of trials for empirical probability estimation
    :param seed: Random seed for reproducibility
    """
    # Determine scan range based on scan variable.
    if scan_variable == "N":
        scan_values = np.arange(10, 501, 10)
        fixed_value = fixed_epsilon
        x_label = "N (number of samples)"
        title = f"Hoeffding Bound vs Empirical (epsilon={fixed_epsilon:.3f})"
    else:  # scan_variable == "epsilon"
        scan_values = np.linspace(0.01, 0.5, 50)
        fixed_value = fixed_N
        x_label = "Epsilon (deviation threshold)"
        title = f"Hoeffding Bound vs Empirical (N={fixed_N})"
    # Compute theoretical bounds and empirical probabilities.
    hoeffding_bounds = []
    empirical_probs = []
    # Compute true mean of the distribution.
    if distribution == "bernoulli":
        true_mean = mu
    elif distribution == "uniform":
        true_mean = 0.5
    elif distribution == "binomial":
        true_mean = mu
    elif distribution == "truncated_gaussian":
        true_mean = mu
    elif distribution == "truncated_exponential":
        true_mean = mu
    else:
        true_mean = 0.5
    for scan_val in scan_values:
        if scan_variable == "N":
            N_current = int(scan_val)
            epsilon_current = fixed_epsilon
        else:
            N_current = fixed_N
            epsilon_current = scan_val
        # Compute Hoeffding bound (capped at 1.0).
        bound = min(1.0, 2 * np.exp(-2 * N_current * epsilon_current**2))
        hoeffding_bounds.append(bound)
        # Compute empirical probability.
        np.random.seed(seed)
        empirical_nus = []
        for trial_idx in range(n_trials):
            trial_samples = _generate_samples_from_distribution(
                distribution, N_current, mu, seed + trial_idx
            )
            trial_nu = np.mean(trial_samples)
            empirical_nus.append(trial_nu)
        empirical_nus = np.array(empirical_nus)
        empirical_prob = np.mean(
            np.abs(empirical_nus - true_mean) >= epsilon_current
        )
        empirical_probs.append(empirical_prob)
    # Create visualization with 2 subplots.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, 5), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    # Plot 1: Bound and empirical probability vs scan variable.
    ax1.plot(
        scan_values,
        hoeffding_bounds,
        linewidth=2.5,
        color="coral",
        label="Hoeffding Bound (theoretical)",
        marker="o",
        markersize=4,
        alpha=0.8,
    )
    ax1.plot(
        scan_values,
        empirical_probs,
        linewidth=2.5,
        color="steelblue",
        label="Empirical Probability",
        marker="s",
        markersize=4,
        alpha=0.8,
    )
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel("Probability", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="best")
    ax1.grid(True, alpha=0.3)
    # Set y-axis to log scale if values span multiple orders of magnitude.
    if np.max(hoeffding_bounds) / np.min(hoeffding_bounds) > 100:
        ax1.set_yscale("log")
        ax1.set_ylabel("Probability (log scale)", fontsize=12)
    else:
        ax1.set_ylim(
            [0, min(1.0, max(max(hoeffding_bounds), max(empirical_probs)) * 1.1)]
        )
    # Plot 2: Comments and explanation.
    ax2.axis("off")
    ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    dist_name = distribution.replace("_", " ").title()
    if scan_variable == "N":
        scan_note = (
            "- As N increases, both bound and empirical probability\n"
            "  decrease exponentially.\n\n"
            "- The exponential decay shows why we need relatively\n"
            "  few samples for good concentration.\n\n"
            "- Empirical probability is always below the bound."
        )
    else:
        scan_note = (
            "- As epsilon increases, both bound and empirical probability\n"
            "  decrease.\n\n"
            "- Larger epsilon means more tolerance for deviation,\n"
            "  so probability of exceeding it decreases.\n\n"
            "- Empirical probability is always below the bound."
        )
    text_content = (
        f"Parameters:\n"
        f"  distribution = {dist_name}\n"
        f"  mu = {mu:.4f}\n"
        f"  true mean = {true_mean:.4f}\n"
        f"  n_trials = {n_trials}\n"
        f"  seed = {seed}\n\n"
        f"Scanning:\n"
        f"  variable = {scan_variable}\n"
        f"  fixed value = {fixed_value}\n\n"
        f"Hoeffding Inequality:\n"
        f"  P(|nu - mean| >= epsilon) <= 2*exp(-2*N*epsilon^2)\n\n"
        f"Observations:\n"
        f"{scan_note}"
    )
    mtumsuti.add_fitted_text_box(ax2, text_content)
    # Adjust layout.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.25
    )
    plt.show()


# #############################################################################
# Cell 6: Empirical Probability vs Hoeffding Bound
# #############################################################################


def cell6_empirical_vs_bound() -> None:
    """
    Create interactive widget showing empirical probability vs Hoeffding bound.

    This visualization shows how the theoretical Hoeffding bound and empirical
    probability change as we vary either N (number of samples) or epsilon
    (deviation threshold) while holding the other fixed.

    Features:
    - Select distribution type (Bernoulli, Uniform, Binomial, etc.)
    - Choose distribution parameter mu
    - Select which variable to scan (N or epsilon)
    - Set fixed value for the non-scanned variable
    - Plots both theoretical bound and empirical probability on same axes

    The plot demonstrates:
    - The bound always holds (empirical <= bound)
    - Exponential decay with respect to both N and epsilon
    - How different distributions behave under the same bound
    """
    mu_init = 0.6
    fixed_N_init = 100
    fixed_epsilon_init = 0.1
    seed_init = 42
    # Create distribution selector.
    distribution_dropdown = ipywidgets.Dropdown(
        options=[
            ("Bernoulli", "bernoulli"),
            ("Uniform [0, 1]", "uniform"),
            ("Binomial (scaled)", "binomial"),
            ("Truncated Gaussian", "truncated_gaussian"),
            ("Truncated Exponential", "truncated_exponential"),
        ],
        value="bernoulli",
        description="Distribution:",
        style={"description_width": "150px"},
        layout=ipywidgets.Layout(width="500px"),
    )
    # Create scan variable selector.
    scan_dropdown = ipywidgets.Dropdown(
        options=[
            ("Scan N (fix epsilon)", "N"),
            ("Scan epsilon (fix N)", "epsilon"),
        ],
        value="N",
        description="Scan variable:",
        style={"description_width": "150px"},
        layout=ipywidgets.Layout(width="500px"),
    )
    # Create widgets.
    mu_slider, mu_box = mtumsuti.build_widget_control(
        name="mu",
        description="distribution parameter",
        min_val=0.1,
        max_val=0.9,
        step=0.05,
        initial_value=mu_init,
        is_float=True,
    )
    fixed_N_slider, fixed_N_box = mtumsuti.build_widget_control(
        name="fixed_N",
        description="N (when scanning epsilon)",
        min_val=10,
        max_val=500,
        step=10,
        initial_value=fixed_N_init,
        is_float=False,
    )
    fixed_epsilon_slider, fixed_epsilon_box = mtumsuti.build_widget_control(
        name="fixed_epsilon",
        description="epsilon (when scanning N)",
        min_val=0.01,
        max_val=0.5,
        step=0.01,
        initial_value=fixed_epsilon_init,
        is_float=True,
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
    output = ipywidgets.Output()

    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)
            _plot_hoeffding_inequality_demo2(
                distribution=distribution_dropdown.value,
                mu=mu_slider.value,
                scan_variable=scan_dropdown.value,
                fixed_N=fixed_N_slider.value,
                fixed_epsilon=fixed_epsilon_slider.value,
                seed=seed_slider.value,
            )

    def update_visibility(change=None):
        """Show/hide fixed value widgets based on scan variable."""
        scan_var = scan_dropdown.value
        if scan_var == "N":
            fixed_epsilon_box.layout.visibility = "visible"
            fixed_N_box.layout.visibility = "hidden"
        else:
            fixed_epsilon_box.layout.visibility = "hidden"
            fixed_N_box.layout.visibility = "visible"
        update_plot()

    # Observe changes.
    distribution_dropdown.observe(update_plot, names="value")
    mu_slider.observe(update_plot, names="value")
    scan_dropdown.observe(update_visibility, names="value")
    fixed_N_slider.observe(update_plot, names="value")
    fixed_epsilon_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    # Initial visibility setup.
    fixed_N_box.layout.visibility = "hidden"
    # Display widgets and initial plot.
    display(
        ipywidgets.VBox(
            [
                distribution_dropdown,
                scan_dropdown,
                mu_box,
                fixed_N_box,
                fixed_epsilon_box,
                seed_box,
                output,
            ]
        )
    )
    update_plot()


# #############################################################################
# Cell 7: Helper - Plot Hoeffding Bound Surface
# #############################################################################


def _plot_hoeffding_bound_surface(
    *,
    N_max: int = 500,
    epsilon_max: float = 0.5,
    fixed_N: int = None,
    fixed_epsilon: float = None,
    plot_type: str = "heatmap",
) -> None:
    """
    Plot Hoeffding bound as a function of N and epsilon.

    Shows how the bound 2*exp(-2*N*epsilon^2) varies with:
    - N (number of samples)
    - epsilon (deviation threshold)

    :param N_max: Maximum N value for the plot
    :param epsilon_max: Maximum epsilon value for the plot
    :param fixed_N: If provided, plot bound vs epsilon for this fixed N
    :param fixed_epsilon: If provided, plot bound vs N for this fixed epsilon
    :param plot_type: Type of plot - "heatmap", "contour", or "3d"
    """
    # Create grid for N and epsilon.
    N_values = np.linspace(10, N_max, 100)
    epsilon_values = np.linspace(0.01, epsilon_max, 100)
    N_grid, epsilon_grid = np.meshgrid(N_values, epsilon_values)
    # Compute Hoeffding bound for each (N, epsilon) pair.
    bound_grid = 2 * np.exp(-2 * N_grid * epsilon_grid**2)
    # Create visualization based on plot type and fixed parameters.
    if fixed_N is not None:
        # Plot bound vs epsilon for fixed N.
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(18, 5), gridspec_kw={"width_ratios": [1.5, 1]}
        )
        epsilon_plot = np.linspace(0.01, epsilon_max, 200)
        bound_plot = 2 * np.exp(-2 * fixed_N * epsilon_plot**2)
        ax1.plot(epsilon_plot, bound_plot, linewidth=2.5, color="steelblue")
        ax1.fill_between(epsilon_plot, 0, bound_plot, alpha=0.3)
        ax1.set_xlabel("Epsilon (deviation threshold)", fontsize=12)
        ax1.set_ylabel("Hoeffding Bound", fontsize=12)
        ax1.set_title(
            f"Hoeffding Bound vs Epsilon (N={fixed_N})",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylim([0, 1.0])
        ax1.grid(True, alpha=0.3)
        # Add reference lines for common probability thresholds.
        thresholds = [0.05, 0.1, 0.2]
        for thresh in thresholds:
            ax1.axhline(
                y=thresh,
                color="red",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )
            ax1.text(
                epsilon_max * 0.95,
                thresh + 0.01,
                f"p={thresh}",
                ha="right",
                fontsize=9,
                color="red",
            )
        # Comments panel.
        ax2.axis("off")
        ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
        text_content = (
            f"Hoeffding Bound:\n"
            f"  P(|nu - mu| >= epsilon) <= 2*exp(-2*N*epsilon^2)\n\n"
            f"Fixed Parameters:\n"
            f"  N = {fixed_N} (samples)\n"
            f"  N_max = {N_max}\n"
            f"  epsilon_max = {epsilon_max}\n\n"
            f"Observations:\n"
            f"- The bound decreases exponentially as epsilon increases.\n\n"
            f"- Smaller epsilon (stricter requirement) leads to larger\n"
            f"  bound (lower confidence).\n\n"
            f"- For very small epsilon, the bound approaches 2\n"
            f"  (meaningless bound).\n\n"
            f"- For large epsilon, the bound approaches 0\n"
            f"  (very confident)."
        )
        mtumsuti.add_fitted_text_box(ax2, text_content)
    elif fixed_epsilon is not None:
        # Plot bound vs N for fixed epsilon.
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(18, 5), gridspec_kw={"width_ratios": [1.5, 1]}
        )
        N_plot = np.linspace(10, N_max, 200)
        bound_plot = 2 * np.exp(-2 * N_plot * fixed_epsilon**2)
        ax1.plot(N_plot, bound_plot, linewidth=2.5, color="coral")
        ax1.fill_between(N_plot, 0, bound_plot, alpha=0.3, color="coral")
        ax1.set_xlabel("N (number of samples)", fontsize=12)
        ax1.set_ylabel("Hoeffding Bound", fontsize=12)
        ax1.set_title(
            f"Hoeffding Bound vs N (epsilon={fixed_epsilon:.3f})",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylim([0, min(2.0, np.max(bound_plot) * 1.1)])
        ax1.grid(True, alpha=0.3)
        # Add reference lines for common probability thresholds.
        thresholds = [0.05, 0.1, 0.2]
        for thresh in thresholds:
            if thresh < ax1.get_ylim()[1]:
                ax1.axhline(
                    y=thresh,
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                )
                ax1.text(
                    N_max * 0.95,
                    thresh + 0.01,
                    f"p={thresh}",
                    ha="right",
                    fontsize=9,
                    color="red",
                )
        # Comments panel.
        ax2.axis("off")
        ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
        text_content = (
            f"Hoeffding Bound:\n"
            f"  P(|nu - mu| >= epsilon) <= 2*exp(-2*N*epsilon^2)\n\n"
            f"Fixed Parameters:\n"
            f"  epsilon = {fixed_epsilon:.3f}\n"
            f"  N_max = {N_max}\n"
            f"  epsilon_max = {epsilon_max}\n\n"
            f"Observations:\n"
            f"- The bound decreases exponentially as N increases.\n\n"
            f"- Doubling N does not halve the bound; it decreases\n"
            f"  exponentially faster.\n\n"
            f"- Larger N (more samples) leads to smaller bound\n"
            f"  (higher confidence).\n\n"
            f"- The exponential decay shows why we need relatively\n"
            f"  few samples for good concentration."
        )
        mtumsuti.add_fitted_text_box(ax2, text_content)
    elif plot_type == "heatmap":
        # Create heatmap of bound as function of both N and epsilon.
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(18, 5), gridspec_kw={"width_ratios": [1.5, 1]}
        )
        # Use logarithmic scale for better visualization.
        bound_grid_log = np.log10(bound_grid + 1e-10)
        im = ax1.contourf(
            N_grid,
            epsilon_grid,
            bound_grid,
            levels=20,
            cmap="RdYlBu_r",
            alpha=0.9,
        )
        # Add contour lines.
        contour_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
        contours = ax1.contour(
            N_grid,
            epsilon_grid,
            bound_grid,
            levels=contour_levels,
            colors="black",
            linewidths=1.5,
            alpha=0.6,
        )
        ax1.clabel(contours, inline=True, fontsize=9, fmt="%.2f")
        ax1.set_xlabel("N (number of samples)", fontsize=12)
        ax1.set_ylabel("Epsilon (deviation threshold)", fontsize=12)
        ax1.set_title(
            "Hoeffding Bound: P(|nu - mu| >= epsilon)",
            fontsize=14,
            fontweight="bold",
        )
        # Add colorbar.
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label("Bound Value", fontsize=11)
        # Comments panel.
        ax2.axis("off")
        ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
        text_content = (
            f"Hoeffding Bound:\n"
            f"  P(|nu - mu| >= epsilon) <= 2*exp(-2*N*epsilon^2)\n\n"
            f"Parameters:\n"
            f"  N_max = {N_max}\n"
            f"  epsilon_max = {epsilon_max}\n\n"
            f"Color Interpretation:\n"
            f"- Blue: Low bound (high confidence)\n"
            f"- Yellow/Orange: Medium bound\n"
            f"- Red: High bound (low confidence)\n\n"
            f"Contour Lines:\n"
            f"- Black lines show constant probability levels\n"
            f"- Labels indicate the bound value\n\n"
            f"Key Insights:\n"
            f"- Lower-left (small N, large epsilon): high bound\n"
            f"- Upper-right (large N, small epsilon): varies\n"
            f"- The bound is exponentially sensitive to both N\n"
            f"  and epsilon."
        )
        mtumsuti.add_fitted_text_box(ax2, text_content)
    elif plot_type == "contour":
        # Create contour plot.
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(18, 5), gridspec_kw={"width_ratios": [1.5, 1]}
        )
        contour_levels = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        contours = ax1.contour(
            N_grid,
            epsilon_grid,
            bound_grid,
            levels=contour_levels,
            cmap="viridis",
            linewidths=2,
        )
        ax1.clabel(contours, inline=True, fontsize=10, fmt="%.3f")
        ax1.set_xlabel("N (number of samples)", fontsize=12)
        ax1.set_ylabel("Epsilon (deviation threshold)", fontsize=12)
        ax1.set_title(
            "Hoeffding Bound Contours",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        # Comments panel.
        ax2.axis("off")
        ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
        text_content = (
            f"Hoeffding Bound:\n"
            f"  P(|nu - mu| >= epsilon) <= 2*exp(-2*N*epsilon^2)\n\n"
            f"Parameters:\n"
            f"  N_max = {N_max}\n"
            f"  epsilon_max = {epsilon_max}\n\n"
            f"Contour Interpretation:\n"
            f"- Each line represents constant bound value\n"
            f"- Labels show the bound probability\n\n"
            f"Trade-offs:\n"
            f"- To maintain constant bound while decreasing epsilon,\n"
            f"  N must increase quadratically.\n\n"
            f"- The curves show (N, epsilon) pairs that achieve\n"
            f"  the same confidence level."
        )
        mtumsuti.add_fitted_text_box(ax2, text_content)
    # Adjust layout.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.25
    )
    plt.show()


# #############################################################################
# Cell 7: Hoeffding Bound as a Function of N and Epsilon
# #############################################################################


def cell7_bound_surface_heatmap() -> None:
    """
    Create interactive visualization of Hoeffding bound surface.

    Allows exploration of how the bound 2*exp(-2*N*epsilon^2) changes with:
    - N (number of samples)
    - epsilon (deviation threshold)

    Provides three view modes:
    - Fix N and vary epsilon
    - Fix epsilon and vary N
    - Show full 2D heatmap/contour of both
    """
    N_max_init = 500
    epsilon_max_init = 0.5
    # Create widgets for controlling the plot.
    N_max_slider, N_max_box = mtumsuti.build_widget_control(
        name="N_max",
        description="max N for plot",
        min_val=100,
        max_val=1000,
        step=50,
        initial_value=N_max_init,
        is_float=False,
    )
    epsilon_max_slider, epsilon_max_box = mtumsuti.build_widget_control(
        name="epsilon_max",
        description="max epsilon for plot",
        min_val=0.1,
        max_val=1.0,
        step=0.05,
        initial_value=epsilon_max_init,
        is_float=True,
    )
    # Create mode selector.
    mode_dropdown = ipywidgets.Dropdown(
        options=[
            ("Heatmap (both N and epsilon)", "heatmap"),
            ("Fix N, vary epsilon", "fixed_N"),
            ("Fix epsilon, vary N", "fixed_epsilon"),
            ("Contour plot (both N and epsilon)", "contour"),
        ],
        value="heatmap",
        description="View mode:",
        style={"description_width": "150px"},
        layout=ipywidgets.Layout(width="500px"),
    )
    # Create sliders for fixed values.
    fixed_N_slider, fixed_N_box = mtumsuti.build_widget_control(
        name="fixed_N",
        description="fixed N value",
        min_val=10,
        max_val=500,
        step=10,
        initial_value=100,
        is_float=False,
    )
    fixed_epsilon_slider, fixed_epsilon_box = mtumsuti.build_widget_control(
        name="fixed_epsilon",
        description="fixed epsilon value",
        min_val=0.01,
        max_val=0.5,
        step=0.01,
        initial_value=0.1,
        is_float=True,
    )
    # Create output widget.
    output = ipywidgets.Output()

    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)
            mode = mode_dropdown.value
            if mode == "fixed_N":
                _plot_hoeffding_bound_surface(
                    N_max=N_max_slider.value,
                    epsilon_max=epsilon_max_slider.value,
                    fixed_N=fixed_N_slider.value,
                    fixed_epsilon=None,
                )
            elif mode == "fixed_epsilon":
                _plot_hoeffding_bound_surface(
                    N_max=N_max_slider.value,
                    epsilon_max=epsilon_max_slider.value,
                    fixed_N=None,
                    fixed_epsilon=fixed_epsilon_slider.value,
                )
            else:
                _plot_hoeffding_bound_surface(
                    N_max=N_max_slider.value,
                    epsilon_max=epsilon_max_slider.value,
                    fixed_N=None,
                    fixed_epsilon=None,
                    plot_type=mode,
                )

    def update_visibility(change=None):
        """Show/hide fixed value sliders based on mode."""
        mode = mode_dropdown.value
        if mode == "fixed_N":
            fixed_N_box.layout.visibility = "visible"
            fixed_epsilon_box.layout.visibility = "hidden"
        elif mode == "fixed_epsilon":
            fixed_N_box.layout.visibility = "hidden"
            fixed_epsilon_box.layout.visibility = "visible"
        else:
            fixed_N_box.layout.visibility = "hidden"
            fixed_epsilon_box.layout.visibility = "hidden"
        update_plot()

    # Observe changes.
    mode_dropdown.observe(update_visibility, names="value")
    N_max_slider.observe(update_plot, names="value")
    epsilon_max_slider.observe(update_plot, names="value")
    fixed_N_slider.observe(update_plot, names="value")
    fixed_epsilon_slider.observe(update_plot, names="value")
    # Initial visibility setup.
    fixed_N_box.layout.visibility = "hidden"
    fixed_epsilon_box.layout.visibility = "hidden"
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                mode_dropdown,
                N_max_box,
                epsilon_max_box,
                fixed_N_box,
                fixed_epsilon_box,
                output,
            ]
        )
    )
    update_plot()


# #############################################################################
# Cell 8: Helper - Plot 3D Hoeffding Bound
# #############################################################################


def _plot_hoeffding_bound_3d(
    *,
    N_max: int = 500,
    epsilon_max: float = 0.5,
    elevation: float = 30,
    azimuth: float = 45,
    use_log_scale: bool = False,
) -> None:
    """
    Plot 3D surface of Hoeffding bound as a function of N and epsilon.

    Creates an interactive 3D visualization showing how the bound
    2*exp(-2*N*epsilon^2) varies across the (N, epsilon) space.

    :param N_max: Maximum N value for the plot
    :param epsilon_max: Maximum epsilon value for the plot
    :param elevation: Elevation angle for 3D view (degrees)
    :param azimuth: Azimuth angle for 3D view (degrees)
    :param use_log_scale: If True, use log scale for Z-axis (bound values)
    """
    # Import 3D plotting toolkit.

    # Create grid for N and epsilon.
    N_values = np.linspace(10, N_max, 100)
    epsilon_values = np.linspace(0.01, epsilon_max, 100)
    N_grid, epsilon_grid = np.meshgrid(N_values, epsilon_values)
    # Compute Hoeffding bound for each (N, epsilon) pair.
    bound_grid = 2 * np.exp(-2 * N_grid * epsilon_grid**2)
    # Apply log scale if requested.
    if use_log_scale:
        Z_grid = np.log10(bound_grid + 1e-10)
        z_label = "log10(Bound)"
    else:
        Z_grid = bound_grid
        z_label = "Bound Value"
    # Create 3D plot.
    fig = plt.figure(figsize=(16, 6))
    # Left subplot: 3D surface.
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(
        N_grid,
        epsilon_grid,
        Z_grid,
        cmap="viridis",
        alpha=0.9,
        edgecolor="none",
        linewidth=0,
        antialiased=True,
    )
    # Add contour lines on the surface.
    if not use_log_scale:
        contour_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
        for level in contour_levels:
            ax1.contour(
                N_grid,
                epsilon_grid,
                bound_grid,
                levels=[level],
                colors="red",
                linewidths=1.5,
                alpha=0.6,
                offset=level,
            )
    # Set labels and title.
    ax1.set_xlabel("N (samples)", fontsize=11, labelpad=10)
    ax1.set_ylabel("Epsilon", fontsize=11, labelpad=10)
    ax1.set_zlabel(z_label, fontsize=11, labelpad=10)
    ax1.set_title(
        "3D Surface: Hoeffding Bound",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    # Set viewing angle.
    ax1.view_init(elev=elevation, azim=azimuth)
    # Add colorbar.
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
    # Right subplot: Comments panel.
    ax2 = fig.add_subplot(122)
    ax2.axis("off")
    ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Generate interpretation text.
    scale_note = (
        "Note: Using log scale for better visibility."
        if use_log_scale
        else "Note: Using linear scale."
    )
    text_content = (
        f"Hoeffding Bound:\n"
        f"  P(|nu - mu| >= epsilon) <= 2*exp(-2*N*epsilon^2)\n\n"
        f"Parameters:\n"
        f"  N_max = {N_max}\n"
        f"  epsilon_max = {epsilon_max}\n"
        f"  elevation = {elevation} degrees\n"
        f"  azimuth = {azimuth} degrees\n\n"
        f"Visualization:\n"
        f"- {scale_note}\n"
        f"- Surface color indicates bound value\n"
        f"- Red contour lines show key probability levels\n\n"
        f"Key Features:\n"
        f"- Exponential decay in N direction (x-axis)\n"
        f"- Exponential decay in epsilon direction (y-axis)\n"
        f"- Steepest descent along the diagonal\n\n"
        f"Interpretation:\n"
        f"- Near N=0: bound approaches 2 (meaningless)\n"
        f"- Large N, large epsilon: bound approaches 0\n"
        f"- The surface shows trade-off between N and epsilon\n"
        f"  for achieving desired confidence."
    )
    mtumsuti.add_fitted_text_box(ax2, text_content)
    # Adjust layout.
    plt.tight_layout()
    plt.show()


# #############################################################################
# Cell 8: 3D Surface Visualization of Hoeffding Bound
# #############################################################################


def cell8_bound_3d_surface() -> None:
    """
    Create interactive 3D surface visualization of Hoeffding bound.

    Shows the bound 2*exp(-2*N*epsilon^2) as a 3D surface where:
    - X-axis: N (number of samples)
    - Y-axis: epsilon (deviation threshold)
    - Z-axis: Bound value (probability)

    Allows interactive control of:
    - Plot range (N_max, epsilon_max)
    - Viewing angle (elevation, azimuth)
    - Scale (linear or logarithmic for Z-axis)
    """
    N_max_init = 500
    epsilon_max_init = 0.5
    elevation_init = 30
    azimuth_init = 45
    # Create widgets for controlling the plot.
    N_max_slider, N_max_box = mtumsuti.build_widget_control(
        name="N_max",
        description="max N for plot",
        min_val=100,
        max_val=1000,
        step=50,
        initial_value=N_max_init,
        is_float=False,
    )
    epsilon_max_slider, epsilon_max_box = mtumsuti.build_widget_control(
        name="epsilon_max",
        description="max epsilon for plot",
        min_val=0.1,
        max_val=1.0,
        step=0.05,
        initial_value=epsilon_max_init,
        is_float=True,
    )
    elevation_slider, elevation_box = mtumsuti.build_widget_control(
        name="elevation",
        description="viewing angle (up/down)",
        min_val=0,
        max_val=90,
        step=5,
        initial_value=elevation_init,
        is_float=True,
    )
    azimuth_slider, azimuth_box = mtumsuti.build_widget_control(
        name="azimuth",
        description="viewing angle (rotation)",
        min_val=0,
        max_val=360,
        step=5,
        initial_value=azimuth_init,
        is_float=True,
    )
    # Create checkbox for log scale.
    log_scale_checkbox = ipywidgets.Checkbox(
        value=False,
        description="Use log scale for Z-axis",
        style={"description_width": "200px"},
        layout=ipywidgets.Layout(width="300px"),
    )
    # Create output widget.
    output = ipywidgets.Output()

    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)
            _plot_hoeffding_bound_3d(
                N_max=N_max_slider.value,
                epsilon_max=epsilon_max_slider.value,
                elevation=elevation_slider.value,
                azimuth=azimuth_slider.value,
                use_log_scale=log_scale_checkbox.value,
            )

    # Observe changes.
    N_max_slider.observe(update_plot, names="value")
    epsilon_max_slider.observe(update_plot, names="value")
    elevation_slider.observe(update_plot, names="value")
    azimuth_slider.observe(update_plot, names="value")
    log_scale_checkbox.observe(update_plot, names="value")
    # Display widgets.
    display(
        ipywidgets.VBox(
            [
                N_max_box,
                epsilon_max_box,
                elevation_box,
                azimuth_box,
                log_scale_checkbox,
                output,
            ]
        )
    )
    update_plot()
