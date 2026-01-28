"""
Utility functions for Learning Theory lesson.

Import as:

import msml610.tutorials.utils_Lesson05_Learning_Theory.old as mtullthol
"""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)

# Suppress FutureWarnings from seaborn and other libraries.
warnings.filterwarnings("ignore", category=FutureWarning)


# #############################################################################
# Hoeffding Inequality
# #############################################################################


def simulate_marble_sampling(mu: float, N: int) -> float:
    """
    Simulate sampling N marbles from a bin with true proportion mu of red marbles.

    :param mu: True proportion of red marbles in the bin (0 <= mu <= 1)
    :param N: Number of marbles to sample
    :return: Sample proportion nu of red marbles
    """
    hdbg.dassert_lte(0.0, mu)
    hdbg.dassert_lte(mu, 1.0)
    hdbg.dassert_lt(0, N)
    # Sample N marbles: 1 = red, 0 = green.
    samples = np.random.binomial(1, mu, N)
    # Calculate sample proportion.
    nu = np.mean(samples)
    return nu


def calculate_hoeffding_bound(epsilon: float, N: int) -> float:
    r"""
    Calculate the Hoeffding inequality bound.

    The Hoeffding inequality states:

    $$P(|\nu - \mu| > \varepsilon) \le 2e^{-2\varepsilon^2 N}$$

    :param epsilon: Error tolerance
    :param N: Sample size
    :return: Hoeffding bound
    """
    hdbg.dassert_lt(0, epsilon)
    hdbg.dassert_lt(0, N)
    bound = 2 * np.exp(-2 * epsilon**2 * N)
    return bound


def run_hoeffding_experiment(
    mu: float, N: int, epsilon: float, n_trials: int
) -> Dict[str, Any]:
    """
    Run multiple Hoeffding experiments and collect statistics.

    :param mu: True proportion of red marbles
    :param N: Sample size per trial
    :param epsilon: Error tolerance
    :param n_trials: Number of trials to run
    :return: Dictionary with statistics including:
        - nus: Array of sample proportions from all trials
        - violation_count: Number of times |nu - mu| > epsilon
        - violation_rate: Empirical violation rate
        - theoretical_bound: Hoeffding bound
    """
    hdbg.dassert_lte(0.0, mu)
    hdbg.dassert_lte(mu, 1.0)
    hdbg.dassert_lt(0, N)
    hdbg.dassert_lt(0, epsilon)
    hdbg.dassert_lte(epsilon, 1.0)
    hdbg.dassert_lt(0, n_trials)
    # Run n_trials experiments.
    nus = np.array([simulate_marble_sampling(mu, N) for _ in range(n_trials)])
    # Count violations: |nu - mu| > epsilon.
    violations = np.abs(nus - mu) > epsilon
    violation_count = np.sum(violations)
    violation_rate = violation_count / n_trials
    # Calculate Hoeffding bound.
    theoretical_bound = calculate_hoeffding_bound(epsilon, N)
    # Return statistics.
    results = {
        "nus": nus,
        "violation_count": violation_count,
        "violation_rate": violation_rate,
        "theoretical_bound": theoretical_bound,
        "mu": mu,
        "N": N,
        "epsilon": epsilon,
        "n_trials": n_trials,
    }
    return results


def _plot_marble_bin(ax: plt.Axes, results: Dict[str, Any]) -> None:
    """
    Plot marble bin visualization showing a single sample.

    :param ax: Matplotlib axis
    :param results: Results dictionary from run_hoeffding_experiment
    """
    mu = results["mu"]
    N = results["N"]
    # Run a single sample to visualize.
    sample = np.random.binomial(1, mu, N)
    nu = np.mean(sample)
    # Create visualization of marbles.
    # Show marbles in a grid layout.
    grid_size = int(np.ceil(np.sqrt(N)))
    # Pad sample to fit grid.
    padded_sample = np.pad(sample, (0, grid_size**2 - N), constant_values=-1)
    marble_grid = padded_sample.reshape(grid_size, grid_size)
    # Create color map: red for 1, green for 0, white for padding.
    colors = ["white", "green", "red"]
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    # Plot the grid.
    ax.imshow(marble_grid, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    # Add title and information.
    ax.set_title(
        f"Marble Sampling Visualization\n(Single Sample of N={N} marbles)",
        fontsize=12,
        fontweight="bold",
    )
    # Add text box with information.
    info_text = (
        f"True proportion: $\\mu = {mu:.2f}$\n"
        f"Sample proportion: $\\nu = {nu:.2f}$\n"
        f"Difference: $|\\nu - \\mu| = {abs(nu - mu):.3f}$"
    )
    ax.text(
        0.5,
        -0.15,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def _plot_sampling_distribution(ax: plt.Axes, results: Dict[str, Any]) -> None:
    """
    Plot histogram of sample proportions across all trials.

    :param ax: Matplotlib axis
    :param results: Results dictionary from run_hoeffding_experiment
    """
    mu = results["mu"]
    epsilon = results["epsilon"]
    nus = results["nus"]
    n_trials = results["n_trials"]
    # Plot histogram of sample proportions.
    ax.hist(
        nus,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label=f"Empirical distribution\n(n={n_trials} trials)",
    )
    # Add vertical lines for mu and bounds.
    ax.axvline(
        mu, color="red", linestyle="--", linewidth=2, label=f"$\\mu = {mu:.2f}$"
    )
    ax.axvline(
        mu - epsilon,
        color="orange",
        linestyle=":",
        linewidth=2,
        label="$\\mu \\pm \\varepsilon$",
    )
    ax.axvline(mu + epsilon, color="orange", linestyle=":", linewidth=2)
    # Shade the acceptable region.
    ax.axvspan(
        mu - epsilon,
        mu + epsilon,
        alpha=0.2,
        color="green",
        label="Acceptable region",
    )
    # Add title and labels.
    ax.set_title(
        f"Sampling Distribution of $\\nu$\n(across {n_trials} trials)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Sample proportion $\\nu$", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_hoeffding_bound_curve(ax: plt.Axes, results: Dict[str, Any]) -> None:
    """
    Plot Hoeffding bound as a function of sample size N.

    :param ax: Matplotlib axis
    :param results: Results dictionary from run_hoeffding_experiment
    """
    epsilon = results["epsilon"]
    N = results["N"]
    theoretical_bound = results["theoretical_bound"]
    violation_rate = results["violation_rate"]
    # Generate range of N values.
    N_values = np.linspace(10, max(1000, N * 2), 200)
    bounds = [calculate_hoeffding_bound(epsilon, n) for n in N_values]
    # Plot the bound curve.
    ax.plot(
        N_values,
        bounds,
        color="blue",
        linewidth=2,
        label="Hoeffding bound:\n$2e^{-2\\varepsilon^2 N}$",
    )
    # Mark current N position.
    ax.plot(
        N,
        theoretical_bound,
        "ro",
        markersize=10,
        label=f"Current N={N}\nBound={theoretical_bound:.4f}",
    )
    # Add horizontal line for empirical violation rate.
    ax.axhline(
        violation_rate,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Empirical violation rate\n{violation_rate:.4f}",
    )
    # Add title and labels.
    ax.set_title(
        f"Hoeffding Bound vs Sample Size\n($\\varepsilon = {epsilon:.2f}$)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Sample size N", fontsize=10)
    ax.set_ylabel("Probability bound", fontsize=10)
    ax.set_ylim(0, min(1.0, max(bounds) * 1.1))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    # Use log scale for y-axis if bound is very small.
    if theoretical_bound < 0.01:
        ax.set_yscale("log")


def _plot_comments(ax: plt.Axes, results: Dict[str, Any]) -> None:
    """
    Plot comments panel with interpretation of results.

    :param ax: Matplotlib axis
    :param results: Results dictionary from run_hoeffding_experiment
    """
    mu = results["mu"]
    N = results["N"]
    epsilon = results["epsilon"]
    n_trials = results["n_trials"]
    violation_count = results["violation_count"]
    violation_rate = results["violation_rate"]
    theoretical_bound = results["theoretical_bound"]
    # Turn off axis.
    ax.axis("off")
    # Build interpretation text.
    interpretation = []
    interpretation.append("HOEFFDING INEQUALITY INTERPRETATION")
    interpretation.append("=" * 45)
    interpretation.append("")
    interpretation.append("Parameters:")
    interpretation.append(f"  True proportion (mu): {mu:.2f}")
    interpretation.append(f"  Sample size (N): {N}")
    interpretation.append(f"  Error tolerance (epsilon): {epsilon:.2f}")
    interpretation.append(f"  Number of trials: {n_trials}")
    interpretation.append("")
    interpretation.append("Results:")
    interpretation.append(f"  Violations: {violation_count}/{n_trials}")
    interpretation.append(f"  Empirical rate: {violation_rate:.4f}")
    interpretation.append(f"  Theoretical bound: {theoretical_bound:.4f}")
    interpretation.append("")
    # Add interpretation based on results.
    if violation_rate <= theoretical_bound:
        interpretation.append("OBSERVATION:")
        interpretation.append(
            f"  The empirical violation rate ({violation_rate:.4f})"
        )
        interpretation.append(
            f"  is LESS than the Hoeffding bound ({theoretical_bound:.4f})."
        )
        interpretation.append("  This confirms the bound is valid!")
    else:
        interpretation.append("OBSERVATION:")
        interpretation.append("  The empirical rate slightly exceeds the bound.")
        interpretation.append(
            "  This can happen due to statistical fluctuation."
        )
        interpretation.append(
            "  Try increasing n_trials for more stable results."
        )
    interpretation.append("")
    interpretation.append("KEY INSIGHTS:")
    # Insight about sample size.
    if N < 100:
        interpretation.append(f"  With N={N}, the bound is relatively loose.")
        interpretation.append(
            "  Try increasing N to see exponential improvement!"
        )
    else:
        interpretation.append(f"  With N={N}, the bound is tight, showing")
        interpretation.append("  exponential convergence with sample size.")
    # Insight about epsilon.
    if epsilon < 0.05:
        interpretation.append(
            f"  Small epsilon ({epsilon:.2f}) requires large N"
        )
        interpretation.append("  for good confidence.")
    elif epsilon > 0.2:
        interpretation.append(
            f"  Large epsilon ({epsilon:.2f}) is more tolerant,"
        )
        interpretation.append("  allowing smaller sample sizes.")
    interpretation.append("")
    interpretation.append("CONNECTION TO MACHINE LEARNING:")
    interpretation.append("  mu = Out-of-sample error E_out(h)")
    interpretation.append("  nu = In-sample error E_in(h)")
    interpretation.append("  Hoeffding tells us: E_in tracks E_out")
    interpretation.append("  with high probability!")
    interpretation.append("")
    interpretation.append("  This is why ML generalization works!")
    # Join and display.
    text = "\n".join(interpretation)
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )


def plot_hoeffding_interactive(
    mu: float = 0.6,
    N: int = 100,
    epsilon: float = 0.1,
    n_trials: int = 1000,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Interactive visualization of Hoeffding Inequality.

    Creates 4 panels showing:
    1. Marble sampling visualization
    2. Sampling distribution histogram
    3. Hoeffding bound curve
    4. Comments and interpretation

    :param mu: True proportion of red marbles (0 < mu < 1)
    :param N: Sample size (number of marbles drawn)
    :param epsilon: Error tolerance
    :param n_trials: Number of experiments to run
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Run the experiment.
    results = run_hoeffding_experiment(mu, N, epsilon, n_trials)
    # Create figure with 4 subplots.
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    # Panel 1: Marble bin visualization.
    _plot_marble_bin(axes[0], results)
    # Panel 2: Sampling distribution.
    _plot_sampling_distribution(axes[1], results)
    # Panel 3: Hoeffding bound curve.
    _plot_hoeffding_bound_curve(axes[2], results)
    # Panel 4: Comments.
    _plot_comments(axes[3], results)
    plt.tight_layout()
    plt.show()


def _plot_single_sample(ax: plt.Axes, mu: float, N: int) -> None:
    """
    Plot visualization of a single Bernoulli sample.

    :param ax: Matplotlib axis
    :param mu: True probability of success
    :param N: Number of samples
    """
    # Generate single sample.
    sample = np.random.binomial(1, mu, N)
    nu = np.mean(sample)
    # Create bar chart showing the sample.
    x_vals = np.arange(N)
    colors = ["red" if s == 1 else "green" for s in sample]
    ax.bar(x_vals, sample, color=colors, edgecolor="black", alpha=0.7)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Sample index", fontsize=10)
    ax.set_ylabel("Outcome (1=success, 0=failure)", fontsize=10)
    ax.set_title(
        f"Single Bernoulli Sample\n(N={N})",
        fontsize=12,
        fontweight="bold",
    )
    # Add text box with information.
    info_text = (
        f"True prob: $\\mu = {mu:.2f}$\n"
        f"Sample mean: $\\nu = {nu:.2f}$\n"
        f"Successes: {int(np.sum(sample))}/{N}"
    )
    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )
    ax.grid(True, alpha=0.3, axis="y")


def _plot_empirical_distribution_nu(
    ax: plt.Axes,
    nus: np.ndarray,
    mu: float,
    N: int,
    n_trials: int,
) -> None:
    """
    Plot empirical distribution of sample means nu.

    :param ax: Matplotlib axis
    :param nus: Array of sample means from all trials
    :param mu: True probability of success
    :param N: Sample size per trial
    :param n_trials: Number of trials
    """
    # Plot histogram.
    ax.hist(
        nus,
        bins=40,
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        label=f"Empirical (n={n_trials})",
    )
    # Add vertical line for true mean.
    ax.axvline(
        mu,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"$\\mu = {mu:.2f}$",
    )
    # Add vertical line for sample mean.
    sample_mean = np.mean(nus)
    ax.axvline(
        sample_mean,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Sample mean = {sample_mean:.3f}",
    )
    ax.set_xlabel("Sample proportion $\\nu$", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"Empirical Distribution of $\\nu$\n({n_trials} trials)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_expected_distribution_nu(
    ax: plt.Axes,
    mu: float,
    N: int,
) -> None:
    """
    Plot expected distribution of nu using normal approximation.

    :param ax: Matplotlib axis
    :param mu: True probability of success
    :param N: Sample size
    """
    # Calculate theoretical mean and std dev.
    # For Bernoulli: variance = mu * (1 - mu)
    # For sample mean: std_dev = sqrt(variance / N)
    theoretical_std = np.sqrt(mu * (1 - mu) / N)
    # Generate x values.
    x_vals = np.linspace(mu - 4 * theoretical_std, mu + 4 * theoretical_std, 200)
    # Calculate normal distribution.
    y_vals = norm.pdf(x_vals, loc=mu, scale=theoretical_std)
    # Plot the distribution.
    ax.plot(
        x_vals,
        y_vals,
        color="darkblue",
        linewidth=2,
        label=f"Normal($\\mu={mu:.2f}$, $\\sigma={theoretical_std:.3f}$)",
    )
    ax.fill_between(x_vals, y_vals, alpha=0.3, color="skyblue")
    # Add vertical line for mean.
    ax.axvline(
        mu,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mu:.2f}",
    )
    # Shade 1 std dev region.
    ax.axvspan(
        mu - theoretical_std,
        mu + theoretical_std,
        alpha=0.2,
        color="yellow",
        label="$\\pm 1\\sigma$ region",
    )
    ax.set_xlabel("Sample proportion $\\nu$", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"Expected Distribution (Normal Approx)\n(N={N})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_study_comments_cell1(
    ax: plt.Axes,
    mu: float,
    N: int,
    n_trials: int,
    nus: np.ndarray,
) -> None:
    """
    Plot comments for Cell 1 study.

    :param ax: Matplotlib axis
    :param mu: True probability
    :param N: Sample size
    :param n_trials: Number of trials
    :param nus: Array of sample means
    """
    # Turn off axis.
    ax.axis("off")
    # Calculate statistics.
    empirical_mean = np.mean(nus)
    empirical_std = np.std(nus)
    theoretical_std = np.sqrt(mu * (1 - mu) / N)
    # Build comments.
    comments = []
    comments.append("EMPIRICAL VS EXPECTED DISTRIBUTION")
    comments.append("=" * 45)
    comments.append("")
    comments.append("Setup:")
    comments.append(f"  True probability: mu = {mu:.2f}")
    comments.append(f"  Sample size: N = {N}")
    comments.append(f"  Number of trials: {n_trials}")
    comments.append("")
    comments.append("Theoretical (Law of Large Numbers):")
    comments.append(f"  Expected mean: E[nu] = {mu:.3f}")
    comments.append(f"  Expected std dev: {theoretical_std:.4f}")
    comments.append("")
    comments.append("Empirical (from trials):")
    comments.append(f"  Observed mean: {empirical_mean:.3f}")
    comments.append(f"  Observed std dev: {empirical_std:.4f}")
    comments.append("")
    comments.append("Observations:")
    # Check if empirical matches theoretical.
    mean_diff = abs(empirical_mean - mu)
    std_diff = abs(empirical_std - theoretical_std)
    if mean_diff < 0.01:
        comments.append("  Empirical mean closely matches mu!")
    else:
        comments.append(f"  Mean differs by {mean_diff:.3f}")
        comments.append("  (Try more trials for convergence)")
    if std_diff / theoretical_std < 0.1:
        comments.append("  Std dev matches theory well!")
    else:
        comments.append("  Std dev shows some variation")
    comments.append("")
    comments.append("Key Insights:")
    if N < 50:
        comments.append(f"  N={N} is small - expect high variance")
        comments.append("  Increase N for tighter distribution")
    elif N < 200:
        comments.append(f"  N={N} gives moderate precision")
        comments.append("  Distribution shows clear peak at mu")
    else:
        comments.append(f"  N={N} is large - very tight distribution")
        comments.append("  Sample means cluster tightly around mu")
    comments.append("")
    comments.append("Connection to Learning:")
    comments.append("  This shows nu (sample statistic)")
    comments.append("  converges to mu (population parameter)")
    comments.append("  as N increases - basis of ML!")
    # Join and display.
    text = "\n".join(comments)
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )


def plot_hoeffding_study_empirical_vs_expected(
    mu: float = 0.6,
    N: int = 100,
    n_trials: int = 1000,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Interactive study of Hoeffding inequality: empirical vs expected distribution.

    Creates 4 panels showing:
    1. Single sample visualization
    2. Empirical distribution of nu from trials
    3. Expected distribution from normal approximation
    4. Comments and interpretation

    :param mu: True probability of success (0 < mu < 1)
    :param N: Number of samples per trial
    :param n_trials: Number of trials to run
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Run trials to collect sample means.
    nus = np.array([simulate_marble_sampling(mu, N) for _ in range(n_trials)])
    # Create figure with 4 subplots.
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    # Panel 1: Single sample visualization.
    _plot_single_sample(axes[0], mu, N)
    # Panel 2: Empirical distribution.
    _plot_empirical_distribution_nu(axes[1], nus, mu, N, n_trials)
    # Panel 3: Expected distribution.
    _plot_expected_distribution_nu(axes[2], mu, N)
    # Panel 4: Comments.
    _plot_study_comments_cell1(axes[3], mu, N, n_trials, nus)
    plt.tight_layout()
    plt.show()


def _plot_difference_distribution(
    ax: plt.Axes,
    differences: np.ndarray,
    mu: float,
    N: int,
    n_trials: int,
) -> None:
    """
    Plot distribution of mu - nu.

    :param ax: Matplotlib axis
    :param differences: Array of (mu - nu) values
    :param mu: True probability
    :param N: Sample size
    :param n_trials: Number of trials
    """
    # Plot histogram.
    ax.hist(
        differences,
        bins=40,
        density=True,
        alpha=0.6,
        color="purple",
        edgecolor="black",
        label=f"Empirical (n={n_trials})",
    )
    # Add vertical line at zero.
    ax.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Zero error",
    )
    # Add expected distribution.
    theoretical_std = np.sqrt(mu * (1 - mu) / N)
    x_vals = np.linspace(differences.min(), differences.max(), 200)
    y_vals = norm.pdf(x_vals, loc=0, scale=theoretical_std)
    ax.plot(
        x_vals,
        y_vals,
        color="darkblue",
        linewidth=2,
        linestyle=":",
        label=f"Expected N(0, {theoretical_std:.3f})",
    )
    ax.set_xlabel("Error: $\\mu - \\nu$", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"Distribution of $\\mu - \\nu$\n({n_trials} trials)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_cumulative_abs_error(
    ax: plt.Axes,
    abs_differences: np.ndarray,
    n_trials: int,
) -> None:
    """
    Plot cumulative distribution of absolute errors.

    :param ax: Matplotlib axis
    :param abs_differences: Array of |mu - nu| values
    :param n_trials: Number of trials
    """
    # Sort absolute differences.
    sorted_abs_diff = np.sort(abs_differences)
    # Calculate cumulative probability.
    cumulative_prob = np.arange(1, n_trials + 1) / n_trials
    # Plot cumulative distribution.
    ax.plot(
        sorted_abs_diff,
        cumulative_prob,
        color="darkgreen",
        linewidth=2,
        label="Empirical CDF",
    )
    ax.fill_between(
        sorted_abs_diff,
        cumulative_prob,
        alpha=0.3,
        color="lightgreen",
    )
    # Add horizontal lines for common percentiles.
    for percentile, label_text in [(0.68, "68%"), (0.95, "95%"), (0.99, "99%")]:
        ax.axhline(
            percentile,
            color="gray",
            linestyle=":",
            linewidth=1,
            alpha=0.7,
        )
        ax.text(
            ax.get_xlim()[1] * 0.95,
            percentile,
            label_text,
            fontsize=8,
            verticalalignment="bottom",
        )
    ax.set_xlabel("Absolute error: $|\\mu - \\nu|$", fontsize=10)
    ax.set_ylabel("Cumulative probability", fontsize=10)
    ax.set_title(
        "CDF of Absolute Errors\n(What fraction < epsilon?)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_hoeffding_comparison(
    ax: plt.Axes,
    abs_differences: np.ndarray,
    N: int,
    n_trials: int,
) -> None:
    """
    Plot Hoeffding bound comparison.

    :param ax: Matplotlib axis
    :param abs_differences: Array of |mu - nu| values
    :param N: Sample size
    :param n_trials: Number of trials
    """
    # Generate range of epsilon values.
    epsilon_vals = np.linspace(0.01, 0.5, 100)
    # Calculate empirical violation rate for each epsilon.
    empirical_rates = []
    for eps in epsilon_vals:
        violations = np.sum(abs_differences > eps)
        empirical_rates.append(violations / n_trials)
    # Calculate Hoeffding bounds.
    hoeffding_bounds = [
        calculate_hoeffding_bound(eps, N) for eps in epsilon_vals
    ]
    # Plot empirical rates.
    ax.plot(
        epsilon_vals,
        empirical_rates,
        color="red",
        linewidth=2,
        label="Empirical violation rate",
    )
    # Plot Hoeffding bound.
    ax.plot(
        epsilon_vals,
        hoeffding_bounds,
        color="blue",
        linewidth=2,
        linestyle="--",
        label="Hoeffding bound",
    )
    # Shade region between them.
    ax.fill_between(
        epsilon_vals,
        empirical_rates,
        hoeffding_bounds,
        alpha=0.3,
        color="green",
        label="Safety margin",
    )
    ax.set_xlabel("Error tolerance $\\varepsilon$", fontsize=10)
    ax.set_ylabel("Probability of violation", fontsize=10)
    ax.set_title(
        f"Hoeffding Bound Validation\n(N={N})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")


def _plot_study_comments_cell2(
    ax: plt.Axes,
    mu: float,
    N: int,
    n_trials: int,
    differences: np.ndarray,
) -> None:
    """
    Plot comments for Cell 2 study.

    :param ax: Matplotlib axis
    :param mu: True probability
    :param N: Sample size
    :param n_trials: Number of trials
    :param differences: Array of (mu - nu) values
    """
    # Turn off axis.
    ax.axis("off")
    # Calculate statistics.
    abs_differences = np.abs(differences)
    mean_abs_error = np.mean(abs_differences)
    max_abs_error = np.max(abs_differences)
    percentile_95 = np.percentile(abs_differences, 95)
    theoretical_std = np.sqrt(mu * (1 - mu) / N)
    # Build comments.
    comments = []
    comments.append("DISTRIBUTION OF mu - nu")
    comments.append("=" * 45)
    comments.append("")
    comments.append("Setup:")
    comments.append(f"  True probability: mu = {mu:.2f}")
    comments.append(f"  Sample size: N = {N}")
    comments.append(f"  Number of trials: {n_trials}")
    comments.append("")
    comments.append("Error Statistics:")
    comments.append(f"  Mean absolute error: {mean_abs_error:.4f}")
    comments.append(f"  Max absolute error: {max_abs_error:.4f}")
    comments.append(f"  95th percentile: {percentile_95:.4f}")
    comments.append(f"  Theoretical std: {theoretical_std:.4f}")
    comments.append("")
    comments.append("Key Observations:")
    # Check distribution centering.
    mean_diff = np.mean(differences)
    if abs(mean_diff) < 0.01:
        comments.append("  Errors centered at zero (unbiased)")
    else:
        comments.append(f"  Mean error: {mean_diff:.4f}")
    # Check spread.
    if mean_abs_error < theoretical_std:
        comments.append("  Typical error < theoretical std")
        comments.append("  Good agreement with theory!")
    else:
        comments.append("  Errors within expected range")
    comments.append("")
    comments.append("Hoeffding Inequality Check:")
    comments.append("  Empirical violation rates should")
    comments.append("  stay BELOW Hoeffding bound")
    comments.append("  This validates the PAC framework!")
    comments.append("")
    comments.append("What This Means for ML:")
    comments.append("  The difference mu - nu represents")
    comments.append("  generalization error")
    comments.append("")
    comments.append("  Hoeffding tells us this difference")
    comments.append("  is small with high probability")
    comments.append("  when N is large enough")
    comments.append("")
    comments.append("  This is why test error (nu)")
    comments.append("  predicts true error (mu)!")
    # Join and display.
    text = "\n".join(comments)
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )


def plot_hoeffding_study_difference_distribution(
    mu: float = 0.6,
    N: int = 100,
    n_trials: int = 1000,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Interactive study of Hoeffding inequality: distribution of mu - nu.

    Creates 4 panels showing:
    1. Distribution of errors (mu - nu)
    2. Cumulative distribution of absolute errors
    3. Comparison with Hoeffding bound
    4. Comments and interpretation

    :param mu: True probability of success (0 < mu < 1)
    :param N: Number of samples per trial
    :param n_trials: Number of trials to run
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Run trials to collect sample means.
    nus = np.array([simulate_marble_sampling(mu, N) for _ in range(n_trials)])
    # Calculate differences.
    differences = mu - nus
    abs_differences = np.abs(differences)
    # Create figure with 4 subplots.
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    # Panel 1: Distribution of differences.
    _plot_difference_distribution(axes[0], differences, mu, N, n_trials)
    # Panel 2: Cumulative distribution of absolute errors.
    _plot_cumulative_abs_error(axes[1], abs_differences, n_trials)
    # Panel 3: Hoeffding bound comparison.
    _plot_hoeffding_comparison(axes[2], abs_differences, N, n_trials)
    # Panel 4: Comments.
    _plot_study_comments_cell2(axes[3], mu, N, n_trials, differences)
    plt.tight_layout()
    plt.show()
