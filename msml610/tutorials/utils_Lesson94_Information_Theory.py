"""
Utility functions for Information Theory lesson.

Import as:

import msml610.tutorials.utils_Lesson94_Information_Theory as mtulinth
"""

import logging
import textwrap
import warnings
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)

# Suppress FutureWarnings from seaborn and other libraries.
warnings.filterwarnings("ignore", category=FutureWarning)


# #############################################################################
# Entropy calculations
# #############################################################################


def calculate_entropy(probabilities: Union[List[float], np.ndarray]) -> float:
    r"""
    Calculate Shannon entropy for a discrete probability distribution.

    Entropy $H(X)$ of a discrete random variable $X$ is defined as:

    $$H(X) = -\sum_x p(x) \log_2 p(x)$$

    :param probabilities: Array of probabilities (must sum to 1)
    :return: Entropy in bits
    """

    # Convert to numpy array.
    probabilities = np.array(probabilities)
    # Check that probabilities sum to 1.
    prob_sum = np.sum(probabilities)
    hdbg.dassert_lte(
        abs(prob_sum - 1.0),
        1e-6,
        "Probabilities must sum to 1, got sum=%s",
        prob_sum,
    )
    # Filter out zero probabilities to avoid log(0).
    probabilities = probabilities[probabilities > 0]
    # Calculate entropy using log base 2.
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def binary_entropy(p: float) -> float:
    r"""
    Calculate entropy of a binary random variable.

    Entropy $H(p)$ is defined as:

    $$H(p) = -p \log_2 p - (1-p) \log_2 (1-p)$$

    :param p: Probability of outcome 1
    :return: Binary entropy H(p)
    """
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def calculate_joint_entropy(joint_prob: np.ndarray) -> float:
    r"""
    Calculate joint entropy H(X,Y) from joint probability distribution.

    Joint Entropy $H(X, Y)$ of two variables $X$ and $Y$ is defined as:

    $$H(X, Y) = -\sum_{x,y} p(x,y) \log_2 p(x,y)$$

    :param joint_prob: 2D array of joint probabilities p(x,y)
    :return: Joint entropy in bits
    """
    joint_prob = np.array(joint_prob)
    # Flatten the joint probability distribution.
    joint_prob_flat = joint_prob.flatten()
    # Use calculate_entropy() for consistent calculation.
    return calculate_entropy(joint_prob_flat)


def calculate_conditional_entropy(joint_prob: np.ndarray) -> float:
    r"""
    Calculate conditional entropy H(Y|X) from joint probability distribution.

    Conditional Entropy $H(Y|X)$ measures uncertainty in $Y$ after
    observing $X$:

    $$H(Y|X) = -\sum_{x,y} p(x,y) \log_2 p(y|x) = \sum_x p(x) H(Y|X=x)$$

    :param joint_prob: 2D array of joint probabilities p(x,y)
    :return: Conditional entropy H(Y|X) in bits
    """
    joint_prob = np.array(joint_prob)
    # Calculate marginal p(x).
    p_x = joint_prob.sum(axis=1)
    conditional_entropy = 0
    for i, px in enumerate(p_x):
        if px > 0:
            # Calculate p(y|x) for this x.
            p_y_given_x = joint_prob[i, :] / px
            # Calculate H(Y|X=x).
            p_y_given_x = p_y_given_x[p_y_given_x > 0]
            h_y_given_x = -np.sum(p_y_given_x * np.log2(p_y_given_x))
            # Weight by p(x).
            conditional_entropy += px * h_y_given_x
    return conditional_entropy


# #############################################################################
# Mutual information
# #############################################################################


def calculate_mutual_information(joint_prob: np.ndarray) -> float:
    r"""
    Calculate mutual information I(X;Y) from joint probability distribution.

    Mutual Information $I(X;Y)$ measures how much knowing one variable
    reduces uncertainty about the other:

    $$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$$

    :param joint_prob: 2D array of joint probabilities p(x,y)
    :return: Mutual information in bits
    """
    joint_prob = np.array(joint_prob)
    # Calculate marginals.
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    # Calculate entropies.
    h_x = calculate_entropy(p_x)
    h_y = calculate_entropy(p_y)
    h_xy = calculate_joint_entropy(joint_prob)
    # Mutual information.
    mi = h_x + h_y - h_xy
    return mi


def create_correlated_joint_distribution(
    *, correlation: float = 0.5
) -> np.ndarray:
    """
    Create a 2x2 joint distribution with specified correlation.

    :param correlation: Correlation strength (0=independent, 1=perfectly
        correlated)
    :return: 2x2 joint probability matrix
    """
    # Create joint distribution with correlation.
    p11 = 0.25 + correlation * 0.25
    p00 = 0.25 + correlation * 0.25
    p10 = 0.25 - correlation * 0.25
    p01 = 0.25 - correlation * 0.25
    joint_prob = np.array([[p00, p01], [p10, p11]])
    return joint_prob


def plot_joint_entropy_interactive(
    *,
    dependence: float = 0.5,
    n_samples: int = 100,
    figsize: Optional[tuple] = None,
) -> None:
    """
    Interactive visualization of joint entropy with dependence control.

    :param dependence: Dependence strength between variables (0=independent,
        1=perfectly correlated)
    :param n_samples: Number of samples to generate for scatter plot
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Create joint distribution with specified dependence.
    joint_prob = create_correlated_joint_distribution(correlation=dependence)
    # Convert to DataFrame for better visualization.
    joint_df = pd.DataFrame(
        joint_prob, index=["X=0", "X=1"], columns=["Y=0", "Y=1"]
    )
    # Calculate marginals.
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    # Calculate entropy metrics.
    h_x = calculate_entropy(p_x)
    h_y = calculate_entropy(p_y)
    h_xy = calculate_joint_entropy(joint_prob)
    mi = calculate_mutual_information(joint_prob)
    # Create DataFrame for entropy metrics (without I(X;Y)).
    metrics_df = pd.DataFrame(
        {
            "Metric": ["H(X)", "H(Y)", "H(X,Y)"],
            "Value": [h_x, h_y, h_xy],
        }
    )
    # Determine interpretation message based on dependence.
    if dependence < 0.1:
        interpretation = (
            "Independence: H(X,Y) ≈ H(X) + H(Y) (maximum joint entropy)"
        )
    elif dependence > 0.9:
        interpretation = (
            "Perfect dependence: H(X,Y) ≈ H(X) = H(Y) (minimum joint entropy)"
        )
    else:
        interpretation = f"Partial dependence: Joint entropy reduced by {mi:.4f} bits due to shared information"
    # Generate samples from the joint distribution.
    # Flatten joint probabilities and sample from the 4 outcomes.
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    probs = joint_prob.flatten()
    # Sample indices and convert to (x, y) pairs.
    sampled_indices = np.random.choice(len(outcomes), size=n_samples, p=probs)
    samples = [outcomes[i] for i in sampled_indices]
    x_samples = [s[0] for s in samples]
    y_samples = [s[1] for s in samples]
    # Add jitter for better visualization.
    jitter_amount = 0.05
    x_jittered = x_samples + np.random.normal(0, jitter_amount, n_samples)
    y_jittered = y_samples + np.random.normal(0, jitter_amount, n_samples)
    # Create DataFrame for samples.
    samples_df = pd.DataFrame({"X": x_jittered, "Y": y_jittered})
    # Create visualization with 4 subplots in a single row.
    # Use gridspec_kw to set fixed width ratios for consistent layout.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 1, 1.2]}
    )
    # Plot 1: Joint distribution heatmap using seaborn.
    sns.heatmap(
        joint_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0,
        vmax=0.5,
        cbar=True,
        ax=ax1,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )
    ax1.set_xlabel("Variable Y", fontsize=12)
    ax1.set_ylabel("Variable X", fontsize=12)
    ax1.set_title(
        f"Joint Distribution P(X,Y)\nDependence = {dependence:.2f}",
        fontsize=14,
        fontweight="bold",
    )
    # Plot 2: Entropy metrics comparison using seaborn.
    colors_metrics = ["steelblue", "coral", "purple"]
    sns.barplot(
        data=metrics_df,
        x="Metric",
        y="Value",
        hue="Metric",
        palette=colors_metrics,
        alpha=0.7,
        edgecolor="black",
        legend=False,
        ax=ax2,
    )
    ax2.set_ylabel("Information [bits]", fontsize=12)
    ax2.set_xlabel("")
    ax2.set_title("Entropy Metrics", fontsize=14, fontweight="bold")
    ax2.set_ylim([0, 2.2])
    # Add value labels on bars.
    for i, (metric, value) in enumerate(
        zip(metrics_df["Metric"], metrics_df["Value"])
    ):
        ax2.text(
            i,
            value + 0.05,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Plot 3: Scatter plot of sampled realizations using seaborn.
    sns.scatterplot(
        data=samples_df,
        x="Y",
        y="X",
        alpha=0.6,
        s=50,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
        ax=ax3,
    )
    ax3.set_xlabel("Variable Y", fontsize=12)
    ax3.set_ylabel("Variable X", fontsize=12)
    ax3.set_title(
        f"Sampled Realizations (n={n_samples})",
        fontsize=14,
        fontweight="bold",
    )
    ax3.set_xlim([-0.3, 1.3])
    ax3.set_ylim([-0.3, 1.3])
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.grid(True, alpha=0.3)
    # Plot 4: Comments and explanation text.
    ax4.axis("off")
    ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Wrap interpretation text to fixed width to ensure consistent dimensions.
    wrapped_interpretation = textwrap.fill(interpretation, width=40)
    # Add explanation text.
    text_content = (
        f"Entropy Values:\n"
        f"  • H(X) = {h_x:.4f} bits\n"
        f"  • H(Y) = {h_y:.4f} bits\n"
        f"  • H(X,Y) = {h_xy:.4f} bits\n\n"
        f"Interpretation:\n"
        f"  {wrapped_interpretation}\n\n"
        f"Verification:\n"
        f"  H(X,Y) + I(X;Y) =\n"
        f"      H(X) + H(Y)\n"
        f"  {h_xy:.4f} + {mi:.4f} =\n"
        f"      {h_x:.4f} + {h_y:.4f}\n"
        f"  {h_xy + mi:.4f} = {h_x + h_y:.4f}"
    )
    ax4.text(
        0.05,
        0.95,
        text_content,
        transform=ax4.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
        wrap=True,
    )
    # Use subplots_adjust with fixed parameters instead of tight_layout.
    # This ensures consistent spacing and dimensions across all frames.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.25
    )
    plt.show()


def plot_conditional_entropy_interactive(
    *, dependence: float = 0.5, figsize: Optional[tuple] = None
) -> None:
    """
    Interactive visualization of conditional entropy with dependence control.

    :param dependence: Dependence strength between variables (0=independent,
        1=perfectly correlated)
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Create joint distribution with specified dependence.
    joint_prob = create_correlated_joint_distribution(correlation=dependence)
    # Convert to DataFrame for better visualization.
    joint_df = pd.DataFrame(
        joint_prob, index=["X=0", "X=1"], columns=["Y=0", "Y=1"]
    )
    # Calculate marginals.
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    # Calculate conditional distributions P(Y|X).
    p_y_given_x0 = (
        joint_prob[0, :] / p_x[0] if p_x[0] > 0 else np.array([0.5, 0.5])
    )
    p_y_given_x1 = (
        joint_prob[1, :] / p_x[1] if p_x[1] > 0 else np.array([0.5, 0.5])
    )
    # Calculate entropy metrics.
    h_x = calculate_entropy(p_x)
    h_y = calculate_entropy(p_y)
    h_xy = calculate_joint_entropy(joint_prob)
    h_y_given_x = calculate_conditional_entropy(joint_prob)
    h_x_given_y = calculate_conditional_entropy(joint_prob.T)
    # Determine interpretation message based on dependence.
    if dependence < 0.1:
        interpretation = (
            "Independence: H(Y|X) = H(Y)\n"
            "Knowing X provides no information about Y.\n"
            "Conditional distributions P(Y|X=0) and P(Y|X=1) are identical."
        )
    elif dependence > 0.9:
        interpretation = (
            "Perfect dependence: H(Y|X) = 0\n"
            "Knowing X completely determines Y.\n"
            "Each conditional distribution is deterministic (spike)."
        )
    else:
        reduction = h_y - h_y_given_x
        percentage = (reduction / h_y * 100) if h_y > 0 else 0
        interpretation = (
            f"Partial dependence: H(Y|X) < H(Y)\n"
            f"Knowing X reduces uncertainty about Y by {reduction:.4f} bits ({percentage:.1f}%).\n"
            f"Conditional distributions differ but retain uncertainty."
        )
    # Create visualization with 4 subplots in a single row.
    # Use gridspec_kw to set fixed width ratios for consistent layout.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 1, 1.2]}
    )
    # Plot 1: Joint distribution heatmap using seaborn.
    sns.heatmap(
        joint_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0,
        vmax=0.5,
        cbar=True,
        ax=ax1,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )
    ax1.set_xlabel("Variable Y", fontsize=12)
    ax1.set_ylabel("Variable X", fontsize=12)
    ax1.set_title(
        f"Joint Distribution P(X,Y)\nDependence = {dependence:.2f}",
        fontsize=14,
        fontweight="bold",
    )
    # Plot 2: Conditional distributions P(Y|X=0) and P(Y|X=1).
    # Create DataFrame for conditional distributions.
    cond_df = pd.DataFrame(
        {
            "P(Y|X=0)": p_y_given_x0,
            "P(Y|X=1)": p_y_given_x1,
        },
        index=["Y=0", "Y=1"],
    )
    # Plot grouped bar chart.
    x_pos = np.arange(len(cond_df.index))
    width = 0.35
    ax2.bar(
        x_pos - width / 2,
        cond_df["P(Y|X=0)"],
        width,
        label="P(Y|X=0)",
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    ax2.bar(
        x_pos + width / 2,
        cond_df["P(Y|X=1)"],
        width,
        label="P(Y|X=1)",
        alpha=0.7,
        color="coral",
        edgecolor="black",
    )
    ax2.set_xlabel("Variable Y", fontsize=12)
    ax2.set_ylabel("Conditional Probability", fontsize=12)
    ax2.set_title(
        "Conditional Distributions P(Y|X)", fontsize=14, fontweight="bold"
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cond_df.index)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis="y")
    # Add value labels on bars.
    for i, (val0, val1) in enumerate(
        zip(cond_df["P(Y|X=0)"], cond_df["P(Y|X=1)"])
    ):
        ax2.text(
            i - width / 2,
            val0 + 0.02,
            f"{val0:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax2.text(
            i + width / 2,
            val1 + 0.02,
            f"{val1:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Plot 3: Entropy metrics comparison using seaborn.
    metrics_df = pd.DataFrame(
        {
            "Metric": ["H(X)", "H(Y)", "H(Y|X)", "H(X,Y)"],
            "Value": [h_x, h_y, h_y_given_x, h_xy],
        }
    )
    colors_metrics = ["steelblue", "coral", "green", "purple"]
    sns.barplot(
        data=metrics_df,
        x="Metric",
        y="Value",
        hue="Metric",
        palette=colors_metrics,
        alpha=0.7,
        edgecolor="black",
        legend=False,
        ax=ax3,
    )
    ax3.set_ylabel("Information [bits]", fontsize=12)
    ax3.set_xlabel("")
    ax3.set_title("Entropy Metrics", fontsize=14, fontweight="bold")
    ax3.set_ylim([0, 2.2])
    # Add value labels on bars.
    for i, (metric, value) in enumerate(
        zip(metrics_df["Metric"], metrics_df["Value"])
    ):
        ax3.text(
            i,
            value + 0.05,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Highlight H(Y|X) as the main focus.
    ax3.get_children()[2].set_linewidth(3)
    ax3.get_children()[2].set_edgecolor("darkgreen")
    # Plot 4: Comments text panel.
    ax4.axis("off")
    ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Wrap interpretation text to fixed width to ensure consistent dimensions.
    wrapped_interpretation = textwrap.fill(interpretation, width=40)
    # Add explanation text.
    text_content = (
        f"Conditional Entropy:\n"
        f"  H(Y|X) = {h_y_given_x:.4f} bits\n\n"
        f"Interpretation:\n"
        f"  {wrapped_interpretation}\n\n"
        f"Chain Rule Verification:\n"
        f"  H(X,Y) = H(X) + H(Y|X)\n"
        f"  {h_xy:.4f} = {h_x:.4f} + {h_y_given_x:.4f}\n"
        f"  {h_xy:.4f} = {h_x + h_y_given_x:.4f}\n\n"
        f"Symmetry:\n"
        f"  H(X|Y) = {h_x_given_y:.4f} bits\n"
        f"  H(X,Y) = H(Y) + H(X|Y)\n"
        f"  {h_xy:.4f} = {h_y:.4f} + {h_x_given_y:.4f}"
    )
    ax4.text(
        0.1,
        0.9,
        text_content,
        transform=ax4.transAxes,
        fontsize=11,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=1", facecolor="wheat", alpha=0.3),
    )
    # Use subplots_adjust with fixed parameters instead of tight_layout.
    # This ensures consistent spacing and dimensions across all frames.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.25
    )
    plt.show()


# #############################################################################
# KL divergence and cross-entropy
# #############################################################################


def calculate_kl_divergence(
    p: Union[List[float], np.ndarray], q: Union[List[float], np.ndarray]
) -> float:
    r"""
    Calculate KL divergence D_KL(P || Q).

    Kullback-Leibler (KL) Divergence $D_{KL}(P \| Q)$ measures how one
    distribution differs from another:

    $$D_{KL}(P \| Q) = \sum_x P(x) \log_2 \frac{P(x)}{Q(x)}$$

    :param p: True distribution P
    :param q: Approximating distribution Q
    :return: KL divergence in bits
    """
    p = np.array(p)
    q = np.array(q)
    # Avoid log(0) by filtering.
    mask = (p > 0) & (q > 0)
    kl = np.sum(p[mask] * np.log2(p[mask] / q[mask]))
    return kl


def calculate_cross_entropy(
    p: Union[List[float], np.ndarray], q: Union[List[float], np.ndarray]
) -> float:
    r"""
    Calculate cross-entropy H(P, Q).

    Cross-Entropy $H(P, Q)$ measures the average number of bits needed to
    encode data from $P$ using code optimized for $Q$:

    $$H(P, Q) = -\sum_x P(x) \log_2 Q(x)$$

    Relationship: $H(P, Q) = H(P) + D_{KL}(P \| Q)$

    :param p: True distribution P
    :param q: Model distribution Q
    :return: Cross-entropy in bits
    """
    p = np.array(p)
    q = np.array(q)
    # Avoid log(0).
    mask = (p > 0) & (q > 0)
    ce = -np.sum(p[mask] * np.log2(q[mask]))
    return ce


# #############################################################################
# Visualization functions
# #############################################################################


def plot_distribution_with_stats(
    *,
    values: np.ndarray,
    probabilities: np.ndarray,
    title: str,
    ax: Optional[Any] = None,
    figsize: tuple = (6, 4),
    save_fig: Optional[str] = None,
) -> None:
    """
    Plot a probability distribution with mean, variance, and entropy statistics.

    :param values: Array of outcome values
    :param probabilities: Array of probabilities for each outcome
    :param title: Title for the plot
    :param ax: Matplotlib axis to plot on (creates new if None)
    :param figsize: Figure size as (width, height) tuple, default (10, 3)
    :param save_fig: Optional filename to save the figure (e.g., 'plot.png')
    """
    # Calculate statistics.
    mean = np.sum(values * probabilities)
    variance = np.sum(probabilities * (values - mean) ** 2)
    entropy = calculate_entropy(probabilities)
    # Create axis if not provided.
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Plot probability distribution.
    ax.bar(
        values,
        probabilities,
        alpha=0.7,
        edgecolor="black",
        color="steelblue",
    )
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    # Add statistics text box.
    stats_text = (
        f"Mean: {mean:.2f}\n"
        f"Variance: {variance:.2f}\n"
        f"Entropy: {entropy:.4f} bits"
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    # Save figure if requested.
    if save_fig is not None and fig is not None:
        fig.savefig(save_fig, dpi=150, bbox_inches="tight")
        _LOG.info("Saved figure to: %s", save_fig)
    # Show plot if not using existing axis.
    if fig is not None:
        plt.tight_layout()
        plt.show()


def plot_binary_entropy_interactive(
    *, p: float = 0.5, n: int = 100, figsize: Optional[tuple] = None
) -> None:
    """
    Plot binary entropy function with current value highlighted.

    :param p: Probability of outcome 1 (slider controlled)
    :param n: Number of samples to draw over time (default 100)
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Create array of probabilities.
    p_values = np.linspace(0.001, 0.999, 1000)
    entropy_values = [binary_entropy(p_val) for p_val in p_values]
    # Generate n samples from the binary distribution.
    np.random.seed(42)  # For reproducibility
    samples = np.random.binomial(1, p, n)
    # Calculate information content.
    if p > 0 and p < 1:
        info_0 = -np.log2(1 - p)
        info_1 = -np.log2(p)
    else:
        info_0 = 0.0
        info_1 = 0.0
    # Create figure with 4 subplots in a single row.
    # Use gridspec_kw to set fixed width ratios for consistent layout.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 1, 1.2]}
    )
    # Plot 1: Binary entropy function.
    ax1.plot(p_values, entropy_values, "b-", linewidth=2, label="H(p)")
    ax1.axvline(
        p, color="red", linestyle="--", linewidth=2, label=f"p = {p:.2f}"
    )
    ax1.axhline(
        binary_entropy(p), color="red", linestyle="--", linewidth=1, alpha=0.5
    )
    ax1.scatter([p], [binary_entropy(p)], color="red", s=100, zorder=5)
    ax1.set_xlabel("Probability p", fontsize=12)
    ax1.set_ylabel("Entropy H(p) [bits]", fontsize=12)
    ax1.set_title("Binary Entropy Function", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim([-0.05, 1.1])
    # Plot 2: Probability distribution.
    outcomes = ["Outcome 0", "Outcome 1"]
    probs = [1 - p, p]
    colors = ["skyblue", "coral"]
    ax2.bar(outcomes, probs, color=colors, alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_title(
        f"Probability Distribution\nEntropy = {binary_entropy(p):.4f} bits",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis="y")
    # Add probability values on bars.
    for i, (outcome, prob) in enumerate(zip(outcomes, probs)):
        ax2.text(
            i,
            prob + 0.02,
            f"{prob:.2f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    # Plot 3: Samples over time.
    time_indices = np.arange(n)
    # Create color array for samples.
    sample_colors = ["skyblue" if s == 0 else "coral" for s in samples]
    ax3.scatter(
        time_indices,
        samples,
        c=sample_colors,
        alpha=0.6,
        s=50,
        edgecolor="black",
        linewidth=0.5,
    )
    # Add horizontal lines at 0 and 1.
    ax3.axhline(
        0,
        color="skyblue",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Outcome 0",
    )
    ax3.axhline(
        1,
        color="coral",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Outcome 1",
    )
    ax3.set_xlabel("Time (sample index)", fontsize=12)
    ax3.set_ylabel("Outcome", fontsize=12)
    ax3.set_title(
        f"Samples Over Time (n={n})\nObserved: {samples.sum()}/{n} ones ({samples.sum() / n:.2%})",
        fontsize=14,
        fontweight="bold",
    )
    ax3.set_ylim([-0.3, 1.3])
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["0", "1"])
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(fontsize=11)
    # Plot 4: Comments and explanation text.
    ax4.axis("off")
    ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Wrap text content to ensure consistent dimensions.
    text_content = (
        f"Information Content:\n"
        f"  • Outcome 0: {info_0:.4f} bits\n"
        f"  • Outcome 1: {info_1:.4f} bits\n\n"
        f"Entropy:\n"
        f"  • Expected information:\n"
        f"    {binary_entropy(p):.4f} bits\n\n"
        f"Samples:\n"
        f"  • Drawn: {n}\n"
        f"  • Outcome 1 appeared:\n"
        f"    {samples.sum()} times\n"
        f"    ({samples.sum() / n:.2%})"
    )
    ax4.text(
        0.05,
        0.95,
        text_content,
        transform=ax4.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
        wrap=True,
    )
    # Use subplots_adjust with fixed parameters instead of tight_layout.
    # This ensures consistent spacing and dimensions across all frames.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.25
    )
    plt.show()


def visualize_information_decomposition(joint_prob: np.ndarray) -> None:
    """
    Visualize the decomposition of joint entropy into components.

    :param joint_prob: 2D array of joint probabilities
    """
    joint_prob = np.array(joint_prob)
    # Calculate all components.
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    h_x = calculate_entropy(p_x)
    h_y = calculate_entropy(p_y)
    h_xy = calculate_joint_entropy(joint_prob)
    h_y_given_x = calculate_conditional_entropy(joint_prob)
    h_x_given_y = calculate_conditional_entropy(joint_prob.T)
    mi = calculate_mutual_information(joint_prob)
    # Create Venn diagram visualization.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Plot 1: Information measures.
    measures = ["H(X)", "H(Y)", "H(X,Y)", "H(Y|X)", "H(X|Y)", "I(X;Y)"]
    values = [h_x, h_y, h_xy, h_y_given_x, h_x_given_y, mi]
    colors_bars = [
        "steelblue",
        "coral",
        "purple",
        "lightblue",
        "lightsalmon",
        "green",
    ]
    bars = ax1.bar(
        measures, values, color=colors_bars, alpha=0.7, edgecolor="black"
    )
    ax1.set_ylabel("Information [bits]", fontsize=12)
    ax1.set_title("Information Measures", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", rotation=45)
    # Add values on bars.
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Plot 2: Relationship diagram.
    ax2.text(
        0.5,
        0.9,
        "Information Relationships",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )
    ax2.text(
        0.5,
        0.75,
        "H(X, Y) = H(X) + H(Y|X)",
        ha="center",
        fontsize=12,
    )
    ax2.text(
        0.5,
        0.68,
        f"{h_xy:.3f} = {h_x:.3f} + {h_y_given_x:.3f}",
        ha="center",
        fontsize=11,
        color="blue",
    )
    ax2.text(
        0.5,
        0.58,
        "I(X;Y) = H(X) + H(Y) - H(X,Y)",
        ha="center",
        fontsize=12,
    )
    ax2.text(
        0.5,
        0.51,
        f"{mi:.3f} = {h_x:.3f} + {h_y:.3f} - {h_xy:.3f}",
        ha="center",
        fontsize=11,
        color="green",
    )
    ax2.text(
        0.5,
        0.41,
        "I(X;Y) = H(Y) - H(Y|X)",
        ha="center",
        fontsize=12,
    )
    ax2.text(
        0.5,
        0.34,
        f"{mi:.3f} = {h_y:.3f} - {h_y_given_x:.3f}",
        ha="center",
        fontsize=11,
        color="green",
    )
    ax2.text(0.5, 0.24, "Comments:", ha="center", fontsize=12, fontweight="bold")
    ax2.text(
        0.5,
        0.17,
        f"Knowing X reduces uncertainty about Y by {mi:.3f} bits",
        ha="center",
        fontsize=11,
    )
    ax2.text(
        0.5,
        0.10,
        f"({(mi / h_y) * 100:.1f}% of total uncertainty in Y)",
        ha="center",
        fontsize=11,
        style="italic",
    )
    ax2.axis("off")
    plt.tight_layout()
    plt.show()


def plot_mutual_info_interactive(
    *, correlation: float = 0.5, figsize: Optional[tuple] = None
) -> None:
    """
    Interactive plot showing how correlation affects mutual information.

    :param correlation: Correlation strength between variables
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    joint_prob = create_correlated_joint_distribution(correlation=correlation)
    # Calculate metrics.
    mi = calculate_mutual_information(joint_prob)
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    h_x = calculate_entropy(p_x)
    h_y = calculate_entropy(p_y)
    h_xy = calculate_joint_entropy(joint_prob)
    h_y_given_x = calculate_conditional_entropy(joint_prob)
    # Determine interpretation message based on correlation.
    if correlation < 0.1:
        interpretation = (
            "Independence: I(X;Y) ≈ 0\n"
            "Variables are nearly independent.\n"
            "Knowing X provides no information about Y."
        )
    elif correlation > 0.9:
        interpretation = (
            "Strong correlation: I(X;Y) ≈ H(Y)\n"
            "Variables are nearly perfectly correlated.\n"
            "Knowing X almost completely determines Y."
        )
    else:
        percentage = (mi / h_y * 100) if h_y > 0 else 0
        interpretation = (
            f"Partial correlation:\n"
            f"Knowing X reduces uncertainty about Y\n"
            f"by {mi:.4f} bits ({percentage:.1f}% of H(Y))."
        )
    # Create visualization with 4 subplots in a single row.
    # Use gridspec_kw to set fixed width ratios for consistent layout.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 1, 1.2]}
    )
    # Plot 1: Joint distribution heatmap.
    im = ax1.imshow(joint_prob, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Y=0", "Y=1"])
    ax1.set_yticklabels(["X=0", "X=1"])
    ax1.set_xlabel("Variable Y", fontsize=12)
    ax1.set_ylabel("Variable X", fontsize=12)
    ax1.set_title(
        f"Joint Distribution P(X,Y)\nCorrelation = {correlation:.2f}",
        fontsize=14,
        fontweight="bold",
    )
    # Add text annotations.
    for i in range(2):
        for j in range(2):
            text = ax1.text(
                j,
                i,
                f"{joint_prob[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold",
            )
    plt.colorbar(im, ax=ax1)
    # Plot 2: Information metrics vs correlation.
    correlations = np.linspace(0, 1, 50)
    mis = []
    for corr in correlations:
        jp = create_correlated_joint_distribution(correlation=corr)
        mis.append(calculate_mutual_information(jp))
    ax2.plot(correlations, mis, "b-", linewidth=2, label="I(X;Y)")
    ax2.axvline(
        correlation,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Current: {correlation:.2f}",
    )
    ax2.scatter([correlation], [mi], color="red", s=100, zorder=5)
    ax2.set_xlabel("Correlation", fontsize=12)
    ax2.set_ylabel("Mutual Information [bits]", fontsize=12)
    ax2.set_title(
        "Mutual Information vs Correlation", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 1.1])
    # Plot 3: Entropy metrics comparison using seaborn.
    metrics_df = pd.DataFrame(
        {
            "Metric": ["H(X)", "H(Y)", "I(X;Y)", "H(X,Y)"],
            "Value": [h_x, h_y, mi, h_xy],
        }
    )
    colors_metrics = ["steelblue", "coral", "green", "purple"]
    sns.barplot(
        data=metrics_df,
        x="Metric",
        y="Value",
        hue="Metric",
        palette=colors_metrics,
        alpha=0.7,
        edgecolor="black",
        legend=False,
        ax=ax3,
    )
    ax3.set_ylabel("Information [bits]", fontsize=12)
    ax3.set_xlabel("")
    ax3.set_title("Entropy Metrics", fontsize=14, fontweight="bold")
    ax3.set_ylim([0, 2.2])
    # Add value labels on bars.
    for i, (metric, value) in enumerate(
        zip(metrics_df["Metric"], metrics_df["Value"])
    ):
        ax3.text(
            i,
            value + 0.05,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Highlight I(X;Y) bar.
    ax3.get_children()[2].set_linewidth(3)
    ax3.get_children()[2].set_edgecolor("darkgreen")
    # Plot 4: Comments and explanation text.
    ax4.axis("off")
    ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Wrap interpretation text to fixed width to ensure consistent dimensions.
    wrapped_interpretation = textwrap.fill(interpretation, width=40)
    # Add explanation text.
    text_content = (
        f"Mutual Information:\n"
        f"  I(X;Y) = {mi:.4f} bits\n\n"
        f"Interpretation:\n"
        f"  {wrapped_interpretation}\n\n"
        f"Key Relationships:\n"
        f"  I(X;Y) = H(X) + H(Y) - H(X,Y)\n"
        f"  {mi:.4f} = {h_x:.4f} + {h_y:.4f}\n"
        f"           - {h_xy:.4f}\n"
        f"  {mi:.4f} = {h_x + h_y - h_xy:.4f}\n\n"
        f"  I(X;Y) = H(Y) - H(Y|X)\n"
        f"  {mi:.4f} = {h_y:.4f} - {h_y_given_x:.4f}"
    )
    ax4.text(
        0.05,
        0.95,
        text_content,
        transform=ax4.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
        wrap=True,
    )
    # Use subplots_adjust with fixed parameters instead of tight_layout.
    # This ensures consistent spacing and dimensions across all frames.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.25
    )
    plt.show()


def plot_mutual_information_venn_interactive(
    *,
    dependence: float = 0.5,
    scenario: str = "Binary",
    figsize: Optional[tuple] = None,
) -> None:
    """
    Enhanced interactive visualization of mutual information with Venn-style decomposition.

    Shows the relationship between entropy components and mutual information
    with clear visual decomposition.

    :param dependence: Dependence strength between variables (0=independent,
        1=perfectly correlated)
    :param scenario: "Binary" for 2x2 distribution or "Weather" for 3x3
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Create joint distribution based on scenario.
    if scenario == "Binary":
        # Binary variables X, Y in {0, 1}.
        joint_prob = create_correlated_joint_distribution(correlation=dependence)
        x_labels = ["X=0", "X=1"]
        y_labels = ["Y=0", "Y=1"]
        scenario_name = "Binary Variables"
    else:
        # Weather/Activity scenario (3x3).
        # Create a 3x3 joint distribution.
        # Base independent distribution.
        p_x_base = np.array([0.4, 0.3, 0.3])  # Sunny, Rainy, Cloudy.
        p_y_base = np.array([0.35, 0.35, 0.3])  # Park, Cinema, Home.
        # Create correlated version.
        # Strong diagonal for dependence (Sunny->Park, Rainy->Cinema, Cloudy->Home).
        joint_prob = np.outer(p_x_base, p_y_base)
        # Add correlation by shifting probability to diagonal.
        diagonal_boost = dependence * 0.2
        for i in range(3):
            # Boost diagonal elements.
            joint_prob[i, i] += diagonal_boost
        # Renormalize.
        joint_prob = joint_prob / joint_prob.sum()
        x_labels = ["Sunny", "Rainy", "Cloudy"]
        y_labels = ["Park", "Cinema", "Home"]
        scenario_name = "Weather & Activity"
    # Convert to DataFrame for visualization.
    joint_df = pd.DataFrame(joint_prob, index=x_labels, columns=y_labels)
    # Calculate marginals.
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    # Calculate all entropy metrics.
    h_x = calculate_entropy(p_x)
    h_y = calculate_entropy(p_y)
    h_xy = calculate_joint_entropy(joint_prob)
    h_y_given_x = calculate_conditional_entropy(joint_prob)
    h_x_given_y = calculate_conditional_entropy(joint_prob.T)
    mi = calculate_mutual_information(joint_prob)
    # Determine interpretation based on dependence.
    if dependence < 0.1:
        interpretation = (
            "Independence:\n"
            f"  I(X;Y) = {mi:.4f} bits (nearly 0)\n"
            f"  H(X,Y) = {h_xy:.4f} ≈ H(X) + H(Y)\n"
            f"  = {h_x:.4f} + {h_y:.4f}\n\n"
            "Knowing X provides no\n"
            "information about Y.\n\n"
            "The Venn circles have\n"
            "minimal overlap."
        )
    elif dependence > 0.9:
        interpretation = (
            "Strong Dependence:\n"
            f"  I(X;Y) = {mi:.4f} bits\n"
            f"  H(Y|X) = {h_y_given_x:.4f} (nearly 0)\n\n"
            "Knowing X almost completely\n"
            "determines Y.\n\n"
            f"Mutual information is {(mi / h_y * 100):.1f}%\n"
            f"of H(Y).\n\n"
            "The Venn circles have\n"
            "maximum overlap."
        )
    else:
        reduction = h_y - h_y_given_x
        percentage = (reduction / h_y * 100) if h_y > 0 else 0
        interpretation = (
            "Partial Dependence:\n"
            f"  I(X;Y) = {mi:.4f} bits\n\n"
            f"Knowing X reduces uncertainty\n"
            f"about Y by {reduction:.4f} bits\n"
            f"({percentage:.1f}% of H(Y)).\n\n"
            "Key relationships:\n"
            f"  I(X;Y) = H(X) - H(X|Y)\n"
            f"  {mi:.3f} = {h_x:.3f} - {h_x_given_y:.3f}\n\n"
            f"  I(X;Y) = H(Y) - H(Y|X)\n"
            f"  {mi:.3f} = {h_y:.3f} - {h_y_given_x:.3f}"
        )
    # Create visualization with 4 subplots in a single row.
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes
    # Plot 1: Joint distribution heatmap.
    sns.heatmap(
        joint_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0,
        vmax=joint_prob.max() * 1.2,
        cbar=True,
        ax=ax1,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    ax1.set_xlabel("Variable Y", fontsize=12)
    ax1.set_ylabel("Variable X", fontsize=12)
    ax1.set_title(
        f"Joint Distribution P(X,Y)\n{scenario_name}\nDependence = {dependence:.2f}",
        fontsize=13,
        fontweight="bold",
    )
    # Plot 2: Entropy decomposition using stacked visualization.
    # Show H(X,Y) decomposed into I(X;Y) + H(X|Y) + H(Y|X) - I(X;Y).
    # Simpler: show H(X), H(Y), H(X,Y), with I(X;Y) as overlap.
    categories = ["H(X)", "H(Y)", "H(X,Y)", "I(X;Y)"]
    values = [h_x, h_y, h_xy, mi]
    colors_bars = ["steelblue", "coral", "purple", "green"]
    bars = ax2.bar(
        categories,
        values,
        color=colors_bars,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    ax2.set_ylabel("Information [bits]", fontsize=12)
    ax2.set_xlabel("")
    ax2.set_title(
        "Entropy Components\n(Mutual Info in Green)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3, axis="y")
    max_entropy = max(h_x, h_y, h_xy)
    ax2.set_ylim([0, max_entropy * 1.2])
    # Add value labels on bars.
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max_entropy * 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Highlight I(X;Y) bar.
    bars[3].set_linewidth(3)
    bars[3].set_edgecolor("darkgreen")
    # Plot 3: Venn-style visualization showing overlapping information.
    ax3.set_xlim([0, 10])
    ax3.set_ylim([0, 10])
    ax3.axis("off")
    ax3.set_title(
        "Information Decomposition\n(Venn Diagram)",
        fontsize=13,
        fontweight="bold",
    )
    # Draw two overlapping circles representing H(X) and H(Y).
    # Circle sizes proportional to entropies.
    # Overlap represents I(X;Y).
    from matplotlib.patches import Circle

    # Scale circles based on entropy.
    max_h = max(h_x, h_y, 1)
    radius_x = 2.0 * np.sqrt(h_x / max_h)
    radius_y = 2.0 * np.sqrt(h_y / max_h)
    # Position circles with overlap based on MI.
    center_x = [3.5, 6.5]
    center_y = [5, 5]
    # Adjust center positions to show overlap.
    overlap_factor = mi / max(h_x, h_y, 0.01)
    separation = 3.0 * (1 - overlap_factor * 0.8)
    center_x = [5 - separation / 2, 5 + separation / 2]
    # Draw circles.
    circle_x = Circle(
        (center_x[0], center_y[0]),
        radius_x,
        facecolor="steelblue",
        alpha=0.4,
        linewidth=2,
        edgecolor="steelblue",
        label=f"H(X)={h_x:.3f}",
    )
    circle_y = Circle(
        (center_x[1], center_y[1]),
        radius_y,
        facecolor="coral",
        alpha=0.4,
        linewidth=2,
        edgecolor="coral",
        label=f"H(Y)={h_y:.3f}",
    )
    ax3.add_patch(circle_x)
    ax3.add_patch(circle_y)
    # Add labels.
    ax3.text(
        center_x[0] - 0.8,
        7.5,
        "H(X)",
        fontsize=14,
        fontweight="bold",
        color="steelblue",
    )
    ax3.text(
        center_x[1] + 0.8,
        7.5,
        "H(Y)",
        fontsize=14,
        fontweight="bold",
        color="coral",
    )
    # Label the overlap region.
    ax3.text(
        5,
        5,
        f"I(X;Y)\n{mi:.3f}",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    # Add conditional entropy labels.
    ax3.text(
        center_x[0] - 1.2,
        5,
        f"H(X|Y)\n{h_x_given_y:.2f}",
        fontsize=9,
        ha="center",
        va="center",
        color="navy",
    )
    ax3.text(
        center_x[1] + 1.2,
        5,
        f"H(Y|X)\n{h_y_given_x:.2f}",
        fontsize=9,
        ha="center",
        va="center",
        color="darkred",
    )
    # Add relationship equations.
    ax3.text(
        5,
        2.0,
        f"H(X,Y) = {h_xy:.3f}",
        fontsize=10,
        ha="center",
        fontweight="bold",
        color="purple",
    )
    ax3.text(
        5,
        1.2,
        "= I(X;Y) + H(X|Y) + H(Y|X)",
        fontsize=8,
        ha="center",
        style="italic",
    )
    # Plot 4: Comments text panel.
    ax4.axis("off")
    ax4.set_title("Comments", fontsize=13, fontweight="bold", pad=20)
    # Add explanation text.
    text_content = (
        f"{interpretation}\n\n"
        f"Fundamental Relations:\n"
        f"  H(X,Y) = H(X) + H(Y|X)\n"
        f"  {h_xy:.3f} = {h_x:.3f} + {h_y_given_x:.3f}\n\n"
        f"  I(X;Y) = H(X) + H(Y) - H(X,Y)\n"
        f"  {mi:.3f} = {h_x:.3f} + {h_y:.3f} - {h_xy:.3f}\n\n"
        f"Symmetry:\n"
        f"  I(X;Y) = I(Y;X) = {mi:.3f}"
    )
    ax4.text(
        0.1,
        0.95,
        text_content,
        transform=ax4.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=1", facecolor="wheat", alpha=0.3),
    )
    plt.tight_layout()
    plt.show()


def plot_cross_entropy_interactive(
    *, p1: float = 0.7, q1: float = 0.5, figsize: Optional[tuple] = None
) -> None:
    """
    Interactive visualization of cross-entropy between two binary distributions.

    :param p1: Probability for true distribution P
    :param q1: Probability for model distribution Q
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Create distributions.
    p = np.array([1 - p1, p1])
    q = np.array([1 - q1, q1])
    # Calculate metrics.
    h_p = calculate_entropy(p)
    h_pq = calculate_cross_entropy(p, q)
    kl_pq = calculate_kl_divergence(p, q)
    # Determine interpretation based on cross-entropy value.
    entropy_ratio = h_pq / h_p if h_p > 0 else 1.0
    extra_bits = h_pq - h_p
    if abs(extra_bits) < 0.01:
        interpretation = (
            "Optimal encoding!\n"
            "Model Q matches true distribution P.\n"
            "H(P,Q) = H(P): No extra bits needed."
        )
        quality = "Perfect"
    elif extra_bits < 0.1:
        interpretation = (
            "Near-optimal encoding.\n"
            "Model Q is very close to P.\n"
            f"Only {extra_bits:.3f} extra bits per symbol."
        )
        quality = "Excellent"
    elif extra_bits < 0.5:
        interpretation = (
            "Moderate encoding cost.\n"
            "Model Q differs from P.\n"
            f"Requires {extra_bits:.3f} extra bits per symbol."
        )
        quality = "Good"
    else:
        interpretation = (
            "High encoding cost!\n"
            "Model Q poorly matches P.\n"
            f"Requires {extra_bits:.3f} extra bits per symbol."
        )
        quality = "Poor"
    # Create visualization with 4 subplots in a single row.
    # Use gridspec_kw to set fixed width ratios for consistent layout.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 1, 1.2]}
    )
    # Plot 1: Distributions comparison.
    x = np.arange(2)
    width = 0.35
    bars1 = ax1.bar(
        x - width / 2,
        p,
        width,
        label="P (True)",
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    bars2 = ax1.bar(
        x + width / 2,
        q,
        width,
        label="Q (Model)",
        alpha=0.7,
        color="coral",
        edgecolor="black",
    )
    ax1.set_xlabel("Outcome", fontsize=12)
    ax1.set_ylabel("Probability", fontsize=12)
    ax1.set_title(
        f"Distribution Comparison\nP: [{1 - p1:.2f}, {p1:.2f}] vs Q: [{1 - q1:.2f}, {q1:.2f}]",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(["0", "1"])
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, axis="y")
    # Add value labels.
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    # Plot 2: Cross-entropy heatmap.
    p_range = np.linspace(0.05, 0.95, 30)
    q_range = np.linspace(0.05, 0.95, 30)
    ce_matrix = np.zeros((len(p_range), len(q_range)))
    for i, p_val in enumerate(p_range):
        for j, q_val in enumerate(q_range):
            p_dist = np.array([1 - p_val, p_val])
            q_dist = np.array([1 - q_val, q_val])
            ce_matrix[i, j] = calculate_cross_entropy(p_dist, q_dist)
    im = ax2.contourf(q_range, p_range, ce_matrix, levels=20, cmap="YlOrRd")
    ax2.scatter(
        [q1],
        [p1],
        color="red",
        s=200,
        marker="*",
        edgecolor="black",
        linewidth=2,
        label=f"Current: H(P,Q)={h_pq:.3f}",
        zorder=5,
    )
    ax2.set_xlabel("Q (Model probability for outcome 1)", fontsize=12)
    ax2.set_ylabel("P (True probability for outcome 1)", fontsize=12)
    ax2.set_title(
        "Cross-Entropy H(P,Q) Landscape", fontsize=14, fontweight="bold"
    )
    ax2.legend(fontsize=11, loc="upper left")
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Cross-Entropy [bits]", fontsize=11)
    # Add diagonal line (where P=Q, optimal encoding).
    ax2.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5)
    ax2.text(
        0.5,
        0.55,
        "P=Q line\n(optimal)",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )
    # Plot 3: Information metrics comparison.
    metrics_df = pd.DataFrame(
        {
            "Metric": ["H(P)", "H(P,Q)", "D_KL(P||Q)"],
            "Value": [h_p, h_pq, kl_pq],
        }
    )
    colors_metrics = ["steelblue", "purple", "coral"]
    sns.barplot(
        data=metrics_df,
        x="Metric",
        y="Value",
        hue="Metric",
        palette=colors_metrics,
        alpha=0.7,
        edgecolor="black",
        legend=False,
        ax=ax3,
    )
    ax3.set_ylabel("Information [bits]", fontsize=12)
    ax3.set_xlabel("")
    ax3.set_title(
        f"Information Metrics\nEncoding Quality: {quality}",
        fontsize=14,
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3, axis="y")
    # Add value labels on bars.
    for i, (metric, value) in enumerate(
        zip(metrics_df["Metric"], metrics_df["Value"])
    ):
        ax3.text(
            i,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Highlight H(P,Q) bar as the main focus.
    if len(ax3.patches) >= 2:
        ax3.patches[1].set_linewidth(3)
        ax3.patches[1].set_edgecolor("darkviolet")
    # Plot 4: Comments text panel.
    ax4.axis("off")
    ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Wrap interpretation text to fixed width to ensure consistent dimensions.
    wrapped_interpretation = textwrap.fill(interpretation, width=40)
    # Add comprehensive explanation text.
    text_content = (
        f"Distributions:\n"
        f"  True P:     [{1 - p1:.2f}, {p1:.2f}]\n"
        f"  Model Q:    [{1 - q1:.2f}, {q1:.2f}]\n\n"
        f"Cross-Entropy:\n"
        f"  H(P,Q) = {h_pq:.4f} bits\n"
        f"  (bits needed to encode P\n"
        f"   using code for Q)\n\n"
        f"Optimal Entropy:\n"
        f"  H(P) = {h_p:.4f} bits\n"
        f"  (minimum bits needed)\n\n"
        f"Extra Cost:\n"
        f"  D_KL(P||Q) = {kl_pq:.4f} bits\n"
        f"  ({(kl_pq / h_p * 100):.1f}% inefficiency)\n\n"
        f"Interpretation:\n"
        f"  {wrapped_interpretation}\n\n"
        f"Verification:\n"
        f"  H(P,Q) = H(P) + D_KL(P||Q)\n"
        f"  {h_pq:.4f} = {h_p:.4f} + {kl_pq:.4f}\n"
        f"  {h_pq:.4f} = {h_p + kl_pq:.4f}"
    )
    ax4.text(
        0.05,
        0.95,
        text_content,
        transform=ax4.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
        wrap=True,
    )
    # Use subplots_adjust with fixed parameters instead of tight_layout.
    # This ensures consistent spacing and dimensions across all frames.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.25
    )
    plt.show()


def plot_kl_divergence_interactive(
    *, p1: float = 0.7, q1: float = 0.5, figsize: Optional[tuple] = None
) -> None:
    """
    Interactive visualization of KL divergence between two binary distributions.

    :param p1: Probability for true distribution P
    :param q1: Probability for approximating distribution Q
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Create distributions.
    p = np.array([1 - p1, p1])
    q = np.array([1 - q1, q1])
    # Calculate metrics.
    kl_pq = calculate_kl_divergence(p, q)
    kl_qp = calculate_kl_divergence(q, p)
    ce_pq = calculate_cross_entropy(p, q)
    h_p = calculate_entropy(p)
    # Determine interpretation based on KL divergence value.
    if kl_pq < 0.01:
        interpretation = (
            "Nearly identical distributions!\n"
            "Q is an excellent approximation of P.\n"
            "Almost no information is lost."
        )
        quality = "Excellent"
    elif kl_pq < 0.1:
        interpretation = (
            "Very similar distributions.\n"
            "Q is a good approximation of P.\n"
            "Minimal information loss."
        )
        quality = "Good"
    elif kl_pq < 0.5:
        interpretation = (
            "Moderate divergence.\n"
            "Q differs noticeably from P.\n"
            "Some information is lost."
        )
        quality = "Moderate"
    else:
        interpretation = (
            "Significant divergence!\n"
            "Q is a poor approximation of P.\n"
            "Substantial information loss."
        )
        quality = "Poor"
    # Create visualization with 4 subplots in a single row.
    # Use gridspec_kw to set fixed width ratios for consistent layout.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 1, 1.2]}
    )
    # Plot 1: Distributions comparison.
    x = np.arange(2)
    width = 0.35
    bars1 = ax1.bar(
        x - width / 2,
        p,
        width,
        label="P (True)",
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    bars2 = ax1.bar(
        x + width / 2,
        q,
        width,
        label="Q (Approximation)",
        alpha=0.7,
        color="coral",
        edgecolor="black",
    )
    ax1.set_xlabel("Outcome", fontsize=12)
    ax1.set_ylabel("Probability", fontsize=12)
    ax1.set_title(
        f"Distribution Comparison\nP: [{1 - p1:.2f}, {p1:.2f}] vs Q: [{1 - q1:.2f}, {q1:.2f}]",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(["0", "1"])
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, axis="y")
    # Add value labels.
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    # Plot 2: KL divergence metrics.
    metrics = ["H(P)", "H(P,Q)", "D_KL(P||Q)", "D_KL(Q||P)"]
    values = [h_p, ce_pq, kl_pq, kl_qp]
    colors_m = ["steelblue", "purple", "red", "orange"]
    bars = ax2.bar(metrics, values, color=colors_m, alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Information [bits]", fontsize=12)
    ax2.set_title(
        f"Information Metrics\nApproximation Quality: {quality}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Plot 3: KL divergence heatmap.
    p_range = np.linspace(0.05, 0.95, 30)
    q_range = np.linspace(0.05, 0.95, 30)
    kl_matrix = np.zeros((len(p_range), len(q_range)))
    for i, p_val in enumerate(p_range):
        for j, q_val in enumerate(q_range):
            p_dist = np.array([1 - p_val, p_val])
            q_dist = np.array([1 - q_val, q_val])
            kl_matrix[i, j] = calculate_kl_divergence(p_dist, q_dist)
    im = ax3.contourf(q_range, p_range, kl_matrix, levels=20, cmap="RdYlBu_r")
    ax3.scatter(
        [q1],
        [p1],
        color="red",
        s=200,
        marker="*",
        edgecolor="black",
        linewidth=2,
        label=f"Current: D_KL(P||Q)={kl_pq:.3f}",
        zorder=5,
    )
    ax3.set_xlabel("Q (Approximation probability for outcome 1)", fontsize=12)
    ax3.set_ylabel("P (True probability for outcome 1)", fontsize=12)
    ax3.set_title(
        "KL Divergence D_KL(P||Q) Heatmap", fontsize=14, fontweight="bold"
    )
    ax3.legend(fontsize=11, loc="upper left")
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label("KL Divergence [bits]", fontsize=11)
    # Add diagonal line (where P=Q, KL=0).
    ax3.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5)
    ax3.text(
        0.5,
        0.55,
        "P=Q line\n(KL=0)",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )
    # Plot 4: Comments text panel.
    ax4.axis("off")
    ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Wrap interpretation text to fixed width to ensure consistent dimensions.
    wrapped_interpretation = textwrap.fill(interpretation, width=40)
    # Add comprehensive explanation text.
    verification_text = (
        "(verified)" if abs(h_p + kl_pq - ce_pq) < 0.001 else "(error!)"
    )
    text_content = (
        f"Distributions:\n"
        f"  True P:     [{1 - p1:.2f}, {p1:.2f}]\n"
        f"  Approx Q:   [{1 - q1:.2f}, {q1:.2f}]\n\n"
        f"Entropy & Cross-Entropy:\n"
        f"  H(P) = {h_p:.4f} bits\n"
        f"  H(P,Q) = {ce_pq:.4f} bits\n\n"
        f"KL Divergence (Asymmetric!):\n"
        f"  D_KL(P||Q) = {kl_pq:.4f} bits\n"
        f"  D_KL(Q||P) = {kl_qp:.4f} bits\n\n"
        f"Interpretation:\n"
        f"  {wrapped_interpretation}\n\n"
        f"Verification:\n"
        f"  H(P,Q) = H(P) + D_KL(P||Q)\n"
        f"  {ce_pq:.4f} = {h_p:.4f} + {kl_pq:.4f}\n"
        f"  {ce_pq:.4f} = {h_p + kl_pq:.4f}\n"
        f"  {verification_text}"
    )
    ax4.text(
        0.05,
        0.95,
        text_content,
        transform=ax4.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
        wrap=True,
    )
    # Use subplots_adjust with fixed parameters instead of tight_layout.
    # This ensures consistent spacing and dimensions across all frames.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.25
    )
    plt.show()


def create_markov_chain_distribution(
    *, noise_level: float = 0.2, scenario: str = "Compression"
) -> tuple:
    """
    Create distributions for Markov chain X -> Y -> Z with controllable noise.

    :param noise_level: Noise level in Y->Z transition (0.0=clean, 1.0=maximum)
    :param scenario: Processing scenario - "Compression", "Quantization", or
        "Binary"
    :return: Tuple of (p_x, p_y_given_x, p_z_given_y, p_xy, p_yz, p_xz)
    """
    if scenario == "Compression":
        # Compression scenario: 4 states -> 2 states -> 2 states.
        # X: Original signal with 4 distinct states.
        p_x = np.array([0.4, 0.3, 0.2, 0.1])
        # P(Y|X): Compression groups (0,1) -> Y=0 and (2,3) -> Y=1.
        p_y_given_x = np.array(
            [
                [0.9, 0.1],  # X=0 -> mostly Y=0
                [0.85, 0.15],  # X=1 -> mostly Y=0
                [0.1, 0.9],  # X=2 -> mostly Y=1
                [0.05, 0.95],  # X=3 -> mostly Y=1
            ]
        )
    elif scenario == "Quantization":
        # Quantization scenario: Simulating continuous -> discrete conversion.
        # X: 4 levels representing quantized continuous values.
        p_x = np.array([0.35, 0.35, 0.15, 0.15])
        # P(Y|X): Further quantization to 2 levels.
        p_y_given_x = np.array(
            [
                [0.95, 0.05],  # X=0 -> very likely Y=0
                [0.80, 0.20],  # X=1 -> likely Y=0
                [0.20, 0.80],  # X=2 -> likely Y=1
                [0.05, 0.95],  # X=3 -> very likely Y=1
            ]
        )
    else:  # Binary
        # Binary symmetric channel scenario.
        # X: 4 message types (2 bits).
        p_x = np.array([0.25, 0.25, 0.25, 0.25])
        # P(Y|X): Binary symmetric channel with grouping.
        p_y_given_x = np.array(
            [
                [0.88, 0.12],  # X=0 -> Y=0
                [0.88, 0.12],  # X=1 -> Y=0
                [0.12, 0.88],  # X=2 -> Y=1
                [0.12, 0.88],  # X=3 -> Y=1
            ]
        )
    # P(Z|Y): Add noise controlled by noise_level parameter.
    # noise_level = 0.0: clean channel (0.95 correct transmission).
    # noise_level = 1.0: maximum noise (0.5 = random).
    clean_prob = 0.95
    noisy_prob = 0.5
    transition_prob = clean_prob - noise_level * (clean_prob - noisy_prob)
    p_z_given_y = np.array(
        [
            [transition_prob, 1 - transition_prob],  # Y=0 -> mostly Z=0
            [1 - transition_prob, transition_prob],  # Y=1 -> mostly Z=1
        ]
    )
    # Calculate joint distributions.
    # P(X,Y).
    p_xy = p_x[:, np.newaxis] * p_y_given_x
    # Marginal P(Y).
    p_y = p_xy.sum(axis=0)
    # P(Y,Z).
    p_yz = p_y[:, np.newaxis] * p_z_given_y
    # Marginal P(Z).
    p_z = p_yz.sum(axis=0)
    # Calculate P(X,Z) through marginalization over Y.
    p_xz = np.zeros((4, 2))
    for i in range(4):
        for k in range(2):
            for j in range(2):
                p_xz[i, k] += p_x[i] * p_y_given_x[i, j] * p_z_given_y[j, k]
    return p_x, p_y_given_x, p_z_given_y, p_xy, p_yz, p_xz


def plot_data_processing_inequality_interactive(
    *,
    noise_level: float = 0.2,
    scenario: str = "Compression",
    figsize: Optional[tuple] = None,
) -> None:
    """
    Interactive visualization of Data Processing Inequality.

    Shows how information degrades through processing pipeline X -> Y -> Z.
    Demonstrates that I(X;Z) <= I(X;Y) for Markov chain X -> Y -> Z.

    :param noise_level: Noise in Y->Z transition (0.0=clean, 1.0=maximum)
    :param scenario: Processing scenario - "Compression", "Quantization", or
        "Binary"
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Create distributions for the Markov chain.
    p_x, p_y_given_x, p_z_given_y, p_xy, p_yz, p_xz = (
        create_markov_chain_distribution(
            noise_level=noise_level, scenario=scenario
        )
    )
    # Calculate marginals.
    p_y = p_xy.sum(axis=0)
    p_z = p_yz.sum(axis=0)
    # Calculate all entropy metrics.
    h_x = calculate_entropy(p_x)
    h_y = calculate_entropy(p_y)
    h_z = calculate_entropy(p_z)
    # Calculate mutual informations.
    mi_xy = calculate_mutual_information(p_xy)
    mi_yz = calculate_mutual_information(p_yz)
    mi_xz = calculate_mutual_information(p_xz)
    # Calculate information loss.
    info_loss = mi_xy - mi_xz
    info_retention_pct = (mi_xz / mi_xy * 100) if mi_xy > 0 else 100.0
    # Verify inequality.
    inequality_satisfied = mi_xz <= mi_xy + 1e-6
    # Determine interpretation based on noise level.
    if noise_level < 0.1:
        interpretation = (
            "Clean Processing:\n"
            f"  Noise Level: {noise_level:.2f}\n"
            f"  Information Loss: {info_loss:.4f} bits\n"
            f"  Retention: {info_retention_pct:.1f}%\n\n"
            "Minimal information loss through\n"
            "the Y->Z transition.\n\n"
            "I(X;Z) is close to I(X;Y),\n"
            "demonstrating that clean\n"
            "processing preserves information."
        )
    elif noise_level > 0.7:
        interpretation = (
            "High Noise:\n"
            f"  Noise Level: {noise_level:.2f}\n"
            f"  Information Loss: {info_loss:.4f} bits\n"
            f"  Retention: {info_retention_pct:.1f}%\n\n"
            "Substantial information loss\n"
            "through the Y->Z transition.\n\n"
            "I(X;Z) << I(X;Y),\n"
            "showing how noise degrades\n"
            "information in a pipeline."
        )
    else:
        interpretation = (
            "Moderate Noise:\n"
            f"  Noise Level: {noise_level:.2f}\n"
            f"  Information Loss: {info_loss:.4f} bits\n"
            f"  Retention: {info_retention_pct:.1f}%\n\n"
            "Partial information loss\n"
            "through the Y->Z transition.\n\n"
            "I(X;Z) < I(X;Y),\n"
            "demonstrating the fundamental\n"
            "inequality."
        )
    # Create visualization with 4 subplots in a single row.
    # Use gridspec_kw to set fixed width ratios for consistent layout.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 1, 1.2]}
    )
    # Plot 1: Processing pipeline diagram.
    ax1.set_xlim([0, 10])
    ax1.set_ylim([0, 10])
    ax1.axis("off")
    ax1.set_title(
        f"Processing Pipeline\n{scenario} Scenario",
        fontsize=14,
        fontweight="bold",
    )
    # Draw pipeline: X -> Y -> Z.
    # X box.
    ax1.add_patch(
        plt.Rectangle(
            (0.5, 7), 2, 1.5, facecolor="steelblue", alpha=0.6, edgecolor="black"
        )
    )
    ax1.text(
        1.5, 7.75, "X", ha="center", va="center", fontsize=16, fontweight="bold"
    )
    ax1.text(
        1.5,
        6.3,
        "Original\n(4 states)",
        ha="center",
        va="top",
        fontsize=9,
    )
    # Y box.
    ax1.add_patch(
        plt.Rectangle(
            (4, 7), 2, 1.5, facecolor="coral", alpha=0.6, edgecolor="black"
        )
    )
    ax1.text(
        5, 7.75, "Y", ha="center", va="center", fontsize=16, fontweight="bold"
    )
    ax1.text(
        5,
        6.3,
        "Compressed\n(2 states)",
        ha="center",
        va="top",
        fontsize=9,
    )
    # Z box.
    ax1.add_patch(
        plt.Rectangle(
            (7.5, 7),
            2,
            1.5,
            facecolor="lightgreen",
            alpha=0.6,
            edgecolor="black",
        )
    )
    ax1.text(
        8.5, 7.75, "Z", ha="center", va="center", fontsize=16, fontweight="bold"
    )
    ax1.text(
        8.5,
        6.3,
        "Noisy\n(2 states)",
        ha="center",
        va="top",
        fontsize=9,
    )
    # Arrows.
    ax1.arrow(
        2.5,
        7.75,
        1.3,
        0,
        head_width=0.3,
        head_length=0.2,
        fc="black",
        ec="black",
    )
    ax1.arrow(
        6.0,
        7.75,
        1.3,
        0,
        head_width=0.3,
        head_length=0.2,
        fc="black",
        ec="black",
    )
    # Information metrics.
    ax1.text(
        3.15,
        8.5,
        f"I(X;Y)={mi_xy:.3f}",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )
    ax1.text(
        6.65,
        8.5,
        f"I(Y;Z)={mi_yz:.3f}",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )
    # Key result.
    ax1.text(
        5,
        4.5,
        "Data Processing Inequality:",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    ax1.text(5, 3.8, "I(X;Z) <= I(X;Y)", ha="center", fontsize=11)
    inequality_color = "green" if inequality_satisfied else "red"
    inequality_symbol = "check" if inequality_satisfied else "times"
    ax1.text(
        5,
        3.1,
        f"{mi_xz:.4f} <= {mi_xy:.4f}",
        ha="center",
        fontsize=11,
        color=inequality_color,
        fontweight="bold",
    )
    ax1.text(
        5,
        2.4,
        f"I(X;Z) = {mi_xz:.4f} bits",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.6),
    )
    ax1.text(
        5,
        1.5,
        f"Information Loss: {info_loss:.4f} bits\n({100 - info_retention_pct:.1f}%)",
        ha="center",
        fontsize=9,
    )
    # Plot 2: Joint distributions heatmaps.
    ax2_left = plt.subplot(1, 8, 3)
    ax2_right = plt.subplot(1, 8, 4)
    # P(X,Y) heatmap.
    p_xy_df = pd.DataFrame(
        p_xy, index=[f"X={i}" for i in range(4)], columns=["Y=0", "Y=1"]
    )
    sns.heatmap(
        p_xy_df,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=0.4,
        cbar=False,
        ax=ax2_left,
        annot_kws={"fontsize": 9, "fontweight": "bold"},
    )
    ax2_left.set_xlabel("Y", fontsize=10)
    ax2_left.set_ylabel("X", fontsize=10)
    ax2_left.set_title("P(X,Y)", fontsize=11, fontweight="bold")
    # P(Y,Z) heatmap.
    p_yz_df = pd.DataFrame(p_yz, index=["Y=0", "Y=1"], columns=["Z=0", "Z=1"])
    sns.heatmap(
        p_yz_df,
        annot=True,
        fmt=".3f",
        cmap="Oranges",
        vmin=0,
        vmax=0.6,
        cbar=False,
        ax=ax2_right,
        annot_kws={"fontsize": 9, "fontweight": "bold"},
    )
    ax2_right.set_xlabel("Z", fontsize=10)
    ax2_right.set_ylabel("Y", fontsize=10)
    ax2_right.set_title("P(Y,Z)", fontsize=11, fontweight="bold")
    # Plot 3: Entropy and mutual information metrics.
    metrics_df = pd.DataFrame(
        {
            "Metric": ["H(X)", "H(Y)", "H(Z)", "I(X;Y)", "I(Y;Z)", "I(X;Z)"],
            "Value": [h_x, h_y, h_z, mi_xy, mi_yz, mi_xz],
            "Type": [
                "Entropy",
                "Entropy",
                "Entropy",
                "Mutual Info",
                "Mutual Info",
                "Mutual Info",
            ],
        }
    )
    colors_metrics = [
        "steelblue",
        "coral",
        "lightgreen",
        "purple",
        "orange",
        "green",
    ]
    bars = ax3.bar(
        range(len(metrics_df)),
        metrics_df["Value"],
        color=colors_metrics,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax3.set_ylabel("Information [bits]", fontsize=12)
    ax3.set_xlabel("")
    ax3.set_title("Information Metrics", fontsize=14, fontweight="bold")
    ax3.set_xticks(range(len(metrics_df)))
    ax3.set_xticklabels(
        metrics_df["Metric"], rotation=45, ha="right", fontsize=10
    )
    ax3.grid(True, alpha=0.3, axis="y")
    max_val = max(metrics_df["Value"])
    ax3.set_ylim([0, max_val * 1.2])
    # Add value labels on bars.
    for i, (bar, val) in enumerate(zip(bars, metrics_df["Value"])):
        height = bar.get_height()
        ax3.text(
            i,
            height + max_val * 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    # Highlight I(X;Z) bar.
    bars[5].set_linewidth(3)
    bars[5].set_edgecolor("darkgreen")
    # Plot 4: Comments text panel.
    ax4.axis("off")
    ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Add comprehensive explanation text.
    text_content = (
        f"{interpretation}\n\n"
        f"Entropies:\n"
        f"  H(X) = {h_x:.4f} bits\n"
        f"  H(Y) = {h_y:.4f} bits\n"
        f"  H(Z) = {h_z:.4f} bits\n\n"
        f"Mutual Information:\n"
        f"  I(X;Y) = {mi_xy:.4f} bits\n"
        f"  I(Y;Z) = {mi_yz:.4f} bits\n"
        f"  I(X;Z) = {mi_xz:.4f} bits\n\n"
        f"Inequality Verification:\n"
        f"  I(X;Z) <= I(X;Y)\n"
        f"  {mi_xz:.4f} <= {mi_xy:.4f}\n"
        f"  Status: {'Satisfied' if inequality_satisfied else 'VIOLATED'}"
    )
    ax4.text(
        0.05,
        0.95,
        text_content,
        transform=ax4.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
        wrap=True,
    )
    # Use subplots_adjust with fixed parameters instead of tight_layout.
    # This ensures consistent spacing and dimensions across all frames.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.35
    )
    plt.show()


def demonstrate_data_processing_inequality() -> None:
    """
    Demonstrate the data processing inequality with a simple example.
    """
    # Create Markov chain X -> Y -> Z.
    # X: Original signal (4 states).
    p_x = np.array([0.4, 0.3, 0.2, 0.1])
    # Y: Compressed version (2 states) - groups (0,1) and (2,3).
    # Z: Further processed (2 states) - adds noise.
    # Transition probabilities.
    # P(Y|X): X states 0,1 -> Y=0 with high prob; X states 2,3 -> Y=1.
    p_y_given_x = np.array(
        [
            [0.9, 0.1],  # X=0 -> mostly Y=0
            [0.85, 0.15],  # X=1 -> mostly Y=0
            [0.1, 0.9],  # X=2 -> mostly Y=1
            [0.05, 0.95],  # X=3 -> mostly Y=1
        ]
    )
    # P(Z|Y): Add noise.
    p_z_given_y = np.array(
        [
            [0.8, 0.2],  # Y=0 -> mostly Z=0
            [0.2, 0.8],  # Y=1 -> mostly Z=1
        ]
    )
    # Calculate joint distributions.
    p_xy = p_x[:, np.newaxis] * p_y_given_x  # Joint P(X,Y)
    p_y = p_xy.sum(axis=0)  # Marginal P(Y)
    p_yz = p_y[:, np.newaxis] * p_z_given_y  # Joint P(Y,Z)
    p_z = p_yz.sum(axis=0)  # Marginal P(Z)
    # Calculate P(X,Z) through Y.
    p_xz = np.zeros((4, 2))
    for i in range(4):
        for k in range(2):
            for j in range(2):
                p_xz[i, k] += p_x[i] * p_y_given_x[i, j] * p_z_given_y[j, k]
    # Calculate mutual informations.
    mi_xy = calculate_mutual_information(p_xy)
    mi_yz = calculate_mutual_information(p_yz)
    mi_xz = calculate_mutual_information(p_xz)
    # Visualize.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Plot 1: Mutual information values.
    stages = ["I(X;Y)", "I(Y;Z)", "I(X;Z)"]
    mi_values = [mi_xy, mi_yz, mi_xz]
    colors_stages = ["steelblue", "coral", "lightgreen"]
    bars = ax1.bar(
        stages,
        mi_values,
        color=colors_stages,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    ax1.set_ylabel("Mutual Information [bits]", fontsize=12)
    ax1.set_title(
        "Data Processing Inequality\nX → Y → Z", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="y")
    # Add values and inequality annotations.
    for bar, val in zip(bars, mi_values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax1.axhline(mi_xy, color="steelblue", linestyle="--", alpha=0.5, linewidth=2)
    ax1.text(1.5, mi_xy + 0.03, "I(X;Y) bound", fontsize=10, style="italic")
    # Plot 2: Information flow diagram.
    ax2.text(
        0.5,
        0.9,
        "Data Processing Inequality",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )
    ax2.text(
        0.5,
        0.75,
        "Markov Chain: X → Y → Z",
        ha="center",
        fontsize=13,
        style="italic",
    )
    ax2.text(
        0.5,
        0.65,
        f"I(X;Y) = {mi_xy:.4f} bits",
        ha="center",
        fontsize=12,
        color="steelblue",
    )
    ax2.text(
        0.5,
        0.58,
        f"I(Y;Z) = {mi_yz:.4f} bits",
        ha="center",
        fontsize=12,
        color="coral",
    )
    ax2.text(
        0.5,
        0.51,
        f"I(X;Z) = {mi_xz:.4f} bits",
        ha="center",
        fontsize=12,
        color="green",
    )
    ax2.text(
        0.5,
        0.40,
        "Data Processing Inequality:",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    ax2.text(0.5, 0.33, "I(X;Z) ≤ I(X;Y)", ha="center", fontsize=12)
    ax2.text(
        0.5,
        0.26,
        f"{mi_xz:.4f} ≤ {mi_xy:.4f} ✓",
        ha="center",
        fontsize=12,
        color="green" if mi_xz <= mi_xy else "red",
        fontweight="bold",
    )
    ax2.text(0.5, 0.16, "Comments:", ha="center", fontsize=12, fontweight="bold")
    ax2.text(
        0.5,
        0.09,
        f"Information lost from X to Z: {mi_xy - mi_xz:.4f} bits",
        ha="center",
        fontsize=11,
    )
    ax2.text(
        0.5,
        0.03,
        f"({((mi_xy - mi_xz) / mi_xy) * 100:.1f}% of original information)",
        ha="center",
        fontsize=10,
        style="italic",
    )
    ax2.axis("off")
    plt.tight_layout()
    plt.show()


# #############################################################################
# Minimum Description Length (MDL)
# #############################################################################


def generate_mdl_data(
    *,
    n_samples: int = 50,
    true_degree: int = 3,
    noise_level: float = 0.3,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic polynomial data for MDL demonstration.

    :param n_samples: Number of data points to generate
    :param true_degree: Degree of the true underlying polynomial
    :param noise_level: Standard deviation of Gaussian noise
    :param seed: Random seed for reproducibility
    :return: Tuple of (x values, y values, true coefficients)
    """
    np.random.seed(seed)
    # Generate x values uniformly in [-1, 1].
    x = np.linspace(-1, 1, n_samples)
    # Generate true polynomial coefficients.
    # Use smaller coefficients for higher degrees to keep y values reasonable.
    true_coeffs = np.random.randn(true_degree + 1) * 0.5
    # Generate true y values.
    y_true = np.polyval(true_coeffs, x)
    # Add Gaussian noise.
    noise = np.random.normal(0, noise_level, n_samples)
    y = y_true + noise
    return x, y, true_coeffs


def fit_polynomial_model(*, x: np.ndarray, y: np.ndarray, degree: int) -> tuple:
    """
    Fit polynomial of given degree to data.

    :param x: Input x values
    :param y: Target y values
    :param degree: Degree of polynomial to fit
    :return: Tuple of (fitted coefficients, predictions, MSE)
    """
    # Fit polynomial.
    coeffs = np.polyfit(x, y, degree)
    # Generate predictions.
    y_pred = np.polyval(coeffs, x)
    # Calculate mean squared error.
    mse = np.mean((y - y_pred) ** 2)
    return coeffs, y_pred, mse


def calculate_mdl_components(
    *, n_samples: int, n_params: int, mse: float
) -> dict:
    """
    Calculate MDL components for model selection.

    :param n_samples: Number of data samples
    :param n_params: Number of model parameters
    :param mse: Mean squared error of the model
    :return: Dictionary with 'model_cost', 'data_cost', 'total_mdl'
    """
    # L(H): Model description length.
    # Penalizes complexity using BIC-like penalty: k * log(n).
    # Each parameter costs log2(n) bits to describe.
    model_cost = n_params * np.log2(n_samples)
    # L(D|H): Data encoding cost given the model.
    # Cost to encode residuals, related to log(MSE).
    # Add small epsilon to avoid log(0).
    epsilon = 1e-10
    data_cost = (n_samples / 2) * np.log2(mse + epsilon)
    # Total MDL.
    total_mdl = model_cost + data_cost
    return {
        "model_cost": model_cost,
        "data_cost": data_cost,
        "total_mdl": total_mdl,
    }


def plot_mdl_interactive(
    *,
    degree: int = 3,
    n_samples: int = 50,
    true_degree: int = 3,
    noise_level: float = 0.3,
    figsize: Optional[tuple] = None,
) -> None:
    """
    Interactive visualization of Minimum Description Length principle.

    :param degree: Current polynomial degree to fit (slider controlled)
    :param n_samples: Number of data samples
    :param true_degree: True underlying polynomial degree
    :param noise_level: Noise level in data generation
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Generate data.
    x, y, true_coeffs = generate_mdl_data(
        n_samples=n_samples,
        true_degree=true_degree,
        noise_level=noise_level,
        seed=42,
    )
    # Fit model with current degree.
    coeffs, y_pred, mse = fit_polynomial_model(x=x, y=y, degree=degree)
    # Calculate MDL for current model.
    n_params = degree + 1  # Polynomial of degree d has d+1 parameters.
    mdl_current = calculate_mdl_components(
        n_samples=n_samples, n_params=n_params, mse=mse
    )
    # Calculate MDL for all degrees (1 to 8) to show the curve.
    max_degree = 8
    degrees = np.arange(1, max_degree + 1)
    mdl_values = []
    model_costs = []
    data_costs = []
    for d in degrees:
        _, _, mse_d = fit_polynomial_model(x=x, y=y, degree=d)
        mdl_d = calculate_mdl_components(
            n_samples=n_samples, n_params=d + 1, mse=mse_d
        )
        mdl_values.append(mdl_d["total_mdl"])
        model_costs.append(mdl_d["model_cost"])
        data_costs.append(mdl_d["data_cost"])
    # Find optimal degree (minimum MDL).
    optimal_idx = np.argmin(mdl_values)
    optimal_degree = degrees[optimal_idx]
    # Determine interpretation based on current degree.
    if degree < optimal_degree - 1:
        status = "Underfitting"
        interpretation = (
            f"Model too simple (degree {degree}).\n"
            f"High data encoding cost due to poor fit.\n"
            f"Total MDL = {mdl_current['total_mdl']:.2f} bits > optimal.\n"
            f"Recommendation: Increase complexity."
        )
    elif degree > optimal_degree + 1:
        status = "Overfitting"
        interpretation = (
            f"Model too complex (degree {degree}).\n"
            f"High model cost due to many parameters.\n"
            f"Total MDL = {mdl_current['total_mdl']:.2f} bits > optimal.\n"
            f"Recommendation: Reduce complexity."
        )
    else:
        status = "Optimal/Near-Optimal"
        interpretation = (
            f"Well-balanced model (degree {degree}).\n"
            f"Good tradeoff between fit and complexity.\n"
            f"Total MDL = {mdl_current['total_mdl']:.2f} bits ~ minimum.\n"
            f"Optimal degree: {optimal_degree}."
        )
    # Create visualization with 4 subplots in a single row.
    # Use gridspec_kw to set fixed width ratios for consistent layout.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 1, 1.2]}
    )
    # Plot 1: Data and Model Fit.
    ax1.scatter(x, y, alpha=0.6, s=50, color="steelblue", label="Data", zorder=3)
    # Sort x for smooth curve plotting.
    x_sorted = np.sort(x)
    y_fit_sorted = np.polyval(coeffs, x_sorted)
    ax1.plot(
        x_sorted,
        y_fit_sorted,
        "r-",
        linewidth=2,
        label=f"Fit (deg={degree})",
        zorder=2,
    )
    # Plot residuals as light lines.
    for xi, yi, yi_pred in zip(x, y, y_pred):
        ax1.plot(
            [xi, xi], [yi, yi_pred], "gray", alpha=0.3, linewidth=1, zorder=1
        )
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title(
        f"Data and Model Fit\nMSE = {mse:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(True, alpha=0.3)
    # Plot 2: MDL Components Bar Chart.
    components = ["L(H)\nModel", "L(D|H)\nData", "Total\nMDL"]
    values_bar = [
        mdl_current["model_cost"],
        mdl_current["data_cost"],
        mdl_current["total_mdl"],
    ]
    colors_mdl = ["coral", "skyblue", "purple"]
    bars = ax2.bar(
        components,
        values_bar,
        color=colors_mdl,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    ax2.set_ylabel("Description Length [bits]", fontsize=12)
    ax2.set_title("MDL Components", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    # Add value labels on bars.
    for bar, val in zip(bars, values_bar):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(values_bar) * 0.02,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    # Plot 3: MDL vs Model Complexity Curve.
    ax3.plot(
        degrees,
        mdl_values,
        "b-",
        linewidth=2,
        marker="o",
        markersize=6,
        label="Total MDL",
    )
    # Highlight current degree.
    ax3.scatter(
        [degree],
        [mdl_current["total_mdl"]],
        color="red",
        s=150,
        zorder=5,
        label=f"Current (deg={degree})",
    )
    # Highlight optimal degree.
    ax3.scatter(
        [optimal_degree],
        [mdl_values[optimal_idx]],
        color="green",
        s=150,
        marker="*",
        zorder=5,
        label=f"Optimal (deg={optimal_degree})",
    )
    ax3.set_xlabel("Model Complexity (Polynomial Degree)", fontsize=12)
    ax3.set_ylabel("Total MDL [bits]", fontsize=12)
    ax3.set_title("MDL vs Complexity", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=10, loc="upper center")
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(degrees)
    # Plot 4: Comments Panel.
    ax4.axis("off")
    ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Wrap interpretation text for consistent layout.
    wrapped_interpretation = textwrap.fill(interpretation, width=40)
    # Add explanation text.
    text_content = (
        f"Model Status: {status}\n"
        f"{'=' * 30}\n\n"
        f"Current Model (degree {degree}):\n"
        f"  • Parameters: {n_params}\n"
        f"  • L(H) = {mdl_current['model_cost']:.2f} bits\n"
        f"  • L(D|H) = {mdl_current['data_cost']:.2f} bits\n"
        f"  • Total MDL = {mdl_current['total_mdl']:.2f}\n\n"
        f"Analysis:\n"
        f"  {wrapped_interpretation}\n\n"
        f"MDL Principle:\n"
        f"  Select model minimizing\n"
        f"  MDL = L(H) + L(D|H)\n"
        f"  Balances complexity vs fit."
    )
    ax4.text(
        0.05,
        0.95,
        text_content,
        transform=ax4.transAxes,
        fontsize=9,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
        wrap=True,
    )
    # Use subplots_adjust with fixed parameters for consistent dimensions.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.25
    )
    plt.show()


# #############################################################################
# Kolmogorov Complexity
# #############################################################################


def generate_string_by_type(
    *, string_type: str = "Random", length: int = 64, seed: int = 42
) -> np.ndarray:
    """
    Generate binary string of specified complexity type.

    :param string_type: Type of string to generate
        ("All Zeros", "Repeating 01", "Fibonacci", "Random", "Semi-random")
    :param length: Length of binary string
    :param seed: Random seed for reproducible random strings
    :return: Binary string as numpy array of 0s and 1s
    """
    np.random.seed(seed)
    if string_type == "All Zeros":
        # Highly compressible pattern.
        return np.zeros(length, dtype=int)
    elif string_type == "Repeating 01":
        # Simple repeating pattern.
        pattern = np.array([0, 1])
        return np.tile(pattern, length // 2 + 1)[:length]
    elif string_type == "Fibonacci":
        # Generate Fibonacci sequence and mark positions.
        # More complex but still structured.
        fib = [1, 1]
        while fib[-1] < length:
            fib.append(fib[-1] + fib[-2])
        # Create binary string with 1s at Fibonacci positions.
        result = np.zeros(length, dtype=int)
        for f in fib:
            if f < length:
                result[f] = 1
        return result
    elif string_type == "Random":
        # Incompressible random string.
        return np.random.randint(0, 2, length)
    elif string_type == "Semi-random":
        # Mix of pattern and randomness.
        # First half is patterned, second half is random.
        first_half = np.tile([0, 0, 0, 0, 1, 1, 1, 1], length // 16 + 1)[
            : length // 2
        ]
        second_half = np.random.randint(0, 2, length - len(first_half))
        return np.concatenate([first_half, second_half])
    else:
        # Default to random.
        return np.random.randint(0, 2, length)


def calculate_description_length(*, string_type: str, length: int) -> int:
    """
    Calculate approximate description length (K-complexity proxy).

    :param string_type: Type of string
    :param length: Length of string
    :return: Approximate description length in bits
    """
    # Base overhead for any program (interpreter, basic syntax).
    base_overhead = 20
    if string_type == "All Zeros":
        # Description: "print('0' * n)" where n is encoded in log(n) bits.
        return base_overhead + int(np.ceil(np.log2(length + 1)))
    elif string_type == "Repeating 01":
        # Description: "print('01' * (n//2))" - constant pattern + log(n).
        pattern_bits = 2  # Encode "01" pattern.
        count_bits = int(np.ceil(np.log2(length + 1)))
        return base_overhead + pattern_bits + count_bits
    elif string_type == "Fibonacci":
        # Description: Fibonacci algorithm + length parameter.
        algorithm_bits = 40  # Fib generation algorithm.
        count_bits = int(np.ceil(np.log2(length + 1)))
        return base_overhead + algorithm_bits + count_bits
    elif string_type == "Random":
        # Description: Must include entire string (incompressible).
        return length  # No compression possible.
    elif string_type == "Semi-random":
        # Description: Pattern for first half + full second half.
        pattern_bits = length // 2 // 4  # Some compression on first half.
        random_bits = length // 2  # No compression on second half.
        return base_overhead + pattern_bits + random_bits
    else:
        return length


def get_program_description(*, string_type: str, length: int) -> tuple:
    """
    Get human-readable program description and its length.

    :param string_type: Type of string
    :param length: Length of string
    :return: Tuple of (description text, description length)
    """
    desc_length = calculate_description_length(
        string_type=string_type, length=length
    )
    if string_type == "All Zeros":
        description = (
            f"# Program to generate string:\n"
            f"def generate():\n"
            f"    return '0' * {length}\n\n"
            f"Description length: ~{desc_length} bits\n"
            f"(Base overhead + log2({length}))"
        )
    elif string_type == "Repeating 01":
        description = (
            f"# Program to generate string:\n"
            f"def generate():\n"
            f"    return '01' * {length // 2}\n\n"
            f"Description length: ~{desc_length} bits\n"
            f"(Base + pattern + log2({length}))"
        )
    elif string_type == "Fibonacci":
        description = (
            f"# Program to generate string:\n"
            f"def generate():\n"
            f"    fib = [1,1]\n"
            f"    while fib[-1] < {length}:\n"
            f"        fib.append(fib[-1]+fib[-2])\n"
            f"    result = ['0']*{length}\n"
            f"    for f in fib:\n"
            f"        if f < {length}: result[f]='1'\n"
            f"    return ''.join(result)\n\n"
            f"Description: ~{desc_length} bits"
        )
    elif string_type == "Random":
        # Must include actual bits (show truncated).
        description = (
            f"# Program to generate string:\n"
            f"def generate():\n"
            f"    # No pattern - must store all bits\n"
            f"    return '[actual bits...]'\n\n"
            f"Description length: ~{desc_length} bits\n"
            f"(Full string - incompressible!)"
        )
    elif string_type == "Semi-random":
        description = (
            f"# Program to generate string:\n"
            f"def generate():\n"
            f"    # First half: pattern\n"
            f"    p = '00001111' * {length // 16}\n"
            f"    # Second half: random bits\n"
            f"    r = '[random bits...]'\n"
            f"    return p + r\n\n"
            f"Description: ~{desc_length} bits"
        )
    else:
        description = f"Unknown type\nLength: {desc_length} bits"
    return description, desc_length


def plot_kolmogorov_complexity_interactive(
    *,
    string_type: str = "Random",
    length: int = 64,
    figsize: Optional[tuple] = None,
) -> None:
    """
    Interactive visualization of Kolmogorov Complexity concept.

    :param string_type: Type of binary string to generate
    :param length: Length of binary string
    :param figsize: Figure size as (width, height) in inches; defaults to
        (20, 5) if not specified
    """
    # Set default figsize if not provided.
    if figsize is None:
        figsize = (20, 5)
    # Generate string.
    binary_string = generate_string_by_type(
        string_type=string_type, length=length, seed=42
    )
    # Get program description.
    program_desc, desc_length = get_program_description(
        string_type=string_type, length=length
    )
    # Calculate compression ratio.
    compression_ratio = desc_length / length if length > 0 else 1.0
    # Determine complexity interpretation.
    if compression_ratio < 0.3:
        complexity_level = "Very Low"
        interpretation = (
            f"Highly structured pattern.\n"
            f"Short program generates long string.\n"
            f"Compression ratio: {compression_ratio:.2%}\n"
            f"K-complexity << string length"
        )
        color_status = "green"
    elif compression_ratio < 0.6:
        complexity_level = "Low to Medium"
        interpretation = (
            f"Some structure present.\n"
            f"Moderate compression possible.\n"
            f"Compression ratio: {compression_ratio:.2%}\n"
            f"K-complexity < string length"
        )
        color_status = "yellowgreen"
    elif compression_ratio < 0.9:
        complexity_level = "Medium to High"
        interpretation = (
            f"Limited structure.\n"
            f"Minimal compression achieved.\n"
            f"Compression ratio: {compression_ratio:.2%}\n"
            f"K-complexity approaching length"
        )
        color_status = "orange"
    else:
        complexity_level = "Very High (Random)"
        interpretation = (
            f"No compressible pattern!\n"
            f"Description equals string length.\n"
            f"Compression ratio: {compression_ratio:.2%}\n"
            f"K-complexity ≈ string length"
        )
        color_status = "red"
    # Create visualization with 4 subplots in a single row.
    # Use gridspec_kw to set fixed width ratios for consistent layout.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 1, 1.2]}
    )
    # Plot 1: String Visualization.
    # Display binary string as colored grid.
    n_rows = min(8, int(np.ceil(np.sqrt(length))))
    n_cols = int(np.ceil(length / n_rows))
    # Pad string to fill grid.
    padded_length = n_rows * n_cols
    padded_string = np.pad(
        binary_string,
        (0, padded_length - length),
        mode="constant",
        constant_values=-1,
    )
    grid = padded_string.reshape(n_rows, n_cols)
    # Create color map: 0=blue, 1=red, -1=gray (padding).
    cmap_colors = ["#3498db", "#e74c3c", "#95a5a6"]  # Blue, Red, Gray
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(cmap_colors)
    im = ax1.imshow(grid, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
    ax1.set_title(
        f"Binary String Visualization\n({string_type}, n={length})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Bit Position (columns)", fontsize=11)
    ax1.set_ylabel("Rows", fontsize=11)
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Add legend.
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3498db", label="0 bit"),
        Patch(facecolor="#e74c3c", label="1 bit"),
    ]
    ax1.legend(
        handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9
    )
    # Plot 2: Program Description.
    ax2.axis("off")
    ax2.set_title(
        "Shortest Program Description",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax2.text(
        0.05,
        0.95,
        program_desc,
        transform=ax2.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3),
        wrap=True,
    )
    # Plot 3: Complexity Comparison.
    metrics = ["String\nLength", "Description\nLength"]
    values = [length, desc_length]
    colors_bar = ["steelblue", "coral"]
    bars = ax3.bar(
        metrics,
        values,
        color=colors_bar,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    ax3.set_ylabel("Length [bits]", fontsize=12)
    ax3.set_title("Complexity Comparison", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    # Add value labels on bars.
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(values) * 0.02,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    # Add compression ratio as text annotation.
    ax3.text(
        0.5,
        0.5,
        f"Compression\nRatio:\n{compression_ratio:.1%}",
        transform=ax3.transAxes,
        fontsize=12,
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=color_status, alpha=0.3),
    )
    # Plot 4: Comments Panel.
    ax4.axis("off")
    ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    # Wrap interpretation text for consistent layout.
    wrapped_interpretation = textwrap.fill(interpretation, width=40)
    # Add explanation text.
    text_content = (
        f"Kolmogorov Complexity:\n"
        f"{'=' * 30}\n\n"
        f"String Type: {string_type}\n"
        f"String Length: {length} bits\n"
        f"Description Length: {desc_length} bits\n\n"
        f"Complexity Level:\n"
        f"  {complexity_level}\n\n"
        f"Analysis:\n"
        f"  {wrapped_interpretation}\n\n"
        f"Key Insight:\n"
        f"  K(x) = length of shortest\n"
        f"  program generating x.\n"
        f"  Random strings: K(x) ≈ |x|\n"
        f"  Patterned: K(x) << |x|"
    )
    ax4.text(
        0.05,
        0.95,
        text_content,
        transform=ax4.transAxes,
        fontsize=9,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
        wrap=True,
    )
    # Use subplots_adjust with fixed parameters for consistent dimensions.
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.25
    )
    plt.show()
