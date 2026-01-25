"""
Utility functions for Information Theory lesson.

Import as:

import Lesson94_Information_Theory_utils as litutils
"""

import logging
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
    import helpers.hdbg as hdbg

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


def create_correlated_joint_distribution(*, correlation: float = 0.5) -> np.ndarray:
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
    *, dependence: float = 0.5, n_samples: int = 100
) -> None:
    """
    Interactive visualization of joint entropy with dependence control.

    :param dependence: Dependence strength between variables (0=independent,
        1=perfectly correlated)
    :param n_samples: Number of samples to generate for scatter plot
    """
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
        interpretation = "Independence: H(X,Y) ≈ H(X) + H(Y) (maximum joint entropy)"
    elif dependence > 0.9:
        interpretation = "Perfect dependence: H(X,Y) ≈ H(X) = H(Y) (minimum joint entropy)"
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
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
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
    ax4.set_title("Explanation", fontsize=14, fontweight="bold", pad=20)
    # Add explanation text.
    text_content = (
        f"Entropy Values:\n"
        f"  • H(X) = {h_x:.4f} bits\n"
        f"  • H(Y) = {h_y:.4f} bits\n"
        f"  • H(X,Y) = {h_xy:.4f} bits\n\n"
        f"Interpretation:\n"
        f"  {interpretation}\n\n"
        f"Verification:\n"
        f"  H(X,Y) + I(X;Y) = H(X) + H(Y)\n"
        f"  {h_xy:.4f} + {mi:.4f} = {h_x:.4f} + {h_y:.4f}\n"
        f"  {h_xy + mi:.4f} = {h_x + h_y:.4f}"
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
    plt.tight_layout()
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


def plot_binary_entropy_interactive(*, p: float = 0.5) -> None:
    """
    Plot binary entropy function with current value highlighted.

    :param p: Probability of outcome 1 (slider controlled)
    """
    # Create array of probabilities.
    p_values = np.linspace(0.001, 0.999, 1000)
    entropy_values = [binary_entropy(p_val) for p_val in p_values]
    # Create figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
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
    ax1.set_title("Binary Entropy Function", fontsize=14)
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
    plt.tight_layout()
    plt.show()
    # Print information content.
    if p > 0 and p < 1:
        info_0 = -np.log2(1 - p)
        info_1 = -np.log2(p)
        print(f"Information content of Outcome 0: {info_0:.4f} bits")
        print(f"Information content of Outcome 1: {info_1:.4f} bits")
        print(f"Expected information (Entropy): {binary_entropy(p):.4f} bits")


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
        f"H(X, Y) = H(X) + H(Y|X)",
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
        f"I(X;Y) = H(X) + H(Y) - H(X,Y)",
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
        f"I(X;Y) = H(Y) - H(Y|X)",
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
    ax2.text(
        0.5, 0.24, "Interpretation:", ha="center", fontsize=12, fontweight="bold"
    )
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
        f"({(mi/h_y)*100:.1f}% of total uncertainty in Y)",
        ha="center",
        fontsize=11,
        style="italic",
    )
    ax2.axis("off")
    plt.tight_layout()
    plt.show()


def plot_mutual_info_interactive(*, correlation: float = 0.5) -> None:
    """
    Interactive plot showing how correlation affects mutual information.

    :param correlation: Correlation strength between variables
    """
    joint_prob = create_correlated_joint_distribution(correlation=correlation)
    # Calculate metrics.
    mi = calculate_mutual_information(joint_prob)
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    h_x = calculate_entropy(p_x)
    h_y = calculate_entropy(p_y)
    # Create visualization.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    # Plot 1: Joint distribution heatmap.
    im = ax1.imshow(joint_prob, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Y=0", "Y=1"])
    ax1.set_yticklabels(["X=0", "X=1"])
    ax1.set_xlabel("Variable Y", fontsize=12)
    ax1.set_ylabel("Variable X", fontsize=12)
    ax1.set_title(
        f"Joint Distribution p(X,Y)\nCorrelation = {correlation:.2f}",
        fontsize=14,
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
    # Plot 2: Marginal distributions.
    x_pos = np.arange(2)
    width = 0.35
    ax2.bar(
        x_pos - width / 2, p_x, width, label="P(X)", alpha=0.7, color="steelblue"
    )
    ax2.bar(
        x_pos + width / 2, p_y, width, label="P(Y)", alpha=0.7, color="coral"
    )
    ax2.set_xlabel("Outcome", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_title("Marginal Distributions", fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["0", "1"])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    # Plot 3: Information metrics vs correlation.
    correlations = np.linspace(0, 1, 50)
    mis = []
    for corr in correlations:
        jp = create_correlated_joint_distribution(correlation=corr)
        mis.append(calculate_mutual_information(jp))
    ax3.plot(correlations, mis, "b-", linewidth=2, label="I(X;Y)")
    ax3.axvline(
        correlation,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Current: {correlation:.2f}",
    )
    ax3.scatter([correlation], [mi], color="red", s=100, zorder=5)
    ax3.set_xlabel("Correlation", fontsize=12)
    ax3.set_ylabel("Mutual Information [bits]", fontsize=12)
    ax3.set_title("Mutual Information vs Correlation", fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()
    # Print metrics.
    print(f"Entropy H(X) = {h_x:.4f} bits")
    print(f"Entropy H(Y) = {h_y:.4f} bits")
    print(f"Mutual Information I(X;Y) = {mi:.4f} bits")
    if h_y > 0:
        print(
            f"Percentage of Y's entropy explained by X: {(mi/h_y)*100:.2f}%"
        )


def plot_kl_divergence_interactive(*, p1: float = 0.7, q1: float = 0.5) -> None:
    """
    Interactive visualization of KL divergence between two binary distributions.

    :param p1: Probability for true distribution P
    :param q1: Probability for approximating distribution Q
    """
    # Create distributions.
    p = np.array([1 - p1, p1])
    q = np.array([1 - q1, q1])
    # Calculate metrics.
    kl_pq = calculate_kl_divergence(p, q)
    kl_qp = calculate_kl_divergence(q, p)
    ce_pq = calculate_cross_entropy(p, q)
    h_p = calculate_entropy(p)
    # Create visualization.
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    # Plot 1: Distributions comparison.
    ax1 = fig.add_subplot(gs[0, 0])
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
    ax1.set_title("Distribution Comparison", fontsize=14, fontweight="bold")
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
            )
    # Plot 2: KL divergence metrics.
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ["H(P)", "H(P,Q)", "D_KL(P||Q)", "D_KL(Q||P)"]
    values = [h_p, ce_pq, kl_pq, kl_qp]
    colors_m = ["steelblue", "purple", "red", "orange"]
    bars = ax2.bar(
        metrics, values, color=colors_m, alpha=0.7, edgecolor="black"
    )
    ax2.set_ylabel("Information [bits]", fontsize=12)
    ax2.set_title("Information Metrics", fontsize=14, fontweight="bold")
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
    ax3 = fig.add_subplot(gs[1, :])
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
        label=f"Current: D_KL={kl_pq:.3f}",
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
    # Add diagonal line (where P=Q).
    ax3.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5, label="P=Q (KL=0)")
    plt.tight_layout()
    plt.show()
    # Print analysis.
    print(f"True distribution P: [{1-p1:.2f}, {p1:.2f}]")
    print(f"Approximation Q:    [{1-q1:.2f}, {q1:.2f}]")
    print()
    print(f"Entropy H(P) = {h_p:.4f} bits")
    print(f"Cross-Entropy H(P,Q) = {ce_pq:.4f} bits")
    print(f"KL Divergence D_KL(P||Q) = {kl_pq:.4f} bits")
    print(
        f"KL Divergence D_KL(Q||P) = {kl_qp:.4f} bits (note: asymmetric!)"
    )
    print()
    print(
        f"Verification: H(P) + D_KL(P||Q) = {h_p:.4f} + {kl_pq:.4f} = {h_p+kl_pq:.4f}"
    )
    print(
        f"Should equal H(P,Q) = {ce_pq:.4f} ✓"
        if abs(h_p + kl_pq - ce_pq) < 0.001
        else f"Should equal H(P,Q) = {ce_pq:.4f} ✗"
    )


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
    ax1.axhline(
        mi_xy, color="steelblue", linestyle="--", alpha=0.5, linewidth=2
    )
    ax1.text(1.5, mi_xy + 0.03, f"I(X;Y) bound", fontsize=10, style="italic")
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
    ax2.text(
        0.5, 0.33, f"I(X;Z) ≤ I(X;Y)", ha="center", fontsize=12
    )
    ax2.text(
        0.5,
        0.26,
        f"{mi_xz:.4f} ≤ {mi_xy:.4f} ✓",
        ha="center",
        fontsize=12,
        color="green" if mi_xz <= mi_xy else "red",
        fontweight="bold",
    )
    ax2.text(
        0.5, 0.16, "Interpretation:", ha="center", fontsize=12, fontweight="bold"
    )
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
        f"({((mi_xy-mi_xz)/mi_xy)*100:.1f}% of original information)",
        ha="center",
        fontsize=10,
        style="italic",
    )
    ax2.axis("off")
    plt.tight_layout()
    plt.show()
    print("Data Processing Inequality Demonstration")
    print("=" * 60)
    print(f"Original signal X has {len(p_x)} states")
    print(f"Compressed Y has 2 states")
    print(f"Processed Z has 2 states")
    print()
    print(f"I(X;Y) = {mi_xy:.4f} bits (information between X and Y)")
    print(f"I(Y;Z) = {mi_yz:.4f} bits (information between Y and Z)")
    print(f"I(X;Z) = {mi_xz:.4f} bits (information between X and Z)")
    print()
    print(f"Data Processing Inequality: I(X;Z) ≤ I(X;Y)")
    print(
        f"{mi_xz:.4f} ≤ {mi_xy:.4f}: {'✓ Satisfied' if mi_xz <= mi_xy + 1e-6 else '✗ Violated'}"
    )
    print()
    print(
        f"Information lost: {mi_xy - mi_xz:.4f} bits ({((mi_xy-mi_xz)/mi_xy)*100:.1f}%)"
    )
