"""
Utility functions for Learning Theory lesson - Bin Analogy of ML.

Import as:

import msml610.tutorials.utils_Lesson05_Learning_Theory_Bin_Analogy_ML as mtulltba
"""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets
from IPython.display import display

import helpers.hdbg as hdbg
import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)

# Suppress FutureWarnings from seaborn and other libraries.
warnings.filterwarnings("ignore", category=FutureWarning)


# #############################################################################
# Cell 1: Visual Bin: Population of Marbles.
# #############################################################################


def _draw_bin_with_marbles(mu: float, seed: int) -> None:
    """
    Draw a 2D bin filled with red and green marbles.

    The marbles are arranged in a grid pattern, with colors determined by the
    proportion parameter mu.

    :param mu: True proportion of red marbles (0-1)
    :param seed: Random seed for reproducibility
    """
    # Create figure.
    fig, ax = plt.subplots(figsize=(6, 6))
    # Set up the bin dimensions.
    bin_width = 10
    bin_height = 10
    # Draw bin outline.
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            bin_width,
            bin_height,
            fill=False,
            edgecolor="black",
            linewidth=3,
        )
    )
    # Calculate number of marbles to display.
    n_marbles_x = 15
    n_marbles_y = 15
    total_marbles = n_marbles_x * n_marbles_y
    n_red_marbles = int(mu * total_marbles)
    # Create marble positions in a grid.
    x_positions = np.linspace(0.5, bin_width - 0.5, n_marbles_x)
    y_positions = np.linspace(0.5, bin_height - 0.5, n_marbles_y)
    # Create list of all marble positions.
    marble_positions = []
    for x in x_positions:
        for y in y_positions:
            marble_positions.append((x, y))
    # Shuffle positions randomly to distribute colors uniformly.
    np.random.seed(seed)
    np.random.shuffle(marble_positions)
    # Draw marbles.
    marble_radius = 0.25
    for i, (x, y) in enumerate(marble_positions):
        # Determine color based on position in shuffled list.
        if i < n_red_marbles:
            color = "red"
        else:
            color = "green"
        # Draw marble as circle.
        circle = plt.Circle(
            (x, y), marble_radius, color=color, ec="black", linewidth=0.5
        )
        ax.add_patch(circle)
    # Set axis properties.
    ax.set_xlim(-0.5, bin_width + 0.5)
    ax.set_ylim(-0.5, bin_height + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    # Add title with mu value.
    ax.set_title(
        f"Population: Unknown $\\mu$ = {mu:.2f}\n"
        f"Red marbles: {n_red_marbles}/{total_marbles}",
        fontsize=14,
        pad=20,
    )
    plt.tight_layout()
    plt.show()


def cell1_draw_bin_with_marbles_interactive() -> None:
    """
    Create interactive visualization of bin with marbles.

    Sets up an interactive widget with sliders for mu and seed parameters that
    control the proportion of red marbles and random arrangement in the bin.
    """
    mu_init = 0.5
    seed_init = 42
    # Create slider for seed.
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="",
        min_val=0,
        max_val=1000,
        step=1,
        initial_value=seed_init,
        is_float=False,
    )
    # Create slider for mu.
    mu_slider, mu_box = mtumsuti.build_widget_control(
        name="mu",
        description="",
        min_val=0.0,
        max_val=1.0,
        step=0.01,
        initial_value=mu_init,
        is_float=True,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _draw_bin_with_marbles, {"mu": mu_slider, "seed": seed_slider}
    )
    # Display widgets.
    display(ipywidgets.VBox([seed_box, mu_box, output]))


# #############################################################################
# Cell 2: Single Experiment: Is nu Close to mu?
# #############################################################################


def _plot_single_experiment(mu: float, N: int, seed: int) -> None:
    """
    Run a single sampling experiment and compare sample mean nu with true mean mu.

    :param mu: True proportion of red marbles (0-1)
    :param N: Number of samples to draw
    :param seed: Random seed for reproducibility
    """
    # Generate samples.
    np.random.seed(seed)
    samples = np.random.binomial(1, mu, size=N)
    nu = np.mean(samples)
    error = abs(nu - mu)
    # Determine color coding based on error.
    if error < 0.1:
        bar_color = "green"
        closeness = "was"
        color_desc = "close"
    elif error < 0.2:
        bar_color = "yellow"
        closeness = "was somewhat"
        color_desc = "medium distance"
    else:
        bar_color = "red"
        closeness = "was not"
        color_desc = "far"
    # Create visualization with 2 subplots.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1]}
    )
    # Plot 1: Bar chart comparing mu and nu.
    labels = ["mu\n(population)", "nu\n(sample)"]
    values = [mu, nu]
    colors = ["steelblue", bar_color]
    bars = ax1.bar(
        labels, values, color=colors, alpha=0.85, edgecolor="black", linewidth=2
    )
    # Add value labels on bars.
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax1.set_ylabel("Proportion", fontsize=12)
    ax1.set_title(
        f"Population vs Sample\nN = {N}, seed = {seed}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3, axis="y")
    # Add error display.
    ax1.text(
        0.5,
        0.95,
        f"Error: |nu - mu| = {error:.3f}",
        ha="center",
        va="top",
        transform=ax1.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    # Plot 2: Insight box.
    ax2.axis("off")
    ax2.set_title("Interpretation", fontsize=14, fontweight="bold", pad=20)
    # Generate insight text.
    text_content = (
        f"Parameters:\n"
        f"  mu = {mu:.3f} (true proportion)\n"
        f"  N = {N} (sample size)\n"
        f"  seed = {seed}\n\n"
        f"Single Experiment Result:\n"
        f"  nu = {nu:.3f} (sample proportion)\n"
        f"  |nu - mu| = {error:.3f}\n\n"
        f"Assessment:\n"
        f"  In this run, nu {closeness} close to mu.\n"
        f"  Color: {color_desc}\n\n"
        f"Key Insight:\n"
        f"- A single experiment gives us one estimate.\n"
        f"- Sometimes nu is close to mu, sometimes not.\n"
        f"- We need to understand: How OFTEN is nu close?\n"
        f"- Try different seeds to see variation!"
    )
    mtumsuti.add_fitted_text_box(ax2, text_content)
    plt.tight_layout()
    plt.show()


def cell2_plot_single_experiment_interactive() -> None:
    """
    Create interactive visualization of single sampling experiment.

    Sets up an interactive widget with sliders for seed, mu, and N parameters
    that shows how close the sample proportion nu is to the true proportion mu.
    """
    mu_init = 0.6
    N_init = 100
    seed_init = 42
    # Create interactive widgets.
    seed_slider, seed_box = mtumsuti.build_widget_control(
        name="seed",
        description="seed",
        min_val=0,
        max_val=1000,
        step=1,
        initial_value=seed_init,
        is_float=False,
    )
    mu_slider, mu_box = mtumsuti.build_widget_control(
        name="mu",
        description="mu",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=mu_init,
        is_float=True,
    )
    N_slider, N_box = mtumsuti.build_widget_control(
        name="N",
        description="N",
        min_val=10,
        max_val=1000,
        step=10,
        initial_value=N_init,
        is_float=False,
    )
    # Create interactive output.
    output = ipywidgets.interactive_output(
        _plot_single_experiment,
        {"mu": mu_slider, "N": N_slider, "seed": seed_slider},
    )
    # Display widgets.
    display(ipywidgets.VBox([seed_box, mu_box, N_box, output]))
