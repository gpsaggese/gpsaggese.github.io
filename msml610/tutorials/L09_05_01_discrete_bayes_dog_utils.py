"""
Utility functions for MSML610 course tutorials.

Import as:

import msml610.tutorials.L09_05_01_discrete_bayes_dog_utils as mtl00dbdu
"""

import logging
import copy

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


_LOG = logging.getLogger(__name__)


def plot_belief(
    belief: np.ndarray,
    *,
    title: str = "Belief",
    y_lim: Tuple[float, float] = (0, 1),
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Plot a belief distribution.

    :param belief: The belief array
    :param ax: The axis to plot on
    :param title: The title for the belief distribution
    """
    if ax is None:
        _, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    ax.bar(range(len(belief)), belief,
            color="#1F77B4")
    ax.set_ylim(y_lim)
    # Show all integers on x-axis.
    ax.set_xticks(range(len(belief)))
    # Grid lines at every integer (vertical).
    ax.grid(axis="x", color="gray", linestyle="-", linewidth=0.8, alpha=0.6)
    # Horizontal grid lines
    ax.grid(axis="y", color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")
    ax.set_title(title)


def plot_beliefs(
    belief1: np.ndarray,
    belief2: np.ndarray,
    *,
    title1: str = "Belief1",
    title2: str = "Belief2",
    y_lim: Tuple[float, float] = (0, 1),
    same_plot: bool = True,
) -> None:
    """
    Plot two belief distributions, either side by side or together as bars
    with different colors and a legend.

    :param belief1: The first belief array
    :param belief2: The second belief array
    :param title1: The title for the first belief distribution (also used as label if same_plot=True)
    :param title2: The title for the second belief distribution (also used as label if same_plot=True)
    :param y_lim: The limits for the y-axis
    :param same_plot: If True, show both beliefs on the same axes with legend
    """
    if not same_plot:
        figsize = copy.deepcopy(plt.rcParams["figure.figsize"])
        figsize[0] = figsize[0] * 2
        _, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        plot_belief(belief1, title=title1, y_lim=y_lim, ax=axes[0])
        plot_belief(belief2, title=title2, y_lim=y_lim, ax=axes[1])
        plt.tight_layout()
    else:
        n = len(belief1)
        # TODO(ai_gp): Use a dassert.
        if len(belief2) != n:
            raise ValueError("belief1 and belief2 must have the same length when plotting on same plot.")
        indices = np.arange(n)
        width = 0.4
        _, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
        # Plot first belief (shift left)
        ax.bar(
            indices - width / 2,
            belief1,
            width=width,
            label=title1,
            color="#1F77B4",  # blue
            zorder=2,
        )
        # Plot second belief (shift right)
        ax.bar(
            indices + width / 2,
            belief2,
            width=width,
            label=title2,
            color="#79B3D9",  # light blue
            zorder=2,
        )
        ax.set_ylim(y_lim if y_lim is not None else (0, 1))
        ax.set_xticks(indices)
        ax.set_xlabel("State")
        ax.set_ylabel("Probability")
        # No global main_title (no plt.suptitle); use legend with labels
        ax.grid(axis="x", color="gray", linestyle="-", linewidth=0.8, alpha=0.6, zorder=1)
        ax.grid(axis="y", color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
        ax.legend()
        plt.tight_layout()
