"""
Utility functions for MSML610 course tutorials.

Import as:

import msml610.tutorials.L09_05_01_discrete_bayes_dog_utils as mtl00dbdu
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


_LOG = logging.getLogger(__name__)


def plot_belief(belief: np.ndarray, *, title: str = "Belief Distribution", y_lim: Tuple[float, float] = (0, 1), ax: Optional[plt.Axes] = None) -> None:
    """
    Plot a belief distribution.

    :param belief: The belief array
    :param ax: The axis to plot on
    :param title: The title for the belief distribution
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.bar(range(len(belief)), belief)
    if y_lim is not None:
        y_lim = (0, 1)
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


def plot_beliefs(belief1: np.ndarray, belief2: np.ndarray, *, title1: str = "Belief Distribution 1", title2: str = "Belief Distribution 2", y_lim: Tuple[float, float] = (0, 1)) -> None:
    """
    Plot two belief distributions side by side in a 1x2 figure.

    :param belief1: The first belief array
    :param belief2: The second belief array
    :param title1: The title for the first belief distribution
    :param title2: The title for the second belief distribution
    """
    _ , axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    plot_belief(belief1, title=title1, y_lim=y_lim, ax=axes[0])
    plot_belief(belief2, title=title2, y_lim=y_lim, ax=axes[1])
    plt.tight_layout()
