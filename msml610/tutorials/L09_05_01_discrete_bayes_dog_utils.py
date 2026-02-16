"""
Utility functions for MSML610 course tutorials.

Import as:

import msml610.tutorials.L09_05_01_discrete_bayes_dog_utils as mtl00dbdu
"""

import logging

import matplotlib.pyplot as plt
import numpy as np


_LOG = logging.getLogger(__name__)


def plot_belief(belief: np.ndarray) -> None:
    _, ax = plt.subplots()
    ax.bar(range(len(belief)), belief)
    ax.set_ylim(0, 1)
    # show all integers on x-axis
    ax.set_xticks(range(len(belief)))
    # grid lines at every integer (vertical)
    ax.grid(axis="x", color="gray", linestyle="-", linewidth=0.8, alpha=0.6)
    # optional: horizontal grid lines
    ax.grid(axis="y", color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")
    ax.set_title("Belief Distribution")
