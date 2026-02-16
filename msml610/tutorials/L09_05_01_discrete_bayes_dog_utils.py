"""
Utility functions for MSML610 course tutorials.

Import as:

import msml610.tutorials.msml610_utils as mtumsuti
"""

import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)


def plot_belief(belief: np.ndarray) -> None:
    _, ax = plt.subplots()
    ax.bar(range(len(belief)), belief)
    ax.set_ylim(0, 1)
    # show all integers on x-axis
    ax.set_xticks(range(len(belief)))
    # grid lines at every integer (vertical)
    ax.grid(axis='x', color='gray', linestyle='-', linewidth=0.8, alpha=0.6)
    # optional: horizontal grid lines
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")
    ax.set_title("Belief Distribution")
