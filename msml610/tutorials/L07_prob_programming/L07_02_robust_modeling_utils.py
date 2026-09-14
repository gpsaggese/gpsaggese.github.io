"""
Utility functions for the robust modeling tutorial (L07_02_robust_modeling).

Import as:

import msml610.tutorials.L07_prob_programming.L07_02_robust_modeling_utils as mtlppl0rmu
"""

import logging
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

_LOG = logging.getLogger(__name__)


# #############################################################################
# Cell 1.6: The Student-t distribution's tails
# #############################################################################


def plot_student_t_sweep(
    dfs: Sequence[float] = (0.1, 0.5, 1, 2, 5, 10, 30),
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the Student-t PDF for several degrees of freedom against a Gaussian.

    :param dfs: degrees of freedom `nu` to sweep over
    :param figsize: optional figure size
    """
    # Points to be used to sample the PDF.
    x_values = np.linspace(-10, 10, 500)
    _, ax = plt.subplots(figsize=figsize)
    for df in dfs:
        # Student-t with df degrees of freedom.
        distr = stats.t(df)
        x_pdf = distr.pdf(x_values)
        ax.plot(x_values, x_pdf, label=f"nu={df}")
    # Plot the Gaussian limit (nu -> infinity) for comparison.
    x_pdf = stats.norm.pdf(x_values)
    ax.plot(x_values, x_pdf, "k--", label="Gauss / nu=infty")
    ax.set_xlim(-5, 5)
    ax.legend()
