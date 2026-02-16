"""
Utility functions for MSML610 course tutorials.

Import as:

import msml610.tutorials.L09_05_01_discrete_bayes_dog_utils as mtl00dbdu
"""

import copy
import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


_LOG = logging.getLogger(__name__)

# The map of the office is the following.
HALLWAY = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])


def plot_belief(
    belief: np.ndarray,
    *,
    title: str = "Belief",
    y_lim: Tuple[float, float] = (0, 1),
    ax: Optional[plt.Axes] = None,
    use_hallway: bool = True,
) -> None:
    """
    Plot a belief distribution.

    :param belief: The belief array
    :param ax: The axis to plot on
    :param title: The title for the belief distribution
    :param use_hallway: If True, mark door positions from HALLWAY constant
    """
    if ax is None:
        _, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    ax.bar(range(len(belief)), belief, color="#1F77B4")
    ax.set_ylim(y_lim)
    # Show all integers on x-axis.
    ax.set_xticks(range(len(belief)))
    # Grid lines at every integer (vertical).
    ax.grid(
        axis="x", color="gray", linestyle="-", linewidth=0.8, alpha=0.6
    )
    # Horizontal grid lines.
    ax.grid(
        axis="y", color="gray", linestyle="--", linewidth=0.8, alpha=0.5
    )
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    # Mark door positions.
    if use_hallway:
        doors = np.where(HALLWAY == 1)[0]
        for d in doors:
            ax.axvline(
                d, color="red", linestyle="--", linewidth=2, alpha=0.8
            )


def plot_beliefs(
    belief1: np.ndarray,
    belief2: np.ndarray,
    *,
    title1: str = "Belief1",
    title2: str = "Belief2",
    y_lim: Tuple[float, float] = (0, 1),
    same_plot: bool = True,
    use_hallway: bool = True,
) -> None:
    """
    Plot two belief distributions, either side by side or together as bars
    with different colors and a legend.

    :param belief1: The first belief array
    :param belief2: The second belief array
    :param title1: The title for the first belief distribution (also used
        as label if same_plot=True)
    :param title2: The title for the second belief distribution (also used
        as label if same_plot=True)
    :param y_lim: The limits for the y-axis
    :param same_plot: If True, show both beliefs on the same axes with
        legend
    :param use_hallway: If True, mark door positions from HALLWAY constant
    """
    if not same_plot:
        figsize = copy.deepcopy(plt.rcParams["figure.figsize"])
        figsize[0] = figsize[0] * 2
        _, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        plot_belief(
            belief1,
            title=title1,
            y_lim=y_lim,
            ax=axes[0],
            use_hallway=use_hallway,
        )
        plot_belief(
            belief2,
            title=title2,
            y_lim=y_lim,
            ax=axes[1],
            use_hallway=use_hallway,
        )
        plt.tight_layout()
    else:
        n = len(belief1)
        # TODO(ai_gp): Use a dassert.
        if len(belief2) != n:
            raise ValueError(
                "belief1 and belief2 must have the same length."
            )
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
        # No global main_title (no plt.suptitle); use legend with labels.
        ax.grid(
            axis="x",
            color="gray",
            linestyle="-",
            linewidth=0.8,
            alpha=0.6,
            zorder=1,
        )
        ax.grid(
            axis="y",
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            zorder=1,
        )
        ax.legend()
        # Mark door positions.
        if use_hallway:
            doors = np.where(HALLWAY == 1)[0]
            for d in doors:
                ax.axvline(
                    d, color="red", linestyle="--", linewidth=2, alpha=0.8
                )
        plt.tight_layout()


# #########################################################
# Cell 2
# #########################################################

from filterpy.discrete_bayes import predict, update


def discrete_bayes_sim(prior, kernel, measurements, z_prob, hallway):
    posterior = np.array([.1]*10)
    priors, posteriors = [], []
    for i, z in enumerate(measurements):
        prior = predict(posterior, 1, kernel)
        priors.append(prior)

        likelihood = lh_hallway(hallway, z, z_prob)
        posterior = update(likelihood, prior)
        posteriors.append(posterior)
    return priors, posteriors


def plot_posterior(hallway, posteriors, i):
    plt.title('Posterior')
    book_plots.bar_plot(hallway, c='k')
    book_plots.bar_plot(posteriors[i], ylim=(0, 1.0))
    plt.axvline(i % len(hallway), lw=5)
    plt.show()
    
def plot_prior(hallway, priors, i):
    plt.title('Prior')
    book_plots.bar_plot(hallway, c='k')
    book_plots.bar_plot(priors[i], ylim=(0, 1.0), c='#ff8015')
    plt.axvline(i % len(hallway), lw=5)
    plt.show()

def animate_discrete_bayes(hallway, priors, posteriors):
    def animate(step):
        step -= 1
        i = step // 2    
        if step % 2 == 0:
            plot_prior(hallway, priors, i)
        else:
            plot_posterior(hallway, posteriors, i)
    
    return animate


def cell2_interactive():
    # change these numbers to alter the simulation
    kernel = (.1, .8, .1)
    z_prob = 1.0
    hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])

    # measurements with no noise
    zs = [hallway[i % len(hallway)] for i in range(50)]

    priors, posteriors = discrete_bayes_sim(prior, kernel, zs, z_prob, hallway)
    interact(animate_discrete_bayes(hallway, priors, posteriors), step=IntSlider(value=1, max=len(zs)*2));