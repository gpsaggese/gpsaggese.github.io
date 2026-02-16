"""
Utility functions for MSML610 course tutorials.

Import as:

import msml610.tutorials.L09_05_01_discrete_bayes_dog_utils as mtl00dbdu
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import helpers.hdbg as hdbg

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
        # Assert that both beliefs have the same length.
        assert len(belief2) == n, (
            f"belief1 and belief2 must have the same length. "
            f"Got {n} and {len(belief2)}"
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


# ############################################################################
# Cell 2
# ############################################################################

from filterpy.discrete_bayes import predict, update
from ipywidgets import IntSlider, interact


def lh_hallway(hall: np.ndarray, z: int, z_prob: float) -> np.ndarray:
    """
    Compute likelihood that a measurement matches positions in the hallway.

    Creates a likelihood array where positions matching the measurement z
    are scaled according to the measurement probability.

    :param hall: Array representing the hallway map (0=wall, 1=door)
    :param z: Measurement value (0 or 1)
    :param z_prob: Probability that the measurement is correct
    :return: Likelihood array for all positions
    """
    try:
        scale = z_prob / (1.0 - z_prob)
    except ZeroDivisionError:
        scale = 1e8
    likelihood = np.ones(len(hall))
    likelihood[hall == z] *= scale
    return likelihood

    
def dassert_sensor_info(sensor_info: Dict[str, List]) -> None:
    hdbg.dassert_eq(len(sensor_info["positions"]), len(sensor_info["z_moves"]))
    hdbg.dassert_eq(len(sensor_info["positions"]), len(sensor_info["z_doors"]))


def get_dog_movements1() -> List[int]:
    positions = [i % len(HALLWAY) for i in range(50)]
    return positions


def get_sensor_info(positions: List[int]) -> Dict[str, List]:
    """
    Get the movements of the dog and the measurements.

    The dog moves 1 position to the right at each step.
    """
    z_doors = [HALLWAY[z] for z in positions]
    z_moves = [0] + [positions[i] - positions[i-1] for i in range(1, len(positions))]
    sensor_info = {
        "positions": positions,
        "z_doors": z_doors,
        "z_moves": z_moves,
    }
    dassert_sensor_info(sensor_info)
    return z_moves, z_doors


def discrete_bayes_sim(
    prior: np.ndarray,
    # TODO(gp): door_sensor_prob
    kernel: tuple,
    sensor_info: Dict[str, List],
    # 
    z_prob: float,
    hallway: np.ndarray,
) -> Tuple[list, list]:
    """
    Run discrete Bayes filter simulation.

    Performs predict-update cycles for each measurement, tracking both prior and
    posterior beliefs at each step.

    :param prior: Initial belief distribution
    :param kernel: Motion model kernel (probabilities for undershoot,
        correct, overshoot)
    :param dog_info: Dictionary containing the dog's movements and measurements
    :param z_prob: Probability that sensor measurement is correct
    :return: Tuple of (priors, posteriors) lists for each time step
    """
    posterior = prior.copy()
    priors, posteriors = [], []
    dassert_sensor_info(sensor_info)
    for i in range(len(sensor_info["z_doors"])):
        # Predict step.
        prior = predict(posterior, sensor_info["z_moves"][i], kernel)
        priors.append(prior)
        # Update step.
        likelihood = lh_hallway(hallway, sensor_info["z_doors"][i], z_prob)
        posterior = update(likelihood, prior)
        posteriors.append(posterior)
    return priors, posteriors


def plot_posterior(
    hallway: np.ndarray, posteriors: list, i: int
) -> None:
    """
    Plot posterior belief at step i with dog position marker.

    :param hallway: Map of the hallway (0=wall, 1=door)
    :param posteriors: List of posterior belief distributions
    :param i: Time step index
    """
    plot_belief(
        posteriors[i],
        title="Posterior",
        y_lim=(0, 1.0),
        use_hallway=True,
    )
    # Mark current dog position.
    plt.axvline(i % len(hallway), color="green", linewidth=5, alpha=0.5)
    plt.show()


def plot_prior(hallway: np.ndarray, priors: list, i: int) -> None:
    """
    Plot prior belief at step i with dog position marker.

    :param hallway: Map of the hallway (0=wall, 1=door)
    :param priors: List of prior belief distributions
    :param i: Time step index
    """
    plot_belief(
        priors[i],
        title="Prior",
        y_lim=(0, 1.0),
        use_hallway=True,
    )
    # Mark current dog position.
    plt.axvline(i % len(hallway), color="green", linewidth=5, alpha=0.5)
    plt.show()


def animate_discrete_bayes(
    hallway: np.ndarray, priors: list, posteriors: list
):
    """
    Create animation function for discrete Bayes filter.

    Returns a function that alternates between plotting priors and
    posteriors as the step parameter changes.

    :param hallway: Map of the hallway (0=wall, 1=door)
    :param priors: List of prior belief distributions
    :param posteriors: List of posterior belief distributions
    :return: Animation function for use with ipywidgets
    """

    def animate(step: int) -> None:
        """
        Display belief distribution for given step.

        :param step: Step number (1-indexed, alternates between prior/posterior)
        """
        step -= 1
        i = step // 2
        if step % 2 == 0:
            plot_prior(hallway, priors, i)
        else:
            plot_posterior(hallway, posteriors, i)

    return animate


def cell2_interactive() -> None:
    """
    Interactive visualization of discrete Bayes filter tracking a dog.

    Creates an interactive widget that animates the belief update process
    as the dog moves through a hallway with noisy sensors.
    """
    # Change these numbers to alter the simulation.
    kernel = (0.1, 0.8, 0.1)
    z_prob = 1.0
    # Initial uniform belief.
    prior = np.array([0.1] * 10)
    # Get the sensor info.
    positions = get_dog_movements1()
    sensor_info = get_sensor_info(positions)
    # Run simulation.
    priors, posteriors = discrete_bayes_sim(
        prior, kernel, sensor_info, z_prob, HALLWAY
    )
    # Create interactive widget.
    interact(
        animate_discrete_bayes(HALLWAY, priors, posteriors),
        step=IntSlider(value=1, max=len(sensor_info["z_doors"]) * 2),
    )