"""
Utility functions for MSML610 course tutorials.

Import as:

import msml610.tutorials.L09_05_01_discrete_bayes_dog_utils as mtl00dbdu
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

from ipywidgets import Dropdown, VBox, interactive_output
from filterpy.discrete_bayes import predict, update
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

import helpers.hdbg as hdbg
import msml610_utils as mtumsuti

_LOG = logging.getLogger(__name__)

HALLWAY_LEN = 10

def get_hallway1() -> np.ndarray:
    hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    hdbg.dassert_eq(len(hallway), HALLWAY_LEN)
    return hallway


def get_hallway2() -> np.ndarray:
    hallway = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    hdbg.dassert_eq(len(hallway), HALLWAY_LEN)
    return hallway


def plot_belief(
    belief: np.ndarray,
    *,
    title: str = "Belief",
    y_lim: Tuple[float, float] = (0, 1),
    ax: Optional[plt.Axes] = None,
    hallway: np.ndarray = None,
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
        hdbg.dassert_is_not(hallway, None)
        doors = np.where(hallway == 1)[0]
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
    hallway: np.ndarray = None,
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
        hdbg.dassert_eq(len(belief2), n)
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
            hdbg.dassert_is_not(hallway, None)
            doors = np.where(hallway == 1)[0]
            for d in doors:
                ax.axvline(
                    d, color="red", linestyle="--", linewidth=2, alpha=0.8
                )
        plt.tight_layout()


# ############################################################################
# Cell 2
# ############################################################################

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
    """
    The dog runs around the office for 50 steps.
    """
    positions = [i % HALLWAY_LEN for i in range(50)]
    return positions


def get_dog_movements2() -> List[int]:
    """
    The dog runs between door 1 and door 2 for 50 steps.
    """
    positions = [0, 1] * 25
    return positions


def get_dog_movements3() -> List[int]:
    """
    The dog runs between positions 0 and 3 for 12 steps, then back for 12 steps.
    """
    positions = [0, 0] + [0, 1, 2, 3, 3, 2, 1, 0] * 6
    return positions


def get_sensor_info(positions: List[int], hallway: np.ndarray) -> Dict[str, List]:
    """
    Get the movements of the dog and the measurements.
    """
    z_doors = [hallway[z] for z in positions]
    z_moves = [0] + [positions[i] - positions[i-1] for i in range(1, len(positions))]
    sensor_info = {
        "positions": positions,
        "z_doors": z_doors,
        "z_moves": z_moves,
    }
    dassert_sensor_info(sensor_info)
    return sensor_info


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
    hallway: np.ndarray, posteriors: list, i: int, positions: List[int]
) -> None:
    """
    Plot posterior belief at step i with dog position marker.

    :param hallway: Map of the hallway (0=wall, 1=door)
    :param posteriors: List of posterior belief distributions
    :param i: Time step index
    :param positions: List of actual dog positions
    """
    plot_belief(
        posteriors[i],
        title="Posterior",
        y_lim=(0, 1.0),
        use_hallway=True,
    )
    # Mark current dog position.
    plt.axvline(positions[i], color="green", linewidth=5, alpha=0.5)
    plt.show()


def plot_prior(hallway: np.ndarray, priors: list, i: int, positions: List[int]) -> None:
    """
    Plot prior belief at step i with dog position marker.

    :param hallway: Map of the hallway (0=wall, 1=door)
    :param priors: List of prior belief distributions
    :param i: Time step index
    :param positions: List of actual dog positions
    """
    plot_belief(
        priors[i],
        title="Prior",
        y_lim=(0, 1.0),
        use_hallway=True,
    )
    # Mark current dog position.
    plt.axvline(positions[i], color="green", linewidth=5, alpha=0.5)
    plt.show()


def animate_discrete_bayes(
    hallway: np.ndarray, priors: list, posteriors: list, sensor_info: Dict[str, List]
):
    """
    Create animation function for discrete Bayes filter.

    Returns a function that alternates between plotting priors and
    posteriors as the step parameter changes.

    :param hallway: Map of the hallway (0=wall, 1=door)
    :param priors: List of prior belief distributions
    :param posteriors: List of posterior belief distributions
    :param sensor_info: Dictionary containing dog positions and measurements
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
            plot_prior(hallway, priors, i, sensor_info["positions"])
        else:
            plot_posterior(hallway, posteriors, i, sensor_info["positions"])

    return animate


def plot_dog_movement(
    positions: List[int],
    step_idx: int,
    *,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Plot dog movement trajectory with current position highlighted.

    :param positions: List of dog positions over time
    :param step_idx: Current step index to highlight
    :param ax: The axis to plot on
    """
    if ax is None:
        _, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    # Plot all positions as a line.
    steps = range(len(positions))
    ax.plot(steps, positions, marker="o", linestyle="-", color="blue", alpha=0.5)
    # Highlight current position.
    if step_idx < len(positions):
        ax.plot(
            step_idx,
            positions[step_idx],
            marker="o",
            markersize=12,
            color="green",
            alpha=0.8,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Position")
    ax.set_title("Dog Movement")
    ax.set_ylim(-0.5, HALLWAY_LEN - 0.5)
    ax.set_yticks(range(HALLWAY_LEN))
    ax.grid(True, alpha=0.3)


def animate_discrete_bayes_with_movement(
    hallway: np.ndarray,
    priors: list,
    posteriors: list,
    sensor_info: Dict[str, List],
):
    """
    Create animation function for discrete Bayes filter with movement plot.

    Returns a function that alternates between plotting priors and
    posteriors as the step parameter changes, along with a plot showing
    the dog's movement trajectory.

    :param hallway: Map of the hallway (0=wall, 1=door)
    :param priors: List of prior belief distributions
    :param posteriors: List of posterior belief distributions
    :param sensor_info: Dictionary containing dog positions and measurements
    :return: Animation function for use with ipywidgets
    """

    def animate(step: int) -> None:
        """
        Display belief distribution and movement for given step.

        :param step: Step number (1-indexed, alternates between prior/posterior)
        """
        step -= 1
        i = step // 2
        # Create stacked plots (2 rows, 1 column).
        figsize = copy.deepcopy(plt.rcParams["figure.figsize"])
        figsize[1] = figsize[1] * 2
        _, axes = plt.subplots(2, 1, figsize=figsize)
        # Plot belief (prior or posterior).
        if step % 2 == 0:
            plot_belief(
                priors[i],
                title="Prior",
                y_lim=(0, 1.0),
                ax=axes[0],
                hallway=hallway,
                use_hallway=True,
            )
            # Mark current dog position.
            axes[0].axvline(
                sensor_info["positions"][i], color="green", linewidth=5, alpha=0.5
            )
        else:
            plot_belief(
                posteriors[i],
                title="Posterior",
                y_lim=(0, 1.0),
                ax=axes[0],
                hallway=hallway,
                use_hallway=True,
            )
            # Mark current dog position.
            axes[0].axvline(
                sensor_info["positions"][i], color="green", linewidth=5, alpha=0.5
            )
        # Plot dog movement.
        plot_dog_movement(sensor_info["positions"], i, ax=axes[1])
        plt.tight_layout()
        plt.show()

    return animate


def cell2_interactive() -> None:
    """
    Interactive visualization of discrete Bayes filter tracking a dog.

    Creates an interactive widget that animates the belief update process
    as the dog moves through a hallway with noisy sensors. Includes controls
    for movement function, initial prior, and sensor probability.
    """
    hallway = get_hallway1()
    # Create widgets for controls.
    # Movement function selector.
    movement_dropdown = Dropdown(
        options={
            "Movement 1 (around office)": "get_dog_movements1",
            "Movement 2 (between doors)": "get_dog_movements2",
            "Movement 3 (between positions 0 and 3 and back)": "get_dog_movements3",
        },
        value="get_dog_movements1",
        description="Movement:",
        style={"description_width": "initial"},
    )
    # Prior distribution selector.
    prior_dropdown = Dropdown(
        options={
            "Flat (uniform)": "flat",
            "All in position 3": "position_3",
            "All in position 8": "position_8",
        },
        value="flat",
        description="Initial Prior:",
        style={"description_width": "initial"},
    )
    # z_prob slider with +/- buttons.
    z_prob_slider, z_prob_box = mtumsuti.build_widget_control(
        name="z_prob",
        description="Sensor accuracy",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=1.0,
        is_float=True,
    )
    # Step slider with +/- buttons.
    step_slider, step_box = mtumsuti.build_widget_control(
        name="step",
        description="Animation step",
        min_val=1,
        max_val=100,
        step=1,
        initial_value=1,
        is_float=False,
    )

    def update_visualization(movement_func: str, prior_type: str, z_prob: float, step: int) -> None:
        """
        Update the visualization based on widget values.

        :param movement_func: Name of movement function to use
        :param prior_type: Type of initial prior distribution
        :param z_prob: Sensor measurement probability
        :param step: Current step in the animation
        """
        # Change these numbers to alter the simulation.
        kernel = (0.1, 0.8, 0.1)
        # Get initial prior based on selection.
        if prior_type == "flat":
            prior = np.array([0.1] * 10)
        elif prior_type == "position_3":
            prior = np.zeros(10)
            prior[3] = 1.0
        elif prior_type == "position_8":
            prior = np.zeros(10)
            prior[8] = 1.0
        else:
            raise ValueError(f"Invalid prior type: {prior_type}")
        # Get positions based on movement function.
        if movement_func == "get_dog_movements1":
            positions = get_dog_movements1()
        elif movement_func == "get_dog_movements2":
            positions = get_dog_movements2()
        elif movement_func == "get_dog_movements3":
            positions = get_dog_movements3()
        else:
            raise ValueError(f"Invalid movement function: {movement_func}")
        sensor_info = get_sensor_info(positions, hallway)
        # Update step slider max value.
        step_slider.max = len(sensor_info["z_doors"]) * 2
        # Run simulation.
        priors, posteriors = discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Animate.
        animate_fn = animate_discrete_bayes_with_movement(
            hallway, priors, posteriors, sensor_info
        )
        animate_fn(step)

    # Create interactive output.
    output = interactive_output(
        update_visualization,
        {
            "movement_func": movement_dropdown,
            "prior_type": prior_dropdown,
            "z_prob": z_prob_slider,
            "step": step_slider,
        },
    )
    # Display widgets above plots.
    controls = VBox([movement_dropdown, prior_dropdown, z_prob_box, step_box])
    display(VBox([controls, output]))