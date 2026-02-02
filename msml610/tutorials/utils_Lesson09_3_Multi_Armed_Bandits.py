"""
Utility functions for Multi-Armed Bandits lesson.

Import as:

import msml610.tutorials.utils_Lesson09_3_Multi_Armed_Bandits as mtulmaba
"""

import logging
import textwrap
from typing import List, Tuple

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

import helpers.hdbg as hdbg
import msml610_utils as ut

_LOG = logging.getLogger(__name__)


# #############################################################################
# Cell 1: Introduction - Casino Slot Machines
# #############################################################################


def cell1_casino_slot_machines() -> None:
    """
    Interactive casino slot machine visualization.

    Display 3 slot machines with fixed true means generating random rewards.
    User can:
    - Choose which machine to play
    - Toggle showing true means
    - Reset total winnings and coin budget
    """
    # Initialize state.
    state = {
        "total_winnings": 0.0,
        "coins_remaining": 10,
        "initial_coins": 10,
        "machine_results": ["?", "?", "?"],
        "machine_rewards": [[], [], []],  # Store all rewards for each machine.
        "true_means": [-0.2, 0.0, 0.5],
        "show_true_means": False,
    }

    # Create seed widget.
    seed_slider, seed_box = ut.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )

    # Create coins widget.
    coins_slider, coins_box = ut.build_widget_control(
        name="coins",
        description="number of coins",
        min_val=5,
        max_val=50,
        step=1,
        initial_value=10,
        is_float=False,
    )

    # Create widgets for machine selection.
    machine_selector = ipywidgets.Dropdown(
        options=["Machine 1", "Machine 2", "Machine 3"],
        value="Machine 1",
        description="Select Machine:",
        style={"description_width": "120px"},
    )

    # Create widgets for showing true means.
    show_means_toggle = ipywidgets.Checkbox(
        value=False,
        description="Show True Means",
        style={"description_width": "120px"},
    )

    # Create action buttons.
    play_button = ipywidgets.Button(
        description="Play Selected Machine",
        button_style="success",
        layout={"width": "200px"},
    )

    reset_button = ipywidgets.Button(
        description="Reset Game",
        button_style="warning",
        layout={"width": "200px"},
    )

    # Output widget for plots.
    output = ipywidgets.Output()

    def update_plot() -> None:
        """Update the visualization."""
        with output:
            clear_output(wait=True)

            # Create figure with subplots.
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Plot slot machines.
            for i in range(3):
                ax = axes[i]
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")

                # Draw slot machine frame.
                machine_rect = plt.Rectangle(
                    (0.1, 0.2),
                    0.8,
                    0.6,
                    linewidth=3,
                    edgecolor="black",
                    facecolor="lightgray",
                )
                ax.add_patch(machine_rect)

                # Display result or question mark.
                result_text = state["machine_results"][i]
                ax.text(
                    0.5,
                    0.5,
                    result_text,
                    ha="center",
                    va="center",
                    fontsize=32,
                    weight="bold",
                )

                # Machine label.
                ax.text(
                    0.5,
                    0.9,
                    f"Machine {i+1}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    weight="bold",
                )

                # Calculate and display sample mean and number of pulls.
                rewards = state["machine_rewards"][i]
                num_pulls = len(rewards)
                if num_pulls > 0:
                    sample_mean = np.mean(rewards)
                    stats_text = f"n={num_pulls}, mean={sample_mean:.2f}"
                else:
                    stats_text = "n=0, mean=?"

                ax.text(
                    0.5,
                    0.05,
                    stats_text,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="blue",
                    weight="bold",
                )

                # Show true mean if enabled.
                if state["show_true_means"]:
                    ax.text(
                        0.5,
                        -0.05,
                        f"true mu={state['true_means'][i]:.2f}",
                        ha="center",
                        va="top",
                        fontsize=9,
                        color="red",
                        weight="bold",
                    )

            # Add game status as title.
            fig.suptitle(
                f"Total Winnings: {state['total_winnings']:.2f} | "
                f"Coins Remaining: {state['coins_remaining']}",
                fontsize=16,
                weight="bold",
            )

            plt.tight_layout()
            plt.show()

    def on_play_clicked(b) -> None:
        """Handle play button click."""
        if state["coins_remaining"] <= 0:
            _LOG.warning("No coins remaining!")
            return

        # Set random seed.
        np.random.seed(seed_slider.value)

        # Get selected machine index (0, 1, 2).
        machine_idx = int(machine_selector.value.split()[-1]) - 1

        # Generate reward from machine (sample from distribution centered at true mean).
        true_mean = state["true_means"][machine_idx]
        # Use normal distribution with std=0.3 clipped to [-1, 1].
        reward = np.clip(
            np.random.normal(true_mean, 0.3),
            -1.0,
            1.0,
        )

        # Update state.
        state["total_winnings"] += reward
        state["coins_remaining"] -= 1
        state["machine_results"][machine_idx] = f"{reward:.2f}"
        state["machine_rewards"][machine_idx].append(reward)

        # Increment seed for next play.
        seed_slider.value = seed_slider.value + 1

        # Update plot.
        update_plot()

    def on_reset_clicked(b) -> None:
        """Handle reset button click."""
        state["total_winnings"] = 0.0
        state["initial_coins"] = coins_slider.value
        state["coins_remaining"] = coins_slider.value
        state["machine_results"] = ["?", "?", "?"]
        state["machine_rewards"] = [[], [], []]
        update_plot()

    def on_show_means_changed(change) -> None:
        """Handle toggle for showing true means."""
        state["show_true_means"] = change["new"]
        update_plot()

    # Connect callbacks.
    play_button.on_click(on_play_clicked)
    reset_button.on_click(on_reset_clicked)
    show_means_toggle.observe(on_show_means_changed, names="value")

    # Layout widgets.
    controls = ipywidgets.VBox(
        [
            seed_box,
            coins_box,
            show_means_toggle,
            machine_selector,
            ipywidgets.HBox([play_button, reset_button]),
        ]
    )

    # Display widgets and initial plot.
    display(controls, output)
    update_plot()
