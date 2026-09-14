"""
Utility functions for Multi-Armed Bandits lesson.

Import as:

import msml610.tutorials.L09_multi_armed_bandits.L09_03_multi_armed_bandits_utils as mtlmabl0mabu
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

import helpers.htutorial as htutori
import L09_03_multi_armed_bandits_sim as sim

_LOG = logging.getLogger(__name__)


def _beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Compute the Beta(alpha, beta) PDF at points x, without a scipy dependency.

    f(x; alpha, beta) = x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta), where
    B(alpha, beta) = Gamma(alpha) * Gamma(beta) / Gamma(alpha + beta).

    :param x: points in [0, 1] at which to evaluate the PDF
    :param alpha: Beta shape parameter
    :param beta: Beta shape parameter
    :return: PDF values at `x`
    """
    x = np.asarray(x, dtype=float)
    const = math.gamma(alpha + beta) / (math.gamma(alpha) * math.gamma(beta))
    return const * x ** (alpha - 1) * (1 - x) ** (beta - 1)


# #############################################################################
# Cell 1: Introduction: Casino Slot Machines
# #############################################################################


def cell1_casino_slot_machines() -> None:
    """
    Interactive casino slot machine visualization.

    Display 3 slot machines with fixed true means generating random rewards.
    User can:
    - Click a machine directly to play it.
    - Toggle showing true means.
    - Reset total winnings and coin budget.
    """
    # Initialize state.
    true_means = [-0.2, 0.0, 0.5]
    state = {
        "total_winnings": 0.0,
        "coins_remaining": 10,
        "initial_coins": 10,
        "machine_last_rewards": [None, None, None],
        "bandit": None,
        "show_true_means": False,
    }
    # Create seed widget.
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    # Create coins widget.
    coins_slider, coins_box = htutori.build_widget_control(
        name="coins",
        description="number of coins",
        min_val=5,
        max_val=50,
        step=1,
        initial_value=10,
        is_float=False,
    )
    # Create widgets for showing true means.
    show_means_toggle = ipywidgets.Checkbox(
        value=False,
        description="Show True Means",
        style={"description_width": "120px"},
    )
    # Create one button per machine so the user clicks the machine directly
    # to play it, instead of picking it from a selector.
    machine_buttons = [
        ipywidgets.Button(
            description=f"Play Machine {i + 1}",
            button_style="success",
            layout={"width": "150px"},
        )
        for i in range(3)
    ]
    reset_button = ipywidgets.Button(
        description="Reset Game",
        button_style="warning",
        layout={"width": "200px"},
    )
    # Output widget for plots.
    output = ipywidgets.Output()

    def update_plot() -> None:
        """
        Update the visualization.
        """
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
                # Display a fixed placeholder (the machine face never reveals
                # the reward value; the reward is printed below the machine
                # instead).
                ax.text(
                    0.5,
                    0.5,
                    "?",
                    ha="center",
                    va="center",
                    fontsize=32,
                    weight="bold",
                )
                # Machine label.
                ax.text(
                    0.5,
                    0.9,
                    f"Machine {i + 1}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    weight="bold",
                )
                # Display the last reward obtained from this machine, below
                # the machine (not inside it).
                last_reward = state["machine_last_rewards"][i]
                if last_reward is None:
                    reward_text = "Last reward: ?"
                else:
                    reward_text = f"Last reward: {last_reward:.2f}"
                ax.text(
                    0.5,
                    0.13,
                    reward_text,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="darkgreen",
                    weight="bold",
                )
                # Calculate and display sample mean and number of pulls.
                if state["bandit"] is not None:
                    rewards = state["bandit"].machine_rewards[i]
                    num_pulls = state["bandit"].machine_pulls[i]
                    if num_pulls > 0:
                        sample_mean = np.mean(rewards)
                        stats_text = f"n={num_pulls}, mean={sample_mean:.2f}"
                    else:
                        stats_text = "n=0, mean=?"
                else:
                    stats_text = "n=0, mean=?"
                ax.text(
                    0.5,
                    0.03,
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
                        -0.08,
                        f"true mu={true_means[i]:.2f}",
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

    def make_on_play_clicked(machine_idx: int):
        """
        Build a click handler that plays a specific machine.

        :param machine_idx: index of the machine this button plays (0 to 2)
        :return: click handler for `machine_buttons[machine_idx]`
        """

        def on_play_clicked(b: ipywidgets.Button) -> None:
            """
            Handle a click on one machine's play button.

            :param b: button widget (unused)
            """
            if state["coins_remaining"] <= 0:
                _LOG.warning("No coins remaining!")
                return
            # Initialize bandit if needed.
            if state["bandit"] is None:
                state["bandit"] = sim.MultiArmedBandit(
                    k_machines=3,
                    mu_values=true_means,
                    seed=seed_slider.value,
                    width=0.3,
                )
            # Pull the machine.
            reward = state["bandit"].pull(machine_idx)
            # Update state.
            state["total_winnings"] += reward
            state["coins_remaining"] -= 1
            state["machine_last_rewards"][machine_idx] = reward
            # Increment seed for next play.
            seed_slider.value = seed_slider.value + 1
            # Update plot.
            update_plot()

        return on_play_clicked

    def on_reset_clicked(b: ipywidgets.Button) -> None:
        """
        Handle reset button click.

        :param b: button widget (unused)
        """
        state["total_winnings"] = 0.0
        state["initial_coins"] = coins_slider.value
        state["coins_remaining"] = coins_slider.value
        state["machine_last_rewards"] = [None, None, None]
        # Reset bandit with current seed.
        state["bandit"] = sim.MultiArmedBandit(
            k_machines=3,
            mu_values=true_means,
            seed=seed_slider.value,
            width=0.3,
        )
        update_plot()

    def on_show_means_changed(change: dict) -> None:
        """
        Handle toggle for showing true means.

        :param change: dictionary with change information
        """
        state["show_true_means"] = change["new"]
        update_plot()

    # Connect callbacks.
    for machine_idx, button in enumerate(machine_buttons):
        button.on_click(make_on_play_clicked(machine_idx))
    reset_button.on_click(on_reset_clicked)
    show_means_toggle.observe(on_show_means_changed, names="value")
    # Layout widgets.
    controls = ipywidgets.VBox(
        [
            seed_box,
            coins_box,
            show_means_toggle,
            ipywidgets.HBox(machine_buttons + [reset_button]),
        ]
    )
    # Display widgets and initial plot.
    display(controls, output)
    update_plot()


# TODO(ai_gp): Make it private, if possible.
def plot_epsilon_sweep(
    *,
    sweep_results: dict,
    n_coins: int,
) -> None:
    """
    Plot comparison of strategies across epsilon values.

    :param sweep_results: results from BanditSimulation.epsilon_sweep()
    :param n_coins: number of coins used in simulation
    """
    epsilon_values = sweep_results["epsilon_values"]
    exploration = sweep_results["exploration"]
    exploitation = sweep_results["exploitation"]
    balanced_list = sweep_results["balanced"]
    # Create figure with 2 subplots.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Plot 1: Mean final reward vs epsilon.
    ax = axes[0]
    balanced_means = [b["mean_final"] for b in balanced_list]
    balanced_stds = [b["std_final"] for b in balanced_list]
    ax.errorbar(
        epsilon_values,
        balanced_means,
        yerr=balanced_stds,
        label="Epsilon-Greedy",
        marker="o",
        linewidth=2,
        capsize=5,
    )
    ax.axhline(
        exploration["mean_final"],
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Pure Exploration (mean={exploration['mean_final']:.2f})",
    )
    ax.axhline(
        exploitation["mean_final"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Pure Exploitation (mean={exploitation['mean_final']:.2f})",
    )
    ax.set_xlabel("Epsilon", fontsize=12)
    ax.set_ylabel("Mean Final Reward", fontsize=12)
    ax.set_title("Strategy Performance vs Epsilon", fontsize=14, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    # Plot 2: Cumulative reward over time for best epsilon.
    ax = axes[1]
    best_idx = np.argmax(balanced_means)
    best_epsilon = epsilon_values[best_idx]
    best_balanced = balanced_list[best_idx]
    # Plot mean with error bands.
    trials = np.arange(1, n_coins + 1)
    ax.plot(
        trials,
        exploration["mean_cumulative"],
        label="Pure Exploration",
        color="blue",
        linewidth=2,
        alpha=0.8,
    )
    ax.fill_between(
        trials,
        exploration["mean_cumulative"] - exploration["std_cumulative"],
        exploration["mean_cumulative"] + exploration["std_cumulative"],
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        trials,
        exploitation["mean_cumulative"],
        label="Pure Exploitation",
        color="red",
        linewidth=2,
        alpha=0.8,
    )
    ax.fill_between(
        trials,
        exploitation["mean_cumulative"] - exploitation["std_cumulative"],
        exploitation["mean_cumulative"] + exploitation["std_cumulative"],
        color="red",
        alpha=0.2,
    )
    ax.plot(
        trials,
        best_balanced["mean_cumulative"],
        label=f"Balanced (epsilon={best_epsilon:.1f})",
        color="green",
        linewidth=2,
        alpha=0.8,
    )
    ax.fill_between(
        trials,
        best_balanced["mean_cumulative"] - best_balanced["std_cumulative"],
        best_balanced["mean_cumulative"] + best_balanced["std_cumulative"],
        color="green",
        alpha=0.2,
    )
    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("Mean Cumulative Reward", fontsize=12)
    ax.set_title(
        "Best Strategy Performance Over Time", fontsize=14, weight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()


# #############################################################################
# Cell 2: Exploration vs Exploitation Dilemma
# #############################################################################


def cell2_exploration_vs_exploitation() -> None:
    """
    Demonstrate exploration vs exploitation tradeoff with three strategies.

    Visualize three strategies:
    - Pure exploration: randomly select machines.
    - Pure exploitation: always select best known machine.
    - Balanced (epsilon-greedy): explore with probability epsilon.

    Show cumulative rewards over multiple trials to compare performance.
    """
    # Initialize state.
    true_means = [-0.2, 0.0, 0.5]
    state = {
        "num_machines": 3,
        "num_trials": 100,
    }

    # Create seed widget.
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )

    # Create coins widget.
    coins_slider, coins_box = htutori.build_widget_control(
        name="coins",
        description="number of coins",
        min_val=10,
        max_val=200,
        step=10,
        initial_value=100,
        is_float=False,
    )

    # Create epsilon slider.
    epsilon_slider, epsilon_box = htutori.build_widget_control(
        name="epsilon",
        description="exploration probability",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.1,
        is_float=True,
    )

    # Output widget for plots.
    output = ipywidgets.Output()

    def run_strategy_experiment(
        num_coins: int,
        seed: int,
        strategy: sim.Strategy,
    ) -> Tuple[List[float], List[float]]:
        """
        Run experiment with given strategy.

        :param num_coins: number of coins to play
        :param seed: random seed
        :param strategy: Strategy instance to use
        :return: (rewards, cumulative_rewards)
        """
        bandit = sim.MultiArmedBandit(
            k_machines=state["num_machines"],
            mu_values=true_means,
            seed=seed,
            width=0.3,
        )
        experiment = sim.BanditExperiment(
            bandit=bandit,
            strategy=strategy,
            n_coins=num_coins,
        )
        rewards, cumulative_rewards, _ = experiment.run()
        return rewards, cumulative_rewards

    def update_plot() -> None:
        """
        Update the visualization showing all three strategies.
        """
        with output:
            clear_output(wait=True)

            # Run all three strategies.
            seed = seed_slider.value
            num_coins = coins_slider.value
            epsilon = epsilon_slider.value

            # Create strategies.
            exploration_strategy = sim.ExplorationStrategy(seed=seed)
            exploitation_strategy = sim.ExploitationStrategy()
            balanced_strategy = sim.EpsilonGreedyStrategy(
                epsilon=epsilon,
                seed=seed + 2,
            )

            # Run experiments.
            _, exploration_cumulative = run_strategy_experiment(
                num_coins, seed, exploration_strategy
            )
            _, exploitation_cumulative = run_strategy_experiment(
                num_coins, seed + 1, exploitation_strategy
            )
            _, balanced_cumulative = run_strategy_experiment(
                num_coins, seed + 2, balanced_strategy
            )

            # Create figure with 2 subplots.
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: All three strategies on same plot.
            ax = axes[0]
            ax.plot(
                range(1, num_coins + 1),
                exploration_cumulative,
                label="Pure Exploration (Random)",
                color="blue",
                linewidth=2,
                alpha=0.8,
            )
            ax.plot(
                range(1, num_coins + 1),
                exploitation_cumulative,
                label="Pure Exploitation (Greedy)",
                color="red",
                linewidth=2,
                alpha=0.8,
            )
            ax.plot(
                range(1, num_coins + 1),
                balanced_cumulative,
                label=f"Balanced (epsilon={epsilon:.2f})",
                color="green",
                linewidth=2,
                alpha=0.8,
            )
            ax.set_xlabel("Trial", fontsize=12)
            ax.set_ylabel("Cumulative Reward", fontsize=12)
            ax.set_title(
                "Exploration vs Exploitation Strategies",
                fontsize=14,
                weight="bold",
            )
            ax.set_ylim(0, num_coins)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=10)

            # Plot 2: Comments box comparing all strategies.
            ax = axes[1]
            ax.axis("off")

            # Calculate final rewards.
            final_exploration = exploration_cumulative[-1]
            final_exploitation = exploitation_cumulative[-1]
            final_balanced = balanced_cumulative[-1]

            # Create comparison text.
            comment_lines = [
                "Strategy Comparison",
                "=" * 35,
                "",
                f"Pure Exploration: {final_exploration:.2f}",
                "- Tries all machines randomly",
                "- Learns but earns little",
                "",
                f"Pure Exploitation: {final_exploitation:.2f}",
                "- Sticks with first good option",
                "- Can get stuck on suboptimal choice",
                "",
                f"Balanced (epsilon={epsilon:.2f}): {final_balanced:.2f}",
                f"- Explores {epsilon * 100:.0f}% of time",
                f"- Exploits {(1 - epsilon) * 100:.0f}% of time",
                "- Balance is key!",
                "",
                "True means:",
                f"Machine 1: {true_means[0]:.2f}",
                f"Machine 2: {true_means[1]:.2f}",
                f"Machine 3: {true_means[2]:.2f} (best)",
            ]

            comment_text = "\n".join(comment_lines)
            ax.text(
                0.05,
                0.95,
                comment_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()
            plt.show()

    def on_widget_change(change) -> None:
        """
        Handle widget value changes.
        """
        update_plot()

    # Connect callbacks to automatically update when widgets change.
    seed_slider.observe(on_widget_change, names="value")
    coins_slider.observe(on_widget_change, names="value")
    epsilon_slider.observe(on_widget_change, names="value")

    # Layout widgets.
    controls = ipywidgets.VBox(
        [
            seed_box,
            coins_box,
            epsilon_box,
        ]
    )

    # Display widgets and initial plot.
    display(controls, output)
    update_plot()


# #############################################################################
# Cell 3: Greedy Algorithm Failure
# #############################################################################


def cell3_greedy_algorithm_failure() -> None:
    """
    Show the greedy algorithm getting stuck on a suboptimal arm.

    Run the greedy algorithm (`ExploitationStrategy`) on 3 Bernoulli arms with
    true means mu = (0.4, 0.7, 0.5). User can:
    - Set the seed and number of coins.
    - Click "Run Greedy Algorithm" to replay the experiment.
    """
    # True means are fixed to match the classic "greedy gets stuck" story:
    # Machine 1 (mu=0.4) is pulled first, gets lucky, and greedy never tries
    # the truly best Machine 2 (mu=0.7) again.
    true_means = [0.4, 0.7, 0.5]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    # Create seed widget (must be first, per convention).
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=200,
        step=1,
        initial_value=42,
        is_float=False,
    )
    # Create coins widget.
    coins_slider, coins_box = htutori.build_widget_control(
        name="coins",
        description="number of coins",
        min_val=10,
        max_val=50,
        step=5,
        initial_value=20,
        is_float=False,
    )
    run_button = ipywidgets.Button(
        description="Run Greedy Algorithm",
        button_style="success",
        layout={"width": "200px"},
    )
    output = ipywidgets.Output()

    def on_run_clicked(b: ipywidgets.Button) -> None:
        """
        Run the greedy algorithm and plot the resulting timeline.

        :param b: button widget (unused)
        """
        with output:
            clear_output(wait=True)
            n_coins = coins_slider.value
            # Run the greedy algorithm (pure exploitation after init).
            bandit = sim.MultiArmedBandit(
                k_machines=3,
                mu_values=true_means,
                seed=seed_slider.value,
                reward_type="bernoulli",
            )
            strategy = sim.ExploitationStrategy()
            experiment = sim.BanditExperiment(
                bandit=bandit, strategy=strategy, n_coins=n_coins
            )
            rewards, _, _ = experiment.run()
            choices = experiment.machine_choices
            rounds = list(range(1, n_coins + 1))
            # Track the empirical mean of each machine after every round
            # (NaN before the machine's first pull, so the line only starts
            # once there is data).
            running_sums = [0.0] * 3
            running_counts = [0] * 3
            empirical_means = [[np.nan] * n_coins for _ in range(3)]
            for t, (choice, reward) in enumerate(zip(choices, rewards)):
                running_sums[choice] += reward
                running_counts[choice] += 1
                for i in range(3):
                    if running_counts[i] > 0:
                        empirical_means[i][t] = (
                            running_sums[i] / running_counts[i]
                        )
            # Create figure with 3 panels: timeline, empirical means, comments.
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
            # Panel 1: timeline of which machine was pulled and its reward.
            for i in range(3):
                idx = [t for t, c in enumerate(choices) if c == i]
                ax1.scatter(
                    [rounds[t] for t in idx],
                    [rewards[t] for t in idx],
                    color=colors[i],
                    label=f"Machine {i + 1}",
                    s=60,
                    zorder=3,
                )
            ax1.set_xlabel("Round t")
            ax1.set_ylabel("Reward")
            ax1.set_yticks([0, 1])
            ax1.set_title("Pull timeline: color = machine, y = reward")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="center right", fontsize=9)
            # Panel 2: empirical mean estimate per machine over time, with the
            # (normally hidden) true means shown as light dotted lines.
            for i in range(3):
                ax2.plot(
                    rounds,
                    empirical_means[i],
                    color=colors[i],
                    linewidth=2,
                    label=f"Machine {i + 1} empirical mean",
                )
                ax2.axhline(
                    true_means[i],
                    color=colors[i],
                    linestyle=":",
                    alpha=0.5,
                )
            ax2.set_xlabel("Round t")
            ax2.set_ylabel("Empirical mean")
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_title("Empirical mean estimates (dotted = true mu)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best", fontsize=9)
            # Panel 3: comments describing the current run's outcome.
            pull_counts = [choices.count(i) for i in range(3)]
            best_machine = int(np.argmax(true_means))
            stuck = pull_counts[best_machine] == 1 and choices[0] != best_machine
            comment_lines = [
                "State:",
                f"  seed: {seed_slider.value}",
                f"  coins: {n_coins}",
                "",
                "True means:",
            ]
            for i in range(3):
                comment_lines.append(
                    f"  Machine {i + 1}: mu={true_means[i]:.2f}"
                )
            comment_lines += [
                "",
                "First 3 pulls (initialization):",
                f"  Machine 1: reward={rewards[0]:.0f}",
                f"  Machine 2: reward={rewards[1]:.0f}",
                f"  Machine 3: reward={rewards[2]:.0f}",
                "",
                "Pull counts:",
            ]
            for i in range(3):
                comment_lines.append(f"  Machine {i + 1}: {pull_counts[i]}")
            if stuck:
                comment_lines += [
                    "",
                    f"Stuck: greedy locked onto Machine {choices[0] + 1}",
                    f"after 1 pull, never revisited Machine {best_machine + 1}",
                    "(the true best arm).",
                ]
            else:
                comment_lines += [
                    "",
                    f"Greedy locked onto Machine {choices[-1] + 1}",
                    f"(true best is Machine {best_machine + 1}).",
                ]
            ax3.axis("off")
            htutori.add_fitted_text_box(
                ax3, "\n".join(comment_lines), max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    run_button.on_click(on_run_clicked)
    controls = ipywidgets.VBox([seed_box, coins_box, run_button])
    display(controls, output)
    on_run_clicked(run_button)


# #############################################################################
# Cell 4: Epsilon-Greedy Algorithm
# #############################################################################


def cell4_epsilon_greedy() -> None:
    """
    Show how epsilon-greedy balances exploration and exploitation.

    Run `EpsilonGreedyStrategy` on the same 3 Bernoulli arms as Cell 3
    (mu = (0.4, 0.7, 0.5)). User can:
    - Set the seed, number of coins, and epsilon.
    - Click "Run Epsilon-Greedy" to replay the experiment.
    """
    true_means = [0.4, 0.7, 0.5]
    action_colors = {
        "init": "gray", "explore": "tab:blue", "exploit": "tab:green"
    }
    # Create seed widget (must be first, per convention).
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=200,
        step=1,
        initial_value=42,
        is_float=False,
    )
    coins_slider, coins_box = htutori.build_widget_control(
        name="coins",
        description="number of coins",
        min_val=10,
        max_val=100,
        step=5,
        initial_value=50,
        is_float=False,
    )
    epsilon_slider, epsilon_box = htutori.build_widget_control(
        name="epsilon",
        description="exploration probability",
        min_val=0.0,
        max_val=0.5,
        step=0.05,
        initial_value=0.1,
        is_float=True,
    )
    run_button = ipywidgets.Button(
        description="Run Epsilon-Greedy",
        button_style="success",
        layout={"width": "200px"},
    )
    output = ipywidgets.Output()

    def on_run_clicked(b: ipywidgets.Button) -> None:
        """
        Run epsilon-greedy and plot the resulting timeline.

        :param b: button widget (unused)
        """
        with output:
            clear_output(wait=True)
            n_coins = coins_slider.value
            epsilon = epsilon_slider.value
            bandit = sim.MultiArmedBandit(
                k_machines=3,
                mu_values=true_means,
                seed=seed_slider.value,
                reward_type="bernoulli",
            )
            strategy = sim.EpsilonGreedyStrategy(
                epsilon=epsilon, seed=seed_slider.value + 1
            )
            experiment = sim.BanditExperiment(
                bandit=bandit, strategy=strategy, n_coins=n_coins
            )
            _, cumulative_rewards, _ = experiment.run()
            choices = experiment.machine_choices
            action_types = strategy.action_types
            rounds = list(range(1, n_coins + 1))
            # Create a 1x4 layout: timeline, pull counts, cumulative reward,
            # comments.
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
            # Panel 1: timeline colored by decision type (explore vs exploit).
            for action_type, color in action_colors.items():
                idx = [
                    t
                    for t, a in enumerate(action_types)
                    if a == action_type
                ]
                if not idx:
                    continue
                ax1.scatter(
                    [rounds[t] for t in idx],
                    [choices[t] for t in idx],
                    color=color,
                    label=action_type,
                    s=50,
                    zorder=3,
                )
            ax1.set_xlabel("Round t")
            ax1.set_ylabel("Machine pulled")
            ax1.set_yticks([0, 1, 2])
            ax1.set_yticklabels(["Machine 1", "Machine 2", "Machine 3"])
            ax1.set_title(f"Pull timeline (epsilon={epsilon:.2f})")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="best", fontsize=9)
            # Panel 2: pull counts per machine.
            pull_counts = [choices.count(i) for i in range(3)]
            ax2.bar(
                ["Machine 1", "Machine 2", "Machine 3"],
                pull_counts,
                color=["tab:blue", "tab:orange", "tab:green"],
                alpha=0.8,
            )
            ax2.set_ylabel("Number of pulls")
            ax2.set_title("Pull counts per machine")
            ax2.grid(True, alpha=0.3, axis="y")
            # Panel 3: cumulative reward over time.
            ax3.plot(rounds, cumulative_rewards, color="black", linewidth=2)
            ax3.set_xlabel("Round t")
            ax3.set_ylabel("Cumulative reward")
            ax3.set_title("Cumulative reward")
            ax3.grid(True, alpha=0.3)
            # Panel 4: comments.
            num_explore = action_types.count("explore")
            num_exploit = action_types.count("exploit")
            comment_lines = [
                "State:",
                f"  seed: {seed_slider.value}",
                f"  coins: {n_coins}",
                f"  epsilon: {epsilon:.2f}",
                "",
                "Decisions:",
                f"  init: {action_types.count('init')}",
                f"  explore: {num_explore}",
                f"  exploit: {num_exploit}",
                "",
                "Pull counts:",
            ]
            for i in range(3):
                comment_lines.append(f"  Machine {i + 1}: {pull_counts[i]}")
            comment_lines += [
                "",
                f"Cumulative reward: {cumulative_rewards[-1]:.1f}",
            ]
            ax4.axis("off")
            htutori.add_fitted_text_box(
                ax4, "\n".join(comment_lines), max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    run_button.on_click(on_run_clicked)
    controls = ipywidgets.VBox(
        [seed_box, coins_box, epsilon_box, run_button]
    )
    display(controls, output)
    on_run_clicked(run_button)


# #############################################################################
# Cell 5: Confidence Intervals for Each Arm
# #############################################################################


def _hoeffding_half_width(n: int, confidence: float) -> float:
    """
    Compute a Hoeffding confidence-interval half-width for a mean in [0, 1].

    For n iid samples in [0, 1] with confidence `1 - delta`, the sample mean
    is within `sqrt(log(2 / delta) / (2 n))` of the true mean with
    probability at least `1 - delta`.

    :param n: number of samples
    :param confidence: confidence level in (0, 1), e.g. 0.95
    :return: half-width of the confidence interval
    """
    delta = 1.0 - confidence
    return float(np.sqrt(np.log(2.0 / delta) / (2.0 * n)))


def cell5_confidence_intervals() -> None:
    """
    Show empirical means and confidence intervals shrinking with more pulls.

    Pull 3 Bernoulli arms (mu = (0.3, 0.5, 0.7)) the same number of times
    each, and display the empirical mean plus a Hoeffding confidence
    interval. User can:
    - Set the seed and number of pulls per machine.
    - Choose the confidence level (90%, 95%, 99%).
    - Toggle showing the true means.
    """
    true_means = [0.3, 0.5, 0.7]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    # Create seed widget (must be first, per convention).
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=200,
        step=1,
        initial_value=42,
        is_float=False,
    )
    n_pulls_slider, n_pulls_box = htutori.build_widget_control(
        name="N",
        description="pulls per machine",
        min_val=2,
        max_val=200,
        step=2,
        initial_value=10,
        is_float=False,
    )
    confidence_dropdown = ipywidgets.Dropdown(
        options=[("90%", 0.90), ("95%", 0.95), ("99%", 0.99)],
        value=0.95,
        description="confidence:",
        style={"description_width": "100px"},
    )
    show_means_toggle = ipywidgets.Checkbox(
        value=False,
        description="Show True Means",
        style={"description_width": "120px"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[dict] = None) -> None:
        """
        Recompute empirical means and confidence intervals, then redraw.

        :param change: dictionary with change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            n_pulls = n_pulls_slider.value
            confidence = confidence_dropdown.value
            bandit = sim.MultiArmedBandit(
                k_machines=3,
                mu_values=true_means,
                seed=seed_slider.value,
                reward_type="bernoulli",
            )
            # Pull each machine exactly n_pulls times.
            for i in range(3):
                for _ in range(n_pulls):
                    bandit.pull(i)
            empirical_means = bandit.get_empirical_means()
            half_width = _hoeffding_half_width(n_pulls, confidence)
            # Create 1x3 layout: CI bar chart, half-width vs N, comments.
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
            # Panel 1: empirical mean with confidence interval error bars.
            x_pos = np.arange(3)
            ax1.bar(
                x_pos,
                empirical_means,
                yerr=half_width,
                capsize=10,
                color=colors,
                alpha=0.8,
            )
            if show_means_toggle.value:
                for i in range(3):
                    ax1.hlines(
                        true_means[i],
                        x_pos[i] - 0.4,
                        x_pos[i] + 0.4,
                        color="black",
                        linestyle=":",
                        linewidth=2,
                    )
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(["Machine 1", "Machine 2", "Machine 3"])
            ax1.set_ylabel("Empirical mean +/- CI")
            ax1.set_ylim(0, 1)
            ax1.set_title(f"N={n_pulls} pulls/machine, {confidence:.0%} CI")
            ax1.grid(True, alpha=0.3, axis="y")
            # Panel 2: half-width vs N, with a marker at the current N.
            n_range = list(range(2, 201))
            half_widths = [
                _hoeffding_half_width(n, confidence) for n in n_range
            ]
            ax2.plot(n_range, half_widths, color="black", linewidth=2)
            ax2.scatter(
                [n_pulls], [half_width], color="red", s=80, zorder=3
            )
            ax2.set_xlabel("N (pulls per machine)")
            ax2.set_ylabel("CI half-width")
            ax2.set_title("More pulls shrink the CI")
            ax2.grid(True, alpha=0.3)
            # Panel 3: comments.
            comment_lines = [
                "State:",
                f"  seed: {seed_slider.value}",
                f"  N (pulls/machine): {n_pulls}",
                f"  confidence: {confidence:.0%}",
                "",
                "Empirical means:",
            ]
            for i in range(3):
                comment_lines.append(
                    f"  Machine {i + 1}: {empirical_means[i]:.2f}"
                    f" +/- {half_width:.2f}"
                )
            comment_lines += [
                "",
                f"CI half-width: {half_width:.3f}",
            ]
            ax3.axis("off")
            htutori.add_fitted_text_box(
                ax3, "\n".join(comment_lines), max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    # Register callbacks so the plot updates live as widgets change.
    seed_slider.observe(update_plot, names="value")
    n_pulls_slider.observe(update_plot, names="value")
    confidence_dropdown.observe(update_plot, names="value")
    show_means_toggle.observe(update_plot, names="value")
    controls = ipywidgets.VBox(
        [seed_box, n_pulls_box, confidence_dropdown, show_means_toggle]
    )
    display(controls, output)
    update_plot()


# #############################################################################
# Cell 6: Upper Confidence Bound (UCB) Intuition
# #############################################################################


def cell6_ucb_intuition() -> None:
    """
    Show the UCB index as empirical mean plus exploration bonus.

    Uses a fixed illustrative scenario with 3 machines that have different
    pull counts and empirical means, so the effect of `t` on the exploration
    bonus is isolated. User can:
    - Move the time slider `t` to see the bonus (and the argmax) change.
    """
    # Fixed illustrative scenario: machine 2 has the highest empirical mean,
    # but machine 3 has been pulled the fewest times, so its bonus is large.
    empirical_means = [0.60, 0.65, 0.50]
    pull_counts = [20, 30, 3]
    colors_mean = "tab:blue"
    colors_bonus = "tab:orange"
    t_slider, t_box = htutori.build_widget_control(
        name="t",
        description="current round",
        min_val=4,
        max_val=100,
        step=1,
        initial_value=10,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[dict] = None) -> None:
        """
        Recompute the UCB index for each machine and redraw.

        :param change: dictionary with change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            t = t_slider.value
            bonuses = [
                sim.ucb_bonus(t, pull_counts[i]) for i in range(3)
            ]
            ucb_values = [empirical_means[i] + bonuses[i] for i in range(3)]
            best_machine = int(np.argmax(ucb_values))
            # Create 1x2 layout: stacked bar chart, comments.
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            # Panel 1: stacked bar chart of mean (blue) + bonus (orange).
            x_pos = np.arange(3)
            ax1.bar(
                x_pos, empirical_means, color=colors_mean, label="empirical mean"
            )
            ax1.bar(
                x_pos,
                bonuses,
                bottom=empirical_means,
                color=colors_bonus,
                label="exploration bonus",
            )
            for i in range(3):
                marker = " *" if i == best_machine else ""
                ax1.text(
                    x_pos[i],
                    ucb_values[i] + 0.05,
                    f"U={ucb_values[i]:.2f}{marker}",
                    ha="center",
                    fontsize=10,
                    weight="bold",
                )
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(["Machine 1", "Machine 2", "Machine 3"])
            ax1.set_ylabel("UCB index")
            ax1.set_ylim(0, max(ucb_values) + 0.5)
            ax1.set_title(f"UCB index at t={t} (* = highest)")
            ax1.legend(loc="upper right", fontsize=9)
            ax1.grid(True, alpha=0.3, axis="y")
            # Panel 2: comments.
            comment_lines = ["State:", f"  t: {t}", ""]
            for i in range(3):
                comment_lines.append(
                    f"Machine {i + 1}: N={pull_counts[i]},"
                    f" mean={empirical_means[i]:.2f}"
                )
                comment_lines.append(
                    f"  bonus={bonuses[i]:.2f}, UCB={ucb_values[i]:.2f}"
                )
            comment_lines += [
                "",
                f"Highest UCB: Machine {best_machine + 1}",
            ]
            ax2.axis("off")
            htutori.add_fitted_text_box(
                ax2, "\n".join(comment_lines), max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    t_slider.observe(update_plot, names="value")
    controls = ipywidgets.VBox([t_box])
    display(controls, output)
    update_plot()


# #############################################################################
# Cell 7: UCB Algorithm Simulation
# #############################################################################


def cell7_ucb_simulation() -> None:
    """
    Run `UCBStrategy` on 4 Bernoulli arms and show pull counts and regret.

    True means mu = (0.3, 0.5, 0.7, 0.4), so Machine 3 (mu=0.7) is optimal.
    User can:
    - Set the seed and time horizon T.
    - Click "Run UCB Algorithm" to replay the experiment.
    """
    true_means = [0.3, 0.5, 0.7, 0.4]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    best_machine = int(np.argmax(true_means))
    # Create seed widget (must be first, per convention).
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=200,
        step=1,
        initial_value=42,
        is_float=False,
    )
    t_horizon_slider, t_horizon_box = htutori.build_widget_control(
        name="T",
        description="time horizon",
        min_val=20,
        max_val=300,
        step=10,
        initial_value=100,
        is_float=False,
    )
    run_button = ipywidgets.Button(
        description="Run UCB Algorithm",
        button_style="success",
        layout={"width": "200px"},
    )
    output = ipywidgets.Output()

    def on_run_clicked(b: ipywidgets.Button) -> None:
        """
        Run UCB1 and plot the pull timeline, pull counts, and regret.

        :param b: button widget (unused)
        """
        with output:
            clear_output(wait=True)
            t_horizon = t_horizon_slider.value
            bandit = sim.MultiArmedBandit(
                k_machines=4,
                mu_values=true_means,
                seed=seed_slider.value,
                reward_type="bernoulli",
            )
            strategy = sim.UCBStrategy()
            experiment = sim.BanditExperiment(
                bandit=bandit, strategy=strategy, n_coins=t_horizon
            )
            experiment.run()
            choices = experiment.machine_choices
            cumulative_regret = sim.compute_regret(true_means, choices)
            rounds = list(range(1, t_horizon + 1))
            # Track N_i(t): running pull count per arm over time.
            running_counts = [[0] * t_horizon for _ in range(4)]
            counts_so_far = [0] * 4
            for t, choice in enumerate(choices):
                counts_so_far[choice] += 1
                for i in range(4):
                    running_counts[i][t] = counts_so_far[i]
            # Create a 1x4 layout: timeline, N_i(t), regret, comments.
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 5))
            # Panel 1: timeline of which machine was pulled.
            for i in range(4):
                idx = [t for t, c in enumerate(choices) if c == i]
                ax1.scatter(
                    [rounds[t] for t in idx],
                    [i] * len(idx),
                    color=colors[i],
                    s=25,
                    label=f"Machine {i + 1}",
                )
            ax1.set_xlabel("Round t")
            ax1.set_yticks(range(4))
            ax1.set_yticklabels([f"Machine {i + 1}" for i in range(4)])
            ax1.set_title("Pull timeline")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="center right", fontsize=8)
            # Panel 2: N_i(t) for each arm.
            for i in range(4):
                ax2.plot(
                    rounds, running_counts[i], color=colors[i], linewidth=2,
                    label=f"Machine {i + 1}",
                )
            ax2.set_xlabel("Round t")
            ax2.set_ylabel("N_i(t)")
            ax2.set_title("Pull counts over time")
            ax2.legend(loc="upper left", fontsize=8)
            ax2.grid(True, alpha=0.3)
            # Panel 3: cumulative regret.
            ax3.plot(rounds, cumulative_regret, color="black", linewidth=2)
            ax3.set_xlabel("Round t")
            ax3.set_ylabel("Cumulative regret L_t")
            ax3.set_title("Cumulative regret")
            ax3.grid(True, alpha=0.3)
            # Panel 4: comments.
            pull_counts = [choices.count(i) for i in range(4)]
            comment_lines = [
                "State:",
                f"  seed: {seed_slider.value}",
                f"  T: {t_horizon}",
                "",
                "Pull counts:",
            ]
            for i in range(4):
                comment_lines.append(f"  Machine {i + 1}: {pull_counts[i]}")
            comment_lines += [
                "",
                f"Optimal arm: Machine {best_machine + 1}"
                f" (mu={true_means[best_machine]:.2f})",
                f"Final regret L_T: {cumulative_regret[-1]:.2f}",
            ]
            ax4.axis("off")
            htutori.add_fitted_text_box(
                ax4, "\n".join(comment_lines), max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    run_button.on_click(on_run_clicked)
    controls = ipywidgets.VBox([seed_box, t_horizon_box, run_button])
    display(controls, output)
    on_run_clicked(run_button)


# #############################################################################
# Cell 8: UCB Exploration Bonus Decay
# #############################################################################


def cell8_ucb_bonus_decay() -> None:
    """
    Show how the UCB exploration bonus sqrt(2 log(t) / N_i) decays with N_i.

    User can:
    - Move the time slider `t`.
    - Move the pulls slider `N_i` to see the bonus value at that point.
    """
    t_slider, t_box = htutori.build_widget_control(
        name="t",
        description="current round",
        min_val=2,
        max_val=1000,
        step=10,
        initial_value=100,
        is_float=False,
    )
    n_slider, n_box = htutori.build_widget_control(
        name="N_i",
        description="pulls of arm i",
        min_val=1,
        max_val=100,
        step=1,
        initial_value=10,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[dict] = None) -> None:
        """
        Recompute the bonus curve and redraw.

        :param change: dictionary with change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            t = t_slider.value
            n_i = n_slider.value
            bonus = sim.ucb_bonus(t, n_i)
            n_range = list(range(1, 101))
            bonus_curve = [sim.ucb_bonus(t, n) for n in n_range]
            # Create 1x2 layout: bonus vs N_i curve, comments.
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            # Panel 1: bonus vs N_i, with a marker at the current N_i.
            ax1.plot(n_range, bonus_curve, color="tab:orange", linewidth=2)
            ax1.scatter([n_i], [bonus], color="red", s=80, zorder=3)
            ax1.set_xlabel("N_i (number of pulls)")
            ax1.set_ylabel("Exploration bonus")
            ax1.set_title(f"Bonus decays as 1/sqrt(N_i), at t={t}")
            ax1.grid(True, alpha=0.3)
            # Panel 2: comments.
            comment_lines = [
                "State:",
                f"  t: {t}",
                f"  N_i: {n_i}",
                "",
                f"Bonus = sqrt(2 log({t}) / {n_i})",
                f"      = {bonus:.4f}",
            ]
            ax2.axis("off")
            htutori.add_fitted_text_box(
                ax2, "\n".join(comment_lines), max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    t_slider.observe(update_plot, names="value")
    n_slider.observe(update_plot, names="value")
    controls = ipywidgets.VBox([t_box, n_box])
    display(controls, output)
    update_plot()


# #############################################################################
# Cell 9: Regret Accumulation
# #############################################################################


def _build_strategy(algorithm: str, *, seed: int) -> sim.Strategy:
    """
    Build a `Strategy` instance by algorithm name.

    :param algorithm: one of "Random", "Greedy", "Epsilon-Greedy", "UCB"
    :param seed: random seed for strategies that need one
    :return: the corresponding `Strategy` instance
    """
    if algorithm == "Random":
        strategy = sim.ExplorationStrategy(seed=seed)
    elif algorithm == "Greedy":
        strategy = sim.ExploitationStrategy()
    elif algorithm == "Epsilon-Greedy":
        strategy = sim.EpsilonGreedyStrategy(epsilon=0.1, seed=seed)
    elif algorithm == "UCB":
        strategy = sim.UCBStrategy()
    elif algorithm == "Thompson Sampling":
        strategy = sim.ThompsonSamplingStrategy(seed=seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return strategy


def cell9_regret_accumulation() -> None:
    """
    Visualize per-step and cumulative regret for a chosen algorithm.

    Uses 3 Bernoulli arms with mu = (0.3, 0.5, 0.7). User can:
    - Choose the algorithm (Random, Greedy, Epsilon-Greedy, UCB).
    - Set the seed and time horizon T.
    - Click "Run Algorithm" to replay the experiment.
    """
    true_means = [0.3, 0.5, 0.7]
    algorithm_dropdown = ipywidgets.Dropdown(
        options=["Random", "Greedy", "Epsilon-Greedy", "UCB"],
        value="UCB",
        description="algorithm:",
        style={"description_width": "100px"},
    )
    # Create seed widget (must be first slider, per convention).
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=200,
        step=1,
        initial_value=42,
        is_float=False,
    )
    t_horizon_slider, t_horizon_box = htutori.build_widget_control(
        name="T",
        description="time horizon",
        min_val=20,
        max_val=300,
        step=10,
        initial_value=100,
        is_float=False,
    )
    run_button = ipywidgets.Button(
        description="Run Algorithm",
        button_style="success",
        layout={"width": "200px"},
    )
    output = ipywidgets.Output()

    def on_run_clicked(b: ipywidgets.Button) -> None:
        """
        Run the chosen algorithm and plot per-step and cumulative regret.

        :param b: button widget (unused)
        """
        with output:
            clear_output(wait=True)
            t_horizon = t_horizon_slider.value
            algorithm = algorithm_dropdown.value
            bandit = sim.MultiArmedBandit(
                k_machines=3,
                mu_values=true_means,
                seed=seed_slider.value,
                reward_type="bernoulli",
            )
            strategy = _build_strategy(algorithm, seed=seed_slider.value + 1)
            experiment = sim.BanditExperiment(
                bandit=bandit, strategy=strategy, n_coins=t_horizon
            )
            experiment.run()
            choices = experiment.machine_choices
            mu_star = max(true_means)
            instantaneous_regret = [
                mu_star - true_means[c] for c in choices
            ]
            cumulative_regret = sim.compute_regret(true_means, choices)
            rounds = list(range(1, t_horizon + 1))
            # Create a 1x3 layout: per-step regret, cumulative regret,
            # comments.
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
            # Panel 1: instantaneous regret per step, colored by whether the
            # optimal arm was chosen.
            best_machine = int(np.argmax(true_means))
            bar_colors = [
                "tab:green" if c == best_machine else "tab:red"
                for c in choices
            ]
            ax1.bar(rounds, instantaneous_regret, color=bar_colors)
            ax1.set_xlabel("Round t")
            ax1.set_ylabel("Instantaneous regret")
            ax1.set_title(f"{algorithm}: per-step regret (green = optimal arm)")
            ax1.grid(True, alpha=0.3)
            # Panel 2: cumulative regret.
            ax2.plot(rounds, cumulative_regret, color="black", linewidth=2)
            ax2.set_xlabel("Round t")
            ax2.set_ylabel("Cumulative regret L_t")
            ax2.set_title("Cumulative regret")
            ax2.grid(True, alpha=0.3)
            # Panel 3: comments.
            num_optimal = choices.count(best_machine)
            comment_lines = [
                "State:",
                f"  algorithm: {algorithm}",
                f"  seed: {seed_slider.value}",
                f"  T: {t_horizon}",
                "",
                f"Optimal arm: Machine {best_machine + 1}",
                f"Pulled optimal arm: {num_optimal}/{t_horizon} rounds",
                f"Final regret L_T: {cumulative_regret[-1]:.2f}",
            ]
            ax3.axis("off")
            htutori.add_fitted_text_box(
                ax3, "\n".join(comment_lines), max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    run_button.on_click(on_run_clicked)
    controls = ipywidgets.VBox(
        [seed_box, algorithm_dropdown, t_horizon_box, run_button]
    )
    display(controls, output)
    on_run_clicked(run_button)


# #############################################################################
# Cell 10: Comparing Algorithms: Regret Curves
# #############################################################################

_ALGORITHM_BIG_O = {
    "Random": "Theta(T)",
    "Greedy": "Theta(T)",
    "Epsilon-Greedy": "O(T^(2/3))",
    "UCB": "O(log T)",
    "Thompson Sampling": "O(log T)",
}


def cell10_regret_comparison() -> None:
    """
    Compare cumulative regret curves of several algorithms on a log-t axis.

    Arms are generated as `mu_i = linspace(0.9, 0.1, K)`, so Machine 1 is
    always optimal. Each algorithm's curve is averaged over a few trials
    (different seeds) to smooth out per-run noise. User can:
    - Check which algorithms to compare.
    - Set the number of arms K and the time horizon T.
    - Click "Run Comparison".
    """
    n_trials = 5
    algorithm_names = list(_ALGORITHM_BIG_O.keys())
    algorithm_checkboxes = {
        name: ipywidgets.Checkbox(value=True, description=name)
        for name in algorithm_names
    }
    # Create seed widget (must be first, per convention).
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="base random seed",
        min_val=0,
        max_val=200,
        step=1,
        initial_value=42,
        is_float=False,
    )
    k_slider, k_box = htutori.build_widget_control(
        name="K",
        description="number of arms",
        min_val=2,
        max_val=10,
        step=1,
        initial_value=3,
        is_float=False,
    )
    # T spans two orders of magnitude, so use a log2 slider (128 to 8192).
    t_exp_slider, t_box = htutori.build_log_widget_control(
        name="log2(T)",
        description="T (time horizon)",
        min_exp=7,
        max_exp=13,
        initial_exp=10,
        base=2,
    )
    run_button = ipywidgets.Button(
        description="Run Comparison",
        button_style="success",
        layout={"width": "200px"},
    )
    output = ipywidgets.Output()

    def on_run_clicked(b: ipywidgets.Button) -> None:
        """
        Run the selected algorithms and plot averaged regret curves.

        :param b: button widget (unused)
        """
        with output:
            clear_output(wait=True)
            k_machines = k_slider.value
            t_horizon = 2**t_exp_slider.value
            base_seed = seed_slider.value
            true_means = np.linspace(0.9, 0.1, k_machines).tolist()
            selected = [
                name
                for name in algorithm_names
                if algorithm_checkboxes[name].value
            ]
            rounds = list(range(1, t_horizon + 1))
            # Create 1x2 layout: regret curves (log-t axis), comments.
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            final_regrets = {}
            for name in selected:
                # Average the cumulative regret curve over n_trials seeds.
                regret_sum = np.zeros(t_horizon)
                for trial_idx in range(n_trials):
                    trial_seed = base_seed + trial_idx * 1000
                    bandit = sim.MultiArmedBandit(
                        k_machines=k_machines,
                        mu_values=true_means,
                        seed=trial_seed,
                        reward_type="bernoulli",
                    )
                    strategy = _build_strategy(name, seed=trial_seed + 1)
                    experiment = sim.BanditExperiment(
                        bandit=bandit, strategy=strategy, n_coins=t_horizon
                    )
                    experiment.run()
                    regret_sum += np.array(
                        sim.compute_regret(
                            true_means, experiment.machine_choices
                        )
                    )
                mean_regret = regret_sum / n_trials
                final_regrets[name] = mean_regret[-1]
                ax1.plot(rounds, mean_regret, linewidth=2, label=name)
            ax1.set_xscale("log")
            ax1.set_xlabel("Round t (log scale)")
            ax1.set_ylabel("Mean cumulative regret L_t")
            ax1.set_title(
                f"K={k_machines} arms, T={t_horizon}, {n_trials} trials"
            )
            ax1.legend(loc="upper left", fontsize=9)
            ax1.grid(True, alpha=0.3)
            # Panel 2: comments with the theoretical growth rate per algorithm.
            comment_lines = [
                "State:",
                f"  K: {k_machines}",
                f"  T: {t_horizon}",
                f"  seed: {base_seed}",
                "",
                "Final regret (this run):",
            ]
            for name in selected:
                comment_lines.append(
                    f"  {name}: {final_regrets[name]:.1f}"
                    f" [{_ALGORITHM_BIG_O[name]}]"
                )
            ax2.axis("off")
            htutori.add_fitted_text_box(
                ax2, "\n".join(comment_lines), max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    run_button.on_click(on_run_clicked)
    controls = ipywidgets.VBox(
        [seed_box, k_box, t_box]
        + list(algorithm_checkboxes.values())
        + [run_button]
    )
    display(controls, output)
    on_run_clicked(run_button)


# #############################################################################
# Cell 11: Bayesian Bandits: Prior and Posterior
# #############################################################################


def cell11_bayesian_prior_posterior() -> None:
    """
    Show a Beta prior updating into a Beta posterior as data arrives.

    A single hidden Bernoulli arm (mu=0.6) is pulled on demand. User can:
    - Set the prior parameters alpha, beta.
    - Click "Pull Arm" to draw one more success/failure and update the
      posterior.
    - Click "Reset" to clear the observed data.
    """
    true_mu = 0.6
    # Create seed widget (must be first, per convention).
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=200,
        step=1,
        initial_value=42,
        is_float=False,
    )
    alpha_slider, alpha_box = htutori.build_widget_control(
        name="alpha",
        description="prior alpha",
        min_val=1,
        max_val=10,
        step=1,
        initial_value=1,
        is_float=False,
    )
    beta_slider, beta_box = htutori.build_widget_control(
        name="beta",
        description="prior beta",
        min_val=1,
        max_val=10,
        step=1,
        initial_value=1,
        is_float=False,
    )
    pull_button = ipywidgets.Button(
        description="Pull Arm", button_style="success"
    )
    reset_button = ipywidgets.Button(
        description="Reset", button_style="warning"
    )
    output = ipywidgets.Output()
    # Observed data state: number of successes and failures so far.
    state = {"successes": 0, "failures": 0, "rng": np.random.RandomState(42)}

    def update_plot() -> None:
        """
        Redraw the prior/posterior PDFs given the current observed data.
        """
        with output:
            clear_output(wait=True)
            alpha = alpha_slider.value
            beta = beta_slider.value
            s = state["successes"]
            f = state["failures"]
            post_alpha = alpha + s
            post_beta = beta + f
            x = np.linspace(0.001, 0.999, 500)
            prior_pdf = _beta_pdf(x, alpha, beta)
            post_pdf = _beta_pdf(x, post_alpha, post_beta)
            # Create 1x2 layout: PDF plot, comments.
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            # Panel 1: prior (light, dotted) and posterior (solid, shaded).
            ax1.plot(
                x, prior_pdf, color="tab:blue", linestyle=":", alpha=0.6,
                linewidth=2, label=f"Prior Beta({alpha},{beta})",
            )
            ax1.plot(
                x, post_pdf, color="tab:orange", linewidth=2.5,
                label=f"Posterior Beta({post_alpha:.0f},{post_beta:.0f})",
            )
            ax1.fill_between(x, post_pdf, alpha=0.2, color="tab:orange")
            ax1.set_xlabel("mu (success probability)")
            ax1.set_ylabel("Density")
            ax1.set_title(f"After {s + f} pulls: {s} successes, {f} failures")
            ax1.legend(loc="upper left", fontsize=9)
            ax1.grid(True, alpha=0.3)
            # Panel 2: comments.
            post_mean = post_alpha / (post_alpha + post_beta)
            post_var = (post_alpha * post_beta) / (
                (post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1)
            )
            comment_lines = [
                "State:",
                f"  prior: Beta({alpha}, {beta})",
                f"  successes s: {s}",
                f"  failures f: {f}",
                "",
                f"Posterior: Beta({post_alpha:.0f}, {post_beta:.0f})",
                f"  mean: {post_mean:.3f}",
                f"  variance: {post_var:.4f}",
            ]
            ax2.axis("off")
            htutori.add_fitted_text_box(
                ax2, "\n".join(comment_lines), max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    def on_pull_clicked(b: ipywidgets.Button) -> None:
        """
        Draw one Bernoulli(true_mu) sample and update the observed counts.

        :param b: button widget (unused)
        """
        reward = float(state["rng"].random_sample() < true_mu)
        if reward == 1.0:
            state["successes"] += 1
        else:
            state["failures"] += 1
        update_plot()

    def on_reset_clicked(b: ipywidgets.Button) -> None:
        """
        Clear the observed data and reseed the random draws.

        :param b: button widget (unused)
        """
        state["successes"] = 0
        state["failures"] = 0
        state["rng"] = np.random.RandomState(seed_slider.value)
        update_plot()

    pull_button.on_click(on_pull_clicked)
    reset_button.on_click(on_reset_clicked)
    seed_slider.observe(
        lambda change: on_reset_clicked(reset_button), names="value"
    )
    controls = ipywidgets.VBox(
        [
            seed_box,
            alpha_box,
            beta_box,
            ipywidgets.HBox([pull_button, reset_button]),
        ]
    )
    display(controls, output)
    update_plot()


# #############################################################################
# Cell 12: Thompson Sampling Algorithm
# #############################################################################


def _replay_thompson_sampling(
    *,
    k_machines: int,
    true_means: List[float],
    seed: int,
    n_rounds: int,
) -> dict:
    """
    Run Thompson Sampling and reconstruct the per-round posterior history.

    :param k_machines: number of arms (K)
    :param true_means: true Bernoulli success probability of each arm
    :param seed: random seed
    :param n_rounds: number of rounds to run
    :return: dict with "choices", "rewards", "samples_history",
        "alphas_history", "betas_history" (one entry per round)
    """
    bandit = sim.MultiArmedBandit(
        k_machines=k_machines,
        mu_values=true_means,
        seed=seed,
        reward_type="bernoulli",
    )
    strategy = sim.ThompsonSamplingStrategy(seed=seed + 1)
    experiment = sim.BanditExperiment(
        bandit=bandit, strategy=strategy, n_coins=n_rounds
    )
    rewards, _, _ = experiment.run()
    choices = experiment.machine_choices
    # Reconstruct the Beta(alpha, beta) parameters used *before* each round's
    # pull, by replaying the successes/failures accumulated so far.
    running_s = [0] * k_machines
    running_f = [0] * k_machines
    alphas_history = []
    betas_history = []
    for t in range(n_rounds):
        alphas_history.append([1.0 + s for s in running_s])
        betas_history.append([1.0 + f for f in running_f])
        if rewards[t] == 1.0:
            running_s[choices[t]] += 1
        else:
            running_f[choices[t]] += 1
    return {
        "choices": choices,
        "rewards": rewards,
        "samples_history": strategy.samples_history,
        "alphas_history": alphas_history,
        "betas_history": betas_history,
    }


def cell12_thompson_sampling() -> None:
    """
    Show Thompson Sampling drawing from each arm's posterior and picking argmax.

    User can:
    - Set the seed, number of arms K, and number of rounds to run.
    - Click "Run Thompson Sampling" to replay the experiment.
    - Move the step slider to browse the posterior at each round.
    """
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    # Create seed widget (must be first, per convention).
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=200,
        step=1,
        initial_value=42,
        is_float=False,
    )
    k_slider, k_box = htutori.build_widget_control(
        name="K",
        description="number of arms",
        min_val=2,
        max_val=5,
        step=1,
        initial_value=3,
        is_float=False,
    )
    n_rounds_slider, n_rounds_box = htutori.build_widget_control(
        name="rounds",
        description="number of rounds",
        min_val=5,
        max_val=50,
        step=5,
        initial_value=20,
        is_float=False,
    )
    step_slider, step_box = htutori.build_widget_control(
        name="step",
        description="round to inspect",
        min_val=1,
        max_val=20,
        step=1,
        initial_value=1,
        is_float=False,
    )
    run_button = ipywidgets.Button(
        description="Run Thompson Sampling", button_style="success"
    )
    output = ipywidgets.Output()
    state: Dict[str, Any] = {
        "result": None,
        "true_means": None,
        "k_machines": None,
    }

    def draw_step() -> None:
        """
        Draw the posterior curves and sampled dots for the current step.
        """
        result = state["result"]
        if result is None:
            return
        with output:
            clear_output(wait=True)
            k_machines = state["k_machines"]
            true_means = state["true_means"]
            t = step_slider.value - 1
            alphas = result["alphas_history"][t]
            betas = result["betas_history"][t]
            samples = result["samples_history"][t]
            choice = result["choices"][t]
            x = np.linspace(0.001, 0.999, 300)
            # Create 1x2 layout: posterior curves with sampled dots, comments.
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            for i in range(k_machines):
                pdf = _beta_pdf(x, alphas[i], betas[i])
                ax1.plot(
                    x, pdf, color=colors[i], linewidth=2,
                    label=f"Machine {i + 1} (mu={true_means[i]:.2f})",
                )
                # Mark the sampled theta_i for this arm on its curve.
                sample_pdf = _beta_pdf(
                    np.array([samples[i]]), alphas[i], betas[i]
                )[0]
                marker = "*" if i == choice else "o"
                marker_size = 250 if i == choice else 100
                ax1.scatter(
                    [samples[i]], [sample_pdf], color=colors[i],
                    marker=marker, s=marker_size, edgecolor="black",
                    zorder=3,
                )
            ax1.set_xlabel("mu")
            ax1.set_ylabel("Posterior density")
            ax1.set_title(
                f"Round {t + 1}: sampled thetas (* = selected Machine"
                f" {choice + 1})"
            )
            ax1.legend(loc="upper left", fontsize=8)
            ax1.grid(True, alpha=0.3)
            # Panel 2: comments.
            comment_lines = ["State:", f"  round: {t + 1}", ""]
            for i in range(k_machines):
                comment_lines.append(
                    f"Machine {i + 1}: Beta({alphas[i]:.0f},{betas[i]:.0f})"
                )
                comment_lines.append(f"  theta sample={samples[i]:.3f}")
            comment_lines += ["", f"Selected: Machine {choice + 1}"]
            ax2.axis("off")
            htutori.add_fitted_text_box(
                ax2, "\n".join(comment_lines), max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    def on_run_clicked(b: ipywidgets.Button) -> None:
        """
        Run Thompson Sampling for the chosen K and number of rounds.

        :param b: button widget (unused)
        """
        k_machines = k_slider.value
        n_rounds = n_rounds_slider.value
        true_means = np.linspace(0.9, 0.1, k_machines).tolist()
        state["result"] = _replay_thompson_sampling(
            k_machines=k_machines,
            true_means=true_means,
            seed=seed_slider.value,
            n_rounds=n_rounds,
        )
        state["true_means"] = true_means
        state["k_machines"] = k_machines
        step_slider.max = n_rounds
        step_slider.value = 1
        draw_step()

    run_button.on_click(on_run_clicked)
    step_slider.observe(lambda change: draw_step(), names="value")
    controls = ipywidgets.VBox(
        [seed_box, k_box, n_rounds_box, run_button, step_box]
    )
    display(controls, output)
    on_run_clicked(run_button)


# #############################################################################
# Cell 13: Thompson Sampling: Probability Matching
# #############################################################################


def _monte_carlo_prob_optimal(
    alphas: List[float], betas: List[float], *, n_samples: int, seed: int
) -> List[float]:
    """
    Estimate Pr(arm i is optimal | data) via Monte Carlo posterior sampling.

    :param alphas: Beta posterior alpha parameter of each arm
    :param betas: Beta posterior beta parameter of each arm
    :param n_samples: number of Monte Carlo draws
    :param seed: random seed
    :return: estimated probability that each arm has the highest sample
    """
    rng = np.random.RandomState(seed)
    k_machines = len(alphas)
    # Draw n_samples thetas per arm, then count how often each arm wins.
    draws = np.array(
        [
            rng.beta(alphas[i], betas[i], size=n_samples)
            for i in range(k_machines)
        ]
    )
    winners = np.argmax(draws, axis=0)
    counts = np.bincount(winners, minlength=k_machines)
    return (counts / n_samples).tolist()


def cell13_probability_matching() -> None:
    """
    Compare theoretical Pr(arm optimal) with the empirical selection frequency.

    Builds a fixed posterior from `n_pulls` observed pulls per arm, then:
    - Computes the theoretical Pr(arm i is optimal | data) with a large
      Monte Carlo estimate (50000 draws).
    - Lets the user click "Run 1000 Steps" to draw 1000 fresh samples from
      that same fixed posterior and record how often each arm wins, which
      should match the theoretical probability (probability matching).
    """
    true_means = [0.3, 0.5, 0.7]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    # Create seed widget (must be first, per convention).
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=200,
        step=1,
        initial_value=42,
        is_float=False,
    )
    n_pulls_slider, n_pulls_box = htutori.build_widget_control(
        name="N",
        description="pulls per arm (data)",
        min_val=5,
        max_val=50,
        step=5,
        initial_value=15,
        is_float=False,
    )
    run_button = ipywidgets.Button(
        description="Run 1000 Steps", button_style="success"
    )
    output = ipywidgets.Output()

    def on_run_clicked(b: ipywidgets.Button) -> None:
        """
        Build the posterior, then compare theoretical vs empirical Pr(optimal).

        :param b: button widget (unused)
        """
        with output:
            clear_output(wait=True)
            seed = seed_slider.value
            n_pulls = n_pulls_slider.value
            k_machines = len(true_means)
            bandit = sim.MultiArmedBandit(
                k_machines=k_machines,
                mu_values=true_means,
                seed=seed,
                reward_type="bernoulli",
            )
            # Build a fixed posterior from n_pulls observations per arm.
            for i in range(k_machines):
                for _ in range(n_pulls):
                    bandit.pull(i)
            alphas = [1.0 + sum(r) for r in bandit.machine_rewards]
            betas = [
                1.0 + len(r) - sum(r) for r in bandit.machine_rewards
            ]
            theoretical_prob = _monte_carlo_prob_optimal(
                alphas, betas, n_samples=50000, seed=seed + 1
            )
            empirical_freq = _monte_carlo_prob_optimal(
                alphas, betas, n_samples=1000, seed=seed + 2
            )
            # Create 1x3 layout: theoretical bars, empirical bars, comments.
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
            labels = [f"Machine {i + 1}" for i in range(k_machines)]
            ax1.bar(labels, theoretical_prob, color=colors[:k_machines])
            ax1.set_ylabel("Pr(arm is optimal)")
            ax1.set_ylim(0, 1)
            ax1.set_title("Theoretical (50000 MC draws)")
            ax1.grid(True, alpha=0.3, axis="y")
            ax2.bar(labels, empirical_freq, color=colors[:k_machines])
            ax2.set_ylabel("Selection frequency")
            ax2.set_ylim(0, 1)
            ax2.set_title("Empirical (1000 steps)")
            ax2.grid(True, alpha=0.3, axis="y")
            # Panel 3: comments.
            comment_lines = [
                "State:",
                f"  seed: {seed}",
                f"  N pulls/arm: {n_pulls}",
                "",
                "alpha, beta per arm:",
            ]
            for i in range(k_machines):
                comment_lines.append(
                    f"  Machine {i + 1}: Beta({alphas[i]:.0f},{betas[i]:.0f})"
                )
            comment_lines += ["", "Theoretical vs empirical:"]
            for i in range(k_machines):
                comment_lines.append(
                    f"  Machine {i + 1}: {theoretical_prob[i]:.3f}"
                    f" vs {empirical_freq[i]:.3f}"
                )
            ax3.axis("off")
            htutori.add_fitted_text_box(
                ax3, "\n".join(comment_lines), max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    run_button.on_click(on_run_clicked)
    controls = ipywidgets.VBox([seed_box, n_pulls_box, run_button])
    display(controls, output)
    on_run_clicked(run_button)


# #############################################################################
# Cell 14: UCB vs Thompson Sampling Comparison
# #############################################################################


def cell14_ucb_vs_thompson() -> None:
    """
    Compare UCB1 and Thompson Sampling on the same bandit environment.

    Arm 1 has mean mu*=0.7; the other K-1 arms have mean mu* - Delta. User
    can:
    - Set the number of arms K and the suboptimality gap Delta.
    - Set the time horizon T.
    - Click "Run Both Algorithms".
    """
    best_mu = 0.7
    colors = {"UCB": "tab:blue", "Thompson Sampling": "tab:orange"}
    # Create seed widget (must be first, per convention).
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=200,
        step=1,
        initial_value=42,
        is_float=False,
    )
    k_slider, k_box = htutori.build_widget_control(
        name="K",
        description="number of arms",
        min_val=2,
        max_val=6,
        step=1,
        initial_value=3,
        is_float=False,
    )
    delta_slider, delta_box = htutori.build_widget_control(
        name="Delta",
        description="suboptimality gap",
        min_val=0.05,
        max_val=0.5,
        step=0.05,
        initial_value=0.2,
        is_float=True,
    )
    t_horizon_slider, t_horizon_box = htutori.build_widget_control(
        name="T",
        description="time horizon",
        min_val=20,
        max_val=300,
        step=10,
        initial_value=150,
        is_float=False,
    )
    run_button = ipywidgets.Button(
        description="Run Both Algorithms", button_style="success"
    )
    output = ipywidgets.Output()

    def on_run_clicked(b: ipywidgets.Button) -> None:
        """
        Run UCB1 and Thompson Sampling and compare their regret curves.

        :param b: button widget (unused)
        """
        with output:
            clear_output(wait=True)
            k_machines = k_slider.value
            delta = delta_slider.value
            t_horizon = t_horizon_slider.value
            seed = seed_slider.value
            true_means = [best_mu] + [best_mu - delta] * (k_machines - 1)
            results = {}
            for name, strategy in [
                ("UCB", sim.UCBStrategy()),
                (
                    "Thompson Sampling",
                    sim.ThompsonSamplingStrategy(seed=seed + 1),
                ),
            ]:
                bandit = sim.MultiArmedBandit(
                    k_machines=k_machines,
                    mu_values=true_means,
                    seed=seed,
                    reward_type="bernoulli",
                )
                experiment = sim.BanditExperiment(
                    bandit=bandit, strategy=strategy, n_coins=t_horizon
                )
                experiment.run()
                choices = experiment.machine_choices
                results[name] = {
                    "regret": sim.compute_regret(true_means, choices),
                    "pull_counts": [
                        choices.count(i) for i in range(k_machines)
                    ],
                }
            rounds = list(range(1, t_horizon + 1))
            # Create 1x3 layout: regret curves, pull counts, comments.
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
            # Panel 1: regret curves for both algorithms, overlaid.
            for name, color in colors.items():
                ax1.plot(
                    rounds, results[name]["regret"], color=color,
                    linewidth=2, label=name,
                )
            ax1.set_xlabel("Round t")
            ax1.set_ylabel("Cumulative regret L_t")
            ax1.set_title(f"K={k_machines}, Delta={delta:.2f}")
            ax1.legend(loc="upper left", fontsize=9)
            ax1.grid(True, alpha=0.3)
            # Panel 2: pull counts per arm, grouped by algorithm.
            x_pos = np.arange(k_machines)
            width = 0.35
            ax2.bar(
                x_pos - width / 2, results["UCB"]["pull_counts"], width,
                color=colors["UCB"], label="UCB",
            )
            ax2.bar(
                x_pos + width / 2,
                results["Thompson Sampling"]["pull_counts"],
                width,
                color=colors["Thompson Sampling"],
                label="Thompson Sampling",
            )
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f"Machine {i + 1}" for i in range(k_machines)])
            ax2.set_ylabel("Pull count")
            ax2.set_title("Pull counts per arm")
            ax2.legend(loc="best", fontsize=9)
            ax2.grid(True, alpha=0.3, axis="y")
            # Panel 3: comments.
            comment_lines = [
                "State:",
                f"  seed: {seed}",
                f"  K: {k_machines}",
                f"  Delta: {delta:.2f}",
                f"  T: {t_horizon}",
                "",
                "Final regret L_T:",
                f"  UCB: {results['UCB']['regret'][-1]:.2f}",
                f"  Thompson Sampling: "
                f"{results['Thompson Sampling']['regret'][-1]:.2f}",
            ]
            ax3.axis("off")
            htutori.add_fitted_text_box(
                ax3, "\n".join(comment_lines), max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    run_button.on_click(on_run_clicked)
    controls = ipywidgets.VBox(
        [seed_box, k_box, delta_box, t_horizon_box, run_button]
    )
    display(controls, output)
    on_run_clicked(run_button)
