"""
Utility functions for Multi-Armed Bandits lesson.

Import as:

import msml610.tutorials.utils_Lesson09_3_Multi_Armed_Bandits as mtul3maba
"""

import abc
import logging
from typing import List, Optional, Tuple

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

import helpers.hdbg as hdbg
import msml610_utils as ut

_LOG = logging.getLogger(__name__)


# #############################################################################
# MultiArmedBandit
# #############################################################################


class MultiArmedBandit:
    """
    Multi-armed bandit environment with K machines.

    Each machine has an unknown probability distribution with mean mu_i.
    Rewards are drawn from a uniform distribution in [mu_i - width, mu_i +
    width], clipped to [-1, 1].
    """

    def __init__(
        self,
        *,
        k_machines: int,
        mu_values: List[float],
        seed: int,
        width: float = 0.3,
    ) -> None:
        """
        Initialize multi-armed bandit.

        :param k_machines: number of machines (K)
        :param mu_values: true mean values for each machine
        :param seed: random seed for reproducibility
        :param width: half-width of uniform distribution around mean
        """
        hdbg.dassert_eq(
            len(mu_values),
            k_machines,
            "Number of mu values must equal k_machines:",
            len(mu_values),
            k_machines,
        )
        hdbg.dassert_lte(1, k_machines, "Must have at least 1 machine")
        hdbg.dassert_lt(0.0, width, "Width must be positive")
        self.k_machines = k_machines
        self.mu_values = list(mu_values)
        self.width = width
        self.seed = seed
        # Initialize random state.
        self._rng = np.random.RandomState(seed)
        # Track statistics per machine.
        self.machine_pulls = [0] * k_machines
        self.machine_rewards = [[] for _ in range(k_machines)]

    def pull(self, machine_idx: int) -> float:
        """
        Pull a specific machine and get reward.

        :param machine_idx: index of machine to pull (0 to K-1)
        :return: reward value in [-1, 1]
        """
        hdbg.dassert_lte(
            0,
            machine_idx,
            "Machine index must be non-negative:",
            machine_idx,
        )
        hdbg.dassert_lt(
            machine_idx,
            self.k_machines,
            "Machine index out of range:",
            machine_idx,
        )
        # Generate reward from uniform distribution.
        true_mean = self.mu_values[machine_idx]
        reward = np.clip(
            self._rng.uniform(true_mean - self.width, true_mean + self.width),
            -1.0,
            1.0,
        )
        # Update statistics.
        self.machine_pulls[machine_idx] += 1
        self.machine_rewards[machine_idx].append(reward)
        return reward

    def get_empirical_means(self) -> List[float]:
        """
        Get empirical mean reward for each machine.

        :return: list of empirical means (or 0.0 if machine not pulled)
        """
        means = []
        for rewards in self.machine_rewards:
            if len(rewards) > 0:
                means.append(np.mean(rewards))
            else:
                means.append(0.0)
        return means

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset all statistics but keep mu values.

        :param seed: optional new seed; if None, use original seed
        """
        if seed is not None:
            self.seed = seed
        self.machine_pulls = [0] * self.k_machines
        self.machine_rewards = [[] for _ in range(self.k_machines)]
        # Reset random state.
        self._rng = np.random.RandomState(self.seed)


# #############################################################################
# Strategy
# #############################################################################


class Strategy(abc.ABC):
    """
    Abstract base class for bandit selection strategies.
    """

    @abc.abstractmethod
    def select_machine(
        self,
        bandit: MultiArmedBandit,
    ) -> int:
        """
        Select which machine to pull next.

        :param bandit: MultiArmedBandit instance with current state
        :return: index of machine to pull (0 to K-1)
        """
        pass

    def reset(self) -> None:
        """
        Reset any internal state of the strategy.
        """
        pass


# #############################################################################
# ExplorationStrategy
# #############################################################################


class ExplorationStrategy(Strategy):
    """
    Pure exploration strategy - randomly select machines.
    """

    def __init__(self, *, seed: int) -> None:
        """
        Initialize exploration strategy.

        :param seed: random seed for machine selection
        """
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    def select_machine(
        self,
        bandit: MultiArmedBandit,
    ) -> int:
        """
        Randomly select a machine with equal probability.

        :param bandit: MultiArmedBandit instance
        :return: randomly selected machine index
        """
        return self._rng.randint(0, bandit.k_machines)

    def reset(self) -> None:
        """
        Reset random state.
        """
        self._rng = np.random.RandomState(self.seed)


# #############################################################################
# ExploitationStrategy
# #############################################################################


class ExploitationStrategy(Strategy):
    """
    Pure exploitation strategy - always select best known machine.

    Start with one pull of each machine for initialization.
    """

    def __init__(self) -> None:
        """
        Initialize exploitation strategy.
        """
        self.initialized = False

    def select_machine(
        self,
        bandit: MultiArmedBandit,
    ) -> int:
        """
        Select machine with highest empirical mean.

        Initially pulls each machine once for initialization.

        :param bandit: MultiArmedBandit instance
        :return: machine index with highest empirical mean
        """
        # Initialize by pulling each machine once.
        if not self.initialized:
            for machine_idx in range(bandit.k_machines):
                if bandit.machine_pulls[machine_idx] == 0:
                    return machine_idx
            self.initialized = True
        # Select machine with highest empirical mean.
        empirical_means = bandit.get_empirical_means()
        return int(np.argmax(empirical_means))

    def reset(self) -> None:
        """
        Reset initialization state.
        """
        self.initialized = False


# #############################################################################
# EpsilonGreedyStrategy
# #############################################################################


class EpsilonGreedyStrategy(Strategy):
    """
    Epsilon-greedy strategy - explore with probability epsilon.

    Balances exploration and exploitation.
    """

    def __init__(self, *, epsilon: float, seed: int) -> None:
        """
        Initialize epsilon-greedy strategy.

        :param epsilon: exploration probability (0 to 1)
        :param seed: random seed for exploration decisions
        """
        hdbg.dassert_lte(
            0.0,
            epsilon,
            "Epsilon must be non-negative:",
            epsilon,
        )
        hdbg.dassert_lte(
            epsilon,
            1.0,
            "Epsilon must be at most 1.0:",
            epsilon,
        )
        self.epsilon = epsilon
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self.initialized = False

    def select_machine(
        self,
        bandit: MultiArmedBandit,
    ) -> int:
        """
        Select machine using epsilon-greedy policy.

        With probability epsilon, explore (random selection).
        With probability 1-epsilon, exploit (best known machine).

        :param bandit: MultiArmedBandit instance
        :return: selected machine index
        """
        # Initialize by pulling each machine once.
        if not self.initialized:
            for machine_idx in range(bandit.k_machines):
                if bandit.machine_pulls[machine_idx] == 0:
                    return machine_idx
            self.initialized = True
        # Epsilon-greedy selection.
        if self._rng.random() < self.epsilon:
            # Explore.
            return self._rng.randint(0, bandit.k_machines)
        else:
            # Exploit.
            empirical_means = bandit.get_empirical_means()
            return int(np.argmax(empirical_means))

    def reset(self) -> None:
        """
        Reset random state and initialization.
        """
        self._rng = np.random.RandomState(self.seed)
        self.initialized = False


# #############################################################################
# BanditExperiment
# #############################################################################


class BanditExperiment:
    """
    Run a single experiment with a bandit and strategy.
    """

    def __init__(
        self,
        *,
        bandit: MultiArmedBandit,
        strategy: Strategy,
        n_coins: int,
    ) -> None:
        """
        Initialize experiment.

        :param bandit: MultiArmedBandit instance
        :param strategy: Strategy instance
        :param n_coins: number of coins to play (N)
        """
        hdbg.dassert_lte(1, n_coins, "Must play at least 1 coin")
        self.bandit = bandit
        self.strategy = strategy
        self.n_coins = n_coins

    def run(self) -> Tuple[List[float], List[float], float]:
        """
        Run the experiment for n_coins trials.

        :return: tuple of (rewards, cumulative_rewards, final_total)
        """
        # Reset bandit and strategy state.
        self.bandit.reset()
        self.strategy.reset()
        rewards = []
        cumulative_rewards = []
        cumulative = 0.0
        # Run trials.
        for _ in range(self.n_coins):
            # Strategy selects machine.
            machine_idx = self.strategy.select_machine(self.bandit)
            # Pull machine and get reward.
            reward = self.bandit.pull(machine_idx)
            # Track results.
            rewards.append(reward)
            cumulative += reward
            cumulative_rewards.append(cumulative)
        return rewards, cumulative_rewards, cumulative


# #############################################################################
# BanditSimulation
# #############################################################################


class BanditSimulation:
    """
    Run multiple experiments for statistical analysis.
    """

    def __init__(
        self,
        *,
        k_machines: int,
        mu_values: List[float],
        n_coins: int,
        base_seed: int = 0,
    ) -> None:
        """
        Initialize simulation parameters.

        :param k_machines: number of machines (K)
        :param mu_values: true mean values for each machine
        :param n_coins: number of coins per experiment (N)
        :param base_seed: base seed for reproducibility
        """
        self.k_machines = k_machines
        self.mu_values = mu_values
        self.n_coins = n_coins
        self.base_seed = base_seed

    def run_trials(
        self,
        *,
        strategy_class: type,
        strategy_params: dict,
        n_trials: int,
    ) -> dict:
        """
        Run n_trials experiments with the same setup, varying seed.

        :param strategy_class: Strategy class to instantiate
        :param strategy_params: parameters to pass to strategy
        :param n_trials: number of trials to run
        :return: dictionary with statistics and results
        """
        hdbg.dassert_lte(1, n_trials, "Must run at least 1 trial")
        final_totals = []
        all_cumulative_rewards = []
        # Run trials with different seeds.
        for trial_idx in range(n_trials):
            # Create bandit with unique seed.
            bandit_seed = self.base_seed + trial_idx
            bandit = MultiArmedBandit(
                k_machines=self.k_machines,
                mu_values=self.mu_values,
                seed=bandit_seed,
            )
            # Create strategy with unique seed if needed.
            if "seed" in strategy_params:
                strategy_params_trial = strategy_params.copy()
                strategy_params_trial["seed"] = bandit_seed + 1000
                strategy = strategy_class(**strategy_params_trial)
            else:
                strategy = strategy_class(**strategy_params)
            # Run experiment.
            experiment = BanditExperiment(
                bandit=bandit,
                strategy=strategy,
                n_coins=self.n_coins,
            )
            _, cumulative_rewards, final_total = experiment.run()
            final_totals.append(final_total)
            all_cumulative_rewards.append(cumulative_rewards)
        # Compute statistics.
        final_totals_array = np.array(final_totals)
        all_cumulative_array = np.array(all_cumulative_rewards)
        return {
            "final_totals": final_totals,
            "mean_final": np.mean(final_totals_array),
            "std_final": np.std(final_totals_array),
            "all_cumulative_rewards": all_cumulative_rewards,
            "mean_cumulative": np.mean(all_cumulative_array, axis=0),
            "std_cumulative": np.std(all_cumulative_array, axis=0),
        }

    def epsilon_sweep(
        self,
        *,
        n_trials: int,
        epsilon_values: List[float] = None,
    ) -> dict:
        """
        Run simulations for multiple epsilon values.

        Compare exploration, exploitation, and balanced strategies.

        :param n_trials: number of trials per epsilon value
        :param epsilon_values: list of epsilon values to test
        :return: dictionary with results for each epsilon
        """
        if epsilon_values is None:
            epsilon_values = np.arange(0.0, 1.1, 0.1).tolist()
        results = {
            "epsilon_values": epsilon_values,
            "exploration": None,
            "exploitation": None,
            "balanced": [],
        }
        # Run pure exploration.
        exploration_results = self.run_trials(
            strategy_class=ExplorationStrategy,
            strategy_params={"seed": self.base_seed},
            n_trials=n_trials,
        )
        results["exploration"] = exploration_results
        # Run pure exploitation.
        exploitation_results = self.run_trials(
            strategy_class=ExploitationStrategy,
            strategy_params={},
            n_trials=n_trials,
        )
        results["exploitation"] = exploitation_results
        # Run balanced for each epsilon.
        for epsilon in epsilon_values:
            balanced_results = self.run_trials(
                strategy_class=EpsilonGreedyStrategy,
                strategy_params={"epsilon": epsilon, "seed": self.base_seed},
                n_trials=n_trials,
            )
            results["balanced"].append(balanced_results)
        return results


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
# Cell 1: Introduction - Casino Slot Machines
# #############################################################################


def cell1_casino_slot_machines() -> None:
    """
    Interactive casino slot machine visualization.

    Display 3 slot machines with fixed true means generating random rewards.
    User can:
    - Choose which machine to play.
    - Toggle showing true means.
    - Reset total winnings and coin budget.
    """
    # Initialize state.
    true_means = [-0.2, 0.0, 0.5]
    state = {
        "total_winnings": 0.0,
        "coins_remaining": 10,
        "initial_coins": 10,
        "machine_results": ["?", "?", "?"],
        "bandit": None,
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
                    f"Machine {i + 1}",
                    ha="center",
                    va="center",
                    fontsize=14,
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

    def on_play_clicked(b) -> None:
        """
        Handle play button click.
        """
        if state["coins_remaining"] <= 0:
            _LOG.warning("No coins remaining!")
            return

        # Initialize bandit if needed.
        if state["bandit"] is None:
            state["bandit"] = MultiArmedBandit(
                k_machines=3,
                mu_values=true_means,
                seed=seed_slider.value,
                width=0.3,
            )

        # Get selected machine index (0, 1, 2).
        machine_idx = int(machine_selector.value.split()[-1]) - 1

        # Pull the machine.
        reward = state["bandit"].pull(machine_idx)

        # Update state.
        state["total_winnings"] += reward
        state["coins_remaining"] -= 1
        state["machine_results"][machine_idx] = f"{reward:.2f}"

        # Increment seed for next play.
        seed_slider.value = seed_slider.value + 1

        # Update plot.
        update_plot()

    def on_reset_clicked(b) -> None:
        """
        Handle reset button click.
        """
        state["total_winnings"] = 0.0
        state["initial_coins"] = coins_slider.value
        state["coins_remaining"] = coins_slider.value
        state["machine_results"] = ["?", "?", "?"]
        # Reset bandit with current seed.
        state["bandit"] = MultiArmedBandit(
            k_machines=3,
            mu_values=true_means,
            seed=seed_slider.value,
            width=0.3,
        )
        update_plot()

    def on_show_means_changed(change) -> None:
        """
        Handle toggle for showing true means.
        """
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
        min_val=10,
        max_val=200,
        step=10,
        initial_value=100,
        is_float=False,
    )

    # Create epsilon slider.
    epsilon_slider, epsilon_box = ut.build_widget_control(
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
        strategy: Strategy,
    ) -> Tuple[List[float], List[float]]:
        """
        Run experiment with given strategy.

        :param num_coins: number of coins to play
        :param seed: random seed
        :param strategy: Strategy instance to use
        :return: (rewards, cumulative_rewards)
        """
        bandit = MultiArmedBandit(
            k_machines=state["num_machines"],
            mu_values=true_means,
            seed=seed,
            width=0.3,
        )
        experiment = BanditExperiment(
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
            exploration_strategy = ExplorationStrategy(seed=seed)
            exploitation_strategy = ExploitationStrategy()
            balanced_strategy = EpsilonGreedyStrategy(
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
# BanditEnsemble
# #############################################################################


class BanditEnsemble:
    """
    Average results over multiple random mu_i configurations.
    """

    def __init__(
        self,
        *,
        k_machines: int,
        n_coins: int,
        mu_range: Tuple[float, float] = (-0.5, 0.5),
        base_seed: int = 0,
    ) -> None:
        """
        Initialize ensemble parameters.

        :param k_machines: number of machines (K)
        :param n_coins: number of coins per experiment (N)
        :param mu_range: range for random mu values (min, max)
        :param base_seed: base seed for reproducibility
        """
        self.k_machines = k_machines
        self.n_coins = n_coins
        self.mu_range = mu_range
        self.base_seed = base_seed

    def run_ensemble(
        self,
        *,
        strategy_class: type,
        strategy_params: dict,
        n_trials: int,
        n_mu_configs: int,
    ) -> dict:
        """
        Run trials across multiple random mu configurations.

        :param strategy_class: Strategy class to instantiate
        :param strategy_params: parameters to pass to strategy
        :param n_trials: number of trials per mu configuration
        :param n_mu_configs: number of random mu configurations
        :return: dictionary with aggregated statistics
        """
        hdbg.dassert_lte(1, n_trials, "Must run at least 1 trial")
        hdbg.dassert_lte(1, n_mu_configs, "Must run at least 1 mu config")
        all_mean_finals = []
        all_std_finals = []
        # Generate random mu configurations.
        mu_rng = np.random.RandomState(self.base_seed)
        for mu_config_idx in range(n_mu_configs):
            # Generate random mu values.
            mu_values = mu_rng.uniform(
                self.mu_range[0],
                self.mu_range[1],
                self.k_machines,
            ).tolist()
            # Run simulation for this mu configuration.
            simulation = BanditSimulation(
                k_machines=self.k_machines,
                mu_values=mu_values,
                n_coins=self.n_coins,
                base_seed=self.base_seed + mu_config_idx * 10000,
            )
            results = simulation.run_trials(
                strategy_class=strategy_class,
                strategy_params=strategy_params,
                n_trials=n_trials,
            )
            all_mean_finals.append(results["mean_final"])
            all_std_finals.append(results["std_final"])
        # Aggregate statistics across mu configurations.
        all_mean_finals_array = np.array(all_mean_finals)
        return {
            "mean_finals_per_config": all_mean_finals,
            "overall_mean": np.mean(all_mean_finals_array),
            "overall_std": np.std(all_mean_finals_array),
            "std_finals_per_config": all_std_finals,
        }

    def compare_strategies_ensemble(
        self,
        *,
        n_trials: int,
        n_mu_configs: int,
        epsilon: float = 0.1,
    ) -> dict:
        """
        Compare strategies averaged over random mu configurations.

        :param n_trials: number of trials per mu configuration
        :param n_mu_configs: number of random mu configurations
        :param epsilon: epsilon value for balanced strategy
        :return: dictionary with results for each strategy
        """
        results = {}
        # Run exploration.
        results["exploration"] = self.run_ensemble(
            strategy_class=ExplorationStrategy,
            strategy_params={"seed": self.base_seed},
            n_trials=n_trials,
            n_mu_configs=n_mu_configs,
        )
        # Run exploitation.
        results["exploitation"] = self.run_ensemble(
            strategy_class=ExploitationStrategy,
            strategy_params={},
            n_trials=n_trials,
            n_mu_configs=n_mu_configs,
        )
        # Run balanced.
        results["balanced"] = self.run_ensemble(
            strategy_class=EpsilonGreedyStrategy,
            strategy_params={"epsilon": epsilon, "seed": self.base_seed},
            n_trials=n_trials,
            n_mu_configs=n_mu_configs,
        )
        return results

    def plot_ensemble_comparison(
        self,
        *,
        ensemble_results: dict,
        epsilon: float = 0.1,
    ) -> None:
        """
        Plot comparison of strategies across random mu configurations.

        :param ensemble_results: results from compare_strategies_ensemble()
        :param epsilon: epsilon value used for balanced strategy
        """
        strategies = ["exploration", "exploitation", "balanced"]
        labels = [
            "Pure Exploration",
            "Pure Exploitation",
            f"Balanced (epsilon={epsilon:.1f})",
        ]
        colors = ["blue", "red", "green"]
        means = [ensemble_results[s]["overall_mean"] for s in strategies]
        stds = [ensemble_results[s]["overall_std"] for s in strategies]
        # Create bar plot.
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(strategies))
        bars = ax.bar(
            x_pos, means, yerr=stds, capsize=10, color=colors, alpha=0.7
        )
        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel(
            "Mean Final Reward (averaged over mu configs)", fontsize=12
        )
        ax.set_title(
            "Strategy Comparison Across Random Mu Configurations",
            fontsize=14,
            weight="bold",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis="y")
        # Add value labels on bars.
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{mean:.2f}\n±{std:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )
        plt.tight_layout()
        plt.show()


# #############################################################################
# Interactive Widget Functions
# #############################################################################


def cell3_strategy_comparison() -> None:
    """
    Interactive widget for comparing strategies with epsilon sweep.

    User can:
    - Set number of machines and coins.
    - Configure mu values.
    - Set number of trials.
    - Run epsilon sweep and visualize results.
    """
    # Initialize state.
    state = {
        "k_machines": 3,
        "mu_values": [-0.2, 0.0, 0.5],
        "n_coins": 100,
        "n_trials": 10,
        "results": None,
    }
    # Create widgets.
    k_machines_slider, k_machines_box = ut.build_widget_control(
        name="k_machines",
        description="number of machines (K)",
        min_val=2,
        max_val=10,
        step=1,
        initial_value=3,
        is_float=False,
    )
    n_coins_slider, n_coins_box = ut.build_widget_control(
        name="n_coins",
        description="number of coins (N)",
        min_val=50,
        max_val=500,
        step=50,
        initial_value=100,
        is_float=False,
    )
    n_trials_slider, n_trials_box = ut.build_widget_control(
        name="n_trials",
        description="number of trials",
        min_val=5,
        max_val=50,
        step=5,
        initial_value=10,
        is_float=False,
    )
    seed_slider, seed_box = ut.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    # Create mu values text input.
    mu_text = ipywidgets.Text(
        value="-0.2, 0.0, 0.5",
        description="mu values:",
        style={"description_width": "120px"},
    )
    # Create run button.
    run_button = ipywidgets.Button(
        description="Run Epsilon Sweep",
        button_style="success",
        layout={"width": "200px"},
    )
    # Output widget.
    output = ipywidgets.Output()

    def on_run_clicked(b) -> None:
        """
        Handle run button click.
        """
        with output:
            clear_output(wait=True)
            # Parse mu values.
            try:
                mu_values = [float(x.strip()) for x in mu_text.value.split(",")]
                k_machines = k_machines_slider.value
                if len(mu_values) != k_machines:
                    _LOG.warning(
                        "Number of mu values (%d) must equal k_machines (%d)",
                        len(mu_values),
                        k_machines,
                    )
                    return
            except ValueError:
                _LOG.warning(
                    "Invalid mu values format. Use comma-separated numbers."
                )
                return
            # Update state.
            state["k_machines"] = k_machines
            state["mu_values"] = mu_values
            state["n_coins"] = n_coins_slider.value
            state["n_trials"] = n_trials_slider.value
            # Run simulation.
            _LOG.info(
                "Running epsilon sweep with %d trials...", state["n_trials"]
            )
            simulation = BanditSimulation(
                k_machines=state["k_machines"],
                mu_values=state["mu_values"],
                n_coins=state["n_coins"],
                base_seed=seed_slider.value,
            )
            results = simulation.epsilon_sweep(
                n_trials=state["n_trials"],
            )
            state["results"] = results
            # Plot results.
            plot_epsilon_sweep(sweep_results=results, n_coins=state["n_coins"])

    # Connect callback.
    run_button.on_click(on_run_clicked)
    # Layout widgets.
    controls = ipywidgets.VBox(
        [
            k_machines_box,
            mu_text,
            n_coins_box,
            n_trials_box,
            seed_box,
            run_button,
        ]
    )
    # Display widgets.
    display(controls, output)


def cell4_ensemble_comparison() -> None:
    """
    Interactive widget for comparing strategies over random mu configs.

    User can:
    - Set number of machines and coins
    - Configure number of trials and mu configurations
    - Set epsilon value
    - Run ensemble comparison and visualize results
    """
    # Initialize state.
    state = {
        "k_machines": 3,
        "n_coins": 100,
        "n_trials": 10,
        "n_mu_configs": 10,
        "epsilon": 0.1,
        "results": None,
    }
    # Create widgets.
    k_machines_slider, k_machines_box = ut.build_widget_control(
        name="k_machines",
        description="number of machines (K)",
        min_val=2,
        max_val=10,
        step=1,
        initial_value=3,
        is_float=False,
    )
    n_coins_slider, n_coins_box = ut.build_widget_control(
        name="n_coins",
        description="number of coins (N)",
        min_val=50,
        max_val=500,
        step=50,
        initial_value=100,
        is_float=False,
    )
    n_trials_slider, n_trials_box = ut.build_widget_control(
        name="n_trials",
        description="number of trials",
        min_val=5,
        max_val=50,
        step=5,
        initial_value=10,
        is_float=False,
    )
    n_mu_configs_slider, n_mu_configs_box = ut.build_widget_control(
        name="n_mu_configs",
        description="number of mu configs",
        min_val=5,
        max_val=50,
        step=5,
        initial_value=10,
        is_float=False,
    )
    epsilon_slider, epsilon_box = ut.build_widget_control(
        name="epsilon",
        description="epsilon value",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.1,
        is_float=True,
    )
    seed_slider, seed_box = ut.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    # Create run button.
    run_button = ipywidgets.Button(
        description="Run Ensemble Comparison",
        button_style="success",
        layout={"width": "200px"},
    )
    # Output widget.
    output = ipywidgets.Output()

    def on_run_clicked(b) -> None:
        """
        Handle run button click.
        """
        with output:
            clear_output(wait=True)
            # Update state.
            state["k_machines"] = k_machines_slider.value
            state["n_coins"] = n_coins_slider.value
            state["n_trials"] = n_trials_slider.value
            state["n_mu_configs"] = n_mu_configs_slider.value
            state["epsilon"] = epsilon_slider.value
            # Run ensemble.
            _LOG.info(
                "Running ensemble with %d trials and %d mu configs...",
                state["n_trials"],
                state["n_mu_configs"],
            )
            ensemble = BanditEnsemble(
                k_machines=state["k_machines"],
                n_coins=state["n_coins"],
                base_seed=seed_slider.value,
            )
            results = ensemble.compare_strategies_ensemble(
                n_trials=state["n_trials"],
                n_mu_configs=state["n_mu_configs"],
                epsilon=state["epsilon"],
            )
            state["results"] = results
            # Plot results.
            ensemble.plot_ensemble_comparison(
                ensemble_results=results,
                epsilon=state["epsilon"],
            )

    # Connect callback.
    run_button.on_click(on_run_clicked)
    # Layout widgets.
    controls = ipywidgets.VBox(
        [
            k_machines_box,
            n_coins_box,
            n_trials_box,
            n_mu_configs_box,
            epsilon_box,
            seed_box,
            run_button,
        ]
    )
    # Display widgets.
    display(controls, output)
