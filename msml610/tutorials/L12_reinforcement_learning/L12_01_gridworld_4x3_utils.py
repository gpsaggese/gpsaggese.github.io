"""
Utility functions for the 4x3 grid world reinforcement learning lesson.

Implements the canonical AIMA 4x3 grid world from scratch (no gymnasium):
- The stochastic environment (states, transition model, rewards).
- Exact planning (value iteration, policy iteration).
- Model-free learning (Q-learning).
- Interactive notebook cells built on top of these primitives.

Import as:

import msml610.tutorials.L12_reinforcement_learning.L12_01_gridworld_4x3_utils as mtlrll0g4u
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import ipywidgets
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output, display

import helpers.htutorial as htutori

import msml610.tutorials.L12_reinforcement_learning.L12_01_utils as mtlrll0ut

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    mtlrll0ut.init_loggers(notebook_log)


# #############################################################################
# GridWorld
# #############################################################################


class GridWorld:
    """
    The canonical AIMA 4x3 grid world as a Markov decision process.

    - Coordinates are 1-indexed `(col, row)` with `(1, 1)` at the bottom-left.
    - The agent intends an action but slips perpendicular with some
      probability, and bumping a wall or boundary leaves it in place
    - Rewards are collected on arrival: a small living reward for non-terminal
      cells and +1 / -1 for the two terminal cells.
    """

    def __init__(
        self,
        *,
        r_step: float = -0.04,
        gamma: float = 1.0,
        p_intended: float = 0.8,
    ) -> None:
        """
        Build the 4x3 grid world.

        :param r_step: living reward for entering a non-terminal cell
        :param gamma: discount factor
        :param p_intended: probability the intended action succeeds
        """
        self.n_cols = 4
        self.n_rows = 3
        self.start = (1, 1)
        # Terminal cells map to the reward collected on entering them.
        self.terminals = {(4, 3): 1.0, (4, 2): -1.0}
        self.walls = {(2, 2)}
        self.r_step = r_step
        self.gamma = gamma
        self.p_intended = p_intended
        self.actions = list(mtlrll0ut.ACTIONS)
        # Enumerate every reachable cell (all cells except walls).
        self.states = [
            (c, r)
            for r in range(1, self.n_rows + 1)
            for c in range(1, self.n_cols + 1)
            if (c, r) not in self.walls
        ]
        self.nonterminal_states = [
            s for s in self.states if s not in self.terminals
        ]

    def is_terminal(self, s: Tuple[int, int]) -> bool:
        """
        Return whether `s` is a terminal (episode-ending) cell.
        """
        return s in self.terminals

    def reward(self, s: Tuple[int, int]) -> float:
        """
        Return the reward collected on arriving in cell `s`.
        """
        if s in self.terminals:
            return self.terminals[s]
        return self.r_step

    def _attempt_move(
        self, s: Tuple[int, int], direction: str
    ) -> Tuple[int, int]:
        """
        Move one step in `direction`, bouncing back on a wall or boundary.

        :param s: current cell
        :param direction: one of the action names
        :return: resulting cell (equal to `s` if the move is blocked)
        """
        d_col, d_row = mtlrll0ut.ACTION_DELTA[direction]
        target = (s[0] + d_col, s[1] + d_row)
        # Reject moves off the grid or into a wall: the agent stays put.
        off_grid = not (
            1 <= target[0] <= self.n_cols and 1 <= target[1] <= self.n_rows
        )
        if off_grid or target in self.walls:
            return s
        return target

    def transitions(
        self, s: Tuple[int, int], a: str
    ) -> Dict[Tuple[int, int], float]:
        """
        Return the next-state distribution `Pr(s' | s, a)`.

        :param s: current cell
        :param a: intended action
        :return: dict mapping each reachable next cell to its probability
        """
        # Terminal states are absorbing: no transitions leave them.
        if self.is_terminal(s):
            return {}
        p_perp = (1.0 - self.p_intended) / 2.0
        # The intended action and its two perpendicular slips.
        outcome_probs = [(a, self.p_intended)]
        for perp in mtlrll0ut.PERPENDICULAR[a]:
            outcome_probs.append((perp, p_perp))
        # Accumulate probability mass, merging outcomes that land on the
        # same cell (e.g. several blocked moves all bounce back to `s`).
        dist: Dict[Tuple[int, int], float] = {}
        for direction, prob in outcome_probs:
            s2 = self._attempt_move(s, direction)
            dist[s2] = dist.get(s2, 0.0) + prob
        return dist

    def q_value(
        self,
        s: Tuple[int, int],
        a: str,
        u: Dict[Tuple[int, int], float],
    ) -> float:
        """
        Compute the expected one-step value of action `a` in state `s`.

        Uses the reward-on-arrival convention:
        `Q(s, a) = sum_s' Pr(s' | s, a) [R(s') + gamma U(s')]`.

        :param s: current cell
        :param a: action to evaluate
        :param u: current utility estimate over states
        :return: action value
        """
        q = 0.0
        for s2, prob in self.transitions(s, a).items():
            q += prob * (self.reward(s2) + self.gamma * u[s2])
        return q

    def sample_next(
        self,
        s: Tuple[int, int],
        a: str,
        rng: np.random.RandomState,
    ) -> Tuple[int, int]:
        """
        Sample a next state from `Pr(s' | s, a)` (used by model-free learning).

        :param s: current cell
        :param a: chosen action
        :param rng: random state for reproducible sampling
        :return: sampled next cell
        """
        dist = self.transitions(s, a)
        cells = list(dist.keys())
        probs = list(dist.values())
        idx = rng.choice(len(cells), p=probs)
        return cells[idx]

    def to_grid(
        self,
        values: Dict[Tuple[int, int], float],
        *,
        fill: float = np.nan,
    ) -> np.ndarray:
        """
        Convert a dict over states into a 2D array for heatmap plotting.

        Row 3 (top of the grid) is placed in the first array row so the
        heatmap orientation matches the drawn grid.

        :param values: mapping from cell to a scalar value
        :param fill: value used for walls / missing cells
        :return: array of shape `(n_rows, n_cols)`
        """
        return mtlrll0ut.to_grid(values, self.n_rows, self.n_cols, fill=fill)


# #############################################################################
# Exact planning: value iteration and policy iteration
# #############################################################################


def value_iteration(
    env: GridWorld,
    *,
    max_sweeps: int = 100,
    tol: float = 1e-6,
) -> Tuple[List[Dict[Tuple[int, int], float]], List[float]]:
    """
    Run value iteration, recording the utilities after each sweep.

    :param env: grid world to solve
    :param max_sweeps: maximum number of Bellman sweeps
    :param tol: stop once the max per-state change drops below this
    :return: (snapshots, deltas) where
        - snapshots[0] is the all-zero initialization
        - snapshots[i] is the utility dict after sweep i
        - deltas[i] is the max change during sweep i
    """
    # Initialize utilities to zero everywhere (terminals stay at zero).
    u = {s: 0.0 for s in env.states}
    snapshots = [dict(u)]
    deltas: List[float] = []
    for _ in range(max_sweeps):
        u_new = dict(u)
        delta = 0.0
        for s in env.nonterminal_states:
            best = max(env.q_value(s, a, u) for a in env.actions)
            delta = max(delta, abs(best - u[s]))
            u_new[s] = best
        u = u_new
        snapshots.append(dict(u))
        deltas.append(delta)
        if delta < tol:
            break
    return snapshots, deltas


def extract_policy(
    env: GridWorld,
    u: Dict[Tuple[int, int], float],
) -> Dict[Tuple[int, int], str]:
    """
    Extract the greedy policy with respect to utilities `u`.

    :param env: grid world
    :param u: utility estimate over states
    :return: mapping from each non-terminal cell to its greedy action
    """
    policy = {}
    for s in env.nonterminal_states:
        policy[s] = max(env.actions, key=lambda a: env.q_value(s, a, u))
    return policy


def policy_evaluation(
    env: GridWorld,
    policy: Dict[Tuple[int, int], str],
) -> Dict[Tuple[int, int], float]:
    """
    Exactly evaluate a fixed policy by solving the linear Bellman system.

    With the action fixed per state the `max` disappears and the Bellman
    equations become linear, solvable in one shot with `numpy.linalg.solve`.

    :param env: grid world
    :param policy: action to take in each non-terminal state
    :return: utility `U^pi(s)` for every state (terminals are zero)
    """
    states = env.nonterminal_states
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    # Build (I - gamma P) U = b restricted to non-terminal states.
    a_mat = np.eye(n)
    b_vec = np.zeros(n)
    for s in states:
        i = index[s]
        for s2, prob in env.transitions(s, policy[s]).items():
            b_vec[i] += prob * env.reward(s2)
            # Terminal next states contribute zero future utility.
            if s2 in index:
                a_mat[i, index[s2]] -= env.gamma * prob
    solution = np.linalg.solve(a_mat, b_vec)
    u = {s: 0.0 for s in env.states}
    for s in states:
        u[s] = float(solution[index[s]])
    return u


def policy_iteration(
    env: GridWorld,
    *,
    initial_policy: Optional[Dict[Tuple[int, int], str]] = None,
    max_iters: int = 50,
) -> Tuple[List[Dict[Tuple[int, int], str]], List[int]]:
    """
    Run policy iteration, recording the policy after each improvement.

    :param env: grid world to solve
    :param initial_policy: starting policy (defaults to "Up" everywhere)
    :param max_iters: maximum evaluate/improve rounds
    :return: (policies, changes) where policies[i] is the policy after round i
        (policies[0] is the initial policy) and changes[i] is the number of
        states whose action changed during round i
    """
    if initial_policy is None:
        policy = {s: "Up" for s in env.nonterminal_states}
    else:
        policy = dict(initial_policy)
    policies = [dict(policy)]
    changes: List[int] = []
    for _ in range(max_iters):
        u = policy_evaluation(env, policy)
        new_policy = extract_policy(env, u)
        n_changed = sum(
            1 for s in env.nonterminal_states if new_policy[s] != policy[s]
        )
        policy = new_policy
        policies.append(dict(policy))
        changes.append(n_changed)
        # Convergence: the policy is stable and therefore optimal.
        if n_changed == 0:
            break
    return policies, changes


# #############################################################################
# Model-free learning: Q-learning
# #############################################################################


def _greedy_action(
    env: GridWorld,
    q: Dict[Tuple[Tuple[int, int], str], float],
    s: Tuple[int, int],
) -> str:
    """
    Return the action with the highest current Q-value at state `s`.
    """
    return max(env.actions, key=lambda a: q[(s, a)])


def _greedy_policy_from_q(
    env: GridWorld,
    q: Dict[Tuple[Tuple[int, int], str], float],
) -> Dict[Tuple[int, int], str]:
    """
    Derive the greedy policy implied by a Q-table.
    """
    return {s: _greedy_action(env, q, s) for s in env.nonterminal_states}


def q_learning(
    env: GridWorld,
    *,
    n_episodes: int,
    alpha: float,
    epsilon: float,
    seed: int,
    max_steps: int = 100,
    snapshot_every: int = 0,
) -> Dict[str, Any]:
    """
    Learn the optimal policy from experience tuples with tabular Q-learning.

    :param env: grid world (its transition model is used only to sample
        experience, never read directly by the learner)
    :param n_episodes: number of training episodes
    :param alpha: learning rate
    :param epsilon: exploration probability for epsilon-greedy action choice
    :param seed: random seed for reproducibility
    :param max_steps: step cap per episode (guards against non-terminating runs)
    :param snapshot_every: if > 0, store the greedy policy every this many
        episodes
    :return: dict with the final Q-table, per-episode returns, visit counts,
        the final greedy policy, and optional policy snapshots
    """
    rng = np.random.RandomState(seed)
    # Initialize all Q-values to zero.
    q = {(s, a): 0.0 for s in env.states for a in env.actions}
    visit_counts = {s: 0 for s in env.states}
    returns: List[float] = []
    snapshots: List[Tuple[int, Dict[Tuple[int, int], str]]] = []
    for episode in range(n_episodes):
        s = env.start
        total_reward = 0.0
        for _ in range(max_steps):
            visit_counts[s] += 1
            # Epsilon-greedy action selection.
            if rng.rand() < epsilon:
                a = env.actions[rng.randint(len(env.actions))]
            else:
                a = _greedy_action(env, q, s)
            s2 = env.sample_next(s, a, rng)
            r = env.reward(s2)
            total_reward += r
            # TD target uses the best next-state Q (zero past a terminal).
            if env.is_terminal(s2):
                td_target = r
            else:
                best_next = max(q[(s2, a2)] for a2 in env.actions)
                td_target = r + env.gamma * best_next
            q[(s, a)] += alpha * (td_target - q[(s, a)])
            s = s2
            if env.is_terminal(s):
                break
        returns.append(total_reward)
        if snapshot_every > 0 and (episode + 1) % snapshot_every == 0:
            snapshots.append((episode + 1, _greedy_policy_from_q(env, q)))
    result = {
        "q": q,
        "returns": returns,
        "visit_counts": visit_counts,
        "policy": _greedy_policy_from_q(env, q),
        "snapshots": snapshots,
    }
    return result


def max_q_values(
    env: GridWorld,
    q: Dict[Tuple[Tuple[int, int], str], float],
) -> Dict[Tuple[int, int], float]:
    """
    Return `max_a Q(s, a)` per state, the learned value estimate.
    """
    return {s: max(q[(s, a)] for a in env.actions) for s in env.states}


# #############################################################################
# Cell 1.1: The 4x3 grid and its states
# #############################################################################


def cell1_1_show_grid(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Draw the 4x3 grid world layout that every later algorithm reasons about.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (7, 5)
    env = GridWorld()
    _, ax = plt.subplots(figsize=figsize)
    mtlrll0ut.draw_grid_base(env, ax, annotate_coords=True)
    ax.set_title(
        "The 4x3 grid world (11 reachable states)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    print("env.states (n=%d):" % len(env.states), env.states)
    print("env.terminals=", env.terminals)
    print("env.walls=", env.walls)


# #############################################################################
# Cell 1.2: Stochastic action model
# #############################################################################


def cell1_2_stochastic_action(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Interactively show how an intended action spreads probability mass.

    Interactive controls (ipywidgets):
    :param action: the intended action (``Up``, ``Down``, ``Left``, ``Right``)
    :param p_intended: probability the intended action succeeds (0.5 to 1.0)
    :param figsize: optional figure size

    Layout is a 2x2 grid: controls | HTML description, grid plot | comments.
    """
    if figsize is None:
        figsize = (14, 5)
    env = GridWorld()
    state_dropdown = ipywidgets.Dropdown(
        options=[str(s) for s in env.nonterminal_states],
        value=str(env.start),
        description="state:",
        style={"description_width": "initial"},
    )
    action_dropdown = ipywidgets.Dropdown(
        options=mtlrll0ut.ACTIONS,
        value="Up",
        description="action:",
        style={"description_width": "initial"},
    )
    p_slider, p_box = htutori.build_widget_control(
        name="p_intended",
        description="probability intended action succeeds",
        min_val=0.5,
        max_val=1.0,
        step=0.05,
        initial_value=0.8,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.p_intended = p_slider.value
            action = action_dropdown.value
            focal = eval(state_dropdown.value)
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            # Panel left: grid plot with probability arrows.
            mtlrll0ut.draw_grid_base(env, ax1, highlight=focal)
            for direction in mtlrll0ut.ACTIONS:
                s2 = env._attempt_move(focal, direction)
                if direction == action:
                    prob = env.p_intended
                elif direction in mtlrll0ut.PERPENDICULAR[action]:
                    prob = (1.0 - env.p_intended) / 2.0
                else:
                    prob = 0.0
                if prob == 0.0:
                    continue
                d_x, d_y = mtlrll0ut.ACTION_DELTA[direction]
                if s2 == focal:
                    ax1.text(
                        focal[0],
                        focal[1],
                        "stay\n%.2f" % prob,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="firebrick",
                    )
                else:
                    ax1.annotate(
                        "",
                        xy=(focal[0] + d_x * 0.45, focal[1] + d_y * 0.45),
                        xytext=(focal[0], focal[1]),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color="steelblue",
                            linewidth=1.0 + 8.0 * prob,
                        ),
                    )
                    ax1.text(
                        focal[0] + d_x * 0.62,
                        focal[1] + d_y * 0.62,
                        "%.2f" % prob,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="steelblue",
                    )
            ax1.set_title(
                "Intended action '%s' from cell %s" % (action, focal),
                fontsize=13,
                fontweight="bold",
            )
            ax1.set_xlabel(
                "col\n\n"
                "the intended action and the three "
                "possible outcomes, with arrow thickness encoding probability",
                fontsize=9,
            )
            # Panel right: comments.
            p_perp = (1.0 - env.p_intended) / 2.0
            text = (
                "Parameters:\n"
                "  action: %s\n"
                "  p_intended: %.2f\n\n"
                "Outcome probabilities:\n"
                "  intended: %.2f\n"
                "  each perpendicular: %.2f"
                % (action, env.p_intended, env.p_intended, p_perp)
            )
            mtlrll0ut.comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "state": "which cell the agent is currently in",
            "action": "intended direction (<code>Up</code>, <code>Down</code>, <code>Left</code>, <code>Right</code>)",
            "p_intended": "probability the intended action succeeds (0.50 to 1.00); the remaining 1&minus;p is split evenly across the two perpendicular moves",
        }
    )
    state_dropdown.observe(update_plot, names="value")
    action_dropdown.observe(update_plot, names="value")
    p_slider.observe(update_plot, names="value")
    update_plot()
    # Build the 2x2 grid layout using the standard template pattern:
    #   Top row: controls on the left, info box on the right.
    #   Bottom row: the plot and comments (as a 1x2 matplotlib figure).
    controls = ipywidgets.VBox(
        [
            state_dropdown,
            action_dropdown,
            p_box,
        ],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 1.3: Transition model as an explicit table
# #############################################################################


def cell1_3_full_transition_model(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Print the full transition model Pr(s' | s, a) for every state-action pair.

    Displays a multi-row table where each row shows the probability
    distribution over next states for one (s, a) pair.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (14, 5)
    env = GridWorld()
    # Build a table with one row per (s, a, s') triple.
    rows = []
    for s in env.nonterminal_states:
        for a in env.actions:
            dist = env.transitions(s, a)
            for s2, prob in sorted(dist.items(), key=lambda x: -x[1]):
                rows.append(
                    {
                        "state": str(s),
                        "action": a,
                        "next state s'": str(s2),
                        "Pr(s' | s, a)": round(prob, 3),
                    }
                )
    df = pd.DataFrame(rows)
    # Also show a heatmap of all transitions per action as subplots.
    n_actions = len(env.actions)
    _, axes = plt.subplots(1, n_actions, figsize=figsize)
    for idx, a in enumerate(env.actions):
        ax = axes[idx]
        # Build a mapping from (s, s2) to probability for this action.
        prob_grid_val = {}
        for s in env.nonterminal_states:
            dist = env.transitions(s, a)
            for s2, prob in dist.items():
                prob_grid_val[(s, s2)] = prob_grid_val.get((s, s2), 0.0) + prob
        # Create a matrix where rows = from-state index, cols = to-state index.
        state_list = env.states
        n_states = len(state_list)
        mat = np.zeros((n_states, n_states))
        for i, s in enumerate(state_list):
            if s in env.nonterminal_states:
                dist = env.transitions(s, a)
                for s2, prob in dist.items():
                    j = state_list.index(s2)
                    mat[i, j] = prob
        sns.heatmap(
            mat,
            ax=ax,
            cmap="Blues",
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 5},
            cbar=False,
            linewidths=0.5,
            linecolor="black",
            vmin=0.0,
            vmax=1.0,
            xticklabels=[str(s) for s in state_list],
            yticklabels=[str(s) for s in state_list],
        )
        ax.set_title("Action: %s" % a, fontsize=12, fontweight="bold")
        ax.set_xlabel("next state s'")
        ax.set_ylabel("state s")
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=6
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    plt.suptitle(
        "Transition model Pr(s' | s, a) for each action",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    print("\nFull transition probability table:")
    display(df)


def cell1_3_show_transition_table(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Display the transition model as a concrete probability table for a fixed
    state-action pair.

    Shows the Pr(s' | s, a) distribution for the START state with the Up action
    as a pandas DataFrame, making the abstract model explicit.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (7, 5)
    env = GridWorld()
    s = env.start
    a = "Up"
    dist = env.transitions(s, a)
    # Build the probability table as a sorted DataFrame.
    rows = []
    for s2, prob in sorted(dist.items(), key=lambda x: -x[1]):
        rows.append(
            {
                "state s": str(s),
                "action a": a,
                "next state s'": str(s2),
                "Pr(s' | s, a)": round(prob, 3),
            }
        )
    df = pd.DataFrame(rows)
    # Also draw the grid shaded by transition probability.
    prob_map = {cell: 0.0 for cell in env.states}
    for s2, p in dist.items():
        prob_map[s2] = p
    _, ax = plt.subplots(figsize=figsize)
    grid = env.to_grid(prob_map, fill=np.nan)
    sns.heatmap(
        grid,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar=False,
        linewidths=1.0,
        linecolor="black",
        vmin=0.0,
        vmax=1.0,
        mask=np.isnan(grid),
    )
    # Highlight the source state.
    ax.add_patch(
        mpatches.Rectangle(
            (s[0] - 1, env.n_rows - s[1]),
            1.0,
            1.0,
            fill=False,
            edgecolor="darkorange",
            linewidth=3.5,
        )
    )
    # Mark the wall cell with a grey patch and label.
    for wall in env.walls:
        ax.add_patch(
            mpatches.Rectangle(
                (wall[0] - 1, env.n_rows - wall[1]),
                1.0,
                1.0,
                facecolor=mtlrll0ut.COLOR_WALL,
                edgecolor="black",
            )
        )
        ax.text(
            wall[0] - 0.5,
            env.n_rows - wall[1] + 0.5,
            "WALL",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_title(
        "Pr(s' | s=%s, a=%s)" % (s, a),
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticklabels([str(c) for c in range(1, env.n_cols + 1)])
    ax.set_yticklabels([str(r) for r in range(env.n_rows, 0, -1)], rotation=0)
    ax.set_xlabel(
        "col\n\nreachable next states shaded by probability",
        fontsize=9,
    )
    ax.set_ylabel("row")
    plt.tight_layout()
    plt.show()
    # Display the probability table as a DataFrame.
    display(df)


def cell1_3_transition_table(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Display the explicit `Pr(s' | s, a)` row for a chosen state and action.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (14, 5)
    env = GridWorld()
    state_dropdown = ipywidgets.Dropdown(
        options=[str(s) for s in env.nonterminal_states],
        value=str(env.start),
        description="state:",
        style={"description_width": "initial"},
    )
    action_dropdown = ipywidgets.Dropdown(
        options=mtlrll0ut.ACTIONS,
        value="Up",
        description="action:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            s = eval(state_dropdown.value)
            a = action_dropdown.value
            dist = env.transitions(s, a)
            # Build the probability row as a DataFrame for the comments panel.
            df = pd.DataFrame(
                {
                    "next_state": [str(s2) for s2 in dist],
                    "probability": [round(p, 3) for p in dist.values()],
                }
            ).sort_values("probability", ascending=False)
            # Two-panel layout: heatmap + comments with probability table.
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            # Panel 1: grid shaded by transition probability.
            prob_map = {cell: 0.0 for cell in env.states}
            for s2, p in dist.items():
                prob_map[s2] = p
            grid = env.to_grid(prob_map, fill=np.nan)
            # Mask wall cells; zero-prob cells kept unmasked so annotations
            # show, then overlaid with white patches below.
            wall_mask = np.isnan(grid)
            sns.heatmap(
                grid,
                ax=ax1,
                cmap="Blues",
                annot=True,
                fmt=".2f",
                cbar=False,
                linewidths=1.0,
                linecolor="black",
                vmin=0.0,
                vmax=1.0,
                mask=wall_mask,
            )
            # Overlay white patches on zero-probability cells.
            for col in range(1, env.n_cols + 1):
                for row in range(1, env.n_rows + 1):
                    cell = (col, row)
                    if cell not in env.walls and prob_map[cell] == 0.0:
                        ax1.add_patch(
                            mpatches.Rectangle(
                                (col - 1, env.n_rows - row),
                                1.0,
                                1.0,
                                facecolor="white",
                                edgecolor="black",
                                linewidth=1.0,
                            )
                        )
                        ax1.text(
                            col - 0.5,
                            env.n_rows - row + 0.5,
                            "0.00",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="lightgray",
                        )
            # Mark wall cells with grey patches and "WALL" label, matching
            # the style from cell1_2.
            for wall in env.walls:
                ax1.add_patch(
                    mpatches.Rectangle(
                        (wall[0] - 1, env.n_rows - wall[1]),
                        1.0,
                        1.0,
                        facecolor=mtlrll0ut.COLOR_WALL,
                        edgecolor="black",
                    )
                )
                ax1.text(
                    wall[0] - 0.5,
                    env.n_rows - wall[1] + 0.5,
                    "WALL",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
            # Highlight the source state.
            ax1.add_patch(
                mpatches.Rectangle(
                    (s[0] - 1, env.n_rows - s[1]),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="darkorange",
                    linewidth=3.5,
                )
            )
            ax1.set_title(
                "Pr(s' | s=%s, a=%s)" % (s, a),
                fontsize=13,
                fontweight="bold",
            )
            ax1.set_xticklabels([str(c) for c in range(1, env.n_cols + 1)])
            ax1.set_yticklabels(
                [str(r) for r in range(env.n_rows, 0, -1)], rotation=0
            )
            ax1.set_xlabel(
                "col\n\nreachable next states shaded by probability",
                fontsize=9,
            )
            ax1.set_ylabel("row")
            # Panel 2: comments with the probability table.
            table_text = "Probability table:\n\n"
            for _, row in df.iterrows():
                table_text += "  %s -> %s:  %.3f\n" % (
                    s,
                    row["next_state"],
                    row["probability"],
                )
            table_text += "\nsum of probabilities: %.6f" % round(
                sum(dist.values()), 6
            )
            text = "Parameters:\n  state: %s\n  action: %s\n\n%s" % (
                s,
                a,
                table_text,
            )
            mtlrll0ut.comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "state": "which non-terminal cell to query the model for",
            "action": "intended direction (<code>Up</code>, <code>Down</code>, <code>Left</code>, <code>Right</code>)",
        }
    )
    state_dropdown.observe(update_plot, names="value")
    action_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            state_dropdown,
            action_dropdown,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 1.4: Rewards and episode returns
# #############################################################################


def _sample_trajectory(
    env: GridWorld,
    policy: Dict[Tuple[int, int], str],
    *,
    seed: int,
    max_steps: int = 30,
) -> List[Tuple[int, int]]:
    """
    Roll out a trajectory from START following `policy` under the slip model.
    """
    rng = np.random.RandomState(seed)
    s = env.start
    path: List[Tuple[int, int]] = [s]
    for _ in range(max_steps):
        if env.is_terminal(s):
            break
        s = env.sample_next(s, policy[s], rng)
        path.append(s)
    return path


def cell1_4_rewards_and_returns(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show per-cell rewards and the discounted return of a sample trajectory.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (14, 5)
    env = GridWorld()
    # A fixed sensible policy used only to produce an illustrative path.
    base_policy = {
        (1, 1): "Up",
        (1, 2): "Up",
        (1, 3): "Right",
        (2, 3): "Right",
        (3, 3): "Right",
        (2, 1): "Right",
        (3, 1): "Up",
        (4, 1): "Up",
        (3, 2): "Up",
    }
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="trajectory seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=1,
        is_float=False,
    )
    r_slider, r_box = htutori.build_widget_control(
        name="r_step",
        description="living reward per step",
        min_val=-1.0,
        max_val=0.0,
        step=0.02,
        initial_value=-0.04,
        is_float=True,
    )
    gamma_slider, gamma_box = htutori.build_widget_control(
        name="gamma",
        description="discount factor",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.9,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.r_step = r_slider.value
            gamma = gamma_slider.value
            path = _sample_trajectory(env, base_policy, seed=seed_slider.value)
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            # Panel 1: grid annotated with rewards and the trajectory path.
            mtlrll0ut.draw_grid_base(env, ax1)
            for col in range(1, env.n_cols + 1):
                for row in range(1, env.n_rows + 1):
                    cell = (col, row)
                    if cell in env.walls:
                        continue
                    ax1.text(
                        col,
                        row - 0.05,
                        "%.2f" % env.reward(cell),
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
            xs = [c[0] for c in path]
            ys = [c[1] for c in path]
            ax1.plot(xs, ys, "-o", color="darkblue", linewidth=2.0, markersize=6)
            ax1.set_title(
                "Rewards and a sample trajectory",
                fontsize=13,
                fontweight="bold",
            )
            ax1.set_xlabel(
                "col\n\nper-cell rewards with a sample path from START",
                fontsize=9,
            )
            # Panel 2: comments with running discounted return.
            running = 0.0
            lines = []
            for t in range(1, len(path)):
                r_t = env.reward(path[t])
                running += (gamma**t) * r_t
                if t <= 8:
                    lines.append("  t=%d enter %s r=%.2f" % (t, path[t], r_t))
            text = (
                "Parameters:\n"
                "  r_step: %.2f\n"
                "  gamma: %.2f\n\n"
                "Discounted return G = %.3f\n\n"
                "First steps:\n%s"
                % (env.r_step, gamma, running, "\n".join(lines))
            )
            mtlrll0ut.comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "seed": "random seed controlling the trajectory path taken",
            "r_step": "living reward collected for each non-terminal step (more negative = stronger penalty for wandering)",
            "gamma": "discount factor (near 1 values distant rewards; near 0 cares only about immediate reward)",
        }
    )
    r_slider.observe(update_plot, names="value")
    gamma_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            seed_box,
            r_box,
            gamma_box,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.2 (equations): Bellman optimality equation system for all states
# #############################################################################


def cell2_2_bellman_equations(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Interactively show the Bellman optimality equation for a chosen state.

    For the selected state:
        U(s) = max_a sum_{s'} Pr(s' | s, a) [R(s') + gamma * U(s')]

    With the optimal utilities plugged in so the equation is satisfied by
    the best action.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (14, 8)
    env = GridWorld()
    state_dropdown = ipywidgets.Dropdown(
        options=[str(s) for s in env.nonterminal_states],
        value=str(env.start),
        description="state:",
        style={"description_width": "initial"},
    )
    gamma_slider, gamma_box = htutori.build_widget_control(
        name="gamma",
        description="discount factor",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.9,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.gamma = gamma_slider.value
            s = eval(state_dropdown.value)
            snapshots, _ = value_iteration(env)
            u = snapshots[-1]
            policy = extract_policy(env, u)
            a_best = policy[s]
            q_a = env.q_value(s, a_best, u)
            dist = env.transitions(s, a_best)
            term_lines = []
            for s2, prob in sorted(dist.items(), key=lambda x: -x[1]):
                r = env.reward(s2)
                term = "%.2f * (%.2f + %.2f * %.3f)" % (
                    prob,
                    r,
                    env.gamma,
                    u[s2],
                )
                term_lines.append("         %s" % term)
            eq = (
                "Bellman optimality equation for %s:\n\n"
                "U(%s) = max_a Q(%s, a) = %.3f\n\n"
                "Best action: %s\n\n"
                "sum_{s'} Pr(s'|s,%s)[R(s') + gamma U(s')]:\n"
                "%s"
                % (
                    s,
                    s,
                    s,
                    q_a,
                    a_best,
                    a_best,
                    "\n".join(term_lines),
                )
            )
            _, ax = plt.subplots(figsize=figsize)
            ax.axis("off")
            ax.set_title(
                "Bellman optimality equation for %s" % (s,),
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            htutori.add_fitted_text_box(ax, eq, max_fontsize=13, min_fontsize=9)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "state": "which cell to inspect the Bellman equation for",
            "gamma": "discount factor (higher gamma weights future rewards more)",
        }
    )
    state_dropdown.observe(update_plot, names="value")
    gamma_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            state_dropdown,
            gamma_box,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.1: The Bellman equation for one state
# #############################################################################


def cell2_1_bellman_one_state(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show the value of each action at one state under converged utilities.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (16, 5)
    env = GridWorld()
    state_dropdown = ipywidgets.Dropdown(
        options=[str(s) for s in env.nonterminal_states],
        value="(3, 1)",
        description="state:",
        style={"description_width": "initial"},
    )
    gamma_slider, gamma_box = htutori.build_widget_control(
        name="gamma",
        description="discount factor",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.9,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.gamma = gamma_slider.value
            s = eval(state_dropdown.value)
            # Solve the MDP to get utilities used in the one-step lookahead.
            snapshots, _ = value_iteration(env)
            u = snapshots[-1]
            q_vals = {a: env.q_value(s, a, u) for a in env.actions}
            best_action = max(q_vals, key=lambda a: q_vals[a])
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: grid with the inspected state highlighted.
            mtlrll0ut.draw_grid_base(env, ax1, highlight=s)
            ax1.set_title(
                "Inspected state %s" % (s,), fontsize=13, fontweight="bold"
            )
            ax1.set_xlabel(
                "col\n\nthe highlighted cell on the grid",
                fontsize=9,
            )
            # Panel 2: bar chart of action values, best one highlighted.
            colors = [
                "seagreen" if a == best_action else "steelblue"
                for a in env.actions
            ]
            ax2.bar(env.actions, [q_vals[a] for a in env.actions], color=colors)
            ax2.axhline(0.0, color="black", linewidth=0.8)
            ax2.set_ylabel("expected action value Q(s, a)")
            ax2.set_title(
                "Action values (max kept)", fontsize=13, fontweight="bold"
            )
            ax2.set_xlabel(
                "action\n\na bar per action, the best (max) highlighted",
                fontsize=9,
            )
            ax2.grid(True, alpha=0.3)
            # Panel 3: comments.
            text = (
                "State: %s\n"
                "gamma: %.2f\n\n"
                "Action values:\n%s\n\n"
                "Best action: %s\n"
                "U(s) = %.3f"
                % (
                    s,
                    env.gamma,
                    "\n".join(
                        "  %-6s %.3f" % (a, q_vals[a]) for a in env.actions
                    ),
                    best_action,
                    q_vals[best_action],
                )
            )
            mtlrll0ut.comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "state": "which cell to inspect the action values for",
            "gamma": "discount factor controlling how future rewards are weighted",
        }
    )
    state_dropdown.observe(update_plot, names="value")
    gamma_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            state_dropdown,
            gamma_box,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.3: Value iteration converging over sweeps
# #############################################################################


def cell2_3_value_iteration(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Step through value iteration sweeps and watch utilities converge.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (16, 5)
    env = GridWorld()
    iter_slider, iter_box = htutori.build_widget_control(
        name="iteration",
        description="value iteration sweep",
        min_val=0,
        max_val=30,
        step=1,
        initial_value=0,
        is_float=False,
    )
    gamma_slider, gamma_box = htutori.build_widget_control(
        name="gamma",
        description="discount factor",
        min_val=0.5,
        max_val=1.0,
        step=0.05,
        initial_value=0.9,
        is_float=True,
    )
    r_slider, r_box = htutori.build_widget_control(
        name="r_step",
        description="living reward",
        min_val=-1.0,
        max_val=0.0,
        step=0.02,
        initial_value=-0.04,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.gamma = gamma_slider.value
            env.r_step = r_slider.value
            snapshots, deltas = value_iteration(env, max_sweeps=30)
            # Clamp the requested sweep to what is available.
            i = min(iter_slider.value, len(snapshots) - 1)
            u = snapshots[i]
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            mtlrll0ut.draw_value_heatmap(
                env, ax1, u, title="Utilities after sweep %d" % i
            )
            ax1.set_xlabel(
                "col\n\neach cell annotated with its current U(s)",
                fontsize=9,
            )
            # Panel 2: convergence curve of the max change per sweep.
            ax2.plot(
                range(1, len(deltas) + 1),
                deltas,
                "-o",
                color="darkorange",
            )
            if 1 <= i <= len(deltas):
                ax2.axvline(i, color="gray", linestyle="--")
            ax2.set_xlabel(
                "sweep\n\nmax |U_{i+1} - U_i| per sweep",
                fontsize=9,
            )
            ax2.set_ylabel("max |U_{i+1} - U_i|")
            ax2.set_title("Convergence", fontsize=13, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            # Panel 3: comments with current state information only.
            converged = len(deltas)
            text = (
                "Parameters:\n"
                "  sweep: %d / %d\n"
                "  gamma: %.2f\n"
                "  r_step: %.2f\n\n"
                "U(start) = %.3f\n"
                "sweeps to converge: %d"
                % (
                    i,
                    len(snapshots) - 1,
                    env.gamma,
                    env.r_step,
                    u[env.start],
                    converged,
                )
            )
            mtlrll0ut.comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "iteration": "which Bellman sweep to display (value propagates backward from terminals one ring per sweep)",
            "gamma": "discount factor (higher gamma propagates value further but converges more slowly)",
            "r_step": "living reward collected for each non-terminal step",
        }
    )
    iter_slider.observe(update_plot, names="value")
    gamma_slider.observe(update_plot, names="value")
    r_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            iter_box,
            gamma_box,
            r_box,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.4: Extracting the optimal policy
# #############################################################################


def cell2_4_extract_policy(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show the greedy policy extracted from converged utilities.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (14, 5)
    env = GridWorld()
    r_slider, r_box = htutori.build_widget_control(
        name="r_step",
        description="living reward",
        min_val=-2.0,
        max_val=0.0,
        step=0.05,
        initial_value=-0.04,
        is_float=True,
    )
    gamma_slider, gamma_box = htutori.build_widget_control(
        name="gamma",
        description="discount factor",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.9,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.r_step = r_slider.value
            env.gamma = gamma_slider.value
            snapshots, _ = value_iteration(env)
            u = snapshots[-1]
            policy = extract_policy(env, u)
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            # Panel 1: utilities heatmap with policy arrows overlaid.
            mtlrll0ut.draw_value_heatmap(
                env, ax1, u, title="Optimal policy over utilities"
            )
            mtlrll0ut.draw_policy_arrows_heatmap(env, ax1, policy)
            ax1.set_xlabel(
                "col\n\nan arrow per cell over the utility heatmap",
                fontsize=9,
            )
            # Panel 2: comments.
            text = (
                "Parameters:\n"
                "  r_step: %.2f\n"
                "  gamma: %.2f\n\n"
                "U(start) = %.3f" % (env.r_step, env.gamma, u[env.start])
            )
            mtlrll0ut.comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "r_step": "living reward per step; a large negative value makes the agent take the short risky path, a near-zero value makes it take the long safe path",
            "gamma": "discount factor (higher gamma weights future rewards more, changing which actions are greedy)",
        }
    )
    r_slider.observe(update_plot, names="value")
    gamma_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            r_box,
            gamma_box,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 3.1: Policy evaluation for a fixed policy
# #############################################################################


# Preset policies used to illustrate policy evaluation.
def _preset_policy(env: GridWorld, name: str) -> Dict[Tuple[int, int], str]:
    """
    Return one of several named preset policies over the non-terminal states.
    """
    if name == "always-up":
        return {s: "Up" for s in env.nonterminal_states}
    if name == "always-right":
        return {s: "Right" for s in env.nonterminal_states}
    if name == "random":
        rng = np.random.RandomState(0)
        return {
            s: env.actions[rng.randint(len(env.actions))]
            for s in env.nonterminal_states
        }
    # Hand-tuned reasonable policy.
    return {
        (1, 1): "Up",
        (1, 2): "Up",
        (1, 3): "Right",
        (2, 3): "Right",
        (3, 3): "Right",
        (2, 1): "Left",
        (3, 1): "Up",
        (4, 1): "Up",
        (3, 2): "Up",
    }


def cell3_1_policy_evaluation(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Evaluate a fixed policy by solving the linear Bellman system.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (14, 5)
    env = GridWorld()
    policy_dropdown = ipywidgets.Dropdown(
        options=["random", "always-up", "always-right", "hand-tuned"],
        value="hand-tuned",
        description="policy:",
        style={"description_width": "initial"},
    )
    gamma_slider, gamma_box = htutori.build_widget_control(
        name="gamma",
        description="discount factor",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.9,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.gamma = gamma_slider.value
            policy = _preset_policy(env, policy_dropdown.value)
            u = policy_evaluation(env, policy)
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            mtlrll0ut.draw_value_heatmap(
                env, ax1, u, title="U^pi for policy '%s'" % policy_dropdown.value
            )
            mtlrll0ut.draw_policy_arrows_heatmap(env, ax1, policy)
            ax1.set_xlabel(
                "col\n\nthe fixed policy as arrows over its evaluated utilities",
                fontsize=9,
            )
            text = (
                "Parameters:\n"
                "  policy: %s\n"
                "  gamma: %.2f\n\n"
                "U(start) = %.3f"
                % (policy_dropdown.value, env.gamma, u[env.start])
            )
            mtlrll0ut.comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "policy": "which fixed policy to evaluate (<code>hand-tuned</code>, <code>always-up</code>, <code>always-right</code>, <code>random</code>)",
            "gamma": "discount factor weighting future rewards",
        }
    )
    policy_dropdown.observe(update_plot, names="value")
    gamma_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            policy_dropdown,
            gamma_box,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 3.2: Policy improvement and iteration to optimality
# #############################################################################


def cell3_2_policy_iteration(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Step through policy iteration rounds and watch arrows flip to optimal.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (16, 5)
    env = GridWorld()
    # Start from a deliberately poor policy so improvement is visible.
    initial = {s: "Down" for s in env.nonterminal_states}
    policies, changes = policy_iteration(env, initial_policy=initial)
    iter_slider, iter_box = htutori.build_widget_control(
        name="iteration",
        description="evaluate/improve round",
        min_val=0,
        max_val=len(policies) - 1,
        step=1,
        initial_value=0,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            i = min(iter_slider.value, len(policies) - 1)
            before = policies[max(i - 1, 0)]
            after = policies[i]
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            mtlrll0ut.draw_grid_base(env, ax1)
            mtlrll0ut.draw_policy_arrows(env, ax1, before, color="gray")
            ax1.set_title(
                "Policy before round %d" % i, fontsize=13, fontweight="bold"
            )
            ax1.set_xlabel(
                "col\n\nthe current policy arrows",
                fontsize=9,
            )
            mtlrll0ut.draw_grid_base(env, ax2)
            mtlrll0ut.draw_policy_arrows(env, ax2, after, color="darkblue")
            ax2.set_title(
                "Policy after round %d" % i, fontsize=13, fontweight="bold"
            )
            ax2.set_xlabel(
                "col\n\nthe improved policy arrows",
                fontsize=9,
            )
            n_changed = changes[i - 1] if 1 <= i <= len(changes) else 0
            text = (
                "Round: %d / %d\n\n"
                "States changed action: %d\n"
                "rounds to converge: %d"
                % (i, len(policies) - 1, n_changed, len(changes))
            )
            mtlrll0ut.comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "iteration": "which evaluate/improve round to display (the policy converges in very few rounds)",
        }
    )
    iter_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            iter_box,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 3.3: Value iteration vs policy iteration
# #############################################################################


def cell3_3_compare_solvers(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Compare convergence of value iteration and policy iteration.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (16, 5)
    env = GridWorld()
    gamma_slider, gamma_box = htutori.build_widget_control(
        name="gamma",
        description="discount factor",
        min_val=0.5,
        max_val=0.99,
        step=0.01,
        initial_value=0.9,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.gamma = gamma_slider.value
            _, vi_deltas = value_iteration(env, max_sweeps=200)
            initial = {s: "Down" for s in env.nonterminal_states}
            _, pi_changes = policy_iteration(env, initial_policy=initial)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: value iteration convergence.
            ax1.plot(
                range(1, len(vi_deltas) + 1),
                vi_deltas,
                "-o",
                color="darkorange",
            )
            ax1.set_xlabel(
                "sweep\n\nutility change per sweep",
                fontsize=9,
            )
            ax1.set_ylabel("max utility change")
            ax1.set_title("Value iteration", fontsize=13, fontweight="bold")
            ax1.grid(True, alpha=0.3)
            # Panel 2: policy iteration convergence.
            ax2.plot(
                range(1, len(pi_changes) + 1),
                pi_changes,
                "-s",
                color="seagreen",
            )
            ax2.set_xlabel(
                "round\n\nchanged-action count per round",
                fontsize=9,
            )
            ax2.set_ylabel("states changed action")
            ax2.set_title("Policy iteration", fontsize=13, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            # Panel 3: comments with a small summary table.
            text = (
                "gamma: %.2f\n\n"
                "value iteration sweeps: %d\n"
                "policy iteration rounds: %d"
                % (env.gamma, len(vi_deltas), len(pi_changes))
            )
            mtlrll0ut.comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "gamma": "discount factor; as gamma approaches 1, value iteration needs many more sweeps while policy iteration stays at a handful of rounds",
        }
    )
    gamma_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            gamma_box,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 4.1: Why reinforcement learning is harder than planning
# #############################################################################


def cell4_1_planning_vs_learning(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Contrast planning (knows the model) with learning (must experience it).

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (14, 5)
    env = GridWorld()
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # Panel 1: the grid with the transition table marked unknown.
    mtlrll0ut.draw_grid_base(env, ax1)
    ax1.text(
        2.5,
        2.0,
        "Pr(s' | s, a)\nUNKNOWN\nto the agent",
        ha="center",
        va="center",
        fontsize=14,
        color="firebrick",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    ax1.set_title("Same world, blindfolded", fontsize=13, fontweight="bold")
    # Panel 2: environment parameters only.
    text = "Environment:\n  r_step: %.2f\n  gamma: %.2f\n  p_intended: %.2f" % (
        env.r_step,
        env.gamma,
        env.p_intended,
    )
    mtlrll0ut.comment_panel(ax2, text)
    plt.tight_layout()
    plt.show()


# #############################################################################
# Cell 4.2: The Q-learning update rule
# #############################################################################


def cell4_2_q_update_rule(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show how a single experience tuple nudges one Q-value via the TD update.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (14, 5)
    env = GridWorld()
    alpha_slider, alpha_box = htutori.build_widget_control(
        name="alpha",
        description="learning rate",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.5,
        is_float=True,
    )
    gamma_slider, gamma_box = htutori.build_widget_control(
        name="gamma",
        description="discount factor",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.9,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.gamma = gamma_slider.value
            alpha = alpha_slider.value
            # A concrete illustrative experience tuple s -a-> s' with reward r.
            s, a, s2 = (3, 1), "Up", (3, 2)
            r = env.reward(s2)
            # Use converged utilities as a stand-in for current Q(s', .).
            snapshots, _ = value_iteration(env)
            u = snapshots[-1]
            q_old = 0.0
            # Use the converged utility of s' as the current best next value.
            best_next = u[s2]
            td_target = r + env.gamma * best_next
            td_error = td_target - q_old
            q_new = q_old + alpha * td_error
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            # Panel 1: the focused transition diagram.
            mtlrll0ut.draw_grid_base(env, ax1, highlight=s)
            ax1.annotate(
                "",
                xy=(s2[0], s2[1] - 0.1),
                xytext=(s[0], s[1] + 0.1),
                arrowprops=dict(
                    arrowstyle="-|>", color="darkblue", linewidth=3.0
                ),
            )
            ax1.text(
                (s[0] + s2[0]) / 2 + 0.25,
                (s[1] + s2[1]) / 2,
                "a=%s\nr=%.2f" % (a, r),
                ha="left",
                va="center",
                fontsize=10,
                color="darkblue",
            )
            ax1.set_title("One experience tuple", fontsize=13, fontweight="bold")
            ax1.set_xlabel(
                "col\n\n"
                "one transition $s \\rightarrow s'$ (action $a$) "
                "with reward $r$",
                fontsize=9,
            )
            # Panel 2: comments breaking the update into parts.
            text = (
                "Update:\n"
                "Q <- Q + alpha[r + gamma\n"
                "      max Q(s',.) - Q]\n\n"
                "Parameters:\n"
                "  alpha: %.2f\n"
                "  gamma: %.2f\n\n"
                "For tuple (%s, %s, %.2f, %s):\n"
                "  old Q: %.3f\n"
                "  TD target: %.3f\n"
                "  TD error: %.3f\n"
                "  new Q: %.3f"
                % (
                    alpha,
                    env.gamma,
                    s,
                    a,
                    r,
                    s2,
                    q_old,
                    td_target,
                    td_error,
                    q_new,
                )
            )
            mtlrll0ut.comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "alpha": "learning rate; near 0 the estimate barely moves, near 1 it jumps to the TD target",
            "gamma": "discount factor weighting the next-state value in the TD target",
        }
    )
    alpha_slider.observe(update_plot, names="value")
    gamma_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            alpha_box,
            gamma_box,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 4.3: Exploration vs exploitation with epsilon-greedy
# #############################################################################


def cell4_3_exploration(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Compare state coverage under low vs high exploration.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (16, 5)
    env = GridWorld()
    epsilon_slider, epsilon_box = htutori.build_widget_control(
        name="epsilon",
        description="exploration probability",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.1,
        is_float=True,
    )
    n_exp_slider, n_box = htutori.build_log_widget_control(
        name="log(n_episodes)",
        description="number of training episodes",
        min_exp=4,
        max_exp=11,
        initial_exp=7,
        base=2,
    )
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            n_episodes = 2**n_exp_slider.value
            epsilon = epsilon_slider.value
            seed = seed_slider.value
            # Run two agents: the chosen epsilon and a high-exploration agent.
            low = q_learning(
                env,
                n_episodes=n_episodes,
                alpha=0.5,
                epsilon=epsilon,
                seed=seed,
            )
            high = q_learning(
                env,
                n_episodes=n_episodes,
                alpha=0.5,
                epsilon=0.9,
                seed=seed,
            )
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            mtlrll0ut.draw_visit_heatmap(
                env,
                ax1,
                low["visit_counts"],
                title="Visits at epsilon=%.2f" % epsilon,
            )
            ax1.set_xlabel(
                "col\n\nvisit counts under the chosen exploration rate",
                fontsize=9,
            )
            mtlrll0ut.draw_visit_heatmap(
                env,
                ax2,
                high["visit_counts"],
                title="Visits at epsilon=0.90",
            )
            ax2.set_xlabel(
                "col\n\nvisit counts under broad exploration",
                fontsize=9,
            )
            text = (
                "Parameters:\n"
                "  epsilon: %.2f\n"
                "  n_episodes: %d\n"
                "  seed: %d" % (epsilon, n_episodes, seed)
            )
            mtlrll0ut.comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "epsilon": "exploration probability; low epsilon concentrates visits on a narrow corridor, high epsilon spreads visits broadly",
            "n_episodes": "number of training episodes (log scale)",
            "seed": "random seed for reproducible training runs",
        }
    )
    epsilon_slider.observe(update_plot, names="value")
    n_exp_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            epsilon_box,
            n_box,
            seed_box,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 4.4: Watching Q-learning learn the optimal policy
# #############################################################################


def cell4_4_q_learning_converges(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Train Q-learning and compare its policy to the value iteration optimum.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (18, 5)
    env = GridWorld()
    n_exp_slider, n_box = htutori.build_log_widget_control(
        name="log(n_episodes)",
        description="training episodes",
        min_exp=4,
        max_exp=12,
        initial_exp=9,
        base=2,
    )
    alpha_slider, alpha_box = htutori.build_widget_control(
        name="alpha",
        description="learning rate",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.5,
        is_float=True,
    )
    epsilon_slider, epsilon_box = htutori.build_widget_control(
        name="epsilon",
        description="exploration probability",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.2,
        is_float=True,
    )
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    output = ipywidgets.Output()
    # The planning-optimal policy for comparison (computed once).
    vi_snapshots, _ = value_iteration(env)
    optimal_policy = extract_policy(env, vi_snapshots[-1])

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            n_episodes = 2**n_exp_slider.value
            result = q_learning(
                env,
                n_episodes=n_episodes,
                alpha=alpha_slider.value,
                epsilon=epsilon_slider.value,
                seed=seed_slider.value,
            )
            learned = result["policy"]
            returns = result["returns"]
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: learned greedy policy.
            mtlrll0ut.draw_grid_base(env, ax1)
            mtlrll0ut.draw_policy_arrows(env, ax1, learned, color="darkblue")
            ax1.set_title("Q-learning policy", fontsize=13, fontweight="bold")
            ax1.set_xlabel(
                "col\n\nthe greedy policy derived from the learned Q-table",
                fontsize=9,
            )
            # Panel 2: learning curve (smoothed return per episode).
            window = max(1, len(returns) // 50)
            smooth = pd.Series(returns).rolling(window, min_periods=1).mean()
            ax2.plot(smooth, color="seagreen")
            ax2.set_xlabel(
                "episode\n\nsmoothed total return per episode",
                fontsize=9,
            )
            ax2.set_ylabel("return (smoothed)")
            ax2.set_title("Learning curve", fontsize=13, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            # Panel 3: comments with agreement vs the optimal policy.
            n_match = sum(
                1
                for s in env.nonterminal_states
                if learned[s] == optimal_policy[s]
            )
            n_total = len(env.nonterminal_states)
            text = (
                "Parameters:\n"
                "  n_episodes: %d\n"
                "  alpha: %.2f\n"
                "  epsilon: %.2f\n\n"
                "Policy match vs value\n"
                "iteration: %d / %d states"
                % (
                    n_episodes,
                    alpha_slider.value,
                    epsilon_slider.value,
                    n_match,
                    n_total,
                )
            )
            mtlrll0ut.comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "n_episodes": "number of training episodes (log scale); more episodes give the Q-table time to converge",
            "alpha": "learning rate controlling how much each experience updates the Q-value",
            "epsilon": "exploration probability for epsilon-greedy action selection",
            "seed": "random seed for reproducible training",
        }
    )
    n_exp_slider.observe(update_plot, names="value")
    alpha_slider.observe(update_plot, names="value")
    epsilon_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            n_box,
            alpha_box,
            epsilon_box,
            seed_box,
        ],
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))
