"""
Utility functions for the 4x3 grid world reinforcement learning lesson
implemented as a gymnasium environment (mirrors the from-scratch version).

Uses the standard FrozenLake convention: `P[s][a]` exposes the transition
model as `[(prob, s', r, terminated)]` for planning, while Q-learning uses
the `env.step()` API.

Import as:

import msml610.tutorials.L12_reinforcement_learning.L12_02_gridworld_4x3_gymnasium_utils as mtlrll0g4gu
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import ipywidgets
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output, display

import helpers.hprint as hprint
import helpers.htutorial as htutori

import msml610.tutorials.L12_reinforcement_learning.L12_01_utils as mtlrll0ut

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    mtlrll0ut.init_loggers(notebook_log)


# #############################################################################
# GridWorldEnv
# #############################################################################


class GridWorldEnv(gym.Env):
    """
    The canonical AIMA 4x3 grid world as a gymnasium environment.

    - States are integers 0-10 mapped to (col, row) tuples
    - Actions are 0=Up,1=Down,2=Left,3=Right
    - The intended action succeeds with probability `p_intended`; the agent
      slips perpendicular with probability (1-p_int)/2 each
    - Wall hits and boundary violations leave the agent in place.

    The transition model is exposed as `P[s][a] -> list of (prob, s', reward,
    terminated)` for use by planning algorithms.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def _attempt_move(
        self, cell: Tuple[int, int], direction: str
    ) -> Tuple[int, int]:
        """
        Try moving one step; bounce back on walls and boundaries.
        """
        d_col, d_row = mtlrll0ut.ACTION_DELTA[direction]
        target = (cell[0] + d_col, cell[1] + d_row)
        off_grid = not (
            1 <= target[0] <= self.n_cols and 1 <= target[1] <= self.n_rows
        )
        if off_grid or target in self.wall_cells:
            return cell
        _LOG.debug("_attempt_move: %s -> %s", cell, target)
        return target

    def _compute_transitions(
        self, cell: Tuple[int, int], a_id: int
    ) -> List[Tuple[float, int, float, bool]]:
        """
        Compute `[(prob, s', reward, terminated)]` for a state-action pair.

        Terminal states are absorbing (stay in place with zero reward).
        """
        # Note: `hprint.func_signature_to_str()` looks up the caller by name
        # in the module globals, so it cannot be called from inside a method.
        # Terminal states are absorbing: stay in place with zero reward.
        if cell in self.terminal_cells:
            s_id = self._cell_to_id[cell]
            return [(1.0, s_id, 0.0, True)]
        # Decompose the intended action into a distribution over actual moves,
        # with probability p_intended for the chosen direction and equal
        # probability for each perpendicular slip.
        action = mtlrll0ut.ACTION_ID_TO_NAME[a_id]
        p_perp = (1.0 - self.p_intended) / 2.0
        outcome_probs = [(action, self.p_intended)]
        for perp in mtlrll0ut.PERPENDICULAR[action]:
            outcome_probs.append((perp, p_perp))
        # Aggregate by destination cell (blocked/wall moves that bounce back
        # to the same cell merge their probabilities into one entry).
        dist: Dict[Tuple[int, int], float] = {}
        for direction, prob in outcome_probs:
            s2_cell = self._attempt_move(cell, direction)
            dist[s2_cell] = dist.get(s2_cell, 0.0) + prob
        # Convert the aggregated distribution into the flat outcome list
        # expected by the FrozenLake P[s][a] convention.
        result: List[Tuple[float, int, float, bool]] = []
        for s2_cell, prob in dist.items():
            s2_id = self._cell_to_id[s2_cell]
            reward = (
                self.terminal_cells[s2_cell]
                if s2_cell in self.terminal_cells
                else self.r_step
            )
            terminated = s2_cell in self.terminal_cells
            result.append((prob, s2_id, reward, terminated))
        return result

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        r_step: float = -0.04,
        gamma: float = 1.0,
        p_intended: float = 0.8,
    ) -> None:
        # Note: `hprint.func_signature_to_str()` looks up the caller by name
        # in the module globals, so it cannot be called from inside a method.
        super().__init__()
        self.n_cols = 4
        self.n_rows = 3
        self.r_step = r_step
        self.gamma = gamma
        self.p_intended = p_intended
        self.render_mode = render_mode

        # Build state <-> coordinate mappings.
        self.start_cell = (1, 1)
        self.terminal_cells = {(4, 3): 1.0, (4, 2): -1.0}
        self.wall_cells = {(2, 2)}
        self.all_cells = [
            (c, r)
            for r in range(1, self.n_rows + 1)
            for c in range(1, self.n_cols + 1)
            if (c, r) not in self.wall_cells
        ]
        self._cell_to_id = {cell: i for i, cell in enumerate(self.all_cells)}
        self._id_to_cell = {i: cell for cell, i in self._cell_to_id.items()}
        self._terminal_ids = {self._cell_to_id[c] for c in self.terminal_cells}

        # Action and observation spaces.
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(len(self.all_cells))

        # Pre-compute the transition model P[s][a].
        self.P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}
        for s_id in range(len(self.all_cells)):
            cell = self._id_to_cell[s_id]
            self.P[s_id] = {}
            for a_id in range(4):
                self.P[s_id][a_id] = self._compute_transitions(cell, a_id)

        # Internal state.
        self._state: int = self._cell_to_id[self.start_cell]

    # --- Core gymnasium API ------------------------------------------------

    def _get_obs(self) -> int:
        return self._state

    def _get_info(self) -> Dict[str, Any]:
        cell = self._id_to_cell[self._state]
        return {"cell": cell}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Apply action, sample the next state, and return the transition.
        """
        _LOG.debug(hprint.to_str("action"))
        # Sample from the pre-computed transition distribution.
        outcomes = self.P[self._state][action]
        probs = [o[0] for o in outcomes]
        idx = self.np_random.choice(len(outcomes), p=probs)
        _, s2_id, reward, terminated = outcomes[idx]
        self._state = s2_id
        terminated_bool = bool(terminated)
        truncated_bool = False
        _LOG.debug(hprint.to_str("s2_id reward terminated_bool"))
        return s2_id, reward, terminated_bool, truncated_bool, self._get_info()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        _LOG.debug(hprint.to_str("seed"))
        super().reset(seed=seed)
        self._state = self._cell_to_id[self.start_cell]
        _LOG.debug("return=%s", self._state)
        return self._get_obs(), self._get_info()

    # --- Drawing helpers ---------------------------------------------------

    def _draw_base(
        self,
        ax: plt.Axes,
        *,
        highlight: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Draw the grid layout with coloured special cells.

        Delegates to the shared :func:`l12_utils.draw_grid_base` in ``L12_01_utils``.
        """
        _LOG.debug(hprint.to_str("highlight"))
        mtlrll0ut.draw_grid_base(self, ax, highlight=highlight)

    def render(self) -> None:
        """
        Draw the grid.

        In `human` mode, show via plt.show().
        """
        _LOG.debug(hprint.to_str("self.render_mode"))
        _, ax = plt.subplots(figsize=(6, 4))
        self._draw_base(ax, highlight=self._id_to_cell[self._state])
        ax.set_title("GridWorldEnv", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if self.render_mode == "human":
            plt.show()

    # --- Transition model (FrozenLake-style P[s][a] dict) ------------------

    def _cell_to_id_safe(self, cell: Tuple[int, int]) -> Optional[int]:
        """
        Return state id, or None if `cell` is a wall.
        """
        return self._cell_to_id.get(cell)

    # --- Drawing helpers exposed for the interactive cells -----------------

    def id_to_cell(self, state_id: int) -> Tuple[int, int]:
        return self._id_to_cell[state_id]

    def cell_to_id(self, cell: Tuple[int, int]) -> int:
        return self._cell_to_id[cell]

    def nonterminal_ids(self) -> List[int]:
        return [
            i for i in range(len(self.all_cells)) if i not in self._terminal_ids
        ]

    def is_terminal_id(self, state_id: int) -> bool:
        return state_id in self._terminal_ids

    def to_grid(
        self,
        values: Dict[Tuple[int, int], float],
        *,
        fill: float = np.nan,
    ) -> np.ndarray:
        """Convert a dict over (col, row) cells into a 2D array for heatmap
        plots.
        """
        return mtlrll0ut.to_grid(values, self.n_rows, self.n_cols, fill=fill)


# #############################################################################
# Exact planning helpers (use the gymnasium env's P[s][a] transition model)
# #############################################################################


def _q_value(
    env: GridWorldEnv,
    s_id: int,
    a_id: int,
    u: Dict[int, float],
) -> float:
    """
    Compute expected one-step value of action `a_id` in state `s_id`.

    Uses the convention: Q(s,a) = sum_{outcomes} prob * (reward + gamma * u[s']).
    """
    _LOG.debug(hprint.to_str("s_id a_id"))
    q = 0.0
    for prob, s2_id, reward, _ in env.P[s_id][a_id]:
        q += prob * (reward + env.gamma * u[s2_id])
    _LOG.debug("return=%s", q)
    return q


def value_iteration(
    env: GridWorldEnv,
    *,
    max_sweeps: int = 100,
    tol: float = 1e-6,
) -> Tuple[List[Dict[int, float]], List[float]]:
    """
    Run value iteration using the gymnasium env's P[s][a] model.

    :return: (snapshots, deltas) where snapshots[i] are utilities after sweep i
        and deltas[i] is the max per-state change during sweep i.
    """
    _LOG.debug(hprint.func_signature_to_str())
    n_states = env.observation_space.n
    u: Dict[int, float] = {i: 0.0 for i in range(n_states)}
    # Initialize snapshots with the zero-utility starting point.
    snapshots = [dict(u)]
    deltas: List[float] = []
    nonterm = env.nonterminal_ids()
    # Perform Bellman sweeps: update each non-terminal state's utility to
    # the max over actions of the expected discounted return.
    for _ in range(max_sweeps):
        delta = 0.0
        u_new = dict(u)
        for s_id in nonterm:
            best = max(_q_value(env, s_id, a, u) for a in range(4))
            delta = max(delta, abs(best - u[s_id]))
            u_new[s_id] = best
        u = u_new
        snapshots.append(dict(u))
        deltas.append(delta)
        if delta < tol:
            break
    _LOG.debug(hprint.to_str("len(snapshots) deltas[-1]"))
    return snapshots, deltas


def extract_policy(
    env: GridWorldEnv,
    u: Dict[int, float],
) -> Dict[Tuple[int, int], str]:
    """
    Extract the greedy policy (as cell->action dict) from utilities `u`.
    """
    _LOG.debug(hprint.func_signature_to_str())
    policy: Dict[Tuple[int, int], str] = {}
    for c in env.all_cells:
        if c in env.terminal_cells:
            continue
        s_id = env.cell_to_id(c)
        best_a = max(range(4), key=lambda a: _q_value(env, s_id, a, u))
        policy[c] = mtlrll0ut.ACTION_ID_TO_NAME[best_a]
    _LOG.debug(hprint.to_str("len(policy)"))
    return policy


def policy_evaluation(
    env: GridWorldEnv,
    policy: Dict[Tuple[int, int], str],
) -> Dict[int, float]:
    """
    Exactly evaluate a fixed policy by solving the linear Bellman system.

    With a fixed action per state the max disappears and the Bellman equations
    are linear; solved in one shot with numpy.linalg.solve.
    """
    _LOG.debug(hprint.func_signature_to_str())
    n_states = env.observation_space.n
    nonterm_ids = env.nonterminal_ids()
    # Build the linear system (I - gamma * P^pi) * u = r^pi.
    # Only non-terminal states are solved; terminal states have fixed utilities.
    idx_map = {s_id: i for i, s_id in enumerate(nonterm_ids)}
    n = len(nonterm_ids)
    a_mat = np.eye(n)
    b_vec = np.zeros(n)
    # Map each cell to the fixed action dictated by the policy.
    policy_action_ids: Dict[int, int] = {}
    for cell, a_name in policy.items():
        if cell in env.terminal_cells or cell in env.wall_cells:
            continue
        policy_action_ids[env.cell_to_id(cell)] = mtlrll0ut.NAME_TO_ACTION_ID[
            a_name
        ]
    # Fill the matrix and vector using the transition model P[s][policy(s)].
    for s_id in nonterm_ids:
        i = idx_map[s_id]
        a_id = policy_action_ids[s_id]
        for prob, s2_id, reward, _ in env.P[s_id][a_id]:
            b_vec[i] += prob * reward
            if s2_id in idx_map:
                a_mat[i, idx_map[s2_id]] -= env.gamma * prob
    solution = np.linalg.solve(a_mat, b_vec)
    u = {i: 0.0 for i in range(n_states)}
    for s_id in nonterm_ids:
        u[s_id] = float(solution[idx_map[s_id]])
    _LOG.debug("return: n_states=%d", len(u))
    return u


def policy_iteration(
    env: GridWorldEnv,
    *,
    initial_action: int = 1,  # default "Down" for poor initial policy
    max_iters: int = 50,
) -> Tuple[List[Dict[Tuple[int, int], str]], List[int]]:
    """
    Run policy iteration, recording per-round snapshots.
    """
    _LOG.debug(hprint.to_str("initial_action max_iters"))
    # Start with a uniform policy (all states take the same default action).
    policy: Dict[Tuple[int, int], str] = {}
    for c in env.all_cells:
        if c not in env.terminal_cells:
            policy[c] = mtlrll0ut.ACTION_ID_TO_NAME[initial_action]
    policies = [dict(policy)]
    changes: List[int] = []
    # Repeat evaluate-then-improve until the policy stabilizes.
    for _ in range(max_iters):
        u = policy_evaluation(env, policy)
        new_policy = extract_policy(env, u)
        n_changed = sum(
            1
            for c in env.all_cells
            if c not in env.terminal_cells and new_policy[c] != policy[c]
        )
        policy = new_policy
        policies.append(dict(policy))
        changes.append(n_changed)
        if n_changed == 0:
            break
    _LOG.debug(hprint.to_str("len(policies) len(changes)"))
    return policies, changes


# #############################################################################
# Q-learning (uses env.step() — no access to P[s][a])
# #############################################################################


def _greedy_action_id(
    q: Dict[int, float],
    s_id: int,
) -> int:
    """
    Return the action with the highest Q-value at state `s_id`.

    Query by flattened key `(s_id * 4 + a_id)`.
    """
    _LOG.debug(hprint.to_str("s_id"))
    best_a = max(range(4), key=lambda a: q[s_id * 4 + a])
    _LOG.debug("return=%s", best_a)
    return best_a


def q_learning(
    env: GridWorldEnv,
    *,
    n_episodes: int,
    alpha: float,
    epsilon: float,
    seed: int,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """
    Learn the optimal policy from experience using tabular Q-learning.

    The agent sees only (s, a, r, s') tuples via env.step() — it never reads
    env.P (the transition model).

    :return: dict with Q-table (flat key), per-episode returns, visit counts, the
        final greedy policy as a cell->action dict, per-episode greedy policies.
    """
    _LOG.debug(hprint.to_str("n_episodes alpha epsilon seed"))
    n_states = env.observation_space.n
    n_actions = 4
    rng = np.random.RandomState(seed)
    q: Dict[int, float] = {}
    for s_id in range(n_states):
        for a_id in range(n_actions):
            q[s_id * n_actions + a_id] = 0.0
    visit_counts: Dict[int, int] = {s_id: 0 for s_id in range(n_states)}
    returns: List[float] = []
    policies: List[Dict[Tuple[int, int], str]] = []
    # Each episode rolls out a trajectory by interacting with the environment
    # through env.step(), performing on-policy TD updates at every step.
    for episode in range(n_episodes):
        s_id, _ = env.reset(seed=seed + episode if seed else None)
        total_reward = 0.0
        for _ in range(max_steps):
            visit_counts[s_id] += 1
            # Epsilon-greedy: explore randomly with prob epsilon, otherwise greedy.
            if rng.rand() < epsilon:
                a_id = rng.randint(n_actions)
            else:
                a_id = _greedy_action_id(q, s_id)
            s2_id, reward, terminated, truncated, _ = env.step(a_id)
            total_reward += reward
            # Compute the TD target: for terminal states it's just the reward;
            # for non-terminal it's reward + gamma * max Q(s', a').
            if terminated or truncated:
                td_target = reward
            else:
                best_next = max(
                    q[s2_id * n_actions + a2] for a2 in range(n_actions)
                )
                td_target = reward + env.gamma * best_next
            # Incremental Q-update: Q(s,a) += alpha * (TD_target - Q(s,a)).
            q[s_id * n_actions + a_id] += alpha * (
                td_target - q[s_id * n_actions + a_id]
            )
            s_id = s2_id
            if terminated or truncated:
                break
        returns.append(total_reward)
        # Record the greedy policy after this episode.
        policy: Dict[Tuple[int, int], str] = {}
        for c in env.all_cells:
            if c not in env.terminal_cells:
                sid = env.cell_to_id(c)
                ba = _greedy_action_id(q, sid)
                policy[c] = mtlrll0ut.ACTION_ID_TO_NAME[ba]
        policies.append(policy)
    final_policy = policies[-1] if policies else {}
    result = {
        "q": q,
        "returns": returns,
        "visit_counts": visit_counts,
        "policy": final_policy,
        "policies": policies,
    }
    _LOG.debug(hprint.to_str("len(q) len(returns)"))
    return result


def _sample_trajectory(
    env: GridWorldEnv,
    policy: Dict[Tuple[int, int], str],
    *,
    seed: int,
    max_steps: int = 30,
) -> List[Tuple[int, int]]:
    """
    Roll out a trajectory using env.step() for realism.
    """
    _LOG.debug(hprint.to_str("seed max_steps"))
    s_id, _ = env.reset(seed=seed)
    path = [env.id_to_cell(s_id)]
    for _ in range(max_steps):
        current_cell = env.id_to_cell(s_id)
        if current_cell in env.terminal_cells:
            break
        a_id = mtlrll0ut.NAME_TO_ACTION_ID[policy[current_cell]]
        s2_id, _, terminated, truncated, _ = env.step(a_id)
        s_id = s2_id
        path.append(env.id_to_cell(s_id))
        if terminated or truncated:
            break
    _LOG.debug("return: len(path)=%d", len(path))
    return path


# #############################################################################
# Cell 1.1: The 4x3 grid and its states
# #############################################################################


def cell1_1_show_grid(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Draw the grid and show the gymnasium observation/action spaces.
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (7, 5)
    env = GridWorldEnv()
    _, ax = plt.subplots(figsize=figsize)
    env._draw_base(ax)
    ax.set_title(
        "The 4x3 grid world (gymnasium environment)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("n_states=%d" % env.observation_space.n)
    print("state id -> cell mapping:")
    for i in range(env.observation_space.n):
        print("  %d -> %s" % (i, env.id_to_cell(i)))
    print("terminal ids:", sorted(env._terminal_ids))


# #############################################################################
# Cell 1.2: Stochastic action model
# #############################################################################


def cell1_2_stochastic_action(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show how an intended action spreads probability mass through env.P.

    Interactive controls (ipywidgets):
    :param action: the intended action (``Up``, ``Down``, ``Left``, ``Right``)
    :param p_intended: probability the intended action succeeds (0.5 to 1.0)
    :param figsize: optional figure size
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (14, 5)
    env = GridWorldEnv()
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
    focal = (3, 2)

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.p_intended = p_slider.value
            action = action_dropdown.value
            a_id = mtlrll0ut.NAME_TO_ACTION_ID[action]
            focal_id = env.cell_to_id(focal)
            outcomes = env.P[focal_id][a_id]
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            env._draw_base(ax1, highlight=focal)
            # Annotate each outcome with its probability.
            for prob, s2_id, reward, _ in outcomes:
                s2_cell = env.id_to_cell(s2_id)
                if s2_cell == focal:
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
                    d_x = s2_cell[0] - focal[0]
                    d_y = s2_cell[1] - focal[1]
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
                "Intended '%s' from %s (gymnasium)" % (action, focal),
                fontsize=13,
                fontweight="bold",
            )
            ax1.set_xlabel(
                "col\n\n"
                "_Action spread_: the intended action and perpendicular "
                "slips with arrow thickness encoding probability",
                fontsize=9,
            )
            p_perp = (1.0 - env.p_intended) / 2.0
            text = (
                "Parameters:\n"
                "  action: %s\n"
                "  p_intended: %.2f\n\n"
                "Outcome probabilities:\n"
                "  intended: %.2f\n"
                "  each perpendicular: %.2f\n\n"
                "Source: env.P[%d][%d]"
                % (
                    action,
                    env.p_intended,
                    env.p_intended,
                    p_perp,
                    focal_id,
                    a_id,
                )
            )
            mtlrll0ut.comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = mtlrll0ut.make_param_info(
        {
            "action": "intended direction (<code>Up</code>, <code>Down</code>, <code>Left</code>, <code>Right</code>)",
            "p_intended": "probability the intended action succeeds (0.50 to 1.00); the remaining 1&minus;p is split evenly across the two perpendicular moves",
        }
    )
    action_dropdown.observe(update_plot, names="value")
    p_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            action_dropdown,
            p_box,
        ]
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 1.3: Transition model from env.P
# #############################################################################


def cell1_3_transition_table(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """Display the explicit transition row from env.P for a chosen (state,
    action).
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (7, 5)
    env = GridWorldEnv()
    state_dropdown = ipywidgets.Dropdown(
        options=[str(c) for c in env.all_cells if c not in env.terminal_cells],
        value=str(env.start_cell),
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
            cell = eval(state_dropdown.value)
            a = action_dropdown.value
            s_id = env.cell_to_id(cell)
            a_id = mtlrll0ut.NAME_TO_ACTION_ID[a]
            outcomes = env.P[s_id][a_id]
            # DataFrame.
            df = pd.DataFrame(
                [
                    {
                        "next_state": str(env.id_to_cell(s2)),
                        "prob": round(p, 3),
                        "reward": round(r, 3),
                        "terminal": t,
                    }
                    for p, s2, r, t in outcomes
                ]
            ).sort_values("prob", ascending=False)
            # Grid overlay.
            prob_map = {
                env.id_to_cell(i): 0.0 for i in range(env.observation_space.n)
            }
            for p, s2, _, _ in outcomes:
                prob_map[env.id_to_cell(s2)] = p
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
            ax.add_patch(
                mpatches.Rectangle(
                    (cell[0] - 1, env.n_rows - cell[1]),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="darkorange",
                    linewidth=3.5,
                )
            )
            ax.set_title(
                "Pr(s' | s=%s, a=%s) from env.P" % (cell, a),
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xticklabels([str(c) for c in range(1, env.n_cols + 1)])
            ax.set_yticklabels(
                [str(r) for r in range(env.n_rows, 0, -1)], rotation=0
            )
            ax.set_xlabel(
                "col\n\n"
                "_Transition table_: each row shows the probability distribution "
                "over next states for a given state and action",
                fontsize=9,
            )
            ax.set_ylabel("row")
            plt.tight_layout()
            plt.show()
            display(df.reset_index(drop=True))
            prob_sum = round(sum(outcome[0] for outcome in outcomes), 6)
            print("sum of probabilities=", prob_sum)

    param_info = mtlrll0ut.make_param_info(
        {
            "state": "which non-terminal cell to query env.P[s][a] for",
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
        ]
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 1.4: Rewards and episode returns
# #############################################################################


def cell1_4_rewards_and_returns(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show per-cell rewards and the discounted return of a sample trajectory.
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (14, 5)
    env = GridWorldEnv()
    # Fixed policy for the sample trajectory.
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
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="trajectory seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=1,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.r_step = r_slider.value
            env.gamma = gamma_slider.value
            env.p_intended = 1.0  # deterministic for a clean illustration
            # Recompute transitions to update rewards.
            # Changes to r_step or gamma require rebuilding env.P since
            # the reward values are baked into each transition outcome.
            for s_id in range(len(env.all_cells)):
                cell = env.id_to_cell(s_id)
                for a_id in range(4):
                    env.P[s_id][a_id] = env._compute_transitions(cell, a_id)
            path = _sample_trajectory(env, base_policy, seed=seed_slider.value)
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            env._draw_base(ax1)
            for col in range(1, env.n_cols + 1):
                for row in range(1, env.n_rows + 1):
                    cell = (col, row)
                    if cell in env.wall_cells:
                        continue
                    ax1.text(
                        col,
                        row - 0.05,
                        "%.2f"
                        % (
                            env.terminal_cells[cell]
                            if cell in env.terminal_cells
                            else env.r_step
                        ),
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
            xs = [c[0] for c in path]
            ys = [c[1] for c in path]
            ax1.plot(xs, ys, "-o", color="darkblue", linewidth=2.0, markersize=6)
            ax1.set_title(
                "Rewards and trajectory (gymnasium)",
                fontsize=13,
                fontweight="bold",
            )
            ax1.set_xlabel(
                "col\n\n"
                "_Rewards and trajectory_: per-cell rewards with a sample "
                "path from START",
                fontsize=9,
            )
            running = 0.0
            lines = []
            for t in range(1, len(path)):
                cell = path[t]
                r_t = (
                    env.terminal_cells[cell]
                    if cell in env.terminal_cells
                    else env.r_step
                )
                running += (env.gamma**t) * r_t
                if t <= 8:
                    lines.append("  t=%d enter %s r=%.2f" % (t, path[t], r_t))
            text = (
                "Parameters:\n"
                "  r_step: %.2f\n"
                "  gamma: %.2f\n\n"
                "First steps:\n%s\n\n"
                "Key idea:\n- Same return computation\n"
                "  as the from-scratch env.\n"
                "- Rewards come from env's\n"
                "  reward model."
                "Discounted return G = %.3f\n\n"
                % (env.r_step, env.gamma, "\n".join(lines), running)
            )
            mtlrll0ut.comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    r_slider.observe(update_plot, names="value")
    gamma_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    param_info = mtlrll0ut.make_param_info(
        {
            "r_step": "living reward collected for each non-terminal step (more negative = stronger penalty for wandering)",
            "gamma": "discount factor (near 1 values distant rewards; near 0 cares only about immediate reward)",
            "seed": "random seed controlling the trajectory path taken",
        }
    )
    controls = ipywidgets.VBox(
        [
            r_box,
            gamma_box,
            seed_box,
        ]
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
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (16, 5)
    env = GridWorldEnv()
    state_dropdown = ipywidgets.Dropdown(
        options=[str(c) for c in env.all_cells if c not in env.terminal_cells],
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
            cell = eval(state_dropdown.value)
            s_id = env.cell_to_id(cell)
            snapshots, _ = value_iteration(env)
            u = snapshots[-1]
            q_vals = {
                a: _q_value(env, s_id, mtlrll0ut.NAME_TO_ACTION_ID[a], u)
                for a in mtlrll0ut.ACTIONS
            }
            best_action = max(q_vals, key=lambda a: q_vals[a])
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            env._draw_base(ax1, highlight=cell)
            ax1.set_title(
                "Inspected state %s" % (cell,), fontsize=13, fontweight="bold"
            )
            ax1.set_xlabel(
                "col\n\n_Inspected state_: the highlighted cell on the grid",
                fontsize=9,
            )
            colors = [
                "seagreen" if a == best_action else "steelblue"
                for a in mtlrll0ut.ACTIONS
            ]
            ax2.bar(
                mtlrll0ut.ACTIONS,
                [q_vals[a] for a in mtlrll0ut.ACTIONS],
                color=colors,
            )
            ax2.axhline(0.0, color="black", linewidth=0.8)
            ax2.set_ylabel("expected action value Q(s,a)")
            ax2.set_title(
                "Action values (max kept)", fontsize=13, fontweight="bold"
            )
            ax2.set_xlabel(
                "action\n\n"
                "_Action values_: a bar per action, the best (max) highlighted",
                fontsize=9,
            )
            ax2.grid(True, alpha=0.3)
            text = (
                "State: %s\n"
                "gamma: %.2f\n\n"
                "Action values:\n%s\n\n"
                "Best action: %s\n"
                "U(s) = %.3f\n\n"
                "Source: uses env.P[s][a]"
                % (
                    cell,
                    env.gamma,
                    "\n".join(
                        "  %-6s %.3f" % (a, q_vals[a]) for a in mtlrll0ut.ACTIONS
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
        ]
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.2: Value iteration converging over sweeps
# #############################################################################


def cell2_2_value_iteration(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Step through value iteration sweeps and watch utilities converge.

    Uses the gymnasium env's `value_iteration()` planner which reads env.P.
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (16, 5)
    env = GridWorldEnv()
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
            # Recompute transitions so rewards propagate through env.P.
            # Any change to r_step or gamma requires rebuilding the
            # transition model since rewards are baked into each outcome.
            for s_id in range(len(env.all_cells)):
                cell = env.id_to_cell(s_id)
                for a_id in range(4):
                    env.P[s_id][a_id] = env._compute_transitions(cell, a_id)
            snapshots, deltas = value_iteration(env, max_sweeps=30)
            i = min(iter_slider.value, len(snapshots) - 1)
            u_ids = snapshots[i]
            # Convert id-keyed u to cell-keyed for heatmap.
            u_cells = {env.id_to_cell(s_id): val for s_id, val in u_ids.items()}
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            mtlrll0ut.draw_value_heatmap(
                env, ax1, u_cells, title="Utilities after sweep %d" % i
            )
            ax1.set_xlabel(
                "col\n\n"
                "_Utility heatmap_: each cell annotated with its current U(s)",
                fontsize=9,
            )
            ax2.plot(
                range(1, len(deltas) + 1),
                deltas,
                "-o",
                color="darkorange",
            )
            if 1 <= i <= len(deltas):
                ax2.axvline(i, color="gray", linestyle="--")
            ax2.set_xlabel(
                "sweep\n\n_Convergence_: max utility change per sweep",
                fontsize=9,
            )
            ax2.set_ylabel("max |U_{i+1} - U_i|")
            ax2.set_title("Convergence", fontsize=13, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            start_id = env.cell_to_id(env.start_cell)
            text = (
                "Parameters:\n"
                "  sweep: %d / %d\n"
                "  gamma: %.2f\n"
                "  r_step: %.2f\n\n"
                "U(start) = %.3f\n"
                "sweeps to converge: %d\n\n"
                "Source: env.P[s][a] model."
                % (
                    i,
                    len(snapshots) - 1,
                    env.gamma,
                    env.r_step,
                    u_ids[start_id],
                    len(deltas),
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
        ]
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.3: Extracting the optimal policy
# #############################################################################


def cell2_3_extract_policy(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show the greedy policy extracted from converged utilities.
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (14, 5)
    env = GridWorldEnv()
    r_slider, r_box = htutori.build_widget_control(
        name="r_step",
        description="living reward",
        min_val=-2.0,
        max_val=0.0,
        step=0.05,
        initial_value=-0.04,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            env.r_step = r_slider.value
            for s_id in range(len(env.all_cells)):
                cell = env.id_to_cell(s_id)
                for a_id in range(4):
                    env.P[s_id][a_id] = env._compute_transitions(cell, a_id)
            snapshots, _ = value_iteration(env)
            u_ids = snapshots[-1]
            u_cells = {env.id_to_cell(s_id): val for s_id, val in u_ids.items()}
            policy = extract_policy(env, u_ids)
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            mtlrll0ut.draw_value_heatmap(
                env, ax1, u_cells, title="Optimal policy over utilities"
            )
            mtlrll0ut.draw_policy_arrows_heatmap(env, ax1, policy)
            ax1.set_xlabel(
                "col\n\n"
                "_Policy over utilities_: an arrow per cell over the utility "
                "heatmap",
                fontsize=9,
            )
            start_id = env.cell_to_id(env.start_cell)
            text = (
                "Parameters:\n"
                "  r_step: %.2f\n\n"
                "U(start) = %.3f\n\n"
                "Key idea:\n"
                "- Same optimal policy as\n"
                "  the from-scratch env.\n"
                "- Policy is greedy with\n"
                "  respect to env.P-based\n"
                "  utility estimates." % (env.r_step, u_ids[start_id])
            )
            mtlrll0ut.comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    r_slider.observe(update_plot, names="value")
    update_plot()
    param_info = mtlrll0ut.make_param_info(
        {
            "r_step": "living reward per step; a large negative value makes the agent take the short risky path, a near-zero value makes it take the long safe path",
        }
    )
    controls = ipywidgets.VBox(
        [
            r_box,
        ]
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 3.1: Policy evaluation for a fixed policy
# #############################################################################


def _preset_policy(env: GridWorldEnv, name: str) -> Dict[Tuple[int, int], str]:
    """
    Return a named preset policy.
    """
    _LOG.debug(hprint.to_str("name"))
    if name == "always-up":
        return {s: "Up" for s in env.all_cells if s not in env.terminal_cells}
    if name == "always-right":
        return {s: "Right" for s in env.all_cells if s not in env.terminal_cells}
    if name == "random":
        rng = np.random.RandomState(0)
        return {
            s: mtlrll0ut.ACTIONS[rng.randint(4)]
            for s in env.all_cells
            if s not in env.terminal_cells
        }
    policy = {
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
    _LOG.debug("return: %s actions", len(policy))
    return policy


def cell3_1_policy_evaluation(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Evaluate a fixed policy by solving the linear Bellman system.
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (14, 5)
    env = GridWorldEnv()
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
            for s_id in range(len(env.all_cells)):
                cell = env.id_to_cell(s_id)
                for a_id in range(4):
                    env.P[s_id][a_id] = env._compute_transitions(cell, a_id)
            policy = _preset_policy(env, policy_dropdown.value)
            u_ids = policy_evaluation(env, policy)
            u_cells = {env.id_to_cell(s_id): val for s_id, val in u_ids.items()}
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            mtlrll0ut.draw_value_heatmap(
                env, ax1, u_cells, title="U^pi for '%s'" % policy_dropdown.value
            )
            mtlrll0ut.draw_policy_arrows_heatmap(env, ax1, policy)
            ax1.set_xlabel(
                "col\n\n"
                "_Policy utilities_: the fixed policy as arrows over its "
                "evaluated utilities",
                fontsize=9,
            )
            start_id = env.cell_to_id(env.start_cell)
            text = (
                "Parameters:\n"
                "  policy: %s\n"
                "  gamma: %.2f\n\n"
                "U(start) = %.3f\n\n"
                "Source: policy_evaluation()\n"
                "reads env.P."
                % (policy_dropdown.value, env.gamma, u_ids[start_id])
            )
            mtlrll0ut.comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    policy_dropdown.observe(update_plot, names="value")
    gamma_slider.observe(update_plot, names="value")
    update_plot()
    param_info = mtlrll0ut.make_param_info(
        {
            "policy": "which fixed policy to evaluate (<code>hand-tuned</code>, <code>always-up</code>, <code>always-right</code>, <code>random</code>)",
            "gamma": "discount factor weighting future rewards",
        }
    )
    controls = ipywidgets.VBox(
        [
            policy_dropdown,
            gamma_box,
        ]
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
    Step through policy iteration rounds, watch arrows flip.
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (16, 5)
    env = GridWorldEnv()
    policies, changes = policy_iteration(env, initial_action=1)
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
            env._draw_base(ax1)
            mtlrll0ut.draw_policy_arrows(env, ax1, before, color="gray")
            ax1.set_title(
                "Policy before round %d" % i, fontsize=13, fontweight="bold"
            )
            ax1.set_xlabel(
                "col\n\n_Before_: the current policy arrows",
                fontsize=9,
            )
            env._draw_base(ax2)
            mtlrll0ut.draw_policy_arrows(env, ax2, after, color="darkblue")
            ax2.set_title(
                "Policy after round %d" % i, fontsize=13, fontweight="bold"
            )
            ax2.set_xlabel(
                "col\n\n_After_: the improved policy arrows",
                fontsize=9,
            )
            n_changed = changes[i - 1] if 1 <= i <= len(changes) else 0
            text = (
                "Round: %d / %d\n\n"
                "States changed: %d\n"
                "rounds to converge: %d\n\n"
                "Source: uses env.P for\n"
                "evaluation + improvement."
                % (i, len(policies) - 1, n_changed, len(changes))
            )
            mtlrll0ut.comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    iter_slider.observe(update_plot, names="value")
    update_plot()
    param_info = mtlrll0ut.make_param_info(
        {
            "iteration": "which evaluate/improve round to display (the policy converges in very few rounds)",
        }
    )
    controls = ipywidgets.VBox(
        [
            iter_box,
        ]
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
    Compare convergence of the two exact methods.
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (16, 5)
    env = GridWorldEnv()
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
            for s_id in range(len(env.all_cells)):
                cell = env.id_to_cell(s_id)
                for a_id in range(4):
                    env.P[s_id][a_id] = env._compute_transitions(cell, a_id)
            _, vi_deltas = value_iteration(env, max_sweeps=200)
            _, pi_changes = policy_iteration(env, initial_action=1)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            ax1.plot(
                range(1, len(vi_deltas) + 1), vi_deltas, "-o", color="darkorange"
            )
            ax1.set_xlabel(
                "sweep\n\n_Value iteration_: utility change per sweep",
                fontsize=9,
            )
            ax1.set_ylabel("max utility change")
            ax1.set_title("Value iteration", fontsize=13, fontweight="bold")
            ax1.grid(True, alpha=0.3)
            ax2.plot(
                range(1, len(pi_changes) + 1), pi_changes, "-s", color="seagreen"
            )
            ax2.set_xlabel(
                "round\n\n_Policy iteration_: changed-action count per round",
                fontsize=9,
            )
            ax2.set_ylabel("states changed action")
            ax2.set_title("Policy iteration", fontsize=13, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            text = (
                "gamma: %.2f\n\n"
                "VI sweeps: %d\n"
                "PI rounds: %d\n\n"
                "Source: both use env.P.\n"
                "Both converge to the same\n"
                "optimal policy." % (env.gamma, len(vi_deltas), len(pi_changes))
            )
            mtlrll0ut.comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    gamma_slider.observe(update_plot, names="value")
    update_plot()
    param_info = mtlrll0ut.make_param_info(
        {
            "gamma": "discount factor; as gamma approaches 1, value iteration needs many more sweeps while policy iteration stays at a handful of rounds",
        }
    )
    controls = ipywidgets.VBox(
        [
            gamma_box,
        ]
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 4.1: Why RL is harder than planning
# #############################################################################


def cell4_1_planning_vs_learning(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Contrast planning (reads env.P) with learning (calls env.step()).
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (14, 5)
    env = GridWorldEnv()
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    env._draw_base(ax1)
    ax1.text(
        2.5,
        2.0,
        "env.P\nUNKNOWN\nto the agent",
        ha="center",
        va="center",
        fontsize=14,
        color="firebrick",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    ax1.set_title("Same world, blindfolded", fontsize=13, fontweight="bold")
    text = (
        "Planning (Parts 2-3):\n"
        "- Reads env.P[s][a].\n"
        "- Solves Bellman equations.\n\n"
        "Learning (Part 4):\n"
        "- Calls env.step().\n"
        "- Sees only (s,a,r,s').\n"
        "- Must act to discover rules."
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
    Show how a single env.step() tuple nudges a Q-value.
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (14, 5)
    env = GridWorldEnv()
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
            # Illustrative tuple from gymnasium env.step().
            cell, a_cell, s2_cell = (3, 1), "Up", (3, 2)
            s_id = env.cell_to_id(cell)
            a_id = mtlrll0ut.NAME_TO_ACTION_ID[a_cell]
            s2_id = env.cell_to_id(s2_cell)
            r = env.r_step  # living reward since s2 is non-terminal
            snapshots, _ = value_iteration(env)
            u = snapshots[-1]
            q_old = 0.0
            best_next = u[s2_id]
            td_target = r + env.gamma * best_next
            td_error = td_target - q_old
            q_new = q_old + alpha * td_error
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            env._draw_base(ax1, highlight=cell)
            ax1.annotate(
                "",
                xy=(s2_cell[0], s2_cell[1] - 0.1),
                xytext=(cell[0], cell[1] + 0.1),
                arrowprops=dict(
                    arrowstyle="-|>", color="darkblue", linewidth=3.0
                ),
            )
            ax1.text(
                (cell[0] + s2_cell[0]) / 2 + 0.25,
                (cell[1] + s2_cell[1]) / 2,
                "a=%s\nr=%.2f" % (a_cell, r),
                ha="left",
                va="center",
                fontsize=10,
                color="darkblue",
            )
            ax1.set_title("One env.step() tuple", fontsize=13, fontweight="bold")
            ax1.set_xlabel(
                "col\n\n"
                "_Transition diagram_: one transition from env.step() "
                "with reward r",
                fontsize=9,
            )
            text = (
                "Update: Q <- Q + alpha[r +\n"
                "      gamma max Q(s',.) - Q]\n\n"
                "Parameters:\n"
                "  alpha: %.2f  gamma: %.2f\n\n"
                "For step from %s:\n"
                "  old Q: %.3f\n"
                "  TD target: %.3f\n"
                "  TD error: %.3f\n"
                "  new Q: %.3f\n\n"
                "Source: env.step() tuple."
                % (alpha, env.gamma, cell, q_old, td_target, td_error, q_new)
            )
            mtlrll0ut.comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    alpha_slider.observe(update_plot, names="value")
    gamma_slider.observe(update_plot, names="value")
    update_plot()
    param_info = mtlrll0ut.make_param_info(
        {
            "alpha": "learning rate; near 0 the estimate barely moves, near 1 it jumps to the TD target",
            "gamma": "discount factor weighting the next-state value in the TD target",
        }
    )
    controls = ipywidgets.VBox(
        [
            alpha_box,
            gamma_box,
        ]
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
    Compare state coverage under low vs high epsilon using q_learning().
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (16, 5)
    env = GridWorldEnv()
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
        description="training episodes",
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
            eps = epsilon_slider.value
            seed = seed_slider.value
            low = q_learning(
                env, n_episodes=n_episodes, alpha=0.5, epsilon=eps, seed=seed
            )
            high = q_learning(
                env, n_episodes=n_episodes, alpha=0.5, epsilon=0.9, seed=seed
            )
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            mtlrll0ut.draw_visit_heatmap(
                env,
                ax1,
                low["visit_counts"],
                title="Visits at epsilon=%.2f" % eps,
            )
            ax1.set_xlabel(
                "col\n\n"
                "_Low-epsilon visits_: visit counts under the chosen "
                "exploration rate",
                fontsize=9,
            )
            mtlrll0ut.draw_visit_heatmap(
                env,
                ax2,
                high["visit_counts"],
                title="Visits at epsilon=0.90",
            )
            ax2.set_xlabel(
                "col\n\n"
                "_High-epsilon visits_: visit counts under broad exploration",
                fontsize=9,
            )
            text = (
                "Parameters:\n"
                "  epsilon: %.2f\n"
                "  n_episodes: %d\n"
                "  seed: %d\n\n"
                "Source: q_learning() uses\n"
                "  env.step() - no model.\n\n"
                "Key idea:\n"
                "- Low epsilon = narrow path.\n"
                "- High epsilon = broad.\n"
                "- Balance is key." % (eps, n_episodes, seed)
            )
            mtlrll0ut.comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    epsilon_slider.observe(update_plot, names="value")
    n_exp_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    param_info = mtlrll0ut.make_param_info(
        {
            "epsilon": "exploration probability; low epsilon concentrates visits on a narrow corridor, high epsilon spreads visits broadly",
            "n_episodes": "number of training episodes (log scale)",
            "seed": "random seed for reproducible training runs",
        }
    )
    controls = ipywidgets.VBox(
        [
            epsilon_box,
            n_box,
            seed_box,
        ]
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
    Train Q-learning via env.step() and compare to the planning optimum.
    """
    _LOG.debug(hprint.to_str("figsize"))
    if figsize is None:
        figsize = (18, 5)
    env = GridWorldEnv()
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
    # Planning optimum for comparison.
    opt_u, _ = value_iteration(env)
    optimal_policy = extract_policy(env, opt_u[-1])

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
            env._draw_base(ax1)
            mtlrll0ut.draw_policy_arrows(env, ax1, learned, color="darkblue")
            ax1.set_title(
                "Q-learning policy (gymnasium)", fontsize=13, fontweight="bold"
            )
            ax1.set_xlabel(
                "col\n\n"
                "_Q-learning policy_: the greedy policy derived from "
                "the learned Q-table",
                fontsize=9,
            )
            window = max(1, len(returns) // 50)
            smooth = pd.Series(returns).rolling(window, min_periods=1).mean()
            ax2.plot(smooth, color="seagreen")
            ax2.set_xlabel(
                "episode\n\n_Learning curve_: smoothed total return per episode",
                fontsize=9,
            )
            ax2.set_ylabel("return (smoothed)")
            ax2.set_title("Learning curve", fontsize=13, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            n_match = sum(
                1
                for c in env.all_cells
                if c not in env.terminal_cells
                and learned.get(c) == optimal_policy.get(c)
            )
            n_total = len(
                [c for c in env.all_cells if c not in env.terminal_cells]
            )
            text = (
                "Parameters:\n"
                "  n_episodes: %d\n"
                "  alpha: %.2f\n"
                "  epsilon: %.2f\n\n"
                "Policy match vs VI: %d / %d\n\n"
                "Source: q_learning() uses\n"
                "  env.step() exclusively.\n"
                "Same payoff: optimal policy\n"
                "without ever reading env.P."
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

    n_exp_slider.observe(update_plot, names="value")
    alpha_slider.observe(update_plot, names="value")
    epsilon_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    param_info = mtlrll0ut.make_param_info(
        {
            "n_episodes": "number of training episodes (log scale); more episodes give the Q-table time to converge",
            "alpha": "learning rate controlling how much each experience updates the Q-value",
            "epsilon": "exploration probability for epsilon-greedy action selection",
            "seed": "random seed for reproducible training",
        }
    )
    controls = ipywidgets.VBox(
        [
            n_box,
            alpha_box,
            epsilon_box,
            seed_box,
        ]
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))
