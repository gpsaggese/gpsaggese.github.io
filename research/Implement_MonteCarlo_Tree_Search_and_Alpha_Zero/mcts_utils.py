"""
Monte Carlo Tree Search (MCTS): a game-agnostic tree search engine.

Implements the MCTS algorithm (selection, expansion, rollout,
backpropagation) against the `Game` interface defined in `game.py`, used to
play any two-player, zero-sum, perfect-information game.

See `README.md` for a description of every file in this directory.

Import as:

import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.mcts_utils as rimtsaazmu
"""

import logging
import math
import random
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import helpers.hdbg as hdbg
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game as rimtsaazg
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.search_algorithms_utils as rimtsaazsau

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


# Exploration constant `C` in the UCT formula: higher values favor trying
# less-visited moves over exploiting the current best estimate.
EXPLORATION_CONSTANT = math.sqrt(2)

# Default number of MCTS simulations run per move when the caller does not
# override it.
DEFAULT_NUM_SIMULATIONS = 500

# Default number of random playouts per root action for flat Monte Carlo
# (Lesson09.8: "For each action a at s0: Run N random playouts").
DEFAULT_NUM_ROLLOUTS = 200


# #############################################################################
# MCTSNode
# #############################################################################


class MCTSNode:
    """
    A single node in the MCTS search tree.

    Tracks the game state it represents, its position in the tree, and the
    running visit count / value total used by the UCT formula. This is a
    different shape than `search_algorithms_utils.SearchNode`: MCTS grows one
    node per simulation and stores a running `visit_count` / `value_sum`,
    rather than visiting every node once and backing up a final `value` in a
    single recursive pass.
    """

    def __init__(
        self,
        state: rimtsaazg.State,
        *,
        parent: Optional["MCTSNode"] = None,
        move: Optional[rimtsaazg.Move] = None,
        untried_moves: Optional[List[rimtsaazg.Move]] = None,
    ) -> None:
        """
        Initialize a node for `state`.

        :param state: game state this node represents
        :param parent: predecessor node in the tree
            - Default: `None` (root node)
        :param move: move applied to `parent` to reach `state`
            - Default: `None` (root node)
        :param untried_moves: legal moves from `state` not yet expanded
            - Default: `None` (treated as no moves, e.g., a terminal state)
        """
        self.state = state
        self.parent = parent
        self.move = move
        self.children: Dict[rimtsaazg.Move, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.untried_moves: List[rimtsaazg.Move] = list(untried_moves or [])

    @property
    def is_fully_expanded(self) -> bool:
        """
        Check whether every legal move from this node already has a child.
        """
        fully_expanded = not self.untried_moves
        return fully_expanded

    @property
    def mean_value(self) -> float:
        """
        Average simulation outcome observed through this node so far.
        """
        mean = self.value_sum / self.visit_count if self.visit_count else 0.0
        return mean

    def __str__(self) -> str:
        """
        Return a human-readable summary of the node for debugging/logging.

        :return: node summary, e.g.,
            "MCTSNode(move=3, visits=12, mean_value=0.42, children=2,
            untried=1)"
        """
        txt = (
            "MCTSNode(move=%s, visits=%d, mean_value=%.2f, children=%d,"
            " untried=%d)"
            % (
                self.move,
                self.visit_count,
                self.mean_value,
                len(self.children),
                len(self.untried_moves),
            )
        )
        return txt


# #############################################################################
# MCTS phases
# #############################################################################


def _uct_score(
    child: MCTSNode,
    parent_visit_count: int,
    *,
    exploration_constant: float = EXPLORATION_CONSTANT,
) -> float:
    """
    Score a child node using the UCT formula:
    `Q/N + C * sqrt(ln(N_parent) / N)`.

    An unvisited child gets an infinite score so every move is tried at least
    once before any move is revisited.

    :param child: candidate child node
    :param parent_visit_count: visit count of the parent node
    :param exploration_constant: the `C` in the UCT formula, trades off
        exploration (higher) against exploitation (lower)
        - Default: `EXPLORATION_CONSTANT`
    :return: UCT score, higher is more promising to explore
    """
    if child.visit_count == 0:
        score = math.inf
    else:
        exploitation = child.mean_value
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visit_count) / child.visit_count
        )
        score = exploitation + exploration
    return score


def _select(
    node: MCTSNode,
    game: rimtsaazg.Game,
    *,
    exploration_constant: float = EXPLORATION_CONSTANT,
) -> MCTSNode:
    """
    Descend the tree from `node`, following the highest-UCT child at each
    step, until reaching a node that still has untried moves or is terminal.

    :param node: node to start the descent from (typically the root)
    :param game: game rules, used to check terminality
    :param exploration_constant: the `C` in the UCT formula, forwarded to
        `_uct_score()`
        - Default: `EXPLORATION_CONSTANT`
    :return: node ready for expansion (or simulation, if terminal)
    """
    while not game.is_terminal(node.state) and node.is_fully_expanded:
        node = max(
            node.children.values(),
            key=lambda child: _uct_score(
                child,
                node.visit_count,
                exploration_constant=exploration_constant,
            ),
        )
    return node


def _expand(node: MCTSNode, game: rimtsaazg.Game) -> MCTSNode:
    """
    Add one new child to `node` for a randomly chosen untried move.

    :param node: node with at least one untried move
    :param game: game rules, used to apply the move
    :return: newly created child node
    """
    move_idx = random.randrange(len(node.untried_moves))
    move = node.untried_moves.pop(move_idx)
    child_state = game.apply_move(node.state, move)
    child = MCTSNode(
        child_state,
        parent=node,
        move=move,
        untried_moves=game.get_legal_moves(child_state),
    )
    node.children[move] = child
    return child


def random_rollout(game: rimtsaazg.Game, state: rimtsaazg.State) -> int:
    """
    Play out `state` to a terminal state using uniformly random moves.

    This is MCTS's default policy (used below by the simulation phase); it
    is also flat Monte Carlo's only policy (see "Flat Monte Carlo" below),
    since flat MC is MCTS with a tree of depth one and reuses this same
    function instead of duplicating it.

    :param game: game rules, used to generate and apply moves
    :param state: state to roll out from
    :return: outcome of the rollout: `1`, `-1`, or `0` for a draw
    """
    while not game.is_terminal(state):
        moves = game.get_legal_moves(state)
        move = random.choice(moves)
        state = game.apply_move(state, move)
    winner = game.get_winner(state)
    return winner


def _backpropagate(node: MCTSNode, value: float) -> None:
    """
    Propagate a simulation result up the tree to the root.

    `value` is the outcome from the perspective of the player who made the
    move leading into `node`. Since players alternate at every ply, that
    perspective flips (the sign is negated) at each step up the tree.

    :param node: node the simulation was run from
    :param value: simulation outcome (`1.0` win / `-1.0` loss / `0.0` draw)
        from the perspective of the player who moved into `node`
    """
    current: Optional[MCTSNode] = node
    while current is not None:
        current.visit_count += 1
        current.value_sum += value
        value = -value
        current = current.parent


# #############################################################################
# Public API
# #############################################################################


def build_mcts_tree(
    game: rimtsaazg.Game,
    state: rimtsaazg.State,
    *,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    exploration_constant: float = EXPLORATION_CONSTANT,
) -> MCTSNode:
    """
    Run MCTS from `state` and return the fully-built search tree.

    Runs `num_simulations` iterations of selection, expansion, random-rollout
    simulation, and backpropagation from a fresh root, then returns that
    root node itself, rather than only the best move: `run_mcts()` picks the
    best move from it, and a notebook can instead inspect the tree (e.g., to
    visualize visit counts and mean values per child).

    :param game: game-agnostic rules implementation
    :param state: state to search from, must not be terminal
    :param num_simulations: number of MCTS iterations to run
        - Default: `DEFAULT_NUM_SIMULATIONS`
    :param exploration_constant: the `C` in the UCT formula, forwarded to
        `_select()`
        - Default: `EXPLORATION_CONSTANT`
    :return: root node of the search tree, with `num_simulations` simulations
        backpropagated through it
    """
    hdbg.dassert(
        not game.is_terminal(state), "Cannot run MCTS from a terminal state"
    )
    root = MCTSNode(state, untried_moves=game.get_legal_moves(state))
    for _ in range(num_simulations):
        leaf = _select(root, game, exploration_constant=exploration_constant)
        if not game.is_terminal(leaf.state):
            leaf = _expand(leaf, game)
        winner = random_rollout(game, leaf.state)
        # `value` must be from the perspective of the player who moved into
        # `leaf`, i.e., the opponent of the player to move at `leaf.state`.
        mover_into_leaf = -game.get_current_player(leaf.state)
        if winner == 0:
            value = 0.0
        elif winner == mover_into_leaf:
            value = 1.0
        else:
            value = -1.0
        _backpropagate(leaf, value)
    return root


def run_mcts(
    game: rimtsaazg.Game,
    state: rimtsaazg.State,
    *,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    exploration_constant: float = EXPLORATION_CONSTANT,
) -> rimtsaazg.Move:
    """
    Run MCTS from `state` and return the most-visited move at the root.

    Builds the search tree via `build_mcts_tree()`, then returns the root
    child with the highest visit count (the standard, low-variance choice,
    as opposed to the child with the highest mean value).

    :param game: game-agnostic rules implementation
    :param state: state to search from, must not be terminal
    :param num_simulations: number of MCTS iterations to run
        - Default: `DEFAULT_NUM_SIMULATIONS`
    :param exploration_constant: the `C` in the UCT formula, trades off
        exploration (higher) against exploitation (lower)
        - Default: `EXPLORATION_CONSTANT`
    :return: move with the highest visit count at the root
    """
    root = build_mcts_tree(
        game,
        state,
        num_simulations=num_simulations,
        exploration_constant=exploration_constant,
    )
    best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[
        0
    ]
    return best_move


def random_player(
    game: rimtsaazg.Game, state: rimtsaazg.State
) -> rimtsaazg.Move:
    """
    Select a uniformly random legal move.

    Matches the `(game, state) -> move` signature expected by `play_game()`.

    :param game: game-agnostic rules implementation
    :param state: current game state, must not be terminal
    :return: randomly chosen legal move
    """
    moves = game.get_legal_moves(state)
    hdbg.dassert(moves, "No legal moves available to choose from")
    move = random.choice(moves)
    return move


def make_mcts_player(
    *, num_simulations: int = DEFAULT_NUM_SIMULATIONS
) -> Callable[[rimtsaazg.Game, rimtsaazg.State], rimtsaazg.Move]:
    """
    Build a `(game, state) -> move` player function backed by MCTS.

    :param num_simulations: number of MCTS simulations to run per move
        - Default: `DEFAULT_NUM_SIMULATIONS`
    :return: player function suitable for `play_game()`
    """

    def player(game: rimtsaazg.Game, state: rimtsaazg.State) -> rimtsaazg.Move:
        move = run_mcts(game, state, num_simulations=num_simulations)
        return move

    return player


def play_game(
    game: rimtsaazg.Game,
    player1: Callable[[rimtsaazg.Game, rimtsaazg.State], rimtsaazg.Move],
    player2: Callable[[rimtsaazg.Game, rimtsaazg.State], rimtsaazg.Move],
    *,
    verbose: bool = False,
) -> Tuple[int, List[rimtsaazg.State]]:
    """
    Play one full game between two players, alternating moves.

    :param game: game-agnostic rules implementation
    :param player1: move-selection function for player `1`
    :param player2: move-selection function for player `-1`
    :param verbose: if True, print the board after every move
        - Default: `False`
    :return: tuple `(winner, history)`
        - `winner`: `1`, `-1`, or `0` for a draw
        - `history`: states visited, from the initial state to the terminal
          state (inclusive)
    """
    state = game.get_initial_state()
    history = [state]
    players = {1: player1, -1: player2}
    if verbose:
        print(game.render(state))
    while not game.is_terminal(state):
        current_player = game.get_current_player(state)
        move = players[current_player](game, state)
        state = game.apply_move(state, move)
        history.append(state)
        if verbose:
            print()
            print(game.render(state))
    winner = game.get_winner(state)
    return winner, history


# #############################################################################


def evaluate_win_rate(
    game: rimtsaazg.Game,
    player_under_test: Callable[
        [rimtsaazg.Game, rimtsaazg.State], rimtsaazg.Move
    ],
    opponent: Callable[[rimtsaazg.Game, rimtsaazg.State], rimtsaazg.Move],
    *,
    num_games: int = 200,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Play many games of `player_under_test` (as player `1`) against
    `opponent` (as player `-1`) and report outcome statistics.

    :param game: game-agnostic rules implementation
    :param player_under_test: move-selection function to evaluate, plays as
        player `1`
    :param opponent: move-selection function to evaluate against, plays as
        player `-1`
    :param num_games: number of games to play
        - Default: `200`
    :param show_progress: if True, display a `tqdm` progress bar
        - Default: `True`
    :return: dict with keys `num_games`, `win_rate`, `loss_rate`, `draw_rate`
        (rates measured from `player_under_test`'s perspective)
    """
    outcomes = np.zeros(num_games)
    game_range = tqdm.trange(num_games) if show_progress else range(num_games)
    for i in game_range:
        winner, _ = play_game(game, player_under_test, opponent)
        outcomes[i] = winner
    results = {
        "num_games": float(num_games),
        "win_rate": float(np.mean(outcomes == 1)),
        "loss_rate": float(np.mean(outcomes == -1)),
        "draw_rate": float(np.mean(outcomes == 0)),
    }
    return results


def plot_win_rate_results(
    results: Dict[str, float], *, figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Plot a bar chart of win / draw / loss rates from `evaluate_win_rate()`.

    :param results: dict as returned by `evaluate_win_rate()`
    :param figsize: figure size passed to `plt.subplots()`
        - Default: `None` (uses `plt.rcParams["figure.figsize"]`)
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    labels = ["win", "draw", "loss"]
    rates = [results["win_rate"], results["draw_rate"], results["loss_rate"]]
    colors = ["#4C9A2A", "#B0B0B0", "#C0392B"]
    _, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, rates, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of games")
    ax.set_title(f"MCTS vs. random over {int(results['num_games'])} games")
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.02,
            f"{rate:.1%}",
            ha="center",
        )
    plt.show()


# #############################################################################
# Flat Monte Carlo
# #############################################################################


def _estimate_action_value(
    game: rimtsaazg.Game,
    state: rimtsaazg.State,
    move: rimtsaazg.Move,
    *,
    num_rollouts: int,
) -> float:
    """
    Estimate `Q_hat(state, move)` as the mean outcome, from the mover's
    perspective, over `num_rollouts` random playouts through `move`.

    Lesson09.8: "For each action a at s0: Run N random playouts to a
    terminal state, estimate Q_hat(s0, a) as the mean terminal outcome."
    Reuses `random_rollout()` above, MCTS's own default policy.

    :param game: game-agnostic rules implementation
    :param state: state the action is taken from
    :param move: action to evaluate
    :param num_rollouts: number of random playouts to average over
    :return: mean outcome in `[-1, 1]` from the mover's perspective
    """
    mover = game.get_current_player(state)
    total = 0.0
    for _ in range(num_rollouts):
        rollout_state = game.apply_move(state, move)
        winner = random_rollout(game, rollout_state)
        if winner == mover:
            total += 1.0
        elif winner != 0:
            total -= 1.0
    q_hat = total / num_rollouts
    return q_hat


def build_flat_mc_tree(
    game: rimtsaazg.Game,
    state: rimtsaazg.State,
    *,
    num_rollouts: int = DEFAULT_NUM_ROLLOUTS,
) -> rimtsaazsau.SearchNode:
    """
    Score every legal move at `state` by flat Monte Carlo and return the
    resulting depth-one search tree.

    Lesson09.8: "Flat Monte Carlo is exactly MCTS with a tree of depth
    one": one child per root action, no further expansion. Reuses
    `search_algorithms_utils.SearchNode` (rather than `MCTSNode` above) so
    the tree renders with the same `build_tree_graph()` used for minimax,
    alpha-beta, and depth-limited search.

    :param game: game-agnostic rules implementation
    :param state: state to search from, must not be terminal
    :param num_rollouts: number of random playouts averaged per action
        - Default: `DEFAULT_NUM_ROLLOUTS`
    :return: root node with one scored child per legal move
    """
    hdbg.dassert(
        not game.is_terminal(state), "Cannot search from a terminal state"
    )
    root = rimtsaazsau.SearchNode(state)
    child_values = []
    for move in game.get_legal_moves(state):
        child_state = game.apply_move(state, move)
        child = root.add_child(move, child_state)
        value = _estimate_action_value(
            game, state, move, num_rollouts=num_rollouts
        )
        child.value = value
        child_values.append(value)
    current_player = game.get_current_player(state)
    root.value = max(child_values) if current_player == 1 else min(child_values)
    return root


def run_flat_mc(
    game: rimtsaazg.Game,
    state: rimtsaazg.State,
    *,
    num_rollouts: int = DEFAULT_NUM_ROLLOUTS,
) -> rimtsaazg.Move:
    """
    Run flat Monte Carlo from `state` and return `argmax_a Q_hat(s0, a)`.

    :param game: game-agnostic rules implementation
    :param state: state to search from, must not be terminal
    :param num_rollouts: number of random playouts averaged per action
        - Default: `DEFAULT_NUM_ROLLOUTS`
    :return: best-estimated move for the player to move at `state`
    """
    root = build_flat_mc_tree(game, state, num_rollouts=num_rollouts)
    best_move = rimtsaazsau.pick_best_move(game, root)
    return best_move


def make_flat_mc_player(
    *, num_rollouts: int = DEFAULT_NUM_ROLLOUTS
) -> Callable[[rimtsaazg.Game, rimtsaazg.State], rimtsaazg.Move]:
    """
    Build a `(game, state) -> move` player function backed by flat Monte
    Carlo.

    :param num_rollouts: number of random playouts averaged per action
        - Default: `DEFAULT_NUM_ROLLOUTS`
    :return: player function suitable for `play_game()`
    """

    def player(game: rimtsaazg.Game, state: rimtsaazg.State) -> rimtsaazg.Move:
        move = run_flat_mc(game, state, num_rollouts=num_rollouts)
        return move

    return player
