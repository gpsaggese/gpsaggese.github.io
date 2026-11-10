"""
Represent, learn, evaluate, and search with a policy/value network and PUCT.

States remain the immutable tuples owned by the game implementations. Encodings
are model inputs only: never pass an encoded board back to the game rules.
Callers supply reachable states and legal moves from the existing game API.
Evaluators return legal action priors and player-to-move values; terminal
states have an all-zero policy and an exact outcome.
Search stores values in each node's player-to-move perspective and negates
them between parent and child in strictly alternating two-player games.
Supervised minibatches train a CPU MLP on fixed policy and value targets.
Self-play collects search policies and completed-game outcomes without updates.
The training loop alternates collection and replay updates on one CPU model.

Import as:

import research.Implement_AlphaZero.alphazero_utils as rialzut
"""

import collections
import dataclasses
import functools
import logging
import math
import numbers
import os
import random
import tempfile
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange

import helpers.hdbg as hdbg
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game as rimtsaazg
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.mcts_utils as rimtsaazmu
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.search_algorithms_utils as rimtsaazsau

_LOG = logging.getLogger(__name__)


def encode_state(game: rimtsaazg.Game, state: rimtsaazg.State) -> np.ndarray:
    """
    Encode a flat board from the perspective of its player to move.

    This encoder assumes cells are `0` (empty), `1` (X), or `-1` (O), as in
    the existing board games. For example, after X plays cell 0, O sees
    `[-1, 0, 0, 0, 0, 0, 0, 0, 0]`. Cell positions never change.

    :param game: rules supplying the player to move, including at terminality
    :param state: reachable board in the game's original representation
    :return: fresh flat `float32` array, shape `(len(state),)`, containing
        `+1` for own pieces, `-1` for opponent pieces, and `0` for empty cells
    """
    _LOG.debug("Encoding state='%s'", state)
    player = game.get_current_player(state)
    # Multiplication allocates an independent input and preserves action indices.
    encoded = np.asarray(state, dtype=np.float32) * player
    _LOG.debug("Encoded state='%s'", encoded)
    return encoded


def get_legal_action_mask(
    game: rimtsaazg.Game, state: rimtsaazg.State, action_size: int
) -> np.ndarray:
    """
    Mark legal moves in a fixed action space, including occupied positions.

    This assumes game moves are integer indices in `[0, action_size)`.
    Tic-Tac-Toe has 9 cell actions; Connect Four has 7 column actions,
    even though its board has 42 cells. A mask is not a probability vector.

    :param game: rules supplying legal action indices
    :param state: reachable game state
    :param action_size: total number of possible action indices, positive
    :return: fresh boolean array of shape `(action_size,)`; all False for
        terminal states, even if the board still has empty cells
    """
    _LOG.debug("Masking state='%s', action_size='%s'", state, action_size)
    hdbg.dassert_lt(0, action_size, "The action space must be nonempty")
    mask = np.zeros(action_size, dtype=np.bool_)
    # Ask the game for legality: empty cells alone do not detect a finished game.
    for move in game.get_legal_moves(state):
        hdbg.dassert_lte(0, move, "Actions must be nonnegative indices")
        hdbg.dassert_lt(move, action_size, "Action exceeds the fixed space")
        mask[move] = True
    _LOG.debug("Legal action mask='%s'", mask)
    return mask


def get_terminal_value(
    game: rimtsaazg.Game, state: rimtsaazg.State
) -> Optional[float]:
    """
    Return the exact outcome from the perspective of the player to move.

    For a finished game, the player to move means the player who would move
    next under the game's turn convention. After X wins, O is next and the
    value is `-1.0`. Alternating-player search must negate this value
    when backing it up to the parent.

    :param game: two-player zero-sum rules with players `1` and `-1`
    :param state: reachable game state in its original representation
    :return: `None` for an unfinished game; otherwise `+1.0` for a win,
        `-1.0` for a loss, or `0.0` for a draw for the player to move
    """
    _LOG.debug("Evaluating terminal state='%s'", state)
    # A draw and an unfinished position both have winner 0 in the game API.
    value = None
    if game.is_terminal(state):
        value = float(game.get_winner(state) * game.get_current_player(state))
    _LOG.debug("Terminal value='%s'", value)
    return value


# #############################################################################
# Policy normalization
# #############################################################################


def _as_policy_array(policy: np.ndarray) -> np.ndarray:
    """
    Copy a finite, nonnegative weight vector into floating-point storage.

    :param policy: nonempty one-dimensional real weights, not logits
    :return: independent `float64` array with validated weights
    """
    hdbg.dassert(np.isrealobj(policy), "Policy weights must be real")
    weights = np.array(policy, dtype=np.float64, copy=True)
    hdbg.dassert_eq(weights.ndim, 1, "Policy must be a flat action vector")
    hdbg.dassert_lt(0, weights.size, "The action space must be nonempty")
    hdbg.dassert(np.isfinite(weights).all(), "Policy weights must be finite")
    hdbg.dassert((weights >= 0).all(), "Policy weights must be nonnegative")
    return weights


def normalize_policy(
    policy: np.ndarray, legal_action_mask: np.ndarray
) -> np.ndarray:
    """
    Mask illegal actions and normalize nonnegative weights over legal actions.

    For example, weights `[2, 9, 1]` and mask `[True, False, True]` yield
    `[2/3, 0, 1/3]`. Inputs are not modified. All weights, including illegal
    entries, must be finite and nonnegative; logits require conversion to
    weights before calling this function.

    :param policy: nonempty flat real weight vector
    :param legal_action_mask: boolean vector of the same shape
    :return: fresh `float64` probability vector; uniform over legal actions
        if their total weight is zero, or all zeros if no actions are legal
    """
    _LOG.debug("Normalizing policy with shape='%s'", np.shape(policy))
    weights = _as_policy_array(policy)
    mask = np.asarray(legal_action_mask)
    hdbg.dassert_eq(mask.dtype, np.dtype(bool), "Legality must be boolean")
    hdbg.dassert_eq(mask.shape, weights.shape, "Mask must match action space")
    # Scale by the largest legal weight before summing to avoid overflow.
    normalized = np.where(mask, weights, 0.0)
    scale = normalized.max()
    if scale > 0:
        normalized /= scale
        normalized /= normalized.sum()
    elif mask.any():
        # Zero legal mass carries no preference; use a uniform legal prior.
        normalized = mask.astype(np.float64) / np.count_nonzero(mask)
    # With no legal moves, the zero vector represents absence of a policy.
    return normalized


# #############################################################################
# PolicyValuePrediction
# #############################################################################


@dataclasses.dataclass
class PolicyValuePrediction:
    """
    Hold an action policy and a scalar value for the player to move.

    `policy` is a nonempty flat vector summing to one, or all zeros for a
    terminal state. `value` is a finite real number in `[-1, 1]`. The
    constructor validates these numerical constraints and owns a copy of the
    policy. The evaluator is responsible for game-specific legality and for
    using the zero policy only at terminality.
    """

    policy: np.ndarray
    value: float

    def __post_init__(self) -> None:
        """
        Validate the prediction at the evaluator output boundary.
        """
        self.policy = _as_policy_array(self.policy)
        # A normalized policy cannot contain an entry greater than one.
        hdbg.dassert((self.policy <= 1).all(), "Probabilities must be <= 1")
        mass = self.policy.sum()
        hdbg.dassert(
            mass == 0 or np.isclose(mass, 1.0, rtol=1e-6, atol=1e-8),
            "Policy must sum to one, or be zero at terminality",
        )
        hdbg.dassert_isinstance(
            self.value, numbers.Real, "Value must be scalar"
        )
        hdbg.dassert(np.isfinite(self.value), "Value must be finite")
        hdbg.dassert_lte(-1.0, self.value, "Value cannot be below a loss")
        hdbg.dassert_lte(self.value, 1.0, "Value cannot exceed a win")
        self.value = float(self.value)


# Evaluators share one signature; consumers need not know how values are obtained.
PolicyValueEvaluator = Callable[
    [rimtsaazg.Game, rimtsaazg.State], PolicyValuePrediction
]


# #############################################################################
# UniformEvaluator
# #############################################################################


class UniformEvaluator:
    """
    Return uniform legal priors and a neutral estimate for unfinished games.

    The neutral value `0.0` expresses no preference; it does not assert that
    the game will draw. Terminal values instead come directly from the rules.
    This baseline performs no search, random sampling, or learning.
    """

    def __init__(self, action_size: int) -> None:
        """
        Set the fixed action space used by every prediction.

        :param action_size: positive action count, e.g., 9 cells for
            Tic-Tac-Toe or 7 columns for Connect Four
        """
        hdbg.dassert_isinstance(
            action_size, int, "Action count must be an integer"
        )
        hdbg.dassert_lt(0, action_size, "The action space must be nonempty")
        self.action_size = action_size

    def __call__(
        self, game: rimtsaazg.Game, state: rimtsaazg.State
    ) -> PolicyValuePrediction:
        """
        Evaluate a reachable state using the game's original representation.

        :param game: rules with integer actions and players `1` and `-1`
        :param state: original game state, not a player-relative encoding
        :return: legal policy and player-to-move value; terminal predictions
            have an all-zero policy and the exact game outcome
        """
        _LOG.debug("Evaluating state='%s'", state)
        mask = get_legal_action_mask(game, state, self.action_size)
        value = get_terminal_value(game, state)
        if value is None:
            hdbg.dassert(
                mask.any(), "An unfinished game must have a legal action"
            )
            value = 0.0
        policy = normalize_policy(np.ones(self.action_size), mask)
        prediction = PolicyValuePrediction(policy, value)
        return prediction


# #############################################################################
# AlphaZeroNode
# #############################################################################


class AlphaZeroNode:
    """
    Store a game state, incoming policy prior, and accumulated search results.

    `value_sum` and `mean_value` are from the player-to-move perspective in
    this node's state. A parent therefore values this child as `-mean_value`.
    `prior` is the probability assigned to the incoming action by the parent;
    the root uses `1.0`. Children map original action indices to nodes.
    """

    def __init__(self, state: rimtsaazg.State, prior: float) -> None:
        """
        Initialize an unvisited, unexpanded node.

        :param state: original immutable game state
        :param prior: incoming action probability
        """
        self.state = state
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[rimtsaazg.Move, "AlphaZeroNode"] = {}

    @property
    def mean_value(self) -> float:
        """
        Return the average backed-up value, or zero before the first visit.
        """
        value = self.value_sum / self.visit_count if self.visit_count else 0.0
        return value


# #############################################################################
# PUCT selection and leaf evaluation
# #############################################################################


def get_puct_scores(
    node: AlphaZeroNode, exploration_constant: float
) -> Dict[rimtsaazg.Move, float]:
    """
    Score each child from its parent's perspective.

    The score is `-child.mean_value + c * child.prior *
    sqrt(max(1, node.visit_count)) / (1 + child.visit_count)`.
    Using a numerator count of one at an unvisited parent makes its first
    selection honor the priors. An unvisited child's mean value is zero.

    :param node: parent whose children are candidates; leaves return `{}`
    :param exploration_constant: finite nonnegative PUCT exploration weight
    :return: action-indexed scores; larger scores are preferred
    """
    hdbg.dassert_isinstance(
        exploration_constant, numbers.Real, "Exploration weight must be real"
    )
    hdbg.dassert(
        np.isfinite(exploration_constant), "Exploration must be finite"
    )
    hdbg.dassert_lte(
        0.0, exploration_constant, "Exploration cannot be negative"
    )
    # Child values favor the opponent; the minus sign changes perspective.
    parent_scale = math.sqrt(max(1, node.visit_count))
    scores = {
        move: -child.mean_value
        + exploration_constant
        * child.prior
        * parent_scale
        / (1 + child.visit_count)
        for move, child in node.children.items()
    }
    return scores


def _evaluate_search_leaf(
    node: AlphaZeroNode,
    game: rimtsaazg.Game,
    evaluator: PolicyValueEvaluator,
    action_size: int,
) -> float:
    """
    Get a leaf value and expand every legal action if the game is unfinished.

    :param node: unexpanded node or previously visited terminal leaf
    :param game: strictly alternating two-player game rules
    :param evaluator: callable supplying nonterminal policy/value predictions
    :param action_size: fixed number of policy entries
    :return: leaf value from its player-to-move perspective
    """
    value = get_terminal_value(game, node.state)
    if value is None:
        # Terminal outcomes bypass the evaluator, so estimates cannot override wins.
        prediction = evaluator(game, node.state)
        hdbg.dassert_isinstance(
            prediction,
            PolicyValuePrediction,
            "Evaluator must return a prediction",
        )
        hdbg.dassert_eq(
            prediction.policy.shape,
            (action_size,),
            "Policy must match action space",
        )
        mask = get_legal_action_mask(game, node.state, action_size)
        hdbg.dassert(mask.any(), "An unfinished game must have a legal action")
        # Restrict priors to the legal actions before creating child nodes.
        priors = normalize_policy(prediction.policy, mask)
        for move in np.flatnonzero(mask):
            move = int(move)
            state = game.apply_move(node.state, move)
            node.children[move] = AlphaZeroNode(state, float(priors[move]))
        value = prediction.value
    return value


# #############################################################################
# Search and visit policy
# #############################################################################


def _validate_root_noise(dirichlet_alpha: float, noise_fraction: float) -> None:
    """
    Validate the concentration and mixing weight for legal root noise.
    """
    hdbg.dassert(
        np.isfinite(dirichlet_alpha) and dirichlet_alpha > 0,
        "Dirichlet concentration must be finite and positive",
    )
    hdbg.dassert(
        np.isfinite(noise_fraction) and 0 <= noise_fraction <= 1,
        "Noise fraction must lie in [0, 1]",
    )


def build_search_tree(
    game: rimtsaazg.Game,
    state: rimtsaazg.State,
    evaluator: PolicyValueEvaluator,
    *,
    action_size: int,
    num_simulations: int,
    exploration_constant: float = 1.0,
    root_noise_fraction: float = 0.0,
    dirichlet_alpha: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> AlphaZeroNode:
    """
    Build a fresh PUCT tree using an explicitly supplied policy/value evaluator.

    Expand the nonterminal root once before the counted simulations. Its
    initial value estimate is not backed up. Each simulation descends by PUCT
    to a leaf, obtains an exact terminal value or an evaluator prediction,
    and updates every node on its path. Thus root visits and the sum of root
    child visits both equal `num_simulations`. Equal scores choose the lowest
    action index. Optional Dirichlet noise changes root child priors before
    selection; deeper priors remain untouched. No rollouts or tree reuse occur.

    :param game: strictly alternating two-player, zero-sum, deterministic rules
    :param state: reachable nonterminal state in its original representation
    :param evaluator: callable returning nonterminal action priors and a
        player-to-move value; terminal leaves never call it
    :param action_size: fixed positive action count, independent of board size
    :param num_simulations: nonnegative integer simulation budget; zero expands
        only the root, allowing inspection of priors
    :param exploration_constant: finite nonnegative exploration weight
        - Default: `1.0`
    :param root_noise_fraction: root prior mixing weight in `[0, 1]`;
        zero disables noise and consumes no random numbers
    :param dirichlet_alpha: finite positive concentration for each legal action
    :param rng: caller-owned NumPy generator, required when noise is enabled
    :return: root with inspectable priors, states, visits, values, and children
    """
    _LOG.debug(
        "Searching state='%s', num_simulations='%s'", state, num_simulations
    )
    hdbg.dassert_isinstance(action_size, int, "Action count must be an integer")
    hdbg.dassert_lt(0, action_size, "The action space must be nonempty")
    hdbg.dassert_isinstance(
        num_simulations, int, "Simulation count must be an integer"
    )
    hdbg.dassert_lte(0, num_simulations, "Simulation count cannot be negative")
    hdbg.dassert(not game.is_terminal(state), "Cannot search a terminal root")
    _validate_root_noise(dirichlet_alpha, root_noise_fraction)
    if root_noise_fraction > 0:
        hdbg.dassert_isinstance(
            rng, np.random.Generator, "Root noise requires an explicit RNG"
        )
    root = AlphaZeroNode(state, 1.0)
    # Validate exploration before any evaluator call, including with zero budget.
    get_puct_scores(root, exploration_constant)
    _evaluate_search_leaf(root, game, evaluator, action_size)
    if root_noise_fraction > 0:
        # Children retain ascending action order; sample only legal entries.
        noise = rng.dirichlet(np.full(len(root.children), dirichlet_alpha))
        hdbg.dassert(np.isfinite(noise).all(), "Root noise must be finite")
        for child, noise_prior in zip(root.children.values(), noise):
            child.prior = float(
                (1 - root_noise_fraction) * child.prior
                + root_noise_fraction * noise_prior
            )
    for _ in range(num_simulations):
        node = root
        path = [root]
        # A node with no children is either unexpanded or terminal.
        while node.children:
            scores = get_puct_scores(node, exploration_constant)
            move = max(scores, key=lambda action: (scores[action], -action))
            node = node.children[move]
            path.append(node)
        value = _evaluate_search_leaf(node, game, evaluator, action_size)
        # Start at the leaf's own perspective, then alternate at every parent.
        for visited in reversed(path):
            visited.visit_count += 1
            visited.value_sum += value
            value = -value
    return root


def get_visit_policy(
    root: AlphaZeroNode, action_size: int, *, temperature: float = 1.0
) -> np.ndarray:
    """
    Convert root child visit counts into a fixed-size action distribution.

    Positive temperature returns weights proportional to `N(s, a) ** (1/tau)`.
    At temperature one this is the normalized visit count; at zero it is a
    one-hot policy on the lowest-index most-visited action. With no visits,
    apply the same temperature rule to legal priors instead. Unvisited actions
    keep zero mass whenever any action has been visited.

    :param root: expanded nonterminal root returned by `build_search_tree()`
    :param action_size: fixed positive action count used to build the tree
    :param temperature: finite nonnegative sampling temperature; default one
        preserves the usual normalized visit policy
    :return: fresh `float64` policy with zero mass on illegal actions
    """
    hdbg.dassert(
        np.isfinite(temperature) and temperature >= 0,
        "Temperature must be finite and nonnegative",
    )
    hdbg.dassert_isinstance(action_size, int, "Action count must be an integer")
    hdbg.dassert_lt(0, action_size, "The action space must be nonempty")
    hdbg.dassert(
        root.children, "A visit policy needs an expanded nonterminal root"
    )
    counts = np.zeros(action_size)
    priors = np.zeros(action_size)
    mask = np.zeros(action_size, dtype=bool)
    for move, child in root.children.items():
        hdbg.dassert_lte(0, move, "Action indices must be nonnegative")
        hdbg.dassert_lt(move, action_size, "Action exceeds the fixed space")
        counts[move] = child.visit_count
        priors[move] = child.prior
        mask[move] = True
    weights = counts if counts.any() else priors
    policy = normalize_policy(weights, mask)
    if temperature == 0:
        best_action = int(np.argmax(policy))
        policy[:] = 0
        policy[best_action] = 1.0
    elif temperature != 1:
        # Shift logs before division: small tau can underflow losing actions,
        # but the largest entry always stays exp(0), so mass never disappears.
        positive = policy > 0
        log_weights = np.log(policy[positive])
        with np.errstate(over="ignore", under="ignore"):
            policy[positive] = np.exp(
                (log_weights - log_weights.max()) / temperature
            )
        policy /= policy.sum()
    return policy


# #############################################################################
# PolicyValueNetwork
# #############################################################################


class PolicyValueNetwork(nn.Module):
    """
    Map current-player board encodings to policy logits and bounded values.

    Two shared ReLU layers feed a linear policy head and a tanh value head.
    This small CPU float32 MLP has no dropout or batch normalization, so
    predictions are identical in training and evaluation modes. Initialization
    uses a local seed without consuming the caller's CPU random stream.
    """

    def __init__(
        self, input_size: int, action_size: int, *, hidden_size: int, seed: int
    ) -> None:
        """
        Build a reproducible network with independent input and action sizes.

        :param input_size: flat board size, e.g., 9 or 42
        :param action_size: fixed action count, e.g., 9 or 7
        :param hidden_size: positive width of both shared hidden layers
        :param seed: CPU initialization seed
        """
        super().__init__()
        for size in (input_size, action_size, hidden_size):
            hdbg.dassert_isinstance(size, int, "Layer sizes must be integers")
            hdbg.dassert_lt(0, size, "Layer sizes must be positive")
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        # Restore the caller's CPU RNG state when initialization finishes.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            layer_options = {"device": "cpu", "dtype": torch.float32}
            self.trunk = nn.Sequential(
                nn.Linear(input_size, hidden_size, **layer_options),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size, **layer_options),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(
                hidden_size, action_size, **layer_options
            )
            self.value_head = nn.Linear(hidden_size, 1, **layer_options)

    def forward(
        self, encoded_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict raw action logits and values without applying game rules.

        :param encoded_states: finite CPU float32 tensor of shape
            `(input_size,)` or `(batch_size, input_size)`
        :return: logits of shape `(action_size,)` or `(batch_size, action_size)`
            and tanh values of shape `()` or `(batch_size,)`, respectively;
            autograd remains available for training
        """
        hdbg.dassert_in(encoded_states.ndim, (1, 2), "Use a board or a batch")
        hdbg.dassert_eq(
            encoded_states.shape[-1], self.input_size, "Wrong board size"
        )
        hdbg.dassert_eq(encoded_states.device.type, "cpu", "Use CPU inputs")
        hdbg.dassert_eq(encoded_states.dtype, torch.float32, "Use float32")
        hdbg.dassert(
            torch.isfinite(encoded_states).all().item(), "Inputs must be finite"
        )
        features = self.trunk(encoded_states)
        logits = self.policy_head(features)
        values = torch.tanh(self.value_head(features)).squeeze(-1)
        return logits, values


# #############################################################################
# NetworkEvaluator
# #############################################################################


class NetworkEvaluator:
    """
    Adapt a CPU network to the game's policy/value evaluator contract.

    Retain the model by reference, so later optimizer steps affect predictions.
    Inference builds no gradient graph and does not change model mode or grads.
    Terminal states bypass the network and use the exact game outcome.
    """

    def __init__(self, network: PolicyValueNetwork) -> None:
        """
        Wrap a network whose dimensions match the intended game.

        :param network: CPU float32 model with board and action dimensions
        """
        self.network = network

    def __call__(
        self, game: rimtsaazg.Game, state: rimtsaazg.State
    ) -> PolicyValuePrediction:
        """
        Encode a state, mask logits, and return legal priors and a value.

        Mask before softmax: even an enormous illegal logit must not erase
        the relative probabilities of legal actions through underflow.

        :param game: rules providing legal integer action indices
        :param state: reachable original state, not an encoded board
        :return: fresh legal policy and player-to-move value; terminal
            predictions contain zero policy and the exact outcome
        """
        action_size = self.network.action_size
        value = get_terminal_value(game, state)
        if value is not None:
            return PolicyValuePrediction(np.zeros(action_size), value)
        mask = get_legal_action_mask(game, state, action_size)
        hdbg.dassert(mask.any(), "An unfinished game needs a legal action")
        encoded = torch.from_numpy(encode_state(game, state))
        with torch.inference_mode():
            logits, predicted_value = self.network(encoded)
            hdbg.dassert(
                torch.isfinite(logits).all().item(), "Logits must be finite"
            )
            # Float64 softmax preserves differences between large finite logits.
            masked_logits = logits.to(torch.float64).masked_fill(
                ~torch.from_numpy(mask), -torch.inf
            )
            policy = torch.softmax(masked_logits, dim=-1).numpy()
            value = predicted_value.item()
        return PolicyValuePrediction(policy, value)


# #############################################################################
# Supervised minibatch learning
# #############################################################################


def train_batch(
    network: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    encoded_states: np.ndarray,
    target_policies: np.ndarray,
    target_values: np.ndarray,
    *,
    l2_coefficient: float,
) -> Dict[str, float]:
    """
    Take one optimizer step on fixed policy/value targets.

    The objective is mean policy cross-entropy plus mean squared value error
    plus `l2_coefficient * sum(parameter ** 2)`, including biases. Policy
    cross-entropy uses unmasked logits over every action; illegal actions
    should have zero target mass. This penalizes probability assigned to them.
    The function receives encodings, not rules, so callers ensure target
    legality. Terminal all-zero policies are not training distributions.

    :param network: CPU float32 model to update
    :param optimizer: optimizer over this model's parameters; reuse between
        calls to preserve momentum; set weight decay to zero because L2 is
        included explicitly in the objective
    :param encoded_states: nonempty array `(batch_size, input_size)` with
        current-player encodings; inputs are copied and never modified
    :param target_policies: finite nonnegative distributions of shape
        `(batch_size, action_size)`, each summing to one; soft targets allowed
    :param target_values: finite player-to-move labels `(batch_size,)` in
        `[-1, 1]`, not values from a fixed player's perspective
    :param l2_coefficient: finite nonnegative regularization coefficient
    :return: pre-update `loss`, `policy_loss`, `value_loss`, and weighted
        `l2_loss` as Python floats; gradients remain available for inspection
    """
    hdbg.dassert(
        np.isfinite(l2_coefficient) and l2_coefficient >= 0,
        "L2 coefficient must be finite and nonnegative",
    )
    # Copy at the boundary: caller arrays cannot be mutated by the optimizer.
    arrays = []
    for data in (encoded_states, target_policies, target_values):
        hdbg.dassert(np.isrealobj(data), "Training arrays must be real")
        copied = np.array(data, dtype=np.float32, copy=True)
        hdbg.dassert(
            np.isfinite(copied).all(), "Training arrays must be finite"
        )
        arrays.append(torch.from_numpy(copied))
    states, policies, values = arrays
    hdbg.dassert_eq(states.ndim, 2, "Training requires a batch of boards")
    batch_size = states.shape[0]
    hdbg.dassert_lt(0, batch_size, "Training batch must be nonempty")
    hdbg.dassert_eq(
        states.shape[1], network.input_size, "Wrong training board size"
    )
    hdbg.dassert_eq(
        tuple(policies.shape),
        (batch_size, network.action_size),
        "Policy targets must match the batch and action space",
    )
    hdbg.dassert_eq(
        tuple(values.shape), (batch_size,), "Values need one label per board"
    )
    hdbg.dassert((policies >= 0).all().item(), "Targets must be nonnegative")
    hdbg.dassert(
        torch.allclose(policies.sum(dim=1), torch.ones(batch_size), atol=1e-6),
        "Each target policy must sum to one",
    )
    hdbg.dassert((values.abs() <= 1).all().item(), "Values must lie in [-1, 1]")
    # Stable log-softmax handles soft distributions without taking log(0).
    logits, predicted_values = network(states)
    policy_loss = -(policies * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()
    value_loss = F.mse_loss(predicted_values, values)
    l2_loss = l2_coefficient * sum(
        parameter.square().sum() for parameter in network.parameters()
    )
    loss = policy_loss + value_loss + l2_loss
    hdbg.dassert(torch.isfinite(loss).item(), "Training loss must be finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Fail before updating if invalid gradients would corrupt the parameters.
    for parameter in network.parameters():
        hdbg.dassert_is_not(
            parameter.grad, None, "Every parameter needs a gradient"
        )
        hdbg.dassert(
            torch.isfinite(parameter.grad).all().item(),
            "Gradients must be finite",
        )
    optimizer.step()
    metrics = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "l2_loss": l2_loss.item(),
    }
    _LOG.debug("Training metrics='%s'", metrics)
    return metrics


# #############################################################################
# Self-play records and configuration
# #############################################################################


@dataclasses.dataclass
class TrainingExample:
    """
    Own one original state, its search policy, and its completed-game outcome.

    Policy entries retain fixed action coordinates. `value` is the winner
    times this state's player to move, hence exactly -1, 0, or +1. The state
    precedes its selected move and is nonterminal; callers supply reachable
    states and legal policies. Encode states only when assembling a minibatch.
    """

    state: rimtsaazg.State
    policy: np.ndarray
    value: float

    def __post_init__(self) -> None:
        """
        Copy the state and policy and validate the numerical training target.
        """
        prediction = PolicyValuePrediction(self.policy, self.value)
        hdbg.dassert_lt(0, prediction.policy.sum(), "Targets need policy mass")
        hdbg.dassert_in(
            prediction.value, (-1.0, 0.0, 1.0), "Use a completed-game outcome"
        )
        self.state = tuple(self.state)
        self.policy = prediction.policy
        self.value = prediction.value


@dataclasses.dataclass(frozen=True)
class SelfPlayConfig:
    """
    Hold explicit search, exploration, and termination settings for one game.

    `temperature` applies to the first `temperature_moves` plies (moves by
    either player); subsequent policies use temperature zero. Dirichlet noise
    applies to each new root throughout the game, independently of temperature.
    `max_moves` is a guard: an unfinished game at the limit raises instead of
    producing fabricated draw labels. No truncated examples are returned.
    """

    num_simulations: int
    temperature: float
    temperature_moves: int
    dirichlet_alpha: float
    root_noise_fraction: float
    max_moves: int
    exploration_constant: float = 1.0

    def __post_init__(self) -> None:
        """
        Reject invalid configurations before any search or RNG consumption.
        """
        for name in ("num_simulations", "temperature_moves", "max_moves"):
            value = getattr(self, name)
            hdbg.dassert_isinstance(
                value, int, "Move/budget counts must be integers"
            )
        hdbg.dassert_lt(
            0, self.num_simulations, "Self-play needs a positive search budget"
        )
        hdbg.dassert_lte(
            0, self.temperature_moves, "Temperature duration cannot be negative"
        )
        hdbg.dassert_lt(0, self.max_moves, "Move limit must be positive")
        hdbg.dassert(
            np.isfinite(self.temperature) and self.temperature >= 0,
            "Temperature must be finite and nonnegative",
        )
        hdbg.dassert(
            np.isfinite(self.exploration_constant)
            and self.exploration_constant >= 0,
            "Exploration must be finite and nonnegative",
        )
        _validate_root_noise(self.dirichlet_alpha, self.root_noise_fraction)


@dataclasses.dataclass
class SelfPlayResult:
    """
    Return aligned examples and moves plus the final original state and winner.

    `examples[i].state` is the position before `moves[i]`. The terminal state
    is recorded separately and never used as a policy training example.
    """

    examples: List[TrainingExample]
    moves: List[rimtsaazg.Move]
    final_state: rimtsaazg.State
    winner: int


# #############################################################################
# Self-play collection
# #############################################################################


def play_self_play_game(
    game: rimtsaazg.Game,
    evaluator: PolicyValueEvaluator,
    config: SelfPlayConfig,
    rng: np.random.Generator,
    *,
    action_size: int,
) -> SelfPlayResult:
    """
    Collect a complete game using one fixed evaluator for both players.

    Start from the game's initial state and build a fresh PUCT tree each ply.
    Store the temperature-adjusted visit policy used to sample that move.
    Only after terminality assign outcomes from each stored state's player
    perspective. The collector does not train, alter evaluator parameters,
    use global random state, or retain trees across moves. Caller evaluators
    should be deterministic and side-effect free for seeded reproducibility.

    :param game: terminating, strictly alternating two-player zero-sum rules
    :param evaluator: fixed callable supplying priors and player-to-move values
    :param config: validated search, sampling, and maximum game length settings
    :param rng: caller-owned generator shared by root noise and move sampling
    :param action_size: fixed positive action count, independent of board size
    :return: aligned states, policies, final labels, played moves, and outcome;
        an initially terminal game returns no examples or moves
    """
    hdbg.dassert_isinstance(rng, np.random.Generator, "Supply an explicit RNG")
    hdbg.dassert_isinstance(action_size, int, "Action count must be an integer")
    hdbg.dassert_lt(0, action_size, "Action space must be nonempty")
    state = game.get_initial_state()
    pending = []
    moves = []
    while not game.is_terminal(state):
        hdbg.dassert_lt(
            len(moves),
            config.max_moves,
            "Game exceeded max_moves before reaching a terminal outcome",
        )
        root = build_search_tree(
            game,
            state,
            evaluator,
            action_size=action_size,
            num_simulations=config.num_simulations,
            exploration_constant=config.exploration_constant,
            root_noise_fraction=config.root_noise_fraction,
            dirichlet_alpha=config.dirichlet_alpha,
            rng=rng,
        )
        temperature = (
            config.temperature if len(moves) < config.temperature_moves else 0.0
        )
        policy = get_visit_policy(root, action_size, temperature=temperature)
        # Greedy moves consume no random numbers; positive tau samples pi.
        move = (
            int(rng.choice(action_size, p=policy))
            if temperature > 0
            else int(np.argmax(policy))
        )
        pending.append((state, policy, game.get_current_player(state)))
        moves.append(move)
        state = game.apply_move(state, move)
    winner = game.get_winner(state)
    hdbg.dassert_in(winner, (-1, 0, 1), "Winner must identify a player or draw")
    examples = [
        TrainingExample(board, policy, float(winner * player))
        for board, policy, player in pending
    ]
    result = SelfPlayResult(examples, moves, state, winner)
    _LOG.debug("Collected moves='%s', winner='%s'", len(moves), winner)
    return result


# #############################################################################
# ReplayBuffer
# #############################################################################


class ReplayBuffer:
    """
    Keep the most recent completed-game examples in bounded FIFO storage.

    Insertions and reads copy records so caller mutations cannot alter replay.
    Sampling is uniform with replacement: a full minibatch is available even
    when the buffer holds fewer records than the requested batch size. Records
    must come from the same game and use matching board and action dimensions.
    """

    def __init__(self, capacity: int) -> None:
        """
        Allocate empty replay storage with a fixed positive record limit.

        :param capacity: maximum number of positions, not number of games
        """
        hdbg.dassert_isinstance(capacity, int, "Capacity must be an integer")
        hdbg.dassert_lt(0, capacity, "Replay capacity must be positive")
        self._examples: Deque[TrainingExample] = collections.deque(
            maxlen=capacity
        )

    @property
    def capacity(self) -> int:
        """
        Return the configured maximum record count.
        """
        return self._examples.maxlen

    def __len__(self) -> int:
        """
        Return the current number of retained positions.
        """
        return len(self._examples)

    def add(self, example: TrainingExample) -> None:
        """
        Copy a record into replay, evicting the oldest if storage is full.

        :param example: completed-game target from the same game as prior data
        """
        owned = TrainingExample(example.state, example.policy, example.value)
        if self._examples:
            oldest = self._examples[0]
            hdbg.dassert_eq(
                len(owned.state),
                len(oldest.state),
                "Replay board sizes must match",
            )
            hdbg.dassert_eq(
                owned.policy.shape,
                oldest.policy.shape,
                "Replay action sizes must match",
            )
        self._examples.append(owned)

    def get_examples(self) -> List[TrainingExample]:
        """
        Return independent records in oldest-to-newest order for inspection.
        """
        return [
            TrainingExample(e.state, e.policy, e.value) for e in self._examples
        ]

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> List[TrainingExample]:
        """
        Draw independent record copies uniformly with replacement.

        :param batch_size: positive number of examples to return
        :param rng: explicit caller-owned generator, advanced by sampling
        :return: copied examples in sampled order; duplicates are allowed
        """
        hdbg.dassert_isinstance(
            batch_size, int, "Batch size must be an integer"
        )
        hdbg.dassert_lt(0, batch_size, "Batch size must be positive")
        hdbg.dassert_lt(0, len(self), "Cannot sample empty replay")
        hdbg.dassert_isinstance(
            rng, np.random.Generator, "Supply an explicit RNG"
        )
        examples = list(self._examples)
        indices = rng.choice(len(examples), size=batch_size, replace=True)
        return [
            TrainingExample(
                examples[i].state, examples[i].policy, examples[i].value
            )
            for i in indices
        ]


# #############################################################################
# Training configuration and metrics
# #############################################################################


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """
    Set explicit CPU training budgets and the self-play configuration.

    Each iteration collects `games_per_iteration` complete games with fixed
    parameters, adds their positions to replay, then takes
    `updates_per_iteration` minibatch steps. L2 is supplied to `train_batch()`;
    the optimizer must use zero weight decay to avoid double regularization.
    """

    iterations: int
    games_per_iteration: int
    updates_per_iteration: int
    batch_size: int
    l2_coefficient: float
    self_play: SelfPlayConfig

    def __post_init__(self) -> None:
        """
        Reject invalid budgets and loss settings before training begins.
        """
        for name in (
            "iterations",
            "games_per_iteration",
            "updates_per_iteration",
            "batch_size",
        ):
            value = getattr(self, name)
            hdbg.dassert_isinstance(
                value, int, "Training counts must be integers"
            )
            hdbg.dassert_lt(0, value, "Training counts must be positive")
        hdbg.dassert(
            np.isfinite(self.l2_coefficient) and self.l2_coefficient >= 0,
            "L2 coefficient must be finite and nonnegative",
        )
        hdbg.dassert_isinstance(
            self.self_play, SelfPlayConfig, "Supply self-play settings"
        )


@dataclasses.dataclass(frozen=True)
class TrainingMetrics:
    """
    Summarize one completed collection/update iteration.

    Loss fields are means of pre-update minibatch losses within the iteration,
    not a fixed validation-set loss or a measure of playing strength.
    `generated_examples` counts positions before any replay eviction.
    Iteration numbering starts at one for each call to `train_alphazero()`.
    """

    iteration: int
    games: int
    generated_examples: int
    replay_size: int
    updates: int
    loss: float
    policy_loss: float
    value_loss: float
    l2_loss: float


# #############################################################################
# Training loop
# #############################################################################


def train_alphazero(
    game: rimtsaazg.Game,
    network: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: TrainingConfig,
    rng: np.random.Generator,
    *,
    show_progress: bool = True,
) -> List[TrainingMetrics]:
    """
    Alternate complete self-play games and replay updates on one CPU network.

    The supplied model, optimizer, replay, and RNG are updated in place. Reuse
    all four across calls to continue an in-memory run, preserving optimizer
    moments, old examples, and the random stream. Replay sampling uses the
    same generator as self-play; model initialization has its own explicit seed.
    There is no arena selection or separate opponent network.

    :param game: the same terminating two-player game for all replay records
    :param network: CPU float32 policy/value model matching the game dimensions
    :param optimizer: optimizer owning exactly this network's parameters,
        with weight decay zero; its state persists across all minibatches
    :param replay: bounded storage, optionally containing earlier game examples
    :param config: explicit collection and update budgets
    :param rng: explicit generator for root noise, actions, and replay sampling
    :param show_progress: display iteration progress; disable in automated tests
    :return: one metric record per completed iteration; training errors propagate
        and previously completed games or optimizer steps are not rolled back
    """
    hdbg.dassert_isinstance(rng, np.random.Generator, "Supply an explicit RNG")
    initial_state = game.get_initial_state()
    hdbg.dassert(
        not game.is_terminal(initial_state),
        "Training needs a nonterminal initial state",
    )
    hdbg.dassert_eq(
        len(initial_state), network.input_size, "Network must match board size"
    )
    parameters = list(network.parameters())
    optimizer_parameters = [
        p for group in optimizer.param_groups for p in group["params"]
    ]
    hdbg.dassert_eq(
        len(optimizer_parameters),
        len(parameters),
        "Optimizer must own all model parameters exactly once",
    )
    hdbg.dassert_eq(
        {id(p) for p in optimizer_parameters},
        {id(p) for p in parameters},
        "Optimizer belongs to a different model",
    )
    for group in optimizer.param_groups:
        hdbg.dassert_eq(
            group.get("weight_decay", 0),
            0,
            "Use explicit L2, not optimizer weight decay",
        )
    if len(replay):
        example = replay.get_examples()[0]
        hdbg.dassert_eq(
            len(example.state),
            network.input_size,
            "Replay board size must match model",
        )
        hdbg.dassert_eq(
            example.policy.shape,
            (network.action_size,),
            "Replay action size must match model",
        )
    evaluator = NetworkEvaluator(network)
    history = []
    for iteration in trange(
        config.iterations, desc="AlphaZero training", disable=not show_progress
    ):
        generated_examples = 0
        # Keep weights fixed throughout collection, then update from replay.
        for _ in range(config.games_per_iteration):
            result = play_self_play_game(
                game,
                evaluator,
                config.self_play,
                rng,
                action_size=network.action_size,
            )
            for example in result.examples:
                replay.add(example)
            generated_examples += len(result.examples)
        losses = []
        for _ in range(config.updates_per_iteration):
            batch = replay.sample(config.batch_size, rng)
            inputs = np.stack([encode_state(game, e.state) for e in batch])
            policies = np.stack([e.policy for e in batch])
            values = np.array([e.value for e in batch], dtype=np.float32)
            metrics = train_batch(
                network,
                optimizer,
                inputs,
                policies,
                values,
                l2_coefficient=config.l2_coefficient,
            )
            losses.append(metrics)
        mean_losses = {
            name: float(np.mean([m[name] for m in losses]))
            for name in losses[0]
        }
        summary = TrainingMetrics(
            iteration=iteration + 1,
            games=config.games_per_iteration,
            generated_examples=generated_examples,
            replay_size=len(replay),
            updates=config.updates_per_iteration,
            **mean_losses,
        )
        history.append(summary)
        _LOG.debug("Training iteration='%s'", summary)
    return history


# #############################################################################
# Network checkpoints
# #############################################################################


def save_checkpoint(network: PolicyValueNetwork, path: str) -> None:
    """
    Atomically save network dimensions and CPU weights for later inference.

    The versioned payload contains tensors and primitive metadata only. This
    is a model checkpoint: optimizer state, replay data, RNG state, gradients,
    and training progress are not stored. Save between optimizer steps.

    :param network: finite CPU float32 policy/value network to snapshot
    :param path: destination file in an existing directory; replace it only
        after serialization succeeds
    """
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in network.state_dict().items()
    }
    hdbg.dassert(
        all(torch.isfinite(t).all().item() for t in state_dict.values()),
        "Checkpoint parameters must be finite",
    )
    payload = {
        "format_version": 1,
        "architecture": {
            "input_size": network.input_size,
            "action_size": network.action_size,
            "hidden_size": network.hidden_size,
        },
        "state_dict": state_dict,
    }
    directory = os.path.dirname(os.path.abspath(path))
    # A sibling temporary file makes replacement atomic on the same filesystem.
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=".alphazero-", suffix=".pt", delete=False
        ) as stream:
            temporary_path = stream.name
            torch.save(payload, stream)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_checkpoint(path: str) -> PolicyValueNetwork:
    """
    Recreate a CPU model from a versioned network checkpoint.

    Load tensors with `weights_only=True` and CPU mapping, then require an
    exact state-dictionary match. The returned independent model is in eval
    mode with no gradients. Optimizer/replay/RNG state is not restored, so this
    is not an exact training-resume API.

    :param path: file created by `save_checkpoint()`
    :return: CPU float32 network with the saved dimensions and parameters
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    hdbg.dassert_eq(
        payload["format_version"], 1, "Unsupported checkpoint version"
    )
    architecture = payload["architecture"]
    network = PolicyValueNetwork(
        architecture["input_size"],
        architecture["action_size"],
        hidden_size=architecture["hidden_size"],
        seed=0,
    )
    network.load_state_dict(payload["state_dict"], strict=True)
    hdbg.dassert(
        all(torch.isfinite(p).all().item() for p in network.parameters()),
        "Checkpoint parameters must be finite",
    )
    network.eval()
    return network


# #############################################################################
# Evaluation agents
# #############################################################################


EvaluationAgent = Callable[
    [rimtsaazg.Game, rimtsaazg.State, np.random.Generator], rimtsaazg.Move
]


def _call_legacy_player(
    player: Callable[[rimtsaazg.Game, rimtsaazg.State], rimtsaazg.Move],
    game: rimtsaazg.Game,
    state: rimtsaazg.State,
    rng: np.random.Generator,
) -> rimtsaazg.Move:
    """
    Seed a reference player and restore its Python global RNG after the call.

    Reference random/MCTS players use Python's module-level RNG. Isolate their
    calls without changing their algorithms. This adapter is for serial use;
    it must not run concurrently with other Python global-RNG consumers.
    """
    previous_state = random.getstate()
    try:
        random.seed(int(rng.integers(0, 2**32)))
        return player(game, state)
    finally:
        random.setstate(previous_state)


def make_random_agent() -> EvaluationAgent:
    """
    Adapt the reference uniform-random legal player to explicit seeded use.

    :return: serial evaluation agent consuming only its supplied random stream
    """

    def agent(
        game: rimtsaazg.Game, state: rimtsaazg.State, rng: np.random.Generator
    ) -> rimtsaazg.Move:
        return _call_legacy_player(rimtsaazmu.random_player, game, state, rng)

    return agent


def make_mcts_agent(num_simulations: int) -> EvaluationAgent:
    """
    Adapt the original rollout MCTS at an explicit per-move simulation budget.

    Keep its UCT constant sqrt(2), incoming-player values, random expansions,
    rollouts, and most-visited move selection unchanged. Calls are serial.

    :param num_simulations: positive number of reference MCTS simulations
    :return: agent with seeded random expansion/rollout choices
    """
    hdbg.dassert_isinstance(num_simulations, int, "Budget must be an integer")
    hdbg.dassert_lt(0, num_simulations, "MCTS needs a positive budget")
    player = rimtsaazmu.make_mcts_player(num_simulations=num_simulations)

    def agent(
        game: rimtsaazg.Game, state: rimtsaazg.State, rng: np.random.Generator
    ) -> rimtsaazg.Move:
        return _call_legacy_player(player, game, state, rng)

    return agent


def make_minimax_agent() -> EvaluationAgent:
    """
    Adapt exact minimax using the reference alpha-beta implementation.

    Cache selected actions by game object and state to reuse solved positions.
    This is suitable for Tic-Tac-Toe; exhaustive search is not a practical
    general baseline for larger games. Game rules must not change while cached.

    :return: deterministic exact-play agent that consumes no random numbers
    """
    solve = functools.lru_cache(maxsize=10000)(rimtsaazsau.run_alpha_beta)

    def agent(
        game: rimtsaazg.Game, state: rimtsaazg.State, rng: np.random.Generator
    ) -> rimtsaazg.Move:
        return solve(game, state)

    return agent


def make_policy_agent(evaluator: PolicyValueEvaluator) -> EvaluationAgent:
    """
    Select the highest-probability legal action without search or sampling.

    :param evaluator: fixed deterministic policy/value evaluator; value is unused
    :return: greedy policy agent; equal probabilities choose the lowest index
    """

    def agent(
        game: rimtsaazg.Game, state: rimtsaazg.State, rng: np.random.Generator
    ) -> rimtsaazg.Move:
        hdbg.dassert(
            not game.is_terminal(state), "Cannot choose a terminal move"
        )
        prediction = evaluator(game, state)
        mask = get_legal_action_mask(game, state, len(prediction.policy))
        hdbg.dassert(mask.any(), "A move requires a legal action")
        policy = normalize_policy(prediction.policy, mask)
        return int(np.argmax(policy))

    return agent


def make_search_agent(
    evaluator: PolicyValueEvaluator,
    *,
    action_size: int,
    num_simulations: int,
    exploration_constant: float = 1.0,
) -> EvaluationAgent:
    """
    Build a greedy PUCT evaluation agent with root noise disabled.

    :param evaluator: fixed deterministic evaluator, either uniform or learned
    :param action_size: positive fixed action count
    :param num_simulations: positive per-move simulation budget
    :param exploration_constant: nonnegative finite PUCT weight
    :return: deterministic agent selecting most-visited actions, lowest-index ties
    """
    hdbg.dassert_isinstance(action_size, int, "Action count must be an integer")
    hdbg.dassert_lt(0, action_size, "Action count must be positive")
    hdbg.dassert_isinstance(num_simulations, int, "Budget must be an integer")
    hdbg.dassert_lt(0, num_simulations, "Search needs a positive budget")
    hdbg.dassert(
        np.isfinite(exploration_constant) and exploration_constant >= 0,
        "Exploration must be finite and nonnegative",
    )

    def agent(
        game: rimtsaazg.Game, state: rimtsaazg.State, rng: np.random.Generator
    ) -> rimtsaazg.Move:
        root = build_search_tree(
            game,
            state,
            evaluator,
            action_size=action_size,
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            root_noise_fraction=0.0,
        )
        policy = get_visit_policy(root, action_size, temperature=0.0)
        return int(np.argmax(policy))

    return agent


# #############################################################################
# Evaluation records and paired-seat matches
# #############################################################################


@dataclasses.dataclass(frozen=True)
class EvaluationGame:
    """
    Record one complete game from the evaluated agent's perspective.

    `agent_player` is +1 for X or -1 for O; `winner` uses game coordinates.
    `outcome = winner * agent_player` is +1 for a win, 0 for a draw, -1 for
    a loss. Moves start from the game's initial state; no training occurs.
    """

    seed: int
    agent_player: int
    winner: int
    outcome: int
    moves: Tuple[rimtsaazg.Move, ...]
    final_state: rimtsaazg.State


@dataclasses.dataclass
class EvaluationResult:
    """
    Hold per-game evidence and summarize agent-relative wins, draws, and losses.
    """

    games: List[EvaluationGame]

    def summary(
        self, *, agent_player: Optional[int] = None
    ) -> Dict[str, Union[int, float]]:
        """
        Count outcomes and score draws as one-half, optionally for one seat.

        :param agent_player: +1 for X, -1 for O, or None to combine both seats
        :return: games, wins, draws, losses, and score_rate = (wins + draws/2)/games
        """
        hdbg.dassert_in(
            agent_player, (None, 1, -1), "Choose X, O, or both seats"
        )
        selected = [
            g
            for g in self.games
            if agent_player is None or g.agent_player == agent_player
        ]
        hdbg.dassert(selected, "No completed games for this summary")
        wins = sum(g.outcome == 1 for g in selected)
        draws = sum(g.outcome == 0 for g in selected)
        losses = sum(g.outcome == -1 for g in selected)
        return {
            "games": len(selected),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score_rate": (wins + 0.5 * draws) / len(selected),
        }


def evaluate_agent(
    game: rimtsaazg.Game,
    agent: EvaluationAgent,
    opponent: EvaluationAgent,
    *,
    seeds: Sequence[int],
    max_moves: int,
) -> EvaluationResult:
    """
    Play two complete games per seed, evaluating the agent once in each seat.

    For each seed, play as X then O. Independent per-role streams derive from
    SeedSequence([seed, seat_index]). Matching seed/seat pairs across agents
    receive the same opponent stream; an agent's random consumption cannot
    advance its opponent's stream. Seeds affect stochastic agents only: repeated
    deterministic matches can be identical and are not independent evidence.

    Agents must be fixed during evaluation. This runner does not update models,
    add self-play noise, sample training targets, or alter global RNGs. Use the
    supplied factories for noise-free greedy network/PUCT evaluation. Legacy
    random/MCTS adapters temporarily isolate Python's RNG and require serial use.

    :param game: terminating two-player zero-sum rules with players +1 and -1
    :param agent: agent to measure, accepting (game, state, its_rng)
    :param opponent: fixed reference agent with the same signature
    :param seeds: nonempty sequence of distinct nonnegative integer seeds
    :param max_moves: positive game-length cap; unfinished games raise, not draw
    :return: complete moves/outcomes with summaries available overall and by seat
    """
    hdbg.dassert_isinstance(max_moves, int, "Move limit must be an integer")
    hdbg.dassert_lt(0, max_moves, "Move limit must be positive")
    seeds = list(seeds)
    hdbg.dassert(seeds, "Evaluation requires at least one seed")
    for seed in seeds:
        hdbg.dassert_isinstance(
            seed, numbers.Integral, "Seeds must be integers"
        )
        hdbg.dassert_lte(0, seed, "Seeds must be nonnegative")
    hdbg.dassert_eq(
        len(set(seeds)), len(seeds), "Use distinct evaluation seeds"
    )
    hdbg.dassert(
        not game.is_terminal(game.get_initial_state()),
        "Evaluation needs a nonterminal initial state",
    )
    records = []
    for seed in seeds:
        for seat_index, agent_player in enumerate((1, -1)):
            streams = np.random.SeedSequence([int(seed), seat_index]).spawn(2)
            agent_rng, opponent_rng = [
                np.random.default_rng(s) for s in streams
            ]
            state = game.get_initial_state()
            moves = []
            while not game.is_terminal(state):
                hdbg.dassert_lt(
                    len(moves),
                    max_moves,
                    "Evaluation exceeded max_moves before terminality",
                )
                player = game.get_current_player(state)
                hdbg.dassert_in(
                    player, (-1, 1), "Game must use players +1 and -1"
                )
                actor, rng = (
                    (agent, agent_rng)
                    if player == agent_player
                    else (opponent, opponent_rng)
                )
                move = actor(game, state, rng)
                hdbg.dassert_isinstance(
                    move,
                    numbers.Integral,
                    "Agent must return an integer action",
                )
                hdbg.dassert_in(
                    move,
                    game.get_legal_moves(state),
                    "Agent returned an illegal move",
                )
                moves.append(int(move))
                state = game.apply_move(state, int(move))
            winner = game.get_winner(state)
            hdbg.dassert_in(
                winner, (-1, 0, 1), "Winner must identify a player or draw"
            )
            records.append(
                EvaluationGame(
                    int(seed),
                    agent_player,
                    winner,
                    winner * agent_player,
                    tuple(moves),
                    state,
                )
            )
    return EvaluationResult(records)
