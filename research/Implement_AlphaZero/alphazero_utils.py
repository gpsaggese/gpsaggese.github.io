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

Import as:

import research.Implement_AlphaZero.alphazero_utils as rialzut
"""

import dataclasses
import logging
import math
import numbers
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import helpers.hdbg as hdbg
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game as rimtsaazg

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


def build_search_tree(
    game: rimtsaazg.Game,
    state: rimtsaazg.State,
    evaluator: PolicyValueEvaluator,
    *,
    action_size: int,
    num_simulations: int,
    exploration_constant: float = 1.0,
) -> AlphaZeroNode:
    """
    Build a fresh PUCT tree using an explicitly supplied policy/value evaluator.

    Expand the nonterminal root once before the counted simulations. Its
    initial value estimate is not backed up. Each simulation descends by PUCT
    to a leaf, obtains an exact terminal value or an evaluator prediction,
    and updates every node on its path. Thus root visits and the sum of root
    child visits both equal `num_simulations`. Equal scores choose the lowest
    action index. No random rollouts, root noise, or tree reuse are performed.

    :param game: strictly alternating two-player, zero-sum, deterministic rules
    :param state: reachable nonterminal state in its original representation
    :param evaluator: callable returning nonterminal action priors and a
        player-to-move value; terminal leaves never call it
    :param action_size: fixed positive action count, independent of board size
    :param num_simulations: nonnegative integer simulation budget; zero expands
        only the root, allowing inspection of priors
    :param exploration_constant: finite nonnegative exploration weight
        - Default: `1.0`
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
    root = AlphaZeroNode(state, 1.0)
    # Validate exploration before any evaluator call, including with zero budget.
    get_puct_scores(root, exploration_constant)
    _evaluate_search_leaf(root, game, evaluator, action_size)
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


def get_visit_policy(root: AlphaZeroNode, action_size: int) -> np.ndarray:
    """
    Convert root child visit counts into a fixed-size action distribution.

    Visited roots return `N(s, a) / sum_b N(s, b)`. With no simulations, use
    the root's legal priors instead. Action selection can use `argmax` of the
    result; NumPy breaks equal probabilities by the lowest action index.

    :param root: expanded nonterminal root returned by `build_search_tree()`
    :param action_size: fixed positive action count used to build the tree
    :return: fresh `float64` policy with zero mass on illegal actions
    """
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
