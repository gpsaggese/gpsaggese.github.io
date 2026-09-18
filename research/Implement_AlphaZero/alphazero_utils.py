"""
Represent board games as inputs and outcomes for AlphaZero.

States remain the immutable tuples owned by the game implementations. Encodings
are model inputs only: never pass an encoded board back to the game rules.
Callers supply reachable states and legal moves from the existing game API.

Import as:

import research.Implement_AlphaZero.alphazero_utils as rialzut
"""

import logging
from typing import Optional

import numpy as np

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
