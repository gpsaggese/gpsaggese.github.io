"""
`Game`: the game-agnostic interface shared by every search algorithm in this
project.

Any two-player, zero-sum, perfect-information game that implements this
6-method contract can be searched (without touching search implementation) by:
- `search_algorithms_utils.py` (minimax, alpha-beta pruning, depth-limited
  search)
- `mcts_utils.py` (flat Monte Carlo, MCTS)

Import as:

import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game as rimtsaazg
"""

import abc
from typing import List, Tuple

# A game state is a flat tuple of cell values and a move is an integer
# identifying the action (e.g., a cell index or a column); see
# `game_examples.py` for concrete `Game` implementations.
State = Tuple[int, ...]
Move = int


# #############################################################################
# Game
# #############################################################################


class Game(abc.ABC):
    """
    Game-agnostic interface for a two-player, zero-sum, perfect-information
    game usable by any search algorithm in this project.

    - A concrete game (e.g., `TicTacToe`, `ConnectFour`) implements these six
      methods
        - The search code only depends on this interface, so a new game can be
          plugged in without touching the search logic
    - Players are represented as `1` and `-1`; a draw is `0`
    """

    @abc.abstractmethod
    def get_initial_state(self) -> State:
        """
        Build the starting state of a new game.

        :return: initial game state
        """

    @abc.abstractmethod
    def get_legal_moves(self, state: State) -> List[Move]:
        """
        List the moves available from `state`.

        :param state: game state to query
        :return: legal moves, empty if `state` is terminal
        """

    @abc.abstractmethod
    def apply_move(self, state: State, move: Move) -> State:
        """
        Apply `move` to `state` and return the resulting state.

        :param state: game state before the move
        :param move: move to apply, must be legal in `state`
        :return: new game state after the move
        """

    @abc.abstractmethod
    def is_terminal(self, state: State) -> bool:
        """
        Check whether `state` ends the game.

        :param state: game state to query
        :return: True if no more moves can be made from `state`
        """

    @abc.abstractmethod
    def get_winner(self, state: State) -> int:
        """
        Determine the outcome of a terminal state.

        :param state: terminal game state
        :return: `1` or `-1` for the winning player, `0` for a draw
        """

    @abc.abstractmethod
    def get_current_player(self, state: State) -> int:
        """
        Report whose turn it is to move in `state`.

        :param state: game state to query
        :return: `1` or `-1`
        """

    @abc.abstractmethod
    def render(self, state: State) -> str:
        """
        Render `state` as a human-readable board.

        :param state: game state to render
        :return: printable board representation
        """
