"""
Concrete `Game` implementations: tic-tac-toe, Connect Four, and Connect
Three.

All three games plug into any search algorithm in this project (minimax,
alpha-beta pruning, depth-limited search, flat Monte Carlo, MCTS) by
implementing the `Game` interface from `game.py`. `ConnectThree` is
`ConnectFour`'s gravity-drop rule on a 3x3 board (3-in-a-row instead of
4-in-a-row), small enough for a full search tree to render, used by
`search_algorithms.02.API.ipynb`.

See `README.md` for a description of every file in this directory.

Import as:

import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game_examples as rimtsaazge
"""

import logging
from typing import List, Optional, Tuple, cast

import helpers.hdbg as hdbg
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game as rimtsaazg

_LOG = logging.getLogger(__name__)

# TODO(gp): Consider is_terminal -> is_goal to align to the slides

# #############################################################################
# Shared board helpers
# #############################################################################

# Cell values shared by every game in this module: `0` empty, `1` player X
# (moves first), `-1` player O.
_EMPTY = 0
_PLAYER_X = 1
_PLAYER_O = -1

_CELL_SYMBOLS = {_EMPTY: ".", _PLAYER_X: "X", _PLAYER_O: "O"}


# TODO(gp): This is inefficient since we could keep this info in the state,
# instead of recomputing it all the times.
def _infer_current_player(state: rimtsaazg.State) -> int:
    """
    Infer whose turn it is from how many cells are filled.

    Works for any board size, since players strictly alternate starting with
    `1` (X).

    :param state: game state to query
    :return: `1` (X) if an even number of moves have been played, `-1` (O)
        otherwise
    """
    num_moves_played = sum(1 for cell in state if cell != _EMPTY)
    current_player = _PLAYER_X if num_moves_played % 2 == 0 else _PLAYER_O
    return current_player


def _render_board(state: rimtsaazg.State, num_cols: int) -> str:
    """
    Render `state` as rows of space-separated symbols.

    :param state: game state to render
    :param num_cols: number of columns per row
    :return: multi-line string, one row per line
    """
    rows = [
        " ".join(_CELL_SYMBOLS[cell] for cell in state[row : row + num_cols])
        for row in range(0, len(state), num_cols)
    ]
    board_str = "\n".join(rows)
    return board_str


def _get_line_winner(
    state: rimtsaazg.State, win_lines: List[Tuple[int, ...]]
) -> int:
    """
    Check `win_lines` for a completed line of same-player marks.

    :param state: game state to check
    :param win_lines: index tuples, each a candidate winning line
    :return: `1` or `-1` if some line in `win_lines` is filled by that
        player, else `0`
    """
    winner = _EMPTY
    for line in win_lines:
        first_cell = state[line[0]]
        if first_cell != _EMPTY and all(state[i] == first_cell for i in line):
            winner = first_cell
            break
    return winner


# #############################################################################
# TicTacToe
# #############################################################################


# TODO(ai_gp): Inline
_TICTACTOE_NUM_COLS = 3

# TODO(ai_gp): Move inside the class.
# The 8 index triples that constitute a win: 3 rows, 3 columns, 2 diagonals.
_TICTACTOE_WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


# #############################################################################
# TicTacToe
# #############################################################################


class TicTacToe(rimtsaazg.Game):
    """
    3x3 tic-tac-toe.

    - A state is a length-9 tuple of cell values (`0` empty, `1` X, `-1` O)
      indexed row-major from the top-left corner
    - Player `1` (X) moves first
    """

    def get_initial_state(self) -> rimtsaazg.State:
        """
        Build an empty 3x3 board.

        :return: length-9 tuple of `0`s
        """
        state = (_EMPTY,) * 9
        return state

    def get_legal_moves(self, state: rimtsaazg.State) -> List[rimtsaazg.Move]:
        """
        List the empty cells, if any.

        :param state: game state to query
        :return: indices of empty cells, empty if `state` is terminal
        """
        if self.is_terminal(state):
            moves: List[rimtsaazg.Move] = []
        else:
            moves = [i for i, cell in enumerate(state) if cell == _EMPTY]
        return moves

    def apply_move(
        self, state: rimtsaazg.State, move: rimtsaazg.Move
    ) -> rimtsaazg.State:
        """
        Place the current player's mark at `move`.

        :param state: game state before the move
        :param move: index of the empty cell to play
        :return: new state with the current player's mark at `move`
        """
        hdbg.dassert_eq(
            state[move], _EMPTY, "Move %s targets an occupied cell", move
        )
        player = self.get_current_player(state)
        new_state = state[:move] + (player,) + state[move + 1 :]
        return new_state

    def is_terminal(self, state: rimtsaazg.State) -> bool:
        """
        Check whether the board has a winner or is full.

        :param state: game state to query
        :return: True if the board is a win for either player or a draw
        """
        is_over = (
            _get_line_winner(state, _TICTACTOE_WIN_LINES) != _EMPTY
            or _EMPTY not in state
        )
        return is_over

    def get_winner(self, state: rimtsaazg.State) -> int:
        """
        Determine the outcome of a finished board.

        :param state: terminal game state
        :return: `1` or `-1` for three-in-a-row, `0` for a full board with
            no winner
        """
        hdbg.dassert(
            self.is_terminal(state), "get_winner() requires a terminal state"
        )
        winner = _get_line_winner(state, _TICTACTOE_WIN_LINES)
        return winner

    def get_current_player(self, state: rimtsaazg.State) -> int:
        """
        Infer whose turn it is from how many cells are filled.

        :param state: game state to query
        :return: `1` (X) if an even number of moves have been played, `-1`
            (O) otherwise
        """
        current_player = _infer_current_player(state)
        return current_player

    def render(self, state: rimtsaazg.State) -> str:
        """
        Render the board as 3 rows of space-separated symbols.

        :param state: game state to render
        :return: 3-line string, e.g.:
            ```
            X . O
            . X .
            O . X
            ```
        """
        board_str = _render_board(state, _TICTACTOE_NUM_COLS)
        return board_str


def evaluate_tic_tac_toe(state: rimtsaazg.State) -> float:
    """
    Heuristic evaluation of a (possibly non-terminal) tic-tac-toe state.

    Used by `search_algorithms_utils.build_depth_limited_tree()` to score a
    cut node once the search depth limit is reached before a terminal state.

    The heuristic is a local, cheap feature:
    - counts, over the 8 win lines, how many are still "open" (contain no mark
      from the opponent) for each player
    - returns the normalized difference
    It is also an imperfect heuristic since it scores each line independently,
    so it cannot see a fork (two lines that share an empty cell and cannot both
    be blocked), and can therefore rank a winning position no higher than a
    merely promising one.

    :param state: game state to evaluate, need not be terminal
    :return: heuristic score in `[-1, 1]` from X's perspective, on the same
        scale as the exact `{-1, 0, 1}` values `get_winner()` returns
    """
    x_open = sum(
        1
        for line in _TICTACTOE_WIN_LINES
        if all(state[i] != _PLAYER_O for i in line)
    )
    o_open = sum(
        1
        for line in _TICTACTOE_WIN_LINES
        if all(state[i] != _PLAYER_X for i in line)
    )
    score = (x_open - o_open) / len(_TICTACTOE_WIN_LINES)
    return score


# #############################################################################
# ConnectFour
# #############################################################################


# TODO(ai_gp): Inline or move inside the class.
_CONNECT_FOUR_NUM_ROWS = 6
_CONNECT_FOUR_NUM_COLS = 7
_CONNECT_FOUR_LINE_LENGTH = 4


# TODO(ai_gp): Make it static inside ConnectFour
def _build_win_lines(
    num_rows: int, num_cols: int, line_length: int
) -> List[Tuple[int, ...]]:
    """
    Enumerate every `line_length`-in-a-row window on a `num_rows` x
    `num_cols` board.

    Shared by `ConnectFour` (6x7, 4-in-a-row) and `ConnectThree` (3x3,
    3-in-a-row): covers all horizontal, vertical, and both diagonal
    directions. State is indexed row-major with row `0` at the top and row
    `num_rows - 1` at the bottom.

    :param num_rows: number of rows on the board
    :param num_cols: number of columns on the board
    :param line_length: number of same-player marks in a row that wins
    :return: index tuples, each a candidate winning line of length
        `line_length`
    """

    def idx(row: int, col: int) -> int:
        return row * num_cols + col

    n = line_length
    lines: List[Tuple[int, ...]] = []
    for row in range(num_rows):
        for col in range(num_cols - n + 1):
            # Horizontal window starting at (row, col).
            lines.append(tuple(idx(row, col + k) for k in range(n)))
    for col in range(num_cols):
        for row in range(num_rows - n + 1):
            # Vertical window starting at (row, col).
            lines.append(tuple(idx(row + k, col) for k in range(n)))
    for row in range(num_rows - n + 1):
        for col in range(num_cols - n + 1):
            # Diagonal window, top-left to bottom-right.
            lines.append(tuple(idx(row + k, col + k) for k in range(n)))
            # Diagonal window, top-right to bottom-left.
            lines.append(tuple(idx(row + k, col + n - 1 - k) for k in range(n)))
    return lines


_CONNECT_FOUR_WIN_LINES = _build_win_lines(
    _CONNECT_FOUR_NUM_ROWS, _CONNECT_FOUR_NUM_COLS, _CONNECT_FOUR_LINE_LENGTH
)


# #############################################################################
# ConnectFour
# #############################################################################


class ConnectFour(rimtsaazg.Game):
    """
    Standard 7-column x 6-row Connect Four.

    - A state is a length-42 tuple of cell values (`0` empty, `1` X, `-1` O)
      indexed row-major from the top-left corner; row `0` is the top row
    - A move is a column index `0`-`6`; the mark drops to the lowest empty cell
      in that column (gravity), so unlike tic-tac-toe not every empty cell is a
      legal move
    - Player `1` (X) moves first
    """

    def get_initial_state(self) -> rimtsaazg.State:
        """
        Build an empty 6x7 board.

        :return: length-42 tuple of `0`s
        """
        state = (_EMPTY,) * (_CONNECT_FOUR_NUM_ROWS * _CONNECT_FOUR_NUM_COLS)
        return state

    def get_legal_moves(self, state: rimtsaazg.State) -> List[rimtsaazg.Move]:
        """
        List the columns that still have room, if any.

        A column has room iff its top cell (row `0`) is still empty.

        :param state: game state to query
        :return: column indices with room, empty if `state` is terminal
        """
        if self.is_terminal(state):
            moves: List[rimtsaazg.Move] = []
        else:
            moves = [
                col
                for col in range(_CONNECT_FOUR_NUM_COLS)
                if state[col] == _EMPTY
            ]
        return moves

    def _get_landing_row(self, state: rimtsaazg.State, col: int) -> int:
        """
        Find the lowest empty row in `col` (gravity drop target).

        :param state: game state before the move
        :param col: column to drop into, must have room
        :return: row index the mark will land on
        """
        landing_row: Optional[int] = None
        for row in range(_CONNECT_FOUR_NUM_ROWS - 1, -1, -1):
            if state[row * _CONNECT_FOUR_NUM_COLS + col] == _EMPTY:
                landing_row = row
                break
        hdbg.dassert_is_not(landing_row, None, "Column %s is full", col)
        return cast(int, landing_row)

    def apply_move(
        self, state: rimtsaazg.State, move: rimtsaazg.Move
    ) -> rimtsaazg.State:
        """
        Drop the current player's mark into column `move`.

        :param state: game state before the move
        :param move: column to drop into, must have room
        :return: new state with the current player's mark at the landing
            cell
        """
        row = self._get_landing_row(state, move)
        player = self.get_current_player(state)
        cell_idx = row * _CONNECT_FOUR_NUM_COLS + move
        new_state = state[:cell_idx] + (player,) + state[cell_idx + 1 :]
        return new_state

    def is_terminal(self, state: rimtsaazg.State) -> bool:
        """
        Check whether the board has a 4-in-a-row winner or is full.

        :param state: game state to query
        :return: True if the board is a win for either player or a draw
        """
        is_over = (
            _get_line_winner(state, _CONNECT_FOUR_WIN_LINES) != _EMPTY
            or _EMPTY not in state
        )
        return is_over

    def get_winner(self, state: rimtsaazg.State) -> int:
        """
        Determine the outcome of a finished board.

        :param state: terminal game state
        :return: `1` or `-1` for four-in-a-row, `0` for a full board with no
            winner
        """
        hdbg.dassert(
            self.is_terminal(state), "get_winner() requires a terminal state"
        )
        winner = _get_line_winner(state, _CONNECT_FOUR_WIN_LINES)
        return winner

    def get_current_player(self, state: rimtsaazg.State) -> int:
        """
        Infer whose turn it is from how many cells are filled.

        :param state: game state to query
        :return: `1` (X) if an even number of moves have been played, `-1`
            (O) otherwise
        """
        current_player = _infer_current_player(state)
        return current_player

    def render(self, state: rimtsaazg.State) -> str:
        """
        Render the board as 6 rows of space-separated symbols.

        :param state: game state to render
        :return: 6-line string, top row first
        """
        board_str = _render_board(state, _CONNECT_FOUR_NUM_COLS)
        return board_str


# #############################################################################
# ConnectThree
# #############################################################################


_CONNECT_THREE_NUM_ROWS = 3
_CONNECT_THREE_NUM_COLS = 3
_CONNECT_THREE_LINE_LENGTH = 3

_CONNECT_THREE_WIN_LINES = _build_win_lines(
    _CONNECT_THREE_NUM_ROWS, _CONNECT_THREE_NUM_COLS, _CONNECT_THREE_LINE_LENGTH
)


# #############################################################################
# ConnectThree
# #############################################################################


class ConnectThree(rimtsaazg.Game):
    """
    3-column x 3-row Connect Three: `ConnectFour`'s gravity-drop rule on a
    board small enough to search exhaustively.

    - A state is a length-9 tuple of cell values (`0` empty, `1` X, `-1` O)
      indexed row-major from the top-left corner; row `0` is the top row
    - A move is a column index `0`-`2`; the mark drops to the lowest empty
      cell in that column (gravity), exactly like `ConnectFour`, just on a
      3x3 board with a 3-in-a-row win instead of a 4-in-a-row one
    - Player `1` (X) moves first
    - Used by `search_algorithms.02.API.ipynb` to show the step-by-step
      minimax tree-building widget on a second, gravity-drop game, without
      paying `ConnectFour`'s far larger game tree
    """

    def get_initial_state(self) -> rimtsaazg.State:
        """
        Build an empty 3x3 board.

        :return: length-9 tuple of `0`s
        """
        state = (_EMPTY,) * (_CONNECT_THREE_NUM_ROWS * _CONNECT_THREE_NUM_COLS)
        return state

    def get_legal_moves(self, state: rimtsaazg.State) -> List[rimtsaazg.Move]:
        """
        List the columns that still have room, if any.

        A column has room iff its top cell (row `0`) is still empty.

        :param state: game state to query
        :return: column indices with room, empty if `state` is terminal
        """
        if self.is_terminal(state):
            moves: List[rimtsaazg.Move] = []
        else:
            moves = [
                col
                for col in range(_CONNECT_THREE_NUM_COLS)
                if state[col] == _EMPTY
            ]
        return moves

    def _get_landing_row(self, state: rimtsaazg.State, col: int) -> int:
        """
        Find the lowest empty row in `col` (gravity drop target).

        :param state: game state before the move
        :param col: column to drop into, must have room
        :return: row index the mark will land on
        """
        landing_row: Optional[int] = None
        for row in range(_CONNECT_THREE_NUM_ROWS - 1, -1, -1):
            if state[row * _CONNECT_THREE_NUM_COLS + col] == _EMPTY:
                landing_row = row
                break
        hdbg.dassert_is_not(landing_row, None, "Column %s is full", col)
        return cast(int, landing_row)

    def apply_move(
        self, state: rimtsaazg.State, move: rimtsaazg.Move
    ) -> rimtsaazg.State:
        """
        Drop the current player's mark into column `move`.

        :param state: game state before the move
        :param move: column to drop into, must have room
        :return: new state with the current player's mark at the landing
            cell
        """
        row = self._get_landing_row(state, move)
        player = self.get_current_player(state)
        cell_idx = row * _CONNECT_THREE_NUM_COLS + move
        new_state = state[:cell_idx] + (player,) + state[cell_idx + 1 :]
        return new_state

    def is_terminal(self, state: rimtsaazg.State) -> bool:
        """
        Check whether the board has a 3-in-a-row winner or is full.

        :param state: game state to query
        :return: True if the board is a win for either player or a draw
        """
        is_over = (
            _get_line_winner(state, _CONNECT_THREE_WIN_LINES) != _EMPTY
            or _EMPTY not in state
        )
        return is_over

    def get_winner(self, state: rimtsaazg.State) -> int:
        """
        Determine the outcome of a finished board.

        :param state: terminal game state
        :return: `1` or `-1` for three-in-a-row, `0` for a full board with
            no winner
        """
        hdbg.dassert(
            self.is_terminal(state), "get_winner() requires a terminal state"
        )
        winner = _get_line_winner(state, _CONNECT_THREE_WIN_LINES)
        return winner

    def get_current_player(self, state: rimtsaazg.State) -> int:
        """
        Infer whose turn it is from how many cells are filled.

        :param state: game state to query
        :return: `1` (X) if an even number of moves have been played, `-1`
            (O) otherwise
        """
        current_player = _infer_current_player(state)
        return current_player

    def render(self, state: rimtsaazg.State) -> str:
        """
        Render the board as 3 rows of space-separated symbols.

        :param state: game state to render
        :return: 3-line string, top row first
        """
        board_str = _render_board(state, _CONNECT_THREE_NUM_COLS)
        return board_str
