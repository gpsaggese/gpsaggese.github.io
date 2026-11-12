"""
Notebook utilities for `game.01.API.ipynb`.

See `game.py` for the `Game` interface, `game_examples.py` for the concrete
`TicTacToe` and `ConnectFour` implementations these widgets operate on, and
`README.md` for a description of every file in this directory.

Import as:

import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game_API_utils as rimtsaazgau
"""

import logging
from typing import Callable, Dict, Optional, Tuple

import ipywidgets
import matplotlib.axes
import matplotlib.patches
import matplotlib.pyplot as plt
from IPython.display import clear_output

import helpers.hprint as hprint
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game as rimtsaazg

_LOG = logging.getLogger(__name__)


# #############################################################################
# Cell 2.4: Interactive two-player board
# #############################################################################


def cell2_4_build_play_widget(game: rimtsaazg.Game) -> "ipywidgets.VBox":
    """
    Build a clickable board where the user plays both `X` and `O`.

    Every click applies a legal move via `Game.apply_move()`, re-renders the
    3x3 grid of buttons, and updates the status line; a "New game" button
    resets to the initial state.

    :param game: game-agnostic rules implementation for a 3x3 board (e.g.,
        `TicTacToe`)
    :return: widget container to `display()` in a notebook cell
    """
    _LOG.debug(hprint.to_str("game"))
    # A cell's value is `0` (empty), `1` (X), or `-1` (O); see `Game`'s
    # docstring in `game.py` for the player-encoding convention.
    # Use a single space (not "") for the empty label: some `ipywidgets`
    # frontends don't repaint a `Button.description` change into an empty
    # string, which would leave stale `X`/`O` labels on the board after
    # "New game".
    _empty = 0
    _player_symbols = {0: " ", 1: "X", -1: "O"}
    game_state: Dict[str, rimtsaazg.State] = {"state": game.get_initial_state()}
    status = ipywidgets.HTML()
    cell_buttons = [
        ipywidgets.Button(
            description="",
            layout=ipywidgets.Layout(width="60px", height="60px"),
        )
        for _ in range(9)
    ]

    def _update_view() -> None:
        """
        Refresh button labels and the status line from `game_state`.
        """
        state = game_state["state"]
        for button, cell in zip(cell_buttons, state):
            button.description = _player_symbols[cell]
        if game.is_terminal(state):
            winner = game.get_winner(state)
            if winner == _empty:
                status.value = "<b>Game over: draw</b>"
            else:
                status.value = (
                    f"<b>Game over: {_player_symbols[winner]} wins</b>"
                )
        else:
            player = game.get_current_player(state)
            status.value = f"Current turn: <b>{_player_symbols[player]}</b>"

    def _make_on_click(index: int) -> Callable[["ipywidgets.Button"], None]:
        """
        Build the click handler for the cell at `index`.
        """

        def _on_click(_button: "ipywidgets.Button") -> None:
            state = game_state["state"]
            if game.is_terminal(state) or state[index] != _empty:
                # Ignore clicks on a finished game or an occupied cell.
                return
            game_state["state"] = game.apply_move(state, index)
            _update_view()

        return _on_click

    # Wire up one click handler per cell.
    for index, button in enumerate(cell_buttons):
        button.on_click(_make_on_click(index))

    def _on_reset_click(_button: "ipywidgets.Button") -> None:
        game_state["state"] = game.get_initial_state()
        _update_view()

    reset_button = ipywidgets.Button(description="New game")
    reset_button.on_click(_on_reset_click)
    # Lay out the 9 buttons as a 3x3 grid.
    grid = ipywidgets.VBox(
        [ipywidgets.HBox(cell_buttons[row : row + 3]) for row in range(0, 9, 3)]
    )
    _update_view()
    widget = ipywidgets.VBox([status, grid, reset_button])
    return widget


# #############################################################################
# Part 3: Connect Four
# #############################################################################


# Board geometry mirrors `game_examples.ConnectFour`'s hardcoded 6x7 layout;
# kept as separate constants here since this module only deals with
# presentation, not game rules.
_CONNECT_FOUR_NUM_ROWS = 6
_CONNECT_FOUR_NUM_COLS = 7

# Disc colors follow the physical game (red vs. yellow discs on a blue
# board), not the muted palette used for charts elsewhere in this module:
# a literal, recognizable board is the whole point of a board picture.
_CONNECT_FOUR_DISC_COLORS = {0: "#F5F5F5", 1: "#D6402C", -1: "#F4C542"}
_CONNECT_FOUR_BOARD_COLOR = "#2F6FBB"
_CONNECT_FOUR_PLAYER_NAMES = {1: "red", -1: "yellow"}


def plot_connect_four_board(
    state: rimtsaazg.State,
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Draw `state` as a Connect Four board: colored discs on a blue grid.

    A picture reads a Connect Four position far faster than the text grid
    `ConnectFour.render()` produces, which is why every Connect Four cell in
    `game.01.API.ipynb` and `mcts.03.API.ipynb` shows the board this way.

    :param state: length-42 Connect Four state (see
        `game_examples.ConnectFour`), row-major with row `0` at the top
    :param ax: axes to draw into
        - Default: `None` (create a standalone figure and `plt.show()` it;
          pass an `ax` instead to embed the board in a larger figure, e.g.,
          a multi-panel widget)
    :param figsize: figure size, only used when `ax` is `None`
        - Default: `None` (uses `plt.rcParams["figure.figsize"]`)
    """
    standalone = ax is None
    if standalone:
        if figsize is None:
            figsize = plt.rcParams["figure.figsize"]
        _, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(_CONNECT_FOUR_BOARD_COLOR)
    for row in range(_CONNECT_FOUR_NUM_ROWS):
        for col in range(_CONNECT_FOUR_NUM_COLS):
            cell = state[row * _CONNECT_FOUR_NUM_COLS + col]
            # Flip the row so row 0 (top of the board) is drawn at the top
            # of the axes instead of matplotlib's default bottom-up y-axis.
            y = _CONNECT_FOUR_NUM_ROWS - 1 - row
            disc = matplotlib.patches.Circle(
                (col, y),
                0.42,
                facecolor=_CONNECT_FOUR_DISC_COLORS[cell],
                edgecolor="#1B4A80",
                linewidth=1.0,
                zorder=2,
            )
            ax.add_patch(disc)
    ax.set_xlim(-0.6, _CONNECT_FOUR_NUM_COLS - 0.4)
    ax.set_ylim(-0.6, _CONNECT_FOUR_NUM_ROWS - 0.4)
    ax.set_aspect("equal")
    ax.set_xticks(range(_CONNECT_FOUR_NUM_COLS))
    ax.set_xlabel("column")
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if standalone:
        plt.show()


# #############################################################################
# Cell 3.3: Interactive Connect Four board
# #############################################################################


def cell3_3_build_connect_four_play_widget(
    game: rimtsaazg.Game,
) -> "ipywidgets.VBox":
    """
    Build a clickable Connect Four board where the user plays both colors.

    One button per column drops the current player's disc via
    `Game.apply_move()`, gravity included; the board picture
    (`plot_connect_four_board()`) and status line refresh after every move;
    a "New game" button resets to the initial state.

    :param game: game-agnostic rules implementation for Connect Four
        (`ConnectFour`)
    :return: widget container to `display()` in a notebook cell
    """
    _LOG.debug(hprint.to_str("game"))
    game_state: Dict[str, rimtsaazg.State] = {"state": game.get_initial_state()}
    status = ipywidgets.HTML()
    output = ipywidgets.Output()

    def _update_view() -> None:
        """
        Redraw the board picture and refresh the status line from
        `game_state`.
        """
        state = game_state["state"]
        with output:
            clear_output(wait=True)
            plot_connect_four_board(state, figsize=(6, 5))
        if game.is_terminal(state):
            winner = game.get_winner(state)
            if winner == 0:
                status.value = "<b>Game over: draw</b>"
            else:
                status.value = f"<b>Game over: {_CONNECT_FOUR_PLAYER_NAMES[winner]} wins</b>"
        else:
            player = game.get_current_player(state)
            status.value = (
                f"Current turn: <b>{_CONNECT_FOUR_PLAYER_NAMES[player]}</b>"
            )

    def _make_on_click(col: int) -> Callable[["ipywidgets.Button"], None]:
        """
        Build the click handler that drops a disc into column `col`.
        """

        def _on_click(_button: "ipywidgets.Button") -> None:
            state = game_state["state"]
            if game.is_terminal(state) or col not in game.get_legal_moves(state):
                # Ignore clicks on a finished game or a full column.
                return
            game_state["state"] = game.apply_move(state, col)
            _update_view()

        return _on_click

    # Wire up one click handler per column.
    column_buttons = [
        ipywidgets.Button(
            description=f"drop {col}",
            layout=ipywidgets.Layout(width="70px"),
        )
        for col in range(_CONNECT_FOUR_NUM_COLS)
    ]
    for col, button in enumerate(column_buttons):
        button.on_click(_make_on_click(col))

    def _on_reset_click(_button: "ipywidgets.Button") -> None:
        game_state["state"] = game.get_initial_state()
        _update_view()

    reset_button = ipywidgets.Button(description="New game")
    reset_button.on_click(_on_reset_click)
    buttons_row = ipywidgets.HBox(column_buttons)
    _update_view()
    widget = ipywidgets.VBox([status, buttons_row, output, reset_button])
    return widget
