"""
Utility functions for the game theory refresher notebook.

Implements the interactive widgets, visualizations, and equilibrium
computations built around a single reusable 2x2 payoff-matrix sandbox, plus
the sequential-game, evolutionary-game, and network-routing cells that build
on it.

Import as:

import book_cs_refreshers.tutorials.notebooks.L95_05_game_theory_utils as bcrtnl0gtu
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import ipywidgets
import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import clear_output, display

import helpers.hnotebook as hnotebo
import helpers.htutorial as htutori

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger to the utils logger.

    :param notebook_log: Logger created in the notebook
    """
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


def _fmt_num(x: float) -> str:
    """
    Format a payoff number without a trailing ".0" for whole numbers.

    :param x: Payoff value
    :return: `"4"` for whole numbers, `"4.50"` otherwise
    """
    if float(x).is_integer():
        return f"{x:.0f}"
    return f"{x:.2f}"


# #############################################################################
# Shared 2x2 game representation (used by Cells 1.1-4.2).
# #############################################################################

# Canonical 2x2 games as (row player payoffs, column player payoffs), each a
# 2x2 array indexed [row action, column action].
_PRESETS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "Prisoners' Dilemma": (
        np.array([[-1.0, -3.0], [0.0, -2.0]]),
        np.array([[-1.0, 0.0], [-3.0, -2.0]]),
    ),
    "Battle of the Sexes": (
        np.array([[2.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 2.0]]),
    ),
    "Stag Hunt": (
        np.array([[4.0, 0.0], [3.0, 3.0]]),
        np.array([[4.0, 3.0], [0.0, 3.0]]),
    ),
    "Matching Pennies": (
        np.array([[1.0, -1.0], [-1.0, 1.0]]),
        np.array([[-1.0, 1.0], [1.0, -1.0]]),
    ),
}
# Action names shown on the matrix axes for each preset (row labels, column
# labels are the same two actions since every preset here is symmetric).
_ACTION_LABELS: Dict[str, Tuple[str, str]] = {
    "Prisoners' Dilemma": ("Cooperate", "Defect"),
    "Battle of the Sexes": ("Opera", "Football"),
    "Stag Hunt": ("Stag", "Hare"),
    "Matching Pennies": ("Heads", "Tails"),
    "Custom": ("Action 1", "Action 2"),
}


def _analyze_game(p1: np.ndarray, p2: np.ndarray) -> Dict[str, Any]:
    """
    Compute best responses, pure Nash equilibria, and dominant strategies.

    :param p1: 2x2 payoff array for the row player, `p1[i, j]` is the payoff
        when row plays action `i` and column plays action `j`
    :param p2: 2x2 payoff array for the column player, same indexing
    :return: dict with keys:
        - "row_best": 2x2 bool array, True where the row payoff is a best
          response to that column
        - "col_best": 2x2 bool array, True where the column payoff is a best
          response to that row
        - "nash_cells": list of (i, j) pure Nash equilibrium cells
        - "row_dominant": index of the row player's dominant action, or None
        - "col_dominant": index of the column player's dominant action, or
          None
        - "is_zero_sum": True if `p1 + p2` is constant across all 4 cells
    """
    row_best = np.zeros((2, 2), dtype=bool)
    col_best = np.zeros((2, 2), dtype=bool)
    # A cell is a row best response if it is the max of its column, and a
    # column best response if it is the max of its row.
    for j in range(2):
        row_best[np.argmax(p1[:, j]), j] = True
    for i in range(2):
        col_best[i, np.argmax(p2[i, :])] = True
    nash_cells = [
        (i, j)
        for i in range(2)
        for j in range(2)
        if row_best[i, j] and col_best[i, j]
    ]
    # A row (column) is dominant if it strictly beats the other row (column)
    # against every opponent action.
    row_dominant: Optional[int] = None
    if np.all(p1[0, :] > p1[1, :]):
        row_dominant = 0
    elif np.all(p1[1, :] > p1[0, :]):
        row_dominant = 1
    col_dominant: Optional[int] = None
    if np.all(p2[:, 0] > p2[:, 1]):
        col_dominant = 0
    elif np.all(p2[:, 1] > p2[:, 0]):
        col_dominant = 1
    totals = p1 + p2
    is_zero_sum = bool(np.allclose(totals, totals.flat[0]))
    analysis = {
        "row_best": row_best,
        "col_best": col_best,
        "nash_cells": nash_cells,
        "row_dominant": row_dominant,
        "col_dominant": col_dominant,
        "is_zero_sum": is_zero_sum,
    }
    return analysis


def _is_pareto_optimal(p1: np.ndarray, p2: np.ndarray, i: int, j: int) -> bool:
    """
    Check whether cell `(i, j)` is Pareto optimal.

    A cell is Pareto optimal if no other cell gives both players a payoff at
    least as high, with at least one player strictly better off.

    :param p1: 2x2 row player payoff array
    :param p2: 2x2 column player payoff array
    :param i: Row index of the cell to check
    :param j: Column index of the cell to check
    :return: True if no other cell Pareto-dominates `(i, j)`
    """
    for a in range(2):
        for b in range(2):
            if (a, b) == (i, j):
                continue
            at_least_as_good = p1[a, b] >= p1[i, j] and p2[a, b] >= p2[i, j]
            strictly_better = p1[a, b] > p1[i, j] or p2[a, b] > p2[i, j]
            if at_least_as_good and strictly_better:
                return False
    return True


def _render_matrix_panel(
    ax: matplotlib.axes.Axes,
    p1: np.ndarray,
    p2: np.ndarray,
    row_labels: Tuple[str, str],
    col_labels: Tuple[str, str],
    analysis: Dict[str, Any],
    *,
    title: str = "Payoff matrix",
    show_best_response: bool = False,
    show_dominant: bool = False,
) -> None:
    """
    Draw one 2x2 payoff matrix, optionally highlighted.

    Each cell shows `(row payoff, col payoff)`. When `show_best_response` is
    set, the row payoff is boxed light blue where it is the row player's best
    response and the column payoff is boxed light orange where it is the
    column player's best response; cells where both hold (pure Nash
    equilibria) get a gold background. When `show_dominant` is set, the
    dominant row and/or column (if any) is outlined.

    :param ax: Axes to draw on
    :param p1: 2x2 row player payoff array
    :param p2: 2x2 column player payoff array
    :param row_labels: Names of the row player's two actions
    :param col_labels: Names of the column player's two actions
    :param analysis: Output of `_analyze_game(p1, p2)`
    :param title: Panel title
    :param show_best_response: If True, highlight best responses and Nash
        equilibrium cells
    :param show_dominant: If True, outline the dominant row/column
    """
    # Shade the background by total surplus (row payoff + column payoff) so
    # students see at a glance which outcomes are jointly better.
    totals = p1 + p2
    ax.imshow(totals, cmap="RdYlGn", alpha=0.35, vmin=-6, vmax=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(col_labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("column player")
    ax.set_ylabel("row player")
    ax.set_title(title, fontsize=13, fontweight="bold")
    for i in range(2):
        for j in range(2):
            is_nash = (i, j) in analysis["nash_cells"]
            if show_best_response and is_nash:
                nash_rect = mpatches.Rectangle(
                    (j - 0.48, i - 0.48),
                    0.96,
                    0.96,
                    facecolor="gold",
                    alpha=0.5,
                    edgecolor="none",
                    zorder=1,
                )
                ax.add_patch(nash_rect)
            # Row payoff (left number), boxed light blue if it is the row
            # player's best response to this column.
            row_bbox = None
            if show_best_response and analysis["row_best"][i, j]:
                row_bbox = dict(
                    boxstyle="round,pad=0.2",
                    facecolor="#AED6F1",
                    edgecolor="none",
                )
            ax.text(
                j - 0.18,
                i,
                _fmt_num(p1[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                bbox=row_bbox,
                zorder=3,
            )
            ax.text(j, i, ",", ha="center", va="center", fontsize=12, zorder=3)
            # Column payoff (right number), boxed light orange if it is the
            # column player's best response to this row.
            col_bbox = None
            if show_best_response and analysis["col_best"][i, j]:
                col_bbox = dict(
                    boxstyle="round,pad=0.2",
                    facecolor="#F5CBA7",
                    edgecolor="none",
                )
            ax.text(
                j + 0.18,
                i,
                _fmt_num(p2[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                bbox=col_bbox,
                zorder=3,
            )
    # Outline the dominant row and/or column, if requested and present.
    if show_dominant and analysis["row_dominant"] is not None:
        i = analysis["row_dominant"]
        ax.add_patch(
            mpatches.Rectangle(
                (-0.5, i - 0.5),
                2.0,
                1.0,
                fill=False,
                edgecolor="steelblue",
                linewidth=3,
                zorder=2,
            )
        )
    if show_dominant and analysis["col_dominant"] is not None:
        j = analysis["col_dominant"]
        ax.add_patch(
            mpatches.Rectangle(
                (j - 0.5, -0.5),
                1.0,
                2.0,
                fill=False,
                edgecolor="darkorange",
                linewidth=3,
                zorder=2,
            )
        )
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)


# #############################################################################
# Shared 2x2 sandbox controls (used by Cells 1.2, 2.1, 2.2).
# #############################################################################

# Cell name suffixes in row-major order: "11"=(row0,col0), "12"=(row0,col1),
# "21"=(row1,col0), "22"=(row1,col1).
_CELL_NAMES = ["11", "12", "21", "22"]


def _build_sandbox_controls(
    *, initial_game: str = "Prisoners' Dilemma"
) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Build the preset dropdown and 8 payoff sliders for the 2x2 sandbox.

    :param initial_game: Preset loaded when the sandbox first renders
    :return: Tuple of (game dropdown, dict of 8 sliders keyed "p1_11" etc.,
        VBox layout containing the dropdown and both players' sliders)
    """
    game_dd = ipywidgets.Dropdown(
        options=["Custom"] + list(_PRESETS.keys()),
        value=initial_game,
        description="game:",
        style={"description_width": "initial"},
    )
    p1_init, p2_init = _PRESETS[initial_game]
    sliders: Dict[str, Any] = {}
    boxes: List[Any] = [game_dd]
    # Row player's 4 payoff sliders, then the column player's 4.
    for player, payoffs in (("p1", p1_init), ("p2", p2_init)):
        for idx, name in enumerate(_CELL_NAMES):
            i, j = idx // 2, idx % 2
            slider, box = htutori.build_widget_control(
                name=f"{player}_{name}",
                description=f"payoff at ({i + 1},{j + 1})",
                min_val=-5,
                max_val=5,
                step=1,
                initial_value=int(payoffs[i, j]),
                is_float=False,
            )
            sliders[f"{player}_{name}"] = slider
            boxes.append(box)
    controls = ipywidgets.VBox(boxes)
    return game_dd, sliders, controls


def _read_sandbox_payoffs(
    sliders: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the 8 sandbox sliders into 2x2 row/column payoff arrays.

    :param sliders: Dict of 8 sliders keyed "p1_11", "p1_12", ..., "p2_22"
    :return: Tuple of (row player 2x2 array, column player 2x2 array)
    """
    p1 = np.array(
        [
            [sliders["p1_11"].value, sliders["p1_12"].value],
            [sliders["p1_21"].value, sliders["p1_22"].value],
        ],
        dtype=float,
    )
    p2 = np.array(
        [
            [sliders["p2_11"].value, sliders["p2_12"].value],
            [sliders["p2_21"].value, sliders["p2_22"].value],
        ],
        dtype=float,
    )
    return p1, p2


def _format_sandbox_comments(
    p1: np.ndarray,
    p2: np.ndarray,
    analysis: Dict[str, Any],
    row_labels: Tuple[str, str],
    col_labels: Tuple[str, str],
) -> str:
    """
    Format the comments panel text for the plain sandbox (Cell 1.2).
    """
    _ = row_labels, col_labels
    lines = [
        "Payoffs (row, col):",
        f"  ({_fmt_num(p1[0, 0])},{_fmt_num(p2[0, 0])})"
        f"   ({_fmt_num(p1[0, 1])},{_fmt_num(p2[0, 1])})",
        f"  ({_fmt_num(p1[1, 0])},{_fmt_num(p2[1, 0])})"
        f"   ({_fmt_num(p1[1, 1])},{_fmt_num(p2[1, 1])})",
        "",
        f"zero-sum: {analysis['is_zero_sum']}",
    ]
    return "\n".join(lines)


def _format_dominance_comments(
    p1: np.ndarray,
    p2: np.ndarray,
    analysis: Dict[str, Any],
    row_labels: Tuple[str, str],
    col_labels: Tuple[str, str],
) -> str:
    """
    Format the comments panel text for the dominance sandbox (Cell 2.1).
    """
    _ = p1, p2
    row_dom = analysis["row_dominant"]
    col_dom = analysis["col_dominant"]
    row_txt = row_labels[row_dom] if row_dom is not None else "none"
    col_txt = col_labels[col_dom] if col_dom is not None else "none"
    lines = [
        "Dominant strategies:",
        f"  row player: {row_txt}",
        f"  column player: {col_txt}",
    ]
    return "\n".join(lines)


def _format_nash_comments(
    p1: np.ndarray,
    p2: np.ndarray,
    analysis: Dict[str, Any],
    row_labels: Tuple[str, str],
    col_labels: Tuple[str, str],
) -> str:
    """
    Format the comments panel text for the Nash sandbox (Cell 2.2).
    """
    _ = p1, p2
    nash = analysis["nash_cells"]
    lines = [f"Pure Nash equilibria: {len(nash)}"]
    if nash:
        for i, j in nash:
            lines.append(f"  ({row_labels[i]}, {col_labels[j]})")
    else:
        lines.append("  (none: best responses cycle)")
    return "\n".join(lines)


def _run_sandbox_cell(
    *,
    figsize: Optional[Tuple[float, float]],
    show_best_response: bool,
    show_dominant: bool,
    comments_fn: Callable[
        [
            np.ndarray,
            np.ndarray,
            Dict[str, Any],
            Tuple[str, str],
            Tuple[str, str],
        ],
        str,
    ],
) -> None:
    """
    Shared driver for the sandbox-based interactive cells (1.2, 2.1, 2.2).

    Builds the preset dropdown and 8 payoff sliders, then renders a matrix
    panel plus a comments panel recomputed by `comments_fn` on every change.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    :param show_best_response: Forwarded to `_render_matrix_panel`
    :param show_dominant: Forwarded to `_render_matrix_panel`
    :param comments_fn: Function turning (p1, p2, analysis, row_labels,
        col_labels) into the comments panel text
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    game_dd, sliders, controls = _build_sandbox_controls()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the matrix and comments panel from the current sliders.
        """
        _ = change
        with output:
            clear_output(wait=True)
            p1, p2 = _read_sandbox_payoffs(sliders)
            game = game_dd.value
            action_labels = _ACTION_LABELS.get(game, _ACTION_LABELS["Custom"])
            row_labels = col_labels = action_labels
            analysis = _analyze_game(p1, p2)
            _, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1]}
            )
            _render_matrix_panel(
                ax1,
                p1,
                p2,
                row_labels,
                col_labels,
                analysis,
                title=game,
                show_best_response=show_best_response,
                show_dominant=show_dominant,
            )
            ax2.axis("off")
            ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text = comments_fn(p1, p2, analysis, row_labels, col_labels)
            htutori.add_fitted_text_box(
                ax2, text, max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    def on_preset_change(change: Any) -> None:
        """
        Overwrite all 8 sliders when a preset is selected from the dropdown.
        """
        _ = change
        game = game_dd.value
        if game == "Custom":
            return
        p1_new, p2_new = _PRESETS[game]
        for idx, name in enumerate(_CELL_NAMES):
            i, j = idx // 2, idx % 2
            sliders[f"p1_{name}"].value = int(p1_new[i, j])
            sliders[f"p2_{name}"].value = int(p2_new[i, j])

    game_dd.observe(on_preset_change, names="value")
    for slider in sliders.values():
        slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([controls, output]))


# #############################################################################
# Cell 1.1: Reading a Payoff Matrix
# #############################################################################


def cell1_1_show_payoff_matrix(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Display the Prisoners' Dilemma payoff matrix as a static reference.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    p1, p2 = _PRESETS["Prisoners' Dilemma"]
    row_labels = col_labels = _ACTION_LABELS["Prisoners' Dilemma"]
    analysis = _analyze_game(p1, p2)
    _, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1]}
    )
    _render_matrix_panel(
        ax1, p1, p2, row_labels, col_labels, analysis, title="Prisoners' Dilemma"
    )
    ax2.axis("off")
    ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    text = (
        "Reading convention:\n"
        "  cell = (row payoff, col payoff)\n\n"
        "Row player picks the row,\n"
        "column player picks the column"
    )
    htutori.add_fitted_text_box(ax2, text, max_fontsize=13, min_fontsize=9)
    plt.tight_layout()
    plt.show()


# #############################################################################
# Cell 1.2: The 2x2 Game Sandbox: Editable Payoffs
# #############################################################################


def cell1_2_game_sandbox(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Interactive 2x2 payoff-matrix sandbox with 8 editable payoffs.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    _run_sandbox_cell(
        figsize=figsize,
        show_best_response=False,
        show_dominant=False,
        comments_fn=_format_sandbox_comments,
    )


# #############################################################################
# Cell 2.1: Highlighting Dominant and Dominated Strategies
# #############################################################################


def cell2_1_dominant_strategy_widget(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Sandbox variant that outlines each player's dominant strategy, if any.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    _run_sandbox_cell(
        figsize=figsize,
        show_best_response=False,
        show_dominant=True,
        comments_fn=_format_dominance_comments,
    )


# #############################################################################
# Cell 2.2: Finding Nash Equilibria via Best Response
# #############################################################################


def cell2_2_nash_equilibrium_widget(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Sandbox variant that underlines best responses and shades Nash cells.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    _run_sandbox_cell(
        figsize=figsize,
        show_best_response=True,
        show_dominant=False,
        comments_fn=_format_nash_comments,
    )


# #############################################################################
# Cell 3.1: Prisoners' Dilemma, Battle of the Sexes, and Stag Hunt Side by Side
# #############################################################################


def cell3_1_classic_games_gallery(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Compare three canonical games with best-response highlighting.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    highlight_toggle = ipywidgets.ToggleButton(
        value=True, description="highlight best responses"
    )
    output = ipywidgets.Output()
    games = ["Prisoners' Dilemma", "Battle of the Sexes", "Stag Hunt"]

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the three matrices with the current highlight setting.
        """
        _ = change
        with output:
            clear_output(wait=True)
            _, axes = plt.subplots(1, 3, figsize=figsize)
            for ax, game in zip(axes, games):
                p1, p2 = _PRESETS[game]
                row_labels = col_labels = _ACTION_LABELS[game]
                analysis = _analyze_game(p1, p2)
                _render_matrix_panel(
                    ax,
                    p1,
                    p2,
                    row_labels,
                    col_labels,
                    analysis,
                    title=game,
                    show_best_response=highlight_toggle.value,
                )
                # One-line summary of the Nash equilibria and their Pareto
                # status, placed under the matrix.
                nash = analysis["nash_cells"]
                per_ne = [
                    f"({row_labels[i]},{col_labels[j]}): "
                    + (
                        "Pareto"
                        if _is_pareto_optimal(p1, p2, i, j)
                        else "not Pareto"
                    )
                    for i, j in nash
                ]
                summary = "; ".join(per_ne) if per_ne else "no pure NE"
                ax.text(
                    0.5,
                    -0.32,
                    summary,
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=8,
                )
            plt.tight_layout()
            plt.show()

    highlight_toggle.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([highlight_toggle, output]))


# #############################################################################
# Cell 3.2: Stag Hunt: Payoff Dominance vs Risk Dominance
# #############################################################################

# Hare's payoff is fixed regardless of the opponent's action (see the Stag
# Hunt preset above: (Hare,Stag)=3 and (Hare,Hare)=3).
_STAG_HUNT_HARE_PAYOFF = 3.0


def cell3_2_stag_hunt_dominance(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Interactive belief-threshold plot for Stag Hunt equilibrium selection.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    v_slider, v_box = htutori.build_widget_control(
        name="V",
        description="stag payoff (Stag,Stag)",
        min_val=2,
        max_val=6,
        step=0.5,
        initial_value=4,
        is_float=True,
    )
    q_slider, q_box = htutori.build_widget_control(
        name="q",
        description="belief opponent plays Stag",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.5,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the expected-payoff-vs-belief plot from the current sliders.
        """
        _ = change
        with output:
            clear_output(wait=True)
            v = v_slider.value
            q = q_slider.value
            q_grid = np.linspace(0, 1, 200)
            e_stag = q_grid * v
            e_hare = np.full_like(q_grid, _STAG_HUNT_HARE_PAYOFF)
            _, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.6, 1]}
            )
            ax1.plot(q_grid, e_stag, label="E[Stag]", color="firebrick")
            ax1.plot(q_grid, e_hare, label="E[Hare]", color="steelblue")
            q_star = min(_STAG_HUNT_HARE_PAYOFF / v, 1.0)
            ax1.axvline(
                q_star,
                color="gray",
                linestyle="--",
                label=f"threshold q*={q_star:.2f}",
            )
            ax1.scatter([q], [q * v], color="firebrick", zorder=5)
            ax1.scatter(
                [q], [_STAG_HUNT_HARE_PAYOFF], color="steelblue", zorder=5
            )
            ax1.set_xlabel("belief q that opponent plays Stag")
            ax1.set_ylabel("expected payoff")
            ax1.set_title("Expected payoff vs belief")
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax2.axis("off")
            ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            better = "Stag" if q * v > _STAG_HUNT_HARE_PAYOFF else "Hare"
            text = (
                f"V (stag payoff): {v:.2f}\n"
                f"Hare payoff (fixed): {_STAG_HUNT_HARE_PAYOFF:.0f}\n"
                f"q: {q:.2f}\n\n"
                f"Threshold q*: {q_star:.2f}\n"
                f"Currently better: {better}"
            )
            htutori.add_fitted_text_box(
                ax2, text, max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    v_slider.observe(update_plot, names="value")
    q_slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([v_box, q_box, output]))


# #############################################################################
# Cell 4.1: Why Randomize? Matching Pennies Has No Pure Equilibrium
# #############################################################################


def cell4_1_matching_pennies_best_response(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Show the best-response cycle in Matching Pennies (no pure Nash).

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    p1, p2 = _PRESETS["Matching Pennies"]
    row_labels = col_labels = _ACTION_LABELS["Matching Pennies"]
    analysis = _analyze_game(p1, p2)
    _, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1]}
    )
    _render_matrix_panel(
        ax1,
        p1,
        p2,
        row_labels,
        col_labels,
        analysis,
        title="Matching Pennies",
        show_best_response=True,
    )
    ax2.axis("off")
    ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    text = (
        f"Pure Nash equilibria: {len(analysis['nash_cells'])}\n\n"
        "Best responses cycle:\n"
        "Heads -> Tails -> Heads"
    )
    htutori.add_fitted_text_box(ax2, text, max_fontsize=13, min_fontsize=9)
    plt.tight_layout()
    plt.show()


# #############################################################################
# Cell 4.2: Computing the Equilibrium Mix by Indifference
# #############################################################################


def cell4_2_mixed_equilibrium_indifference(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Interactive indifference plot for the Matching Pennies mixed equilibrium.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    stakes_slider, stakes_box = htutori.build_widget_control(
        name="stakes",
        description="win/lose payoff magnitude",
        min_val=1,
        max_val=5,
        step=0.5,
        initial_value=1,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the expected-payoff-vs-q plot from the current stakes slider.
        """
        _ = change
        with output:
            clear_output(wait=True)
            m = stakes_slider.value
            q = np.linspace(0, 1, 200)
            e_heads = m * (2 * q - 1)
            e_tails = m * (1 - 2 * q)
            _, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.6, 1]}
            )
            ax1.plot(q, e_heads, label="E[Heads]", color="firebrick")
            ax1.plot(q, e_tails, label="E[Tails]", color="steelblue")
            ax1.axvline(0.5, color="gray", linestyle="--", label="q*=0.50")
            ax1.set_xlabel("q = opponent's probability of Heads")
            ax1.set_ylabel("expected payoff")
            ax1.set_title("Indifference: expected payoff vs q")
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax2.axis("off")
            ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text = (
                f"stakes: {m:.2f}\n\n"
                "Equilibrium q*: 0.50\n"
                "Expected value at equilibrium: 0.00"
            )
            htutori.add_fitted_text_box(
                ax2, text, max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    stakes_slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([stakes_box, output]))


# #############################################################################
# Cell 5.1: Solving the Entry Game by Backward Induction
# #############################################################################

# Fixed layout and edges for the market-entry game tree (shared by Cells
# 5.1 and 5.2).
_ENTRY_POS: Dict[str, Tuple[float, float]] = {
    "Entrant": (0.5, 2.0),
    "Out": (0.0, 1.0),
    "Incumbent": (1.0, 1.0),
    "Accommodate": (0.5, 0.0),
    "Fight": (1.5, 0.0),
}
_ENTRY_EDGES: List[Tuple[str, str]] = [
    ("Entrant", "Out"),
    ("Entrant", "Incumbent"),
    ("Incumbent", "Accommodate"),
    ("Incumbent", "Fight"),
]
_ENTRY_EDGE_LABELS: Dict[Tuple[str, str], str] = {
    ("Entrant", "Out"): "Out",
    ("Entrant", "Incumbent"): "Enter",
    ("Incumbent", "Accommodate"): "Accommodate",
    ("Incumbent", "Fight"): "Fight",
}


def _solve_entry_game(out_payoff: float, fight_payoff: float) -> Dict[str, Any]:
    """
    Solve the market-entry game by backward induction.

    Fixed payoffs: Accommodate = (Entrant=2, Incumbent=1), Fight =
    (Entrant=-1, Incumbent=`fight_payoff`), Out = (Entrant=`out_payoff`,
    Incumbent=2).

    :param out_payoff: Entrant's payoff if it stays Out
    :param fight_payoff: Incumbent's payoff if it Fights after entry
    :return: dict with keys "leaf_payoffs" (name -> (entrant, incumbent)
        payoff pair), "incumbent_choice", "entrant_choice", and "path" (list
        of edges from the root to the solved leaf)
    """
    leaf_payoffs = {
        "Out": (out_payoff, 2.0),
        "Accommodate": (2.0, 1.0),
        "Fight": (-1.0, fight_payoff),
    }
    # Solve the Incumbent's node first, since it is closer to the leaves.
    incumbent_choice = "Accommodate" if 1.0 >= fight_payoff else "Fight"
    entrant_enter_payoff = leaf_payoffs[incumbent_choice][0]
    # Solve the Entrant's node using the Incumbent's resolved choice.
    entrant_choice = "Enter" if entrant_enter_payoff > out_payoff else "Out"
    if entrant_choice == "Out":
        path = [("Entrant", "Out")]
    else:
        path = [("Entrant", "Incumbent"), ("Incumbent", incumbent_choice)]
    solved = {
        "leaf_payoffs": leaf_payoffs,
        "incumbent_choice": incumbent_choice,
        "entrant_choice": entrant_choice,
        "path": path,
    }
    return solved


def _entry_game_entrant_payoff(solved: Dict[str, Any]) -> float:
    """
    Read the Entrant's realized payoff at the leaf reached by `solved`.

    `solved["entrant_choice"]` is "Out" or "Enter", neither of which is a key
    of `solved["leaf_payoffs"]` (whose keys are "Out", "Accommodate", and
    "Fight"), so the reached leaf name must be read from the last edge of the
    solved path instead.

    :param solved: Output of `_solve_entry_game()`
    :return: Entrant's payoff at the leaf actually reached
    """
    reached_leaf = solved["path"][-1][1]
    return solved["leaf_payoffs"][reached_leaf][0]


def _render_entry_tree(
    ax: matplotlib.axes.Axes,
    solved: Dict[str, Any],
    *,
    title: str = "",
    mark_noncredible: bool = False,
) -> None:
    """
    Draw the market-entry game tree with the backward-induction solution.

    The solved path is drawn in green; when `mark_noncredible` is set, the
    Incumbent's Fight branch is drawn dashed red and annotated, regardless of
    whether it lies on the solved path.

    :param ax: Axes to draw on
    :param solved: Output of `_solve_entry_game()`
    :param title: Panel title
    :param mark_noncredible: If True, mark the Fight branch as non-credible
    """
    graph = nx.DiGraph(_ENTRY_EDGES)
    path_set = set(solved["path"])
    leaf_payoffs = solved["leaf_payoffs"]
    reached_leaf = solved["path"][-1][1] if solved["path"] else None
    # Color each node: decision nodes light gray, the reached leaf green, the
    # Fight leaf pink when marked non-credible, all other leaves white.
    node_colors = []
    node_labels = {}
    for node in graph.nodes():
        if node in ("Entrant", "Incumbent"):
            node_colors.append("#D5D8DC")
            node_labels[node] = node
        else:
            entrant_pay, incumbent_pay = leaf_payoffs[node]
            node_labels[node] = (
                f"{node}\n({_fmt_num(entrant_pay)}, {_fmt_num(incumbent_pay)})"
            )
            if node == reached_leaf:
                node_colors.append("#A9DFBF")
            elif mark_noncredible and node == "Fight":
                node_colors.append("#F5B7B1")
            else:
                node_colors.append("#FFFFFF")
    edge_colors = []
    edge_styles = []
    for edge in graph.edges():
        if edge in path_set:
            edge_colors.append("#229954")
            edge_styles.append("solid")
        elif mark_noncredible and edge == ("Incumbent", "Fight"):
            edge_colors.append("#CB4335")
            edge_styles.append("dashed")
        else:
            edge_colors.append("#AEB6BF")
            edge_styles.append("solid")
    nx.draw_networkx_nodes(
        graph,
        _ENTRY_POS,
        ax=ax,
        node_color=node_colors,
        node_size=2600,
        edgecolors="black",
    )
    nx.draw_networkx_labels(
        graph, _ENTRY_POS, labels=node_labels, ax=ax, font_size=9
    )
    for edge, color, style in zip(graph.edges(), edge_colors, edge_styles):
        nx.draw_networkx_edges(
            graph,
            _ENTRY_POS,
            ax=ax,
            edgelist=[edge],
            edge_color=color,
            style=style,
            width=2.5,
            arrows=True,
            arrowsize=15,
        )
    nx.draw_networkx_edge_labels(
        graph, _ENTRY_POS, ax=ax, edge_labels=_ENTRY_EDGE_LABELS, font_size=9
    )
    if mark_noncredible:
        ax.text(
            1.5, -0.35, "non-credible", ha="center", fontsize=9, color="#CB4335"
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def _format_entry_comments(solved: Dict[str, Any]) -> str:
    """
    Format the comments panel text for the entry game (Cell 5.1).
    """
    lp = solved["leaf_payoffs"]
    lines = [
        "Leaf payoffs (Entrant, Incumbent):",
        f"  Out: ({_fmt_num(lp['Out'][0])}, {_fmt_num(lp['Out'][1])})",
        f"  Accommodate: ({_fmt_num(lp['Accommodate'][0])}, "
        f"{_fmt_num(lp['Accommodate'][1])})",
        f"  Fight: ({_fmt_num(lp['Fight'][0])}, {_fmt_num(lp['Fight'][1])})",
        "",
        f"Incumbent's node solves to: {solved['incumbent_choice']}",
        f"Entrant's node solves to: {solved['entrant_choice']}",
    ]
    return "\n".join(lines)


def cell5_1_entry_game_backward_induction(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Interactive backward-induction solver for the market-entry game.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    out_slider, out_box = htutori.build_widget_control(
        name="out_payoff",
        description="Entrant's payoff if Out",
        min_val=-2,
        max_val=2,
        step=0.5,
        initial_value=0,
        is_float=True,
    )
    fight_slider, fight_box = htutori.build_widget_control(
        name="fight_payoff",
        description="Incumbent's payoff if Fight",
        min_val=-3,
        max_val=1,
        step=0.5,
        initial_value=-1,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Resolve and redraw the entry game from the current sliders.
        """
        _ = change
        with output:
            clear_output(wait=True)
            solved = _solve_entry_game(out_slider.value, fight_slider.value)
            _, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.6, 1]}
            )
            _render_entry_tree(
                ax1, solved, title="Entry game: backward induction"
            )
            ax2.axis("off")
            ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            htutori.add_fitted_text_box(
                ax2,
                _format_entry_comments(solved),
                max_fontsize=13,
                min_fontsize=9,
            )
            plt.tight_layout()
            plt.show()

    out_slider.observe(update_plot, names="value")
    fight_slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([out_box, fight_box, output]))


# #############################################################################
# Cell 5.2: Non-Credible Threats and Subgame Perfection
# #############################################################################


def cell5_2_credible_commitment(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Contrast the baseline SPE with a toggleable credible-commitment device.

    The left panel always shows the baseline subgame perfect equilibrium
    (empty threat, Fight payoff = -1). The right panel shows the same tree
    with the commitment device off (Fight payoff = -1, threat still empty and
    marked non-credible) or on (Fight payoff raised to 2, making Fight
    genuinely optimal and flipping the solved path).

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    commitment_toggle = ipywidgets.ToggleButton(
        value=False,
        description="commitment device (raise Fight payoff to 2)",
        layout={"width": "320px"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw both trees from the current commitment toggle.
        """
        _ = change
        with output:
            clear_output(wait=True)
            baseline = _solve_entry_game(out_payoff=0.0, fight_payoff=-1.0)
            fight_payoff = 2.0 if commitment_toggle.value else -1.0
            contrasted = _solve_entry_game(
                out_payoff=0.0, fight_payoff=fight_payoff
            )
            _, (ax1, ax2, ax3) = plt.subplots(
                1,
                3,
                figsize=figsize,
                gridspec_kw={"width_ratios": [1.4, 1.4, 1]},
            )
            _render_entry_tree(ax1, baseline, title="Baseline: threat is empty")
            right_title = (
                "With commitment device"
                if commitment_toggle.value
                else "Threat restated (still empty)"
            )
            _render_entry_tree(
                ax2,
                contrasted,
                title=right_title,
                mark_noncredible=not commitment_toggle.value,
            )
            ax3.axis("off")
            ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text = (
                f"commitment device: {commitment_toggle.value}\n\n"
                f"Baseline SPE: {baseline['entrant_choice']}, "
                f"{baseline['incumbent_choice']}\n"
                f"Right-panel SPE: {contrasted['entrant_choice']}, "
                f"{contrasted['incumbent_choice']}\n\n"
                f"Entrant payoff (baseline): "
                f"{_fmt_num(_entry_game_entrant_payoff(baseline))}\n"
                f"Entrant payoff (right panel): "
                f"{_fmt_num(_entry_game_entrant_payoff(contrasted))}"
            )
            htutori.add_fitted_text_box(
                ax3, text, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    commitment_toggle.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([commitment_toggle, output]))


# #############################################################################
# Cell 6.1: Hawk-Dove and Evolutionarily Stable Strategies
# #############################################################################


def _hawk_dove_matrix(v: float, c: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the Hawk-Dove payoff matrix for resource value `v` and cost `c`.

    :param v: Value of the contested resource
    :param c: Cost of losing a fight
    :return: Tuple of (row player 2x2 array, column player 2x2 array),
        actions ordered (Hawk, Dove)
    """
    hh = (v - c) / 2.0
    p1 = np.array([[hh, v], [0.0, v / 2.0]])
    p2 = np.array([[hh, 0.0], [v, v / 2.0]])
    return p1, p2


def _ess_fraction(v: float, c: float) -> float:
    """
    Compute the evolutionarily stable fraction of Hawks.

    :param v: Value of the contested resource
    :param c: Cost of losing a fight
    :return: 1.0 (pure Hawk) if `v >= c`, else the interior mix `v / c`
    """
    if v >= c:
        return 1.0
    return v / c


def cell6_1_hawk_dove_ess(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Interactive Hawk-Dove matrix and fitness-crossing plot.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    v_slider, v_box = htutori.build_widget_control(
        name="V",
        description="resource value",
        min_val=1,
        max_val=10,
        step=0.5,
        initial_value=4,
        is_float=True,
    )
    c_slider, c_box = htutori.build_widget_control(
        name="C",
        description="fight cost",
        min_val=1,
        max_val=10,
        step=0.5,
        initial_value=6,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the Hawk-Dove matrix and fitness plot from V and C.
        """
        _ = change
        with output:
            clear_output(wait=True)
            v = v_slider.value
            c = c_slider.value
            p1, p2 = _hawk_dove_matrix(v, c)
            analysis = _analyze_game(p1, p2)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            _render_matrix_panel(
                ax1,
                p1,
                p2,
                ("Hawk", "Dove"),
                ("Hawk", "Dove"),
                analysis,
                title="Hawk-Dove payoffs",
            )
            x = np.linspace(0, 1, 200)
            f_hawk = x * (v - c) / 2.0 + (1 - x) * v
            f_dove = (1 - x) * v / 2.0
            ax2.plot(x, f_hawk, label="Hawk fitness", color="firebrick")
            ax2.plot(x, f_dove, label="Dove fitness", color="steelblue")
            x_star = _ess_fraction(v, c)
            ax2.axvline(
                x_star,
                color="gray",
                linestyle="--",
                label=f"ESS x*={x_star:.2f}",
            )
            ax2.set_xlabel("population fraction playing Hawk (x)")
            ax2.set_ylabel("expected fitness")
            ax2.set_title("Fitness vs population share")
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax3.axis("off")
            ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            ess_kind = "pure Hawk" if v >= c else "mixed"
            text = (
                f"V: {v:.2f}\nC: {c:.2f}\nV/C: {v / c:.2f}\n\n"
                f"ESS type: {ess_kind}\n"
                f"ESS fraction Hawk: {x_star:.2f}"
            )
            htutori.add_fitted_text_box(
                ax3, text, max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    v_slider.observe(update_plot, names="value")
    c_slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([v_box, c_box, output]))


# #############################################################################
# Cell 6.2: Replicator Dynamics: Watching a Population Converge
# #############################################################################


def _replicator_trajectory(
    x0: float, v: float, c: float, n_generations: int, *, dt: float = 0.15
) -> np.ndarray:
    """
    Simulate the Hawk-Dove replicator dynamics via forward Euler integration.

    :param x0: Initial fraction of the population playing Hawk
    :param v: Value of the contested resource
    :param c: Cost of losing a fight
    :param n_generations: Number of discrete Euler steps to simulate
    :param dt: Step size for the Euler integration
    :return: Array of length `n_generations + 1` with the Hawk fraction at
        each generation, starting from `x0`
    """
    xs = [x0]
    x = x0
    for _ in range(n_generations):
        f_hawk = x * (v - c) / 2.0 + (1 - x) * v
        f_dove = (1 - x) * v / 2.0
        dx = x * (1 - x) * (f_hawk - f_dove)
        x = float(np.clip(x + dt * dx, 0.0, 1.0))
        xs.append(x)
    return np.array(xs)


def cell6_2_replicator_dynamics(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Interactive replicator-dynamics trajectory for the Hawk-Dove population.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    x0_slider, x0_box = htutori.build_widget_control(
        name="x0",
        description="initial Hawk fraction",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.1,
        is_float=True,
    )
    v_slider, v_box = htutori.build_widget_control(
        name="V",
        description="resource value",
        min_val=1,
        max_val=10,
        step=0.5,
        initial_value=4,
        is_float=True,
    )
    c_slider, c_box = htutori.build_widget_control(
        name="C",
        description="fight cost",
        min_val=1,
        max_val=10,
        step=0.5,
        initial_value=6,
        is_float=True,
    )
    ngen_slider, ngen_box = htutori.build_widget_control(
        name="n_generations",
        description="generations simulated",
        min_val=10,
        max_val=200,
        step=10,
        initial_value=60,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Resimulate and redraw the replicator trajectory from the sliders.
        """
        _ = change
        with output:
            clear_output(wait=True)
            x0 = x0_slider.value
            v = v_slider.value
            c = c_slider.value
            n = int(ngen_slider.value)
            xs = _replicator_trajectory(x0, v, c, n)
            x_star = _ess_fraction(v, c)
            _, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.6, 1]}
            )
            ax1.plot(np.arange(len(xs)), xs, color="firebrick", linewidth=2)
            ax1.axhline(
                x_star,
                color="gray",
                linestyle="--",
                label=f"ESS x*={x_star:.2f}",
            )
            ax1.set_xlabel("generation")
            ax1.set_ylabel("fraction playing Hawk")
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_title("Population trajectory")
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax2.axis("off")
            ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text = (
                f"generation: {n}\n"
                f"current Hawk fraction: {xs[-1]:.4f}\n"
                f"ESS fraction: {x_star:.4f}\n"
                f"distance to ESS: {abs(xs[-1] - x_star):.4f}"
            )
            htutori.add_fitted_text_box(
                ax2, text, max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    for slider in (x0_slider, v_slider, c_slider, ngen_slider):
        slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([x0_box, v_box, c_box, ngen_box, output]))


# #############################################################################
# Cell 7.1: Braess's Paradox: Adding a Road That Hurts Everyone
# #############################################################################

# Fixed layout for the 4-node Braess network (source, two middle nodes, sink).
_BRAESS_POS: Dict[str, Tuple[float, float]] = {
    "S": (0.0, 0.5),
    "A": (1.0, 1.0),
    "B": (1.0, 0.0),
    "T": (2.0, 0.5),
}
# Congestion (per-driver) cost on the two congestion-sensitive edges, and the
# fixed cost on the two constant-time edges.
_BRAESS_CONGESTION_SLOPE = 0.1
_BRAESS_CONST_EDGE_COST = 1.0


def _braess_costs(n_drivers: int, shortcut: bool) -> Dict[str, float]:
    """
    Compute equilibrium and social-optimum travel times for the network.

    Without the shortcut, the two routes S-A-T and S-B-T are symmetric, so
    both the Wardrop equilibrium and the social optimum split drivers evenly.
    With the shortcut, S-A-B-T weakly dominates both other routes, so every
    driver funnels through it at equilibrium; the social optimum still splits
    evenly, ignoring the shortcut.

    :param n_drivers: Total number of drivers
    :param shortcut: Whether the zero-cost A-B shortcut edge is present
    :return: dict with "equilibrium_per_driver", "social_optimum_per_driver",
        "equilibrium_total", "social_optimum_total", and "price_of_anarchy"
    """
    n = float(n_drivers)
    social_cost_per_driver = (
        n / 2.0
    ) * _BRAESS_CONGESTION_SLOPE + _BRAESS_CONST_EDGE_COST
    if shortcut:
        # All n drivers cross both congested edges (S-A and B-T).
        equilibrium_cost_per_driver = 2.0 * n * _BRAESS_CONGESTION_SLOPE
    else:
        equilibrium_cost_per_driver = social_cost_per_driver
    costs = {
        "equilibrium_per_driver": equilibrium_cost_per_driver,
        "social_optimum_per_driver": social_cost_per_driver,
        "equilibrium_total": equilibrium_cost_per_driver * n,
        "social_optimum_total": social_cost_per_driver * n,
        "price_of_anarchy": equilibrium_cost_per_driver / social_cost_per_driver,
    }
    return costs


def _render_braess_network(ax: matplotlib.axes.Axes, shortcut: bool) -> None:
    """
    Draw the 4-node Braess network, with the shortcut edge if present.

    :param ax: Axes to draw on
    :param shortcut: Whether to draw the zero-cost A-B shortcut edge
    """
    edges = [("S", "A"), ("A", "T"), ("S", "B"), ("B", "T")]
    labels = {
        ("S", "A"): "x/10",
        ("A", "T"): "1",
        ("S", "B"): "1",
        ("B", "T"): "x/10",
    }
    if shortcut:
        edges.append(("A", "B"))
        labels[("A", "B")] = "0"
    graph = nx.DiGraph(edges)
    edge_colors = [
        "#58D68D" if edge == ("A", "B") else "#5DADE2" for edge in graph.edges()
    ]
    nx.draw_networkx_nodes(
        graph,
        _BRAESS_POS,
        ax=ax,
        node_color="#F7DC6F",
        node_size=1400,
        edgecolors="black",
    )
    nx.draw_networkx_labels(
        graph, _BRAESS_POS, ax=ax, font_size=11, font_weight="bold"
    )
    nx.draw_networkx_edges(
        graph,
        _BRAESS_POS,
        ax=ax,
        edge_color=edge_colors,
        width=2.5,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.15",
    )
    nx.draw_networkx_edge_labels(
        graph, _BRAESS_POS, ax=ax, edge_labels=labels, font_size=9
    )
    ax.set_title(
        "Shortcut " + ("ON" if shortcut else "OFF"),
        fontsize=12,
        fontweight="bold",
    )
    ax.axis("off")


def cell7_1_braess_paradox(
    *, figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Interactive Braess's paradox: network diagram plus travel-time bars.

    :param figsize: Optional figure size, defaults to `plt.rcParams`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    shortcut_toggle = ipywidgets.ToggleButton(
        value=False,
        description="add shortcut edge (A-B, cost 0)",
        layout={"width": "280px"},
    )
    n_slider, n_box = htutori.build_widget_control(
        name="n_drivers",
        description="number of drivers",
        min_val=10,
        max_val=100,
        step=5,
        initial_value=50,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the network and travel-time bars from the current controls.
        """
        _ = change
        with output:
            clear_output(wait=True)
            n = int(n_slider.value)
            shortcut = shortcut_toggle.value
            no_shortcut_costs = _braess_costs(n, False)
            with_shortcut_costs = _braess_costs(n, True)
            current_costs = _braess_costs(n, shortcut)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            _render_braess_network(ax1, shortcut)
            labels = [
                "Equilibrium\n(no shortcut)",
                "Equilibrium\n(with shortcut)",
                "Social\noptimum",
            ]
            values = [
                no_shortcut_costs["equilibrium_total"],
                with_shortcut_costs["equilibrium_total"],
                no_shortcut_costs["social_optimum_total"],
            ]
            colors = ["#5DADE2", "#EC7063", "#58D68D"]
            ax2.bar(labels, values, color=colors)
            ax2.set_ylabel("total travel time")
            ax2.set_title("Total travel time")
            for idx, val in enumerate(values):
                ax2.text(
                    idx, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9
                )
            ax3.axis("off")
            ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text = (
                f"n_drivers: {n}\n"
                f"shortcut: {shortcut}\n\n"
                f"Equilibrium total: {current_costs['equilibrium_total']:.1f}\n"
                f"Social optimum total: "
                f"{current_costs['social_optimum_total']:.1f}\n"
                f"Price of anarchy: {current_costs['price_of_anarchy']:.2f}"
            )
            htutori.add_fitted_text_box(
                ax3, text, max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    shortcut_toggle.observe(update_plot, names="value")
    n_slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([shortcut_toggle, n_box, output]))
