"""
Notebook utilities for `mcts.03.API.ipynb`.

See `mcts_utils.py` for the game-agnostic MCTS engine, `game.py` for the
`Game` interface, `game_examples.py` for the concrete implementations these
widgets operate on, and `README.md` for a description of every file in this
directory.

Import as:

import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.mcts_API_utils as rimtsaazmau
"""

import logging
import random
from typing import TYPE_CHECKING, Callable, Optional

import ipywidgets
import matplotlib.axes
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

if TYPE_CHECKING:
    import graphviz

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import helpers.htutorial as htutori
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game as rimtsaazg
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game_API_utils as rimtsaazgau
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.mcts_utils as rimtsaazmu

_LOG = logging.getLogger(__name__)


# #############################################################################
# Cell 3.2: Search-tree visualization
# #############################################################################


def _build_tree_graph(
    root: rimtsaazmu.MCTSNode, best_move: rimtsaazg.Move
) -> "graphviz.Digraph":
    """
    Render `root` and its immediate children as a Graphviz tree.

    Only one ply is shown (root + children): deeper subtrees grow too large
    to read, and the immediate children are exactly what `run_mcts()` picks
    the best move from.

    :param root: fully-built MCTS root node (see `build_mcts_tree()`)
    :param best_move: move with the highest visit count, highlighted
    :return: Graphviz graph, rendered natively by Jupyter's `display()`
    """
    import graphviz

    dot = graphviz.Digraph()
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
    dot.node(
        "root",
        f"root\nN={root.visit_count}\nQ={root.mean_value:.2f}",
        fillcolor="#D9E8FB",
    )
    for move, child in sorted(root.children.items()):
        node_id = f"move_{move}"
        fillcolor = "#FDE9C8" if move == best_move else "#EFEFEF"
        dot.node(
            node_id,
            f"move={move}\nN={child.visit_count}\nQ={child.mean_value:.2f}",
            fillcolor=fillcolor,
        )
        dot.edge("root", node_id)
    return dot


def _plot_child_visits(
    ax: matplotlib.axes.Axes,
    root: rimtsaazmu.MCTSNode,
    best_move: rimtsaazg.Move,
) -> None:
    """
    Plot a bar chart of visit_count per root child, highest first.

    :param ax: axes to plot into
    :param root: fully-built MCTS root node
    :param best_move: move with the highest visit count, highlighted
    """
    children = sorted(
        root.children.items(), key=lambda item: item[1].visit_count, reverse=True
    )
    moves = [str(move) for move, _ in children]
    visits = [child.visit_count for _, child in children]
    colors = [
        "#4C9A2A" if move == best_move else "#7FA6D9" for move, _ in children
    ]
    ax.bar(moves, visits, color=colors)
    ax.set_xlabel("move (cell index)")
    ax.set_ylabel("visit count")
    ax.set_title("Visit count per root child")


def _build_tree_widget(
    game: rimtsaazg.Game,
    state: rimtsaazg.State,
    *,
    board_picture_fn: Optional[Callable[..., None]] = None,
) -> "ipywidgets.VBox":
    """
    Interactive widget: run MCTS from `state` and visualize the resulting
    search tree.

    Shared by `cell3_2_build_tree_widget()` (tic-tac-toe, text-only) and
    `cell6_2_build_connect_four_tree_widget()` (Connect Four, with a board
    picture): both let the user tune the two knobs from the UCT formula
    used by `run_mcts()` / `build_mcts_tree()`, Q(s, a) + C * sqrt(ln N(s) /
    N(s, a)):
    - `num_simulations`: the search budget (repeat until the budget, number
      of simulations, is exhausted)
    - `exploration_constant`: the `C` in the formula above
    and see how the visit count N(s, a) and mean value Q(s, a) of each root
    child respond.

    :param game: game-agnostic rules implementation
    :param state: state to search from, must not be terminal
    :param board_picture_fn: if given, a `(state, *, ax) -> None` function
        (e.g., `plot_connect_four_board()`) that draws `state` into an extra
        leftmost panel next to the visit-count bar chart, tying each
        child's move back to the board being searched
        - Default: `None` (bar chart + comments panel only, as used for
          tic-tac-toe, where the Graphviz tree diagram already shows enough
          of the board via move indices)
    :return: widget container to `display()` in a notebook cell
    """
    _LOG.debug(hprint.to_str("game state board_picture_fn"))
    hdbg.dassert(
        not game.is_terminal(state), "Cannot build a tree from a terminal state"
    )
    # Seed goes first, per notebook widget convention.
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=99,
        step=1,
        initial_value=0,
        is_float=False,
    )
    num_sim_exp_slider, num_sim_box = htutori.build_log_widget_control(
        name="log2(num_simulations)",
        description="search budget",
        min_exp=2,
        max_exp=9,
        initial_exp=7,
        base=2,
    )
    c_slider, c_box = htutori.build_widget_control(
        name="C",
        description="exploration constant",
        min_val=0.1,
        max_val=3.0,
        step=0.1,
        initial_value=rimtsaazmu.EXPLORATION_CONSTANT,
        is_float=True,
    )
    output = ipywidgets.Output()

    def update_plot(change: object = None) -> None:
        """
        Rebuild the tree from the current widget values and redraw it.
        """
        _ = change
        with output:
            clear_output(wait=True)
            # Read current widget values.
            seed = int(seed_slider.value)
            num_simulations = 2**num_sim_exp_slider.value
            exploration_constant = float(c_slider.value)
            # Build the tree, seeding first for reproducible rollouts.
            random.seed(seed)
            root = rimtsaazmu.build_mcts_tree(
                game,
                state,
                num_simulations=num_simulations,
                exploration_constant=exploration_constant,
            )
            best_move = max(
                root.children.items(), key=lambda item: item[1].visit_count
            )[0]
            # Tree diagram (Graphviz), then optionally a board picture, a
            # bar chart, and a comments panel.
            display(_build_tree_graph(root, best_move))
            num_panels = 3 if board_picture_fn is not None else 2
            _, axes = plt.subplots(
                1, num_panels, figsize=(5.5 * num_panels, 4.5)
            )
            if board_picture_fn is not None:
                ax_board, ax_bar, ax_comment = axes
                board_picture_fn(state, ax=ax_board)
                ax_board.set_title("Board being searched")
            else:
                ax_bar, ax_comment = axes
            _plot_child_visits(ax_bar, root, best_move)
            best_child = root.children[best_move]
            comment_text = (
                "Parameters:\n"
                f"  seed: {seed}\n"
                f"  num_simulations: {num_simulations}\n"
                f"  C: {exploration_constant:.2f}\n"
                "\n"
                "Root stats:\n"
                f"  root visit_count N(s): {root.visit_count}\n"
                f"  robust child (highest N): {best_move}\n"
                f"  N(s, a) of robust child: {best_child.visit_count}\n"
                f"  Q(s, a) of robust child: {best_child.mean_value:.2f}"
            )
            ax_comment.axis("off")
            ax_comment.set_title(
                "Comments", fontsize=14, fontweight="bold", pad=20
            )
            htutori.add_fitted_text_box(ax_comment, comment_text)
            plt.tight_layout()
            plt.show()

    # Attach observers to all widgets.
    seed_slider.observe(update_plot, names="value")
    num_sim_exp_slider.observe(update_plot, names="value")
    c_slider.observe(update_plot, names="value")
    # Initial render.
    update_plot()
    widget = ipywidgets.VBox([seed_box, num_sim_box, c_box, output])
    return widget


def cell3_2_build_tree_widget(
    game: rimtsaazg.Game, state: rimtsaazg.State
) -> "ipywidgets.VBox":
    """
    Interactive widget: run MCTS from `state` and visualize the resulting
    search tree as a Graphviz diagram plus a visit-count bar chart.

    See `_build_tree_widget()` for the shared implementation.

    :param game: game-agnostic rules implementation
    :param state: state to search from, must not be terminal
    :return: widget container to `display()` in a notebook cell
    """
    widget = _build_tree_widget(game, state)
    return widget


# #############################################################################
# Cell 6.2: Connect Four search-tree visualization
# #############################################################################


def cell6_2_build_connect_four_tree_widget(
    game: rimtsaazg.Game, state: rimtsaazg.State
) -> "ipywidgets.VBox":
    """
    Interactive widget: run MCTS from a Connect Four `state` and visualize
    the resulting search tree next to a picture of the board being
    searched.

    Same knobs and Graphviz/bar-chart layout as `cell3_2_build_tree_widget()`;
    the extra `game_API_utils.plot_connect_four_board()` panel ties each
    child's move (a column) back to where the disc would land.

    :param game: game-agnostic rules implementation, `ConnectFour`
    :param state: state to search from, must not be terminal
    :return: widget container to `display()` in a notebook cell
    """
    widget = _build_tree_widget(
        game, state, board_picture_fn=rimtsaazgau.plot_connect_four_board
    )
    return widget
