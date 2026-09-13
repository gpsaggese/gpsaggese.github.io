"""
Notebook utilities for `search_algorithms.02.API.ipynb`.

See `search_algorithms_utils.py` for `SearchNode` and the classical search
algorithms these widgets replay, `game.py` for the `Game` interface,
`game_examples.py` for the concrete games these widgets operate on, and
`README.md` for a description of every file in this directory.

Import as:

import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.search_algorithms_API_utils as rimtsaazsaau
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import ipywidgets
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

if TYPE_CHECKING:
    import graphviz

import helpers.hprint as hprint
import helpers.htutorial as htutori
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game as rimtsaazg
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.search_algorithms_utils as rimtsaazsau

_LOG = logging.getLogger(__name__)

# A minimax event, in the exact order `search_algorithms_utils._minimax()`
# creates ("visit") and backs up the value of ("value") each `SearchNode`:
# `(event_type, node)`.
_Event = Tuple[str, rimtsaazsau.SearchNode]


# #############################################################################
# Cell 4.2: Replaying how build_minimax_tree() builds a tree
# #############################################################################


def _flatten_minimax_events(root: rimtsaazsau.SearchNode) -> List[_Event]:
    """
    Replay `root`'s construction order as a flat list of events.

    `root` is already fully built (by `build_minimax_tree()`), so every
    `.value` is already known; this only recovers the *order* in which
    `_minimax()` created each node ("visit", pre-order: a node is created
    right before its own subtree is explored) and backed up its value
    ("value", post-order: only after every child is done). Replaying that
    order is what lets the widget below reveal the tree "little by little"
    instead of all at once.

    :param root: root of a tree already built by `build_minimax_tree()`
    :return: chronological list of `(event_type, node)` events
    """
    events: List[_Event] = []

    def walk(node: rimtsaazsau.SearchNode) -> None:
        events.append(("visit", node))
        for child in node.children:
            walk(child)
        events.append(("value", node))

    walk(root)
    return events


def _build_partial_tree_graph(
    events: List[_Event],
    num_events: int,
    *,
    game: Optional[rimtsaazg.Game] = None,
) -> "graphviz.Digraph":
    """
    Render only the prefix `events[:num_events]` as a Graphviz tree.

    Node color follows the same scheme as `build_tree_graph()`, plus one
    extra state `build_tree_graph()` never needs (since it only ever draws
    finished trees): light grey for a node that has been created but whose
    subtree has not finished backing up a value yet.

    :param events: full event list from `_flatten_minimax_events()`
    :param num_events: number of leading events to render
    :param game: switch to also render each node's board via
        `game.render(node.state)`, same knob as `build_tree_graph(game=...)`
        - Default: `None` (no board)
    :return: Graphviz graph, rendered natively by Jupyter's `display()`
    """
    import graphviz

    prefix = events[:num_events]
    # A node has a Graphviz id once it has been "visited" (created); it is
    # "valued" once its own "value" event has also played.
    node_ids: Dict[int, str] = {}
    valued_ids = set()
    for event_type, node in prefix:
        if event_type == "visit":
            node_ids[id(node)] = f"n{len(node_ids)}"
        else:
            valued_ids.add(id(node))

    dot = graphviz.Digraph()
    # A rendered board relies on fixed-width alignment (e.g. "X X ."); the
    # default Helvetica is not monospace and would skew it.
    fontname = "Courier" if game is not None else "Helvetica"
    dot.attr("node", shape="box", fontname=fontname)
    for event_type, node in prefix:
        if event_type != "visit":
            continue
        prefix_label = "root" if node.parent is None else f"move={node.move}"
        if id(node) in valued_ids:
            # The real tree already knows whether this node has children;
            # only *when* that becomes visible in the animation changes.
            fillcolor = "#A9DDB0" if not node.children else "#D9E8FB"
            label = f"{prefix_label}\nvalue={node.value:.2f}"
        else:
            # Created, but its subtree has not finished backing up a value.
            fillcolor = "#EFEFEF"
            label = f"{prefix_label}\nvalue=?"
        if game is not None:
            label = f"{game.render(node.state)}\n{label}"
        dot.node(
            node_ids[id(node)],
            label,
            fillcolor=fillcolor,
            style="rounded,filled",
        )
        if node.parent is not None and id(node.parent) in node_ids:
            dot.edge(node_ids[id(node.parent)], node_ids[id(node)])
    return dot


def _build_minimax_step_widget(
    game: rimtsaazg.Game, states: Dict[str, rimtsaazg.State]
) -> "ipywidgets.VBox":
    """
    Interactive widget: replay `build_minimax_tree()`'s construction of a
    state's search tree one event at a time.

    Shared by `cell4_2_build_minimax_step_widget()` (tic-tac-toe, a
    `demo_state` / `fork_state` switch) and
    `cell5_1_build_connect_three_step_widget()` (Connect Three, a single
    state): both build the *real* tree via `build_minimax_tree()` once,
    then let the user scrub through the exact order `_minimax()` created
    and backed up each node, via `_flatten_minimax_events()` above. A
    "show board" checkbox switches `game.render(node.state)` on each node
    on and off, the same `game=...` knob `build_tree_graph()` takes.

    :param game: game-agnostic rules implementation
    :param states: `{name: state}` choices for the "state" dropdown, e.g.,
        `{"demo_state": demo_state, "fork_state": fork_state}`
    :return: widget container to `display()` in a notebook cell
    """
    _LOG.debug(hprint.to_str("game states"))
    state_dropdown = ipywidgets.Dropdown(
        options=list(states.keys()),
        value=next(iter(states)),
        description="state",
    )
    # Placeholder range; `update_plot()` resizes it to the selected state's
    # actual event count on every call, including the first.
    step_slider, step_box = htutori.build_widget_control(
        name="step",
        description="events replayed",
        min_val=1,
        max_val=1,
        step=1,
        initial_value=1,
        is_float=False,
    )
    show_board_checkbox = ipywidgets.Checkbox(
        value=True, description="show board", indent=False
    )
    output = ipywidgets.Output()
    # Cache `(root, events)` per state name: `build_minimax_tree()` runs
    # once per state, not once per slider tick.
    cache: Dict[str, Tuple[rimtsaazsau.SearchNode, List[_Event]]] = {}

    def get_events(name: str) -> Tuple[rimtsaazsau.SearchNode, List[_Event]]:
        if name not in cache:
            root = rimtsaazsau.build_minimax_tree(game, states[name])
            cache[name] = (root, _flatten_minimax_events(root))
        return cache[name]

    def state_changed(change: object = None) -> None:
        # A new state has its own event count; start its animation from
        # the first event rather than carrying over the old step.
        _ = change
        step_slider.value = 1
        update_plot()

    def update_plot(change: object = None) -> None:
        _ = change
        root, events = get_events(state_dropdown.value)
        step_slider.max = len(events)
        num_events = int(min(step_slider.value, len(events)))
        board_game = game if show_board_checkbox.value else None
        comment_text = (
            "Parameters:\n"
            f"  state: {state_dropdown.value}\n"
            f"  step: {num_events}/{len(events)}\n"
        )
        if num_events >= len(events):
            best_move = rimtsaazsau.pick_best_move(game, root)
            comment_text += (
                "\n"
                "Done:\n"
                f"  root value Q(s0): {root.value:.2f}\n"
                f"  best move: {best_move}"
            )
        with output:
            clear_output(wait=True)
            # The tree diagram (left) and the Comments panel (right) sit
            # side by side: a Graphviz digraph is not a matplotlib artist,
            # so each half gets its own `Output()`, arranged via `HBox`.
            graph_output = ipywidgets.Output()
            with graph_output:
                display(
                    _build_partial_tree_graph(
                        events, num_events, game=board_game
                    )
                )
            comment_output = ipywidgets.Output()
            with comment_output:
                _, ax_comment = plt.subplots(figsize=(4, 4.5))
                ax_comment.axis("off")
                ax_comment.set_title(
                    "Comments", fontsize=14, fontweight="bold", pad=20
                )
                htutori.add_fitted_text_box(ax_comment, comment_text)
                plt.tight_layout()
                plt.show()
            display(ipywidgets.HBox([graph_output, comment_output]))

    state_dropdown.observe(state_changed, names="value")
    step_slider.observe(update_plot, names="value")
    show_board_checkbox.observe(update_plot, names="value")
    update_plot()
    widget = ipywidgets.VBox(
        [state_dropdown, step_box, show_board_checkbox, output]
    )
    return widget


def cell4_2_build_minimax_step_widget(
    game: rimtsaazg.Game, states: Dict[str, rimtsaazg.State]
) -> "ipywidgets.VBox":
    """
    Interactive widget: grow `demo_state` / `fork_state`'s minimax tree one
    node at a time instead of all at once.

    See `_build_minimax_step_widget()` for the shared implementation.

    :param game: game-agnostic rules implementation
    :param states: `{"demo_state": demo_state, "fork_state": fork_state}`
    :return: widget container to `display()` in a notebook cell
    """
    widget = _build_minimax_step_widget(game, states)
    return widget


# #############################################################################
# Cell 5.1: Connect Three step-by-step widget
# #############################################################################


def cell5_1_build_connect_three_step_widget(
    game: rimtsaazg.Game, state: rimtsaazg.State
) -> "ipywidgets.VBox":
    """
    Interactive widget: grow a Connect Three state's minimax tree one node
    at a time, exactly like `cell4_2_build_minimax_step_widget()` above.

    Same widget, a single-state dropdown: the point of this cell is
    generalizing to a second, gravity-drop game, not a second position.

    :param game: game-agnostic rules implementation, `ConnectThree`
    :param state: state to search from, must not be terminal
    :return: widget container to `display()` in a notebook cell
    """
    widget = _build_minimax_step_widget(game, {"connect3_demo_state": state})
    return widget
