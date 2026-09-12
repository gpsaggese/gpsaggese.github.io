"""
Utility functions for the propositional and first-order logic solver lesson.

Runs the lecture's own examples through three engines instead of one:
- `sympy` for symbolic rewriting: truth tables, equivalences, CNF and DNF.
- `PySAT` for propositional satisfiability: DIMACS clauses and real solvers.
- `z3` for first-order reasoning: quantifiers, predicates, and models.

Import as:

import msml610.tutorials.L03_knowledge_representation.L03_06_logic_solvers_utils as mtlkrl0lsu
"""

import logging
import os
import textwrap
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ipywidgets
import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pysat.formula
import pysat.solvers
import sympy
import z3
from IPython.display import clear_output, display

import helpers.hnotebook as hnotebo
import helpers.htutorial as htutori

_LOG = logging.getLogger(__name__)

# #############################################################################
# Colors and display constants
# #############################################################################

# Colors reused across the panels, one per engine so that the same engine
# always reads with the same color.
_COLOR_SYMPY = "#3182bd"
_COLOR_PYSAT = "#e6550d"
_COLOR_Z3 = "#31a354"
_COLOR_NEUTRAL = "#444444"
# The lecture's weather world, used for every propositional example.
_WEATHER_SYMBOL_NAMES = ("Rain", "Cold", "Sunny", "Snow", "Cloudy")
# Names `PySAT` knows its backends by, keyed by the label shown in the widget.
_SOLVER_NAMES = {"Minisat22": "minisat22", "Glucose3": "glucose3"}


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Propositional sentences with `sympy`
# #############################################################################


def weather_symbols() -> Dict[str, Any]:
    """
    Build the lecture's weather proposition symbols.

    Each symbol is atomic: it carries no internal structure and no meaning
    until the notebook grounds it, e.g., `Rain` standing for "it's raining".

    :return: map from symbol name to the `sympy` symbol
    """
    symbols = sympy.symbols(" ".join(_WEATHER_SYMBOL_NAMES))
    result: Dict[str, Any] = {
        str(name): symbol for name, symbol in zip(_WEATHER_SYMBOL_NAMES, symbols)
    }
    return result


def weather_sentences() -> Dict[str, Any]:
    """
    Build the catalog of weather sentences used across the notebook.

    The catalog covers every connective in the lecture: conjunction,
    disjunction, negation, implication, and the biconditional. Each label is
    the sentence as `format_sentence()` prints it, so what a widget shows and
    what a panel prints never drift apart: `sympy` normalizes the argument
    order of `And` and `Or` while the sentence is built.

    :return: map from a display label to the `sympy` sentence
    """
    symbols = weather_symbols()
    rain = symbols["Rain"]
    cold = symbols["Cold"]
    sunny = symbols["Sunny"]
    snow = symbols["Snow"]
    cloudy = symbols["Cloudy"]
    catalog = [
        sympy.And(rain, cloudy),
        sympy.Or(rain, snow),
        sympy.Not(sympy.Or(rain, cloudy)),
        sympy.Implies(rain, sympy.Not(snow)),
        sympy.Implies(rain, sunny),
        sympy.Equivalent(sunny, sympy.Not(cloudy)),
        sympy.Implies(sympy.And(rain, cold), snow),
        sympy.Equivalent(snow, sympy.And(cold, cloudy)),
    ]
    sentences = {format_sentence(sentence): sentence for sentence in catalog}
    return sentences


def format_sentence(sentence: sympy.Basic) -> str:
    """
    Render a `sympy` sentence in the lecture's own infix notation.

    `sympy`'s printer spells out `Implies(A, B)` and `Equivalent(A, B)`,
    which is harder to read than `A => B` and `A <=> B`.

    :param sentence: sentence to render
    :return: string form, e.g., `Rain => ~Snow`
    """
    if isinstance(sentence, sympy.Implies):
        text = " => ".join(_format_arg(arg) for arg in sentence.args)
    elif isinstance(sentence, sympy.Equivalent):
        text = " <=> ".join(_format_arg(arg) for arg in sentence.args)
    elif isinstance(sentence, sympy.And):
        text = " & ".join(_format_arg(arg) for arg in sentence.args)
    elif isinstance(sentence, sympy.Or):
        text = " | ".join(_format_arg(arg) for arg in sentence.args)
    elif isinstance(sentence, sympy.Not):
        text = "~" + _format_arg(sentence.args[0])
    else:
        text = str(sentence)
    return text


def _format_arg(arg: sympy.Basic) -> str:
    """
    Render a sub-sentence, parenthesized when it is itself complex.

    :param arg: sub-sentence to render
    :return: string form, parenthesized if the sub-sentence has a connective
    """
    text = format_sentence(arg)
    is_complex = isinstance(
        arg, (sympy.And, sympy.Or, sympy.Implies, sympy.Equivalent)
    )
    if is_complex:
        text = "(" + text + ")"
    return text


def sentence_symbols(sentence: sympy.Basic) -> List[Any]:
    """
    List the proposition symbols a sentence mentions, in name order.

    :param sentence: sentence to inspect
    :return: symbols sorted by name, e.g., `[Cloudy, Rain]`
    """
    symbols = sorted(sentence.free_symbols, key=str)
    return symbols


def enumerate_assignments(n_vars: int) -> np.ndarray:
    """
    Enumerate every truth assignment over `n_vars` proposition symbols.

    Row `i` holds the binary expansion of `i`, so the rows are exactly the
    $2^n$ models a truth table has to list.

    :param n_vars: number of proposition symbols
    :return: boolean array of shape `(2 ** n_vars, n_vars)`
    """
    index = np.arange(1 << n_vars, dtype=np.int64)
    bits = ((index[:, None] >> np.arange(n_vars)) & 1).astype(bool)
    return bits


def truth_values(
    sentence: sympy.Basic,
    var_order: Sequence[Any],
    bits: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a sentence in every model, by direct substitution.

    This is compositional semantics applied literally: substitute the truth
    values the model assigns to the symbols, then let `sympy` fold the
    connectives from the leaves up.

    :param sentence: sentence to evaluate
    :param var_order: symbols, in the column order of `bits`
    :param bits: assignments from `enumerate_assignments()`
    :return: boolean array with one entry per model
    """
    n_models = bits.shape[0]
    values = np.zeros(n_models, dtype=bool)
    for row in range(n_models):
        subs = [(var, bool(bits[row, i])) for i, var in enumerate(var_order)]
        values[row] = bool(sentence.subs(subs))
    return values


def count_clauses(sentence: sympy.Basic) -> int:
    """
    Count the top-level conjuncts (CNF) or disjuncts (DNF) of a sentence.

    A CNF sentence is a conjunction of clauses, so its clause count is the
    number of `And` arguments; a lone clause counts as one.

    :param sentence: normal-form sentence
    :return: number of clauses
    """
    if isinstance(sentence, (sympy.And, sympy.Or)):
        count = len(sentence.args)
    else:
        count = 1
    return count


def is_equivalent(alpha: sympy.Basic, beta: sympy.Basic) -> bool:
    """
    Check `alpha == beta` the semantic way, as the lecture defines it.

    Two sentences are logically equivalent iff they are true in the same
    models, which holds iff `~(alpha <=> beta)` has no model at all.

    :param alpha: first sentence
    :param beta: second sentence
    :return: True if the two sentences share the same models
    """
    counterexample = sympy.satisfiable(sympy.Not(sympy.Equivalent(alpha, beta)))
    result = counterexample is False
    return result


# #############################################################################
# From `sympy` sentences to DIMACS clauses
# #############################################################################


def cnf_to_clauses(
    sentence: sympy.Basic,
) -> Tuple[List[List[int]], Dict[str, int]]:
    """
    Convert a sentence into the integer clauses a SAT solver consumes.

    The sentence is first rewritten to CNF, then each symbol is replaced by
    its DIMACS id and each negated literal by the negative of that id, e.g.,
    `(Rain | ~Snow) & Cloudy` with `Rain=1, Snow=3, Cloudy=2` becomes
    `[[1, -3], [2]]`.

    :param sentence: sentence to encode, in any form
    :return: tuple of
        - clauses, each a list of signed integers
        - map from symbol name to its DIMACS integer id
    """
    cnf = sympy.to_cnf(sentence, simplify=False)
    symbol_map = {
        str(symbol): index + 1
        for index, symbol in enumerate(sentence_symbols(cnf))
    }
    # A CNF sentence is an `And` of clauses, unless it is a single clause.
    conjuncts = cnf.args if isinstance(cnf, sympy.And) else (cnf,)
    clauses = []
    for conjunct in conjuncts:
        # A clause is an `Or` of literals, unless it is a single literal.
        literals = (
            conjunct.args if isinstance(conjunct, sympy.Or) else (conjunct,)
        )
        clause = []
        for literal in literals:
            if isinstance(literal, sympy.Not):
                clause.append(-symbol_map[str(literal.args[0])])
            else:
                clause.append(symbol_map[str(literal)])
        clauses.append(clause)
    return clauses, symbol_map


def clauses_to_dimacs(clauses: List[List[int]], file_name: str) -> str:
    """
    Write clauses to a DIMACS file and read the file back as text.

    DIMACS is the exchange format SAT solvers and solver competitions have
    shared since the 1990s: one header line, then one clause per line,
    terminated by `0`.

    :param clauses: clauses, each a list of signed integers
    :param file_name: path of the file to write
    :return: content of the written file
    """
    cnf = pysat.formula.CNF(from_clauses=clauses)
    cnf.to_file(file_name)
    with open(file_name, "r") as file_in:
        content = file_in.read()
    return content


def solve_clauses(
    clauses: List[List[int]],
    *,
    solver_label: str = "Minisat22",
) -> Tuple[bool, List[int], float]:
    """
    Decide satisfiability of a set of clauses with a real SAT solver.

    :param clauses: clauses, each a list of signed integers
    :param solver_label: backend to use, a key of `_SOLVER_NAMES`
    :return: tuple of
        - True if the clauses are satisfiable
        - satisfying assignment as signed integers, empty when unsatisfiable
        - elapsed time in seconds
    """
    solver_name = _SOLVER_NAMES[solver_label]
    with pysat.solvers.Solver(
        name=solver_name, bootstrap_with=clauses
    ) as solver:
        # Time the search itself, not the clause loading, so the numbers
        # answer "how hard was this instance" and not "how big was it".
        start = time.time()
        is_sat = solver.solve()
        elapsed = time.time() - start
        raw_model = solver.get_model() if is_sat else None
    model = list(raw_model) if raw_model else []
    return bool(is_sat), model, elapsed


def decode_model(model: Sequence[int], symbol_map: Dict[str, int]) -> str:
    """
    Translate a solver model back from integers to symbol names.

    :param model: satisfying assignment as signed integers
    :param symbol_map: map from symbol name to its DIMACS integer id
    :return: one `name=T/F` entry per symbol, comma separated
    """
    assignment = {abs(literal): literal > 0 for literal in model}
    entries = [
        "%s=%s" % (name, "T" if assignment.get(index, False) else "F")
        for name, index in sorted(symbol_map.items(), key=lambda kv: kv[1])
    ]
    result = ", ".join(entries)
    return result


def entails_by_refutation(
    kb: Sequence[Any],
    alpha: Any,
    *,
    solver_label: str = "Minisat22",
) -> Tuple[bool, str]:
    """
    Decide `KB |= alpha` by refutation, the lecture's proof by contradiction.

    The query is entailed iff `KB & ~alpha` is unsatisfiable: a satisfying
    model of that sentence is a world where the `KB` holds and the query
    fails, i.e., a counterexample.

    :param kb: sentences forming the knowledge base
    :param alpha: query sentence
    :param solver_label: backend to use, a key of `_SOLVER_NAMES`
    :return: tuple of
        - True if `KB` entails `alpha`
        - the counterexample model, or `"none"` when entailment holds
    """
    refutation = sympy.And(*kb, sympy.Not(alpha))
    clauses, symbol_map = cnf_to_clauses(refutation)
    is_sat, model, _ = solve_clauses(clauses, solver_label=solver_label)
    counterexample = decode_model(model, symbol_map) if is_sat else "none"
    entailed = not is_sat
    return entailed, counterexample


# #############################################################################
# Drawing helpers
# #############################################################################


def comment_panel(ax: matplotlib.axes.Axes, text: str) -> None:
    """
    Render a comment panel showing the current variable state.

    :param ax: axes to draw on
    :param text: comment text
    """
    ax.axis("off")
    ax.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    htutori.add_fitted_text_box(ax, text, max_fontsize=12, min_fontsize=6)


def wrap_text(text: str, *, width: int = 46, indent: str = "  ") -> str:
    """
    Wrap a long formula so that it stays inside a comment panel.

    :param text: text to wrap, e.g., a CNF sentence
    :param width: maximum characters per line
    :param indent: prefix added to every continuation line
    :return: wrapped text, with embedded newlines
    """
    wrapped = textwrap.wrap(text, width=width) or [""]
    result = ("\n" + indent).join(wrapped)
    return result


def make_param_info(descriptions: Dict[str, str]) -> ipywidgets.HTML:
    """
    Build a styled HTML info box describing the controls of a cell.

    :param descriptions: map from parameter name to its description text
    :return: styled HTML widget ready for display
    """
    body = "\n".join(
        "<b>%s</b>: %s<br>" % (name, desc) for name, desc in descriptions.items()
    )
    html = (
        "<div style='background:#f5f5f5; padding:10px 14px; "
        "border-radius:4px; border-left:3px solid #4682b4; "
        "font-size:13px; line-height:1.8'>"
        "%s"
        "</div>"
    ) % body
    return ipywidgets.HTML(html)


def draw_text_panel(
    ax: matplotlib.axes.Axes,
    lines: Sequence[Tuple[str, str, str]],
    *,
    title: str,
) -> None:
    """
    Draw a panel of left-aligned monospace lines, one color per line.

    :param ax: axes to draw on
    :param lines: triples of `(text, color, fontweight)`
    :param title: panel title
    """
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")
    # Short panels keep a fixed line spacing instead of spreading out over
    # the whole panel, which would read as unrelated blocks of text.
    step = min(1.0 / (len(lines) + 1), 0.055)
    for index, (text, color, weight) in enumerate(lines):
        ax.text(
            0.02,
            0.97 - index * step,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=9,
            color=color,
            fontweight=weight,
        )


def draw_truth_table(
    ax: matplotlib.axes.Axes,
    bits: np.ndarray,
    var_names: Sequence[str],
    values: np.ndarray,
    *,
    value_name: str,
    highlight_row: int,
    title: str,
) -> None:
    """
    Draw the truth table of a sentence, one row per model.

    The last column holds the sentence's own truth value, and the model the
    toggles select is outlined, so the widget and the table stay connected.

    :param ax: axes to draw on
    :param bits: assignments from `enumerate_assignments()`
    :param var_names: column headers, one per symbol
    :param values: truth value of the sentence in each model
    :param value_name: header of the sentence column
    :param highlight_row: index of the model to outline, `-1` for none
    :param title: panel title
    """
    n_rows, n_vars = bits.shape
    columns = [bits[:, i].astype(int) for i in range(n_vars)]
    columns.append(np.where(values, 2, 3))
    headers = list(var_names) + [value_name]
    image = np.stack(columns, axis=1)
    cmap = mcolors.ListedColormap(
        [
            "#f7f7f7",  # 0: symbol is false.
            "#c6dbef",  # 1: symbol is true.
            "#a1d99b",  # 2: sentence is true.
            "#fcbba1",  # 3: sentence is false.
        ]
    )
    ax.imshow(image, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    # Write the truth values in the cells, the sentence column in bold.
    for row in range(n_rows):
        for col in range(n_vars):
            ax.text(
                col,
                row,
                "T" if bits[row, col] else "F",
                ha="center",
                va="center",
                fontsize=8,
            )
        ax.text(
            n_vars,
            row,
            "T" if values[row] else "F",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    # Outline the model the toggles currently select.
    if 0 <= highlight_row < n_rows:
        ax.add_patch(
            mpatches.Rectangle(
                (-0.5, highlight_row - 0.5),
                n_vars + 1,
                1.0,
                fill=False,
                edgecolor=_COLOR_PYSAT,
                linewidth=2.0,
            )
        )
    ax.set_xticks(range(len(headers)))
    ax.set_xticklabels(headers, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(["m%d" % (row + 1) for row in range(n_rows)], fontsize=7)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("one row per model, outline: the toggled model", fontsize=9)
    ax.grid(False)


# #############################################################################
# Cell 1.1: Sentences as symbols, parsed and evaluated by `sympy`
# #############################################################################


def _node_label(sentence: sympy.Basic) -> str:
    """
    Label a parse-tree node with its connective, or its symbol name.

    :param sentence: sub-sentence at this node
    :return: label, e.g., `&` for a conjunction, `Rain` for a symbol
    """
    if isinstance(sentence, sympy.And):
        label = "&"
    elif isinstance(sentence, sympy.Or):
        label = "|"
    elif isinstance(sentence, sympy.Not):
        label = "~"
    elif isinstance(sentence, sympy.Implies):
        label = "=>"
    elif isinstance(sentence, sympy.Equivalent):
        label = "<=>"
    else:
        label = str(sentence)
    return label


def _collect_parse_tree(
    sentence: sympy.Basic,
    subs: Sequence[Tuple[sympy.Symbol, bool]],
    depth: int,
    next_x: List[float],
    nodes: List[Dict[str, Any]],
    edges: List[Tuple[int, int]],
) -> int:
    """
    Walk a sentence bottom-up, recording node positions and truth values.

    Leaves are laid out left to right in the order they are visited, and an
    internal node sits above the average position of its children, so the
    drawing mirrors the recursive structure of the sentence.

    :param sentence: sub-sentence at this node
    :param subs: substitution pairs fixing every symbol to a truth value
    :param depth: depth of this node, `0` at the root
    :param next_x: single-entry list holding the next free leaf position
    :param nodes: accumulator of node records, mutated in place
    :param edges: accumulator of `(parent, child)` pairs, mutated in place
    :return: index of the node just added
    """
    child_indices = [
        _collect_parse_tree(arg, subs, depth + 1, next_x, nodes, edges)
        for arg in sentence.args
    ]
    if child_indices:
        x_position = float(np.mean([nodes[i]["x"] for i in child_indices]))
    else:
        x_position = next_x[0]
        next_x[0] += 1.0
    # Evaluate this sub-sentence in the current model, which is what makes
    # truth visibly propagate from the leaves to the root.
    node = {
        "x": x_position,
        "y": -float(depth),
        "label": _node_label(sentence),
        "value": bool(sentence.subs(subs)),
    }
    index = len(nodes)
    nodes.append(node)
    for child_index in child_indices:
        edges.append((index, child_index))
    return index


def draw_parse_tree(
    ax: matplotlib.axes.Axes,
    sentence: sympy.Basic,
    subs: Sequence[Tuple[sympy.Symbol, bool]],
    *,
    title: str,
) -> None:
    """
    Draw the parse tree of a sentence, colored by the current model.

    :param ax: axes to draw on
    :param sentence: sentence to parse
    :param subs: substitution pairs fixing every symbol to a truth value
    :param title: panel title
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Tuple[int, int]] = []
    _collect_parse_tree(sentence, subs, 0, [0.0], nodes, edges)
    for parent, child in edges:
        ax.plot(
            [nodes[parent]["x"], nodes[child]["x"]],
            [nodes[parent]["y"], nodes[child]["y"]],
            color="#999999",
            linewidth=1.2,
            zorder=1,
        )
    for node in nodes:
        color = "#a1d99b" if node["value"] else "#fcbba1"
        ax.scatter(
            node["x"],
            node["y"],
            s=900,
            color=color,
            edgecolor=_COLOR_NEUTRAL,
            zorder=2,
        )
        ax.text(
            node["x"],
            node["y"],
            node["label"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=3,
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(
        "green: sub-sentence true in the toggled model, red: false",
        fontsize=9,
    )
    ax.margins(0.2)
    ax.axis("off")


def cell1_1_sentences_and_truth_tables(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Parse a weather sentence with `sympy` and evaluate it in every model.

    Interactive controls (ipywidgets):
    - `sentence`: the weather sentence to parse and evaluate
    - one checkbox per weather symbol: picks the model read off in Comments

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (17, 5)
    sentences = weather_sentences()
    sentence_dropdown = ipywidgets.Dropdown(
        options=list(sentences.keys()),
        value="Rain => ~Snow",
        description="sentence:",
        style={"description_width": "initial"},
    )
    symbol_checkboxes = {
        name: ipywidgets.Checkbox(
            value=(name == "Rain"),
            description=name,
            indent=False,
            layout=ipywidgets.Layout(width="110px"),
        )
        for name in _WEATHER_SYMBOL_NAMES
    }
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            label = sentence_dropdown.value
            sentence = sentences[label]
            var_order = sentence_symbols(sentence)
            var_names = [str(var) for var in var_order]
            bits = enumerate_assignments(len(var_order))
            values = truth_values(sentence, var_order, bits)
            # The toggles fix one model; only the symbols the sentence
            # mentions select a row of its truth table.
            subs = [
                (var, bool(symbol_checkboxes[str(var)].value))
                for var in var_order
            ]
            row_bits = np.array([value for _, value in subs], dtype=bool)
            highlight_row = int(
                np.flatnonzero((bits == row_bits).all(axis=1))[0]
            )
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            draw_truth_table(
                ax1,
                bits,
                var_names,
                values,
                value_name=label,
                highlight_row=highlight_row,
                title="Truth table",
            )
            draw_parse_tree(ax2, sentence, subs, title="Parse tree")
            model_text = ", ".join(
                "%s=%s" % (name, "T" if value else "F")
                for name, value in zip(var_names, row_bits)
            )
            text = (
                "sentence: %s\n\n"
                "atoms: %s\n"
                "models: %d rows (2^%d)\n"
                "models where true: %d\n\n"
                "toggled model:\n  %s\n"
                "value there: %s"
                % (
                    format_sentence(sentence),
                    ", ".join(var_names),
                    bits.shape[0],
                    len(var_order),
                    int(values.sum()),
                    model_text,
                    "True" if values[highlight_row] else "False",
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "sentence": "the weather sentence that gets parsed, evaluated, "
            "and tabulated",
            "Rain, Cold, Sunny, Snow, Cloudy": "one checkbox per symbol, "
            "together fixing the model <code>m</code> whose row is outlined",
        }
    )
    sentence_dropdown.observe(update_plot, names="value")
    for checkbox in symbol_checkboxes.values():
        checkbox.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [
            sentence_dropdown,
            ipywidgets.HBox(list(symbol_checkboxes.values())),
        ],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 1.2: Equivalences and normal forms with `sympy`
# #############################################################################


def equivalence_laws() -> Dict[str, Tuple[Any, Any]]:
    """
    Build the lecture's equivalence laws as `(left, right)` sentence pairs.

    The laws are stated over generic symbols `A`, `B`, `C`, exactly as the
    lecture states them.

    :return: map from law name to the two sides of the equivalence
    """
    a_sym, b_sym, c_sym = sympy.symbols("A B C")
    laws: Dict[str, Tuple[Any, Any]] = {
        "De Morgan": (
            sympy.Not(sympy.And(a_sym, b_sym)),
            sympy.Or(sympy.Not(a_sym), sympy.Not(b_sym)),
        ),
        "Distributivity": (
            sympy.And(a_sym, sympy.Or(b_sym, c_sym)),
            sympy.Or(sympy.And(a_sym, b_sym), sympy.And(a_sym, c_sym)),
        ),
        "Contraposition": (
            sympy.Implies(a_sym, b_sym),
            sympy.Implies(sympy.Not(b_sym), sympy.Not(a_sym)),
        ),
        "Double negation": (sympy.Not(sympy.Not(a_sym)), a_sym),
        "Implication elimination": (
            sympy.Implies(a_sym, b_sym),
            sympy.Or(sympy.Not(a_sym), b_sym),
        ),
        "Biconditional elimination": (
            sympy.Equivalent(a_sym, b_sym),
            sympy.And(sympy.Implies(a_sym, b_sym), sympy.Implies(b_sym, a_sym)),
        ),
    }
    return laws


def cell1_2_equivalences_and_normal_forms(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Check equivalences by model sets, and convert sentences to CNF and DNF.

    Interactive controls (ipywidgets):
    - `law`: the equivalence law checked on the left panel
    - `sentence`: the weather sentence converted to CNF and to DNF

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (17, 5)
    laws = equivalence_laws()
    sentences = weather_sentences()
    law_dropdown = ipywidgets.Dropdown(
        options=list(laws.keys()),
        value="De Morgan",
        description="law:",
        style={"description_width": "initial"},
    )
    sentence_dropdown = ipywidgets.Dropdown(
        options=list(sentences.keys()),
        value="Snow <=> (Cloudy & Cold)",
        description="sentence:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            law_name = law_dropdown.value
            left, right = laws[law_name]
            equivalent = is_equivalent(left, right)
            # Count the models of each side over their shared symbols: the
            # check is about model sets, not about the shape of the strings.
            var_order = sorted(left.free_symbols | right.free_symbols, key=str)
            bits = enumerate_assignments(len(var_order))
            left_models = int(truth_values(left, var_order, bits).sum())
            right_models = int(truth_values(right, var_order, bits).sum())
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            verdict_color = _COLOR_Z3 if equivalent else _COLOR_PYSAT
            lines = [
                ("left :  %s" % format_sentence(left), _COLOR_SYMPY, "bold"),
                ("right:  %s" % format_sentence(right), _COLOR_SYMPY, "bold"),
                ("", _COLOR_NEUTRAL, "normal"),
                (
                    "|M(left)|  = %d of %d models"
                    % (left_models, bits.shape[0]),
                    _COLOR_NEUTRAL,
                    "normal",
                ),
                (
                    "|M(right)| = %d of %d models"
                    % (right_models, bits.shape[0]),
                    _COLOR_NEUTRAL,
                    "normal",
                ),
                ("", _COLOR_NEUTRAL, "normal"),
                (
                    "~(left <=> right) is %s"
                    % ("unsatisfiable" if equivalent else "satisfiable"),
                    _COLOR_NEUTRAL,
                    "normal",
                ),
                (
                    "verdict: %s"
                    % ("equivalent" if equivalent else "not equivalent"),
                    verdict_color,
                    "bold",
                ),
            ]
            draw_text_panel(ax1, lines, title="Equivalence check: %s" % law_name)
            # Convert the chosen sentence to both normal forms, and compare
            # how many clauses each form needs.
            label = sentence_dropdown.value
            sentence = sentences[label]
            cnf = sympy.to_cnf(sentence, simplify=False)
            dnf = sympy.to_dnf(sentence, simplify=False)
            counts = {
                "input\nconnectives": int(sympy.count_ops(sentence)),
                "CNF\nclauses": count_clauses(cnf),
                "DNF\nterms": count_clauses(dnf),
            }
            ax2.bar(
                list(counts.keys()),
                list(counts.values()),
                color=[_COLOR_NEUTRAL, _COLOR_SYMPY, _COLOR_PYSAT],
            )
            for index, value in enumerate(counts.values()):
                ax2.text(index, value, str(value), ha="center", va="bottom")
            ax2.set_title("Size of each form", fontsize=13, fontweight="bold")
            ax2.set_ylabel("count")
            ax2.set_xlabel(
                "`to_cnf()` trades size for a solver-ready shape", fontsize=9
            )
            text = (
                "law: %s\n"
                "  equivalent: %s\n\n"
                "sentence: %s\n\n"
                "CNF:\n  %s\n"
                "DNF:\n  %s\n\n"
                "CNF clauses: %d\n"
                "DNF terms: %d"
                % (
                    law_name,
                    equivalent,
                    format_sentence(sentence),
                    wrap_text(format_sentence(cnf)),
                    wrap_text(format_sentence(dnf)),
                    count_clauses(cnf),
                    count_clauses(dnf),
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "law": "the equivalence checked by comparing the two model sets",
            "sentence": "the sentence converted to CNF and to DNF",
        }
    )
    law_dropdown.observe(update_plot, names="value")
    sentence_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [law_dropdown, sentence_dropdown],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.1: From formula to clauses, encoding CNF for `PySAT`
# #############################################################################


def cell2_1_cnf_to_dimacs(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Encode a `sympy` CNF sentence as DIMACS clauses for `PySAT`.

    Interactive controls (ipywidgets):
    - `sentence`: the weather sentence encoded into clauses

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (17, 5)
    sentences = weather_sentences()
    sentence_dropdown = ipywidgets.Dropdown(
        options=list(sentences.keys()),
        value="Snow <=> (Cloudy & Cold)",
        description="sentence:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            label = sentence_dropdown.value
            sentence = sentences[label]
            clauses, symbol_map = cnf_to_clauses(sentence)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Left panel: the symbol map that has to travel with the file.
            map_lines = [
                (
                    "sentence: %s" % format_sentence(sentence),
                    _COLOR_SYMPY,
                    "bold",
                ),
                ("", _COLOR_NEUTRAL, "normal"),
                ("symbol map (name -> DIMACS id):", _COLOR_NEUTRAL, "bold"),
            ]
            for name, index in sorted(symbol_map.items(), key=lambda kv: kv[1]):
                map_lines.append(
                    ("  %-8s -> %d" % (name, index), _COLOR_NEUTRAL, "normal")
                )
            map_lines.append(("", _COLOR_NEUTRAL, "normal"))
            map_lines.append(("clauses (signed ids):", _COLOR_PYSAT, "bold"))
            for clause in clauses:
                map_lines.append(
                    ("  %s" % clause, _COLOR_PYSAT, "normal"),
                )
            draw_text_panel(ax1, map_lines, title="Symbol map and clauses")
            # Right panel: the same clauses as a DIMACS file on disk.
            file_name = "tmp.L03_06_logic_solvers.cell2_1.cnf"
            content = clauses_to_dimacs(clauses, file_name)
            dimacs_lines = [
                (line, _COLOR_NEUTRAL, "normal")
                for line in content.strip().split("\n")
            ]
            draw_text_panel(
                ax2, dimacs_lines, title="DIMACS file: '%s'" % file_name
            )
            cnf = sympy.to_cnf(sentence, simplify=False)
            text = (
                "sentence: %s\n\n"
                "CNF: %s\n\n"
                "variables: %d\n"
                "clauses: %d\n"
                "DIMACS header: p cnf %d %d\n\n"
                "file: '%s'\n"
                "file size: %d bytes"
                % (
                    format_sentence(sentence),
                    wrap_text(format_sentence(cnf)),
                    len(symbol_map),
                    len(clauses),
                    len(symbol_map),
                    len(clauses),
                    file_name,
                    os.path.getsize(file_name),
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "sentence": "the sentence rewritten to CNF, then encoded as "
            "integer clauses and written out as DIMACS",
        }
    )
    sentence_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [sentence_dropdown],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.2: `PySAT` against brute-force model checking
# #############################################################################

# Largest instance a truth-table enumeration is allowed to actually run: past
# this the $2^n$ rows stop fitting in memory, and the curve is projected.
_MAX_BRUTE_FORCE_VARS = 18
# Solve times are reused across widget updates, since the pigeonhole runs get
# expensive exactly where the lesson is.
_SOLVE_TIME_CACHE: Dict[Tuple[str, int, int], Tuple[bool, float]] = {}


def pigeonhole_clauses(n_pigeons: int, n_holes: int) -> List[List[int]]:
    """
    Encode the pigeonhole principle in CNF.

    Variable `(p, h)` reads "pigeon `p` sits in hole `h`". The encoding says
    every pigeon sits somewhere, and no hole holds two pigeons, so the
    formula is unsatisfiable exactly when `n_pigeons > n_holes`.

    :param n_pigeons: number of pigeons
    :param n_holes: number of holes
    :return: clauses, each a list of signed integers
    """

    def var_id(pigeon: int, hole: int) -> int:
        return pigeon * n_holes + hole + 1

    clauses = []
    # Every pigeon sits in at least one hole.
    for pigeon in range(n_pigeons):
        clauses.append([var_id(pigeon, hole) for hole in range(n_holes)])
    # No hole holds two pigeons.
    for hole in range(n_holes):
        for first in range(n_pigeons):
            for second in range(first + 1, n_pigeons):
                clauses.append([-var_id(first, hole), -var_id(second, hole)])
    return clauses


def brute_force_check(
    clauses: List[List[int]], n_vars: int
) -> Tuple[bool, float]:
    """
    Decide satisfiability by enumerating every model, as a truth table does.

    Every one of the $2^n$ assignments is built at once and each clause is
    checked against all of them, which is the model-checking algorithm of
    the lecture, only vectorized.

    :param clauses: clauses, each a list of signed integers
    :param n_vars: number of variables
    :return: tuple of
        - True if some assignment satisfies every clause
        - elapsed time in seconds
    """
    start = time.time()
    bits = enumerate_assignments(n_vars)
    satisfied = np.ones(bits.shape[0], dtype=bool)
    for clause in clauses:
        clause_mask = np.zeros(bits.shape[0], dtype=bool)
        for literal in clause:
            column = bits[:, abs(literal) - 1]
            clause_mask |= column if literal > 0 else ~column
        satisfied &= clause_mask
    elapsed = time.time() - start
    return bool(satisfied.any()), elapsed


def _timed_solve(
    clauses: List[List[int]], solver_label: str, cache_key: Tuple[str, int, int]
) -> Tuple[bool, float]:
    """
    Solve a cached instance with `PySAT`, timing the call.

    :param clauses: clauses, each a list of signed integers
    :param solver_label: backend to use, a key of `_SOLVER_NAMES`
    :param cache_key: key identifying the instance and the backend
    :return: tuple of
        - True if the clauses are satisfiable
        - elapsed time in seconds
    """
    if cache_key not in _SOLVE_TIME_CACHE:
        is_sat, _, elapsed = solve_clauses(clauses, solver_label=solver_label)
        _SOLVE_TIME_CACHE[cache_key] = (is_sat, elapsed)
    return _SOLVE_TIME_CACHE[cache_key]


def pigeonhole_scaling(n_max: int, solver_label: str) -> pd.DataFrame:
    """
    Time both engines on the unsatisfiable pigeonhole formula, as it grows.

    The brute-force column is measured while the $2^n$ rows still fit, and
    projected from the cost per row past that point.

    :param n_max: largest number of pigeons to time
    :param solver_label: backend to use, a key of `_SOLVER_NAMES`
    :return: one row per instance, with columns
        - `n_pigeons`, `n_vars`, `n_clauses`
        - `solver_secs`, `brute_secs`, `brute_measured`
    """
    rows = []
    cost_per_cell = 0.0
    # Warm the solver up on a trivial instance, so the first timed call does
    # not also pay for loading the backend.
    solve_clauses([[1], [-1, 2]], solver_label=solver_label)
    for n_pigeons in range(3, n_max + 1):
        n_holes = n_pigeons - 1
        clauses = pigeonhole_clauses(n_pigeons, n_holes)
        n_vars = n_pigeons * n_holes
        _, solver_secs = _timed_solve(
            clauses, solver_label, (solver_label, n_pigeons, n_holes)
        )
        if n_vars <= _MAX_BRUTE_FORCE_VARS:
            _, brute_secs = brute_force_check(clauses, n_vars)
            # Remember the measured cost of one (row, clause) pair, so the
            # instances that are too big to run can still be projected.
            cost_per_cell = brute_secs / (2**n_vars * len(clauses))
            brute_measured = True
        else:
            brute_secs = cost_per_cell * 2**n_vars * len(clauses)
            brute_measured = False
        rows.append(
            {
                "n_pigeons": n_pigeons,
                "n_vars": n_vars,
                "n_clauses": len(clauses),
                "solver_secs": solver_secs,
                "brute_secs": brute_secs,
                "brute_measured": brute_measured,
            }
        )
    df = pd.DataFrame(rows)
    return df


def cell2_2_solver_vs_model_checking(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Compare a real SAT solver against enumerating every model.

    Interactive controls (ipywidgets):
    - `n`: number of pigeons, with `n - 1` holes, so the formula is
      unsatisfiable and both engines have to look at the whole search space
    - `solver`: the `PySAT` backend

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (17, 5)
    n_slider, n_box = htutori.build_widget_control(
        "n",
        "n (pigeons, into n-1 holes)",
        3,
        9,
        1,
        7,
        is_float=False,
    )
    solver_dropdown = ipywidgets.Dropdown(
        options=list(_SOLVER_NAMES.keys()),
        value="Minisat22",
        description="solver:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            n_pigeons = int(n_slider.value)
            solver_label = solver_dropdown.value
            df = pigeonhole_scaling(n_pigeons, solver_label)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Left panel: how the two engines scale on the same instances.
            measured = df[df["brute_measured"]]
            # The dashed continuation repeats the last measured point, so it
            # starts exactly where the solid curve stops.
            n_measured = int(df["brute_measured"].sum())
            projected = df.iloc[max(n_measured - 1, 0) :]
            has_projection = n_measured < len(df)
            ax1.plot(
                df["n_pigeons"],
                df["solver_secs"],
                marker="o",
                color=_COLOR_PYSAT,
                label="PySAT %s" % solver_label,
            )
            ax1.plot(
                measured["n_pigeons"],
                measured["brute_secs"],
                marker="s",
                color=_COLOR_SYMPY,
                label="model checking (measured)",
            )
            if has_projection:
                ax1.plot(
                    projected["n_pigeons"],
                    projected["brute_secs"],
                    marker="s",
                    linestyle="--",
                    color=_COLOR_SYMPY,
                    alpha=0.6,
                    label="model checking (projected)",
                )
            ax1.set_yscale("log")
            ax1.set_xticks(df["n_pigeons"].tolist())
            ax1.set_xlabel("n pigeons (n-1 holes, so the formula is UNSAT)")
            ax1.set_ylabel("solve time [s], log scale")
            ax1.set_title("Solve time", fontsize=13, fontweight="bold")
            ax1.legend(fontsize=8)
            # Middle panel: the model the solver returns on the satisfiable
            # twin of the same instance, n pigeons into n holes.
            sat_clauses = pigeonhole_clauses(n_pigeons, n_pigeons)
            is_sat, model, _ = solve_clauses(
                sat_clauses, solver_label=solver_label
            )
            # Decode `var_id(p, h) = p * n_holes + h + 1`, with `n_holes`
            # equal to `n_pigeons` for this satisfiable twin.
            assignment = [
                ((literal - 1) // n_pigeons, (literal - 1) % n_pigeons)
                for literal in model
                if literal > 0
            ]
            model_lines = [
                (
                    "n pigeons into n holes: %s"
                    % ("SAT" if is_sat else "UNSAT"),
                    _COLOR_Z3,
                    "bold",
                ),
                ("", _COLOR_NEUTRAL, "normal"),
                ("one satisfying model:", _COLOR_NEUTRAL, "bold"),
            ]
            for pigeon, hole in sorted(assignment):
                model_lines.append(
                    (
                        "  pigeon %d -> hole %d" % (pigeon, hole),
                        _COLOR_NEUTRAL,
                        "normal",
                    )
                )
            model_lines.append(("", _COLOR_NEUTRAL, "normal"))
            model_lines.append(
                (
                    "n pigeons into n-1 holes: UNSAT",
                    _COLOR_PYSAT,
                    "bold",
                )
            )
            model_lines.append(
                (
                    "  no assignment exists, for any n",
                    _COLOR_PYSAT,
                    "normal",
                )
            )
            draw_text_panel(ax2, model_lines, title="What the solver returns")
            current = df[df["n_pigeons"] == n_pigeons].iloc[0]
            text = (
                "n pigeons: %d\n"
                "holes: %d\n"
                "variables: %d\n"
                "clauses: %d\n"
                "models to enumerate: 2^%d\n\n"
                "solver: %s\n"
                "  verdict: UNSAT\n"
                "  time: %.1f ms\n\n"
                "model checking:\n"
                "  time: %.1f ms (%s)"
                % (
                    n_pigeons,
                    n_pigeons - 1,
                    int(current["n_vars"]),
                    int(current["n_clauses"]),
                    int(current["n_vars"]),
                    solver_label,
                    current["solver_secs"] * 1e3,
                    current["brute_secs"] * 1e3,
                    "measured" if current["brute_measured"] else "projected",
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "n": "pigeons in the pigeonhole formula, always with "
            "<code>n-1</code> holes, so the formula is unsatisfiable",
            "solver": "the <code>PySAT</code> backend that decides the instance",
        }
    )
    n_slider.observe(update_plot, names="value")
    solver_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [n_box, solver_dropdown],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.3: Entailment by refutation, and the random 3-SAT phase transition
# #############################################################################

# Clause-to-variable ratios swept when locating the phase transition.
_PHASE_RATIOS = tuple(np.arange(2.0, 8.5, 0.5))
# The sweep re-solves hundreds of instances, so results are memoized per
# `(n_vars, n_instances, seed, solver)`.
_PHASE_CACHE: Dict[Tuple[int, int, int, str], pd.DataFrame] = {}


def random_3sat_clauses(
    n_vars: int, n_clauses: int, rng: np.random.Generator
) -> List[List[int]]:
    """
    Generate a uniform random 3-SAT instance.

    Each clause picks 3 distinct variables and flips a fair coin for the sign
    of each literal, which is the standard random 3-SAT ensemble used to
    study the satisfiability phase transition.

    :param n_vars: number of variables
    :param n_clauses: number of clauses
    :param rng: random generator, so instances are reproducible
    :return: clauses, each a list of 3 signed integers
    """
    clauses = []
    for _ in range(n_clauses):
        variables = rng.choice(n_vars, size=3, replace=False) + 1
        signs = rng.choice([-1, 1], size=3)
        clauses.append([int(value) for value in variables * signs])
    return clauses


def phase_transition_sweep(
    n_vars: int,
    *,
    n_instances: int = 20,
    seed: int = 0,
    solver_label: str = "Glucose3",
) -> pd.DataFrame:
    """
    Solve random 3-SAT instances across a range of clause-to-variable ratios.

    :param n_vars: number of variables in every instance
    :param n_instances: instances solved at each ratio
    :param seed: random seed, so the sweep is reproducible
    :param solver_label: backend to use, a key of `_SOLVER_NAMES`
    :return: one row per ratio, with columns
        - `ratio`, `n_clauses`
        - `frac_sat`: fraction of the instances that are satisfiable
        - `median_ms`: median solve time in milliseconds
    """
    cache_key = (n_vars, n_instances, seed, solver_label)
    if cache_key not in _PHASE_CACHE:
        rng = np.random.default_rng(seed)
        rows = []
        for ratio in _PHASE_RATIOS:
            n_clauses = int(round(ratio * n_vars))
            outcomes = []
            times = []
            for _ in range(n_instances):
                clauses = random_3sat_clauses(n_vars, n_clauses, rng)
                is_sat, _, elapsed = solve_clauses(
                    clauses, solver_label=solver_label
                )
                outcomes.append(is_sat)
                times.append(elapsed * 1e3)
            rows.append(
                {
                    "ratio": float(ratio),
                    "n_clauses": n_clauses,
                    "frac_sat": float(np.mean(outcomes)),
                    "median_ms": float(np.median(times)),
                }
            )
        _PHASE_CACHE[cache_key] = pd.DataFrame(rows)
    return _PHASE_CACHE[cache_key]


def cell2_3_refutation_and_phase_transition(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Prove entailment by refutation, then locate the 3-SAT phase transition.

    Interactive controls (ipywidgets):
    - `ratio`: clause-to-variable ratio marked on the sweep
    - `n_vars`: variables per random 3-SAT instance
    - `seed`: seed of the instance generator

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (17, 5)
    ratio_slider, ratio_box = htutori.build_widget_control(
        "ratio",
        "ratio (clauses / variables)",
        2.0,
        8.0,
        0.5,
        4.5,
        is_float=True,
    )
    n_vars_slider, n_vars_box = htutori.build_widget_control(
        "n_vars",
        "n_vars (variables per instance)",
        50,
        150,
        50,
        100,
        is_float=False,
    )
    seed_slider, seed_box = htutori.build_widget_control(
        "seed",
        "seed (random instances)",
        0,
        10,
        1,
        0,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            # The lecture's own knowledge base, decided by refutation instead
            # of by enumerating models.
            rain, wet_ground, snow = sympy.symbols("Rain WetGround Snow")
            kb = [rain, sympy.Implies(rain, wet_ground)]
            entailed_wet, counter_wet = entails_by_refutation(kb, wet_ground)
            entailed_snow, counter_snow = entails_by_refutation(kb, snow)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            lines = [
                ("KB:", _COLOR_NEUTRAL, "bold"),
                ("  Rain", _COLOR_NEUTRAL, "normal"),
                ("  Rain => WetGround", _COLOR_NEUTRAL, "normal"),
                ("", _COLOR_NEUTRAL, "normal"),
                ("alpha = WetGround", _COLOR_SYMPY, "bold"),
                (
                    "  KB & ~alpha is %s" % ("UNSAT" if entailed_wet else "SAT"),
                    _COLOR_NEUTRAL,
                    "normal",
                ),
                ("  counterexample: %s" % counter_wet, _COLOR_NEUTRAL, "normal"),
                (
                    "  verdict: %s"
                    % ("entailed" if entailed_wet else "not entailed"),
                    _COLOR_Z3 if entailed_wet else _COLOR_PYSAT,
                    "bold",
                ),
                ("", _COLOR_NEUTRAL, "normal"),
                ("alpha = Snow", _COLOR_SYMPY, "bold"),
                (
                    "  KB & ~alpha is %s"
                    % ("UNSAT" if entailed_snow else "SAT"),
                    _COLOR_NEUTRAL,
                    "normal",
                ),
                (
                    "  counterexample: %s" % counter_snow,
                    _COLOR_NEUTRAL,
                    "normal",
                ),
                (
                    "  verdict: %s"
                    % ("entailed" if entailed_snow else "not entailed"),
                    _COLOR_Z3 if entailed_snow else _COLOR_PYSAT,
                    "bold",
                ),
            ]
            draw_text_panel(ax1, lines, title="Entailment by refutation")
            # Sweep the ratio and plot both curves against it.
            n_vars = int(n_vars_slider.value)
            seed = int(seed_slider.value)
            ratio = float(ratio_slider.value)
            df = phase_transition_sweep(n_vars, seed=seed)
            ax2.plot(
                df["ratio"],
                df["frac_sat"],
                marker="o",
                color=_COLOR_SYMPY,
                label="fraction satisfiable",
            )
            ax2.axvline(
                4.26,
                color=_COLOR_NEUTRAL,
                linestyle=":",
                label="ratio 4.26",
            )
            ax2.axvline(ratio, color=_COLOR_Z3, linestyle="--", label="ratio")
            ax2.set_xlabel("clauses / variables")
            ax2.set_ylabel("fraction satisfiable", color=_COLOR_SYMPY)
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_title(
                "Random 3-SAT phase transition", fontsize=13, fontweight="bold"
            )
            ax2.legend(fontsize=8, loc="center left")
            # The solve time lives on its own axis: the two curves share the
            # x-axis but not the units.
            ax2_time = ax2.twinx()
            ax2_time.plot(
                df["ratio"],
                df["median_ms"],
                marker="s",
                color=_COLOR_PYSAT,
                alpha=0.8,
                label="median solve time",
            )
            ax2_time.set_ylabel("median solve time [ms]", color=_COLOR_PYSAT)
            ax2_time.legend(fontsize=8, loc="upper right")
            current = df.iloc[(df["ratio"] - ratio).abs().argmin()]
            text = (
                "refutation:\n"
                "  KB |= WetGround: %s\n"
                "  KB |= Snow: %s\n\n"
                "sweep:\n"
                "  n_vars: %d\n"
                "  seed: %d\n"
                "  instances per ratio: 20\n\n"
                "at ratio %.1f:\n"
                "  clauses: %d\n"
                "  fraction satisfiable: %.2f\n"
                "  median solve time: %.2f ms"
                % (
                    entailed_wet,
                    entailed_snow,
                    n_vars,
                    seed,
                    current["ratio"],
                    int(current["n_clauses"]),
                    current["frac_sat"],
                    current["median_ms"],
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "ratio": "clause-to-variable ratio marked on the sweep; the "
            "transition sits near <code>4.26</code>",
            "n_vars": "variables per random 3-SAT instance; the transition "
            "sharpens as this grows",
            "seed": "seed of the random instance generator",
        }
    )
    ratio_slider.observe(update_plot, names="value")
    n_vars_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [ratio_box, n_vars_box, seed_box],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 3.1: First-order logic with `z3`
# #############################################################################


def build_domain(domain_size: int) -> Tuple[Any, Any, List[Any], List[str]]:
    """
    Declare a finite domain of objects as a `z3` enumeration sort.

    A finite sort keeps quantifiers decidable, so `z3` can both satisfy and
    refute quantified sentences over it without any search heuristic. Every
    call gets its own `z3` context, since re-declaring the same sort name in
    one context is an error, and the widget rebuilds the domain on each
    update.

    :param domain_size: number of objects in the domain
    :return: tuple of
        - the `z3` context owning every symbol below
        - the `z3` sort
        - the object constants
        - the object names, e.g., `["o0", "o1"]`
    """
    names = ["o%d" % index for index in range(domain_size)]
    ctx = z3.Context()
    sort, consts = z3.EnumSort("Object", names, ctx=ctx)
    return ctx, sort, list(consts), names


def build_query(query_name: str, domain_size: int) -> Dict[str, Any]:
    """
    Build one first-order query over a fresh finite domain.

    Each query carries what the cell needs to both decide it and draw it:
    the formula to check, whether the interesting answer is a model or a
    refutation, and which symbol to read off the model when drawing.

    :param query_name: key of the query, as shown in the widget
    :param domain_size: number of objects in the domain
    :return: dict with the keys
        - `text`: the formula as the lecture writes it
        - `mode`: `"satisfy"` when a model is the answer, `"refute"` when
          unsatisfiability is
        - `goal`: the `z3` formula handed to the solver
        - `draw_assertions`: formulas satisfied to produce the drawn model
        - `draw_kind`: `"relation"` or `"predicate"`
        - `relation`: the binary relation to draw, when `draw_kind` says so
        - `predicates`: the unary predicates to draw, otherwise
        - `ctx`, `consts`, `names`: the context and domain the formula is
          built over
    """
    ctx, sort, consts, names = build_domain(domain_size)
    x_var = z3.Const("x", sort)
    y_var = z3.Const("y", sort)
    bool_sort = z3.BoolSort(ctx)
    query: Dict[str, Any] = {"ctx": ctx, "consts": consts, "names": names}
    if query_name in (
        "Everybody loves somebody",
        "Someone is loved by everyone",
        "Order matters: forall-exists but not exists-forall",
    ):
        loves = z3.Function("Loves", sort, sort, bool_sort)
        forall_exists = z3.ForAll(
            [x_var], z3.Exists([y_var], loves(x_var, y_var))
        )
        exists_forall = z3.Exists(
            [y_var], z3.ForAll([x_var], loves(x_var, y_var))
        )
        if query_name == "Everybody loves somebody":
            goal = forall_exists
            text = ["forall x. exists y. Loves(x, y)"]
        elif query_name == "Someone is loved by everyone":
            goal = exists_forall
            text = ["exists y. forall x. Loves(x, y)"]
        else:
            goal = z3.And(forall_exists, z3.Not(exists_forall))
            text = [
                "forall x. exists y. Loves(x, y)",
                "   and",
                "not (exists y. forall x. Loves(x, y))",
            ]
        # "Everyone loves everyone" satisfies both sentences, so the drawing
        # also asks for a relation that is not total: the verdict stays about
        # the sentence alone, while the picture shows a telling witness.
        not_total = z3.Not(z3.ForAll([x_var, y_var], loves(x_var, y_var)))
        query.update(
            {
                "text": text,
                "mode": "satisfy",
                "goal": goal,
                "draw_assertions": [goal, not_total],
                "draw_kind": "relation",
                "relation": loves,
            }
        )
    elif query_name == "Aristotle: every human is mortal":
        human = z3.Function("Human", sort, bool_sort)
        mortal = z3.Function("Mortal", sort, bool_sort)
        socrates = consts[0]
        axioms = [
            z3.ForAll([x_var], z3.Implies(human(x_var), mortal(x_var))),
            human(socrates),
        ]
        query.update(
            {
                "text": [
                    "forall x. (Human(x) => Mortal(x))",
                    "Human(o0)",
                    "|= Mortal(o0)",
                ],
                "mode": "refute",
                "goal": z3.And(*axioms, z3.Not(mortal(socrates))),
                "draw_assertions": axioms,
                "draw_kind": "predicate",
                "predicates": {"Human": human, "Mortal": mortal},
            }
        )
    else:
        # Quantifier duality, the lecture's own De Morgan rule for
        # quantifiers: refuting it is what proves it holds on this domain.
        p_pred = z3.Function("P", sort, bool_sort)
        duality = z3.Not(
            z3.Not(z3.ForAll([x_var], p_pred(x_var)))
            == z3.Exists([x_var], z3.Not(p_pred(x_var)))
        )
        query.update(
            {
                "text": [
                    "~(forall x. P(x))  <=>  exists x. ~P(x)",
                    "checked by asserting its negation",
                ],
                "mode": "refute",
                "goal": duality,
                "draw_assertions": [z3.Exists([x_var], z3.Not(p_pred(x_var)))],
                "draw_kind": "predicate",
                "predicates": {"P": p_pred},
            }
        )
    return query


def check_query(goal: Any, ctx: Any) -> str:
    """
    Run `z3` on one formula and report the verdict.

    :param goal: formula to check
    :param ctx: `z3` context the formula was built in
    :return: `"sat"`, `"unsat"`, or `"unknown"`
    """
    solver = z3.Solver(ctx=ctx)
    solver.add(goal)
    result = str(solver.check())
    return result


def _model_of(assertions: Sequence[Any], ctx: Any) -> Optional[Any]:
    """
    Find a model of a list of assertions, for drawing.

    :param assertions: formulas the model has to satisfy
    :param ctx: `z3` context the formulas were built in
    :return: the `z3` model, or `None` when the assertions are unsatisfiable
    """
    solver = z3.Solver(ctx=ctx)
    for assertion in assertions:
        solver.add(assertion)
    model = solver.model() if solver.check() == z3.sat else None
    return model


def relation_edges(query: Dict[str, Any], model: Any) -> List[Tuple[str, str]]:
    """
    Read a binary relation off a `z3` model, one pair of objects at a time.

    :param query: query dict from `build_query()`, of kind `"relation"`
    :param model: model to read the interpretation from
    :return: pairs of object names the relation holds between
    """
    names = query["names"]
    consts = query["consts"]
    relation = query["relation"]
    edges = []
    for source_index, source in enumerate(consts):
        for target_index, target in enumerate(consts):
            value = model.eval(relation(source, target), model_completion=True)
            if z3.is_true(value):
                edges.append((names[source_index], names[target_index]))
    return edges


def draw_domain_graph(
    ax: matplotlib.axes.Axes,
    query: Dict[str, Any],
    model: Optional[Any],
    *,
    title: str,
) -> None:
    """
    Draw the domain `z3` returned, as objects plus the relation over them.

    :param ax: axes to draw on
    :param query: query dict from `build_query()`
    :param model: model to read the interpretation from, or `None`
    :param title: panel title
    """
    names = query["names"]
    consts = query["consts"]
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    node_colors = ["#c6dbef" for _ in names]
    if model is None:
        ax.text(
            0.5,
            0.5,
            "no model: the assertions are unsatisfiable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
    elif query["draw_kind"] == "relation":
        graph.add_edges_from(relation_edges(query, model))
    else:
        # Color every object by the unary predicates it satisfies.
        labels = []
        for index, const in enumerate(consts):
            holds = [
                name
                for name, predicate in query["predicates"].items()
                if z3.is_true(
                    model.eval(predicate(const), model_completion=True)
                )
            ]
            labels.append(",".join(holds) if holds else "-")
            node_colors[index] = "#a1d99b" if holds else "#f7f7f7"
        graph = nx.relabel_nodes(
            graph,
            {
                name: "%s\n%s" % (name, label)
                for name, label in zip(names, labels)
            },
        )
    positions = nx.circular_layout(graph)
    nx.draw_networkx(
        graph,
        pos=positions,
        ax=ax,
        node_color=node_colors,
        edgecolors=_COLOR_NEUTRAL,
        node_size=1800,
        font_size=8,
        arrows=True,
        arrowsize=14,
        connectionstyle="arc3,rad=0.15",
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")


def cell3_1_z3_quantifiers(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Check quantified first-order sentences with `z3` over a finite domain.

    Interactive controls (ipywidgets):
    - `query`: the first-order sentence handed to `z3`
    - `domain_size`: number of objects in the domain

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (17, 5)
    query_dropdown = ipywidgets.Dropdown(
        options=[
            "Everybody loves somebody",
            "Someone is loved by everyone",
            "Order matters: forall-exists but not exists-forall",
            "Aristotle: every human is mortal",
            "Quantifier duality",
        ],
        value="Order matters: forall-exists but not exists-forall",
        description="query:",
        style={"description_width": "initial"},
        layout=ipywidgets.Layout(width="440px"),
    )
    size_slider, size_box = htutori.build_widget_control(
        "domain_size",
        "domain_size (objects)",
        2,
        6,
        1,
        3,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            query_name = query_dropdown.value
            domain_size = int(size_slider.value)
            query = build_query(query_name, domain_size)
            verdict = check_query(query["goal"], query["ctx"])
            model = _model_of(query["draw_assertions"], query["ctx"])
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            draw_domain_graph(
                ax1, query, model, title="Domain and relation from the model"
            )
            expected = "unsat" if query["mode"] == "refute" else "sat"
            reading = {
                (
                    "satisfy",
                    "sat",
                ): "a model exists: the sentence is satisfiable",
                ("satisfy", "unsat"): "no model on this domain",
                (
                    "refute",
                    "unsat",
                ): "the negation has no model: the claim holds",
                ("refute", "sat"): "a countermodel exists: the claim fails",
            }[(query["mode"], verdict)]
            lines = [("formula:", _COLOR_NEUTRAL, "bold")]
            lines.extend(
                [("  %s" % line, _COLOR_Z3, "normal") for line in query["text"]]
            )
            lines.extend(
                [
                    ("", _COLOR_NEUTRAL, "normal"),
                    (
                        "asserted: %s"
                        % (
                            "the sentence"
                            if query["mode"] == "satisfy"
                            else "its negation"
                        ),
                        _COLOR_NEUTRAL,
                        "normal",
                    ),
                    ("z3 verdict: %s" % verdict, _COLOR_Z3, "bold"),
                    ("expected: %s" % expected, _COLOR_NEUTRAL, "normal"),
                    ("", _COLOR_NEUTRAL, "normal"),
                    ("reading:", _COLOR_NEUTRAL, "bold"),
                    ("  %s" % reading, _COLOR_NEUTRAL, "normal"),
                ]
            )
            draw_text_panel(ax2, lines, title="Query and verdict")
            if model is not None and query["draw_kind"] == "relation":
                drawn = "%d edges in Loves" % len(relation_edges(query, model))
            elif model is not None:
                drawn = "predicate membership per object"
            else:
                drawn = "none"
            text = (
                "query: %s\n\n"
                "domain size: %d objects\n"
                "quantifier pattern:\n  %s\n\n"
                "asserted: %s\n"
                "z3 verdict: %s\n\n"
                "model drawn: %s"
                % (
                    query_name,
                    domain_size,
                    query["text"][0],
                    "the sentence"
                    if query["mode"] == "satisfy"
                    else "its negation",
                    verdict,
                    drawn,
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "query": "the first-order sentence <code>z3</code> checks, and "
            "whose model is drawn on the left",
            "domain_size": "number of objects in the finite domain the "
            "quantifiers range over",
        }
    )
    query_dropdown.observe(update_plot, names="value")
    size_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [query_dropdown, size_box],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 3.2: One question, three engines
# #############################################################################


def _propositional_entailment_row(engine: str) -> Dict[str, str]:
    """
    Decide the propositional query with one engine, and time the call.

    The query is the lecture's own: `{Rain, Rain => WetGround} |= WetGround`,
    decided by refutation, which every one of the three engines can express.

    :param engine: `"sympy"`, `"PySAT"`, or `"z3"`
    :return: one row of the comparison table
    """
    rain, wet_ground = sympy.symbols("Rain WetGround")
    kb = [rain, sympy.Implies(rain, wet_ground)]
    start = time.time()
    if engine == "sympy":
        method = "satisfiable(KB & ~a)"
        refutation = sympy.And(*kb, sympy.Not(wet_ground))
        entailed = sympy.satisfiable(refutation) is False
    elif engine == "PySAT":
        method = "clauses -> solve()"
        entailed, _ = entails_by_refutation(kb, wet_ground)
    else:
        method = "Solver().check()"
        rain_z3, wet_z3 = z3.Bools("Rain WetGround")
        solver = z3.Solver()
        solver.add(rain_z3, z3.Implies(rain_z3, wet_z3), z3.Not(wet_z3))
        entailed = solver.check() == z3.unsat
    elapsed_ms = (time.time() - start) * 1e3
    row = {
        "engine": engine,
        "layer": "propositional",
        "method": method,
        "verdict": "entailed" if entailed else "not entailed",
        "time [ms]": "%.2f" % elapsed_ms,
    }
    return row


def _first_order_entailment_row(engine: str) -> Dict[str, str]:
    """
    Decide the first-order query with one engine, and time the call.

    The query is the Socrates argument: from `forall x. Human(x) =>
    Mortal(x)` and `Human(Socrates)`, does `Mortal(Socrates)` follow? Only
    `z3` can state the universal rule; the propositional engines see two
    unrelated atoms.

    :param engine: `"sympy"`, `"PySAT"`, or `"z3"`
    :return: one row of the comparison table
    """
    start = time.time()
    if engine in ("sympy", "PySAT"):
        # The quantified rule cannot be written down, so the best a
        # propositional engine can do is treat each ground atom as its own
        # symbol, which loses the link between them.
        human_socrates, mortal_socrates = sympy.symbols(
            "Human_Socrates Mortal_Socrates"
        )
        kb = [human_socrates]
        if engine == "sympy":
            method = "satisfiable(KB & ~a)"
            refutation = sympy.And(*kb, sympy.Not(mortal_socrates))
            entailed = sympy.satisfiable(refutation) is False
        else:
            method = "clauses -> solve()"
            entailed, _ = entails_by_refutation(kb, mortal_socrates)
        verdict = "not entailed (rule lost)"
    else:
        method = "ForAll + check()"
        # Each call gets a fresh context, so re-running the cell does not
        # re-declare `Object` in a context that already has it.
        ctx = z3.Context()
        sort = z3.DeclareSort("Object", ctx=ctx)
        bool_sort = z3.BoolSort(ctx)
        human = z3.Function("Human", sort, bool_sort)
        mortal = z3.Function("Mortal", sort, bool_sort)
        socrates = z3.Const("Socrates", sort)
        x_var = z3.Const("x", sort)
        solver = z3.Solver(ctx=ctx)
        solver.add(z3.ForAll([x_var], z3.Implies(human(x_var), mortal(x_var))))
        solver.add(human(socrates))
        solver.add(z3.Not(mortal(socrates)))
        entailed = solver.check() == z3.unsat
        verdict = "entailed" if entailed else "not entailed"
    elapsed_ms = (time.time() - start) * 1e3
    row = {
        "engine": engine,
        "layer": "first-order",
        "method": method,
        "verdict": verdict,
        "time [ms]": "%.2f" % elapsed_ms,
    }
    return row


def compare_engines(query_name: str) -> pd.DataFrame:
    """
    Run the same entailment question through all three engines.

    :param query_name: `"propositional"` or `"first-order"`
    :return: one row per engine, with columns `engine`, `layer`, `method`,
        `verdict`, `time [ms]`
    """
    engines = ["sympy", "PySAT", "z3"]
    # Warm `z3` up, so the first timed call does not also pay for building
    # its first context.
    z3.Solver().check()
    if query_name == "propositional":
        rows = [_propositional_entailment_row(engine) for engine in engines]
    else:
        rows = [_first_order_entailment_row(engine) for engine in engines]
    df = pd.DataFrame(rows)
    return df


def draw_comparison_table(
    ax: matplotlib.axes.Axes, df: pd.DataFrame, *, title: str
) -> None:
    """
    Render the engine comparison as a table panel.

    :param ax: axes to draw on
    :param df: comparison table from `compare_engines()`
    :param title: panel title
    """
    ax.axis("off")
    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=list(df.columns),
        colWidths=[0.13, 0.17, 0.26, 0.30, 0.14],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    # Color each engine row, so the same engine reads the same everywhere.
    engine_colors = {
        "sympy": _COLOR_SYMPY,
        "PySAT": _COLOR_PYSAT,
        "z3": _COLOR_Z3,
    }
    for row_index, engine in enumerate(df["engine"], start=1):
        for col_index in range(len(df.columns)):
            cell = table[row_index, col_index]
            cell.set_edgecolor(engine_colors[engine])
    ax.set_title(title, fontsize=13, fontweight="bold")


def draw_scope_diagram(ax: matplotlib.axes.Axes, *, title: str) -> None:
    """
    Draw which logic each engine reaches, as nested boxes.

    :param ax: axes to draw on
    :param title: panel title
    """
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Outer box: first-order logic, which only `z3` reaches.
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.06, 0.12),
            0.88,
            0.72,
            boxstyle="round,pad=0.02",
            facecolor="#e5f5e0",
            edgecolor=_COLOR_Z3,
            linewidth=2.0,
        )
    )
    ax.text(
        0.5,
        0.78,
        "first-order logic: quantifiers, predicates, functions",
        ha="center",
        va="center",
        fontsize=10,
        color=_COLOR_Z3,
        fontweight="bold",
    )
    # Inner box: propositional logic, where all three engines work.
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.16, 0.22),
            0.68,
            0.42,
            boxstyle="round,pad=0.02",
            facecolor="#deebf7",
            edgecolor=_COLOR_SYMPY,
            linewidth=2.0,
        )
    )
    ax.text(
        0.5,
        0.56,
        "propositional logic: symbols and connectives",
        ha="center",
        va="center",
        fontsize=10,
        color=_COLOR_SYMPY,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.44,
        "sympy: rewriting, CNF, DNF\nPySAT: fast satisfiability\nz3: both layers",
        ha="center",
        va="center",
        fontsize=9,
        color=_COLOR_NEUTRAL,
    )
    ax.text(
        0.5,
        0.05,
        "z3 reaches the outer box; sympy and PySAT stop at the inner one",
        ha="center",
        va="center",
        fontsize=9,
        color=_COLOR_NEUTRAL,
        style="italic",
    )
    ax.set_title(title, fontsize=13, fontweight="bold")


def cell3_2_three_engines(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Route one entailment question to whichever engine can answer it.

    Interactive controls (ipywidgets):
    - `query`: the propositional question, or its first-order generalization

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (17, 5)
    query_dropdown = ipywidgets.Dropdown(
        options=["propositional", "first-order"],
        value="propositional",
        description="query:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            query_name = query_dropdown.value
            df = compare_engines(query_name)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            if query_name == "propositional":
                question = "KB = {Rain, Rain => WetGround} |= WetGround"
            else:
                question = (
                    "KB = {forall x. Human(x) => Mortal(x), Human(Socrates)}"
                    " |= Mortal(Socrates)"
                )
            draw_comparison_table(ax1, df, title="Three engines, one question")
            draw_scope_diagram(ax2, title="What each engine reaches")
            answered = df[df["verdict"] == "entailed"]["engine"].tolist()
            text = (
                "query: %s\n\n"
                "question:\n  %s\n\n"
                "engines answering 'entailed':\n  %s\n\n"
                "verdicts:\n%s"
                % (
                    query_name,
                    question,
                    ", ".join(answered) if answered else "none",
                    "\n".join(
                        "  %-6s %s" % (row["engine"], row["verdict"])
                        for _, row in df.iterrows()
                    ),
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "query": "the propositional entailment, or its first-order "
            "generalization with a quantified rule",
        }
    )
    query_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [query_dropdown],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))
