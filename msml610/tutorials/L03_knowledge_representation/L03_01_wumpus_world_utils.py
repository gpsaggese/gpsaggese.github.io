"""
Utility functions for the wumpus world knowledge representation lesson.

Builds a knowledge-based agent for the classic wumpus world and uses it to
make the model-theoretic definition of entailment concrete:
- The hidden world and the local percepts it generates.
- A propositional knowledge base built with `TELL` and queried with `ASK`.
- Model enumeration, entailment as model inclusion, and inference traces.
- Soundness and completeness, seen by deliberately breaking each.
- Brute-force model checking against a SAT solver as the grid grows.
- Interactive notebook cells built on top of these primitives.

Import as:

import msml610.tutorials.L03_knowledge_representation.L03_01_wumpus_world_utils as mtlkrl0wwu
"""

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ipywidgets
import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sympy
import sympy.logic.boolalg as slboolal
import sympy.logic.inference as slinfere
from IPython.display import clear_output, display

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.htutorial as htutori

_LOG = logging.getLogger(__name__)

# #############################################################################
# Colors and display constants
# #############################################################################

# Cell fill colors used when drawing the hidden world.
_COLOR_START = "#cfe8ff"
_COLOR_PIT = "#f4a6a6"
_COLOR_WUMPUS = "#d8b4e2"
_COLOR_GOLD = "#ffe680"
_COLOR_EMPTY = "#ffffff"
# Cell fill colors used when drawing what the `KB` knows.
_COLOR_UNKNOWN = "#e8e8e8"
_COLOR_SAFE = "#a8e6a3"
_COLOR_VISITED = "#7fb3d5"
# Colors used to mark reasoner verdicts against the entailment ground truth.
_COLOR_CORRECT = "#a8e6a3"
_COLOR_FALSE_POSITIVE = "#f4a6a6"
_COLOR_FALSE_NEGATIVE = "#ffd699"
# Neighbor offsets `(d_col, d_row)`, fixed so that layouts are reproducible.
_NEIGHBOR_DELTA = ((0, 1), (0, -1), (-1, 0), (1, 0))
# Default layout seed, picked because its agent walk reaches the gold, smells
# the wumpus on the way, and narrows the candidate models down to one.
_DEFAULT_SEED = 29
# Enumerating more than this many variables exhausts memory, so runtimes past
# this point are extrapolated instead of measured (see `cell3_2_scaling()`).
_MAX_ENUMERATED_VARS = 18


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger so notebook cells see the
    debug and info output of these utility functions.

    :param notebook_log: logger owned by the notebook
    """
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# The hidden wumpus world
# #############################################################################


class WumpusWorld:
    """
    The hidden wumpus world that generates the agent's percepts.

    - Coordinates are 1-indexed `(col, row)` with `(1, 1)` at the bottom left
    - The start cell `(1, 1)` never holds a pit or the wumpus, so the agent
      survives its first step
    - Percepts are purely local:
      - A breeze in every cell adjacent to a pit
      - A stench in the wumpus cell and in every cell adjacent to it
      - A glitter in the cell holding the gold
    - The agent never sees this object: it only sees the percepts.
    """

    def __init__(
        self,
        *,
        n_cols: int = 4,
        n_rows: int = 4,
        seed: int = _DEFAULT_SEED,
        pit_prob: float = 0.2,
    ) -> None:
        """
        Sample a hidden layout of pits, the wumpus, and the gold.

        :param n_cols: number of grid columns
        :param n_rows: number of grid rows
        :param seed: random seed for the layout
        :param pit_prob: probability that a non-start cell holds a pit
        """
        _LOG.debug("n_cols=%s n_rows=%s seed=%s", n_cols, n_rows, seed)
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.start = (1, 1)
        # Enumerate every cell in row-major order, bottom row first.
        self.cells = [
            (col, row)
            for row in range(1, n_rows + 1)
            for col in range(1, n_cols + 1)
        ]
        # Every cell except the start can hold a pit.
        rng = np.random.RandomState(seed)
        candidates = [cell for cell in self.cells if cell != self.start]
        draws = rng.rand(len(candidates))
        self.pits = {
            cell for cell, draw in zip(candidates, draws) if draw < pit_prob
        }
        # Guarantee at least one pit, otherwise no breeze percept ever fires
        # and the entailment examples become degenerate.
        if not self.pits:
            self.pits = {candidates[int(rng.randint(len(candidates)))]}
        # The wumpus and the gold sit on cells that do not hold a pit.
        free = [cell for cell in candidates if cell not in self.pits]
        hdbg.dassert_lte(
            2, len(free), "Layout leaves no room for the wumpus and the gold"
        )
        picks = rng.choice(len(free), size=2, replace=False)
        self.wumpus = free[int(picks[0])]
        self.gold = free[int(picks[1])]

    def neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Return the cells orthogonally adjacent to `cell`, inside the grid.

        :param cell: cell as `(col, row)`
        :return: adjacent cells, e.g., `(1, 2)` -> `[(1, 3), (1, 1), (2, 2)]`
        """
        col, row = cell
        out = []
        for d_col, d_row in _NEIGHBOR_DELTA:
            candidate = (col + d_col, row + d_row)
            inside = 1 <= candidate[0] <= self.n_cols
            inside = inside and 1 <= candidate[1] <= self.n_rows
            if inside:
                out.append(candidate)
        return out

    def percept(self, cell: Tuple[int, int]) -> Dict[str, bool]:
        """
        Return the percept the agent receives when standing in `cell`.

        :param cell: cell as `(col, row)`
        :return: dict with the `breeze`, `stench`, and `glitter` flags
            ```

            {'breeze': True, 'stench': False, 'glitter': False}

            ```
        """
        neighbors = self.neighbors(cell)
        percept = {
            "breeze": any(n in self.pits for n in neighbors),
            "stench": cell == self.wumpus
            or any(n == self.wumpus for n in neighbors),
            "glitter": cell == self.gold,
        }
        return percept

    def percept_label(self, cell: Tuple[int, int]) -> str:
        """
        Return a compact label for the percept at `cell`.

        :param cell: cell as `(col, row)`
        :return: one letter per active percept, or `none`, e.g., `B S`
        """
        percept = self.percept(cell)
        letters = []
        if percept["breeze"]:
            letters.append("B")
        if percept["stench"]:
            letters.append("S")
        if percept["glitter"]:
            letters.append("G")
        label = " ".join(letters) if letters else "none"
        return label

    def truth_label(self, cell: Tuple[int, int]) -> str:
        """
        Return the hidden content of `cell`, which the agent cannot see.

        :param cell: cell as `(col, row)`
        :return: role of the cell, e.g., `PIT`, `WUMPUS`, `GOLD`, `START`
        """
        if cell in self.pits:
            label = "PIT"
        elif cell == self.wumpus:
            label = "WUMPUS"
        elif cell == self.gold:
            label = "GOLD"
        elif cell == self.start:
            label = "START"
        else:
            label = ""
        return label


# #############################################################################
# Propositional encoding of the world
# #############################################################################


def pit_symbol(cell: Tuple[int, int]) -> sympy.Symbol:
    """
    Return the proposition "there is a pit in `cell`".

    :param cell: cell as `(col, row)`
    :return: symbol named `P_<col>_<row>`, e.g., `P_2_3`
    """
    return sympy.Symbol("P_%d_%d" % cell)


def breeze_symbol(cell: Tuple[int, int]) -> sympy.Symbol:
    """
    Return the proposition "the agent feels a breeze in `cell`".

    :param cell: cell as `(col, row)`
    :return: symbol named `B_<col>_<row>`, e.g., `B_1_2`
    """
    return sympy.Symbol("B_%d_%d" % cell)


def breeze_axiom(world: WumpusWorld, cell: Tuple[int, int]) -> sympy.Basic:
    """
    Build the breeze axiom for `cell`, the rule that links percept to cause.

    The axiom is the biconditional
    `B_cell <=> (P_n1 | P_n2 | ...)` over the neighbors of `cell`, e.g.,
    `Equivalent(B_1_2, P_1_1 | P_1_3 | P_2_2)`.

    :param world: world supplying the grid geometry
    :param cell: cell the axiom is written for
    :return: the biconditional as a `sympy` sentence
    """
    neighbor_pits = [pit_symbol(n) for n in world.neighbors(cell)]
    axiom = sympy.Equivalent(breeze_symbol(cell), sympy.Or(*neighbor_pits))
    return axiom


def format_sentence(sentence: sympy.Basic) -> str:
    """
    Render a `sympy` sentence as ASCII text suitable for a plot panel.

    :param sentence: sentence to render
    :return: string form, e.g., `Equivalent(B_1_2, P_1_1 | P_1_3 | P_2_2)`
    """
    return sympy.sstr(sentence)


# #############################################################################
# Model checking by explicit enumeration
# #############################################################################


def enumerate_assignments(n_vars: int) -> np.ndarray:
    """
    Enumerate every truth assignment over `n_vars` propositional variables.

    Row `i` holds the binary expansion of `i`, so the rows are exactly the
    $2^n$ candidate worlds a model checker has to examine.

    :param n_vars: number of variables
    :return: boolean array of shape `(2 ** n_vars, n_vars)`
    """
    hdbg.dassert_lte(
        n_vars,
        _MAX_ENUMERATED_VARS,
        "Enumerating this many variables exhausts memory",
    )
    index = np.arange(1 << n_vars, dtype=np.int64)
    bits = ((index[:, None] >> np.arange(n_vars)) & 1).astype(bool)
    return bits


def to_clauses(
    sentences: Sequence[sympy.Basic],
    var_order: Sequence[sympy.Symbol],
) -> List[List[Tuple[int, bool]]]:
    """
    Convert sentences into CNF clauses over variable indices.

    Each clause is a list of `(variable index, is_positive)` literals, which
    makes model checking a handful of vectorized boolean operations.

    :param sentences: sentences forming the knowledge base
    :param var_order: variables, in the column order of the model table
    :return: clauses, e.g., `Or(P_1_1, P_2_2)` -> `[[(0, True), (1, True)]]`
    """
    index: Dict[Any, int] = {var: i for i, var in enumerate(var_order)}
    clauses: List[List[Tuple[int, bool]]] = []
    for sentence in sentences:
        cnf = slboolal.to_cnf(sentence, simplify=False)
        # A sentence that reduces to `True` constrains nothing.
        if cnf == sympy.true:
            continue
        # A sentence that reduces to `False` makes the `KB` unsatisfiable,
        # which the empty clause encodes.
        if cnf == sympy.false:
            clauses.append([])
            continue
        for conjunct in slboolal.conjuncts(cnf):
            literals = []
            for literal in slboolal.disjuncts(conjunct):
                if isinstance(literal, sympy.Not):
                    symbol = literal.args[0]
                    is_positive = False
                else:
                    symbol = literal
                    is_positive = True
                hdbg.dassert_in(
                    symbol, index, "Sentence mentions a variable outside the table"
                )
                literals.append((index[symbol], is_positive))
            clauses.append(literals)
    return clauses


def satisfying_mask(
    sentences: Sequence[sympy.Basic],
    var_order: Sequence[sympy.Symbol],
    bits: np.ndarray,
) -> np.ndarray:
    """
    Mark the assignments in `bits` that satisfy every sentence.

    This is the model checking algorithm: it implements the definition of
    entailment directly, at the cost of touching all $2^n$ rows.

    :param sentences: sentences forming the knowledge base
    :param var_order: variables, in the column order of `bits`
    :param bits: assignments from `enumerate_assignments()`
    :return: boolean mask of shape `(2 ** n_vars,)`, one entry per assignment
    """
    clauses = to_clauses(sentences, var_order)
    mask = evaluate_clauses(clauses, bits)
    return mask


def evaluate_clauses(
    clauses: Sequence[Sequence[Tuple[int, bool]]],
    bits: np.ndarray,
) -> np.ndarray:
    """
    Evaluate CNF clauses against every assignment, one clause at a time.

    Kept separate from `to_clauses()` so that timing experiments measure the
    enumeration cost alone, without the CNF conversion overhead.

    :param clauses: clauses from `to_clauses()`
    :param bits: assignments from `enumerate_assignments()`
    :return: boolean mask marking the satisfying assignments
    """
    mask = np.ones(bits.shape[0], dtype=bool)
    for clause in clauses:
        # A clause holds when at least one of its literals holds.
        clause_holds = np.zeros(bits.shape[0], dtype=bool)
        for var_index, is_positive in clause:
            column = bits[:, var_index]
            clause_holds |= column if is_positive else ~column
        mask &= clause_holds
    return mask


def entailment_verdict(kb_mask: np.ndarray, alpha_mask: np.ndarray) -> str:
    """
    Decide entailment by comparing the two model sets.

    $KB \\models \\alpha$ holds exactly when $M(KB) \\subseteq M(\\alpha)$.

    :param kb_mask: mask marking $M(KB)$
    :param alpha_mask: mask marking $M(\\alpha)$
    :return: one of `entailed`, `contradicted`, `unknown`, `inconsistent`
    """
    if not kb_mask.any():
        # No model satisfies the `KB`, so every query is vacuously entailed.
        verdict = "inconsistent"
    elif bool(np.all(alpha_mask[kb_mask])):
        verdict = "entailed"
    elif not bool(np.any(alpha_mask[kb_mask])):
        verdict = "contradicted"
    else:
        verdict = "unknown"
    return verdict


# #############################################################################
# The knowledge base: TELL and ASK
# #############################################################################


class KnowledgeBase:
    """
    A propositional knowledge base over the pit and breeze variables.

    - `tell()` appends a sentence, and nothing is ever removed
    - `ask()` answers by model checking: enumerate every assignment to the pit
      variables and compare $M(KB)$ with $M(\\alpha)$
    - Observed breeze values are substituted into their axioms before model
      checking, so the models range over the pit variables alone
      - This is the same model set as the full one, projected onto the pit
        variables, and it keeps the enumeration to $2^{16}$ rows on a 4x4 grid
    """

    def __init__(
        self,
        world: WumpusWorld,
        *,
        var_order: Optional[Sequence[sympy.Symbol]] = None,
    ) -> None:
        """
        Create an empty knowledge base for `world`.

        :param world: world whose geometry supplies the axioms
        :param var_order: pit variables spanning the model table
            - Default: one pit variable per grid cell, row-major
        """
        self.world = world
        if var_order is None:
            var_order = [pit_symbol(cell) for cell in world.cells]
        self.var_order = list(var_order)
        # Sentences in the order they were told, used for display.
        self.sentences: List[sympy.Basic] = []
        # Cells whose percept has been told, in visit order.
        self.told_cells: List[Tuple[int, int]] = []
        # Observed breeze values, substituted before model checking.
        self.breeze_obs: Dict[Tuple[int, int], bool] = {}
        self._bits = enumerate_assignments(len(self.var_order))

    def tell(self, sentence: sympy.Basic) -> None:
        """
        Add one sentence to the knowledge base.

        :param sentence: sentence to add
        """
        _LOG.debug("tell '%s'", format_sentence(sentence))
        self.sentences.append(sentence)

    def tell_percept(self, cell: Tuple[int, int]) -> List[str]:
        """
        Tell everything the agent learns by standing in `cell`.

        Three sentences are added:
        - The agent survived, so `cell` holds no pit
        - The breeze axiom relating `cell` to its neighbors
        - The observed breeze literal, positive or negative

        :param cell: cell the agent perceives from
        :return: rendered sentences that were told, for display
        """
        _LOG.debug("cell=%s", str(cell))
        told = []
        # The agent is standing in the cell and survived, so it holds no pit.
        if cell not in self.told_cells:
            self.tell(sympy.Not(pit_symbol(cell)))
            told.append(format_sentence(sympy.Not(pit_symbol(cell))))
            self.told_cells.append(cell)
        # The axiom is general knowledge, but it only matters once the cell is
        # actually perceived, so it is told alongside the percept.
        axiom = breeze_axiom(self.world, cell)
        self.tell(axiom)
        told.append(format_sentence(axiom))
        # Tell the observed breeze value.
        has_breeze = self.world.percept(cell)["breeze"]
        literal = breeze_symbol(cell)
        sentence = literal if has_breeze else sympy.Not(literal)
        self.tell(sentence)
        told.append(format_sentence(sentence))
        self.breeze_obs[cell] = has_breeze
        return told

    def pit_sentences(self) -> List[sympy.Basic]:
        """
        Return the told sentences with the observed breeze values substituted.

        Substituting `B_1_2 = True` into `Equivalent(B_1_2, P_1_1 | P_2_2)`
        leaves `P_1_1 | P_2_2`, a sentence over pit variables only.

        :return: sentences over the pit variables
        """
        substitutions = {
            breeze_symbol(cell): sympy.true if value else sympy.false
            for cell, value in self.breeze_obs.items()
        }
        out = []
        for sentence in self.sentences:
            grounded = sentence.subs(substitutions)
            # Told breeze literals collapse to `True` and constrain nothing.
            if grounded == sympy.true:
                continue
            out.append(grounded)
        return out

    def models(self) -> np.ndarray:
        """
        Return the mask marking $M(KB)$ over the pit variables.

        :return: boolean mask, one entry per candidate assignment
        """
        mask = satisfying_mask(self.pit_sentences(), self.var_order, self._bits)
        return mask

    def count_models(self) -> int:
        """
        Count the assignments still consistent with everything told.

        :return: $|M(KB)|$
        """
        return int(self.models().sum())

    def n_candidate_models(self) -> int:
        """
        Return the total number of candidate assignments, told or not.

        :return: $2^n$ where $n$ is the number of pit variables
        """
        return int(self._bits.shape[0])

    def ask(self, alpha: sympy.Basic) -> str:
        """
        Ask whether the knowledge base entails `alpha`.

        :param alpha: query sentence
        :return: one of `entailed`, `contradicted`, `unknown`, `inconsistent`
        """
        kb_mask = self.models()
        alpha_mask = satisfying_mask([alpha], self.var_order, self._bits)
        verdict = entailment_verdict(kb_mask, alpha_mask)
        _LOG.debug("alpha='%s' verdict='%s'", format_sentence(alpha), verdict)
        return verdict

    def pit_verdicts(self) -> Dict[Tuple[int, int], str]:
        """
        Ask "is this cell pit-free?" for every cell, reusing one enumeration.

        :return: map from cell to `no pit`, `pit`, or `?`
        """
        kb_mask = self.models()
        verdicts = {}
        for cell in self.world.cells:
            symbol = pit_symbol(cell)
            # A cell outside the model table is never constrained.
            if symbol not in self.var_order:
                verdicts[cell] = "?"
                continue
            alpha_mask = satisfying_mask(
                [sympy.Not(symbol)], self.var_order, self._bits
            )
            verdict = entailment_verdict(kb_mask, alpha_mask)
            if verdict == "entailed":
                verdicts[cell] = "no pit"
            elif verdict == "contradicted":
                verdicts[cell] = "pit"
            else:
                verdicts[cell] = "?"
        return verdicts


# #############################################################################
# Drawing helpers
# #############################################################################


def comment_panel(ax: matplotlib.axes.Axes, text: str) -> None:
    """
    Render a wheat-colored comment panel with a bold "Comments" title.

    :param ax: axes to draw on
    :param text: comment text showing the current variable state
    """
    ax.axis("off")
    ax.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    htutori.add_fitted_text_box(ax, text, max_fontsize=12, min_fontsize=7)


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


def draw_grid_cells(
    ax: matplotlib.axes.Axes,
    n_cols: int,
    n_rows: int,
    facecolors: Dict[Tuple[int, int], str],
    labels: Dict[Tuple[int, int], str],
    sublabels: Dict[Tuple[int, int], str],
    *,
    highlight: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Draw a grid of colored cells with a label and a sublabel in each.

    :param ax: axes to draw on
    :param n_cols: number of grid columns
    :param n_rows: number of grid rows
    :param facecolors: fill color per cell
    :param labels: bold text drawn in the upper part of each cell
    :param sublabels: small text drawn in the lower part of each cell
    :param highlight: optional cell to outline in bold
    """
    for col in range(1, n_cols + 1):
        for row in range(1, n_rows + 1):
            cell = (col, row)
            rect = mpatches.Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor=facecolors.get(cell, _COLOR_EMPTY),
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(rect)
            label = labels.get(cell, "")
            if label:
                ax.text(
                    col,
                    row + 0.22,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
            sublabel = sublabels.get(cell, "")
            if sublabel:
                ax.text(
                    col,
                    row - 0.18,
                    sublabel,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="dimgray",
                )
            # Always show the coordinates so cells can be named in the text.
            ax.text(
                col - 0.42,
                row - 0.40,
                "%d,%d" % cell,
                ha="left",
                va="center",
                fontsize=6,
                color="gray",
            )
    if highlight is not None:
        rect = mpatches.Rectangle(
            (highlight[0] - 0.5, highlight[1] - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor="darkblue",
            linewidth=3.5,
        )
        ax.add_patch(rect)
    ax.set_xlim(0.4, n_cols + 0.6)
    ax.set_ylim(0.4, n_rows + 0.6)
    ax.set_xticks(range(1, n_cols + 1))
    ax.set_yticks(range(1, n_rows + 1))
    ax.set_aspect("equal")
    ax.grid(False)


def draw_truth_grid(
    world: WumpusWorld,
    ax: matplotlib.axes.Axes,
    *,
    highlight: Optional[Tuple[int, int]] = None,
    title: str = "Hidden world (the agent cannot see this)",
) -> None:
    """
    Draw the hidden layout of pits, the wumpus, and the gold.

    :param world: world to draw
    :param ax: axes to draw on
    :param highlight: optional cell to outline in bold
    :param title: panel title
    """
    facecolors = {}
    labels = {}
    sublabels = {}
    for cell in world.cells:
        if cell in world.pits:
            facecolors[cell] = _COLOR_PIT
        elif cell == world.wumpus:
            facecolors[cell] = _COLOR_WUMPUS
        elif cell == world.gold:
            facecolors[cell] = _COLOR_GOLD
        elif cell == world.start:
            facecolors[cell] = _COLOR_START
        else:
            facecolors[cell] = _COLOR_EMPTY
        labels[cell] = world.truth_label(cell)
        sublabels[cell] = world.percept_label(cell)
    draw_grid_cells(
        ax,
        world.n_cols,
        world.n_rows,
        facecolors,
        labels,
        sublabels,
        highlight=highlight,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(
        "col\n\nthe true layout, with the percept each cell generates below it",
        fontsize=9,
    )


def draw_kb_grid(
    kb: KnowledgeBase,
    ax: matplotlib.axes.Axes,
    *,
    agent: Optional[Tuple[int, int]] = None,
    title: str = "What the KB knows (told percepts only)",
) -> None:
    """
    Draw what the knowledge base knows, cell by cell.

    Visited cells show the percept that was told, and every cell shows the
    answer to "is this cell pit-free?" obtained by `ASK`.

    :param kb: knowledge base to draw
    :param ax: axes to draw on
    :param agent: optional cell holding the agent, outlined in bold
    :param title: panel title
    """
    verdicts = kb.pit_verdicts()
    facecolors = {}
    labels = {}
    sublabels = {}
    for cell in kb.world.cells:
        if cell in kb.told_cells:
            facecolors[cell] = _COLOR_VISITED
        elif verdicts[cell] == "no pit":
            facecolors[cell] = _COLOR_SAFE
        elif verdicts[cell] == "pit":
            facecolors[cell] = _COLOR_PIT
        else:
            facecolors[cell] = _COLOR_UNKNOWN
        labels[cell] = verdicts[cell]
        if cell in kb.told_cells:
            sublabels[cell] = kb.world.percept_label(cell)
        else:
            sublabels[cell] = ""
    draw_grid_cells(
        ax,
        kb.world.n_cols,
        kb.world.n_rows,
        facecolors,
        labels,
        sublabels,
        highlight=agent,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(
        "col\n\nblue: visited, green: proved pit-free, grey: still unknown",
        fontsize=9,
    )


def draw_model_table(
    ax: matplotlib.axes.Axes,
    bits: np.ndarray,
    var_names: Sequence[str],
    kb_mask: np.ndarray,
    *,
    alpha_mask: Optional[np.ndarray] = None,
    alpha_name: str = "alpha",
    title: str = "Model table",
) -> None:
    """
    Draw the truth table over the model variables, shading the model sets.

    Each row is one candidate world. Rows in $M(KB)$ get a blue wash, and rows
    in $M(\\alpha)$ get a dashed orange outline, so inclusion and the gaps
    that break it are visible at a glance.

    :param ax: axes to draw on
    :param bits: assignments from `enumerate_assignments()`
    :param var_names: column headers, one per variable
    :param kb_mask: mask marking $M(KB)$
    :param alpha_mask: optional mask marking $M(\\alpha)$
    :param alpha_name: header used for the $M(\\alpha)$ indicator column
    :param title: panel title
    """
    n_models, n_vars = bits.shape
    # Build the image: variable columns, then one indicator column per set.
    columns = [bits[:, i].astype(int) for i in range(n_vars)]
    headers = list(var_names)
    columns.append(np.where(kb_mask, 2, 3))
    headers.append("M(KB)")
    if alpha_mask is not None:
        columns.append(np.where(alpha_mask, 4, 5))
        headers.append("M(%s)" % alpha_name)
    image = np.stack(columns, axis=1)
    cmap = mcolors.ListedColormap(
        [
            # 0: False, 1: True.
            "#f7f7f7",
            "#c6dbef",
            # 2: in M(KB), 3: outside M(KB).
            "#3182bd",
            "#ffffff",
            # 4: in M(alpha), 5: outside M(alpha).
            "#e6550d",
            "#ffffff",
        ]
    )
    ax.imshow(image, cmap=cmap, vmin=0, vmax=5, aspect="auto")
    # Wash the rows that belong to M(KB) so the shaded subset reads as a set.
    for row_index in np.flatnonzero(kb_mask):
        ax.add_patch(
            mpatches.Rectangle(
                (-0.5, row_index - 0.5),
                image.shape[1],
                1.0,
                facecolor="#3182bd",
                alpha=0.18,
                edgecolor="none",
            )
        )
    # Outline the rows that belong to M(alpha).
    if alpha_mask is not None:
        for row_index in np.flatnonzero(alpha_mask):
            ax.add_patch(
                mpatches.Rectangle(
                    (-0.5, row_index - 0.5),
                    image.shape[1],
                    1.0,
                    fill=False,
                    edgecolor="#e6550d",
                    linestyle="--",
                    linewidth=1.2,
                )
            )
    # Annotate the truth values only while they stay legible.
    if n_models <= 32:
        for row_index in range(n_models):
            for col_index in range(n_vars):
                ax.text(
                    col_index,
                    row_index,
                    "T" if bits[row_index, col_index] else "F",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    ax.set_xticks(range(len(headers)))
    ax.set_xticklabels(headers, rotation=90, fontsize=8)
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(
        "one row per candidate world\n\n"
        "blue wash: row satisfies the KB, dashed outline: row satisfies alpha",
        fontsize=9,
    )
    ax.grid(False)


def draw_count_bars(
    ax: matplotlib.axes.Axes,
    labels: Sequence[str],
    counts: Sequence[int],
    colors: Sequence[str],
    *,
    title: str,
    ylabel: str,
) -> None:
    """
    Draw a small bar chart of model or verdict counts, annotated with values.

    :param ax: axes to draw on
    :param labels: bar labels
    :param counts: bar heights
    :param colors: bar colors
    :param title: panel title
    :param ylabel: y-axis label
    """
    bars = ax.bar(list(labels), list(counts), color=list(colors))
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(list(counts) + [1]) * 1.25)
    ax.tick_params(axis="x", labelsize=9)


# #############################################################################
# Cell 1.1: The wumpus world grid and the knowledge base
# #############################################################################


def _parse_cell(text: str) -> Tuple[int, int]:
    """
    Parse a cell rendered as `(col, row)` back into a tuple.

    :param text: cell label, e.g., `(2, 3)`
    :return: cell as `(col, row)`, e.g., `(2, 3)`
    """
    col, row = text.strip("()").split(",")
    return (int(col), int(row))


def cell1_1_world_and_kb(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show the hidden world next to everything the knowledge base has been told.

    Interactive controls (ipywidgets):
    - `cell`: the cell whose percept is told into the `KB`
    - `TELL percept`: run `TELL` for the selected cell
    - `reset KB`: forget everything, keeping the same hidden layout
    - `seed`: random seed for the hidden pit, wumpus, and gold layout

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (17, 5)
    world = WumpusWorld()
    cell_dropdown = ipywidgets.Dropdown(
        options=[str(cell) for cell in world.cells],
        value=str(world.start),
        description="cell:",
        style={"description_width": "initial"},
    )
    tell_button = ipywidgets.Button(
        description="TELL percept",
        button_style="primary",
        layout=ipywidgets.Layout(width="140px"),
    )
    reset_button = ipywidgets.Button(
        description="reset KB",
        layout=ipywidgets.Layout(width="140px"),
    )
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed for the hidden layout",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=_DEFAULT_SEED,
        is_float=False,
    )
    output = ipywidgets.Output()
    state: Dict[str, Any] = {}

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            kb = state["kb"]
            selected = _parse_cell(cell_dropdown.value)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: the hidden truth, which the agent never sees.
            draw_truth_grid(state["world"], ax1, highlight=selected)
            # Panel 2: the same grid as reconstructed from the `KB` alone.
            draw_kb_grid(kb, ax2, agent=selected)
            # Panel 3: comments on the current state of the `KB`.
            told = state["told"]
            text = (
                "Parameters:\n"
                "  cell: %s\n"
                "  seed: %d\n\n"
                "Percept at the cell:\n"
                "  %s\n\n"
                "Last TELL:\n"
                "  %s\n\n"
                "KB state:\n"
                "  cells told: %d\n"
                "  sentences in KB: %d\n"
                "  |M(KB)|: %d of %d"
                % (
                    str(selected),
                    seed_slider.value,
                    state["world"].percept_label(selected),
                    "\n  ".join(told) if told else "(nothing told yet)",
                    len(kb.told_cells),
                    len(kb.sentences),
                    kb.count_models(),
                    kb.n_candidate_models(),
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    def rebuild(change: Optional[Any] = None) -> None:
        """
        Sample a fresh layout and forget everything the `KB` was told.
        """
        _ = change
        state["world"] = WumpusWorld(seed=seed_slider.value)
        state["kb"] = KnowledgeBase(state["world"])
        state["told"] = []
        update_plot()

    def on_tell(_button: Any) -> None:
        """
        Tell the percept at the selected cell into the `KB`.
        """
        cell = _parse_cell(cell_dropdown.value)
        state["told"] = state["kb"].tell_percept(cell)
        update_plot()

    param_info = make_param_info(
        {
            "cell": "the cell whose percept is told into the <code>KB</code>",
            "TELL percept": "adds three sentences: the cell holds no pit, "
            "the breeze axiom for the cell, and the observed breeze literal",
            "reset KB": "forgets every told sentence and resamples the layout",
            "seed": "random seed for the hidden pit, wumpus, and gold layout",
        }
    )
    cell_dropdown.observe(update_plot, names="value")
    tell_button.on_click(on_tell)
    reset_button.on_click(rebuild)
    seed_slider.observe(rebuild, names="value")
    rebuild()
    # Top row: controls on the left, info box on the right.
    # Bottom row: the two grids and the comments panel.
    controls = ipywidgets.VBox(
        [
            cell_dropdown,
            ipywidgets.HBox([tell_button, reset_button]),
            seed_box,
        ],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.1: Models and the breeze axiom
# #############################################################################

# The cell whose breeze percept drives the running entailment example.
_BREEZE_CELL = (1, 2)
# Pit variables spanning the model table, the three axiom variables first.
_MODEL_TABLE_CELLS = [(1, 1), (2, 2), (1, 3), (2, 1), (3, 1), (1, 4)]
# Number of pit variables used by the entailment and reasoner cells, small
# enough that every row of the truth table stays individually readable.
_ENTAILMENT_N_VARS = 4


def model_table_var_order(n_vars: int) -> List[sympy.Symbol]:
    """
    Return the first `n_vars` pit variables of the model table.

    The three variables of the breeze axiom come first, so the axiom is always
    inside the enumerated table.

    :param n_vars: number of pit variables to enumerate
    :return: pit variables, e.g., `[P_1_1, P_2_2, P_1_3]`
    """
    hdbg.dassert_lte(
        n_vars, len(_MODEL_TABLE_CELLS), "Not enough cells in the model table"
    )
    return [pit_symbol(cell) for cell in _MODEL_TABLE_CELLS[:n_vars]]


def breeze_example_sentences() -> Tuple[List[sympy.Basic], List[sympy.Basic]]:
    """
    Build the running `KB` for the breeze example at cell `(1, 2)`.

    The `KB` holds three sentences:
    - The breeze axiom `Equivalent(B_1_2, P_1_1 | P_1_3 | P_2_2)`
    - The observed percept `B_1_2`, a breeze was felt
    - The fact `~P_1_1`, the agent started in `(1, 1)` and survived

    :return: tuple of
        - told sentences, as displayed to the reader
        - the same sentences grounded on the pit variables, for model checking
    """
    world = WumpusWorld()
    axiom = breeze_axiom(world, _BREEZE_CELL)
    percept = breeze_symbol(_BREEZE_CELL)
    safe_start = sympy.Not(pit_symbol(world.start))
    told = [axiom, percept, safe_start]
    # Substitute the observed breeze value to leave sentences over pits only.
    grounded = [axiom.subs({percept: sympy.true}), safe_start]
    return told, grounded


def cell2_1_models_and_axiom(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Enumerate every model of the breeze axiom and shade the ones satisfying it.

    Interactive controls (ipywidgets):
    - `log2(models)`: the number of pit variables `n`, on a logarithmic slider
      because the number of models is $2^n$

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (16, 6)
    # The slider moves the exponent, so the displayed value is the model count.
    n_slider, n_box = htutori.build_log_widget_control(
        name="log2(models)",
        description="n (pit variables enumerated)",
        min_exp=3,
        max_exp=len(_MODEL_TABLE_CELLS),
        initial_exp=3,
        base=2,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            n_vars = n_slider.value
            var_order = model_table_var_order(n_vars)
            told, grounded = breeze_example_sentences()
            bits = enumerate_assignments(n_vars)
            kb_mask = satisfying_mask(grounded, var_order, bits)
            n_shaded = int(kb_mask.sum())
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: every candidate world, with the models of the KB shaded.
            draw_model_table(
                ax1,
                bits,
                [str(var) for var in var_order],
                kb_mask,
                title="Model table over %d pit variables" % n_vars,
            )
            # Panel 2: how much of the table survives the KB.
            draw_count_bars(
                ax2,
                ["satisfies KB", "violates KB"],
                [n_shaded, bits.shape[0] - n_shaded],
                ["#3182bd", "#d9d9d9"],
                title="Model count",
                ylabel="number of models",
            )
            # Panel 3: comments on the current enumeration.
            text = (
                "Parameters:\n"
                "  n (pit variables): %d\n\n"
                "Model counts:\n"
                "  candidate models 2^n: %d\n"
                "  models of KB (shaded): %d\n"
                "  ruled out: %d\n\n"
                "KB sentences:\n"
                "  %s"
                % (
                    n_vars,
                    bits.shape[0],
                    n_shaded,
                    bits.shape[0] - n_shaded,
                    "\n  ".join(format_sentence(s) for s in told),
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "log2(models)": "the number of pit variables <code>n</code> that "
            "the table enumerates; the row count is <code>2^n</code>, so the "
            "slider moves the exponent rather than the count",
        }
    )
    n_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [n_box], layout=ipywidgets.Layout(padding="0px 8px 0px 0px")
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.2: Entailment as model inclusion
# #############################################################################


def entailment_queries() -> Dict[str, sympy.Basic]:
    """
    Build the candidate queries used by the entailment and reasoner cells.

    Every query is a single clause over the first four pit variables, so that
    it can be decided both by model checking and by clause subsumption.

    :return: map from query label to query sentence
        ```

        {'no pit at (1,1)': ~P_1_1, 'pit at (1,1)': P_1_1, ...}

        ```
    """
    queries = {
        "no pit at (1,1)": sympy.Not(pit_symbol((1, 1))),
        "pit at (1,1)": pit_symbol((1, 1)),
        "no pit at (2,2)": sympy.Not(pit_symbol((2, 2))),
        "pit at (2,2)": pit_symbol((2, 2)),
        "pit at (2,2) or pit at (1,3)": sympy.Or(
            pit_symbol((2, 2)), pit_symbol((1, 3))
        ),
        "no pit at (2,1)": sympy.Not(pit_symbol((2, 1))),
    }
    return queries


def true_entailment_verdicts() -> Dict[str, str]:
    """
    Decide every candidate query by model checking, the entailment reference.

    :return: map from query label to verdict
    """
    var_order = model_table_var_order(_ENTAILMENT_N_VARS)
    _, grounded = breeze_example_sentences()
    bits = enumerate_assignments(_ENTAILMENT_N_VARS)
    kb_mask = satisfying_mask(grounded, var_order, bits)
    verdicts = {}
    for label, alpha in entailment_queries().items():
        alpha_mask = satisfying_mask([alpha], var_order, bits)
        verdicts[label] = entailment_verdict(kb_mask, alpha_mask)
    return verdicts


def cell2_2_entailment(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Decide `KB |= alpha` by checking whether `M(KB)` sits inside `M(alpha)`.

    Interactive controls (ipywidgets):
    - `alpha`: the query sentence whose model set is compared with `M(KB)`

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (16, 6)
    queries = entailment_queries()
    alpha_dropdown = ipywidgets.Dropdown(
        options=list(queries.keys()),
        value="pit at (2,2) or pit at (1,3)",
        description="alpha:",
        style={"description_width": "initial"},
        layout=ipywidgets.Layout(width="420px"),
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            label = alpha_dropdown.value
            alpha = queries[label]
            var_order = model_table_var_order(_ENTAILMENT_N_VARS)
            told, grounded = breeze_example_sentences()
            bits = enumerate_assignments(_ENTAILMENT_N_VARS)
            kb_mask = satisfying_mask(grounded, var_order, bits)
            alpha_mask = satisfying_mask([alpha], var_order, bits)
            verdict = entailment_verdict(kb_mask, alpha_mask)
            # Rows that satisfy the KB but violate alpha are the counterexamples
            # that break entailment.
            counterexamples = int((kb_mask & ~alpha_mask).sum())
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: the same table as before, now with two shaded sets.
            draw_model_table(
                ax1,
                bits,
                [str(var) for var in var_order],
                kb_mask,
                alpha_mask=alpha_mask,
                alpha_name="alpha",
                title="M(KB) against M(alpha)",
            )
            # Panel 2: the three counts that decide the verdict.
            draw_count_bars(
                ax2,
                ["M(KB)", "M(KB) and M(alpha)", "counterexamples"],
                [
                    int(kb_mask.sum()),
                    int((kb_mask & alpha_mask).sum()),
                    counterexamples,
                ],
                ["#3182bd", "#74c476", "#de2d26"],
                title="Model inclusion",
                ylabel="number of models",
            )
            # Panel 3: comments on the current query.
            text = (
                "Parameters:\n"
                "  alpha: %s\n"
                "  alpha sentence: %s\n\n"
                "Model sets:\n"
                "  |M(KB)|: %d\n"
                "  |M(alpha)|: %d\n"
                "  |M(KB) and M(alpha)|: %d\n"
                "  counterexample rows: %d\n\n"
                "M(KB) subset of M(alpha): %s\n"
                "Verdict: KB %s alpha\n\n"
                "KB sentences:\n"
                "  %s"
                % (
                    label,
                    format_sentence(alpha),
                    int(kb_mask.sum()),
                    int(alpha_mask.sum()),
                    int((kb_mask & alpha_mask).sum()),
                    counterexamples,
                    "yes" if counterexamples == 0 else "no",
                    verdict,
                    "\n  ".join(format_sentence(s) for s in told),
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "alpha": "the query sentence; entailment holds only when every "
            "row of <code>M(KB)</code> also lies in <code>M(alpha)</code>, so "
            "a single counterexample row is enough to break it",
        }
    )
    alpha_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [alpha_dropdown], layout=ipywidgets.Layout(padding="0px 8px 0px 0px")
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.3: Implication, entailment, and inference
# #############################################################################


def _draw_text_lines(
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
    step = 1.0 / (len(lines) + 1)
    for index, (text, color, weight) in enumerate(lines):
        ax.text(
            0.02,
            0.97 - index * step,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=10,
            color=color,
            fontweight=weight,
        )


def inference_trace() -> List[Tuple[str, str, str]]:
    """
    Build the step-by-step proof that walks from the `KB` to the conclusion.

    Each step names the rule that licenses it, so the trace is a procedure a
    computer can run, not a semantic claim about all models.

    :return: triples of `(text, color, fontweight)` ready for a text panel
    """
    told, _ = breeze_example_sentences()
    axiom, percept, safe_start = told
    world = WumpusWorld()
    neighbor_pits = [pit_symbol(n) for n in world.neighbors(_BREEZE_CELL)]
    # Modus ponens detaches the disjunction over all three neighbors.
    disjunction = sympy.Or(*neighbor_pits)
    # Resolution against the safe start removes the start cell from it.
    resolved = sympy.Or(
        *[p for p in neighbor_pits if p != pit_symbol(world.start)]
    )
    lines = [
        ("1. TELL   %s" % format_sentence(percept), "black", "normal"),
        ("          (percept: a breeze was felt)", "dimgray", "normal"),
        ("2. TELL   %s" % format_sentence(axiom), "black", "normal"),
        ("          (breeze axiom for the cell)", "dimgray", "normal"),
        ("3. TELL   %s" % format_sentence(safe_start), "black", "normal"),
        ("          (the agent stood in (1,1) and survived)", "dimgray", "normal"),
        ("4. Biconditional elimination on 2:", "#2166ac", "bold"),
        ("          B_1_2 >> (%s)" % format_sentence(disjunction), "#2166ac", "normal"),
        ("5. Modus ponens on 1 and 4:", "#2166ac", "bold"),
        ("          %s" % format_sentence(disjunction), "#2166ac", "normal"),
        ("6. Resolution on 3 and 5:", "#2166ac", "bold"),
        ("          %s" % format_sentence(resolved), "#2166ac", "normal"),
        ("Conclusion: at least one neighbor other than (1,1)", "#b2182b", "bold"),
        ("holds a pit, and neither one is entailed on its own.", "#b2182b", "bold"),
    ]
    return lines


def implication_view() -> List[Tuple[str, str, str]]:
    """
    Build the panel that reads the breeze axiom as one sentence in the logic.

    :return: triples of `(text, color, fontweight)` ready for a text panel
    """
    told, _ = breeze_example_sentences()
    axiom = told[0]
    lines = [
        ("The sentence itself, one object inside the logic:", "black", "bold"),
        ("  %s" % format_sentence(axiom), "black", "normal"),
        ("", "black", "normal"),
        ("Its connectives:", "black", "bold"),
        ("  Equivalent(.,.)  biconditional, 'if and only if'", "#2166ac", "normal"),
        ("  |                disjunction, 'or'", "#2166ac", "normal"),
        ("", "black", "normal"),
        ("Its atoms:", "black", "bold"),
        ("  B_1_2            a breeze is felt in (1,2)", "#006d2c", "normal"),
        ("  P_1_1            a pit sits in (1,1)", "#006d2c", "normal"),
        ("  P_1_3            a pit sits in (1,3)", "#006d2c", "normal"),
        ("  P_2_2            a pit sits in (2,2)", "#006d2c", "normal"),
        ("", "black", "normal"),
        ("Syntax only: nothing here mentions models.", "#b2182b", "bold"),
    ]
    return lines


def cell2_3_three_views(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Separate implication, entailment, and inference on the same example.

    Interactive controls (ipywidgets):
    - `view`: which of the three ideas the left panel shows

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (15, 6)
    definitions = {
        "implication": "a connective inside one sentence (syntax)",
        "entailment": "truth that follows in every model (semantics)",
        "inference": "a procedure a computer runs (computation)",
    }
    view_dropdown = ipywidgets.Dropdown(
        options=list(definitions.keys()),
        value="implication",
        description="view:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            view = view_dropdown.value
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            if view == "implication":
                _draw_text_lines(
                    ax1, implication_view(), title="Implication (syntactic)"
                )
            elif view == "entailment":
                # The same shaded table as Cell 2.2, for the entailed query.
                var_order = model_table_var_order(_ENTAILMENT_N_VARS)
                _, grounded = breeze_example_sentences()
                bits = enumerate_assignments(_ENTAILMENT_N_VARS)
                kb_mask = satisfying_mask(grounded, var_order, bits)
                alpha = entailment_queries()["pit at (2,2) or pit at (1,3)"]
                alpha_mask = satisfying_mask([alpha], var_order, bits)
                draw_model_table(
                    ax1,
                    bits,
                    [str(var) for var in var_order],
                    kb_mask,
                    alpha_mask=alpha_mask,
                    title="Entailment (semantic)",
                )
            else:
                _draw_text_lines(
                    ax1, inference_trace(), title="Inference (computational)"
                )
            # Panel 2: comments naming the active view and its definition.
            text = (
                "Parameters:\n"
                "  view: %s\n\n"
                "Definition:\n"
                "  %s\n\n"
                "Where it lives:\n"
                "  implication: inside one sentence\n"
                "  entailment: across all models\n"
                "  inference: inside an algorithm\n\n"
                "Same example throughout:\n"
                "  breeze at (1,2), start (1,1) known safe"
                % (view, definitions[view])
            )
            comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "view": "which of the three ideas to show: "
            "<code>implication</code> is syntax inside one sentence, "
            "<code>entailment</code> is truth across all models, and "
            "<code>inference</code> is the procedure that tracks it",
        }
    )
    view_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [view_dropdown], layout=ipywidgets.Layout(padding="0px 8px 0px 0px")
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 3.1: Soundness and completeness by breaking them
# #############################################################################


def _clause_literals(sentence: sympy.Basic) -> Optional[frozenset]:
    """
    Return the literal set of a sentence that is a single clause.

    :param sentence: sentence to inspect
    :return: frozen set of literals, or `None` if the sentence is not one
        clause, e.g., `P_1_1 | P_2_2` -> `frozenset({P_1_1, P_2_2})`
    """
    cnf = slboolal.to_cnf(sentence, simplify=False)
    conjuncts = list(slboolal.conjuncts(cnf))
    literals = None
    if len(conjuncts) == 1:
        literals = frozenset(slboolal.disjuncts(conjuncts[0]))
    return literals


def _subsumes(
    derived_literals: Sequence[frozenset],
    target: Optional[frozenset],
) -> bool:
    """
    Check whether any derived clause subsumes `target`.

    A clause subsumes another when its literals are a subset of theirs, since
    then the shorter clause already implies the longer one.

    :param derived_literals: literal sets of the derived clauses
    :param target: literal set to settle, or `None` when it is not one clause
    :return: True when some derived clause subsumes the target
    """
    found = False
    if target is not None:
        found = any(literals <= target for literals in derived_literals)
    return found


def verdict_from_derived(
    derived: Sequence[sympy.Basic],
    alpha: sympy.Basic,
) -> str:
    """
    Report what a rule-based reasoner can say about `alpha`.

    A reasoner only knows what it actually derived, so a derived clause settles
    `alpha` only when it subsumes `alpha`, that is when its literals are a
    subset of the literals of `alpha`.

    :param derived: sentences the reasoner derived
    :param alpha: query sentence, which must be a single clause
    :return: one of `entailed`, `contradicted`, `unknown`
    """
    alpha_literals = _clause_literals(alpha)
    hdbg.dassert_is_not(alpha_literals, None, "Query must be a single clause")
    negated_literals = _clause_literals(sympy.Not(alpha))
    # Sentences that are not a single clause are beyond what these reasoners
    # can use, so they are dropped.
    derived_literals = [
        literals
        for literals in (_clause_literals(d) for d in derived)
        if literals is not None
    ]
    verdict = "unknown"
    if _subsumes(derived_literals, alpha_literals):
        verdict = "entailed"
    elif _subsumes(derived_literals, negated_literals):
        verdict = "contradicted"
    return verdict


def derived_by_modus_ponens() -> List[sympy.Basic]:
    """
    Derive everything forward chaining with modus ponens alone can reach.

    The reasoner eliminates the biconditional and detaches its consequent, but
    it has no resolution rule, so it cannot combine the resulting disjunction
    with the fact that the start cell is safe.

    :return: derived sentences
    """
    world = WumpusWorld()
    told, _ = breeze_example_sentences()
    safe_start = told[2]
    neighbor_pits = [pit_symbol(n) for n in world.neighbors(_BREEZE_CELL)]
    derived = [safe_start, sympy.Or(*neighbor_pits)]
    return derived


def derived_by_affirming_consequent() -> List[sympy.Basic]:
    """
    Derive everything a reasoner that affirms the consequent reaches.

    The reasoner reads the biconditional as the one-way implication
    "a pit nearby causes a breeze", then commits the classic fallacy: a breeze
    was felt, so it affirms the consequent and declares every neighbor that is
    not already known safe to hold a pit.

    :return: derived sentences
    """
    world = WumpusWorld()
    told, _ = breeze_example_sentences()
    safe_start = told[2]
    derived = [safe_start]
    for cell in world.neighbors(_BREEZE_CELL):
        if cell != world.start:
            derived.append(pit_symbol(cell))
    return derived


def reasoner_verdicts(name: str) -> Tuple[Dict[str, str], List[sympy.Basic]]:
    """
    Run one of the three reasoners over every candidate query.

    :param name: reasoner name shown in the dropdown
    :return: tuple of
        - map from query label to the verdict the reasoner reports
        - the sentences the reasoner worked from
    """
    queries = entailment_queries()
    if name == "correct (model checking)":
        _, derived = breeze_example_sentences()
        verdicts = true_entailment_verdicts()
    elif name == "unsound (affirms consequent)":
        derived = derived_by_affirming_consequent()
        verdicts = {
            label: verdict_from_derived(derived, alpha)
            for label, alpha in queries.items()
        }
    elif name == "incomplete (modus ponens only)":
        derived = derived_by_modus_ponens()
        verdicts = {
            label: verdict_from_derived(derived, alpha)
            for label, alpha in queries.items()
        }
    else:
        raise ValueError(f"Unknown reasoner: {name}")
    return verdicts, derived


def _classify_verdict(true_verdict: str, reported: str) -> str:
    """
    Compare a reported verdict against the entailment ground truth.

    :param true_verdict: verdict from model checking
    :param reported: verdict the reasoner reported
    :return: one of `correct`, `false positive`, `false negative`
    """
    if reported == true_verdict:
        kind = "correct"
    elif reported == "unknown":
        # The reasoner failed to derive an entailed conclusion.
        kind = "false negative"
    else:
        # The reasoner asserted something entailment does not support.
        kind = "false positive"
    return kind


def _draw_verdict_table(
    ax: matplotlib.axes.Axes,
    true_verdicts: Dict[str, str],
    reported: Dict[str, str],
    *,
    title: str,
) -> None:
    """
    Draw one row per query, comparing the reasoner against the ground truth.

    :param ax: axes to draw on
    :param true_verdicts: verdicts from model checking
    :param reported: verdicts the reasoner reported
    :param title: panel title
    """
    kind_to_color = {
        "correct": _COLOR_CORRECT,
        "false positive": _COLOR_FALSE_POSITIVE,
        "false negative": _COLOR_FALSE_NEGATIVE,
    }
    labels = list(true_verdicts.keys())
    n_rows = len(labels)
    ax.axis("off")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, n_rows + 1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    # Header row sits above the query rows.
    for col, header in enumerate(["query alpha", "entailment", "reasoner"]):
        ax.text(
            col + 0.05,
            n_rows + 0.5,
            header,
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    for index, label in enumerate(labels):
        y = n_rows - 1 - index
        kind = _classify_verdict(true_verdicts[label], reported[label])
        ax.add_patch(
            mpatches.Rectangle(
                (0, y),
                3.0,
                1.0,
                facecolor=kind_to_color[kind],
                edgecolor="white",
                linewidth=1.5,
            )
        )
        ax.text(0.05, y + 0.5, label, ha="left", va="center", fontsize=8)
        ax.text(
            1.05,
            y + 0.5,
            true_verdicts[label],
            ha="left",
            va="center",
            fontsize=8,
        )
        ax.text(
            2.05,
            y + 0.5,
            reported[label],
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xlabel(
        "green: matches entailment, red: false positive, orange: false negative",
        fontsize=9,
    )


def cell3_1_soundness_completeness(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Run a correct, an unsound, and an incomplete reasoner on the same queries.

    Interactive controls (ipywidgets):
    - `reasoner`: which reasoner answers the queries

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (17, 5)
    reasoner_dropdown = ipywidgets.Dropdown(
        options=[
            "correct (model checking)",
            "unsound (affirms consequent)",
            "incomplete (modus ponens only)",
        ],
        value="correct (model checking)",
        description="reasoner:",
        style={"description_width": "initial"},
        layout=ipywidgets.Layout(width="420px"),
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            name = reasoner_dropdown.value
            truth = true_entailment_verdicts()
            reported, derived = reasoner_verdicts(name)
            kinds = [
                _classify_verdict(truth[label], reported[label])
                for label in truth
            ]
            n_false_positive = kinds.count("false positive")
            n_false_negative = kinds.count("false negative")
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: the verdicts, marked against the entailment reference.
            _draw_verdict_table(
                ax1, truth, reported, title="Reported verdicts vs entailment"
            )
            # Panel 2: the two failure modes, counted.
            draw_count_bars(
                ax2,
                ["correct", "false positive", "false negative"],
                [kinds.count("correct"), n_false_positive, n_false_negative],
                [_COLOR_CORRECT, _COLOR_FALSE_POSITIVE, _COLOR_FALSE_NEGATIVE],
                title="Failure modes",
                ylabel="number of queries",
            )
            # Panel 3: comments on the reasoner and its failure counts.
            text = (
                "Parameters:\n"
                "  reasoner: %s\n\n"
                "Counts over %d queries:\n"
                "  correct: %d\n"
                "  false positives (unsound): %d\n"
                "  false negatives (incomplete): %d\n\n"
                "Sound: %s\n"
                "Complete: %s\n\n"
                "Sentences the reasoner works from:\n"
                "  %s"
                % (
                    name,
                    len(truth),
                    kinds.count("correct"),
                    n_false_positive,
                    n_false_negative,
                    "yes" if n_false_positive == 0 else "no",
                    "yes" if n_false_negative == 0 else "no",
                    "\n  ".join(format_sentence(s) for s in derived),
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "reasoner": "which procedure answers the queries: the "
            "<code>correct</code> one enumerates models, the "
            "<code>unsound</code> one affirms the consequent, and the "
            "<code>incomplete</code> one has modus ponens but no resolution",
        }
    )
    reasoner_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [reasoner_dropdown], layout=ipywidgets.Layout(padding="0px 8px 0px 0px")
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 3.2: Model checking does not scale
# #############################################################################


def build_scaling_problem(
    size: int,
    *,
    seed: int = 42,
) -> Tuple[List[sympy.Basic], List[sympy.Symbol], sympy.Basic]:
    """
    Build the same safety query on a `size` x `size` grid.

    The `KB` holds the grounded breeze axioms of the start cell and of its
    neighbors, plus the fact that the start cell is safe. The query asks
    whether the first neighbor of the start cell is provably pit-free.

    :param size: number of rows and columns of the grid
    :param seed: random seed for the hidden layout
    :return: tuple of
        - sentences over the pit variables
        - pit variables, one per grid cell
        - the query sentence
    """
    world = WumpusWorld(n_cols=size, n_rows=size, seed=seed)
    var_order = [pit_symbol(cell) for cell in world.cells]
    sentences: List[sympy.Basic] = [sympy.Not(pit_symbol(world.start))]
    # Ground the axiom of every perceived cell with its observed breeze value.
    for cell in [world.start] + world.neighbors(world.start):
        axiom = breeze_axiom(world, cell)
        has_breeze = world.percept(cell)["breeze"]
        value = sympy.true if has_breeze else sympy.false
        sentences.append(axiom.subs({breeze_symbol(cell): value}))
    alpha = sympy.Not(pit_symbol(world.neighbors(world.start)[0]))
    return sentences, var_order, alpha


def time_call(func: Any, *, min_secs: float = 0.02) -> float:
    """
    Time a call accurately by repeating it until the total is measurable.

    A single call on a small grid finishes faster than the clock resolution, so
    the call is repeated until `min_secs` of work has accumulated and the total
    is divided by the number of repetitions. One warm-up call runs first, so
    that one-off import and cache costs stay out of the measurement.

    :param func: zero-argument callable to time
    :param min_secs: minimum total duration to accumulate
    :return: seconds spent per call
    """
    func()
    n_calls = 0
    elapsed = 0.0
    start = time.perf_counter()
    while elapsed < min_secs:
        func()
        n_calls += 1
        elapsed = time.perf_counter() - start
    return elapsed / n_calls


def measure_scaling(size: int, *, seed: int = _DEFAULT_SEED) -> Dict[str, float]:
    """
    Time brute-force model checking and a SAT solver on the same query.

    Model checking enumerates all $2^n$ assignments, so it is only measured
    while the enumeration fits in memory. The SAT solver answers the same
    question by searching, never materializing the model set.

    :param size: number of rows and columns of the grid
    :param seed: random seed for the hidden layout
    :return: dict with the grid size, variable count, and both runtimes
        ```

        {'size': 4.0, 'n_vars': 16.0, 'model_secs': 0.01, 'sat_secs': 0.004}

        ```
    """
    sentences, var_order, alpha = build_scaling_problem(size, seed=seed)
    n_vars = len(var_order)
    # Time the SAT solver on `KB and not alpha`, the standard entailment query.
    query = sympy.And(*sentences, sympy.Not(alpha))
    sat_secs = time_call(
        lambda: slinfere.satisfiable(query, algorithm="dpll2")
    )
    # Time the enumeration alone, with the CNF conversion left outside so that
    # the measurement tracks the $2^n$ growth and nothing else.
    model_secs = float("nan")
    if n_vars <= _MAX_ENUMERATED_VARS:
        kb_clauses = to_clauses(sentences, var_order)
        alpha_clauses = to_clauses([alpha], var_order)
        bits = enumerate_assignments(n_vars)

        def check_all_models() -> None:
            kb_mask = evaluate_clauses(kb_clauses, bits)
            alpha_mask = evaluate_clauses(alpha_clauses, bits)
            entailment_verdict(kb_mask, alpha_mask)

        model_secs = time_call(check_all_models)
    out = {
        "size": float(size),
        "n_vars": float(n_vars),
        "model_secs": model_secs,
        "sat_secs": sat_secs,
    }
    _LOG.debug("size=%s out=%s", size, str(out))
    return out


def model_checking_reference(
    measurements: Sequence[Dict[str, float]],
) -> List[float]:
    """
    Build the $2^n$ reference curve that model checking has to follow.

    Each extra variable doubles the number of models, so the curve anchors on
    the largest measured runtime and scales it by $2^{n - n_{ref}}$. Past the
    point where the enumeration stops fitting in memory this is the only way
    left to state the cost.

    :param measurements: results from `measure_scaling()`, by increasing size
    :return: reference runtime for every measurement, measured or not
    """
    measured = [m for m in measurements if not np.isnan(m["model_secs"])]
    hdbg.dassert_lte(1, len(measured), "At least one runtime must be measured")
    reference = measured[-1]
    projected = []
    for measurement in measurements:
        factor = 2.0 ** (measurement["n_vars"] - reference["n_vars"])
        projected.append(reference["model_secs"] * factor)
    return projected


def cell3_2_scaling(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Compare brute-force model checking against a SAT solver as the grid grows.

    Interactive controls (ipywidgets):
    - `grid_size`: the largest grid included in the runtime curve

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (15, 5)
    size_slider, size_box = htutori.build_widget_control(
        name="grid_size",
        description="largest grid side included in the curve",
        min_val=2,
        max_val=6,
        step=1,
        initial_value=4,
        is_float=False,
    )
    output = ipywidgets.Output()
    # Timings are reused across slider moves, so each size is measured once.
    cache: Dict[int, Dict[str, float]] = {}

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            sizes = list(range(2, size_slider.value + 1))
            for size in sizes:
                if size not in cache:
                    cache[size] = measure_scaling(size)
            measurements = [cache[size] for size in sizes]
            projected = model_checking_reference(measurements)
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            # Panel 1: runtime curves. Measured runtimes are solid, and the
            # 2^n reference that continues past them is dotted.
            ax1.plot(
                sizes,
                projected,
                marker="o",
                markerfacecolor="white",
                linestyle=":",
                linewidth=2.0,
                color="#b2182b",
                alpha=0.7,
                label="model checking (2^n reference)",
            )
            ax1.plot(
                sizes,
                [m["model_secs"] for m in measurements],
                marker="o",
                linewidth=2.5,
                color="#b2182b",
                label="model checking (measured)",
            )
            ax1.plot(
                sizes,
                [m["sat_secs"] for m in measurements],
                marker="s",
                linewidth=2.5,
                color="#2166ac",
                label="SAT solver (measured)",
            )
            ax1.set_yscale("log")
            ax1.set_xticks(sizes)
            ax1.set_xticklabels(["%dx%d" % (s, s) for s in sizes])
            ax1.set_xlabel(
                "grid size\n\n"
                "runtime of the same safety query, on a log scale",
                fontsize=9,
            )
            ax1.set_ylabel("runtime (s)", fontsize=11)
            ax1.set_title(
                "Runtime vs grid size", fontsize=13, fontweight="bold"
            )
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            # Panel 2: comments on the largest grid in the curve.
            last = measurements[-1]
            model_secs = last["model_secs"]
            if np.isnan(model_secs):
                model_text = "%.3g s (projected)" % projected[-1]
            else:
                model_text = "%.3g s (measured)" % model_secs
            text = (
                "Parameters:\n"
                "  grid_size: %dx%d\n\n"
                "At the largest grid:\n"
                "  pit variables n: %d\n"
                "  candidate models 2^n: %.3g\n"
                "  model checking: %s\n"
                "  SAT solver: %.3g s\n\n"
                "Speedup at this size: %.1fx\n\n"
                "Runtimes above %d variables are projected,\n"
                "since the enumeration no longer fits in memory."
                % (
                    int(last["size"]),
                    int(last["size"]),
                    int(last["n_vars"]),
                    2.0 ** last["n_vars"],
                    model_text,
                    last["sat_secs"],
                    (
                        projected[-1]
                        if np.isnan(model_secs)
                        else model_secs
                    )
                    / last["sat_secs"],
                    _MAX_ENUMERATED_VARS,
                )
            )
            comment_panel(ax2, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "grid_size": "the largest grid side included in the curve; the "
            "number of pit variables is <code>size^2</code>, so the model "
            "count grows as <code>2^(size^2)</code>",
        }
    )
    size_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [size_box], layout=ipywidgets.Layout(padding="0px 8px 0px 0px")
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 4.1: From propositional rules to first-order sentences
# #############################################################################


def first_order_instantiation(
    world: WumpusWorld,
    cell: Tuple[int, int],
) -> List[Tuple[str, str, str]]:
    """
    Walk the first-order breeze rule down to the propositional axiom.

    :param world: world supplying the grid geometry
    :param cell: cell the universal rule is instantiated for
    :return: triples of `(text, color, fontweight)` ready for a text panel
    """
    neighbors = world.neighbors(cell)
    ground_or = " or ".join("Pit%s" % str(n) for n in neighbors)
    lines = [
        ("One universal rule for the whole grid:", "black", "bold"),
        ("  forall x, y:", "#2166ac", "normal"),
        ("    Breeze(x, y) <=>", "#2166ac", "normal"),
        ("      exists x', y':", "#2166ac", "normal"),
        ("        Adjacent(x, y, x', y') and Pit(x', y')", "#2166ac", "normal"),
        ("", "black", "normal"),
        ("Universal instantiation, {x/%d, y/%d}:" % cell, "black", "bold"),
        ("  Breeze%s <=>" % str(cell), "#006d2c", "normal"),
        ("    exists x', y':", "#006d2c", "normal"),
        ("      Adjacent(%d, %d, x', y') and Pit(x', y')" % cell, "#006d2c", "normal"),
        ("", "black", "normal"),
        ("Existential instantiation over the neighbors:", "black", "bold"),
        ("  Breeze%s <=> %s" % (str(cell), ground_or), "#006d2c", "normal"),
        ("", "black", "normal"),
        ("The propositional axiom from Cell 2.1:", "black", "bold"),
        ("  %s" % format_sentence(breeze_axiom(world, cell)), "#b2182b", "bold"),
    ]
    return lines


def cell4_1_first_order(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Rewrite the breeze axiom as one quantified sentence, then ground it.

    Interactive controls (ipywidgets):
    - `cell`: the cell the universal rule is instantiated for

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (17, 5)
    world = WumpusWorld()
    cell_dropdown = ipywidgets.Dropdown(
        options=[str(cell) for cell in world.cells],
        value=str((2, 1)),
        description="cell:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            cell = _parse_cell(cell_dropdown.value)
            neighbors = world.neighbors(cell)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: the quantified rule and its two instantiation steps.
            _draw_text_lines(
                ax1,
                first_order_instantiation(world, cell),
                title="From one rule to one axiom",
            )
            # Panel 2: the ground literals the instantiation produced.
            facecolors = {}
            labels = {}
            sublabels = {}
            for grid_cell in world.cells:
                if grid_cell == cell:
                    facecolors[grid_cell] = _COLOR_START
                    labels[grid_cell] = "Breeze"
                    sublabels[grid_cell] = "%d,%d" % grid_cell
                elif grid_cell in neighbors:
                    facecolors[grid_cell] = _COLOR_GOLD
                    labels[grid_cell] = "Pit?"
                    sublabels[grid_cell] = "%d,%d" % grid_cell
                else:
                    facecolors[grid_cell] = _COLOR_EMPTY
                    labels[grid_cell] = ""
                    sublabels[grid_cell] = ""
            draw_grid_cells(
                ax2,
                world.n_cols,
                world.n_rows,
                facecolors,
                labels,
                sublabels,
                highlight=cell,
            )
            ax2.set_title(
                "Grounded instance", fontsize=13, fontweight="bold"
            )
            ax2.set_xlabel(
                "col\n\nblue: the cell bound to (x, y), "
                "yellow: the witnesses bound to (x', y')",
                fontsize=9,
            )
            # Panel 3: comments on the chosen instantiation.
            text = (
                "Parameters:\n"
                "  cell: %s\n\n"
                "Instantiation:\n"
                "  {x/%d, y/%d}\n"
                "  witnesses: %s\n\n"
                "Ground sentence:\n"
                "  %s\n\n"
                "Propositional axioms replaced\n"
                "by this one rule: %d"
                % (
                    str(cell),
                    cell[0],
                    cell[1],
                    ", ".join(str(n) for n in neighbors),
                    format_sentence(breeze_axiom(world, cell)),
                    len(world.cells),
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "cell": "the cell the universal rule is instantiated for; "
            "universal instantiation binds <code>x, y</code> to it, and "
            "existential instantiation picks witnesses among its neighbors",
        }
    )
    cell_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [cell_dropdown], layout=ipywidgets.Layout(padding="0px 8px 0px 0px")
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 4.2: The agent loop, KB grows and candidate models shrink
# #############################################################################


def stench_suspects(kb: KnowledgeBase) -> set:
    """
    Return the cells that might still hold the wumpus.

    The propositional `KB` reasons about pits only, so wumpus avoidance uses
    the stench percepts directly: any unvisited neighbor of a cell that smelled
    of wumpus is a suspect and is left alone.

    :param kb: knowledge base holding the told percepts
    :return: set of suspect cells
    """
    suspects = set()
    for cell in kb.told_cells:
        if kb.world.percept(cell)["stench"]:
            suspects.update(
                n for n in kb.world.neighbors(cell) if n not in kb.told_cells
            )
    return suspects


def next_safe_move(
    kb: KnowledgeBase,
    visited: Sequence[Tuple[int, int]],
    current: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """
    Pick an unvisited cell that the `KB` proves pit-free, if one exists.

    Neighbors of the current cell come first so the walk stays contiguous, and
    the frontier of every visited cell is considered after that.

    :param kb: knowledge base answering the safety queries
    :param visited: cells already visited, in visit order
    :param current: cell the agent is standing in
    :return: the chosen cell, or `None` when nothing is provably safe
    """
    verdicts = kb.pit_verdicts()
    suspects = stench_suspects(kb)
    candidates = list(kb.world.neighbors(current))
    for cell in visited:
        candidates.extend(kb.world.neighbors(cell))
    choice = None
    for cell in candidates:
        provably_safe = verdicts[cell] == "no pit" and cell not in suspects
        if cell not in visited and provably_safe:
            choice = cell
            break
    _LOG.debug("current=%s choice=%s", str(current), str(choice))
    return choice


def cell4_2_agent_loop(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Step a knowledge-based agent and watch the candidate model set shrink.

    Interactive controls (ipywidgets):
    - `step`: move the agent to one provably safe cell and `TELL` its percept
    - `reset`: return the agent to the start and forget everything
    - `seed`: random seed for the hidden layout

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = (17, 5)
    step_button = ipywidgets.Button(
        description="step",
        button_style="primary",
        layout=ipywidgets.Layout(width="140px"),
    )
    reset_button = ipywidgets.Button(
        description="reset",
        layout=ipywidgets.Layout(width="140px"),
    )
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed for the hidden layout",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=_DEFAULT_SEED,
        is_float=False,
    )
    output = ipywidgets.Output()
    state: Dict[str, Any] = {}

    def update_plot(change: Optional[Any] = None) -> None:
        _ = change
        with output:
            clear_output(wait=True)
            kb = state["kb"]
            history = state["history"]
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: the agent and every percept told so far.
            draw_kb_grid(
                kb,
                ax1,
                agent=state["agent"],
                title="Agent at %s after %d steps"
                % (str(state["agent"]), len(state["visited"])),
            )
            # Panel 2: the candidate model count after each TELL.
            ax2.plot(
                range(len(history)),
                history,
                marker="o",
                linewidth=2.5,
                color="#2166ac",
            )
            ax2.set_yscale("log")
            ax2.set_xlabel(
                "TELL number\n\n"
                "every TELL can only remove models, never add them",
                fontsize=9,
            )
            ax2.set_ylabel("|M(KB)|", fontsize=11)
            ax2.set_title(
                "Candidate models remaining", fontsize=13, fontweight="bold"
            )
            ax2.grid(True, alpha=0.3)
            # Panel 3: comments on the current step.
            verdicts = kb.pit_verdicts()
            n_proved_safe = sum(1 for v in verdicts.values() if v == "no pit")
            text = (
                "Parameters:\n"
                "  seed: %d\n\n"
                "Agent state:\n"
                "  step: %d\n"
                "  position: %s\n"
                "  cells visited: %d\n\n"
                "Last action:\n"
                "  %s\n\n"
                "KB state:\n"
                "  sentences in KB: %d\n"
                "  |M(KB)|: %d of %d\n"
                "  cells proved pit-free: %d of %d"
                % (
                    seed_slider.value,
                    len(state["visited"]),
                    str(state["agent"]),
                    len(state["visited"]),
                    state["last"],
                    len(kb.sentences),
                    kb.count_models(),
                    kb.n_candidate_models(),
                    n_proved_safe,
                    len(kb.world.cells),
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    def rebuild(change: Optional[Any] = None) -> None:
        """
        Put the agent back in the start cell with an empty knowledge base.
        """
        _ = change
        world = WumpusWorld(seed=seed_slider.value)
        kb = KnowledgeBase(world)
        state["world"] = world
        state["kb"] = kb
        state["agent"] = world.start
        state["visited"] = []
        # The first entry is the empty KB, when every assignment is possible.
        state["history"] = [kb.n_candidate_models()]
        # The agent perceives as soon as it enters the start cell.
        told = kb.tell_percept(world.start)
        state["visited"].append(world.start)
        state["history"].append(kb.count_models())
        state["last"] = "TELL at %s: %s" % (str(world.start), "; ".join(told))
        update_plot()

    def on_step(_button: Any) -> None:
        """
        Move the agent one cell and tell the percept it finds there.
        """
        kb = state["kb"]
        choice = next_safe_move(kb, state["visited"], state["agent"])
        if choice is None:
            state["last"] = (
                "no provably safe move left: the KB entails safety\n"
                "  for no unvisited frontier cell"
            )
        else:
            told = kb.tell_percept(choice)
            state["agent"] = choice
            state["visited"].append(choice)
            state["history"].append(kb.count_models())
            state["last"] = "move to %s and TELL: %s" % (
                str(choice),
                "; ".join(told),
            )
        update_plot()

    param_info = make_param_info(
        {
            "step": "moves the agent to one cell the <code>KB</code> proves "
            "pit-free, then <code>TELL</code>s the percept found there",
            "reset": "returns the agent to the start with an empty "
            "<code>KB</code>",
            "seed": "random seed for the hidden pit, wumpus, and gold layout",
        }
    )
    step_button.on_click(on_step)
    reset_button.on_click(rebuild)
    seed_slider.observe(rebuild, names="value")
    rebuild()
    controls = ipywidgets.VBox(
        [
            ipywidgets.HBox([step_button, reset_button]),
            seed_box,
        ],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))
