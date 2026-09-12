"""
Utility functions for the ontology and description-logic reasoning lesson.

Builds OWL ontologies with `owlready2` and calls the HermiT description-logic
reasoner on them:
- A university ontology of classes, properties, and one cardinality axiom.
- A subset of the Manchester Pizza ontology, used for subsumption and for a
  deliberately inconsistent vegetarian pizza.
- The asserted class hierarchy compared against the inferred one.
- An axiom editor that re-runs the reasoner after every edit.
- The unsatisfiable `FlyingPenguin` concept.
- Instance-level realization, retrieval, and the open world assumption.
- Reasoner runtime as cardinality constraints and property chains are added.
- Interactive notebook cells built on top of these primitives.

The Pizza ontology is rebuilt here as a small subset instead of being
downloaded, so that the cells run with no network access and so that every
axiom the reasoner uses is visible in this file.

Every builder returns a fresh `owlready2.World`, since `sync_reasoner()`
writes the inferred triples back into the world it is given. Rebuilding is
what keeps repeated runs independent of each other.

Import as:

import msml610.tutorials.L03_knowledge_representation.L03_04_ontology_reasoning_utils as mtlkrl0oru
"""

import dataclasses
import logging
import textwrap
import time
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import ipywidgets
import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import owlready2
import pandas as pd
from IPython.display import clear_output, display

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.htutorial as htutori

_LOG = logging.getLogger(__name__)

# #############################################################################
# Colors and display constants
# #############################################################################

# Colors marking where a subclass edge or a class comes from.
_COLOR_ASSERTED = "#a9cce3"
_COLOR_INFERRED = "#a8e6a3"
_COLOR_PLAIN = "#e8e8e8"
_COLOR_UNSATISFIABLE = "#f4a6a6"
_COLOR_SELECTED = "#f5b041"
_COLOR_INDIVIDUAL = "#d8b4e2"
# Edge colors matching the node colors above.
_EDGE_ASSERTED = "#5499c7"
_EDGE_INFERRED = "#52be80"
_EDGE_PLAIN = "#999999"
# Default figure size for a 1x3 panel row, wide enough for the graphs.
_FIGSIZE = (17, 5)
# Default figure size for a 1x4 panel row.
_FIGSIZE_WIDE = (19, 5)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger so notebook cells see the
    debug and info output of these utility functions.

    :param notebook_log: logger owned by the notebook
    """
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Ontology plumbing
# #############################################################################

# Base IRI shared by every ontology built here.
_BASE_IRI = "http://umd.edu/msml610/%s.owl"


def new_ontology(name: str) -> Tuple[owlready2.World, owlready2.Ontology]:
    """
    Create an empty ontology inside a fresh, private world.

    :param name: short ontology name, e.g., `pizza`
    :return: the world holding the ontology, and the ontology itself
    """
    world = owlready2.World()
    onto = world.get_ontology(_BASE_IRI % name)
    return world, onto


# Symbol printed for each kind of OWL restriction, in ASCII Manchester syntax.
_RESTRICTION_LABELS = {
    owlready2.SOME: "some",
    owlready2.ONLY: "only",
    owlready2.EXACTLY: "exactly",
    owlready2.MIN: "min",
    owlready2.MAX: "max",
    owlready2.VALUE: "value",
}


def render_concept(concept: Any) -> str:
    """
    Render a class expression in ASCII Manchester syntax.

    The DL renderer shipped with `owlready2` emits unicode operators, which
    the notebook conventions forbid, so the few constructs used in this lesson
    are rendered here instead.

    :param concept: class, restriction, or boolean combination of them
    :return: rendering, e.g., `hasTopping only (CheeseTopping or
        VegetableTopping)`
    """
    if isinstance(concept, owlready2.Restriction):
        label = _RESTRICTION_LABELS[concept.type]
        filler = render_concept(concept.value)
        if concept.type in (owlready2.EXACTLY, owlready2.MIN, owlready2.MAX):
            # Qualified number restrictions carry a cardinality too.
            out = "%s %s %s %s" % (
                concept.property.name,
                label,
                concept.cardinality,
                filler,
            )
        else:
            out = "%s %s %s" % (concept.property.name, label, filler)
    elif isinstance(concept, owlready2.And):
        out = "(%s)" % " and ".join(render_concept(c) for c in concept.Classes)
    elif isinstance(concept, owlready2.Or):
        out = "(%s)" % " or ".join(render_concept(c) for c in concept.Classes)
    elif isinstance(concept, owlready2.Not):
        out = "not %s" % render_concept(concept.Class)
    elif isinstance(concept, owlready2.ThingClass):
        out = concept.name
    else:
        # Individuals used as the filler of a `value` restriction land here.
        out = getattr(concept, "name", str(concept))
    return out


def count_axioms(onto: owlready2.Ontology) -> int:
    """
    Count the class-level axioms asserted in an ontology.

    Subclass axioms, equivalence axioms, and disjointness axioms are all
    counted, since each of them is a statement the reasoner can use.

    :param onto: ontology to inspect
    :return: number of asserted class axioms
    """
    n_axioms = 0
    for cls in onto.classes():
        n_axioms += len(cls.is_a)
        n_axioms += len(cls.equivalent_to)
    n_axioms += len(list(onto.disjoint_classes()))
    return n_axioms


# #############################################################################
# Reasoning
# #############################################################################


@dataclasses.dataclass(frozen=True)
class Hierarchy:
    """
    A class hierarchy, as a set of subclass edges over named classes.
    """

    # Subclass edges as `(parent, child)`, e.g., `("Pizza", "Margherita")`.
    edges: Tuple[Tuple[str, str], ...]
    # Every named class in the ontology, including the ones with no edge.
    classes: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Classification:
    """
    What the reasoner did to one ontology: the hierarchy before and after.
    """

    # Hierarchy as the author wrote it.
    asserted: Hierarchy
    # Hierarchy after the reasoner has classified the ontology.
    inferred: Hierarchy
    # Classes the reasoner proved cannot have any member.
    unsatisfiable: Tuple[str, ...]
    # Wall-clock time of the `sync_reasoner()` call, in seconds.
    seconds: float

    def new_edges(self) -> Tuple[Tuple[str, str], ...]:
        """
        Return the subclass edges the reasoner added to the asserted ones.

        :return: edges present after classification and absent before
        """
        asserted = set(self.asserted.edges)
        return tuple(e for e in self.inferred.edges if e not in asserted)


def extract_hierarchy(onto: owlready2.Ontology) -> Hierarchy:
    """
    Read the subclass edges currently stored in an ontology.

    Called once before `sync_reasoner()` and once after it, so that the
    asserted and the inferred hierarchy can be compared.

    :param onto: ontology to read
    :return: hierarchy over the named classes of the ontology
    """
    classes = sorted(cls.name for cls in onto.classes())
    edges = set()
    for cls in onto.classes():
        for parent in cls.is_a:
            # Skip the anonymous class expressions: only named parents are
            # edges of the hierarchy.
            if isinstance(parent, owlready2.ThingClass):
                edges.add((parent.name, cls.name))
    # Two classes the reasoner proves equivalent point at each other, which
    # would turn the hierarchy into a cyclic graph. Keep one of the two
    # directions so that the graph stays acyclic and can be laid out in rows.
    edges = {(p, c) for p, c in edges if (c, p) not in edges or p < c}
    return Hierarchy(tuple(sorted(edges)), tuple(classes))


def run_reasoner(world: owlready2.World) -> float:
    """
    Run HermiT on a world and write the inferred facts back into it.

    :param world: world holding the ontology to classify
    :return: wall-clock time of the reasoner call, in seconds
    """
    start = time.perf_counter()
    # `debug=0` keeps the HermiT command line out of the notebook output.
    owlready2.sync_reasoner(world, debug=0)
    seconds = time.perf_counter() - start
    _LOG.debug("reasoner ran in '%.3f' s", seconds)
    return seconds


def classify(onto: owlready2.Ontology) -> Classification:
    """
    Classify an ontology and report the hierarchy before and after.

    :param onto: ontology to classify, in its own world
    :return: asserted hierarchy, inferred hierarchy, and unsatisfiable classes
    """
    asserted = extract_hierarchy(onto)
    seconds = run_reasoner(onto.world)
    # `owl.Nothing` is unsatisfiable by definition, so it says nothing about
    # the ontology and is dropped from the report.
    unsatisfiable = tuple(
        sorted(
            cls.name
            for cls in onto.world.inconsistent_classes()
            if cls.name != "Nothing"
        )
    )
    inferred = extract_hierarchy(onto)
    # An unsatisfiable class is a subclass of every class at once, so its
    # inferred edges would swamp the diagram without saying anything. Keep the
    # edges its author wrote and mark the class itself instead.
    asserted_edges = set(asserted.edges)
    edges = tuple(
        e
        for e in inferred.edges
        if e[1] not in unsatisfiable or e in asserted_edges
    )
    inferred = Hierarchy(edges, inferred.classes)
    return Classification(asserted, inferred, unsatisfiable, seconds)


def find_justification(
    classify_fn: Callable[[Tuple[str, ...]], Classification],
    axioms: Sequence[str],
    class_name: str,
) -> Tuple[str, ...]:
    """
    Find the axioms responsible for a class being unsatisfiable.

    Uses the reasoner as an oracle, which is the black-box way of computing a
    justification: an axiom belongs to the justification when dropping it, and
    nothing else, makes the class satisfiable again.

    :param classify_fn: classifies the ontology built from a set of axioms
    :param axioms: axiom keys that are candidates for the justification
    :param class_name: unsatisfiable class to explain, e.g.,
        `VegetarianAmericanPizza`
    :return: keys of the axioms in the justification
    """
    hdbg.dassert_in(class_name, classify_fn(tuple(axioms)).unsatisfiable)
    justification = []
    for key in axioms:
        # Rebuild the ontology without this one axiom and ask again.
        without = tuple(k for k in axioms if k != key)
        if class_name not in classify_fn(without).unsatisfiable:
            justification.append(key)
    _LOG.debug("justification for '%s'='%s'", class_name, justification)
    return tuple(justification)


# #############################################################################
# Drawing helpers
# #############################################################################


def wrap_comment(text: str, *, indent: str = "    ") -> str:
    """
    Wrap one long comment line so that it stays inside the comments panel.

    :param text: line to wrap, e.g., a full axiom with its filler
    :param indent: prefix added to every continuation line
    :return: wrapped text, e.g., `VegetarianPizza EquivalentTo\\n    Pizza ...`
    """
    return textwrap.fill(text, width=44, subsequent_indent=indent)


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


def draw_table(
    ax: matplotlib.axes.Axes,
    rows: Sequence[Sequence[str]],
    col_labels: Sequence[str],
    *,
    title: str,
    row_colors: Sequence[str] = (),
    col_widths: Sequence[float] = (),
    font_size: int = 9,
) -> None:
    """
    Draw a table of strings inside an axes, with one color per row.

    :param ax: axes to draw on
    :param rows: table body, one sequence of cells per row
    :param col_labels: column headers
    :param title: bold title drawn above the table
    :param row_colors: fill color per row, empty to leave the rows white
    :param col_widths: relative column widths, empty for equal widths
    :param font_size: font size of the table cells
    """
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if not rows:
        # `ax.table()` cannot render an empty body, so say so explicitly.
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center", fontsize=11)
    else:
        table = ax.table(
            cellText=[list(row) for row in rows],
            colLabels=list(col_labels),
            colWidths=list(col_widths) if col_widths else None,
            cellLoc="left",
            loc="upper center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 1.45)
        # Style the header row.
        for col in range(len(col_labels)):
            cell = table[(0, col)]
            cell.set_facecolor("#d9d9d9")
            cell.set_text_props(fontweight="bold")
        # Color each body row by its status.
        for row_index, color in enumerate(row_colors):
            for col in range(len(col_labels)):
                table[(row_index + 1, col)].set_facecolor(color)


def build_hierarchy_graph(
    edges: Sequence[Tuple[str, str]],
    classes: Sequence[str] = (),
) -> nx.DiGraph:
    """
    Build the directed graph of a class hierarchy, parents pointing at children.

    :param edges: subclass edges as `(parent, child)`
    :param classes: classes to keep as isolated nodes even with no edge
    :return: graph whose nodes are class names
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(classes)
    graph.add_edges_from(edges)
    return graph


def attach_orphans_to_root(graph: nx.DiGraph, root: str) -> nx.DiGraph:
    """
    Hang every parentless class of a graph under the root class.

    Classification replaces the `Thing` parent of a class as soon as it proves
    a more specific one, and that more specific parent can be a class the
    diagram leaves out. Re-attaching keeps such a class inside the tree
    instead of floating next to the root.

    :param graph: hierarchy graph, possibly with parentless classes
    :param root: class every other class hangs from, e.g., `Thing`
    :return: the same graph with the missing root edges added
    """
    for node in list(graph.nodes):
        if node != root and graph.in_degree(node) == 0:
            graph.add_edge(root, node)
    return graph


def hierarchy_positions(graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """
    Lay out a class hierarchy in rows, one row per distance from the root.

    The graph flows downward, so a class always sits below every one of its
    superclasses.

    :param graph: acyclic hierarchy graph to lay out
    :return: position per node, e.g., `{'Margherita': (1.0, -2.0)}`
    """
    hdbg.dassert(nx.is_directed_acyclic_graph(graph), "the graph has a cycle")
    depth: Dict[str, int] = {}
    for node in nx.topological_sort(graph):
        parents = list(graph.predecessors(node))
        depth[node] = 0 if not parents else max(depth[p] for p in parents) + 1
    # Spread the classes of each row horizontally, centered on zero.
    positions: Dict[str, Tuple[float, float]] = {}
    for level in sorted(set(depth.values())):
        row = sorted(node for node in graph.nodes if depth[node] == level)
        for index, node in enumerate(row):
            positions[node] = (
                float(index) - (len(row) - 1) / 2.0,
                float(-level),
            )
    return positions


def draw_hierarchy(
    ax: matplotlib.axes.Axes,
    graph: nx.DiGraph,
    positions: Dict[str, Tuple[float, float]],
    node_colors: Dict[str, str],
    *,
    title: str,
    edge_colors: Optional[Dict[Tuple[str, str], str]] = None,
    crossed_out: Sequence[str] = (),
    font_size: int = 7,
) -> None:
    """
    Draw a class hierarchy, one rounded box per class.

    Each class is drawn as a text box rather than as a marker, so that a long
    name such as `PeperoniSausageTopping` still fits inside its node.

    :param ax: axes to draw on
    :param graph: hierarchy graph to draw
    :param positions: position per node
    :param node_colors: fill color per node
    :param title: bold title drawn above the graph
    :param edge_colors: color per edge, gray for the edges left out
    :param crossed_out: classes to draw crossed out, i.e., unsatisfiable ones
    :param font_size: font size of the class labels
    """
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    edge_colors = edge_colors or {}
    # Draw the edges first, so that the class boxes cover their endpoints.
    for edge in graph.edges:
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[edge],
            edge_color=edge_colors.get(edge, _EDGE_PLAIN),
            width=2.0 if edge in edge_colors else 1.0,
            node_size=900,
            ax=ax,
        )
    for node in graph.nodes:
        x, y = positions[node]
        ax.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            fontsize=font_size,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=node_colors.get(node, _COLOR_PLAIN),
                edgecolor="#555555",
            ),
        )
        if node in crossed_out:
            # An unsatisfiable class can never have a member, so its box is
            # struck through.
            ax.plot(
                [x - 0.42, x + 0.42],
                [y, y],
                color="#c0392b",
                linewidth=1.8,
                zorder=5,
            )
    # `ax.text()` does not grow the data limits, so set them from the layout.
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)


def draw_property_edges(
    ax: matplotlib.axes.Axes,
    positions: Dict[str, Tuple[float, float]],
    properties: Sequence[Tuple[str, str, str]],
) -> None:
    """
    Draw the object properties of an ontology as labelled dashed arrows.

    :param ax: axes already holding the class hierarchy
    :param positions: position per class, as used for the hierarchy
    :param properties: properties as `(domain, name, range)`, e.g.,
        `("Student", "takesCourse", "Course")`
    """
    for domain, name, range_ in properties:
        x0, y0 = positions[domain]
        x1, y1 = positions[range_]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                linestyle="dashed",
                color="#8e44ad",
                connectionstyle="arc3,rad=0.25",
                shrinkA=22,
                shrinkB=22,
            ),
        )
        ax.text(
            (x0 + x1) / 2.0,
            (y0 + y1) / 2.0 + 0.18,
            name,
            ha="center",
            va="center",
            fontsize=6,
            color="#8e44ad",
            style="italic",
        )


# #############################################################################
# Cell 1.1: The university ontology
# #############################################################################

# Object properties of the university ontology, as `(domain, name, range)`.
UNIVERSITY_PROPERTIES = (
    ("Student", "takesCourse", "Course"),
    ("Professor", "teachesCourse", "Course"),
    ("Person", "belongsToDepartment", "Department"),
)


def build_university_ontology(
    *,
    cardinality_axiom: bool = True,
) -> Tuple[owlready2.World, owlready2.Ontology]:
    """
    Build the university ontology of the lecture: classes, properties, axiom.

    The one cardinality axiom says that every `Course` is taught by exactly
    one `Professor`, which is the axiom a plain database schema can enforce
    but cannot reason with.

    :param cardinality_axiom: add the "exactly one professor" axiom
    :return: the world holding the ontology, and the ontology itself
    """
    world, onto = new_ontology("university")
    with onto:

        class Person(owlready2.Thing):
            pass

        class Course(owlready2.Thing):
            pass

        class Department(owlready2.Thing):
            pass

        class Student(Person):
            pass

        class Professor(Person):
            pass

        class teachesCourse(owlready2.ObjectProperty):
            domain = [Person]
            range = [Course]

        class isTaughtBy(owlready2.ObjectProperty):
            domain = [Course]
            range = [Person]
            inverse_property = teachesCourse

        class takesCourse(owlready2.ObjectProperty):
            domain = [Student]
            range = [Course]

        class belongsToDepartment(owlready2.ObjectProperty):
            domain = [Person]
            range = [Department]

        # A person is either a student or a professor, never both, and the
        # three top classes describe different kinds of thing.
        owlready2.AllDisjoint([Student, Professor])
        owlready2.AllDisjoint([Person, Course, Department])
        if cardinality_axiom:
            course: Any = Course
            course.is_a.append(isTaughtBy.exactly(1, Professor))
    return world, onto


# What each way of modelling a domain offers, used by the comparison table.
_MODELLING_COMPARISON = (
    ("class hierarchy", "flat tables", "yes", "yes", "yes"),
    ("named relations", "foreign keys", "no", "yes", "yes"),
    ("logical axioms", "no", "no", "yes", "yes"),
    ("derives new facts", "no", "no", "yes", "yes"),
    ("holds instances", "yes", "no", "no", "yes"),
    ("open world", "no", "n/a", "yes", "yes"),
)


def cell1_1_university_ontology(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show the university ontology next to the other ways of modelling a domain.

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    _, onto = build_university_ontology()
    hierarchy = extract_hierarchy(onto)
    n_classes = len(hierarchy.classes)
    n_properties = len(list(onto.object_properties()))
    n_axioms = count_axioms(onto)
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    # Panel 1: the classes, with the object properties drawn on top of them.
    graph = build_hierarchy_graph(hierarchy.edges, hierarchy.classes)
    positions = hierarchy_positions(graph)
    colors = {name: _COLOR_ASSERTED for name in hierarchy.classes}
    draw_hierarchy(
        ax1,
        graph,
        positions,
        colors,
        title="Class hierarchy and properties",
        font_size=8,
    )
    draw_property_edges(ax1, positions, UNIVERSITY_PROPERTIES)
    # Panel 2: what each way of modelling the same domain can do.
    draw_table(
        ax2,
        [list(row) for row in _MODELLING_COMPARISON],
        ["property", "db schema", "taxonomy", "ontology", "knowledge base"],
        title="Ontology vs schema vs taxonomy vs KB",
        col_widths=[0.26, 0.19, 0.17, 0.17, 0.21],
        font_size=8,
    )
    # Panel 3: comments on the ontology just built.
    course: Any = onto.Course
    cardinality = [
        render_concept(c)
        for c in course.is_a
        if isinstance(c, owlready2.Restriction)
    ]
    text = (
        "Ontology: university\n\n"
        "Size:\n"
        "  classes: %d\n"
        "  object properties: %d\n"
        "  class axioms: %d\n\n"
        "Classes:\n"
        "  %s\n\n"
        "Properties:\n"
        "  %s\n\n"
        "Cardinality axiom:\n"
        "  Course SubClassOf\n    %s"
        % (
            n_classes,
            n_properties,
            n_axioms,
            "\n  ".join(hierarchy.classes),
            "\n  ".join(name for _, name, _ in UNIVERSITY_PROPERTIES),
            cardinality[0],
        )
    )
    comment_panel(ax3, text)
    plt.tight_layout()
    plt.show()


# #############################################################################
# Cell 2.1: Subsumption and an inconsistent vegetarian pizza
# #############################################################################

# Axioms of the pizza ontology that the cells switch on and off, keyed by a
# short id and rendered in ASCII Manchester syntax. Every other axiom of the
# ontology is always present.
PIZZA_AXIOMS: Dict[str, str] = {
    "disjoint_toppings": (
        "disjoint(CheeseTopping, MeatTopping, VegetableTopping)"
    ),
    "vegetarian_definition": (
        "VegetarianPizza EquivalentTo Pizza and "
        "(hasTopping only (CheeseTopping or VegetableTopping))"
    ),
    "american_has_meat": (
        "AmericanPizza SubClassOf hasTopping some PeperoniSausageTopping"
    ),
    "veg_american_is_american": (
        "VegetarianAmericanPizza SubClassOf AmericanPizza"
    ),
    "veg_american_is_vegetarian": (
        "VegetarianAmericanPizza SubClassOf VegetarianPizza"
    ),
    "closure_margherita": (
        "Margherita SubClassOf hasTopping only "
        "(MozzarellaTopping or TomatoTopping)"
    ),
    "closure_mushroom": (
        "MushroomPizza SubClassOf hasTopping only "
        "(MozzarellaTopping or TomatoTopping or MushroomTopping)"
    ),
    "interesting_pizza": (
        "InterestingPizza EquivalentTo Pizza and (hasTopping min 3 PizzaTopping)"
    ),
}

# The pizza ontology as it is authored, with every optional axiom present.
DEFAULT_PIZZA_AXIOMS: Tuple[str, ...] = tuple(PIZZA_AXIOMS)

# Classes drawn in the hierarchy panels. The topping classes are left out of
# the diagrams to keep them readable; they still take part in the reasoning.
PIZZA_GRAPH_CLASSES = (
    "Thing",
    "Pizza",
    "NamedPizza",
    "Margherita",
    "MushroomPizza",
    "AmericanPizza",
    "VegetarianAmericanPizza",
    "VegetarianPizza",
    "InterestingPizza",
)

# Classes offered for testing in the subsumption cell.
PIZZA_TEST_CLASSES = (
    "Margherita",
    "MushroomPizza",
    "AmericanPizza",
    "VegetarianAmericanPizza",
    "VegetarianPizza",
    "InterestingPizza",
)


def build_pizza_ontology(
    axioms: Sequence[str] = DEFAULT_PIZZA_AXIOMS,
) -> Tuple[owlready2.World, owlready2.Ontology]:
    """
    Build a subset of the Manchester Pizza ontology.

    The subset keeps what the classic Protege tutorial uses to teach
    classification:
    - Three disjoint topping categories, with two named toppings each.
    - Named pizzas listing the toppings they have, plus a closure axiom
      saying they have no other topping.
    - `VegetarianPizza` defined by a universal restriction, so that
      membership in it is something the reasoner derives.
    - `VegetarianAmericanPizza`, deliberately asserted to be both an
      `AmericanPizza` and a `VegetarianPizza`.

    :param axioms: keys from `PIZZA_AXIOMS` to include
    :return: the world holding the ontology, and the ontology itself
    """
    world, onto = new_ontology("pizza")
    with onto:

        class Pizza(owlready2.Thing):
            pass

        class PizzaTopping(owlready2.Thing):
            pass

        class hasTopping(owlready2.ObjectProperty):
            domain = [Pizza]
            range = [PizzaTopping]

        # The three topping categories, each with its named toppings.
        cheese = types.new_class("CheeseTopping", (PizzaTopping,))
        meat = types.new_class("MeatTopping", (PizzaTopping,))
        vegetable = types.new_class("VegetableTopping", (PizzaTopping,))
        mozzarella = types.new_class("MozzarellaTopping", (cheese,))
        peperoni = types.new_class("PeperoniSausageTopping", (meat,))
        types.new_class("HamTopping", (meat,))
        tomato = types.new_class("TomatoTopping", (vegetable,))
        mushroom = types.new_class("MushroomTopping", (vegetable,))
        # The named toppings are always pairwise distinct. Without this the
        # reasoner could not count three different toppings on a pizza.
        owlready2.AllDisjoint([mozzarella, peperoni, tomato, mushroom])
        if "disjoint_toppings" in axioms:
            owlready2.AllDisjoint([cheese, meat, vegetable])
        # The named pizzas, each listing the toppings it has.
        named_pizza = types.new_class("NamedPizza", (Pizza,))
        margherita = types.new_class("Margherita", (named_pizza,))
        margherita.is_a.append(hasTopping.some(mozzarella))
        margherita.is_a.append(hasTopping.some(tomato))
        if "closure_margherita" in axioms:
            # The closure axiom is what turns "has these toppings" into "has
            # only these toppings", which is what the reasoner needs to prove
            # a pizza vegetarian.
            margherita.is_a.append(hasTopping.only(mozzarella | tomato))
        mushroom_pizza = types.new_class("MushroomPizza", (named_pizza,))
        mushroom_pizza.is_a.append(hasTopping.some(mozzarella))
        mushroom_pizza.is_a.append(hasTopping.some(tomato))
        mushroom_pizza.is_a.append(hasTopping.some(mushroom))
        if "closure_mushroom" in axioms:
            mushroom_pizza.is_a.append(
                hasTopping.only(mozzarella | tomato | mushroom)
            )
        american = types.new_class("AmericanPizza", (named_pizza,))
        american.is_a.append(hasTopping.some(mozzarella))
        american.is_a.append(hasTopping.some(tomato))
        if "american_has_meat" in axioms:
            american.is_a.append(hasTopping.some(peperoni))
        american.is_a.append(hasTopping.only(mozzarella | tomato | peperoni))
        # Defined classes, whose members the reasoner works out.
        vegetarian = types.new_class("VegetarianPizza", (owlready2.Thing,))
        if "vegetarian_definition" in axioms:
            vegetarian.equivalent_to = [
                Pizza & hasTopping.only(cheese | vegetable)
            ]
        interesting = types.new_class("InterestingPizza", (owlready2.Thing,))
        if "interesting_pizza" in axioms:
            interesting.equivalent_to = [Pizza & hasTopping.min(3, PizzaTopping)]
        # The deliberately broken pizza: an American pizza asserted to be
        # vegetarian as well.
        veg_american = types.new_class(
            "VegetarianAmericanPizza", (owlready2.Thing,)
        )
        if "veg_american_is_american" in axioms:
            veg_american.is_a.append(american)
        if "veg_american_is_vegetarian" in axioms:
            veg_american.is_a.append(vegetarian)
    return world, onto


def show_pizza_axioms() -> None:
    """
    Display the switchable axioms of the pizza ontology as a table.
    """
    axioms_df = pd.DataFrame(
        {
            "key": list(PIZZA_AXIOMS),
            "axiom": [PIZZA_AXIOMS[key] for key in PIZZA_AXIOMS],
        }
    )
    # The axioms are longer than the default column width.
    with pd.option_context("display.max_colwidth", 120):
        display(axioms_df)


# Classification of the pizza ontology, keyed by the set of axioms used. The
# reasoner is slow enough that re-running it on every widget change would be
# visible, and its answer only depends on the axioms.
_PIZZA_CACHE: Dict[Tuple[str, ...], Classification] = {}


def classify_pizza(axioms: Sequence[str]) -> Classification:
    """
    Classify the pizza ontology built from a set of axioms, with caching.

    :param axioms: keys from `PIZZA_AXIOMS` to include
    :return: asserted hierarchy, inferred hierarchy, and unsatisfiable classes
    """
    key = tuple(sorted(axioms))
    if key not in _PIZZA_CACHE:
        _, onto = build_pizza_ontology(axioms)
        _PIZZA_CACHE[key] = classify(onto)
    return _PIZZA_CACHE[key]


def pizza_axiom_text(keys: Sequence[str]) -> List[str]:
    """
    Render a set of pizza axiom keys as readable axioms.

    :param keys: keys from `PIZZA_AXIOMS`
    :return: one ASCII Manchester axiom per key
    """
    return [PIZZA_AXIOMS[key] for key in keys]


def _pizza_subgraph(
    classification: Classification,
) -> Tuple[nx.DiGraph, Dict[str, Tuple[float, float]]]:
    """
    Build the drawable part of the inferred pizza hierarchy.

    :param classification: result of classifying the pizza ontology
    :return: hierarchy graph over `PIZZA_GRAPH_CLASSES`, and its layout
    """
    edges = [
        edge
        for edge in classification.inferred.edges
        if edge[0] in PIZZA_GRAPH_CLASSES and edge[1] in PIZZA_GRAPH_CLASSES
    ]
    graph = build_hierarchy_graph(edges, PIZZA_GRAPH_CLASSES)
    return graph, hierarchy_positions(graph)


def cell2_1_subsumption_and_inconsistency(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Test one pizza class at a time and read the reasoner's verdict.

    Interactive controls (ipywidgets):
    - `pizza_class`: the class to test, including the deliberately broken one

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    class_dropdown = ipywidgets.Dropdown(
        options=list(PIZZA_TEST_CLASSES),
        value="Margherita",
        description="pizza_class:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()
    # The ontology is the same for every choice, so classify it once.
    classification = classify_pizza(DEFAULT_PIZZA_AXIOMS)

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the hierarchy and the explanation for the selected class.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            tested = class_dropdown.value
            is_unsatisfiable = tested in classification.unsatisfiable
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: the classified hierarchy, tested class highlighted.
            graph, positions = _pizza_subgraph(classification)
            colors = {name: _COLOR_PLAIN for name in graph.nodes}
            for name in classification.unsatisfiable:
                if name in colors:
                    colors[name] = _COLOR_UNSATISFIABLE
            colors[tested] = (
                _COLOR_UNSATISFIABLE if is_unsatisfiable else _COLOR_SELECTED
            )
            new_edges = {
                edge
                for edge in classification.new_edges()
                if edge in set(graph.edges)
            }
            draw_hierarchy(
                ax1,
                graph,
                positions,
                colors,
                title="Classified hierarchy",
                edge_colors={edge: _EDGE_INFERRED for edge in new_edges},
                crossed_out=classification.unsatisfiable,
            )
            # Panel 2: why the reasoner reached its verdict.
            if is_unsatisfiable:
                justification = find_justification(
                    classify_pizza, DEFAULT_PIZZA_AXIOMS, tested
                )
                rows = [[axiom] for axiom in pizza_axiom_text(justification)]
                draw_table(
                    ax2,
                    [[textwrap.fill(row[0], width=44)] for row in rows],
                    ["axioms that cannot all hold at once"],
                    title="Explanation: %s is unsatisfiable" % tested,
                    row_colors=[_COLOR_UNSATISFIABLE] * len(rows),
                    font_size=7,
                )
            else:
                # A satisfiable class is explained by what subsumes it.
                asserted = {
                    parent
                    for parent, child in classification.asserted.edges
                    if child == tested
                }
                inferred = {
                    parent
                    for parent, child in classification.inferred.edges
                    if child == tested
                }
                rows = []
                colors_rows = []
                for parent in sorted(inferred):
                    is_new = parent not in asserted
                    rows.append([parent, "inferred" if is_new else "asserted"])
                    colors_rows.append(
                        _COLOR_INFERRED if is_new else _COLOR_ASSERTED
                    )
                draw_table(
                    ax2,
                    rows,
                    ["superclass of %s" % tested, "source"],
                    title="Explanation: %s is satisfiable" % tested,
                    row_colors=colors_rows,
                    col_widths=[0.6, 0.4],
                )
            # Panel 3: comments on the class under test.
            verdict = "unsatisfiable" if is_unsatisfiable else "satisfiable"
            text = (
                "Parameters:\n"
                "  pizza_class: %s\n\n"
                "Reasoner verdict:\n"
                "  %s: %s\n"
                "  unsatisfiable classes: %s\n"
                "  reasoning time: %.2f s\n\n"
                "Ontology:\n"
                "  classes: %d\n"
                "  asserted edges: %d\n"
                "  inferred edges: %d"
                % (
                    tested,
                    tested,
                    verdict,
                    ", ".join(classification.unsatisfiable) or "none",
                    classification.seconds,
                    len(classification.inferred.classes),
                    len(classification.asserted.edges),
                    len(classification.new_edges()),
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "pizza_class": "class handed to the reasoner; "
            "<code>VegetarianAmericanPizza</code> is the one asserted to be "
            "both an American and a vegetarian pizza",
        }
    )
    class_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [class_dropdown],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.2: Asserted hierarchy vs inferred hierarchy
# #############################################################################

# The three ways of looking at the hierarchy offered by the `view` control.
_HIERARCHY_VIEWS = ("asserted only", "inferred only", "both")


def cell2_2_asserted_vs_inferred(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Compare the pizza hierarchy as authored against the classified one.

    Interactive controls (ipywidgets):
    - `view`: show the asserted edges, the newly inferred ones, or both

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    view_toggle = ipywidgets.ToggleButtons(
        options=list(_HIERARCHY_VIEWS),
        value="both",
        description="view:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()
    classification = classify_pizza(DEFAULT_PIZZA_AXIOMS)
    # The diff of the two graphs is what the reasoner added. `nx.difference()`
    # needs both graphs to have the same nodes, which they do here.
    asserted_graph = build_hierarchy_graph(
        classification.asserted.edges, classification.asserted.classes
    )
    inferred_graph = build_hierarchy_graph(
        classification.inferred.edges, classification.asserted.classes
    )
    diff_graph = nx.difference(inferred_graph, asserted_graph)

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the hierarchy for the selected view.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            view = view_toggle.value
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: the drawable classes, edges colored by their source.
            graph, positions = _pizza_subgraph(classification)
            asserted_edges = set(classification.asserted.edges)
            drawn = nx.DiGraph()
            drawn.add_nodes_from(graph.nodes)
            edge_colors = {}
            for edge in graph.edges:
                is_asserted = edge in asserted_edges
                if view == "asserted only" and not is_asserted:
                    continue
                if view == "inferred only" and is_asserted:
                    continue
                drawn.add_edge(*edge)
                edge_colors[edge] = (
                    _EDGE_ASSERTED if is_asserted else _EDGE_INFERRED
                )
            colors = {name: _COLOR_PLAIN for name in graph.nodes}
            for name in classification.unsatisfiable:
                if name in colors:
                    colors[name] = _COLOR_UNSATISFIABLE
            draw_hierarchy(
                ax1,
                drawn,
                positions,
                colors,
                title="Hierarchy (%s)" % view,
                edge_colors=edge_colors,
                crossed_out=classification.unsatisfiable,
            )
            # Panel 2: every edge the reasoner added, over all the classes.
            rows = [
                [child, parent]
                for parent, child in sorted(
                    classification.new_edges(), key=lambda e: (e[1], e[0])
                )
            ]
            draw_table(
                ax2,
                rows,
                ["class", "newly inferred superclass"],
                title="What classification added",
                row_colors=[_COLOR_INFERRED] * len(rows),
                col_widths=[0.5, 0.5],
            )
            # Panel 3: comments on the size of the gap.
            n_dropped = len(nx.difference(asserted_graph, inferred_graph).edges)
            text = (
                "Parameters:\n"
                "  view: %s\n\n"
                "Edge counts (all classes):\n"
                "  asserted: %d\n"
                "  after classification: %d\n"
                "  newly inferred: %d\n"
                "  dropped as redundant: %d\n\n"
                "Diagram shows the pizza-side classes\n"
                "only, so it holds fewer edges than\n"
                "the counts above.\n\n"
                "Reasoning time: %.2f s"
                % (
                    view,
                    len(classification.asserted.edges),
                    len(classification.inferred.edges),
                    diff_graph.number_of_edges(),
                    n_dropped,
                    classification.seconds,
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "view": "<code>asserted only</code> is the hierarchy as written, "
            "<code>inferred only</code> is what classification added, "
            "<code>both</code> overlays them in two colors",
        }
    )
    view_toggle.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [view_toggle],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.3: Interactive axiom editor
# #############################################################################

# Axioms the editor can add and remove. The rest of the pizza ontology stays
# fixed, so that each edit changes exactly one thing.
EDITABLE_AXIOMS = (
    "disjoint_toppings",
    "closure_margherita",
    "interesting_pizza",
)


def cell2_3_axiom_editor(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Add or remove one axiom at a time and re-run the reasoner after each edit.

    Interactive controls (ipywidgets):
    - `axiom_template`: the axiom to add or remove
    - `apply`: apply the selected axiom and classify again

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    axiom_dropdown = ipywidgets.Dropdown(
        options=list(EDITABLE_AXIOMS),
        value=EDITABLE_AXIOMS[0],
        description="axiom_template:",
        style={"description_width": "initial"},
    )
    apply_button = ipywidgets.Button(description="apply", button_style="primary")
    output = ipywidgets.Output()
    # Mutable editor state: the axioms currently in the ontology, and what the
    # last edit was.
    state: Dict[str, Any] = {
        "axioms": set(DEFAULT_PIZZA_AXIOMS),
        "last_edit": "none yet",
        "edges_before": len(classify_pizza(DEFAULT_PIZZA_AXIOMS).inferred.edges),
    }

    def redraw() -> None:
        """
        Classify the edited ontology and redraw the hierarchy.
        """
        with output:
            clear_output(wait=True)
            axioms = tuple(sorted(state["axioms"]))
            classification = classify_pizza(axioms)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: the hierarchy after the edit.
            graph, positions = _pizza_subgraph(classification)
            colors = {name: _COLOR_PLAIN for name in graph.nodes}
            for name in classification.unsatisfiable:
                if name in colors:
                    colors[name] = _COLOR_UNSATISFIABLE
            asserted_edges = set(classification.asserted.edges)
            edge_colors = {
                edge: _EDGE_INFERRED
                for edge in graph.edges
                if edge not in asserted_edges
            }
            draw_hierarchy(
                ax1,
                graph,
                positions,
                colors,
                title="Hierarchy after the edit",
                edge_colors=edge_colors,
                crossed_out=classification.unsatisfiable,
            )
            # Panel 2: which editable axioms are in the ontology right now.
            rows = []
            row_colors = []
            for key in EDITABLE_AXIOMS:
                is_on = key in state["axioms"]
                rows.append(
                    [
                        textwrap.fill(PIZZA_AXIOMS[key], width=38),
                        "in" if is_on else "out",
                    ]
                )
                row_colors.append(_COLOR_ASSERTED if is_on else _COLOR_PLAIN)
            draw_table(
                ax2,
                rows,
                ["axiom", "status"],
                title="Editable axioms",
                row_colors=row_colors,
                col_widths=[0.78, 0.22],
                font_size=7,
            )
            # Panel 3: comments on the edit just applied.
            n_edges = len(classification.inferred.edges)
            text = (
                "Parameters:\n"
                "  axiom_template: %s\n\n"
                "Last edit:\n"
                "  %s\n\n"
                "Hierarchy edges:\n"
                "  before the edit: %d\n"
                "  after the edit: %d\n"
                "  newly inferred: %d\n\n"
                "Unsatisfiable classes:\n"
                "  %s\n\n"
                "Reasoning time: %.2f s"
                % (
                    axiom_dropdown.value,
                    wrap_comment(state["last_edit"]),
                    state["edges_before"],
                    n_edges,
                    len(classification.new_edges()),
                    ", ".join(classification.unsatisfiable) or "none",
                    classification.seconds,
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    def on_apply(button: Any) -> None:
        """
        Toggle the selected axiom and classify the edited ontology.

        :param button: button that was clicked (unused)
        """
        _ = button
        key = axiom_dropdown.value
        state["edges_before"] = len(
            classify_pizza(tuple(sorted(state["axioms"]))).inferred.edges
        )
        if key in state["axioms"]:
            state["axioms"].remove(key)
            state["last_edit"] = "removed %s" % PIZZA_AXIOMS[key]
        else:
            state["axioms"].add(key)
            state["last_edit"] = "added %s" % PIZZA_AXIOMS[key]
        redraw()

    param_info = make_param_info(
        {
            "axiom_template": "axiom to add if it is out of the ontology, or "
            "to remove if it is in",
            "apply": "apply the edit and run the reasoner again",
        }
    )
    apply_button.on_click(on_apply)
    redraw()
    controls = ipywidgets.VBox(
        [axiom_dropdown, apply_button],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 3.1: Unsatisfiable concepts, the flying penguin
# #############################################################################

# Axioms of the penguin ontology that the cell switches on and off.
PENGUIN_AXIOMS: Dict[str, str] = {
    "penguins_cannot_fly": "Penguin SubClassOf not FlyingThing",
    "flying_penguin_definition": (
        "FlyingPenguin EquivalentTo Penguin and FlyingThing"
    ),
}

# The penguin ontology as it is authored, with every optional axiom present.
DEFAULT_PENGUIN_AXIOMS: Tuple[str, ...] = tuple(PENGUIN_AXIOMS)


def build_penguin_ontology(
    axioms: Sequence[str] = DEFAULT_PENGUIN_AXIOMS,
) -> Tuple[owlready2.World, owlready2.Ontology]:
    """
    Build the bird ontology holding the `FlyingPenguin` concept.

    :param axioms: keys from `PENGUIN_AXIOMS` to include
    :return: the world holding the ontology, and the ontology itself
    """
    world, onto = new_ontology("penguin")
    with onto:

        class Bird(owlready2.Thing):
            pass

        class FlyingThing(owlready2.Thing):
            pass

        penguin: Any = types.new_class("Penguin", (Bird,))
        sparrow: Any = types.new_class("Sparrow", (Bird,))
        # A sparrow is a bird that does fly, which is what makes the penguin
        # the exception rather than the rule.
        sparrow.is_a.append(FlyingThing)
        if "penguins_cannot_fly" in axioms:
            penguin.is_a.append(owlready2.Not(FlyingThing))
        flying_penguin = types.new_class("FlyingPenguin", (owlready2.Thing,))
        if "flying_penguin_definition" in axioms:
            flying_penguin.equivalent_to = [penguin & FlyingThing]
    return world, onto


# Classification of the penguin ontology, keyed by the set of axioms used.
_PENGUIN_CACHE: Dict[Tuple[str, ...], Classification] = {}


def classify_penguin(axioms: Sequence[str]) -> Classification:
    """
    Classify the penguin ontology built from a set of axioms, with caching.

    :param axioms: keys from `PENGUIN_AXIOMS` to include
    :return: asserted hierarchy, inferred hierarchy, and unsatisfiable classes
    """
    key = tuple(sorted(axioms))
    if key not in _PENGUIN_CACHE:
        _, onto = build_penguin_ontology(axioms)
        _PENGUIN_CACHE[key] = classify(onto)
    return _PENGUIN_CACHE[key]


def cell3_1_flying_penguin(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show that `FlyingPenguin` is unsatisfiable, and what makes it so.

    Interactive controls (ipywidgets):
    - `penguins_cannot_fly`: add or remove the axiom that causes the conflict

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    axiom_checkbox = ipywidgets.Checkbox(
        value=True,
        description="penguins_cannot_fly",
        indent=False,
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the hierarchy and the explanation for the current axiom set.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            axioms = ["flying_penguin_definition"]
            if axiom_checkbox.value:
                axioms.append("penguins_cannot_fly")
            classification = classify_penguin(axioms)
            is_unsatisfiable = "FlyingPenguin" in classification.unsatisfiable
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: the classified hierarchy.
            graph = build_hierarchy_graph(
                classification.inferred.edges, classification.inferred.classes
            )
            positions = hierarchy_positions(graph)
            colors = {name: _COLOR_PLAIN for name in graph.nodes}
            colors["FlyingPenguin"] = (
                _COLOR_UNSATISFIABLE if is_unsatisfiable else _COLOR_INFERRED
            )
            draw_hierarchy(
                ax1,
                graph,
                positions,
                colors,
                title="Classified hierarchy",
                crossed_out=classification.unsatisfiable,
                font_size=8,
            )
            # Panel 2: the axioms that cannot hold together.
            if is_unsatisfiable:
                justification = find_justification(
                    classify_penguin, axioms, "FlyingPenguin"
                )
                rows = [
                    [textwrap.fill(PENGUIN_AXIOMS[key], width=40)]
                    for key in justification
                ]
                draw_table(
                    ax2,
                    rows,
                    ["axioms that cannot all hold at once"],
                    title="Explanation: FlyingPenguin is unsatisfiable",
                    row_colors=[_COLOR_UNSATISFIABLE] * len(rows),
                    font_size=8,
                )
            else:
                rows = [
                    [textwrap.fill(PENGUIN_AXIOMS[key], width=40)]
                    for key in axioms
                ]
                draw_table(
                    ax2,
                    rows,
                    ["axioms currently asserted"],
                    title="FlyingPenguin is satisfiable",
                    row_colors=[_COLOR_ASSERTED] * len(rows),
                    font_size=8,
                )
            # Panel 3: comments on the current state.
            text = (
                "Parameters:\n"
                "  penguins_cannot_fly: %s\n\n"
                "Reasoner verdict:\n"
                "  FlyingPenguin satisfiable: %s\n"
                "  unsatisfiable classes: %s\n\n"
                "Axioms asserted: %d\n"
                "Reasoning time: %.2f s"
                % (
                    axiom_checkbox.value,
                    not is_unsatisfiable,
                    ", ".join(classification.unsatisfiable) or "none",
                    len(axioms),
                    classification.seconds,
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "penguins_cannot_fly": "assert <code>Penguin SubClassOf not "
            "FlyingThing</code>; clear it and the conflict disappears",
        }
    )
    axiom_checkbox.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [axiom_checkbox],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 4.1: Realization, retrieval, and the open world assumption
# #############################################################################

# Individuals that can be realized, and what each of them is asserted to be.
UNIVERSITY_INDIVIDUALS = ("gp", "alice", "bob", "carol")

# Classes that can be used as a retrieval query.
UNIVERSITY_QUERY_CLASSES = (
    "TeachingAssistant",
    "Student",
    "Professor",
    "Course",
)

# Prefix of the named complement classes. They exist so that a query can
# answer "provably false", and are plumbing rather than part of the domain,
# so they are hidden from the panels.
_COMPLEMENT_PREFIX = "Not"


@dataclasses.dataclass(frozen=True)
class InstanceFacts:
    """
    What the reasoner knows about the individuals of the university ontology.
    """

    # Classes each individual is asserted to belong to.
    asserted_types: Dict[str, Tuple[str, ...]]
    # Classes each individual belongs to after realization.
    inferred_types: Dict[str, Tuple[str, ...]]
    # The most specific of the inferred classes, per individual.
    most_specific: Dict[str, Tuple[str, ...]]
    # Individuals retrieved for each query class, before and after reasoning.
    asserted_instances: Dict[str, Tuple[str, ...]]
    inferred_instances: Dict[str, Tuple[str, ...]]
    # Three-valued answer per `(individual, class)` question.
    answers: Dict[Tuple[str, str], str]
    # Class hierarchy after classification.
    hierarchy: Hierarchy


def build_populated_university() -> Tuple[owlready2.World, owlready2.Ontology]:
    """
    Extend the university ontology with a defined class and four individuals.

    The individuals are chosen to separate the three answers an open world
    query can give:
    - `alice` teaches, so she is provably a teaching assistant.
    - `bob` has no teaching assertion at all, so nothing is provable about it.
    - `carol` is explicitly said to teach nothing, which closes the world
      around her and makes the negative answer provable.

    :return: the world holding the ontology, and the ontology itself
    """
    world, onto = build_university_ontology()
    with onto:
        student: Any = onto.Student
        professor: Any = onto.Professor
        course: Any = onto.Course
        teaches_course: Any = onto.teachesCourse
        # A teaching assistant is any student who teaches a course. Nobody is
        # asserted to be one: the reasoner works out who qualifies.
        assistant = types.new_class("TeachingAssistant", (owlready2.Thing,))
        assistant.equivalent_to = [student & teaches_course.some(course)]
        # Naming the complement of each query class is what lets a query
        # answer "provably false" instead of only "not provable".
        for cls in (assistant, student, professor, course):
            complement = types.new_class(
                "%s%s" % (_COMPLEMENT_PREFIX, cls.name), (owlready2.Thing,)
            )
            complement.equivalent_to = [owlready2.Not(cls)]
        data605 = course("data605")
        msml610 = course("msml610")
        gp = professor("gp")
        gp.teachesCourse = [data605, msml610]
        alice = student("alice")
        alice.takesCourse = [msml610]
        alice.teachesCourse = [data605]
        bob = student("bob")
        bob.takesCourse = [data605, msml610]
        carol = student("carol")
        carol.takesCourse = [data605]
        carol.is_a.append(owlready2.Not(teaches_course.some(course)))
    return world, onto


# Facts about the populated university ontology, computed once.
_INSTANCE_CACHE: List[InstanceFacts] = []


def realize_university() -> InstanceFacts:
    """
    Realize the populated university ontology and answer the queries on it.

    :return: asserted and inferred types, retrieval results, query answers
    """
    if not _INSTANCE_CACHE:
        _, onto = build_populated_university()
        asserted_types = {}
        asserted_instances = {}
        for name in UNIVERSITY_INDIVIDUALS:
            individual: Any = onto[name]
            asserted_types[name] = tuple(
                sorted(
                    cls.name
                    for cls in individual.is_a
                    if isinstance(cls, owlready2.ThingClass)
                )
            )
        for query in UNIVERSITY_QUERY_CLASSES:
            asserted_instances[query] = tuple(
                sorted(ind.name for ind in onto.search(type=onto[query]))
            )
        run_reasoner(onto.world)
        inferred_types = {}
        most_specific = {}
        answers = {}
        for name in UNIVERSITY_INDIVIDUALS:
            individual = onto[name]
            classes = {
                cls
                for cls in individual.INDIRECT_is_a
                if isinstance(cls, owlready2.ThingClass)
            }
            inferred_types[name] = tuple(sorted(cls.name for cls in classes))
            # A class is most specific for this individual when no other
            # class it belongs to sits below it.
            specific = [
                cls.name
                for cls in classes
                if not cls.name.startswith(_COMPLEMENT_PREFIX)
                and not any(
                    other is not cls and cls in other.ancestors()
                    for other in classes
                )
            ]
            most_specific[name] = tuple(sorted(specific))
            for query in UNIVERSITY_QUERY_CLASSES:
                names = inferred_types[name]
                if query in names:
                    answer = "true"
                elif "%s%s" % (_COMPLEMENT_PREFIX, query) in names:
                    answer = "false"
                else:
                    answer = "unknown"
                answers[(name, query)] = answer
        inferred_instances = {}
        for query in UNIVERSITY_QUERY_CLASSES:
            inferred_instances[query] = tuple(
                sorted(ind.name for ind in onto.search(type=onto[query]))
            )
        _INSTANCE_CACHE.append(
            InstanceFacts(
                asserted_types,
                inferred_types,
                most_specific,
                asserted_instances,
                inferred_instances,
                answers,
                extract_hierarchy(onto),
            )
        )
    return _INSTANCE_CACHE[0]


# Classes drawn in the realization panel: the complement classes are left out
# to keep the diagram readable.
_REALIZATION_GRAPH_CLASSES = (
    "Thing",
    "Person",
    "Student",
    "Professor",
    "TeachingAssistant",
    "Course",
    "Department",
)


def cell4_1_realization_and_owa(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Realize one individual, retrieve the members of a class, and ask an
    unasserted question under the open world assumption.

    Interactive controls (ipywidgets):
    - `individual`: individual to realize
    - `query_class`: class whose members are retrieved
    - `query_absent_fact`: ask whether the individual is in the query class

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE_WIDE
    individual_dropdown = ipywidgets.Dropdown(
        options=list(UNIVERSITY_INDIVIDUALS),
        value="alice",
        description="individual:",
        style={"description_width": "initial"},
    )
    query_dropdown = ipywidgets.Dropdown(
        options=list(UNIVERSITY_QUERY_CLASSES),
        value="TeachingAssistant",
        description="query_class:",
        style={"description_width": "initial"},
    )
    ask_button = ipywidgets.Button(
        description="query_absent_fact", button_style="primary"
    )
    output = ipywidgets.Output()
    facts = realize_university()
    # Whether the question has been asked yet, so that the answer appears on
    # the button press rather than before it.
    state: Dict[str, bool] = {"asked": False}

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the realization, retrieval, and open world panels.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            individual = individual_dropdown.value
            query = query_dropdown.value
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: where the individual lands in the hierarchy.
            edges = [
                edge
                for edge in facts.hierarchy.edges
                if edge[0] in _REALIZATION_GRAPH_CLASSES
                and edge[1] in _REALIZATION_GRAPH_CLASSES
            ]
            graph = build_hierarchy_graph(edges, _REALIZATION_GRAPH_CLASSES)
            graph = attach_orphans_to_root(graph, "Thing")
            positions = hierarchy_positions(graph)
            colors = {}
            for name in graph.nodes:
                if name in facts.most_specific[individual]:
                    colors[name] = _COLOR_INDIVIDUAL
                elif name in facts.inferred_types[individual]:
                    colors[name] = _COLOR_INFERRED
                elif name in facts.asserted_types[individual]:
                    colors[name] = _COLOR_ASSERTED
                else:
                    colors[name] = _COLOR_PLAIN
            draw_hierarchy(
                ax1,
                graph,
                positions,
                colors,
                title="Realization of %s" % individual,
                font_size=8,
            )
            # Panel 2: the individuals the query class retrieves.
            rows = []
            row_colors = []
            for name in UNIVERSITY_INDIVIDUALS:
                in_asserted = name in facts.asserted_instances[query]
                in_inferred = name in facts.inferred_instances[query]
                rows.append(
                    [
                        name,
                        "yes" if in_asserted else "no",
                        "yes" if in_inferred else "no",
                    ]
                )
                if in_inferred and not in_asserted:
                    row_colors.append(_COLOR_INFERRED)
                elif in_inferred:
                    row_colors.append(_COLOR_ASSERTED)
                else:
                    row_colors.append(_COLOR_PLAIN)
            draw_table(
                ax2,
                rows,
                ["individual", "asserted", "retrieved"],
                title="Retrieval: %s" % query,
                row_colors=row_colors,
                col_widths=[0.4, 0.3, 0.3],
            )
            # Panel 3: the same question under the two assumptions.
            answer = facts.answers[(individual, query)]
            asserted = query in facts.asserted_types[individual]
            if state["asked"]:
                cwa_answer = "true" if answer == "true" else "false"
                owa_answer = answer
            else:
                cwa_answer = "not asked yet"
                owa_answer = "not asked yet"
            draw_table(
                ax3,
                [
                    ["asserted in the KB", "yes" if asserted else "no"],
                    ["closed world answer", cwa_answer],
                    ["open world answer", owa_answer],
                ],
                ["question: %s is a %s" % (individual, query), "answer"],
                title="Open world query",
                row_colors=[
                    _COLOR_ASSERTED,
                    _COLOR_PLAIN,
                    _COLOR_INFERRED if state["asked"] else _COLOR_PLAIN,
                ],
                col_widths=[0.55, 0.45],
                font_size=8,
            )
            # Panel 4: comments on the current individual and query.
            text = (
                "Parameters:\n"
                "  individual: %s\n"
                "  query_class: %s\n\n"
                "Realization of %s:\n"
                "  asserted types: %s\n"
                "  inferred types: %s\n"
                "  most specific: %s\n\n"
                "Retrieval of %s:\n"
                "  asserted: %s\n"
                "  retrieved: %s\n\n"
                "Query answer: %s"
                % (
                    individual,
                    query,
                    individual,
                    ", ".join(facts.asserted_types[individual]),
                    ", ".join(
                        name
                        for name in facts.inferred_types[individual]
                        if not name.startswith(_COMPLEMENT_PREFIX)
                    ),
                    ", ".join(facts.most_specific[individual]),
                    query,
                    ", ".join(facts.asserted_instances[query]) or "none",
                    ", ".join(facts.inferred_instances[query]) or "none",
                    owa_answer,
                )
            )
            comment_panel(ax4, text)
            plt.tight_layout()
            plt.show()

    def on_ask(button: Any) -> None:
        """
        Ask the question about the selected individual and class.

        :param button: button that was clicked (unused)
        """
        _ = button
        state["asked"] = True
        update_plot()

    def on_change(change: Optional[Any] = None) -> None:
        """
        Reset the query when the individual or the class changes.

        :param change: widget change record (unused)
        """
        state["asked"] = False
        update_plot(change)

    param_info = make_param_info(
        {
            "individual": "individual handed to realization; "
            "<code>bob</code> and <code>carol</code> differ only in whether "
            "the world was closed around them",
            "query_class": "class used for retrieval and for the query",
            "query_absent_fact": "ask whether the individual is in the query "
            "class, which for most pairs was never asserted",
        }
    )
    individual_dropdown.observe(on_change, names="value")
    query_dropdown.observe(on_change, names="value")
    ask_button.on_click(on_ask)
    update_plot()
    controls = ipywidgets.VBox(
        [individual_dropdown, query_dropdown, ask_button],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 5.1: Expressiveness vs tractability
# #############################################################################

# Numbers of cardinality constraints measured for the runtime curve.
SCALING_STEPS = (0, 2, 4, 6, 8, 10)

# Largest cardinality used by the added constraints. The reasoner has to build
# a model with that many distinct individuals per constrained class, which is
# where the cost comes from.
_MAX_CARDINALITY = 8

# Size of the population used to make the property chain do visible work.
_N_UNITS = 8
_N_STAFF = 40


def build_scaling_ontology(
    n_cardinality: int,
    *,
    add_property_chain: bool,
) -> Tuple[owlready2.World, owlready2.Ontology]:
    """
    Build a university ontology with extra expressive constructs.

    Two knobs make the ontology more expressive:
    - `n_cardinality` qualified number restrictions, each forcing the reasoner
      to build a larger model.
    - One property chain `worksIn o partOf -> worksIn`, which propagates the
      staff of a unit up to every unit above it.

    :param n_cardinality: number of cardinality constraints to add
    :param add_property_chain: add the property chain axiom
    :return: the world holding the ontology, and the ontology itself
    """
    # The "exactly one professor" axiom of Cell 1.1 is left out, so that the
    # restrictions added here are the only cardinality constraints and the
    # measurement is about them alone.
    world, onto = build_university_ontology(cardinality_axiom=False)
    with onto:
        course: Any = onto.Course
        professor: Any = onto.Professor
        person: Any = onto.Person
        is_taught_by: Any = onto.isTaughtBy

        class Unit(owlready2.Thing):
            pass

        class partOf(owlready2.ObjectProperty, owlready2.TransitiveProperty):
            domain = [Unit]
            range = [Unit]

        class worksIn(owlready2.ObjectProperty):
            domain = [person]
            range = [Unit]

        for index in range(n_cardinality):
            # Each new course class needs more professors than the previous
            # one, up to a cap that keeps the runtime bounded.
            cardinality = min(index + 2, _MAX_CARDINALITY)
            cls = types.new_class("Course%d" % index, (course,))
            cls.is_a.append(is_taught_by.min(cardinality, professor))
            cls.is_a.append(is_taught_by.max(cardinality + 1, person))
        # A chain of units, with staff attached to the lowest ones.
        units = [Unit("unit%d" % index) for index in range(_N_UNITS)]
        for index in range(_N_UNITS - 1):
            units[index].partOf = [units[index + 1]]
        for index in range(_N_STAFF):
            staff = professor("staff%d" % index)
            staff.worksIn = [units[index % _N_UNITS]]
        # Membership in this class is what the property chain changes.
        works_in_top = types.new_class("WorksInTopUnit", (owlready2.Thing,))
        works_in_top.equivalent_to = [worksIn.value(units[-1])]
        if add_property_chain:
            worksIn.property_chain.append(
                owlready2.PropertyChain([worksIn, partOf])
            )
    return world, onto


@dataclasses.dataclass(frozen=True)
class ScalingMeasurement:
    """
    One point of the runtime curve.
    """

    # Number of cardinality constraints in the ontology.
    n_cardinality: int
    # Whether the property chain axiom is present.
    property_chain: bool
    # Wall-clock time of the reasoner call, in seconds.
    seconds: float
    # Individuals inferred to work in the top unit.
    n_inferred: int


# Runtime measurements, keyed by `(n_cardinality, property_chain)`.
_SCALING_CACHE: Dict[Tuple[int, bool], ScalingMeasurement] = {}

# Whether the reasoner has already run once in this process.
_JVM_WARMED_UP: List[bool] = []


def _warm_up_reasoner() -> None:
    """
    Run the reasoner once on a throwaway ontology.

    HermiT runs in a JVM, and the first call in a process pays for starting
    it. Paying that cost before the measurements keeps it out of the curve.
    """
    if not _JVM_WARMED_UP:
        _, onto = build_university_ontology()
        run_reasoner(onto.world)
        _JVM_WARMED_UP.append(True)


def measure_reasoning_time(
    n_cardinality: int,
    *,
    add_property_chain: bool,
) -> ScalingMeasurement:
    """
    Time the reasoner on one ontology of the scaling family, with caching.

    :param n_cardinality: number of cardinality constraints to add
    :param add_property_chain: add the property chain axiom
    :return: the measurement for this point of the curve
    """
    key = (n_cardinality, add_property_chain)
    if key not in _SCALING_CACHE:
        _warm_up_reasoner()
        _, onto = build_scaling_ontology(
            n_cardinality, add_property_chain=add_property_chain
        )
        seconds = run_reasoner(onto.world)
        # An unsatisfiable class would let the reasoner stop early, which
        # would measure the wrong thing.
        hdbg.dassert_eq(
            [
                cls.name
                for cls in onto.world.inconsistent_classes()
                if cls.name != "Nothing"
            ],
            [],
        )
        n_inferred = len(onto.search(type=onto.WorksInTopUnit))
        _SCALING_CACHE[key] = ScalingMeasurement(
            n_cardinality, add_property_chain, seconds, n_inferred
        )
    return _SCALING_CACHE[key]


def cell5_1_expressiveness_vs_tractability(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Measure how reasoner runtime grows as expressive constructs are added.

    Interactive controls (ipywidgets):
    - `n`: number of cardinality constraints added to the ontology
    - `add_property_chain`: add one property chain axiom

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    n_slider, n_box = htutori.build_widget_control(
        name="n",
        description="n (cardinality constraints)",
        min_val=0,
        max_val=10,
        step=2,
        initial_value=4,
        is_float=False,
    )
    chain_checkbox = ipywidgets.Checkbox(
        value=False,
        description="add_property_chain",
        indent=False,
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the runtime curve for the current controls.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            n_cardinality = n_slider.value
            with_chain = chain_checkbox.value
            # Measuring the whole curve is what makes the trend visible; the
            # results are cached, so only the first pass pays for it.
            curves = {}
            for chain in (False, True):
                curves[chain] = [
                    measure_reasoning_time(n, add_property_chain=chain)
                    for n in SCALING_STEPS
                ]
            current = measure_reasoning_time(
                n_cardinality, add_property_chain=with_chain
            )
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: runtime against the number of constraints.
            for chain, style in ((False, "--"), (True, "-")):
                ax1.plot(
                    [m.n_cardinality for m in curves[chain]],
                    [m.seconds for m in curves[chain]],
                    style,
                    marker="o",
                    linewidth=2.0,
                    color=_EDGE_INFERRED if chain else _EDGE_ASSERTED,
                    label="property chain: %s" % chain,
                )
            ax1.plot(
                current.n_cardinality,
                current.seconds,
                marker="*",
                markersize=18,
                color="#c0392b",
                linestyle="none",
                label="current setting",
            )
            ax1.set_xlabel("n (cardinality constraints)", fontsize=10)
            ax1.set_ylabel("reasoning time [s]", fontsize=10)
            ax1.set_title("Reasoner runtime", fontsize=13, fontweight="bold")
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=8)
            # Panel 2: what the property chain buys, in inferred facts.
            labels = ["chain off", "chain on"]
            values = [
                measure_reasoning_time(
                    n_cardinality, add_property_chain=False
                ).n_inferred,
                measure_reasoning_time(
                    n_cardinality, add_property_chain=True
                ).n_inferred,
            ]
            bars = ax2.bar(
                labels, values, color=[_EDGE_ASSERTED, _EDGE_INFERRED]
            )
            for bar, value in zip(bars, values):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value,
                    str(value),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
            ax2.set_ylabel("individuals in WorksInTopUnit", fontsize=10)
            ax2.set_title(
                "What the property chain infers",
                fontsize=13,
                fontweight="bold",
            )
            ax2.grid(True, alpha=0.3, axis="y")
            # Panel 3: comments on the current measurement.
            baseline = curves[with_chain][0].seconds
            text = (
                "Parameters:\n"
                "  n: %d\n"
                "  add_property_chain: %s\n\n"
                "Current measurement:\n"
                "  construct: %s\n"
                "  reasoning time: %.2f s\n"
                "  time at n=0: %.2f s\n"
                "  slowdown: %.1fx\n\n"
                "Inferred members of WorksInTopUnit:\n"
                "  chain off: %d\n"
                "  chain on: %d"
                % (
                    n_cardinality,
                    with_chain,
                    "qualified number restrictions"
                    + (" plus property chain" if with_chain else ""),
                    current.seconds,
                    baseline,
                    current.seconds / baseline,
                    values[0],
                    values[1],
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "n": "number of qualified number restrictions added to the "
            "ontology, each one asking for more professors than the last",
            "add_property_chain": "add <code>worksIn o partOf -> "
            "worksIn</code>, which propagates staff up the unit chain",
        }
    )
    n_slider.observe(update_plot, names="value")
    chain_checkbox.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [n_box, chain_checkbox],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))
