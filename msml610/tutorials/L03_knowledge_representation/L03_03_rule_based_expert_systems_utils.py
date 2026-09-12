"""
Utility functions for the rule-based expert system lesson.

Implements the match-conflict resolution-act cycle of a MYCIN-style production
system from scratch, using Winston's animal-identification rules:
- A rule base, a working memory, and a reflex agent baseline.
- Forward (data-driven) chaining, one fired rule at a time.
- Backward (goal-driven) chaining with an AND-OR proof tree.
- Conflict resolution by specificity, recency, and priority.
- Explanation traces and MYCIN certainty factors.
- Non-monotonic default reasoning and the closed world assumption.
- A decision tree trained on the same domain, for comparison.
- Interactive notebook cells built on top of these primitives.

The engine is written here instead of being imported from a production-system
package (e.g., `experta`) because those packages are unmaintained and do not
run on modern Python, and because the point of the lesson is to expose the
recognize-act cycle itself.

Import as:

import msml610.tutorials.L03_knowledge_representation.L03_03_rule_based_expert_systems_utils as mtlkrl0rbesu
"""

import dataclasses
import itertools
import logging
import textwrap
import time
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import ipywidgets
import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sklearn.metrics as skmetri
import sklearn.model_selection as skmodsel
import sklearn.tree as sktree
from IPython.display import clear_output, display

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.htutorial as htutori

_LOG = logging.getLogger(__name__)

# #############################################################################
# Colors and display constants
# #############################################################################

# Colors marking where a fact came from.
_COLOR_ASSERTED = "#a9cce3"
_COLOR_DERIVED = "#a8e6a3"
_COLOR_MISSING = "#e8e8e8"
_COLOR_RETRACTED = "#f4a6a6"
# Colors marking the role a node plays in a graph or a proof tree.
_COLOR_RULE = "#ffe680"
_COLOR_RULE_FIRED = "#f5b041"
_COLOR_GOAL = "#d8b4e2"
_COLOR_FAILED = "#f4a6a6"
# Colors used to separate the three conflict resolution strategies.
_STRATEGY_COLORS = {
    "specificity": "#5dade2",
    "recency": "#58d68d",
    "priority": "#f5b041",
}
# Default figure size for a 1x3 panel row, wide enough for the graphs.
_FIGSIZE = (17, 5)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger so notebook cells see the
    debug and info output of these utility functions.

    :param notebook_log: logger owned by the notebook
    """
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Rules and working memory
# #############################################################################


@dataclasses.dataclass(frozen=True)
class Rule:
    """
    One production rule: if every premise holds, assert the conclusion.

    A premise prefixed with `~` is a negative premise, which holds when the
    fact is absent from working memory. This is negation as failure, so it
    only makes sense under the closed world assumption (see
    `cell4_2_cwa_vs_owa()`).
    """

    # Short label, e.g., `R9`, used in traces and graphs.
    name: str
    # Premises, e.g., `("carnivore", "tawny", "~leopard")`.
    premises: Tuple[str, ...]
    # Fact asserted when the rule fires, e.g., `cheetah`.
    conclusion: str
    # Explicit priority, used by the `priority` conflict resolution strategy.
    salience: int = 0
    # Certainty factor of the rule itself, in $[0, 1]$.
    cf: float = 1.0

    def positive_premises(self) -> Tuple[str, ...]:
        """
        Return the premises that must be present in working memory.

        :return: premise facts, e.g., `("carnivore", "tawny")`
        """
        return tuple(p for p in self.premises if not p.startswith("~"))

    def negative_premises(self) -> Tuple[str, ...]:
        """
        Return the premises that must be absent from working memory.

        :return: premise facts, e.g., `("leopard",)` for `~leopard`
        """
        return tuple(p[1:] for p in self.premises if p.startswith("~"))

    def as_text(self) -> str:
        """
        Render the rule the way a domain expert would read it.

        :return: rule text, e.g., `R9: carnivore AND tawny -> cheetah`
        """
        premises = " AND ".join(self.premises)
        return "%s: %s -> %s" % (self.name, premises, self.conclusion)


@dataclasses.dataclass(frozen=True)
class Fact:
    """
    One fact in working memory, with the provenance needed to explain it.
    """

    # Fact name, e.g., `carnivore`.
    name: str
    # Logical time at which the fact entered working memory, used by recency.
    step: int
    # Rule that derived the fact, empty when the fact was asserted by a user.
    rule: str
    # Propagated certainty factor of the fact, in $[0, 1]$.
    cf: float


@dataclasses.dataclass(frozen=True)
class Activation:
    """
    One turn of the recognize-act cycle: what matched, what was picked, what
    was added.
    """

    # 1-based index of the cycle that produced this activation.
    step: int
    # Rule selected by the conflict resolution strategy.
    rule: Rule
    # Names of every rule that matched on this cycle, the conflict set.
    conflict_set: Tuple[str, ...]
    # Fact added to working memory by firing the rule.
    conclusion: str
    # Certainty factor propagated to the conclusion.
    cf: float


def _parse_premise(premise: str) -> Tuple[str, bool]:
    """
    Split a premise into its fact and its sign.

    :param premise: premise text, e.g., `carnivore` or `~leopard`
    :return: fact and whether the premise is positive, e.g.,
        `("leopard", False)`
    """
    is_positive = not premise.startswith("~")
    fact = premise if is_positive else premise[1:]
    return fact, is_positive


# #############################################################################
# Winston's animal identification rule base
# #############################################################################

# Winston's classic animal identification rules, in the usual textbook order.
# The rules chain in three layers: observable features produce a class
# (`mammal`, `bird`), the class plus more features produces a sub-class
# (`carnivore`, `ungulate`), and that produces a species.
ANIMAL_RULES: Tuple[Rule, ...] = (
    Rule("R1", ("has_hair",), "mammal"),
    Rule("R2", ("gives_milk",), "mammal"),
    Rule("R3", ("has_feathers",), "bird"),
    Rule("R4", ("flies", "lays_eggs"), "bird"),
    Rule("R5", ("mammal", "eats_meat"), "carnivore"),
    Rule(
        "R6", ("mammal", "pointed_teeth", "claws", "forward_eyes"), "carnivore"
    ),
    Rule("R7", ("mammal", "hoofs"), "ungulate"),
    Rule("R8", ("mammal", "chews_cud"), "ungulate"),
    Rule("R9", ("carnivore", "tawny", "dark_spots"), "cheetah"),
    Rule("R10", ("carnivore", "tawny", "black_stripes"), "tiger"),
    Rule(
        "R11",
        ("ungulate", "long_legs", "long_neck", "tawny", "dark_spots"),
        "giraffe",
    ),
    Rule("R12", ("ungulate", "white", "black_stripes"), "zebra"),
    Rule("R13", ("bird", "does_not_fly", "swims", "black_and_white"), "penguin"),
    Rule("R14", ("bird", "good_flier"), "albatross"),
)

# Facts the rules can conclude, as opposed to facts an agent observes.
_DERIVED_FACTS: Tuple[str, ...] = tuple(
    dict.fromkeys(rule.conclusion for rule in ANIMAL_RULES)
)

# Species any rule base in this lesson can identify. `leopard` is not one of
# Winston's rules: it is added in `cell2_3_conflict_resolution()` to create a
# deliberate conflict with the cheetah rule.
SPECIES: Tuple[str, ...] = (
    "cheetah",
    "tiger",
    "giraffe",
    "zebra",
    "penguin",
    "albatross",
    "leopard",
)

# Facts an agent observes directly, rather than deriving them with a rule.
OBSERVABLE_FEATURES: Tuple[str, ...] = tuple(
    dict.fromkeys(
        premise
        for rule in ANIMAL_RULES
        for premise in rule.positive_premises()
        if premise not in _DERIVED_FACTS
    )
)


def show_rule_base(rules: Sequence[Rule] = ANIMAL_RULES) -> None:
    """
    Display a rule base as a table of premises and conclusions.

    :param rules: rule base to show
    """
    rules_df = pd.DataFrame(
        [
            {
                "rule": rule.name,
                "premises": " AND ".join(rule.premises),
                "concludes": rule.conclusion,
            }
            for rule in rules
        ]
    )
    display(rules_df)


def rules_for_conclusion(
    conclusion: str, rules: Sequence[Rule] = ANIMAL_RULES
) -> List[Rule]:
    """
    Collect every rule that can conclude a given fact.

    :param conclusion: fact to look for, e.g., `mammal`
    :param rules: rule base to search
    :return: rules concluding the fact, e.g., `[R1, R2]` for `mammal`
    """
    return [rule for rule in rules if rule.conclusion == conclusion]


# #############################################################################
# The rule engine: match, conflict resolution, act
# #############################################################################

# Conflict resolution strategies implemented by `RuleEngine.select_rule()`.
STRATEGIES: Tuple[str, ...] = ("specificity", "recency", "priority")


def compute_strata(rules: Sequence[Rule]) -> Dict[str, int]:
    """
    Assign each rule the stratum at which negation as failure is safe.

    A rule that reads `~f` may only fire once every rule concluding `f` has
    had its chance, otherwise a default conclusion is drawn before the fact
    that would block it. Putting the negating rule one stratum higher than
    every producer of `f` enforces that order.

    Mutually negating rules (e.g., `~a -> b` together with `~b -> a`) have no
    stratification, and there every rule is put in stratum 0 so that conflict
    resolution alone decides the outcome.

    :param rules: rule base to stratify
    :return: stratum per rule name, e.g., `{'D1': 1, 'D3': 0}`
    """
    strata = {rule.name: 0 for rule in rules}
    # Each pass lifts a rule above the rules producing its negated premises.
    # A stratifiable rule base settles within one pass per rule.
    for _ in range(len(rules) + 1):
        changed = False
        for rule in rules:
            level = 0
            for premise in rule.premises:
                fact, is_positive = _parse_premise(premise)
                for producer in rules_for_conclusion(fact, rules):
                    offset = 0 if is_positive else 1
                    level = max(level, strata[producer.name] + offset)
            if level != strata[rule.name]:
                strata[rule.name] = level
                changed = True
        if not changed:
            break
    else:
        # The loop ran to exhaustion, so the negation is cyclic.
        _LOG.debug("Rule base is not stratifiable, falling back to stratum 0")
        strata = {rule.name: 0 for rule in rules}
    return strata


class RuleEngine:
    """
    A forward-chaining production system with an explicit recognize-act cycle.

    Each call to `step()` runs one full cycle:
    1. Match: find every rule whose premises hold in working memory.
    2. Conflict resolution: pick one rule out of that conflict set.
    3. Act: assert the rule's conclusion into working memory.
    """

    def __init__(
        self,
        rules: Sequence[Rule],
        *,
        strategy: str = "specificity",
    ) -> None:
        """
        Build an engine over a rule base, with an empty working memory.

        :param rules: rule base the engine fires
        :param strategy: conflict resolution strategy, one of `STRATEGIES`
        """
        hdbg.dassert_in(
            strategy, STRATEGIES, "Unknown conflict resolution strategy"
        )
        self.rules = tuple(rules)
        self.strategy = strategy
        # Stratum per rule, so that a rule reading `~f` waits for every rule
        # concluding `f`.
        self.strata = compute_strata(self.rules)
        # Working memory, keyed by fact name so membership tests are cheap.
        self.facts: Dict[str, Fact] = {}
        # One entry per fired rule, in firing order.
        self.trace: List[Activation] = []
        # Logical clock, incremented every time a fact enters working memory.
        self._clock = 0

    def assert_fact(self, name: str, *, cf: float = 1.0) -> None:
        """
        Put an observed fact into working memory.

        :param name: fact to assert, e.g., `has_hair`
        :param cf: certainty factor of the observation
        """
        if name not in self.facts:
            self.facts[name] = Fact(name, self._clock, "", cf)
            self._clock += 1

    def assert_facts(self, names: Sequence[str]) -> None:
        """
        Assert several facts, in the order given.

        The order matters: it is exactly what the `recency` conflict
        resolution strategy reads.

        :param names: facts to assert, e.g., `["has_hair", "eats_meat"]`
        """
        for name in names:
            self.assert_fact(name)

    def matches(self, rule: Rule) -> bool:
        """
        Check whether a rule is ready to fire against working memory.

        A rule matches when every positive premise is present, every negative
        premise is absent, and the conclusion is not already known. The last
        condition is refraction: it stops a rule from firing forever on the
        same facts.

        :param rule: rule to test
        :return: True if the rule belongs in the conflict set
        """
        holds = rule.conclusion not in self.facts
        for premise in rule.premises:
            fact, is_positive = _parse_premise(premise)
            if (fact in self.facts) != is_positive:
                holds = False
                break
        return holds

    def conflict_set(self) -> List[Rule]:
        """
        Collect every rule that matches the current working memory.

        Only the lowest stratum that has a match is returned, so a default
        rule never fires ahead of the rules that could block it.

        :return: matching rules, in rule base order
        """
        matching = [rule for rule in self.rules if self.matches(rule)]
        if matching:
            lowest = min(self.strata[rule.name] for rule in matching)
            matching = [
                rule for rule in matching if self.strata[rule.name] == lowest
            ]
        return matching

    def _selection_key(self, rule: Rule) -> Any:
        """
        Build the sort key that the active strategy maximizes.

        :param rule: rule in the conflict set
        :return: key whose maximum identifies the selected rule
        """
        # Break every tie by rule base order, so selection is deterministic.
        tie_break = -self.rules.index(rule)
        if self.strategy == "specificity":
            # The most detailed rule wins, i.e., the one with most premises.
            key: Any = (len(rule.positive_premises()), tie_break)
        elif self.strategy == "recency":
            # The rule matching the newest facts wins. Timestamps are compared
            # newest first, so a tie on the newest fact is broken by the next
            # newest, exactly like the OPS5 `LEX` strategy.
            stamps = sorted(
                (self.facts[p].step for p in rule.positive_premises()),
                reverse=True,
            )
            key = (tuple(stamps), tie_break)
        else:
            # The rule author's explicit priority wins.
            key = (rule.salience, tie_break)
        return key

    def select_rule(self, conflict_set: Sequence[Rule]) -> Rule:
        """
        Pick one rule out of the conflict set with the active strategy.

        :param conflict_set: rules that currently match
        :return: rule to fire next
        """
        hdbg.dassert_lte(
            1, len(conflict_set), "Cannot select from an empty conflict set"
        )
        return max(conflict_set, key=self._selection_key)

    def step(self) -> Optional[Activation]:
        """
        Run one recognize-act cycle.

        :return: the activation produced, or `None` at the fixed point where
            no rule matches any more
        """
        conflict_set = self.conflict_set()
        if not conflict_set:
            activation = None
        else:
            rule = self.select_rule(conflict_set)
            # Propagate certainty: a conclusion is no stronger than its
            # weakest premise, discounted by the rule's own certainty.
            premise_cfs = [self.facts[p].cf for p in rule.positive_premises()]
            cf = rule.cf * (min(premise_cfs) if premise_cfs else 1.0)
            self.facts[rule.conclusion] = Fact(
                rule.conclusion, self._clock, rule.name, cf
            )
            self._clock += 1
            activation = Activation(
                step=len(self.trace) + 1,
                rule=rule,
                conflict_set=tuple(r.name for r in conflict_set),
                conclusion=rule.conclusion,
                cf=cf,
            )
            self.trace.append(activation)
            _LOG.debug("fired '%s' -> '%s'", rule.name, rule.conclusion)
        return activation

    def run(self, *, max_steps: int = 50) -> List[Activation]:
        """
        Fire rules until working memory reaches a fixed point.

        :param max_steps: safety bound on the number of cycles
        :return: every activation produced, in firing order
        """
        while len(self.trace) < max_steps:
            if self.step() is None:
                break
        return self.trace

    def asserted_facts(self) -> List[str]:
        """
        List the facts an agent observed, in assertion order.

        :return: fact names, e.g., `["has_hair", "eats_meat"]`
        """
        facts = [f for f in self.facts.values() if not f.rule]
        return [f.name for f in sorted(facts, key=lambda f: f.step)]

    def derived_facts(self) -> List[str]:
        """
        List the facts the rules concluded, in derivation order.

        :return: fact names, e.g., `["mammal", "carnivore", "cheetah"]`
        """
        facts = [f for f in self.facts.values() if f.rule]
        return [f.name for f in sorted(facts, key=lambda f: f.step)]

    def species_found(self) -> List[str]:
        """
        List the species the engine identified, in derivation order.

        :return: species names, e.g., `["cheetah"]`
        """
        return [f for f in self.derived_facts() if f in SPECIES]

    def recompute(self) -> List[str]:
        """
        Rebuild every derived fact from the asserted facts alone.

        This is the truth maintenance step: a conclusion whose rule no longer
        matches simply fails to come back, which is how a default conclusion
        gets retracted (see `cell4_1_non_monotonic()`).

        :return: facts that were derived before but are no longer supported
        """
        previous = set(self.derived_facts())
        asserted = {name: f for name, f in self.facts.items() if not f.rule}
        self.facts = asserted
        self.trace = []
        self._clock = max((f.step for f in asserted.values()), default=-1) + 1
        self.run()
        retracted = sorted(previous - set(self.derived_facts()))
        _LOG.debug("retracted=%s", retracted)
        return retracted

    def explanation_chain(self, fact: str) -> List[Activation]:
        """
        Answer MYCIN's "how" question for a fact: which rules produced it.

        :param fact: fact to explain, e.g., `cheetah`
        :return: activations supporting the fact, conclusion first
        """
        by_conclusion = {a.conclusion: a for a in self.trace}
        chain: List[Activation] = []
        pending = [fact]
        seen = set()
        while pending:
            current = pending.pop(0)
            if current in by_conclusion and current not in seen:
                seen.add(current)
                activation = by_conclusion[current]
                chain.append(activation)
                pending.extend(activation.rule.positive_premises())
        return chain

    def why_needed(self, fact: str) -> List[Rule]:
        """
        Answer MYCIN's "why" question for a fact: which rules consume it.

        :param fact: fact to explain, e.g., `carnivore`
        :return: rules that use the fact as a premise
        """
        return [rule for rule in self.rules if fact in rule.positive_premises()]


def run_rules(
    asserted: Sequence[str],
    *,
    rules: Sequence[Rule] = ANIMAL_RULES,
    strategy: str = "specificity",
) -> RuleEngine:
    """
    Assert facts into a fresh engine and run it to its fixed point.

    :param asserted: facts to assert, in observation order
    :param rules: rule base to fire
    :param strategy: conflict resolution strategy
    :return: the engine, with its working memory and trace filled in
    """
    engine = RuleEngine(rules, strategy=strategy)
    engine.assert_facts(asserted)
    engine.run()
    return engine


# #############################################################################
# Backward chaining
# #############################################################################


def backward_chain(
    goal: str,
    facts: Sequence[str],
    rules: Sequence[Rule] = ANIMAL_RULES,
) -> Tuple[nx.DiGraph, str, bool, List[str]]:
    """
    Prove a goal by goal-driven backward chaining, recording the search.

    The returned graph is the AND-OR proof tree:
    - An OR node is a subgoal, proved if any rule below it succeeds.
    - An AND node is a rule, proved if every premise below it succeeds.

    :param goal: fact to prove, e.g., `cheetah`
    :param facts: facts already known, e.g., `["has_hair", "eats_meat"]`
    :param rules: rule base to search
    :return: proof tree, root node id, whether the goal was proved, and the
        names of the rules expanded, in expansion order
    """
    known = set(facts)
    graph = nx.DiGraph()
    explored: List[str] = []
    counter = itertools.count()

    def _prove(subgoal: str, ancestors: FrozenSet[str]) -> Tuple[str, bool]:
        """
        Prove one subgoal, adding its subtree to the graph.

        :param subgoal: fact to prove
        :param ancestors: subgoals already open on this branch
        :return: node id created for the subgoal, and whether it was proved
        """
        # Node ids must be unique, since the same subgoal can appear on
        # several branches of the tree.
        node = "%s#%d" % (subgoal, next(counter))
        graph.add_node(node, label=subgoal, kind="goal", status="failed")
        if subgoal in known:
            # The subgoal is a fact already in the knowledge base: a leaf.
            graph.nodes[node]["kind"] = "fact"
            graph.nodes[node]["status"] = "proved"
            proved = True
        elif subgoal in ancestors:
            # The subgoal is already open higher up this branch, so expanding
            # it again would loop forever.
            graph.nodes[node]["kind"] = "cycle"
            proved = False
        else:
            proved = False
            candidates = rules_for_conclusion(subgoal, rules)
            if not candidates:
                # No rule concludes the subgoal and it is not a known fact.
                graph.nodes[node]["kind"] = "fact"
            for rule in candidates:
                explored.append(rule.name)
                rule_node = "%s#%d" % (rule.name, next(counter))
                graph.add_node(
                    rule_node, label=rule.name, kind="rule", status="failed"
                )
                graph.add_edge(node, rule_node)
                # A rule node is an AND node: every premise must be proved.
                rule_proved = True
                for premise in rule.positive_premises():
                    child, child_proved = _prove(premise, ancestors | {subgoal})
                    graph.add_edge(rule_node, child)
                    rule_proved = rule_proved and child_proved
                if rule_proved:
                    graph.nodes[rule_node]["status"] = "proved"
                    proved = True
            if proved:
                graph.nodes[node]["status"] = "proved"
        return node, proved

    root, proved = _prove(goal, frozenset())
    _LOG.debug("goal='%s' proved=%s explored=%s", goal, proved, explored)
    return graph, root, proved, explored


# #############################################################################
# Certainty factors
# #############################################################################


def combine_cf(cf_a: float, cf_b: float) -> float:
    """
    Combine two positive certainty factors the way MYCIN does.

    Two independent pieces of evidence reinforce each other without ever
    exceeding 1: $CF = CF_a + CF_b (1 - CF_a)$.

    :param cf_a: certainty already accumulated for the conclusion
    :param cf_b: certainty contributed by another rule
    :return: combined certainty factor, e.g., `0.8` for `0.5` and `0.6`
    """
    return cf_a + cf_b * (1.0 - cf_a)


def _order_rules_by_dependency(rules: Sequence[Rule]) -> List[Rule]:
    """
    Sort rules so that each rule comes after the rules producing its premises.

    :param rules: rule base to sort
    :return: rules in dependency order
    """
    graph = nx.DiGraph()
    for rule in rules:
        graph.add_node(rule.name)
        for premise in rule.positive_premises():
            for producer in rules_for_conclusion(premise, rules):
                graph.add_edge(producer.name, rule.name)
    by_name = {rule.name: rule for rule in rules}
    return [by_name[name] for name in nx.topological_sort(graph)]


def propagate_certainty(
    facts: Sequence[str],
    rules: Sequence[Rule],
) -> Dict[str, float]:
    """
    Propagate certainty factors from observed facts through the rule base.

    A conclusion is no stronger than the weakest premise supporting it,
    discounted by the certainty of the rule itself. Two rules concluding the
    same fact are combined with `combine_cf()`.

    :param facts: observed facts, each taken as certain
    :param rules: rule base carrying the per-rule certainty factors
    :return: certainty factor per fact, e.g.,
        `{'mammal': 0.9, 'carnivore': 0.72}`
    """
    cf: Dict[str, float] = {fact: 1.0 for fact in facts}
    # One pass in dependency order is enough, and guarantees that each rule
    # contributes its evidence exactly once.
    for rule in _order_rules_by_dependency(rules):
        premises = rule.positive_premises()
        if all(premise in cf for premise in premises):
            value = rule.cf * min(cf[premise] for premise in premises)
            if rule.conclusion in cf:
                cf[rule.conclusion] = combine_cf(cf[rule.conclusion], value)
            else:
                cf[rule.conclusion] = value
    return cf


# #############################################################################
# Drawing helpers
# #############################################################################


def wrap_comment(text: str, *, indent: str = "    ") -> str:
    """
    Wrap one long comment line so that it stays inside the comments panel.

    :param text: line to wrap, e.g., a full rule with all its premises
    :param indent: prefix added to every continuation line
    :return: wrapped text, e.g., `R16: carnivore AND dark_spots AND\\n    ...`
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
) -> None:
    """
    Draw a table of strings inside an axes, with one color per row.

    :param ax: axes to draw on
    :param rows: table body, one sequence of cells per row
    :param col_labels: column headers
    :param row_colors: fill color per row, empty to leave the rows white
    :param title: bold title drawn above the table
    :param col_widths: relative column widths, empty for equal widths
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
        table.set_fontsize(9)
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


def draw_fact_panel(
    ax: matplotlib.axes.Axes,
    facts: Dict[str, Fact],
    universe: Sequence[str],
    *,
    title: str,
    retracted: Sequence[str] = (),
) -> None:
    """
    Draw working memory as a table of facts with their provenance.

    :param ax: axes to draw on
    :param facts: working memory, keyed by fact name
    :param universe: every fact worth showing, present or not
    :param title: bold title drawn above the table
    :param retracted: facts that used to hold and no longer do
    """
    rows = []
    colors = []
    for name in universe:
        if name in retracted:
            rows.append([name, "retracted", "support gone"])
            colors.append(_COLOR_RETRACTED)
        elif name in facts:
            fact = facts[name]
            source = fact.rule if fact.rule else "observed"
            rows.append([name, "true", source])
            colors.append(_COLOR_DERIVED if fact.rule else _COLOR_ASSERTED)
        else:
            rows.append([name, "-", "-"])
            colors.append(_COLOR_MISSING)
    draw_table(
        ax,
        rows,
        ["fact", "status", "from"],
        title=title,
        row_colors=colors,
        col_widths=[0.42, 0.28, 0.3],
    )


def build_rule_graph(rules: Sequence[Rule]) -> nx.DiGraph:
    """
    Build the bipartite dependency graph of a rule base.

    Fact nodes and rule nodes alternate: every premise points at its rule, and
    every rule points at its conclusion.

    :param rules: rule base to draw
    :return: graph whose nodes carry a `kind` of `fact` or `rule`
    """
    graph = nx.DiGraph()
    for rule in rules:
        graph.add_node(rule.name, kind="rule", label=rule.name)
        for premise in rule.positive_premises():
            graph.add_node(premise, kind="fact", label=premise)
            graph.add_edge(premise, rule.name)
        graph.add_node(rule.conclusion, kind="fact", label=rule.conclusion)
        graph.add_edge(rule.name, rule.conclusion)
    return graph


def layered_positions(graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """
    Lay out a directed acyclic graph in rows by longest path from a source.

    The graph flows downward, like the proof trees, so that premises sit above
    the rules that consume them.

    :param graph: acyclic graph to lay out
    :return: position per node, e.g., `{'mammal': (0.0, -2.0)}`
    """
    level: Dict[str, int] = {}
    for node in nx.topological_sort(graph):
        parents = list(graph.predecessors(node))
        level[node] = 0 if not parents else max(level[p] for p in parents) + 1
    # Spread the nodes of each row horizontally, centered on zero.
    positions: Dict[str, Tuple[float, float]] = {}
    for depth in sorted(set(level.values())):
        row = [node for node in graph.nodes if level[node] == depth]
        for index, node in enumerate(row):
            positions[node] = (
                float(index) - (len(row) - 1) / 2.0,
                float(-depth),
            )
    return positions


def tree_positions(
    graph: nx.DiGraph, root: str
) -> Dict[str, Tuple[float, float]]:
    """
    Lay out a tree top down, placing each parent above its children.

    :param graph: tree to lay out
    :param root: node to put on top
    :return: position per node
    """
    positions: Dict[str, Tuple[float, float]] = {}
    next_leaf = itertools.count()

    def _place(node: str, depth: int) -> float:
        """
        Place one subtree and return the x coordinate of its root.

        :param node: subtree root
        :param depth: distance from the root of the whole tree
        :return: x coordinate assigned to `node`
        """
        children = list(graph.successors(node))
        if not children:
            # Leaves are laid out left to right in visit order.
            x = float(next(next_leaf))
        else:
            x = float(np.mean([_place(child, depth + 1) for child in children]))
        positions[node] = (x, float(-depth))
        return x

    _place(root, 0)
    return positions


def draw_graph(
    ax: matplotlib.axes.Axes,
    graph: nx.DiGraph,
    positions: Dict[str, Tuple[float, float]],
    node_colors: Dict[str, str],
    *,
    title: str,
    highlight_edges: Sequence[Tuple[str, str]] = (),
    font_size: int = 7,
) -> None:
    """
    Draw a labelled graph, with rule nodes as boxes and facts as rounded boxes.

    Each node is drawn as a text box rather than as a marker, so that a long
    fact name such as `black_stripes` still fits inside its node.

    :param ax: axes to draw on
    :param graph: graph to draw
    :param positions: position per node
    :param node_colors: fill color per node
    :param title: bold title drawn above the graph
    :param highlight_edges: edges to draw thick and orange
    :param font_size: font size of the node labels
    """
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    # Draw the edges first, so that the node boxes cover their endpoints.
    highlight = set(highlight_edges)
    plain = [edge for edge in graph.edges if edge not in highlight]
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=plain,
        edge_color="#999999",
        node_size=900,
        ax=ax,
    )
    if highlight:
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=list(highlight),
            edge_color="#d35400",
            width=2.5,
            node_size=900,
            ax=ax,
        )
    for node, data in graph.nodes(data=True):
        x, y = positions[node]
        # Rules get square boxes, facts get rounded ones.
        is_rule = data.get("kind") == "rule"
        ax.text(
            x,
            y,
            data.get("label", node),
            ha="center",
            va="center",
            fontsize=font_size,
            bbox=dict(
                boxstyle="square,pad=0.4" if is_rule else "round,pad=0.4",
                facecolor=node_colors.get(node, _COLOR_MISSING),
                edgecolor="#555555",
            ),
        )
    # `ax.text()` does not grow the data limits, so set them from the layout.
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)


# #############################################################################
# Cell 1.1: Rules, working memory, and the reflex agent baseline
# #############################################################################

# Facts the two agents can observe in this cell, in the order the checkboxes
# are laid out.
_CELL1_PERCEPTS: Tuple[str, ...] = (
    "has_hair",
    "eats_meat",
    "tawny",
    "dark_spots",
)

# The reflex agent's whole program: one action per percept, no memory.
_REFLEX_TABLE: Dict[str, str] = {
    "has_hair": "report: furry animal",
    "eats_meat": "report: predator",
    "tawny": "report: tawny animal",
    "dark_spots": "report: spotted animal",
}


def _draw_agent_pipelines(
    ax: matplotlib.axes.Axes, agent_type: str, *, action: str
) -> None:
    """
    Draw the reflex and the rule-based pipelines, highlighting the active one.

    :param ax: axes to draw on
    :param agent_type: active agent, `reflex` or `rule-based`
    :param action: action the active agent produced, shown in the last box
    """
    ax.set_title("Agent architectures", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    pipelines = (
        ("reflex", 7.5, ("percept", "lookup table", "act")),
        (
            "rule-based",
            3.0,
            ("percept", "working memory", "match", "fire", "act"),
        ),
    )
    for name, y, stages in pipelines:
        is_active = name == agent_type
        color = "#5dade2" if is_active else "#ededed"
        edge = "#1b4f72" if is_active else "#bbbbbb"
        text_color = "black" if is_active else "#999999"
        width = 9.0 / len(stages)
        ax.text(
            0.2,
            y + 1.5,
            "%s agent" % name,
            fontsize=11,
            fontweight="bold",
            color=text_color,
        )
        for index, stage in enumerate(stages):
            x = 0.3 + index * width
            box = mpatches.FancyBboxPatch(
                (x, y - 0.5),
                width * 0.82,
                1.1,
                boxstyle="round,pad=0.08",
                facecolor=color,
                edgecolor=edge,
            )
            ax.add_patch(box)
            ax.text(
                x + width * 0.41,
                y + 0.05,
                stage,
                ha="center",
                va="center",
                # The rule-based pipeline has more, hence narrower, stages.
                fontsize=8 if len(stages) <= 3 else 7,
                color=text_color,
            )
            # Draw the arrow linking this stage to the next one.
            if index + 1 < len(stages):
                ax.annotate(
                    "",
                    xy=(x + width, y + 0.05),
                    xytext=(x + width * 0.82, y + 0.05),
                    arrowprops=dict(arrowstyle="->", color=edge),
                )
    ax.text(
        0.2,
        0.6,
        "action: %s" % action,
        fontsize=10,
        fontweight="bold",
        color="#7d3c98",
    )


def cell1_1_reflex_vs_rule_based(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Contrast a reflex agent with a rule-based agent on the same percepts.

    Interactive controls (ipywidgets):
    - `agent`: which agent reacts to the percepts
    - one checkbox per percept fact

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    agent_dropdown = ipywidgets.Dropdown(
        options=["reflex", "rule-based"],
        value="reflex",
        description="agent:",
        style={"description_width": "initial"},
    )
    checkboxes = {
        fact: ipywidgets.Checkbox(
            value=fact == "has_hair",
            description=fact,
            indent=False,
            layout=ipywidgets.Layout(width="160px"),
        )
        for fact in _CELL1_PERCEPTS
    }
    output = ipywidgets.Output()
    # The reflex agent has no memory, so it reacts only to the newest percept.
    state: Dict[str, str] = {"last": "has_hair"}

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw both panels for the current agent and percept set.

        :param change: widget change record, used to spot the newest percept
        """
        if change is not None and isinstance(change, dict):
            owner = change.get("owner")
            if owner is not None and change.get("new") is True:
                state["last"] = owner.description
        with output:
            clear_output(wait=True)
            agent_type = agent_dropdown.value
            selected = [f for f in _CELL1_PERCEPTS if checkboxes[f].value]
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Run the rule-based agent over every percept currently true.
            engine = run_rules(selected)
            if agent_type == "reflex":
                # The reflex agent never chains, so it has no derived facts.
                last = state["last"] if state["last"] in selected else ""
                action = _REFLEX_TABLE.get(last, "(no percept)")
                visible: Dict[str, Fact] = {
                    name: fact
                    for name, fact in engine.facts.items()
                    if not fact.rule and name == last
                }
            else:
                action = (
                    "report: %s" % engine.species_found()[0]
                    if engine.species_found()
                    else "keep sensing"
                )
                visible = engine.facts
            # Panel 1: the two agent pipelines, the active one highlighted.
            _draw_agent_pipelines(ax1, agent_type, action=action)
            # Panel 2: what each agent holds about the animal right now.
            draw_fact_panel(
                ax2,
                visible,
                _CELL1_PERCEPTS + ("mammal", "carnivore", "cheetah"),
                title="Working memory",
            )
            # Panel 3: comments on the current state.
            text = (
                "Parameters:\n"
                "  agent: %s\n"
                "  percepts on: %s\n\n"
                "Reflex agent:\n"
                "  newest percept: %s\n"
                "  action: %s\n\n"
                "Rule-based agent:\n"
                "  facts in memory: %d\n"
                "  derived: %s\n"
                "  rules fired: %s\n"
                "  species: %s"
                % (
                    agent_type,
                    ", ".join(selected) if selected else "(none)",
                    state["last"] if state["last"] in selected else "(none)",
                    _REFLEX_TABLE.get(
                        state["last"] if state["last"] in selected else "",
                        "(none)",
                    ),
                    len(engine.facts),
                    ", ".join(engine.derived_facts()) or "(none)",
                    ", ".join(a.rule.name for a in engine.trace) or "(none)",
                    ", ".join(engine.species_found()) or "(none yet)",
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "agent": "<code>reflex</code> maps the newest percept straight to "
            "an action, <code>rule-based</code> stores percepts in working "
            "memory and chains rules over them",
            "percepts": "the facts the agent observes; the rule-based agent "
            "needs <code>has_hair</code>, <code>eats_meat</code>, "
            "<code>tawny</code>, and <code>dark_spots</code> together to "
            "reach <code>cheetah</code>",
        }
    )
    agent_dropdown.observe(update_plot, names="value")
    for checkbox in checkboxes.values():
        checkbox.observe(update_plot, names="value")
    update_plot()
    # Top row: controls on the left, info box on the right.
    # Bottom row: the pipelines, the fact table, and the comments panel.
    controls = ipywidgets.VBox(
        [agent_dropdown, ipywidgets.VBox(list(checkboxes.values()))],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.1: Forward chaining, watching working memory fill up
# #############################################################################

# Percepts that can be switched on in the forward chaining cell.
_CELL2_PERCEPTS: Tuple[str, ...] = (
    "has_hair",
    "eats_meat",
    "tawny",
    "dark_spots",
    "black_stripes",
)


def _relevant_rules(percepts: Sequence[str]) -> List[Rule]:
    """
    Keep the rules that could ever fire from a set of observable percepts.

    The graph drawn in `cell2_1_forward_chaining()` must keep its shape while
    the checkboxes change, so relevance is computed once from the full percept
    universe rather than from the percepts currently on.

    :param percepts: every percept the cell can observe
    :return: rules whose premises are all reachable from those percepts
    """
    reachable = set(percepts)
    changed = True
    while changed:
        changed = False
        for rule in ANIMAL_RULES:
            premises = set(rule.positive_premises())
            if premises <= reachable and rule.conclusion not in reachable:
                reachable.add(rule.conclusion)
                changed = True
    return [
        rule
        for rule in ANIMAL_RULES
        if set(rule.positive_premises()) <= reachable
    ]


def draw_wm_timeline(
    ax: matplotlib.axes.Axes,
    engine: RuleEngine,
    universe: Sequence[str],
    n_steps: int,
    *,
    title: str,
) -> None:
    """
    Draw working memory as a fact-by-step grid, one column per fired rule.

    :param ax: axes to draw on
    :param engine: engine whose trace and facts are drawn
    :param universe: facts to show as rows, in display order
    :param n_steps: number of rule-firing columns to draw
    :param title: bold title drawn above the grid
    """
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    # Column 0 is the initial working memory, before any rule has fired.
    n_cols = n_steps + 1
    # A fact appears in column `c` when it entered at or before step `c`.
    entry_step = {}
    for name, fact in engine.facts.items():
        entry_step[name] = (
            0
            if not fact.rule
            else engine.trace.index(
                next(a for a in engine.trace if a.conclusion == name)
            )
            + 1
        )
    for row, name in enumerate(universe):
        for col in range(n_cols):
            present = name in entry_step and entry_step[name] <= col
            color = (
                _COLOR_ASSERTED if name not in _DERIVED_FACTS else _COLOR_DERIVED
            )
            rect = mpatches.Rectangle(
                (col, -row),
                0.92,
                0.86,
                facecolor=color if present else _COLOR_MISSING,
                edgecolor="#aaaaaa",
            )
            ax.add_patch(rect)
    ax.set_yticks([-row + 0.43 for row in range(len(universe))])
    ax.set_yticklabels(universe, fontsize=8)
    ax.set_xticks([col + 0.46 for col in range(n_cols)])
    labels = ["init"] + [a.rule.name for a in engine.trace[:n_steps]]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("working memory after each fired rule", fontsize=9)
    ax.set_xlim(-0.1, n_cols)
    ax.set_ylim(-len(universe) + 0.05, 1.0)
    ax.grid(False)


def cell2_1_forward_chaining(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Fire forward-chaining rules one at a time and watch working memory grow.

    Interactive controls (ipywidgets):
    - `fire next rule`: run exactly one recognize-act cycle
    - `run to fixed point`: fire until no rule matches
    - `reset`: empty working memory and start over
    - one checkbox per initial percept

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    checkboxes = {
        fact: ipywidgets.Checkbox(
            value=fact != "black_stripes",
            description=fact,
            indent=False,
            layout=ipywidgets.Layout(width="170px"),
        )
        for fact in _CELL2_PERCEPTS
    }
    step_button = ipywidgets.Button(
        description="fire next rule",
        button_style="primary",
        layout=ipywidgets.Layout(width="150px"),
    )
    run_button = ipywidgets.Button(
        description="run to fixed point",
        layout=ipywidgets.Layout(width="150px"),
    )
    reset_button = ipywidgets.Button(
        description="reset",
        layout=ipywidgets.Layout(width="150px"),
    )
    output = ipywidgets.Output()
    state: Dict[str, Any] = {}
    # The rule subset and its graph never change, so build them once.
    relevant = _relevant_rules(_CELL2_PERCEPTS)
    graph = build_rule_graph(relevant)
    positions = layered_positions(graph)
    # Rows of the timeline: the percepts first, then everything the relevant
    # rules can conclude.
    timeline_facts = list(_CELL2_PERCEPTS) + list(
        dict.fromkeys(rule.conclusion for rule in relevant)
    )

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the timeline and the dependency graph for the current trace.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            engine = state["engine"]
            trace = engine.trace
            last = trace[-1] if trace else None
            # The dependency graph needs more room than the other two panels.
            _, (ax1, ax2, ax3) = plt.subplots(
                1,
                3,
                figsize=figsize,
                gridspec_kw={"width_ratios": [1, 1.4, 0.9]},
            )
            # Panel 1: one column of working memory per fired rule.
            draw_wm_timeline(
                ax1,
                engine,
                timeline_facts,
                len(trace),
                title="Working memory timeline",
            )
            # Panel 2: the rule dependency graph, with the last rule fired
            # highlighted along with the fact it produced.
            node_colors = {}
            for node, data in graph.nodes(data=True):
                if data["kind"] == "rule":
                    fired = any(a.rule.name == node for a in trace)
                    node_colors[node] = _COLOR_RULE if fired else _COLOR_MISSING
                else:
                    node_colors[node] = (
                        _COLOR_DERIVED
                        if node in engine.facts and engine.facts[node].rule
                        else _COLOR_ASSERTED
                        if node in engine.facts
                        else _COLOR_MISSING
                    )
            highlight: List[Tuple[str, str]] = []
            if last is not None:
                node_colors[last.rule.name] = _COLOR_RULE_FIRED
                highlight = [
                    (p, last.rule.name) for p in last.rule.positive_premises()
                ] + [(last.rule.name, last.conclusion)]
            draw_graph(
                ax2,
                graph,
                positions,
                node_colors,
                title="Rule dependency graph",
                highlight_edges=highlight,
            )
            # Panel 3: comments on the step just taken.
            text = (
                "Parameters:\n"
                "  initial facts: %s\n\n"
                "Cycle:\n"
                "  step: %d\n"
                "  rule fired: %s\n"
                "  new fact: %s\n"
                "  conflict set: %s\n\n"
                "Working memory:\n"
                "  facts: %d\n"
                "  derived: %s\n"
                "  at fixed point: %s"
                % (
                    ", ".join(engine.asserted_facts()) or "(none)",
                    len(trace),
                    wrap_comment(last.rule.as_text())
                    if last
                    else "(nothing fired yet)",
                    last.conclusion if last else "(none)",
                    ", ".join(last.conflict_set) if last else "(none)",
                    len(engine.facts),
                    ", ".join(engine.derived_facts()) or "(none)",
                    not engine.conflict_set(),
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    def rebuild(change: Optional[Any] = None) -> None:
        """
        Start a fresh run from the percepts currently checked.

        :param change: widget change record (unused)
        """
        _ = change
        engine = RuleEngine(relevant)
        engine.assert_facts([f for f in _CELL2_PERCEPTS if checkboxes[f].value])
        state["engine"] = engine
        update_plot()

    def on_step(_button: Any) -> None:
        """
        Run exactly one recognize-act cycle.

        :param _button: clicked button (unused)
        """
        state["engine"].step()
        update_plot()

    def on_run(_button: Any) -> None:
        """
        Run cycles until working memory stops growing.

        :param _button: clicked button (unused)
        """
        state["engine"].run()
        update_plot()

    param_info = make_param_info(
        {
            "fire next rule": "runs one match, conflict resolution, act cycle",
            "run to fixed point": "fires until no rule matches any more",
            "reset": "empties working memory and reasserts the checked facts",
            "initial facts": "the percepts working memory starts with",
        }
    )
    step_button.on_click(on_step)
    run_button.on_click(on_run)
    reset_button.on_click(rebuild)
    for checkbox in checkboxes.values():
        checkbox.observe(rebuild, names="value")
    rebuild()
    controls = ipywidgets.VBox(
        [
            ipywidgets.HBox([step_button, run_button, reset_button]),
            ipywidgets.VBox(list(checkboxes.values())),
        ],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.2: Backward chaining and the AND-OR proof tree
# #############################################################################

# The facts both chaining directions start from in this cell.
_CELL2_2_FACTS: Tuple[str, ...] = (
    "has_hair",
    "eats_meat",
    "tawny",
    "dark_spots",
)


def cell2_2_backward_chaining(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Prove a goal by backward chaining and compare the work against forward
    chaining on the same facts.

    Interactive controls (ipywidgets):
    - `goal`: the fact backward chaining tries to prove

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    goal_dropdown = ipywidgets.Dropdown(
        options=["cheetah", "tiger", "carnivore", "mammal", "zebra"],
        value="cheetah",
        description="goal:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the proof tree and the rules-explored comparison.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            goal = goal_dropdown.value
            graph, root, proved, explored = backward_chain(goal, _CELL2_2_FACTS)
            forward = run_rules(_CELL2_2_FACTS)
            # The proof tree is the widest panel, so give it extra room.
            _, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.9, 1, 1]}
            )
            # Panel 1: the AND-OR tree the backward chainer walked.
            node_colors = {}
            for node, data in graph.nodes(data=True):
                if data["status"] == "failed":
                    node_colors[node] = _COLOR_FAILED
                elif data["kind"] == "rule":
                    node_colors[node] = _COLOR_RULE
                elif node == root:
                    node_colors[node] = _COLOR_GOAL
                else:
                    node_colors[node] = _COLOR_DERIVED
            draw_graph(
                ax1,
                graph,
                tree_positions(graph, root),
                node_colors,
                title="AND-OR proof tree for '%s'" % goal,
            )
            # Panel 2: how many rules each direction touched.
            counts = pd.Series(
                {
                    "forward\n(rules fired)": len(forward.trace),
                    "backward\n(rules expanded)": len(explored),
                }
            )
            ax2.bar(
                counts.index,
                counts.to_numpy(),
                color=["#5dade2", "#f5b041"],
                edgecolor="#555555",
            )
            for index, value in enumerate(counts.to_numpy()):
                ax2.text(
                    index, value + 0.05, str(value), ha="center", fontsize=11
                )
            ax2.set_ylabel("rules explored", fontsize=10)
            ax2.set_title(
                "Search effort", fontsize=13, fontweight="bold", pad=12
            )
            ax2.set_ylim(0, max(counts.max() + 1, 2))
            # Panel 3: comments on the proof just built.
            text = (
                "Parameters:\n"
                "  goal: %s\n"
                "  facts known: %s\n\n"
                "Backward chaining:\n"
                "  proved: %s\n"
                "  rules expanded: %d\n"
                "  order: %s\n\n"
                "Forward chaining:\n"
                "  rules fired: %d\n"
                "  facts derived: %s\n"
                "  goal reached: %s"
                % (
                    goal,
                    ", ".join(_CELL2_2_FACTS),
                    proved,
                    len(explored),
                    ", ".join(explored) or "(none)",
                    len(forward.trace),
                    ", ".join(forward.derived_facts()) or "(none)",
                    goal in forward.facts,
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "goal": "the fact backward chaining tries to prove; "
            "<code>zebra</code> is not provable from these facts, so its "
            "tree shows a failed branch",
            "facts known": "<code>%s</code>" % ", ".join(_CELL2_2_FACTS),
        }
    )
    goal_dropdown.observe(update_plot, names="value")
    update_plot()
    top_row = ipywidgets.HBox(
        [
            ipywidgets.VBox(
                [goal_dropdown],
                layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
            ),
            param_info,
        ]
    )
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 2.3: Conflict resolution by specificity, recency, and priority
# #############################################################################

# A deliberately conflicting rule pair: a tawny, spotted, tree-climbing
# carnivore matches both the cheetah rule and the leopard rule, and each rule
# blocks the other with a negative premise. Exactly one of them can fire, so
# the conflict resolution strategy alone decides the species.
_CONFLICT_RULES: Tuple[Rule, ...] = (
    Rule("R1", ("has_hair",), "mammal"),
    Rule("R5", ("mammal", "eats_meat"), "carnivore"),
    Rule("R9", ("carnivore", "tawny", "dark_spots", "~leopard"), "cheetah"),
    Rule(
        "R16",
        ("carnivore", "dark_spots", "solitary", "climbs_trees", "~cheetah"),
        "leopard",
        salience=5,
    ),
)

# Assertion order matters: `tawny` arrives after `solitary` and
# `climbs_trees`, which is what lets recency and specificity disagree.
_CONFLICT_FACTS: Tuple[str, ...] = (
    "has_hair",
    "eats_meat",
    "solitary",
    "climbs_trees",
    "tawny",
    "dark_spots",
)


def _conflict_rules_with_priority(priority: int) -> Tuple[Rule, ...]:
    """
    Rebuild the conflicting rule base with a chosen priority for `R9`.

    :param priority: salience given to the cheetah rule `R9`
    :return: rule base with `R9` at the requested salience
    """
    return tuple(
        dataclasses.replace(rule, salience=priority)
        if rule.name == "R9"
        else rule
        for rule in _CONFLICT_RULES
    )


def cell2_3_conflict_resolution(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show how the conflict resolution strategy decides the final fact set.

    Interactive controls (ipywidgets):
    - `strategy`: which rule the engine picks out of the conflict set
    - `prio`: salience of the cheetah rule `R9`, used by `priority`

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    strategy_dropdown = ipywidgets.Dropdown(
        options=list(STRATEGIES),
        value="specificity",
        description="strategy:",
        style={"description_width": "initial"},
    )
    prio_slider, prio_box = htutori.build_widget_control(
        name="prio",
        description="priority of the cheetah rule R9",
        min_val=0,
        max_val=10,
        step=1,
        initial_value=0,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the conflict set and the per-strategy outcomes.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            strategy = strategy_dropdown.value
            rules = _conflict_rules_with_priority(prio_slider.value)
            # Replay the run up to the cycle where the conflict shows up, so
            # the conflict set panel shows more than one matching rule.
            engine = RuleEngine(rules, strategy=strategy)
            engine.assert_facts(_CONFLICT_FACTS)
            while len(engine.conflict_set()) < 2 and engine.conflict_set():
                engine.step()
            conflict_set = engine.conflict_set()
            selected = engine.select_rule(conflict_set) if conflict_set else None
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: every rule that currently matches, selected one first.
            rows = []
            colors = []
            for rule in conflict_set:
                is_selected = selected is not None and rule.name == selected.name
                rows.append(
                    [
                        rule.name,
                        str(len(rule.positive_premises())),
                        str(rule.salience),
                        rule.conclusion,
                    ]
                )
                colors.append(
                    _COLOR_RULE_FIRED if is_selected else _COLOR_MISSING
                )
            draw_table(
                ax1,
                rows,
                ["rule", "premises", "salience", "concludes"],
                title="Conflict set (selected rule highlighted)",
                row_colors=colors,
                col_widths=[0.2, 0.25, 0.25, 0.3],
            )
            # Panel 2: run each strategy to completion on the same facts.
            outcomes = {}
            for name in STRATEGIES:
                run = RuleEngine(rules, strategy=name)
                run.assert_facts(_CONFLICT_FACTS)
                run.run()
                outcomes[name] = run
            heights = [
                len(outcomes[name].derived_facts()) for name in STRATEGIES
            ]
            ax2.bar(
                list(STRATEGIES),
                heights,
                color=[_STRATEGY_COLORS[name] for name in STRATEGIES],
                edgecolor="#555555",
            )
            for index, name in enumerate(STRATEGIES):
                species = outcomes[name].species_found()
                ax2.text(
                    index,
                    heights[index] + 0.05,
                    species[0] if species else "(none)",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )
            ax2.set_ylabel("facts derived", fontsize=10)
            ax2.set_ylim(0, max(heights) + 1)
            ax2.set_title(
                "Final fact set per strategy",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )
            # Panel 3: comments on the current strategy and conflict.
            text = (
                "Parameters:\n"
                "  strategy: %s\n"
                "  prio (R9): %d\n\n"
                "Conflict:\n"
                "  conflict set size: %d\n"
                "  rules matching: %s\n"
                "  selected: %s\n\n"
                "Outcome per strategy:\n"
                "  specificity: %s\n"
                "  recency: %s\n"
                "  priority: %s"
                % (
                    strategy,
                    prio_slider.value,
                    len(conflict_set),
                    ", ".join(r.name for r in conflict_set) or "(none)",
                    wrap_comment(selected.as_text()) if selected else "(none)",
                    ", ".join(outcomes["specificity"].species_found())
                    or "(none)",
                    ", ".join(outcomes["recency"].species_found()) or "(none)",
                    ", ".join(outcomes["priority"].species_found()) or "(none)",
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "strategy": "<code>specificity</code> picks the rule with most "
            "premises, <code>recency</code> the rule matching the newest "
            "facts, <code>priority</code> the rule with highest salience",
            "prio": "salience of the cheetah rule <code>R9</code>; the "
            "leopard rule <code>R16</code> sits at 5, so raising this past 5 "
            "flips the <code>priority</code> outcome",
        }
    )
    strategy_dropdown.observe(update_plot, names="value")
    prio_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [strategy_dropdown, prio_box],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 3.1: Explaining conclusions and certainty factors
# #############################################################################

# Rules used for the certainty factor demo. The negative premises are dropped
# so that every candidate species fires and can be ranked.
_CF_RULES: Tuple[Rule, ...] = (
    Rule("R1", ("has_hair",), "mammal", cf=0.9),
    Rule("R5", ("mammal", "eats_meat"), "carnivore", cf=0.8),
    Rule("R9", ("carnivore", "tawny", "dark_spots"), "cheetah", cf=0.7),
    Rule("R10", ("carnivore", "tawny", "black_stripes"), "tiger", cf=0.75),
    Rule(
        "R16",
        ("carnivore", "dark_spots", "solitary", "climbs_trees"),
        "leopard",
        cf=0.6,
    ),
)

# Facts that make all three species rules fire at once, so the ranking is a
# real differential diagnosis rather than a single answer.
_CF_FACTS: Tuple[str, ...] = (
    "has_hair",
    "eats_meat",
    "tawny",
    "dark_spots",
    "black_stripes",
    "solitary",
    "climbs_trees",
)


def _cf_rules_with_values(
    cf1: float, cf2: float, cf3: float
) -> Tuple[Rule, ...]:
    """
    Rebuild the certainty factor rule base with three slider values.

    :param cf1: certainty of `R1`, `has_hair -> mammal`
    :param cf2: certainty of `R5`, `mammal AND eats_meat -> carnivore`
    :param cf3: certainty of `R9`, the cheetah rule
    :return: rule base carrying the requested certainty factors
    """
    overrides = {"R1": cf1, "R5": cf2, "R9": cf3}
    return tuple(
        dataclasses.replace(rule, cf=overrides.get(rule.name, rule.cf))
        for rule in _CF_RULES
    )


def _explanation_graph(engine: RuleEngine, conclusion: str) -> nx.DiGraph:
    """
    Build the proof graph behind one conclusion, for the "how" answer.

    :param engine: engine that already ran to its fixed point
    :param conclusion: fact to explain, e.g., `cheetah`
    :return: graph of the facts and rules supporting the conclusion
    """
    graph = nx.DiGraph()
    graph.add_node(conclusion, kind="fact", label=conclusion)
    for activation in engine.explanation_chain(conclusion):
        rule = activation.rule
        graph.add_node(rule.name, kind="rule", label=rule.name)
        graph.add_edge(rule.name, activation.conclusion)
        for premise in rule.positive_premises():
            graph.add_node(premise, kind="fact", label=premise)
            graph.add_edge(premise, rule.name)
    return graph


def cell3_1_explanation_and_certainty(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Trace a conclusion back to its rules, then rank diagnoses by certainty.

    Interactive controls (ipywidgets):
    - `conclusion`: the fact to explain
    - `cf1`, `cf2`, `cf3`: certainty factors of `R1`, `R5`, and `R9`

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    conclusion_dropdown = ipywidgets.Dropdown(
        options=["cheetah", "leopard", "tiger", "carnivore", "mammal"],
        value="cheetah",
        description="conclusion:",
        style={"description_width": "initial"},
    )
    cf_sliders = []
    cf_boxes = []
    for name, description, initial in (
        ("cf1", "certainty of R1 (has_hair -> mammal)", 0.9),
        ("cf2", "certainty of R5 (-> carnivore)", 0.8),
        ("cf3", "certainty of R9 (-> cheetah)", 0.7),
    ):
        slider, box = htutori.build_widget_control(
            name=name,
            description=description,
            min_val=0.0,
            max_val=1.0,
            step=0.05,
            initial_value=initial,
            is_float=True,
        )
        cf_sliders.append(slider)
        cf_boxes.append(box)
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the explanation trace and the certainty ranking.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            conclusion = conclusion_dropdown.value
            rules = _cf_rules_with_values(*[s.value for s in cf_sliders])
            engine = run_rules(_CF_FACTS, rules=rules)
            cf = propagate_certainty(_CF_FACTS, rules)
            # The explanation trace is the widest panel.
            _, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1, 1]}
            )
            # Panel 1: the rules and facts behind the chosen conclusion.
            graph = _explanation_graph(engine, conclusion)
            node_colors = {
                node: _COLOR_RULE if data["kind"] == "rule" else _COLOR_DERIVED
                for node, data in graph.nodes(data=True)
            }
            node_colors[conclusion] = _COLOR_GOAL
            draw_graph(
                ax1,
                graph,
                layered_positions(graph),
                node_colors,
                title="How '%s' was concluded" % conclusion,
            )
            # Panel 2: candidate species ranked by propagated certainty.
            ranking = pd.Series(
                {name: cf[name] for name in ("cheetah", "leopard", "tiger")}
            ).sort_values(ascending=False)
            ax2.barh(
                list(ranking.index)[::-1],
                ranking.to_numpy()[::-1],
                color="#5dade2",
                edgecolor="#555555",
            )
            for index, value in enumerate(ranking.to_numpy()[::-1]):
                ax2.text(
                    value + 0.01, index, "%.3f" % value, va="center", fontsize=10
                )
            ax2.set_xlim(0, 1.05)
            ax2.set_xlabel("propagated certainty factor", fontsize=10)
            ax2.set_title(
                "Diagnosis ranking", fontsize=13, fontweight="bold", pad=12
            )
            # Panel 3: the "how" and "why" answers, plus the certainty values.
            chain = engine.explanation_chain(conclusion)
            how = (
                "\n    ".join(
                    wrap_comment(a.rule.as_text(), indent="        ")
                    for a in chain
                )
                if chain
                else "observed directly"
            )
            why = ", ".join(r.name for r in engine.why_needed(conclusion))
            text = (
                "Parameters:\n"
                "  conclusion: %s\n"
                "  cf1 (R1): %.2f\n"
                "  cf2 (R5): %.2f\n"
                "  cf3 (R9): %.2f\n\n"
                "HOW was it concluded:\n"
                "    %s\n\n"
                "WHY is it needed:\n"
                "  premise of: %s\n\n"
                "Certainty factors:\n"
                "  mammal: %.3f\n"
                "  carnivore: %.3f\n"
                "  top diagnosis: %s (%.3f)"
                % (
                    conclusion,
                    cf_sliders[0].value,
                    cf_sliders[1].value,
                    cf_sliders[2].value,
                    how,
                    why or "(nothing, it is a leaf conclusion)",
                    cf["mammal"],
                    cf["carnivore"],
                    ranking.index[0],
                    ranking.iloc[0],
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "conclusion": "the fact whose derivation is traced back to the "
            "rules that produced it",
            "cf1, cf2": "certainty of the two shared rules, which scale every "
            "diagnosis at once and never reorder them",
            "cf3": "certainty of the cheetah rule alone, which does reorder "
            "the ranking once it drops below the leopard rule",
        }
    )
    conclusion_dropdown.observe(update_plot, names="value")
    for slider in cf_sliders:
        slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [conclusion_dropdown] + cf_boxes,
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 4.1: Non-monotonic reasoning, Tweety the penguin
# #############################################################################

# The default reasoning rule base. `D1` is the default rule "birds fly",
# guarded by a negative premise: it holds only while nothing marks the bird as
# abnormal.
_DEFAULT_RULES: Tuple[Rule, ...] = (
    Rule("D1", ("bird", "~abnormal"), "flies"),
    Rule("D2", ("penguin",), "bird"),
    Rule("D3", ("penguin",), "abnormal"),
    Rule("D4", ("flies",), "reaches_tree_nest"),
)

# Facts shown in both panels, whether they hold or not.
_TWEETY_UNIVERSE: Tuple[str, ...] = (
    "bird",
    "penguin",
    "abnormal",
    "flies",
    "reaches_tree_nest",
)


def cell4_1_non_monotonic(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Retract a default conclusion by adding one fact.

    Interactive controls (ipywidgets):
    - `Tweety is a penguin`: assert the new fact and re-derive
    - `reset`: go back to the initial scenario

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    penguin_button = ipywidgets.Button(
        description="Tweety is a penguin",
        button_style="primary",
        layout=ipywidgets.Layout(width="190px"),
    )
    reset_button = ipywidgets.Button(
        description="reset",
        layout=ipywidgets.Layout(width="190px"),
    )
    output = ipywidgets.Output()
    state: Dict[str, Any] = {}

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the before and after panels of working memory.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            engine = state["engine"]
            retracted = state["retracted"]
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: the initial scenario, frozen for comparison.
            draw_fact_panel(
                ax1,
                state["before"],
                _TWEETY_UNIVERSE,
                title="Before: only 'Tweety is a bird'",
            )
            # Panel 2: working memory after the new fact, with the
            # conclusions that lost their support marked as retracted.
            draw_fact_panel(
                ax2,
                engine.facts,
                _TWEETY_UNIVERSE,
                title="After: %s"
                % (
                    "'Tweety is a penguin' added"
                    if "penguin" in engine.facts
                    else "nothing added yet"
                ),
                retracted=retracted,
            )
            # Panel 3: comments on what the new fact did.
            text = (
                "Parameters:\n"
                "  penguin asserted: %s\n\n"
                "Working memory:\n"
                "  asserted: %s\n"
                "  derived: %s\n"
                "  retracted: %s\n\n"
                "Default rule D1:\n"
                "  bird AND ~abnormal -> flies\n"
                "  fires now: %s\n\n"
                "Does Tweety fly: %s"
                % (
                    "penguin" in engine.facts,
                    ", ".join(engine.asserted_facts()),
                    ", ".join(engine.derived_facts()) or "(none)",
                    ", ".join(retracted) or "(nothing)",
                    "flies" in engine.facts,
                    "flies" in engine.facts,
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    def rebuild(change: Optional[Any] = None) -> None:
        """
        Restore the initial scenario, where only the default rule applies.

        :param change: widget change record (unused)
        """
        _ = change
        engine = RuleEngine(_DEFAULT_RULES)
        engine.assert_fact("bird")
        engine.run()
        state["engine"] = engine
        # Freeze a copy of working memory, to show as the "before" panel.
        state["before"] = dict(engine.facts)
        state["retracted"] = []
        update_plot()

    def on_penguin(_button: Any) -> None:
        """
        Assert that Tweety is a penguin and re-derive everything.

        :param _button: clicked button (unused)
        """
        engine = state["engine"]
        engine.assert_fact("penguin")
        state["retracted"] = engine.recompute()
        update_plot()

    param_info = make_param_info(
        {
            "Tweety is a penguin": "asserts one new fact, which makes "
            "<code>abnormal</code> hold and removes the support for "
            "<code>flies</code>",
            "reset": "restores the scenario where only "
            "<code>Tweety is a bird</code> is known",
        }
    )
    penguin_button.on_click(on_penguin)
    reset_button.on_click(rebuild)
    rebuild()
    controls = ipywidgets.VBox(
        [penguin_button, reset_button],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 4.2: Closed world vs open world on the same query
# #############################################################################

# Facts observed about the animal being identified in this cell.
_CWA_FACTS: Tuple[str, ...] = ("has_hair", "eats_meat", "tawny", "dark_spots")

# Queries worth asking: two facts that hold, two that are simply absent, and
# one species the knowledge base cannot derive.
_CWA_QUERIES: Tuple[str, ...] = (
    "mammal",
    "cheetah",
    "swims",
    "black_stripes",
    "tiger",
)


def cell4_2_cwa_vs_owa(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Answer the same query under the closed and the open world assumption.

    Interactive controls (ipywidgets):
    - `fact`: the fact to query
    - `assumption`: which of the two answers the agent acts on

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    fact_dropdown = ipywidgets.Dropdown(
        options=list(_CWA_QUERIES),
        value="mammal",
        description="fact:",
        style={"description_width": "initial"},
    )
    assumption_dropdown = ipywidgets.Dropdown(
        options=["CWA", "OWA"],
        value="CWA",
        description="assumption:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()
    # The fact base is the same for both assumptions, so build it once.
    engine = run_rules(_CWA_FACTS)

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the two answer panels for the current query.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            query = fact_dropdown.value
            assumption = assumption_dropdown.value
            present = query in engine.facts
            # Under the closed world assumption, anything not derivable is
            # taken to be false. Under the open world assumption it is only
            # unknown.
            cwa_answer = "true" if present else "false"
            owa_answer = "true" if present else "unknown"
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            for ax, title, absent_answer in (
                (ax1, "Closed world assumption", "false"),
                (ax2, "Open world assumption", "unknown"),
            ):
                # Both assumptions read the same fact base, and differ only on
                # what they answer for a fact that is not in it.
                rows = []
                colors = []
                for name in _CWA_QUERIES:
                    in_base = name in engine.facts
                    rows.append(
                        [
                            name,
                            "yes" if in_base else "no",
                            "true" if in_base else absent_answer,
                        ]
                    )
                    if name == query:
                        colors.append(_COLOR_RULE_FIRED)
                    else:
                        colors.append(
                            _COLOR_DERIVED if in_base else _COLOR_MISSING
                        )
                is_active = assumption == (
                    "CWA" if absent_answer == "false" else "OWA"
                )
                draw_table(
                    ax,
                    rows,
                    ["fact", "in fact base", "ASK answer"],
                    title="%s%s" % (title, " (active)" if is_active else ""),
                    row_colors=colors,
                    col_widths=[0.4, 0.3, 0.3],
                )
            # Panel 3: comments on the query and the divergence.
            text = (
                "Parameters:\n"
                "  fact: %s\n"
                "  assumption: %s\n\n"
                "Fact base:\n"
                "  observed: %s\n"
                "  derived: %s\n\n"
                "Answers:\n"
                "  in fact base: %s\n"
                "  CWA answer: %s\n"
                "  OWA answer: %s\n"
                "  answers agree: %s\n\n"
                "Answer acted on: %s"
                % (
                    query,
                    assumption,
                    ", ".join(engine.asserted_facts()),
                    ", ".join(engine.derived_facts()),
                    present,
                    cwa_answer,
                    owa_answer,
                    cwa_answer == owa_answer,
                    cwa_answer if assumption == "CWA" else owa_answer,
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "fact": "the fact to query; <code>mammal</code> and "
            "<code>cheetah</code> are derivable, the rest are absent",
            "assumption": "<code>CWA</code> reads an absent fact as false, "
            "<code>OWA</code> reads it as unknown",
        }
    )
    fact_dropdown.observe(update_plot, names="value")
    assumption_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [fact_dropdown, assumption_dropdown],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))


# #############################################################################
# Cell 5.1: Rules vs a learned classifier
# #############################################################################

# The observable features that define each species, taken straight from the
# premises of Winston's rules.
_SPECIES_FEATURES: Dict[str, Tuple[str, ...]] = {
    "cheetah": (
        "has_hair",
        "gives_milk",
        "eats_meat",
        "pointed_teeth",
        "claws",
        "forward_eyes",
        "tawny",
        "dark_spots",
    ),
    "tiger": (
        "has_hair",
        "gives_milk",
        "eats_meat",
        "pointed_teeth",
        "claws",
        "forward_eyes",
        "tawny",
        "black_stripes",
    ),
    "giraffe": (
        "has_hair",
        "gives_milk",
        "hoofs",
        "chews_cud",
        "long_legs",
        "long_neck",
        "tawny",
        "dark_spots",
    ),
    "zebra": (
        "has_hair",
        "gives_milk",
        "hoofs",
        "chews_cud",
        "white",
        "black_stripes",
    ),
    "penguin": (
        "has_feathers",
        "lays_eggs",
        "does_not_fly",
        "swims",
        "black_and_white",
    ),
    "albatross": ("has_feathers", "lays_eggs", "flies", "good_flier"),
}


def make_animal_dataset(n_samples: int, noise: float, seed: int) -> pd.DataFrame:
    """
    Sample animals, then corrupt a fraction of the observed features.

    Noise is applied to the percepts rather than to the labels, because that
    is what a sensor actually gets wrong, and it is what makes the brittle
    conjunctions in the rule base fail.

    :param n_samples: number of animals to sample
    :param noise: probability that any one observed feature is flipped
    :param seed: random seed
    :return: dataframe with one boolean column per observable feature and a
        `species` column, e.g.,

        ```

        has_hair  gives_milk  ...  species
        True      True        ...  cheetah

        ```
    """
    hdbg.dassert_lte(1, n_samples, "Need at least one sample")
    rng = np.random.default_rng(seed)
    species = rng.choice(list(_SPECIES_FEATURES), size=n_samples)
    index = {name: i for i, name in enumerate(OBSERVABLE_FEATURES)}
    rows = np.zeros((n_samples, len(OBSERVABLE_FEATURES)), dtype=bool)
    for row, name in enumerate(species):
        for feature in _SPECIES_FEATURES[name]:
            rows[row, index[feature]] = True
    # Flip each observed bit independently, which is what a noisy sensor does.
    rows ^= rng.random(rows.shape) < noise
    df = pd.DataFrame(rows, columns=pd.Index(OBSERVABLE_FEATURES))
    df["species"] = species
    return df


def rule_engine_predict(row: pd.Series) -> str:
    """
    Identify one animal by running the rule base on its observed features.

    :param row: one row of `make_animal_dataset()`
    :return: species identified, or `unknown` when no rule chain completes
    """
    engine = RuleEngine(ANIMAL_RULES)
    engine.assert_facts([f for f in OBSERVABLE_FEATURES if bool(row[f])])
    engine.run()
    species = engine.species_found()
    return species[0] if species else "unknown"


def _score_models(
    n_samples: int, noise: float, seed: int
) -> Tuple[Dict[str, float], sktree.DecisionTreeClassifier, List[str]]:
    """
    Train a decision tree and score it against the rule engine.

    :param n_samples: size of the generated dataset
    :param noise: probability that any one observed feature is flipped
    :param seed: random seed
    :return: metrics per model, the fitted tree, and the class names
    """
    df = make_animal_dataset(n_samples, noise, seed)
    features = list(OBSERVABLE_FEATURES)
    splits = skmodsel.train_test_split(
        df, test_size=0.25, random_state=seed, stratify=df["species"]
    )
    train_df: pd.DataFrame = splits[0]
    test_df: pd.DataFrame = splits[1]
    # Fit the learned baseline on the same noisy percepts the rules see.
    tree = sktree.DecisionTreeClassifier(max_depth=6, random_state=seed)
    tree.fit(train_df[features], train_df["species"])
    # Time the tree over the whole test set, then per prediction.
    start = time.perf_counter()
    tree_pred = tree.predict(test_df[features])
    tree_latency = (time.perf_counter() - start) / len(test_df) * 1e6
    # Time the rule engine the same way: it re-runs the cycle per animal.
    start = time.perf_counter()
    rule_pred = [rule_engine_predict(row) for _, row in test_df.iterrows()]
    rule_latency = (time.perf_counter() - start) / len(test_df) * 1e6
    metrics = {
        "rule_accuracy": skmetri.accuracy_score(test_df["species"], rule_pred),
        "tree_accuracy": skmetri.accuracy_score(test_df["species"], tree_pred),
        "rule_latency_us": rule_latency,
        "tree_latency_us": tree_latency,
        "rule_abstained": float(np.mean([p == "unknown" for p in rule_pred])),
        "n_train": float(len(train_df)),
        "n_test": float(len(test_df)),
    }
    return metrics, tree, sorted(df["species"].unique())


def cell5_1_rules_vs_learned(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Compare the rule engine against a decision tree on the same animals.

    Interactive controls (ipywidgets):
    - `N`: number of training examples, on a log scale
    - `noise`: probability that any one observed feature is flipped
    - `seed`: random seed

    :param figsize: optional figure size
    """
    if figsize is None:
        figsize = _FIGSIZE
    n_exp_slider, n_box = htutori.build_log_widget_control(
        name="log(N)",
        description="N (dataset size)",
        min_exp=6,
        max_exp=12,
        initial_exp=9,
        base=2,
    )
    noise_slider, noise_box = htutori.build_widget_control(
        name="noise",
        description="probability a percept is flipped",
        min_val=0.0,
        max_val=0.3,
        step=0.02,
        initial_value=0.0,
        is_float=True,
    )
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=100,
        step=1,
        initial_value=42,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Retrain the tree and redraw the comparison for the current settings.

        :param change: widget change record (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            n_samples = 2**n_exp_slider.value
            noise = noise_slider.value
            seed = seed_slider.value
            metrics, tree, class_names = _score_models(n_samples, noise, seed)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: accuracy and latency, side by side.
            rows = [
                [
                    "accuracy",
                    "%.3f" % metrics["rule_accuracy"],
                    "%.3f" % metrics["tree_accuracy"],
                ],
                [
                    "latency (us)",
                    "%.1f" % metrics["rule_latency_us"],
                    "%.1f" % metrics["tree_latency_us"],
                ],
                ["audit trail", "rule chain", "split thresholds"],
                ["abstains", "%.2f" % metrics["rule_abstained"], "0.00"],
            ]
            draw_table(
                ax1,
                rows,
                ["metric", "rule engine", "decision tree"],
                title="Rule engine vs decision tree",
                row_colors=[_COLOR_MISSING] * len(rows),
                col_widths=[0.32, 0.34, 0.34],
            )
            # Panel 2: the top of the learned tree, for visual comparison
            # with the rule dependency graph of Cell 2.1.
            sktree.plot_tree(
                tree,
                max_depth=2,
                feature_names=list(OBSERVABLE_FEATURES),
                class_names=class_names,
                filled=True,
                impurity=False,
                fontsize=6,
                ax=ax2,
            )
            ax2.set_title(
                "Learned decision tree (top 2 levels)",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )
            # Panel 3: comments on the current dataset and scores.
            text = (
                "Parameters:\n"
                "  N (dataset): %d\n"
                "  noise: %.2f\n"
                "  seed: %d\n\n"
                "Split:\n"
                "  train: %d\n"
                "  test: %d\n\n"
                "Accuracy:\n"
                "  rule engine: %.3f\n"
                "  decision tree: %.3f\n"
                "  rule engine abstains: %.2f\n\n"
                "Latency per animal:\n"
                "  rule engine: %.1f us\n"
                "  decision tree: %.1f us"
                % (
                    n_samples,
                    noise,
                    seed,
                    int(metrics["n_train"]),
                    int(metrics["n_test"]),
                    metrics["rule_accuracy"],
                    metrics["tree_accuracy"],
                    metrics["rule_abstained"],
                    metrics["rule_latency_us"],
                    metrics["tree_latency_us"],
                )
            )
            comment_panel(ax3, text)
            plt.tight_layout()
            plt.show()

    param_info = make_param_info(
        {
            "N": "number of animals sampled; a quarter is held out for the "
            "test set both models are scored on",
            "noise": "probability that any one observed feature is flipped, "
            "which is what breaks the conjunctions in the rule base",
            "seed": "random seed for the sample and the tree",
        }
    )
    n_exp_slider.observe(update_plot, names="value")
    noise_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    controls = ipywidgets.VBox(
        [n_box, noise_box, seed_box],
        layout=ipywidgets.Layout(padding="0px 8px 0px 0px"),
    )
    top_row = ipywidgets.HBox([controls, param_info])
    display(ipywidgets.VBox([top_row, output]))
