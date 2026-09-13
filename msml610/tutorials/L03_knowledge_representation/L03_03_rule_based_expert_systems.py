# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MYCIN Redux: A Rule-Based Expert System You Can Debug
#
# - This notebook implements the match-conflict resolution-act cycle from
#   scratch, using Winston's classic animal-identification rules, then
#   contrasts the result with a learned classifier
# - The pedagogical arc:
#   - Reflex agent vs rule-based agent with working memory
#   - Forward chaining (data-driven) vs backward chaining (goal-driven)
#   - Conflict resolution strategies and their effect on the derived facts
#   - Explanation and certainty factors
#   - Non-monotonic default reasoning
#   - Closed world vs open world assumption
#   - Rule engine vs learned classifier on explainability and accuracy
# - The engine lives in `L03_03_rule_based_expert_systems_utils.py` and is
#   written from scratch, so that every step of the recognize-act cycle is
#   visible and can be single-stepped

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
# !pip install -q networkx==3.6.1 scikit-learn==1.9.1

import networkx
print("networkx version: ", networkx.__version__)
import sklearn
print("sklearn version: ", sklearn.__version__)

# %%
import helpers.hnotebook as hnotebook

import L03_03_rule_based_expert_systems_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# %%
# Explain Winston's animal identification rules
#
# Patrick Winston's classic animal-identification rule base is a canonical
# example in AI education (Winston, 1977). It uses observable features of
# animals to derive increasingly specific classifications through a chain of
# inference rules. The rules demonstrate both forward chaining (data-driven)
# and how to organize knowledge hierarchically.
#
# Observable features (what an agent can perceive directly):
print("Observable features (percepts):")
print(f"  {', '.join(utils.OBSERVABLE_FEATURES)}")
print()
# Derived facts (what the rules conclude):
print("Derived facts (classes and species):")
print(f"  Classes: mammal, bird, carnivore, ungulate")
print(f"  Species: {', '.join(utils.SPECIES)}")
print()

# %% [markdown]
# ## The Rule Base
#
# - Every cell below reasons over Winston's animal identification rules
# - The rules chain in three layers:
#   - Observable features produce a class, e.g.,
#     `has_hair` $\to$ `mammal`
#   - A class plus more features produces a sub-class, e.g.,
#     `mammal` $\land$ `eats_meat` $\to$ `carnivore`
#   - A sub-class plus more features produces a species, e.g.,
#     `carnivore` $\land$ `tawny` $\land$ `dark_spots` $\to$ `cheetah`

# %%
# Show the rules the engine fires, in rule base order.
utils.show_rule_base()
# Outcome: 14 rules, whose premises are either observable features or facts
# concluded by an earlier rule.

# %% [markdown]
# # Part 1: Rules and Working Memory

# %% [markdown]
# ## Cell 1.1: Rules, working memory, and the reflex agent baseline
#
# **Goal**:
# - Contrast a simple reflex agent (percept only) with a rule-based agent
#   (percept plus working memory)
# - Introduce Winston's animal-identification rules as the running example
#
# **Explanation of Widget**
# - _Agent diagrams_: the reflex pipeline (percept $\to$ action) and the
#   rule-based pipeline (percept $\to$ working memory $\to$ match $\to$ fire
#   $\to$ act), with the active agent highlighted
# - _Working memory_: current fact table, showing which facts are observed,
#   which are derived, and which rule derived them
# - _Comments_: current agent type, facts currently true, action produced
#
# **Parameters**:
# - `agent`: `reflex` or `rule-based`
# - `has_hair`, `eats_meat`, `tawny`, `dark_spots`: one checkbox per fact

# %%
# Toggle the agent type and the percepts, and compare what each agent reaches.
utils.cell1_1_reflex_vs_rule_based()

# %% [markdown]
# **Key observations**:
# - The reflex agent can only react to a single percept and never reaches
#   `cheetah`, since that conclusion requires chaining `mammal` and
#   `carnivore` first
# - The rule-based agent's working memory is what makes multi-step derivation
#   possible at all
# - Adding one fact at a time shows exactly which intermediate conclusion
#   becomes available next
#
# - Try turning on `has_hair` alone: the rule-based agent reaches `mammal` and
#   stops, because no further rule has all its premises
# - Try turning on all four percepts: only then does the third layer of rules
#   become reachable

# %% [markdown]
# # Part 2: Forward and Backward Chaining

# %% [markdown]
# ## Cell 2.1: Forward chaining: watching working memory fill up
#
# **Goal**:
# - Animate forward (data-driven) chaining over Winston's rules, one fired
#   rule per frame
#
# **Explanation of Widget**
# - _Working memory timeline_: the facts held after each fired rule, one column
#   per step, starting from the initial percepts
# - _Rule dependency graph_: the premise $\to$ rule $\to$ conclusion graph, with
#   the just-fired rule and its edges highlighted in orange
# - _Comments_: step number, rule fired this step, new fact added, conflict set
#
# **Parameters**:
# - `fire next rule`: run exactly one recognize-act cycle
# - `run to fixed point`: fire until no rule matches
# - `initial facts`: the percepts working memory starts with

# %%
# Fire one rule at a time and watch the new fact enable the next rule.
utils.cell2_1_forward_chaining()

# %% [markdown]
# **Key observations**:
# - Each fired rule's conclusion can become the premise of the next rule,
#   chaining `has_hair` $\to$ `mammal` $\to$ `carnivore` $\to$ `cheetah`
# - Forward chaining halts at a fixed point: it stops the moment no rule's
#   premises are fully satisfied by the current working memory
# - Forward chaining derives every consequence of the facts, whether or not
#   any of them was asked for
#
# - Try switching `black_stripes` on together with `dark_spots`: both `R9` and
#   `R10` become ready, so the conflict set grows to two and the engine has to
#   choose, which is exactly the problem Cell 2.3 attacks
# - Firing the same rule twice is blocked by refraction: once a conclusion is
#   in working memory, its rule no longer matches

# %% [markdown]
# ## Cell 2.2: Backward chaining: the AND-OR proof tree for a goal
#
# **Goal**:
# - Run goal-driven backward chaining on a goal such as `cheetah` and contrast
#   which rules it explores against forward chaining on the same facts
#
# **Explanation of Widget**
# - _AND-OR tree_: the tree rooted at the goal, where a fact node is an OR node
#   (any rule below it suffices) and a rule node is an AND node (every premise
#   below it is required); green is proved, red is failed
# - _Rules-fired counter_: rules explored by forward vs backward chaining for the
#   same query
# - _Comments_: current goal, facts known, rules explored so far
#
# **Parameters**:
# - `goal`: one of `cheetah`, `tiger`, `carnivore`, `mammal`, `zebra`

# %%
# Prove a goal backward and compare the effort against forward chaining.
utils.cell2_2_backward_chaining()

# %% [markdown]
# **Key observations**:
# - Backward chaining only explores rules that could plausibly support the
#   chosen goal, ignoring rules irrelevant to it: with goal `mammal` it
#   expands 2 rules where forward chaining fires 3
# - Forward chaining derives every consequence of the facts, whether or not it
#   is useful for a specific goal
# - The rules-fired counter makes the cost trade-off concrete: fewer rules
#   explored does not always mean less work overall, and with goal `cheetah`
#   backward chaining actually expands more rules, since it re-derives
#   `mammal` once per candidate rule for `carnivore`
#
# - Try goal `zebra`: the whole subtree fails in red, because `ungulate` is
#   not derivable from these percepts
# - Try goal `tiger`: the `black_stripes` leaf fails, and one failed AND child
#   is enough to fail the rule above it, even though `carnivore` was proved

# %% [markdown]
# ## Cell 2.3: Conflict resolution: specificity, recency, and priority
#
# **Goal**:
# - Show that when multiple rules match the same working memory, the strategy
#   used to pick one changes the final derived fact set
# - The rule base here adds a leopard rule that conflicts with the cheetah
#   rule: each blocks the other with a negative premise, so exactly one of the
#   two can ever fire
#
# **Explanation of Widget**
# - _Conflict set_: all rules whose premises currently match, with the
#   strategy-selected rule highlighted in orange
# - _Final fact set_: the facts derived per strategy, run to completion, each
#   bar labelled with the species that strategy reaches
# - _Comments_: current strategy, size of conflict set, rule selected this step
#
# **Parameters**:
# - `strategy`: `specificity`, `recency`, or `priority`
# - `prio`: salience of the cheetah rule `R9`, read by `priority`

# %%
# Switch strategy and watch the same working memory reach a different species.
utils.cell2_3_conflict_resolution()

# %% [markdown]
# **Key observations**:
# - The same working memory can produce different final fact sets depending on
#   which conflicting rule fires first: `specificity` reaches `leopard` and
#   `recency` reaches `cheetah`
# - Specificity favors the most detailed matching rule, recency favors the rule
#   matching the newest fact, priority is set explicitly by the rule author
# - The deliberately contradictory rule pair exposes the weak point of each
#   strategy: none of them is more correct than the others, they simply encode
#   different assumptions about which evidence should dominate
#
# - Try raising `prio` above 5 under the `priority` strategy: the cheetah rule
#   overtakes the leopard rule and the conclusion flips
# - The conflict set is where the engine stops being a logic and starts being
#   a program: the rules alone do not determine the answer

# %% [markdown]
# # Part 3: Explanation and Uncertainty

# %% [markdown]
# ## Cell 3.1: Explaining conclusions and certainty factors
#
# **Goal**:
# - Answer MYCIN's two explanation questions for any conclusion, then extend
#   crisp rules with certainty factors and compare the resulting diagnosis
#   ranking
#
# **Explanation of Widget**
# - _Explanation trace_: the rules and facts behind the chosen conclusion, with
#   the conclusion itself in purple
# - _Diagnosis ranking_: the candidate diagnoses ranked by propagated certainty
#   factor
# - _Comments_: chosen conclusion, "how" and "why" answers, certainty values
#
# **Parameters**:
# - `conclusion`: the fact to explain
# - `cf1`, `cf2`, `cf3`: certainty factors of `R1`, `R5`, and `R9`, in
#   $[0, 1]$
#
# The certainty of a conclusion is the certainty of the rule times the
# certainty of its weakest premise:
# $$CF(c) = CF(\text{rule}) \cdot \min_i CF(p_i)$$
# and two rules supporting the same conclusion combine as
# $$CF = CF_a + CF_b (1 - CF_a)$$

# %%
# Trace a conclusion back to its rules, then rank the competing diagnoses.
utils.cell3_1_explanation_and_certainty()

# %% [markdown]
# **Key observations**:
# - Every conclusion has a full, inspectable chain of rules behind it, which is
#   the core explainability advantage of symbolic systems
# - Certainty factors reorder the ranking of candidate diagnoses without
#   changing which crisp conclusions are logically derivable: all three
#   species stay in working memory at every slider setting
# - Small changes to one rule's certainty factor can flip which diagnosis
#   ranks first
#
# - Try raising `cf3` above $0.75$: `cheetah` overtakes `tiger`, because the
#   tiger rule carries a certainty of $0.75$
# - Try lowering `cf1` or `cf2`: every diagnosis drops by the same factor and
#   the ranking never changes, since both rules sit on the shared part of the
#   chain
# - Certainty multiplies along a chain, so a long derivation is always less
#   certain than a short one

# %% [markdown]
# # Part 4: Non-Classical Reasoning

# %% [markdown]
# ## Cell 4.1: Non-monotonic reasoning: Tweety the penguin
#
# **Goal**:
# - Run the classic default-reasoning case: a default rule concludes Tweety
#   flies, then a new fact retracts that conclusion
# - The default rule is `bird` $\land$ `~abnormal` $\to$ `flies`, where
#   `~abnormal` holds as long as nothing marks the bird as abnormal
#
# **Explanation of Widget**
# - _Before panel_: working memory in the initial scenario, where only
#   "Tweety is a bird" is known
# - _After panel_: working memory once the new fact is asserted, with the
#   conclusions that lost their support marked in red as retracted
# - _Comments_: facts currently asserted, whether "Tweety flies" currently holds
#
# **Parameters**:
# - `Tweety is a penguin`: assert the new fact and re-derive
# - `reset`: restore the initial scenario

# %%
# Assert one new fact and watch an earlier conclusion disappear.
utils.cell4_1_non_monotonic()

# %% [markdown]
# **Key observations**:
# - Classical logical conclusions only accumulate: adding a fact never removes
#   an earlier conclusion
# - The default rule "birds fly" behaves differently: one new fact overturns an
#   earlier, previously justified conclusion
# - This is exactly what "non-monotonic" means: the set of held conclusions can
#   shrink as well as grow
#
# - The retraction cascades: `reaches_tree_nest` was derived from `flies`, so
#   it disappears too, even though no rule mentions penguins
# - Retracting requires provenance: the engine can only remove `flies` because
#   it recorded which rule produced it and can check whether that rule still
#   matches

# %% [markdown]
# ## Cell 4.2: Closed world vs open world on the same query
#
# **Goal**:
# - Query a fact that is simply absent from working memory under the closed
#   world assumption and under the open world assumption, side by side
#
# **Explanation of Widget**
# - _CWA panel_: the fact base, with the answer each fact gets when "not told"
#   is read as "false"
# - _OWA panel_: the same fact base, with the answer each fact gets when
#   "not told" is read as "unknown"
# - _Comments_: fact queried, assumption active, answer produced
#
# **Parameters**:
# - `fact`: the fact to query, present or absent
# - `assumption`: `CWA` or `OWA`

# %%
# Ask the same question of the same fact base under both assumptions.
utils.cell4_2_cwa_vs_owa()

# %% [markdown]
# **Key observations**:
# - Identical fact base and identical query, yet the two assumptions produce
#   different answers whenever the fact is absent
# - CWA and OWA agree whenever the fact is actually asserted; they only diverge
#   on missing information
# - Choosing CWA implicitly treats "not told" as "false", which is convenient
#   but not always correct
#
# - Try querying `tiger`: under CWA the agent concludes the animal is not a
#   tiger, when all it really knows is that it never checked for stripes
# - The negative premises used by the rules in Cell 2.3 and Cell 4.1 are only
#   meaningful under CWA: `~leopard` means "leopard was not derived", not
#   "leopard is false in the world"

# %% [markdown]
# # Part 5: Rules vs Learning

# %% [markdown]
# ## Cell 5.1: Rules vs learned classifier: explainability vs accuracy
#
# **Goal**:
# - Compare the hand-built rule engine against a decision tree trained on the
#   same animal dataset, on accuracy, latency, and auditability
# - Noise is applied to the percepts rather than to the labels, since that is
#   what a real sensor gets wrong, and it is what breaks the conjunctions the
#   rules rely on
#
# **Explanation of Widget**
# - _Comparison table_: accuracy, latency, auditability, and abstention rate for
#   the rule engine vs the decision tree
# - _Decision tree diagram_: the top of the trained tree, for visual comparison
#   with the rule dependency graph of Cell 2.1
# - _Comments_: dataset size, noise level, current accuracy for each model
#
# **Parameters**:
# - `N`: number of sampled animals, on a $\log_2$ scale
# - `noise`: probability that any one observed feature is flipped, in
#   $[0, 0.3]$
# - `seed`: random seed

# %%
# Train a decision tree on the same animals the rule engine identifies.
utils.cell5_1_rules_vs_learned()

# %% [markdown]
# **Key observations**:
# - The learned classifier can match or exceed the rule engine's accuracy,
#   especially as noise increases: at `noise` $= 0.1$ the tree scores about
#   $0.84$ against the rule engine's $0.70$
# - The rule engine's every decision traces to an explicit rule; the decision
#   tree's split thresholds are harder for a domain expert to audit directly
# - The trade-off is not fixed: as dataset size and noise change, which system
#   looks more attractive changes too
#
# - Try `noise` $= 0$: both reach perfect accuracy, and the only difference
#   left is latency, where the tree is about 20 times faster per animal
# - Try a small `N` with noise on: the rule engine wins, because it needs no
#   training data at all, while the tree has too few examples
# - The rule engine fails by abstaining, not by guessing: when a conjunction
#   breaks, no species rule fires and it answers `unknown`, which is a very
#   different failure mode from a confident wrong label

# %% [markdown]
# # Summary: The Mental Model
#
# - A production system holds facts in working memory and rules in a rule
#   base, and runs a recognize-act cycle: match every rule, resolve the
#   conflict set down to one rule, fire it, repeat until a fixed point
# - Forward chaining is data-driven and derives every consequence of the
#   facts; backward chaining is goal-driven and explores only the rules that
#   could support the query, but neither is uniformly cheaper
# - When several rules match, the rules alone do not determine the answer: the
#   conflict resolution strategy does, and specificity, recency, and priority
#   can each reach a different conclusion from identical facts
# - Explainability is structural, not an add-on: because each fact records the
#   rule that produced it, the engine can answer "how" for any conclusion and
#   "why" for any subgoal
# - Certainty factors rank competing conclusions without changing which ones
#   are derivable, and certainty multiplies along a chain
# - Default rules make reasoning non-monotonic: a new fact can remove an
#   earlier conclusion, and the retraction cascades to everything derived from
#   it
# - The closed world assumption reads "not told" as "false" and is what makes
#   negative premises meaningful; the open world assumption answers "unknown"
#   instead
# - Against a learned classifier, the rule engine trades accuracy under noise
#   for an auditable trace and for needing no training data
