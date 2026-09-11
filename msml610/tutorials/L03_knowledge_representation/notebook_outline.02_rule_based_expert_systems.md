# MYCIN Redux: A Rule-Based Expert System You Can Debug

- Source: `Lesson03.1-Knowledge_representation.smd`,
  `Lesson03.3-Non_classical_logics.smd`
- This notebook implements the match-conflict-resolution-act cycle from scratch using
  Winston's classic animal-identification rules, then contrasts it with a learned
  classifier
- The pedagogical arc:
  - Reflex agent vs rule-based agent with working memory
  - Forward chaining (data-driven) vs backward chaining (goal-driven)
  - Conflict resolution strategies and their effect on the derived facts
  - Explanation and certainty factors
  - Non-monotonic default reasoning
  - Closed world vs open world assumption
  - Rule engine vs learned classifier on explainability and accuracy

## Cell 1: Rules, Working Memory, and the Reflex Agent Baseline

**Goal**:
- Contrast a simple reflex agent (percept only) with a rule-based agent (percept plus
  working memory)
- Introduce Winston's animal-identification rules as the running example
**Plots and their descriptions**:
- _Agent diagrams_: two flow diagrams side by side, reflex agent (percept $\to$
  action) and rule-based agent (percept $\to$ working memory $\to$ match $\to$ fire
  $\to$ act)
- _Working memory_: current fact table (`has_hair`, `eats_meat`, `tawny`,
  `dark_spots`)
- _Comments_: current agent type, facts currently true
**Widgets**:
- `agent_type`: toggle between "reflex" and "rule-based"
- `has_hair`, `eats_meat`, `tawny`, `dark_spots`: checkboxes for each fact
**Key observations** (post-visualization):
- The reflex agent can only react to a single percept and never reaches `cheetah`,
  since that conclusion requires chaining `mammal` and `carnivore` first
- The rule-based agent's working memory is what makes multi-step derivation possible
  at all
- Adding one fact at a time shows exactly which intermediate conclusion becomes
  available next
**Implementation**: `experta` for the rule engine, `matplotlib` flow diagrams,
`ipywidgets.Checkbox` for facts

## Cell 2: Forward Chaining: Watching Working Memory Fill Up

**Goal**:
- Animate forward (data-driven) chaining over Winston's rules, one fired rule per
  frame
**Plots and their descriptions**:
- _Working memory timeline_: table of facts after each fired rule, one column per
  step
- _Rule dependency graph_: `graphviz` diagram with the just-fired rule highlighted
- _Comments_: step number, rule fired this step, new fact added
**Widgets**:
- `step`: button to fire the next matching rule
- `initial_facts`: checkboxes for the starting fact set
**Key observations** (post-visualization):
- Each fired rule's conclusion can become the premise of the next rule, chaining
  `has_hair` -> `mammal` -> `carnivore` -> `cheetah`
- Forward chaining halts at a fixed point: it stops the moment no rule's premises are
  fully satisfied by the current working memory
- Adding a rule that contradicts an existing one causes the engine to loop or produce
  inconsistent facts, which forward chaining alone cannot detect
**Implementation**: `experta.KnowledgeEngine` for forward chaining,
`graphviz.Digraph` for the dependency graph

## Cell 3: Backward Chaining: the AND-OR Proof Tree for a Goal

**Goal**:
- Run goal-driven backward chaining on the goal `cheetah` and contrast which rules it
  explores against forward chaining on the same facts
**Plots and their descriptions**:
- _AND-OR tree_: `graphviz` tree rooted at the goal, expanding subgoals as AND/OR
  nodes
- _Rules-fired counter_: bar chart comparing rules explored by forward vs backward
  chaining for the same query
- _Comments_: current goal, facts known, rules explored so far
**Widgets**:
- `goal`: dropdown for "cheetah", "carnivore", "mammal"
**Key observations** (post-visualization):
- Backward chaining only explores rules that could plausibly support the chosen goal,
  ignoring rules irrelevant to it
- Forward chaining derives every consequence of the facts, whether or not it is
  useful for a specific goal
- The rules-fired counter makes the cost trade-off concrete: fewer rules explored
  does not always mean less work overall
**Implementation**: `experta` backward-chaining pattern (goal rules),
`graphviz.Digraph` for the AND-OR tree

## Cell 4: Conflict Resolution: Specificity, Recency, and Priority

**Goal**:
- Show that when multiple rules match the same working memory, the strategy used to
  pick one changes the final derived fact set
**Plots and their descriptions**:
- _Conflict set_: table of all rules whose premises currently match, with the
  strategy-selected rule highlighted
- _Final fact set_: bar chart of derived facts per strategy, run to completion
- _Comments_: current strategy, size of conflict set, rule selected this step
**Widgets**:
- `strategy`: dropdown for "specificity", "recency", "priority"
- `rule_order`: slider to reorder rule priority when "priority" is selected
**Key observations** (post-visualization):
- The same working memory can produce different final fact sets depending on which
  conflicting rule fires first
- Specificity favors the most detailed matching rule, recency favors the rule
  matching the newest fact, priority is set explicitly by the rule author
- A deliberately contradictory rule pair exposes each strategy's weak point: some
  strategies resolve the conflict cleanly, others loop or leave inconsistent facts in
  place
**Implementation**: `experta` salience/priority mechanism, `pandas.DataFrame` for the
conflict set table

## Cell 5: Explaining Conclusions and Certainty Factors

**Goal**:
- Answer MYCIN's two explanation questions for any conclusion, then extend crisp
  rules with certainty factors and compare the resulting diagnosis ranking
**Plots and their descriptions**:
- _Explanation trace_: `graphviz` tree tracing a chosen conclusion back to the rules
  that fired
- _Diagnosis ranking_: bar chart of candidate diagnoses ranked by propagated
  certainty factor
- _Comments_: chosen conclusion, "how" and "why" answers, certainty factor values
**Widgets**:
- `conclusion`: dropdown to pick a conclusion to explain
- `cf_rule_1`, `cf_rule_2`, `cf_rule_3`: sliders for each rule's certainty factor
  (0.0-1.0)
**Key observations** (post-visualization):
- Every conclusion has a full, inspectable chain of rules behind it, which is the
  core explainability advantage of symbolic systems
- Certainty factors reorder the ranking of candidate diagnoses without changing which
  crisp conclusions are logically derivable
- Small changes to one rule's certainty factor can flip which diagnosis ranks first
**Implementation**: `experta` fact provenance for the trace, custom certainty factor
propagation in Python, `graphviz.Digraph`

## Cell 6: Non-Monotonic Reasoning: Tweety the Penguin

**Goal**:
- Run the classic default-reasoning case: a default rule concludes Tweety flies, then
  a new fact retracts that conclusion
**Plots and their descriptions**:
- _Working memory timeline_: before/after panels, with the retracted conclusion shown
  crossed out after the new fact is added
- _Comments_: facts currently asserted, whether "Tweety flies" currently holds
**Widgets**:
- `add_penguin_fact`: button to assert "Tweety is a penguin"
- `reset`: button to restore the initial scenario (default rule only)
**Key observations** (post-visualization):
- Classical logical conclusions only accumulate: adding a fact never removes an
  earlier conclusion
- The default rule "birds fly" behaves differently: one new fact overturns an
  earlier, previously justified conclusion
- This is exactly what "non-monotonic" means: the set of held conclusions can shrink
  as well as grow
**Implementation**: `experta` fact retraction (`self.retract`), `matplotlib` timeline
panel

## Cell 7: Closed World vs Open World on the Same Query

**Goal**:
- Query a fact that is simply absent from the working memory under the closed world
  assumption and under the open world assumption, side by side
**Plots and their descriptions**:
- _CWA panel_: fact base plus the query answer "false" for an absent fact
- _OWA panel_: same fact base plus the query answer "unknown" for the same absent
  fact
- _Comments_: fact queried, assumption active, answer produced
**Widgets**:
- `fact`: dropdown to pick a fact to query (present or absent)
- `assumption`: toggle between "CWA" and "OWA"
**Key observations** (post-visualization):
- Identical fact base and identical query, yet the two assumptions produce different
  answers whenever the fact is absent
- CWA and OWA agree whenever the fact is actually asserted; they only diverge on
  missing information
- Choosing CWA implicitly treats "not told" as "false", which is convenient but not
  always correct
**Implementation**: `experta` for the CWA query, a small custom OWA checker that
returns "unknown" for unasserted facts

## Cell 8: Rules vs Learned Classifier: Explainability vs Accuracy

**Goal**:
- Compare the hand-built rule engine against a decision tree trained on the same
  symptom dataset, on accuracy, latency, and auditability
**Plots and their descriptions**:
- _Comparison table_: accuracy and latency for the rule engine vs the decision tree
- _Decision tree diagram_: trained tree structure next to the rule dependency graph
  from Cell 2, for visual comparison
- _Comments_: dataset size, noise level, current accuracy for each model
**Widgets**:
- `dataset_size`: log-scale slider for number of training examples
- `noise_level`: slider for label noise (0.0-0.3)
**Key observations** (post-visualization):
- The learned classifier can match or exceed the rule engine's accuracy, especially
  as noise increases
- The rule engine's every decision traces to an explicit rule; the decision tree's
  split thresholds are harder for a domain expert to audit directly
- The trade-off is not fixed: as dataset size and noise change, which system looks
  more attractive changes too
**Implementation**: `scikit-learn.tree.DecisionTreeClassifier` for the learned
baseline, `graphviz` export via `sklearn.tree.export_graphviz`