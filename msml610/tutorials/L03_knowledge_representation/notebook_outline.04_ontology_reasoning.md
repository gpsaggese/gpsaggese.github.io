# Ontology Lab: Pizzas, Penguins, and a Reasoner

- Source: `Lesson03.1-Knowledge_representation.smd`,
  `Lesson03.3-Non_classical_logics.smd`
- This notebook builds an ontology from classes, individuals, properties, and axioms,
  then lets a description-logic reasoner classify and query it
- The pedagogical arc:
  - Ontology vs database schema vs taxonomy vs knowledge base
  - Subsumption and inconsistency on the Manchester Pizza ontology
  - Asserted hierarchy vs inferred hierarchy
  - Interactive axiom editing
  - Unsatisfiable concepts
  - Instance-level reasoning and the open world assumption
  - Expressiveness vs tractability in reasoner runtime

## Cell 1: Building the University Ontology

**Goal**:
- Construct the lecture's university ontology in code: classes, properties, and one
  cardinality axiom
- Distinguish an ontology from a plain database schema, a taxonomy, and a knowledge
  base
**Plots and their descriptions**:
- _Class hierarchy_: interactive `ipycytoscape` diagram of `Student`, `Professor`,
  `Course`, `Department`, connected by `takesCourse`, `teachesCourse`,
  `belongsToDepartment`
- _Comparison table_: ontology vs database schema vs taxonomy vs knowledge base, one
  row per property (has axioms, supports inference, has instances)
- _Comments_: number of classes, properties, and axioms currently defined
**Widgets**: none for this reference cell
**Key observations** (post-visualization):
- A database schema constrains data shape but cannot derive new facts; the ontology's
  axioms let a reasoner do exactly that
- A taxonomy is only the class hierarchy; the ontology adds properties and axioms on
  top of it
- The cardinality axiom "every `Course` is taught by exactly one `Professor`" is a
  constraint a plain schema could enforce, but only the ontology can use it to draw
  new conclusions
**Implementation**: `owlready2` for classes, properties, and the cardinality axiom,
`ipycytoscape.CytoscapeWidget` for the hierarchy diagram

## Cell 2: the Pizza Ontology: Subsumption and an Inconsistent Vegetarian

**Goal**:
- Load the Manchester Pizza ontology and run class-level reasoning
- Ask the reasoner to classify a `VegetarianPizza` that lists a meat topping and read
  its explanation
**Plots and their descriptions**:
- _Class hierarchy_: Pizza ontology class tree with the tested class highlighted, red
  if the reasoner marks it inconsistent
- _Explanation panel_: the specific axioms in conflict, printed as short logical
  statements
- _Comments_: class tested, reasoner verdict (consistent/inconsistent)
**Widgets**:
- `pizza_class`: dropdown to pick which pizza class to test (including one
  deliberately misclassified with a meat topping)
**Key observations** (post-visualization):
- Subsumption ("is `A` more general than `B`?") is answered structurally by the
  reasoner from the axioms, never by reading class names
- An inconsistency is a proof that no individual can satisfy every asserted axiom on
  that class at once
- The explanation panel names the exact axioms responsible, not just the final
  "inconsistent" verdict
**Implementation**: `owlready2.get_ontology()` loading the Pizza ontology `.owl`
file, `owlready2.sync_reasoner()` (HermiT), `owlready2` inconsistency-explanation API

## Cell 3: Asserted Hierarchy vs Inferred Hierarchy

**Goal**:
- Compare the class hierarchy as authored against the hierarchy after classification,
  on the same ontology
**Plots and their descriptions**:
- _Diff graph_: `networkx` graph with asserted edges in one color and newly inferred
  edges in a second color
- _Comments_: number of asserted edges, number of newly inferred edges
**Widgets**:
- `view`: toggle between "asserted only", "inferred only", "both"
**Key observations** (post-visualization):
- Classification can discover subclass relationships the author never wrote
  explicitly
- The gap between the asserted graph and the inferred graph is exactly the value the
  reasoner adds
- An ontology with no gap between the two graphs is one where the author already
  anticipated every consequence of their own axioms
**Implementation**: `owlready2` for asserted vs inferred `is_a` relations,
`networkx.difference()` for the diff, `matplotlib` two-color graph rendering

## Cell 4: Interactive Axiom Editor

**Goal**:
- Let students add or remove one axiom and re-run the reasoner live, watching which
  edges the hierarchy gains or loses
**Plots and their descriptions**:
- _Class hierarchy_: same diagram as Cell 3, redrawn after each edit with newly
  inferred edges highlighted
- _Comments_: axiom just applied, number of hierarchy edges before and after
**Widgets**:
- `axiom_template`: dropdown for a disjointness axiom or a cardinality restriction to
  add or remove
- `apply`: button to apply the selected axiom and re-run the reasoner
**Key observations** (post-visualization):
- One added axiom can ripple into many new inferred edges elsewhere in the hierarchy
- Some axioms make a previously satisfiable class unsatisfiable, which the diagram
  shows immediately
- Small, deliberate edits make the reasoner's behavior predictable instead of a black
  box
**Implementation**: `owlready2` dynamic axiom addition/removal,
`owlready2.sync_reasoner()` re-invoked on each edit

## Cell 5: Unsatisfiable Concepts: the Flying Penguin

**Goal**:
- Define `FlyingPenguin` as `Penguin` and `Flying`, and confirm the reasoner marks
  the concept unsatisfiable
**Plots and their descriptions**:
- _Class hierarchy_: `FlyingPenguin` node rendered empty/crossed out
- _Explanation panel_: the two conflicting axioms (`Penguin`
  $\sqsubseteq \lnot Flying$ and `FlyingPenguin` $\sqsubseteq Flying \sqcap Penguin$)
- _Comments_: whether `FlyingPenguin` is currently satisfiable
**Widgets**:
- `penguins_cannot_fly`: toggle to add or remove the "penguin cannot fly" axiom
**Key observations** (post-visualization):
- Satisfiability is a per-class question, answerable before anyone tries to create an
  individual of that class
- An unsatisfiable class can never have members, no matter how the ontology is
  populated later
- Removing the "penguin cannot fly" axiom flips `FlyingPenguin` back to satisfiable,
  showing the axiom is exactly what caused the conflict
**Implementation**: `owlready2` class disjointness declaration,
`owlready2.sync_reasoner()` unsatisfiable-class report

## Cell 6: Instance-Level Reasoning: Realization, Retrieval, and Open World

**Goal**:
- Perform realization (most specific class of an individual) and retrieval
  (individuals satisfying a class), then demonstrate the open world assumption
**Plots and their descriptions**:
- _Realization panel_: individual `GP` with its full inferred class chain highlighted
  in the hierarchy
- _Retrieval table_: individuals satisfying the chosen class query
- _OWA panel_: query for an absent fact, returning "unknown" rather than "false"
- _Comments_: individual/class currently queried, result
**Widgets**:
- `individual`: dropdown to pick an individual for realization
- `query_class`: dropdown to pick a class for retrieval (e.g., `TeachingAssistant`)
- `query_absent_fact`: button to query a fact that was never asserted
**Key observations** (post-visualization):
- Realization narrows an individual down to its most specific known class or classes,
  not just its asserted type
- Retrieval lists every individual the reasoner can prove satisfies the class,
  including ones never explicitly typed that way
- Under the open world assumption, an absent fact returns "unknown", never a hard
  "false": absence of information is not evidence of falsehood
**Implementation**: `owlready2` instance reasoning (`instance.is_a` after
`sync_reasoner()`), custom OWA query wrapper

## Cell 7: Expressiveness vs Tractability: Reasoner Runtime as Axioms Grow

**Goal**:
- Measure how reasoner runtime grows as cardinality constraints and property chains
  are added to the ontology
**Plots and their descriptions**:
- _Runtime curve_: reasoning time vs number of added expressive constructs
  (cardinality constraints, then a property chain)
- _Comments_: current construct count, construct type, measured runtime
**Widgets**:
- `n_cardinality_constraints`: slider for how many cardinality constraints are added
  (0-10)
- `add_property_chain`: toggle to add one property-chain axiom
**Key observations** (post-visualization):
- Runtime grows noticeably once cardinality constraints and property chains are
  introduced, compared to the plain class hierarchy in Cell 1
- This is the same expressiveness vs tractability tension raised for propositional vs
  first-order logic, now visible in a description-logic reasoner
- Richer constructs buy more inferential power at a measurable runtime cost
**Implementation**: `owlready2` cardinality restrictions and property chains,
`time.perf_counter()` around `owlready2.sync_reasoner()`, `matplotlib` runtime plot