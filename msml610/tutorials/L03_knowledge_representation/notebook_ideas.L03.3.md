# Notebook Ideas: Non-classical Logics and Knowledge Representation (L03.3)

- Source material:
  - `msml610/lectures_source/Lesson03.3-Non_classical_logics.smd`
- Each idea is one interactive Jupyter notebook that teaches the lecture concepts
  through:
  - Visualization
  - Interaction
  - Exploration
- Coverage of the lecture arc:
  - Idea 1: non-monotonic logic, default reasoning, belief revision
  - Idea 2: ontological and epistemological commitment, closed vs open world
    assumption, common sense reasoning
  - Idea 3: description logics (ALC, SHOIN), OWL variants, reasoning tasks
  - Idea 4: RDF, SPARQL, the semantic web stack, WikiData, DBpedia
  - Idea 5: semantic networks (WordNet, ConceptNet) and inductive logic programming

## 1. Tweety, Nixon, and the Yale Shooting: Reasoning That Takes Things Back

### Goal

- Students will:
  - Gain intuitive understanding of non-monotonic logic by watching a conclusion get
    retracted the moment a new fact arrives
  - Explore the relationship between the number of known facts and the number of
    derivable conclusions, which grows in classical logic and does not in
    non-monotonic logic

### Learning Objectives

- Encode a default rule as "conclude `B` from `A` unless `not B` is derivable"
- Implement negation as failure and see why it is not classical negation
- Plot the conclusion set as facts are added, and detect the monotonicity violation
- Produce multiple extensions for one knowledge base, and choose between them
- Trace which earlier belief a new fact killed, using a justification graph
- Connect default reasoning to the frame problem in temporal reasoning

### Core Concepts

- Monotonicity: if `KB |= alpha`, then `KB + beta |= alpha`, and how defaults break
  it
- Default logic: prerequisite, justification, and consequent of a default rule
- Circumscription: minimize the extension of an abnormality predicate
- Negation as failure vs classical negation, and the unique name assumption
- Extensions of a default theory, and the case where more than one exists
- Truth maintenance: justifications, `IN` and `OUT` labels, and dependency-directed
  backtracking

### Key Packages

- **clingo**: answer set programming, which gives defaults and negation as failure
  directly
- **pyDatalog**: Datalog rules with stratified negation, for the simpler cases
- **networkx**: draw the justification graph and mark the retracted nodes
- **ipywidgets**: toggle a fact on and off, and re-run the solver live

### Learning Activities

- Build the Tweety knowledge base: `bird(tweety)`, and the default "birds fly unless
  abnormal"
- Add `penguin(tweety)` and `abnormal(X) :- penguin(X)`, then watch `flies(tweety)`
  disappear from the answer set
- Plot `|conclusions|` against `|facts|` for a classical encoding and a default
  encoding on the same domain, and read the difference off the two curves
- Run the Nixon Diamond, the standard multiple-extension example:
  - `quaker(nixon)` and `republican(nixon)`
  - Quakers are pacifists by default, Republicans are not
  - Enumerate both extensions and show that neither is preferred without extra
    information
- Add a priority between the two defaults and confirm that one extension survives
- Re-build the lecture's university case:
  - Default: every Computer Science student takes every department course
  - Conclude `takesCourse(alice, cs101)`
  - Add "Alice lacks the prerequisites" and watch the conclusion flip
- Run the Yale Shooting Problem to see the frame problem concretely:
  - Load the gun, wait, shoot
  - Show the unintended model where the gun spontaneously unloads
  - Repair the encoding with an inertia default
- Build a small justification-based truth maintenance system:
  - Assert facts one at a time
  - Color nodes `IN` or `OUT` after each step
  - Click a node to see the chain of justifications behind it
- Measure how the answer set count grows as more conflicting defaults are added
- Compare the conclusion after every step against what classical logic would say,
  side by side in one table

## 2. Two Assumptions That Change Every Answer: Commitments and World Closure

### Goal

- Students will:
  - Gain intuitive understanding of ontological and epistemological commitment by
    asking one question in five formalisms and collecting five different answers
  - Explore the relationship between world closure and query results, by running the
    same query against a relational database (closed world) and a triple store (open
    world)

### Learning Objectives

- State the ontological commitment of propositional, first-order, higher-order, and
  temporal logic on the same domain
- State the epistemological commitment of logic (true, false, unknown) against
  probability theory (degree of belief in `[0, 1]`)
- Predict and verify the answer to an unasserted query under CWA and under OWA
- Connect negation as failure, `NOT EXISTS` in SQL, and `FILTER NOT EXISTS` in SPARQL
- Explain why counting is well defined under CWA and is not under OWA
- Encode a common sense script and show where its default expectations fail

### Core Concepts

- Ontological commitment: what the language says the world is made of
- Epistemological commitment: what belief states the agent may hold per fact
- Closed world assumption: what is not asserted is false
- Open world assumption: what is not asserted is unknown
- Unique name assumption, and why OWL drops it
- Common sense knowledge as implicit defaults about everyday situations
- Scripts as stereotyped event sequences, e.g., the restaurant script

### Key Packages

- **rdflib**: open-world triple store with SPARQL
- **owlready2**: OWL reasoning, plus explicit `AllDifferent` to restore unique names
- **duckdb**: closed-world relational queries over the same data
- **clingo**: closed-world answer set semantics for the logical encoding
- **ipywidgets**: dropdown to pick the formalism and see the query answer change

### Learning Activities

- Encode "every human is mortal" four times and compare what each encoding can say:
  - Propositional: one symbol per human, no quantifier
  - First-order: `forall x: Human(x) => Mortal(x)`
  - Higher-order: a statement about the relation itself, e.g., transitivity
  - Temporal: the fact holds over an interval, e.g., "the light is on at `t1`"
- Add probability theory as a fifth row: replace true/false/unknown with
  `Pr(X = 6) = 0.3`, and show that entailment becomes conditioning
- Build the lecture's enrollment example in two stores:
  - Assert only "Alice takes CS101"
  - Ask "does Bob take CS101?" in SQL and get `false`
  - Ask the same in SPARQL and get "no binding", which means unknown
- Write the OWA query that actually distinguishes "known false" from "unknown", using
  an explicit negative assertion
- Count students per course under both assumptions and show that the CWA count is a
  fact while the OWA count is only a lower bound
- Interactive closure slider: start from the open world, close one predicate at a
  time, and watch answers flip from unknown to false
- Drop the unique name assumption: assert `alice` and `a_smith` as separate names,
  ask whether the course has one student or two, then add `sameAs` and re-ask
- Encode the restaurant script from the lecture:
  - Bob enters, Bob sits, therefore Bob intends to order
  - Add "Bob asks for directions" and retract the order expectation
- Test the script against a handful of real restaurant reviews and count how many
  violate at least one default step
- Ask a large language model the same unasserted question and classify its answer as
  CWA-style or OWA-style, then discuss which behavior a knowledge base should have

## 3. From ALC to OWL: Buying Expressiveness with Compute

### Goal

- Students will:
  - Gain intuitive understanding of description logics by building concepts from
    constructors and letting a tableau reasoner classify them
  - Explore the relationship between expressiveness and reasoning cost, by adding one
    SHOIN letter at a time and measuring the runtime

### Learning Objectives

- Read and write ALC concept expressions with `and`, `or`, `not`, `exists`, `forall`
- Interpret a concept as a set and a role as a binary relation on a small model
- Run the four reasoning tasks: subsumption, satisfiability, instance checking, and
  classification
- Identify which SHOIN letter each axiom uses: transitivity, role hierarchy,
  nominals, inverse roles, cardinality
- Explain why OWL Lite, OWL DL, and OWL Full sit at different points of the
  decidability line
- Trace a tableau proof and see where it closes with a contradiction

### Core Concepts

- Concepts, roles, individuals, and the `TBox` and `ABox` split
- Set-theoretic semantics: a concept denotes a subset of the domain
- Subsumption `A subset of B`, and classification as the inferred hierarchy
- Unsatisfiable concept: a class that no individual can ever belong to
- The SHOIN letters and the axioms each one licenses
- Decidability vs expressiveness, and why OWL Full gives up decidability

### Key Packages

- **owlready2**: build the ontology in Python and call the HermiT reasoner
- **rdflib**: serialize the same ontology to Turtle and inspect the raw triples
- **ipycytoscape**: collapsible view of the asserted and the inferred class tree
- **networkx**: diff the asserted hierarchy against the inferred one

### Learning Activities

- Build a tiny finite model by hand:
  - Domain of six individuals
  - Roles `hasChild` and `takes` as explicit sets of pairs
  - Evaluate `Man and exists hasChild.Person` by direct set computation
- Define the lecture examples as ALC concepts and let the reasoner check them:
  - `Father = Man and exists hasChild.Person`
  - `Mother = Woman and exists hasChild.Thing`
  - `Student = exists takes.Course`
- Interactive concept builder: assemble an expression from constructor buttons and
  see the matching individuals highlighted in the model
- Walk up the SHOIN ladder one letter at a time on the university ontology:
  - `S`: make `ancestorOf` transitive and query the closure
  - `H`: assert `hasSon subset of hasChild` and re-classify
  - `O`: define a class by the nominal `{alice, bob}`
  - `I`: declare `isChildOf` the inverse of `hasChild`
  - `N`: assert `Person subset of (= 2 hasChild.Thing)` and find the conflict
- Time the reasoner after each letter and plot runtime against expressiveness
- Define `FlyingPenguin` as `Penguin and exists canFly.Thing`, confirm the reasoner
  marks it unsatisfiable, and read the explanation it returns
- Load the Manchester Pizza ontology, the standard teaching ontology, and make a
  `VegetarianPizza` inconsistent by adding one meat topping
- Hand-trace a tableau expansion for `A and not A` and for
  `exists R.C and forall R. not C`, then compare with the reasoner trace
- Compare the three OWL variants on one ontology:
  - OWL Lite: keep only hierarchies and simple cardinality
  - OWL DL: add the full constructor set
  - OWL Full: treat a class as an individual and observe that reasoning no longer
    terminates reliably
- Export to Turtle and RDF/XML and confirm the same ontology re-opens in Protege

## 4. The Semantic Web Stack: Triples, SPARQL, and Real Knowledge Graphs

### Goal

- Students will:
  - Gain intuitive understanding of RDF by building a graph one triple at a time and
    querying it with SPARQL
  - Explore the relationship between a small hand-built graph and the public
    billion-triple graphs that WikiData and DBpedia expose

### Learning Objectives

- Model any fact as a (`subject`, `predicate`, `object`) triple with IRIs and
  literals
- Write the four SPARQL query forms: `SELECT`, `CONSTRUCT`, `ASK`, `DESCRIBE`
- Build basic graph patterns from triple patterns and variables
- Derive new triples by RDFS entailment, e.g., `subClassOf` transitivity
- Read WikiData item and property identifiers, e.g., `Q42` and `P31`
- Measure incompleteness in a real knowledge graph and relate it to the open world
  assumption

### Core Concepts

- RDF as a directed labeled graph, and IRIs as global names
- Literals vs resources, and why the distinction matters for joins
- Triple patterns, basic graph patterns, and variable binding
- SPARQL endpoints, federated queries, and result serialization
- The semantic web layer cake: URIs, RDF, RDFS, OWL, SPARQL, applications
- Knowledge graph as an ontology plus a large instance layer

### Key Packages

- **rdflib**: in-memory triple store, Turtle and JSON-LD serialization, local SPARQL
- **SPARQLWrapper**: send queries to the live WikiData and DBpedia endpoints
- **pyvis**: draggable, zoomable subgraph rendering inside the notebook
- **networkx**: shortest paths, transitive closure, and degree distributions

### Learning Activities

- Type the lecture's book example as five triples and render it as a graph:
  - `Book123 hasTitle "The Great Gatsby"`
  - `Book123 hasAuthor Author456`
  - `Author456 hasName "F. Scott Fitzgerald"`
- Serialize the same graph to Turtle, N-Triples, and JSON-LD, and compare the three
  files
- Interactive query builder: pick subject, predicate, and limit with widgets, show
  the generated SPARQL text, then run it
- Run all four query forms on the same pattern and compare what each one returns
- Query WikiData for `Q42` (Douglas Adams) and walk the statements `P31` (instance
  of) and `P106` (occupation), including qualifiers and references
- Query DBpedia for Berlin and pull `dbo:country` and `dbo:populationTotal`, matching
  the lecture example
- Compute a Kevin Bacon number from a WikiData cast subgraph by breadth-first search
- Add an RDFS schema with `subClassOf` and `subPropertyOf`, then show the triples
  that appear only after entailment is switched on
- Federate one query across WikiData and DBpedia, then list the entities that failed
  to align
- Measure incompleteness: pick 100 people in WikiData and count how many lack a
  birthplace, then explain why the missing value is unknown and not absent
- Plot the degree distribution of the retrieved subgraph and identify the hub
  entities
- Break a query on purpose by using a literal where a resource is required, and read
  the empty result as a modeling error rather than a knowledge gap

## 5. Where Symbolic Knowledge Comes From: Curated, Crowdsourced, and Learned

### Goal

- Students will:
  - Gain intuitive understanding of semantic networks by traversing WordNet and
    ConceptNet and measuring what each one does and does not contain
  - Explore the relationship between hand-curated knowledge and learned knowledge, by
    using inductive logic programming to induce the same kind of rule from examples

### Learning Objectives

- Traverse a semantic network along typed edges, e.g., `is-a`, `part-of`, `UsedFor`
- Compute similarity from graph structure and compare it with human judgments
- Contrast a curated lexical resource with a crowdsourced commonsense resource
- Induce a first-order rule from positive examples, negative examples, and background
  knowledge
- Read an induced rule as a default rule with exceptions, linking back to Idea 1
- Identify the failure modes of ILP: noisy data and large hypothesis spaces

### Core Concepts

- Semantic network: nodes are concepts, edges are labeled relations
- Synsets in WordNet, hypernym chains, antonymy, and meronymy
- ConceptNet relations: `IsA`, `PartOf`, `UsedFor`, `CapableOf`, `Causes`
- Spreading activation as inference by graph traversal
- Inductive logic programming: background knowledge, examples, hypothesis space
- Coverage and consistency of an induced clause, and rule readability as a feature

### Key Packages

- **nltk**: WordNet synsets, hypernym paths, and Wu-Palmer similarity
- **conceptnet-lite**: local ConceptNet queries without the rate-limited web API
- **popper-ilp**: modern ILP system that learns recursive rules and handles
  exceptions
- **aleph** or **pyswip**: the classic Prolog-based ILP route for comparison
- **ipycytoscape**: interactive, expandable neighborhood view of a concept

### Learning Activities

- Walk the WordNet hypernym chain from `dog` up to `entity` and draw the chain
- Rebuild the lecture's temperature network in code, with `hot`, `cold`, `arctic`,
  and the `attribute`, `antonym`, `hypernym`, and `similar` edges
- Query ConceptNet for `knife` and list the `UsedFor` and `CapableOf` neighbors, then
  do the same for `bicycle` to match the lecture example
- Interactive concept explorer: click a node to expand its neighbors, and filter the
  edges by relation type
- Compare the two resources on the same 50 word pairs:
  - WordNet Wu-Palmer similarity
  - ConceptNet edge-weighted path similarity
  - Plot both against human ratings and inspect the largest disagreements
- Find the commonsense facts that ConceptNet has and WordNet cannot express, e.g., "a
  knife is used for cutting"
- Run spreading activation from two seed concepts and report where the activations
  meet
- Induce rules with ILP on the classic benchmarks:
  - Michalski East-West trains: learn which trains go east
  - Family relations: learn `grandparent` from `parent`, including the recursive
    `ancestor` rule
- Re-run the lecture's bird example as an ILP task:
  - Background: birds have wings
  - Positive: Tweety is a bird that flies
  - Negative: a penguin does not fly
  - Read the induced rule and note that it is exactly a default with an exception
- Inject label noise into the examples and plot induced-rule accuracy against the
  noise rate, to make the ILP brittleness claim concrete
- Grow the background knowledge and measure how search time scales with hypothesis
  space size
- Compare the induced rules against a decision tree on the same data, on accuracy and
  on whether a human can read the model