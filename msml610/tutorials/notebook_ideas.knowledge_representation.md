# Notebook Ideas: Knowledge Representation

- Source material: `msml610/lectures_source/Lesson03.1-Knowledge_representation.smd`
- Each idea is one interactive Jupyter notebook that teaches the lecture concepts
  through:
  - Visualization
  - Interaction
  - Exploration
- Coverage of the lecture arc:
  - Ideas 1: semantics, entailment, inference, knowledge-based agents
  - Idea 2: rule-based systems and reflex agents
  - Idea 3: ontologies and description-logic reasoning tasks
  - Idea 4: knowledge graphs, grounding, and open-world queries
  - Idea 5: symbolic vs sub-symbolic representation and conceptual spaces

## 1. Wumpus World: From Percepts to Proofs

### Goal

- Students will:
  - Gain intuitive understanding of knowledge bases, models, and entailment by
    building a knowledge-based agent that survives the Wumpus World
  - Explore the relationship between what a $KB$ means (semantics) and what an
    algorithm can derive from it (inference)

### Learning Objectives

- Encode percepts as propositional sentences with `TELL` and query them with `ASK`
- Enumerate all possible worlds and compute $M(KB)$ and $M(\alpha)$
- Verify entailment as the set inclusion $M(KB) \subseteq M(\alpha)$
- Separate implication, entailment, and inference on concrete examples
- Observe soundness (no false positives) and completeness (no false negatives)
- Measure how model checking degrades as the number of variables grows

### Core Concepts

- Syntax vs semantics of a knowledge representation language
- Model as a full assignment to all variables, and satisfaction of a sentence
- Logical entailment $KB \models \alpha$ and its model-theoretic definition
- Model checking as a sound and complete brute-force algorithm
- Forward chaining, backward chaining, and resolution as alternatives
- Expressiveness vs tractability: atomic, factored, and structured states

### Key Packages

- **sympy**: propositional formulas, truth tables, and symbolic simplification
- **python-sat**: DPLL and CDCL solver backends for the same knowledge base
- **ipywidgets**: click a cell of the grid to `TELL` a percept and re-run `ASK`
- **networkx**: draw the proof graph produced by forward chaining

### Learning Activities

- Build the classic 4x4 Wumpus World grid with pits, a wumpus, and gold
- Encode the breeze axiom $B_{1,2} \iff (P_{1,1} \lor P_{2,2} \lor P_{1,3})$ for
  every cell
- Enumerate all $2^n$ models in a table, and shade the rows that satisfy the $KB$
- Query _"is cell (2,2) provably safe?"_ and read the answer off the shaded rows
- Explore the three-way distinction with a widget:
  - Implication: a sentence inside the logic
  - Entailment: truth preserved across every shaded row
  - Inference: the procedure that walks from $KB$ to $\alpha$
- Test a deliberately unsound rule (affirming the consequent) and count the wrong
  conclusions it produces
- Test an incomplete rule set (modus ponens only, no resolution) and count the
  entailed facts it misses
- Measure runtime of model checking vs a SAT solver as the grid grows from 2x2 to 6x6
- Explore the random 3-SAT phase transition near a clause-to-variable ratio of
  $4.26$, where the hardest instances live
- Interactive agent loop: step the agent, watch the $KB$ grow, and watch the set of
  candidate models shrink

## 2. MYCIN Redux: A Rule-Based Expert System You Can Debug

### Goal

- Students will:
  - Gain intuitive understanding of rule-based systems by implementing the
    match-conflict-resolution-act cycle from scratch
  - Explore the relationship between explainability and predictive accuracy by
    comparing a rule engine against a learned classifier on the same data

### Learning Objectives

- Encode expert knowledge as if-then rules over a working memory of facts
- Implement forward chaining (data-driven) and backward chaining (goal-driven)
- Compare conflict resolution strategies: specificity, recency, and priority
- Trace and explain every conclusion back to the rules that fired
- Extend crisp rules with certainty factors to handle uncertainty
- Contrast a reflex agent (percept only) with a rule-based agent (working memory)

### Core Concepts

- Rule-based system components: knowledge base, inference engine, working memory
- Forward chaining vs backward chaining, and when each is cheaper
- Conflict resolution and rule-ordering effects on the final fact set
- Explainability as a first-class property of symbolic systems
- Declarative vs procedural encoding of the same behavior
- Rule-based trade-offs: brittleness, maintenance cost, and uncertainty handling

### Key Packages

- **experta**: a production-rule engine in the CLIPS tradition
- **graphviz**: render the rule dependency graph and the inference tree
- **ipywidgets**: toggle patient symptoms and watch the diagnosis change live
- **scikit-learn**: train the learned baseline that the rule engine competes with

### Learning Activities

- Re-build a small MYCIN-style diagnostic KB, the 1970s Stanford system that
  recommended antibiotics for blood infections
- Encode Winston's classic animal-identification rules:

  ```text
  IF has_hair THEN mammal
  IF mammal AND eats_meat THEN carnivore
  IF carnivore AND tawny AND dark_spots THEN cheetah
  ```

- Run forward chaining and animate the working memory filling up, one fired rule per
  frame
- Run backward chaining on the goal `cheetah` and draw the AND-OR proof tree it
  explores
- Measure how many rules each strategy fires on the same query
- Answer the two MYCIN explanation questions for any conclusion:
  - _"How did you conclude this?"_
  - _"Why are you asking me this?"_
- Add certainty factors to rules and propagate them, then compare the ranking of
  candidate diagnoses
- Interactive conflict resolution: reorder the rules with a slider and observe when
  the derived facts change
- Train a decision tree on a symptom dataset and compare it to the rule engine on
  accuracy, on latency, and on whether a doctor can audit the reasoning
- Break the KB by adding a contradictory rule, and watch the engine loop or produce
  inconsistent facts

## 3. Ontology Lab: Pizzas, Penguins, and a Reasoner

### Goal

- Students will:
  - Gain intuitive understanding of ontologies by authoring classes, individuals,
    properties, and axioms, then letting a description-logic reasoner do the work
  - Explore the relationship between asserted knowledge and inferred knowledge

### Learning Objectives

- Build an ontology from the lecture components: classes, individuals, properties,
  attributes, constraints, axioms, and hierarchies
- Run class-level reasoning: subsumption, satisfiability, and classification
- Run instance-level reasoning: instance checking, consistency, and realization
- Distinguish an ontology from a database schema, a taxonomy, and a knowledge base
- Watch reasoning time grow as the ontology moves to more expressive constructs

### Core Concepts

- OWL and RDF as concrete knowledge representation languages
- Description logic constructs and the expressiveness vs tractability curve
- Subsumption: _"is class `A` more general than class `B`?"_
- Unsatisfiable concepts, e.g., `FlyingPenguin` requires flying and cannot fly
- The open-world assumption: unknown is not the same as false
- Asserted hierarchy vs inferred hierarchy after classification

### Key Packages

- **owlready2**: build OWL ontologies in Python and call the HermiT reasoner
- **rdflib**: serialize and query the same ontology as RDF triples
- **networkx**: diff the asserted class graph against the inferred one
- **ipycytoscape**: interactive, collapsible class hierarchy view

### Learning Activities

- Load the Manchester Pizza ontology, the standard Protege teaching ontology, and
  browse its class tree
- Ask the reasoner to classify a `VegetarianPizza` that lists a meat topping, and
  read the explanation for the resulting inconsistency
- Re-build the lecture's university ontology in code:
  - Classes: `Student`, `Professor`, `Course`, `Department`
  - Properties: `takesCourse`, `teachesCourse`, `belongsToDepartment`
  - Axioms: every `Course` is taught by exactly one `Professor`
- Define `FlyingPenguin` and confirm the reasoner marks the concept unsatisfiable
- Perform realization: ask for the most specific class of the individual `GP`
- Perform retrieval: list every individual that satisfies `TeachingAssistant`
- Interactive axiom editor: add or remove one axiom and re-run the reasoner, with
  newly inferred edges highlighted in the hierarchy view
- Demonstrate the open-world assumption: query for a fact that is simply absent and
  get "unknown" rather than "false"
- Measure reasoner runtime as cardinality constraints and property chains are added
- Export the ontology to Turtle and RDF/XML, then re-open the file in Protege

## 4. Knowledge Graphs in the Wild: WikiData, DBpedia, and Bacon Numbers

### Goal

- Students will:
  - Gain intuitive understanding of large-scale knowledge representation by querying
    public knowledge graphs with millions of grounded facts
  - Explore the relationship between a hand-built ontology and the messy, incomplete
    graphs that real systems query

### Learning Objectives

- Model knowledge as (`subject`, `predicate`, `object`) triples
- Write SPARQL queries against live DBpedia and WikiData endpoints
- Implement path-based reasoning, e.g., transitive closure of `ancestorOf`
- See grounding in practice: entity identifiers that point to real-world things
- Detect and repair schema mismatches when merging two knowledge sources
- Compare structured query answering against embedding-based similarity search

### Core Concepts

- Knowledge graph as a knowledge base built from an ontology
- RDF triples, IRIs, and the linked-data idea
- SPARQL as the declarative query language: state _what_, not _how_
- Transitive and symmetric properties, and inference by graph traversal
- Grounding: symbols in the graph refer to entities in the world
- Incomplete and noisy knowledge, and the limits of open-world querying

### Key Packages

- **rdflib**: local in-memory triple store and SPARQL over it
- **SPARQLWrapper**: send queries to the DBpedia and WikiData endpoints
- **networkx**: shortest paths, transitive closure, and centrality
- **pyvis**: interactive, draggable subgraph visualization in the notebook

### Learning Activities

- Hand-build the university knowledge graph from the lecture as RDF triples, and
  query it with SPARQL
- Query DBpedia for the cast of a film and render the actor-film bipartite graph
- Compute the Six Degrees of Kevin Bacon number for any actor by breadth-first search
  on the queried subgraph
- Compute an Erdos number from a co-authorship subgraph and compare the two
  small-world structures
- Query WikiData for _"Nobel laureates in Physics born in Germany"_ and inspect the
  entity and property identifiers that ground each answer
- Interactive query builder: pick a subject, a property, and a limit with widgets,
  then read the generated SPARQL before it runs
- Reason over paths: derive `ancestorOf` from repeated `parentOf` edges, and verify
  against a direct query
- Merge a DBpedia subgraph and a WikiData subgraph, then find the entities that
  failed to align
- Measure how answer counts change when one property is missing, to make the
  open-world assumption concrete
- Compare a SPARQL query against nearest-neighbor search over graph embeddings on the
  same question

## 5. Symbols, Vectors, and Conceptual Spaces

### Goal

- Students will:
  - Gain intuitive understanding of symbolic vs sub-symbolic representation by
    encoding the same concepts as a WordNet hierarchy and as learned vectors
  - Explore the relationship between geometric similarity and discrete structure,
    which is the core claim of conceptual spaces

### Learning Objectives

- Traverse a large hand-built semantic network and measure symbolic similarity
- Train or load word embeddings and measure cosine similarity on the same pairs
- Show that discrete symbols carry no similarity structure on their own
- Build a low-dimensional conceptual space and test whether concepts form regions
- Connect symbols to sensory data, i.e., grounding
- Assemble a minimal neuro-symbolic pipeline: retrieve with vectors, check with rules

### Core Concepts

- Symbolic KR: discrete, interpretable, brittle under ambiguity
- Sub-symbolic KR: distributed, robust, opaque
- Neuro-symbolic KR: reason with logic over learned concepts
- Conceptual spaces: interpretable dimensions, concepts as convex regions, similarity
  as distance
- Grounding as the bridge between a symbol and the world
- Sapir-Whorf in data: vocabulary boundaries differ across languages

### Key Packages

- **nltk**: WordNet synsets, hypernym paths, and Wu-Palmer similarity
- **gensim**: Word2Vec and GloVe vectors, analogy arithmetic
- **umap-learn**: project high-dimensional embeddings to 2D for inspection
- **scikit-image**: convert colors between RGB and the perceptual CIELAB space
- **ipywidgets**: sliders for the two axes of a conceptual space

### Learning Activities

- Walk the WordNet hypernym path from `dog` up to `entity`, and draw the chain
- Compute Wu-Palmer path similarity for pairs such as (`car`, `bicycle`) and (`car`,
  `democracy`)
- Compute cosine similarity for the same pairs in embedding space, then plot symbolic
  similarity against vector similarity and inspect the disagreements
- Run the analogy `king - man + woman` and then find the pairs where the same
  arithmetic fails
- Rebuild the lecture's transportation conceptual space:
  - Axes: `Environmental friendliness` and `Technological advancement`
  - Place `Bicycle`, `Motorbike`, `Elevator`, and check which fall inside the
    `Vehicle` region
- Test the convexity criterion: sample points between two members of a concept and
  ask whether the midpoint is still a member
- Build a color conceptual space in CIELAB and draw English color-name regions, then
  draw the Russian split between _siniy_ (dark blue) and _goluboy_ (light blue) as
  two basic regions
- Ground symbols in perception using the fact that ImageNet labels are WordNet
  synsets: pick a synset, retrieve its images, and discuss what the mapping assumes
- Interactive ambiguity demo: query the symbol `spring` in WordNet, get several
  synsets, and compare with the single vector a static embedding assigns to it
- Neuro-symbolic mini-task:
  - Retrieve candidate facts by embedding similarity
  - Filter the candidates with an explicit logical rule
  - Measure precision before and after the symbolic filter