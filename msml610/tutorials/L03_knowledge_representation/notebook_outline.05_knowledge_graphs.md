# Knowledge Graphs in the Wild: WikiData, DBpedia, and Bacon Numbers

- Source: `Lesson03.3-Non_classical_logics.smd`
- This notebook moves from a hand-built knowledge graph to live public knowledge
  graphs with millions of grounded facts
- The pedagogical arc:
  - Triples and SPARQL on a small hand-built knowledge graph
  - Grounded entities on live DBpedia queries
  - Path-based reasoning: Bacon numbers and Erdos numbers by BFS
  - An interactive SPARQL query builder against WikiData
  - Transitive closure derived by traversal vs a direct query
  - Merging two knowledge graphs and finding schema mismatches
  - The open world assumption and structured vs embedding-based search

## Cell 1: From Ontology to Knowledge Graph: Triples You Can Query

**Goal**:
- Represent the lecture's university ontology as (`subject`, `predicate`, `object`)
  triples
- Write and run the first SPARQL query against the local triple store
**Plots and their descriptions**:
- _Triple graph_: `pyvis` interactive graph of the university knowledge graph, edges
  labeled by predicate
- _Triple table_: raw (`subject`, `predicate`, `object`) rows backing the graph
- _Comments_: query text entered, number of result rows
**Widgets**:
- `sparql_query`: text box for a SPARQL query against the local store
- `run`: button to execute the query and show the result table
**Key observations** (post-visualization):
- A knowledge graph is a knowledge base stored literally as (`subject`, `predicate`,
  `object`) triples
- SPARQL states what pattern to match, not how to search for it: the same query works
  regardless of how the triples are stored internally
- The graph view and the triple table are two views of the exact same data
**Implementation**: `rdflib.Graph` for the local triple store,
`rdflib.plugins.sparql` for query execution, `pyvis.network.Network` for the graph
view

## Cell 2: Querying DBpedia: Grounded Entities, Not Just Strings

**Goal**:
- Query DBpedia for the cast of a chosen film and render the actor-film graph
- Make the idea of grounding concrete: a symbol in the graph points to one specific
  real-world entity
**Plots and their descriptions**:
- _Bipartite graph_: `pyvis` graph with films on one side and actors on the other
- _Identifier panel_: raw DBpedia IRIs (e.g., `dbr:Actor_Name`) next to their
  human-readable labels
- _Comments_: film queried, number of cast members returned
**Widgets**:
- `film`: dropdown or text box to pick a film title
- `fetch`: button to query DBpedia and redraw the graph
**Key observations** (post-visualization):
- Every node is grounded to a stable identifier, not merely a display string
- Two different display strings can resolve to the same grounded entity, and the same
  string can appear on unrelated entities
- Grounding is what lets the next cell walk from one actor to another with confidence
  the edges refer to real relationships
**Implementation**: `SPARQLWrapper.SPARQLWrapper` against the DBpedia endpoint,
`pyvis.network.Network` for the bipartite graph

## Cell 3: Six Degrees of Kevin Bacon: Path Reasoning by BFS

**Goal**:
- Compute the Bacon number of a chosen actor by breadth-first search on the queried
  actor-film subgraph
**Plots and their descriptions**:
- _Shortest path_: `pyvis` graph with the BFS path from the chosen actor to Kevin
  Bacon highlighted
- _Degree distribution_: bar chart of node degrees in the searched subgraph
- _Comments_: starting actor, computed Bacon number, nodes visited during BFS
**Widgets**:
- `actor`: dropdown or text box to pick a starting actor
- `max_depth`: slider capping how many BFS hops to search (1-6)
**Key observations** (post-visualization):
- A knowledge graph answers multi-hop questions ("who links to whom, and how far")
  that a single triple query cannot answer directly
- The path length BFS finds is exactly the actor's Bacon number
- Raising `max_depth` beyond the true answer changes nothing: BFS already finds the
  shortest path
**Implementation**: `networkx.shortest_path` (BFS) on the subgraph built from DBpedia
queries, `pyvis` path highlighting

## Cell 4: Erdos Numbers: the Same Small-World Structure Elsewhere

**Goal**:
- Compute an Erdos number from a co-authorship subgraph and compare its small-world
  structure to the Bacon graph
**Plots and their descriptions**:
- _Degree-distribution comparison_: side-by-side histograms for the Bacon graph and
  the Erdos graph
- _Comments_: starting author, computed Erdos number, comparison summary statistics
**Widgets**:
- `author`: dropdown or text box to pick a starting author
- `graph_choice`: toggle between "show Bacon graph" and "show Erdos graph"
**Key observations** (post-visualization):
- Two independently built graphs, one of movies and one of papers, show the same
  small-world statistics: short average path length despite huge size
- The BFS technique from Cell 3 transfers unchanged across domains: only the
  underlying triples differ
- Comparing the two histograms side by side is more convincing than any single
  numeric claim about "six degrees"
**Implementation**: `networkx.shortest_path` for the Erdos BFS, `SPARQLWrapper` for
the co-authorship subgraph, `matplotlib` histogram

## Cell 5: Interactive SPARQL Query Builder

**Goal**:
- Let students assemble a SPARQL query from three simple widget choices and see the
  generated query before it runs against WikiData
**Plots and their descriptions**:
- _Generated query panel_: the SPARQL text assembled live from the widget values
- _Result table_: rows returned once the query runs
- _Comments_: subject, property, and limit currently selected
**Widgets**:
- `subject`: dropdown for the query subject (e.g., "Nobel laureates in Physics")
- `property`: dropdown for a filtering property (e.g., "born in Germany")
- `limit`: numeric input for the SPARQL `LIMIT` clause
**Key observations** (post-visualization):
- The generated query panel demystifies SPARQL: three widget choices map directly
  onto query clauses
- Every returned row carries its own entity and property identifiers, which ground
  the answer back to specific WikiData items
- Changing `limit` only truncates the result list; it never changes which facts are
  entailed by the graph
**Implementation**: `SPARQLWrapper.SPARQLWrapper` against the WikiData endpoint,
string templating for the generated query panel

## Cell 6: Deriving ancestorOf: Transitive Closure by Graph Traversal

**Goal**:
- Derive the `ancestorOf` relation from repeated `parentOf` edges and verify the
  result against a direct WikiData query
**Plots and their descriptions**:
- _Traversal graph_: `parentOf` edges plus derived `ancestorOf` edges overlaid in a
  distinct style
- _Agreement table_: derived `ancestorOf` pairs next to the directly queried pairs,
  with mismatches (if any) flagged
- _Comments_: number of hops allowed, number of derived pairs, number that match the
  direct query
**Widgets**:
- `max_hops`: slider for how many `parentOf` hops to traverse (1-5)
**Key observations** (post-visualization):
- A transitive property does not need to be stored explicitly: repeated traversal
  computes it from a single base relation
- The derived and directly queried answers agree, confirming the graph encodes the
  same relation both ways
- Capping `max_hops` too low silently truncates the derived relation, an
  open-world-style failure mode worth noticing here
**Implementation**: `networkx` transitive closure via repeated BFS/DFS on `parentOf`
edges, `SPARQLWrapper` for the direct verification query

## Cell 7: Merging Two Knowledge Graphs: Schema Mismatches

**Goal**:
- Merge a DBpedia subgraph and a WikiData subgraph on the same topic, and find the
  entities that fail to align
**Plots and their descriptions**:
- _Merged graph_: nodes colored by source (DBpedia, WikiData, or both after
  alignment)
- _Mismatch table_: entities present in one source but not linked to a counterpart in
  the other
**Widgets**:
- `alignment_strategy`: dropdown for "exact label match" vs "`owl:sameAs` link"
- `merge`: button to merge and recolor the graph under the chosen strategy
**Key observations** (post-visualization):
- Independently built knowledge graphs rarely share identifiers directly, even when
  describing the same real-world entities
- Alignment quality depends entirely on which signal is used: label matching is
  noisier but always available, `owl:sameAs` links are precise but sparse
- The mismatch table exposes exactly where a downstream query would silently miss
  facts if the merge is trusted blindly
**Implementation**: `rdflib.Graph` union of both sources, `networkx` for the merged
graph, `pyvis` two-color rendering by source

## Cell 8: Open World in Practice, and Structured vs Embedding Search

**Goal**:
- Show how answer counts change when a property is missing under the open world
  assumption, and compare structured SPARQL answers against embedding-based
  nearest-neighbor search on the same question
**Plots and their descriptions**:
- _Answer-count bar_: number of query results with the property present vs missing
- _Agreement table_: SPARQL results next to the top-$k$ nearest neighbors from a
  graph embedding, with agreement highlighted
- _Comments_: property toggled, $k$ used for nearest-neighbor search
**Widgets**:
- `property_present`: toggle to include or remove one property from the graph
- `k`: slider for the number of nearest neighbors retrieved (1-10)
**Key observations** (post-visualization):
- Removing one property silently shrinks the SPARQL answer set rather than raising an
  error: this is the open world assumption in practice
- Structured query answering returns only what is explicitly represented, while
  embedding-based search can surface plausible matches the graph never states
  directly
- The gap between the two result lists is exactly the trade-off between precision
  (SPARQL) and recall on incomplete knowledge (embeddings)
**Implementation**: `rdflib`/`SPARQLWrapper` for structured answers, `node2vec`-style
embeddings via `gensim` for the graph, nearest-neighbor search via
`scikit-learn.neighbors.NearestNeighbors`