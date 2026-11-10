// git_hash=6da0f339-2dk timestamp=20260925_153613
// Import AIMA style formatting and macros.
#import "/helpers_root/dev_scripts_helpers/typst/aima_style.typ": (
  aima-style, algorithm, chapter, glossary, styled-table,
)
// Import the custom citation/bibliography system.
#import "/helpers_root/dev_scripts_helpers/typst/umd_references.typ": (
  cite, references,
)

// Document metadata
#set document(
  title: "L06.2: Using Bayesian Networks",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L06.2: Using Bayesian Networks")

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:12 '* Roadmap'
// Slide: Roadmap
= Roadmap

#strong[Bayesian networks] provide a compact representation of a joint
distribution by encoding conditional independence relationships in a directed
acyclic graph. This chapter covers how to read and construct such networks from
domain knowledge, formalizing the way variables depend on one another through
local conditional probability tables rather than a single monolithic joint
table.

A central concept in this framework is the #strong[Markov blanket]: the minimal
set of nodes (a variable's parents, children, and children's other parents) that
renders a variable conditionally independent of every other node in the network.
Understanding the Markov blanket clarifies both the semantics of the graph and
the compact representations available for conditional probability tables.

The chapter then turns to inference, the task of computing posterior
probabilities given observed evidence. #strong[Exact inference] methods,
including enumeration and variable elimination, exploit the network's factored
structure to answer queries without enumerating the full joint. When exact
computation becomes intractable, #strong[approximate inference] via Monte Carlo
sampling offers a practical alternative. The sampling toolkit covered here
includes rejection sampling, importance sampling, Markov chain Monte Carlo
(MCMC), Gibbs sampling, and Metropolis-Hastings, each trading exactness for
scalability in different ways.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:24 '# Semantics of Bayesian Networks'
// Slide: Semantics of Bayesian Networks
= Semantics of Bayesian Networks

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:28 '* Bayesian Networks: Semantics'
// Slide: Bayesian Networks: Semantics
#strong[Bayesian Networks: Semantics]

A #strong[Bayesian network] admits two equivalent semantic interpretations
#cite("pearl1988probabilistic"). The first is the #emph[joint distribution
  view]: the network encodes the full joint probability distribution over all
variables as a product of local conditional probabilities,

$ P(X_1, ..., X_n) = product_(i=1)^n P(X_i | "Parents"(X_i)) $

This factorization is what makes Bayesian networks compact. Instead of storing
an exponentially large joint table, each variable contributes only a small
conditional probability table (CPT) conditioned on its parents in the graph. The
result is a complete specification of the joint distribution from which any
query can, in principle, be answered.

The second is the #emph[conditional independence view]: the graph's structure
directly encodes conditional independence relationships among the variables.
Specifically, each variable is conditionally independent of all its
non-descendants given its parents. This reading is what makes inference
tractable, because it tells the reasoning algorithm which variables can be
ignored when computing a particular query. The two views are mathematically
equivalent: the independence assumptions encoded by the graph are exactly those
needed to justify the product factorization, and vice versa. In practice, the
joint distribution view is most useful when constructing a model and
understanding its overall behavior, while the conditional independence view
guides the design of efficient inference algorithms.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:50 '* Chain Rule for a Joint Distribution'
// Slide: Chain Rule for a Joint Distribution
#strong[Chain Rule for a Joint Distribution]

I need to understand the context better to address this TODO. The comment
`// TODO(ai_gp): Issue with colors inside math` suggests there's a problem with
using color commands inside Typst math mode (`$...$`).

Could you provide more context? Specifically:

1. #strong[Where is this TODO located?] (Which file, what surrounding code?)
2. #strong[What's the actual symptom?] For example:
  - Does `#text(fill: red)[$x$]` not work as expected?
  - Is there a problem with `$#text(fill: red)[x] + y$` inside math?
  - Is `aima_style.typ` applying a color function that breaks inside `$...$`?
3. #strong[What's the desired behavior?] (e.g., coloring a single variable in a
  formula, highlighting a term in a display equation)

In general, Typst's math mode has known quirks with inline styling. The typical
patterns are:

```typst
// Color an entire formula from outside:
#text(fill: red)[$x + y$]

// Color one symbol inside math (use the # escape):
$#text(fill: red)[x] + y$

// Using a show rule to color a specific symbol everywhere:
#show math.equation: set text(fill: blue)
```

If you can share the file or the surrounding code, I can give you a concrete
fix.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:84 '* Evaluate a Bayesian Network'
// Slide: Evaluate a Bayesian Network
#strong[Evaluate a Bayesian Network]

Computing the joint probability distribution over all variables in a Bayesian
network follows a four-step procedure that exploits the network's graphical
structure to turn what would otherwise be an exponential-sized table into a
compact, factored product.

#algorithm("Joint Probability in a Bayesian Network", [
  1. *Sort* the nodes in topological order consistent with the directed edges
    (parents before children). Multiple valid orderings exist; any one produces
    the same result.\
  2. *Apply the chain rule* using that topological ordering:
  $ Pr(X_1, ..., X_n) = product_(i=1)^n Pr(X_i | X_(i-1), ..., X_1) $
  3. *Simplify each conditional* by the #emph[Markov assumption]: every node is
    conditionally independent of all its non-descendant predecessors given its
    parents, so
  $ Pr(X_i | X_(i-1), ..., X_1) = Pr(X_i | "Parents"(X_i)) $
  4. *Express the joint* as a product of entries from the #strong[Conditional
      Probability Tables] (CPTs):
  $ Pr(X_1, ..., X_n) = product_(i=1)^n Pr(X_i | "Parents"(X_i)) $
])

The power of this factorization comes from the conditional independencies
encoded in the network's structure. Without those independencies, writing down
the joint distribution for $n$ binary variables would require specifying
$2^n - 1$ independent parameters. The Bayesian network replaces that monolithic
table with a collection of small CPTs, one per node, each conditioned only on
the node's parents. Because most nodes have far fewer parents than total
variables, the number of parameters drops from exponential to something
manageable. The CPTs alone, together with the graph topology, fully specify
$Pr(X_1, ..., X_n)$: no additional information is needed to answer any
probabilistic query about the domain.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:111 '* Evaluate a Bayesian Network: Example'
// Slide: Evaluate a Bayesian Network: Example
#strong[Evaluate a Bayesian Network: Example]

Consider a concrete scenario using the classic burglary-alarm network. Suppose
you want to find the probability that the alarm has sounded, that neither a
burglary nor an earthquake has occurred ($not "Burglary" and not "Earthquake"$),
and that both John and Mary call ($"JohnCalls" and "MaryCalls"$).
@fig:evaluateabayesiannetworkexample shows the network structure for this
problem: Burglary and Earthquake feed into Alarm, which in turn influences
whether John and Mary call.

// rendered_images:begin
// ```graphviz[width=70%]
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//     Burglary [label="Burglary", fillcolor="#A6C8F4"];
//     Earthquake [label="Earthquake", fillcolor="#FFD1A6"];
//     Alarm [label="Alarm", fillcolor="#B2E2B2"];
//     JohnCalls [label="JohnCalls", fillcolor="#C6A6F4"];
//     MaryCalls [label="MaryCalls", fillcolor="#C6A6F4"];
// 
//     { rank = same; Burglary; Earthquake; }
//     { rank = same; JohnCalls; MaryCalls; }
// 
//     Burglary -> Alarm;
//     Earthquake -> Alarm;
//     Alarm -> JohnCalls;
//     Alarm -> MaryCalls;
// }
// ```
// label=fig:evaluateabayesiannetworkexample
// caption=Diagram relating Burglary,
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.1.png",
    width: 70%,
  ),
  caption: [Diagram relating Burglary,],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:evaluateabayesiannetworkexample>
// render_images:end
Earthquake, Alarm and JohnCalls

The chain rule for Bayesian networks lets you decompose this joint probability
into a product of terms, each conditioned only on its parents in the graph:

$ Pr("JohnCalls", "MaryCalls", "Alarm", not "Burglary", not "Earthquake") $
$
  = Pr("JohnCalls" | "Alarm") dot.op Pr("MaryCalls" | "Alarm") dot.op Pr("Alarm" | not "Burglary" and not "Earthquake") dot.op Pr(not "Burglary") dot.op Pr(not "Earthquake")
$

Each factor in this product corresponds to exactly one node's conditional
probability table. $Pr("JohnCalls" | "Alarm")$ and $Pr("MaryCalls" | "Alarm")$
come from the tables for John and Mary, conditioned on the alarm state.
$Pr("Alarm" | not "Burglary" and not "Earthquake")$ is the alarm's own table
entry for the case where neither cause is present. Finally, $Pr(not "Burglary")$
and $Pr(not "Earthquake")$ are the prior probabilities of no burglary and no
earthquake, read directly from the root nodes. Because each variable appears in
exactly one factor (the one where it sits on the left-hand side, conditioned on
its parents), the full joint probability reduces to a simple multiplication of
five numbers that you can look up directly from the network's tables.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:161 '# Constructing a Bayesian Network'
// Slide: Constructing a Bayesian Network
= Constructing a Bayesian Network

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:163 '* Steps to Construct a Bayesian Network'
// Slide: Steps to Construct a Bayesian Network
#strong[Steps to Construct a Bayesian Network]

Building a Bayesian network from scratch follows a structured, four-step process
#cite("russell2020aima"). First, gather domain knowledge by listing all relevant
random variables needed to describe the system and identifying the key
interactions among them. This step sets the vocabulary of the model: every
quantity that matters to the domain should appear as a node.

Second, order the nodes according to cause-effect dependencies. A careful causal
ordering ensures that the resulting network is minimal, meaning it contains no
superfluous edges. In practice, this means placing causes before their effects
so that each variable conditions only on things that temporally or logically
precede it.

Third, for each node $X_i$, pick the minimum set of parents $"Parents"(X_i)$.
Add edges only where a genuine dependency exists and avoid redundant
connections. If a variable is conditionally independent of a candidate parent
given the other parents already selected, that edge is unnecessary. Keeping the
parent sets small is what makes Bayesian networks computationally tractable.

Fourth, estimate the conditional probability $Pr(X_i | "Parents"(X_i))$ for each
node. These conditional probability tables (CPTs) can come from data, from
expert elicitation, or from a combination of both. When data are plentiful,
standard statistical or maximum-likelihood techniques work well; when data are
scarce, structured interviews with domain experts fill the gap.

Finally, validate the completed model. Have domain experts review the structure
to catch missing or spurious edges. Confirm that the graph is a directed acyclic
graph (DAG); any cycle would make the joint distribution ill-defined. Then test
the network by predicting known outcomes and comparing the predictions against
actual data, iterating on both structure and parameters until the model performs
reliably. @fig:stepstoconstructabayesiannetwork summarizes the four construction
steps and their logical flow from domain knowledge gathering through CPT
estimation.

// rendered_images:begin
// ```graphviz
// digraph ConstructionSteps {
//     splines=true;
//     nodesep=0.5;
//     ranksep=0.4;
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
//           penwidth=1.4, fillcolor="#A0D6D1"];
// 
//     Gather [label="1. Gather\ndomain knowledge"];
//     Order [label="2. Order\nthe nodes"];
//     Pick [label="3. Pick\nminimal parents"];
//     Estimate [label="4. Estimate\nCPTs"];
//     Validate [label="5. Validate\nthe model"];
// 
//     Gather -> Order -> Pick -> Estimate -> Validate;
// }
// ```
// label=fig:stepstoconstructabayesiannetwork
// caption=Diagram relating 1. Gather
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.2.png",
    width: 70%,
  ),
  caption: [Diagram relating 1. Gather],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:stepstoconstructabayesiannetwork>
// render_images:end
domain knowledge, 2. Order the nodes, 3. Pick minimal parents and 4. Estimate
CPTs

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:209 '* Bayesian Networks: Properties'
// Slide: Bayesian Networks: Properties
#strong[Bayesian Networks: Properties]

Bayesian networks offer three properties that make them a practical tool for
probabilistic reasoning. First, they are #strong[complete]: a Bayesian network
encodes every piece of information contained in the full joint probability
distribution over its variables, so nothing is lost by working with the network
instead of the joint table. Second, they are #strong[consistent], meaning
non-redundant. Because each conditional probability table is specified
independently given the variable's parents, there is no way for a domain expert
constructing the network to accidentally introduce values that violate the
axioms of probability. Every valid parameterization of the network automatically
defines a legitimate distribution. Third, and most consequentially for
scalability, Bayesian networks are #strong[compact]. Each variable interacts
directly with only a limited number of other variables (its parents in the
graph), so the number of parameters typically grows linearly with the number of
variables rather than exponentially. In practice, modelers sometimes even choose
to omit a real-world dependency from the graph to keep the structure simple,
accepting a small approximation in exchange for a much more tractable model.

This compactness is what distinguishes a Bayesian network from a #emph[fully
  connected] system, in which every variable is influenced by every other
variable. A fully connected graph offers no structural savings: it requires the
same number of parameters as the raw joint probability table. The Bayesian
network's power lies precisely in its #emph[sparsity]; by encoding only the
dependencies that genuinely matter, it captures the same distributional
information with far fewer numbers, making both storage and inference feasible
for large domains.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:228 '* Ordering of Nodes'
// Slide: Ordering of Nodes
#strong[Ordering of Nodes]

The #strong[complexity] of a Bayesian network, measured by the number of edges
and the size of the conditional probability tables, depends heavily on the order
in which nodes are added to the graph. Three different orderings of the same
five variables illustrate this clearly: @fig:orderingofnodes shows the causal
order, @fig:orderingofnodes-2 shows the reversed causal order, and
@fig:orderingofnodes-3 shows an arbitrary order.

// rendered_images:begin
// ```graphviz
// digraph BayesianNetwork {
//     splines=true;
//     nodesep=0.8;
//     ranksep=0.8;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     // Nodes
//     Burglary   [label="Burglary", fillcolor="#A6C8F4"];
//     Alarm      [label="Alarm", fillcolor="#FFD1A6"];
//     JohnCalls   [label="JohnCalls", fillcolor="#B2E2B2"];
//     MaryCalls   [label="MaryCalls", fillcolor="#B2E2B2"];
//     Earthquake [label="Earthquake", fillcolor="#A6C8F4"];
// 
//     // Edges
//     Burglary -> Alarm;
//     Earthquake -> Alarm;
//     Alarm -> JohnCalls;
//     Alarm -> MaryCalls;
// }
// ```
// label=fig:orderingofnodes
// caption=Diagram relating Burglary, Alarm, JohnCalls
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.3.png",
    width: 70%,
  ),
  caption: [Diagram relating Burglary, Alarm, JohnCalls],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:orderingofnodes>
// render_images:end
and MaryCalls

// rendered_images:begin
// ```graphviz
// digraph BayesianNetwork {
//     splines=true;
//     nodesep=0.8;
//     ranksep=0.8;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     // Nodes
//     Burglary   [label="Burglary", fillcolor="#A6C8F4"];
//     Alarm      [label="Alarm", fillcolor="#FFD1A6"];
//     JohnCalls   [label="JohnCalls", fillcolor="#B2E2B2"];
//     MaryCalls   [label="MaryCalls", fillcolor="#B2E2B2"];
//     Earthquake [label="Earthquake", fillcolor="#A6C8F4"];
// 
//     // Edges
//     MaryCalls -> Alarm;
//     JohnCalls -> Alarm;
//     Alarm -> Burglary;
//     Alarm -> Earthquake;
//     Alarm -> MaryCalls;
//     Alarm -> JohnCalls;
// 
//     Burglary -> Alarm;
//     Earthquake -> Alarm;
// }
// ```
// label=fig:orderingofnodes-2
// caption=Diagram relating Burglary, Alarm, JohnCalls
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.4.png",
    width: 70%,
  ),
  caption: [Diagram relating Burglary, Alarm, JohnCalls],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:orderingofnodes-2>
// render_images:end
and MaryCalls

// rendered_images:begin
// ```graphviz
// digraph BayesianNetwork {
//     splines=true;
//     nodesep=0.8;
//     ranksep=0.8;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     // Nodes
//     Burglary   [label="Burglary", fillcolor="#A6C8F4"];
//     Alarm      [label="Alarm", fillcolor="#FFD1A6"];
//     JohnCalls   [label="JohnCalls", fillcolor="#B2E2B2"];
//     MaryCalls   [label="MaryCalls", fillcolor="#B2E2B2"];
//     Earthquake [label="Earthquake", fillcolor="#A6C8F4"];
// 
//     // Edges
//     MaryCalls -> Earthquake;
//     MaryCalls -> Burglary;
//     MaryCalls -> JohnCalls;
//     JohnCalls -> Earthquake;
//     Earthquake -> Burglary;
//     Earthquake -> Alarm;
//     Burglary -> Alarm;
//     Alarm -> MaryCalls;
//     Alarm -> JohnCalls;
// }
// ```
// label=fig:orderingofnodes-3
// caption=Diagram relating Burglary, Alarm, JohnCalls
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.5.png",
    width: 70%,
  ),
  caption: [Diagram relating Burglary, Alarm, JohnCalls],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:orderingofnodes-3>
// render_images:end
and MaryCalls

When variables are introduced in #emph[causal order], each node conditions only
on its direct causes, producing the fewest edges and the most compact network.
Reversing that order, as in @fig:orderingofnodes-2, forces the addition of
redundant edges: the network must encode the same joint distribution, so
conditioning on effects rather than causes requires larger parent sets to
capture the same dependencies. An #emph[arbitrary order], shown in
@fig:orderingofnodes-3, tends to produce the densest graph of all, because
variables that are neither causes nor effects of one another still end up linked
to preserve consistency with the true distribution. The graph is #emph[minimal]
in terms of connectivity when every edge runs from cause to effect, exactly as
in the causal-order diagram. This is a practical reason to prefer causal
knowledge when constructing Bayesian networks: it naturally yields the sparsest,
most interpretable structure.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:324 '* Causal vs Diagnostic Models'
// Slide: Causal vs Diagnostic Models
#strong[Causal vs Diagnostic Models]

A #strong[causal model] represents dependencies that flow from causes to
symptoms. For instance, a burglary causes an alarm to sound, written
$"Burglary" arrow.r "Alarm"$. Causal models tend to be simpler because they
involve fewer parameters and the dependencies they encode are more reliable:
the probability that an alarm goes off given a burglary is a relatively stable
physical fact about the alarm system, unlikely to shift with context.

A #strong[diagnostic model] reverses that direction, running from symptoms back
to causes. Here the reasoning starts from an observation (Mary calls) and works
backward toward what might have produced it ($"MaryCalls" arrow.r "Alarm"$ or
$"Alarm" arrow.r "Burglary"$). These conditional probabilities are more tenuous
because they depend on base rates and competing explanations that can change
across populations or settings, making them harder to estimate reliably.

Despite their instability, diagnostic probabilities are exactly what we care
about in practice: given that the alarm went off, what is the probability of a
burglary? The key insight is that we do not need to estimate these fragile
diagnostic probabilities directly. Instead, we specify the easier causal
direction ($P("Alarm" | "Burglary")$) and then apply Bayes' rule to invert it,
obtaining the diagnostic quantity ($P("Burglary" | "Alarm")$) from the causal
one. This inversion is the central reason Bayesian networks are built with
causal arrows: the network encodes the simple, stable, causal conditionals, and
probabilistic inference handles the reversal automatically.
@fig:causalvsdiagnosticmodels contrasts these two modeling directions.

// rendered_images:begin
// ```graphviz
// digraph CausalModel {
//     splines=true;
//     nodesep=2.0;
//     ranksep=1.5;
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
//     // Node styles
//     Causes [fillcolor="#B2E2B2"];
//     Symptoms [fillcolor="#F4A6A6"];
//     // Edges
//     Causes -> Symptoms [xlabel=<Causal<BR/>Model>, fontname="Helvetica"];
//     Symptoms -> Causes [xlabel=<Diagnostic<BR/>Model>, fontname="Helvetica"];
// }
// ```
// label=fig:causalvsdiagnosticmodels
// caption=Diagram illustrating Causal vs
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.6.png",
    width: 70%,
  ),
  caption: [Diagram illustrating Causal vs],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:causalvsdiagnosticmodels>
// render_images:end
Diagnostic Models

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:357 '# Markov Blankets'
// Slide: Markov Blankets
= Markov Blankets

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:359 '* Markov Blanket of a Node'
// Slide: Markov Blanket of a Node
#strong[Markov Blanket of a Node]

The #strong[Markov blanket] #cite("pearl1988probabilistic") of a node $X$ is the
minimal set of other nodes that, once their values are known, renders $X$
conditionally independent of every remaining variable in the network. It
consists of three groups:

1. #emph[Parents of $X$]: the nodes with direct edges into $X$, capturing every
  variable that directly influences it.
2. #emph[Children of $X$]: the nodes that $X$ directly points to, representing
  the variables it directly influences.
3. #emph[Spouses of $X$]: any other node that is also a parent of one of $X$'s
  children (a "co-parent"). These matter because observing a child activates an
  explaining-away pathway between its parents, so knowing a spouse's value can
  change what $X$'s child tells us about $X$.

Together, these three groups form a shield around $X$: no information from the
rest of the graph can reach $X$ without passing through at least one member of
its Markov blanket. @fig:markovblanketofanode shows this structure: parents,
children, and spouses surround $X$ and mediate all
probabilistic influence between it and the wider network.

// rendered_images:begin
// ```graphviz
// digraph CausalModel {
//   // Set overall graph properties
//   bgcolor="transparent";
//   rankdir=TB;
//   node [shape=ellipse, style=filled, fontname="Arial"];
// 
//   // Regular nodes
//   U1 [label=<U<SUB>1</SUB>>, fillcolor="#FF9999"];
//   Um [label=<U<SUB>m</SUB>>, fillcolor="#FF9999"];
//   X [label="X", fillcolor="#999999"];
//   Z1j [label=<Z<SUB>1j</SUB>>, fillcolor="#99CCFF"];
//   Znj [label=<Z<SUB>nj</SUB>>, fillcolor="#99CCFF"];
//   Y1 [label=<Y<SUB>1</SUB>>, fillcolor="#99FF99"];
//   Yn [label=<Y<SUB>n</SUB>>, fillcolor="#99FF99"];
// 
//   // Dots/ellipses as dummy nodes
//   node [shape=plaintext, style=solid, fontname="Arial", fillcolor=transparent];
//   dummy1 [label="..."];
//   dummy2 [label="..."];
//   dummy3 [label="..."];
//   dummy4 [label="..."];
//   dummy5 [label="..."];
//   dummy6 [label="..."];
//   dummy7 [label="..."];
//   dummy8 [label="..."];
//   dummy9 [label="..."];
//   dummy10 [label="..."];
//   dummy11 [label="..."];
//   dummy12 [label="..."];
// 
//   // Restore style for main nodes
//   node [shape=ellipse, style=filled, fontname="Arial"];
// 
//   // Define main edges
//   U1 -> X;
//   Um -> X;
//   X -> Y1;
//   X -> Yn;
//   Z1j -> Y1;
//   Znj -> Yn;
// 
//   dummy1 -> U1;
//   dummy2 -> Um;
//   dummy5 -> Z1j;
//   dummy6 -> Znj;
//   edge [style=solid];
// 
//   // Optional layout helpers
//   U1 -> dummy3;
//   Um -> dummy4;
//   Z1j -> dummy7;
//   Znj -> dummy8;
//   Y1 -> dummy9;
//   Y1 -> dummy10;
//   Yn -> dummy11;
//   Yn -> dummy12;
// }
// ```
// label=fig:markovblanketofanode
// caption=Diagram relating X and ...
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.7.png",
    width: 70%,
  ),
  caption: [Diagram relating X and ...],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:markovblanketofanode>
// render_images:end

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:437 '* Conditional Independence on Markov Blanket'
// Slide: Conditional Independence on Markov Blanket
#strong[Conditional Independence on Markov Blanket]

In a Bayesian network, each variable is #strong[conditionally independent] of
its predecessors given its parents. More broadly, a variable is conditionally
independent of every other node in the network given its #emph[Markov blanket]:
the set comprising its parents, its children, and its children's other parents
(sometimes called its "spouses"). These two properties are what make Bayesian
networks computationally tractable; without them, reasoning over a joint
distribution with many variables would require tracking an exponential number of
entries.

The #strong[Markov blanket] of a node $X_i$ contains exactly the nodes needed to
predict $X_i$'s state, rendering the rest of the network irrelevant. Once the
values of every node in the blanket are known, no other variable anywhere in the
graph can change the posterior distribution over $X_i$. This locality is what
enables #emph[efficient and localized inference]: an algorithm updating its
belief about $X_i$ only needs to gather evidence from the blanket, not from the
entire network. @fig:conditionalindependenceonmarkovblanket shows the
Markov blanket forming a shield around a node, separating it informationally from
every variable outside that boundary.

// rendered_images:begin
// ```graphviz
// digraph CausalModel {
//   // Set overall graph properties
//   bgcolor="transparent";
//   rankdir=TB;
//   node [shape=ellipse, style=filled, fontname="Arial"];
// 
//   // Regular nodes
//   U1 [label=<U<SUB>1</SUB>>, fillcolor="#FF9999"];
//   Um [label=<U<SUB>m</SUB>>, fillcolor="#FF9999"];
//   X [label="X", fillcolor="#999999"];
//   Z1j [label=<Z<SUB>1j</SUB>>, fillcolor="#99CCFF"];
//   Znj [label=<Z<SUB>nj</SUB>>, fillcolor="#99CCFF"];
//   Y1 [label=<Y<SUB>1</SUB>>, fillcolor="#99FF99"];
//   Yn [label=<Y<SUB>n</SUB>>, fillcolor="#99FF99"];
// 
//   // Dots/ellipses as dummy nodes
//   node [shape=plaintext, style=solid, fontname="Arial", fillcolor=transparent];
//   dummy1 [label="..."];
//   dummy2 [label="..."];
//   dummy3 [label="..."];
//   dummy4 [label="..."];
//   dummy5 [label="..."];
//   dummy6 [label="..."];
//   dummy7 [label="..."];
//   dummy8 [label="..."];
//   dummy9 [label="..."];
//   dummy10 [label="..."];
//   dummy11 [label="..."];
//   dummy12 [label="..."];
// 
//   // Restore style for main nodes
//   node [shape=ellipse, style=filled, fontname="Arial"];
// 
//   // Define main edges
//   U1 -> X;
//   Um -> X;
//   X -> Y1;
//   X -> Yn;
//   Z1j -> Y1;
//   Znj -> Yn;
// 
//   dummy1 -> U1;
//   dummy2 -> Um;
//   dummy5 -> Z1j;
//   dummy6 -> Znj;
//   edge [style=solid];
// 
//   // Optional layout helpers
//   U1 -> dummy3;
//   Um -> dummy4;
//   Z1j -> dummy7;
//   Znj -> dummy8;
//   Y1 -> dummy9;
//   Y1 -> dummy10;
//   Yn -> dummy11;
//   Yn -> dummy12;
// }
// ```
// label=fig:conditionalindependenceonmarkovblanket
// caption=Diagram relating X and
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.8.png",
    width: 70%,
  ),
  caption: [Diagram relating X and],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:conditionalindependenceonmarkovblanket>
// render_images:end
...

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:514 '* How Can a Node Be Influenced by Its Children?'
// Slide: How Can a Node Be Influenced by Its Children?
#strong[How Can a Node Be Influenced by Its Children?]

How can a node be influenced by its children? At first glance, the directed
edges in a Bayesian network point from parents to children, suggesting that
influence flows only downward. Yet a descendant can influence its ancestor
indirectly through a phenomenon called #strong[explaining away]. When you
observe evidence about a descendant, that evidence can change what you believe
about the ancestor by activating dependent paths that were previously blocked.
Information therefore flows both ways in a Bayesian network: generatively from
causes to effects through the graph's edges, and diagnostically from observed
effects back up to their possible causes through Bayesian inference.

Consider the Garden World shown in @fig:howcananodebeinfluencedbyitschildren.
Suppose you observe that the grass is wet ($"WetGrass"$). This observation
increases the probability of both possible causes: $"Rain"$ and $"Sprinkler"$.
Each cause becomes more plausible because either one could have produced the
observed effect. Now suppose you additionally learn that the $"Sprinkler"$ was
on. This new piece of evidence #emph[explains away] the wet grass: because the
sprinkler already accounts for the observation, the need to invoke rain as an
explanation diminishes, and the probability of $"Rain"$ drops back down. The
descendant node $"WetGrass"$ has effectively updated your belief about the
ancestor node $"Rain"$, even though no directed edge points from $"WetGrass"$ to
$"Rain"$.

// rendered_images:begin
// ```graphviz
// digraph BayesianFlow {
//     // rankdir=LR;
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
//     // Node styles
//     Rain [fillcolor="#A6C8F4", label="Rain"];
//     WetGrass [fillcolor="#B2E2B2", label="WetGrass"];
//     Sprinkler [fillcolor="#A6E7F4", label="Sprinkler"];
//     // Force ranks
//     // Edges
//     Rain -> WetGrass;
//     Sprinkler -> WetGrass;
// }
// ```
// label=fig:howcananodebeinfluencedbyitschildren
// caption=Diagram relating Rain,
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.9.png",
    width: 70%,
  ),
  caption: [Diagram relating Rain,],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:howcananodebeinfluencedbyitschildren>
// render_images:end
WetGrass and Sprinkler

This pattern arises whenever two or more causes share a common effect. Observing
the effect activates the path between those causes (opening a previously blocked
collider), and learning the state of one cause then changes the posterior
probability of the other. Explaining away is one of the most important reasoning
patterns in probabilistic inference, and it is a direct consequence of the way
Bayes' theorem propagates evidence through the joint distribution encoded by the
network's structure.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:556 '* Markov Blanket: Medical Example'
// Slide: Markov Blanket: Medical Example
#strong[Markov Blanket: Medical Example]

Consider the risk factors and outcomes for heart disease. The #strong[target
  node] in this example is $"HeartDisease"$. Its #emph[parent nodes] are the
risk factors that directly influence whether heart disease develops: $"Age"$,
$"Genetics"$, $"Diet"$, and $"Exercise"$. Each of these has a direct causal link
to $H$ (heart disease).

// rendered_images:begin
// ```graphviz
// digraph HeartDiseaseGraph {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     // Nodes
//     H [label="HeartDisease", fillcolor="#F4A6A6"];
//     A [label="Age", fillcolor="#A6C8F4"];
//     G [label="Genetics", fillcolor="#A6C8F4"];
//     D [label="Diet", fillcolor="#A6C8F4"];
//     E [label="Exercise", fillcolor="#A6C8F4"];
//     BP [label="BloodPressure", fillcolor="#B2E2B2"];
//     C [label="Cholesterol", fillcolor="#B2E2B2"];
// 
//     // Risk factors influencing Heart Disease
//     A -> H;
//     G -> H;
//     D -> H;
//     E -> H;
// 
//     // Heart Disease influencing outcomes
//     H -> BP;
//     H -> C;
// 
//     // Risk factors also influencing outcomes directly
//     A -> BP;
//     A -> C;
//     G -> BP;
//     G -> C;
//     D -> BP;
//     D -> C;
//     E -> BP;
//     E -> C;
// 
//     // Force ranks
//     {rank=same; A; G; D; E}
//     {rank=same; BP; C}
// }
// ```
// label=fig:markovblanketmedicalexample
// caption=Diagram relating HeartDisease,
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.10.png",
    width: 70%,
  ),
  caption: [Diagram relating HeartDisease,],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:markovblanketmedicalexample>
// render_images:end
Age, Genetics and Diet

The #emph[spouse nodes] in this network happen to coincide with the parent
nodes, because $"Age"$, $"Genetics"$, $"Diet"$, and $"Exercise"$ also directly
influence the outcomes $"BloodPressure"$ and $"Cholesterol"$. Those two outcome
variables are the #emph[children nodes] of $"HeartDisease"$: they represent
observable consequences that heart disease produces.
@fig:markovblanketmedicalexample shows the parents, children, and spouses
together forming the Markov Blanket around $"HeartDisease"$.

The practical implication is clear: once you know the state of every node in
this blanket (the risk factors, the outcomes, and the shared parents of those
outcomes), you can compute the probability of $"HeartDisease"$ without any other
information from the rest of the network. No matter how many additional
variables the full Bayesian network contains (lifestyle habits, medical history,
demographic data), they become conditionally irrelevant once the Markov Blanket
is observed.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:623 '* Markov Blanket: Economic Example'
// Slide: Markov Blanket: Economic Example
#strong[Markov Blanket: Economic Example]

Consider the factors affecting house prices in a particular region. The
#strong[target node] in this example is $"HousePrices"$, the variable whose
behavior we want to predict or explain.

The #emph[parent nodes], those that directly influence house prices, are
$"EconomicGrowth"$, $"InterestRate"$, and $"UnemploymentRate"$. Each of these
has a clear causal pathway to the target: strong economic growth tends to push
prices up, higher interest rates make mortgages more expensive and dampen
prices, and rising unemployment reduces the pool of qualified buyers.
@fig:markovblanketeconomicexample shows these three variables feeding
directly into the $"HousePrices"$ node in the network.

// rendered_images:begin
// ```graphviz
// digraph HousePriceGraph {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     // Nodes
//     HP [label="HousePrices", fillcolor="#F4A6A6"];
//     E [label="EconomicGrowth", fillcolor="#A6C8F4"];
//     IR [label="InterestRate", fillcolor="#A6C8F4"];
//     UE [label="UnemploymentRate", fillcolor="#A6C8F4"];
//     DI [label="DisposableIncome", fillcolor="#B2E2B2"];
//     D [label="HousingDemand", fillcolor="#B2E2B2"];
// 
//     // Edges
//     E -> HP;
//     IR -> HP;
//     UE -> HP;
// 
//     HP -> DI;
//     HP -> D;
// 
//     // Force ranks
//     {rank=same; E; IR; UE}
//     {rank=same; DI; D}
// }
// ```
// label=fig:markovblanketeconomicexample
// caption=Diagram relating HousePrices,
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.11.png",
    width: 70%,
  ),
  caption: [Diagram relating HousePrices,],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:markovblanketeconomicexample>
// render_images:end
EconomicGrowth, InterestRate and UnemploymentRate

The #emph[children nodes], variables that $"HousePrices"$ itself influences,
include $"DisposableIncome"$ and $"HousingDemand"$. House prices affect how much
money people have left after covering housing costs, so $"DisposableIncome"$
depends on the target. Similarly, higher prices can reduce the number of buyers
willing or able to enter the market, making $"HousingDemand"$ a downstream
consequence of the target node as well.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:676 '* Markov Blanket: Finance Example'
// Slide: Markov Blanket: Finance Example
#strong[Markov Blanket: Finance Example]

Consider the factors that affect an individual company's stock price. In this
example, the #strong[target node] is $S P$ (Stock Price), the variable whose
behavior we want to understand or predict.

The #emph[parent nodes], those that directly influence stock price, are $I P$
(Industry Performance), $E P S$ (Earnings Per Share), and $M S$ (Market
Sentiment). These three variables feed information directly into $S P$: a rising
industry tide, strong quarterly earnings, or a shift in investor mood each has a
direct causal path to the stock's price.

The #emph[child node] of $S P$ is $T V$ (Trading Volume). Changes in stock price
influence how much stock is being traded: a sharp price move, up or down,
typically triggers a burst of buying and selling activity, so $T V$ sits
downstream of $S P$ in the graph.

// rendered_images:begin
// ```graphviz
// digraph StockPriceGraph {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     // Nodes with abbreviations
//     SP [label="Stock Price", fillcolor="#F4A6A6"];
//     EPS [label="Earnings Per Share", fillcolor="#A6C8F4"];
//     IP [label="Industry Performance", fillcolor="#A6C8F4"];
//     MS [label="Market Sentiment", fillcolor="#A6C8F4"];
//     TV [label="Trading Volume", fillcolor="#B2E2B2"];
//     RC [label="Regulatory Changes", fillcolor="#C6A6F4"];
//     GE [label="Global Economic Conditions", fillcolor="#C6A6F4"];
// 
//     // Edges
//     EPS -> SP;
//     IP -> SP;
//     MS -> SP;
// 
//     SP -> TV;
// 
//     RC -> EPS;
//     RC -> IP;
//     GE -> EPS;
//     GE -> MS;
// 
//     // Force ranks
//     {rank=same; EPS; IP; MS}
//     {rank=same; RC; GE}
//     {rank=same; TV}
// }
// ```
// label=fig:markovblanketfinanceexample
// caption=Diagram relating Stock Price,
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.12.png",
    width: 70%,
  ),
  caption: [Diagram relating Stock Price,],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:markovblanketfinanceexample>
// render_images:end
Earnings Per Share, Industry Performance and Market Sentiment

@fig:markovblanketfinanceexample shows the network extending further.
The #emph[grandparent nodes], $R C$ (Regulatory Changes in the technology
sector) and $G E$ (Global Economic Conditions), sit one layer above the parents.
Regulatory changes influence $I P$ and $E P S$ but have no direct edge to $T V$.
Likewise, global economic conditions shape $M S$ and $E P S$ without directly
affecting trading volume.

The key insight is that you do not need to know $R C$ or $G E$ to estimate Stock
Price. Once you condition on the Markov blanket of $S P$ (its parents $I P$,
$E P S$, $M S$, its child $T V$, and any other parents of $T V$), the
grandparent nodes become conditionally independent of the target. All the
information they carry about $S P$ is already mediated through the blanket. This
is exactly what makes the Markov blanket so useful in practice: it tells you the
minimal set of variables you actually need to observe, letting you safely ignore
everything outside it without losing predictive power.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:738 '# Compact Representation of Conditional Probability Tables'
// Slide: Compact Representation of Conditional Probability Tables
= Compact Representation of Conditional Probability Tables

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:742 '* Specifying a Conditional Probability Table'
// Slide: Specifying a Conditional Probability Table
#strong[Specifying a Conditional Probability Table]

A #strong[conditional probability table] (CPT) for a node with $k$ parents
requires $O(2^k)$ entries in the worst case #cite("koller2009pgm"), one row for
every combination of parent values. Even a modest number of parents makes the
table unwieldy to specify by hand.

Consider the $"Alarm"$ node from the burglary network, which has just $k = 2$
parents. Even this simple case already demands $2^2 = 4$ rows to fully specify
$Pr("Alarm" = T mid B, E)$:

#figure(
  styled-table(
    headers: ([Burglary], [Earthquake], [$Pr("Alarm"=T mid B, E)$]),
    rows: (
      ([T], [T], [0.95]),
      ([T], [F], [0.94]),
      ([T], [F], [0.29]),
      ([F], [F], [0.001]),
    ),
  ),
  caption: [CPT for the alarm node with two binary parents.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:alarmcpt>

@tab:alarmcpt shows that a node with only two parents is already a four-row
table. With five parents the table balloons to 32 rows; with ten, over a
thousand. Eliciting that many probabilities from a domain expert is impractical,
and estimating them from data demands correspondingly large sample sizes.

Fortunately, real-world relationships among variables are rarely completely
arbitrary. Several structural patterns let us compress a CPT far below its
worst-case size:

- #emph[Deterministic nodes], whose value is a fixed function of their parents,
  need no probability table at all.
- #emph[Noisy logical relationships] (such as noisy-OR or noisy-AND), which
  model a default logical combination perturbed by independent noise, replace an
  exponential table with a number of parameters that grows only linearly in $k$.
- #emph[Context-specific independence], where a node ignores some parents once
  another parent takes a particular value, lets entire blocks of the CPT
  collapse into shared entries.

Each of these patterns exploits regularity in the domain to make large Bayesian
networks feasible to build and maintain.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:764 '* Deterministic Nodes'
// Slide: Deterministic Nodes
#strong[Deterministic Nodes]

A #strong[deterministic node] is a node whose value is fully specified by its
parents, with no uncertainty involved. Unlike ordinary chance nodes,
deterministic nodes do not involve randomness or probability; their values
follow directly from a fixed rule applied to their inputs. They appear
frequently in models as a way to simplify relationships and computations by
separating the deterministic logic from the genuinely stochastic parts of the
network.

Deterministic relationships come in two common forms. The first is a
#emph[logical relationship], useful when a condition holds if any of several
sub-conditions are true. For instance, a node representing North American
citizenship might be defined as:

$ "IsNorthAmerican" = "IsCanadian" or "IsUS" or "IsMexican" $

The second is a #emph[numerical relationship], where the node's value is
computed from its parents by a mathematical function. A best-price node, for
example, simply takes the minimum over a set of price inputs:

$ "BestPrice" = min("Price"_i) $

In both cases the node adds no new source of variation to the model: once its
parents are known, its own value is completely determined.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:777 '* Noisy Logical Relationships'
// Slide: Noisy Logical Relationships
#strong[Noisy Logical Relationships]

#strong[Noisy logical relationships], such as noisy-OR and noisy-MAX, provide a
probabilistic version of a logical relationship. Rather than stating that a
child variable is deterministically true whenever any parent is true, these
models allow each parent to contribute to the child's activation with some
probability of failure. This formulation can be substantially simpler to specify
than a full conditional probability table when a node has $k$ parents, because
the number of parameters grows linearly in $k$ rather than exponentially.

To see where noisy logical relationships come from, consider how propositional
logic handles causation. A logical statement such as
$"Fever" arrow.l.r.double "Cold" or "Flu" or "Malaria"$ asserts that fever is
present if and only if at least one of these diseases is present. The
relationship is crisp: any single cause suffices, and there is no room for
uncertainty.

Bayesian networks soften this picture by introducing three assumptions. First,
all possible causes of the child node are listed among its parents (a #emph[leak
  node] can be added to capture #emph[miscellaneous causes] not explicitly
modeled). Second, each parent's ability to produce the effect is uncertain: a
patient with the flu might or might not develop a fever. Third, the inhibition
probabilities of different parents are independent of one another, so the chance
that cold fails to cause fever does not depend on whether flu also fails to
cause fever.

Under these three assumptions the conditional probability of fever given its
parents decomposes into a product of independent "failure" terms. Specifically:

$
  Pr("Fever" | "parents"("Fever")) = 1 - &Pr(not "Fever" | "Cold", not "Flu", not "Malaria") dot.op \
  &Pr(not "Fever" | not "Cold", "Flu", not "Malaria") dot.op \
  &Pr(not "Fever" | not "Cold", not "Flu", "Malaria")
$

Each factor in the product is the probability that a single active cause
#emph[fails] to produce fever when it is the only cause present. The overall
probability of fever is then one minus the probability that #emph[every] active
cause independently fails. This is exactly the probabilistic analogue of a
logical OR: the child is "off" only when all causes fail to fire, and it is "on"
otherwise. Because each cause contributes just one inhibition parameter,
specifying the full noisy-OR model for $k$ parents requires only $k$ numbers
instead of the $2^k$ entries a general conditional probability table would need.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:801 '* Context-Specific Independence'
// Slide: Context-Specific Independence
#strong[Context-Specific Independence]

A variable exhibits #strong[context-specific independence] (CSI) if it is
conditionally independent of one or more of its parents given certain values of
other variables. In other words, the full conditional probability table may
simplify dramatically once a particular context is fixed, even though the
general case requires conditioning on every parent.

Consider the variable $"Damage"$, which depends on both the $"Ruggedness"$ of a
car and whether an $"Accident"$ occurred in a given period:

$
  Pr("Damage" | "Ruggedness", "Accident") = cases(
    d_1 & "if Accident" = "True",
    d_2("Ruggedness") & "if Accident" = "False",
  )
$

where $d_1$ and $d_2$ are distributions. When an accident has occurred, the
distribution over damage is just $d_1$, which does not involve $"Ruggedness"$ at
all. That is, $"Damage"$ is conditionally independent of $"Ruggedness"$ given
$"Accident" = "True"$. Only in the no-accident context does the ruggedness of
the car matter (through $d_2$). This is a textbook instance of context-specific
independence: the independence holds in one specific context
($"Accident" = "True"$) but not globally. Exploiting CSI lets inference
algorithms skip irrelevant parent combinations, reducing the effective size of
conditional probability tables and speeding up computation without sacrificing
exactness.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:818 '* Bayesian Networks with Continuous Variables'
// Slide: Bayesian Networks with Continuous Variables
#strong[Bayesian Networks with Continuous Variables]

Many real-world problems involve continuous quantities: height, mass,
temperature, money. These variables take values from uncountable domains, so the
discrete machinery developed so far needs adaptation.

The core difficulty is that a Conditional Probability Table (CPT) cannot
represent a continuous random variable. A CPT enumerates every combination of
parent values and assigns a probability to each; when a variable can take
infinitely many values, that enumeration is impossible. Two broad strategies
address this. First, #emph[discretization] bins the continuous range into
intervals and treats each bin as a discrete value. This works but sacrifices
accuracy (the bin boundaries are arbitrary) and can produce very large CPTs when
fine granularity is needed. Second, one can work with #emph[continuous
  variables] directly by specifying a family of probability density functions,
such as the Gaussian distribution, whose parameters (mean and variance)
compactly encode the entire conditional distribution. When no standard
parametric family fits, #emph[non-parametric] density estimates offer a flexible
alternative at the cost of more data and computation.

In practice, many domains mix both kinds of variable. A #strong[hybrid Bayesian
  network] contains both discrete and continuous nodes in the same graph. For
instance, a customer buys some number of apples (a discrete count) whose
distribution depends on the price per kilogram (a continuous cost variable).
Similarly, an insurance company deciding the annual premium to charge for a
vehicle (continuous) conditions on discrete applicant information such as the
car's maker and model. Hybrid networks require inference algorithms that can
handle the interplay between discrete conditioning and continuous densities, a
topic the next sections develop in detail.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:839 '* Bayesian Network: Car Insurance Company (1/2)'
// Slide: Bayesian Network: Car Insurance Company (1/2)
#strong[Bayesian Network: Car Insurance Company (1/2)]

Consider a car insurance company that receives an application from an individual
to insure a specific vehicle. The company must analyze information about the
applicant and the car, decide on an appropriate annual premium to charge, and
later pay out claims based on the type of incident. This is a classic decision
problem under uncertainty: the insurer cannot observe every factor that
determines risk, yet must still set a price that reflects expected costs.

The solution is to build a Bayesian network that captures the #emph[causal
  structure] of the insurance domain. Rather than treating every variable as an
opaque feature in a flat regression, a Bayesian network encodes which factors
cause or influence which others, letting the insurer reason about unobserved
quantities through their observable consequences.

The network's nodes fall into three natural groups. First, the #emph[input
  information] the company can directly observe:

- #emph[About the applicant]: $"Age"$, $"YearsWithLicense"$, $"DrivingRecord"$,
  $"GoodStudent"$
- #emph[About the vehicle]: $"MakeModel"$, $"VehicleYear"$, $"Airbag"$,
  $"SafetyFeatures"$
- #emph[About the driving situation]: $"Mileage"$, $"HasGarage"$

Second, #strong[unobservable information] plays a critical role. Some of the
most important factors driving risk are never directly available on an
application form. $"RiskAversion"$ captures how cautiously a person behaves in
general, while $"DrivingBehavior"$ reflects their actual habits behind the wheel
(speeding, tailgating, distracted driving). Neither appears on any form the
applicant fills out, yet both strongly influence accident probability. The
Bayesian network handles this gracefully: by connecting these latent variables
to their observable consequences (a clean driving record, a good-student
discount, choice of a vehicle with extra safety features), the network can infer
likely values for $"RiskAversion"$ and $"DrivingBehavior"$ from the evidence
that is available.

Third, the network models three distinct #emph[types of claims] the insurer must
pay:

- $"MedicalCost"$: injuries sustained by the applicant themselves
- $"LiabilityCost"$: lawsuits filed by other parties against the applicant
- $"PropertyCost"$: vehicle damage to either party, as well as theft of the
  insured vehicle

Each claim type has different parent variables in the network. Medical costs
depend heavily on airbag presence and safety features; liability costs depend on
driving behavior and the severity of accidents involving other parties; property
costs depend on the vehicle's value and whether it is garaged. By separating
these three cost nodes rather than lumping them into a single "total claim"
variable, the network can price each component of the premium according to the
specific risk factors that drive it, producing a more accurate and justifiable
rate.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:863 '* Bayesian Network: Car Insurance Company (2/2)'
// Slide: Bayesian Network: Car Insurance Company (2/2)
#strong[Bayesian Network: Car Insurance Company (2/2)]

In this network, the nodes are color-coded to distinguish their roles.
#emph[Blue nodes] represent information provided directly by the applicants,
such as age, driving history, and vehicle type. #emph[Brown nodes] represent
hidden variables that are not directly observable but influence the outcomes,
capturing latent risk factors the insurer cannot measure from the application
alone. #emph[Violet nodes] represent the target variables the insurance company
ultimately wants to predict, such as the likelihood of a claim or the expected
cost of coverage. @fig:bayesiannetworkcarinsurancecompany22 shows how
these three categories of variables connect within the Bayesian network for the
car insurance domain, making explicit which quantities are observed, which must
be inferred, and which constitute the prediction targets.

// rendered_images:begin
// ```graphviz[width=80%]
// digraph InsuranceRiskModel {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.2];
// 
//     // Define node colors
//     Age [fillcolor="#A6C8F4"];
//     GoodStudent [fillcolor="#A6C8F4"];
//     YearsLicensed [fillcolor="#A6C8F4"];
//     DrivingRecord [fillcolor="#A6C8F4"];
//     Mileage [fillcolor="#A6C8F4"];
//     SafetyFeatures [fillcolor="#A6C8F4"];
//     MakeModel [fillcolor="#A6C8F4"];
//     VehicleYear [fillcolor="#A6C8F4"];
//     CarValue [fillcolor="#A6C8F4"];
//     Airbag [fillcolor="#A6C8F4"];
//     AntiTheft [fillcolor="#A6C8F4"];
//     Garaged [fillcolor="#A6C8F4"];
//     ExtraCar [fillcolor="#A6C8F4"];
// 
//     RiskAversion [fillcolor="#FFD1A6"];
//     DrivingSkill [fillcolor="#FFD1A6"];
//     DrivingBehavior [fillcolor="#FFD1A6"];
//     Ruggedness [fillcolor="#FFD1A6"];
//     Theft [fillcolor="#FFD1A6"];
//     Cushioning [fillcolor="#FFD1A6"];
//     OwnCarDamage [fillcolor="#FFD1A6"];
//     OtherCost [fillcolor="#FFD1A6"];
//     Accident [fillcolor="#FFD1A6"];
//     SocioEcon [fillcolor="#FFD1A6"];
// 
//     MedicalCost [fillcolor="#C6A6F4"];
//     LiabilityCost [fillcolor="#C6A6F4"];
//     PropertyCost [fillcolor="#C6A6F4"];
//     OwnCarCost [fillcolor="#C6A6F4"];
// 
//     // Define edges
//     Age -> YearsLicensed;
//     Age -> DrivingSkill;
//     Age -> GoodStudent;
//     Age -> RiskAversion;
// 
//     YearsLicensed -> DrivingSkill;
//     DrivingSkill -> DrivingRecord;
//     DrivingSkill -> DrivingBehavior;
// 
//     DrivingRecord -> DrivingBehavior;
//     DrivingBehavior -> Accident;
// 
//     RiskAversion -> Garaged;
//     RiskAversion -> AntiTheft;
// 
//     Garaged -> Theft;
//     AntiTheft -> Theft;
// 
//     Mileage -> Ruggedness;
//     SafetyFeatures -> Ruggedness;
// 
//     SocioEcon -> RiskAversion;
//     SocioEcon -> MakeModel;
//     SocioEcon -> ExtraCar;
// 
//     MakeModel -> VehicleYear;
//     MakeModel -> SafetyFeatures;
//     MakeModel -> Ruggedness;
// 
//     VehicleYear -> CarValue;
//     CarValue -> Ruggedness;
//     CarValue -> Airbag;
// 
//     Ruggedness -> OwnCarDamage;
//     Airbag -> Cushioning;
//     Cushioning -> Accident;
// 
//     Accident -> MedicalCost;
//     Accident -> LiabilityCost;
//     Accident -> PropertyCost;
//     Accident -> OtherCost;
// 
//     OwnCarDamage -> OwnCarCost;
//     OwnCarCost -> PropertyCost;
//     Theft -> OwnCarDamage;
// }
// ```
// label=fig:bayesiannetworkcarinsurancecompany22
// caption=Diagram illustrating
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.13.png",
    width: 80%,
  ),
  caption: [Diagram illustrating],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:bayesiannetworkcarinsurancecompany22>
// render_images:end
Bayesian Network: Car Insurance Company (2/2)

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:956 '# Exact Inference in Bayesian Networks'
// Slide: Exact Inference in Bayesian Networks
= Exact Inference in Bayesian Networks

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:960 '* The Inference Problem'
// Slide: The Inference Problem
#strong[The Inference Problem]

How do we compute the #strong[posterior] $P(X | bold(E) = bold(e))$ for a query
variable $X$, given observed evidence $bold(e)$? The setup involves three groups
of variables: the query variable $X$ whose distribution we want, the evidence
variables $bold(E) = {E_1, dots, E_m}$ whose values we have observed, and the
hidden variables $bold(Y) = {Y_1, dots, Y_ell}$ that are neither queried nor
observed #cite("russell2020aima").

The most straightforward approach is #strong[inference by enumeration]: use the
full joint distribution and marginalize (sum) over all hidden variables:

$ P(X | e) = alpha sum_Y P(X, e, Y) $

This computes the correct answer by exhaustively iterating over every
combination of values for the hidden variables, but the cost grows exponentially
with the number of those variables.

#strong[Variable elimination] improves on enumeration by caching intermediate
results and eliminating variables systematically, avoiding the redundant
summations that make naive enumeration expensive. A key optimization is removing
irrelevant variables entirely: any variable that is not an ancestor of either
the query or the evidence in the Bayes net can be safely ignored, since it
contributes nothing to the posterior.

These exact inference methods come with real limitations. For tree-structured
networks, exact inference runs in $O(n)$ time, but for general networks the
problem is intractable at $O(2^n)$. Exact methods also do not extend to
continuous variables without additional machinery (such as discretization or
closed-form parametric assumptions). When exact computation becomes impractical,
these algorithms nonetheless serve as the conceptual foundation for the
approximate methods (such as sampling) that take their place.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:985 '* Exact Inference in Bayesian Networks: Example'
// Slide: Exact Inference in Bayesian Networks: Example
#strong[Exact Inference in Bayesian Networks: Example]

Suppose you receive a call from both John and Mary: what is the probability of a
burglary? Formally, the query is
$P("Burglary" | "JohnCalls" = "True", "MaryCalls" = "True")$. Computing this
from the network requires summing over the full joint distribution. Any
conditional probability can be obtained by marginalizing over the hidden
variables:

$
  Pr(X | bold(e)) = alpha Pr(X, bold(e)) = alpha sum_(bold(y)) Pr(X, bold(e), bold(y))
$

where $alpha$ is a normalizing constant and the sum runs over every combination
of values for the variables $bold(y)$ not mentioned in the query or the
evidence. @fig:exactinferenceinbayesiannetworksexample shows the five-node
network whose conditional probability tables (CPTs) supply every factor needed
for this calculation.

// rendered_images:begin
// ```graphviz[width=70%]
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//     Burglary [label="Burglary", fillcolor="#A6C8F4"];
//     Earthquake [label="Earthquake", fillcolor="#FFD1A6"];
//     Alarm [label="Alarm", fillcolor="#B2E2B2"];
//     JohnCalls [label="JohnCalls", fillcolor="#C6A6F4"];
//     MaryCalls [label="MaryCalls", fillcolor="#C6A6F4"];
// 
//     { rank = same; Burglary; Earthquake; }
//     { rank = same; JohnCalls; MaryCalls; }
// 
//     Burglary -> Alarm;
//     Earthquake -> Alarm;
//     Alarm -> JohnCalls;
//     Alarm -> MaryCalls;
// }
// ```
// label=fig:exactinferenceinbayesiannetworksexample
// caption=Diagram relating
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.14.png",
    width: 70%,
  ),
  caption: [Diagram relating],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:exactinferenceinbayesiannetworksexample>
// render_images:end
Burglary, Earthquake, Alarm and JohnCalls

The joint distribution decomposes into a product of conditional probabilities,
one per node in the Bayesian network, thanks to the chain rule encoded by the
graph's structure. Applying this to the burglary query gives:

$ Pr(b | j, m) = alpha Pr(B, j, m) = alpha sum_e sum_a Pr(B, j, m, e, a) $

Each term inside the sum is then rewritten using the CPTs that define the
network:

$
  Pr(b | j, m) = alpha sum_e sum_a Pr(b) Pr(e) Pr(a | b, e) Pr(j | a) Pr(m | a)
$

Reading this expression from left to right, $Pr(b)$ is the prior probability of
burglary, $Pr(e)$ is the prior for earthquake, $Pr(a | b, e)$ is the alarm's CPT
conditioned on its two parents, and $Pr(j | a)$ and $Pr(m | a)$ are the calling
probabilities conditioned on the alarm. Evaluating the double sum means
iterating over all combinations of the hidden variables $e$ (earthquake) and $a$
(alarm), multiplying the five CPT entries for each combination, and accumulating
the results. After summing, the normalizing constant $alpha$ scales the two
values $Pr(b = "true" | j, m)$ and $Pr(b = "false" | j, m)$ so they add to one,
yielding the posterior distribution over Burglary.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1038 '# Approximate Inference in Bayesian Networks'
// Slide: Approximate Inference in Bayesian Networks
= Approximate Inference in Bayesian Networks

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1040 '* Monte Carlo Algorithms'
// Slide: Monte Carlo Algorithms
#strong[Monte Carlo Algorithms]

#strong[Monte Carlo algorithms] #cite("koller2009pgm") are randomized sampling
algorithms used to estimate quantities that are difficult to calculate exactly.
Rather than computing a closed-form solution, these methods draw random samples
from a distribution and use those samples to build up an empirical approximation
of the target quantity. A common application is drawing samples from the
posterior probability distribution of a Bayes network, where exact inference may
be intractable due to the network's size or structure.

The appeal of Monte Carlo methods lies in their flexibility: the accuracy of the
approximation depends on the number of samples generated, and with enough
samples one can get arbitrarily close to the true probability distribution. This
convergence guarantee, grounded in the law of large numbers, makes the approach
applicable across many branches of science, from statistical physics to
computational biology to financial modeling. The tradeoff is that Monte Carlo
methods can be computationally intensive, particularly in high-dimensional
spaces where many samples are needed before the estimates stabilize. They also
make it difficult to understand how the variables interact, since the method
produces numerical estimates rather than an interpretable analytical expression
relating the variables to one another.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1057 '* Sampling from Arbitrary Distributions'
// Slide: Sampling from Arbitrary Distributions
#strong[Sampling from Arbitrary Distributions]

How do you actually draw a sample from a probability distribution, whether
discrete or continuous? The core idea is to start with a uniform random number
$r in [0, 1]$, which every programming language can produce, and then transform
it through the #strong[cumulative distribution function] (CDF)
$F(x) = Pr(X lt.eq x)$ to obtain a sample that follows the target distribution.

For #emph[discrete distributions], you build a table of outcomes alongside their
cumulative probabilities, then find the smallest outcome whose cumulative
probability exceeds $r$. For instance, if a die has faces with probabilities
0.1, 0.2, 0.3, 0.2, 0.1, 0.1, the cumulative sums are 0.1, 0.3, 0.6, 0.8, 0.9,
1.0; drawing $r = 0.45$ lands in the third bucket, so you return outcome 3.

For #emph[continuous distributions], the technique is the #strong[inverse
  transform method]: set $x = F^(-1)(r)$. Because $F$ maps the distribution's
domain onto $[0, 1]$, its inverse maps a uniform draw back onto a properly
distributed sample. As a concrete case, the exponential distribution has CDF
$F(x) = 1 - e^(-lambda x)$, which inverts to

$ x = F^(-1)(r) = -1/lambda ln(1 - r) $

so a single logarithm turns a uniform draw into an exponential sample. When the
inverse $F^(-1)$ has no closed-form expression, as happens with the Gaussian
CDF, you fall back on #emph[numerical methods] such as bisection search on $F$
or purpose-built approximations like the Box–Muller transform.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1079 '* Sampling Bayesian Network Without Evidence'
// Slide: Sampling Bayesian Network Without Evidence
#strong[Sampling Bayesian Network Without Evidence]

How can we generate samples from a Bayesian network when no evidence is
observed? The answer is #strong[prior sampling], a straightforward procedure
that walks the network in topological order and draws each variable from its
local distribution.

The process works as follows. Start with the source nodes, the variables that
have no parents. These have known unconditional probability distributions that
can be sampled directly: for instance, $Pr("Rain") = 0.5$ lets us flip a fair
coin to decide whether it is raining. Then move to each child node in turn. A
conditional variable's distribution depends on whatever values its parents were
assigned in previous steps: if Rain was set to true, we might sample WetGrass
from $Pr("WetGrass" | "Rain" = T) = 0.1$. By the time every node has been
visited, we have one complete assignment of values to all variables in the
network.

// rendered_images:begin
// ```graphviz
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//     Rain       [label="Rain",       fillcolor="#A6C8F4"];
//     Sprinkler  [label="Sprinkler",  fillcolor="#FFD1A6"];
//     WetGrass   [label="WetGrass",   fillcolor="#B2E2B2"];
// 
//     { rank = same; Rain; Sprinkler; }
// 
//     Rain      -> WetGrass;
//     Sprinkler -> WetGrass;
// }
// ```
// label=fig:samplingbayesiannetworkwithoutevidence
// caption=Diagram relating Rain,
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.15.png",
    width: 70%,
  ),
  caption: [Diagram relating Rain,],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:samplingbayesiannetworkwithoutevidence>
// render_images:end
Sprinkler and WetGrass

@fig:samplingbayesiannetworkwithoutevidence shows a small network with
Rain, Sprinkler, and WetGrass, with the parent-child dependencies that
determine the sampling order.

The key insight is that prior sampling directly implements the Bayesian
network's semantics. Each sample is drawn from the full joint probability
distribution:

$ f_("PS")(x_1, ..., x_n) = product_(i=1)^n Pr(x_i | "parents"(X_i)) $

where PS stands for #emph[prior sampling]. Because the network's joint
distribution factorizes as exactly this product of conditional probability
tables, sampling each variable conditioned on its already-sampled parents
produces draws from the correct joint. As the number of samples grows, the
empirical frequencies of any event converge to its true probability under the
model. The method is simple and unbiased, though it says nothing yet about how
to handle observed evidence; that requires the rejection and
likelihood-weighting extensions discussed next.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1124 '* Consistency of Sampling'
// Slide: Consistency of Sampling
#strong[Consistency of Sampling]

#strong[Consistency of estimation] guarantees that the distribution obtained
from prior sampling converges to the true joint probability as the number of
samples grows without bound. Concretely, if $N_(P S)$ denotes the number of
times a particular event $(x_1, dots, x_n)$ occurs across $N$ total samples,
then

$ lim_(N arrow.r oo) frac(N_(P S)(x_1, dots, x_n), N) = Pr(x_1, dots, x_n) $

This is a direct consequence of the law of large numbers: each sample is drawn
independently from the joint distribution defined by the Bayesian network, so
the empirical frequency of any complete assignment must converge to its true
probability. In practice, of course, we work with a finite sample budget, so we
use the approximation

$ Pr(x_1, dots, x_n) approx frac(N_(P S)(x_1, dots, x_n), N) $

which converges at a rate of $O(1 slash sqrt(N))$. That rate has a concrete
implication: to cut the estimation error in half, you need roughly four times as
many samples. For low-dimensional problems this is perfectly workable, but in
high-dimensional networks, where the number of possible joint assignments grows
exponentially, the count $N_(P S)$ for any single assignment can remain very
small even for large $N$, making the finite-sample estimate unreliable despite
the asymptotic guarantee.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1140 '* Rejection Sampling'
// Slide: Rejection Sampling
#strong[Rejection Sampling]

#strong[Rejection sampling] is a method for drawing samples from a distribution
that is difficult to sample from directly. The core goal is to compute the
conditional probability $Pr(X = x | E = e)$ when the evidence $e$ is rare,
making direct estimation impractical.

The algorithm proceeds in three steps:

1. #emph[Generate samples from the prior distribution.] Draw a large number of
  complete variable assignments from the Bayesian network's joint distribution,
  with no conditioning. These samples collectively estimate $Pr(x, e)$, the
  joint probability of every combination of query and evidence values.
2. #emph[Reject samples that do not match the observed evidence.] Any sample
  where $E eq.not e$ is discarded. The surviving samples, those where $E = e$,
  now form an empirical estimate of $Pr(X, E = e)$.
3. #emph[Count occurrences of the query value among the surviving samples.]
  Within the retained set, tally how many samples satisfy $X = x$. The fraction
  of retained samples with $X = x$ gives the estimate of $Pr(X = x | E = e)$.

The intuition is straightforward: by throwing away every sample that disagrees
with the evidence, the remaining pool behaves as if it had been drawn from the
conditional distribution all along. As the number of prior samples grows, the
estimate converges to the true conditional probability. The practical
difficulty, however, is that when $e$ is a low-probability event, the vast
majority of samples are rejected, and the method needs an enormous number of
draws before enough survivors accumulate to produce a reliable estimate.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1158 '* Rejection Sampling: Estimating Pi'
// Slide: Rejection Sampling: Estimating Pi
#strong[Rejection Sampling: Estimating Pi]

Rejection sampling is a straightforward technique for drawing samples from a
distribution that is difficult to sample from directly. The core idea is to use
a simpler #strong[proposal distribution] $q(x)$ from which samples can be
generated easily, and then accept or reject each sample based on how well it
matches the target distribution $p(x)$.

The procedure works as follows. First, choose a proposal distribution $q(x)$ and
find a constant $M$ such that $M dot.op q(x) gt.eq p(x)$ for all $x$. Then, for
each sample: draw a candidate $x$ from $q(x)$, draw a uniform random number $u$
from $[0, 1]$, and accept $x$ if $u lt.eq p(x) / (M dot.op q(x))$; otherwise,
reject it and try again. The accepted samples follow the target distribution
$p(x)$ exactly, with no approximation error.

#figure(
  image("../lectures_source/figures/L06.2.Rejection_Sampling.png", width: 80%),
  caption: [Rejection Sampling],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:rejectionsampling>

@fig:rejectionsampling shows the method: sample uniformly
under the scaled proposal envelope $M dot.op q(x)$ and keep only those points
that fall beneath the target density $p(x)$. The acceptance rate equals $1 / M$,
so the tighter the envelope fits the target, the fewer samples are wasted. In
low dimensions with a well-chosen proposal, rejection sampling is both simple to
implement and provably correct. The tradeoff is that in high-dimensional spaces,
even the best constant $M$ tends to be very large, causing the acceptance rate
to drop exponentially and making the method impractical without more
sophisticated alternatives such as MCMC.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1164 '* Rejection Sampling: Garden World Example'
// Slide: Rejection Sampling: Garden World Example
#strong[Rejection Sampling: Garden World Example]

Suppose you want to estimate $Pr("Rain" | "Sprinkler" = T)$. You draw 100
samples from the full joint distribution of the network, but only those samples
where Sprinkler is true are relevant to the query. Of the 100 samples, 73 have
$not "Sprinkler"$ and must be discarded, leaving just 27 usable samples where
Sprinkler is true. Among those 27, 8 happen to have Rain and 19 have
$not "Rain"$. The estimate is therefore

$ Pr("Rain" | "Sprinkler") = 8 / 27 $

This example exposes the core inefficiency of rejection sampling: nearly three
quarters of the computational work went into generating samples that were
immediately thrown away because they did not match the evidence. The usable
sample size (27 out of 100) is small enough that the resulting estimate carries
substantial variance. Had the evidence involved a rarer event, the acceptance
rate would drop even further, and thousands or millions of total samples might
be needed to accumulate a handful of accepted ones. The method remains correct
in the limit, since every accepted sample is a genuine draw from the conditional
distribution, but the cost of reaching that limit grows rapidly as the
probability of the evidence shrinks.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1176 '* Rejection Sampling: Pros and Cons'
// Slide: Rejection Sampling: Pros and Cons
#strong[Rejection Sampling: Pros and Cons]

Rejection sampling offers a clear tradeoff between simplicity and efficiency. On
one hand, it produces a #emph[consistent estimate]: as the number of generated
samples grows, the approximation converges to the true posterior value. This is
a desirable theoretical guarantee, since it means the method is at least correct
in the limit.

On the other hand, the practical cost can be severe. Many samples are rejected
outright because they fail to match the observed evidence $e$. How many depends
on how rare $Pr(E = e)$ is: the less probable the evidence, the more samples the
algorithm must discard before finding one that agrees with it. Worse, the
fraction of accepted samples #emph[decreases exponentially] as the number of
evidence variables grows. This is a manifestation of the #emph[curse of
  dimensionality]: each additional observed variable tightens the filter, making
it exponentially harder for a randomly generated sample to pass. For complex
systems with many observations, rejection sampling becomes impractical because
nearly every sample is thrown away.

Continuous variables introduce a further difficulty. In theory, the probability
of a continuous variable taking any single exact value is zero
($Pr(E = e) = 0$), so a naive check for equality would reject every sample. In
practice, floating-point precision limits make exact matches essentially
impossible, meaning that rejection sampling requires binning or tolerance
thresholds to work with continuous evidence at all. These workarounds add
complexity and introduce their own approximation errors, undermining the
method's original appeal as a straightforward sampling strategy.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1190 '* Importance Sampling'
// Slide: Importance Sampling
#strong[Importance Sampling]

#strong[Importance sampling] is a technique that draws samples from a simpler,
more convenient distribution $Q(X)$ rather than the target distribution $Pr(X)$.
To correct for the mismatch between these two distributions, each sample is
assigned an #emph[importance weight] $w = Pr(X) / Q(X)$. The expected value of
any function $f(X)$ under the true distribution can then be estimated by
averaging these weighted samples:

$ E[f(X)] approx 1 / N sum_(i=1)^N w_i f(X_i) $

#figure(
  image("../lectures_source/figures/L06.2.Importance_Sampling.png", width: 80%),
  caption: [Importance Sampling],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:importancesampling>

@fig:importancesampling captures the core idea: concentrate sampling
effort where it matters most. Consider estimating $Pr(A | E = e)$ when the event
$E = e$ is rare. Standard forward sampling would generate enormous numbers of
samples before stumbling onto one where $E = e$, wasting nearly all
computational effort. Importance sampling sidesteps this by choosing a proposal
distribution $Q$ that places more probability mass near $E = e$, then correcting
via the importance weights so the final estimate remains unbiased with respect
to the true distribution.

A useful analogy: suppose you are running a survey but your random sample
drastically underrepresents certain demographic groups. Rather than discarding
the data and resampling (as rejection sampling would), you keep every response
but give underrepresented groups proportionally higher weights when computing
your statistics. The result is a corrected estimate that uses all collected
data.

This approach improves inference efficiency over rejection sampling, which
discards every sample that fails to match the desired condition. Because
importance sampling retains and reweights all drawn samples, it extracts more
information per sample and converges faster, particularly in problems where the
evidence or query event occupies a small region of the joint distribution.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1216 '* Markov Chain Monte Carlo'
// Slide: Markov Chain Monte Carlo
#strong[Markov Chain Monte Carlo]

MCMC sampling stands as one of the #emph[most influential algorithms] in
computer science, ranked alongside Quicksort and the Fast Fourier Transform. Its
origins trace back to the Manhattan Project in the 1940s, where Stanislaw Ulam,
John von Neumann, Nicholas Metropolis, and others invented it to solve problems
involving high-dimensional integrals and, eventually, Bayesian inference.

The goal of MCMC is #strong[approximate inference] for Bayesian networks in
cases where exact inference is computationally intractable. Unlike rejection
sampling or importance sampling, which generate each sample independently from a
proposal distribution, MCMC works by making random changes to the preceding
sample. Each new sample is a small perturbation of the last, creating a chain of
correlated draws that, over time, converges to the target distribution. This
sequential dependence is what makes MCMC a fundamentally different strategy:
rather than hoping that independent proposals land in high-probability regions,
MCMC explores the distribution by wandering through it one step at a time. The
deeper insight connecting these ideas is striking: two very different
mathematical objects, Markov chains (a memoryless stochastic process defined by
transition probabilities) and Bayesian networks (a structured probabilistic
graphical model encoding conditional independencies), turn out to be intimately
linked. A Markov chain whose stationary distribution matches the posterior of a
Bayesian network gives us a way to draw samples from that posterior without ever
computing it exactly.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1231 '* Markov Chain Construction'
// Slide: Markov Chain Construction
#strong[Markov Chain Construction]

A #strong[Markov chain] is a random walk through a state space in which the
future depends only on the present state, not on how the chain arrived there.
The chain produces a sequence of states $x^((0)), x^((1)), x^((2)), dots$
starting from some initial configuration. At each step, a #emph[transition
  operator] moves the chain from its current state $x$ to a new state $x'$
according to transition probabilities $Pr(x arrow.r x')$. After $t$ steps, the
chain induces a distribution $pi_t (x)$ over states. When that distribution
stops changing, so that $pi_t (x) = pi_(t+1)(x)$, the chain has reached its
#strong[stationary distribution], the fixed point of the transition dynamics
illustrated in @fig:markovchainconstruction.

// rendered_images:begin
// ```graphviz[width=90%]
// digraph MarkovChain {
//     rankdir=LR;
//     splines=true;
//     nodesep=0.6;
//     ranksep=0.6;
//     node [shape=circle, style=filled, fontname="Helvetica", fontsize=12,
//           penwidth=1.4, fillcolor="#A0D6D1"];
// 
//     x0 [label=<x<SUP>(0)</SUP>>];
//     x1 [label=<x<SUP>(1)</SUP>>];
//     x2 [label=<x<SUP>(2)</SUP>>];
//     dots [shape=plaintext, label="...", fillcolor=transparent];
//     xt [label=<x<SUP>(t)</SUP>>, fillcolor="#A6C8F4"];
// 
//     x0 -> x1 [label="Pr(x->x')"];
//     x1 -> x2;
//     x2 -> dots;
//     dots -> xt [label="stationary"];
// }
// ```
// label=fig:markovchainconstruction
// caption=Diagram relating ..., Pr(x->x') and
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.16.png",
    width: 90%,
  ),
  caption: [Diagram relating ..., Pr(x->x') and],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:markovchainconstruction>
// render_images:end
stationary

Two widely used transition operators turn a Bayesian network into a Markov chain
that targets the posterior distribution. #emph[Gibbs sampling] resamples a
single variable conditioned on its Markov blanket, leaving all other variables
unchanged; cycling through variables one at a time produces a new full state at
each step. #emph[Metropolis–Hastings] takes a different approach: it proposes a
candidate state from some proposal distribution, then accepts or rejects that
candidate with a probability computed from the ratio of the target densities at
the proposed and current states. Both operators define valid Markov chains whose
samples, over time, approximate the distribution of interest.

For the stationary distribution of the chain to equal the posterior distribution
over non-evidence variables given the evidence, two regularity conditions must
hold:

- #emph[Ergodicity]: the chain must be able to reach every state with positive
  probability from every other state, so no region of the state space is
  permanently cut off.
- #emph[Aperiodicity]: the chain must not cycle deterministically through a
  fixed sequence of states; it needs the freedom to revisit any state at
  irregular intervals.

When both conditions are satisfied, the chain's long-run distribution converges
to the true posterior regardless of where it started, which is what makes MCMC a
practical tool for approximate inference in complex Bayesian networks.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1277 '* Markov Chain Monte Carlo: Mixing'
// Slide: Markov Chain Monte Carlo: Mixing
#strong[Markov Chain Monte Carlo: Mixing]

#strong[Mixing] describes how quickly a Markov chain forgets its starting point
and explores the entire state space efficiently. A well-mixed chain moves
frequently between different high-probability regions and produces samples with
low autocorrelation. Poor mixing, by contrast, means the chain becomes trapped
in a single mode for many iterations, yielding biased estimates and high
variance in any quantities computed from those samples.

In practice, even a well-designed sampler needs time to reach its stationary
distribution. The standard remedy is to discard an initial set of samples known
as the #emph[burn-in period], treating only the post-convergence draws as
approximate samples from the true posterior. How long that burn-in must be
depends directly on how quickly the chain mixes: a fast-mixing chain needs fewer
discarded samples before its output becomes trustworthy.

Consider sampling from a bimodal distribution as a concrete illustration. A
poorly mixing chain tends to settle into one of the two peaks and remain there,
drastically under-representing the other mode. A well-mixing chain, on the other
hand, jumps between both peaks often enough that the collected samples
faithfully reflect the true posterior's shape. @fig:mcmcmixing shows this
contrast: a chain that transitions freely between modes produces a
far more accurate picture of the target distribution than one that lingers in a
single region.

#figure(
  image("../lectures_source/figures/L06.2.MCMC_mixing.png", width: 80%),
  caption: [MCMC mixing],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:mcmcmixing>

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1304 '* Gibbs Sampling in Bayesian Networks'
// Slide: Gibbs Sampling in Bayesian Networks
#strong[Gibbs Sampling in Bayesian Networks]

#strong[Gibbs sampling] is a special case of Markov Chain Monte Carlo (MCMC)
that samples one variable at a time, cycling through the non-evidence variables
while holding the observed evidence fixed. The procedure starts with an initial
complete assignment to all non-evidence variables. Then, for each non-evidence
variable $X_i$, a new value is drawn from $Pr(X_i | M B(X_i))$, where $M B(X_i)$
is the #emph[Markov blanket] of $X_i$: its parents, children, and children's
other parents (spouses). Because each update conditions only on the Markov
blanket, every sampling step is a purely local computation, regardless of the
network's overall size.

Consider the classic weather network with variables $"Cloudy"$, $"Sprinkler"$,
$"Rain"$, and $"WetGrass"$. Suppose we observe $"WetGrass" = "true"$ and
$"Sprinkler" = "true"$. Gibbs sampling fixes those two evidence variables and
iteratively resamples the remaining ones: first draw a new value for $"Cloudy"$
from $Pr("Cloudy" | "Sprinkler", "Rain", "WetGrass")$, then draw $"Rain"$ from
its own Markov-blanket conditional, and repeat. Over many iterations the
sequence of sampled states converges to the true posterior distribution over
$"Cloudy"$ and $"Rain"$ given the evidence.

The method is straightforward to implement for any Bayesian network because the
only distribution one ever needs to evaluate is a single variable's conditional
given its Markov blanket, and that conditional can be computed from the local
CPTs without a global inference pass. This locality also means updates scale
well to large, complex graphs. The tradeoff is that Gibbs sampling can mix
slowly when variables are highly correlated: the sampler gets "stuck" in one
region of the joint distribution and takes a long time to explore alternatives.
In practice this means many samples may be required before the empirical
averages become accurate estimates of the posterior, particularly in tightly
coupled networks where a single local change barely shifts the global
configuration.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1327 '* Metropolis–Hastings Sampling'
// Slide: Metropolis–Hastings Sampling
#strong[Metropolis–Hastings Sampling]

#strong[Metropolis–Hastings sampling] is a more general Markov Chain Monte Carlo
method than Gibbs sampling. Where Gibbs sampling updates one variable at a time
by drawing from its exact conditional, Metropolis–Hastings allows arbitrary
proposal moves and decides whether to accept or reject each one, making it
applicable to a far wider class of distributions.

The algorithm proceeds as follows:

1. Start at a current state $bold(x)$.
2. Propose a new state $bold(x')$ from a #emph[proposal distribution]
  $q(x' | x)$. For instance, with 95% probability the proposal might be a Gibbs
  sampling step, while the remaining 5% uses importance sampling to attempt
  larger jumps.
3. Compute the #emph[acceptance probability]:
  $ A(x, x') = min(1, (pi(x') q(x | x')) / (pi(x) q(x' | x))) $
4. Move to $bold(x')$ with probability $A(x, x')$; otherwise remain at
  $bold(x)$.

The ratio inside the acceptance formula compares how "good" the proposed state
is (under the target distribution $pi$) against how likely the proposal
mechanism was to suggest this particular move versus its reverse. When the new
state has higher target probability and is no harder to reach than to return
from, the ratio exceeds one and the move is always accepted. When the new state
has lower probability, the move is accepted only sometimes, in proportion to how
much lower it is.

This accept/reject mechanism is the key to balancing exploration and
exploitation. Uphill moves toward higher-probability regions are always taken,
while occasional downhill moves into #emph[lower-probability states] prevent the
chain from getting trapped in a single local mode. Over many iterations the
chain visits states in proportion to $pi$, regardless of where it started.

Metropolis–Hastings is very flexible: it works with any proposal distribution,
and it can handle high-dimensional spaces where direct sampling is infeasible.
The tradeoff is that performance depends heavily on the choice of proposal. If
proposal steps are too small, the chain explores the space slowly (poor
"mixing"); if they are too large, most proposals land in low-probability regions
and are rejected, so the chain barely moves at all. Practical use therefore
requires careful tuning of the proposal distribution, balancing step size
against acceptance rate to achieve efficient sampling.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1355 '* Summary'
// Slide: Summary
= Summary

The central insight of Bayesian networks is that a joint distribution over many
variables can be represented compactly by exploiting conditional independencies
encoded in a directed acyclic graph. Rather than storing an exponentially large
joint probability table, each variable carries only a small conditional
probability table (CPT) conditioned on its parents in the graph, and the full
joint is recovered as the product of these local terms.

Several foundational ideas make this representation practical. The
#emph[semantics] of a Bayesian network rest on a factorization:
$Pr(X_1, dots, X_n) = product_(i=1)^n Pr(X_i | "Parents"(X_i))$, where each
variable is conditionally independent of all its non-descendants given its
parents. The #emph[construction] process follows a causal ordering: nodes are
added in an order that reflects causal priority, each node is connected to the
minimal set of parents that captures its direct influences, and the CPTs are
then estimated from data or domain knowledge. The #strong[Markov blanket] of a
node, consisting of its parents, children, and children's other parents
(spouses), identifies the minimal set of variables needed to predict that node
given everything else in the network.

As the number of parents grows, even local CPTs can become unwieldy. Several
techniques keep their size manageable: #emph[deterministic nodes] whose values
are fixed functions of their parents require no table at all; #emph[noisy-OR]
and #emph[noisy-MAX] models decompose a multi-parent interaction into
independent failure probabilities, reducing an exponential table to a linear
number of parameters; and #emph[context-specific independence] allows certain
parent configurations to be grouped together when they produce the same
conditional distribution, further compressing the representation.

Answering queries against a Bayesian network requires inference. #emph[Exact
  inference] methods such as enumeration and variable elimination compute
posterior probabilities by summing out hidden variables in a carefully chosen
order. These approaches are efficient when the network has a tree-like
structure, but for densely connected graphs the problem is intractable in the
worst case. This motivates #emph[approximate inference] via Monte Carlo
sampling. Rejection sampling generates complete assignments from the prior and
discards those inconsistent with the evidence, while importance sampling
reweights samples rather than discarding them, improving efficiency. Markov
chain Monte Carlo (MCMC) methods, including Gibbs sampling (which resamples each
variable in turn from its Markov blanket) and Metropolis-Hastings (which
proposes arbitrary moves and accepts them with a correction ratio), provide a
general-purpose framework for drawing samples from complex posterior
distributions.

// From: msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd:1376 '* References'
// Slide: References
= References

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
