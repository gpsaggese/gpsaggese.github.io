// git_hash=2117c1bd timestamp=20260915_101833
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
  title: "L03.3: Non-classical Logics and Knowledge Representation",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L03.3: Non-classical Logics and Knowledge Representation")

= Roadmap

This chapter explores logical systems that relax the assumptions of classical
propositional and first-order logic to handle real-world reasoning challenges:
incomplete information, evolving knowledge, and default assumptions.
#emph[Non-monotonic and default reasoning] allow conclusions to be retracted when new
facts arrive, enabling systems to work with incomplete information and common-sense
knowledge. #emph[Inductive logic programming] turns the problem around, learning
logical rules directly from examples and background knowledge. #emph[Description
  logics] provide decidable fragments of first-order logic tailored for ontologies and
knowledge organization. #emph[Knowledge representation standards] like RDF and OWL
enable the Semantic Web, allowing machines to understand and reason over structured
data shared across the internet. Finally,
#emph[semantic networks and knowledge graphs] ground these formal systems in
large-scale, practical implementations that power modern AI applications.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:13 '# Non-classical Logics'
// Slide: Non-classical Logics
= Non-classical Logics

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:15 '* Motivation'
// Slide: Motivation
= Motivation

Classical logic, whether propositional or first-order, assumes a complete, static
world in which conclusions, once proven, never need to be retracted. Real-world
knowledge rarely cooperates with that assumption: it is incomplete, riddled with
exceptions, and evolves as new facts arrive. The stock example is the default rule
#emph[birds fly]: it holds perfectly well until you learn about penguins, at which
point a conclusion you had already drawn must be withdrawn. Classical logic has no
mechanism for that withdrawal, because adding a new axiom can only ever increase the
set of derivable theorems, never shrink it.

The key idea behind the techniques in this chapter is to relax classical logic along
two axes. The first is #strong[ontological commitment]: what kinds of entities and
relationships the logic assumes exist in the world. The second is
#strong[epistemological commitment]: what an agent is allowed to believe about facts,
including the possibility that some facts are simply unknown rather than true or
false. By loosening one or both commitments, we obtain formalisms that can handle
defaults, exceptions, and incomplete information gracefully.

This chapter surveys several such formalisms and the practical standards built on top
of them:

- #emph[Non-monotonic and default reasoning], which allow conclusions to be retracted
  when new information arrives, along with the related notion of common-sense
  reasoning and the distinction between open-world and closed-world assumptions.
- #emph[Inductive logic programming], which induces logical rules from positive and
  negative examples together with background knowledge, combining learning and
  reasoning in a single formalism.
- #emph[Description logics] (notably the ALC and SHOIN families) and the Web Ontology
  Language (OWL), which provide decidable fragments of first-order logic tailored for
  defining and reasoning about concept hierarchies.
- #emph[Knowledge representation standards] such as RDF and SPARQL, which give a
  concrete syntax and query language for encoding and retrieving structured knowledge
  on the web.
- #emph[Knowledge graphs and semantic networks], including WordNet, ConceptNet,
  WikiData, and DBpedia, which organize large-scale real-world knowledge into graph
  structures that both humans and machines can navigate and query.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:37 '* Ontological Commitment'
// Slide: Ontological Commitment
= Ontological Commitment

#strong[Ontological commitments] are the assumptions about reality that a formal
language makes #cite("russell2020aima"). Every logical system carries an implicit
stance on what exists in the world and how truth is determined; different formalisms
carve up reality in fundamentally different ways.

- #emph[Propositional logic] assumes the world consists of facts that are either true
  or false. A sentence such as $P or Q$ asserts a disjunction between two atomic
  propositions, with no internal structure beyond their truth values.
- #emph[First-order logic] enriches this picture by assuming the world contains
  objects that stand in relations to one another, where each relation either holds or
  does not. The sentence $forall x: "Human"(x) arrow.r.double "Mortal"(x)$ quantifies
  over individual objects and predicates properties of them, something propositional
  logic cannot express.
- #emph[Higher-order logic] goes a step further: relations themselves become objects
  that can be quantified over and reasoned about. One can write assertions such as
  "all relations are transitive," treating the relation itself as an entity in the
  domain.
- #emph[Temporal logic] adds a time dimension, allowing facts to hold at particular
  times or intervals. A statement like "rain occurred at time $t$" is expressible
  directly, whereas the other formalisms would need auxiliary encoding to capture the
  same idea.

Each step in this hierarchy expands the ontology: propositional logic commits only to
bare facts, first-order logic adds objects and relations, higher-order logic promotes
relations to first-class citizens, and temporal logic layers time over any of these.
The tradeoff is that richer ontological commitments buy more expressive power at the
cost of greater computational complexity in reasoning. As @fig:ontologicalcommitment
illustrates, these logical systems form a progression of increasingly detailed
assumptions about the structure of reality.

// rendered_images:begin
// ```graphviz
// digraph OntologicalLevels {
//   graph [rankdir=LR, splines=curved, bgcolor="transparent",
//          ranksep="1.1 equally", nodesep=0.26, pad=0.4, fontname="Helvetica"];
//   node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=15,
//          fontcolor="#26215C", color="#D8D6EE", penwidth=1.2,
//          height=0.52, margin="0.24,0.10"];
//   edge  [arrowhead=vee, penwidth=1.6, color="#B9B6D6"];
//
//   root [label="Logical Systems", shape=box, style="rounded,filled",
//         fillcolor="#26215C", fontcolor="white", fontsize=18, penwidth=0,
//         margin="0.34,0.20"];
//
//   prop [label="Propositional Logic\n(facts: true/false)", fillcolor="#EFEDFC", color="#7C74D6"];
//   fol [label="First-Order Logic\n(objects & relations)", fillcolor="#EFEDFC", color="#7C74D6"];
//   hol [label="Higher-Order Logic\n(relations as objects)", fillcolor="#E8F1FB", color="#3E86C8"];
//   temp [label="Temporal Logic\n(facts at times)", fillcolor="#E8F1FB", color="#3E86C8"];
//
//   root -> prop [color="#7C74D6", penwidth=2.0];
//   root -> fol [color="#7C74D6", penwidth=2.0];
//   root -> hol [color="#3E86C8", penwidth=2.0];
//   root -> temp [color="#3E86C8", penwidth=2.0];
// }
// ```
// label=fig:ontologicalcommitment
// caption=Logical systems ranked by increasing ontological commitment, from propositional to temporal logic.
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.1.png",
    width: 70%,
  ),
  caption: [Logical systems ranked by increasing ontological commitment, from
    propositional to temporal logic.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:ontologicalcommitment>
// render_images:end

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:88 '* Epistemological Commitment'
// Slide: Epistemological Commitment
= Epistemological Commitment

An #strong[epistemological commitment] describes the possible states of knowledge an
agent can hold with respect to each individual fact. Where the ontological commitment
fixes what the world is made of, the epistemological commitment fixes what a reasoner
is allowed to believe about it. Different representation languages draw that line in
different places:

- #emph[Propositional] and #emph[first-order logic] admit exactly three states of
  belief: a sentence is true, false, or unknown.
- #emph[Probability theory] takes a finer-grained stance, assigning each sentence a
  degree of belief anywhere in the interval $[0, 1]$. For instance, $Pr(X = 6) = 0.3$
  expresses moderate uncertainty about a single outcome rather than forcing a hard
  true-or-false judgment.
- #emph[Temporal logic] keeps the same three belief states as propositional and
  first-order logic, but attaches them to facts indexed by time. The sentence "the
  light is ON at time $t_1$" is true, false, or unknown independently of what holds
  at $t_2$.
- #emph[Fuzzy logic] replaces uncertainty about a fact with a known degree of truth in
  $[0, 1]$. The sentence "the room is warm" holds to degree $0.7$: this is not a bet
  on whether the room is warm, but an assertion that warmth itself is a matter of
  degree.

The last two are worth separating carefully. Probability theory and fuzzy logic both
attach a number in $[0, 1]$ to a sentence, but they mean different things by it: a
probability quantifies ignorance about a sharp fact, while a fuzzy truth value
quantifies how well a vague predicate applies to a perfectly known situation.

// Book-only grouping: the source .smd dropped this heading, but the book keeps it to
// group the three slides below.
== Non-monotonic Reasoning and Common Sense

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:104 '* Non-monotonic Logic'
// Slide: Non-monotonic Logic
#strong[Non-monotonic logic] is a logic in which adding new information can
invalidate conclusions that were previously derived #cite(
  "mccarthy1980circumscription",
). This stands in sharp contrast to classical (monotonic) logic, where once a
statement is proven from a set of axioms, no additional axiom can ever retract it. In
classical logic, the set of theorems only grows as premises are added; in
non-monotonic logic, that guarantee is deliberately abandoned so that a reasoner can
revise its beliefs when better evidence arrives.

The motivation is straightforward: real-world reasoning almost always operates on
incomplete or evolving knowledge. A doctor's diagnosis changes when lab results come
back; a robot's plan changes when it discovers a blocked corridor. Non-monotonic
logic gives formal systems the flexibility to draw tentative conclusions from what is
known now, then retract and replace those conclusions gracefully when new facts
emerge.

// rendered_images:begin
// ```graphviz
// digraph NonMonotonicReasoning {
//     rankdir=TB;
//     node [shape=box, style=filled, fontname=Helvetica];
//     edge [fontname=Helvetica];
//
//     kb1 [label="Initial KB:\nBirds typically fly\nTweety is a bird", fillcolor="#FFD1A6"];
//     conc1 [label="Conclusion:\nTweety can fly", fillcolor="#B2E2B2"];
//
//     newFact [label="New Fact:\nTweety is a penguin\nPenguins cannot fly", fillcolor="#F4A6A6"];
//
//     kb2 [label="Updated KB", fillcolor="#FFD1A6"];
//     conc2 [label="Revised Conclusion:\nTweety CANNOT fly", fillcolor="#A6E7F4"];
//
//     kb1 -> conc1 [label="reasoning"];
//     conc1 -> newFact [label="conflict"];
//     newFact -> kb2 [label="update"];
//     kb2 -> conc2 [label="revise"];
// }
// ```
// label=fig:nonmonotoniclogic
// caption=A non-monotonic reasoner revising its conclusion about Tweety after learning she is a penguin.
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.2.png",
    width: 70%,
  ),
  caption: [A non-monotonic reasoner revising its conclusion about Tweety after
    learning she is a penguin.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:nonmonotoniclogic>
// render_images:end

Consider the classic example illustrated in @fig:nonmonotoniclogic. Suppose the
knowledge base contains two statements: "birds typically fly" and "Tweety is a bird."
A non-monotonic reasoner concludes that Tweety can fly, because nothing contradicts
the default rule. Now a new fact arrives: "Tweety is a penguin," together with the
rule "penguins cannot fly." The reasoner retracts its earlier conclusion and derives
instead that Tweety cannot fly. In a monotonic system, the original conclusion would
persist alongside the new facts, producing an outright contradiction. Non-monotonic
logic avoids this by treating the initial inference as defeasible: it held only in
the absence of more specific information.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:125 '* Default Reasoning'
// Slide: Default Reasoning
#strong[Default reasoning] makes assumptions in the absence of contrary evidence,
allowing a system to draw conclusions based on what is typical rather than waiting for
complete information #cite("reiter1980default").

Reiter's formulation encodes each default as a rule with three parts:

$ (alpha : beta) / gamma $

read as: if the prerequisite $alpha$ holds and the justification $beta$ is consistent
with everything currently known, then conclude $gamma$. The justification is what
makes the rule defeasible. As long as nothing contradicts $beta$, the conclusion
$gamma$ stands; the moment new information rules $beta$ out, $gamma$ is withdrawn.
This is exactly what allows a system to function reasonably without complete
information: it commits to the typical case, but keeps that commitment revisable.

Consider an access-control example. The default rule states that, typically, employees
can access the building. The system learns that Sam is an employee, so the
prerequisite holds, nothing contradicts the justification, and it concludes that Sam
can access the building. Later a new fact arrives: Sam's badge was reported lost. The
justification no longer survives, the earlier conclusion is retracted, and the system
now holds that Sam cannot access the building. This kind of retraction is what makes
default reasoning #emph[non-monotonic]: adding new information can shrink, not just
grow, the set of beliefs.

Real-world agents rarely have access to every relevant fact before they need to act or
answer a query. By encoding what is normally true and handling exceptions as they
arise, a default reasoner behaves sensibly in the common case while remaining open to
correction.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:144 '* Non-Monotonic Logic: University Example'
// Slide: Non-Monotonic Logic: University Example
A university-registration scenario makes this pattern concrete. The initial facts
establish that $"Alice"$ is a $"Student"$ belonging to the $"ComputerScience"$
department, and that $"CS101"$ is a $"Course"$ offered by that same department. A
default rule states that every student in a department takes all courses that
department offers. Under this rule, since Alice is a computer science student, the
system concludes $"takesCourse"("Alice", "CS101")$. However,
when new information arrives indicating that Alice does not meet the prerequisites
for $"CS101"$, the earlier default conclusion is retracted. The revised reasoning
yields $not "takesCourse"("Alice", "CS101")$, overriding the previous default with
the more specific exception. This illustrates a core feature of non-monotonic
reasoning: adding new facts can invalidate previously drawn conclusions rather than
merely extending them.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:156 '* Common Sense Reasoning'
// Slide: Common Sense Reasoning
=== Common Sense Reasoning

Much of what humans know about the world is never stated explicitly: if you drop a
glass, it will likely break; people eat food when hungry. These facts feel too
obvious to mention, yet they underpin virtually every inference a person makes in
daily life. #strong[Common sense reasoning] is the capacity to make assumptions and
draw conclusions based on this kind of everyday knowledge about the world, filling in
gaps that formal logic alone cannot bridge.

Common sense reasoning has several distinctive characteristics. It deals with
incomplete, uncertain, and ambiguous information rather than clean, fully specified
inputs. It relies on defaults, heuristics, and typical patterns instead of strict
proofs, which means it trades guarantees for practical coverage. It is also flexible
and tolerant of exceptions: a system using common sense can accept that birds
typically fly without crashing when it encounters a penguin.

These same qualities make common sense reasoning extraordinarily difficult to
automate. The underlying knowledge is vast, informal, and imprecisely defined, so
there is no compact axiom set to start from. Encoding that knowledge in a
machine-readable form has proven to be one of AI's most persistent bottlenecks;
decades of effort on projects like Cyc showed just how much manual work is involved.
Even when some knowledge is captured, handling exceptions and contradictions adds
another layer of complexity: defaults interact, contexts shift, and a rule that holds
"usually" can fail in ways that are hard to anticipate.

Researchers have attacked the problem from several angles. #emph[Knowledge graphs]
organize facts as structured networks of entities and relations, making at least some
common sense retrievable. #emph[Non-monotonic logic] allows conclusions to be
retracted when new information arrives, modeling the defeasible nature of everyday
reasoning. #emph[Probabilistic reasoning] quantifies uncertainty directly, assigning
likelihoods to default assumptions. More recently, #emph[machine learning],
particularly large language models trained on broad text corpora, has shown
surprising facility with common sense tasks, though whether these models truly
"understand" common sense or merely pattern-match remains an open question.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:177 '* Common Sense Reasoning: Restaurant Example'
// Slide: Common Sense Reasoning: Restaurant Example
The restaurant scenario illustrates how common sense reasoning works in practice.
Suppose the initial facts are that Bob enters a restaurant and sits at a table.
Drawing on everyday knowledge, a reasoning system knows that customers who sit down
typically intend to order, that waiters bring a menu before taking an order, and that
customers pay before leaving. From Bob sitting at the table, the system infers that
he intends to eat, and therefore that he will receive a menu and place an order. As
@fig:commonsensereasoningrestaurantexample shows, this chain of inference flows
naturally from the initial observations through common sense defaults to a concrete
prediction about Bob's behavior.

// rendered_images:begin
// ```graphviz
// digraph RestaurantExample {
//     rankdir=TB;
//     node [shape=box, style=filled, fontname=Helvetica];
//     edge [fontname=Helvetica, fontsize=10];
//
//     bob_enters [label="Bob enters\nRestaurant", fillcolor="#FFD1A6"];
//     bob_sits [label="Bob sits at\nTable", fillcolor="#FFD1A6"];
//
//     common_sense [label="Common Sense:\nSit at table\n→ intends to eat", fillcolor="#C6A6F4"];
//
//     infer1 [label="Infer: Bob\nintends to eat", fillcolor="#B2E2B2"];
//     menu [label="Server brings\nMenu", fillcolor="#B2E2B2"];
//     order [label="Bob places\nOrder", fillcolor="#B2E2B2"];
//
//     new_info [label="NEW: Bob asks\nfor directions\nonly", fillcolor="#F4A6A6"];
//     revise [label="REVISE:\nNo order!", fillcolor="#A6E7F4"];
//
//     bob_enters -> bob_sits;
//     bob_sits -> common_sense [label="apply"];
//     common_sense -> infer1 [label="conclude"];
//     infer1 -> menu;
//     menu -> order;
//
//     order -> new_info [label="contradicts"];
//     new_info -> revise [label="revise"];
// }
// ```
// label=fig:commonsensereasoningrestaurantexample
// caption=Common-sense inference chain from Bob sitting down to placing an order, revised on new information.
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.3.png",
    width: 70%,
  ),
  caption: [Common-sense inference chain from Bob sitting down to placing an order,
    revised on new information.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:commonsensereasoningrestaurantexample>
// render_images:end

What makes this example interesting is what happens next. New information arrives:
Bob actually just asks for directions. At that point, the system must revise its
earlier conclusion. Bob will #emph[not] order after all. This is #strong[non-monotonic
  reasoning] in action: adding a new fact does not merely extend the set of
conclusions but retracts one that was previously valid. Classical logic is monotonic,
meaning that once something is proved it stays proved regardless of what else is
learned. Common sense reasoning cannot afford that rigidity; it must treat earlier
inferences as defeasible defaults, open to revision when contradicted by more
specific or more recent evidence.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:188 '* Open World vs Closed World Assumptions'
// Slide: Open World vs Closed World Assumptions
A related question in knowledge representation is how a system should treat a fact it
was never told: should silence be read as denial, or merely as ignorance? These two
stances, the #strong[closed-world] and #strong[open-world assumptions], give opposite
answers.

#grid(
  columns: (1fr, 50%),
  column-gutter: 1em,
)[
  The distinction between these two assumptions becomes concrete with a simple
  enrollment example. Suppose the only recorded fact is "Alice takes CS101." Under
  the #emph[closed-world assumption], silence is denial: because nothing is said
  about Bob, the system concludes "Bob does not take CS101." The absence of a
  positive statement is treated as a negative one. Under the #emph[open-world
    assumption], silence is ignorance: the system acknowledges that Bob's enrollment
  status is simply unknown, and he may or may not be enrolled. Neither answer is
  assumed until evidence arrives. @tab:openworldvsclosedworldassumptions summarizes
  how the two assumptions diverge across several key aspects, including how they
  treat missing information, what unstated facts imply, and where each assumption is
  most naturally applied.
][
  #figure(
    styled-table(
      headers: (
        "Aspect",
        "Closed World Assumption (CWA)",
        "Open World Assumption (OWA)",
      ),
      rows: (
        ("Missing info", "False by default", "Unknown (not false)"),
        (
          "Example: Bob takes CS101?",
          "FALSE (not stated → false)",
          "UNKNOWN (not stated → unknown)",
        ),
        (
          "DB systems",
          "Relational DB (SQL), logic programs",
          "Semantic Web (RDF, OWL)",
        ),
        ("Best for", "Complete, static knowledge", "Incomplete, evolving data"),
      ),
    ),
    caption: [Closed world assumption vs open world assumption],
    kind: "table",
    supplement: [Table.],
    placement: auto,
  ) <tab:openworldvsclosedworldassumptions>
]

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:212 '* Inductive Logic Programming'
// Slide: Inductive Logic Programming
A different challenge in knowledge representation is not how to treat unstated facts,
but how to learn new logical rules directly from data. #strong[Inductive logic
  programming] (ILP) learns logical rules from a combination of examples and
background knowledge. Rather than fitting numerical parameters, ILP systems search
through a space of possible logical hypotheses to find rules that explain the
observed data while remaining consistent with what is already known.

Consider a simple illustration. Suppose the background knowledge states that birds
have wings, and that penguins are birds. The system is then given positive examples:
Tweety, a bird, can fly; a parrot, also a bird, can fly. It also receives negative
examples: a penguin cannot fly, and neither can an ostrich. From these, the ILP
system might induce the rule "birds fly unless they are penguins or ostriches,"
capturing the default with its exceptions in a single readable clause.
@fig:inductivelogicprogramming illustrates how these three inputs (background
knowledge, positive examples, and negative examples) feed into the hypothesis space
from which the system selects its learned rules.

// rendered_images:begin
// ```graphviz
// digraph ILP {
//   graph [rankdir=TD, splines=curved, bgcolor="transparent",
//          ranksep="1.1 equally", nodesep=0.26, pad=0.4, fontname="Helvetica"];
//   node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=14,
//          fontcolor="#26215C", color="#D8D6EE", penwidth=1.2,
//          height=0.52, margin="0.24,0.10"];
//   edge  [arrowhead=vee, penwidth=1.6, color="#B9B6D6"];
//
//   background [label="Background Knowledge",
//               fillcolor="#FBEEE2", color="#C07A45"];
//   pos_ex [label="Positive Examples",
//           fillcolor="#E5F4EE", color="#2F9678"];
//   neg_ex [label="Negative Examples",
//           fillcolor="#FFF3E0", color="#D97706"];
//   hypothesis [label="Hypothesis Space", fillcolor="#EFEDFC", color="#7C74D6"];
//   learned_rule [label="Learned Rules", fillcolor="#E8F1FB", color="#3E86C8"];
//
//   background -> hypothesis [color="#C07A45", penwidth=2.0];
//   pos_ex -> hypothesis [color="#2F9678", penwidth=2.0];
//   neg_ex -> hypothesis [color="#D97706", penwidth=2.0];
//   hypothesis -> learned_rule [color="#3E86C8", penwidth=2.0];
// }
// ```
// label=fig:inductivelogicprogramming
// caption=Inputs feeding the hypothesis space from which an ILP system selects its learned rules
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.4.png",
    width: 70%,
  ),
  caption: [Inputs feeding the hypothesis space from which an ILP system selects its
    learned rules],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:inductivelogicprogramming>
// render_images:end

ILP offers several genuine advantages. The rules it produces are human-readable, so a
domain expert can inspect, critique, and refine what the system has learned rather
than treating it as a black box. Because the learning process operates over logical
representations, it naturally integrates learning with reasoning: the same formalism
used to express background knowledge is used to state the hypothesis, making it
straightforward to chain learned rules with existing facts. ILP also supports the
direct incorporation of background knowledge, which means the system does not have to
rediscover structure that is already well understood.

These strengths come with real costs, however. Searching through the space of
possible logical rules grows combinatorially with the number of predicates and
variables, so ILP can struggle with computational complexity on large datasets. The
framework also assumes that the training examples and background knowledge are
largely correct, so it struggles with noisy data: a single mislabeled example can
derail the search for a consistent hypothesis. These limitations have
kept ILP most useful in domains where data is relatively clean and structured, such
as drug design and bioinformatics, rather than in the noisier settings where
statistical learners dominate.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:266 '# Knowledge Representation Frameworks'
// Slide: Knowledge Representation Frameworks
= Knowledge Representation Frameworks

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:268 '## Description Logics'
// Slide: Description Logics
== Description Logics

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:272 '* Description Logic'
// Slide: Description Logic
#strong[Description logic] represents structured knowledge about a domain #cite(
  "baader2003dlhandbook",
). It occupies a carefully chosen middle ground in the expressivity spectrum: more
expressive than propositional logic, which cannot talk about individuals and their
relationships, yet more tractable than full first-order logic, where even basic
reasoning tasks become undecidable. That balance between what you can say and what
you can still compute is the defining design choice behind description logics.

The framework rests on three building blocks. #emph[Classes] (also called concepts)
capture abstract groups such as $"Person"$ or $"Animal"$. #emph[Properties] (also
called roles) are binary relations linking individuals, for example $"hasChild"$ or
$"ownsPet"$. #emph[Instances] are specific objects in the domain, such as $"GP"$ or
$"Nuvolo"$. From these primitives, a description logic builds complex expressions
using logical constructors: intersection ($inter.sq$), union ($union.sq$), negation
($not$), universal restriction ($forall$), and existential restriction ($exists$).
For instance, one can define
$"Father" equiv "Man" inter.sq exists "hasChild"."Person"$, capturing exactly those
men who have at least one child that is a person.

Two central reasoning tasks give description logic its practical power. #emph[Concept
  subsumption] asks whether every member of one class is necessarily a member of
another: "Is $A$ a subset of $B$?" #emph[Instance checking] asks whether a specific
individual belongs to a given class: "Does $a$ belong to $A$?" Both tasks can be
decided algorithmically in well-studied description logic families, which is
precisely why the formalism is so widely adopted.

// rendered_images:begin
// ```graphviz
// digraph DescriptionLogic {
//     rankdir=TB;
//     node [shape=box, style=filled, fontname=Helvetica];
//     edge [fontname=Helvetica];
//
//     person [label="Class: Person", fillcolor="#C6A6F4"];
//     pet_owner [label="PetOwner\n(Person with pet)", fillcolor="#A0D6D1"];
//     gp [label="Instance: GP\n(a PetOwner)", fillcolor="#A6E7F4"];
//     nuvolo [label="Instance: Nuvolo\n(GP's dog)", fillcolor="#A6E7F4"];
//
//     person -> pet_owner [label="subclass"];
//     pet_owner -> gp [label="instance"];
//     gp -> nuvolo [label="ownsPet"];
//
//     syntax_note [label="Syntax:\nPetOwner = Person AND exists ownsPet.Dog", fillcolor="#B2E2B2", shape=note];
// }
// ```
// label=fig:descriptionlogic
// caption=A description-logic class hierarchy relating Person, PetOwner, and two instances.
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.5.png",
    width: 70%,
  ),
  caption: [A description-logic class hierarchy relating Person, PetOwner, and two
    instances.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:descriptionlogic>
// render_images:end

@fig:descriptionlogic illustrates how these pieces fit together in a concrete
scenario: the class $"Person"$ is refined into a subclass $"PetOwner"$ (a person who
owns a pet), the instance $"GP"$ is recognized as a $"PetOwner"$, and the instance
$"Nuvolo"$ is GP's dog. This kind of structured, machine-readable representation is
the foundation of modern ontology languages. The Web Ontology Language (OWL), for
example, is built directly on description logic and is the standard for encoding
domain knowledge on the Semantic Web.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:293 '* ALC'
// Slide: ALC
The family of description logics that ontology languages build on starts with a
foundational member: #strong[Attributive Concept Language with Complements (ALC)] is
a basic but expressive description logic that serves as a foundation for knowledge
representation in ontologies. ALC combines familiar logical operators to describe
concepts and their relationships: intersection ($inter.sq$, analogous to "and"),
union ($union.sq$, analogous to "or"), and negation ($not$, analogous to "not").
Beyond these Boolean connectives, ALC supports existential quantification
($exists R.C$, meaning "there exists a related individual in class $C$") and
universal quantification ($forall R.C$, meaning "all related individuals belong to
class $C$"). The underlying semantics are set-theoretic: classes correspond to sets
of individuals, and properties correspond to binary relations over those individuals.

To see how these constructs work in practice, consider the statement "all students
take some course." In ALC this is captured as
$"Student" equiv exists "takes"."Course"$, which says that every individual in the
Student concept is related by the #emph[takes] relation to at least one individual in
the Course concept. Similarly, "a mother is a woman with at least one child" becomes
$"Mother" equiv "Woman" inter.sq exists "hasChild".top$, where $top$ (the universal
concept) means the child can be any individual at all; what matters is that the
#emph[hasChild] relation holds for at least one entity.

ALC occupies a deliberate sweet spot in the landscape of description logics. It is
#emph[decidable], meaning that reasoning tasks such as checking whether one concept
subsumes another or whether a knowledge base is consistent are guaranteed to
terminate. At the same time, it is expressive enough to capture a wide range of
real-world domain constraints. This balance between expressiveness and computational
tractability is precisely why ALC serves as the foundation for more complex logics
used in the Web Ontology Language (OWL): richer OWL profiles extend ALC with
additional constructors (number restrictions, role hierarchies, nominals) while
inheriting its core reasoning architecture.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:315 '* SHOIN'
// Slide: SHOIN
#strong[SHOIN] is a description logic that extends ALC with several expressive
features #cite("horrocks2003owl"). Where ALC provides the core Boolean connectives
over concepts and existential and universal role restrictions, SHOIN adds five
additional capabilities, each named by a letter in the acronym:

- $cal(S)$: transitive properties, allowing roles like $"ancestorOf"$ to chain across
  multiple steps so that an ancestor of an ancestor is still recognized as an
  ancestor.
- $cal(H)$: role hierarchies, letting one role be declared a sub-role of another (for
  instance, $"hasSon" subset.sq.eq "hasChild"$), so that any individual filling the
  more specific role automatically fills the general one.
- $cal(O)$: nominals, which bring specific named individuals into concept expressions
  (for example, singleton sets like ${"John"}$), bridging the gap between the class
  level and the instance level.
- $cal(I)$: inverse roles, so that if $"hasChild"$ links a parent to a child,
  $"isChildOf"$ is automatically available as its inverse without redundant axioms.
- $cal(N)$: number restrictions (cardinality constraints), enabling statements such
  as "exactly one" or "at most three" fillers for a given role.

To see number restrictions at work, consider the assertion
$"Person" subset.sq.eq (= 2 "hasChild".top)$, which states that every person has
exactly two children. ALC cannot express this constraint at all; it can say "there
exists at least one child" or "all children satisfy some concept," but it cannot
count.

The added expressiveness comes with a cost: reasoning in SHOIN is computationally
harder than in ALC, because the tableau algorithms must track transitive closures,
role subset entailments, individual identity, inverse navigation, and counting
simultaneously. The payoff is the ability to model considerably richer real-world
scenarios, capturing domain knowledge that simpler logics would flatten or lose.
SHOIN serves as the formal foundation for OWL DL, the decidable fragment of the Web
Ontology Language widely used in semantic web applications. @fig:shoin illustrates
how each lettered extension builds on the ALC base, showing the layered architecture
from basic concept logic up through transitive properties, role hierarchies, and
nominals.

// rendered_images:begin
// ```graphviz
// digraph SHOINFeatures {
//     rankdir=LR;
//     node [shape=box, style=filled, fontname=Helvetica, fontsize=9];
//     edge [fontname=Helvetica, fontsize=8];
//
//     alc [label="ALC\n(base)", fillcolor="#A0D6D1"];
//
//     s_node [label="S:\nTransitive\nProps\n(ancestorOf)", fillcolor="#FFD1A6"];
//     h_node [label="H:\nRole\nHierarchies\n(hasSon sub-role of hasChild)", fillcolor="#FFD1A6"];
//     o_node [label="O:\nNominals\n(John:individual)", fillcolor="#FFD1A6"];
//     i_node [label="I:\nInverse\nRoles\n(isChildOf)", fillcolor="#FFD1A6"];
//     n_node [label="N:\nCardinality\n(=2 children)", fillcolor="#FFD1A6"];
//
//     alc -> s_node;
//     s_node -> h_node;
//     h_node -> o_node;
//     o_node -> i_node;
//     i_node -> n_node;
//
//     shoin [label="SHOIN\n(full)", fillcolor="#A6C8F4"];
//     n_node -> shoin;
// }
// ```
// label=fig:shoin
// caption=SHOIN's five lettered extensions layered on top of the ALC base.
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.6.png",
    width: 70%,
  ),
  caption: [SHOIN's five lettered extensions layered on top of the ALC base.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:shoin>
// render_images:end

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:337 '* OWL'
// Slide: OWL
#strong[Web Ontology Language (OWL)] #cite("mcguinness2004owl") is a semantic web
language designed for representing complex knowledge with formal semantics that
machines can reason over. Built on the description logic SHOIN, OWL provides a rich
framework for encoding ontologies directly on the web, going well beyond what simpler
formats like RDF Schema can express. Its core building blocks are classes (concepts
such as "Cat" or "Mammal"), properties (relationships between individuals or between
individuals and data values), individuals (specific instances of classes), and axioms
(logical statements constraining how these elements relate).

To see OWL in action, consider the statement "every cat is a mammal." In OWL's
description-logic notation this is written as $"Cat" subset.eq "Mammal"$, a
subsumption axiom asserting that the class Cat is a subclass of Mammal. A reasoner
can then automatically infer that any individual classified as a Cat must also be a
Mammal, without that fact being stated explicitly. This kind of automated inference
powers practical applications ranging from #emph[semantic search] engines that
understand query intent rather than just matching keywords, to biomedical data
integration platforms where ontologies like SNOMED CT and the Gene Ontology let
researchers query across heterogeneous datasets using shared formal vocabularies.

OWL comes in three variants that trade expressiveness for computational tractability,
as @fig:owl illustrates. #emph[OWL Lite] is the simplest profile, restricted largely
to classification hierarchies and simple constraints; it is suitable when the
ontology needs only straightforward taxonomic relationships and fast reasoning is
essential. #emph[OWL DL] (Description Logic) retains the full expressiveness of the
underlying SHOIN logic while guaranteeing that reasoning remains decidable: every
valid query is guaranteed to terminate with a correct answer, making it the most
widely used variant in practice. #emph[OWL Full] removes all syntactic restrictions
and allows maximum expressiveness, including the ability to treat classes
simultaneously as individuals, but at the cost of undecidability: there is no general
algorithm guaranteed to answer every reasoning query in finite time.

// rendered_images:begin
// ```graphviz
// digraph OWLVariants {
//     rankdir=TB;
//     node [shape=box, style=filled, fontname=Helvetica];
//     edge [fontname=Helvetica];
//
//     owl [label="Web Ontology\nLanguage (OWL)", fillcolor="#C6A6F4"];
//
//     lite [label="OWL Lite\n(Simpler)\nfor classification", fillcolor="#A0D6D1"];
//     dl [label="OWL DL\n(Full expressiveness)\nDecidable reasoning", fillcolor="#A6E7F4"];
//     full [label="OWL Full\n(Maximum)\nUndecidable", fillcolor="#FFD1A6"];
//
//     owl -> lite;
//     owl -> dl;
//     owl -> full;
//
//     lite -> dl [style=invis];
//     dl -> full [style=invis];
//
//     note1 [label="Lite:\nHierarchies only", fillcolor="#B2E2B2", shape=note, fontsize=8];
//     note2 [label="DL:\nDecidable\n= practical", fillcolor="#B2E2B2", shape=note, fontsize=8];
//     note3 [label="Full:\nTuring\ncomplete", fillcolor="#F4A6A6", shape=note, fontsize=8];
// }
// ```
// label=fig:owl
// caption=OWL's three variants, trading expressiveness for decidability.
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.7.png",
    width: 70%,
  ),
  caption: [OWL's three variants, trading expressiveness for decidability.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:owl>
// render_images:end

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:355 '* Example of OWL in RDF'
// Slide: Example of OWL in RDF
A single axiom shows how the pieces fit together: "every $"Student"$ is a $"Person"$
with at least one $"hasAdvisor"$", which in description-logic notation reads

$ "Student" subset.sq.eq "Person" inter.sq (gt.eq 1 thin "hasAdvisor") $

The same axiom has two useful renderings. @fig:exampleofowlinrdf gives the
class/property/restriction graph: $"Student"$ is declared a subclass of $"Person"$,
and a restriction on the $"hasAdvisor"$ object property requires a minimum cardinality
of one, so every individual classified as a $"Student"$ must have at least one
advisor. This combination of class hierarchy and cardinality restriction is exactly
the kind of constraint that propositional or plain first-order logic cannot express as
compactly.

// rendered_images:begin
// ```graphviz
// digraph OWL_Example {
//     rankdir=TD;
//     node [shape=ellipse, style=filled, fillcolor=lightgray];
//
//     Person [label="Person (Class)"];
//     Student [label="Student (Class)"];
//     hasAdvisor [label="hasAdvisor (ObjectProperty)", shape=box, fillcolor=lightblue];
//     Restriction [label="Restriction: minCardinality 1", shape=diamond, fillcolor=lightyellow];
//
//     Student -> Person [label="subClassOf"];
//     Student -> Restriction [label="subClassOf"];
//     Restriction -> hasAdvisor [label="onProperty"];
// }
// ```
// label=fig:exampleofowlinrdf
// caption=The Student-Person-hasAdvisor axiom as a class/property/restriction graph.
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.8.png",
    width: 70%,
  ),
  caption: [The Student-Person-hasAdvisor axiom as a class/property/restriction
    graph.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:exampleofowlinrdf>
// render_images:end

@fig:rdfexample gives the second rendering: the same axiom serialized as RDF/XML for
machine exchange. The `owl:Class` and `rdfs:subClassOf` elements encode the
Student-is-a-Person relationship directly, and the nested `owl:Restriction` block
spells out the minimum-cardinality constraint on `hasAdvisor` in full, showing how
compact the graphical notation is by comparison.

#figure(
  image("../lectures_source/figures/L03.RDF_example.png", width: 80%),
  caption: [RDF/XML serialization of the Student-Person-hasAdvisor example],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:rdfexample>

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:389 '## Knowledge Representation Standards'
// Slide: Knowledge Representation Standards
== Knowledge Representation Standards

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:391 '* Resource Description Framework'
// Slide: Resource Description Framework
#strong[Resource Description Framework (RDF)], standardized in 2004
#cite("klyne2004rdf"), is a standard model for data interchange on the web. It
provides a way to represent structured information in a machine-readable format,
enabling different systems and applications to share and combine data without loss of
meaning.

#grid(
  columns: (1fr, 55%),
  column-gutter: 1em,
)[
  The basic building block of RDF is the #strong[triple], a three-part statement that
  links two pieces of information through a named relationship:

  - #emph[Subject]: the entity being described (e.g., `Nuvolo`)
  - #emph[Predicate]: the property or relationship (e.g., `isA`)
  - #emph[Object]: the value or related entity (e.g., `Dog`)

  As @tab:rdftriples illustrates, each row captures one atomic fact about an entity.
  The triple `(Book123, hasAuthor, Author456)` connects a book to its author, while
  `(Author456, hasName, "F. Scott Fitzgerald")` attaches a human-readable name to
  that author entity. By chaining triples together this way, RDF builds up a rich
  description from minimal primitives.
][
  #figure(
    styled-table(
      headers: (
        "Subject",
        "Predicate",
        "Object",
      ),
      rows: (
        ("Book123", "hasTitle", [_"The Great Gatsby"_]),
        ("Book123", "hasAuthor", "Author456"),
        ("Author456", "hasName", [_"F. Scott Fitzgerald"_]),
        ("Book123", "publishedYear", [_"1925"_]),
        ("Book123", "belongsToGenre", [_"Fiction"_]),
      ),
    ),
    caption: [RDF triples describing a book and its author],
    kind: "table",
    supplement: [Table.],
    placement: auto,
  ) <tab:rdftriples>
]

These statements naturally form directed graphs, where subjects and objects are nodes
and predicates are the labeled edges connecting them. To ensure global uniqueness and
avoid naming collisions across different datasets, RDF identifies its components
using URIs (e.g., `http://example.org/Nuvolo`) or literal values for concrete data
like strings and dates. This URI-based naming is what makes it possible to merge RDF
data from completely independent sources without ambiguity.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:412 '* RDF: Example'
// Slide: RDF: Example
@fig:rdftriplesgraph draws the five triples of @tab:rdftriples as that directed graph.
The subject `Book123` sits at the center, with one outgoing edge per predicate. Three
of those edges terminate in literals (a title, a year, a genre), while `hasAuthor`
terminates in a second entity, `Author456`, which is itself the subject of a further
triple. Nothing in the graph is new information: it is the same table read as nodes
and edges rather than as rows, which is precisely the point of the triple model.

// rendered_images:begin
// ```graphviz
// digraph RDFExample {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.5;
//     ranksep=0.6;
//     rankdir=LR;
//
//     node [shape=box,
//           style="rounded,filled",
//           penwidth=1.8,
//           fontname="Helvetica",
//           fontsize=11,
//           margin="0.20,0.12",
//           height=0.48];
//
//     edge [color="#A3B1C0",
//           penwidth=1.3,
//           arrowhead=vee,
//           arrowsize=0.75,
//           fontname="Helvetica",
//           fontsize=9,
//           fontcolor="#7B8794"];
//
//     // Nodes representing the RDF example
//     Book [label="Book123\n(Subject)", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     Author [label="Author456\n(Entity)", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     Title [label="\"The Great Gatsby\"\n(Literal)", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Year [label="\"1925\"\n(Literal)", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Genre [label="\"Fiction\"\n(Literal)", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Name [label="\"F. Scott Fitzgerald\"\n(Literal)", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//
//     // Edges with predicates
//     Book -> Title [label="hasTitle"];
//     Book -> Author [label="hasAuthor"];
//     Author -> Name [label="hasName"];
//     Book -> Year [label="publishedYear"];
//     Book -> Genre [label="belongsToGenre"];
// }
// ```
// label=fig:rdftriplesgraph
// caption=The RDF triples of a book and its author, drawn as a directed graph
// width=80%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.9.png",
    width: 80%,
  ),
  caption: [The RDF triples of a book and its author, drawn as a directed graph],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:rdftriplesgraph>
// render_images:end

RDF serves as the foundation for several practical applications: building knowledge
graphs that organize large-scale structured information, powering semantic search
systems that understand the meaning behind queries rather than just matching
keywords, and supporting ontologies that formally define the concepts and
relationships within a domain.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:473 '* SPARQL'
// Slide: SPARQL
#strong[SPARQL], standardized in 2008 #cite("prudhommeaux2008sparql"), is the query
language for RDF data. It provides a standardized way to retrieve and manipulate
information stored in RDF format, much as SQL serves relational databases.

A SPARQL query is built from a few core components. #emph[Triple patterns] are query
fragments that match triples in the RDF graph. A #emph[basic graph pattern] is a set
of triple patterns combined together. #emph[Variables], written with a leading
question mark (e.g., `?person`, `?animal`), stand in for the unknown parts of a
triple that the query engine should fill in.

SPARQL supports four main query types:

- `SELECT`: retrieves bindings for specific variables.
- `CONSTRUCT`: builds new RDF triples from the query results.
- `ASK`: returns a simple boolean indicating whether a matching pattern exists in the
  graph.
- `DESCRIBE`: returns an RDF graph that describes the matched resources.

For instance, to find all resources whose type is `Bird`, one writes:

```
SELECT ?animal WHERE { ?animal rdf:type ex:Bird }
```

This query binds the variable `?animal` to every subject that has an `rdf:type` arc
pointing to `ex:Bird`, returning each matching resource as a row in the result set.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:495 '# Knowledge Graphs & The Semantic Web'
// Slide: Knowledge Graphs & The Semantic Web
= Knowledge Graphs & The Semantic Web

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:497 '* Semantic Web'
// Slide: Semantic Web
The #strong[Semantic Web] extends the ordinary World Wide Web so that machines, not
just humans, can understand and process its content. Traditional HTML is designed for
visual presentation: a browser knows how to render headings, links, and paragraphs,
but it has no way to determine that two pages describe the same entity or that one
dataset's "author" field means the same thing as another's "creator." The Semantic
Web addresses this gap by layering explicit meaning (semantics) on top of web
resources, enabling automated integration, discovery, and reasoning across
independently published data.

The architecture rests on three core components:

- #emph[RDF (Resource Description Framework)]: the base data model, which represents
  all information as triples of the form subject-predicate-object. Each element is
  identified by a URI, giving every concept a globally unique name.
- #emph[SPARQL]: the standard query language for retrieving and manipulating RDF
  data, analogous to SQL for relational databases but designed around the graph
  structure of triples.
- #emph[OWL (Web Ontology Language)]: a richer layer for defining ontologies,
  expressing complex relationships such as class hierarchies, cardinality
  constraints, and equivalences between concepts from different vocabularies.

@fig:semanticweb illustrates how these layers build on one another: URIs and Unicode
provide the foundation for globally unambiguous identifiers, RDF supplies the
triple-based data model on top of that, RDFS adds a schema layer with classes and
properties, and OWL sits at the top, enabling expressive ontological reasoning.

// rendered_images:begin
// ```graphviz
// digraph SemanticWebStack {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.4;
//     ranksep=0.6;
//     rankdir=TB;
//
//     node [shape=box,
//           style="rounded,filled",
//           penwidth=1.8,
//           fontname="Helvetica",
//           fontsize=12,
//           margin="0.22,0.14",
//           height=0.50];
//
//     edge [color="#A3B1C0",
//           penwidth=1.3,
//           arrowhead=vee,
//           arrowsize=0.75,
//           fontname="Helvetica",
//           fontsize=10,
//           fontcolor="#7B8794"];
//
//     foundation [label="Foundation\nURIs and Unicode", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     rdf [label="RDF\nData Model", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     rdfs [label="RDFS\nSchema Layer", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     owl [label="OWL\nOntology Language", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     sparql [label="SPARQL\nQuery Language", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     apps [label="Applications\nSemantic Search, KGs", fillcolor="#C7ECF0", color="#7CC6D0", fontcolor="#1F4E56"];
//
//     foundation -> rdf;
//     foundation -> rdfs;
//     rdf -> owl;
//     rdf -> sparql;
//     owl -> apps;
//     sparql -> apps;
// }
// ```
// label=fig:semanticweb
// caption=The Semantic Web stack, from URIs and Unicode up to the applications built on it
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.10.png",
    width: 70%,
  ),
  caption: [The Semantic Web stack, from URIs and Unicode up to the applications built
    on it],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:semanticweb>
// render_images:end

The core ideas of the Semantic Web have been widely adopted in practice. Schema.org
markup powers rich search results across major engines, knowledge graphs underpin
products from Google and Wikidata, and linked-data principles shape how government
agencies and scientific repositories publish open datasets. That said, the full
original vision of a seamlessly interlinked, machine-readable web remains only
partially realized.

Several challenges account for the gap. Building and maintaining ontologies is
complex, requiring specialized expertise that many organizations lack. Privacy and
data-ownership concerns arise once information becomes easily discoverable and
combinable by machines. Standardization across domains is slow, scalability of
reasoning over very large triple stores remains an active research problem, and
tensions between decentralized linked data and the reality of centralized platform
control continue to shape the ecosystem's evolution.

// Book-only grouping: the source .smd dropped this heading, but the book keeps it to
// group the slides below.
== Semantic Network Implementations

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:567 '* WikiData'
// Slide: WikiData
#strong[Wikidata], launched in 2012, is a free, collaborative knowledge base that
serves as the structured-data backbone for Wikipedia #cite("vrandecic2014wikidata").
Unlike Wikipedia's prose articles, Wikidata stores facts in a graph-based data model
that machines can query directly through SPARQL APIs, making it one of the largest
openly accessible knowledge graphs in existence.

The data model revolves around five core components:

- #emph[Item]: a uniquely identified entity, such as `Q42` for Douglas Adams.
- #emph[Property]: a named attribute that connects an item to a value, such as `P31`
  ("instance of") or `P106` ("occupation").
- #emph[Value]: the datum attached through a property. For example, the triple `Q42`
  $arrow.r$ `P31` $arrow.r$ `Q5` encodes the fact that Douglas Adams is a human,
  while `Q42` $arrow.r$ `P106` $arrow.r$ "science-fiction writer" records his
  occupation.
- #emph[Reference]: a citation that supports the claim, linking it back to a
  verifiable source.
- #emph[Qualifier]: contextual metadata that refines a statement, such as the year a
  property held or the specific role in which it applied.

@fig:wikidata illustrates how these components fit together for the Douglas Adams
example, showing items linked to values through properties in a directed graph.

// rendered_images:begin
// ```graphviz
// digraph WikiDataStructure {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.5;
//     ranksep=0.6;
//     rankdir=TD;
//
//     node [shape=box,
//           style="rounded,filled",
//           penwidth=1.8,
//           fontname="Helvetica",
//           fontsize=11,
//           margin="0.20,0.12",
//           height=0.48];
//
//     edge [color="#A3B1C0",
//           penwidth=1.3,
//           arrowhead=vee,
//           arrowsize=0.75,
//           fontname="Helvetica",
//           fontsize=9,
//           fontcolor="#7B8794"];
//
//     // Item (orange: source/entity)
//     q42 [label="Item: Q42\n(Douglas Adams)", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//
//     // Properties (teal: stateful/structural element)
//     p31 [label="Property: P31\n(instance of)", fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//     p106 [label="Property: P106\n(occupation)", fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//
//     // Values (blue: output/result)
//     q5 [label="Value: Q5\n(Human)", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     q36180 [label="Value: Q36180\n(Sci-fi writer)", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//
//     // Metadata (purple: annotations/references)
//     ref [label="Reference:\nSupporting citation", fillcolor="#E8D9F7", color="#9B7DB1", fontcolor="#4A2E5C", shape=note];
//     qual [label="Qualifier:\nYear = 1952", fillcolor="#E8D9F7", color="#9B7DB1", fontcolor="#4A2E5C", shape=note];
//
//     q42 -> p31 [label="  statement  "];
//     p31 -> q5 [label="  value  "];
//
//     q42 -> p106 [label="  statement  "];
//     p106 -> q36180 [label="  value  "];
//
//     p31 -> ref;
//     p106 -> qual;
// }
// ```
// label=fig:wikidata
// caption=Wikidata's item, property, value, reference and qualifier structure for Douglas Adams
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.11.png",
    width: 70%,
  ),
  caption: [Wikidata's item, property, value, reference and qualifier structure for
    Douglas Adams],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:wikidata>
// render_images:end

This structure makes Wikidata a practical foundation for knowledge graph
construction, semantic search, and AI reasoning. A semantic search engine can resolve
an ambiguous query by following property links to disambiguate entities, while a
reasoning system can chain triples together to infer facts not stated explicitly, for
instance deducing that Douglas Adams, being human, was also a mammal, given the right
ontological links upstream from `Q5`.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:645 '* DBPedia'
// Slide: DBPedia
#strong[DBpedia], started in 2007, extracts structured content from Wikipedia to
create a large-scale, multilingual knowledge graph #cite("lehmann2015dbpedia"). Rather
than treating encyclopedia articles as flat text, DBpedia parses infoboxes,
categories, and other semi-structured elements into machine-readable form, making the
wealth of Wikipedia accessible to automated reasoning systems.

The underlying data is stored as RDF triples, each following the pattern
#emph[(Subject, Predicate, Object)]. For instance, the entity "Berlin" might be
linked to Germany through the predicate `dbo:country`, and its population recorded
via `dbo:populationTotal 3.5M`. These triples collectively form a dense web of
factual relationships spanning millions of entities across dozens of languages.

DBpedia serves several practical purposes. It exposes a public SPARQL endpoint,
allowing users to run structured semantic queries over Wikipedia's content: asking,
for example, for all European capitals with populations above one million, something
no keyword search could answer reliably. It also plays a central role in the broader
Semantic Web, acting as a hub that other linked-data sources connect to for entity
disambiguation and cross-referencing. Beyond web infrastructure, DBpedia is widely
used to enhance AI models with real-world knowledge, providing grounding facts that
improve tasks such as question answering, entity linking, and relation extraction.
@fig:dbpedia traces one such chain of extracted triples: the European Union has
Germany as a member, Germany has Berlin as its capital, and Berlin carries literal
values for its population and coordinates. A query for "European capitals with more
than one million inhabitants" walks exactly these edges.

// rendered_images:begin
// ```graphviz
// digraph DBPediaExample {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.5;
//     ranksep=0.6;
//     rankdir=TB;
//
//     node [shape=box,
//           style="rounded,filled",
//           penwidth=1.8,
//           fontname="Helvetica",
//           fontsize=11,
//           margin="0.20,0.12",
//           height=0.48];
//
//     edge [color="#A3B1C0",
//           penwidth=1.3,
//           arrowhead=vee,
//           arrowsize=0.75,
//           fontname="Helvetica",
//           fontsize=9,
//           fontcolor="#7B8794"];
//
//     EU [label="European Union", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Germany [label="Germany", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     Berlin [label="Berlin", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     Pop [label="3.5 million", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     Coords [label="52° 31' N, 13° 24' E", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//
//     EU -> Germany [label="member"];
//     Germany -> Berlin [label="capital"];
//     Berlin -> Pop [label="population"];
//     Berlin -> Coords [label="coordinates"];
// }
// ```
// label=fig:dbpedia
// caption=A chain of DBpedia triples linking the European Union to Berlin's population and coordinates
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.12.png",
    width: 70%,
  ),
  caption: [A chain of DBpedia triples linking the European Union to Berlin's
    population and coordinates],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:dbpedia>
// render_images:end

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:707 '* Semantic Networks'
// Slide: Semantic Networks
Wikidata and DBpedia are both instances of a broader, older formalism.
#strong[Semantic networks] represent knowledge as graphs of concepts and relations.
In this formalism, nodes stand for entities or concepts, while edges encode the
semantic relations between them: typical edge labels include "is-a," "part-of," and
"has-property." Two widely used examples are WordNet, a lexical database organizing
English words into synonym sets linked by hypernymy and meronymy, and ConceptNet, a
commonsense knowledge graph connecting everyday concepts with labeled relations.

Semantic networks are easy to visualize and traverse, which makes them a natural fit
for explaining how concepts relate to one another. Reasoning reduces to path
traversal: to determine whether a dog is an animal, one follows the "is-a" edges from
#emph[Dog] to #emph[Mammal] to #emph[Animal], inheriting properties along the way.
This representation powered many early AI systems and remains the backbone of modern
knowledge graphs. As @fig:semanticnetworks illustrates, even a small network
capturing the relationships among animals, dogs, cats, and mammals reveals the
hierarchical structure that makes inheritance-based inference straightforward.

// rendered_images:begin
// ```graphviz
// digraph SemanticNetwork {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.5;
//     ranksep=0.6;
//     rankdir=TB;
//
//     node [shape=ellipse,
//           style="rounded,filled",
//           penwidth=1.8,
//           fontname="Helvetica",
//           fontsize=11,
//           margin="0.20,0.12"];
//
//     edge [color="#A3B1C0",
//           penwidth=1.3,
//           arrowhead=vee,
//           arrowsize=0.75,
//           fontname="Helvetica",
//           fontsize=9,
//           fontcolor="#7B8794"];
//
//     animal [label="Animal", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     mammal [label="Mammal", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     dog [label="Dog", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     cat [label="Cat", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     fido [label="Fido\n(instance)", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     property [label="Has 4 legs", fillcolor="#C7ECF0", color="#7CC6D0", fontcolor="#1F4E56"];
//
//     animal -> mammal [label="is-a"];
//     animal -> dog [label="is-a"];
//     animal -> cat [label="is-a"];
//     dog -> fido [label="instance"];
//     dog -> property [label="property"];
// }
// ```
// label=fig:semanticnetworks
// caption=A small semantic network of is-a and property relations among animals.
// width=70%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.13.png",
    width: 70%,
  ),
  caption: [A small semantic network of is-a and property relations among
    animals.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:semanticnetworks>
// render_images:end

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:771 '* WordNet'
// Slide: WordNet
#strong[WordNet], described in its 1995 reference paper #cite("miller1995wordnet"), is
a large lexical database of English words that models semantic relationships between
them. Unlike resources built automatically from corpora, WordNet was manually curated
by linguists, which gives it high precision but also means it can be incomplete when
it comes to domain-specific or rapidly evolving terminology.

The database is organized as a graph structure. Nodes in this graph are
#emph[synsets]: sets of synonyms that together express a single distinct concept. For
instance, the words "car" and "automobile" belong to the same synset because they
refer to the same concept despite being different surface forms. The edges connecting
synsets encode several types of semantic relations:

- #emph[Is-a (hypernymy/hyponymy)]: a taxonomic link indicating that one concept is a
  specialization of another (e.g., "dog" is an "animal").
- #emph[Part-whole (meronymy)]: a compositional link indicating that one concept is a
  component of another (e.g., "wheel" is part of "car").
- #emph[Opposites (antonymy)]: a link between concepts with contrasting meanings.
- #emph[Attribute]: a link from an adjective to the noun whose value it describes.
- #emph[Similar to]: a link between an adjective and a narrower adjective that shades
  its meaning.

@fig:wordnet shows a fragment of the network around the noun "temperature." The
adjectives "hot" and "cold" both point at it through #emph[attribute] edges and at
each other through an #emph[antonym] edge. Two noun synsets, "body temperature, blood
heat" and "coldness, cold, low temperature, frigidity, frigidness," sit under
"temperature" as #emph[hypernym] links, and "cold" shades into the narrower synset
"arctic, frigid, gelid, glacial, icy, polar" through a #emph[similar] edge. A handful
of relation types, applied consistently, is enough to organize the whole lexicon.

// rendered_images:begin
// ```graphviz
// digraph WordNet {
//     rankdir=LR;
//     node [shape=ellipse, style=filled, fontname="Helvetica", fontsize=12];
//     edge [fontname="Helvetica-Italic", fontsize=11, fontcolor=black];
//
//     temperature [label="temperature", fillcolor="#b7c9e8"];
//     hot [label="hot", fillcolor="#f5e6a3"];
//     cold [label="cold", fillcolor="#f5e6a3"];
//     bodytemp [label="body temperature,\nblood heat", fillcolor="#b7c9e8"];
//     coldness [label="coldness, cold, low\ntemperature, frigidity,\nfrigidness", fillcolor="#b7c9e8"];
//     arctic [label="arctic, frigid, gelid,\nglacial, icy, polar", fillcolor="#d9c2e0"];
//
//     hot -> temperature [label="attribute"];
//     cold -> temperature [label="attribute"];
//     hot -> cold [dir=both, label="antonym"];
//     bodytemp -> temperature [label="hypernym"];
//     coldness -> temperature [label="hypernym"];
//     cold -> arctic [label="similar"];
// }
// ```
// label=fig:wordnet
// caption=A WordNet fragment around the noun "temperature" and its related synsets
// width=80%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.14.png",
    width: 80%,
  ),
  caption: [A WordNet fragment around the noun "temperature" and its related synsets],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:wordnet>
// render_images:end

These structured relationships make WordNet useful across a range of NLP tasks. In
#emph[word sense disambiguation], the synset structure helps determine which meaning
of a polysemous word is intended in a given context. Semantic similarity measures can
be computed by traversing the graph and measuring path length or shared ancestors
between synsets. WordNet also serves as a backbone for information retrieval and
question answering systems, where understanding that a query about "automobiles"
should also match documents about "cars" directly improves recall.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:818 '* ConceptNet'
// Slide: ConceptNet
The second of WordNet's two sibling examples takes a different focus.
#strong[ConceptNet], whose 5.5 release appeared in 2017 #cite("speer2017conceptnet"),
is a large knowledge graph that connects words and phrases through labeled semantic
relationships, serving as a structured repository of commonsense knowledge. Rather
than encoding narrow, domain-specific facts, ConceptNet captures the kind of everyday
understanding that humans take for granted: that dogs are animals, that wheels are
parts of cars, and that fire causes smoke.

The graph's structure is straightforward. Nodes represent concepts, which can be
individual words or short phrases. Edges carry typed semantic relationships that
specify how two concepts relate to each other:

- #emph[IsA]: taxonomic membership, such as ("dog", "animal").
- #emph[PartOf]: meronymic composition, such as ("wheel", "car").
- #emph[UsedFor]: functional purpose, such as ("knife", "cutting").
- #emph[CapableOf]: an ability or typical action, such as ("bird", "fly").
- #emph[Causes]: causal links, such as ("fire", "smoke").

A concrete entry in the graph might be the triple ("bicycle", "UsedFor",
"transportation"), asserting that bicycles serve a transportation purpose. By
aggregating millions of such triples from crowd-sourced resources, expert databases,
and multilingual dictionaries, ConceptNet builds a broad web of everyday reasoning
that no single curated ontology could practically cover on its own.

@fig:conceptnet illustrates the structure of ConceptNet, showing how concepts link to
one another through these typed edges to form a dense, navigable graph of commonsense
associations.

#figure(
  image("../lectures_source/figures/L03.ConceptNet.png", width: 80%),
  caption: [ConceptNet],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:conceptnet>

Because it encodes the implicit background knowledge that most text assumes readers
already have, ConceptNet finds use across a range of applications: natural language
understanding systems that need to resolve ambiguity or fill in unstated context,
question answering platforms and chatbots that must reason beyond what is literally
stated, commonsense AI reasoning tasks where pure statistical models fall short, and
semantic search or recommendation engines that benefit from understanding conceptual
similarity rather than relying solely on keyword overlap.

// Book-only grouping: the source .smd dropped this heading, but the book keeps it to
// group the slides below.
== Knowledge Graphs

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:851 '* Knowledge Graphs'
// Slide: Knowledge Graphs
#strong[Knowledge graphs] represent entities and their relationships as graph
structures #cite("hogan2021knowledgegraphs"). In a knowledge graph, #emph[nodes]
correspond to entities (people, places, concepts) and #emph[edges] encode the
relations between them. A simple triple such as "Paris → isCapitalOf → France"
captures a single fact; millions of such triples, linked together, form a rich web of
structured knowledge that machines can traverse and reason over.

Knowledge graphs support expressive information retrieval through query languages
like SPARQL, which lets users pose complex questions that span multiple hops across
the graph. Beyond simple lookup, they enable reasoning through #strong[path
  traversal] (following chains of edges to discover implicit connections) and
#strong[schema inference] (deriving new facts from the types and constraints declared
in the graph's ontology).

These capabilities make knowledge graphs a foundational component in several
practical systems:

- #emph[Question answering]: translating a natural-language question into a graph
  query that retrieves a precise answer.
- #emph[Recommendation]: exploiting entity connections (e.g., shared genres,
  co-authorship) to surface relevant items.
- #emph[Semantic search]: moving beyond keyword matching to understand the meaning
  behind a query.

Major technology companies rely heavily on knowledge graphs: Google's Knowledge Graph
powers the information panels that appear beside search results, Facebook's social
graph drives friend and content recommendations, and academic search engines like
Semantic Scholar use them to link papers, authors, and concepts.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:868 '* Knowledge Graph: University Example'
// Slide: Knowledge Graph: University Example
@fig:knowledgegraphuniversityexample puts these ideas together in a university
setting. The solid edges are the class-level schema: $"Student"$ and $"Professor"$
both #emph[belong] to a $"Department"$, a $"Department"$ #emph[offers] a $"Course"$,
and students #emph[take] the courses professors #emph[teach]. The dashed edges
instantiate that schema with individuals: Alice takes CS101, Dr. Smith teaches CS101,
and both belong to the Computer Science department, which in turn offers CS101. A
query engine can walk these edges to answer questions no single triple states
outright, such as which department is responsible for every course a given student
takes.

// rendered_images:begin
// ```graphviz
// digraph UniversityOntology {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.4;
//     ranksep=0.5;
//     rankdir=LR;
//
//     node [shape=box,
//           style="rounded,filled",
//           penwidth=1.8,
//           fontname="Helvetica",
//           fontsize=11,
//           margin="0.20,0.12",
//           height=0.48];
//
//     edge [color="#A3B1C0",
//           penwidth=1.3,
//           arrowhead=vee,
//           arrowsize=0.75,
//           fontname="Helvetica",
//           fontsize=9,
//           fontcolor="#7B8794"];
//
//     // Classes (orange)
//     Student [label="Student", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Professor [label="Professor", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Course [label="Course", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Department [label="Department", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//
//     // Individuals (green)
//     Alice [label="Alice", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     Bob [label="Bob", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     DrSmith [label="Dr. Smith", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     DrLee [label="Dr. Lee", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     CS101 [label="CS101", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     MATH201 [label="MATH201", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     CompSci [label="Computer Science", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     Math [label="Mathematics", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//
//     // Class-level relationships
//     Student -> Course [label="takes"];
//     Professor -> Course [label="teaches"];
//     Student -> Department [label="belongs"];
//     Professor -> Department [label="belongs"];
//     Department -> Course [label="offers"];
//
//     // Individual-level relationships
//     Alice -> CS101 [label="takes", style=dashed];
//     Bob -> MATH201 [label="takes", style=dashed];
//     DrSmith -> CS101 [label="teaches", style=dashed];
//     DrLee -> MATH201 [label="teaches", style=dashed];
//     Alice -> CompSci [label="belongs", style=dashed];
//     Bob -> Math [label="belongs", style=dashed];
//     DrSmith -> CompSci [label="belongs", style=dashed];
//     DrLee -> Math [label="belongs", style=dashed];
//     CompSci -> CS101 [label="offers", style=dashed];
//     Math -> MATH201 [label="offers", style=dashed];
// }
// ```
// label=fig:knowledgegraphuniversityexample
// caption=A university knowledge graph, with class-level edges solid and individual-level edges dashed
// width=80%
// placement=auto
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson03.3-Non_classical_logics.typ.figs/Lesson03.3-Non_classical_logics.15.png",
    width: 80%,
  ),
  caption: [A university knowledge graph, with class-level edges solid and
    individual-level edges dashed],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:knowledgegraphuniversityexample>
// render_images:end

= Summary

This chapter relaxed the assumptions of classical logic along two axes:
ontological commitment (what exists) and epistemological commitment (what an
agent may believe). #emph[Non-monotonic and default reasoning] let a system
retract conclusions as new facts arrive, the #emph[open-world] and
#emph[closed-world assumptions] fix how silence about a fact is interpreted,
and #emph[inductive logic programming] learns such default-laden rules
directly from labeled examples. #emph[Description logics], from the
foundational ALC up through the more expressive SHOIN, trade expressiveness
for decidable reasoning and underpin the #emph[Web Ontology Language (OWL)].
#emph[RDF] and #emph[SPARQL] give the Semantic Web a concrete triple-based
data model and query language, and large-scale #emph[knowledge graphs] and
#emph[semantic networks], from WordNet and ConceptNet to WikiData and
DBpedia, apply these ideas at the scale of real-world question answering,
search, and recommendation.

// From: msml610/lectures_source/Lesson03.3-Non_classical_logics.smd:932 '* References'
// Slide: References
= References

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
