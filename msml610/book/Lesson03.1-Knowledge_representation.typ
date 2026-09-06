// git_hash=bf95ac2c8-l69 timestamp=20260905_164943
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
  title: "L03.1: Knowledge Representation",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L03.1: Knowledge Representation")

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:7 '# Knowledge Representation'
// Slide: Knowledge Representation
#strong[Knowledge Representation]

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:9 '* Roadmap'
// Slide: Roadmap
#strong[Roadmap]

Knowledge representation is the study of how to encode what an agent knows so
that it can reason, plan, and act effectively. The progression here moves from
foundational motivations to concrete mechanisms: first understanding #emph[why]
explicit representation matters, then examining #emph[how] knowledge is
structured, interpreted, and put to work.

The foundations establish what knowledge representation actually is and the core
design tradeoffs every representation scheme must navigate: expressiveness
versus computational tractability, generality versus domain specificity, and
formal precision versus ease of authoring. From there, the discussion turns to
the #emph[languages] available for encoding knowledge, spanning natural
language, programming languages, propositional logic, and first-order logic,
each offering a different point in that tradeoff space. Semantics then pins down
what it means for a knowledge base to be "about" the world: how symbols are
grounded in referents, what a model is, and what it means for a sentence to be
satisfied. Reasoning builds on that semantic foundation by defining entailment
(what follows from what), inference (the mechanical process of deriving new
sentences), and the twin guarantees of soundness and completeness that connect
the two. With these pieces in place, the focus shifts to agents that actually
use represented knowledge: from simple reflex agents, through rule-based
systems, to full knowledge-based agents that maintain an internal knowledge base
and query it before acting. Finally, ontologies provide the large-scale
organizational scaffolding, specifying the categories, relations, and axioms
that let knowledge be shared and reused across tasks and domains.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:21 '## What Is Knowledge Representation'
// Slide: What Is Knowledge Representation
== What Is Knowledge Representation

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:23 '* Defining Knowledge Representation'
// Slide: Defining Knowledge Representation
#strong[Defining Knowledge Representation]

#strong[Knowledge Representation (KR)] is the study of how to formally encode
information so that machines can reason with it. Rather than storing raw data,
KR provides a structured language for capturing facts, relationships, and rules
in a form that automated systems can manipulate. Common formalisms include
production rules, first-order logic, ontologies, and semantic networks, each
offering different tradeoffs between expressiveness and computational
tractability.

KR defines two essential aspects of any encoding. #emph[Structure] determines
how knowledge is organized: whether as a flat set of propositions, a hierarchy
of classes and instances, or a graph of interconnected concepts.
#emph[Semantics] determines what the encoded statements actually mean: the
formal interpretation that lets a reasoner distinguish valid inferences from
invalid ones. Without clear semantics, a knowledge base is just syntax; without
clear structure, it becomes unwieldy as the domain grows.

Machines rely on knowledge representation to carry out core reasoning tasks.
Given a well-formed knowledge base, an inference engine can #emph[draw
  conclusions] that were not stated explicitly, deriving new facts from existing
ones through logical entailment. KR also supports #emph[planning], where an
agent searches through possible action sequences by reasoning about
preconditions and effects encoded in its knowledge base. Finally, KR enables
systems to #emph[answer queries] posed by users or other software components,
retrieving not just stored facts but also their logical consequences.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:38 '* Why Knowledge Representation Matters'
// Slide: Why Knowledge Representation Matters
#strong[Why Knowledge Representation Matters]

Why is knowledge representation essential to AI? There are several compelling
reasons that go beyond what learning from data alone can achieve.

First, learning alone is not enough. Machines need to #emph[reason] about the
world, not just recognize patterns. A medical AI trained on patient data can
#emph[predict] diseases with impressive accuracy, but to #emph[explain] a
diagnosis to a doctor, it needs structured knowledge that captures relationships
between symptoms, conditions, and treatments.

Second, knowledge representation bridges perception and reasoning. Sensors
provide #emph[perception] in the form of raw data, but that data is not
inherently meaningful. Knowledge representation turns raw data into
#emph[actionable knowledge] through #emph[reasoning], allowing a system to move
from "these pixels form a stop sign" to "I should decelerate the vehicle."

Third, knowledge representation enables explainability. When a system's
knowledge is explicitly represented, users can understand #emph[why] it made a
particular decision. This is critical for high-stakes domains such as
healthcare, law, and autonomous systems, where a black-box prediction is often
insufficient and sometimes legally unacceptable.

Fourth, knowledge representation enables planning and abstract reasoning. Robots
plan actions using abstract symbolic knowledge: they do not re-derive the
concept of "door" from pixel data every time they need to navigate a room.
Similarly, conversational agents reason about intent and context, mapping a
user's words to goals and selecting responses that advance those goals rather
than simply pattern-matching against a training corpus.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:54 '## Design Choices in Knowledge Representation'
// Slide: Design Choices in Knowledge Representation
== Design Choices in Knowledge Representation

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:56 '* Expressiveness Vs. Tractability'
// Slide: Expressiveness Vs. Tractability
#strong[Expressiveness Vs. Tractability]

In knowledge representation, a fundamental #strong[trade-off] exists between two
competing goals. #emph[Expressiveness] refers to the richness of concepts a
language can capture: how much detail and nuance it can encode about the world.
#emph[Tractability] refers to whether reasoning in that language can be
performed efficiently, ideally in polynomial time or better. As
@fig:expressivenessvstractability illustrates, these two properties pull in
opposite directions: more expressive languages lead to harder computation, often
pushing reasoning problems into undecidable or intractable territory. Choosing
the right knowledge representation therefore depends heavily on the application
and the balance it demands between descriptive power and computational
feasibility.

// rendered_images:begin
// ```tikz
// \begin{axis}[
//   width=12cm,
//   height=8cm,
//   axis lines=middle,
//   xlabel style={
//     at={(axis description cs:0.5,-0.05)},
//     anchor=north,
//     font=\Huge\bfseries
//   },
//   ylabel style={
//     at={(axis description cs:-0.02,0.5)},
//     anchor=south,
//     rotate=90,
//     font=\Huge\bfseries
//   },
//   xlabel={\textbf{Expressiveness}},
//   ylabel={\textbf{Tractability}},
//   xtick=\empty,
//   ytick=\empty,
//   xmin=0.5, xmax=10,
//   ymin=0, ymax=10,
//   domain=1:9,
//   samples=100,
//   enlargelimits=true,
//   clip=false,
//   ]
// 
//   % Tradeoff curve (dashed, ultra thick hyperbola)
//   \addplot[domain=1:9, ultra thick, dashed, blue] {10 / x};
// 
//   % Points
//   \addplot[only marks, mark=*] coordinates {(2,5)} node[above right, font=\huge\bfseries] {\textbf{Atomic}};
//   \addplot[only marks, mark=*] coordinates {(5,2)} node[above right, font=\huge\bfseries] {\textbf{Factored}};
//   \addplot[only marks, mark=*] coordinates {(8,1.25)} node[below right, font=\huge\bfseries] {\textbf{Structured}};
// \end{axis}
// ```
// label=fig:expressivenessvstractability
// caption=Diagram illustrating the tradeoff between expressiveness and tractability across atomic, factored, and structured representations.
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson03.1-Knowledge_representation.typ.figs/Lesson03.1-Knowledge_representation.1.png"),
  caption: [Diagram illustrating the tradeoff between expressiveness and tractability across atomic, factored, and structured representations.],
) <fig:expressivenessvstractability>
// render_images:end
Expressiveness Vs. Tractability

Three broad levels of representation illustrate this spectrum:

- #emph[Atomic representations] treat each state as a single, indivisible
  entity. A chess position, for instance, is one opaque state in a game-tree
  search. This is simple and computationally friendly, but it cannot express
  internal structure or relationships between parts of a state.
- #emph[Factored representations] capture simple relationships between variables
  without imposing deeper structure. Propositional logic is the canonical
  example: a sentence like $B_(1,2) arrow.l.r (P_(1,1) or P_(2,2) or P_(1,3))$
  encodes the Wumpus World rule that a breeze at cell (1,2) holds if and only if
  a pit occupies one of its adjacent cells. Each variable is a Boolean fact, and
  reasoning stays decidable, though it can be expensive in the worst case.
- #emph[Structured representations] add the most expressive power by allowing
  objects, relations, and quantification. First-order logic, for example, lets
  us write
  $forall x space forall y space "Father"(x, y) arrow.r.double "Parent"(x, y)$, stating
  that any father of any person is also that person's parent. This single
  sentence covers every pair of individuals at once, something propositional
  logic cannot do without enumerating each pair explicitly. The cost is that
  reasoning in full first-order logic is undecidable in general.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:123 '* Symbolic Vs. Sub-symbolic Representation'
// Slide: Symbolic Vs. Sub-symbolic Representation
#strong[Symbolic Vs. Sub-symbolic Representation]

#strong[Symbolic knowledge representation] uses discrete, human-readable symbols
to encode what a system knows. Logic formulas and knowledge graphs are classic
examples: a fact like `parent(alice, bob)` is immediately interpretable by a
human reader, and a rule engine can chain such facts together to derive new
conclusions. This transparency makes symbolic representations well suited for
rule-based reasoning, where every inference step can be inspected and justified.
The tradeoff is that symbolic systems struggle with ambiguity; real-world
language and perception are full of graded, context-dependent meanings that do
not reduce neatly to crisp logical predicates.

#strong[Sub-symbolic knowledge representation] takes the opposite path, encoding
knowledge as learned, distributed representations rather than explicit symbols.
Vector embeddings are the prototypical example: a word, sentence, or image is
mapped to a point in a high-dimensional space, and similarity in that space
captures semantic relationships that would be tedious to hand-code. Sub-symbolic
representations excel at handling the very ambiguity that defeats symbolic
methods, but they lack transparency. A 768-dimensional vector for the concept
"dog" does not explain _why_ it sits near "wolf" and far from "democracy" in any
way a domain expert can audit.

@fig:symbolicvssubsymbolic illustrates the contrast between these two paradigms,
highlighting how symbolic and sub-symbolic representations differ in their
structure, interpretability, and tolerance for ambiguity.

#figure(
  image(
    "../lectures_source/figures/L03.symbolic_vs_subsymbolic.png",
    width: 80%,
  ),
  caption: [symbolic vs subsymbolic],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:symbolicvssubsymbolic>

#strong[Neuro-symbolic knowledge representation] blends both approaches, aiming
to get the best of each. A neuro-symbolic system might learn distributed
representations from raw data (the sub-symbolic side) and then reason over those
learned concepts using structured logical rules (the symbolic side). For
instance, a vision model could learn to recognize objects as embeddings, while a
symbolic planner uses those recognized objects as typed constants in a planning
domain. The result is a system that can handle perceptual ambiguity through its
neural components while still producing interpretable, auditable reasoning
chains through its symbolic layer.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:150 '* Conceptual Spaces'
// Slide: Conceptual Spaces
#strong[Conceptual Spaces]

#strong[Conceptual spaces] are frameworks for representing knowledge using
geometric structures. Each dimension of the space corresponds to an
interpretable feature, and similarity between objects is modeled directly by
spatial distance: the closer two points lie, the more alike the objects they
represent. A #strong[concept], in this framework, is simply a region in that
multidimensional space.

This geometric grounding gives conceptual spaces a natural way to handle
similarity and vagueness. Symbolic systems, by contrast, represent categories
with discrete tokens such as `Car` or `Bicycle` that carry no built-in
similarity structure; nothing in the symbol itself tells you how alike a car and
a bicycle are. In a conceptual space, their proximity along shared dimensions
(number of wheels, passenger capacity, speed) does that work automatically.

Consider transportation methods as a concrete illustration. Suppose the space
has two dimensions: #emph[Environmental Friendliness] and #emph[Technological
  Advancement]. In this space, `Wooden` artifacts (shown as a red region in
@fig:conceptualspaces) and `Vehicle` (a blue region) overlap at the point
labeled `dugout canoe`, which is both wooden and a vehicle. Items like
`Bicycle`, `Motorbike`, and `Car` form nested subregions inside the `Vehicle`
region, ordered by increasing technological advancement, while `Elevator` sits
inside `Vehicle` but outside any of those subregions, reflecting its distinct
feature profile.

#figure(
  image("../lectures_source/figures/L03.conceptual_spaces.png", width: 80%),
  caption: [conceptual spaces],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:conceptualspaces>

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:180 '## Languages for Representing Knowledge'
// Slide: Languages for Representing Knowledge
== Languages for Representing Knowledge

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:182 '* Natural Languages'
// Slide: Natural Languages
#strong[Natural Languages]

#strong[Natural languages], such as English or Italian, differ fundamentally
from formal languages in three key respects. First, they are _expressive_:
natural language evolved as a medium for communication between people rather
than as a system for precise representation of knowledge. Second, they are
_ambiguous_: a single word can carry multiple unrelated meanings. The word
"spring," for instance, refers both to a season and to a coiled object that
stores mechanical energy. Third, they are _context-dependent_: the meaning of an
utterance shifts depending on the sentence it appears in and the situation in
which it is used. A one-word exclamation like "Look!" conveys entirely different
information depending on whether the speaker is pointing at a sunset or warning
of an oncoming car.

The #strong[Sapir-Whorf hypothesis] is the contested claim that "the language
you speak shapes how you perceive, think about, and experience the world,"
extending even to arbitrary grammatical features such as the grammatical gender
assigned to nouns. The strong version of the hypothesis, which holds that
language _determines_ what a person can think, has been rejected by the
evidence. What survives is a weaker form: language exerts a modest influence on
habitual attention, nudging speakers to notice or categorize certain
distinctions more readily without making other thoughts impossible.

Consider a few illustrations. Some languages lack dedicated words for concepts
that other languages name easily; certain Indigenous Australian languages, for
example, use absolute cardinal directions where English speakers would say
"left" or "right." Conversely, some languages carve a domain into finer
categories than others. Russian treats dark blue (#emph[siniy]) and light blue
(#emph[goluboy]) as two distinct basic color terms rather than as shades of a
single color, and experimental evidence suggests Russian speakers are slightly
faster at discriminating blues that cross that boundary. A fictional but vivid
case is Newspeak in George Orwell's _1984_, a language deliberately stripped of
vocabulary so that certain politically dangerous concepts become harder to
articulate. Orwell's premise rests on the strong form of the hypothesis: if you
remove the words, you remove the capacity for the thought itself. While that
extreme claim does not hold up empirically, the example dramatizes how
vocabulary can channel or constrain the ease with which ideas are expressed and
shared.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:207 '* Procedural vs Declarative Approaches'
// Slide: Procedural vs Declarative Approaches
#strong[Procedural vs Declarative Approaches]

#strong[Procedural approach] focuses on _how_ a task is done: it encodes the
desired behavior directly into the program as an explicit sequence of
instructions. Consider a robot navigating a maze: in the procedural style, the
programmer writes out every turn and straight segment the robot should follow,
step by step. The result is precise and predictable, but brittle: if the maze
changes, the entire program must be rewritten.

#strong[Declarative approach], by contrast, specifies _what_ the goal is without
dictating how to reach it. Instead of hand-coding each movement, you describe
the relationships between actions, states, and goals, then let the system search
for a solution on its own. For the same maze robot, a declarative specification
simply states "reach the exit," and the robot's inference engine figures out
which corridors to take. This buys flexibility and modularity at the cost of
less direct control over execution and a heavier computational burden: the
system needs a sufficiently powerful search or inference mechanism to turn that
abstract goal into concrete behavior.

#figure(
  styled-table(
    headers: ("Approach", "Strengths", "Weaknesses"),
    rows: (
      (
        "Procedural",
        "More control over execution; explicit steps",
        "Less flexible; harder to modify or extend",
      ),
      (
        "Declarative",
        "More abstract; easier to modify, extend, and reason about goals",
        "Less control; harder to optimize; may require more powerful inference engines",
      ),
    ),
    bold-first-col: true,
  ),
  caption: [Comparison of procedural and declarative approaches to encoding
    knowledge.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:proceduraldeclarative>

@tab:proceduraldeclarative summarizes the core tradeoff: procedural
representations give the designer fine-grained control but resist change, while
declarative representations are easier to extend and reason about but demand
more from the underlying solver.

In practice, many successful AI systems use a hybrid of both styles. Declarative
knowledge can be _compiled_ into procedural code: a classical planner, for
instance, takes a declarative goal specification and automatically generates a
procedure (a plan) that achieves it. The declarative layer keeps the system's
knowledge modular and human-readable, while the compiled procedural output
delivers efficient execution. This compilation step bridges the gap between the
two paradigms, capturing the strengths of each.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:230 '* Limits of Programming Languages'
// Slide: Limits of Programming Languages
#strong[Limits of Programming Languages]

A #strong[programming language] such as C++ or Python is a formal, procedural
language in which data structures represent facts about the world and code
updates those structures in domain-specific ways. This approach is powerful but
comes with real limitations. First, programming languages lack a general
mechanism for deriving new facts from existing ones; any such derivation must be
hand-coded by the programmer using their own domain knowledge. Second, they
cannot gracefully handle partial information. A variable holds a single value or
remains unknown entirely; there is no built-in way to express something like "a
white knight is on b1 or on f6" or to quantify uncertainty about which square it
occupies.

A #strong[declarative language] addresses these shortcomings by separating
knowledge from inference. In a declarative framework, knowledge captures the
domain-specific problem: the particular facts, rules, and relationships relevant
to the task at hand. Crucially, the meaning of any sentence in the language is
built up from the meanings of its parts, a property known as #emph[compositional
  semantics]. Inference, by contrast, is domain-independent: it is a
general-purpose engine that derives new conclusions from whatever knowledge it
is given, without needing custom code for each application. Propositional logic
and first-order logic are canonical examples of declarative languages that
embody this clean separation.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:252 '* Propositional Logic'
// Slide: Propositional Logic
#strong[Propositional Logic]

#strong[Propositional logic] uses atomic statements, called propositions, and
logical connectives to represent knowledge. Its syntax is built from atomic
formulas such as $P$ and $Q$, combined through connectives: NOT ($not$), AND
($and$), OR ($or$), and IMPLIES ($arrow.r.double$). The semantics of
propositional logic rest on truth tables, where each proposition takes a binary
truth value, either true or false, and the truth value of any compound sentence
is determined entirely by the truth values of its atoms.

Two inference mechanisms are central to reasoning in propositional logic.
#emph[Modus ponens] lets you derive $Q$ whenever you know both $P$ and
$P arrow.r.double Q$. #emph[Resolution] works by deriving contradictions: you
negate the desired conclusion, add that negation to the knowledge base, and show
that a contradiction follows, thereby confirming the original conclusion.

Propositional logic is best suited to closed, well-defined environments where
the set of relevant facts is fixed and finite. Practical applications include:

- #emph[Digital circuit design], where each gate's output is a Boolean function
  of its inputs.
- #emph[Rule-based systems], where domain knowledge is encoded as if-then rules
  over binary conditions.
- #emph[Simplified AI models], where the world can be captured by a finite list
  of true-or-false statements.

These strengths come with real costs. Propositional logic cannot represent
objects, relations between objects, or quantified statements ("for all" or
"there exists"). A sentence like "every student enrolled in the course has
completed the prerequisite" has no direct encoding: you would need a separate
proposition for each individual student. This makes propositional logic
unsuitable for open or dynamic domains where the number of entities is unknown
or changes over time, motivating the move to more expressive formalisms such as
first-order logic.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:276 '* First-Order Logic (FOL)'
// Slide: First-Order Logic (FOL)
#strong[First-Order Logic (FOL)]

#strong[First-order logic] extends propositional logic by introducing the
machinery needed to talk about individual objects and their properties, rather
than just manipulating whole propositions as atomic units. Its syntax retains
the atomic formulas and connectives of propositional logic but adds three
critical ingredients: #emph[variables] such as $x$, #emph[predicates] such as
$"Human"(x)$ that express properties of or relations among objects, and
#emph[quantifiers] that range over those objects. The universal quantifier
$forall$ ("for all") and the existential quantifier $exists$ ("there exists")
let us make statements about entire collections at once.

Semantically, first-order logic can represent far more complex and structured
knowledge than propositional logic alone. It models properties of individual
objects, relationships between them, and quantified claims over whole domains.
For instance, the formula $forall x ("Human"(x) arrow.r.double "Mortal"(x))$ encodes
the statement "all humans are mortal," binding the variable $x$ so the
implication applies to every entity in the domain rather than to a single fixed
proposition.

First-order logic comes equipped with several inference mechanisms that make
automated reasoning possible:

- #emph[Unification]: matches predicates by finding substitutions for variables
  that make two expressions identical, enabling general rules to apply to
  specific cases.
- #emph[Resolution]: combines pairs of clauses to deduce new facts from known
  statements, forming the backbone of many automated theorem provers.
- #emph[Model checking]: verifies whether a given statement holds true under a
  specific interpretation, assigning concrete objects and relations to the
  logic's symbols.

These capabilities make first-order logic a workhorse across several areas of AI
and computer science: representing structured knowledge in expert systems and
databases, powering automated theorem proving where new conclusions must be
derived mechanically from axioms, and underpinning the semantic web and
ontologies that give machine-readable meaning to data shared across the
internet.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:299 '## Knowledge Bases and Their Semantics'
// Slide: Knowledge Bases and Their Semantics
== Knowledge Bases and Their Semantics

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:301 '* Syntax and Semantics of a KB'
// Slide: Syntax and Semantics of a KB
#strong[Syntax and Semantics of a KB]

A #strong[knowledge base (KB)] is a set of sentences and rules that together
encode what is known about the world. The sentences, often denoted α, express
assertions that may be observed directly, assumed as background knowledge, or
derived through reasoning. For instance, sentences like "it rains," "the ground
is dry," or "the ground is wet" each state a fact about the current state of
affairs. Rules, on the other hand, capture relationships between facts: "if it
rains, the ground gets wet" connects one assertion to another via a conditional
dependency. Together, these two components give a knowledge-based agent the raw
material it needs for reasoning.

To build a knowledge base, one needs a #strong[knowledge representation
  language]: a formal system for writing sentences about the world. Every such
language has two parts. Its #emph[syntax] specifies which strings count as
well-formed sentences. In arithmetic, for example, "$x + y = 4$" is
syntactically valid, while "$x 4 y + =$" is not. Its #emph[semantics] assigns
meaning to each well-formed sentence by saying whether it is true or false in
each possible world. The sentence $x + y = 4$ is true in the world where $x = 2$
and $y = 2$, but false in the world where $x = 1$ and $y = 1$. Syntax without
semantics gives you strings you can manipulate but cannot interpret; semantics
without syntax gives you meanings you cannot write down.

An #strong[axiom] is a sentence that is taken as given, not derived from other
sentences. Axioms form the foundational assumptions of a knowledge base, the
starting points from which everything else follows. #strong[Inference] is the
complementary process: deriving new sentences from existing ones by applying
logical reasoning rules. A sound inference procedure guarantees that any
sentence it produces is true whenever the sentences it started from are true, so
the knowledge base grows without introducing falsehoods.

Finally, not every logic treats truth the same way. In classical logic, each
sentence is simply true or false with no middle ground. #strong[Fuzzy logic]
relaxes this binary view by allowing degrees of truth, so a sentence α might
have $"Truth"(alpha) = 0.5$, indicating partial truth. #strong[Probabilistic
  logic] takes a different approach: rather than assigning a truth degree, it
assigns a probability that the sentence is true, such as $Pr(alpha) = 0.3$.
These alternatives matter in practice because real-world knowledge is rarely
black and white; choosing the right truth-value framework shapes what kinds of
uncertainty a knowledge-based agent can represent and reason about.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:329 '* Grounding'
// Slide: Grounding
#strong[Grounding]

#strong[Grounding] connects abstract symbols to real-world entities or
observations. In a knowledge base, a symbol like `Apple` is just a string with
no inherent meaning; grounding is the bridge that links it to the actual fruit
you can see and hold. Without this connection, a reasoning system manipulates
tokens according to syntactic rules but has no way to verify whether its
conclusions correspond to anything in the physical world.

The goal of grounding is to make representations meaningful beyond their syntax.
An agent that can ground its symbols is able to act meaningfully in the real
world: it recognizes that the symbol `Apple` refers to the red object on the
table, not just to a node in a graph. Without grounding, even a logically
flawless inference engine is performing purely symbolic manipulation with no
real-world relevance.

Grounding is far from straightforward. Sensory data is noisy and incomplete: a
camera image of an apple may be partially occluded, poorly lit, or taken from an
unusual angle. The mapping from raw inputs to abstract concepts is also
context-dependent. The word "bank" grounds to a financial institution in one
conversation and to the edge of a river in another, and resolving this requires
situational knowledge that goes well beyond pattern matching on pixels or
characters.

These challenges make grounding a central concern across several applied
domains. In robotics, grounding enables object recognition and manipulation: a
robot must connect its internal label for "cup" to the specific object it needs
to grasp. In natural language understanding, grounding ties words and phrases to
the entities and events they describe, which is essential for tasks like visual
question answering or instruction following. More broadly, autonomous agents and
cognitive systems depend on grounded representations to close the loop between
perception, reasoning, and action in open-ended environments.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:360 '* Can We Trust Grounding?'
// Slide: Can We Trust Grounding?
#strong[Can We Trust Grounding?]

How do we know that the symbols in our knowledge base actually correspond to
anything real? Strictly speaking, we cannot be certain: this touches on deep
philosophical questions about the nature of reality itself. In practice, we
simply #strong[assume] that the knowledge base is correct, that the mapping
between symbols and the world they represent is trustworthy.

This assumption rests on the idea that an agent's sensors faithfully translate
real-world events into symbolic sentences. For instance, when the agent detects
a burning smell, it creates a knowledge base entry:

```
IF smell = burning THEN food_is_burning
```

From accumulated entries like this, the agent learns general rules and acts on
them:

```
IF food_is_burning THEN turn_off_stove
```

The agent moves from particular observations to general rules, a process we also
assume to be typically correct. Yet learning is still fallible: the smell of
burning might not mean the food is on fire at all. Perhaps a neighbor is cooking
on an outdoor grill. The symbol `food_is_burning` would then be asserted in the
knowledge base even though nothing in the agent's own environment is actually
burning. This gap between what the sensors report and what is truly happening is
a persistent limitation of any knowledge-based system, and it is one reason why
robust agents need mechanisms for revising their beliefs when new evidence
contradicts old conclusions.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:381 '* Models and Possible Worlds'
// Slide: Models and Possible Worlds
#strong[Models and Possible Worlds]

Consider a world with two Boolean variables, _rain_ and _wet ground_. Each
possible world (or #strong[model]) assigns a truth value to every relevant
variable, producing a complete snapshot of how things might be. With just these
two variables there are four models: $("Rain" = T, "WetGround" = T)$,
$("Rain" = T, "WetGround" = F)$, $("Rain" = F, "WetGround" = T)$, and
$("Rain" = F, "WetGround" = F)$. A model $m$ is the mathematical abstraction that
captures one such possible world; for instance, $m$ might be the assignment
$("Rain" = F, "WetGround" = T)$, representing a world where the ground is wet even
though it has not rained.

// rendered_images:begin
// ```graphviz
// digraph G {
//   rankdir=TD;
//   nodesep=3.5;
//   node [shape=box, style="rounded,filled", fillcolor="#f7f7f7"];
// 
//   Model [label="Model"];
//   Worlds [label="Possible\nworlds"];
// 
//   Model -> Worlds [
//     dir=both
//     fontsize=10
//     penwidth=2
//     label="   Grounding"
//     labeldistance=3.0
//     labelangle=0
//   ];
// }
// ```
// label=fig:modelsandpossibleworlds
// caption=Diagram relating a model to the possible worlds it grounds.
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson03.1-Knowledge_representation.typ.figs/Lesson03.1-Knowledge_representation.2.png"),
  caption: [Diagram relating a model to the possible worlds it grounds.],
) <fig:modelsandpossibleworlds>
// render_images:end
worlds and Grounding

As @fig:modelsandpossibleworlds illustrates, a model serves as the formal bridge
between the abstract notion of "possible world" and the concrete variable
assignments that ground our reasoning.

Now consider a richer scenario: men and women sitting at a table. Here the model
represents every possible world as "there are $x$ men and $y$ women." A sentence
such as $x + y = 4$ is true in some of these worlds and false in others. In a
world where $x = 2$ and $y = 2$, the sentence holds; in a world where $x = 3$
and $y = 3$, it does not. The key insight is the same as in the rain example: a
sentence is not true or false in isolation, but true or false _with respect to a
particular model_. Once you fix the model, every well-formed sentence receives a
definite truth value.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:426 '* Satisfaction'
// Slide: Satisfaction
#strong[Satisfaction]

Recall that a #strong[model] $m$ fixes the values of every variable
$x_1, dots, x_n$ that appears in the sentences of interest. For instance, the
assignment $(italic("Rain") = T, italic("WetGround") = T)$ is one model over two
Boolean variables; it pins down a single complete state of affairs.

Given a model, we can ask whether a particular sentence comes out true under
that assignment. If a sentence α is true in model $m$, we say that #strong[the
  model $m$ satisfies α], sometimes written $m models alpha$. For example, the
model $(italic("Rain") = T, italic("WetGround") = F)$ satisfies the sentence
$alpha: italic("Rain") = T$, because Rain is indeed true in that assignment,
regardless of what WetGround happens to be. The phrasing may feel backwards at
first: in everyday reasoning we usually think of the world as fixed and then
check which claims hold, but in logic the convention runs the other way,
treating the sentence as given and asking which worlds make it true.

This convention motivates a useful piece of notation. We write $M(alpha)$ for
#strong[the set of all models in which α is true]. Continuing the example above,
suppose the language has just two Boolean variables, Rain and WetGround, so
there are four models in total. The sentence $alpha: italic("Rain") = T$ is true
in exactly two of them:

$
  M(italic("Rain") = T) = {(italic("Rain") = T, italic("WetGround") = T), thin (italic("Rain") = T, italic("WetGround") = F)}
$

The two models that do not appear in $M(alpha)$ are the ones where Rain is
false. Thinking of $M(alpha)$ as a subset of the full model space will be
central to defining entailment and validity in the sections ahead: a sentence
that is true in every model has $M(alpha)$ equal to the entire space, while a
contradiction has $M(alpha) = emptyset$.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:442 '## Entailment and Inference'
// Slide: Entailment and Inference
== Entailment and Inference

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:444 '* Logical Entailment'
// Slide: Logical Entailment
#strong[Logical Entailment]

#strong[Logical entailment] between sentences is the relationship that holds
when one sentence follows logically from another in a knowledge base. Formally,
"α entails β" (written $alpha models beta$) means that in every model in which α
is true, β is also true. An equivalent way to state this is
$M(alpha) subset.eq M(beta)$: the set of models satisfying α is a subset of the
models satisfying β.

Consider the "rain and wet ground" world. Suppose
$"KB" = {"Rain", "Rain" arrow.r "WetGround"}$. This knowledge base entails
$"WetGround"$ because in every model where $"Rain"$ holds and
$"Rain" arrow.r "WetGround"$ holds, $"WetGround"$ must also hold. There is
simply no model that satisfies the KB yet violates $"WetGround"$.

As a second illustration, take a simple arithmetic world where α is "$x = 0$"
and β is "$x dot.op y = 0$." Here α entails β because in any model where $x = 0$
is true, $x dot.op y = 0$ is necessarily true regardless of the value of $y$.
The truth of α constrains the world tightly enough that β cannot fail.

Entailment is not tied to any particular proof procedure; it simply preserves
truth across all models. Think of it as a constraint on consistent belief: if
you believe the sentences in your KB, you #emph[must] believe the entailed
sentences as well. No valid interpretation of the world lets you accept the
premises while rejecting the conclusion.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:468 '* Entailment, Implication, and Inference'
// Slide: Entailment, Implication, and Inference
#strong[Entailment, Implication, and Inference]

Entailment, implication, and inference are three concepts that often get
conflated, but they operate at different levels. #emph[Implication] is a
relationship between two statements that lives inside the logic itself.
#emph[Logical entailment] is a meta-level guarantee that truth necessarily
follows from known facts across every possible world. #emph[Inference] is the
reasoning process an agent actually uses to derive new truths from what it
already knows.

#strong[Implication] ($A arrow.r.double B$) is the logical statement "if $A$ is
true, then $B$ is true." It does not, by itself, guarantee that either $A$ or
$B$ holds; it only constrains their relationship. The truth table for
implication declares it false in exactly one case: when $A$ is true and $B$ is
false. In every other combination, the implication holds. For instance, let $A$
be "it is raining" and $B$ be "the ground is wet." The statement
$A arrow.r.double B$ can be true even when it is not raining, because the
implication makes no claim about what happens when $A$ is false.

#strong[Logical entailment] ($"KB" models alpha$) is a stronger, meta-level
notion: it means that $alpha$ is true in every model where the knowledge base
$"KB"$ is true. Consider whether $"Rain"$ entails $"WetGround"$. For that to
hold, every single model in which $"Rain"$ is true must also make $"WetGround"$
true. If there exists even one model where $"Rain" = T$ and $"WetGround" = F$,
the entailment fails. So $"Rain"$ does not entail $"WetGround"$ in general
unless additional rules (like $"Rain" arrow.r.double "WetGround"$) are part of
the knowledge base.

#strong[Inference] is what an agent, whether a person or a computer, actually
does when it figures out new truths by reasoning from what it knows. It starts
with some established truths and applies reasoning steps to arrive at new ones.
The goal of any sound inference procedure is to track entailment: every
conclusion it produces should be one that is genuinely entailed by the premises.

To summarize: implication is a statement that lives inside the logic, entailment
is a meta-level relationship asserting that truth follows across all models, and
inference is the reasoning process that, when done correctly, mirrors
entailment.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:498 '* Inference Engines'
// Slide: Inference Engines
#strong[Inference Engines]

An #strong[inference engine] is the mechanism that applies logical rules to a
knowledge base to derive new conclusions or answer queries. Three classical
strategies drive this process:

- #emph[Forward chaining] starts from known facts and repeatedly applies
  inference rules to extract new data. For instance, given $A arrow.r.double B$
  and the fact $A$, the engine infers $B$, then checks whether that new fact
  triggers further rules.
- #emph[Backward chaining] works in the opposite direction: it begins with a
  goal and searches backward for rules and facts that would establish it. To
  prove $B$, the engine looks for a rule $A arrow.r.double B$ and then attempts
  to prove $A$.
- #emph[Resolution] is a single, complete inference rule applicable to both
  propositional and first-order logic. By converting sentences to conjunctive
  normal form and systematically resolving complementary literals, it can
  determine whether a query follows from the knowledge base, making it a
  workhorse of automated theorem proving.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:512 '* Model Checking'
// Slide: Model Checking
#strong[Model Checking]

Recall that $M("KB")$ represents the set of all models, or possible worlds, in
which our knowledge base $"KB"$ is true. The central question is whether a
sentence α is entailed by the knowledge base, written $"KB" models alpha$. By
definition, this holds exactly when α is true in every model where $"KB"$ is
true, that is, when $M("KB") subset.eq M(alpha)$.

#strong[Model checking] provides a brute-force algorithm for answering this
question:

1. Enumerate all possible models (all assignments of truth values to the
  propositional symbols in the language).
2. Identify which of those models satisfy the knowledge base, giving the set
  $M("KB")$.
3. Verify that α is true in every model belonging to $M("KB")$.

If every model in $M("KB")$ also makes α true, then $"KB" models alpha$ holds. If
even a single model in $M("KB")$ falsifies α, the entailment fails. The procedure
is conceptually straightforward but computationally expensive: the number of
models grows exponentially with the number of propositional symbols, making
exhaustive enumeration impractical for large knowledge bases. Nevertheless,
model checking serves as the reference baseline against which more efficient
inference methods are measured.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:532 '* Soundness and Completeness'
// Slide: Soundness and Completeness
#strong[Soundness and Completeness]

#strong[Inference] is a syntactic process of deriving new sentences from others
using the formal rules of a proof system (such as modus ponens, resolution, and
related techniques). Where entailment is a semantic relationship about what must
be true in all models, inference is a mechanical, symbol-pushing procedure: you
apply rules to the sentences you already have and produce new ones. Consider a
familiar example: you know "if it rains, the ground gets wet," you observe "it
is raining," and you infer "the ground must be wet." That final step is
inference: a syntactic derivation that produces a new sentence from existing
ones, following a rule (here, modus ponens) without ever inspecting models
directly.

Two properties determine whether an inference algorithm can be trusted. A
#strong[sound] inference algorithm derives only sentences that are entailed by
the knowledge base. In other words, whatever the algorithm finds is correct: it
never asserts something as following from $"KB"$ when it does not. There are no
false positives. A #strong[complete] inference algorithm can derive every
sentence entailed by the knowledge base. It never misses a valid conclusion:
there are no false negatives.

For instance, model checking (the brute-force enumeration of all possible truth
assignments) is both sound and complete when the space of models is finite. It
is sound because every model it finds consistent with $"KB" and alpha$ genuinely
makes α true, so it never asserts a false entailment. It is complete because
enumerating all finitely many models guarantees that every entailed sentence
will be discovered. No valid conclusion slips through the exhaustive search.

The ideal inference algorithm achieves both properties simultaneously: it finds
everything that follows from the knowledge base, and nothing that does not.
Soundness without completeness means the algorithm is cautious but potentially
ignorant; completeness without soundness means it finds everything but may
hallucinate conclusions. In practice, designing algorithms that are both sound
and complete is a central challenge in knowledge representation and reasoning,
and much of the theory of propositional and first-order logic is devoted to
identifying exactly which inference procedures achieve this ideal.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:558 '* Representation Mirrors the World'
// Slide: Representation Mirrors the World
#strong[Representation Mirrors the World]

A sound and complete inference algorithm guarantees that every conclusion it
derives is true in any world where the premises hold. Soundness means the
algorithm never produces a falsehood from true premises; completeness means it
eventually finds every conclusion that logically follows. Together, these
properties ensure that purely mechanical symbol manipulation can reliably track
real-world truths.

This guarantee rests on a crucial correspondence: even though the inference
engine operates entirely on syntax (the internal representation), the formal
structure mirrors reality in two specific ways. First, sentences in the
representation correspond to aspects of the real world. Second, entailment
between sentences in the representation corresponds to implication between
aspects of the real world. If sentence α entails sentence β within the formal
system, then whatever real-world fact α describes genuinely implies whatever β
describes. The syntactic derivation "follows" the semantic relationship, so
reasoning carried out inside the machine tracks reasoning about the actual
world.

// rendered_images:begin
// ```graphviz
// digraph EntailmentSemantics {
//   rankdir=TB;
//   node [shape=box, style=filled, fillcolor=lightgray];
// 
//   // Representation layer
//   Sentence1 -> Sentence2 [label="Entails", style=dashed];
// 
//   // World layer
//   RealWorld1 -> RealWorld2 [label="Follows", style=dashed];
// 
//   // Semantics arrows
//   Sentence1 -> RealWorld1 [label="Semantics", style=dashed];
//   Sentence2 -> RealWorld2 [label="Semantics", style=dashed];
// 
//   // Invisible edges to align Representation and World vertically
//   {rank=same; Sentence1; Sentence2}
//   {rank=same; RealWorld1; RealWorld2}
// 
//   // Labels for layers (optional)
//   subgraph cluster_representation {
//     label="Representation";
//     style=dotted;
//     Sentence1;
//     Sentence2;
//   }
// 
//   subgraph cluster_world {
//     label="World";
//     style=dotted;
//     RealWorld1;
//     RealWorld2;
//   }
// }
// ```
// label=fig:representationmirrorstheworld
// caption=Diagram relating entailment between sentences, consequence between world states, and the semantics that link representation to the world.
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson03.1-Knowledge_representation.typ.figs/Lesson03.1-Knowledge_representation.3.png"),
  caption: [Diagram relating entailment between sentences, consequence between world states, and the semantics that link representation to the world.],
) <fig:representationmirrorstheworld>
// render_images:end
Follows, Semantics and Representation

@fig:representationmirrorstheworld illustrates this parallel: on one side, the
semantics of the world determines what is actually true; on the other, the
representation and its entailment relation determine what can be derived. The
diagram shows that the "follows" relation within the formal system is anchored,
via semantics, to genuine facts about reality. This is what makes logic-based
agents trustworthy: their internal derivations are not arbitrary symbol games
but faithful mirrors of the relationships they represent.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:616 '## Logical Agents and Rule-Based Systems'
// Slide: Logical Agents and Rule-Based Systems
== Logical Agents and Rule-Based Systems

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:618 '* Reflex and Rule-Based Agents'
// Slide: Reflex and Rule-Based Agents
#strong[Reflex and Rule-Based Agents]

A #strong[reflex agent] acts on the current percept alone, mapping what it sees
right now to an action through a predefined condition-action rule: "if
condition, then action." Consider a thermostat: if the temperature drops below a
threshold, it turns on the heater. That is the entire decision process. A reflex
agent ignores percept history entirely; it has no memory of past states, so its
behavior depends only on what is true at this instant.

A #strong[rule-based system] generalizes the reflex idea by maintaining an
explicit set of "if-then" rules together with a memory of current facts. Three
components work together: a #emph[knowledge base] that stores both facts and
rules, an #emph[inference engine] that applies rules to known facts in order to
infer new facts or trigger actions, and a #emph[working memory] that holds the
facts currently under consideration.

The inference engine cycles through four steps:

1. #emph[Match]: find every rule whose conditions are satisfied by the current
  facts in working memory.
2. #emph[Conflict resolution]: when multiple rules match simultaneously, decide
  which one to fire (strategies range from choosing the most specific rule to
  prioritizing recently added facts).
3. #emph[Act]: apply the chosen rule, which may add new facts to working memory
  or trigger an external action.
4. #emph[Repeat]: return to step 1 and continue until no rule's conditions
  match, at which point the system halts.

To see this cycle in action, suppose the knowledge base contains the rule "if a
patient has a fever and a rash, then suggest measles." Working memory currently
holds two facts: the patient has a fever, and the patient has a rash. On the
first pass the engine matches both conditions, fires the rule, and adds the
conclusion "suggest measles" to working memory. If no further rules match the
updated facts, the cycle terminates with that diagnosis.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:643 '* Rule-Based Systems: Trade-offs'
// Slide: Rule-Based Systems: Trade-offs
#strong[Rule-Based Systems: Trade-offs]

Simple reflex agents offer several practical advantages. They are fast and
efficient in well-defined environments, since each percept maps directly to an
action with no deliberation overhead. Their condition-action rules are easy to
modify and update when the domain changes, and because every decision traces
back to an explicit rule, the agent's reasoning is fully transparent and
explainable. These properties make them a natural fit whenever expert knowledge
can be clearly articulated as a finite set of if-then rules.

The tradeoffs, however, are real. Without working memory, a simple reflex agent
cannot plan ahead or learn from past experience; it reacts to the current
percept alone. Scaling to large or complex domains is difficult because the rule
table grows combinatorially with the number of distinguishable situations. The
basic architecture also cannot handle uncertainty: extending it to noisy or
partially observable settings requires bolting on probabilistic reasoning or
other mechanisms that go beyond pure condition-action matching. As the rule set
grows, conflicts between rules and the sheer cost of maintaining consistency
become increasingly burdensome.

Despite these limitations, simple reflex agents see wide use in practice. Fully
observable control loops such as thermostats are the textbook case: the current
temperature reading is all the information the agent needs. Expert systems for
medical diagnosis and technical troubleshooting encode specialist knowledge as
condition-action rules and were among the earliest commercial successes of AI.
Business rule engines apply the same pattern to automate policy decisions in
domains like insurance underwriting or loan approval. Game AI for simple
opponents often relies on reflex rules mapping game states to moves. Legal
reasoning tools, where statutes and regulations can be expressed as structured
conditionals, represent another natural application of the architecture.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:665 '* Knowledge-based Agents'
// Slide: Knowledge-based Agents
#strong[Knowledge-based Agents]

Intelligence in this sense is achieved by _reasoning_ over an internal
representation of knowledge, not by reacting to raw sensor data with fixed
rules. The system builds a model of its world and draws conclusions from that
model before it acts.

A #strong[knowledge-based agent] maintains a knowledge base and operates through
two core operations. The first, TELL, adds a new percept or learned fact to the
knowledge base. The second, ASK, queries the knowledge base to determine what
action to take next. Between these two operations the agent performs
#emph[inference]: it combines existing knowledge with the new percept to derive
facts that were never stated explicitly, then uses those derived facts to select
an action.

This architecture gives the agent several distinctive capabilities. It can
accept tasks framed as high-level goal descriptions rather than step-by-step
instructions, because it reasons about how to reach the goal from its current
state of knowledge. It continuously updates that knowledge using information
from its sensors, so its picture of the world stays current. It can also explain
and justify its actions by pointing to the chain of reasoning that produced
them. A medical diagnosis system, for instance, infers which diseases are
consistent with a patient's symptoms and then suggests treatments grounded in
that inference. A chess program consults a database of positions and moves to
plan a strategy several turns ahead. When the available information is
incomplete or uncertain, the agent does not simply give up; it falls back on
probabilistic reasoning to weigh alternatives and choose the action most likely
to succeed given what it does know.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:685 '## Ontologies'
// Slide: Ontologies
== Ontologies

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:687 '* Ontologies'
// Slide: Ontologies
#strong[Ontologies]

An #strong[ontology] is a formal and explicit representation of a domain. It
describes the types of things that exist and how they relate to one another,
organized around three core building blocks: #emph[classes] (the types of
things), #emph[individuals] (specific objects that belong to those classes), and
#emph[properties] (the relationships that connect them).

Consider a medical ontology that defines how diseases, symptoms, and treatments
relate to each other: "influenza" is a class of disease, a particular patient's
flu episode is an individual, and "is treated by" is a property linking a
disease to a medication. A geographical ontology works the same way, describing
how cities belong to states and states belong to countries through hierarchical
containment properties.

The purpose of building an ontology is threefold. First, it provides a shared
vocabulary for a domain of knowledge, so that different systems and teams use
the same terms with the same meanings. Second, it enables reasoning about
entities and their relationships, letting an inference engine derive new facts
from the structure already encoded. Third, it allows machines and humans to
understand and share information consistently, bridging the gap between how a
database stores data and how a domain expert thinks about it.

Several related concepts are worth distinguishing. A #emph[database schema]
defines columns with fixed types and is generally more rigid than an ontology;
it enforces structure but says little about meaning. A #emph[taxonomy] is a
simpler, tree-like hierarchical classification (the Linnaean system for
biological species is a classic example) that captures "is-a" relationships but
typically lacks the richer property vocabulary an ontology supports. A
#emph[knowledge base] is a collection of facts and rules, and it is sometimes
built on top of an ontology: the ontology supplies the vocabulary and structural
constraints, while the knowledge base populates that structure with concrete
assertions and inference rules.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:711 '* Components of an Ontology'
// Slide: Components of an Ontology
#strong[Components of an Ontology]

An ontology is built from a set of recurring structural elements that, together,
let you represent any domain's knowledge precisely.

- #emph[Classes] (also called concepts) capture the general categories in a
  domain: `Person`, `City`, or `Car`. They define what kinds of things exist.
- #emph[Individuals] (or instances) are specific, concrete members of those
  classes: `GP` is an instance of `Person`, `Rome` is an instance of `City`, and
  `Ferrari 458` is an instance of `Car`.
- #emph[Properties] (or relations) describe how classes and instances interact
  with one another: `isMortal`, `locatedIn`, and `hasAge` each link a subject to
  an object, forming the edges of the knowledge graph.
- #emph[Attributes] (or data values) attach concrete data to instances rather
  than linking them to other entities: the triple (`GP`, `hasAge`,
  `<your_guess>`) records a numeric fact about a particular individual.
- #emph[Constraints] restrict the values a property may take. For instance,
  `hasColor` applied to `Car` might be constrained to range over a fixed set
  such as \{`red`, `yellow`, ...\}, preventing nonsensical assignments.
- #emph[Axioms] are logical statements that encode rules and invariances the
  ontology must respect. A classic example is "all humans are mortal," written
  formally as $forall x ("Human"(x) arrow.r.double "Mortal"(x))$. Axioms let a reasoner
  derive new facts from existing ones.
- #emph[Hierarchies] organize classes and properties into parent-child trees.
  Declaring that `Student` is a subclass of `Person` means every student
  automatically inherits the properties defined for persons, giving the ontology
  its characteristic layered structure.

These components do not exist in isolation; they interlock. Classes sit in
hierarchies, individuals instantiate classes, properties connect individuals to
one another or to data values, constraints and axioms govern what those
connections may look like, and the whole assembly forms a machine-readable model
of a domain that supports both querying and automated inference.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:730 '* Example: University Ontology'
// Slide: Example: University Ontology
#strong[Example: University Ontology]

An ontology organizes knowledge into a formal structure built from a few core
building blocks. The first of these is #strong[classes], which represent the
categories of things in the domain. In a university setting, the natural classes
are `Student`, `Professor`, `Course`, and `Department`. Each class groups
together all individuals that share a common role in the institution.

// rendered_images:begin
// ```graphviz
// digraph UniversityOntology {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//     penwidth=1.4];
// 
//     // Node styles
//     Student [label="Student", fillcolor="#A6C8F4"];
//     Professor [label="Professor", fillcolor="#A6E7F4"];
//     Course [label="Course", fillcolor="#B2E2B2"];
//     Department [label="Department", fillcolor="#FFD1A6"];
// 
//     Alice [label="Alice", fillcolor="#C6A6F4"];
//     Bob [label="Bob", fillcolor="#C6A6F4"];
//     GP [label="GP", fillcolor="#C6A6F4"];
//     DrNo [label="DrNo", fillcolor="#C6A6F4"];
//     DATA605 [label="DATA605", fillcolor="#D2B48C"];
//     MSML610 [label="MSML610", fillcolor="#D2B48C"];
//     ComputerScience [label="ComputerScience", fillcolor="#F4A6A6"];
//     Mathematics [label="Mathematics", fillcolor="#F4A6A6"];
// 
//     // Force ranks
//     //{rank=same; Student; Professor; Course; Department;}
// 
//     // Edges
//     Alice -> Student [label="instance of"];
//     Bob -> Student [label="instance of"];
//     GP -> Professor [label="instance of"];
//     DrNo -> Professor [label="instance of"];
//     DATA605 -> Course [label="instance of"];
//     MSML610 -> Course [label="instance of"];
//     ComputerScience -> Department [label="instance of"];
//     Mathematics -> Department [label="instance of"];
// 
//     Student -> Course [label="takesCourse"];
//     Professor -> Course [label="teachesCourse"];
//     Student -> Department [label="belongsToDepartment"];
//     Professor -> Department [label="belongsToDepartment"];
//     Department -> Course [label="offersCourse"];
// }
// ```
// label=fig:exampleuniversityontology
// caption=Diagram relating Student, Professor, Course, and Department entities and instances in an example university ontology.
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson03.1-Knowledge_representation.typ.figs/Lesson03.1-Knowledge_representation.4.png"),
  caption: [Diagram relating Student, Professor, Course, and Department entities and instances in an example university ontology.],
) <fig:exampleuniversityontology>
// render_images:end
Course and Department

The second building block is #strong[properties], which capture the
relationships between classes. For instance, `takesCourse` links a `Student` to
a `Course`, `teachesCourse` links a `Professor` to a `Course`, and
`belongsToDepartment` connects both students and professors to a `Department`.
These directed relationships, shown in @fig:exampleuniversityontology, make the
structure of the domain explicit: rather than leaving it implicit that
professors teach courses, the ontology encodes that link as a first-class
element that reasoners can query and check.

Third, #strong[individuals] are specific instances of classes. Alice and Bob are
individuals belonging to the `Student` class; GP and DrNo are individuals of
type `Professor`; DATA605 and MSML610 are individual `Course` instances.
Individuals populate the ontology with concrete facts that ground the abstract
class hierarchy in real data.

Finally, #strong[axioms] are logical rules that every valid state of the
ontology must satisfy. Two axioms illustrate the idea here: every `Course` must
be taught by exactly one `Professor`, and every `Student` must belong to exactly
one `Department`. These constraints do more than document expectations; a
reasoner can use them to detect inconsistencies (a course with no instructor, a
student claimed by two departments) and to infer missing facts (if a course
exists and only one professor is linked to it, that professor must be its
instructor). Together, classes, properties, individuals, and axioms give an
ontology enough structure to support both human understanding and automated
reasoning over the domain.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:805 '## Reasoning in Ontologies'
// Slide: Reasoning in Ontologies
== Reasoning in Ontologies

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:807 '* Class-Level Reasoning Tasks'
// Slide: Class-Level Reasoning Tasks
#strong[Class-Level Reasoning Tasks]

#strong[Subsumption] asks whether one class is a subclass of another: "Is class
`A` a subclass of `B`?" This check determines whether one concept is more
general than another. For instance, if `Person` subsumes `Student`, then every
`Student` is necessarily a `Person`. Subsumption reasoning is fundamental to
building taxonomies and ontologies, because it establishes the hierarchical
relationships that give a knowledge base its organizational backbone.

#strong[Satisfiability] asks a different question: "Can an instance of a concept
exist?" This test checks whether a concept is logically consistent, meaning its
defining conditions do not contradict one another. Consider the concept
`FlyingPenguin`: it requires the ability to fly, yet penguins are defined as
birds that cannot fly. Because these constraints conflict, `FlyingPenguin` is
unsatisfiable; no individual could ever fulfill all of its requirements
simultaneously. Catching such contradictions early prevents a knowledge base
from containing meaningless or degenerate categories.

#strong[Classification] builds on subsumption by automatically organizing
concepts into a hierarchy. Given a set of concept definitions, a reasoner checks
all pairwise subsumption relationships and arranges concepts accordingly. For
example, given definitions of `Animal`, `Bird`, and `Penguin`, classification
places `Penguin` under `Bird` and `Bird` under `Animal`, producing a clean
taxonomic tree without manual effort.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:828 '* Instance-Level Reasoning Tasks'
// Slide: Instance-Level Reasoning Tasks
#strong[Instance-Level Reasoning Tasks]

#strong[Instance checking] asks whether a specific individual belongs to a given
concept. For example, one might query whether `GP` is an instance of `Student`;
the reasoner examines the assertions and axioms in the knowledge base to confirm
or deny membership.

#strong[Consistency checking] determines whether the entire knowledge base is
free of contradictions. A well-formed ontology should never allow an individual
to satisfy mutually exclusive conditions simultaneously; for instance, no
`Person` should be classified as both `Alive` and `Dead` at the same time. If a
reasoner detects such a contradiction, the knowledge base is
#emph[inconsistent], meaning every possible query becomes trivially true,
rendering the ontology useless for practical inference.

#strong[Realization] identifies the most specific class to which an individual
belongs. Rather than stopping at a general classification, realization narrows
the answer as far down the hierarchy as the axioms permit. For example, given
enough role assertions and class restrictions, a reasoner might discover that
`GP` is not merely a `Human` but more precisely a `Professor`.

#strong[Retrieval] finds all individuals in the knowledge base that satisfy a
given concept description. Where instance checking tests one individual against
one concept, retrieval sweeps across every known individual and returns the
complete set of matches. For example, retrieving all instances classified as
`TeachingAssistant` produces a list of every individual the reasoner can prove
belongs to that class.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:846 '* Advanced Reasoning Tasks'
// Slide: Advanced Reasoning Tasks
#strong[Advanced Reasoning Tasks]

#strong[Query answering] is the task of answering complex questions against the
knowledge base. For instance, one might ask: find every `Person` who studies at
the university yet is not classified as a `Student`. The system must traverse
the graph's edges and apply logical conditions to return a correct result set.

#strong[Abduction] asks a different question: given an observation, what is the
best explanation? Suppose you see a `Person` carrying a backpack and wearing
flip-flops in the snow. Abductive reasoning infers that this person is likely a
`Student`, since that hypothesis best accounts for the observed combination of
traits.

#strong[Deduction] moves in the opposite direction, inferring consequences that
logically follow from known facts and rules. If `John` is a `Student` in
`ComputerScience`, and the rules state that computer science students may attend
`MSML610`, then deduction concludes that John can attend `MSML610`. Unlike
abduction, deduction is truth-preserving: if the premises hold, the conclusion
must hold.

#strong[Belief revision] handles the messy reality that knowledge bases change.
When new, possibly conflicting information arrives, the system must update its
beliefs consistently. For example, suppose the knowledge base initially contains
a rule that every student in `ComputerScience` can take `MSML610`. Later, a
prerequisite policy is added that restricts enrollment. Belief revision retracts
or weakens the earlier rule so the knowledge base remains coherent.

#strong[Temporal reasoning] introduces the dimension of time into inference. If
`EventA` happens before `EventB`, then `EventB` cannot be the cause of `EventA`.
Encoding and respecting temporal order lets the system rule out impossible
causal chains and answer questions about sequences of events.

#strong[Causal reasoning] goes further by inferring cause-and-effect
relationships among entities or events. Combining temporal knowledge (the storm
preceded the flooding) with physical knowledge (storms produce heavy rainfall
that overwhelms drainage), the system can infer the triple (`Storm`, `Cause`,
`Flooding`). Causal reasoning builds on both temporal ordering and
domain-specific rules to move beyond correlation toward genuine explanatory
claims.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:875 '* Protege'
// Slide: Protege
#strong[Protege]

#strong[Protégé] is a free, open-source platform for building ontologies,
developed at Stanford University. It allows users to construct and visualize
ontologies by defining classes, properties, individuals, and the relationships
among them through a graphical interface. Beyond construction, Protégé enables
reasoning over ontologies through a plugin architecture: reasoner plugins can
check an ontology for logical consistency and infer new knowledge that was not
explicitly stated but follows from the axioms already present.

#figure(
  image("../lectures_source/figures/L03.Protege_OWL.jpg", width: 80%),
  caption: [Protege OWL],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:protegeowl>

As @fig:protegeowl illustrates, Protégé provides a visual workspace for
inspecting and editing ontology elements. The platform supports the major
ontology languages, including OWL (Web Ontology Language) and RDF (Resource
Description Framework), along with multiple serialization formats such as
RDF/XML, Turtle, and OWL Functional Syntax. This broad format support makes it
straightforward to exchange ontologies with other tools in the Semantic Web
ecosystem.

Protégé's use cases span a wide range of domains:

- #emph[Domain-specific knowledge modeling]: fields such as biomedicine and law
  rely on Protégé to capture complex terminologies and regulatory hierarchies in
  machine-readable form.
- #emph[Semantic Web applications]: ontologies built in Protégé serve as the
  backbone for linked-data systems that publish and query structured information
  on the web.
- #emph[AI systems requiring structured knowledge]: any AI pipeline that needs a
  formal, queryable representation of domain concepts (for instance, a clinical
  decision-support system that reasons about drug interactions) can import an
  OWL ontology authored in Protégé directly into its knowledge base.

// From: msml610/lectures_source/Lesson03.1-Knowledge_representation.smd:909 '* Summary'
// Slide: Summary
#strong[Summary]

Knowledge representation serves as the bridge between raw perception and
structured reasoning: it takes what is implicitly known and makes it explicit,
organized, and amenable to machine processing. Without this bridge, an agent may
perceive the world richly but lack the internal structure to draw conclusions,
plan actions, or communicate its understanding.

Several core themes run through the study of knowledge representation. First,
every representational choice involves #emph[design tradeoffs]: more expressive
languages can capture subtler distinctions but tend to make inference harder,
while simpler languages keep reasoning tractable at the cost of what they can
say. The tension between symbolic approaches (which manipulate discrete,
human-readable structures) and sub-symbolic ones (which operate over continuous
numerical representations) reflects the same balancing act.

Second, the #emph[languages] available for encoding knowledge span a wide
spectrum. Natural language is maximally expressive but riddled with ambiguity.
Programming languages are precise and executable but typically lack the
declarative semantics needed for general reasoning. Propositional logic offers
clean truth-functional semantics but cannot quantify over objects, while
first-order logic adds variables, quantifiers, and predicates, bringing enough
expressiveness to formalize most of the reasoning patterns that knowledge-based
AI requires.

Third, the #emph[semantics] of a knowledge base give its sentences meaning. A KB
is a set of sentences grounded in the world through an interpretation function;
whether a sentence is true or false is evaluated over models, which are possible
configurations of the domain. This model-theoretic grounding is what separates a
knowledge base from a mere collection of strings.

Fourth, #emph[reasoning] in this framework is governed by entailment: a sentence
follows from a KB exactly when every model that satisfies the KB also satisfies
that sentence. Sound inference never produces a conclusion that fails to follow;
complete inference never misses one that does. Together, soundness and
completeness guarantee that the mechanical process of inference perfectly tracks
the semantic notion of logical consequence.

Finally, the progression from simple reflex rules to full #emph[knowledge-based
  agents] shows how richer internal representations unlock more capable
behavior. Ontologies provide the shared, structured vocabularies that let agents
(and teams of agents) organize their knowledge into coherent categories, reason
over those categories, and communicate unambiguously about the world they
inhabit.
