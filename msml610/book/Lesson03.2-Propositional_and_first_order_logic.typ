// git_hash=9b64c438-xu0 timestamp=20260909_100210
// Import AIMA style formatting and macros.
#import "/helpers_root/dev_scripts_helpers/typst/aima_style.typ": (
  aima-style, algorithm, chapter, glossary, styled-table, wrap-content,
)
// Import the custom citation/bibliography system.
#import "/helpers_root/dev_scripts_helpers/typst/umd_references.typ": (
  cite, references,
)

// Document metadata
#set document(
  title: "L03.2: Propositional and First Order Logic",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L03.2: Propositional and First Order Logic")

= Roadmap

This chapter explores two foundational logical systems for knowledge representation:
#emph[propositional logic] and #emph[first-order logic]. Propositional logic provides
a clean, tractable framework for reasoning about fixed facts, using syntax and
semantics grounded in truth tables and #emph[model checking]. First-order logic
extends this with variables, predicates, and quantifiers, dramatically increasing
expressiveness at the cost of computational complexity. Both systems form the
theoretical backbone for automated reasoning, #emph[constraint satisfaction], and
knowledge-based AI systems. This chapter walks through their formal definitions,
inference procedures, and practical applications.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:13 '# Propositional logic'
// Slide: Propositional logic
= Propositional logic

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:15 '## Syntax'
// Slide: Syntax
== Syntax

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:19 '* Propositional Logic'
// Slide: Propositional Logic
#strong[Propositional logic] is a formal system for reasoning about statements that
can be either true or false. Two components define the system. #emph[Syntax]
specifies the allowable sentences: every well-formed formula is built from
proposition symbols and logical connectives (for instance, $P and Q$).
#emph[Semantics] determines how truth values are assigned to those sentences: truth
tables or deduction rules evaluate complex formulas by combining the truth values of
their parts. If $P$ is true and $Q$ is false, for example, then $P and Q$ evaluates
to false.

Propositional logic underpins several practical technologies. #emph[SAT solvers]
determine whether a given propositional formula can be satisfied by some assignment
of truth values; they are workhorses in hardware verification and scheduling.
#emph[Expert systems] encode domain knowledge as logic rules to mimic human
decision-making, with medical diagnosis systems being a classic example.
#emph[Rule-based agents] operate from a set of predefined rules to select actions, as
in automated customer-service chatbots that follow branching decision trees.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:43 '* Proposition Symbol'
// Slide: Proposition Symbol
These applications all rest on the same basic building block. A #strong[proposition
  symbol] is the most basic unit of propositional logic: an atomic sentence
consisting of a single symbol such as $P$, $Q$, or $N o r t h$. On its own, a
proposition symbol carries no truth value; it is simply a placeholder for a
real-world statement and requires grounding to acquire semantics. What it stands for
is a proposition that can be either true or false. For instance, $K_(E,5)$ might mean
"the Knight is in square E5." Despite the suggestive subscript notation, $K_(E,5)$ is
not composed of smaller symbols; it is a single atomic symbol whose internal
typography is just a naming convention, not logical structure.

The constants $T r u e$ and $F a l s e$ are #strong[built-in symbols]: proposition
symbols that, unlike ordinary ones, come with inherent, fixed truth values rather
than needing an external interpretation to assign one.

Propositional logic is therefore an #emph[atomic representation]: every proposition
is an indivisible unit with no internal structure that the logic can inspect or
decompose. This is both the source of propositional logic's simplicity and the root
of its expressiveness limitations, since there is no way to talk about objects,
properties, or relations within a single proposition.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:57 '* Sentences'
// Slide: Sentences
These atomic symbols combine into two kinds of sentences. An #strong[atomic sentence]
consists of a single proposition symbol, such as $P$, with no internal structure to
decompose. A #strong[complex sentence], by contrast, is built from simpler sentences
using parentheses and logical connectives, for example $(P and Q) or R$. Because the
definition is recursive, any complex sentence can itself become a building block
inside a still larger one, giving propositional logic the ability to express
arbitrarily elaborate statements from a small set of primitives.

The #strong[logical connectives] that glue sentences together are:

- #emph[Not] ($not$): negation, flipping a sentence's truth value.
- #emph[And] ($and$): conjunction, true only when both operands are true. The
  symbol's shape resembles the letter "A," a useful mnemonic for "and."
- #emph[Or] ($or$): disjunction, true when at least one operand is true. The symbol
  derives from the Latin word #emph[vel], meaning "or."
- #emph[Implies] ($arrow.r.double$): material implication, false only when the
  premise is true and the conclusion is false.
- #emph[If and only if] ($arrow.l.r.double$): biconditional, true exactly when both
  sides share the same truth value.

Regardless of whether a sentence is atomic or complex, it evaluates to exactly one of
two values: true or false. There is no middle ground, no "partially true." This
strict #emph[bivalence] is what makes propositional logic tractable: every
well-formed sentence, no matter how deeply nested, ultimately reduces to a single
truth value once the truth values of its atomic components are fixed.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:76 '* Propositional Logic: Weather Example'
// Slide: Propositional Logic: Weather Example
A concrete weather example makes these definitions tangible. #strong[Atomic
  sentences] are the simplest propositions in propositional logic, each representing
a single declarative claim that is either true or false. For instance, $"Rain"$
stands for "it's raining," $"Cold"$ for "it's cold," $"Sunny"$ for "it's sunny,"
$"Snow"$ for "it's snowing," and $"Cloudy"$ for "it's cloudy." An atomic sentence can
appear in positive form ($"Rain"$) or be negated with the $not$ operator:
$not "Rain"$ means "it's not raining."

From these atoms, compound sentences are built using logical connectives. A
#strong[conjunction] joins two sentences with $and$ ("and"), so $"Rain" and "Cloudy"$
reads "it's raining and it's cloudy," and is true only when both parts hold. A
#strong[disjunction] joins them with $or$ ("or"), so $"Rain" or "Snow"$ reads "it's
either raining or snowing," and is true whenever at least one part holds.
#strong[Negation] applies to any sentence, not just atoms: $not ("Rain" or "Cloudy")$
means "it is not the case that it's raining or cloudy," which is true only when
neither raining nor cloudy.

An #strong[implication] (also called an #emph[if-then statement] or a #emph[rule])
has the form "premise $arrow.r.double$ conclusion." For example,
$"Rain" arrow.r.double not "Snow"$ reads "if it's raining, then it's not snowing."
The implication is false only when the premise is true and the conclusion is false;
in every other case it holds. Finally, a #strong[biconditional] captures "if and only
if" by combining two implications: $A arrow.l.r.double B$ is shorthand for
$(A arrow.r.double B) and (B arrow.r.double A)$. The sentence
$"Sunny" arrow.l.r.double not "Cloudy"$ asserts that being sunny and not being cloudy
always go together: each one guarantees the other.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:103 '* Grammar in BNF form'
// Slide: Grammar in BNF form
Beyond such informal examples, the syntax of propositional logic is specified
formally. #strong[Backus Normal Form] (BNF) provides a formal way to represent the
grammar of propositional logic #cite("backus1959syntax"). By specifying exactly which
strings of symbols count as well-formed sentences, BNF removes any guesswork about
whether an expression is syntactically valid.

A grammar alone, however, does not resolve every reading of a sentence. The
expression $not A or B$ could be parsed as $(not A) or B$ or as $not (A or B)$, two
formulas with very different truth conditions. This kind of ambiguity arises whenever
the grammar permits more than one parse tree for the same string.

#wrap-content(
  [
    #figure(
      image(
        "../lectures_source/figures/L03.propositional_logic_BNF.png",
        width: 100%,
      ),
      caption: [BNF grammar for propositional logic sentences, disambiguated by
        operator precedence.],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:propositionallogicbnf>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 40%),
)[
  The standard fix is to layer #emph[operator precedence] on top of the grammar
  rules. Negation ($not$) binds more tightly than conjunction ($and$) and disjunction
  ($or$), so $not A or B$ is read unambiguously as $(not A) or B$. Additional
  conventions (left-to-right associativity, explicit parenthesization for remaining
  edge cases) ensure that every well-formed sentence has exactly one parse.
  @fig:propositionallogicbnf summarizes the BNF production rules that, together with
  these precedence conventions, fully specify the syntax of propositional logic.
]

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:122 '## Semantics'
// Slide: Semantics
== Semantics

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:124 '* Semantics of Propositional Logic'
// Slide: Semantics of Propositional Logic
#strong[Semantics] are the rules for determining the truth of a sentence α with
respect to a model $m$. Where syntax tells you whether a sentence is well-formed,
semantics tells you whether it is _true or false_ given a particular possible world.

More precisely, a #strong[model] in propositional logic is an assignment that fixes
the truth value (true or false) for every atomic sentence in the language. Models are
abstractions: they have no built-in connection to any specific real-world situation.
The symbol $P_(1,2)$ is, on its own, just a symbol. It could mean "there is a pit in
square [1, 2]," or it could equally well mean "I'm in Paris today and tomorrow." The
mapping from symbols to real-world propositions requires #emph[grounding], a separate
step that ties the abstract model to the domain you care about. Until that grounding
is established, the logic manipulates symbols purely by their formal relationships,
indifferent to what they "really" refer to.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:138 '* Computing the Truth Value of a Sentence'
// Slide: Computing the Truth Value of a Sentence
Once a model is in place, grounded or not, computing a sentence's truth value follows
purely mechanical rules. The #strong[truth value of a sentence] is derived
recursively from a model $m$, which assigns a truth value to every proposition
symbol. For instance, the model

$ m = { P_(1,2) = F, P_(2,2) = F, P_(3,1) = T } $

fixes the truth of three atomic propositions, and from those assignments the truth of
any compound sentence follows mechanically. Every sentence α is built from atomic
sentences (whose values come directly from $m$) and five connectives, each with a
precise definition:

- $not P$ is true if and only if $P$ is false in $m$.
- $P and Q$ is true if and only if both $P$ and $Q$ are true in $m$.
- $P or Q$ is true if and only if at least one of $P$ or $Q$ is true in $m$.
- $P arrow.r.double Q$ is true unless $P$ is true and $Q$ is false in $m$.
- $P arrow.l.r.double Q$ is true if and only if $P$ and $Q$ share the same truth
  value in $m$.

These rules let you evaluate any formula bottom-up: start at the atomic symbols, look
up their values in $m$, then apply each connective's rule until you reach the
outermost operator.

A #strong[truth table] collects the truth value of a sentence for every possible
model. To build one, enumerate every combination of truth values for the atomic
symbols and compute the sentence's value in each row. Consider the sentence
$X = A and B or C$. Three proposition symbols yield $2^3 = 8$ rows. Working through
them, $X$ comes out true whenever $C$ is true (regardless of $A$ and $B$), or
whenever both $A$ and $B$ are true (regardless of $C$).
@tab:computingthetruthvalueofasentence shows the complete enumeration, confirming
that exactly five of the eight models satisfy $X$.

#figure(
  styled-table(
    headers: ("A", "B", "C", "X"),
    rows: (
      ("F", "F", "F", "F"),
      ("F", "F", "T", "T"),
      ("F", "T", "F", "F"),
      ("F", "T", "T", "T"),
      ("T", "F", "F", "F"),
      ("T", "F", "T", "T"),
      ("T", "T", "F", "T"),
      ("T", "T", "T", "T"),
    ),
    bold-first-col: false,
  ),
  caption: [Truth table for the formula X = A and B or C, showing all $2^3 = 8$
    models with their truth values.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:computingthetruthvalueofasentence>

Truth tables are conceptually simple but grow exponentially: $n$ symbols produce
$2^n$ rows. For small formulas they remain the most transparent way to verify
equivalences, check validity, or identify satisfying assignments, but for larger
knowledge bases more efficient inference methods become essential.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:184 '* Interpretation of Implication'
// Slide: Interpretation of Implication
One connective's truth table trips up newcomers more than the rest: in a logical
implication $P arrow.r.double Q$, there is #strong[no causation] between $P$ and $Q$.
The statement "$P arrow.r.double Q$" says only: "if $P$ is true, I claim that $Q$ is
true; otherwise I am making no claim at all." This purely truth-functional reading
leads to results that feel strange at first. For instance, "5 is odd arrow.r.double
that Tokyo is the capital of Japan" is a true sentence in propositional logic,
because both the antecedent and the consequent happen to be true, even though oddness
of a number has nothing to do with geography. The key insight is that an implication
is true whenever its antecedent is false, regardless of the consequent. So "5 is even
arrow.r.double pigs fly" is also true: because 5 is not even, the antecedent is
false, and the implication makes no claim at all, which by convention counts as true.
These #emph[vacuously true] implications are a frequent source of confusion, but they
follow directly from the truth table for material implication and carry no causal or
explanatory content whatsoever.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:195 '## Inference and Proof'
// Slide: Inference and Proof
== Inference and Proof

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:197 '* Model Checking is Sound and Complete'
// Slide: Model Checking is Sound and Complete
#wrap-content(
  [
    // rendered_images:begin
    //             ```graphviz
    //             digraph ModelChecking {
    //               graph [rankdir=TB, bgcolor="transparent", nodesep=0.25, ranksep=0.35,
    //                      fontname="Helvetica"];
    //               node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
    //                     fontcolor="#26215C", color="#7F77DD", penwidth=1.2];
    //               edge [color="#888888", penwidth=1.2];
    //
    //               enumerate [label="Enumerate all\nmodels", fillcolor="#A0D6D1"];
    //               filter [label="Keep models\nwhere KB true", fillcolor="#A6E7F4"];
    //               check [label="Check alpha true\nin all of them", fillcolor="#A6C8F4"];
    //
    //               enumerate -> filter -> check;
    //             }
    //             ```
    //             label=fig:modelcheckingissoundandcomplete
    //             caption=The three-stage model-checking pipeline for testing whether KB entails a query.
    // width=100%
    // placement=auto
    // rendered_images:end
    // render_images:begin
    #figure(
      image(
        "Lesson03.2-Propositional_and_first_order_logic.typ.figs/Lesson03.2-Propositional_and_first_order_logic.1.png",
        width: 100%,
      ),
      caption: [The three-stage model-checking pipeline for testing whether KB
        entails a query.],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:modelcheckingissoundandcomplete>
    // render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 40%),
)[
  The #strong[model checking algorithm] works by enumerating every possible model
  (that is, every row of the truth table for the variables involved), filtering to
  keep only those models in which the knowledge base $K B$ is true, and then
  verifying that the query sentence α holds in every one of those surviving models.
  @fig:modelcheckingissoundandcomplete illustrates this three-stage pipeline:
  enumerate all models, retain those satisfying the knowledge base, then confirm the
  query across what remains.
]

This procedure has two important guarantees in propositional logic. It is
#strong[sound], meaning any inference it produces is correct: if the algorithm
reports that $K B$ entails α, that relationship genuinely holds, because the
algorithm directly implements the definition of entailment by checking every model.
It is also #strong[complete], meaning every true entailment will be found: the
algorithm works for any knowledge base and any query sentence, and it always
terminates because the number of propositional models is finite.

These guarantees come at a cost. With $n$ propositional variables, there are $2^n$
possible truth assignments, so the #emph[time complexity] of model checking is
$O(2^n)$, placing it in the NP-complete class. The worst case is genuinely
exponential, though average-case performance on structured problems can be
considerably better. The #emph[space complexity], by contrast, is only $O(n)$,
because the enumeration can proceed depth-first: at any point the algorithm needs to
store only a single assignment of values to the $n$ variables, not the full table of
$2^n$ rows.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:238 '* Inference in Propositional Logic'
// Slide: Inference in Propositional Logic
Model checking is one way to establish entailment; a lighter-weight alternative
reasons symbolically instead. #strong[Inference rules] are the rules of reasoning
that allow an agent to derive correct new statements from the statements already
present in a knowledge base. Each rule captures a pattern of valid deduction: if the
premises match what is known, the conclusion is guaranteed to follow.

The most familiar rule is #emph[Modus Ponens]: if $p arrow.r.double q$ and $p$ both
hold, infer $q$. In everyday language: "If it rains, the ground will be wet. It
rains. Therefore, the ground is wet." Its mirror image is #emph[Modus Tollens]: if
$p arrow.r.double q$ and $not q$, infer $not p$. Continuing the example: "If it
rains, the ground will be wet. The ground is not wet. Therefore, it did not rain."

Beyond these two workhorses, several structural rules round out a basic deductive
toolkit:

- #emph[Syllogism] (transitivity): from $p arrow.r.double q$ and
  $q arrow.r.double r$, infer $p arrow.r.double r$.
- #emph[Disjunctive syllogism]: from $p or q$ and $not p$, infer $q$.
- #emph[Addition]: from $p$, infer $p or q$ for any $q$.
- #emph[Simplification]: from $p and q$, infer $p$ (or $q$).
- #emph[Conjunction]: from $p$ and $q$ separately, infer $p and q$.
- #emph[Resolution rule] #cite("robinson1965resolution"): from $(p or q)$ and
  $(not p or r)$, infer $(q or r)$.

Resolution deserves special attention. Introduced by Robinson in 1965, it is the
single rule on which most #emph[automated theorem provers] are built, because every
other rule in this list can be derived as a special case of resolution when
statements are first converted to clausal form.

#figure(
  styled-table(
    headers: ("Rule", "From", "Infer"),
    rows: (
      ("Modus Ponens", [$p arrow.r.double q$, $p$], [$q$]),
      ("Modus Tollens", [$p arrow.r.double q$, $not q$], [$not p$]),
      (
        "Syllogism",
        [$p arrow.r.double q$, $q arrow.r.double r$],
        [$p arrow.r.double r$],
      ),
      ("Disjunctive Syllogism", [$p or q$, $not p$], [$q$]),
      ("Addition", [$p$], [$p or q$]),
      ("Simplification", [$p and q$], [$p$ (or $q$)]),
      ("Conjunction", [$p$, $q$], [$p and q$]),
      ("Resolution Rule", [$(p or q)$, $(not p or r)$], [$(q or r)$]),
    ),
    bold-first-col: true,
  ),
  caption: [Summary of standard propositional inference rules.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:inferencerules>

@tab:inferencerules collects all eight rules in one place for quick reference,
showing the premises each rule requires and the conclusion it licenses.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:271 '* Propositional Theorem Proving'
// Slide: Propositional Theorem Proving
Chaining these inference rules together is exactly what #strong[propositional theorem
  proving] does: the central task is to prove a sentence α from a knowledge base KB
by applying rules of inference to construct a formal proof. Given a knowledge base
and a query α, exactly one of three entailment statuses holds: the query is
#emph[entailed] ($"KB" models alpha$), meaning it follows from what is known; it is
#emph[refuted] ($"KB" models not alpha$), meaning its negation follows; or it is
#emph[unknown], meaning KB entails neither α nor ¬α.

Two broad strategies exist for establishing entailment. #emph[Model checking]
enumerates all possible truth assignments and verifies that α is true in every model
where KB is true. #emph[Propositional theorem proving], by contrast, builds a
stepwise proof from KB to α using inference rules, without exhaustively listing
models. When a short proof exists, theorem proving can be dramatically more efficient
than model checking, since it avoids the exponential enumeration of all assignments.
When proofs are long or hard to find, however, the two approaches may converge in
cost.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:288 '* Logical Equivalence of Sentences'
// Slide: Logical Equivalence of Sentences
Both proof strategies rely on a shared notion of sameness between sentences. Two
sentences α and β are #strong[logically equivalent], written $alpha equiv beta$, when
they are true in exactly the same set of models:

$ M(alpha) = M(beta) $

An equivalent way to state this: α and β are logically equivalent if and only if each
entails the other:

$ alpha models beta and beta models alpha $

For instance, $P or Q equiv Q or P$: commutativity of disjunction holds because
swapping the disjuncts does not change which models satisfy the sentence.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:302 '* Logical Equivalences (1/2)'
// Slide: Logical Equivalences (1/2)
The fundamental equivalences of propositional logic provide a toolkit for rewriting
sentences into forms that are easier to work with, without changing their truth
value. #strong[Commutativity] of conjunction and disjunction tells us that the order
of operands does not matter:

$ (alpha and beta) equiv (beta and alpha) $
$ (alpha or beta) equiv (beta or alpha) $

#strong[Associativity] extends this idea to chains of the same connective, letting us
drop parentheses entirely when three or more operands are joined by the same
operator:

$
  (alpha and beta) and gamma equiv alpha and (beta and gamma) equiv alpha and beta and gamma
$
$
  (alpha or beta) or gamma equiv alpha or (beta or gamma) equiv alpha or beta or gamma
$

The two #strong[distributivity] laws describe how conjunction and disjunction
interact with each other. Conjunction distributes over disjunction in a way that
mirrors multiplication over addition in arithmetic:

$ alpha and (beta or gamma) equiv (alpha and beta) or (alpha and gamma) $

The dual form, where disjunction distributes over conjunction, has no familiar
arithmetic analogue but is equally valid:

$ alpha or (beta and gamma) equiv (alpha or beta) and (alpha or gamma) $

These two distributivity laws are essential for converting between conjunctive normal
form (CNF) and disjunctive normal form (DNF), a step that appears repeatedly in
automated reasoning and SAT solving.

Finally, #strong[double negation] elimination states that negating a sentence twice
returns it to its original truth value:

$ not (not alpha) equiv alpha $

This equivalence is straightforward yet indispensable: many proof procedures and
simplification routines introduce double negations as intermediate steps, and this
rule lets them be cleaned up immediately.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:336 '* Logical Equivalences (2/2)'
// Slide: Logical Equivalences (2/2)
Four more equivalences round out this toolkit. #strong[Contraposition] states that an
implication and its contrapositive are logically equivalent: if α arrow.r.double β,
then not-β arrow.r.double not-α, and vice versa.

$ (alpha arrow.r.double beta) equiv (not beta arrow.r.double not alpha) $

This equivalence is the foundation of proof by contrapositive, a technique where
instead of showing "if α then β" directly, you show "if not β then not α," which can
sometimes be more straightforward.

#strong[Implication elimination] rewrites a conditional as a disjunction: α
arrow.r.double β is equivalent to saying either α is false or β is true.

$ (alpha arrow.r.double beta) equiv (not alpha or beta) $

This equivalence is particularly useful when converting sentences to conjunctive
normal form (CNF), since it removes the implication connective entirely in favor of
negation and disjunction.

#strong[Biconditional elimination] breaks a biconditional into two separate
implications: α if and only if β means that α arrow.r.double β and β arrow.r.double
α.

$
  (alpha arrow.l.r.double beta) equiv (alpha arrow.r.double beta) and (beta arrow.r.double alpha)
$

Each of those resulting implications can then be further rewritten using implication
elimination, making biconditional elimination a key step in any systematic
normalization procedure.

The #strong[De Morgan laws] describe how negation distributes over conjunction and
disjunction. Negating a conjunction yields a disjunction of the individual negations,
and negating a disjunction yields a conjunction of the individual negations:

$ not (alpha and beta) equiv (not alpha or not beta) $

$ not (alpha or beta) equiv (not alpha and not beta) $

Together, these five equivalences form a small but complete toolkit for rewriting any
propositional formula into an equivalent form that uses only negation, conjunction,
and disjunction. In practice, applying implication elimination first, then
biconditional elimination, then De Morgan's laws, and finally double-negation
elimination, is exactly the sequence used to convert an arbitrary sentence into CNF
for use in resolution-based theorem provers.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:359 '* Deduction Theorem'
// Slide: Deduction Theorem
Beyond rewriting individual sentences, some sentences carry a special status
regardless of model. A #strong[valid sentence] (also called a #strong[tautology]) is
a sentence α that is true in every model. The classic example is $P or not P$: no
matter what truth value P takes, the disjunction holds. Because a tautology cannot
fail to be true, every tautology is logically equivalent to the sentence $"True"$.

The mirror image of validity is #strong[contradiction]: a sentence α that is false in
every model. The paradigmatic case is $P and not P$, which no assignment of truth
values can satisfy. Every contradiction is equivalent to the sentence $"False"$.

These two extremes, always true and always false, anchor a powerful result known as
the #strong[deduction theorem]: the sentence α entails β (written
$alpha models beta$) if and only if the sentence $alpha arrow.r.double beta$ is a
tautology. In other words, to check whether one sentence semantically follows from
another, it suffices to check whether their material implication is valid.

#wrap-content(
  [
    // rendered_images:begin
    //         ```graphviz
    //         digraph DeductionBridge {
    //           graph [rankdir=TB, bgcolor="transparent", nodesep=0.3, ranksep=0.4,
    //                  fontname="Helvetica"];
    //           node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
    //                 fontcolor="#26215C", color="#7F77DD", penwidth=1.2];
    //           edge [color="#888888", penwidth=1.4, fontname="Helvetica", fontsize=10,
    //                 fontcolor="#45296B"];
    //
    //           entailment [label="Entailment\n(semantic)", fillcolor="#A6C8F4"];
    //           implication [label="Implication\n(syntactic)", fillcolor="#A0D6D1"];
    //
    //           entailment -> implication [dir=both, label="Deduction\nTheorem"];
    //         }
    //         ```
    //         label=fig:deductiontheorem
    //         caption=The deduction theorem bridging semantic entailment and syntactic implication.
    // width=100%
    // placement=auto
    // rendered_images:end
    // render_images:begin
    #figure(
      image(
        "Lesson03.2-Propositional_and_first_order_logic.typ.figs/Lesson03.2-Propositional_and_first_order_logic.2.png",
        width: 100%,
      ),
      caption: [The deduction theorem bridging semantic entailment and syntactic
        implication.],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:deductiontheorem>
    // render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 40%),
)[
  Why is this result so useful? It acts as a bridge between two ideas that look
  similar but live at different levels of the logic. #emph[Entailment] ($models$) is
  a semantic notion: it talks about truth across all models, asserting that in every
  possible world where α is true, β is also true. #emph[Implication]
  ($arrow.r.double$) is a syntactic notion: $alpha arrow.r.double beta$ is just
  another formula inside the logic, built from the same connectives as any other
  sentence. The deduction theorem tells us these two perspectives coincide: the
  semantic relationship "α makes β unavoidable" holds exactly when the syntactic
  object $alpha arrow.r.double beta$ is a tautology. @fig:deductiontheorem
  illustrates how the deduction theorem connects the semantic concept of entailment
  with the syntactic concept of implication, unifying these two perspectives into a
  single equivalence.
]

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:405 '* Satisfiability'
// Slide: Satisfiability
The deduction theorem reduces entailment to a syntactic check on a single formula; a
related question asks whether a sentence can be true at all. A sentence α is
#strong[satisfiable] if and only if it is true in at least one model. Checking
whether a given sentence has such a model is the #strong[SAT problem]: one could, in
principle, enumerate every possible model until finding one that makes α true. This
brute-force approach highlights why the problem is hard: Cook showed in 1971 that
propositional satisfiability is NP-complete #cite("cook1971complexity"), meaning no
known algorithm solves every instance in polynomial time.

#wrap-content(
  [
    // rendered_images:begin
    //         ```graphviz
    //         digraph Satisfiability {
    //           graph [bgcolor="transparent", fontname="Helvetica"];
    //           node [fontname="Helvetica", fontsize=10, fontcolor="#26215C"];
    //
    //           subgraph cluster_all {
    //             label="All sentences";
    //             style="rounded,filled"; color="#F4A6A6"; fillcolor="#FBEAEA";
    //             labelloc=b; fontsize=11; fontcolor="#7A2E2E";
    //
    //             subgraph cluster_sat {
    //               label="Satisfiable";
    //               style="rounded,filled"; color="#A0D6D1"; fillcolor="#E7F6F5";
    //               labelloc=b; fontsize=11; fontcolor="#1F5A55";
    //
    //               valid [label="Valid\n(tautologies)", shape=box,
    //                      style="rounded,filled", fillcolor="#A6C8F4", penwidth=0];
    //             }
    //           }
    //         }
    //         ```
    //         label=fig:satisfiability
    //         caption=Nested regions of all sentences, satisfiable sentences, and valid tautologies.
    // width=100%
    // placement=auto
    // rendered_images:end
    // render_images:begin
    #figure(
      image(
        "Lesson03.2-Propositional_and_first_order_logic.typ.figs/Lesson03.2-Propositional_and_first_order_logic.3.png",
        width: 100%,
      ),
      caption: [Nested regions of all sentences, satisfiable sentences, and valid
        tautologies.],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:satisfiability>
    // render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 40%),
)[
  On the other end of the spectrum, a sentence α is #emph[unsatisfiable] if no model
  makes it true; it is a contradiction. Satisfiability and validity turn out to be
  two sides of the same coin: α is valid (a tautology) if and only if $not alpha$ is
  unsatisfiable. By contrapositive, α is satisfiable if and only if $not alpha$ is
  not valid. These equivalences are practically useful because they let us reduce a
  validity question to a satisfiability question and vice versa, reusing whichever
  solver we already have. @fig:satisfiability illustrates the relationship: the set
  of valid sentences (tautologies) sits inside the larger set of satisfiable
  sentences, which in turn sits inside the set of all sentences, with unsatisfiable
  sentences lying entirely outside the satisfiable region.
]

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:451 '* Proof by Contradiction'
// Slide: Proof by Contradiction
These satisfiability and validity equivalences underlie a proof technique of their
own. A #strong[proof by contradiction] (also called #emph[refutation]) rests on a
clean equivalence: the entailment $alpha models beta$ holds if and only if the
conjunction $alpha and not beta$ is unsatisfiable. In other words, there is no
possible world in which the premises are true and the conclusion is false; any
attempt to construct one collapses into contradiction.

The method translates that equivalence into a step-by-step argument:

#algorithm("Proof by Contradiction", (
  [Assume the premises α.],
  [Assume that the target conclusion β is false.],
  [Derive a contradiction from these two assumptions taken together.],
  [Conclude that β must be true whenever α is.],
))

The power of this technique is that searching for a contradiction is often easier
than constructing a direct derivation, because negating the conclusion gives the
reasoner an extra assumption to work with. Resolution-based theorem provers exploit
exactly this idea: they negate the query, convert everything to clausal form, and
then repeatedly resolve pairs of clauses until the empty clause (a direct
representation of contradiction) appears.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:464 '* Propositional Logic: Pros and Cons'
// Slide: Propositional Logic: Pros and Cons
Having surveyed its syntax, semantics, and proof machinery, it is worth weighing
propositional logic's balance sheet. Propositional logic offers several genuine
strengths. It is #emph[declarative]: its semantics rests on a well-defined relation
between sentences and possible worlds, so the meaning of a statement does not depend
on how it will be executed. It handles #emph[partial information] gracefully; for
instance, "a white knight is in b1 or in f6" is captured directly as
$W K 1_(b 1) or W K 2_(f 6)$, with no need to commit to one square or the other. Its
semantics is #emph[compositional]: the meaning of a complex sentence is built up
systematically from the meanings of its parts. And because every well-formed formula
is #emph[context-independent] and #emph[unambiguous], two reasoners working from the
same knowledge base will always draw the same conclusions.

These advantages come with real costs, however. Propositional logic has no concise
way to describe environments that contain many objects. Saying "the pawn is in a cell
around b1" requires enumerating every qualifying square as a separate disjunction,
and the formula grows quickly as the board (or world) gets larger. More
fundamentally, propositional logic cannot represent #emph[uncertainty]. A claim such
as "there is a 50% probability that the pawn is in b1" simply has no counterpart in
the language: every proposition is either true or false in a given world, with no
room for degrees of belief. These limitations motivate the move to richer formalisms,
including first-order logic for compactly quantifying over objects and probabilistic
logics for reasoning under uncertainty.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:491 '# First-order Logic'
// Slide: First-order Logic
= First-order Logic

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:495 '## Syntax'
// Slide: Syntax
== Syntax

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:497 '* First-Order Logic (FOL)'
// Slide: First-Order Logic (FOL)
#strong[First-order logic] (FOL) extends propositional logic by introducing two key
capabilities: #emph[quantifiers] such as $forall$ (for all) and $exists$ (there
exists), and #emph[predicates] that represent properties of objects and relations
among them. Where propositional logic can only assert that whole statements are true
or false, FOL lets us talk directly about the objects in a domain and make claims
that apply to some or all of them.

This combination gives FOL the expressiveness of natural language while retaining the
formal precision of propositional logic. Because FOL is built around objects and
relations, it can capture statements like "some humans have green eyes"
($exists x: "Human"(x) and "GreenEyes"(x)$) or "chess pieces around the Queen are at
risk" without resorting to a separate propositional variable for every possible
instance. FOL thus provides the expressive power needed to represent structured,
relational knowledge in a way that supports rigorous inference.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:509 '* First-Order Logic: Syntax'
// Slide: First-Order Logic: Syntax
That structured, relational knowledge is assembled from a small set of syntactic
ingredients. #strong[Constants] name specific objects in the domain, such as
$"Socrates"$. #strong[Predicates] describe properties of objects or relations among
them: $"Human"(x)$ asserts that $x$ is human. #strong[Functions] map tuples of
objects to other objects; for instance, $"Mother"(x)$ returns the mother of $x$.
#strong[Variables] such as $x$ and $y$ serve as placeholders that can refer to any
object in the domain.

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L03.FOL_Syntax.png", width: 100%),
      caption: [Syntax tree for a first-order logic sentence, from terms and
        predicates up through quantifiers.],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:folsyntax>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 40%),
)[
  The real expressive power of first-order logic comes from its #strong[quantifiers].
  The universal quantifier $forall x$ lets us state that a property holds for every
  object in the domain, while the existential quantifier $exists x$ asserts that at
  least one object satisfies the property. Together, these elements let us write
  sentences that generalize far beyond what propositional logic can express.
  @fig:folsyntax summarizes the full syntactic structure showing how these components
  combine to form well-formed formulas.
]

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:532 '* Sentences'
// Slide: Sentences
These components combine into sentences built from more basic pieces. A #strong[term]
is a logical expression that refers to an object. For instance, $R i c h a r d$ is a
term that names a specific individual in the domain.

An #strong[atomic sentence] consists of a predicate symbol followed by a
parenthesized list of terms, written as
$P r e d i c a t e(T e r m 1, T e r m 2, ...)$. Predicates correspond to relations
over objects in the domain. For example, $B r o t h e r(R i c h a r d, J o h n)$
asserts that Richard is the brother of John under a given interpretation, while
$M a r r i e d(F a t h e r(R i c h a r d), M o t h e r(J o h n))$ asserts that the
father of Richard and the mother of John are married. Notice that terms can be
nested: $F a t h e r(R i c h a r d)$ is itself a term (built from a function symbol)
that refers to an object, and it appears as an argument inside the outer predicate.

A #strong[complex sentence] combines atomic sentences using the same logical
connectives available in propositional logic (conjunction, disjunction, negation,
implication, biconditional). The syntax and semantics carry over directly: if
$B r o t h e r(R i c h a r d, J o h n)$ and
$M a r r i e d(F a t h e r(R i c h a r d), M o t h e r(J o h n))$ are each atomic
sentences, then their conjunction, disjunction, or any other propositional
combination is a well-formed complex sentence whose truth value is determined by the
truth values of its parts in the usual way.

A #strong[variable] is a term that stands for a possible, but unspecified, object.
Variables are typically written as lowercase letters such as $x$, $y$, or $z$, and
they can appear anywhere a constant or function term would. For example,
$L e f t L e g(x)$ uses the variable $x$ as the argument of a function symbol,
denoting the left leg of whatever object $x$ happens to refer to. Variables become
essential once quantifiers are introduced, because they let a single sentence range
over many objects at once.

The #strong[equality symbol] expresses that two terms refer to the same object.
Writing $F a t h e r(J o h n) = H e n r y$ asserts that the object denoted by the
function term $F a t h e r(J o h n)$ is identical to the object named $H e n r y$.
Equality is built into the logic itself rather than being just another predicate,
which means standard inference rules can exploit it directly (for instance,
substituting equals for equals within any sentence).

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:554 '* Quantifiers'
// Slide: Quantifiers
Beyond individual terms and sentences, #strong[quantifiers] express properties of
entire collections of objects at once, rather than enumerating each object by name as
propositional logic requires. First-order logic provides two quantifiers that between
them cover the two fundamental claims one can make about a collection.

The #strong[universal quantifier], written $forall x space P(x)$, asserts that every
object in the domain satisfies the predicate $P$. The statement is true precisely
when $P(x)$ holds for all possible values of $x$. For instance, "all cats are
mammals" applies to every cat without needing to list them individually.

The #strong[existential quantifier], written $exists x space P(x)$, asserts that some
object in the domain satisfies $P$, without specifying which one. The statement is
true whenever $P(x)$ holds for at least one value of $x$. Saying "there exists a
prime number greater than 100" makes a claim about the collection of integers without
pointing to a specific witness.

A variable that falls within the scope of a quantifier is said to be #strong[bound];
one that does not is #strong[free] (unbound). In the sentence
$forall x (C a t(x) arrow.r.double M a m m a l(x))$, the variable $x$ is bound by
$forall$, so the formula makes a complete, evaluable claim. Free variables, by
contrast, leave a formula open: its truth value depends on what object is assigned to
the unbound variable. Recognizing whether a variable is bound or free is essential
for correct inference, because only sentences with no free variables (called
#emph[closed sentences]) have a definite truth value in a given interpretation.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:571 '* Nested Quantifiers'
// Slide: Nested Quantifiers
Single quantifiers are only the beginning: complex sentences can also be expressed
using #strong[multiple quantifiers]. When more than one quantifier appears in a
formula, their order matters: swapping two quantifiers can change the meaning of the
sentence entirely. Parentheses and careful scoping help clarify which variable each
quantifier binds.

Consider the claim "brothers are siblings." This is a universally quantified
conditional over two variables:

$ forall x, y space "Brother"(x, y) arrow.r.double "Sibling"(x, y) $

The sibling relationship itself is symmetric, meaning it holds in both directions:

$ forall x, y space "Sibling"(x, y) arrow.l.r.double "Sibling"(y, x) $

The difference quantifier order makes becomes vivid with two related English
sentences. "Everybody loves somebody" says that for each person, there exists at
least one person they love:

$ forall x space exists y space "Loves"(x, y) $

Here $forall$ scopes over $exists$: the loved person $y$ can differ depending on who
$x$ is. Contrast this with "there is someone loved by everyone," which places the
existential quantifier first:

$ exists y space forall x space "Loves"(x, y) $

Now a single individual $y$ must be loved by every $x$. The first sentence is almost
trivially true in most domains; the second makes a far stronger claim. This pair is
the classic illustration of why quantifier order is not interchangeable:
$forall x space exists y$ and $exists y space forall x$ are genuinely different
logical statements, and confusing them is one of the most common errors in
translating natural language into first-order logic.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:586 '* Connection between $\forall$ and $\exists$'
// Slide: Connection between $\forall$ and $\exists$
Order is not the only subtlety these two quantifiers hide: they are also
#strong[connected] through negation, following De Morgan's rules for quantifiers.
These equivalences let you freely convert between "for all" and "there exists"
statements by pushing negation inward or outward, much as De Morgan's laws for
propositional logic swap conjunctions and disjunctions under negation.

$ forall x space not P(x) arrow.l.r.double not exists x: P(x) $

$ not (forall x space P(x)) arrow.l.r.double exists x space not P(x) $

$ forall x space P(x) arrow.l.r.double not exists x space not P(x) $

$ exists x space P(x) arrow.l.r.double not (forall x space not P(x)) $

The first equivalence states that asserting $P$ is false of every element is the same
as denying that any element satisfies $P$. The second, perhaps the most frequently
used in practice, says that refuting a universal claim is equivalent to producing a
single counterexample. The third and fourth complete the picture by expressing each
quantifier purely in terms of the other plus negation. Together, these four
identities mean that first-order logic could, in principle, get by with only one
quantifier; the second is definable from the first. In practice, retaining both keeps
formulas readable and closer to natural-language intuition.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:603 '## Semantics'
// Slide: Semantics
== Semantics

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:605 '* First-order Logic: Semantics'
// Slide: First-order Logic: Semantics
#strong[Semantics] in first-order logic define how sentences are interpreted within a
particular domain. Where syntax gives us the grammar for writing well-formed
formulas, semantics tell us what those formulas actually _mean_ by connecting symbols
to objects and relations in the world.

The vocabulary of a first-order language is built from three kinds of symbols:

- #emph[Constant symbols] represent specific, named objects in the domain: $"Alice"$,
  $G P$, $C S 101$.
- #emph[Predicate symbols] represent relationships among objects. A predicate takes
  one or more arguments and evaluates to true or false:
  $"EnrolledIn"("Student", "Class")$, $"Teaches"("Professor", "Class")$,
  $"IsStudent"(x)$, $"IsProfessor"(x)$.
- #emph[Function symbols] represent mappings from objects to objects. Unlike
  predicates, a function returns an object rather than a truth value:
  $"AdvisorOf"("Student")$, $"DepartmentOf"("Professor")$.

These symbols are purely syntactic until we say what they refer to. An
#strong[interpretation] (also called a #emph[grounding]) is the bridge between the
formal language and the world: it maps each constant symbol to a specific object,
each predicate symbol to a specific relation over objects, and each function symbol
to a specific mapping. Going the other direction, it also tells us which symbol
corresponds to a given real-world entity. Many different interpretations are possible
for the same set of symbols. The constant $G P$ could refer to any object in the
domain; the predicate $"Human"(x)$ could pick out any subset of objects. Among all
these possibilities, the #emph[intended interpretation] is the one that matches the
natural, common-sense reading we have in mind when we write the formulas. For
instance, mapping $G P$ to the course instructor, or reading
$forall x ("Human"(x) arrow.r.double "Mortal"(x))$ as the claim that every human is
mortal, reflects the intended interpretation rather than some arbitrary alternative
assignment.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:628 '* Representing Knowledge in FOL'
// Slide: Representing Knowledge in FOL
Interpreted this way, first-order logic offers several representational strengths. It
can express general rules such as $forall x ("Bird"(x) arrow.r.double "CanFly"(x))$,
stating that every bird can fly, alongside specific facts like $"Bird"("Tweety")$
that ground those rules in particular individuals.

Beyond simple properties, FOL captures complex #strong[relations] through
multi-argument predicates: $"Loves"("Romeo", "Juliet")$ and $"GreaterThan"(3, 2)$
each link two objects in a named relationship that propositional logic could not
express without enumerating every possible pair. #strong[Functions] add another layer
by constructing new objects from existing ones: $"FatherOf"("John")$ refers to a
specific individual (John's father) without requiring a separate constant for that
person.

Together, these building blocks assemble into a #strong[knowledge base] of axioms and
facts from which an inference engine can derive new conclusions. The real payoff is
#emph[reasoning]: given the rule that all birds fly and the fact that Tweety is a
bird, the system automatically concludes $"CanFly"("Tweety")$ without that conclusion
ever being stated explicitly.

#grid(
  columns: (1fr, 50%),
  column-gutter: 1em,
)[
  Interpreted this way, first-order logic offers several representational strengths.
  It can express general rules such as
  $forall x ("Bird"(x) arrow.r.double "CanFly"(x))$, stating that every bird can fly,
  alongside specific facts like $"Bird"("Tweety")$ that ground those rules in
  particular individuals.

  Beyond simple properties, FOL captures complex #strong[relations] through
  multi-argument predicates: $"Loves"("Romeo", "Juliet")$ and $"GreaterThan"(3, 2)$
  each link two objects in a named relationship that propositional logic could not
  express without enumerating every possible pair. #strong[Functions] add another
  layer by constructing new objects from existing ones: $"FatherOf"("John")$ refers
  to a specific individual (John's father) without requiring a separate constant for
  that person.

  Together, these building blocks assemble into a #strong[knowledge base] of axioms
  and facts from which an inference engine can derive new conclusions. The real
  payoff is #emph[reasoning]: given the rule that all birds fly and the fact that
  Tweety is a bird, the system automatically concludes $"CanFly"("Tweety")$ without
  that conclusion ever being stated explicitly. @tab:folcategories summarizes these
  four categories, illustrating how a single formalism covers universal
  generalizations, ground facts, relational statements, and functional terms.
][
  #figure(
    styled-table(
      headers: ("Category", "Example"),
      rows: (
        ("General rules", [$forall x ("Bird"(x) arrow.r.double "CanFly"(x))$]),
        ("Specific facts", [$"Bird"("Tweety")$]),
        ("Relations", [$"Loves"("Romeo", "Juliet")$, $"GreaterThan"(3, 2)$]),
        ("Functions", [$"FatherOf"("John")$]),
      ),
    ),
    caption: [Representative FOL expressions by category.],
    kind: "table",
    supplement: [Table.],
    placement: auto,
  ) <tab:folcategories>
]

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:653 '* First-Order Logic: Inference'
// Slide: First-Order Logic: Inference
A knowledge base full of axioms and facts is only useful if we can reason over it.
The central goal of inference in first-order logic is to derive new sentences from
existing ones using sound rules, so that every conclusion is guaranteed to be true
whenever the premises are.

#strong[Universal instantiation] lets you move from a universally quantified
statement to a specific instance: from $forall x thin P(x)$ you may infer $P(c)$ for
any constant $c$ already in the domain. If every bird has wings, and Tweety is a
bird, you can conclude Tweety has wings. #strong[Existential instantiation] works in
the opposite direction of quantification: from $exists x thin P(x)$ you may infer
$P(c)$, but only by introducing a fresh constant $c$ that has not appeared elsewhere
in the proof. This fresh-name requirement (sometimes called a Skolem constant)
prevents you from accidentally identifying the unknown witness with some object that
already carries other commitments.

Beyond these two quantifier-specific rules, the standard propositional inference
rules carry over directly into first-order logic. #emph[Modus ponens], for instance,
still lets you infer $Q$ whenever you have both $P$ and $P arrow.r Q$; #emph[modus
  tollens], #emph[resolution], and the other familiar rules apply in the same way
once quantifiers have been instantiated away. @tab:folinferencerules summarizes the
core rules side by side.

#figure(
  styled-table(
    headers: ("Rule", "From", "Infer"),
    rows: (
      (
        "Universal Instantiation",
        [$forall x thin P(x)$],
        [$P(c)$, any constant $c$],
      ),
      (
        "Existential Instantiation",
        [$exists x thin P(x)$],
        [$P(c)$, a new constant $c$],
      ),
      ("Modus Ponens", [$P$, $P arrow.r Q$], [$Q$]),
    ),
  ),
  caption: [Core inference rules for first-order logic.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:folinferencerules>

One important theoretical result frames everything that follows: FOL inference is
#strong[semi-decidable] #cite("turing1936computable"). If a sentence is entailed by
the knowledge base, a sound proof procedure will eventually find a proof. If the
sentence is not entailed, however, the search may run forever without terminating.
This asymmetry, established by Turing in 1936, means there is no general algorithm
that can always tell you "no, this does not follow" in finite time. Practical FOL
provers therefore combine completeness guarantees with heuristic search strategies to
keep proof search tractable for the cases that arise in practice.

= Summary

Propositional logic and first-order logic form the theoretical backbone of automated
reasoning. Propositional logic provides a clean, decidable framework for reasoning
about fixed facts through truth-functional semantics and complete inference
procedures like model checking and resolution. Its strength is #emph[tractability];
its weakness is #emph[expressiveness]: it cannot quantify over objects or express
relations compactly.

First-order logic overcomes these limitations by introducing variables, predicates,
and quantifiers, allowing statements that range over collections of objects and
express complex relational structures. This dramatic increase in expressiveness comes
at a cost: reasoning becomes #emph[semi-decidable] rather than decidable, and proof
search strategies become essential to keep inference tractable. Despite these
challenges, FOL serves as the foundation for logic programming languages, automated
theorem provers, and knowledge representation systems across AI.

Both logics rest on the same core principles: clear syntax-semantics separation, the
notion of entailment as truth preservation across models, and the ideal of
#emph[sound and complete inference]. These principles, developed rigorously over
decades, continue to guide the design of modern AI reasoning systems, even as richer
and more practical formalisms build upon them.

// From: msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd:679 '* References'
// Slide: References
= References

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
