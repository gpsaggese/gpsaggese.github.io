// git_hash=aa74b8a4-wl8 timestamp=20260918_152157
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
  title: "L06.1: Bayesian Networks",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L06.1: Bayesian Networks")

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:7 '* Roadmap'
// Slide: Roadmap
= Roadmap

This lesson begins with the challenge of building logic-based AI systems that
must operate under uncertainty: when the world is only partially observable and
outcomes are non-deterministic, rules with exceptions cannot adequately describe
what an agent should believe or do. Probability and utility theory provide the
formal tools needed to fill that gap.

From there, the discussion moves to probabilistic reasoning. The full joint
distribution over a set of random variables encodes everything an agent could
want to know, but it is exponentially large. Independence and conditional
independence are the structural properties that make this representation compact
enough to be practical.

The lesson concludes with Bayesian networks, which combine directed acyclic
graphs with conditional probability tables to represent these independence
relationships explicitly. Two running examples, the Garden World and the classic
burglar alarm network, show how these graphs are constructed and how they
support inference.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:19 '# Logic-Based AI Under Uncertainty'
// Slide: Logic-Based AI Under Uncertainty
= Logic-Based AI Under Uncertainty

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:21 '## Limits of Logic-Based Systems'
// Slide: Limits of Logic-Based Systems
== Limits of Logic-Based Systems

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:23 '* Logic Rules Under Uncertainty'
// Slide: Logic Rules Under Uncertainty
#strong[Logic Rules Under Uncertainty]

#strong[Logic-based AI systems] rest on propositional logic and represent an
agent's repertoire of actions as rules of the form "if preconditions $P$ hold,
then action $A$ causes effect $E$." For instance, a rule might state that
turning the car key starts the engine. In a tidy, fully specified world that
rule works perfectly, but real situations are messier: the battery might be
dead, the fuel tank empty, or the starter motor broken. Any of those hidden
conditions can silently violate the rule's preconditions and make the predicted
effect fail to materialize.

The deeper issue is that real-world agents face #strong[uncertainty] from three
interacting sources:

1. #emph[Partial observability]: the agent cannot see the full state of the
  world, so some preconditions are simply unknown at decision time.
2. #emph[Non-determinism]: even when the agent does act, the outcome is not
  always deterministic or predictable; the same action in apparently the same
  state can produce different results.
3. #emph[Adversarial conditions]: other agents, whether human opponents,
  competing software systems, or the environment itself, may interfere with the
  agent's plans in ways a fixed rule base cannot anticipate.

Together, these factors mean that a purely logical rule set, no matter how
carefully engineered, will eventually encounter a situation it cannot handle.
The car-key example is a miniature version of the general problem: a single
deterministic rule captures the happy path but says nothing about the many ways
the world can deviate from it. Handling that deviation is what motivates the
move from classical logic-based reasoning toward probabilistic and
decision-theoretic frameworks.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:42 '* Belief States and Contingent Plans'
// Slide: Belief States and Contingent Plans
#strong[Belief States and Contingent Plans]

One natural response is to adapt classical logic so it can cope with incomplete
information. Three strategies suggest themselves, though each falls short on its
own.

- #emph[Belief states]: maintain the set of all possible current world states
  and reason over that set rather than over a single known state. The difficulty
  is that enumerating and tracking every member of the belief set quickly
  becomes intractable as the world grows.

- #emph[Causal and exhaustive augmentation]: strengthen the logical rules so
  that every precondition for acting is spelled out explicitly, including remote
  or unlikely ones. The cost is that the rule base balloons: the agent must
  anticipate conditions it will almost certainly never encounter, yet cannot
  safely ignore any of them within a purely logical framework.

- #emph[Contingent plans]: construct a plan that branches on every possible
  sensor report and belief configuration, covering every eventuality the agent
  might face. Plans built this way grow large and complex rapidly. Worse, there
  may be no plan that guarantees success under every contingency, yet the agent
  still needs to act; a framework that demands certainty before committing to
  action can leave the agent paralyzed.

Each of these ideas captures a real insight (track what you don't know, be
explicit about preconditions, plan for contingencies), but none scales
gracefully. The underlying problem is that classical logic treats every
proposition as fully true or fully false, offering no principled way to say
"this is probably true enough to act on." That limitation motivates the move
toward probabilistic and decision-theoretic methods, where degrees of belief
replace the all-or-nothing commitment that makes these naive patches so
unwieldy.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:56 '* Starting a Car'
// Slide: Starting a Car
#strong[Starting a Car]

The goal seems simple: build a system that starts a car when the driver
turns the key. But even this everyday action rests on a surprisingly deep stack
of preconditions.

Some preconditions are obvious. The car needs fuel, the battery must be charged,
and the key must be the correct one for the ignition. These are the conditions
any driver would check first when the engine refuses to turn over.

Other preconditions are far less visible. The electrical system must be intact
so the starter motor receives current. The fuel pump must be operational to
deliver fuel from the tank. The fuel lines cannot be clogged or cracked. The
engine oil level must be sufficient to avoid immediate damage once the engine
fires. Each of these sits behind the scenes, quietly enabling the outcome we
take for granted.

// rendered_images:begin
// ```graphviz[width=75%]
// digraph BayesianFlow {
//     rankdir=LR;
//     compound=true;
//     splines=true;
//     nodesep=0.1;
//     ranksep=0.5;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//     penwidth=1.4, fillcolor="#FFD1A6", height=0.3];
// 
//     Goal [label="Engine starts", fillcolor="#A6C8F4"];
// 
//     subgraph cluster_obvious {
//         label="Obvious"; labeljust=l; style="rounded,dashed"; color="#888888";
//         fontname="Helvetica"; fontsize=12;
//         O1 [label="Fuel"];
//         O2 [label="Battery charged"];
//         O3 [label="Right key"];
//     }
// 
//     subgraph cluster_hidden {
//         label="Less obvious"; labeljust=l; style="rounded,dashed"; color="#888888";
//         fontname="Helvetica"; fontsize=12;
//         H1 [label="Electrical system"];
//         H2 [label="Fuel pump"];
//         H3 [label="Fuel lines clear"];
//         H4 [label="Oil level"];
//     }
// 
//     O2 -> Goal [ltail=cluster_obvious];
//     H2 -> Goal [ltail=cluster_hidden];
// }
// ```
// label=fig:startingacar
// caption=Obvious and less obvious preconditions for the engine to start.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.1.png",
    width: 75%,
  ),
  caption: [Obvious and less obvious preconditions for the engine to start.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:startingacar>
// render_images:end

As @fig:startingacar shows, even a partial diagram of these dependencies
fans out quickly: the engine starting depends on fuel, a charged battery, and
more, each of which has its own enabling conditions. This is the core
problem: every precondition itself requires something else to be true before the
goal can be achieved. Fuel in the tank presupposes a working fuel gauge (or a
recent fill-up), a charged battery presupposes a functioning alternator and
intact wiring, and so on. The dependency chain does not terminate neatly; it
branches and deepens, making a complete, closed specification of "how to start a
car" far harder than it first appears. What looks like a single task is really
a web of interrelated subgoals, each with its own failure modes.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:116 '## The Qualification Problem'
// Slide: The Qualification Problem
== The Qualification Problem

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:118 '* Causal and Exhaustive Rules'
// Slide: Causal and Exhaustive Rules
#strong[Causal and Exhaustive Rules]

To use propositional logic under uncertainty, one must augment the left side of
every rule $X arrow.r.double Y$ so that it satisfies two properties. First, the
rule must be #strong[causal]: $X$ should capture a genuine cause-effect
relationship with $Y$, not merely a statistical correlation. Second, the rule
must be #strong[exhaustive]: every possible condition $X$ that can lead to
outcome $Y$ needs to be enumerated on the left-hand side.

In practice, both requirements run into serious trouble. Every action can hide
an unbounded chain of preconditions that must hold for it to succeed. Consider
starting a car: turning the key presupposes the starter motor is working, which
presupposes the battery is charged, which presupposes the battery has not
degraded, and so on. Each link in the chain reveals another condition that was
silently assumed. Worse, those preconditions are not fixed: they shift depending
on the situation. A charged battery may be sufficient in mild weather, but in
freezing temperatures an additional condition (that it is not too cold for the
battery chemistry to deliver enough current) suddenly matters. The set of
relevant antecedents is therefore open-ended and context-dependent, making true
exhaustiveness unattainable in any realistic domain. This is a core reason why
classical propositional logic, on its own, is inadequate for reasoning under
uncertainty: no finite conjunction of premises can guarantee that every hidden
condition has been captured.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:134 '* Why Qualification Fails'
// Slide: Why Qualification Fails
#strong[Why Qualification Fails]

The #strong[logical qualification problem] refers to the need to enumerate every
precondition that must hold for an action to succeed #cite(
  "mccarthy1980circumscription",
). In principle, a logical agent would need a complete list of conditions
guaranteeing that its planned action will work; in practice, that list is never
finished.

Several reasons make solving the logical qualification problem effectively
impossible #cite("russell2020aima"):

1. #emph[Laziness]: listing every possible rule and exception demands more
  effort than any engineering team can sustain at scale.
2. #emph[Theoretical ignorance]: science itself lacks a complete theory of many
  domains. Medical science, for instance, does not know all the "rules"
  governing disease and treatment.
3. #emph[Practical ignorance]: even when the rules are known in principle, the
  information needed to apply them may be unavailable. A physician may know
  which test would resolve a diagnosis yet be unable to perform that test for a
  particular patient.

These limitations had real historical consequences. The #strong[AI winter] of the late 1980s to early 1990s
followed directly from the mismatch between what rule-based expert systems
promised and what they could deliver. Those systems were built on hand-written
logical rules, and the real world turned out to be far too complex and
open-ended for that approach. Logical rules simply cannot capture all necessary
and sufficient conditions for action in messy, evolving environments, and once
funding agencies and industry recognized the gap, investment dried up for nearly
a decade.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:156 '* Garden World: Rules with Exceptions'
// Slide: Garden World: Rules with Exceptions
#strong[Garden World: Rules with Exceptions]

Consider two propositions: $"Rain"$ means "it rains" and $"WetGrass"$ means "the
grass is wet." Is it true that $"Rain" arrow.r.double "WetGrass"$? In general,
no. If it rains but a protective cover shields the grass, the grass stays dry.
If it rains but the temperature is high, the moisture may evaporate before you
ever notice it. The implication fails because rain alone does not guarantee wet
grass; the surrounding conditions matter.

What about the converse, $"WetGrass" arrow.r.double "Rain"$? This also fails in
general. The grass could be wet because a sprinkler system is running, as
@fig:grasssprinkler shows, or because morning dew has settled on it. Wet
grass simply does not tell you that rain was the cause.

#figure(
  image(
    "../lectures_source/figures/L06.1.grass_sprinkler.png",
    width: 50%,
  ),
  caption: [grass sprinkler],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:grasssprinkler>

To make either rule hold reliably, you need to account for the additional
propositions that capture these interfering conditions: $"Cover"$ ("a protective
cover over the grass"), $"Evaporate"$ ("water evaporates quickly"),
$"Sprinkler"$ ("the sprinkler system is on"), and $"Dew"$ ("morning dew"). Only
by including these factors can you write implications that reflect how the
world actually behaves.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:185 '* Garden World: Causal and Exhaustive Rules'
// Slide: Garden World: Causal and Exhaustive Rules
#strong[Garden World: Causal and Exhaustive Rules]

The goal of writing rules in a knowledge base is to identify every exception and
dependency that make an implication $X arrow.r.double Y$ hold reliably. Two
complementary styles of rule show the difficulty.

First, a #strong[causal rule] traces one cause forward through all the
conditions that must also be true: "if it rains, the grass is not covered, and
the water does not evaporate quickly, then the grass is wet." Formally this
becomes
$"Rain" and not "Cover" and not "Evaporate" and dots.h.c arrow.r.double "WetGrass"$.
Every extra conjunct is a qualification the rule writer had to anticipate;
missing even one (say, the ground is frozen) makes the rule unsound.

Second, an #strong[exhaustive rule] works in the other direction, listing every
possible cause of an observed effect: "the grass is wet if it rained, or the
sprinkler was on, or there is morning dew." In logical form,
$"WetGrass" arrow.r.double "Rain" or "Sprinkler" or "Dew" or dots.h.c$. Here the
open-ended disjunction is the problem: the rule is incomplete whenever a cause
the writer never thought of (a burst pipe, a neighbor's hose) actually wets the
grass.

Both directions, adding qualifications to a causal chain and enumerating all
possible causes, show why purely logical rule-writing is brittle: real-world
domains resist tidy, closed lists of conditions or causes.
@fig:gardenworldcausalandexhaustiverules shows the garden-world example, with
rain connecting to wet grass through intermediate conditions like cover and
evaporation.

// rendered_images:begin
// ```graphviz[width=80%]
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     // Nodes
//     Rain       [label="Rain", fillcolor="#A6C8F4"];
//     WetGrass   [label="WetGrass", fillcolor="#B2E2B2"];
//     Cover      [label="Cover", fillcolor="#FFD1A6"];
//     Evaporate  [label="Evaporate", fillcolor="#F4A6A6"];
//     Sprinkler  [label="Sprinkler", fillcolor="#A0D6D1"];
//     Dew        [label="Dew", fillcolor="#A6E7F4"];
// 
//     // Force ranks
//     { rank=same; Cover; Evaporate; }
//     { rank=same; Sprinkler; Dew; }
// 
//     // Edges
//     Rain -> WetGrass;
//     Rain -> Cover;
//     Rain -> Evaporate;
//     Cover -> WetGrass [label="blocks", style=dashed];
//     Evaporate -> WetGrass [label="blocks", style=dashed];
//     Sprinkler -> WetGrass;
//     Dew -> WetGrass;
// }
// ```
// label=fig:gardenworldcausalandexhaustiverules
// caption=Causes and blocking conditions linking rain to wet grass.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.2.png",
    width: 80%,
  ),
  caption: [Causes and blocking conditions linking rain to wet grass.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:gardenworldcausalandexhaustiverules>
// render_images:end

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:228 '## Probability as the Solution'
// Slide: Probability as the Solution
== Probability as the Solution

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:230 '* Probability and Utility'
// Slide: Probability and Utility
#strong[Probability and Utility]

Propositional logic alone cannot guide action when the world is uncertain. Real
agents rarely have complete information, so they need a richer framework for
choosing what to do.

Acting under uncertainty requires two ingredients working together. The first is
#strong[probability], which lets an agent represent and reason about partial
knowledge: for instance, estimating the likelihood of various outcomes before
committing to a plan. The second is a #strong[utility function], which assigns a
numeric score to each possible outcome reflecting how desirable it is. Utility
functions let the agent weigh competing goals (punctuality, comfort, legal
compliance) on a single scale rather than juggling them informally.

The central idea is #strong[rational choice]: select the plan that maximizes
#emph[expected utility] #cite("vonneumann1944gametheory"). Expected utility
multiplies each outcome's probability by its utility and sums the results,
giving a single number that captures how good a plan is #emph[on average], given
everything the agent currently knows. Maximizing expected utility cannot
guarantee success on any particular occasion, but it gives the agent the best
bet available with the information at hand.

Consider choosing when to leave for the airport, where missing a flight carries
a steep penalty of $-100$. @tab:airportdeparture lays out three options. Leaving
90 minutes early catches the flight 95% of the time with moderate comfort
(utility 10 if caught), yielding an expected utility of 4.50. Leaving 120
minutes early raises the catch probability to 98% at slightly lower comfort
(utility 7), pushing expected utility to 4.86, the best of the three. Leaving a
full 24 hours early virtually guarantees catching the flight (99.99%), but the
discomfort is so large (utility $-20$) that the expected utility plummets to
$-20.01$. The rational choice is 120 minutes: it strikes the best balance
between reliability and cost, even though it is not the option that maximizes
either dimension alone.

#figure(
  styled-table(
    headers: (
      "Leave before",
      "Pr(catch)",
      "Utility if caught",
      "Expected utility",
    ),
    rows: (
      ("90 min", "0.95", "10", "4.50"),
      ([*120 min*], [*0.98*], [*7*], [*4.86*]),
      ("24 h", "0.9999", "-20", "-20.01"),
    ),
  ),
  caption: [When to leave for the airport (missed flight $= -100$).],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:airportdeparture>

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:255 '* Probability as Belief'
// Slide: Probability as Belief
#strong[Probability as Belief]

There is no uncertainty in the actual world: the grass is either wet or dry, and
it either rained or it did not. The uncertainty lives entirely in the observer's
head. #strong[Knowledge is subjective]: probability statements relate to an
agent's knowledge state, not to some objective feature of reality, and updating
that knowledge changes the probabilities accordingly.

Consider a concrete sequence of observations. Initially, you step outside and
observe #emph[wet grass]. Drawing on past data, you estimate
$Pr("Rain" | "WetGrass") = 0.8$, so your current belief is that there is an 80%
chance it rained. This posterior, conditioned on the wet-grass evidence, also
serves as the prior for whatever evidence arrives next.

Now suppose you further observe that the #emph[sprinkler was on]. Wet grass
could be explained by the sprinkler rather than rain, so your belief shifts:
$Pr("Rain" | "WetGrass" and "Sprinkler") = 0.4$, a standard Bayesian
update that redistributes probability mass once a competing explanation enters
the picture.

Finally, a weather report (assumed reliable) says there was no rain. This strong
piece of evidence overwhelms the earlier posterior:

$ Pr("Rain" | "WetGrass" and "Sprinkler" and "WeatherReport") approx 0 $

Each of these three probability statements was correct given the evidence
available at the time it was made. New evidence revised the belief without
making the earlier statements false: an agent who said "80% chance of rain"
after seeing only wet grass was not wrong; that agent simply had less
information. This is the subjective, Bayesian view of probability: beliefs
respond to evidence and change as it accumulates.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:280 '# Probabilistic Reasoning'
// Slide: Probabilistic Reasoning
= Probabilistic Reasoning

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:284 '## Representing Uncertainty'
// Slide: Representing Uncertainty
== Representing Uncertainty

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:286 '* Full Joint Probability Distribution'
// Slide: Full Joint Probability Distribution
#strong[Full Joint Probability Distribution]

The #strong[full joint probability distribution] over a set of random variables
$X_1, X_2, dots, X_n$ assigns a probability to every possible world, where a
possible world is any complete assignment of values to all variables #cite(
  "russell2020aima",
):

$ Pr(X_1 = x_1, X_2 = x_2, dots, X_n = x_n) $

In principle, the full joint distribution can answer any probabilistic query
about the domain: every marginal, every conditional, and every independence
relation is encoded somewhere in that table. The problem is scale. For $n$
variables each taking $k$ values, the table has $k^n$ entries, growing
exponentially with the number of variables. Even a modest domain with a few
dozen binary variables produces a table with billions of rows, making it
impractical to store, let alone to fill in by hand.

Fortunately, real-world domains rarely require all of that machinery. Most
variables are #emph[not fully dependent] on every other variable. A patient's
blood pressure may depend on diet and medication but not on the day's stock
prices. This observation, formalized through #emph[conditional and absolute
  independence] of random variables, is what makes probabilistic reasoning
tractable. Independence relations reduce the number of parameters a model needs,
enabling compact, structured representations such as factorized probabilistic
models and Bayesian networks that capture only the dependencies that actually
exist.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:310 '* Independence of Random Variables'
// Slide: Independence of Random Variables
#strong[Independence of Random Variables]

Two random variables $X$ and $Y$ are #strong[independent] if and only if their
joint probability equals the product of their marginals:

$ Pr(X, Y) = Pr(X) dot.op Pr(Y) $

An equivalent way to state this: knowing $Y$ tells you nothing about $X$, so
conditioning on $Y$ leaves the distribution of $X$ unchanged:

$ Pr(X | Y) = Pr(X) $

For instance, the outcome of a coin flip and tomorrow's weather are independent.
Whether it rains has no bearing on whether the coin lands heads:

$ Pr("Coin" = "Heads" | "Weather" = "Rainy") = Pr("Coin" = "Heads") $

Independence has a practical payoff: it cuts the number of parameters needed
to model a system. If $X_1$ is independent of
$X_2$ and $X_3$, then conditioning on them adds no information:

$ Pr(X_1 | X_2, X_3) = Pr(X_1) $

More broadly, when all three variables are mutually independent, the full joint
distribution factorizes into a simple product:

$ Pr(X_1, X_2, X_3) = Pr(X_1) dot.op Pr(X_2) dot.op Pr(X_3) $

This factorization is what makes independence so valuable in practice. Without
it, specifying a joint distribution over $n$ binary variables requires
enumerating up to $2^n - 1$ entries. With full independence, you need only $n$
marginal distributions. The savings grow exponentially, and much of
probabilistic modeling (from naive Bayes classifiers to the structural
assumptions in Bayesian networks) is built on exploiting independence or its
conditional variants to keep inference tractable.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:336 '## Conditional Independence'
// Slide: Conditional Independence
== Conditional Independence

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:338 '* Conditional Independence'
// Slide: Conditional Independence
#strong[Conditional Independence]

Two random variables $X$ and $Y$ are #strong[conditionally independent] given a
third variable $Z$ if and only if knowing the value of $Z$ renders $X$ and $Y$
independent #cite("koller2009pgm"). Formally,

$ X perp Y | Z <==> Pr(X, Y | Z) = Pr(X | Z) dot.op Pr(Y | Z) $

Consider a concrete scenario: let $X$ represent the event "it is raining today,"
$Y$ the event "a person is carrying an umbrella," and $Z$ the weather forecast.
Without access to the forecast, observing that someone carries an umbrella $Y$
makes rain $X$ more plausible, so the two variables are dependent. Once the
forecast $Z$ is known, however, seeing the umbrella tells you nothing new about
whether it is actually raining: the forecast already accounts for both, so
$X perp Y | Z$.

Conditional independence arises far more frequently in practice than absolute
(marginal) independence, which makes it the workhorse assumption behind compact
probabilistic models. When $X$ and $Y$ are conditionally independent given $Z$,
the joint conditional distribution $Pr(X, Y | Z)$ factorizes into a product of
two simpler terms, $Pr(X | Z) dot.op Pr(Y | Z)$. This factorization is what
allows Bayesian networks and other graphical models to represent
high-dimensional distributions with a manageable number of parameters: instead
of storing the full joint table, each variable's distribution is specified only
in terms of the small set of variables it directly depends on.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:359 '* Fire, Toast, and Alarm'
// Slide: Fire, Toast, and Alarm
#strong[Fire, Toast, and Alarm]

Consider how two events can become independent once we know a third event.
Suppose $"Fire"$ means "there is a fire in the house," $"Toast"$ means "someone
burned toast," $"Alarm"$ means "the fire alarm rings," and $"Call"$ means "a
friend calls to check on you."

The dependencies among these events follow an intuitive chain: $"Alarm"$ depends
on whether $"Fire"$ or $"Toast"$ has occurred, and $"Call"$ depends on whether
$"Alarm"$ rings. At first glance, knowing that a fire broke out might seem
relevant to predicting whether your friend calls, since fire triggers the alarm,
which triggers the call. But that indirect relevance vanishes the moment you
learn the alarm's state.

#wrap-content(
  [
// rendered_images:begin
//     ```graphviz
//     digraph BayesianFlow {
//         splines=true;
//         nodesep=1.0;
//         ranksep=0.75;
//     
//         node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//         penwidth=1.4];
//     
//         Fire   [label="Fire",       fillcolor="#F4A6A6"];
//         Toast  [label="Toast",      fillcolor="#FFD1A6"];
//         Alarm  [label="Alarm",      fillcolor="#A6E7F4"];
//         Call   [label="Call",       fillcolor="#A6C8F4"];
//     
//         { rank = same; Fire; Toast; }
//         { rank = same; Call; }
//     
//         Fire  -> Alarm;
//         Toast -> Alarm;
//         Alarm -> Call;
//     }
//     ```
//     label=fig:firetoastandalarm
//     caption=Fire and toast trigger the alarm, which triggers the call.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.3.png",
    width: 100%,
  ),
  caption: [Fire and toast trigger the alarm, which triggers the call.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:firetoastandalarm>
// render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 45%),
)[
As @fig:firetoastandalarm shows, $"Call"$ is #strong[conditionally
  independent] of both $"Fire"$ and $"Toast"$ given $"Alarm"$. Once you know the
alarm rang, the specific cause (fire or burnt toast) adds no further information
about whether your friend calls:

$ Pr("Call" | "Alarm", "Fire", "Toast") = Pr("Call" | "Alarm") $

The alarm ringing "blocks" the information path from $"Fire"$ and $"Toast"$ to
$"Call"$. Your friend reacts to the alarm itself, not to whatever set it off.
This blocking pattern is a core intuition behind conditional independence in
probabilistic graphical models: an intermediate variable, once observed,
screens off upstream causes from downstream effects.
]

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:408 '* Explaining Away'
// Slide: Explaining Away
#strong[Explaining Away]

Consider $"Rain"$ and $"Sprinkler"$ as two independent causes of $"WetGrass"$
(setting aside $"Weather"$ for the moment). Is
$Pr("Rain" | "Sprinkler") = Pr("Rain")$? Yes: when $"WetGrass"$ has not been
observed, $"Rain"$ and $"Sprinkler"$ are marginally independent
($"Rain" perp "Sprinkler"$). Neither cause tells you anything about the other.

Now condition on the effect. Is
$Pr("Rain" | "Sprinkler", "WetGrass") = Pr("Rain" | "WetGrass")$? No. Once you
know the grass is wet, learning that the sprinkler was on provides an
alternative explanation for the wetness, making rain less likely. Formally,
$"Rain" cancel(perp) "Sprinkler" | "WetGrass"$: the two causes have become dependent
given their common effect.

#wrap-content(
  [
// rendered_images:begin
//     ```graphviz
//     digraph BayesianFlow {
//         splines=true;
//         nodesep=1.0;
//         ranksep=0.75;
//     
//         node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//         penwidth=1.4];
//     
//         Rain       [label="Rain",       fillcolor="#A6C8F4"];
//         Sprinkler  [label="Sprinkler",  fillcolor="#FFD1A6"];
//         WetGrass   [label="WetGrass",   fillcolor="#B2E2B2"];
//     
//         { rank = same; Rain; Sprinkler; }
//     
//         Rain      -> WetGrass;
//         Sprinkler -> WetGrass;
//     }
//     ```
//     label=fig:explainingaway
//     caption=Rain and sprinkler as two independent causes of wet grass.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.4.png",
    width: 100%,
  ),
  caption: [Rain and sprinkler as two independent causes of wet grass.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:explainingaway>
// render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 45%),
)[
This phenomenon is called #strong[explaining away], also known as the collider
effect #cite("pearl1988probabilistic"). It arises whenever two variables jointly
influence a third: observing the effect creates a dependence between the causes
that did not exist before. Evidence supporting one cause "explains" the observed
effect and thereby reduces the need to invoke the other cause. As
@fig:explainingaway shows, observing $"WetGrass"$ "opens" the previously
blocked path between $"Rain"$ and $"Sprinkler"$, allowing information to flow
between them through their shared descendant.
]

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:456 '# Bayesian Networks'
// Slide: Bayesian Networks
= Bayesian Networks

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:458 '## Definition and Intuition'
// Slide: Definition and Intuition
== Definition and Intuition

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:460 '* Bayesian Networks: Definition'
// Slide: Bayesian Networks: Definition
#strong[Bayesian Networks: Definition]

#strong[Bayesian networks] #cite("pearl1988probabilistic"), also known as Bayes
nets, belief networks, or probabilistic networks, are a foundational
representation for reasoning under uncertainty. They belong to the broader class
of graphical models, and when the directed edges carry a causal interpretation
they are sometimes called causal networks.

A Bayesian network is formally a Directed Acyclic Graph (DAG) defined by three
components:

1. #emph[Nodes] $X_i$, each corresponding to a random variable (discrete or
  continuous).
2. #emph[Edges] $X arrow.r Y$, representing direct dependencies among variables.
  An edge from $X$ to $Y$ means $X in "Parents"(Y)$ and $Y in "Children"(X)$.
  The usual family vocabulary (ancestors, descendants, spouses) carries over
  from graph theory.
3. A #emph[conditional probability table] (CPT, sometimes called a conditional
  probability distribution, CPD) for every node $X_i$, quantifying how its
  parents influence it:

$ Pr(X_i | "Parents"(X_i)) $

If a node has no parents, its CPT reduces to an unconditional prior probability
$Pr(X_i)$.

Together, these three ingredients let a Bayesian network encode a full joint
probability distribution compactly. Instead of storing one entry for every
possible combination of all variables, the network factors the joint into a
product of local conditional distributions, one per node. The graph's edges tell
you which conditioning sets matter, and the CPTs supply the numbers. This
factored form is what makes Bayesian networks practical: a domain with dozens of
variables would require an astronomically large joint table, but if each
variable depends directly on only a few parents, the total number of parameters
stays manageable.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:482 '* Bayesian Networks vs Propositional Logic'
// Slide: Bayesian Networks vs Propositional Logic
#strong[Bayesian Networks vs Propositional Logic]

Bayesian networks serve as the probabilistic analogue of propositional logic.
Where propositional logic deals in rigid, all-or-nothing rules (a statement is
either `True` or `False`), Bayesian networks express flexible degrees of belief.
The key shift is replacing a hard implication $X arrow.r.double Y$ with a
conditional probability $Pr(Y | X)$, which captures how strongly observing $X$
should update our belief in $Y$.

#wrap-content(
  [
// rendered_images:begin
//     ```graphviz
//     digraph BayesianFlow {
//         splines=true;
//         nodesep=1.0;
//         ranksep=0.75;
//     
//         node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//         penwidth=1.4];
//     
//         Rain [label="Rain (R)", fillcolor="#A6C8F4", xlabel="P(R) = 0.2"];
//         WetGrass [label="WetGrass (W)", fillcolor="#B2E2B2", xlabel="P(W | R) = 0.9\nP(W | not R) = 0.1"];
//     
//         Rain -> WetGrass;
//     }
//     ```
//     label=fig:bayesiannetworksvspropositionallogic
//     caption=Two-node network with a prior on rain and conditional probabilities for wet grass.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.5.png",
    width: 100%,
  ),
  caption: [Two-node network with a prior on rain and conditional probabilities for wet grass.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:bayesiannetworksvspropositionallogic>
// render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 45%),
)[
Consider a simple "garden world" with two propositions: $R$ ("it is raining")
and $W$ ("the grass is wet"). In propositional logic, if we assert the rule
$R arrow.r.double W$ and observe that $R$ is true, then $W$ must be true; there
is no room for exceptions. A Bayesian network handles the same scenario more
gracefully by assigning $Pr(R = "True") = 0.2$, $Pr(W | R) = 0.9$, and
$Pr(W | not R) = 0.1$. Rain makes wet grass very likely but not certain, and the
grass can occasionally be wet even without rain (perhaps from a sprinkler).
@fig:bayesiannetworksvspropositionallogic shows this small network, with the
prior on $R$ and the two conditional probabilities that link it to $W$.
]

The same contrast appears in medical diagnosis. A propositional rule would state
flatly that "patient has disease $D$" implies "patient has symptom $S$," leaving
no way to express partial evidence or diseases that sometimes present without
the expected symptom. A Bayesian network instead encodes the probability of $S$
given $D$ as high but not certain, acknowledging that symptoms can appear for
other reasons and that a disease does not always produce every textbook sign.
This flexibility makes Bayesian networks the natural formalism for reasoning
under uncertainty. They keep the directed, causal-style structure of logical
implication and replace its brittleness with calibrated probabilities.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:523 '* Bayesian Networks and the Joint Distribution'
// Slide: Bayesian Networks and the Joint Distribution
#strong[Bayesian Networks and the Joint Distribution]

The central result underpinning Bayesian networks is that #strong[graph
  topology] together with #strong[conditional probabilities] suffice to specify
the full joint distribution, often very concisely. Specifically, the joint
probability of all variables decomposes as:

$ Pr(x_1, dots, x_n) = product_(i=1)^n Pr(x_i | "parents"(X_i)) $

This factorization replaces a single exponentially large
joint table with a collection of smaller conditional probability tables, one per
node, each conditioned only on that node's parents in the graph.

The topology of the network, its nodes and directed edges, encodes conditional
independence relationships among the variables. An edge $X arrow.r Y$ means "$X$
has a direct influence on $Y$," though this should not be read as strict
causation without further assumptions. More generally, each node is directly
influenced by its parents and indirectly by its ancestors further up the graph.

Building a Bayesian network involves two complementary steps. First, domain
experts decide which relationships exist among the variables, determining the
graph's topology. This step draws on subject-matter knowledge: a physician knows
that smoking influences lung cancer risk, which in turn influences the result of
an X-ray, and those causal intuitions guide which edges to include. Second, once
the structure is fixed, the conditional probability tables attached to each node
can be specified by experts or estimated from data. The separation of
qualitative structure (the graph) from quantitative parameters (the conditional
probabilities) is one of the framework's key practical advantages: experts can
sketch the topology even when precise numbers must be learned from observations.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:543 '## Conditional Probability Tables'
// Slide: Conditional Probability Tables
== Conditional Probability Tables

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:545 '* Conditional Probability Table'
// Slide: Conditional Probability Table
#strong[Conditional Probability Table]

A #strong[conditional probability table] (CPT) encodes the probability of a node
$X_i$ given its parents, $Pr(X_i | "Parents"(X_i))$. Each row of the CPT
contains the conditional probability of the node for one specific combination of
parent values, so the full table covers every possible conditioning case. This
is what makes a Bayesian network compact: rather than specifying a joint
distribution over all variables at once, each node carries only a small local
table conditioned on its immediate parents.

CPTs arise naturally for discrete variables, where the set of parent value
combinations is finite and each row is just a number (or a short vector for a
multi-valued node). The same idea extends to continuous variables, though the
table is then replaced by a parameterized conditional density. The key insight
is that a finite table of probabilities summarizes a potentially unbounded set
of circumstances. In a purely logical representation, one would need to
enumerate every exception and default case explicitly; a CPT absorbs all of that
variation into a handful of probability values, each row capturing how likely
the child node's values are under one particular parent configuration.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:561 '* Traffic Jam CPT'
// Slide: Traffic Jam CPT
#strong[Traffic Jam CPT]

#wrap-content(
  [
// rendered_images:begin
//     ```graphviz
//     digraph BayesianFlow {
//         splines=true;
//         nodesep=1.0;
//         ranksep=0.75;
//     
//         node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//         penwidth=1.4];
//     
//         Snow [label="Snow", fillcolor="#FFD1A6"];
//         RushHour [label="RushHour", fillcolor="#FFD1A6"];
//         TrafficJam [label="TrafficJam", fillcolor="#B2E2B2", xlabel="P(T | S, R)"];
//     
//         { rank = same; Snow; RushHour; }
//     
//         Snow -> TrafficJam;
//         RushHour -> TrafficJam;
//     }
//     ```
//     label=fig:trafficjamcpt
//     caption=Snow and rush hour as parents of traffic jam.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.6.png",
    width: 100%,
  ),
  caption: [Snow and rush hour as parents of traffic jam.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:trafficjamcpt>
// render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 45%),
)[
Consider three events: $T$ for "Traffic Jam," $S$ for "Snow," and $R$ for "Rush
Hour." A conditional probability table (CPT) models $Pr(T | S, R)$, specifying
the probability of a traffic jam for every combination of its parent variables.

As @fig:trafficjamcpt shows, both Snow and Rush Hour feed into the Traffic
Jam node, and the CPT encodes exactly how their presence or absence shapes the
likelihood of congestion.
]

#figure(
  styled-table(
    headers: (
      $T$,
      $Pr(T|S, R)$,
      $Pr(T|S, not R)$,
      $Pr(T| not S, R)$,
      $Pr(T| not S, not R)$,
    ),
    rows: (
      ("True", "0.80", "0.10", "0.50", "0.05"),
      ("False", "0.20", "0.90", "0.50", "0.95"),
    ),
  ),
  caption: [Conditional probability table for the traffic jam variable.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:trafficjamcpt>

Each column of @tab:trafficjamcpt sums to 1: for any fixed combination of parent
values, the probabilities of $T$ being true and false are complementary. The
entry $Pr(T = "True" | S, R) = 0.80$ means that when it is snowing during rush
hour, there is an 80% chance of a traffic jam. By contrast,
$Pr(T = "True" | not S, not R) = 0.05$ tells us that calm weather outside peak
hours almost never produces congestion. Note that probabilities across different
parent combinations (i.e., across different columns) need not sum to any
particular value; the normalization constraint applies only within each column.

The pattern in the table matches everyday intuition: traffic jams are highly
likely when both snow and rush hour coincide, moderately likely when only one
condition holds (rush hour alone gives a coin-flip 0.50, while snow alone yields
just 0.10), and quite unlikely when neither condition is present. This asymmetry
between the two single-parent cases shows that rush hour alone is a stronger
driver of congestion than snow alone.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:609 '## Garden World Network'
// Slide: Garden World Network
== Garden World Network

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:611 '* Garden World Network'
// Slide: Garden World Network
#strong[Garden World Network]

Consider a small world with five variables. #strong[Weather] captures general
environmental conditions such as sunny or cloudy skies. #strong[Rain] is
directly influenced by Weather: cloudy conditions make rain more probable.
#strong[Sprinkler] is also influenced by Weather, since a homeowner is less
likely to run the sprinkler when it is already raining. #strong[WetGrass]
depends on both Rain and Sprinkler, either of which can leave the lawn wet.
Finally, #strong[StockMarketUp] records whether the stock market closed higher
or lower on a given day.

The first four variables form a natural causal chain: Weather drives both Rain
and Sprinkler, and those two jointly determine WetGrass. StockMarketUp, by
contrast, has no causal connection to any of the others; it is included
precisely to show that a Bayesian network can represent independence
explicitly by leaving a node unconnected. As @fig:gardenworldnetwork shows, the
network encodes each variable's conditional probability given its parents (for
instance, $P(R | W)$ for Rain given Weather), and the absence of an edge between
StockMarketUp and the rest of the graph tells us that the market is marginally
independent of everything in the garden.

// rendered_images:begin
// ```graphviz
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//     penwidth=1.4];
// 
//     Weather       [label="Weather",       fillcolor="#A6E7F4", xlabel="P(W)"];
//     Rain [label="Rain", fillcolor="#A6C8F4", xlabel="P(R | W)"];
//     Sprinkler [label="Sprinkler", fillcolor="#FFD1A6", xlabel="P(S | W)"];
//     WetGrass [label="WetGrass", fillcolor="#B2E2B2", xlabel="P(G | R, S)"];
//     StockMarketUp [label="StockMarketUp", fillcolor="#C6A6F4", xlabel="P(M)"];
// 
//     { rank = same; Rain; Sprinkler; }
// 
//     Weather   -> Rain;
//     Weather   -> Sprinkler;
//     Rain      -> WetGrass;
//     Sprinkler -> WetGrass;
// }
// ```
// label=fig:gardenworldnetwork
// caption=Garden world network with a probability table at each node.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.7.png",
    width: 70%,
  ),
  caption: [Garden world network with a probability table at each node.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:gardenworldnetwork>
// render_images:end

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:650 '* Garden World: Fork Dependence'
// Slide: Garden World: Fork Dependence
#strong[Garden World: Fork Dependence]

$"Rain"$ and $"Sprinkler"$ are #strong[marginally dependent]: knowing whether it
rained changes your belief about whether the sprinkler was on, and vice versa.
Formally, $"Rain" cancel(perp) "Sprinkler"$. Intuitively, if
there is no rain, the weather is likely sunny, and sprinklers are more likely to
be running. The two variables are related because they share a common parent,
$"Weather"$.

However, $"Rain"$ and $"Sprinkler"$ become #strong[conditionally independent] once
$"Weather"$ is observed:

$ "Rain" perp "Sprinkler" | "Weather" $

Once the weather is fixed (say, we know it is sunny), learning whether it
actually rained adds no further information about the sprinkler's state. The
marginal correlation between rain and the sprinkler is entirely #emph[explained
  away] by $"Weather"$, which acts as a confounder. This is the defining
behavior of the fork structure: the common cause induces a marginal association between its
children, but conditioning on it severs that association.
@fig:gardenworldforkdependence shows this fork: $"Weather"$ generates both
$"Rain"$ and $"Sprinkler"$ through its conditional distributions $P(R | W)$ and
$P(S | W)$.

// rendered_images:begin
// ```graphviz
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//     penwidth=1.4];
// 
//     Weather       [label="Weather",       fillcolor="#A6E7F4", xlabel="P(W)"];
//     Rain [label="Rain", fillcolor="#A6C8F4", xlabel="P(R | W)"];
//     Sprinkler [label="Sprinkler", fillcolor="#FFD1A6", xlabel="P(S | W)"];
//     WetGrass [label="WetGrass", fillcolor="#B2E2B2", xlabel="P(G | R, S)"];
//     StockMarketUp [label="StockMarketUp", fillcolor="#C6A6F4", xlabel="P(M)"];
// 
//     { rank = same; Rain; Sprinkler; }
// 
//     Weather   -> Rain;
//     Weather   -> Sprinkler;
//     Rain      -> WetGrass;
//     Sprinkler -> WetGrass;
// }
// ```
// label=fig:gardenworldforkdependence
// caption=Weather as the common cause of rain and sprinkler.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.8.png",
    width: 70%,
  ),
  caption: [Weather as the common cause of rain and sprinkler.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:gardenworldforkdependence>
// render_images:end

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:693 '* Fork and Collider Together'
// Slide: Fork and Collider Together
#strong[Fork and Collider Together]

$"Rain"$ and $"Sprinkler"$ are #strong[dependent] given $"WetGrass"$ when
$"Weather"$ is not observed:

$ "Rain" cancel(perp) "Sprinkler" | "WetGrass" $

Intuitively, if you know the grass is wet and learn the
sprinkler was off, the probability of rain increases, because something had to
make the grass wet. The same reasoning works in reverse; learning it rained
makes the sprinkler less necessary as an explanation.

This dependence persists even after conditioning on $"Weather"$:

$ "Rain" cancel(perp) "Sprinkler" | "WetGrass", "Weather" $

Consider a sunny day: rain is already unlikely and the sprinkler is the more
probable cause of wet grass. If you then observe that the sprinkler is off
despite the grass being wet, rain becomes the only viable explanation, so
$P("Rain" | "WetGrass", "Sprinkler" = "off", "Weather" = "sunny")$ jumps.
Conditioning on $"Weather"$ alone would block the #emph[fork path] (the
common-cause path through $"Weather"$), but observing $"WetGrass"$ opens the
#emph[collider path] (the common-effect path where both $"Rain"$ and
$"Sprinkler"$ feed into $"WetGrass"$). The fork is blocked, but the open
collider path keeps $"Rain"$ and $"Sprinkler"$ dependent even with $"Weather"$
in the conditioning set. @fig:forkandcollidertogether shows both path types in
the same graph.

// rendered_images:begin
// ```graphviz
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//     penwidth=1.4];
// 
//     Weather       [label="Weather",       fillcolor="#A6E7F4", xlabel="P(W)"];
//     Rain [label="Rain", fillcolor="#A6C8F4", xlabel="P(R | W)"];
//     Sprinkler [label="Sprinkler", fillcolor="#FFD1A6", xlabel="P(S | W)"];
//     WetGrass [label="WetGrass", fillcolor="#B2E2B2", xlabel="P(G | R, S)"];
//     StockMarketUp [label="StockMarketUp", fillcolor="#C6A6F4", xlabel="P(M)"];
// 
//     { rank = same; Rain; Sprinkler; }
// 
//     Weather   -> Rain;
//     Weather   -> Sprinkler;
//     Rain      -> WetGrass;
//     Sprinkler -> WetGrass;
// }
// ```
// label=fig:forkandcollidertogether
// caption=Fork and collider paths coexisting in the garden world network.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.9.png",
    width: 70%,
  ),
  caption: [Fork and collider paths coexisting in the garden world network.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:forkandcollidertogether>
// render_images:end

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:738 '* Garden World: Independent Node'
// Slide: Garden World: Independent Node
#strong[Garden World: Independent Node]

The variable $"StockMarketUp"$ is #strong[unconditionally independent] of all
other variables in this network. Graphically, this independence is encoded by
the absence of any edge connecting $"StockMarketUp"$ to another node: no arrow
points into it or out of it. A change in $"StockMarketUp"$ tells us nothing
about $"Weather"$, and vice versa.

Is this independence assumption realistic? Not always. In past data, sunny
weather in New York City has correlated with higher market index returns, and
cold weather drives up demand for heating oil, which in turn moves energy prices
and related equities. If that dependence matters for the
problem at hand, the modeler should add an edge
$"Weather" arrow.r "StockMarketUp"$ to the network, making the weather node a
parent of the stock-market node and allowing the conditional probability table
to capture the relationship. The absence of an edge is itself a modeling choice,
one that asserts "these two quantities carry no information about each other,"
and the network's predictions are only as good as that assertion.
@fig:gardenworldindependentnode shows how the weather node's own probability
table, $P(W)$, feeds into downstream nodes like $"Rain"$ through $P(R | W)$,
while an independent node like $"StockMarketUp"$ sits apart with no such
connection.

// rendered_images:begin
// ```graphviz
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//     penwidth=1.4];
// 
//     Weather       [label="Weather",       fillcolor="#A6E7F4", xlabel="P(W)"];
//     Rain [label="Rain", fillcolor="#A6C8F4", xlabel="P(R | W)"];
//     Sprinkler [label="Sprinkler", fillcolor="#FFD1A6", xlabel="P(S | W)"];
//     WetGrass [label="WetGrass", fillcolor="#B2E2B2", xlabel="P(G | R, S)"];
//     StockMarketUp [label="StockMarketUp", fillcolor="#C6A6F4", xlabel="P(M)"];
// 
//     { rank = same; Rain; Sprinkler; }
// 
//     Weather   -> Rain;
//     Weather   -> Sprinkler;
//     Rain      -> WetGrass;
//     Sprinkler -> WetGrass;
// }
// ```
// label=fig:gardenworldindependentnode
// caption=Stock market node left unconnected from the rest of the network.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.10.png",
    width: 70%,
  ),
  caption: [Stock market node left unconnected from the rest of the network.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:gardenworldindependentnode>
// render_images:end

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:780 '## Burglar Alarm Example'
// Slide: Burglar Alarm Example
== Burglar Alarm Example

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:782 '* Burglar Alarm: Network'
// Slide: Burglar Alarm: Network
#strong[Burglar Alarm: Network]

This is a famous example from Judea Pearl #cite("pearl1988probabilistic") that
shows how Bayesian networks encode conditional independence assumptions
through their graph structure. Consider an $"Alarm"$ system installed at a home
in Los Angeles. The alarm is designed to detect a $"Burglary"$, but it can also
be triggered by a minor $"Earthquake"$. Two neighbors, $"John"$ and $"Mary"$,
have agreed to call the homeowner whenever they hear the alarm go off.

The structure of the network in @fig:burglaralarmnetwork encodes the conditional
independence assumptions of this model. Both $"Burglary"$ and
$"Earthquake"$ are parent nodes of $"Alarm"$, meaning either event can cause the
alarm to sound. $"JohnCalls"$ and $"MaryCalls"$ depend only on
$"Alarm"$, not directly on $"Burglary"$ or $"Earthquake"$. This encodes the
intuition that the neighbors have no way of knowing #emph[why] the alarm is
ringing; they simply hear it and decide whether to call. Once you know the state
of $"Alarm"$, learning whether a burglary or an earthquake occurred provides no
additional information about whether John or Mary will phone. That is exactly
the conditional independence statement $"JohnCalls" perp "Burglary" | "Alarm"$
(and likewise for $"MaryCalls"$), and it is read directly from the graph's
topology rather than computed from a joint probability table.

// rendered_images:begin
// ```graphviz
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//     penwidth=1.4];
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
// label=fig:burglaralarmnetwork
// caption=Burglary and earthquake trigger the alarm, which triggers the neighbors' calls.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.11.png",
    width: 70%,
  ),
  caption: [Burglary and earthquake trigger the alarm, which triggers the neighbors' calls.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:burglaralarmnetwork>
// render_images:end

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:825 '* Burglar Alarm: Probabilities'
// Slide: Burglar Alarm: Probabilities
#strong[Burglar Alarm: Probabilities]

The prior probability of a $"Burglary"$ is 0.001, and the prior probability of
an $"Earthquake"$ is 0.002. The alarm system is fairly reliable at detecting
burglaries, but it also responds to minor earthquakes, producing false
positives. @fig:burglaralarmprobabilities shows these two independent
causes and their prior probabilities feeding into the alarm network.

// rendered_images:begin
// ```graphviz
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//     penwidth=1.4];
// 
//     Burglary [label="Burglary", fillcolor="#A6C8F4", xlabel="P(B) = 0.001"];
//     Earthquake [label="Earthquake", fillcolor="#FFD1A6", xlabel="P(E) = 0.002"];
//     Alarm [label="Alarm", fillcolor="#B2E2B2", xlabel="P(A | B,E)"];
//     JohnCalls    [label="JohnCalls",    fillcolor="#C6A6F4", xlabel="P(J | A)"];
//     MaryCalls    [label="MaryCalls",    fillcolor="#C6A6F4", xlabel="P(M | A)"];
// 
//     { rank = same; Burglary; Earthquake; }
//     { rank = same; JohnCalls; MaryCalls; }
// 
//     Burglary   -> Alarm;
//     Earthquake -> Alarm;
//     Alarm      -> JohnCalls;
//     Alarm      -> MaryCalls;
// }
// ```
// label=fig:burglaralarmprobabilities
// caption=Burglar alarm network annotated with prior and conditional probabilities.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson06.1-Bayesian_Networks.typ.figs/Lesson06.1-Bayesian_Networks.12.png",
    width: 70%,
  ),
  caption: [Burglar alarm network annotated with prior and conditional probabilities.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:burglaralarmprobabilities>
// render_images:end

The conditional probability table for the alarm quantifies this behavior:

#figure(
  styled-table(
    headers: ("Burglary", "Earthquake", "P(Alarm | B, E)"),
    rows: (
      ("True", "True", "0.95"),
      ("True", "False", "0.94"),
      ("False", "True", "0.29"),
      ("False", "False", "0.001"),
    ),
  ),
  caption: [Conditional probability of the alarm given burglary and
    earthquake.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:alarmcpt>

As @tab:alarmcpt shows, when a burglary is occurring the alarm fires with
probability 0.94 or 0.95 regardless of whether an earthquake is also happening.
An earthquake alone, however, triggers the alarm only 29% of the time, and in
the absence of both causes the alarm has a negligible 0.1% false-positive rate.
The high detection rate for burglaries paired with the moderate earthquake
response is what makes this example interesting: the alarm is a noisy-but-useful
sensor whose output conflates two very different causes.

Because $"Burglary"$ and $"Earthquake"$ are independent causes, the marginal
probability of the alarm firing is obtained by summing over all combinations of
their values:

$ Pr("Alarm") = sum_(b, e) Pr("Alarm" | b, e) Pr(b) Pr(e) approx 0.0025 $

In other words, the alarm goes off roughly one time in 400. Most of those
activations come from earthquakes (which are twice as likely as burglaries) or
from the tiny baseline false-positive rate, not from actual burglaries. This low
marginal probability will matter when we later ask the inverse question: given
that the alarm #emph[did] fire, how likely is it that a burglary actually
occurred?

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:881 '* Burglar Alarm: Neighbors'
// Slide: Burglar Alarm: Neighbors
#strong[Burglar Alarm: Neighbors]

Two neighbors, $"John"$ and $"Mary"$, will $"Call"$ when they hear the
$"Alarm"$, but their responses differ in reliability. John almost always calls
when he hears the alarm, with $P("JohnCalls" | "Alarm" = "True") = 0.90$. He
does, however, sometimes confuse the telephone ringing with the alarm and calls
anyway, giving him a false-positive rate of
$P("JohnCalls" | "Alarm" = "False") = 0.05$. Mary, on the other hand, misses the
alarm about 30% of the time, so $P("MaryCalls" | "Alarm" = "True") = 0.70$,
while she rarely calls without cause:
$P("MaryCalls" | "Alarm" = "False") = 0.01$. @tab:neighborcalls summarizes
these probabilities.

#figure(
  grid(
    columns: 2,
    column-gutter: 2em,
    styled-table(
      headers: ("Alarm", "P(JohnCalls)"),
      rows: (
        ("True", "0.90"),
        ("False", "0.05"),
      ),
    ),
    styled-table(
      headers: ("Alarm", "P(MaryCalls)"),
      rows: (
        ("True", "0.70"),
        ("False", "0.01"),
      ),
    ),
  ),
  caption: [Probability that John and Mary call, given the alarm state.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:neighborcalls>

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:915 '* Burglar Alarm: CPTs'
// Slide: Burglar Alarm: CPTs
#strong[Burglar Alarm: CPTs]

A node without parents carries an unconditional (prior) probability: it encodes
our baseline belief before any evidence is observed. For every node, the
probabilities of its possible values must sum to 1 for each combination of
parent values. Because of this constraint, we only need to store
$Pr(X = "True" | dots)$ for a Boolean variable; the complementary probability
$Pr(X = "False" | dots)$ is simply one minus that value, making it redundant to
record explicitly. A Boolean node with $k$ Boolean parents therefore requires
$2^k$ rows in its conditional probability table (CPT), one row per combination
of parent truth values.

#figure(
  grid(
    columns: 1,
    row-gutter: 1em,
    styled-table(
      headers: ("P(Burglary)",),
      rows: (("0.001",),),
    ),
    styled-table(
      headers: ("Alarm", "P(JohnCalls)", "P(¬JohnCalls)"),
      rows: (
        ("True", "0.90", "0.10"),
        ("False", "0.05", "0.95"),
      ),
    ),
    styled-table(
      headers: ("Burglary", "Earthquake", "P(Alarm)"),
      rows: (
        ("T", "T", "0.95"),
        ("T", "F", "0.94"),
        ("F", "T", "0.29"),
        ("F", "F", "0.001"),
      ),
    ),
  ),
  caption: [CPTs for Burglary, JohnCalls, and Alarm.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:cpts>

Consider the three CPTs shown in @tab:cpts. The simplest case is a root node
like Burglary, which has no parents and needs only a single number:
$Pr("Burglary") = 0.001$. A node with one Boolean parent, such as JohnCalls
conditioned on Alarm, needs $2^1 = 2$ rows. When Alarm is true, John calls with
probability 0.90; when Alarm is false, that probability drops to 0.05. The most
complex table here belongs to Alarm itself, which has two Boolean parents
(Burglary and Earthquake) and therefore requires $2^2 = 4$ rows. The table
shows that Alarm is very likely to sound during a burglary
($Pr("Alarm" | B = T, E = T) = 0.95$ and $Pr("Alarm" | B = T, E = F) = 0.94$)
but can also be triggered by an earthquake alone
($Pr("Alarm" | B = F, E = T) = 0.29$), while the false-alarm rate in the absence
of both causes is just 0.001.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:957 '* Summary'
// Slide: Summary
= Summary

Logic-based rules cannot enumerate every precondition of an action in a complex
environment, a difficulty known as the #strong[qualification problem]. Rather
than attempting to list every possible exception, agents represent uncertainty
with probabilities and revise those beliefs as new evidence arrives.
#strong[Bayesian networks] make this approach tractable by exploiting
conditional independence: instead of storing a full joint distribution whose
size grows exponentially, the network encodes the same information with a
directed acyclic graph and one conditional probability table per node.

Several core ideas underpin this framework. First, #emph[logic under
  uncertainty] fails because partial observability, non-determinism, and the
qualification problem together make exhaustive logical rules impossible for
real-world reasoning. Second, #emph[probability] provides a language for
encoding degrees of belief, beliefs that update systematically as evidence
arrives and that combine with utilities to define rational choice. Third,
#emph[independence and conditional independence] tame the exponential cost of
the full joint distribution by letting us factor it into smaller, manageable
pieces. Fourth, the #emph[Bayesian network] formalism captures the full joint
distribution using just a DAG structure plus one CPT per node: a compact
representation that makes inference feasible. Finally, the structural patterns
within these networks, #emph[fork and collider paths], govern how information
flows: observing a common cause in a fork blocks the dependence between its
effects, while observing a common effect in a collider activates a dependence
between its causes through the phenomenon of explaining away.

// From: /Users/saggese/src/umd_classes1/msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd:975 '* References'
// Slide: References
= References

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
