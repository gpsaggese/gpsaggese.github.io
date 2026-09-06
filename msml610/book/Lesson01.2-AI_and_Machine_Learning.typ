// git_hash=b1e45801e-gu7 timestamp=20260830_185619
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
  title: "L01.2: AI and Machine Learning",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L01.2: AI and Machine Learning")

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:12 '# What Is AI?'
// Slide: What Is AI?
= What Is Intelligence? What is AI?

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:14 '* ML, AI, and Intelligence'
// Slide: ML, AI, and Intelligence
== ML, AI, and Intelligence

#strong[Machine Learning] is a subset of Artificial Intelligence (AI). Although
the term is frequently conflated with #emph[deep learning], #emph[large-language
  models], #emph[predictive analytics], and other adjacent fields, each of these
represents a distinct, though overlapping, area of study. Understanding where
machine learning sits in the broader landscape of AI requires first stepping
back and asking a more fundamental question.

What is artificial intelligence? Answering that starts with understanding what
#strong[human intelligence] is. We call ourselves #emph["homo sapiens"] ("wise
man") precisely because intelligence is the trait we believe sets us apart from
other animals. For thousands of years, philosophers, scientists, and more
recently computer scientists have tried to understand how we think, and the
question remains one of the #emph[biggest mysteries] we face. The human brain is
a remarkably small piece of biological matter, yet it has managed to grasp some
of nature's deepest secrets: the theory of relativity, quantum mechanics, and
abstract mathematics (and its limitations like Godel's incompleteness theorems),
the theory of computation, to name just a few.

This raises a profound puzzle: how can a physical system, #emph[the brain],
understand, predict, and manipulate a world that is vastly more complex than
itself? Any serious attempt to build artificial intelligence must grapple with
this question, because the goal, at least in its strongest form, is to replicate
or even surpass that extraordinary capability in a machine.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:33 '* Artificial Intelligence'
// Slide: Artificial Intelligence

#wrap-content(
  [
    #figure(
      grid(
        columns: (1fr,),
        row-gutter: 0.5em,
        image(
          "../lectures_source/figures/L01.2.Dartmouth_Conference_1956.jpg",
          width: 100%,
        ),
        image(
          "../lectures_source/figures/L01.2.Dartmouth_50th_Anniversary.jpg",
          width: 100%,
        ),
      ),
      caption: [The 1956 Dartmouth workshop founders (top) and their 2006
        fiftieth-anniversary reunion (bottom)],
      kind: "figure",
      supplement: [Fig.],
    ) <fig:dartmouthworkshop>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  The term #strong[Artificial Intelligence] was coined in 1956, when John
  McCarthy and colleagues proposed a summer research project at Dartmouth
  College to explore whether every aspect of learning and intelligence could, in
  principle, be described precisely enough for a machine to simulate it #cite(
    "mccarthy1955dartmouth",
  ). That proposal gave the field both its name and its ambitious scope. As
  @fig:dartmouthworkshop shows, the same researchers who launched the field in
  1956 returned five decades later to commemorate the original workshop.
]

#wrap-content(
  [
    #figure(
      image(
        "../lectures_source/figures/L01.2.Richard_Feynman.jpg",
        width: 100%,
      ),
      caption: [Richard Feynman (1965)],
      kind: "figure",
      supplement: [Fig.],
    ) <fig:richardfeynman>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 20%),
)[
  The goals of AI are twofold: to understand human intelligence and to create
  intelligent entities. These goals reinforce each other. Building a system that
  can perceive, reason, and act forces researchers to make their theories of
  cognition precise enough to implement. As @fig:richardfeynman reminds us,
  Richard Feynman captured this idea succinctly: "What I cannot create, I do not
  understand." The act of engineering intelligence is itself a path to
  understanding it.

  What makes AI unique among engineering disciplines is the breadth of its
  ambition paired with the depth of its open questions. AI applies, at least in
  principle, to any human activity and task: from diagnosing diseases to
  composing music, from driving vehicles to proving mathematical theorems. Its
  economic footprint reflects that breadth: AI already generates hundreds of
  billions of dollars annually in market revenue, with trillions in global
  economic impact projected by 2030 #cite("bughin2018aifrontier"). By some
  measures, its societal impact exceeds that of any past historical event,
  including the industrial revolution and the advent of the internet.
]

At the same time, AI remains a discipline with many unresolved problems. This
distinguishes it from fields that possess settled core theories: arithmetic
rests on axioms that have been stable for millennia, and Newtonian mechanics
delivers reliable predictions within its domain. AI has no comparable consensus
on its foundational questions: What is the right representation of knowledge?
How should an agent balance exploration and exploitation? What does it even mean
for a machine to "understand"? These open questions are not signs of immaturity
so much as reflections of the extraordinary difficulty of the problem.
Intelligence, after all, is the most complex phenomenon we have ever tried to
reproduce.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:64 '* AI Formal Definition'
// Slide: AI Formal Definition

=== A Formal Definition of AI

#grid(
  columns: (1fr, 55%),
  column-gutter: 1em,
  align: (left, top),
)[
  Let's now focus on finding a formal definition of AI. AI is characterized
  along #strong[two key axes] #cite("russell2020aima"): the first distinguishes
  whether we care about an agent's internal _thought processes_ or its external
  _behavior_, while the second asks whether the standard of success is fidelity
  to #strong[human] performance or to an #strong[ideal, rational] standard.
  Crossing these two axes yields four distinct ways to define artificial
  intelligence: a machine that can (1) think humanly, (2) think rationally, (3)
  act humanly, or (4) act rationally. @tab:aiformaldefinition lays out these
  four quadrants side by side, making the organizing logic explicit.
][
  #figure(
    styled-table(
      headers: ("", "Human", "Rational"),
      rows: (
        ("Thinking", "Think humanly", "Think rationally"),
        ("Acting", "Act humanly", "Act rationally"),
      ),
      bold-first-col: true,
    ),
    caption: [Table of Human, Rational],
    kind: "table",
    supplement: [Table.],
    placement: auto,
  ) <tab:aiformaldefinition>
]

Which of these definitions best captures what AI should ultimately aim for?
Thinking humanly grounds the field in cognitive science, but human cognition is
riddled with biases and shortcuts that we may not want to replicate. Thinking
rationally appeals to formal logic, yet pure logical reasoning is often
computationally intractable and unable to handle uncertainty. Acting humanly
(the standard behind the Turing Test) is compelling as a benchmark but tells us
little about the internal mechanisms that produce the behavior. #strong[Acting
  rationally] stands out because it sets the bar at doing the right thing given
what the agent knows, regardless of whether the underlying process mirrors human
thought. An agent that acts rationally maximizes its expected performance
measure, can cope with incomplete information, and is not constrained to reason
in any particular style. It simply has to produce good outcomes. This makes
rational action the most inclusive and practically useful target: it subsumes
correct inference when inference is possible, but it also covers fast, reflexive
responses (like pulling a hand from a hot stove) where deliberation would be too
slow. For these reasons, building machines that #emph[act rationally] is widely
regarded as the central goal of modern AI research.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:101 '* 1. AI as Thinking Humanly'
// Slide: 1. AI as Thinking Humanly
=== AI as Thinking Humanly

To build machines that think like humans, we must first #strong[determine how
  humans think], a challenge that sits at the intersection of cognitive science,
psychology, and computer science. The goal is to construct computational models
whose internal reasoning processes mirror those of the human mind, not merely
models that produce the same outputs.

Expressing such a theory as a computer program carries a genuine advantage: it
forces precision. A verbal theory of cognition can hide ambiguities behind
natural language, but a running program must commit to every detail: how
memories are retrieved, how analogies are drawn, how conflicting evidence is
weighed. If the program behaves like a human subject in controlled experiments,
that constitutes real evidence that the underlying theory captures something
true about human cognition.

The tradeoff, however, is substantial. The workings of the human mind remain
largely unknown. Neuroscience can measure blood flow and firing rates of
neurons; psychology can record response times and error patterns; but the
precise algorithms the brain executes are still a matter of active debate.
Building a faithful computational replica of a system we do not yet understand
is difficult, if not impossible. Beyond this empirical gap, the entire framing
is #emph[anthropocentric]: it takes human cognition as the gold standard for
intelligence, quietly assuming that the way humans happen to think is the way
machines #emph[should] think. That assumption is far from obvious. Humans are
subject to systematic biases, limited working memory, and a host of cognitive
shortcuts that evolved for survival on the savanna rather than for optimal
reasoning. Defining machine intelligence in terms of human-likeness risks
importing those limitations as design goals rather than treating them as
constraints to be overcome.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:113 '* 2. AI as Thinking Rationally'
// Slide: 2. AI as Thinking Rationally
=== AI as Thinking Rationally

What are the rules of #strong[correct thinking]? At its core, correct thinking
means that, given correct premises, the process yields correct conclusions.
Nothing false sneaks in along the way.

#wrap-content(
  [
// rendered_images:begin
//     ```graphviz
//     digraph laws_of_thought {
//         bgcolor="transparent";
//         pad="0.15";
//         splines=spline;
//         nodesep=0.4;
//         ranksep=0.5;
//         rankdir=TB;
// 
//         node [shape=box,
//               style="rounded,filled",
//               penwidth=1.8,
//               fontname="Helvetica",
//               fontsize=12,
//               margin="0.22,0.14",
//               height=0.50];
// 
//         edge [color="#A3B1C0",
//               penwidth=1.3,
//               arrowhead=vee,
//               arrowsize=0.75,
//               fontname="Helvetica",
//               fontsize=10,
//               fontcolor="#7B8794"];
// 
//         premises [label="Correct\npremises", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//         logic [label="Logic", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//         conclusion [label="Correct\nconclusions", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
// 
//         premises -> logic -> conclusion;
//     }
//     ```
//     label=fig:2aiasthinkingrationally
//     caption=Diagram relating Correct premises, Logic and Correct conclusions
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson01.2-AI_and_Machine_Learning.typ.figs/Lesson01.2-AI_and_Machine_Learning.1.png"),
  caption: [Diagram relating Correct premises, Logic and Correct conclusions],
) <fig:2aiasthinkingrationally>
// render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 20%),
)[
  #strong[Logic] provides the framework for studying these "laws of thought." It
  allows us to formalize statements about objects in the world and the relations
  among them, expressing claims in a precise symbolic language that removes the
  ambiguity inherent in natural language. Once knowledge is encoded this way,
  purely mechanical rules of inference can derive new truths from old ones. The
  relationship in @fig:2aiasthinkingrationally is straightforward: correct
  premises feed into logic, which in turn produces correct conclusions, making
  the entire chain only as reliable as the premises it starts from.
]

#strong[Automatic theorem proving] takes this idea to its practical limit:
programs accept a problem stated in logical notation and search for a proof.
When a proof exists, such a program will eventually find it. However, there is a
fundamental barrier: first-order validity is only #emph[semi-decidable]. If no
solution exists, the search may run forever without reporting failure. This
means a theorem prover can confirm that a statement is valid but cannot always
confirm that it is #emph[not]. That is a hard ceiling imposed by the mathematics
itself, not by engineering shortcomings.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:168 '* Thinking Rationally: Challenges'
// Slide: Thinking Rationally: Challenges

There are several #strong[challenges with thinking rationally]. The first one is
that #emph[formalizing the informal knowledge] that humans use effortlessly
turns out to be extraordinarily difficult. Consider something as mundane as a
handshake. In plain language, "a handshake occurs when two people extend, grip,
shake hands, then release." Translating even this simple description into formal
logic requires a surprisingly verbose expression:

$
  exists x, y, h_x, h_y : & "Person"(x) and "Person"(y) and x eq.not y and "Hand"(x, h_x) and "Hand"(y, h_y) \ & and "MoveToward"(h_x, h_y) and "Contact"(h_x, h_y) and "Shake"(h_x, h_y) \ & and "Release"(h_x, h_y)
$

This explosion of predicates for a single everyday action hints at why encoding
all of common-sense knowledge remains an open problem.

A second challenge is the #emph[probabilistic nature of knowledge]. Much of what
we know about the world is uncertain rather than categorical. In medicine, for
instance, a patient presenting with fever, cough, and fatigue could have the
flu, COVID-19, or any number of other illnesses. Deterministic logical rules
cannot capture these overlapping, graded possibilities without being augmented
by probability, which adds its own layer of complexity.

Third, there are #emph[scalability challenges]. Even when a problem can be
stated precisely, the search space may grow so large that exact solutions are
computationally intractable. In practice, large problems demand heuristics:
methods that trade guaranteed optimality for the ability to find good-enough
answers in reasonable time.

Finally, intelligence requires more than rational thinking in isolation. An
agent must #emph[interact with the world]: perceive its environment, take
physical or communicative actions, and cope with the consequences. This is the
problem of the embodiment of AI: the recognition that reasoning in a vacuum, no
matter how logically impeccable, falls short of what we mean by intelligent
behavior. A chess engine that cannot parse a spoken question or pick up a piece
is intelligent only in the narrowest sense.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:193 '* 3. AI as Acting Humanly'
// Slide: 3. AI as Acting Humanly
=== AI as Acting Humanly

An #strong[agent] is anything that perceives its environment and acts upon it in
pursuit of a goal. The central ambition of artificial intelligence is to design
agents that can act like humans, not merely follow rigid instructions but
exhibit the flexible, context-sensitive behavior we associate with human
cognition.

How would we know whether a machine has achieved that ambition? The classic
benchmark is the #strong[Turing test] #cite("turing1950computing"). A computer
passes the Turing test if a human interrogator, communicating through text,
cannot reliably tell whether the answers to questions came from a person or from
a computer. The test is deliberately behavioral: it sidesteps philosophical
debates about whether the machine "really" thinks and asks only whether its
outputs are indistinguishable from a human's.

Passing a full, #strong[embodied Turing test] (one that includes physical
interaction, not just text exchange) demands competence across a remarkably
broad set of capabilities:

1. #emph[Natural language processing] to communicate fluently in a human
  language.
2. #emph[Knowledge representation] to store what it knows about the world in a
  structured, retrievable form.
3. #emph[Automated reasoning] to draw conclusions from stored knowledge and
  answer novel questions.
4. #emph[Machine learning] to detect patterns in data and adapt to new
  situations without being explicitly reprogrammed.
5. #emph[Computer vision and speech recognition] to perceive the environment:
  recognizing objects, faces, scenes, and spoken language.
6. #emph[Robotics] to manipulate physical objects and move through space.

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.1.Ex_machina.jpg", width: 100%),
      caption: [Ex machina],
      kind: "figure",
      supplement: [Fig.],
    ) <fig:exmachina>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 20%),
)[
  Each of these areas has grown into a major subfield of AI in its own right,
  yet the Turing test reminds us that genuine human-level intelligence weaves
  them all together seamlessly. A convincing agent must not only see and speak
  but also reason about what it sees and learn from what it hears. As
  @fig:exmachina shows, popular culture has long been fascinated by this vision
  of a machine whose behavior is so fluid and integrated that it becomes
  indistinguishable from a human being.
]

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:223 '* Turing Test: Pros and Cons'
// Slide: Turing Test: Pros and Cons
The #strong[Turing Test] offers a genuinely operational definition of
intelligence: rather than debating abstract qualities, it sets up a concrete
experiment with a clear pass/fail outcome. This sidesteps centuries of
philosophical vagueness about consciousness, subjective experience, and whether
a machine can "really" think. If the interrogator cannot reliably distinguish
the machine from a human, the machine is deemed intelligent. No metaphysics
required.

The tradeoff, however, is significant. Intelligence under this framing is
measured entirely by #emph[anthropomorphic] criteria: the machine must behave in
ways that seem human to a human judge. Yet multiple forms of non-human
intelligence exist (consider the navigational abilities of migratory birds or
the distributed problem-solving of ant colonies), none of which would pass a
test grounded in human conversation. Worse, passing the test ultimately means
#emph[fooling humans] into believing the machine is one of them, which conflates
intelligence with deception and raises the question of whether imitation is
really the goal we should be engineering toward.

An analogy from aeronautical engineering sharpens the point. The Wright brothers
did not succeed by building a machine that flapped its wings like a bird; they
succeeded by studying wind tunnels and aerodynamics: the #emph[principles] of
flight rather than the #emph[appearance] of a flyer. Designing aircraft that
imitate birds would have been the wrong objective entirely. In the same way, a
science of artificial intelligence may be better served by understanding the
principles of rational action than by insisting that machines mimic the surface
behavior of humans.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:248 '* 4. AI as Acting Rationally'
// Slide: 4. AI as Acting Rationally
=== AI as Acting Rationally

A #strong[rational agent] is an agent that does the "right thing" given what it
knows. Rather than requiring perfect knowledge or omniscient foresight,
rationality here is measured against the information actually available to the
agent at the time it acts. If an agent makes the best decision it can with the
evidence at hand, it qualifies as rational, even if hindsight reveals a better
option existed.

What does it take for an agent to act rationally in practice? The requirements
can be distilled into five core capabilities:

1. #emph[Operate autonomously]: the agent should not depend on a human operator
  to dictate every step; it must be able to select actions on its own.
2. #emph[Perceive its environment]: through sensors or data inputs, the agent
  gathers information about the state of the world it inhabits.
3. #emph[Persist over time]: rather than performing a single computation and
  halting, a rational agent maintains an ongoing presence, continuously
  interacting with its environment.
4. #emph[Adapt to change]: environments are rarely static, so the agent must
  update its beliefs and strategies as new information arrives or conditions
  shift.
5. #emph[Create and pursue goals]: autonomy without direction is aimless; a
  rational agent formulates objectives and selects actions that advance those
  objectives.

These five properties work together. Perception feeds the agent's internal model
of the world, persistence gives it the opportunity to observe the consequences
of its actions, adaptation keeps that model accurate, and goal pursuit provides
the criterion for choosing one action over another. Autonomy ties it all
together by ensuring the agent can exercise these capabilities without constant
external intervention. An agent that satisfies all five is well-positioned to
act rationally across a wide range of environments, from a chess board to a
self-driving car navigating city traffic.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:260 '* Acting Rationally as Ultimate Goal of AI'
// Slide: Acting Rationally as Ultimate Goal of AI
=== Acting Rationally as Ultimate Goal of AI

Which definition of AI best captures what we should build? The answer hinges on
two axes: acting versus thinking, and rational versus human-like. Acting is more
fundamental than thinking, because #strong[acting rationally] is a broader
objective: it subsumes correct reasoning as a special case but also covers
situations where an agent must act under uncertainty or time pressure without
the luxury of deliberate thought. Rationality, in turn, is more objective than
human-likeness: it can be defined mathematically through expected-utility
maximization or similar formal criteria, whereas human behavior is shaped by
evolutionary pressures, cognitive biases, and cultural context that are
difficult to pin down as a stable benchmark.

#wrap-content(
  [
    #figure(
      styled-table(
        headers: ("", "Human", "Rational"),
        rows: (
          ("Thinking", "Think humanly", "Think rationally"),
          ("Acting", "Act humanly", [#strong[Act rationally]]),
        ),
        bold-first-col: true,
      ),
      caption: [Table of Human, Rational],
      kind: "table",
      supplement: [Table.],
      placement: auto,
    ) <tab:actingrationallyasultimategoalofai>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 40%),
)[
  These considerations point to a single cell in the classic two-by-two matrix
  shown in @tab:actingrationallyasultimategoalofai: AI should focus on
  #strong[agents acting rationally]. This framing gives the field a clear,
  measurable target: design systems that select actions maximizing expected
  performance given the information available, rather than chasing a moving and
  poorly understood model of human cognition.
]

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:295 '* Rationality Is Not Absolute'
// Slide: Rationality Is Not Absolute

AI aims to build agents that #strong[do the right thing], but what exactly
counts as the "right thing"? This question is far less straightforward than it
first appears, and much of the work in agent design revolves around making it
precise enough to be useful.

Consider a simple scenario: you leave the house and a falling branch strikes
you. Did you act rationally? Almost certainly yes. You had no reason to
anticipate the branch, no low-cost action that would have avoided it, and no
prior information suggesting danger. Rationality does not demand omniscience; it
demands making the best decision given the information available at the time.

Now contrast that with a different scenario: you cross the street without
bothering to check for oncoming traffic, and a car knocks you over. Did you act
rationally? Here the answer tilts toward no. Glancing left and right before
crossing is a near-zero-cost action that dramatically reduces the probability of
a catastrophic outcome. Failing to perform it is hard to justify on any
reasonable weighing of costs and benefits. The asymmetry between the two cases
shows that rational action is not about whether the outcome was good or bad. It
is about whether the #emph[process] of choosing was defensible given what you
knew and what you could have done.

These everyday examples already hint at deeper difficulties, but the stakes
sharpen considerably when we move to autonomous systems. A recurring challenge
in self-driving car design captures this vividly: should a car swerve and hit a
pedestrian to avoid a frontal crash that would kill two occupants? Any answer
forces a choice among competing moral frameworks: utilitarian body counts,
deontological duties not to use a bystander as a means, legal liability, public
trust. No single definition of "the right thing" resolves them all. For an AI
agent, these are not philosophical thought experiments; they are engineering
specifications that must be encoded before the system is deployed. The
difficulty of writing down what "rational" or "right" means in such cases is one
reason that agent design remains as much a conceptual challenge as a technical
one.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:313 '* Problems of a Rational Agent'
// Slide: Problems of a Rational Agent

A rational agent operating in a #strong[probabilistic environment] aims for the
best outcome when the world is deterministic, and for the best #strong[expected
  outcome] when uncertainty is involved. But what does "best" actually mean? The
classical answer is that "best" is determined by an #strong[objective function]:
a cost function, a sum of rewards, a loss function, or a utility that assigns a
numerical score to each possible outcome so that the agent can compare
alternatives.

In practice, however, the picture is more complex, and several limitations
constrain what rationality can deliver.

- #emph[Omniscience vs. no-regrets]: the best decision is based on the
  information available at the time of acting, not on perfect knowledge of the
  world. An agent that makes a reasonable choice given what it knows has nothing
  to regret, even if the outcome turns out poorly.
- #emph[No provably correct action]: in some situations no option can be shown
  to be right, yet the agent must still commit to one. Standing still is itself
  a choice with consequences.
- #emph[Feasibility of perfect reasoning]: even with complete information, full
  rationality may be out of reach for at least three reasons. First, the cost of
  acquiring all relevant data can be prohibitive: ordering every conceivable
  medical test before diagnosing a patient is neither practical nor ethical.
  Second, the computational demands may be staggering: a search tree can have
  more branches than atoms in the observable universe ($tilde 10^{80}$), making
  exhaustive exploration physically impossible. Third, real-time constraints may
  leave no room for deliberation at all: a high-frequency trading system, for
  instance, must decide within a single microsecond.

These constraints motivate the idea of #strong[satisficing] #cite(
  "simon1956satisficing",
): rather than pursuing the theoretically perfect action, a rational agent seeks
one that is #emph[good enough] given the information, computation, and time
actually available. Satisficing reframes rationality not as omniscient
optimality but as acting appropriately under real-world constraints: a standard
that is both achievable and, in many domains, all that can honestly be demanded.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:340 '# What Is Machine Learning?'
// Slide: What Is Machine Learning?
= What Is Machine Learning?

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:342 '* Machine Learning: Definitions'
// Slide: Machine Learning: Definitions

How should we define #strong[machine learning]? The question seems
straightforward, but the field has evolved considerably since its earliest
formulations, and the way we frame the definition shapes how we think about what
these systems can and cannot do.

One of the earliest and most widely cited characterizations comes from Arthur
Samuel, who in 1959 described machine learning as "the field of study that gives
computers the ability to learn without being explicitly programmed" #cite(
  "samuel1959checkers",
). The key insight here is the phrase "without being explicitly programmed":
rather than writing out every rule a system should follow, we instead provide it
with data and let it discover patterns on its own. Samuel's own work
demonstrated this concretely: he built a checkers-playing program that improved
by playing thousands of games against itself, gradually memorizing board
positions associated with wins and losses until it could beat its creator. The
machine was never told which moves were good; it figured that out from
experience.

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.2.Tom_Mitchell.jpg", width: 100%),
      caption: [Tom Mitchell (2025)],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:tommitchell>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 20%),
)[
  Tom Mitchell later sharpened this intuition into something more precise and
  testable. In his 1997 formulation, "a computer program is said to learn from
  experience $E$ with respect to some task $T$ and some performance measure $P$,
  if $P(T)$ improves with experience $E$" #cite("mitchell1997machinelearning").
  This definition is valuable because it gives us three concrete knobs to
  specify for any learning problem: what the task is, what counts as doing well
  at it, and what kind of experience the system trains on. For Samuel's checkers
  program, the task $T$ is playing checkers, the performance measure $P$ is the
  win rate, and the experience $E$ is the corpus of self-play games. As
  @fig:tommitchell shows, Mitchell's contribution to formalizing these ideas
  helped establish machine learning as a rigorous discipline with clearly stated
  objectives rather than a loose collection of heuristics.
]

In practice, today most machine learning's visible applications span
#emph[computer vision] (recognizing objects, faces, and scenes in images),
#emph[speech recognition] (converting spoken language into text, as in virtual
assistants), and #emph[natural language processing] (enabling machines to read,
translate, and generate human language). What unites all of these is the same
core pattern Mitchell identified: a system that improves at a well-defined task
by consuming more data, rather than by a programmer manually encoding every rule
the system needs to follow.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:374 '* The 3 Machine Learning Assumptions'
// Slide: The 3 Machine Learning Assumptions

Machine learning addresses a practical engineering challenge that unfolds in
stages: gathering a dataset, building a statistical model from that dataset
algorithmically, evaluating the model's quality, and finally deploying and
monitoring it in production. Of these stages, most are engineering concerns:
data collection pipelines, evaluation harnesses, deployment infrastructure. But
#emph[building the model] is the research core, the step where genuine
scientific questions arise.

Abu-Mostafa (2012) #cite("abumostafa2012learning") distills this research core
into #emph[three core assumptions] that underpin all of machine learning:

1. A #emph[pattern exists] in the data: there is some regularity worth
  capturing.
2. That pattern #emph[cannot be precisely defined mathematically]: if a
  closed-form solution were available, there would be no need for learning from
  examples.
3. #emph[Data is available]: without observations, no algorithm can discover
  anything.

Which of these assumptions is truly essential? Let's consider each in turn. If
no pattern exists, running a learning algorithm is futile: the model will fit
noise and generalize poorly. Yet in practice we rarely face data that is pure
noise; some structure almost always lurks beneath the surface, even if it is
weak. The second assumption, that mathematics alone cannot pin down the pattern,
is even softer. There are cases where a precise mathematical derivation is
possible, and we still use machine learning anyway because the learning-based
solution is cheaper to develop, more adaptable, or good enough for the task at
hand. Violating this assumption does not block progress; it merely means we had
an alternative route we chose not to take.

The third assumption, however, is non-negotiable. #emph[Without data, no
  progress is possible.] A learning algorithm with no observations has nothing
to generalize from, no signal to extract, no hypothesis to validate. Data
availability is therefore the assumption that is truly essential: the one whose
absence renders the entire enterprise impossible rather than merely suboptimal.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:399 '* AI vs ML vs Deep Learning'
// Slide: AI vs ML vs Deep Learning

#strong[Artificial Intelligence (AI)] refers to machines programmed to reason,
learn, and act in a rational way. #strong[Machine Learning (ML)] is a subset of
AI in which machines become capable of performing tasks without being explicitly
programmed for each one. They improve through experience with data instead. AI
models that are not ML are entirely possible: handcrafted rule-based systems,
such as IBM's Deep Blue chess engine, contain no learning from data whatsoever,
yet they still qualify as AI systems because they encode expert reasoning into
their decision logic.

#wrap-content(
  [
// rendered_images:begin
//     ```tikz
//     % Define colors.
//     \definecolor{AIcolor}{RGB}{244,166,166}    % Red/Pink
//     \definecolor{MLcolor}{RGB}{178,226,178}    % Green
//     \definecolor{DLcolor}{RGB}{160,214,209}    % Teal
//     \definecolor{LLMcolor}{RGB}{198,166,244}   % Purple
// 
//     % Draw AI circle
//     \fill[AIcolor] (0,0) circle (3);
//     \draw (0,0) circle (3);
//     \node[above] at (0,2) {\textbf{AI}};
// 
//     % Draw ML circle inside AI
//     \fill[MLcolor] (0.5,-0.5) circle (2);
//     \draw (0.5,-0.5) circle (2);
//     \node[above] at (0.5,0.5) {\textbf{ML}};
// 
//     % Draw DL circle inside ML
//     \fill[DLcolor] (1,-1) circle (1);
//     \draw (1,-1) circle (1);
//     \node[above] at (1,-0.6) {\textbf{DL}};
// 
//     % Draw LLM circle inside DL
//     \fill[LLMcolor] (1.2,-1.2) circle (0.6);
//     \draw (1.2,-1.2) circle (0.6);
//     \node[above] at (1.2,-1.4) {\textbf{LLMs}};
//     ```
//     label=fig:aivsmlvsdeeplearning
//     caption=Diagram illustrating AI vs ML vs Deep Learning
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson01.2-AI_and_Machine_Learning.typ.figs/Lesson01.2-AI_and_Machine_Learning.2.png"),
  caption: [Diagram illustrating AI vs ML vs Deep Learning],
) <fig:aivsmlvsdeeplearning>
// render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  Within machine learning, #strong[Deep Learning (DL)] denotes the use of a
  particular family of models (neural networks with many layers) to learn
  hierarchical representations of data. #strong[Large Language Models (LLMs)]
  are a specific kind of deep neural network trained on massive text datasets to
  predict text, often further refined with reinforcement learning from human
  feedback (RLHF) #cite("ouyang2022instructgpt"). Just as AI is broader than ML,
  DL is broader than LLMs: a convolutional neural network designed for computer
  vision or speech recognition is a deep learning system, but it is not a large
  language model because it neither processes nor generates natural language in
  the autoregressive sense that defines LLMs. In @fig:aivsmlvsdeeplearning,
  these categories nest concentrically: LLMs sit inside Deep Learning, which
  sits inside Machine Learning, which in turn sits inside the broader field of
  Artificial Intelligence. The nesting makes their subset relationships visually
  clear.
]

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:457 '* Limits of AI Compared to Human Intelligence (1/2)'
// Slide: Limits of AI Compared to Human Intelligence (1/2)

#strong[AI and machine learning systems differ] fundamentally from human
intelligence and machines do not learn the way humans do: large language models,
for instance, process statistical patterns over enormous corpora, yet a child
acquires language from a comparatively tiny stream of input. Whether the brain
uses anything resembling gradient descent remains an open and actively debated
question in computational neuroscience. Reinforcement learning comes closer:
dopaminergic reward signals in the brain bear a striking resemblance to
temporal-difference error signals, so the brain probably does something
analogous to reinforcement learning, at least in a loose sense. Still, the gap
between biological and artificial learning is wide, and it shows up in several
concrete limitations.

- #emph[Fragility to input variations.] Current ML models can fail
  catastrophically when inputs are distorted even slightly. Adversarial attacks
  demonstrate this vividly: altering a single pixel in an image can cause a
  classifier to switch its prediction entirely #cite("su2019onepixel").
  Similarly, a reinforcement-learning agent trained to master a video game may
  collapse if the screen is rotated by a few degrees, whereas a human player
  adapts effortlessly. This brittleness contrasts sharply with the robustness of
  biological perception.

- #emph[Lack of transfer learning.] A physician who learns cardiology can draw
  on that knowledge when studying pulmonology, because human expertise transfers
  fluidly across related domains. ML systems, by contrast, typically cannot
  apply what they have learned in one task to another without substantial
  retraining or architectural redesign. That limitation makes each new problem
  almost as expensive as the first.

- #emph[Massive data and compute requirements.] Training modern ML models
  demands enormous datasets and computational resources. A teenager learns to
  drive competently in a matter of hours behind the wheel, yet self-driving
  systems require billions of compute hours and vast quantities of labelled
  sensor data before they can operate safely. The disparity shows just how
  sample-efficient biological learning remains compared to its artificial
  counterpart.

- #emph[Poor common sense and reasoning.] Perhaps the most stubborn gap is that
  ML systems lack the built-in world knowledge and intuitive logic that humans
  take for granted. A person knows without being told that a cup of coffee
  turned upside down will spill, but an ML model has no such prior unless it has
  been explicitly trained on similar scenarios. This absence of common sense
  makes current systems unreliable in open-ended, real-world situations where
  novel reasoning is required.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:480 '* Limits of AI Compared to Human Intelligence (2/2)'
// Slide: Limits of AI Compared to Human Intelligence (2/2)

- #emph[Lack of transparency.] Many ML models remain largely opaque: they
  produce predictions without exposing the reasoning behind them, which limits
  trust, interpretability, and accountability in high-stakes settings such as
  medical diagnosis or criminal sentencing. Even when a model performs well on
  its benchmark, stakeholders may be unable to verify #emph[why] a particular
  decision was made, making it difficult to catch errors or challenge outcomes.

- #emph[Narrow objectives.] ML systems tend to excel only at narrow,
  well-defined objectives. When the goal is ambiguous or has many conflicting
  parts, optimizing a single proxy metric can backfire spectacularly. A
  recommendation algorithm told to maximize user engagement, for instance, may
  learn that sensational or harmful content keeps people clicking, an outcome
  that satisfies the literal objective while undermining the platform's broader
  mission.

- #emph[Susceptibility to bias.] Because a model can only learn from the data it
  is given, any systematic bias in the training set, whether from historical
  discrimination, sampling gaps, or labeling errors, will be inherited and often
  amplified by the learned function. The result is a system that appears
  objective but quietly encodes the prejudices of its inputs.

- #emph[Lack of embodiment.] Current ML systems lack physical interaction with
  the world. Human cognition is deeply grounded in sensory and motor experience:
  we learn about gravity by dropping things, about heat by touching a stove.
  Disembodied models trained on text or images alone miss this layer of
  understanding, which may explain why they struggle with commonsense physical
  reasoning that even young children handle effortlessly.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:494 '* Key Takeaways'
// Slide: Key Takeaways
= Summary

The central thesis of modern AI research is that the field should focus on
#emph[agents acting rationally], that is, systems that perceive their
environment and take actions that maximize their chances of achieving
well-defined goals, rather than attempting to faithfully mimic human thought
processes or human behavior.

Within this broad vision, several nested disciplines have emerged.
#emph[Artificial Intelligence (AI)] refers to machines programmed to reason,
learn, and act rationally in pursuit of objectives. #emph[Machine Learning (ML)]
is a subset of AI in which systems learn to perform tasks from data rather than
being explicitly programmed with rules for every situation. #emph[Deep Learning
  (DL)] narrows the focus further to ML techniques built on multi-layer neural
networks, which learn hierarchical representations of data. Finally, #emph[Large
  Language Models (LLMs)] are a subset of deep learning: neural networks trained
on massive text datasets to predict the next token in a sequence, often further
refined through Reinforcement Learning from Human Feedback (RLHF) to align their
outputs with human preferences.

Despite the remarkable capabilities of current ML and DL systems (from
superhuman performance on narrow benchmarks to fluent natural-language
conversation), they still fall well short of human intelligence in several
important respects. They tend to be #emph[fragile], breaking down when inputs
shift even slightly from the distribution they were trained on. They struggle
with #emph[transfer learning], finding it difficult to carry knowledge gained in
one domain over to a related but distinct one, something humans do almost
effortlessly. They are notably #emph[data-inefficient], often requiring millions
of labeled examples to master a task a child could learn from a handful of
demonstrations. And they largely lack #emph[common-sense reasoning]: the vast
web of intuitive physical, social, and causal knowledge that humans bring to
bear on everyday decisions without conscious effort. Recognizing these gaps is
essential for understanding both where the field stands today and where the
hardest open problems remain.

// From: msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd:510 '* References'
// Slide: References
= References

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
