// git_hash=b1e45801e-she timestamp=20260830_184941
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
  title: "L01.2: AI and Machine Learning",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter(00, "L01.2: AI and Machine Learning")

// From: msml610/lectures_source/Lesson00.1-Test.smd:10 '# AI and Machine Learning'
// Slide: AI and Machine Learning
#strong[AI and Machine Learning]

// From: msml610/lectures_source/Lesson00.1-Test.smd:12 '## What Is AI?'
// Slide: What Is AI?
== What Is AI?

// From: msml610/lectures_source/Lesson00.1-Test.smd:14 '* ML, AI, and Intelligence'
// Slide: ML, AI, and Intelligence
#strong[ML, AI, and Intelligence]

#strong[Machine Learning] is a subset of Artificial Intelligence (AI). Although
the term is frequently conflated with #emph[deep learning], #emph[large-language
  models], #emph[predictive analytics], and other adjacent fields, each of these
represents a distinct — though overlapping — area of study. Understanding where
machine learning sits in the broader landscape of AI requires first stepping
back and asking a more fundamental question.

What is artificial intelligence? Answering that starts with understanding what
#strong[human intelligence] is. We call ourselves #emph["homo sapiens"] — "wise
man" — precisely because intelligence is the trait we believe sets us apart from
other animals. For thousands of years, philosophers, scientists, and more
recently computer scientists have tried to understand how we think, and the
question remains one of the #emph[biggest mysteries] we face. The human brain is
a remarkably small piece of biological matter, yet it has managed to grasp some
of nature's deepest secrets: the theory of relativity, quantum mechanics, and
the physics of black holes, to name just a few. This raises a profound puzzle:
how can a physical system — the brain — understand, predict, and manipulate a
world that is vastly more complex than itself? Any serious attempt to build
artificial intelligence must grapple with this question, because the goal, at
least in its strongest form, is to replicate or even surpass that extraordinary
capability in a machine.

// From: msml610/lectures_source/Lesson00.1-Test.smd:33 '* Artificial Intelligence'
// Slide: Artificial Intelligence
#strong[Artificial Intelligence]

The term #strong[Artificial Intelligence] was coined in 1956, when John McCarthy
and colleagues proposed a summer research project at Dartmouth College to
explore whether every aspect of learning and intelligence could, in principle,
be described precisely enough for a machine to simulate it #cite(
  "mccarthy1955dartmouth",
). That proposal gave the field both its name and its ambitious scope.

The goals of AI are twofold: to understand human intelligence and to create
intelligent entities. These goals reinforce each other. Building a system that
can perceive, reason, and act forces researchers to make their theories of
cognition precise enough to implement — vague verbal descriptions will not
compile. As @fig:richardfeynman reminds us, Richard Feynman captured this idea
succinctly: "What I cannot create, I do not understand." The act of engineering
intelligence is itself a path to understanding it.

#figure(
  image("../lectures_source/figures/L01.2.Richard_Feynman.jpg", width: 80%),
  caption: [Richard Feynman (1965)],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:richardfeynman>

What makes AI unique among engineering disciplines is the breadth of its
ambition paired with the depth of its open questions. AI applies, at least in
principle, to any human activity and task — from diagnosing diseases to
composing music, from driving vehicles to proving mathematical theorems. Its
economic footprint reflects that breadth: AI already generates hundreds of
billions of dollars annually in market revenue, with trillions in global
economic impact projected by 2030 #cite("bughin2018aifrontier"). By some
measures, its societal impact exceeds that of any past historical event,
including the industrial revolution and the advent of the internet.

At the same time, AI remains a discipline with many unresolved problems. This
distinguishes it from fields that possess settled core theories — arithmetic
rests on axioms that have been stable for millennia, and Newtonian mechanics
delivers reliable predictions within its domain. AI has no comparable consensus
on its foundational questions: What is the right representation of knowledge?
How should an agent balance exploration and exploitation? What does it even mean
for a machine to "understand"? These open questions are not signs of immaturity
so much as reflections of the extraordinary difficulty of the problem.
Intelligence, after all, is the most complex phenomenon we have ever tried to
reproduce.
