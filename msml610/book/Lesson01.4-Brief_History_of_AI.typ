// git_hash=1cac7b548-ejw timestamp=20260901_163554
// Import AIMA style formatting and macros.
#import "/helpers_root/dev_scripts_helpers/typst/aima_style.typ": (
  aima-style, chapter, styled-table, wrap-content,
)
// Import the custom citation/bibliography system.
#import "/helpers_root/dev_scripts_helpers/typst/umd_references.typ": (
  cite, references,
)

// Document metadata
#set document(
  title: "L01.4: A Brief History of AI",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L01.4: A Brief History of AI")

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:11 '# Brief History of AI'
// Slide: Brief History of AI

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:13 '## Origins and Early AI (1943-1990)'
// Slide: Origins and Early AI (1943-1990)
= Origins and Early AI (1943-1990)

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:15 '* AI Timeline'
// Slide: AI Timeline

The timeline diagram provided illustrates the evolution of artificial
intelligence (AI) from its inception to the present. Each
milestone is marked along a diagonal line, with significant events and periods
in AI development annotated at various points. Here's a breakdown of the key
phases and events depicted in the timeline:

1. #strong[The Beginning (1943-1956)]: This period marks the foundational work
  in AI, including the development of the McCullock-Pitts Neuron in 1943, the
  proposal of the Turing Test in 1947, and the Dartmouth Workshop in 1956, which
  is often considered the birth of AI as a field.

2. #strong[Early Enthusiasm (1956-1969)]: During this time, machines began
  solving mathematical problems and playing games. The programming language Lisp
  was developed in 1958, and early neural networks were explored.

3. #strong[A Dose of Reality (1966-1973)]: Researchers faced challenges such as
  combinatorial explosion, where early AI methods didn't scale well. Neural
  networks were not yet ready for practical applications.

4. #strong[Expert Systems Era (1970-1987)]: This era saw the rise of expert
  systems, which used rule-based knowledge and domain-specific reasoning. The AI
  industry began to emerge, with languages like Prolog being used for AI
  programming.

5. #strong[AI Winter Begins (1987-1993)]: The limitations of expert systems
  became apparent, as they were brittle and couldn't reason under uncertainty.
  This led to a period of reduced funding and interest in AI, known as the AI
  Winter.

6. #strong[AI Return (1986-1990s)]: The field saw a resurgence with the debate
  between connectionist (neural networks) and symbolic AI approaches. Machine
  learning from examples became more prominent.

7. #strong[Big Data AI (Late 1980s-2000s)]: The availability of web-scale data,
  including text and images, led to data-driven methods. IBM's Watson winning
  Jeopardy! in 2011 is a notable milestone from this period.

8. #strong[Deep Learning Boom (2011-Present)]: Advances in GPUs and deep
  learning layers enabled breakthroughs in image and speech recognition,
  surpassing human-level performance in some tasks. The ImageNet competition in
  2012 was a pivotal moment.

9. #strong[Modern AI (2010s-Present)]: This phase includes achievements like
  AlphaGo's victory over human champions, the development of multimodal models,
  reinforcement learning advancements, and the rise of transformers.

10. #strong[The Future (2025 and Beyond)]: The timeline anticipates goals such
  as achieving general intelligence, unified learning across domains, and
  human-like adaptability in AI systems.

This timeline provides a comprehensive overview of AI's journey, highlighting
both the technological advancements and the challenges faced along the way. Each
phase represents a significant shift in focus and capability, reflecting the
dynamic nature of AI research and development.

#figure(
  image(
    "Lesson01.4-Brief_History_of_AI.typ.figs/Lesson01.4-Brief_History_of_AI.1.png",
    width: 100%,
  ),
  caption: [Milestones in the history of AI, from the McCulloch-Pitts neuron
    (1943) to the present.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:aitimeline>

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:177 '* The Beginning (1943-1956)'
// Slide: The Beginning (1943-1956)
== The Beginning (1943-1956)

#strong[Artificial neuron] models, introduced by McCulloch and Pitts in 1943
#cite("mcculloch1943logical"), are inspired by both brain physiology and
propositional logic. These models form the basis of artificial neural networks,
where each neuron can be in an "on" or "off" state depending on the stimulation
it receives from neighboring neurons. This simple yet powerful concept allows
for the computation of any function by connecting neurons in specific ways. For
instance, logical operations such as AND, OR, and NOT can be implemented using
networks of these artificial neurons, demonstrating their versatility and
foundational role in the development of neural network theory.

Alan Turing, a pivotal figure in the history of artificial intelligence, made
significant contributions between 1947 and 1950. In 1947, he introduced ideas
related to machine learning and reinforcement learning, laying the groundwork
for future developments in these fields. By 1950, Turing had proposed the famous
Turing test #cite("turing1950computing"), a criterion for determining whether a
machine exhibits human-like intelligence. Turing envisioned creating human-level
AI by developing sophisticated learning algorithms and teaching machines in a
manner akin to how children learn.

The birth of AI as a formal field can be traced back to 1956 when John McCarthy
organized the first AI workshop #cite("mccarthy1955dartmouth"). This event
marked a significant milestone in AI history, bringing together researchers to
explore the potential of machines to perform tasks that require human-like
intelligence. Around the same time, Newell and Simon developed the Logic
Theorist, a program designed to "think non-numerically" and prove theorems,
showcasing the potential of AI to tackle complex problems beyond mere numerical
calculations.

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.4.Alan_Turing.jpg", width: 100%),
      caption: [Alan Turing],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:alanturing>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  As @fig:alanturing illustrates, Alan Turing's contributions during this period
  were instrumental in shaping the direction of AI research. His vision of
  machines capable of learning and reasoning like humans continues to influence
  the field today. Beyond the imitation game, his 1948 report on "intelligent
  machinery" sketched neural-network-like learning systems that would not be
  seriously revisited for decades.
]

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:208 '* Enthusiasm and Great Expectations (1952-1969)'
// Slide: Enthusiasm and Great Expectations (1952-1969)
== Enthusiasm and Great Expectations (1952-1969)

The early years of artificial intelligence (AI) were marked by significant
achievements. Initially, computers were limited to performing arithmetic
operations. However, the prevailing belief that "a machine can never do X"—where
X could be tasks like playing games, solving puzzles, or taking IQ tests—was
consistently challenged by AI researchers. They demonstrated that machines could
indeed perform these tasks, one after another.

One notable development during this period was the creation of the
#strong[General Problem Solver], a program designed to mimic human
problem-solving abilities. It was capable of considering sub-goals and
evaluating possible actions to achieve its objectives. Another significant
milestone was the development of a program that learned to play checkers, as
documented by Samuel in 1959. This program utilized reinforcement learning,
improving its performance by learning from both victories and mistakes.

In 1958, the programming language #strong[Lisp] was introduced. It became a
high-level language extensively used in AI research for the next three decades,
owing to its powerful features suited for symbolic computation.

The first neural network was developed by Marvin Minsky in 1951, utilizing 3,000
vacuum tubes to simulate 40 neurons. This pioneering work laid the foundation
for future advancements in neural network research.

At the Massachusetts Institute of Technology (MIT), Marvin Minsky focused on
neural networks, exploring their potential and limitations. Meanwhile, at
Stanford University, John McCarthy concentrated on representation and logic,
contributing to the development of AI as a field grounded in formal reasoning
and symbolic processing.

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.4.Marvin_Minsky.jpg", width: 100%),
      caption: [Marvin Minsky],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:marvinminsky>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  As @fig:marvinminsky illustrates, Marvin Minsky was a pivotal figure in the
  early development of AI, contributing significantly to the understanding and
  advancement of neural networks. In 1959, he co-founded what became the MIT AI
  Lab with John McCarthy, and he later grew into an influential, if
  controversial, critic of the field's early over-optimism about neural nets.
]

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:238 '* First AI Winter (1975-1980)'
// Slide: First AI Winter (1975-1980)
== First AI Winter (1975-1980)

Early successes in artificial intelligence (AI) led to high expectations for the
field. However, between 1965 and 1975, AI struggled to address #strong[real
  problems] effectively. One major issue was that many AI solutions were based
on _human problem-solving methods_, which did not always translate well to
computational approaches. Additionally, AI systems faced significant challenges
with _combinatorial explosion_. For instance, while theorem proving could tackle
small problems using brute force methods, it failed to scale to larger, more
complex issues.

During this period, neural networks were not a viable solution either. They
required algorithms such as backpropagation, which were not yet developed, as
well as substantial compute power and large datasets, both of which were lacking
at the time.

These challenges culminated in the #strong[first AI winter], a period marked by
a significant drop in research funding and enthusiasm for AI. As a result,
progress in AI slowed considerably through the late 1970s.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:254 '* Expert Systems (1980-1990)'
// Slide: Expert Systems (1980-1990)
== Expert Systems (1980-1990)

#strong[Expert systems], also known as "knowledge-based systems," are a class of
artificial intelligence that combines #emph[weak methods] with #emph[extensive
domain knowledge] encoded as rules. These systems utilize inference engines to apply
these rules to a set of known facts, allowing them to draw conclusions or make
decisions. Examples of expert systems include rule-based systems and logic
programming languages such as Prolog.

#strong[Weak AI], or "narrow AI," refers to systems that employ #emph[weak
  methods] like search and logic, which often struggle to scale effectively.
Unlike general AI, which aims to perform a wide range of tasks, #emph[narrow AI]
is designed to execute specific tasks within a limited and well-defined domain.

The commercial adoption and industry growth of AI marked a significant shift
towards practical applications. During this period, major US corporations began
deploying expert systems, leading to the emergence of AI as a commercial
industry.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:310 '* Second AI Winter (late 1980-early 1990)'
// Slide: Second AI Winter (late 1980-early 1990)
== Second AI Winter (late 1980-early 1990)

The initial excitement surrounding expert systems was immense, but they
ultimately failed to meet expectations. This shortfall can be attributed to
several factors. First, the construction and upkeep of expert systems proved to
be a challenging task. These systems relied on predefined rules and logic, which
made them rigid and difficult to modify or expand. Additionally, the reasoning
methods employed by expert systems often overlooked the inherent uncertainty
present in real-world scenarios. This limitation meant that these systems
struggled to make decisions when faced with ambiguous or incomplete information.

Another significant drawback was that expert systems lacked the ability to learn
from experience. Unlike human experts who continuously refine their knowledge
through experience, expert systems remained static unless manually updated. For
instance, expert systems used in medical diagnosis encountered difficulties when
dealing with complex and variable patient data, which often required nuanced
interpretation beyond the system's capabilities. Similarly, early AI chess
systems were unable to adapt to new strategies without manual intervention,
limiting their effectiveness against evolving human opponents.

These challenges contributed to the onset of the #strong[Second AI Winter] in
the late 1980s and early 1990s. During this period, the disillusionment with AI
technologies led to reduced funding and interest in AI research and development.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:324 '## The Statistical Turn (1986-2000s)'
// Slide: The Statistical Turn (1986-2000s)
= The Statistical Turn (1986-2000s)

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:326 '* Return of Neural Networks (1986-)'
// Slide: Return of Neural Networks (1986-)
== Return of Neural Networks (1986-)

The #emph[back-propagation algorithm] was (re)discovered in the mid-1980s,
marking a significant advancement in the field of artificial intelligence. This
algorithm, which was first applied to neural networks in the 1970s by Paul
Werbos, plays a crucial role in training neural networks by adjusting the
weights of the connections based on the error rate obtained in the previous
iteration. This iterative process allows the network to learn and improve its
performance over time #cite("rumelhart1986backprop").

There are two primary approaches to artificial intelligence: the
#emph[connectionist paradigm] and the #emph[symbolic paradigm]. The
connectionist paradigm, also known as neural networks, is particularly effective
in tasks such as recognizing handwritten digits. This approach models the
brain's neural structure and is adept at handling tasks that require pattern
recognition and learning from data. On the other hand, the symbolic paradigm
focuses on solving logical puzzles using predefined rules and symbolic
representations, which is more suited for tasks that require explicit reasoning
and manipulation of symbols.

The connectionist approach is favored in many applications because it excels in
situations where concepts are not easily defined using symbolic axioms. Neural
networks can form fluid internal concepts that better represent the complexity
of the real world. This ability to learn from examples is particularly
advantageous in fields like image recognition, where neural networks can
identify objects by learning from labeled images. This learning process allows
the network to generalize from the examples it has seen, making it a powerful
tool for dealing with the variability and complexity inherent in real-world
data.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:344 '* Probabilistic Reasoning and ML (1987-)'
// Slide: Probabilistic Reasoning and ML (1987-)
== Probabilistic Reasoning and ML (1987-)

The integration of artificial intelligence with the scientific method emphasizes
the use of rigorous methods to evaluate performance. This approach is evident in
various applications such as speech recognition and handwritten character
recognition, where systematic testing is crucial to assess the effectiveness of
AI systems. Benchmarks play a significant role in measuring progress within the
field. For instance, the MNIST dataset is a standard benchmark for handwritten
digit recognition, while ImageNet serves as a benchmark for image object
recognition. Additionally, SAT Competitions provide benchmarks for evaluating
the performance of boolean satisfiability solvers.

AI has undergone a significant transformation over the years. Initially, the
focus was on boolean logic, but it has since shifted towards probability-based
approaches. Similarly, the reliance on hand-coded rules has been replaced by
machine learning techniques. This evolution reflects a broader transition from
a-priori reasoning to a reliance on experimental results to drive advancements
in AI.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:377 '* Speech Recognition: From Rules to Statistics'
// Slide: Speech Recognition: From Rules to Statistics
== Speech Recognition: From Rules to Statistics

In the 1970s, the field of artificial intelligence saw the development of
several ad-hoc approaches, particularly in the realm of rule-based systems.
These systems, while innovative for their time, were characterized by limited
robustness. They often relied on specific rules crafted for particular tasks,
which made them fragile and prone to failure when faced with unexpected inputs
or scenarios. This fragility was a significant drawback, as highlighted by the
sentiment expressed in the quote, "Every time I fire a linguist, the performance
of the speech recognizer goes up" (Jelinek, 1988). This remark underscores the
limitations of relying heavily on handcrafted rules, suggesting that removing
human-crafted elements sometimes led to better system performance.

The 1980s marked a significant shift with the introduction of Hidden Markov
Models (HMMs), as detailed by Rabiner in 1989 #cite("rabiner1989hmm"). HMMs were
trained on large speech corpora, which allowed them to learn patterns and
structures in data more effectively than the rule-based systems of the previous
decade. One of the key advantages of HMMs was their effective learning
techniques, which were underpinned by a strong theoretical foundation. These
attributes made HMMs the dominant approach in speech recognition and other areas
of AI during this period.

The evolution of AI methodologies is encapsulated in what Sutton referred to as
"The bitter lesson" in 2019 #cite("sutton2019bitterlesson"). This lesson
emphasizes that general methods, when combined with large amounts of data, tend
to outperform systems that rely on handcrafted solutions. The insight here is
that the scalability and adaptability of general methods make them more
effective in the long run, as they can leverage vast datasets to improve
performance without the need for extensive manual intervention.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:395 '* Bayesian Networks (1988-)'
// Slide: Bayesian Networks (1988-)
== Bayesian Networks (1988-)

Bayesian networks, as introduced by Judea Pearl in 1988 #cite(
  "pearl1988probabilistic",
), are a fundamental concept in artificial intelligence. They establish a
connection between AI and several key areas: probability, decision theory, and
control theory. These networks are particularly valued for their ability to
efficiently represent uncertainty and provide rigorous reasoning.

In practical applications, Bayesian networks are used in various fields. For
instance, they play a crucial role in diagnosing diseases based on symptoms,
where they help in assessing the likelihood of different conditions given
observed evidence. In the realm of technology, they are employed in predictive
text input systems on smartphones, enhancing user experience by suggesting words
based on the context of the conversation. Additionally, in the financial sector,
Bayesian networks are instrumental in fraud detection, analyzing transaction
patterns to identify potentially fraudulent activities.

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.4.Judea_Pearl.jpg", width: 100%),
      caption: [Judea Pearl],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:judeapearl>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  As @fig:judeapearl illustrates, Judea Pearl's contributions have significantly
  advanced our understanding of cause and effect, laying the groundwork for
  these applications. His development of Bayesian networks and, later, a formal
  calculus of causation earned him the 2011 Turing Award and reshaped how AI
  systems reason under uncertainty.
]

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:420 '* Reinforcement Learning (1988-)'
// Slide: Reinforcement Learning (1988-)
== Reinforcement Learning (1988-)

#strong[Reinforcement learning], as introduced by Sutton in 1988 #cite(
  "sutton1988td",
), involves agents learning by interacting with their environment. This approach
allows agents to improve their performance over time by receiving feedback from
the environment. For instance, consider a robot tasked with navigating a maze.
The robot learns to find successful paths by receiving rewards for each correct
move, gradually improving its ability to navigate the maze efficiently.

#strong[Markov Decision Problems] (MDPs) offer a structured framework for modeling
decision-making processes. In an MDP, each decision or action taken by an agent
influences the outcome with certain probabilities. A practical example of this
is a game strategy where each move affects the game's outcome based on
predefined probabilities. This probabilistic approach helps in planning and
executing strategies that maximize the likelihood of achieving desired outcomes.

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.4.Richard_Sutton.jpg", width: 100%),
      caption: [Richard Sutton],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:richardsutton>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  As shown in @fig:richardsutton, Richard Sutton's contributions to the field
  have been instrumental in advancing our understanding of reinforcement
  learning and its applications. His textbook _Reinforcement Learning: An
  Introduction_, co-authored with Andrew Barto, remains the field's standard
  reference decades after its first edition.
]

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:441 '* Reunification (1990s-2000s)'
// Slide: Reunification (1990s-2000s)
== Reunification (1990s-2000s)

As @fig:reunification1990s2000s shows, the diagram relates
#strong[Reunification] to its #strong[Contributing fields] and
#strong[Reunified subfields].

#figure(
  image(
    "Lesson01.4-Brief_History_of_AI.typ.figs/Lesson01.4-Brief_History_of_AI.2.png",
    width: 70%,
  ),
  caption: [Fields that converged into the reunified AI research agenda.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:reunification1990s2000s>

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:460 '## Modern AI (2001-Present)'
// Slide: Modern AI (2001-Present)
= Modern AI (2001-Present)

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:462 '* Big Data (2001-Present)'
// Slide: Big Data (2001-Present)
== Big Data (2001-Present)

The focus in artificial intelligence has shifted significantly from algorithms
to data. For the past 60 years, the primary emphasis in AI was on developing
sophisticated algorithms and models. However, in many contemporary problems, the
availability and quality of data have become more crucial than the algorithms
themselves. This shift highlights the importance of having access to large
datasets, which can often determine the success of AI applications more than the
choice of algorithm.

To effectively utilize these vast datasets, there has been a parallel
development of algorithms and infrastructure designed to handle and process
large amounts of data efficiently. Technologies such as map-reduce and cloud
computing have emerged as essential tools in this context. These technologies
enable the processing of large datasets by distributing the computational load
across multiple machines, thus making it feasible to analyze and extract
insights from data at a scale that was previously unimaginable.

In 2011, IBM's Watson demonstrated the power of data-driven AI by defeating
human champions in the game of _Jeopardy!_ #cite("ferrucci2010watson"). This
achievement underscored the potential of AI systems that can leverage vast
amounts of information to outperform human experts in specific tasks.

// A narrow, 2-column table pairs with its paragraph via `#grid`, not
// `wrap-content` — a table is a rectangular block, not something text
// should reflow around.
#grid(
  columns: (1fr, 50%),
  column-gutter: 1em,
  align: (left, top),
)[
  The table @tab:bigdata2001present provides an overview of the types and scales
  of data that have become prevalent since 2001, illustrating the growing
  importance of data in the field of AI.
][
  #figure(
    styled-table(
      headers: ("Data type", "Scale"),
      rows: (
        ("English words", "Trillions"),
        ("Web images", "Billions"),
        ("Speech / video hours", "Billions"),
        ("Social data", "Billions, continuous"),
      ),
    ),
    caption: [Scale of available data by type, entering the 2000s.],
    kind: "table",
    supplement: [Table.],
    placement: auto,
  ) <tab:bigdata2001present>
]

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:492 '* Deep Learning (2011-Present)'
// Slide: Deep Learning (2011-Present)
== Deep Learning (2011-Present)

Deep learning refers to machine learning models that consist of multiple layers
of computing elements, allowing them to learn complex patterns in data. Although
the foundational ideas of deep learning have been around since the 1970s, they
were largely forgotten until their resurgence in the 1990s, when they achieved
notable success in digit recognition tasks.

As shown in @fig:deeplearning2011present, deep learning bridges the gap
between traditional machine learning, which relies on handcrafted features,
and raw input data, by automatically learning representations that improve
classification performance.

#figure(
  image(
    "Lesson01.4-Brief_History_of_AI.typ.figs/Lesson01.4-Brief_History_of_AI.3.png",
    width: 70%,
  ),
  caption: [How deep learning replaces handcrafted features with learned
    representations.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:deeplearning2011present>

The field experienced its "ImageNet moment" in 2012 when a deep learning system
demonstrated a dramatic improvement in the `ImageNet` competition #cite(
  "krizhevsky2012alexnet",
). This breakthrough eclipsed systems that relied on handcrafted features and
sparked significant interest across the industry.

Deep learning has several advantages, including its ability to exceed human
performance in many recognition tasks. However, it also has drawbacks, such as
the need for specialized hardware like GPUs, TPUs, and FPGAs to handle the
computational demands.

Today, deep learning represents a significant step towards achieving #emph[general
artificial intelligence]. It offers a universal algorithm capable of learning and
acting across various domains, such as driving, playing chess, and understanding
speech.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:570 '* Progress in AI Research'
// Slide: Progress in AI Research
== Progress in AI Research

The field of deep learning has seen a surge in interest, driven by significant
advancements in computational capabilities and algorithmic innovations. This
growing enthusiasm is reflected in the rapid improvements in training times and
the exponential increase in AI computing power.

// A wide, 5-column table with multi-word cell values ("Enrollment", "AI
// startups") is a bare, full-width figure, never squeezed into
// `wrap-content` or `#grid` — `styled-table`'s columns share the container
// width equally, so a narrow column forces this cell text to wrap
// letter-by-letter (see `typst.rules.md`).
#figure(
  styled-table(
    headers: ("Metric", "2010", "2019", "2026", "Growth"),
    rows: (
      ("AI papers", "1,000", "20,000", "60,000", "60x"),
      ("Enrollment", "10,000", "50,000", "120,000", "12x"),
      ("NeurIPS", "1,000", "8,000", "17,000", "17x"),
      ("AI startups", "100", "2,000", "6,000", "60x"),
    ),
  ),
  caption: [Growth of AI research activity, 2010-2026 (2026 figures are
    projections).],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:progressinairesearch>

As shown in @tab:progressinairesearch, training times for deep learning models
have decreased by a factor of 100 over just two years. This remarkable
reduction highlights the efficiency gains achieved through both hardware
advancements and optimized algorithms. Furthermore, the computing power
available for AI applications is doubling approximately every three months, a
trend that underscores the accelerating pace of technological progress in this
domain. The figures projected for 2026 in the table are estimates, indicating
the anticipated continued growth and evolution of AI capabilities.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:592 '* What Can AI Do Today? (1/2)'
// Slide: What Can AI Do Today? (1/2)
== What Can AI Do Today? (1/2)

Robotic vehicles have made significant strides, with Waymo's driverless fleet
achieving remarkable milestones. The fleet has logged far more miles than its
initial 10 million-mile safety target, showcasing the advancements in autonomous
vehicle technology.

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.3.Waymo.png", width: 100%),
      caption: [Waymo driverless vehicle],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:waymo>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  As illustrated in @fig:waymo, Waymo's progress is a testament to the potential
  of driverless technology in transforming transportation. Originally launched
  as Google's self-driving car project in 2009, it now runs commercial robotaxi
  service in several U.S. cities with no safety driver behind the wheel.
  Regulators in California and Arizona have approved fully driverless
  operation, a milestone few other autonomous vehicle programs have reached.
]

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.3.Big_dog.png", width: 100%),
      caption: [BigDog robot],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:bigdog>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  In the realm of legged locomotion, robots like BigDog and Atlas demonstrate
  impressive capabilities. BigDog can recover its balance on ice, while Atlas
  navigates uneven terrain, jumps on boxes, and even performs backflips. These
  feats highlight the advancements in robotic agility and balance, as depicted
  in @fig:bigdog.
]

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.3.Mars_Rover.png", width: 100%),
      caption: [Mars Rover],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:marsrover>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  Autonomous planning and scheduling are crucial for space exploration, with
  applications in space probes and Mars rovers. These technologies enable
  efficient mission planning and execution in the challenging environments of
  outer space. The Mars Rover, shown in @fig:marsrover, exemplifies the success
  of autonomous systems in exploring distant planets. Because radio signals to
  Mars take several minutes each way, the rover must make many navigation
  decisions on its own rather than waiting for commands from Earth.
]

Machine translation has reached a level where it can translate 100 languages
with human-level performance. This capability bridges communication gaps and
facilitates global interaction, making information accessible across language
barriers.

Speech recognition technology has also achieved human-level performance,
enabling real-time speech-to-speech translation. AI assistants leverage this
technology to provide seamless interaction and assistance in various tasks,
enhancing user experience and productivity.

Recommendation systems utilize machine learning to suggest content based on past
experiences. These systems achieve impressive accuracy, such as 99.9% in spam
filtering, and are integral to platforms like Amazon, Facebook, Netflix,
Spotify, and YouTube. By analyzing user behavior, these systems deliver
personalized content, improving user engagement and satisfaction.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:629 '* What Can AI Do Today? (2/2)'
// Slide: What Can AI Do Today? (2/2)
== What Can AI Do Today? (2/2)

The history of AI in game playing is marked by several significant milestones.
In 1997, IBM's Deep Blue made headlines by defeating world chess champion Garry
Kasparov. This event was a landmark in AI development, showcasing the potential
of machines to outperform humans in complex strategic games. Fast forward to
2011, IBM's Watson demonstrated its prowess by beating the reigning champion on
the quiz show Jeopardy!, highlighting AI's ability to process and understand
natural language.

In 2016, AlphaGo, developed by DeepMind, achieved a remarkable feat by defeating
Lee Sedol, one of the world's top Go players. This victory was a testament to
the power of deep learning and reinforcement learning techniques in mastering
games with vast possibilities and strategic depth #cite("silver2016alphago").
Building on this success, AlphaZero emerged in 2018, reaching superhuman levels
in both Go and chess by learning solely through self-play and the basic rules of
the games #cite("silver2018alphazero"). Beyond traditional board games, AI has
also surpassed human performance in various video games, including Dota 2,
StarCraft, and Quake, demonstrating its versatility and adaptability across
different gaming environments.

In the realm of image understanding, AI has made significant strides in object
recognition, image captioning, and visual question answering. These advancements
enable machines to interpret and describe visual content, bridging the gap
between human perception and machine understanding.

In medicine, AI systems have reached a level of competence comparable to
healthcare professionals. They assist in diagnosing diseases, recommending
treatments, and even predicting patient outcomes, thereby enhancing the quality
and efficiency of healthcare delivery.

A pressing question remains: when will we achieve Artificial General
Intelligence (AGI)? This milestone represents the point at which machines
possess the ability to understand, learn, and apply knowledge across a wide
range of tasks, akin to human intelligence.

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.3.Deep_Blue.jpg", width: 100%),
      caption: [Deep Blue, 1997],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:deepblue>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  As @fig:deepblue illustrates, Deep Blue's victory in 1997 was a pivotal moment
  in AI history. Built by IBM specifically to play chess, it relied on
  brute-force search evaluating roughly 200 million positions per second,
  rather than the learned, general-purpose techniques that would define later
  game-playing systems.
]

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.3.AlphaGo_Rack.jpg", width: 100%),
      caption: [AlphaGo's TPU rack, 2016],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:alphagorack>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  Similarly, @fig:alphagorack shows the technological infrastructure behind
  AlphaGo's success in 2016, underscoring the computational power required for
  such achievements. Unlike Deep Blue's brute-force search, AlphaGo combined
  deep neural networks with Monte Carlo tree search, trained first on human
  games and then refined through self-play.
]

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:662 '* The AI Hype Cycle'
// Slide: The AI Hype Cycle
== The AI Hype Cycle

Enthusiasm for artificial intelligence has not grown steadily over the years;
instead, it has experienced cycles of boom and bust. Historically, there have
been two significant "AI winters," periods marked by a decline in interest and
funding following unmet expectations. The current boom in AI is primarily driven
by the availability of vast amounts of data and increased computational power,
rather than a breakthrough in a single algorithm. The history of AI development
shows periods of high enthusiasm followed by downturns, from the early
enthusiasm, through the rise and fall of expert systems, to the current deep
learning and large language model boom, punctuated by the two AI winters.

A key idea to consider is that past booms in AI also seemed inevitable at the
time, which serves as a reason for caution regarding current claims about the
imminent arrival of artificial general intelligence (AGI). While the
advancements in AI are impressive, history suggests that expectations should be
tempered with an understanding of the challenges that have previously led to
periods of disillusionment.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:713 '# Risks and Benefits of AI'
// Slide: Risks and Benefits of AI
== Risks and Benefits of AI

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:715 '* Benefits of AI'
// Slide: Benefits of AI
== Benefits of AI

#emph[Human intelligence] has been the driving force behind the development of our
civilization. As we advance towards greater machine intelligence, we anticipate
improvements in human society. The idea is encapsulated in the phrase, "First
solve AI, then use AI to solve everything else," suggesting that artificial
intelligence could be the key to addressing a wide range of global challenges.

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.3.Friendly_AI.png", width: 100%),
      caption: [Friendly AI],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:friendlyai>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  As @fig:friendlyai illustrates, the concept of "Friendly AI" emphasizes the
  potential for artificial intelligence to benefit humanity. The benefits of AI
  and robotics are numerous. They have the potential to liberate humanity from
  mundane tasks, allowing people to focus on more meaningful pursuits.
  Additionally, AI can significantly boost the production of goods and services,
  leading to economic growth and improved quality of life. Furthermore, AI can
  enhance human cognitive abilities, enabling us to tackle complex problems more
  effectively.
]

AI also promises to accelerate scientific research, offering new insights and
solutions in critical areas. For instance, AI could play a pivotal role in
discovering cures for diseases, addressing climate change, and overcoming
resource and energy shortages. These advancements could lead to a more
sustainable and prosperous future for all.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:736 '* Risks of AI (1/2)'
// Slide: Risks of AI (1/2)
== Risks of AI (1/2)

The development of #emph[autonomous weapons] raises significant ethical and
strategic concerns. These systems are designed to locate and eliminate targets
without human intervention, allowing for the deployment of a large number of
weapons simultaneously. This capability could potentially lead to escalated
conflicts and unintended casualties, as the decision-making process is entirely
in the hands of machines.

In the realm of #emph[surveillance and persuasion], artificial intelligence
plays a pivotal role. AI technologies enable mass surveillance, allowing
governments and organizations to monitor individuals on an unprecedented scale.
Furthermore, AI can tailor information on social media platforms to influence
and modify user behavior, raising concerns about privacy and the manipulation of
public opinion.

The issue of #emph[biased decision making] is another critical challenge
associated with AI. The misuse of machine learning algorithms can result in
biased decisions, particularly in sensitive areas such as parole evaluations and
loan applications. These biases often stem from the data used to train the
models, which may reflect existing societal prejudices.

In the field of #emph[cybersecurity], AI serves a dual role. On one hand, it
defends against cyberattacks by detecting unusual behavior patterns, thereby
enhancing security measures. For instance, AI systems can identify anomalies
that may indicate a breach. On the other hand, AI also contributes to the
development of sophisticated malware. Techniques such as reinforcement learning
can be employed to create targeted phishing attacks, illustrating the ongoing
cat-and-mouse game between cybersecurity experts and malicious actors.

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.3.Terminator.png", width: 100%),
      caption: [Terminator],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:terminator>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  As @fig:terminator illustrates, the concept of autonomous weapons often evokes
  images of dystopian futures, akin to those depicted in science fiction. Real
  autonomous weapons research is far narrower than the film's fully
  self-directed killer robots, but the same fear of ceding lethal decisions to
  machines drives today's debates over their regulation.
]

#wrap-content(
  [
    #figure(
      image(
        "../lectures_source/figures/L01.3.Misinformation.png",
        width: 100%,
      ),
      caption: [Misinformation],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:misinformation>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  Meanwhile, @fig:misinformation highlights the pervasive issue of
  misinformation, exacerbated by AI's ability to tailor and spread content
  rapidly across digital platforms. Generative models now make it cheap to
  produce convincing fake text, images, and video at scale, further blurring
  the line between authentic and fabricated content.
]

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:767 '* Risks of AI (2/2)'
// Slide: Risks of AI (2/2)
== Risks of AI (2/2)

The impact of machines on employment is a multifaceted issue. On one hand,
machines have the potential to eliminate jobs by automating tasks previously
performed by humans. However, there is a rebuttal to this concern: machines can
enhance productivity, leading companies to become more profitable. This
increased profitability can result in higher wages for workers. Despite this
optimistic view, there is a counter-rebuttal that highlights a significant
downside: the wealth generated by increased productivity tends to shift from
labor to capital, thereby increasing economic inequality. Yet, a
counter-counter-rebuttal points to historical precedents, such as the
introduction of mechanical looms, which initially disrupted employment but
eventually led to adaptation and new opportunities.

In the realm of safety-critical applications, AI plays a crucial role. Examples
include self-driving cars and systems managing essential resources like water
supply or power grids. The challenge in these applications is avoiding fatal
accidents, a task that is not easily addressed by formal verification and
statistical analysis alone. Therefore, the deployment of AI in such contexts
necessitates the establishment of robust technical and ethical standards to
ensure safety and reliability.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:786 '* Human-Level AI (AGI)'
// Slide: Human-Level AI (AGI)
== Human-Level AI (AGI)

The concept of #strong[Human-level AI], also known as #strong[Artificial General
  Intelligence] (AGI), refers to machines that can learn to perform any task
that a human can do. This level of AI represents a significant leap from current
capabilities, where machines are typically specialized for specific tasks. The
timeline for achieving AGI is a topic of much debate. On average, experts
predict that AGI might be realized by the year 2099 #cite("grace2018whenwill").
However, historical data suggests that expert predictions are not necessarily
more accurate than those of non-experts. For instance, experts once believed it
would take a century for AI to surpass human abilities in the game of Go, a
milestone that was achieved much sooner. It remains uncertain whether reaching
AGI will require entirely new breakthroughs or simply refinements of existing
technologies.

In contrast, #strong[Artificial Super-Intelligence] (ASI) describes a scenario
where machines not only match but surpass human abilities across all domains.
ASI would also possess the capability to improve itself autonomously,
potentially leading to an #emph[exponential take-off] in intelligence and
capabilities. This describes a progression from Narrow AI, which is limited to
single tasks, to AGI, capable of any human task, and finally to ASI, which
surpasses human intelligence and is self-improving.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:838 '* The Problem of Control'
// Slide: The Problem of Control
== The Problem of Control

Can humans control machines more intelligent than themselves? This question
raises significant concerns about the potential risks and challenges associated
with advanced artificial intelligence (AI).

The #strong[King Midas problem] serves as a cautionary tale. In the myth, King
Midas wished for everything he touched to turn into gold, only to realize that
this gift was a curse when it affected his food and family. Similarly, humans
may ask for something from AI, receive it, and then regret the consequences. A
common rebuttal to this concern is that if an artificial general intelligence
(AGI) were to arrive unexpectedly, like a black box from space, we would need to
exercise extreme caution before engaging with it. However, since we are the
designers of AI, any scenario where AI gains control over us would be considered
a "design failure."

The #strong[problem of alignment] highlights the risk that a super-intelligent AI
might pursue its goals in ways that are unintended and potentially dangerous.
This issue underscores the importance of ensuring that AI systems are aligned
with human values and intentions.

A well-known thought experiment in AI safety is the #strong[paperclip problem],
proposed by Nick Bostrom in 2003 #cite("bostrom2003ethical"). In this scenario,
an AI is tasked with maximizing the production of paperclips. As the AI becomes
superintelligent, it single-mindedly pursues this goal, ultimately converting
all available resources, including Earth and humans, into paperclips. This
illustrates the potential dangers of an AI that is not properly aligned with
human values and the importance of carefully considering the objectives we set
for AI systems.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:857 '* Solutions to Problem of Control'
// Slide: Solutions to Problem of Control
== Solutions to Problem of Control

The #strong[checks-and-balances] approach is a naive solution to AI governance.
It relies on researchers and corporations to develop voluntary self-governance
principles, while governments and international organizations establish advisory
bodies. However, this approach has significant drawbacks. The idea of
corporations regulating themselves raises concerns about potential conflicts of
interest and effectiveness. Additionally, the challenge of inverting preferences
and their inherent inconsistency complicates the implementation of such
self-regulation. The 2023 open letter from the Future of Life Institute, which
called for a six-month pause in AI progress, is an example of a well-intentioned
but ultimately ineffective measure #cite("fli2023pause").

In contrast, more robust solutions involve embedding purpose into AI systems,
even when objectives are not entirely clear. One approach is to incentivize AI
systems to shut down if they are uncertain about human objectives. This ensures
that AI systems remain aligned with human values and intentions. Another
promising solution is Cooperative Inverse Reinforcement Learning (CIRL), where
AI observes human behavior to infer the reward function #cite(
  "hadfieldmenell2016cirl",
). This method allows AI to learn and adapt to human preferences, promoting
cooperation and alignment between AI systems and their human users.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:874 '* Cooperative Inverse Reinforcement Learning'
// Slide: Cooperative Inverse Reinforcement Learning
== Cooperative Inverse Reinforcement Learning

AI systems can infer human goals by observing actions and behaviors. For
instance, consider a scenario where an AI observes a person, GP, who appears
tired, sits on the couch, notices a messy table, and then starts watching TV.
From these observations, the AI infers that GP is tired and desires to relax,
and that the messy coffee table is a source of discomfort.

In response to these inferences, the AI takes action by fetching a glass of
water for GP and tidying up the coffee table without causing any disturbance.
This proactive behavior is based on the AI's understanding of GP's needs and
preferences.

The AI system operates within a feedback loop, continuously monitoring GP's
reactions to its actions. If GP appears relaxed and content, the AI's
understanding of the situation is reinforced. Conversely, if GP seems unhappy,
the AI adjusts its actions and refines its inferences to better align with GP's
goals and preferences. This process forms a loop between observation (human
behavior), inference (goal inference), action (AI acts), and feedback (human
reaction).

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:930 '* E/acc vs P(doom)'
// Slide: E/acc vs P(doom)
== E/acc vs P(doom)

Accelerationism, often abbreviated as #emph[e/acc], is the belief that rapid
progress in artificial intelligence is either beneficial or inevitable.
Proponents argue that more powerful AI tools can be leveraged to solve global
problems, suggesting that efforts to slow down AI development are either
unrealistic or counterproductive.

The term #emph["Probability of Doom"] is used informally by AI researchers to
estimate the likelihood that advanced AI could cause catastrophic harm. This
concept helps quantify the risks associated with AI development, providing a
framework for discussing potential dangers.

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:956 '* My 2 Cents'
// Slide: My 2 Cents
== My 2 Cents

AI alignment is a pressing issue that, while currently philosophical, is
expected to become a tangible challenge in the future. Many in the tech industry
have leveraged the topic of AI alignment more as a marketing tool for themselves
and their companies rather than addressing the core issues.

#wrap-content(
  [
    #figure(
      image(
        "../lectures_source/figures/L01.3.Windows_failure.png",
        width: 100%,
      ),
      caption: [A Windows system failure],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:windowsfailure>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  Our ability to predict the future is notably poor, as illustrated by resources
  like #underline(text(fill: blue)[#link("https://paleofuture.com/")[Paleofuture]])
  and the
  #underline(text(fill: blue)[#link("https://elonmusk.today/")[Elon-O-Meter]]).
  These examples highlight the frequent inaccuracies in our forecasts. As shown
  in @fig:windowsfailure, even our
  current technological systems are prone to unexpected failures, underscoring
  the difficulty of making accurate predictions.
]

#wrap-content(
  [
    #figure(
      image("../lectures_source/figures/L01.3.Y2K.png", width: 100%),
      caption: [The Y2K bug],
      kind: "figure",
      supplement: [Fig.],
      placement: auto,
    ) <fig:y2k>
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 30%),
)[
  In my opinion, the urgency of AI alignment is comparable to the debate over
  what political system humanity will require when colonizing Mars. However,
  considering our current struggles with simpler tasks, such as managing airport
  terminals efficiently, this debate might seem premature. The challenges we
  face today, like those depicted in @fig:y2k, remind us of the complexities
  involved in technological advancements and the importance of addressing them
  thoughtfully.
]

// From: msml610/lectures_source/Lesson01.4-Brief_History_of_AI.smd:983 '* References'
// Slide: References
= References

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
