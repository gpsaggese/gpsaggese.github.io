---
title: "L01.2: AI and Machine Learning"
---

<!-- git_hash=2ed4dde-yd7 timestamp=20260828_134220 -->

<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides001.jpg){width=80%}
</center>
<center>

# 2 / 18: ML, AI, and Intelligence

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides002.jpg){width=80%}
</center>

- **Machine Learning** is a subset of Artificial Intelligence (AI)
  - Machine Learning (ML) is a part of AI, but it's not the whole picture. It's
    important to understand that ML is just one way to achieve AI. People often mix
    up ML with other terms like _deep learning_, _large-language models_, and
    _predictive analytics_. These are all related but distinct areas within the
    broader field of AI. ML focuses on creating systems that can learn from data and
    improve over time without being explicitly programmed

- **What is artificial intelligence?**
  - To grasp AI, we first need to understand what **human intelligence** means. AI
    aims to mimic or replicate aspects of human intelligence, so understanding the
    original concept is crucial

- **What is human intelligence?**
  - Humans are known as _"homo sapiens"_, which means "wise man," highlighting our
    intelligence as a defining trait. For centuries, humans have been trying to
    unravel the mystery of how we think and process information. The human brain,
    despite its small size, is capable of understanding complex concepts like the
    theory of relativity and quantum mechanics. This raises intriguing questions
    about how such a small organ can comprehend, predict, and even manipulate a world
    that is far more complex than itself. Understanding this is key to developing AI
    that can perform similar feats
<center>

# 3 / 18: Artificial Intelligence

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides003.jpg){width=80%}
</center>

- The term _"Artificial Intelligence"_ was coined in 1956, and is a field of computer
  science focused on creating systems capable of performing tasks that typically
  require human intelligence. The term was first introduced in 1956 during a
  conference at Dartmouth College, marking the beginning of AI as a distinct academic
  discipline

- Artificial Intelligence has **multiple goals**
  - **Understand human intelligence**: One of the primary goals of AI is to gain
    insights into how human intelligence works. By studying and replicating cognitive
    processes, researchers hope to better understand the complexities of the human
    mind
  - **Create intelligent entities**: Another goal is to develop machines or software
    that can perform tasks intelligently. This includes everything from simple
    automation to complex decision-making processes
  - _"What I cannot create, I do not understand"_ (Feynman, 1988): This quote by
    physicist Richard Feynman emphasizes the idea that true understanding comes only
    from the ability to recreate or simulate a process or phenomenon

- AI has several **important characteristics**
  - AI has the potential to be _applied across a wide range of fields_, from
    healthcare and finance to entertainment and transportation, making it a versatile
    tool for innovation
  - The transformative power of AI is often compared to major historical events like
    the Industrial Revolution, as it has the potential to _fundamentally change how
    we live and work_
  - AI technologies are already a significant economic force, driving growth and
    efficiency in various industries and contributing substantially to the global
    economy for trillions of dollars annually in revenue
  - AI still presents **many unresolved problems**. This sets it apart from
    disciplines like physics and math, where the major foundational concepts have
    already been established. In AI, we are still discovering the basic principles,
    which makes it an unusually open and fast-moving field
<center>

# 4 / 18: AI Formal Definition

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides004.jpg){width=80%}
</center>

- AI can be characterized along **two key axes**
  - **Thinking vs. Acting**: This axis distinguishes whether AI should focus on
    replicating human thought processes or on performing actions. _Thinking_ involves
    processes like reasoning and problem-solving, while _acting_ involves executing
    tasks or making decisions
  - **Human vs. Rational**: This axis differentiates between mimicking human behavior
    and achieving optimal, logical outcomes. _Human_ refers to AI systems that
    emulate human-like behavior, whereas _rational_ refers to systems that aim for
    the best possible performance based on logic and data

- This yields **four possible definitions of AI** as a machine that can:
  1. **Think humanly**: AI that replicates human thought processes, such as
     understanding language or emotions
  2. **Think rationally**: AI that uses logic and data to make decisions, aiming for
     the most effective outcomes
  3. **Act humanly**: AI that behaves like a human, such as in social interactions or
     physical tasks
  4. **Act rationally**: AI that performs actions based on logical reasoning and
     data, striving for optimal results

- Which definition best captures what AI should aim for?
  - This question invites us to consider the ultimate goal of AI development. Should
    AI prioritize human-like behavior, or should it focus on achieving the best
    possible outcomes through rational processes?

- Building machines that **act rationally** should be the ultimate goal of AI
  - The emphasis here is on creating AI systems that make decisions and perform
    actions based on logical reasoning and data analysis. This approach aims to
    maximize efficiency and effectiveness, potentially leading to better solutions
    and advancements in various fields
<center>

# 5 / 18: 1. AI as Thinking Humanly

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides005.jpg){width=80%}
</center>

- Let's focus on the definition "AI as Thinking Humanly"

- _Problem_: To build machines that think like humans, we need to first **determine
  how humans think**
  - This point highlights the challenge of creating AI that mimics human thought
    processes. Before we can program machines to think like us, we must understand
    the intricacies of human cognition. This involves studying psychology,
    neuroscience, and cognitive science to uncover how our minds work

- On the positive side, by attempting to replicate human thinking in machines, we can
  develop a detailed model of the human mind. This model can be expressed as a
  computer program, providing a structured way to understand and simulate human
  thought processes. It can lead to advancements in both AI and cognitive science

- There are several drawbacks:
  - One major drawback is that we still don't fully understand how the human mind
    operates. This lack of knowledge makes it difficult to create accurate AI models
    that truly reflect human thinking
  - Focusing on human-like thinking can be limiting because it centers AI development
    around human capabilities. This anthropocentric approach might overlook other
    forms of intelligence that could be more efficient or effective for certain
    tasks
<center>

# 6 / 18: 2. AI as Thinking Rationally

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides006.jpg){width=80%}
</center>

- How we can ensure that our thinking process is logical and leads to valid
  conclusions?
  - When we start with premises that are true, we should be able to use logical
    reasoning to arrive at conclusions that are also true. This is a fundamental
    aspect of rational thinking and is crucial for developing intelligent systems
    that can make decisions based on logic

- Logic is a branch of philosophy and mathematics that deals with the principles of
  valid reasoning. It involves creating formal systems that can express statements
  about objects and their relationships. By formalizing these statements, we can
  apply logical rules to deduce new information or verify the truth of certain
  propositions

- An application of logic is automatic theorem proving, which involves creating
  computer programs that can solve problems expressed in logical notation. These
  programs attempt to prove or disprove statements by applying logical rules
  However, a challenge with this technique is that if no solution exists, the program
  might run indefinitely, which is related to the halting problem. The halting
  problem is a well-known issue in computer science that deals with determining
  whether a program will eventually stop running or continue indefinitely
<center>

# 7 / 18: Thinking Rationally: Challenges

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides007.jpg){width=80%}
</center>
There are several drawbacks in the approach of creating AI as a system that can think
rationally

- Formalizing **informal knowledge** is difficult. For example, a simple action like
  a handshake involves many steps and conditions that need to be precisely defined
  The logical representation provided in the slide shows how detailed and intricate
  this process can be, requiring us to define each participant, their actions, and
  interactions. This complexity makes it challenging to capture the nuances of human
  knowledge in a formal system

- In many real-world situations, knowledge is not black and white but rather
  **probabilistic**. For instance, symptoms like fever, cough, and fatigue can be
  associated with multiple illnesses such as the flu or COVID-19. This uncertainty
  makes it difficult to apply strict logical reasoning, as we need to consider
  probabilities and make decisions based on incomplete information

- Scalability of knowledge systems is another challenge. As problems grow in size and
  complexity, it becomes increasingly difficult to solve them using purely rational
  methods. Heuristics, or rules of thumb, are often necessary to find practical
  solutions within a reasonable time frame. This is a limitation of thinking purely
  rationally, as it may not always be feasible for large-scale problems

- Rational thinking is just one aspect of intelligence. For an agent to be truly
  intelligent, it must interact with the world, learn from experiences, and adapt to
  new situations. This involves more than just logical reasoning; it requires
  understanding the environment and having a physical presence, or *embodiment*, to
  interact with it. This point emphasizes that intelligence is a multifaceted concept
  that goes beyond mere rationality
<center>

# 8 / 18: 3. AI as Acting Humanly

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides008.jpg){width=80%}
</center>

- An _agent_ in AI is an entity that perceives its environment and takes actions to
  achieve specific goals. This concept is central to AI as it involves creating
  systems that can make decisions and perform tasks autonomously

- The primary aim of AI in this context is to design agents that can mimic human
  behavior. This involves creating systems that can perform tasks in a way that is
  indistinguishable from humans

- The **Turing test** is a classic method for evaluating a machine's ability to
  exhibit intelligent behavior equivalent to, or indistinguishable from, that of a
  human. If a human evaluator cannot reliably distinguish between the machine and a
  human based on their responses, the machine is said to have passed the Turing test

- To pass the Turing test, an AI system needs several capabilities:
  1. **Natural language processing** allows the system to understand and generate
     human language, enabling effective communication
  2. The system must store and manage **knowledge and information** efficiently to
     use it in decision-making processes
  3. **Automated reasoning** involves using stored knowledge to solve problems and
     answer questions logically
  4. The system should be able to learn from data and identify patterns to improve
     its performance over time, which is the goal of **machine learning**
  5. **Computer vision and speech recognition** enable the system to perceive and
     interpret visual and auditory information from the environment
  6. **Robotics** needs to be solved to manipulate physical objects and move,
     allowing the AI system to interact with the physical world
<center>

# 9 / 18: Turing Test: Pros and Cons

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides009.jpg){width=80%}
</center>

- Let's talk about the pros and cons of the Turing Test

- On the positive side:
  - The Turing Test provides an _operational definition of intelligence_. This means
    it offers a practical way to measure if a machine can exhibit intelligent
    behavior similar to a human
  - It helps to _sidestep philosophical vagueness_. Questions like "What is
    consciousness?" or "Can a machine think?" are complex and often debated. The
    Turing Test simplifies this by focusing on whether a machine can mimic human
    conversation well enough to be indistinguishable from a human

- On the negative side
  - The test defines intelligence using **anthropomorphic criteria**, meaning it
    judges machines based on human-like behavior. This is limiting because there are
    many forms of intelligence that don't resemble human intelligence
  - Passing the Turing Test means a machine can **fool humans** into thinking it is
    human, which doesn't necessarily equate to true intelligence or understanding
  - **Example**: In aeronautical engineering, the goal is to focus on _wind tunnels
    and aerodynamics_ rather than trying to design machines that imitate birds
    Similarly, the Turing Test focuses on imitation rather than understanding the
    essence of intelligence
<center>

# 10 / 18: 4. AI as Acting Rationally

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides010.jpg){width=80%}
</center>

- The concept of _rational agents_ in AI refers to systems or entities that make
  decisions aimed at achieving the _best possible outcome based on the information
  they have_. This means they are designed to perform actions that are considered the
  "right thing" in a given situation, maximizing their performance measure

- For an agent to be considered as acting rationally, it should possess several key
  attributes:
  1. **Operate autonomously**: The agent should be capable of functioning
     independently without constant human intervention. This autonomy allows it to
     make decisions and take actions on its own
  2. **Perceive environment**: It must have the ability to sense and interpret its
     surroundings. This perception is crucial for understanding the context in which
     it operates and for making informed decisions
  3. **Persist over a prolonged time period**: The agent should be able to maintain
     its operations and continue functioning effectively over time, rather than being
     limited to short-term tasks
  4. **Adapt to change**: A rational agent should be flexible and capable of
     adjusting its behavior in response to changes in the environment or new
     information. This adaptability is essential for dealing with dynamic and
     unpredictable situations
  5. **Create and pursue goals**: The agent should be able to set objectives and take
     actions to achieve them. This goal-directed behavior is a fundamental aspect of
     acting rationally, as it drives the agent to optimize its actions towards
     desired outcomes
<center>

# 11 / 18: Acting Rationally as Ultimate Goal of AI

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides011.jpg){width=80%}
</center>

- The question is "which definition of AI best captures what we should build?" Should AI
  be designed to act or think like humans, or should it aim to act or think
  rationally? This distinction is crucial because it influences how AI systems are
  developed and evaluated

- **Acting is more fundamental than Thinking**
  - _Acting rationally_ encompasses more than just _thinking rationally_. This means
    that while thinking is important, the ultimate goal is to ensure that AI systems
    can make decisions and take actions that lead to the best outcomes. In real-world
    applications, the ability to act effectively is often more critical than the
    ability to think deeply

- Rationality can be defined using mathematical models and logic, making it a more
  objective standard for AI behavior. In contrast, human behavior is influenced by a
  variety of factors, including emotions and evolutionary pressures, which can lead
  to irrational decisions. By focusing on rationality, AI can be designed to
  consistently make optimal decisions

- The conclusion is that the ultimate goal for AI should be to develop agents that
  act rationally. This means creating systems that can evaluate situations, make
  decisions, and take actions that maximize desired outcomes, based on logical and
  objective criteria
<center>

# 12 / 18: Rationally Is Not Absolute

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides012.jpg){width=80%}
</center>

- Rationally is Not Absolute

- We said that AI aims to build agents that do the right thing, but what is the right
  thing?
  - The main goal of AI is to create systems that make decisions or take actions that
    are considered "right" or "rational."
  - However, defining what is "right" can be complex because it often depends on the
    situation and perspective
  - This raises questions about how to program AI to make decisions that align with
    human values and ethics

- Consider this scenario: you leave the house and a branch strikes you
  - In this scenario, you likely made a valid decision to leave the house, but an
    unforeseen event (the branch) occurred
  - _Did you act rationally?_ - Probably
  - This example illustrates that rational actions can still lead to unexpected
    outcomes

- Now consider this similar scenario: You cross the street and a car knocks you over
  - _Did you act rationally?_ - It depends, but probably not
  - Here, the rationality of your action depends on the context, such as whether you
    checked for traffic
  - This highlights that rationality is often judged based on the information
    available and the outcome

- Also morality and rationality are not necessarily aligned
  - For instance consider if a car should swerve and hit a pedestrian to avoid a
    frontal crash that would kill two people?
  - This presents a moral dilemma where the AI must choose between two negative
    outcomes
  - It underscores the difficulty in programming AI to handle ethical decisions, as
    different people might have different opinions on what the "right" action is
<center>

# 13 / 18: Problems of a Rational Agent

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides013.jpg){width=80%}
</center>

- A rational agent operates in environments that can be either deterministic or
  uncertain. In a deterministic environment, the agent can aim for the best possible
  outcome because the results of actions are predictable. However, in a probabilistic
  or uncertain environment, the agent must aim for the best _expected_ outcome, as
  the results are not guaranteed and involve some level of uncertainty

- **What does "best" mean?**
  - The classical answer is that "best" is determined by the _objective function_
    being used. This could be a cost function, which the agent tries to minimize, or
    a reward function, which the agent tries to maximize. Other examples include loss
    functions in machine learning or utility functions in economics
  - This classical answer is a simplification, though. In practice, deciding what the
    "right" objective function should be, and optimizing it, is itself a hard and
    often ambiguous problem

- **Omniscience vs no-regrets**
  - A rational agent makes decisions based on the information available to it at the
    time, rather than having perfect knowledge of all possible outcomes. This means
    the agent aims to make decisions that it won't regret later, even if it doesn't
    have all the information upfront

- **Sometimes no provably correct action exists**
  - In some situations, there is no action that can be guaranteed to be correct
    Despite this, the agent must still choose an action, often relying on the best
    available information or heuristics

- **Even with perfect information, rationality may not be feasible**
  - Having all the information doesn't always lead to rational decisions. The cost of
    acquiring complete data can be prohibitive, such as in fields like medicine where
    gathering all possible data is expensive
  - The computational power required can also be beyond current capabilities. For
    example, some problems require exploring a search tree with $10^{80}$ nodes,
    which is far more than any computer could ever traverse
  - There are also **real-time demands**. In high-frequency trading, for instance, a
    decision may need to be made within a single microsecond, leaving no room for
    lengthy computation

- A solution is "satisficing"
  - Instead of striving for the perfect solution, which may be unattainable due to
    constraints, agents often aim for a solution that is "good enough." This
    approach, known as satisficing, involves making decisions that are adequate given
    the computational and informational constraints
<center>

# 14 / 18: Limits of AI Compared to Human Intelligence

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides014.jpg){width=80%}
</center>

- **AI and ML differ from human intelligence**
  - Machine learning and artificial intelligence systems operate differently from
    human brains. For example, large language models process information in a way
    that doesn't mimic human learning. Humans learn through experiences and context,
    while machines rely on data patterns and algorithms, and each approach has its
    own limitations
  - Whether the brain uses *gradient descent*, the optimization method behind most ML
    training, is still an open and unclear question
  - Whether the brain uses *reinforcement learning* is more likely, at least in some
    sense, since humans clearly learn from rewards and consequences over time

- **Fragility to input variations**: ML models can be very sensitive to small changes
  in input data. For instance, adversarial attacks can trick a model into making
  errors by changing just one pixel in an image. Similarly, a model trained to play a
  video game might fail if the screen orientation changes slightly, whereas humans
  can adapt to such changes easily

- **Lack of transfer learning**: Unlike humans, ML systems struggle to apply what
  they've learned in one area to a different context without being retrained. Humans
  can often transfer skills and knowledge across different domains seamlessly

- **Massive data and compute requirements**: ML models need vast amounts of data and
  computational power to learn effectively. For example, while a teenager can learn
  to drive in a few hours, developing a self-driving car system requires billions of
  compute hours and extensive datasets

- **Poor common sense and reasoning**: ML models lack the innate world knowledge and
  intuitive logic that humans possess. They don't have built-in common sense or the
  ability to reason through problems in the way humans do
<center>

# 15 / 18: Limits of AI Compared to Human Intelligence

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides015.jpg){width=80%}
</center>

- **Opaque decision-making**: Machine learning models, especially complex ones like
  deep neural networks, often operate as "black boxes." This means that while they
  can make accurate predictions or decisions, the way they arrive at these
  conclusions is not easily understood by humans. This lack of transparency can be
  problematic, especially in critical areas like healthcare or finance, where
  understanding the reasoning behind a decision is crucial for trust and
  accountability. Without clear insight into how decisions are made, it becomes
  challenging to interpret results or hold systems accountable for errors

- **Dependence on narrow objectives**: Machine learning systems are designed to
  perform specific tasks very well, but they struggle when faced with broader, more
  ambiguous goals. For example, an algorithm designed to maximize user engagement on
  a social media platform might inadvertently promote sensational or harmful content
  because it is narrowly focused on increasing clicks or views, without understanding
  the broader implications of its actions. This highlights the importance of
  carefully defining objectives and considering potential unintended consequences

- **Susceptibility to bias and data quality**: Machine learning models learn from the
  data they are trained on. If this data contains biases, the models will likely
  replicate and even amplify these biases. This can lead to unfair or discriminatory
  outcomes, particularly if the data reflects historical inequalities or prejudices
  Ensuring high-quality, unbiased data is crucial for developing fair and accurate
  models

- **Lack of embodiment and physical interaction**: Unlike humans, machine learning
  systems do not (always) have a physical presence or sensory experiences. Human
  intelligence is deeply connected to our physical interactions with the world, which
  help us understand context and develop intuition. This lack of embodiment in ML
  systems means they may struggle with tasks that require a deep understanding of the
  physical world or human experiences, limiting their ability to fully replicate
  human-like intelligence
<center>

# 16 / 18: Machine Learning: Definitions

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides016.jpg){width=80%}
</center>

- After having talked about AI, let's focus on machine learning

- The first question is: how should we define machine learning?

- _"Machine learning is the field of study that gives computers the ability to learn
  without being explicitly programmed"_
  - This classic definition by Arthur Samuel highlights the core idea of machine
    learning: enabling computers to learn from data and experiences rather than
    following hard-coded instructions

- An alternative definition is: machine learning builds machines to do **useful
  things** without being **explicitly programmed**
  - This alternative definition emphasizes the practical aspect of machine learning
    It focuses on creating systems that can perform tasks autonomously
  - Example: A computer learns to play checkers by playing against itself, memorizing
    positions that lead to winning
  - This example illustrates how a machine can improve its performance through
    self-play, learning strategies that lead to success without direct human
    intervention

- _"A computer program is said to learn from experience $E$ with respect to some task
  $T$ and some performance measure $P$, if $P(T)$ improves with experience $E$"_
  (Mitchell, 1998)
  - This formal definition by Tom Mitchell provides a structured way to think about
    machine learning. It introduces the concepts of experience, tasks, and
    performance measures, which are crucial for evaluating learning systems

- Common applications of machine learning include:
  - **Computer vision**: Machines interpret and understand visual information from
    the world, like recognizing objects in images
  - **Speech recognition**: Systems convert spoken language into text, enabling
    voice-activated assistants and transcription services
  - **Natural language processing**: Machines understand and generate human language,
    powering applications like chatbots and translation services
<center>

# 17 / 18: The 3 Machine Learning Assumptions

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides017.jpg){width=80%}
</center>

- Solving a practical machine learning problem involves several steps:
  - **Gathering a dataset** that represents the problem you want to solve
  - **Building a statistical model** from the dataset algorithmically, using
    algorithms that learn from the data to make predictions or decisions without
    being explicitly programmed for the task
  - **Evaluating the model** to check how well it performs
  - **Deploying and monitoring the model** once it's in use
  - And other steps, depending on the project

- According to Abu-Mostafa (2012), most of these phases are "engineering": you know
  what needs to be done, and it's about doing it properly. Building the model,
  however, is the "research" part, since there are hypotheses that need to be
  verified before the model can be trusted

- The **three core assumptions** of machine learning are crucial for its success:
  - _A pattern exists_: This means there is some regularity or structure in the data
    that can be learned and used to make predictions
  - _Pattern cannot be precisely defined mathematically_: Often, the relationships in
    data are too complex and noisy to be captured by simple mathematical formulas,
    which is why machine learning is used
  - _Data is available_: you need access to sufficient and relevant data

- Which of the assumptions is **essential**?
  - _A pattern exists_: If this hypothesis is not true, machine learning will not
    find anything, and the conclusion is that there is nothing to learn from the
    data, which is "fine". So this assumption is not strictly necessary
  - _Pattern cannot be precisely defined_: Even when a pattern can be devised with
    math, we may still choose to use machine learning, for example for efficiency or
    scalability. This assumption is not strictly necessary either
  - _Data is available_: Without data, the model cannot be trained, and no learning
    or progress can occur. Data is the foundation of any machine learning project
    This is the most critical assumption
<center>

# 18 / 18: AI vs ML vs Deep-Learning

</center>
<center>
![](msml610/lectures_commentary/Lesson01.2-AI_and_Machine_Learning.jpg/slides018.jpg){width=80%}
</center>

- This slide is about the relationship between different terms often mixed
  incorrectly (by non-technical persons)
  - The diagram visually represents the relationship between these concepts, showing
    how each is a subset of the previous one, with AI being the broadest category and
    LLMs being a specific application within deep learning

- Artificial Intelligence is a broad field that involves creating machines or systems
  that can perform tasks that typically require human intelligence. These tasks
  include reasoning, learning, and decision-making. AI aims to make machines act in a
  way that seems intelligent and rational

- Machine Learning is a subset of AI, machine learning focuses on enabling machines
  to learn from data and improve their performance over time without being explicitly
  programmed for each task. This means that instead of following a set of pre-defined
  rules, the machine learns patterns from data and makes decisions based on that
  learning

- An example of AI system that doesn't use machine learning is a rule-based systems
  using rules handcrafted by domain experts. These systems, like IBM's Deep Blue,
  which played chess, rely on a set of predefined rules and logic to make decisions,
  rather than learning from data. This shows that AI systems can exist without using
  machine learning at all

- Deep Learning is a specialized area within machine learning that uses neural
  networks with many layers (hence "deep"). Deep learning is particularly powerful
  for tasks like image and speech recognition because it can automatically discover
  intricate patterns in large datasets

- Large Language Models are a type of deep learning model specifically designed to
  understand and generate human language. They are trained on vast amounts of text
  data and often use reinforcement learning from human feedback to improve their
  performance. LLMs are used in applications like chatbots and language translation

- Just as AI systems can exist without machine learning, deep learning systems can
  exist without being large language models. For example, a *convolutional neural
  network* used for computer vision or speech is a deep learning system, but it is
  not an LLM, since it is not trained on text and does not generate language
