// git_hash=418671a2a-nmw timestamp=20260907_200914
// FIXED_BY_CLAUDE_20260907_202440
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
  title: "L02.2: Machine Learning Paradigms",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L02.2: Machine Learning Paradigms")

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:7 '# Machine Learning Paradigms'
// Slide: Machine Learning Paradigms
#strong[Machine Learning Paradigms]

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:9 '## Major Paradigms'
// Slide: Major Paradigms
== Major Paradigms

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:11 '* Machine Learning Paradigms: A Taxonomy'
// Slide: Machine Learning Paradigms: A Taxonomy
#strong[Machine Learning Paradigms: A Taxonomy]

How do the major machine learning paradigms differ in the way they access data
and receive feedback? The answer turns on two axes: whether labeled outputs are
available, and whether the learner interacts with its environment sequentially
or processes a fixed dataset.

#strong[Supervised learning] sits at one end of the label-availability spectrum:
every training example comes paired with a target output, so the learner
receives explicit corrective feedback on each prediction. #strong[Unsupervised
  learning] occupies the opposite end, working with raw data and no labels at
all; its goal is to discover structure (clusters, latent factors, density
estimates) rather than to match a known answer. #strong[Semi-supervised
  learning] falls between the two, combining a small labeled set with a much
larger unlabeled corpus, a practical compromise when annotation is expensive but
raw data is plentiful. #strong[Self-supervised learning] sidesteps human labels
entirely by manufacturing its own supervision from the data's internal
structure: masking a word in a sentence and predicting it, or predicting the
next video frame from previous ones.

Along the second axis, #strong[reinforcement learning] is fundamentally
interactive and sequential. The agent takes an action, observes a reward signal
and a new state, then chooses again; feedback is sparse, delayed, and tied to
the consequences of its own choices rather than handed to it in a tidy label
column. #strong[Active learning] is also interactive but in a different way: the
learner queries an oracle (often a human annotator) for the label of a
strategically chosen example, aiming to reach high accuracy with as few queries
as possible.

@fig:machinelearningparadigmsataxonomy lays out these paradigms along the two
axes, showing how label availability and the sequential/interactive nature of
data access together organize the landscape of learning approaches. Recognizing
where a given problem falls in this taxonomy is the first step toward choosing
an appropriate algorithm.

// rendered_images:begin
// ```graphviz[width=80%]
// digraph LearningParadigms {
//   graph [rankdir=LR, splines=curved, bgcolor="transparent",
//          ranksep="1.0 equally", nodesep=0.24, pad=0.3, fontname="Helvetica"];
//   node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//          fontcolor="#26215C", color="#D8D6EE", penwidth=1.2,
//          height=0.46, margin="0.18,0.09"];
//   edge  [arrowhead=none, penwidth=1.4, color="#B9B6D6"];
// 
//   root [label="Learning Paradigms", shape=box, style="rounded,filled",
//         fillcolor="#26215C", fontcolor="white", fontsize=14, penwidth=0,
//         margin="0.26,0.16"];
// 
//   // Label availability - violet
//   label_avail [label="Label Availability", fillcolor="white", color="#7C74D6", fontcolor="#45296B", penwidth=1.6, fontsize=12];
//   supervised   [label=<<b>Supervised</b><br/><font point-size="9" color="#45296B">Labeled input-output pairs</font>>, fillcolor="#EFEDFC", color="#CBC7F0", fontcolor="#45296B"];
//   unsupervised [label=<<b>Unsupervised</b><br/><font point-size="9" color="#45296B">Unlabeled data, discover structure</font>>, fillcolor="#EFEDFC", color="#CBC7F0", fontcolor="#45296B"];
//   selfsup      [label=<<b>Self-supervised</b><br/><font point-size="9" color="#45296B">Labels derived from data</font>>, fillcolor="#EFEDFC", color="#CBC7F0", fontcolor="#45296B"];
//   semisup      [label=<<b>Semi-supervised</b><br/><font point-size="9" color="#45296B">Mixed labeled + unlabeled</font>>, fillcolor="#EFEDFC", color="#CBC7F0", fontcolor="#45296B"];
// 
//   // Interactive - blue
//   interactive  [label="Interactive/Sequential", fillcolor="white", color="#3E86C8", fontcolor="#1F4E79", penwidth=1.6, fontsize=12];
//   rl           [label=<<b>Reinforcement</b><br/><font point-size="9" color="#1F4E79">Learn from delayed rewards</font>>, fillcolor="#E8F1FB", color="#BFD8F1", fontcolor="#1F4E79"];
//   active       [label=<<b>Active Learning</b><br/><font point-size="9" color="#1F4E79">Request labels on demand</font>>, fillcolor="#E8F1FB", color="#BFD8F1", fontcolor="#1F4E79"];
//   online       [label=<<b>Online Learning</b><br/><font point-size="9" color="#1F4E79">Sequential data, incremental update</font>>, fillcolor="#E8F1FB", color="#BFD8F1", fontcolor="#1F4E79"];
// 
//   root -> label_avail [color="#7C74D6", penwidth=2.0];
//   root -> interactive [color="#3E86C8", penwidth=2.0];
// 
//   label_avail -> {supervised unsupervised selfsup semisup} [color="#A9A3E6"];
//   interactive -> {rl active online}                        [color="#8FB6DE"];
// 
//   { rank=same; supervised; unsupervised; selfsup; semisup; rl; active; online; }
// }
// ```
// label=fig:machinelearningparadigmsataxonomy caption=Diagram relating Learning
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.2-ML_Paradigms.typ.figs/Lesson02.2-ML_Paradigms.1.png",
    width: 80%,
  ),
  caption: [Diagram relating learning paradigms, label availability and interactive/sequential.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:machinelearningparadigmsataxonomy>
// render_images:end

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:53 '* Machine Learning Paradigms: Examples (1/3)'
// Slide: Machine Learning Paradigms: Examples (1/3)
#strong[Machine Learning Paradigms: Examples (1/3)]

How do you set up a machine learning problem? The answer depends on what kind of
data you have and what kind of feedback the learning system receives. There are
five major paradigms, each suited to a different setting.

#strong[Supervised learning] trains a model on labeled data: each input comes
paired with the correct output, and the model learns to predict that output for
new, unseen inputs. Image classification is a canonical example, where
architectures like ResNet learn to map pixel arrays to category labels using the
millions of labeled photographs in ImageNet.

#strong[Unsupervised learning] works without any labels at all. Instead, the
goal is to discover hidden patterns or structure lurking in the data itself.
K-means clustering for customer segmentation is a typical application: the
algorithm groups customers by purchasing behavior without anyone telling it what
the groups should be.

#strong[Reinforcement learning] takes a fundamentally different approach. Rather
than learning from a fixed dataset, the agent learns through interaction with an
environment, taking actions and receiving rewards or punishments that shape its
future behavior. Deep Q-Learning, which learned to play Atari games at
superhuman levels #cite("mnih2015dqn"), demonstrated how powerful this
trial-and-error paradigm can be when combined with deep neural networks.

#strong[Self-supervised learning] bridges the gap between supervised and
unsupervised approaches by generating pseudo-labels directly from unlabeled
data. The model creates its own supervision signal, typically by hiding part of
the input and predicting the missing piece. BERT's masked language modeling
objective #cite("devlin2019bert") is a prominent example: the model masks random
words in a sentence and learns to reconstruct them, building rich language
representations in the process without a single human-provided label.

#strong[Semi-supervised learning] combines a small set of labeled examples with
a much larger pool of unlabeled data to improve performance beyond what either
source alone could provide. Named entity recognition illustrates this well: a
handful of sentences annotated with entity tags (person, organization, location)
can be paired with vast quantities of raw text documents, letting the model
leverage the structure of natural language to generalize far beyond the limited
annotations.

Each paradigm makes different assumptions about the available supervision
signal, and choosing the right one is often the first and most consequential
decision in framing a machine learning problem.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:74 '* Machine Learning Paradigms: Examples (2/3)'
// Slide: Machine Learning Paradigms: Examples (2/3)
#strong[Machine Learning Paradigms: Examples (2/3)]

#strong[Online learning] trains a model incrementally from a stream of data
arriving in real time, rather than requiring the entire dataset up front. A
practical example is online logistic regression for click-through rate
prediction, where the model updates its weights after each user interaction
rather than retraining on a stored batch.

#strong[Multi-task learning] trains a single model to perform multiple related
tasks simultaneously, exploiting shared structure across those tasks. For
instance, a model might learn both sentiment analysis and question answering at
the same time, allowing representations useful for one task to benefit the
other.

#strong[Meta-learning], often described as #emph[learning to learn], equips a
model to adapt quickly to new tasks by leveraging prior experience across many
earlier tasks. A model trained this way can be fine-tuned on a new task using
just a few gradient steps #cite("finn2017maml"), dramatically reducing the data
and compute needed for each new problem.

#strong[Zero-shot and few-shot learning] push generalization further: the model
handles new tasks with no labeled examples at all (zero-shot) or only a handful
(few-shot). Large language models such as GPT-4 demonstrate this by solving
novel tasks purely through zero-shot prompting #cite("openai2023gpt4"), relying
on broad pretraining rather than task-specific supervision.

#strong[Active learning] flips the usual labeling workflow. Instead of passively
receiving labeled data, the model itself selects the most informative samples to
be labeled by an oracle, typically a human annotator #cite("settles2009survey").
A common strategy is to pick samples where the model is least confident,
concentrating labeling effort where it will reduce uncertainty the most.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:99 '* Machine Learning Paradigms: Examples (3/3)'
// Slide: Machine Learning Paradigms: Examples (3/3)
#strong[Machine Learning Paradigms: Examples (3/3)]

#strong[Federated learning] trains models across decentralized devices without
sharing raw data #cite("mcmahan2017federated"). Each participant (a phone, a
hospital, a bank) keeps its data local and sends only model updates to a central
server, which aggregates them into a single improved model. This preserves
privacy by design: the raw records never leave their origin. A typical
application is fraud detection or credit scoring across multiple banks, where
pooling the underlying transaction data would violate regulatory constraints,
yet each institution benefits from patterns visible only in the combined
population.

#strong[Evolutionary learning] optimizes model structures or parameters through
algorithms inspired by natural selection and genetics. A population of candidate
solutions is maintained; each generation, the fittest individuals are selected,
recombined, and mutated to produce offspring that (on average) perform better.
Because the process relies on fitness evaluation rather than gradient
computation, it is #emph[gradient-free], making it applicable to discrete,
non-differentiable, or highly multimodal search spaces where backpropagation
cannot reach. Genetic algorithms are the most familiar instance, but the family
also includes evolution strategies, genetic programming, and neuroevolution.

#strong[Curriculum learning] structures the training process so that a model
encounters easier examples or tasks first, with difficulty increasing gradually
over time #cite("bengio2009curriculum"). The intuition mirrors human education:
mastering arithmetic before calculus builds representations that transfer
upward. In practice, a curriculum can be defined by hand (sorting training
samples by a known difficulty score) or learned automatically (letting the
training loop decide which examples to present next). Robotic control is a
natural fit: a simulated robot can first learn to balance, then walk, then
navigate obstacles, with each stage providing a foundation for the next.

#strong[Multi-agent learning] places multiple agents in a shared environment
where they learn simultaneously, often under game-theoretic dynamics such as
competition, cooperation, or a mix of both. Because each agent's optimal policy
depends on what the others are doing, the learning problem is fundamentally
non-stationary: the "environment" shifts as every participant adapts. DeepMind's
AlphaStar system #cite("vinyals2019alphastar") illustrates the approach at
scale: a league of agents trained against one another in StarCraft II, producing
strategies that reached Grandmaster-level play against human professionals. The
competitive pressure drove the discovery of diverse tactics that no single-agent
training regime would have uncovered.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:122 '## The Three Core Paradigms'
// Slide: The Three Core Paradigms
== The Three Core Paradigms

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:124 '* Supervised Learning'
// Slide: Supervised Learning
#strong[Supervised Learning]

#strong[Supervised learning] learns a function $f: X arrow.r Y$ that maps inputs
to correct outputs #cite("mitchell1997machinelearning"). The training set
consists of example pairs $(bold(x), y)$, where each input $bold(x)$ is paired
with the correct output $y$. Because every training example carries a label
$y_i$, supervised learning requires labeled data, and performance is measured by
the error the learned function makes on a separate test set that was held out
during training.

Two main task types fall under this umbrella. #emph[Classification] produces a
discrete label: an email filter that outputs "Spam" or "Not Spam," a digit
recognizer that returns one of the digits 0 through 9, or a sentiment analyzer
that labels text as positive, negative, or neutral. #emph[Regression], by
contrast, produces a continuous value: predicting house prices from features
like size and location, forecasting oil demand, or estimating future stock
prices. The choice between classification and regression is determined by the
nature of the output variable, not by the algorithm itself; many model families
(linear models, decision trees, neural networks) can be adapted to either
setting.

Common supervised learning algorithms include linear regression, decision trees,
and neural networks, among many others. Each makes different assumptions about
the shape of $f$ and trades off interpretability against expressive power, a
theme that recurs throughout the course.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:145 '* Unsupervised Learning'
// Slide: Unsupervised Learning
#strong[Unsupervised Learning]

#strong[Unsupervised learning] learns from data without labeled outputs. Rather
than receiving explicit feedback or correct answers, the algorithm's goal is to
discover patterns, groupings, or structure hidden within the data itself.
Because there is no ground-truth label to compare against, evaluation tends to
be more qualitative than in supervised settings: a practitioner judges whether
the discovered structure is meaningful, useful, or interpretable for the task at
hand.

Several families of techniques fall under this umbrella:

- #emph[Clustering] groups similar examples together. A retailer might segment
  its customers into behaviorally distinct groups to tailor marketing campaigns,
  or a news aggregator might group articles by topic without anyone having
  pre-assigned topic labels.
- #emph[Dimensionality reduction] compresses data from many variables down to
  fewer ones while preserving as much structure as possible. A common
  application is visualizing high-dimensional datasets in two dimensions using
  methods such as PCA, making it easier to spot clusters or outliers by eye.
- #emph[Density estimation] fits a probability distribution to the observed
  data. One practical use is anomaly detection: after learning what "normal"
  server-log traffic looks like, any observation that falls in a low-density
  region of the estimated distribution can be flagged as suspicious.
- #emph[Association rule learning] uncovers interesting relationships among
  variables. The classic example is market-basket analysis, where the algorithm
  discovers rules like "customers who buy product X also tend to buy product Y,"
  enabling cross-selling and store-layout decisions.

Representative algorithms span all of these families. K-means #cite(
  "macqueen1967kmeans",
) is perhaps the most widely known clustering method, partitioning data into $k$
groups by iteratively assigning points to their nearest centroid. PCA (Principal
Component Analysis) is the workhorse of linear dimensionality reduction,
projecting data onto the directions of greatest variance. Autoencoders, a
neural-network approach, learn a compressed latent representation by training
the network to reconstruct its own input through a narrow bottleneck layer, and
they can serve both dimensionality reduction and density estimation purposes
depending on the architecture.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:167 '* Reinforcement Learning'
// Slide: Reinforcement Learning
#strong[Reinforcement Learning]

#strong[Reinforcement learning] (RL) is a paradigm in which an agent learns by
interacting with an environment to maximize cumulative reward #cite(
  "suttonbarto2018rlbook",
). Rather than learning from a fixed dataset of labeled examples, the agent
discovers which actions yield the best outcomes through trial and error.
Formally, the goal is to learn a policy $pi(s) arrow.r a$ that maximizes
expected reward over time. A central tension in RL is the tradeoff between
#emph[exploration], trying new actions to discover potentially better
strategies, and #emph[exploitation], relying on actions already known to produce
good results. Balancing these two drives is essential: too much exploration
wastes time on poor choices, while too much exploitation risks missing superior
strategies entirely.

One of the practical difficulties of RL is that reward signals can be sparse,
delayed, or hard to specify outside tightly controlled settings such as games
and simulators. When an agent receives a reward only at the end of a long
sequence of decisions, figuring out which earlier actions actually contributed
to success (the #emph[credit assignment] problem) becomes considerably harder.
RL tasks also often involve physical simulation or real-world interaction, which
adds cost and risk to every exploratory step the agent takes.

The core components of any RL system are straightforward. The #emph[agent] is
the learner and decision maker. The #emph[environment] is everything the agent
interacts with, including any dynamics, physics, or rules that govern how the
world responds to the agent's choices. At each time step the environment is in
some #emph[state] $s$, the agent selects an #emph[action] $a$, and the
environment returns a #emph[reward] $r$ along with the next state.
@fig:reinforcementlearning illustrates this interaction loop: the agent observes
the current state, chooses an action, and receives both a reward and a new state
from the environment, then repeats.

// rendered_images:begin
// ```graphviz
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//     // Node styles
//     Agent      [label="Agent", shape=box, fillcolor="#F4A6A6"];
//     Env        [label="Environment", shape=box, fillcolor="#B2E2B2"];
// 
//     // Force ranks
//     //{ rank=same; Agent; Env; }
// 
//     // Edges
//     Agent -> Env [label="  Action", fontcolor=black, labeldistance=2.0];
//     Env -> Agent [label="  State", fontcolor=black, labeldistance=2.0];
//     Env -> Agent [label="  Reward", fontcolor=black, labeldistance=2.0];
// }
// ```
// label=fig:reinforcementlearning caption=Diagram relating Agent, Environment,
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.2-ML_Paradigms.typ.figs/Lesson02.2-ML_Paradigms.2.png",
    width: 70%,
  ),
  caption: [Diagram relating agent, environment and reward signals in the RL loop.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:reinforcementlearning>
// render_images:end

Among the best-known RL algorithms are #emph[Q-learning], which maintains a
table or function approximator estimating the expected future reward for each
state-action pair, and #emph[policy gradient methods], which directly optimize
the policy by adjusting its parameters in the direction that increases expected
reward. These two families represent the main algorithmic divide in RL:
value-based methods that learn what states and actions are worth, versus
policy-based methods that learn the mapping from states to actions directly.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:219 '* Reinforcement Learning: Examples'
// Slide: Reinforcement Learning: Examples
#strong[Reinforcement Learning: Examples]

Reinforcement learning finds natural application in any domain where an agent
must make a sequence of decisions and can learn from the outcomes of those
decisions over time.

#strong[Game playing] stands as one of the most visible success stories: agents
learn strategies through trial and error, with AlphaGo's mastery of Go serving
as a landmark demonstration that RL can surpass human expertise even in games
with enormous state spaces #cite("silver2016alphago"). #strong[Robotics] relies
on RL to learn control policies for movement and manipulation, allowing robots
to acquire motor skills that would be extremely difficult to program by hand.
#strong[Autonomous driving] applies similar principles to learn safe and
efficient driving behaviors, where the agent must continuously balance speed,
safety, and traffic rules in a dynamic environment.

Beyond physical control, RL excels at #strong[resource management] problems that
require optimizing the allocation of limited resources over time. Data center
cooling systems, for instance, can learn energy-efficient temperature policies,
and CPU job schedulers can learn to balance throughput against latency without
explicit hand-tuned heuristics. #strong[Personalized recommendations] represent
another natural fit: a recommendation system adapts its suggestions based on
ongoing user interaction, treating each click or skip as a reward signal that
shapes future rankings (a newsfeed adjusting its story selection in response to
reading patterns is a canonical example). Finally, #strong[healthcare]
applications use RL to optimize treatment plans over time, sequencing
medications or therapies in a way that accounts for a patient's evolving
condition rather than following a single static protocol.

The common thread across all these domains is that decisions are sequential,
feedback is delayed or partial, and the environment shifts in response to the
agent's own actions: precisely the setting where reinforcement learning offers
the most leverage over traditional supervised approaches.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:232 '## Machine Learning in Practice'
// Slide: Machine Learning in Practice
== Machine Learning in Practice

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:234 '* Machine Learning Flow (1/2)'
// Slide: Machine Learning Flow (1/2)
#strong[Machine Learning Flow (1/2)]

Every machine learning project follows a common pipeline whose stages build on
one another. The process begins with a #strong[question]: a concrete problem
statement such as "how can we predict house prices?" That question determines
what #emph[input data] you need; for the house-price example, this would be
historical records of house sales including sale prices and property attributes.
From the raw data you extract #emph[features], the individual measurable
properties the model will learn from: number of bedrooms, neighborhood location,
square footage, and so on. Next you choose a #emph[model], the mathematical
structure that will map features to predictions (a linear regression, a decision
tree, or something more complex). Every model comes with #emph[parameters] that
control how it learns or how complex it becomes: the learning rate that governs
gradient-descent step sizes, or the number of trees in a random forest. Finally,
#emph[evaluation] closes the loop: you measure the trained model's quality with
metrics such as accuracy, precision, and recall, and those results feed back
into refining earlier stages (collecting better data, engineering new features,
or tuning parameters).

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:244 '* Machine Learning Flow (2/2)'
// Slide: Machine Learning Flow (2/2)
#strong[Machine Learning Flow (2/2)]

// rendered_images:begin
// ```graphviz
// digraph BayesianFlow {
//     rankdir=LR;
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
//     // Node styles
//     "Question" [fillcolor="#F4A6A6"];
//     "Input data" [fillcolor="#FFD1A6"];
//     "Features" [fillcolor="#B2E2B2"];
//     "Model" [fillcolor="#A0D6D1"];
//     "Parameters" [fillcolor="#A6E7F4"];
//     "Evaluation" [fillcolor="#A6C8F4"];
//     // Force ranks
//     // Edges
//     "Question" -> "Input data";
//     "Input data" -> "Features";
//     "Features" -> "Model";
//     "Model" -> "Parameters";
//     "Parameters" -> "Evaluation";
// }
// ```
// label=fig:machinelearningflow22 caption=Diagram illustrating Machine Learning
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.2-ML_Paradigms.typ.figs/Lesson02.2-ML_Paradigms.3.png",
    width: 70%,
  ),
  caption: [Diagram illustrating the machine learning workflow: from question to evaluation.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:machinelearningflow22>
// render_images:end

Not all phases in the machine learning workflow carry equal weight. A useful
rule of thumb ranks them as: question, then data, then features, then model. The
clarity of the #emph[question] you ask has the single largest impact on whether
a project succeeds or fails; a vague or poorly scoped question can doom even the
most sophisticated pipeline. Next comes the quality and relevance of the
#emph[data]: no algorithm can extract signal that the data never contained.
Proper #emph[feature] selection simplifies the downstream model and often
improves accuracy more than swapping one learner for another. The #emph[model]
itself, contrary to popular belief, is frequently the least decisive factor. As
@fig:machinelearningflow22 illustrates, the earlier stages of the flow constrain
everything that follows, so investing effort upstream pays disproportionate
dividends compared to tuning the learning algorithm at the end.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:277 '* Framing the Question'
// Slide: Framing the Question
#strong[Framing the Question]

Formulating the question is the most important part of any machine learning
problem. A misunderstanding at this stage cascades into everything that follows:
solving the wrong problem, collecting the wrong data, and producing results that
serve no one. As Einstein reportedly put it, "If I were given one hour to save
the planet, I would spend 59 minutes defining the problem and one minute
resolving it." Whether or not the attribution is genuine, the principle holds:
precision in problem definition determines whether the rest of the work has any
value.

Making the question concrete and precise means aligning it with the actual
business or research objective, then defining the problem in terms specific
enough to guide data collection and model selection. Consider the difference
between a vague question and a well-formed one:

- #emph[Bad]: "How can we improve sales?"
- #emph[Good]: "What factors most significantly impact sales of product X in
  region Y during season Z?"

The first version gives no guidance on what data to gather, what outcome
variable to model, or what "improve" even means. The second version pins down
the product, the geographic scope, and the temporal window, which together
determine the dataset, the target variable, and the evaluation criteria. A
well-posed question is one where you can immediately see what a correct answer
would look like and how you would know if you had found it.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:296 '* Input Data: Fit, Quality, and Scale'
// Slide: Input Data: Fit, Quality, and Scale
#strong[Input Data: Fit, Quality, and Scale]

The data used to build a predictive model must be specific to the prediction
goal. If the task is to predict movie ratings, for instance, the training
examples should be known ratings drawn from the same population whose unseen
ratings the model will eventually estimate. More precisely, the training set
should approximate the test set, which in turn should approximate the data the
model will encounter in production. Any mismatch between these distributions
undermines the validity of the predictions.

The relationship between the available data and the ultimate prediction goal is
not always direct. A business may care about future prices, yet the most
effective modeling strategy might be to predict supply and demand separately and
derive prices from those forecasts. Recognizing these indirect links, and
choosing the right quantity to model, is itself a design decision that shapes
the entire pipeline.

Data quality matters as much as data relevance. Poor-quality inputs, whether
noisy, biased, or incomplete, propagate through every downstream step and
produce unreliable outputs: #emph["garbage in, garbage out."] Equally important
is the discipline to recognize when the data at hand is simply insufficient to
answer the question being asked. As John Tukey cautioned, #emph["desire for an
  answer does not ensure that a reasonable answer can be extracted from the
  given body of data."] Pressing forward with an inadequate dataset does not
yield a weaker answer; it yields a misleading one.

A recurring empirical finding is that #strong[more data often matters more than
  a better model]. The gap between a well-tuned generic algorithm and the best
hand-crafted specialist model is frequently small compared to the gain from
simply adding more, or higher-quality, training examples. The sentiment is
captured by the observation that #emph["it's not who has the best algorithm that
  wins; it's who has the most data"] #cite("banko2001data"). A famous (and only
half-joking) remark from an IBM speech-recognition researcher makes the same
point from the opposite direction: #emph["every time I fire a linguist, the
  performance of the speech recognizer goes up."] The lesson is not that
modeling skill is irrelevant, but that in many practical settings, investing in
data collection and curation yields higher returns than investing in algorithmic
sophistication alone.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:320 '* What Makes a Good Feature'
// Slide: What Makes a Good Feature
#strong[What Makes a Good Feature]

#strong[Features] are higher-level, more compact representations of raw inputs
that capture the information most relevant to a learning task. Rather than
feeding a model every pixel of a scanned digit image, for instance, you might
extract just two numbers: the overall intensity and the vertical symmetry of the
image. Those two quantities compress thousands of pixel values into a pair of
informative signals that a simple classifier can work with directly.

Good features share three characteristics. First, they #emph[retain the
  information that matters] for distinguishing between classes or predicting a
target. Second, they #emph[compress the data], replacing a high-dimensional
input with a much smaller vector and thereby reducing the computational and
statistical burden on the learner. Third, they are #emph[often designed with
  expert knowledge]: a radiologist knows which texture statistics matter in a
chest X-ray, and a linguist knows which syntactic patterns signal sentiment in
text.

Building features by hand or by automated search is, however, a process prone to
several common mistakes:

1. #emph[Automating feature selection without guarding against overfitting.] A
  search over a large feature space can find combinations that fit the training
  set perfectly yet capture nothing stable about the underlying phenomenon. The
  resulting model becomes a black box whose predictions are accurate on
  historical data but can stop working at any time. Google Flu Trends is a
  cautionary example: the system selected query terms that correlated with flu
  rates during one period, but the link between those features and the model's
  predictions was never made transparent, and performance degraded sharply when
  search behavior shifted.

2. #emph[Ignoring data-specific quirks.] Every dataset has idiosyncrasies:
  measurement artifacts, coding conventions, or outliers that look like signal
  but are really noise. Mislabeling an outlier as a legitimate extreme, for
  example, can bias a feature's scale and skew everything downstream.

3. #emph[Unnecessarily discarding information.] Aggressive feature selection or
  overly coarse binning can throw away structure that a model could have
  exploited. Compression is valuable, but it should be guided by relevance to
  the task, not by convenience. Whenever there is doubt about whether a piece of
  information matters, it is safer to keep it and let the learning algorithm
  decide than to discard it prematurely.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:338 '* What Makes a Good Model'
// Slide: What Makes a Good Model
#strong[What Makes a Good Model]

A good predictive model balances several desirable properties that often pull
against one another, as @fig:whatmakesagoodmodel illustrates. Understanding
these tradeoffs is essential for choosing the right model for a given
application.

// rendered_images:begin
// ```graphviz
// digraph ModelTradeoffs {
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
//           margin="0.18,0.10",
//           height=0.45];
// 
//     edge [style=dashed,
//           color="#B23A48",
//           penwidth=1.3,
//           arrowhead=none,
//           fontname="Helvetica",
//           fontsize=9,
//           fontcolor="#6B1F1F"];
// 
//     Accurate      [label="Accurate", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79", penwidth=2.4];
//     Interpretable [label="Interpretable", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     Simple        [label="Simple", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     Fast          [label="Fast", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     Scalable      [label="Scalable", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
// 
//     { rank=same; Interpretable; Simple; Fast; Scalable; }
// 
//     Accurate -> Interpretable [label="  Trade-off  "];
//     Accurate -> Simple [label="  Overfitting risk  "];
//     Accurate -> Fast [label="  Compute cost  "];
//     Accurate -> Scalable [label="  Resources  "];
// }
// ```
// label=fig:whatmakesagoodmodel caption=Diagram relating Accurate, Interpretable,
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.2-ML_Paradigms.typ.figs/Lesson02.2-ML_Paradigms.4.png",
    width: 70%,
  ),
  caption: [Tradeoffs between accuracy, interpretability, simplicity, speed and scalability.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:whatmakesagoodmodel>
// render_images:end

#strong[Simplicity] makes a model easier to implement, maintain, and debug.
Simpler models also carry a lower risk of overfitting, since they have fewer
parameters that can latch onto noise in the training data rather than genuine
patterns.

#strong[Accuracy] is typically what practitioners optimize first, yet pushing
accuracy higher often comes at the expense of every other property. A highly
accurate ensemble of dozens of models may be too slow to deploy, too complex to
explain, and too brittle to maintain.

#strong[Interpretability] lets users understand and trust the decisions a model
produces. Decision trees, for instance, excel here because they output an
explicit chain of reasoning: each prediction can be traced back through a
sequence of human-readable conditions. In domains such as healthcare or criminal
justice, interpretability is not optional; stakeholders need to know #emph[why]
a model reached a particular conclusion.

#strong[Speed] matters both at training time and at inference time. Real-time
applications such as fraud detection or autonomous driving require predictions
within milliseconds, ruling out models whose inference cost scales poorly with
input size.

#strong[Scalability] refers to how well a model handles growing datasets and
user bases. A celebrated cautionary tale comes from the Netflix Prize: the
winning algorithm achieved the best accuracy on the competition's benchmark, yet
Netflix never deployed it in production because it was not scalable enough to
serve its millions of users at acceptable latency. Scalability thus acts as a
hard gate; a model that cannot scale is a model that cannot ship, regardless of
how accurate it is on a static leaderboard.

In practice, model selection is an exercise in navigating these tensions.
Improving one property almost always degrades another, so the "best" model is
always relative to the constraints of the problem at hand.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:407 '## Pipeline Organization'
// Slide: Pipeline Organization
== Pipeline Organization

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:409 '* How Are Machine Learning Systems Organized?'
// Slide: How Are Machine Learning Systems Organized?
#strong[How Are Machine Learning Systems Organized?]

Machine learning systems are organized as a #strong[pipeline]: a sequence of
stages where the problem is broken down into sub-problems, each sub-problem is
solved in turn, and the individual solutions are combined to address the
original problem.

A critical property of pipelines is that overall system performance $p_"system"$
is not simply a weighted sum of the per-stage performances $p_i$. Instead,
#emph[stage errors compound]: a mistake early in the pipeline propagates through
every subsequent stage, degrading the final output far more than the magnitude
of the original error might suggest. This compounding effect means that
improving a late stage may yield little benefit if an earlier stage is already
corrupting its input.

#strong[Ceiling analysis] provides a principled way to measure each stage's
marginal contribution to system performance. The idea is to replace one stage at
a time with a perfect "oracle" and observe how much the end-to-end metric
improves. Formally, the relationship is approximated as

$ Delta p_"system" approx alpha_i dot.op Delta p_i $

where $Delta p_i$ is the improvement in the performance of stage $i$ alone (for
instance, by substituting an oracle for that stage) and $alpha_i$ is the
resulting improvement in overall system performance. Stages with large $alpha_i$
are the bottlenecks worth investing effort in, while stages whose oracle
replacement barely moves the system metric can be deprioritized.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:431 '* Example of Photo OCR System'
// Slide: Example of Photo OCR System
#strong[Example of Photo OCR System]

The goal of #strong[optical character recognition] (OCR) is to build systems
that can read text appearing in a photograph or scanned document. This task is
more challenging than it might first seem: text in natural images varies in
font, size, orientation, lighting, and background clutter, so a robust OCR
pipeline must handle each of these sources of variability systematically.

A typical ML pipeline for OCR breaks the problem into four sequential stages.
First, #emph[text detection] locates regions of the image that contain text.
Second, #emph[character segmentation] splits each detected text region into
individual letter boxes (for instance, the word "hello" becomes five separate
bounding boxes: `h`, `e`, `l`, `l`, `o`). Third, #emph[character classification]
identifies each segmented box as a specific character. Finally, #emph[spelling
  correction] uses linguistic context to fix recognition errors: if the
classifier output `hell0` (with a zero instead of the letter "o"), the
correction step maps it to the intended word "hello." Each stage feeds its
output to the next, so errors can compound; building an accurate classifier at
every stage is therefore critical.

// rendered_images:begin
// ```graphviz
// digraph OCRPipeline {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.35;
//     ranksep=0.5;
//     rankdir=LR;
// 
//     node [shape=box,
//           style="rounded,filled",
//           penwidth=1.8,
//           fontname="Helvetica",
//           fontsize=11,
//           margin="0.18,0.10",
//           height=0.45];
// 
//     edge [color="#A3B1C0",
//           penwidth=1.3,
//           arrowhead=vee,
//           arrowsize=0.75,
//           fontname="Helvetica",
//           fontsize=9,
//           fontcolor="#7B8794"];
// 
//     Image    [label="Input\nimage", shape=ellipse, fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Detect   [label="Text\ndetection", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     Segment  [label="Character\nsegmentation", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     Classify [label="Character\nclassification", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     Spell    [label="Spelling\ncorrection", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     Text     [label="Output\ntext", shape=ellipse, fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
// 
//     Image -> Detect;
//     Detect -> Segment [label="  Boxes  "];
//     Segment -> Classify [label="  h e l l o  "];
//     Classify -> Spell [label="  \"hell0\"  "];
//     Spell -> Text [label="  \"hello\"  "];
// }
// ```
// label=fig:exampleofphotoocrsystem caption=Diagram relating Input image, Text
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.2-ML_Paradigms.typ.figs/Lesson02.2-ML_Paradigms.5.png",
    width: 70%,
  ),
  caption: [Four-stage OCR pipeline: text detection, character segmentation, classification and spelling correction.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:exampleofphotoocrsystem>
// render_images:end

@fig:exampleofphotoocrsystem illustrates how these four stages connect, showing
the flow from a raw input image through detection, segmentation, and
classification to a final text string.

Because the location and size of text in a natural image are unknown in advance,
a #strong[sliding window approach] is used to search for text systematically.
For the text detection stage, the idea is to train a binary classifier that
distinguishes windows containing letters from windows containing only
background. This is a simpler learning problem than recognizing which letter a
window contains; the classifier only needs to say "text" or "not text." At
inference time, the trained classifier is swept across the image in both the
horizontal and vertical directions, evaluated at multiple window scales to
handle text of different sizes. Evaluating the classifier on a single window is
computationally cheap, so this exhaustive scan remains practical even for large
images.

The classifier's output probabilities are assembled into a #emph[text likelihood
  map], essentially a heatmap where bright regions indicate a high probability
of text. Connected high-probability regions are then enclosed in bounding boxes.
Boxes whose aspect ratio is inconsistent with real text (for example, a box that
is taller than it is wide, since valid text regions are almost always wider than
they are tall) are discarded as false positives. The same sliding window
strategy is then reused for the character segmentation stage: a second
classifier, trained to distinguish boundaries between characters from the
interior of a character, is swept across each detected text region to find the
splits between individual letters.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:494 '* Getting More Data'
// Slide: Getting More Data
#strong[Getting More Data]

The ideal recipe for machine learning is straightforward: combine a
#emph[low-bias algorithm] with a #emph[massive amount of data]. A low-bias model
has enough capacity to capture complex patterns, and abundant data prevents it
from overfitting to noise. The challenge lies in making this recipe practical.

How do we know we are using the data effectively? One concrete tool is
#strong[learning curves], which plot training and validation performance as a
function of dataset size. If the validation curve is still climbing steeply when
you run out of data, you have room to benefit from more. If both curves have
plateaued and converged, adding data alone will not help, and you should instead
look at model capacity or feature engineering.

How do we actually get a large amount of data? A useful habit is to ask: "how
much work would it take to get 10x more data than we currently have?" The answer
is often less daunting than expected, and two broad strategies cover most
situations:

- #emph[Artificial data]: synthesize or augment an existing dataset. In computer
  vision this might mean applying rotations, crops, color jitter, or generative
  models to produce realistic variants. In NLP, paraphrasing or back-translation
  can multiply a corpus. The key constraint is that the synthetic examples must
  stay close to the true data distribution; otherwise the extra volume hurts
  more than it helps.
- #emph[Collect and label by hand]: crowdsourcing platforms such as Amazon
  Mechanical Turk let you scale human annotation relatively cheaply, though
  quality control (redundant labeling, gold-standard checks) is essential. More
  recently, large language models have been used to generate or annotate data,
  but this carries real risk: the model's own biases and errors propagate into
  your training set, so any LLM-generated labels should be treated as noisy and
  validated against human judgment.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:511 '* Getting Mode Data: OCR Pipeline'
// Slide: Getting Mode Data: OCR Pipeline
#strong[Getting Mode Data: OCR Pipeline]

How can we increase the size of a training set for a problem like optical
character recognition (OCR)? Several practical strategies exist, ranging from
fully synthetic generation to human-powered collection.

1. #emph[Synthesize a data set from scratch]: use font libraries to render
  characters in many typefaces, paste them against random backgrounds, and apply
  transformations such as scaling, rotation, distortion, and additive noise.
  This can produce virtually unlimited labeled examples at negligible marginal
  cost.
2. #emph[Amplify an existing data set]: take real examples already in hand and
  generate new ones by warping, distorting, or otherwise perturbing them. The
  result is a larger and more varied training set without collecting any new raw
  data.
3. #emph[Distill other models]: use a larger, already-trained model to label a
  pool of unlabeled data, then train a smaller model on those soft or hard
  labels.
4. #emph[Buy or create datasets]: acquire labeled data from commercial providers
  or invest in building a bespoke collection pipeline.
5. #emph[Crowdsourcing and gamification]: distribute labeling tasks to a large
  pool of human workers, sometimes embedding the work inside a game to improve
  engagement and throughput.

For strategies 1 and 2, the choice of transformations and noise must be guided
by the application domain. Gaussian noise, for instance, is a common default,
but it may not reflect the actual degradation a system encounters in practice.
An OCR system reading scanned documents faces ink bleed, uneven lighting, and
paper texture, none of which look like isotropic Gaussian perturbations.
Applying domain-inappropriate augmentations can introduce a mismatch between
training and deployment distributions, ultimately hurting rather than helping
generalization.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:528 '* Ceiling Analysis for ML Pipeline'
// Slide: Ceiling Analysis for ML Pipeline
#strong[Ceiling Analysis for ML Pipeline]

The most valuable resource in any machine learning project is time. Researchers'
time is extremely expensive, and spending months optimizing one component of a
pipeline only to discover that the optimization barely moves the needle on
overall performance is a costly mistake. The central question, then, is: which
part of the pipeline deserves the investment of time and resources?

#strong[Ceiling analysis] provides a principled answer by quantifying how much
each pipeline component limits end-to-end performance. The procedure begins by
choosing a single scalar metric for the entire system, since juggling too many
metrics at once obscures the signal (for an OCR system, for instance, overall
character-level accuracy works well). Then, working through the pipeline in
order, each component is temporarily replaced by an "oracle" that always
produces the correct output while every other component is left untouched. After
each such replacement the end-to-end metric is recomputed. The component whose
oracle produces the largest jump in that metric is the highest-value target for
further work: it is the bottleneck whose errors propagate most heavily through
the rest of the system.

The key discipline ceiling analysis enforces is measurement over intuition. It
is tempting to optimize whichever stage feels most imperfect or most
intellectually interesting, but as Knuth warned, "premature optimization is the
root of all evil" #cite("knuth1974premature"). Without ceiling analysis, teams
routinely pour effort into a stage that is already performing near its ceiling
while ignoring the true bottleneck one or two steps away.

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:552 '* Ceiling Analysis for ML Pipeline: OCT Example'
// Slide: Ceiling Analysis for ML Pipeline: OCT Example
#strong[Ceiling Analysis for ML Pipeline: OCT Example]

Consider a concrete example: an OCR pipeline whose baseline accuracy is 72%. A
ceiling analysis reveals which component deserves attention first. As
@tab:ceilinganalysisformlpipelineoctexample shows, text detection is the
highest-value target: perfecting it alone accounts for most of the achievable
gain. The remaining components, while still imperfect, contribute far less
marginal improvement, so engineering effort spent on them yields diminishing
returns until the text detection bottleneck is resolved.

#figure(
  styled-table(
    headers: ("Component made perfect", "Accuracy", "Gain"),
    rows: (
      ("(baseline system)", "72%", "--"),
      ("Text detection", "89%", "+17%"),
      ("+ Char. segmentation", "90%", "+1%"),
      ("+ Char. classification", "100%", "+10%"),
    ),
  ),
  caption: [Table of Component made perfect, Accuracy, Gain],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:ceilinganalysisformlpipelineoctexample>

// From: msml610/lectures_source/Lesson02.2-ML_Paradigms.smd:580 '* References'
// Slide: References
#strong[References]

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
