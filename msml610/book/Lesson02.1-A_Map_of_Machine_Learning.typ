// git_hash=44abcfb27-twy timestamp=20260903_094703
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
  title: "L02.1: A Map of Machine Learning",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L02.1: A Map of Machine Learning")

// From: msml610/lectures_source/Lesson02.1-A_Map_of_Machine_Learning.smd:13 '# A Map of Machine Learning'
// Slide: A Map of Machine Learning

// From: msml610/lectures_source/Lesson02.1-A_Map_of_Machine_Learning.smd:15 '* Four Branches of Machine Learning'
// Slide: Four Branches of Machine Learning
= Four Branches of Machine Learning

A map of the field can orient before the details: it shows where a new method
fits, which tool suits a given problem, and what this course will and will not
cover.

Machine learning is a sprawling field with many branches #cite(
  "burkov2019hundredpage",
), and one useful way to organize it is along four dimensions, as in
@fig:fourbranchesofmachinelearning.

#strong[Paradigms] describe how learning is set up: whether the system learns
from labeled examples, from rewards, or from unlabeled structure #cite(
  "russell2020aima",
). #strong[Models] are the functional forms that encode hypotheses, from linear
functions to deep neural networks #cite("hastie2009elements").
#strong[Techniques] are the algorithms and processes used to fit those models to
data. Finally, #strong[theory] provides the formal foundations, the guarantees
and limits that tell us when and why a method works.

// rendered_images:begin
// ```mermaid
// %%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#EEEDFE', 'primaryBorderColor': '#7F77DD', 'primaryTextColor': '#26215C', 'lineColor': '#888888', 'fontFamily': 'Helvetica'}}}%%
// mindmap
//   root((Machine Learning))
//     (Paradigms)
//       Supervised
//       Unsupervised
//       Self-supervised
//       Semi-supervised
//       RL
//       Active
//       Online
//     (Models)
//       Linear
//       GLM
//       Neural networks
//       KNN
//       SVM
//       Gaussian processes
//       Graphical models
//     (Techniques)
//       Input processing
//       Model building
//       Performance evaluation
//       Diagnostic
//       Regularization
//       Aggregation
//     (Theory)
//       VC theory
//       Bias-variance decomposition
//       Description length
//       Bayesian
// ```
// label=fig:fourbranchesofmachinelearning
// caption=Diagram relating Machine Learning, Paradigms, Models and Techniques
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.1-A_Map_of_Machine_Learning.typ.figs/Lesson02.1-A_Map_of_Machine_Learning.1.png"),
  caption: [Diagram relating Machine Learning, Paradigms, Models and Techniques],
) <fig:fourbranchesofmachinelearning>
// render_images:end

// From: msml610/lectures_source/Lesson02.1-A_Map_of_Machine_Learning.smd:71 '* Learning Paradigms'
// Slide: Learning Paradigms
= Learning Paradigms

How do you set up a learning problem? The answer depends on what kind of
feedback the learner receives and how it interacts with its environment.
@fig:learningparadigms illustrates the major learning paradigms organized along
two axes: the availability of labels and whether the learning process is
interactive or sequential.

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
// label=fig:learningparadigms
// caption=Diagram relating Learning Paradigms, Label Availability and Interactive/Sequential
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.1-A_Map_of_Machine_Learning.typ.figs/Lesson02.1-A_Map_of_Machine_Learning.2.png"),
  caption: [Diagram relating Learning Paradigms, Label Availability and Interactive/Sequential],
) <fig:learningparadigms>
// render_images:end

In #strong[supervised learning], the learner receives a dataset of input-output
pairs and must discover a mapping from inputs to outputs that generalizes to
unseen examples. The labels are fully provided, making this the most
straightforward setting: given enough labeled data, the learner can directly
measure how well its predictions match the ground truth. In contrast,
#strong[unsupervised learning] operates without any labels at all. The learner
must find structure, patterns, or compressed representations in raw data, with
no explicit signal telling it what the "right answer" is. Clustering,
dimensionality reduction, and density estimation all fall under this heading.

Between these two extremes sits #strong[semi-supervised learning], where only a
small fraction of the data carries labels while the bulk remains unlabeled. The
learner leverages the structure of the unlabeled examples to improve its
predictions beyond what the few labeled points alone would support. A related
but distinct paradigm is #strong[self-supervised learning], in which the learner
constructs its own supervision signal from the data itself (for instance,
predicting a masked word in a sentence or the next frame in a video),
effectively turning an unsupervised problem into a supervised one without human
annotation.

When the learning process becomes interactive, additional paradigms emerge.
#strong[Active learning] allows the learner to query an oracle (often a human
annotator) for the labels of specific examples it finds most informative,
thereby reducing the total labeling effort needed to reach a given performance
level. #strong[Reinforcement learning] goes further: the agent acts within an
environment, receives scalar reward signals rather than explicit labels, and
must learn a policy that maximizes cumulative reward over time. Here the
feedback is not only partial but delayed, since a single action's consequences
may not become apparent until many steps later. This sequential, interactive
nature makes reinforcement learning fundamentally different from the batch,
label-driven paradigms and introduces unique challenges such as the
exploration-exploitation tradeoff.

// From: msml610/lectures_source/Lesson02.1-A_Map_of_Machine_Learning.smd:110 '* Model Families'
// Slide: Model Families
= Model Families

What functional form can a model take? Machine learning models generally fall
into three broad families, each making different assumptions about how the
data-generating process is structured.

#strong[Parametric models] assume that the data can be described by a fixed,
finite set of parameters. Once those parameters are estimated from training
data, the original observations are no longer needed for prediction. Linear
regression, logistic regression, and neural networks are all parametric: you
commit up front to a specific functional form (a line, a sigmoid boundary, a
layered composition of nonlinearities) and then fit its coefficients. The
advantage is computational efficiency and interpretability when the chosen form
is simple; the risk is that a misspecified form underfits the true relationship.

#strong[Non-parametric models] make no such commitment. Instead of fixing the
number of parameters before training, they let model complexity grow with the
size of the dataset. k-Nearest Neighbors is a canonical example: at prediction
time the model consults the training set directly, so adding more data
effectively adds more "parameters." Kernel density estimators and Gaussian
processes also belong here. Non-parametric methods are more flexible and less
prone to structural misspecification, but they can be memory-intensive and
slower at inference because they must retain (or summarize) the training data.

#strong[Graphical models] take a different angle entirely. Rather than focusing
on whether the parameter count is fixed or flexible, they represent the joint
probability distribution over a set of variables as a graph, where nodes are
random variables and edges encode conditional dependencies. Bayesian networks
(directed acyclic graphs) and Markov random fields (undirected graphs) are the
two main sub-families. Graphical models excel at capturing complex multivariate
structure, reasoning under uncertainty, and making conditional independence
assumptions explicit and auditable.

These three families are not mutually exclusive. A Bayesian network, for
instance, can use parametric distributions at each node, or it can use
non-parametric conditional density estimators. The choice of family shapes
everything downstream: how training scales, what inductive biases the learner
brings, and how transparent the resulting model is to inspection.
@fig:modelfamilies summarizes this taxonomy, showing how the three families
branch from the central concept of an ML model.

// rendered_images:begin
// ```graphviz[width=70%]
// digraph MLModelsTaxonomy {
//   graph [rankdir=LR, splines=curved, bgcolor="transparent",
//          ranksep="1.0 equally", nodesep=0.24, pad=0.3, fontname="Helvetica"];
//   node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//          fontcolor="#26215C", color="#D8D6EE", penwidth=1.2,
//          height=0.46, margin="0.18,0.09"];
//   edge  [arrowhead=none, penwidth=1.4, color="#B9B6D6"];
// 
//   root [label="ML Models", shape=box, style="rounded,filled",
//         fillcolor="#26215C", fontcolor="white", fontsize=14, penwidth=0,
//         margin="0.26,0.16"];
// 
//   // Parametric - violet
//   parametric [label="Parametric", fillcolor="white", color="#7C74D6", fontcolor="#45296B", penwidth=1.6, fontsize=12];
//   linear     [label=<<b>Linear models</b><br/><font point-size="9" color="#45296B">Linear, deterministic</font>>, fillcolor="#EFEDFC", color="#CBC7F0", fontcolor="#45296B"];
//   glm        [label=<<b>GLM</b><br/><font point-size="9" color="#45296B">Linear predictor, probabilistic</font>>, fillcolor="#EFEDFC", color="#CBC7F0", fontcolor="#45296B"];
//   nn         [label=<<b>Neural networks</b><br/><font point-size="9" color="#45296B">Non-linear, deterministic</font>>, fillcolor="#EFEDFC", color="#CBC7F0", fontcolor="#45296B"];
// 
//   // Non-parametric - blue
//   nonparam   [label="Non-parametric", fillcolor="white", color="#3E86C8", fontcolor="#1F4E79", penwidth=1.6, fontsize=12];
//   knn        [label=<<b>Nearest neighbors</b><br/><font point-size="9" color="#1F4E79">KNN, non-linear</font>>, fillcolor="#E8F1FB", color="#BFD8F1", fontcolor="#1F4E79"];
//   svm        [label=<<b>SVM</b><br/><font point-size="9" color="#1F4E79">Kernelized, non-linear</font>>, fillcolor="#E8F1FB", color="#BFD8F1", fontcolor="#1F4E79"];
//   gp         [label=<<b>Gaussian processes</b><br/><font point-size="9" color="#1F4E79">Non-linear, probabilistic</font>>, fillcolor="#E8F1FB", color="#BFD8F1", fontcolor="#1F4E79"];
// 
//   // Graphical models - teal
//   graphical  [label="Graphical models", fillcolor="white", color="#2F9678", fontcolor="#1F4E39", penwidth=1.6, fontsize=12];
//   bn         [label=<<b>Bayesian networks</b><br/><font point-size="9" color="#1F4E39">Structured, probabilistic</font>>, fillcolor="#E5F4EE", color="#BCE0D2", fontcolor="#1F4E39"];
//   hmm        [label=<<b>Hidden Markov models</b><br/><font point-size="9" color="#1F4E39">Sequential, probabilistic</font>>, fillcolor="#E5F4EE", color="#BCE0D2", fontcolor="#1F4E39"];
//   kalman     [label=<<b>Kalman filters</b><br/><font point-size="9" color="#1F4E39">Continuous state, probabilistic</font>>, fillcolor="#E5F4EE", color="#BCE0D2", fontcolor="#1F4E39"];
// 
//   root -> parametric [color="#7C74D6", penwidth=2.0];
//   root -> nonparam   [color="#3E86C8", penwidth=2.0];
//   root -> graphical  [color="#2F9678", penwidth=2.0];
// 
//   parametric -> {linear glm nn}   [color="#A9A3E6"];
//   nonparam   -> {knn svm gp}      [color="#8FB6DE"];
//   graphical  -> {bn hmm kalman}   [color="#89C0AC"];
// 
//   { rank=same; linear; glm; nn; knn; svm; gp; bn; hmm; kalman; }
// }
// ```
// label=fig:modelfamilies
// caption=Diagram relating ML Models, Parametric, Non-parametric and Graphical models
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.1-A_Map_of_Machine_Learning.typ.figs/Lesson02.1-A_Map_of_Machine_Learning.3.png"),
  caption: [Diagram relating ML Models, Parametric, Non-parametric and Graphical models],
) <fig:modelfamilies>
// render_images:end

// From: msml610/lectures_source/Lesson02.1-A_Map_of_Machine_Learning.smd:156 '* Stages of an ML Pipeline'
// Slide: Stages of an ML Pipeline
= Stages of an ML Pipeline

What are the #emph[stages] and #emph[common techniques] used in a machine
learning pipeline? A typical pipeline moves through several distinct phases,
each with its own set of tools and concerns.

The process begins with #strong[input processing], where raw data is transformed
into a form suitable for modeling. This includes data cleaning (handling missing
values, correcting errors, removing duplicates), dimensionality reduction
(projecting high-dimensional data into a lower-dimensional space to reduce noise
and computation), and feature engineering (constructing new input variables that
better capture the structure relevant to the prediction task).

Next comes #strong[model building], which involves two intertwined decisions:
choosing a functional form for the hypothesis (a linear model, a decision tree,
a neural network, etc.) and selecting a learning algorithm that fits the
parameters of that model to the training data. The choice of model family
constrains what kinds of patterns the system can represent, while the learning
algorithm determines how efficiently and reliably it finds a good fit within
that family.

Once a model is trained, #strong[performance evaluation] measures how well it
generalizes. Cross-validation is the standard technique here: the data is split
into complementary subsets so the model is always evaluated on examples it was
not trained on, giving a more honest estimate of future performance than
training-set accuracy alone.

#strong[Diagnostics] then help the practitioner understand _why_ performance is
what it is. Bias-variance curves reveal whether errors stem from an overly
simple model (high bias) or from excessive sensitivity to training noise (high
variance). Learning curves plot performance as a function of training-set size,
showing whether collecting more data is likely to help or whether the model's
capacity is the bottleneck.

When diagnostics reveal overfitting, #strong[regularization] techniques
constrain the model's complexity, penalizing large parameter values or limiting
the number of effective degrees of freedom to improve generalization.

Finally, #strong[aggregation] methods combine multiple models to achieve better
performance than any single one. Boosting #cite("freundschapire1997boosting")
trains a sequence of weak learners, each focusing on the mistakes of its
predecessors. Bagging #cite("breiman1996bagging") trains multiple models on
bootstrap samples of the data and averages their predictions to reduce variance.
Stacking learns a meta-model that optimally combines the outputs of several
diverse base learners.

@fig:stagesofanmlpipeline illustrates how these stages connect in sequence, each
feeding its output into the next while diagnostic feedback loops allow the
practitioner to revisit earlier decisions when later stages reveal problems.

// rendered_images:begin
// ```graphviz
// digraph MLPipeline {
//   bgcolor="transparent";
//   pad="0.15";
//   splines=spline;
//   nodesep=0.30;
//   ranksep=0.45;
//   rankdir=LR;
// 
//   node [shape=box,
//         style="rounded,filled",
//         penwidth=1.8,
//         fontname="Helvetica",
//         fontsize=11,
//         margin="0.18,0.12",
//         height=0.46];
// 
//   edge [color="#A3B1C0",
//         penwidth=1.3,
//         arrowhead=vee,
//         arrowsize=0.7,
//         fontname="Helvetica",
//         fontsize=9,
//         fontcolor="#7B8794"];
// 
//   InputProcessing [label=<<b>Input Processing</b><br/><font point-size="9" color="#1F4E79">Cleaning, dim. reduction, feature eng.</font>>,
//                     fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//   ModelBuilding    [label=<<b>Model Building</b><br/><font point-size="9" color="#1F4E79">Choose model, learning algorithm</font>>,
//                     fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//   PerformanceEval  [label=<<b>Performance Evaluation</b><br/><font point-size="9" color="#1F4E79">Cross-validation</font>>,
//                     fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//   Diagnostic       [label=<<b>Diagnostic</b><br/><font point-size="9" color="#1F4E79">Bias-variance curves, learning curves</font>>,
//                     fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//   InputProcessing -> ModelBuilding;
//   ModelBuilding -> PerformanceEval;
//   PerformanceEval -> Diagnostic;
// }
// ```
// label=fig:stagesofanmlpipeline
// caption=Diagram illustrating Stages of an ML Pipeline
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.1-A_Map_of_Machine_Learning.typ.figs/Lesson02.1-A_Map_of_Machine_Learning.4.png"),
  caption: [Diagram illustrating Stages of an ML Pipeline],
) <fig:stagesofanmlpipeline>
// render_images:end

// From: msml610/lectures_source/Lesson02.1-A_Map_of_Machine_Learning.smd:211 '* Theoretical Foundations'
// Slide: Theoretical Foundations
= Theoretical Foundations

What is the theoretical basis underpinning machine learning and its common
phenomena? Several foundational frameworks address this question, each offering
a different lens on why learning from data works and when it fails.

#strong[VC theory] #cite("vapnikchervonenkis1971uniform") measures a model's
#emph["capacity"] by examining the complexity of its hypothesis space. Rather
than counting parameters directly, VC dimension quantifies how many data points
a model class can shatter (classify in every possible way), providing
distribution-free guarantees on generalization. A model with high VC dimension
can fit more patterns but also needs more data to generalize reliably.

The #strong[bias-variance decomposition] #cite("geman1992biasvariance") breaks
prediction error into three components:

$ "Error" = "Bias"^2 + "Variance" + sigma^2 $

#emph[Bias] captures the error introduced when a model's assumptions are too
simplistic to represent the true relationship; for instance, fitting a linear
model to data generated by a quadratic process. #emph[Variance] captures
sensitivity to fluctuations in the training set: a highly flexible model may fit
one sample well but produce very different predictions on another. The third
term, #emph[irreducible noise] ($sigma^2$), reflects inherent randomness in the
data that no model, however powerful, can eliminate. The practical consequence
is a tradeoff: reducing bias (by using a more flexible model) typically
increases variance, and vice versa.

The #strong[description length] framework #cite("rissanen1978mdl") connects
learning to information theory and compression. The #emph[Minimum Description
  Length] (MDL) principle judges a model by the total cost of describing both
the model itself and the data given that model, measured in bits. A model that
is too simple requires many bits to encode the residual errors; one that is too
complex requires many bits to encode the model's own parameters. The best model
strikes a balance, compressing the overall description most efficiently.

The #strong[Bayesian approach] #cite("bishop2006prml") treats machine learning
as probabilistic inference. Instead of selecting a single best hypothesis, it
maintains a distribution over hypotheses, combining prior knowledge with
observed data through Bayes' theorem to update beliefs. This framework naturally
incorporates uncertainty: predictions reflect not just the most likely model but
the entire posterior distribution, and model complexity is penalized implicitly
through the marginal likelihood.

These frameworks are complementary rather than competing: VC theory provides
worst-case guarantees, bias-variance analysis guides model selection, MDL links
learning to compression, and the Bayesian view supplies a coherent probabilistic
calculus. That said, a common difficulty in ML theory is that the assumptions
each framework requires may not align neatly with the messy, high-dimensional
problems encountered in practice. Distribution-free bounds can be too loose to
be informative; bias-variance decompositions assume a squared-error loss that
may not match the task; and Bayesian priors may be chosen for computational
convenience rather than genuine belief. Recognizing these gaps between theory
and practice is essential for applying any of these tools responsibly.

// From: msml610/lectures_source/Lesson02.1-A_Map_of_Machine_Learning.smd:234 '* Adages of Machine Learning'
// Slide: Adages of Machine Learning
= Adages of Machine Learning

Machine learning has accumulated a rich body of folk wisdom, crystallized into
adages that capture recurring lessons practitioners encounter again and again.
These sayings are not rigorous theorems, but they summarize real phenomena that
shape how models are built, evaluated, and deployed.

Some of the oldest principles concern simplicity. Einstein's dictum that
#emph[an explanation of the data should be as simple as possible, but not
  simpler] warns against both over-complicated models and naive ones that throw
away necessary structure. This echoes #emph[Occam's razor]: the simplest model
that fits the data is also the most plausible, a principle formalized in
Bayesian model selection and minimum description length.

Other adages focus on data quality and quantity. #emph[Garbage in, garbage out]
(Fuechse, 1957) reminds us that no algorithm can rescue fundamentally flawed
inputs. George Box's observation that #emph[all models are wrong, but some are
  useful] #cite("box1976science") reframes the goal: we are not seeking truth in
a model's assumptions, only predictive or explanatory value. On the darker side,
Ronald Coase warned in 1982 that #emph[if you torture the data long enough it
  will confess whatever you want], a caution against p-hacking, overfitting, and
confirmation bias in analysis.

More recent adages reflect the data-driven era. Clive Humby's 2006 phrase
#emph[data is the new oil] cast data as a raw resource that, like petroleum,
must be refined to unlock its value. Peter Norvig's complementary claim that
#emph[more data beats clever algorithms] (2006) was later expanded into the
influential paper on #emph[the unreasonable effectiveness of data] #cite(
  "halevy2009unreasonable",
), which argued that simple models trained on massive corpora routinely
outperform sophisticated models trained on small datasets. Finally, Rich
Sutton's #emph[bitter lesson] #cite("sutton2019bitterlesson") sharpened this
point further: across the history of AI, general methods that leverage
computation (search and learning at scale) have consistently won out over
systems that try to encode human domain knowledge by hand. The lesson is
"bitter" because researchers repeatedly invest years in clever,
knowledge-intensive approaches only to be overtaken by brute-force scaling.

Taken together, these adages trace a shift in emphasis: from model elegance and
careful feature engineering toward large-scale data and computation as the
dominant drivers of progress.

// From: msml610/lectures_source/Lesson02.1-A_Map_of_Machine_Learning.smd:252 '* References'
// Slide: References
= References

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
