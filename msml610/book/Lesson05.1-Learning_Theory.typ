// git_hash=aa74b8a4-ctq timestamp=20260918_151057
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
  title: "L05.1: Learning Theory",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L05.1: Learning Theory")

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:7 '* Roadmap'
// Slide: Roadmap
= Roadmap

This lesson asks whether machine learning is even possible: it turns the "possible vs
probable" intuition into the Hoeffding inequality and shows why it is too weak once
we choose among many hypotheses. It then introduces the #strong[growth function],
which counts dichotomies, shattering, and break points to replace the number of
hypotheses $M$ with a finite count. The lesson closes with the #strong[VC dimension]:
the single number that measures the complexity of a hypothesis set, yields
generalization bounds, and tells us how many examples we need. @fig:roadmap
summarizes the progression from feasibility questions through growth functions to the
VC dimension.

// rendered_images:begin
// ```graphviz
// digraph LessonFlow {
//   rankdir=TB;
//   bgcolor="transparent";
//   nodesep=0.3;
//   ranksep=0.4;
//   node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//         penwidth=1.4, fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//   edge [color="#8C8C8C", penwidth=1.2];
// 
//   hoeffding [label=<<b>Hoeffding inequality</b><br/>fixed h: E<SUB>in</SUB> tracks E<SUB>out</SUB>>];
//   union     [label=<<b>Union bound</b><br/>M hypotheses: bound too weak>,
//              fillcolor="#FFB3B3", color="#D64545", fontcolor="#6B1F1F"];
//   growth    [label=<<b>Growth function</b><br/>replace M with m<SUB>H</SUB>(N)>];
//   vc        [label=<<b>VC dimension</b><br/>N &ge; 10 d<SUB>VC</SUB> examples>];
// 
//   hoeffding -> union -> growth -> vc;
// }
// ```
// label=fig:roadmap
// caption=Progression from feasibility questions to the VC dimension.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.1.png",
    width: 70%,
  ),
  caption: [Progression from feasibility questions to the VC dimension.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:roadmap>
// render_images:end

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:47 '# Is Machine Learning Even Possible?'
// Slide: Is Machine Learning Even Possible?
= Is Machine Learning Even Possible?

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:49 '## The Feasibility Problem'
// Slide: The Feasibility Problem
== The Feasibility Problem

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:51 '* Visual ML Experiment: Setup'
// Slide: Visual ML Experiment: Setup
Consider the supervised classification problem shown in @fig:visualmlexperimentsetup.
The input is a 9-bit vector represented as a 3×3 array, where each cell is either
filled or empty. The training set consists of two groups: a blue row of patterns
$bold(x)_1, bold(x)_2, bold(x)_3$ labeled $f(bold(x)) = -1$, and a green row of
patterns $bold(x)_4, bold(x)_5, bold(x)_6$ labeled $f(bold(x)) = +1$. Given a new red
test pattern $bold(x)_0$, the question is whether $f(bold(x)_0) = -1$ or
$f(bold(x)_0) = +1$.

// rendered_images:begin
// ```raw_latex
// \documentclass[border=1pt]{standalone}
// \usepackage{tikz}
// \begin{document}
// 
// \newcommand{\gridpattern}[2]{
//   \begin{tikzpicture}[scale=0.4]
//     \foreach \x in {0,...,2}{
//       \foreach \y in {0,...,2}{
//         \pgfmathsetmacro{\v}{#1[\y*3+\x]}
//         \draw[black] (\x,-\y) rectangle ++(1,-1); % draw grid cell
//         \ifnum \v=1
//           \fill[#2] (\x+0.1,-\y-0.1) rectangle ++(0.8,-0.8); % slightly smaller fill
//         \fi
//       }
//     }
//   \end{tikzpicture}
// }
// 
// %\begin{center}
// \begin{tikzpicture}
//   \matrix[row sep=1em] {
//     \node{\gridpattern{{1,0,0,1,0,1,0,1,0}}{blue}}; &
//     \node{\gridpattern{{1,0,0,0,0,1,1,0,1}}{blue}}; &
//     \node{\gridpattern{{1,0,0,0,0,1,0,0,0}}{blue}}; &
//     \node{\(f = -1\)}; \\
//     \node{\gridpattern{{0,0,1,0,1,0,1,0,0}}{green}}; &
//     \node{\gridpattern{{0,1,0,1,0,1,0,1,0}}{green}}; &
//     \node{\gridpattern{{0,1,1,1,1,0,0,1,1}}{green}}; &
//     \node{\(f = +1\)}; \\
//   };
//   \node at (-0.6,-3.0) {\gridpattern{{1,0,0,0,1,0,0,0,1}}{red}};
//   \node at (1.4,-3.0) {\(f = ?\)};
// \end{tikzpicture}
// %\end{center}
// \end{document}
// ```
// label=fig:visualmlexperimentsetup
// caption=Training patterns and a test pattern in the visual ML experiment.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.2.png",
    width: 70%,
  ),
  caption: [Training patterns and a test pattern in the visual ML experiment.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:visualmlexperimentsetup>
// render_images:end

This setup captures the essence of any supervised classification task: we have
labeled examples from two classes, and we must decide which class a previously unseen
input belongs to. The challenge lies in figuring out what distinguishes the $-1$
patterns from the $+1$ patterns, then applying that learned distinction to the test
pattern. Even with inputs as simple as 3×3 binary grids, the number of possible
classification rules is enormous, so the learner must rely on some form of inductive
bias to choose among them.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:108 '* Visual ML Experiment: Competing Models'
// Slide: Visual ML Experiment: Competing Models
To see why some inductive bias is unavoidable, consider two competing models, each
consistent with the same six training examples. Model 1 classifies a pattern as $+1$
whenever it has an axis of symmetry and $-1$ otherwise; because the test pattern is
symmetrical, this model predicts $f(bold(x)_0) = +1$. Model 2 classifies a pattern as
$+1$ whenever the top-left square is empty and $-1$ otherwise; because the test
pattern has a full top-left square, this model predicts $f(bold(x)_0) = -1$. Both
models agree perfectly on every training point yet disagree on the test point.

This is not a quirk of two hand-picked rules. Many functions fit the six training
examples exactly, and among them some assign $-1$ to the test point while others
assign $+1$. With the training data alone, there is no purely logical way to choose
between them; the data simply does not single out a unique target function.

That observation raises a fundamental question: how can a limited data set reveal
enough information to pin down the entire target function? Put differently, is
learning from finite data even possible? The answer, as later chapters will show, is
that learning does not require identifying the target function exactly. Instead, a
learner commits to a restricted hypothesis class or inductive bias that rules out
most of the functions consistent with the data, then selects the best fit within that
smaller set. The training examples constrain the search, and the bias makes the
remaining ambiguity manageable. Without such a bias, every finite sample is
compatible with infinitely many functions, and no generalization can be justified.
With one, a surprisingly small number of examples can suffice.

// rendered_images:begin
// ```raw_latex
// \documentclass[border=1pt]{standalone}
// \usepackage{tikz}
// \begin{document}
// 
// \newcommand{\gridpattern}[2]{
//   \begin{tikzpicture}[scale=0.4]
//     \foreach \x in {0,...,2}{
//       \foreach \y in {0,...,2}{
//         \pgfmathsetmacro{\v}{#1[\y*3+\x]}
//         \draw[black] (\x,-\y) rectangle ++(1,-1); % draw grid cell
//         \ifnum \v=1
//           \fill[#2] (\x+0.1,-\y-0.1) rectangle ++(0.8,-0.8); % slightly smaller fill
//         \fi
//       }
//     }
//   \end{tikzpicture}
// }
// 
// %\begin{center}
// \begin{tikzpicture}
//   \matrix[row sep=1em] {
//     \node{\gridpattern{{1,0,0,1,0,1,0,1,0}}{blue}}; &
//     \node{\gridpattern{{1,0,0,0,0,1,1,0,1}}{blue}}; &
//     \node{\gridpattern{{1,0,0,0,0,1,0,0,0}}{blue}}; &
//     \node{\(f = -1\)}; \\
//     \node{\gridpattern{{0,0,1,0,1,0,1,0,0}}{green}}; &
//     \node{\gridpattern{{0,1,0,1,0,1,0,1,0}}{green}}; &
//     \node{\gridpattern{{0,1,1,1,1,0,0,1,1}}{green}}; &
//     \node{\(f = +1\)}; \\
//   };
//   \node at (-0.6,-3.0) {\gridpattern{{1,0,0,0,1,0,0,0,1}}{red}};
//   \node at (1.4,-3.0) {\(f = ?\)};
// \end{tikzpicture}
// %\end{center}
// \end{document}
// ```
// label=fig:visualmlexperimentcompetingmodels
// caption=Training patterns and a test pattern shared by both models.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.3.png",
    width: 70%,
  ),
  caption: [Training patterns and a test pattern shared by both models.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:visualmlexperimentcompetingmodels>
// render_images:end

@fig:visualmlexperimentcompetingmodels shows the tension: two models, each perfectly
consistent with the training data, produce opposite predictions on the same unseen
input, which makes the need for an inductive bias concrete.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:174 '* Possible vs Probable'
// Slide: Possible vs Probable
The same difficulty appears whenever we extrapolate: the function can assume any
value outside the data it was trained on. Consider, for instance, a model built from
summer temperature readings: nothing in those observations alone constrains what the
function predicts for winter months, so it could return virtually anything in that
unseen region.

How, then, can we learn an unknown function if estimating its value at unseen points
seems impossible in general? The answer is that we cannot do it without bringing
something extra to the table: assumptions about the function's smoothness, a
parametric model family, or domain knowledge that narrows the space of plausible
behaviors.

This is the core distinction between what is #emph[possible] and what is
#emph[probable]. When we have no knowledge of the unknown function whatsoever, any
continuation is equally valid: the true relationship could be linear, quadratic, a
sine wave, or something far more exotic outside the observed data. In that setting,
generalization is hopeless. Once we introduce even modest domain knowledge or use
historical data patterns, however, we shift from the space of all possible functions
to a much smaller set of #emph[probable] ones. If historical weather data, for
example, traces a sinusoidal pattern year after year, it is reasonable to expect that
unobserved months follow the same cycle. The assumption is not guaranteed to be
correct, but it converts an impossible estimation problem into a tractable one by
ruling out the vast majority of candidate functions that contradict what we already
know.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:193 '## Hoeffding Inequality and the Bin Analogy'
// Slide: Hoeffding Inequality and the Bin Analogy
== Hoeffding Inequality and the Bin Analogy

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:195 '* Bin Analogy: Estimating a Probability'
// Slide: Bin Analogy: Estimating a Probability
Consider a bin containing red and green marbles. The goal is to estimate the
probability of picking a red marble, $Pr("pick a red marble") = mu$, where the value
of $mu$ is unknown. To gather information, we pick $N$ marbles independently with
replacement and record the fraction of red marbles in our sample as $nu$.

#figure(
  image("../lectures_source/figures/L05.1.Bin_with_marbles.png", width: 80%),
  caption: [Bin with marbles],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:binwithmarbles>

Does the sample fraction $nu$ tell us anything about the bin fraction $mu$? At first
glance, the answer seems to be "no": in strict terms, we know nothing about the
marbles we did not pick. The sample could be mostly green while the bin is mostly
red. This scenario is possible, but it is not probable. That distinction is exactly
where probability theory earns its keep. Under certain conditions, the sample
frequency $nu$ is likely to be close to the true frequency $mu$, and the larger the
sample, the more confident we can be in that closeness, as @fig:binwithmarbles shows.

#strong[Hoeffding's inequality] makes the intuition "possible but not probable"
mathematically precise. It provides a bound on how far $nu$ can deviate from $mu$ as
a function of the sample size $N$, turning a vague sense of reliability into a
quantitative guarantee.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:226 '* Hoeffding Inequality'
// Slide: Hoeffding Inequality
To state that bound, consider a Bernoulli random variable $X$ with probability of
success $mu$. To estimate the mean $mu$, draw $N$ independent samples and form the
sample average $nu eq.delta 1/N sum_i X_i$. The central question is how close $nu$ is
likely to be to the true $mu$, and Hoeffding's inequality gives a non-asymptotic
answer.

The #strong[probably approximately correct] (PAC) guarantee #cite(
  "valiant1984theory",
) states #cite("hoeffding1963probability"):

$ Pr(|nu - mu| > epsilon) lt.eq 2 / e^(2 epsilon^2 N) $

The bound has three useful properties. First, it is valid for every finite $N$ and
every $epsilon > 0$: it is not a large-sample approximation like the Central Limit
Theorem, so there is no hidden requirement that $N$ be "large enough." Second, it
applies only when $nu$ is computed from samples drawn at random in the same manner
that defines $mu$; if the sampling process is biased or the draws are dependent, the
guarantee breaks down. Third, the right-hand side does not involve $mu$ at all, which
means the bound holds uniformly regardless of the true success probability.

The interplay among $N$, $epsilon$, and the probability bound reveals a useful
three-way trade-off:

- The bound shrinks exponentially in $N$, so even modest increases in sample size buy
  large reductions in the probability of a bad estimate.
- Tightening the tolerance (making $epsilon$ smaller) demands a proportionally larger
  $N$ to maintain the same confidence level, because $epsilon$ enters the exponent as
  $epsilon^2$.
- Since the inequality guarantees $nu in [mu - epsilon, mu + epsilon]$ with high
  probability, the practical goal is to choose $epsilon$ small while keeping the
  failure probability negligible, which pins down the required sample size.

One subtlety: the bound is formally a statement about the random quantity $nu$, not
about the fixed but unknown $mu$. In practice, however, you read it the other way
around, treating the observed $nu$ as a center and $epsilon$ as a radius to build a
confidence interval for $mu$. That inversion is legitimate precisely because the
inequality is distribution-free in $mu$, but remember that the probabilistic claim
attaches to the sampling procedure, not to any single realized interval.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:250 '* Bin Analogy: Connecting to Learning'
// Slide: Bin Analogy: Connecting to Learning
We now connect the bin analogy, Hoeffding inequality, and feasibility of machine
learning #cite("abumostafa2012learning"). Suppose you know the true target function
$f(bold(x))$ at a set of points $bold(x) in cal(X)$, and you choose a hypothesis
$h: cal(X) arrow.r cal(Y) = {-1, +1}$. Think of each point $bold(x) in cal(X)$ as a
marble. You color the marble green if the hypothesis gets it right
($h(bold(x)) = f(bold(x))$) and red otherwise. Under this coloring, the
#strong[in-sample error] $E_("in")(h)$ is the fraction $nu$ of red marbles in the
sample you have drawn, while the #strong[out-of-sample error] $E_("out")(h)$ is the
fraction $mu$ of red marbles in the entire bin (the full input space). The bridge
between the two is the sampling process: the points $bold(x)_1, ..., bold(x)_N$ are
picked randomly and independently from a distribution over $cal(X)$, which is the
same distribution that defines $E_("out")$.

Because the sample is drawn i.i.d. from the same distribution, the Hoeffding
inequality applies directly and bounds the gap between in-sample and out-of-sample
error for a fixed hypothesis $h$:

$ Pr(|E_("in")(h) - E_("out")(h)| > epsilon) lt.eq 2 e^(-2 epsilon^2 N) $

This tells us that for any single, pre-chosen hypothesis, the probability that its
training error deviates from its true error by more than $epsilon$ shrinks
exponentially with the sample size $N$. In other words, generalization over unknown
points (the unseen marbles still in the bin) is feasible as long as $h$ is fixed
before the data are observed. The in-sample error is a reliable proxy for how well
$h$ performs on the full distribution, and the reliability improves rapidly as we
gather more data.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:274 '## Training, Validation, and Testing'
// Slide: Training, Validation, and Testing
== Training, Validation, and Testing

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:276 '* Learning Phases: College Analogy (1/2)'
// Slide: Learning Phases: College Analogy (1/2)
Machine learning follows a structured progression of phases, each serving a distinct
purpose in building a reliable model. As @fig:learningphasescollegeanalogy12 shows,
the pipeline moves from a learning phase using a training set, through validation and
testing phases with their own held-out data, and finally into an out-of-sample phase
where the model meets real production data.

// rendered_images:begin
// ```graphviz[width=100%]
// digraph ML_Phases {
//   rankdir=LR;
//   node [shape=box, style="rounded,filled", fillcolor=white, fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//   learning   [label="Learning Phase\n(Training Set)"];
//   validation [label="Validation Phase\n(Validation Set)"];
//   testing    [label="Testing Phase\n(Test Set)"];
//   production [label="Out-of-Sample Phase\n(Production)"];
// 
//   learning -> validation -> testing -> production;
// }
// ```
// label=fig:learningphasescollegeanalogy12
// caption=Phases of a machine learning pipeline from training to production.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.4.png",
    width: 100%,
  ),
  caption: [Phases of a machine learning pipeline from training to production.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:learningphasescollegeanalogy12>
// render_images:end

This setup has a natural analogy in a college course. Students begin by studying the
material: reading the textbook, working through examples, and building up their
knowledge base. This corresponds to the #strong[learning phase], where the model is
exposed to labeled training data and adjusts its parameters to capture the underlying
patterns.

Before the final exam, students typically receive practice problems along with their
solutions. Working through these practice problems does not teach entirely new
material; instead, it reveals gaps in understanding and areas that need improvement.
A student who struggles with a particular type of practice problem knows exactly
where to focus additional effort. This step corresponds to the #strong[validation
  set] in machine learning. The model is evaluated on data it was not trained on, and
its performance on that validation data guides decisions such as tuning
hyperparameters, selecting among candidate architectures, or deciding when to stop
training to avoid overfitting. The validation phase, like practice exams, is
diagnostic: it shapes the learning process without being the final judgment of
success.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:303 '* Learning Phases: College Analogy (2/2)'
// Slide: Learning Phases: College Analogy (2/2)
The final exam corresponds to the #strong[testing phase]. The problems on the exam
are different from those encountered during validation (homework and practice sets).
Why not simply hand out the exam problems ahead of time so students can optimize
their performance on them? Because doing well on the exam is not the real objective:
the goal is to learn the course material deeply enough to handle new challenges. The
final exam is not strictly necessary for learning itself; rather, it gauges how well
you have learned, motivates you to study, and only works as a measure precisely
because you have not seen its problems in advance. If you knew the exam questions
beforehand, your score would no longer reflect genuine understanding.

The #strong[out-of-sample phase] corresponds to the experience of using your course
knowledge on real problems after the course ends. None of these problems were
encountered while studying or on the exam: they are new situations drawn from the
world beyond the classroom, and your ability to handle them is the truest test of
whether learning actually occurred.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:321 '## From Validation to Learning'
// Slide: From Validation to Learning
== From Validation to Learning

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:323 '* Validation vs Learning: Many Hypotheses'
// Slide: Validation vs Learning: Many Hypotheses
For a single fixed hypothesis $h$, you have already seen that the in-sample
performance $E_("in")(h) = nu$ needs to stay close to the out-of-sample performance
$E_("out")(h) = mu$. That relationship is the #emph[validation setup]: you have
already learned a model and simply want to check how well it generalizes.

The #strong[learning setup] is different. Here you choose a final hypothesis $g$ from
$M$ candidates, and you need a bound on the out-of-sample performance of whichever
hypothesis the learning algorithm selects, that is, a guarantee that holds for every
$g in cal(H)$. This calls for a Hoeffding-style bound extended to the case of
choosing among multiple hypotheses. The derivation chains three inequalities
together:

$
  forall g in cal(H) = {h_1, dots, h_M} quad Pr(|E_("in")(g) - E_("out")(g)| > epsilon)
$
$
  quad lt.eq Pr(union.big_(i=1)^M (|E_("in")(h_i) - E_("out")(h_i)| > epsilon))
$
$
  quad lt.eq sum_(i=1)^M Pr(|E_("in")(h_i) - E_("out")(h_i)| > epsilon) quad & "(by the union bound)"
$
$ quad lt.eq 2 M exp(-2 epsilon^2 N) quad & "(by Hoeffding)" $

The first step recognizes that $g$ is one of the $h_i$, so the event "$g$ deviates by
more than $epsilon$" is contained in the union of events "some $h_i$ deviates by more
than $epsilon$." The union bound then replaces that union with a sum of individual
probabilities, and Hoeffding's inequality bounds each term.

The result is a valid generalization bound, but it is a weak one. The penalty factor
$M$ grows linearly with the number of hypotheses in the class. For many hypothesis
spaces of practical interest (continuous parameter spaces, neural networks, kernel
machines) $M$ is effectively infinite, which makes $2 M exp(-2 epsilon^2 N)$
vacuously large. This observation motivates the search for tighter complexity
measures, such as the VC dimension, that capture the effective richness of a
hypothesis class without counting every individual hypothesis.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:354 '* Validation Setup: One Coin'
// Slide: Validation Setup: One Coin
A coin-flipping analogy makes the contrast between the two setups concrete. In a
validation setup, consider a single coin whose fairness you want to test. Assume the
coin is unbiased, so its true probability of heads is $mu = 0.5$. You toss it ten
times and record the fraction of heads, $nu$. How likely is it that every single toss
lands heads, making the coin look perfectly biased with $nu = 0$? The answer is:

$ Pr("coin shows" nu = 0) = 1 / 2^(10) = 1 / 1024 approx 0.1% $

That probability is tiny. The chance that the in-sample estimate ($nu = 0.0$)
deviates completely from the true out-of-sample value ($mu = 0.5$) is less than one
tenth of one percent. This is the core intuition behind validation: when a single
hypothesis is tested on a reasonably sized sample, the empirical performance is
unlikely to be wildly misleading. The sample statistic $nu$ tracks the population
parameter $mu$ with high probability, and extreme disagreement between the two is a
rare event.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:371 '* Learning Setup: Many Coins'
// Slide: Learning Setup: Many Coins
In a #strong[learning set-up], the situation changes: you have many coins available
and must choose one, then determine whether it is fair. This mirrors model selection
in machine learning, where you pick a hypothesis from a large class and evaluate it
on limited data.

How likely is it that at least one of 1,000 fair coins appears totally biased after
only 10 flips? In other words, what is the probability that some coin's in-sample
performance ($nu = 0$, all heads or all tails) completely misrepresents its true
out-of-sample behavior? The calculation uses independence and the complement rule:

$
  Pr("at least one coin has" nu = 0) & = 1 - Pr("all coins have" nu eq.not 0) \
                                     & = 1 - (Pr("a coin has" nu eq.not 0))^1000 \
                                     & = 1 - (1 - Pr("a coin has" nu = 0))^1000 \
                                     & = 1 - (1 - 1 slash 2^10)^1000 \
                                     & approx 62%
$

The probability exceeds 50%, which means it is more likely than not that at least one
coin will look completely biased purely by chance. Every single coin is fair, yet the
sheer number of candidates almost guarantees that one of them will produce a
misleading sample. The lesson for learning is direct: when you search over a large
hypothesis space, you should expect some hypothesis to fit the training data well by
accident, even if it has no real predictive power. In-sample performance, by itself,
cannot be trusted as evidence of genuine learning.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:395 '* Hoeffding Bound: Validation vs Learning'
// Slide: Hoeffding Bound: Validation vs Learning
The coin experiments have a direct counterpart in the Hoeffding bound itself. In
#strong[validation and testing], the goal is to assess how well the chosen hypothesis
$g$ approximates the unknown target function $f$. Because $g$ is already fixed at
this stage (no further selection takes place), Hoeffding's inequality applies
directly:

$ Pr(|E_("in") - E_("out")| > epsilon) lt.eq 2 exp(-2 epsilon^2 N) $

where the in-sample error and out-of-sample error are defined as:

$ E_("in")(g) eq.delta 1/N sum_i e(g(bold(x)_i), f(bold(x)_i)) $

$ E_("out")(g) eq.delta EE_(bold(x))[e(g(bold(x)), f(bold(x)))] $

Since $g$ is final and fixed, the bound limits how far $E_("out")$ can stray from
$E_("in")$ for a given sample size $N$ and tolerance $epsilon$.

During #strong[learning], however, the situation changes. The algorithm searches over
a hypothesis set and selects the best of $M$ candidates, so a union-bound correction
is necessary:

$ Pr(|E_("in") - E_("out")| > epsilon) lt.eq 2 M exp(-2 epsilon^2 N) $

The factor of $M$ accounts for the possibility that at least one of the $M$
hypotheses looks deceptively good on the training sample by chance. This makes the
bound considerably weaker: as the number of candidate hypotheses grows, the
right-hand side inflates, and the guarantee on generalization loosens.

Two natural questions arise from this observation. First, is the bound weak because
it #emph[has] to be, or is the looseness an artifact of the union-bound argument?
Second, is it possible to replace this bound with a tighter one that scales more
gracefully with the complexity of the hypothesis set? These questions motivate the
development of more refined complexity measures (such as the VC dimension) that can
provide meaningful guarantees even when $M$ is very large or infinite.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:423 '* Why the Union Bound Is Loose'
// Slide: Why the Union Bound Is Loose
The Hoeffding inequality, combined with the union bound over a training set, gives us

$ Pr(|E_("in") - E_("out")| > epsilon) lt.eq 2 M exp(-2 epsilon^2 N) $

but this bound is #emph[artificially] too loose. To see why, recall where the factor
$M$ came from. For each hypothesis $h_i$ in a finite set of size $M$, define the "bad
event"

$ cal(B)_i = |E_("in")(h_i) - E_("out")(h_i)| > epsilon $

meaning that $h_i$ does not generalize well out of sample. Letting $cal(B)$ denote
the event that the final chosen hypothesis $g$ does not generalize, and noting that
$g in {h_1, h_2, dots, h_M}$, we can write

$ Pr(cal(B)) lt.eq Pr(union.big_i cal(B)_i) lt.eq sum_i Pr(cal(B)_i) $

The last step is the union bound, and it is tight only when the bad events
$cal(B)_1, cal(B)_2, dots, cal(B)_M$ are mutually disjoint. When they overlap
significantly, the sum double-counts shared probability mass and the resulting
estimate becomes highly conservative.

In practice, the bad events overlap enormously. Consider two perceptrons $g_1$ and
$g_2$ with similar weight vectors, classifying two linearly separable classes on a
plane. The out-of-sample error $E_("out")$ for each hypothesis corresponds to the
region where that hypothesis disagrees with the ground truth. Because $g_1$ and $g_2$
are nearly identical, those disagreement regions differ only in a thin strip: the
differential area $Delta E_("out")$. The in-sample error $E_("in")$ counts training
points falling in each disagreement region, and the differential $Delta E_("in")$
counts only the handful of points in $Delta E_("out")$ that change classification
when moving from one hypothesis to the other. Since $Delta E_("out")$ is small, the
bad events $cal(B)_1$ and $cal(B)_2$ are almost the same event: counting them
separately, as the union bound does, vastly inflates the true probability, as
@fig:whytheunionboundisloose shows.

// rendered_images:begin
// ```tikz[width=90%]
// \bfseries
// % Draw the three overlapping colored circles
// \draw[thick, red] (0,0) circle(2cm);         % B1
// \draw[thick, green] (1,0.5) circle(2cm);     % B2
// \draw[thick, blue] (0.5,-1) circle(2cm);     % B3
// 
// % Colored labels
// \node[text=red] at (-2.3,0) {$\mathcal{B}_1$};
// \node[text=green] at (2.2,0.7) {$\mathcal{B}_2$};
// \node[text=blue] at (0.3,-2.5) {$\mathcal{B}_3$};
// ```
// label=fig:whytheunionboundisloose
// caption=Overlapping bad events counted separately by the union bound.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.5.png",
    width: 90%,
  ),
  caption: [Overlapping bad events counted separately by the union bound.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:whytheunionboundisloose>
// render_images:end

This observation is the key motivation for replacing the naive count $M$ (which can
be infinite for continuous hypothesis sets) with a more refined complexity measure
that accounts for how much hypotheses actually differ on a finite sample.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:490 '# Growth Function'
// Slide: Growth Function
= Growth Function

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:492 '## Dichotomies'
// Slide: Dichotomies
== Dichotomies

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:494 '* Dichotomy: Definition'
// Slide: Dichotomy: Definition
The problem is to classify a fixed set of $N$ points $x_1, ..., x_N$ using a
hypothesis set $cal(H)$ of binary classifiers. Given such a set of points, consider a
particular assignment $D$ of class labels $d_1, ..., d_N$ to those points. A
#strong[dichotomy] for hypothesis set $cal(H)$ is any such assignment $D$ for which
there exists some $h in cal(H)$ that achieves exactly that classification. In other
words, a dichotomy is a labeling of the data that is #emph[realizable] by at least
one hypothesis in the class.

To make this concrete, consider four points $A, B, C, D$ lying in a plane, with
binary class labels (say, $circle.small$ and $times$), and let $cal(H)$ be the set of
all two-dimensional perceptrons (linear classifiers). By moving the separating
hyperplane around, you can produce different classifications of the four points, each
one a distinct dichotomy:

- In dichotomy $D_1$, the classifier might label $A$ as $circle.small$ and $B$, $C$,
  $D$ as $times$.
- In dichotomy $D_2$, it might label $A$ as $times$, $B$ as $times$, $C$ as
  $circle.small$, and $D$ as $circle.small$.
- Further orientations of the hyperplane yield still more dichotomies.

Since each of the $N$ points can independently receive one of two labels, there are
at most $2^N$ possible label assignments in total. However, not all of these
assignments are necessarily achievable by a linear separator. The classic example is
the XOR assignment, where diagonally opposite points share a label: no single line in
the plane can separate $A, C$ (labeled $times$) from $B, D$ (labeled $circle.small$)
when the four points form a square. That assignment is therefore #emph[not] a
dichotomy of the perceptron hypothesis set, because no $h in cal(H)$ realizes it.

// rendered_images:begin
// ```tikz
// \bfseries
// % Draw rectangle
// \draw[thick] (0,0) rectangle (5,3.5);
// 
// % Define coordinates for points
// \coordinate (A) at (2.5,3);   % top circle
// \coordinate (B) at (3.8,2);   % right cross
// \coordinate (C) at (2.5,1);   % bottom circle
// \coordinate (D) at (1,1.2);   % left cross
// 
// % Draw symbols
// \node at (A) {\Large $\circ$};
// \node at (B) {\Large $\times$};
// \node at (C) {\Large $\times$};
// \node at (D) {\Large $\times$};
// 
// % Add labels
// \node[above right] at (A) {$A$};
// \node[above left] at (B) {$B$};
// \node[below right] at (C) {$C$};
// \node[below left] at (D) {$D$};
// 
// % Define coordinates for points
// \coordinate (LineStart) at (0, 2.5);
// \coordinate (LineEnd) at (5, 2.5);
// 
// % Draw a line isolating A from the other points
// \draw[red, dotted, thick] (LineStart) -- (LineEnd);
// ```
// label=fig:dichotomydefinition
// caption=A dichotomy of four points produced by a perceptron.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.6.png",
    width: 70%,
  ),
  caption: [A dichotomy of four points produced by a perceptron.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:dichotomydefinition>
// render_images:end

// rendered_images:begin
// ```tikz
// \bfseries
// % Draw rectangle
// \draw[thick] (0,0) rectangle (5,3.5);
// 
// % Define coordinates for points
// \coordinate (A) at (2.5,3);   % top circle
// \coordinate (B) at (3.8,2);   % right cross
// \coordinate (C) at (2.5,1);   % bottom circle
// \coordinate (D) at (1,1.2);   % left cross
// 
// % Draw symbols
// \node at (A) {\Large $\times$};
// \node at (B) {\Large $\times$};
// \node at (C) {\Large $\circ$};
// \node at (D) {\Large $\circ$};
// 
// % Add labels
// \node[above right] at (A) {$A$};
// \node[above left] at (B) {$B$};
// \node[below right] at (C) {$C$};
// \node[below left] at (D) {$D$};
// 
// % Draw single line
// \draw[red, dotted, thick] (5, 0) -- (0, 3.5);
// ```
// label=fig:dichotomydefinition-2
// caption=A second dichotomy of the same four points.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.7.png",
    width: 70%,
  ),
  caption: [A second dichotomy of the same four points.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:dichotomydefinition-2>
// render_images:end

This distinction is central to measuring the expressiveness of a hypothesis class.
@fig:dichotomydefinition and @fig:dichotomydefinition-2 show how different positions
of the decision boundary produce different dichotomies and how certain label patterns
remain out of reach. The number of dichotomies a hypothesis set can produce on $N$
points, rather than the raw count $2^N$, is what will lead us to the growth function
and, eventually, to the VC dimension as a capacity measure for $cal(H)$.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:587 '* Dichotomies vs Hypotheses'
// Slide: Dichotomies vs Hypotheses
Dichotomies are easy to confuse with hypotheses, so it helps to separate the two. A
#strong[hypothesis] is a function that assigns a label to every point in the entire
input space: $cal(X) arrow.r {-1, +1}$. It is a global classifier, defined
everywhere, regardless of whether we ever observe a particular input. A
#strong[dichotomy], by contrast, is a function defined only on a specific finite set
of points: ${bold(x)_1, dots, bold(x)_N} arrow.r {-1, +1}$. You can think of a
dichotomy as a "mini-hypothesis," a hypothesis whose scope has been narrowed to just
the data points at hand.

This distinction matters because, from the perspective of a training set, only the
dichotomy counts. Two hypotheses that draw wildly different decision boundaries
across the full input space are indistinguishable if they assign the same labels to
every training point. In other words, many hypotheses (potentially infinitely many)
can collapse onto a single dichotomy once we restrict attention to a fixed set of $N$
points. As @fig:dichotomiesvshypotheses shows, several distinct hypothesis functions
may produce the same labeling pattern on a given set of points, so the number of
effectively different behaviors a hypothesis set can exhibit on that set is bounded
by the number of distinct dichotomies it can realize.

// rendered_images:begin
// ```graphviz
// digraph HypothesesToDichotomies {
//   rankdir=LR;
//   bgcolor="transparent";
//   nodesep=0.25;
//   ranksep=0.9;
//   node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
//         penwidth=1.4];
//   edge [color="#8C8C8C", penwidth=1.2];
// 
//   subgraph cluster_hyp {
//     label=<Hypotheses<br/>(infinite)>;
//     fontname="Helvetica"; fontsize=12; fontcolor="#3A4A5C";
//     style="rounded,filled"; fillcolor="#F7F9FB"; color="#C7D0DA"; margin=14;
//     node [fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
//     h4 [label=<h<SUB>4</SUB>>];
//     h3 [label=<h<SUB>3</SUB>>];
//     h2 [label=<h<SUB>2</SUB>>];
//     h1 [label=<h<SUB>1</SUB>>];
//   }
// 
//   subgraph cluster_dich {
//     label=<Dichotomies<br/>(at most 2<SUP>N</SUP>)>;
//     fontname="Helvetica"; fontsize=12; fontcolor="#3A4A5C";
//     style="rounded,filled"; fillcolor="#F7F9FB"; color="#C7D0DA"; margin=14;
//     node [fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     d2 [label=<D<SUB>2</SUB>>];
//     d1 [label=<D<SUB>1</SUB>>];
//   }
// 
//   h1 -> d1;
//   h2 -> d1;
//   h3 -> d2;
//   h4 -> d2;
// }
// ```
// label=fig:dichotomiesvshypotheses
// caption=Many hypotheses collapsing onto few dichotomies.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.8.png",
    width: 70%,
  ),
  caption: [Many hypotheses collapsing onto few dichotomies.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:dichotomiesvshypotheses>
// render_images:end

This collapse is precisely why dichotomies become the right unit of analysis when we
try to measure the "effective" complexity of a hypothesis set. The full hypothesis
set may be infinite, yet the number of dichotomies it generates on $N$ points is
always at most $2^N$: a finite, combinatorial quantity we can reason about. Counting
dichotomies rather than hypotheses is the key step toward replacing the (potentially
infinite) hypothesis-set size with a measure that reflects what actually matters for
learning from finite data.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:646 '* Number of Dichotomies'
// Slide: Number of Dichotomies
Since dichotomies are the right unit of analysis, the next step is to count them. The
#strong[number of different dichotomies] on a set of points is written
$|cal(H)(bold(x)_1, ..., bold(x)_N)|$. This count depends on three things: the number
of points $N$, the hypothesis set $cal(H)$ (that is, which family of models we are
considering), and the geometric arrangement of the points themselves. Two hypothesis
sets of the same size can produce very different numbers of dichotomies on the same
points if those hypotheses carve up the input space differently.

A key observation is that the number of dichotomies is always finite, since
$|cal(H)(bold(x)_1, ..., bold(x)_N)| lt.eq 2^N$. There are at most $2^N$ ways to
assign binary labels to $N$ points, so no hypothesis set can produce more dichotomies
than that ceiling. By contrast, the hypothesis set itself is usually infinite:
$|cal(H)| = infinity$. A linear classifier in $bb(R)^2$, for instance, corresponds to
an uncountable family of lines, yet on any finite collection of points it realizes
only a finite number of labelings.

This distinction matters because the "complexity" of $cal(H)$ is tied to how many
dichotomies it can generate, not to how many hypotheses it contains. An infinite
hypothesis set that can shatter large point configurations is more complex, in a
learning-theoretic sense, than one that cannot, even if both sets are equally
infinite. Counting dichotomies rather than hypotheses gives us a finite,
data-dependent measure of capacity that connects directly to generalization bounds.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:661 '## The Growth Function'
// Slide: The Growth Function
== The Growth Function

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:663 '* Growth Function: Definition'
// Slide: Growth Function: Definition
The #strong[growth function] counts the maximum number of distinct dichotomies that a
hypothesis set $cal(H)$ can produce on $N$ points:

$
  m_(cal(H))(N) eq.delta max_(bold(x)_1, dots, bold(x)_N in cal(X)) |cal(H)(bold(x)_1, dots, bold(x)_N)|
$

Why take a maximum? The number of dichotomies a hypothesis set can realize depends on
where the points happen to sit. Some placements may let $cal(H)$ distinguish many
labelings; others may not. The growth function resolves this ambiguity by choosing
the placement that is most favorable to $cal(H)$, giving us a single,
placement-independent measure of the hypothesis set's expressive reach.

Computing $m_(cal(H))(N)$ by brute force is conceptually straightforward but
expensive:

+ Enumerate all possible placements of $N$ points $bold(x)_1, dots, bold(x)_N$ in the
  input space.
+ For each placement, apply every hypothesis $h in cal(H)$ to produce its labeling of
  those $N$ points.
+ Collect the resulting dichotomies and count the distinct ones.
+ Take the maximum count over all placements.

In practice this exhaustive procedure is infeasible for all but the simplest
hypothesis sets, which is exactly why the theoretical bounds we develop next are so
valuable: they let us reason about $m_(cal(H))(N)$ without ever enumerating
placements or hypotheses directly.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:684 '* What Can Vary in a Dichotomy'
// Slide: What Can Vary in a Dichotomy
Before turning to examples, it helps to pin down what the definition of a dichotomy
holds fixed and what it lets vary. Consider a fixed hypothesis set $cal(H)$ (for
example, the set of all two-dimensional perceptrons), a fixed number $N$ of input
points $bold(x)_1, dots, bold(x)_N$, and a particular assignment $D$ of those points
to classes $bold(d)_1, dots, bold(d)_N$. A dichotomy is one way that some hypothesis
$h in cal(H)$ can split those $N$ points into two groups, so the definition of
dichotomy involves several distinct quantities, and it helps to be precise about
which ones are free and which ones are pinned down.

The hypothesis set $cal(H)$ itself is fixed: we choose it once (say, linear
classifiers in $RR^2$) and leave it alone. Because $cal(H)$ determines the kind of
inputs it accepts, the dimensionality of the input space is also fixed implicitly.
The number of points $N$ is the argument we feed to the growth function
$m_(cal(H))(N)$, so it varies from one evaluation of that function to another, but
within any single evaluation it is treated as given.

That leaves two free parameters. The first is how the points are labeled: the class
assignment $bold(d)_1, dots, bold(d)_N$. This freedom is absorbed by the way each
hypothesis in $cal(H)$ partitions the space; we simply count how many distinct
labelings the hypotheses can produce, rather than asking about one labeling in
particular. The second free parameter is where the points sit: their positions
$bold(x)_1, dots, bold(x)_N$. The growth function removes this freedom by taking a
maximum over all possible placements of the $N$ points, so $m_(cal(H))(N)$ reports
the largest number of dichotomies achievable for any configuration. Together, these
two maximizations (over labelings via the hypothesis set, and over point positions
via the $max$) ensure that the growth function captures the worst-case expressive
power of $cal(H)$ as a function of $N$ alone.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:704 '* Growth Function Is Non-Decreasing'
// Slide: Growth Function Is Non-Decreasing
The growth function $m_(cal(H))(N)$ is non-decreasing in $N$: adding more points to
the dataset can never reduce the number of distinct dichotomies, because every
dichotomy achievable on a smaller set extends to at least one dichotomy on the larger
set. For instance, $m_(cal(H))(3) lt.eq m_(cal(H))(4)$, since each dichotomy on three
points corresponds to at least one dichotomy on four points once we ignore the label
assigned to the new point.

Beyond dependence on the sample size $N$, the growth function also increases with the
complexity of the hypothesis class $cal(H)$. A richer, more expressive hypothesis
class can shatter more point configurations, producing a larger count of realizable
dichotomies for any fixed $N$. Similarly, $m_(cal(H))(N)$ grows with the
dimensionality of the input space: higher-dimensional feature spaces give hypotheses
more geometric freedom to separate points in distinct ways, which in turn increases
the number of achievable dichotomies.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:718 '* Growth Function: Examples (1/2)'
// Slide: Growth Function: Examples (1/2)
To make the growth function concrete, consider it for different hypothesis sets
$cal(H)$.

The #strong[perceptron on a plane] provides a familiar starting point. With three
points, every possible dichotomy is achievable, so $m_(cal(H))(3) = 8 = 2^3$. With
four points, however, the perceptron cannot realize XOR-type classifications (where
diagonally opposite points share the same label), which means $m_(cal(H))(4) = 14$
rather than the full $2^4 = 16$. This is the first concrete sign that a linear
separator's expressive power is limited: two of the sixteen possible labelings are
simply unreachable, as @fig:growthfunctionexamples12 shows.

// rendered_images:begin
// ```tikz[width=70%]
// \bfseries
// % Draw rectangle
// \draw[thick] (0,0) rectangle (5,3.5);
// 
// % Define coordinates for points
// \coordinate (A) at (2.5,3);   % top circle
// \coordinate (B) at (3.8,2);   % right cross
// \coordinate (C) at (2.5,1);   % bottom circle
// \coordinate (D) at (1,1.2);   % left cross
// 
// % Draw symbols
// \node at (A) {\Large $\circ$};
// \node at (B) {\Large $\times$};
// \node at (C) {\Large $\circ$};
// \node at (D) {\Large $\times$};
// 
// % Add labels
// \node[above right] at (A) {$A$};
// \node[above left] at (B) {$B$};
// \node[below right] at (C) {$C$};
// \node[below left] at (D) {$D$};
// 
// % Define coordinates for points
// \coordinate (TopCircle) at (5, 0);
// \coordinate (BottomCircle) at (0, 3.5);
// 
// % Draw single line between the circles
// \draw[red, dotted, thick] (TopCircle) -- (BottomCircle);
// ```
// label=fig:growthfunctionexamples12
// caption=Four points and a separating line for the perceptron.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.9.png",
    width: 70%,
  ),
  caption: [Four points and a separating line for the perceptron.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:growthfunctionexamples12>
// render_images:end

#strong[Positive rays] offer an even simpler hypothesis class. Each hypothesis takes
the form $h(x) = "sign"(x - a)$ on $bb(R)$, labeling everything to the right of a
threshold $a$ as $+1$ and everything to the left as $-1$. Given $N$ sorted points on
the real line, the threshold $a$ can sit in any of the $N + 1$ intervals created by
those points (including the two open intervals beyond the outermost points), plus one
additional dichotomy arises from placing $a$ so that all points receive the same
label. The result is $m_(cal(H))(N) = N + 1$, which grows only linearly in $N$, far
below the exponential $2^N$ ceiling. @fig:growthfunctionexamples12-2 shows how the
$N + 1$ possible threshold placements correspond to distinct dichotomies. Because
this growth function is polynomial rather than exponential, positive rays are a
severely restricted hypothesis class, and that restriction is precisely what makes
learning with them feasible even from modest amounts of data.

// rendered_images:begin
// ```tikz
// \bfseries
//     % Draw axis
//     \draw[thick,->] (-1,0) -- (8,0) node[right] {};
// 
//     % Draw negative samples (crosses)
//     \foreach \i in {0, 1, 2, 3} {
//         \draw[thick, red] (\i,0) node[below=3pt] {$x_{\the\numexpr\i+1}$} node {\textsf{x}};
//     }
//     \node at (3.5, -0.3) {$\cdots$};
// 
//     % Draw decision boundary
//     \draw[thick, dotted, blue] (4.5,-0.3) -- (4.5,1.2) node[above] {$a$};
// 
//     % Draw positive samples (circles)
//     \foreach \i in {5, 6, 7} {
//         \draw[thick, blue] (\i,0) circle (3pt);
//     }
//     \node at (7,0) [below=3pt] {$x_N$};
// 
//     % Labels for h(x)
//     \node at (2,0.8) {$h(x) = -1$};
//     \node at (6,0.8) {$h(x) = +1$};
//     \draw[thick,blue,->] (4.5,0.4) -- (7,0.4);
// ```
// label=fig:growthfunctionexamples12-2
// caption=Positive rays on the real line.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.10.png",
    width: 70%,
  ),
  caption: [Positive rays on the real line.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:growthfunctionexamples12-2>
// render_images:end

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:805 '* Growth Function: Examples (2/2)'
// Slide: Growth Function: Examples (2/2)
#strong[Positive intervals] on $bb(R)$ are another example. Here the hypothesis class
$cal(H)$ consists of all intervals $[a, b]$, labeling points inside the interval as
$+1$ and all others as $-1$. Given $N$ points on the real line, the two endpoints
$a < b$ can be placed into any two distinct gaps among the $N + 1$ gaps between (and
around) the sorted points, yielding $binom(N + 1, 2)$ dichotomies. Adding one more
dichotomy for the case where both endpoints fall into the same gap (so that every
point receives label $-1$) gives the growth function

$ m_(cal(H))(N) = binom(N + 1, 2) + 1 tilde N^2. $

This is polynomial in $N$, a much slower growth rate than the $2^N$ ceiling.
@fig:growthfunctionexamples22 shows how the interval endpoints partition the point
set into positive and negative regions.

// rendered_images:begin
// ```tikz
// \bfseries
// % Draw axis
// \draw[very thick] (-0.5,0) -- (9,0);
// 
// % Draw negative samples (crosses)
// \foreach \i/\name in {0/x_1, 1/x_2, 2/x_3} {
//     \draw[thick, red] (\i,0) node[below=3pt] {$\mathit{\name}$} node {\textsf{x}};
// }
// \node at (3, -0.3) {$\cdots$};
// 
// % Draw positive samples (circles)
// \foreach \i in {4, 5, 6} {
//     \draw[thick, blue] (\i,0) circle (3pt);
// }
// 
// % Draw final negative example
// \draw[thick, red] (7,0) node {\textsf{x}};
// \draw[red] node at (7, -0.3) {$x_N$};
// 
// % Draw brackets indicating h(x)=+1 region
// \draw[very thick,blue,<->] (3.6,0.5) -- (6.4,0.5);
// \draw[very thick,blue,rounded corners] (3.6,0.4) -- (3.6,0.6);
// \draw[very thick,blue,rounded corners] (6.4,0.4) -- (6.4,0.6);
// 
// % Labels for h(x)
// \node at (1.5, 0.9) {\color{red}$h(x) = -1$};
// \node at (5, 0.9) {\color{blue}$h(x) = +1$};
// \node at (8.3, 0.9) {\color{red}$h(x) = -1$};
// ```
// label=fig:growthfunctionexamples22
// caption=Positive intervals on the real line.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.11.png",
    width: 70%,
  ),
  caption: [Positive intervals on the real line.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:growthfunctionexamples22>
// render_images:end

#strong[Convex sets on a plane] sit at the opposite extreme. When the hypothesis
class labels a point $+1$ if and only if it lies inside some convex region, and we
arrange $N$ points on a circle, every possible labeling can be realized: simply take
the convex hull of whichever subset should be positive. Because any subset of points
on a circle is in convex position, this construction shatters every set of $N$
points, so

$ m_(cal(H))(N) = 2^N. $

The growth function therefore equals the theoretical maximum for every $N$, meaning
convex sets impose no effective restriction on the number of achievable dichotomies.
As @fig:growthfunctionexamples22-2 shows, placing points on a circle makes this
shattering argument visually clear: for any coloring of the vertices, the convex hull
of the $+1$ vertices cleanly separates them from the rest.

// rendered_images:begin
// ```tikz[width=70%]
// \bfseries
// % Circle radius
// \def\r{3}
// 
// % Draw the outer circle
// \draw[thick] (0,0) circle (\r);
// 
// % Draw the shaded polygonal region inside
// \fill[gray!20,opacity=0.8]
//     ({\r*cos(250)},{\r*sin(250)}) --
//     ({\r*cos(290)},{\r*sin(290)}) --
//     ({\r*cos(30)},{\r*sin(30)}) --
//     ({\r*cos(80)},{\r*sin(80)}) -- cycle;
// 
// % Label inside region
// \node at (0.7,0) {$h(x) = +1$};
// 
// % Draw the points (alternating red circles and blue crosses)
// \foreach \i in {0,...,11} {
//     \pgfmathsetmacro{\angle}{\i * 30}
//     \pgfmathsetmacro{\x}{\r*cos(\angle)}
//     \pgfmathsetmacro{\y}{\r*sin(\angle)}
//     \ifodd\i
//         \node[text=blue] at (\x,\y) {\textsf{x}};
//     \else
//         \draw[thick, red] (\x,\y) circle (3pt);
//     \fi
// }
// ```
// label=fig:growthfunctionexamples22-2
// caption=Convex sets on a plane with points on a circle.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.1-Learning_Theory.typ.figs/Lesson05.1-Learning_Theory.12.png",
    width: 70%,
  ),
  caption: [Convex sets on a plane with points on a circle.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:growthfunctionexamples22-2>
// render_images:end

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:899 '## Break Points'
// Slide: Break Points
== Break Points

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:901 '* Shattering and Break Points'
// Slide: Shattering and Break Points
Given a hypothesis set $cal(H)$, we say that $cal(H)$ #strong[shatters $N$ points] if
and only if $m_(cal(H))(N) = 2^N$. In concrete terms, shattering means there exists
some arrangement of $N$ points for which every one of the $2^N$ possible class
assignments (every binary labeling of those points) can be realized by at least one
hypothesis $h in cal(H)$. Shattering $N$ points requires only that #emph[some]
placement of $N$ points achieves this, not that #emph[every] set of $N$ points can be
classified in every possible way. A single favorable configuration is enough.

The concept of a #strong[break point] captures where that shattering power runs out.
We say $k$ is a break point for $cal(H)$ if and only if $m_(cal(H))(k) < 2^k$,
meaning no data set of size $k$ can be shattered by $cal(H)$. Once a hypothesis set
fails to shatter some size $k$, the growth function can no longer reach the
exponential ceiling at that size, which, as we will see, constrains how fast
$m_(cal(H))(N)$ can grow for all $N gt.eq k$.

The examples from our earlier hypothesis sets show the range of possibilities:

- #emph[2D perceptron]: a break point is $k = 4$, since three non-collinear points
  can be shattered but no arrangement of four points can.
- #emph[Positive rays]: a break point is $k = 2$, reflecting the very limited
  expressive power of a single threshold on the real line.
- #emph[Positive intervals]: a break point is $k = 3$, one step up from positive rays
  thanks to the extra degree of freedom in choosing both endpoints of the interval.
- #emph[Convex sets on a plane]: there is no break point at all. For any $N$, we can
  place $N$ points on a circle and shatter them (any labeled subset sits inside its
  own convex hull), so $m_(cal(H))(N) = 2^N$ for every $N$. This means convex sets
  have infinite VC dimension, a warning sign that the hypothesis class is too rich
  for the growth function alone to guarantee generalization.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:918 '* Break Points and the VC Inequality'
// Slide: Break Points and the VC Inequality
Break points matter because of what they imply for the growth function. If a break
point exists for a hypothesis set $cal(H)$, the growth function $m_(cal(H))(N)$ is
polynomial in $N$. This means we can replace the raw count $M$ of hypotheses (which
may be infinite) with something that grows only polynomially. Recall Hoeffding's
inequality applied to learning:

$ Pr(|E_("in")(g) - E_("out")(g)| > epsilon) lt.eq 2 M e^(-2 epsilon^2 N) $

When $M$ is infinite, the right-hand side is meaningless. The
#strong[Vapnik-Chervonenkis (VC) inequality] #cite(
  "vapnikchervonenkis1971uniform",
) replaces $M$ with the growth function evaluated at $2N$:

$ Pr("bad generalization") lt.eq 4 m_(cal(H))(2N) e^(-1/8 epsilon^2 N) $

The factor $m_(cal(H))(2N)$ appears at twice the sample size because the proof
technique works by comparing $E_("in")$ on the original sample with $E_("in")$ on a
second, independent "ghost sample" of the same size; reasoning about bad events then
requires counting dichotomies on the combined $2N$ points.

Why does this inequality actually help? Because a polynomial times a decaying
exponential still decays to zero. The exponential term $e^(-1/8 epsilon^2 N)$ shrinks
far faster than $m_(cal(H))(2N)$ grows, so for large enough $N$ the bound becomes
tight: the probability of bad generalization can be driven as low as we like by
collecting more data. This is exactly the generalization guarantee that was missing
when we tried to use the raw hypothesis count $M$. With the VC inequality in hand, we
can say that learning works for any hypothesis set that has a finite break point,
regardless of whether that set contains finitely or infinitely many hypotheses.

A hypothesis set can be characterized, from the standpoint of learnability, by the
#emph[existence and value of its break point]. If a break point $k$ exists, the
growth function is bounded by $N^(k-1) + 1$, a polynomial whose degree depends only
on $k$, and the VC inequality guarantees generalization. If no break point exists,
the growth function equals $2^N$ and no such guarantee is available. The break point
(or equivalently the VC dimension, which is $k - 1$) therefore serves as the single
number that separates hypothesis sets that can learn from those that cannot.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:976 '# The VC Dimension'
// Slide: The VC Dimension
= The VC Dimension

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:978 '* VC Dimension: Definition'
// Slide: VC Dimension: Definition
The #strong[VC dimension] of a hypothesis set $cal(H)$, denoted $d_("VC")(cal(H))$,
is the largest value of $N$ for which $m_(cal(H))(N) = 2^N$. In other words, it is
the maximum number of points that $cal(H)$ can shatter #cite(
  "shalevshwartz2014understanding",
). This single number captures the effective complexity of $cal(H)$: a larger VC
dimension means the hypothesis set can realize more diverse labelings and therefore
has greater expressive power, but also greater capacity to overfit.

When $d_("VC")(cal(H)) = N$, several properties follow directly:

- There #emph[exists] some constellation of $N$ points that $cal(H)$ can shatter, but
  not every arrangement of $N$ points need be shatterable. If $N$ points were placed
  at random, there is no guarantee they could be shattered.
- $cal(H)$ can shatter any set of $N'$ points for every $N' lt.eq d_("VC")(cal(H))$:
  shattering is monotone up to the VC dimension.
- The #emph[smallest break point] is $d_("VC") + 1$: this is the first value of $N$
  at which no arrangement of $N$ points can be fully shattered.
- The #emph[growth function] is bounded in terms of the VC dimension by Sauer's lemma
  #cite("sauer1972density"):
  $ m_(cal(H))(N) lt.eq sum_(i=0)^(d_("VC")) binom(N, i) $
  Because this sum is a polynomial of degree $d_("VC")$ in $N$, the VC dimension is
  precisely the #emph[order of the polynomial] that bounds $m_(cal(H))$. Once $N$
  exceeds the VC dimension, the growth function can no longer be exponential; it is
  forced to grow polynomially, which is what makes generalization guarantees
  possible.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:1002 '* VC Dimension: Examples'
// Slide: VC Dimension: Examples
The hypothesis sets examined earlier can now be ranked by their VC dimension, which
provides a clean summary of each one's capacity. @tab:vcdimensionexamples collects
the growth function, break point, and VC dimension for the main examples: positive
rays, positive intervals, convex sets, and perceptrons. The pattern is simple:
hypothesis sets with a finite break point have a polynomial growth function and a
finite VC dimension, while those without a break point (such as convex sets in
$RR^2$) grow exponentially and have an infinite VC dimension.

#figure(
  styled-table(
    headers: (
      "Hypothesis set",
      "Growth function",
      "Break point",
      "VC dimension",
    ),
    rows: (
      ([Positive rays], [$m_H (N) = N + 1$], [2], [1]),
      ([Positive intervals], [$m_H (N) = binom(N + 1, 2) + 1$], [3], [2]),
      ([Perceptron on a plane], [$m_H (3) = 8, m_H (4) = 14$], [4], [3]),
      ([Convex sets on a plane], [$m_H (N) = 2^N$], [none], [$oo$]),
    ),
  ),
  caption: [Break points and VC dimensions of the example hypothesis sets.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:vcdimensionexamples>

The VC dimension $d_("VC")$ is finite if and only if $cal(H)$ has a break point. When
a break point exists, the growth function is bounded by a polynomial in $N$, which in
turn guarantees that the generalization bound holds and learning from a finite sample
is feasible. Conversely, if no break point exists, the growth function equals $2^N$
for every $N$, the bound becomes vacuous, and no finite dataset can reliably
distinguish good hypotheses from bad ones.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:1024 '* VC Dimension: Interpretation'
// Slide: VC Dimension: Interpretation
Beyond the yes-or-no question of learnability, the VC dimension has a graded reading:
it measures the complexity of a hypothesis set in terms of #emph[effective
  parameters], capturing how expressive a model is regardless of how many raw
parameters it carries.

For instance, a perceptron operating in a $d$-dimensional input space has
$d_("VC") = d + 1$, which happens to equal the number of perceptron parameters (the
$d$ weights plus one bias). In the concrete case of a 2D perceptron ($d = 2$), the
break point is 4, so $d_("VC") = 3$: the perceptron can shatter any three points in
the plane but never all four.

The VC dimension treats the model as a black box. It asks how many points $N$ the
model can shatter, not how many tunable coefficients appear in its parameterization.
This distinction matters because not all parameters contribute equally to a model's
degrees of freedom. Consider combining $N$ one-dimensional perceptrons: the resulting
system has $2N$ parameters in total, yet its effective degrees of freedom remain
just 2. The extra parameters are redundant; they do not increase the set of
dichotomies the model can realize.

A more complex hypothesis set $cal(H)$ carries a higher VC dimension $d_("VC")$,
reflecting greater expressive power. That power comes at a cost: more training
examples are needed to pin down a good hypothesis within that larger set. This is the
bias-variance tradeoff made precise through the VC framework: richer models can fit
more patterns but demand correspondingly more data to generalize reliably.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:1044 '## VC Generalization Bounds'
// Slide: VC Generalization Bounds
== VC Generalization Bounds

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:1046 '* VC Generalization Bounds: Bound Term'
// Slide: VC Generalization Bounds: Bound Term
How many data points do we actually need to guarantee that in-sample error tracks
out-of-sample error to within $epsilon$, with failure probability at most $delta$?
The VC inequality gives a concrete, if loose, answer. It states that

$ Pr(|E_"in" - E_"out"| > epsilon) lt.eq 4 m_(cal(H))(2N) e^(-1/8 epsilon^2 N) $

The right-hand side captures the tension between model complexity (through the growth
function $m_(cal(H))(2N)$) and the amount of training data $N$. To make the failure
probability at most $delta$, we set the bound equal to $delta$ and solve for $N$.

The bound's behavior is governed by a term that looks like $N^d e^(-N)$, where $d$ is
the VC dimension. For small $N$, the polynomial $N^d$ dominates and the bound grows;
for large $N$, the exponential $e^(-N)$ takes over and drives the whole expression
toward zero. This is the classic polynomial-versus-exponential race: the exponential
always wins eventually, but a larger $d$ delays the crossover. As
@fig:vcgeneralizationbounds shows, increasing the VC dimension shifts the peak of
$N^d e^(-N)$ to larger values of $N$, meaning a more complex hypothesis set requires
more data before the bound drops below 1 and becomes meaningful. Once $N$ is large
enough, however, the exponential decay guarantees that the bound shrinks, and
generalization is assured.

#figure(
  image(
    "../lectures_source/figures/L05.1.VC_Generalization_Bounds.png",
    width: 80%,
  ),
  caption: [VC Generalization Bounds],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:vcgeneralizationbounds>

In practice, the VC bound is loose: the sample sizes it prescribes are far larger
than what real learning algorithms need. Its value is theoretical rather than
operational; it confirms that generalization #emph[is] achievable for any
finite-VC-dimension hypothesis set, given enough data, even though the "enough" it
quotes is a pessimistic worst case.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:1074 '* VC Generalization Bounds: Sample Size'
// Slide: VC Generalization Bounds: Sample Size
So how much data is "enough"? The number of training examples required for reliable
generalization grows with the VC dimension. Plotting the intersection of $N^d e^(-N)$
with a target probability as a function of $d$ reveals that the needed sample size
$N$ is roughly proportional to $d$. A widely used rule of thumb, drawn from this
analysis, is that $N gt.eq 10 d_("VC")$ suffices for meaningful generalization #cite(
  "abumostafa2012learning",
). @fig:vcgeneralizationbounds2 shows this relationship, showing how the required
sample size scales linearly with the VC dimension to keep the generalization bound
tight.

#figure(
  image(
    "../lectures_source/figures/L05.1.VC_Generalization_Bounds2.png",
    width: 80%,
  ),
  caption: [VC Generalization Bounds2],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:vcgeneralizationbounds2>

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:1092 '* VC Generalization Bounds: Error Bar'
// Slide: VC Generalization Bounds: Error Bar
Solving for the sample size is only one way to read the VC inequality. The VC
inequality

$
  Pr(|E_("in") - E_("out")| > epsilon) lt.eq 4 m_(cal(H))(2N) e^(-1/8 epsilon^2 N)
$

can be used in several ways to relate the tolerance $epsilon$, the confidence level
$delta$, and the sample size $N$. For instance, one might ask: "Given
$epsilon = 0.01$ error tolerance, how many examples $N$ are needed to achieve
$delta = 0.05$?" Alternatively, one could fix $N$ and ask what probability remains
for an error exceeding $epsilon$.

The key manipulation is to set the right-hand side equal to $delta$ and solve for
$epsilon$. Doing so yields the #strong[generalization bound function]

$ Omega(N, cal(H), delta) eq.delta sqrt(8/N ln (4 m_(cal(H))(2N)) / delta) $

With this quantity in hand, the VC inequality becomes a confidence statement:
$|E_("out") - E_("in")| lt.eq Omega(N, cal(H), delta)$ holds with probability at
least $1 - delta$. Equivalently, the out-of-sample error satisfies

$ Pr(E_("out") lt.eq E_("in") + Omega) gt.eq 1 - delta $

This is the standard form of the VC generalization bound. It says that the test error
is at most the training error plus a penalty term $Omega$ that shrinks as $N$ grows
and swells as the hypothesis set $cal(H)$ becomes more complex (through the growth
function $m_(cal(H))$). The bound makes the tradeoff between sample size and model
complexity explicit: a richer $cal(H)$ drives $E_("in")$ down but pushes $Omega$ up,
so the tightest guarantee on $E_("out")$ comes from balancing the two.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:1118 '## Limits of the VC Guarantee'
// Slide: Limits of the VC Guarantee
== Limits of the VC Guarantee

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:1120 '* Data Snooping Voids the VC Guarantee'
// Slide: Data Snooping Voids the VC Guarantee
Consider the case where the data is non-linear: for instance, "o" points clustered in
the center of the feature space and "x" points scattered in the corners. No linear
separator in the original space can distinguish these two classes cleanly. The
standard remedy is to transform the input into a higher-dimensional space $cal(Z)$
via a feature map

$ Phi: bold(x) = (x_0, dots, x_d) arrow.r bold(z) = (z_0, dots, z_(tilde(d))) $

so that a linear boundary in $cal(Z)$ corresponds to a non-linear boundary back in
the original space.

The VC dimension of the linear model in the transformed space satisfies
$d_("VC") lt.eq tilde(d) + 1$, so keeping $tilde(d)$ small directly improves the
generalization bound. A natural starting point for a two-dimensional input is the
full quadratic expansion $bold(z) = (1, x_1, x_2, x_1 x_2, x_1^2, x_2^2)$, which
gives $tilde(d) = 5$. But do we really need all six coordinates? If the boundary is a
circle centered at the origin, only the squared terms matter, so
$bold(z) = (1, x_1^2, x_2^2)$ suffices: $tilde(d) = 2$. Pushing further, if the
radius is the only degree of freedom, we can collapse the two squared terms into
their sum, $bold(z) = (1, x_1^2 + x_2^2)$: $tilde(d) = 1$. And if we already know the
threshold radius (say $0.6$), a single feature $bold(z) = (x_1^2 + x_2^2 - 0.6)$
reduces the problem to $tilde(d) = 0$, a trivial hypothesis set with the best
possible generalization guarantee.

Each simplification, however, required us to look at the data and tailor the feature
map to what we saw. Some model coefficients turned out to be zero and were discarded,
but this pruning was guided by the specific dataset at hand, not by a priori domain
knowledge. The VC analysis that underwrites our generalization bound is a
#emph[warranty]: it applies to the complexity of the #emph[initial] hypothesis set,
the one chosen before any data was examined. The moment we peek at the data to select
a simpler feature map, we forfeit that warranty, a pitfall known as #strong[data
  snooping] #cite(
  "abumostafa2012learning",
). As @fig:voidvcguarantee shows, the VC guarantee becomes void once the hypothesis
set is shaped by the same data used for training. From the standpoint of VC analysis,
the effective complexity remains that of the original, richer hypothesis set,
regardless of how few features survived in the final model.

#figure(
  image("../lectures_source/figures/L05.1.Void_VC_guarantee.png", width: 80%),
  caption: [Void VC guarantee],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:voidvcguarantee>

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:1151 '* Summary'
// Slide: Summary
= Summary

Learning is feasible because, for a fixed hypothesis, in-sample error tracks
out-of-sample error. The #strong[VC dimension] extends this guarantee from a single
hypothesis to an entire hypothesis set, even one containing infinitely many
hypotheses.

The target function is unknown outside the training data, so learning is not certain
but only #emph[probable]: feasibility is inherently a statistical statement. The
#emph[Hoeffding inequality] bounds the gap $|E_("in") - E_("out")|$ for any single
fixed hypothesis, but when the learner chooses among $M$ hypotheses, a union bound
weakens the guarantee by a factor of $M$, making it vacuous for large or infinite
hypothesis sets. The #emph[growth function] resolves this by counting the distinct
dichotomies (labelings) a hypothesis set can actually produce on $N$ points; once a
#emph[break point] exists, that count grows only polynomially in $N$, replacing the
exponential $M$ with a manageable quantity. The #emph[VC dimension] $d_("VC")$
distills everything into a single number: the largest $N$ such that $cal(H)$ can
shatter (realize all $2^N$ dichotomies of) those $N$ points. It serves as a measure
of the effective number of parameters in the model, yields the VC generalization
inequality, and motivates the practical rule of thumb $N gt.eq 10 d_("VC")$ for the
amount of data needed.

One caveat: the entire guarantee rests on the hypothesis set being fixed before the
data are examined. If the learner peeks at the data and then adjusts which hypotheses
to consider, the statistical contract is broken and the VC bound no longer applies.
This is why #emph[data snooping], in any form, voids the theoretical justification
for generalization.

// From: msml610/lectures_source/Lesson05.1-Learning_Theory.smd:1171 '* References'
// Slide: References
= References

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
