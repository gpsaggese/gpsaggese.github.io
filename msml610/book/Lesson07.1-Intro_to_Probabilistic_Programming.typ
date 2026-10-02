// git_hash=7e94b8cb-axw timestamp=20261002_081748
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
  title: "L07.1: Introduction to Probabilistic Programming",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L07.1: Introduction to Probabilistic Programming")

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:27 '* Roadmap'
// Slide: Roadmap
= Roadmap

This chapter begins by examining why #strong[thinking probabilistically] matters:
Bayesian inference offers a coherent framework for updating beliefs with data, rather
than a disconnected collection of statistical recipes. Bayes' theorem provides the
machinery for turning observed data into updated beliefs about model parameters, and
understanding its logic is the foundation for everything that follows.

From there, the chapter works through the #strong[coin-flipping example
  analytically]. By pairing a binomial likelihood with a beta prior, conjugacy
delivers a closed-form posterior for the coin's bias parameter. This example is small
enough to solve exactly, making it the ideal setting for building intuition about how
priors, likelihoods, and posteriors interact.

The discussion then turns to #strong[priors and credible intervals]. Explicit priors
are not a nuisance; they are a feature, encoding what is known (or assumed) before
the data arrive. The chapter covers how to choose priors thoughtfully and how to
interpret Bayesian credible intervals, contrasting them with the frequentist
confidence intervals most readers already know.

Finally, the chapter addresses #strong[probabilistic programming]. Most real-world
models lack the convenient conjugacy that gave us a closed-form answer for the coin.
When no analytical solution exists, the practical path forward is to express the
model in code and solve it numerically. The chapter introduces `PyMC` as the tool for
that job, showing how to specify a model, draw posterior samples, and interpret the
results.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:44 '# Thinking Probabilistically'
// Slide: Thinking Probabilistically
= Thinking Probabilistically

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:58 '## Statistics and Data'
// Slide: Statistics and Data
== Statistics and Data

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:60 '* EDA vs Inferential Statistics'
// Slide: EDA vs Inferential Statistics
#strong[Exploratory data analysis (EDA)] #cite("tukey1977eda") is the practice of
visually inspecting, summarizing, interpreting, and checking a dataset before any
formal modeling begins. It includes computing descriptive statistics and
communicating the results to stakeholders. The goal is to build intuition about the
data's structure, spot anomalies, and identify patterns that guide later analysis.

#strong[Inferential statistics] goes a step further: rather than merely describing
what is in the sample, it draws insights from a limited set of observations to make
predictions about future, unobserved data points. Inferential methods aim to
understand the phenomenon that generated the data and to choose among competing
explanations for the same observations. Where EDA asks "what does this dataset look
like?", inferential statistics asks "what can this dataset tell us about the world
beyond itself?"

As @fig:edavsinferentialstatistics shows, both disciplines start from the same
limited sample of data but diverge in purpose: EDA describes the sample, while
inferential statistics uses the sample to reason about the broader population or
process it was drawn from.

// rendered_images:begin
// ```graphviz
// digraph EDAvsInference {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.4;
//     ranksep=0.55;
//     rankdir=TB;
// 
//     // source : orange | core process : blue | output : sage green
//     node [shape=box,
//           style="rounded,filled",
//           penwidth=1.8,
//           fontname="Helvetica",
//           fontsize=12,
//           margin="0.22,0.14",
//           height=0.50];
// 
//     edge [color="#A3B1C0",
//           penwidth=1.3,
//           arrowhead=vee,
//           arrowsize=0.75,
//           fontname="Helvetica",
//           fontsize=10,
//           fontcolor="#7B8794"];
// 
//     data  [label="Limited sample\nof data", fillcolor="#FBEBD4", color="#D9A85F", fontcolor="#6B4517"];
//     eda   [label="EDA", fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//     inf   [label="Inferential\nstatistics", fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//     desc  [label="Describe\nthe sample", fillcolor="#DFEDE0", color="#8FB79A", fontcolor="#2E5A3D"];
//     gen   [label="Generalize to\nunobserved data", fillcolor="#DFEDE0", color="#8FB79A", fontcolor="#2E5A3D"];
// 
//     { rank=same; eda; inf; }
//     { rank=same; desc; gen; }
// 
//     data -> eda  [label="  Summarize  "];
//     data -> inf  [label="  Generalize  "];
//     eda  -> desc;
//     inf  -> gen;
// }
// ```
// label=fig:edavsinferentialstatistics
// caption=How a limited sample of data diverges into exploratory data analysis and inferential statistics.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson07.1-Intro_to_Probabilistic_Programming.typ.figs/Lesson07.1-Intro_to_Probabilistic_Programming.1.png",
    width: 70%,
  ),
  caption: [How a limited sample of data diverges into exploratory data analysis and inferential statistics.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:edavsinferentialstatistics>
// render_images:end

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:129 '* Statistical Recipes vs Bayesian Inference'
// Slide: Statistical Recipes vs Bayesian Inference
Once the goal is inference, one naive way to get there is to learn a collection of
"statistical recipes": make simplifying assumptions to keep the math tractable, pick
a recipe that looks right for the data and problem at hand, then iterate until the
p-value is low enough (a practice known as p-hacking) or, in a machine learning
context, until out-of-sample performance looks acceptable. The trouble is that the
result ends up depending on which recipe happened to be tried rather than on the
structure of the problem itself.

#strong[Bayesian statistics] offers a more principled alternative #cite(
  "martin2018bayesian",
). By treating inference as a problem of updating probability distributions rather
than selecting from a menu of closed-form formulas, the Bayesian framework removes
the limitations that force classical methods into narrow analytical recipes. This
probabilistic perspective reveals a deep unity beneath methods that look entirely
unrelated on the surface: a linear regression fitted with `statsmodels` and a
decision tree built with `sklearn` are, from the Bayesian viewpoint, two instances of
the same underlying inference machinery, differing only in the likelihood function
and prior each one assumes. Modern computational tools such as `PyMC` make this
practical by using sampling algorithms to solve models that have no closed-form
solution at all, turning problems that were once analytically intractable into
routine computations.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:152 '* Characteristics of Data'
// Slide: Characteristics of Data
Whichever approach to inference is used, it starts from data, the essential raw
material of machine learning: without it, no model can learn, generalize, or make
useful predictions. Understanding where data comes from and what makes it difficult
to work with is therefore a prerequisite for any serious ML practice.

Data can originate from several distinct sources: controlled experiments,
computational simulations, surveys of human respondents, or direct field observations
of natural or engineered systems. Each source carries its own biases and limitations,
and the choice of source shapes every downstream modeling decision.

One pervasive characteristic of real-world data is #strong[uncertainty]. Data is
stochastic for at least three reasons. First, #emph[ontological uncertainty] arises
when the system under study is intrinsically random: quantum phenomena, turbulent
flows, or human decision-making all exhibit irreducible randomness that no amount of
additional measurement can eliminate. Second, #emph[technical uncertainty] stems from
the limited precision or outright noisiness of measurement instruments: a sensor has
finite resolution, a survey question can be misunderstood, and network latency can
corrupt timestamps. Third, #emph[epistemic uncertainty] reflects gaps in our
knowledge about the system itself: we may be missing relevant variables, using an
incomplete model, or simply lacking enough observations to distinguish signal from
noise. Recognizing which type of uncertainty dominates a given dataset guides the
choice of modeling strategy and the interpretation of results.

Data collection also carries real #emph[cost], whether measured in money, time, or
effort. Because resources are finite, it pays to think carefully about what questions
a dataset needs to answer before gathering a single observation. This principle is
formalized in #emph[experiment design], a branch of statistics devoted to planning
data collection so that the resulting observations are maximally informative for the
questions at hand.

Beyond cost, data is rarely #emph[clean]. Real datasets arrive with missing values,
inconsistent formats, duplicated records, mislabeled categories, and outliers that
may be genuine extremes or recording errors. Handling these issues, often grouped
under the umbrella of data cleaning and preprocessing, typically consumes the
majority of a practitioner's time on any applied project.

Finally, raw numbers alone are meaningless without #emph[interpretation]. Every
dataset is viewed through the lens of a mental model (the analyst's intuitions about
what matters) and a formal model (the mathematical structure imposed during
analysis). The same table of numbers can support very different conclusions depending
on which model frames the interpretation, which is why clarity about modeling
assumptions is as important as the data itself.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:173 '## Bayesian Modeling'
// Slide: Bayesian Modeling
== Bayesian Modeling

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:177 '* What Is a Model'
// Slide: What Is a Model
A #strong[model] is a simplified description of a given system or process, designed
to capture the most relevant aspects while abstracting away minor details. More
complexity does not automatically make a model better; in fact, an overly complex
model can hurt generalization by fitting noise rather than signal. The #strong[VC
  dimension] measures the capacity of a model, quantifying how many distinct patterns
a hypothesis set can express. A common rule of thumb holds that #emph[you need at
  least 10 data points per effective degree of freedom of the hypothesis set] #cite(
  "abumostafa2012learning",
), which gives a practical sense of when a model's capacity is well matched to the
available data. The goal, then, is not to build the richest possible description but
to find one that captures the system's essential structure while ignoring details
that contribute more noise than insight.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:190 '* Bayes' Theorem'
// Slide: Bayes' Theorem
Bayesian modeling gives a precise rule for building such a description from data:
#strong[Bayes' theorem] states that for model parameters $theta$ and observed data
$X$,

$ Pr(theta | X) = frac(Pr(X | theta) dot.op Pr(theta), Pr(X)) $

Each term in this equation carries a specific interpretation. The left-hand side,
$Pr(theta | X)$, is the #strong[posterior]: the probability assigned to the
parameters $theta$ after observing the data $X$. The term $Pr(X | theta)$ is the
#strong[likelihood], sometimes called the "statistical model," which measures how
plausible the observed data $X$ would be if the parameters took the value $theta$.
The term $Pr(theta)$ is the #strong[prior], encoding whatever knowledge or belief we
hold about the parameters before seeing any data at all. Finally, $Pr(X)$ is the
#strong[evidence], also known as the "marginal likelihood," representing the total
probability of observing the data $X$. It is called "marginal" because it averages
(marginalizes) the likelihood over all possible parameter values, ensuring that the
posterior is a properly normalized distribution.

The core insight is that Bayes' theorem fuses two distinct sources of information:
what we believed before collecting data, and what the data itself tells us. In
compact form,

$ "Posterior" = frac("Likelihood" dot.op "Prior", "Evidence") $

The prior encodes domain knowledge, regularization assumptions, or previously
collected evidence, while the likelihood lets the current data speak. The evidence
term acts purely as a normalizing constant, scaling the numerator so the posterior
integrates to one. In many practical applications the evidence is intractable to
compute exactly, which motivates approximation methods such as Markov chain Monte
Carlo or variational inference. What matters conceptually, however, is the numerator:
the posterior is proportional to the likelihood times the prior, so stronger data (a
more peaked likelihood) gradually overwhelms a vague prior, while a highly
informative prior dominates when data are scarce. This balance between prior belief
and observed evidence is the central mechanism of Bayesian reasoning.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:223 '* Bayesian Modeling Workflow'
// Slide: Bayesian Modeling Workflow
Two ideas drive the workflow: #strong[probability] provides a coherent language for
measuring uncertainty about parameters, and #strong[Bayes' theorem] gives us a
principled mechanism for updating those probabilities as new data arrive, ideally
reducing our uncertainty in the process.

These two ideas come together in a structured workflow for applied modeling. The
#strong[Bayesian modeling workflow] #cite("gelman2020workflow") proceeds through four
iterative stages:

1. #emph[Design a model] by encoding assumptions about how the data were generated,
  expressed as probability distributions over parameters and observables.
2. #emph[Condition on data] by applying Bayes' theorem to compute (or approximate)
  the posterior distribution, folding the observed evidence into the model.
3. #emph[Validate] the fitted model by checking its predictions against held-out
  data, consulting subject-matter expertise, and comparing it with alternative
  models.
4. #emph[Iterate and backtrack] as needed: fix coding errors discovered during
  validation, refine the model structure, or collect additional or different data
  before returning to step 1.

The workflow is deliberately nonlinear. Validation in step 3 almost always reveals
some mismatch between model and reality, sending the modeler back to an earlier
stage. A posterior predictive check might expose systematic bias, prompting a richer
likelihood; a domain expert might flag an implausible parameter estimate, suggesting
a more informative prior. Each loop through the cycle tightens the alignment between
what the model assumes and what the data actually show, as
@fig:bayesianmodelingworkflow shows.

// rendered_images:begin
// ```graphviz
// digraph BayesianWorkflow {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.5;
//     ranksep=0.45;
//     rankdir=TB;
// 
//     // source : orange | core process : blue | verification : sage green
//     node [shape=box,
//           style="rounded,filled",
//           penwidth=1.8,
//           fontname="Helvetica",
//           fontsize=12,
//           margin="0.22,0.14",
//           height=0.50];
// 
//     edge [color="#A3B1C0",
//           penwidth=1.3,
//           arrowhead=vee,
//           arrowsize=0.75,
//           fontname="Helvetica",
//           fontsize=10,
//           fontcolor="#7B8794"];
// 
//     inputs   [label="Data and\nassumptions", fillcolor="#FBEBD4", color="#D9A85F", fontcolor="#6B4517"];
//     design   [label=<<b>1. Design</b><br/>probabilistic model>, fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//     condition [label=<<b>2. Condition</b><br/>on data (Bayes)>, fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//     validate [label=<<b>3. Validate</b><br/>the model>, fillcolor="#DFEDE0", color="#8FB79A", fontcolor="#2E5A3D"];
// 
//     inputs -> design;
//     design -> condition;
//     condition -> validate;
//     validate -> design [style=dashed, color="#C0455B", fontcolor="#C0455B", label="  Iterate  ", constraint=false];
// }
// ```
// label=fig:bayesianmodelingworkflow
// caption=The iterative Bayesian modeling cycle where design, conditioning, and validation feed back to refine the model.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson07.1-Intro_to_Probabilistic_Programming.typ.figs/Lesson07.1-Intro_to_Probabilistic_Programming.2.png",
    width: 70%,
  ),
  caption: [The iterative Bayesian modeling cycle where design, conditioning, and validation feed back to refine the model.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:bayesianmodelingworkflow>
// render_images:end

This iterative character distinguishes Bayesian modeling from a one-shot "fit and
report" analysis. Rather than treating inference as a single computation, the
workflow treats it as an ongoing conversation between the modeler, the data, and the
model itself.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:288 '# Coin Example: Analytical Approach'
// Slide: Coin Example: Analytical Approach
= Coin Example: Analytical Approach

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:290 '## Distributions'
// Slide: Distributions
== Distributions

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:292 '* Binomial Distribution'
// Slide: Binomial Distribution
The #strong[binomial distribution] models the probability of observing exactly $k$
successes (say, heads) out of $n$ independent trials, each with success probability
$p$. If $X$ follows a binomial distribution, written $X tilde "Binomial"(n, p)$, then
the probability of exactly $k$ successes is

$ Pr(k) = frac(n!, k! (n - k)!) p^k (1 - p)^(n-k) $

The combinatorial prefactor $n! slash (k!(n-k)!)$ counts the number of distinct
orderings in which those $k$ successes can appear among the $n$ trials, while
$p^k (1-p)^(n-k)$ gives the probability of any single such ordering. As
@fig:binomialdistribution shows, the shape of this probability mass function shifts
and spreads as the parameters $n$ and $p$ vary: increasing $n$ widens the
distribution, while changing $p$ slides its peak away from the center.

#figure(
  image(
    "../lectures_source/figures/L07.1.Binomial_distribution.png",
    width: 80%,
  ),
  caption: [Binomial distribution],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:binomialdistribution>

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:319 '* Beta Distribution'
// Slide: Beta Distribution
The binomial distribution describes the data once the bias is known, so the unknown
bias itself needs a distribution: the #strong[Beta distribution] models the
probability of a continuous random variable constrained to the interval $[0, 1]$. If
$X$ follows a Beta distribution with shape parameters $alpha$ and $beta$, written
$X tilde "Beta"(alpha, beta)$, its probability density function is

$
  Pr(theta) eq.delta frac(Gamma(alpha + beta), Gamma(alpha) Gamma(beta)) theta^(alpha - 1) (1 - theta)^(beta - 1)
$

where $Gamma(dot.op)$ is the gamma function, which generalizes the factorial to
continuous arguments. The two parameters $alpha$ and $beta$ control the shape of the
density: when both are equal the distribution is symmetric around $0.5$; when
$alpha > beta$ the mass shifts toward one; and when $alpha < beta$ it shifts toward
zero. The special case $alpha = beta = 1$ recovers the uniform distribution over
$[0, 1]$, encoding complete ignorance about where the true value lies.

This makes the Beta distribution a natural choice whenever the quantity of interest
is itself a probability or a proportion. Consider a coin whose true bias $theta$ is
unknown: placing a $"Beta"(alpha, beta)$ prior on $theta$ lets you express how much
you already know (or don't know) about the coin before flipping it. The same
reasoning applies to a website's click-through rate, a baseball player's batting
average, or the fraction of a population expected to convert in a marketing campaign.
In each case the outcome is bounded between zero and one, and the Beta family
provides a flexible, conjugate prior that updates cleanly as new observations arrive.

@fig:betadistribution plots how the density's shape changes as $alpha$ and $beta$
vary. Concentrated, peaked curves correspond to strong prior beliefs, while flatter
curves express greater uncertainty. Recognizing which parameter regime matches your
prior knowledge is the first step toward choosing a sensible Bayesian model for any
proportion-valued quantity.

#figure(
  image("../lectures_source/figures/L07.1.Beta_distribution.png", width: 80%),
  caption: [Beta distribution],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:betadistribution>

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:350 '* Beta Distribution: Properties'
// Slide: Beta Distribution: Properties
Several properties make the Beta distribution convenient: it is continuous with
support on the interval $[0, 1]$, making it a natural choice for modeling quantities
that represent probabilities, proportions, or rates. It is parameterized by two
positive shape parameters: $alpha$, often called the "success" parameter, and $beta$,
the "failure" parameter.

One of the Beta distribution's most useful properties is its flexibility in shape.
Depending on the values of $alpha$ and $beta$, it can take on several qualitatively
different forms: uniform, monotonically increasing, monotonically decreasing,
bell-shaped (resembling a Gaussian), or U-shaped. When $alpha > beta$, the density
skews toward 1, concentrating probability mass on higher values and reflecting a
greater expected probability of success. When $alpha = beta$, the distribution is
symmetric and centered around 0.5. This versatility lets a single two-parameter
family express a wide range of prior beliefs about an unknown probability.

The Beta distribution also enjoys a key computational property: it is the
#strong[conjugate prior] of the Binomial distribution. This means that if you place a
Beta prior on the success probability $p$ of a Binomial likelihood, the posterior
distribution over $p$ after observing data is again a Beta distribution, just with
updated parameters. Conjugacy keeps Bayesian updating in closed form, avoiding the
need for numerical integration or sampling at each step. @fig:betaconjugate plots how
the shape of the Beta PDF changes as $alpha$ and $beta$ vary, covering the range of
prior beliefs this single family can encode.

#figure(
  image("../lectures_source/figures/L07.1.Beta_distribution.png", width: 80%),
  caption: [Beta distribution: conjugate prior of the Binomial.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:betaconjugate>

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:382 '* Conjugate Prior of a Likelihood'
// Slide: Conjugate Prior of a Likelihood
Pairing a Beta prior with a Binomial likelihood is an instance of a general idea: a
#strong[conjugate prior] of a likelihood $F$ is a prior distribution that, when
combined with that likelihood, yields a posterior sharing the same functional form as
the prior #cite("bishop2006prml"). For instance, a Beta prior paired with a Binomial
likelihood produces a Beta posterior, and a Normal prior paired with a Normal
likelihood produces a Normal posterior.

This conjugate relationship carries several practical advantages: Because the prior
and posterior belong to the same distributional family, the posterior always has a
closed analytical form: there is no need to resort to numerical integration or
sampling to compute it. Updating is straightforward as well; one simply adjusts the
prior's parameters using the observed data, and this update can be applied
iteratively as new batches of data arrive. Most importantly, conjugacy guarantees
tractability of the posterior, which is precisely why conjugate priors remain a
workhorse in Bayesian modeling whenever the likelihood admits one.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:404 '## Analytical Solution'
// Slide: Analytical Solution
== Analytical Solution

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:406 '* Coin Example: Estimating the Bias'
// Slide: Coin Example: Estimating the Bias
Consider a simple experiment: toss a coin $N$ times and record the number of heads
$Y$ and tails $N - Y$. The natural question is, how biased is the coin?

This scenario involves #strong[true uncertainty]. An underlying parameter, the coin
bias $theta$, exists but is unknown to the observer. When $theta = 0$ the coin always
lands tails; when $theta = 1$ it always lands heads; and when $theta = 0.5$ it
produces heads and tails with equal frequency. The goal of inference is to learn
about $theta$ from the observed data, and the Bayesian approach treats $theta$ itself
as a random variable with a distribution that encodes our beliefs about its value.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:420 '* Coin Example: Assumptions'
// Slide: Coin Example: Assumptions
To answer the question about the coin's bias, three assumptions underpin the Bayesian
model. First, the tosses are #strong[independent and identically distributed] (IID):
each flip's outcome is unaffected by any other flip (independence), and the coin's
bias does not change between tosses (identical distribution).

Second, the likelihood of observing $Y$ heads out of $N$ tosses, given the bias
parameter $theta$, follows a #strong[binomial distribution]. The binomial captures
exactly the scenario of counting successes in a fixed number of independent trials
with constant success probability.

Third, the prior over $theta$ is a #strong[beta distribution]. The beta family is
flexible enough to encode a wide range of prior beliefs, from uniform (no preference)
to strongly peaked around a particular value. It also serves as the #emph[conjugate
  prior] of the binomial likelihood, which means the posterior distribution after
observing data is itself another beta distribution. Conjugacy keeps the math
closed-form: instead of solving an integral numerically, you simply update the beta
distribution's two shape parameters with the observed head and tail counts.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:432 '* Coin Example: Analytical Solution'
// Slide: Coin Example: Analytical Solution
Under these assumptions, the #emph[posterior] is proportional to the
#emph[likelihood] times the #emph[prior]:

$ Pr(theta | y) prop Pr(y | theta) Pr(theta) $

Substituting a Binomial likelihood and a Beta prior makes the algebra concrete. The
full joint expression is:

$
  Pr(theta mid Y) &= underbrace(frac(N!, y!(N - y)!) theta^y (1 - theta)^(N - y), "likelihood") dot underbrace(frac(Gamma(alpha + beta), Gamma(alpha) Gamma(beta)) theta^(alpha - 1) (1 - theta)^(beta - 1), "prior")
$

The combinatorial coefficient and the Beta normalizing constant do not depend on
$theta$, so they factor out when we write proportionality:

$
  Pr(theta mid Y) &prop underbrace(theta^y (1 - theta)^(N - y), "likelihood") dot underbrace(theta^(alpha - 1) (1 - theta)^(beta - 1), "prior") \
  &= theta^(y + alpha - 1) (1 - theta)^(N - y + beta - 1)
$

Recognizing this as the kernel of a Beta distribution gives the closed-form
posterior:

$ Pr(theta mid Y) = "Beta"(alpha_"prior" + y, space beta_"prior" + N - y) $

This result is the core of the Bayesian update for the Beta-Binomial model. When the
prior is #emph[conjugate] to the likelihood, the posterior belongs to the same
distributional family as the prior; the entire learning step reduces to updating the
family's parameters rather than changing the functional form. The prior's
pseudo-counts $alpha$ and $beta$ simply absorb the observed data: $alpha$ increases
by the number of successes $y$, and $beta$ increases by the number of failures
$N - y$. No numerical integration or sampling is required; the answer is exact and
immediate.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:467 '* Coin Example: Updating the Beta Prior'
// Slide: Coin Example: Updating the Beta Prior
To see the update at work, consider a concrete case: start with a uniform prior,
$alpha = beta = 1$, and observe $N = 4$ tosses that yield $y = 1$ head. The posterior
is $"Beta"(1 + 1, 1 + 4 - 1) = "Beta"(2, 4)$, with a posterior mean of

$ EE[theta] = frac(2, 2 + 4) approx 0.33 $

A 94% credible interval runs roughly from 0.03 to 0.65. Even with just four
observations, the posterior has already shifted noticeably away from the flat uniform
prior and toward lower values of $theta$, reflecting the fact that only one head
appeared. Sampling the same model with PyMC approximates this exact posterior
numerically, confirming that the analytic conjugate update and the MCMC sampler
agree. @fig:coinexampleupdatingthebetaprior shows how the prior density reshapes into
the posterior after incorporating the observed data.

// rendered_images:begin
// ```graphviz
// digraph BetaUpdate {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.5;
//     ranksep=0.45;
//     rankdir=LR;
// 
//     // prior : orange | data : orange | process : blue | posterior : sage green
//     node [shape=box,
//           style="rounded,filled",
//           penwidth=1.8,
//           fontname="Helvetica",
//           fontsize=12,
//           margin="0.22,0.14",
//           height=0.50];
// 
//     edge [color="#A3B1C0",
//           penwidth=1.3,
//           arrowhead=vee,
//           arrowsize=0.75,
//           fontname="Helvetica",
//           fontsize=10,
//           fontcolor="#7B8794"];
// 
//     prior [label=<<b>Prior</b><br/>Beta(α, β)>, fillcolor="#FBEBD4", color="#D9A85F", fontcolor="#6B4517"];
//     data  [label=<<b>Data</b><br/>y heads,<br/>N − y tails>, fillcolor="#FBEBD4", color="#D9A85F", fontcolor="#6B4517"];
//     post  [label=<<b>Posterior</b><br/>Beta(α + y,<br/>β + N − y)>, fillcolor="#DFEDE0", color="#8FB79A", fontcolor="#2E5A3D"];
// 
//     prior -> post;
//     data  -> post;
// }
// ```
// label=fig:coinexampleupdatingthebetaprior
// caption=How the Beta prior is updated by observed data to produce the posterior distribution.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson07.1-Intro_to_Probabilistic_Programming.typ.figs/Lesson07.1-Intro_to_Probabilistic_Programming.3.png",
    width: 70%,
  ),
  caption: [How the Beta prior is updated by observed data to produce the posterior distribution.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:coinexampleupdatingthebetaprior>
// render_images:end

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:522 '* Coin Example: Effect of Priors (1/2)'
// Slide: Coin Example: Effect of Priors (1/2)
Next, see how the choice of prior changes the update: take a coin whose true bias is
0.35, meaning it lands heads 35% of the time. Three analysts begin with different
prior beliefs about this bias. The first (shown in red in @fig:updatingtheprior)
adopts a uniform prior, treating every possible bias value from 0 to 1 as equally
likely. The second (green) centers a Gaussian-like prior around 0.5, expressing a
belief that the coin is probably close to fair. The third (blue) uses a prior skewed
toward lower values, reflecting a suspicion that the coin favors tails.

Despite these very different starting points, the Bayesian updating procedure is
identical for all three:

1. Observe a batch of coin flips.
2. Compute the likelihood of that data under each possible bias value.
3. Multiply the current prior by the likelihood and renormalize to obtain the
  posterior.
4. Use the posterior as the new prior and repeat as more data arrive.

As @fig:updatingtheprior shows, the three posterior distributions gradually converge
toward the true value of 0.35 as the number of observations grows. Early on, the
posteriors look quite different from one another because the prior still dominates.
After enough flips, though, the likelihood term overwhelms every prior, and all three
analysts reach essentially the same conclusion. This is a useful property of Bayesian
inference: given sufficient data, reasonable priors "wash out," and the posterior
concentrates on the truth regardless of where it started.

#figure(
  image("../lectures_source/figures/L07.1.Updating_the_prior.png", width: 80%),
  caption: [Updating the prior],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:updatingtheprior>

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:548 '* Coin Example: Effect of Priors (2/2)'
// Slide: Coin Example: Effect of Priors (2/2)
The experiment above brings out several properties of Bayesian analysis worth
highlighting. First, the outcome of a Bayesian analysis is not a single point
estimate but an entire #emph[posterior distribution]. This distribution captures both
the best guess and the remaining uncertainty about the parameter. Second, the
#emph[spread] of the posterior is directly proportional to the uncertainty in the
estimate. As more data arrives, the posterior narrows; it narrows faster when the
observed data aligns well with the prior, because confirming evidence concentrates
probability mass more efficiently. Given enough data, models that started from
different priors converge to the same posterior, a reassuring property sometimes
called "washing out the prior." Third, Bayesian updating is #emph[order invariant]:
whether you feed observations to the model one at a time, updating the posterior at
each step, or process them all in a single batch, the final posterior is identical.
@fig:updatingtheprior traces this sequential updating process: how the posterior
after one observation becomes the prior for the next, steadily sharpening as evidence
accumulates.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:569 '# Priors and Credible Intervals'
// Slide: Priors and Credible Intervals
= Priors and Credible Intervals

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:571 '## Frequentist vs Bayesian'
// Slide: Frequentist vs Bayesian
== Frequentist vs Bayesian

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:573 '* Do Priors Bias the Analysis?'
// Slide: Do Priors Bias the Analysis?
Detractors of the Bayesian approach often object that one should "let the data speak
for itself," arguing that imposing a prior distribution contaminates the evidence.
This criticism sounds principled, but it rests on a misunderstanding of what data
actually tells us on its own.

The first counterpoint is that data does not speak; it murmurs. Raw observations
carry no intrinsic meaning. A column of numbers becomes informative only once we
interpret it through some model, whether that model is an informal mental picture of
the data-generating process or a fully specified mathematical likelihood. A prior is
simply one more piece of that interpretive scaffold, making explicit what we believe
before the evidence arrives. Refusing to state a prior does not eliminate
assumptions; it only hides them.

The second counterpoint cuts deeper: every statistical model embeds a prior, whether
the analyst acknowledges it or not. Frequentist procedures are built on their own
structural assumptions (choice of test statistic, significance threshold, sampling
model) that play exactly the same role a Bayesian prior does. The maximum likelihood
estimate offers a concrete example. MLE corresponds precisely to adopting a uniform
(flat) prior over the parameter space and then reporting the mode of the resulting
posterior distribution. Calling the result "prior-free" is a labeling choice, not a
mathematical fact. The uniform prior is still a prior; it asserts that every
parameter value is equally plausible before any data is seen, which is itself a
strong and often unrealistic assumption. Making the prior explicit, as Bayesian
inference requires, at least forces the modeler to defend that assumption openly
rather than smuggling it in through the back door.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:591 '* Advantages of Priors'
// Slide: Advantages of Priors
Defending a prior openly is not only a burden: Bayesian inference offers several real
advantages over frequentist or "hacker ML" approaches. Its assumptions are explicit:
every prior, likelihood, and modeling choice is stated up front rather than buried
inside an algorithm's defaults. This transparency encourages a deeper analysis of
both the problem and the data, because the modeler must commit to a generative story
before seeing any observations. That discipline often surfaces misunderstandings
early, before they silently distort results.

From a statistical standpoint, the prior acts as a regularizer: it shrinks parameter
estimates toward plausible values, reducing overfitting in exactly the same way that
an $L_2$ penalty does in ridge regression, but with a principled probabilistic
justification rather than an ad hoc tuning knob. The posterior distribution also
provides a built-in uncertainty measure: its spread tells you how confident (or
uncertain) you should be about any quantity of interest, without the need for
separate bootstrap or permutation procedures.

Less obvious is the computational payoff. A well-chosen prior can simplify and speed
up inference dramatically, for instance by making a posterior conjugate so that
updates reduce to closed-form algebra instead of expensive sampling. As Gelman
observed, "when you encounter computational problems, there's often an issue with
your model" #cite("gelman2008folk"): difficulty in fitting is frequently a signal
that the model is misspecified, not that the sampler needs more iterations.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:605 '* How to Choose Priors (1/2)'
// Slide: How to Choose Priors (1/2)
Having seen what a prior buys, the practical question is how to choose one, and
beyond the informative and uninformative extremes, practitioners often work with two
intermediate categories of prior that balance flexibility with structure.

#strong[Weakly-informative priors] (sometimes called "flat," "vague," or "diffuse"
priors) supply minimal information while still keeping the posterior well-behaved.
For instance, centering a regression coefficient around zero with a wide spread,
$beta tilde "Normal"(0, 10)$, says little about the coefficient's actual value but
rules out astronomically large estimates that would be physically implausible in most
applications.

#strong[Regularizing priors] encode real prior knowledge about a parameter's
behavior. Common choices include:

- A parameter known to be positive, such as a standard deviation, can receive a
  half-Cauchy prior: $sigma tilde "HalfCauchy"(0, 5)$ #cite("gelman2006prior"). This
  concentrates mass on the positive real line while allowing heavy tails for
  occasionally large values.
- A Laplace prior $beta tilde "Laplace"(0, 1)$ #cite("parkcasella2008lasso") acts as
  the Bayesian analogue of lasso regularization, encouraging sparsity by placing a
  sharp peak at zero so that many coefficients are pulled toward exactly zero during
  estimation.
- A normal prior $beta tilde "Normal"(0, 1)$ with a tighter scale discourages extreme
  coefficient values, functioning much like ridge regularization in frequentist
  settings.

More generally, regularizing priors can express that a parameter is close to zero,
falls above or below some threshold, or lies within a specific range. The key
distinction from weakly-informative priors is intentionality: a regularizing prior is
chosen precisely because the analyst has a reason to believe the parameter should
behave a certain way, and the prior's shape encodes that belief directly into the
model.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:621 '* How to Choose Priors (2/2)'
// Slide: How to Choose Priors (2/2)
At the other end of the spectrum, when strong prior knowledge is available, it makes
sense to encode it directly. #strong[Informative priors] draw on previous
experiments, expert judgment, or earlier datasets to concentrate the prior around
plausible values. For instance, if past experimental data suggest a coefficient is
close to 2.5 with modest uncertainty, one might set
$beta_1 tilde "Normal"(2.5, 0.5)$. Similarly, if historical records show that roughly
5% of cases are positive, a $p tilde "Beta"(2, 38)$ prior centers the distribution
near 0.05 while still allowing the data to shift it.

A complementary strategy is #strong[prior elicitation], which works backward from
qualitative constraints to find the least informative distribution consistent with
them. The idea is to apply the principle of maximum entropy: among all distributions
satisfying the stated constraints, choose the one that adds the least additional
assumption. If a domain expert reports that 90% of the probability mass should fall
between 0.1 and 0.7, one can fit a Beta distribution whose shape parameters satisfy
that interval constraint while remaining as diffuse as possible otherwise. This
yields a prior that faithfully represents what the expert knows without inadvertently
injecting structure the expert never intended.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:633 '## Communicating Results'
// Slide: Communicating Results
== Communicating Results

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:635 '* Communicating a Bayesian Analysis'
// Slide: Communicating a Bayesian Analysis
The first step in any Bayesian analysis is to communicate assumptions and hypotheses
clearly. This means describing the prior distributions and the probabilistic model
that together encode what you believe before seeing data. For the coin-flip example,
the model has two parts: a prior on the bias parameter,
$theta tilde "Beta"(alpha, beta)$, and a likelihood for each observed flip,
$y tilde "Binomial"(n=1, p=theta)$. Writing the model this way forces you to be
explicit about every assumption, from the family of the prior to the number of
observations, so that anyone reading your analysis can scrutinize or adjust those
choices.

The second step is to communicate the result of the Bayesian analysis itself, which
centers on the posterior distribution. Rather than reporting a single point estimate,
you summarize both the location and the dispersion of the posterior. For location,
you might report the mean, mode, or median, choosing whichever best represents the
distribution's center given its shape. For dispersion, the standard deviation is a
natural first choice, though it can be misleading when the posterior is heavily
skewed, because symmetric intervals around the mean may cover regions of negligible
density while missing the distribution's actual mass.

A more informative summary is the #strong[highest density interval] (HDI) #cite(
  "kruschke2015doing",
). The HDI is the shortest contiguous interval that contains a specified portion of
the posterior's probability density, such as 95% or 50%. Every point inside the HDI
has higher density than every point outside it, which makes it the tightest credible
interval for a given coverage level. The exact coverage percentage is a convention,
not a law: the `ArviZ` library, for instance, defaults to 94% rather than 95%. What
matters is that you report which level you chose and interpret the interval as "given
the data and the model, the parameter lies in this range with the stated
probability," a direct probabilistic statement that frequentist confidence intervals
cannot make.

@fig:kruschkediagram draws this workflow as a Kruschke diagram of the coin model,
showing how the prior, likelihood, and posterior connect and where summary statistics
like the HDI sit within the analysis pipeline.

#figure(
  image("../lectures_source/figures/L07.1.Kruschke_diagram.png", width: 80%),
  caption: [Kruschke diagram of the coin model],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:kruschkediagram>

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:670 '* Confidence Intervals vs Credible Intervals'
// Slide: Confidence Intervals vs Credible Intervals
The contrast with frequentist intervals deserves a closer look, because a common
source of confusion is the difference between #emph[frequentist confidence intervals]
and #emph[Bayesian credible intervals] #cite(
  "morey2016fallacy",
). The two sound similar but rest on fundamentally different interpretations of
probability, and mixing them up leads to incorrect statistical reasoning.

In the frequentist framework, a parameter has a single true (but unknown) value; it
is not a random variable. A #strong[confidence interval] is a range computed from the
data that may or may not contain that true value. The correct interpretation of a 95%
confidence interval is: "if we repeated this experiment many times and built an
interval each time, 95% of those intervals would contain the true value." It does
#emph[not] mean "there is a 95% probability that the true value lies in this
particular interval." Once the interval is computed, the true value is either inside
it or not; there is no remaining probability to assign. This subtlety trips up even
experienced practitioners, because the phrasing people want to use (the one about
probability of the true value) is precisely the one the frequentist framework
forbids.

In the Bayesian framework, parameters are themselves random variables endowed with
prior distributions. A #strong[Bayesian credible interval] is defined so that there
is a 95% posterior probability that the parameter lies within the interval, given the
observed data. This is exactly the statement most people intuitively want to make:
"given what I have seen, there is a 95% chance the parameter is in here." The
credible interval directly conditions on the data at hand rather than appealing to
hypothetical repetitions of the experiment. That directness is why many practitioners
find the Bayesian credible interval more intuitive and easier to communicate to
non-statisticians, even though it comes at the cost of specifying a prior.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:689 '* Confidence vs Credible Intervals: Fishing Analogy'
// Slide: Confidence vs Credible Intervals: Fishing Analogy
To make that difference concrete, consider the fishing analogy, a helpful way to
build intuition for frequentist and Bayesian interval estimates. In the frequentist
picture, imagine fishing in a lake where you cannot see beneath the surface. You
throw a net, and a #strong[confidence interval] at the 95% level means: "if I threw
this net 100 times, about 95 of those nets would catch the fish." The subtlety is
that once a particular net has been thrown, the fish is either inside it or not; the
95% figure describes the long-run success rate of the procedure across many
repetitions, not a probability statement about this one net.

The Bayesian picture reframes the problem entirely. Now imagine you have a map that
shows where the fish #emph[probably] is, assembled from past observations and prior
knowledge. A #strong[credible interval] at the 95% level says: "given my map, there
is a 95% chance the fish is inside this region of the lake." Here the fish's true
location is treated as uncertain, and probability directly quantifies your belief
about where it lies, conditioned on the evidence you have gathered so far.

The distinction matters in practice. A confidence interval's coverage guarantee is a
property of the method, not of any single dataset: repeat the experiment many times
and roughly 95% of the intervals you construct will contain the true parameter. A
credible interval, by contrast, makes a direct probability claim about the parameter
given the observed data and the chosen prior. For a well-chosen prior and a
reasonably large sample the two intervals often coincide numerically, but their
interpretations remain fundamentally different: one speaks about the reliability of a
repeated procedure, the other about the plausibility of a parameter value in light of
current evidence.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:710 '# Probabilistic Programming'
// Slide: Probabilistic Programming
= Probabilistic Programming

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:712 '## Probabilistic Programming Languages'
// Slide: Probabilistic Programming Languages
== Probabilistic Programming Languages

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:714 '* Knowns and Unknowns'
// Slide: Knowns and Unknowns
A probabilistic model is built from two kinds of ingredients. The #strong[knowns] are
the model structure, represented as a graph of probability distributions, together
with the observed data, treated as constants once collected. The #strong[unknowns]
are the model parameters, themselves modeled as probability distributions that
express our uncertainty about their values before any data arrives.

The goal is to apply Bayes' theorem to #emph[condition] the unknowns on the knowns,
thereby reducing uncertainty about those parameters. In other words, we combine prior
beliefs with observed evidence to obtain a posterior distribution that concentrates
around parameter values consistent with the data. @fig:knownsandunknowns shows this
flow: knowns and unknowns feed into Bayes' theorem, which produces updated
(posterior) unknowns whose uncertainty has been narrowed by the evidence.

// rendered_images:begin
// ```graphviz
// digraph BayesTheorem {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//     // Node styles
//     knowns       [label="Knowns", fillcolor="#A6C8F4"];
//     unknowns_in  [label="Unknowns", fillcolor="#A6E7F4"];
//     bayes_box    [label="Bayes' Theorem", fillcolor="#FFD1A6"];
//     unknowns_out [label="Updated Unknowns", fillcolor="#B2E2B2"];
// 
//     // Force ranks
//     { rank=same; knowns; unknowns_in }
// 
//     // Edges
//     knowns      -> bayes_box;
//     unknowns_in -> bayes_box;
//     bayes_box   -> unknowns_out;
// }
// ```
// label=fig:knownsandunknowns
// caption=How knowns and unknowns feed into Bayes' theorem to produce updated unknowns with reduced uncertainty.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson07.1-Intro_to_Probabilistic_Programming.typ.figs/Lesson07.1-Intro_to_Probabilistic_Programming.4.png",
    width: 70%,
  ),
  caption: [How knowns and unknowns feed into Bayes' theorem to produce updated unknowns with reduced uncertainty.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:knownsandunknowns>
// render_images:end

For all but the simplest models, however, the posterior distribution has no
closed-form solution: the required integrals over high-dimensional parameter spaces
are analytically intractable. #strong[Probabilistic programming] addresses this
directly. Instead of deriving each model's inference algorithm by hand, a
practitioner specifies the probabilistic model in code, declaring priors,
likelihoods, and observed data, and then hands it to an inference engine that solves
the model using numerical techniques such as Markov Chain Monte Carlo (MCMC) or
variational inference. This separation of model specification from inference
execution makes it practical to work with rich, realistic models that would be
impossible to solve on paper.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:766 '* Probabilistic Programming Languages'
// Slide: Probabilistic Programming Languages
Concrete tools make this separation usable: probabilistic programming provides a
practical framework for Bayesian inference by letting practitioners specify models in
code and delegating the hard work of inference to automated numerical methods. The
core toolchain in the Python ecosystem centers on three libraries. #strong[PyMC]
#cite("abrilpla2023pymc") is a flexible probabilistic programming library that lets
users define generative models with a concise, readable syntax. Under the hood, PyMC
relies on #strong[PyTensor], a library for defining, optimizing, and evaluating
mathematical expressions over tensors, serving as the computational backend that
handles automatic differentiation and graph optimization. Once a model has been fit,
#strong[ArviZ] #cite("kumar2019arviz") provides tools for diagnosing convergence,
summarizing posteriors, and visualizing results in a standardized way.

The workflow follows two steps: first, specify the model using code (priors,
likelihoods, observed data), and second, call an inference engine that solves for the
posterior distribution using numerical methods such as MCMC or variational inference.
The user does not need to derive or implement the sampler; the library selects and
tunes one automatically.

This approach offers several advantages. It can compute posterior distributions even
when no analytical closed-form solution exists, which is the common case for models
of any realistic complexity. It treats model solving as a black box: universal
inference engines handle the sampling, so the practitioner can focus on what matters
most, namely model design, evaluation, and interpretation of results. The tradeoff is
that black-box inference can sometimes be slow or fail to converge for poorly
specified models, so diagnostic tools like ArviZ become essential rather than
optional.

Probabilistic programming languages have been compared to Fortran's impact on
scientific computing. Just as Fortran freed scientists from hand-coding machine
instructions and let them express algorithms at a higher level of abstraction,
probabilistic programming frees statisticians and machine learning engineers from
hand-deriving inference algorithms and lets them focus on the statistical model
itself. The computational details of sampling, gradient computation, and convergence
tuning are handled by the framework, not the user.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:790 '## Coin Example: Numerical Solution'
// Slide: Coin Example: Numerical Solution
== Coin Example: Numerical Solution

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:792 '* Coin Example: Numerical Solution (1/3)'
// Slide: Coin Example: Numerical Solution (1/3)
Suppose you know the true value of $theta$ (which is not the case in practice). This
synthetic setup lets you watch the full Bayesian updating cycle from start to finish,
with no ambiguity about what the "right answer" should be.

The procedure unfolds in five steps:

1. #emph[Model the prior and the likelihood.] Specify a prior over the parameter and
  a likelihood for the observed data:
  $
    cases(theta tilde "Beta"(alpha = 1, beta = 1), Y tilde "Binomial"(n = 1, p = theta))
  $
  The $"Beta"(1,1)$ prior is simply the uniform distribution on $[0,1]$, encoding no
  initial preference for any value of $theta$.

2. #emph[Observe samples.] Draw one or more observations of the binary variable $Y$.

3. #emph[Run inference.] Apply Bayes' rule to combine the prior with the observed
  data, producing the posterior distribution over $theta$.

4. #emph[Generate posterior samples.] Draw samples from the resulting posterior so
  you can visualize and summarize it.

5. #emph[Summarize the posterior.] Condense the posterior into interpretable
  quantities, for instance the Highest Density Interval (HDI), which gives the
  narrowest interval containing a specified probability mass.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:814 '* Coin Example: Numerical Solution (2/3)'
// Slide: Coin Example: Numerical Solution (2/3)
Running the procedure in practice begins by generating data from the ground truth
model, then building a `PyMC` model that matches the mathematical specification.
`PyMC` uses the NUTS sampler #cite(
  "hoffman2014nuts",
) and computes four independent chains. The kernel density estimation (KDE) of the
posterior resembles a Beta distribution, confirming that the numerical approximation
agrees with the analytical result. The traces appear "noisy" in the expected sense:
they explore the parameter space without any chain diverging, which signals healthy
sampling behavior. The numerical summary of the posterior yields a mean
$EE[hat(theta)] approx 0.324$, a standard deviation, and a 94% highest density
interval $Pr(hat(theta) in [0.031, 0.653]) = 0.94$. @fig:coinexamplenumericalsolution
displays the full numerical output, including the KDE plot, trace plots, and summary
statistics for this coin-flipping example.

#figure(
  image(
    "../lectures_source/figures/L07.1.Coin_example_numerical_solution.png",
    width: 80%,
  ),
  caption: [Coin example numerical solution],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:coinexamplenumericalsolution>

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:838 '* Coin Example: Numerical Solution (3/3)'
// Slide: Coin Example: Numerical Solution (3/3)
With the sampler's output in hand, the procedure for validating it follows three
steps. First, compute a single kernel density estimate (KDE) across all chains
combined, pooling their samples into one smooth density curve. Second, generate a
rank plot to verify that the chains are mixing well: the resulting histograms should
appear roughly uniform, indicating that each chain is exploring different regions of
the posterior and that, collectively, they cover the full support. Third, overlay the
single KDE with all relevant summary statistics (mean, credible intervals, etc.) to
confirm that the pooled estimate is consistent and well-behaved.

@fig:coinexamplenumericalsolution2 shows this final step: the combined KDE alongside
the computed statistics for the coin-flipping example, confirming that the sampler
has converged to a stable posterior estimate.

#figure(
  image(
    "../lectures_source/figures/L07.1.Coin_example_numerical_solution_2.png",
    width: 80%,
  ),
  caption: [Coin example numerical solution 2],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:coinexamplenumericalsolution2>

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:858 '* Summary'
// Slide: Summary
= Summary

Bayesian inference works by updating a prior belief with observed data through Bayes'
theorem, producing a posterior distribution that reflects both what we assumed
beforehand and what the evidence tells us. When the posterior has no closed-form
expression, probabilistic programming steps in: it specifies the model as code and
solves for the posterior numerically, typically through Markov chain Monte Carlo
sampling.

This chapter covered several core ideas that together form a complete Bayesian
workflow:

- #emph[Thinking probabilistically]: Bayesian modeling replaces a grab-bag of
  statistical recipes with a single, repeating cycle: write down a generative model,
  condition it on observed data, validate the fit, and iterate.
- #emph[Conjugacy]: when a beta prior is paired with a binomial likelihood, the
  posterior is again a beta distribution, so updating a coin-bias estimate reduces to
  adding observed heads and tails counts directly to the prior's parameters.
- #emph[Priors]: specifying a prior explicitly forces you to state your assumptions
  up front. Priors also regularize estimates (pulling them away from extreme values
  when data are scarce) and quantify uncertainty through the spread of the resulting
  posterior.
- #emph[Credible intervals]: a 95% Bayesian credible interval carries a direct
  probabilistic interpretation: there is a 95% probability that the parameter lies
  inside it, given the model and data. This contrasts with a frequentist confidence
  interval, which describes the long-run coverage rate of the procedure, not a
  probability statement about the parameter itself.
- #emph[Probabilistic programming]: tools such as `PyMC` let you translate a Bayesian
  model into executable code and draw samples from the posterior, while `ArviZ`
  provides diagnostics, summaries, and visualizations to assess whether those samples
  are trustworthy.

Define what you believe, let the data update that belief, and let the computer handle
the cases where pen-and-paper algebra falls short.

// From: msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd:876 '* References'
// Slide: References
= References

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
