// git_hash=9d073f6a-ccf timestamp=20260922_174153
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
  title: "L05.2: Overfitting",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L05.2: Overfitting")

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:9 '* Roadmap'
// Slide: Roadmap
#strong[Roadmap]

I'll convert this slide content into Typst body text following the style guide.

This lesson covers #strong[overfitting]: what it is, why it happens, and how to
recognize it through concrete regression and classification examples.
Understanding overfitting is the first step toward building models that
generalize well to unseen data, rather than merely memorizing the training set.

From there, the chapter derives the #strong[bias-variance decomposition], which
reveals how out-of-sample error splits into three distinct components: bias,
variance, and noise. Each component responds differently to changes in model
complexity, and the decomposition makes precise the intuition that a model can
fail either by being too simple (high bias) or too flexible (high variance).

The chapter closes with #strong[learning curves], which offer a dual perspective
on the same bias-variance tradeoff. Where the decomposition analyzes error as a
function of model complexity with a fixed dataset, learning curves instead fix
the model and ask how performance changes as the training set grows. Together,
these two views provide complementary diagnostic tools for understanding and
improving any learning system.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:19 '# Overfitting'
// Slide: Overfitting
#strong[Overfitting]

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:23 '* Overfitting: Definition'
// Slide: Overfitting: Definition
#strong[Overfitting: Definition]

#strong[Overfitting] occurs when the model fits the training data more than what
is warranted #cite("abumostafa2012learning"). Specifically, the model surpasses
the point where the out-of-sample error $E_"out"$ is minimal (the optimal fit).
This happens when model complexity is too high relative to the amount of data
and the level of noise present, causing the model to mistake noise in the
training set for genuine signal.

Fitting noise instead of signal is not merely useless; it is actively harmful.
When the model infers an in-sample pattern driven by noise and then extrapolates
that pattern to unseen data, its predictions deviate from the true target
function, resulting in poor generalization. The core danger of overfitting is
therefore not that the model learns nothing, but that it learns the #emph[wrong]
thing and confidently applies it where it does not belong.

@fig:overfittingdefinition illustrates this progression: as model complexity
increases beyond the optimal point, training error continues to decrease while
out-of-sample error begins to climb, marking the onset of overfitting.

// rendered_images:begin
// ```tikz
// \bfseries
// % Axis
// \draw[->] (0,0) -- (7,0) node[right] {$\text{VC dimension, } d_{\text{vc}}$};
// \draw[->] (0,0) -- (0,5) node[above] {Error};
// 
// % Dashed line for optimal VC dimension
// \draw[dashed, thick] (2.5,0) -- (2.5,4.5);
// \node at (2.5,-0.3) {$d_{\text{vc}}^*$};
// 
// % In-sample error curve
// \draw[thick, blue] plot[smooth, domain=0.6:6] (\x, {2.0/(0.5*\x^2.0+0.3)});
// 
// % Model complexity (square root curve)
// \draw[thick, violet] plot[smooth, domain=0.6:6] (\x, {1.0*(\x/1.5)^0.6});
// 
// % Out-of-sample error curve (in-sample + model complexity)
// \draw[thick, red] plot[smooth, domain=0.6:6] (\x, {2.0/(0.5*\x^2.0+0.3) + 1.0*(\x/1.5)^0.6});
// 
// % Labels
// \node[blue] at (5.0,0.7) {In-sample Error};
// \node[violet] at (5.5,1.5) {Model Complexity};
// \node[red] at (6.0,2.8) {Out-of-sample Error};
// ```
// label=fig:overfittingdefinition
// caption=Diagram illustrating overfitting.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.2-Overfitting.typ.figs/Lesson05.2-Overfitting.1.png",
    width: 70%,
  ),
  caption: [Diagram illustrating overfitting.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:overfittingdefinition>
// render_images:end

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:70 '* Optimal Fit'
// Slide: Optimal Fit
#strong[Optimal Fit]

The opposite of overfitting is #strong[optimal fit], which means training a
model whose complexity is properly matched to the structure in the data. An
optimally fit model achieves minimal out-of-sample error $E_"out"$, but that
does not mean the generalization error $E_"out" - E_"in"$ is also minimal.
Consider the extreme case: a model that does no training at all has
$E_"in" = E_"out"$ (since it never saw the data), giving a generalization error
of zero, yet its actual performance is terrible. What matters is how well the
model performs on unseen data in absolute terms, not merely the gap between
training and test error.

The #strong[generalization error] is defined as the additional error incurred
when moving from in-sample to out-of-sample evaluation:

$ "Generalization error" eq.delta E_"out" - E_"in" $

This quantity isolates the cost of the model having adapted to the particular
training set rather than to the true underlying pattern. A small generalization
error tells you the model's training performance is a reliable predictor of its
test performance, but it says nothing about whether that performance level is
actually good. Optimal fit targets low $E_"out"$ directly, accepting whatever
generalization gap comes with it, as @fig:optimalfit illustrates.

#figure(
  image("../lectures_source/figures/L05.2.Optimal_fit.png", width: 80%),
  caption: [Optimal fit],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:optimalfit>

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:94 '* Overfitting: Diamond Price Example'
// Slide: Overfitting: Diamond Price Example
#strong[Overfitting: Diamond Price Example]

Consider predicting a diamond's price as a function of its carat size, a classic
regression problem. The #strong[true relationship] is quadratic:

$ "price" tilde ("carat size")^2 + epsilon $

The square term reflects the fact that larger diamonds are disproportionately
rare, so their prices rise convexly rather than linearly. The noise term
$epsilon$ captures everything the model does not see: market fluctuations, cut
quality, clarity, and other missing features.

What happens when we fit models of different complexity to this data? A
#emph[linear fit] (a straight line) underfits the curved relationship. It
carries high bias, because a line simply cannot capture the quadratic shape, so
the error on the training data itself is already large. On the other hand, it
has low variance: no matter which particular sample of diamonds we draw, the
fitted line barely moves.

A #emph[degree-2 polynomial] matches the true generating process and achieves
the right fit. It is flexible enough to capture the curvature without
introducing spurious wiggles.

A #emph[degree-10 polynomial] goes too far. With ten degrees of freedom it can
chase every noisy data point, producing a wiggly curve that fits the training
sample almost perfectly (low bias) but shifts wildly from one sample to the next
(high variance). This is textbook overfitting: the model has memorized the noise
rather than learning the signal.

@fig:diamondpriceexample illustrates all three fits side by side, making it easy
to see how underfitting, right-fitting, and overfitting manifest as the model's
flexibility increases. The progression from line to quadratic to high-degree
polynomial is one of the most concrete ways to build intuition for the
bias-variance tradeoff.

#figure(
  image(
    "../lectures_source/figures/L05.2.Diamond_price_example.png",
    width: 80%,
  ),
  caption: [Diamond price example],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:diamondpriceexample>

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:127 '* Overfitting: Classification Example'
// Slide: Overfitting: Classification Example
#strong[Overfitting: Classification Example]

Consider a concrete scenario: you want to separate two classes using two
features $x_1$ and $x_2$, and the true class boundary happens to be parabolic.
Logistic regression lets you choose the complexity of that boundary by deciding
which polynomial terms to include.

If you use a linear decision boundary, $"logit"(w_0 + w_1 x_1 + w_2 x_2)$, the
model can only draw a straight line through feature space. A line cannot capture
a parabolic boundary, so the model underfits: it carries high bias because it
systematically misrepresents the true shape, but low variance because there are
only three parameters to estimate and they remain stable across different
training samples.

If you instead fit a quadratic boundary,
$"logit"(w_0 + w_1 x_1 + w_2 x_1^2 + w_3 x_1 x_2 + w_4 x_2^2)$, the model's
hypothesis class now includes parabolas. Since the true boundary is itself a
parabola, this is the right level of complexity: the model can represent the
real pattern without wasting capacity on noise.

If you go further and include high powers of $x_1$ and $x_2$, the decision
boundary becomes wiggly, threading through individual training points rather
than tracking the smooth underlying curve. This is overfitting: the model has
low bias because it can, in principle, approximate any shape, but high variance
because those many parameters shift dramatically from one training sample to the
next.

#figure(
  image("../lectures_source/figures/L05.2.Optimal_fit2.png", width: 80%),
  caption: [Optimal fit2],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:optimalfit2>

As @fig:optimalfit2 illustrates, the three regimes line up along a complexity
axis. The linear boundary is too rigid, the high-degree boundary is too
flexible, and the quadratic boundary sits at the sweet spot where model
complexity matches the true complexity of the data-generating process.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:145 '# Bias Variance Analysis'
// Slide: Bias Variance Analysis
#strong[Bias Variance Analysis]

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:147 '## From VC Analysis to Bias-Variance'
// Slide: From VC Analysis to Bias-Variance
== From VC Analysis to Bias-Variance

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:149 '* VC Analysis vs Bias-Variance Analysis'
// Slide: VC Analysis vs Bias-Variance Analysis
#strong[VC Analysis vs Bias-Variance Analysis]

Both VC analysis and bias-variance analysis are tools for understanding
generalization, and both revolve around the choice of hypothesis set $cal(H)$.
They frame the same underlying tension from different angles.

#strong[VC analysis] #cite("vapnikchervonenkis1971uniform") bounds the
out-of-sample error by adding a complexity penalty to the in-sample error:

$ E_"out" lt.eq E_"in" + Omega(cal(H)) $

The term $Omega(cal(H))$ grows with the capacity of the hypothesis set: a richer
$cal(H)$ can fit training data more tightly (lowering $E_"in"$), but the
complexity penalty widens, so the bound loosens. This gives a worst-case,
distribution-free guarantee: no matter what the true target is, the gap between
in-sample and out-of-sample performance is controlled by the VC dimension.

#strong[Bias-variance analysis] decomposes the out-of-sample error into two
competing terms:

$ E_"out" = "bias" + "variance" $

Bias measures how far the best hypothesis in $cal(H)$ is from the true target on
average, while variance measures how much the learned hypothesis fluctuates
across different training sets. A simple $cal(H)$ has high bias (it cannot
approximate the target well) but low variance (it is stable); a complex $cal(H)$
has low bias but high variance.

The two analyses complement each other, as @fig:vcanalysisvsbiasvarianceanalysis
illustrates. VC analysis works with a single, deterministic bound and makes no
assumptions about the data distribution: it tells you the worst that can happen.
Bias-variance analysis, by contrast, averages over the randomness of the
training set and decomposes the expected error into interpretable pieces: it
tells you where the error is coming from. Together they provide both a safety
net (VC) and a diagnostic lens (bias-variance) for choosing the right model
complexity.

#figure(
  image(
    "../lectures_source/figures/L05.2.VC_analysis_vs_Bias_Variance_analysis.png",
    width: 80%,
  ),
  caption: [VC analysis vs Bias Variance analysis],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:vcanalysisvsbiasvarianceanalysis>

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:165 '* Hypothesis Set and Bias-Variance Analysis'
// Slide: Hypothesis Set and Bias-Variance Analysis
#strong[Hypothesis Set and Bias-Variance Analysis]

#strong[Learning] consists in finding a function $g in cal(H)$ such that
$g approx f$, where $f$ is the unknown target function we wish to approximate.
The hypothesis set $cal(H)$ defines the space of candidate functions the learner
is allowed to consider, and the goal is to select the member of that space whose
predictions come closest to $f$ on unseen data.

The central #strong[tradeoff in learning] #cite("geman1992biasvariance") arises
from the tension between two sources of error. At one extreme, a hypothesis set
of low complexity (few free parameters, rigid functional forms) is easy to fit
reliably but may be too restrictive to capture the true structure of $f$; the
result is high bias and low variance. At the other extreme, a highly complex
hypothesis set (many parameters, flexible forms) can mold itself closely to any
training sample, but that flexibility makes it sensitive to the particular data
drawn; the result is low bias and high variance.
@tab:hypothesissetandbiasvarianceanalysis summarizes these two poles side by
side, showing how complexity shifts the balance between underfitting and
overfitting.

#figure(
  styled-table(
    headers: ("Low complexity pole", "High complexity pole"),
    rows: (
      ([Bias], [Variance]),
      ([Underfitting], [Overfitting]),
      ([Less complex $cal(H)$ / $h$], [More complex $cal(H)$ / $h$]),
      ([Approximation (in-sample)], [Generalization (out-of-sample)]),
    ),
    bold-first-col: false,
  ),
  caption: [Table of Low complexity pole, High complexity pole],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:hypothesissetandbiasvarianceanalysis>

Neither extreme is desirable on its own: a learner that always underfits ignores
real signal, while one that always overfits mistakes noise for signal. Effective
learning algorithms navigate between these poles, choosing a level of complexity
matched to the amount and quality of available data.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:185 '## Deriving the Bias-Variance Decomposition'
// Slide: Deriving the Bias-Variance Decomposition
== Deriving the Bias-Variance Decomposition

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:187 '* Decomposing Error in Bias-Variance (1/4)'
// Slide: Decomposing Error in Bias-Variance (1/4)
#strong[Decomposing Error in Bias-Variance (1/4)]

In a regression setup, the target is a real-valued function $f$ that we want to
approximate. We have a hypothesis set
$cal(H) = {h_1(bold(x)), h_2(bold(x)), dots, h_n(bold(x))}$, a training dataset
$cal(D)$ containing $N$ examples, and we measure performance using the squared
error $E_"out" = bb(E)[(g(bold(x)) - f(bold(x)))^2]$. The goal is to choose the
best function $g in cal(H)$ that approximates the unknown $f$.

What happens to the out-of-sample error $E_"out" (g)$ as a function of the
hypothesis set $cal(H)$ for a training set of $N$ examples? This question sits
at the heart of learning theory: the answer reveals a fundamental tension
between choosing a hypothesis set rich enough to capture the true function and
keeping it constrained enough to generalize from a finite sample.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:199 '* Decomposing Error in Bias-Variance (2/4)'
// Slide: Decomposing Error in Bias-Variance (2/4)
#strong[Decomposing Error in Bias-Variance (2/4)]

The final hypothesis $g$ produced by a learning algorithm depends on the
particular training set $D$ it was trained on. To make this dependency explicit,
we write $g^{(D)}$ and define its out-of-sample error as:

$
  E_("out") (g^{(D)}) eq.delta EE_(bold(x)) [(g^{(D)}(bold(x)) - f(bold(x)))^2]
$

This measures how well one specific learned hypothesis, trained on one specific
dataset, generalizes to unseen points drawn from the same distribution.

In practice, though, we care less about what happens with a single hypothesis or
a single dataset and more about the behavior of the entire hypothesis set
$cal(H)$ across the distribution of all possible training sets. Any particular
$D$ of $N$ examples is just one draw from that distribution; a different draw
would produce a different $g^{(D)}$ and a different out-of-sample error. To
remove the dependency on any one training set, we average $E_("out")(g^{(D)})$
over all possible training sets of size $N$:

$
  E_("out") (cal(H)) eq.delta EE_D [E_("out") (g^{(D)})] = EE_D [EE_(bold(x)) [(g^{(D)}(bold(x)) - f(bold(x)))^2]]
$

This double expectation, first over test points $bold(x)$ and then over training
sets $D$, gives a single number that characterizes how well the hypothesis set
$cal(H)$ and the learning algorithm perform #emph[on average], independent of
the luck of any particular data draw. It is this aggregate quantity that the
bias-variance decomposition will break apart into interpretable components.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:219 '* Decomposing Error in Bias-Variance (3/4)'
// Slide: Decomposing Error in Bias-Variance (3/4)
#strong[Decomposing Error in Bias-Variance (3/4)]

Since the squared difference $(g^{(D)}(bold(x)) - f(bold(x)))^2$ is
non-negative, we can switch the order of the expectations by Fubini's theorem,
writing the out-of-sample error as an outer expectation over $bold(x)$ wrapping
an inner expectation over training sets:

$
  E_"out" (cal(H)) = EE_(bold(x)) [ EE_D [ (g^{(D))(bold(x)) - f(bold(x)))^2 ] ]
$

This rewriting is useful because the inner quantity
$EE_D [(g^{(D)}(bold(x)) - f(bold(x)))^2]$ is now a function of $bold(x)$ alone,
and we can analyze it pointwise before averaging over the input space.

To decompose this inner expectation, define the #strong[average hypothesis] over
all possible training sets of a given size:

$ overline(g)(bold(x)) eq.delta EE_D [g^{(D)}(bold(x))] $

This is the hypothesis you would get by averaging the predictions of every model
trained on every possible dataset drawn from the same distribution. It is not a
hypothesis any single training run produces; it is a theoretical construct that
separates what the learning algorithm does on average from how individual runs
fluctuate around that average.

Now add and subtract $overline(g)(bold(x))$ inside the squared term. Expanding
the square and distributing $EE_D$ across the resulting three terms gives:

$
  E_"out" (cal(H))
  &= EE_(bold(x)) [ EE_D [ (g^{(D)}(bold(x)) - f(bold(x)))^2 ] ] \
  &= EE_(bold(x)) space EE_D [ (g^{(D)} - overline(g) + overline(g) - f)^2 ] \
  &= EE_(bold(x)) space EE_D [ (g^{(D)} - overline(g))^2 + (overline(g) - f)^2 + 2(g^{(D)} - overline(g))(overline(g) - f) ]
$

Because $EE_D$ is a linear operator and the term $(overline(g) - f)$ does not
depend on the training set $D$, we can pull it out of the expectation:

$
  E_"out" (cal(H)) = EE_(bold(x)) [ EE_D [(g^{(D)} - overline(g))^2] + (overline(g) - f)^2 + 2 EE_D [(g^{(D)} - overline(g))] (overline(g) - f) ]
$

The critical observation is the cross-term. By the definition of $overline(g)$,
we have $EE_D [g^{(D)} - overline(g)] = overline(g) - overline(g) = 0$, so the
entire cross-term vanishes. This is the same "centering kills the cross-term"
argument that appears whenever you decompose a squared deviation around a mean.
What remains is a clean sum of two non-negative quantities inside the outer
expectation, each with a distinct and interpretable meaning: one measuring how
much individual trained models scatter around the average hypothesis, and the
other measuring how far that average hypothesis sits from the true target
function.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:265 '* Decomposing Error in Bias-Variance (4/4)'
// Slide: Decomposing Error in Bias-Variance (4/4)
#strong[Decomposing Error in Bias-Variance (4/4)]

The cross term in the expansion,

$ EE_D [(g^((D)) - overline(g))] (overline(g) - f) $

disappears entirely. To see why, apply the expectation over $D$ to the first
factor: by definition, $EE_D [g^((D))] = overline(g)$, so
$EE_D [g^((D)) - overline(g)] = 0$. The second factor, $overline(g) - f$, does
not depend on $D$ at all (it is a fixed function of $x$), so the entire product
is zero regardless of $x$.

With the cross term gone, the expected out-of-sample error decomposes cleanly:

$
  E_"out" (cal(H)) &= EE_x [EE_D [(g^((D)) - overline(g))^2] + (overline(g)(x) - f(x))^2] \
  &= EE_x [EE_D [(g^((D)) - overline(g))^2]] + EE_x [(overline(g) - f)^2] quad & "(linearity of " EE_x")" \
  &= EE_x ["var"(x)] + EE_x ["bias"(x)^2] \
  &= "variance" + "bias"
$

The first line simply groups the two surviving squared terms under the outer
expectation over $x$. The second line splits that expectation by linearity,
since neither term involves an interaction between the two quantities. The third
line substitutes the pointwise definitions of variance and bias established
earlier. The final line integrates over $x$ to yield the scalar decomposition:
the expected prediction error of a hypothesis set equals its #strong[variance]
plus its #strong[bias]. Variance measures how much individual learned hypotheses
fluctuate around the consensus $overline(g)$; bias measures how far that
consensus itself sits from the true target $f$. Reducing one typically increases
the other, which is the bias-variance tradeoff at the heart of model selection.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:297 '* Interpretation of Average Hypothesis'
// Slide: Interpretation of Average Hypothesis
#strong[Interpretation of Average Hypothesis]

The #strong[average hypothesis] over all training sets, denoted
$overline(g)(bold(x))$, captures what the learning algorithm would produce "on
average" if we could repeat the entire training process across every possible
dataset of size $N$:

$ overline(g)(bold(x)) eq.delta EE_D [g^((D))(bold(x))] $

This quantity can be interpreted as the best hypothesis that the hypothesis
class $cal(H)$ can deliver when trained on $N$ samples, because it averages out
the noise introduced by any single dataset's idiosyncrasies. One subtlety worth
noting: $overline(g)$ is not necessarily a member of $cal(H)$ itself. The
average of many hypotheses drawn from a class can lie outside that class, just
as the average of several straight lines (each fit to a different sample) could,
in principle, trace out a curve that no single line in the class reproduces.

This idea closely parallels #emph[ensemble learning]. Imagine enumerating every
possible dataset $D$ of size $N$, training a separate hypothesis $g$ on each
one, and then averaging all of those learned hypotheses together. The resulting
predictor would be $overline(g)$. Ensemble methods used in practice (bagging,
for instance) approximate exactly this thought experiment: they cannot enumerate
all possible datasets, so they resample from the data they have and average the
resulting models. The bias-variance decomposition tells us why this averaging
helps: it reduces variance (the sensitivity to which particular dataset was
drawn) without changing the bias (which depends on $cal(H)$ itself, not on any
single sample).

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:311 '* Interpretation of Variance and Bias Terms'
// Slide: Interpretation of Variance and Bias Terms
#strong[Interpretation of Variance and Bias Terms]

The out-of-sample error of a hypothesis set $cal(H)$ decomposes cleanly into two
competing terms:

$ E_"out" (cal(H)) = "bias"^2 + "variance" $

This decomposition reveals exactly where generalization error comes from: part
of it is baked into the choice of model class itself, and part of it arises from
the randomness of whichever finite dataset we happen to train on.

The #strong[bias] term is defined as

$ "bias"^2 eq.delta EE_(bold(x)) [ (overline(g)(bold(x)) - f(bold(x)))^2 ] $

where $overline(g)$ is the average hypothesis, the function you would obtain by
averaging the learned hypotheses over every possible training set of a given
size. Bias does not depend on any particular dataset $D$; it measures how well
$cal(H)$ can approximate the true target $f$ even in the best case, given
infinitely many training sets. A high bias means the model class is too
restrictive to capture the target's shape, no matter how much data you throw at
it.

The #strong[variance] term is defined as

$
  "variance" eq.delta EE_(bold(x)) EE_D [ (g^((D))(bold(x)) - overline(g)(bold(x)))^2 ]
$

This measures how much the learned hypothesis $g^((D))$ fluctuates around
$overline(g)$ as the training set $D$ changes. If we had access to infinitely
many training sets, we could simply use $overline(g)$ and pay no variance cost
at all. In practice we train on a single dataset at a time, so each $g^((D))$
deviates from that ideal average, and that deviation is the price we pay for
having limited data. A complex model class with many degrees of freedom tends to
be highly sensitive to the particular sample it sees, driving variance up.

@fig:biasvariance illustrates the intuition behind these two error sources: bias
reflects a systematic offset from the target, while variance reflects the spread
of individual hypotheses around their own average.
@fig:biasvariancedecomposition shows how the total out-of-sample error
partitions into these two components, making explicit the tradeoff that governs
model selection: simplifying $cal(H)$ reduces variance but may increase bias,
and enriching $cal(H)$ reduces bias but typically increases variance.

#figure(
  image("../lectures_source/figures/L05.2.bias_variance.png", width: 80%),
  caption: [bias variance],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:biasvariance>

#figure(
  image(
    "../lectures_source/figures/L05.2.bias_variance_decomposition.png",
    width: 80%,
  ),
  caption: [bias variance decomposition],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:biasvariancedecomposition>

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:347 '* Variance and Bias Term Varying Cardinality of $\calH$'
// Slide: Variance and Bias Term Varying Cardinality of $\calH$
#strong[Variance and Bias Term Varying Cardinality of $cal(H)$]

Consider the simplest possible hypothesis set: one that contains a single
function, $cal(H) = {h eq.not f}$. Because there is no choice to make, the
learning algorithm always returns the same $h$ regardless of which training set
$D$ it sees. That means the #strong[variance] is exactly zero: different data
sets produce exactly the same output. The price is a potentially large
#strong[bias], since $h$ may be far from the true target $f$, and no amount of
data can close the gap. @fig:biasvariancetradeoffexample1 illustrates this
situation: every run lands on the same hypothesis, but that hypothesis sits at a
fixed distance from the truth.

#figure(
  image(
    "../lectures_source/figures/L05.2.Bias_Variance_tradeoff_example1.png",
    width: 80%,
  ),
  caption: [Bias Variance tradeoff example1],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:biasvariancetradeoffexample1>

Now consider the opposite extreme: a hypothesis set that contains many
functions, $cal(H) = {text("many hypotheses ") h}$. Here the bias can shrink to
zero (for instance, if the true target $f$ happens to live inside $cal(H)$, the
algorithm can in principle find it exactly). The tradeoff is a large
#strong[variance]. Because $cal(H)$ is rich, the particular training set $D$ the
algorithm happens to see steers it toward very different final hypotheses $g$. A
larger $cal(H)$ amplifies this effect: with more candidates to choose from, the
data-dependent wobble grows, and the selected $g$ may end up far from $f$ on any
single run. @fig:biasvariancetradeoffexample2 shows this second regime: the
cloud of possible outputs is centered closer to $f$, but the spread around that
center is wide.

#figure(
  image(
    "../lectures_source/figures/L05.2.Bias_Variance_tradeoff_example2.png",
    width: 80%,
  ),
  caption: [Bias Variance tradeoff example2],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:biasvariancetradeoffexample2>

These two extremes frame the core tension in model selection. A small hypothesis
set keeps variance under control at the cost of bias; a large one reduces bias
but pays in variance. The goal of learning is to find a hypothesis set whose
complexity sits at the sweet spot, minimizing the sum of both contributions to
the expected out-of-sample error.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:387 '## Bias-Variance Tradeoff in Practice'
// Slide: Bias-Variance Tradeoff in Practice
== Bias-Variance Tradeoff in Practice

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:389 '* Bias-Variance Trade-Off: Numerical Example (1/2)'
// Slide: Bias-Variance Trade-Off: Numerical Example (1/2)
#strong[Bias-Variance Trade-Off: Numerical Example (1/2)]

Consider a clean machine learning problem: the target function is
$f(x) = sin(pi x)$ over the interval $x in [-1, 1]$, with no noise added. You
have access to $f(bold(x))$ at only $N = 2$ data points. The question is
straightforward: given so little data, which hypothesis set should you use to
learn this target?

Two candidate hypothesis sets are on the table. The first is $cal(H)_0$, the set
of constant models $h(x) = b$, which can only output a flat horizontal line. The
second is $cal(H)_1$, the set of linear models $h(x) = a x + b$, which can fit a
line with arbitrary slope and intercept.

Which model is best? The answer depends entirely on what "best" means. From the
perspective of #emph[approximation], the linear model $cal(H)_1$ wins: it can
tilt to follow the sinusoid's shape more closely, achieving a smaller error when
you measure how well the best hypothesis in each set matches the true curve.
From the perspective of #emph[learning], though, the picture changes. Learning
means estimating the unknown function from just two data points, and with so few
samples the linear model's extra flexibility becomes a liability: different
pairs of points will produce wildly different fitted lines, so the expected
out-of-sample error may actually be worse than what the humble constant model
achieves on average. This tension between a model's ability to approximate the
target and its ability to learn reliably from limited data sits at the heart of
the bias-variance tradeoff.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:405 '* Bias-Variance Trade-Off: Numerical Example (2/2)'
// Slide: Bias-Variance Trade-Off: Numerical Example (2/2)
#strong[Bias-Variance Trade-Off: Numerical Example (2/2)]

Consider two hypothesis sets fitted to a noisy sinusoid: $g_0$, a constant
model, and $g_1$, a linear model. From a pure approximation standpoint, the
constant model yields $E_("out")(g_0) = 0.5$, while the line achieves
$E_("out")(g_1) = 0.2$. The line has more degrees of freedom and can track the
sinusoid's shape more closely, so its bias is lower. Approximation alone
therefore favors $g_1$.

Learning tells a different story. Suppose the training protocol picks two points
at random as the training set $D$, learns $g$ from $D$, and then averages the
out-of-sample error over all possible such data sets:

$ E_("out") = "bias"^2 + "variance" $

For the constant model, that decomposition gives
$E_("out")(g_0) = 0.5 + 0.25 = 0.75$: modest bias plus modest variance. For the
line model, it gives $E_("out")(g_1) = 0.2 + 1.69 = 1.9$: low bias but enormous
variance. Because a line fitted to just two points swings wildly from one sample
to the next, $g_1$ is far less stable than $g_0$. The constant model
#emph[learns better] than the line model, even though the line model
#emph[approximates better].

#figure(
  image(
    "../lectures_source/figures/L05.2.Bias_Variance_Numerical_Example1.png",
    width: 80%,
  ),
  caption: [Bias Variance Numerical Example1],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:biasvariancenumericalexample1>

#figure(
  image(
    "../lectures_source/figures/L05.2.Bias_Variance_Numerical_Example2.png",
    width: 80%,
  ),
  caption: [Bias Variance Numerical Example2],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:biasvariancenumericalexample2>

This is the bias-variance tradeoff in concrete numbers.
@fig:biasvariancenumericalexample1 shows the approximation side: how well each
hypothesis set can match the target function in principle.
@fig:biasvariancenumericalexample2 shows the learning side: once training-set
randomness is folded in, the variance penalty of the richer model overwhelms its
approximation advantage. A model that looks superior on paper (lower bias) can
perform worse in practice when data is scarce and variance dominates. Choosing
the right complexity is not about finding the best possible fit; it is about
finding the best #emph[learnable] fit given the amount of data available.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:440 '* Bias-Variance Curves'
// Slide: Bias-Variance Curves
#strong[Bias-Variance Curves]

#strong[Bias-variance curves] are plots of $E_("out")$ as a function of
increasing model complexity #cite("hastie2009elements"). They provide one of the
clearest visual summaries of the tradeoff every learner faces: too simple a
model misses the pattern (high bias), while too complex a model chases noise
(high variance).

The typical shape of these curves follows a consistent pattern. Both $E_("in")$
and $E_("out")$ start from the same point when the model is at its simplest. As
complexity grows, $E_("in")$ decreases monotonically and can even reach zero if
the model is flexible enough to interpolate every training point. The
out-of-sample error $E_("out")$, by contrast, is always larger than $E_("in")$
because it reflects performance on data the model has never seen. Since
$E_("out")$ decomposes into the sum of bias and variance, it takes on a
characteristic bowl shape: bias falls as complexity increases (the model can
represent finer structure), while variance rises (the model becomes more
sensitive to the particular training set). The bottom of the bowl marks the
optimal complexity, the point where the combined cost is lowest. To the left of
that minimum lies the #emph[high bias / underfitting] regime, where the model is
too rigid to capture the true relationship. To the right lies the #emph[high
  variance / overfitting] regime, where the model fits training noise at the
expense of generalization.

@fig:biasvariancecurve illustrates this bowl-shaped tradeoff, showing how
$E_("in")$ and $E_("out")$ diverge as complexity increases and where the optimal
fit sits between the two failure modes.

#figure(
  image("../lectures_source/figures/L05.2.bias_variance_curve.png", width: 80%),
  caption: [bias variance curve],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:biasvariancecurve>

@fig:biasvariancedecompositioncurves breaks $E_("out")$ into its bias and variance
components explicitly, making visible the crossover point where the decreasing
bias curve and the increasing variance curve sum to the minimum total error.

#figure(
  image(
    "../lectures_source/figures/L05.2.bias_variance_decomposition.png",
    width: 80%,
  ),
  caption: [bias variance decomposition],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:biasvariancedecompositioncurves>

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:468 '* Bias-Variance Curves and Regularization'
// Slide: Bias-Variance Curves and Regularization
#strong[Bias-Variance Curves and Regularization]

A #strong[model with regularization] learns both the model coefficients
$bold(w)$ and the model's complexity (for example, its VC dimension)
simultaneously. Rather than choosing a hypothesis set of fixed complexity and
then fitting within it, regularization folds the complexity choice directly into
the optimization problem.

The idea is to learn the optimal model $bold(w)(lambda)$ as a function of a
regularization parameter $lambda in {dots, 10^(-1), 1.0, 10, dots}$ by solving:

$
  bold(w)(lambda) = arg min_(bold(w)) E_("aug")(bold(w)) = E_("in")(bold(w)) + Omega(lambda)
$

Here $E_("in")(bold(w))$ is the in-sample error and $Omega(lambda)$ is a penalty
term that grows with model complexity. The parameter $lambda$ controls the
tradeoff between fitting the training data and keeping the model simple.
Sweeping over a grid of $lambda$ values and selecting the one that minimizes
validation error gives a principled way to navigate the bias-variance tradeoff
without explicitly enumerating hypothesis sets of different sizes.

The regularization parameter has a clean interpretation at its extremes. When
$lambda$ is small, the penalty barely constrains the optimization, so the model
remains complex, with low bias but high variance. When $lambda$ is large, the
penalty dominates and forces the coefficients toward zero, yielding a simple
model with high bias but low variance. Neither extreme is desirable on its own:
an intermediate value of $lambda$ strikes the best balance between underfitting
and overfitting, as @fig:regularization illustrates.

#figure(
  image("../lectures_source/figures/L05.2.regularization.png", width: 80%),
  caption: [regularization],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:regularization>

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:502 '* How to Measure the Model Complexity'
// Slide: How to Measure the Model Complexity
#strong[How to Measure the Model Complexity]

The number of features used as inputs to the model is itself a hyperparameter
choice, since adding or removing features changes the hypothesis space the
learner searches. Beyond feature count, several parameters govern the model's
form and its effective degrees of freedom. The #strong[VC dimension] $d_("VC")$
quantifies the capacity of a hypothesis class by measuring the largest set of
points it can shatter. The degree of a polynomial regressor or classifier
controls how flexible its decision boundary can be. In $k$-nearest neighbors,
the value of $k$ determines how many training points vote on each prediction: a
small $k$ yields a jagged, high-variance boundary, while a large $k$ smooths it
toward the majority class. Similarly, the parameter $nu$ in NuSVM upper-bounds
the fraction of training errors and lower-bounds the fraction of support
vectors, giving a more interpretable knob than the classical $C$ penalty.

The #strong[regularization parameter] $lambda$ balances the fit-to-data term
against a complexity penalty, shrinking coefficients toward zero to discourage
overfitting. A large $lambda$ produces a simpler model at the cost of higher
training error; a small $lambda$ lets the model track the training data more
closely but risks poor generalization. For neural networks, the number of
#emph[training epochs] plays an analogous role: too few epochs leave the network
underfitting, while too many allow it to memorize noise in the training set.
Early stopping, which halts training when validation performance plateaus,
effectively turns the epoch count into a regularization device.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:512 '## Noise and Overfitting'
// Slide: Noise and Overfitting
== Noise and Overfitting

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:514 '* Bias-Variance Decomposition with a Noisy Target'
// Slide: Bias-Variance Decomposition with a Noisy Target
#strong[Bias-Variance Decomposition with a Noisy Target]

The bias-variance decomposition extends naturally to the more realistic setting
where the target itself is noisy. Instead of observing $f(bold(x))$ directly, we
observe

$ y = f(bold(x); bold(w)) + epsilon = bold(w)^T bold(x) + epsilon $

where $epsilon$ is a zero-mean noise term that corrupts every measurement. This
matches most real-world data collection: sensors have measurement error, human
labels are inconsistent, and unmodeled variables introduce randomness that no
hypothesis can eliminate.

Carrying through the same decomposition used in the noiseless case, but now
accounting for the additional randomness from $epsilon$, yields a three-term
identity:

$
  E_"out" (cal(H)) = underbrace(bb(E)_(D, bold(x)) [(g^((D)) - overline(g))^2], "variance") + underbrace(bb(E)_(bold(x)) [(overline(g) - f)^2], "bias") + underbrace(bb(E)_(epsilon, bold(x)) [(f - y)^2], "noise")
$

The first two terms are the same variance and bias that appeared before. The
third term, #strong[stochastic noise], is entirely new: it measures how much the
observed target $y$ deviates from the true underlying function $f$. Because
$epsilon$ is independent of the hypothesis set, no amount of model tuning can
reduce this term. It sets an irreducible floor on the expected out-of-sample
error.

Each of the three contributions has a clear geometric interpretation:

- #emph[Variance]: the expected distance from any single learned hypothesis
  $g^((D))$ to the centroid $overline(g)$ of all hypotheses the algorithm could
  produce across different training sets. A complex model with many free
  parameters tends to have high variance because small changes in the data shift
  the solution substantially.
- #emph[Bias]: the distance from that centroid $overline(g)$ to the true
  noiseless function $f$. A model that is too simple cannot place $overline(g)$
  close to $f$ regardless of how much data it sees, so it suffers high bias.
- #emph[Noise]: the distance from the noiseless function $f$ to the actually
  observed target $y$. This component depends only on the data-generating
  process, not on the learner.

The practical consequence is that minimizing out-of-sample error requires
balancing variance against bias (the classic tradeoff), while accepting that the
noise floor cannot be crossed. Adding model complexity reduces bias but
increases variance; simplifying the model does the reverse. The optimal point is
wherever the sum of these two controllable terms is smallest, sitting just above
the irreducible noise level.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:543 '* Bias as Deterministic Noise'
// Slide: Bias as Deterministic Noise
#strong[Bias as Deterministic Noise]

The bias term can be interpreted as #strong[deterministic noise]. Bias is the
part of the target function that the hypothesis set cannot capture:

$ h^*(bold(x)) - f(bold(x)) $

where $h^*()$ is the best approximation of $f(bold(x))$ in the hypothesis set
$cal(H)$ (for instance, the average hypothesis $overline(g)(x)$). Because the
hypothesis set $cal(H)$ lacks the capacity to represent this residual, it can
never be learned no matter how much data is available. In that sense it behaves
exactly like stochastic noise: it contributes to error, yet no amount of
training removes it. @fig:deterministicnoise illustrates how this unlearnable
gap between the best-in-class hypothesis and the true target acts as an
irreducible noise floor imposed by the model's own complexity constraints rather
than by randomness in the data.

#figure(
  image("../lectures_source/figures/L05.2.Deterministic_Noise.png", width: 80%),
  caption: [Deterministic Noise],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:deterministicnoise>

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:568 '* Deterministic vs Stochastic Noise in Practice'
// Slide: Deterministic vs Stochastic Noise in Practice
#strong[Deterministic vs Stochastic Noise in Practice]

#figure(
  styled-table(
    headers: ("Property", "Deterministic noise", "Stochastic noise"),
    rows: (
      ([Fixed for a given $bold(x)$?], [Yes], [No]),
      ([Depends on $cal(H)$?], [Yes], [No]),
      ([Independent of $D$?], [Yes (also of $epsilon$)], [Yes]),
    ),
    bold-first-col: true,
  ),
  caption: [Table of Property, Deterministic noise, Stochastic noise],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:deterministicvsstochasticnoiseinpractice>

In practice, a machine learning algorithm cannot distinguish between stochastic
and deterministic noise, because both the hypothesis set $cal(H)$ and the
dataset $D$ are fixed before learning begins.
@tab:deterministicvsstochasticnoiseinpractice compares the two noise types side
by side, highlighting that their observable effects on training are identical.
Given only the training set, there is no way to tell whether the data was
generated by a noiseless but highly complex target function (whose complexity
$cal(H)$ cannot capture) or by a simpler target corrupted by random noise.
Either source of residual error looks the same to the learner: unexplained
variation that the chosen hypothesis class cannot fit. This equivalence
reinforces why controlling model complexity matters regardless of the noise
source.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:587 '* Deterministic vs Stochastic Noise Example (1/2)'
// Slide: Deterministic vs Stochastic Noise Example (1/2)
#strong[Deterministic vs Stochastic Noise Example (1/2)]

To build intuition for deterministic noise, consider a concrete experimental
setup. The learning task involves two different target functions: a noisy
10th-order polynomial (where stochastic noise has been added to the
observations) and a noiseless 50th-order polynomial (where the data points lie
exactly on the curve, with no random corruption at all). From each target,
$N = 15$ training points are generated.

Two hypothesis sets are then fit to each training set: $cal(H)_2$, a low-order
model restricted to 2nd-order polynomials, and $cal(H)_(10)$, a higher-order
model that can fit up to 10th-order polynomials. The critical observation is
that the learning algorithm sees only the 15 training samples; it has no way to
tell whether the residual error on those samples comes from stochastic noise
corrupting the labels or from deterministic noise (target complexity that the
hypothesis set simply cannot capture).

@fig:deterministicnoiseexample1 and @fig:deterministicnoiseexample2 illustrate
the fits that result from these two scenarios. When $cal(H)_2$ is applied to the
noisy 10th-order target, it underfits: it cannot represent the true 10th-order
shape, so the gap between its best fit and the real function acts exactly like
additional noise from the model's perspective. When $cal(H)_(10)$ is applied to
the noiseless 50th-order target, something similar happens: despite the absence
of any stochastic noise, the 10th-order model still cannot capture the
50th-order structure, and the uncapturable complexity again manifests as an
apparent noise floor. In both cases the learning algorithm confronts residuals
it cannot reduce, and it has no internal signal to distinguish one source from
the other. This equivalence is precisely what makes deterministic noise a useful
concept: overfitting is driven by the total effective noise the model faces,
regardless of whether that noise originates in corrupted measurements or in
target complexity beyond the model's reach.

#figure(
  image(
    "../lectures_source/figures/L05.2.deterministic_noise_example1.png",
    width: 80%,
  ),
  caption: [deterministic noise example1],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:deterministicnoiseexample1>

#figure(
  image(
    "../lectures_source/figures/L05.2.deterministic_noise_example2.png",
    width: 80%,
  ),
  caption: [deterministic noise example2],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:deterministicnoiseexample2>

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:615 '* Deterministic vs Stochastic Noise Example (2/2)'
// Slide: Deterministic vs Stochastic Noise Example (2/2)
#strong[Deterministic vs Stochastic Noise Example (2/2)]

Consider first a noisy, low-order target function. Fitting polynomials from 2nd
to 10th order reveals a familiar pattern: as the model's degrees of freedom
increase, $E_("in")$ drops because the hypothesis can chase every bump in the
training data, but $E_("out")$ climbs sharply because the model is fitting noise
rather than signal. @fig:deterministicnoiseexample3 and
@fig:deterministicnoiseexample5 illustrate this progression, showing how the gap
between in-sample and out-of-sample error widens with model complexity.

#figure(
  image(
    "../lectures_source/figures/L05.2.deterministic_noise_example3.png",
    width: 80%,
  ),
  caption: [deterministic noise example3],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:deterministicnoiseexample3>

#figure(
  image(
    "../lectures_source/figures/L05.2.deterministic_noise_example5.png",
    width: 80%,
  ),
  caption: [deterministic noise example5],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:deterministicnoiseexample5>

Perhaps surprisingly, the same phenomenon appears with a noiseless, high-order
target. Even when the true function is a clean 10th-order polynomial with no
stochastic noise at all, fitting 2nd through 10th order models to a small sample
still produces rising $E_("out")$ as the polynomial order grows: $E_("in")$
decreases with additional degrees of freedom, while $E_("out")$ increases
dramatically. The culprit is deterministic noise: the part of the true target
that the model cannot capture given limited data acts just like stochastic noise
from the learner's perspective. @fig:deterministicnoiseexample4 and
@fig:deterministicnoiseexample6 confirm that this overfitting pattern persists
even in the absence of any random noise.

#figure(
  image(
    "../lectures_source/figures/L05.2.deterministic_noise_example4.png",
    width: 80%,
  ),
  caption: [deterministic noise example4],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:deterministicnoiseexample4>

#figure(
  image(
    "../lectures_source/figures/L05.2.deterministic_noise_example6.png",
    width: 80%,
  ),
  caption: [deterministic noise example6],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:deterministicnoiseexample6>

This leads to a practical question about model selection. The naive, wrong
approach reasons that because the target is a 10th-order polynomial, a
10th-order hypothesis should fit it perfectly. That logic ignores sample size:
with only 15 data points, a 10-parameter model has far too much freedom relative
to the information available, and it overfits badly. The right approach starts
from the number of data points rather than the complexity of the target. A rule
of thumb drawn from VC analysis says that the effective degrees of freedom in
the model should be roughly the number of data points divided by 10:

$ "degrees of freedom" approx N / 10 $

With $N = 15$ training points, this guideline recommends using only one or two
degrees of freedom, a linear or quadratic fit, even when the true target is far
more complex. A simpler model that captures the broad trend will generalize
better than a complex one that memorizes a handful of samples.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:656 '* Amount of Data and Model Complexity'
// Slide: Amount of Data and Model Complexity
#strong[Amount of Data and Model Complexity]

Matching model complexity to the available resources is a central practical
concern. The right level of complexity depends on the #emph[data resources] (how
many training examples you have) and the #emph[signal-to-noise ratio] (how clean
those examples are). A common misconception is that model complexity should
track the complexity of the target function you are trying to learn; in fact, it
should not. A highly complex target function does not justify a highly complex
model if you lack the data to support it. With insufficient data, a complex
model will simply memorize noise rather than capture the true underlying
pattern.

A widely used rule of thumb links the VC dimension to the number of training
points:

$
  d_("VC") ("degrees of freedom of the model") = N ("number of data points") / 10
$

This says that for every free parameter (or degree of freedom) your model has,
you need roughly ten data points to fit it reliably. A model with 20 adjustable
parameters, for instance, calls for at least 200 training examples before you
can trust it to generalize. When the data is noisy, the requirement grows even
steeper: noise effectively dilutes the information each data point carries, so
more examples are needed to pin down the same number of parameters.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:670 '* Overfitting as a Function of Data Resources, Model Complexity, Noise'
// Slide: Overfitting as a Function of Data Resources, Model Complexity, Noise
#strong[Overfitting as a Function of Data Resources, Model Complexity, Noise]

The #strong[relative generalization error] quantifies the degree of overfitting
by comparing how much worse a model performs on unseen data relative to its
out-of-sample error:

$ "Relative Generalization Error" eq.delta frac(E_"out" - E_"in", E_"out") $

When this ratio is close to zero, the model generalizes well: its training error
and test error are nearly the same. As the ratio grows toward one (or beyond),
the model is memorizing training data rather than learning the underlying
pattern. This single number gives a concise summary of the gap between in-sample
and out-of-sample performance, making it easier to compare overfitting severity
across different models or training configurations.

Several factors systematically influence this ratio, as
@tab:overfittingasafunctionofdataresourcesmodelcomplexitynoise summarizes.
Increasing the amount of training data generally reduces overfitting, because a
larger sample leaves less room for the model to latch onto idiosyncratic
patterns. Increasing model complexity, on the other hand, tends to worsen
overfitting: a more flexible hypothesis set can fit training noise more tightly,
driving $E_"in"$ down while $E_"out"$ stays high or rises. Increasing noise in
the data has a similar effect, since noisier targets give an overparameterized
model more spurious structure to memorize. Understanding these three levers
(data size, model complexity, and noise level) is essential for diagnosing
overfitting in practice and choosing the right countermeasure: collecting more
data, simplifying the model, or cleaning the signal.

#figure(
  styled-table(
    headers: ("Increasing factor", "Effect on overfitting"),
    rows: (
      ([Data resources $N$], [Decreases]),
      ([Model complexity $d_(V C)$], [Increases]),
      ([Deterministic noise (target complexity)], [Increases]),
      ([Stochastic noise $sigma^2$], [Increases]),
    ),
    bold-first-col: false,
  ),
  caption: [Table of Increasing factor, Effect on overfitting],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:overfittingasafunctionofdataresourcesmodelcomplexitynoise>

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:688 '# Learning Curves'
// Slide: Learning Curves
#strong[Learning Curves]

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:690 '* Learning Curves vs Bias-Variance Curves'
// Slide: Learning Curves vs Bias-Variance Curves
#strong[Learning Curves vs Bias-Variance Curves]

#strong[Learning curves] are the dual of bias-variance curves, swapping which
quantity is held fixed and which is varied.

In #emph[bias-variance curves], the training set size $N$ stays fixed while the
model changes. The in-sample and out-of-sample errors are functions of model
complexity given a fixed dataset:

$ E_("in"), E_("out") = f(d_(V C) | N) $

The "model changes" axis can take several concrete forms: increasing the VC
dimension $d$, adding more features $p$, or reducing the regularization penalty
$lambda$. All three move the model along the complexity spectrum while the data
it trains on remains the same, producing the familiar U-shaped out-of-sample
error curve.

In #emph[learning curves], the relationship is reversed. The model is held fixed
and the training set size $N$ varies:

$ E_("in"), E_("out") = f(N | d_(V C)) $

As $N$ grows from a handful of examples toward infinity, in-sample error
typically rises (the fixed model can no longer memorize every point) while
out-of-sample error falls (more data gives the model a better picture of the
true pattern). The two curves converge toward the model's best achievable error
for that complexity level. @fig:learningcurvesvsbiasvariancecurves contrasts the
two perspectives side by side, showing how each holds one axis constant while
sweeping the other.

// rendered_images:begin
// ```tikz
// \bfseries
// % Axes
// \draw[->] (0,0) -- (7,0) node[below] {Number of Data Points, $N$};
// \draw[->] (0,0) -- (0,4) node[above] {Expected Error};
// 
// % Dotted convergence line
// \draw[dotted, thick] (0,1.5) -- (6.8,1.5);
// 
// % E_in curve
// \draw[thick, blue] plot[smooth, domain=0.4:6.5] (\x, {1.5 + 1.2/(0.6*\x + 0.4)});
// 
// % E_out curve
// \draw[thick, red] plot[smooth, domain=0.5:6.5] (\x, {1.5 - 1.0/(0.6*\x + 0.4)});
// 
// % Labels
// \node[red] at (5.8,0.95) {$E_{\text{in}}$};
// \node[blue] at (5.8,2.15) {$E_{\text{out}}$};
// ```
// label=fig:learningcurvesvsbiasvariancecurves
// caption=Diagram illustrating learning curves versus bias-variance curves.
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.2-Overfitting.typ.figs/Lesson05.2-Overfitting.2.png",
    width: 70%,
  ),
  caption: [Diagram illustrating learning curves versus bias-variance curves.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:learningcurvesvsbiasvariancecurves>
// render_images:end

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:737 '* Typical Form of Learning Curves'
// Slide: Typical Form of Learning Curves
#strong[Typical Form of Learning Curves]

Learning curves plot $E_"in"$ and $E_"out"$ as a function of the number of
training examples $N$, for a fixed model $h()$. They reveal how a model's
training error and generalization error evolve as more data becomes available. A
fundamental property holds for any $N$: $E_"out" gt.eq E_"in"$, because a model
always fits the data it was trained on at least as well as unseen data.

When $N$ is small, $E_"in"$ is small (potentially zero), because the model has
enough capacity to memorize every training example. This apparent perfection is
misleading: the model is overfitting, capturing noise rather than the underlying
pattern, so $E_"out"$ is large. As $N$ increases, two things happen
simultaneously. First, $E_"in"$ rises because the model can no longer perfectly
fit every example in the growing dataset. Second, $E_"out"$ falls because the
model is forced to learn genuine structure rather than memorize individual
points, and that structure transfers to unseen data. The generalization gap
$E_"out" - E_"in"$ therefore shrinks with increasing $N$.

In the limit as $N arrow.r oo$, both curves level off. $E_"in"$ reaches a
plateau representing the #emph[irreducible error]: the best the model can do
given its functional form, regardless of how much data it sees. $E_"out"$
likewise converges to a minimum. The remaining gap $E_"out" - E_"in"$ depends on
the complexity of the model: a more complex hypothesis class tends to maintain a
larger gap because it retains more capacity to overfit, while a simpler model
converges faster but may plateau at a higher irreducible error.
@fig:learningcurve illustrates this characteristic shape, with $E_"in"$ rising
from below and $E_"out"$ falling from above until they nearly meet.

#figure(
  image("../lectures_source/figures/L05.2.learning_curve.png", width: 80%),
  caption: [learning curve],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:learningcurve>

#figure(
  image("../lectures_source/figures/L05.2.learning_curve2.png", width: 80%),
  caption: [learning curve2],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:learningcurve2>

@fig:learningcurve2 shows a second view of the same phenomenon, reinforcing that
the qualitative behavior of learning curves is consistent across model families:
the gap narrows with more data, and the final plateau is governed by model
complexity relative to the true data-generating process.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:768 '* High-Bias vs High-Variance Regime'
// Slide: High-Bias vs High-Variance Regime
#strong[High-Bias vs High-Variance Regime]

The learning curve reveals two distinct regimes that govern how a model's
performance changes as the training set grows.

In the #strong[high-variance regime], the number of training points $N$ is
small. Here, $E_("in")$ is low because the model has enough capacity to fit the
few available examples closely. However, this close fit comes at a cost: the
learned hypothesis depends heavily on the particular training set $D$ that was
drawn. A different random sample of the same size would produce a noticeably
different model, so $E_("out")$ is much larger than $E_("in")$. The gap between
the two errors is wide, and the most effective remedy is straightforward:
collect more data. Each additional training point constrains the model further,
shrinking the variance and pulling $E_("out")$ closer to $E_("in")$.

#figure(
  image("../lectures_source/figures/L05.2.Irreducible_Error1.png", width: 80%),
  caption: [Irreducible Error1],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:irreducibleerror1>

In the #strong[high-bias regime], the training set is large. Now $E_("in")$ is
no longer small; it has risen and flattened out because the model has already
extracted everything its hypothesis class can represent. Adding more examples
does not help, since the bottleneck is no longer variance but the expressiveness
of the model itself. The gap between $E_("in")$ and $E_("out")$ may be narrow
(indicating good generalization in a statistical sense), yet both errors can
still be unacceptably high if the hypothesis class is too simple for the target
function. Reducing this plateau requires a richer model family, not more data.

#figure(
  image("../lectures_source/figures/L05.2.Irreducible_Error2.png", width: 80%),
  caption: [Irreducible Error2],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:irreducibleerror2>

Between these two extremes lies a transition where the marginal value of each
new training point steadily decreases. As @fig:irreducibleerror1 and
@fig:irreducibleerror2 illustrate, the in-sample error curve rises from near
zero while the out-of-sample error curve descends, and both converge toward a
shared asymptote set by the irreducible noise floor $sigma^2$. Reading the
curves together tells a practitioner which lever to pull: if $E_("in")$ and
$E_("out")$ are both high and flat, the problem is bias (switch to a more
expressive model); if $E_("out")$ is still falling steeply while $E_("in")$
stays low, the problem is variance (gather more data or regularize). This
diagnostic is one of the most practical takeaways from learning theory, turning
an abstract decomposition into a concrete decision about where to invest effort
next.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:855 '* Summary'
// Slide: Summary
#strong[Summary]

Overfitting occurs when a model fits the noise in the training data rather than
the underlying signal, and the bias-variance decomposition provides the formal
explanation for why this happens. The expected out-of-sample error decomposes as
$E_"out" = "bias"^2 + "variance"$, with an additional irreducible noise term
when the target function is noisy.

To summarize the key ideas from this discussion:

- #emph[Overfitting] means pushing model complexity past the point that
  minimizes $E_"out"$, trading better in-sample fit for worse out-of-sample
  generalization.
- The #emph[bias-variance decomposition] splits $E_"out"$ into bias, which
  measures how well the hypothesis set $cal(H)$ can approximate $f$ on average,
  and variance, which captures how much the learned hypothesis $g$ fluctuates
  across different training sets.
- #emph[Complexity control] through regularization and related knobs manages the
  bias-variance tradeoff; the rule of thumb $d_("VC") approx N \/ 10$ provides a
  practical guideline for matching model complexity to available data.
- #emph[Learning curves] offer the dual view of bias-variance analysis, showing
  how $E_"in"$ and $E_"out"$ converge as the training set size $N$ grows.

// From: msml610/lectures_source/Lesson05.2-Overfitting.smd:873 '* References'
// Slide: References
#strong[References]

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
