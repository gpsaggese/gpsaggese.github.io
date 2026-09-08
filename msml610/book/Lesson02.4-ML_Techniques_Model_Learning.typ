// git_hash=418671a2a-o1r timestamp=20260907_201646
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
  title: "L02.4: ML Techniques - Model Learning",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L02.4: ML Techniques - Model Learning")

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:11 '# The Optimization Problem'
// Slide: The Optimization Problem
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[The Optimization Problem]

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:13 '* Minimizing a Function'
// Slide: Minimizing a Function
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[Minimizing a Function]

The goal of optimization in machine learning is to minimize a scalar function
$J(bold(w))$ of $P$ variables $bold(w)$. A common instance is the #strong[in-sample
error]

$ E_(i n)(bold(w)) eq.def 1/N sum_(i=1)^N e(h_(bold(w))(bold(x)_i), y_i) $

which averages the pointwise error of a hypothesis $h_(bold(w))$ over $N$
training examples $(bold(x)_i, y_i)$.

Two broad strategies exist for finding the minimizer. The first is an
// TODO(ai_gp): Use #strong[...] instead of #emph[...] here since this term
// is being formally introduced/defined with the colon-based definition pattern
// (.claude/skills/typst.rules.md:## Highlighting and Emphasis)
#emph[analytical approach]: set the gradient of $J(bold(w))$ equal to zero and
solve for $bold(w)^*$ in closed form. This is elegant when it works, but many
objective functions do not admit a closed-form solution, or the solution is too
expensive to compute for large $P$. The second is a
// TODO(ai_gp): Use #strong[...] instead of #emph[...] here since this term
// is being formally introduced/defined with the colon-based definition pattern
// (.claude/skills/typst.rules.md:## Highlighting and Emphasis)
#emph[numerical approach]:
use an iterative method that updates $bold(w)$ step by step until $J(bold(w))$
reaches its minimum. Gradient descent is the prototypical example. A numerical
method remains applicable even when an analytical solution exists, making it the
more general-purpose tool, and it scales naturally to the high-dimensional
parameter spaces encountered in modern models.

@fig:gradientdescent2 illustrates how gradient descent iteratively moves toward
a minimum of the objective surface.

#figure(
  image("../lectures_source/figures/L02.4.Gradient_descent_2.png", width: 80%),
  // TODO(ai_gp): Caption must describe what the figure shows in a single clause,
  // not just label it. Example: "Gradient descent surface showing convergence to
  // minimum." (.claude/skills/typst.rules.md:## Figures: Required Elements)
  caption: [Gradient descent 2],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:gradientdescent2>

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:43 '* Gradient Descent: Intuition'
// Slide: Gradient Descent: Intuition
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[Gradient Descent: Intuition]

Imagine standing on a hilly surface with the goal of reaching the lowest point.
How do you get there? At each position, you look around to assess the slope in
every direction, then take a step along the #emph[steepest downhill direction]. You
repeat this process until you arrive at a point where no further descent is
possible. This intuitive procedure is precisely what gradient descent formalizes
for mathematical optimization, as @fig:gradientdescent1 illustrates.

#figure(
  image("../lectures_source/figures/L02.4.Gradient_descent_1.png", width: 80%),
  // TODO(ai_gp): Caption must describe what the figure shows, not just label it
  // (.claude/skills/typst.rules.md:## Figures: Required Elements)
  caption: [Gradient descent 1],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:gradientdescent1>

#strong[Gradient descent] is a general technique for minimizing
twice-differentiable functions. Starting from an initial guess, the algorithm
iteratively adjusts its parameters by moving in the direction of the negative
gradient (the direction of steepest decrease). In the general case, gradient
descent converges to a
// TODO(ai_gp): Use #strong[...] instead of #emph[...] here since this term
// is being formally defined with the "a point that is..." pattern
// (.claude/skills/typst.rules.md:## Highlighting and Emphasis)
#emph[local minimum], a point that is lower than all
nearby points but not necessarily the lowest point overall. However, when the
objective function $J(bold(w))$ is convex, every local minimum is also the
global minimum, so gradient descent is guaranteed to find the best possible
solution. Common models whose loss functions are convex include logistic
regression and linear regression, which is one reason these models are so widely
used in practice: their optimization landscape contains no misleading valleys or
plateaus.

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:73 '# Gradient Descent'
// Slide: Gradient Descent
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[Gradient Descent]

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:75 '* Gradient Descent with Fixed Learning Rate (1/3)'
// Slide: Gradient Descent with Fixed Learning Rate (1/3)
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[Gradient Descent with Fixed Learning Rate (1/3)]

Consider the contour plot of a loss function $E_{"in"}$ over weight space.
Gradient descent begins from an initial point $bold(w)(0)$, which might be
chosen randomly or set to the origin. At each step, the algorithm moves a fixed
distance $eta$ (#strong[the learning rate]) through weight space:

$ bold(w)(t + 1) = bold(w)(t) + eta hat(bold(v)) $

where $hat(bold(v))$ is a unit vector indicating the direction of the step. The
goal is to choose $hat(bold(v))$ so that $E_{"in"}(bold(w))$ decreases as much
as possible.

To see which direction accomplishes this, examine the change in the error:

$
  Delta E_{"in"} &= E_{"in"}(bold(w)(t + 1)) - E_{"in"}(bold(w)(t)) \
  &= E_{"in"}(bold(w)(t) + eta hat(bold(v))) - E_{"in"}(bold(w)(t)) \
  &approx eta nabla E_{"in"}(bold(w)(t))^T hat(bold(v)) + O(eta^2)
$

The last line follows from a first-order Taylor expansion around $bold(w)(t)$.
Standard gradient descent retains only the $O(eta)$ term and discards
higher-order contributions. Because $eta$ is a positive scalar, making
$Delta E_{"in"}$ as negative as possible requires choosing $hat(bold(v))$ to
minimize the inner product $nabla E_{"in"}(bold(w)(t))^T hat(bold(v))$; the unit
vector that achieves this points in the direction opposite the gradient. More
advanced methods such as #emph[conjugate gradient] retain the $O(eta^2)$ term as
well, which allows them to account for curvature information and often converge
faster, though at a higher per-step cost.

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:107 '* Gradient Descent with Fixed Learning Rate (2/3)'
// Slide: Gradient Descent with Fixed Learning Rate (2/3)
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[Gradient Descent with Fixed Learning Rate (2/3)]

The change in the in-sample error $E_(i n)$ when moving along a unit direction
$hat(bold(v))$ is approximately

$ Delta E_(i n) approx eta nabla E_(i n) (bold(w)(t))^T hat(bold(v)) $

This expression is a first-order Taylor approximation: the error shift is
proportional to the learning rate $eta$ and to how well the step direction
$hat(bold(v))$ aligns with the gradient. The scalar product $Delta E_(i n)$
reaches its most negative value when $hat(bold(v))$ points exactly opposite to
the gradient, that is, when

$
  hat(bold(v)) = - frac(nabla E_(i n) (bold(w)(t)), \| nabla E_(i n) (bold(w)(t)) \|)
$

At that choice the change equals $- eta \| nabla E_(i n) (bold(w)(t)) \|$, the
steepest possible decrease for a step of size $eta$.

Substituting this optimal direction back into the weight update gives the full
// TODO(ai_gp): Use #emph[...] instead of #strong[...] here since this is not
// a formal definition in a sentence like "Term is/refers to..." but rather a
// label/emphasis (.claude/skills/typst.rules.md:## Highlighting and Emphasis)
rule for the #strong[change in weights]:

$
  Delta bold(w) &= bold(w)(t + 1) - bold(w)(t) \
  &= eta hat(bold(v)) \
  &= - eta frac(nabla E_(i n) (bold(w)(t)), \| nabla E_(i n) (bold(w)(t)) \|)
$

Each iteration therefore moves the weight vector a fixed distance $eta$ in
whichever direction reduces the error most steeply. The method is called
#strong[gradient descent] precisely because it descends along the gradient of
the objective function, following the locally steepest downhill path at every
step.

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:133 '* Gradient Descent with Fixed Learning Rate (3/3)'
// Slide: Gradient Descent with Fixed Learning Rate (3/3)
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[Gradient Descent with Fixed Learning Rate (3/3)]

The gradient descent update formula specifies how each component of the weight
vector changes at every step. The full vector update is

$
  bold(w)(t + 1) = bold(w)(t) - eta frac(nabla E_(i n)(bold(w)(t)), \|nabla E_(i n)(bold(w)(t))\|)
$

and the corresponding per-coordinate form for the $j$-th weight is

$
  w_j (t + 1) = w_j (t) - eta frac(1, \|nabla E_(i n)(bold(w)(t))\|) frac(partial E_(i n)(bold(w)), partial w_j)
$

Each weight moves in the direction that locally reduces the in-sample error,
scaled by the learning rate $eta$ and normalized by the gradient's magnitude. A
critical detail is that the update of all components must be
#emph[simultaneous]: every partial derivative is evaluated at the current point
$bold(w)(t)$ before any component is changed, so no single coordinate's new
value contaminates the computation of another's update within the same step.

An #strong[iteration] of gradient descent refers to one complete application of
this update rule, producing a new weight vector from the current one. Counting
iterations gives a natural measure of computational effort and is the basis for
convergence criteria and stopping conditions.

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:153 '* Gradient Descent: Stopping Criteria'
// Slide: Gradient Descent: Stopping Criteria
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[Gradient Descent: Stopping Criteria]

When should gradient descent stop? In theory, the algorithm terminates when the
change in the in-sample error reaches zero, $Delta E_("in") = bold(0)$. In
practice, however, this exact condition may never be satisfied numerically due
to finite floating-point precision and the iterative nature of the updates. Two
practical stopping criteria are commonly used instead:

- #emph[Threshold on variation]: stop when the change in $E_("in")$ falls below
  a small threshold, $Delta E_("in") < theta$.
- #emph[Iteration budget]: stop after a predetermined number of iterations,
  regardless of whether full convergence has been reached.

Monitoring the optimization process also differs between theory and practice. In
theory, gradient descent requires only the derivatives of the cost function
$J(bold(w))$ to determine each update step. In practice, it is wise to
periodically recompute the cost function $J(bold(w))$ itself and verify that it
is actually decreasing. If the cost plateaus or begins to increase, that signals
a problem: the learning rate may be too large, causing the iterates to
overshoot, or the algorithm may have reached a region of the loss surface where
the gradient provides poor guidance. Periodically logging $J(bold(w))$ provides
a simple but effective diagnostic for catching these issues early.

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:167 '# Tuning Gradient Descent'
// Slide: Tuning Gradient Descent
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[Tuning Gradient Descent]

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:169 '* Choosing the Learning Rate'
// Slide: Choosing the Learning Rate
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[Choosing the Learning Rate]

Consider a one-dimensional convex function whose minimum we seek via gradient
descent. The behavior of the algorithm depends critically on the choice of
learning rate $eta$.

When $eta$ is too small, the linear approximation of $E_(i n)$ remains effective
at each step, so every update does move downhill, but the steps are tiny and
convergence to the minimum requires a large number of iterations.
@fig:gradientdescent3a illustrates this slow, cautious trajectory.

#figure(
  image("../lectures_source/figures/L02.4.Gradient_descent_3a.png", width: 80%),
  // TODO(ai_gp): Caption must describe what the figure shows, not just label it
  // (.claude/skills/typst.rules.md:## Figures: Required Elements)
  caption: [Gradient descent 3a],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:gradientdescent3a>

When $eta$ is too large, the linear approximation breaks down because
higher-order terms in the Taylor expansion dominate the actual function values.
Instead of descending smoothly, the iterates overshoot the minimum and bounce
back and forth across the valley, potentially diverging altogether.
@fig:gradientdescent3b shows this oscillatory behavior.

#figure(
  image("../lectures_source/figures/L02.4.Gradient_descent_3b.png", width: 80%),
  // TODO(ai_gp): Caption must describe what the figure shows, not just label it
  // (.claude/skills/typst.rules.md:## Figures: Required Elements)
  caption: [Gradient descent 3b],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:gradientdescent3b>

A natural idea is to vary $eta$ over the course of optimization. Starting with a
larger learning rate lets the algorithm make rapid initial progress, and then
gradually reducing $eta$ as a function of the iteration count allows it to
settle into a tighter neighborhood of the minimum. Smaller learning rates in
later stages can even help the optimizer find a better local minimum in
non-convex landscapes. The tradeoff is that a #strong[learning-rate schedule] introduces
an additional hyperparameter (or family of hyperparameters, such as a decay rate
and a decay schedule) that must itself be tuned, adding complexity to the
training pipeline. @fig:gradientdescent3c depicts the effect of a well-chosen
adaptive schedule, where early large steps give way to finer adjustments near
the optimum.

#figure(
  image("../lectures_source/figures/L02.4.Gradient_descent_3c.png", width: 80%),
  // TODO(ai_gp): Caption must describe what the figure shows, not just label it
  // (.claude/skills/typst.rules.md:## Figures: Required Elements)
  caption: [Gradient descent 3c],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:gradientdescent3c>

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:202 '* Gradient Descent with Variable Learning Rate'
// Slide: Gradient Descent with Variable Learning Rate
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[Gradient Descent with Variable Learning Rate]

When using #strong[gradient descent with a fixed learning rate], every update
moves the same distance through weight space regardless of how steep the surface
is. The update rule normalizes the gradient so that only its direction matters:

$ Delta bold(w) = - eta frac(nabla J, \|nabla J\|) $

This keeps the step size constant at $eta$, but that uniformity creates a
problem. On steep regions of the loss surface, a fixed step may be too small,
making progress painfully slow. Near a minimum, where the surface flattens out,
that same fixed step may be too large, causing the optimizer to overshoot and
bounce back and forth across the valley instead of settling in.

To converge quickly, the learning rate should adapt to the local geometry. The
optimizer should take large steps when the gradient is large (the surface is
steep and far from a minimum) and small steps when the gradient is small (the
surface is nearly flat and close to a minimum). In other words, the ideal
learning rate is proportional to the gradient magnitude: $eta prop \|nabla J\|$.

#strong[Gradient descent with a variable learning rate] achieves exactly this by
dropping the normalization and using the raw gradient directly:

$ Delta bold(w) = - eta nabla J $

Because the gradient vector $nabla J$ is not divided by its own norm, its
magnitude naturally scales the step size. When the surface is steep,
$\|nabla J\|$ is large and the update moves farther; when the surface is gentle,
$\|nabla J\|$ is small and the update shrinks. The scalar $eta$ still controls
the overall scale, but the gradient's magnitude now acts as a built-in adaptive
factor. This simple change gives the optimizer the variable-rate behavior needed
for efficient convergence: aggressive movement far from the optimum and cautious
refinement near it.

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:222 '* Feature Scaling in Gradient Descent'
// Slide: Feature Scaling in Gradient Descent
// TODO(ai_gp): Replace standalone #strong[...] with real Typst heading syntax
// (= for level-1, == for level-2, etc.) (.claude/skills/typst.rules.md:##
// Structural Hierarchy)
#strong[Feature Scaling in Gradient Descent]

Unscaled features distort the error surface. When one feature ranges from 1 to
1000 while another ranges from 0.01 to 1, the level sets of $E_"in" (bold(w))$
become elongated ellipses rather than circles. A single learning rate cannot
serve both directions well: it is too large along the steep, short-axis
direction (causing the updates to overshoot and bounce) and too small along the
flat, long-axis direction (causing the optimizer to crawl). The result is slow,
unstable convergence that zigzags across the narrow valley instead of heading
straight for the minimum.

The fix is to rescale every feature so that all of them span comparable ranges
before training begins. Two common choices are #emph[min-max scaling], which
maps each feature to the interval $[0, 1]$, and #emph[standardization], which
centers each feature at zero with unit variance. Either approach reshapes the
elongated ellipses into something much closer to circles, so a single learning
rate works reasonably well in every direction and gradient descent converges
faster and more smoothly.

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:236 '# Scaling to Large Datasets'
// Slide: Scaling to Large Datasets
#strong[Scaling to Large Datasets]

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:238 '* Issues with Batch Gradient Descent'
// Slide: Issues with Batch Gradient Descent
#strong[Issues with Batch Gradient Descent]

Consider the squared error over $N$ training samples, defined as

$
  E_(i n)(bold(w)) := 1 / N sum_(i=1)^N (h_(bold(w))(bold(x)_i) - y_i)^2
$

This measures the average discrepancy between the hypothesis $h_(bold(w))$ and
the observed targets across the entire dataset. Minimizing it drives the model
toward a set of weights that fits the training data well.

#strong[Batch Gradient Descent (BGD)] updates the weight vector by stepping in
the direction opposite to the normalized gradient of the in-sample error:

$ bold(w)(t + 1) = bold(w)(t) - eta frac(nabla E_(i n), \|nabla E_(i n)\|) $

Writing out the coordinate-level update for the squared error loss makes the
computation concrete. Each weight component $w_j$ changes according to

$
  w_j (t + 1) = w_j (t) - eta frac(2, N) sum_(i=1)^N (h_(bold(w))(bold(x)_i) - y_i) frac(partial h_(bold(w))(bold(x)_i), partial w_j)
$

The inner sum sweeps over every training example, accumulating the product of
the current residual and the local sensitivity of the hypothesis to $w_j$. Only
after that full pass does the algorithm take a single step.

That full-dataset requirement is the chief drawback of batch gradient descent.
When $N$ is large (say, $N = 10^6$), every single weight update demands a
gradient evaluation that touches all $10^6$ examples, making each iteration
computationally expensive. The entire dataset must also reside in memory at
once, which can be prohibitive for large-scale problems. These costs motivate
the stochastic and mini-batch variants discussed next, which trade a noisier
gradient estimate for dramatically cheaper per-step computation.

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:263 '* Stochastic Gradient Descent'
// Slide: Stochastic Gradient Descent
#strong[Stochastic Gradient Descent]

#strong[Stochastic Gradient Descent] (SGD) updates the weights using only a
single training example chosen at random, rather than computing the gradient
over the entire dataset. This makes each update far cheaper, which is especially
valuable when the training set is large.

// TODO(ai_gp): Use #algorithm("Stochastic Gradient Descent", [...]) instead of
// a bare numbered list for this procedure (.claude/skills/typst.rules.md:##
// Algorithms and Pseudocode)

The procedure works as follows:

1. Pick one example $(bold(x)_n, y_n)$ at random from the training set.
2. Compute the gradient of the loss for that single example,
  $nabla e(h(bold(x)_n), y_n)$, and form the weight update
  $Delta bold(w) = -eta nabla e$.
3. Apply the update to each weight:

$
  w_j (t + 1) = w_j (t) - 2 eta (h_(bold(w))(bold(x)_t) - y_t) frac(partial h_(bold(w))(bold(x)_t), partial w_j)
$

Because each step relies on a single randomly selected example, the path SGD
traces through weight space is noisy: it does not follow the smooth trajectory
of batch gradient descent but instead oscillates, sometimes moving away from the
optimum before correcting course. Near a local minimum the updates do not settle
cleanly; they bounce around the basin rather than converging to a fixed point.

Despite this apparent randomness, SGD works because the gradient $nabla e$ is a
function of a random variable $bold(x)_n$, and its expected value equals the
full-batch gradient. Concretely:

$
  bb(E)[nabla e] = frac(1, N) sum nabla e(h(bold(x)_n), y_n) = nabla frac(1, N) sum e(h(bold(x)_n), y_n) = nabla E_("in")
$

In other words, although any single SGD step may point in a direction that
differs from the true gradient, on average the steps point the same way as the
batch gradient. Over many iterations the random deviations cancel out, and the
algorithm makes steady progress toward a minimum of the in-sample error. The
oscillation that is a drawback in the short term actually confers a benefit: the
noise helps SGD escape shallow local minima that batch descent might get trapped
in, often leading to solutions that generalize better.

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:294 '* Mini-Batch Gradient Descent'
// Slide: Mini-Batch Gradient Descent
#strong[Mini-Batch Gradient Descent]

#strong[Mini-batch gradient descent] brings together characteristics of both
#emph[batch] and #emph[stochastic gradient descent]. Rather than computing the
gradient over the entire training set (as in batch gradient descent) or from a
single example (as in stochastic gradient descent), mini-batch gradient descent
uses a subset of $b$ examples to compute each update to the current weight
vector. Here $b$ is called the #emph[batch size], and common choices are
$b = 32$ or $b = 64$. This middle ground reduces the variance of parameter
updates compared to the single-sample stochastic approach, while remaining far
more computationally tractable per step than a full-batch computation.
Mini-batch updates also benefit from hardware-level parallelism on modern GPUs,
which are optimized for matrix operations on moderately sized batches. The
choice of batch size introduces its own tradeoff: smaller batches inject more
noise into the optimization trajectory (which can help escape shallow local
minima), while larger batches yield smoother, more stable gradient estimates at
the cost of reduced regularization effect and higher memory usage per step.

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:302 '* SGD vs BGD vs Mini-Batch'
// Slide: SGD vs BGD vs Mini-Batch
#strong[SGD vs BGD vs Mini-Batch]

As @tab:sgdvsbgdvsminibatch shows, Table of Aspect, Batch GD, Stochastic GD,
Mini-Batch GD.

#figure(
  styled-table(
    headers: ("Aspect", "Batch GD", "Stochastic GD", "Mini-Batch GD"),
    rows: (
      (
        "Computation",
        "Uses all examples",
        "One random example at a time",
        "A random subset (b examples) at a time",
      ),
      (
        "Memory",
        "Requires all examples in memory",
        "Requires little memory",
        "Requires little memory",
      ),
      (
        "Randomization",
        "More likely to terminate in flat regions",
        "Avoids local minima due to randomness",
        "Some randomness, less noisy than SGD",
      ),
      (
        "Regularization",
        "No implicit regularization",
        "Oscillations act as regularization",
        "Mild implicit regularization",
      ),
      (
        "Parallelization",
        "Can be parallelized",
        "Less parallel-friendly",
        "Can be parallelized within a batch",
      ),
      ("Online learning", "Not suitable", "Suitable", "Suitable"),
    ),
    bold-first-col: true,
  ),
  // TODO(ai_gp): Caption must not list out every column header. Instead, describe
  // what the table shows in one short clause. Example: "Comparison of gradient
  // descent variants." (.claude/skills/typst.rules.md:## Figures: Required Elements)
  caption: [Table of Aspect, Batch GD, Stochastic GD, Mini-Batch GD],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:sgdvsbgdvsminibatch>

// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:320 '* On-Line Learning and Gradient Descent'
// Slide: On-Line Learning and Gradient Descent
#strong[On-Line Learning and Gradient Descent]

In many applications, training data arrives as a
// TODO(ai_gp): Use #emph[...] instead of #strong[...] here since this is
// emphasis/contrast, not a formal definition in a sentence like "Term
// is/refers to..." (.claude/skills/typst.rules.md:## Highlighting and Emphasis)
#strong[continuous stream]
rather than a fixed batch. The model must incorporate new observations on the
fly, updating its parameters as each example (or small group of examples)
becomes available. This contrasts sharply with the #emph[offline setting], where the
entire dataset is collected up front and the model is trained once.

Real-time systems make this requirement especially pressing. A deployed model
needs to adapt to new data points without full retraining, and it must handle
variation in the underlying process dynamics: the distribution generating the
data may shift over time, so yesterday's optimal parameters may no longer be
adequate today.

// rendered_images:begin
// ```graphviz
// digraph StreamModel {
//     rankdir=LR;
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.50;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//     // Node styles
//     DataStream [label=<Stream of data<BR/>x<SUB>i</SUB>>, fillcolor=white, shape=plaintext];
//     Model      [label="Model", fillcolor="#C6A6F4"];
//     Output     [label=<Output<BR/>y<SUB>i</SUB>>, fillcolor=white, shape=plaintext];
// 
//     // Edges
//     DataStream -> Model;
//     Model -> Model [label=<w<SUB>i</SUB>>, fontsize=10];
//     Model -> Output;
// }
// ```
// label=fig:onlinelearningandgradientdescent
// caption=Diagram relating Model
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.4-ML_Techniques_Model_Learning.typ.figs/Lesson02.4-ML_Techniques_Model_Learning.1.png",
    width: 70%,
  ),
  // TODO(ai_gp): Caption is too vague. Describe what the diagram shows more
  // specifically. Example: "Model architecture for streaming data updates in
  // online learning." (.claude/skills/typst.rules.md:## Figures: Required
  // Elements)
  caption: [Diagram relating Model],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:onlinelearningandgradientdescent>
// render_images:end

Stochastic gradient descent and mini-batch gradient descent are naturally
#emph[suitable for online learning], as @fig:onlinelearningandgradientdescent
illustrates. Because these methods update the model one example (or one small
batch) at a time, they fit directly into a #emph[streaming pipeline]: each incoming
observation triggers a parameter update, after which the raw data point can be
discarded. The model itself serves as a "compressed" representation of
everything seen so far, which is valuable when storing every data point is
impractical. Consider stock market prediction, where tick-by-tick price data
accumulates at enormous volume, or training a language model on live chat data,
where older conversations can be dropped after the model has absorbed their
signal.

That said, discarding the data is generally #emph[a bad idea] whenever it can be
avoided. Once raw examples are gone, there is no way to revisit them if the
learning rate was poorly chosen, if the model class changes, or if a subtle bug
corrupted earlier updates. Retaining at least a representative buffer or summary
statistics provides a safety net that pure online updates lack.

// Revert garbled section at top, restore proper context


// From: msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd:369 '* Map-Reduce for Batch Gradient Descent'
// Slide: Map-Reduce for Batch Gradient Descent
#strong[Map-Reduce for Batch Gradient Descent]

Batch gradient descent lends itself naturally to distributed computation because
its core operation, summing gradient contributions across the entire dataset, is
embarrassingly parallel. This makes it a good fit for the #strong[map-reduce]
paradigm, a big data framework that splits distributed computation into two
phases. In the #emph[map step], $k$ worker machines each compute partial
gradient sums over their local shard of the data. In the #emph[reduce step],
those $k$ partial sums are sent to a single node that accumulates them into the
full gradient, which is then used to update the model parameters. As
@fig:mapreduceforbatchgradientdescent illustrates, the shuffle-and-sort stage
routes each worker's partial gradient to the aggregation node, which produces
the aggregated gradient and applies it to the current model parameters. Many
learning algorithms beyond plain gradient descent share this sum-then-update
structure, so map-reduce serves as a general-purpose backbone for large-scale
model training whenever the objective decomposes into independent per-example
terms.

// rendered_images:begin
// ```graphviz
// digraph BayesianFlow {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     // Operation nodes (rounded boxes)
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//     Map_0 [label=<Map<SUB>0</SUB><BR/>Compute Gradient>, fillcolor="#A6C8F4"];
//     Map_1 [label=<Map<SUB>1</SUB><BR/>Compute Gradient>, fillcolor="#A6C8F4"];
//     Map_n [label=<Map<SUB>n</SUB><BR/>Compute Gradient>, fillcolor="#A6C8F4"];
//     ShuffleSort [label="Shuffle & Sort", fillcolor="#C6A6F4"];
//     Update [label="Update Model", fillcolor="#D2B48C"];
// 
//     // Variable/value nodes (plain ellipses with white background)
//     node [shape=ellipse, style=filled, fontname="Helvetica", fontsize=12, penwidth=1.4, fillcolor=white];
// 
//     DataShard_0  [label=<DataShard<SUB>0</SUB>>];
//     DataShard_1  [label=<DataShard<SUB>1</SUB>>];
//     DataShard_n  [label=<DataShard<SUB>n</SUB>>];
// 
//     Gradient_0 [label=<Gradient<SUB>0</SUB>>];
//     Gradient_1 [label=<Gradient<SUB>1</SUB>>];
//     Gradient_n [label=<Gradient<SUB>n</SUB>>];
// 
//     AggregatedGradient [label="Aggregated Gradient"];
//     Model [label="Model Parameters"];
// 
//     // Force ranks
//     { rank=same; DataShard_0; DataShard_1; DataShard_n; }
//     { rank=same; Map_0; Map_1; Map_n; }
//     { rank=same; Gradient_0; Gradient_1; Gradient_n; }
// 
//     // Edges
//     DataShard_0 -> Map_0 -> Gradient_0;
//     DataShard_1 -> Map_1 -> Gradient_1;
//     DataShard_n -> Map_n -> Gradient_n;
// 
//     Gradient_0 -> ShuffleSort;
//     Gradient_1 -> ShuffleSort;
//     Gradient_n -> ShuffleSort;
// 
//     ShuffleSort -> AggregatedGradient;
//     AggregatedGradient -> Update;
//     Model -> Update -> Model;
// }
// ```
// label=fig:mapreduceforbatchgradientdescent
// caption=Diagram relating Shuffle & Reduce
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.4-ML_Techniques_Model_Learning.typ.figs/Lesson02.4-ML_Techniques_Model_Learning.2.png",
    width: 70%,
  ),
  // TODO(ai_gp): Caption is too vague. Describe what the diagram shows more
  // specifically. Example: "Map-reduce architecture for distributed gradient
  // computation across worker nodes." (.claude/skills/typst.rules.md:## Figures:
  // Required Elements)
  caption: [Diagram relating Shuffle & Reduce],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:mapreduceforbatchgradientdescent>
// render_images:end