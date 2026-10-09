# Surrogate Losses as a Differentiable Proxy for Discrete Objectives

## Status
- **Status:**: draft
- **Complete Specs:**: 20%

## Core Idea

- Many real objectives are piecewise-constant or discontinuous in the model
  parameters (0/1 classification accuracy, ranking metrics like NDCG,
  top-$k$ retrieval accuracy, F1 score). These objectives have zero or
  undefined gradient almost everywhere, so gradient descent cannot directly
  optimize them. The standard fix is to replace the true objective with a
  smooth, convex (or at least differentiable) surrogate loss (cross-entropy,
  hinge loss, soft-F1, smoothed top-$k$ loss) that is easy to optimize and
  that, empirically or provably, correlates with the true objective.
- The central hypothesis to test: minimizing a well-chosen surrogate loss
  approximately optimizes the true discrete objective, but this
  correspondence is conditional (on model capacity, data distribution, and
  the specific surrogate) rather than automatic. Classification-calibration
  theory gives conditions under which surrogate risk minimization implies
  0/1 risk minimization in the infinite-data, infinite-capacity limit, but
  says little about finite-sample, finite-capacity behavior, where the two
  objectives can rank hypotheses differently. The research task is to build
  a small, fully controlled setting where both the true and surrogate risks
  can be computed exactly, and to measure exactly when and how their
  optima diverge.
- This approach is distinct from the other differentiable-relaxation
  techniques in the parent brainstorm (Gumbel-Softmax, Sinkhorn, STE,
  continuous relaxation of combinatorial variables). Those relax the
  *decision variables* (turning a discrete choice into a continuous one,
  then rounding). Surrogate losses relax only the *objective/loss function*
  that scores a fixed set of variables, which can remain fully discrete
  (a decision rule, a threshold) or fully continuous (real-valued model
  weights). The non-differentiability being fixed here comes from the
  scoring function ($\mathbb{1}[\cdot]$, rank position, top-$k$ membership),
  not from the underlying parameters being categorical.

## Formalization

### General surrogate-loss setup

- Input space $\mathcal{X}$, label space $\mathcal{Y} = \{-1, +1\}$ for
  binary classification, a scoring function $f_\theta: \mathcal{X} \to
  \mathbb{R}$ parameterized by $\theta$ (continuous, e.g. neural network
  weights), and prediction rule $\hat y = \mathrm{sign}(f_\theta(x))$
- True (discrete) objective, the 0/1 loss:

```latex
\ell_{01}(y, f(x)) = \mathbb{1}[y \, f(x) \le 0], \qquad
R_{01}(\theta) = \mathbb{E}_{(x,y)}\big[\ell_{01}(y, f_\theta(x))\big]
```

  Empirical accuracy is $1 - \hat R_{01}(\theta)$ on a finite sample.
  $R_{01}$ is piecewise-constant in $\theta$: it only changes value when
  some point crosses the decision boundary, so
  $\nabla_\theta R_{01}(\theta) = 0$ almost everywhere.
- A surrogate loss $\phi: \mathbb{R} \to \mathbb{R}_{\ge 0}$ is applied to
  the margin $z = y f(x)$ in place of $\mathbb{1}[z \le 0]$, giving
  surrogate risk $R_\phi(\theta) = \mathbb{E}[\phi(y f_\theta(x))]$. A
  surrogate is useful when it is:
  - an upper bound on the 0/1 loss: $\phi(z) \ge \mathbb{1}[z \le 0]$
  - differentiable (or subdifferentiable) in $z$, hence in $\theta$ via
    the chain rule
  - classification-calibrated: minimizing $\phi$-risk over all measurable
    $f$ also minimizes 0/1 risk (Bartlett, Jordan & McAuliffe 2006 give the
    exact convexity condition: $\phi$ convex, differentiable at 0, and
    $\phi'(0) < 0$)

### Two candidate surrogates

- Logistic / cross-entropy loss (used by logistic regression and neural
  classifiers trained with a sigmoid + BCE head):

```latex
\phi_{ce}(z) = \log(1 + e^{-z})
```

- Hinge loss (used by SVMs, margin-based classifiers):

```latex
\phi_{hinge}(z) = \max(0, 1 - z)
```

  Both are convex, both upper-bound $\mathbb{1}[z \le 0]$ up to a constant
  ($\phi_{ce}(0) = \log 2$, so $\phi_{ce}/\log 2$ upper-bounds it exactly;
  $\phi_{hinge}(0) = 1$), and both are classification-calibrated. They
  differ in how they penalize the margin: cross-entropy is smooth
  everywhere and keeps pushing weight on already-correct, high-confidence
  points ($z \gg 0$ still contributes a small nonzero gradient); hinge is
  exactly zero (and exactly zero gradient) once $z \ge 1$, so it stops
  caring about points that are already safely classified. This difference
  is exactly what is expected to drive divergence between surrogate-optimal
  and accuracy-optimal solutions under outliers and class imbalance.

### Concrete worked example: linear binary classification

- Data: $(x_i, y_i)_{i=1}^n$, $x_i \in \mathbb{R}^2$ drawn from two
  overlapping Gaussians (one per class), so the Bayes-optimal accuracy is
  strictly below 1 and the classes are not linearly separable, forcing
  real tradeoffs rather than a degenerate perfect-separation solution
- Hypothesis class: linear scoring function $f_{w,b}(x) = w^\top x + b$
  with $w \in \mathbb{R}^2$, $b \in \mathbb{R}$ (3 continuous parameters,
  low enough dimension for the true-objective search below to be tractable)
- True objective, restated in this parameterization:
  $R_{01}(w, b) = \frac{1}{n}\sum_i \mathbb{1}[y_i (w^\top x_i + b) \le 0]$,
  a function of continuous $(w, b)$ that is piecewise-constant (it only
  changes as the separating line sweeps past a data point), making the
  point explicit that the *variables* here are continuous while the
  *objective* is what is non-smooth
- Surrogate objectives: $R_{\phi_{ce}}(w, b)$ and $R_{\phi_{hinge}}(w, b)$
  as defined above, both smooth (a.e.) in $(w, b)$ and optimizable by
  gradient descent / subgradient descent

## Key Examples

- **Five known real-world uses of surrogate/smoothed losses**:
  - Binary classification: cross-entropy (logistic loss) as the standard
    differentiable surrogate for 0/1 classification error, used in nearly
    every neural classifier
  - Support vector machines: hinge loss as a margin-based surrogate for
    0/1 misclassification error, with the margin term giving both a
    differentiable objective and a generalization bound
  - Learning to rank: NDCG and MAP are rank-based, non-differentiable
    (they depend on the sort order, not on scores directly); methods like
    RankNet/LambdaRank and ListNet use smooth pairwise or listwise
    surrogate losses that approximate ranking-metric gradients
  - Top-$k$ classification: top-$k$ accuracy (is the true label among the
    $k$ highest-scored classes) is a discrete, non-differentiable
    indicator; smoothed top-$k$ hinge/SVM losses (Lapin, Hein & Schiele
    2016) give a convex differentiable surrogate
  - Non-decomposable metrics (F1, AUC) in robust/imbalanced classification:
    F1 and AUC are not simple sums of per-example losses, so surrogate
    constructions (soft-F1, surrogate AUC losses, ramp loss for
    adversarially robust 0/1 loss) are used to make them optimizable by
    SGD (Eban et al. 2017)
- **Concrete worked example (this idea's testbed)**: overlapping-Gaussian
  binary classification with a linear scorer $f_{w,b}(x) = w^\top x + b$;
  true objective is 0/1 accuracy, candidate surrogates are cross-entropy
  and hinge loss, exactly as formalized above. This is the pair of
  surrogates directly compared in the implementation plan.
- **Failure/divergence mode to characterize**: inject a small number of
  mislabeled or extreme-outlier points into the training set. Cross-entropy
  keeps a nonzero gradient contribution from every point, including
  far-outside-the-margin outliers, so it can pull the decision boundary
  away from the accuracy-maximizing separator; hinge loss saturates to
  zero gradient past margin 1 and is comparatively more robust to
  well-separated outliers but can behave erratically on close ones. Under
  class imbalance, both surrogates (which weight every point equally in
  the empirical risk) can select a boundary that is cross-entropy/hinge
  optimal but clearly accuracy-suboptimal on the minority class, since
  accuracy itself does not penalize majority-class errors and minority-class
  errors asymmetrically the way the surrogate risk implicitly can.

## Questions

1. For this linear/Gaussian testbed, how large is the gap between the
   accuracy achieved by the surrogate-minimizing $(w,b)$ and the
   accuracy-maximizing $(w,b)$ found by direct search, as a function of
   class overlap, outlier fraction, and class imbalance ratio?
2. Does the ranking of cross-entropy vs hinge loss (which one gets closer
   to the true accuracy optimum) stay consistent as noise/outliers/
   imbalance are varied, or does the better surrogate change depending on
   the failure mode being stressed?
3. Classification-calibration theory guarantees surrogate-risk minimization
   implies 0/1-risk minimization only in the limit of unrestricted
   hypothesis classes and infinite data. Given this testbed intentionally
   uses a restricted (linear) hypothesis class and finite data, is the
   observed divergence better explained by the restricted-capacity gap, the
   finite-sample gap, or the surrogate-vs-true-objective gap, and can the
   experiment isolate which one dominates?

## Research Topics

- **Calibration/consistency theory**: read and summarize the excess-risk
  bound of Bartlett, Jordan & McAuliffe (2006) relating $\phi$-risk excess
  to 0/1-risk excess, and check empirically whether the bound's predicted
  ordering (which surrogate gap should be smaller) matches what is observed
  on the testbed
- **Surrogate sensitivity to data pathologies**: systematically vary
  outlier fraction, class overlap, and class imbalance, and map out the
  region of parameter space where surrogate minimization and true-objective
  maximization pick meaningfully different classifiers
- **Alternative/robust surrogates**: extend the comparison to the ramp loss
  (a bounded, non-convex surrogate designed to be more outlier-robust than
  hinge, used in robust optimization framings of the 0/1 loss) and to
  Fisher/squared-error-style surrogates, to see whether non-convex
  surrogates close the gap to the true objective at the cost of harder
  optimization

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: survey, pick the concrete problem, and formalize it
  - Confirm the five real-world surrogate-loss usages above with primary
    sources, and pick the linear/Gaussian binary-classification problem
    as the testbed (already formalized above)
  - Fix the exact data-generating process (Gaussian means/covariances per
    class, sample size, and the outlier/imbalance perturbation knobs used
    later)
  - This is the result: a written formalization (data-generating process,
    hypothesis class, true objective, two surrogate losses) precise enough
    to implement directly, with no open parameter choices left informal

- Milestone 2: implement the direct discrete-objective baseline in Python
  - Implement exact accuracy computation $R_{01}(w,b)$ on the fixed sample
  - Implement a direct search over $(w,b)$ that optimizes accuracy without
    using gradients: grid search over a bounded $(w,b)$ range, refined by
    random/local search (simulated annealing or coordinate-wise search),
    since exhaustive grid search is tractable at this dimensionality (3
    parameters)
  - This is the result: a reference classifier that (near-)maximizes true
    accuracy on the training sample, usable as the ground-truth baseline
    for all subsequent comparisons

- Milestone 3: implement gradient descent on 2+ surrogate losses
  - Implement $R_{\phi_{ce}}(w,b)$ and $R_{\phi_{hinge}}(w,b)$ and minimize
    each with (sub)gradient descent / Adam from several random
    initializations
  - Log both the surrogate loss curve and the true accuracy at every
    iteration, so the two trajectories can be compared directly
  - This is the result: two trained classifiers (cross-entropy-optimal and
    hinge-optimal), each with a full training curve of surrogate loss vs.
    true accuracy over time

- Milestone 4: compare true-objective performance and characterize
  divergence
  - Compare final accuracy of the discrete-search baseline vs. both
    surrogate-trained classifiers on clean data, then repeat under
    injected outliers and class imbalance (per the failure mode in Key
    Examples)
  - Produce accuracy-gap plots as a function of outlier fraction/imbalance
    ratio for each surrogate, and inspect the specific misclassified points
    that separate the surrogate-optimal from the accuracy-optimal boundary
  - This is the result: a quantitative characterization of when and by how
    much surrogate minimization diverges from true-objective optimization,
    and which of the two surrogates degrades faster under which pathology

## References
- Bartlett, P., Jordan, M., & McAuliffe, J., _Convexity, Classification,
  and Risk Bounds_. (2006)
- Cortes, C., & Vapnik, V., _Support-Vector Networks_. (1995)
- Zhang, T., _Statistical Behavior and Consistency of Classification
  Methods Based on Convex Risk Minimization_. (2004)
- Lapin, M., Hein, M., & Schiele, B., _Loss Functions for Top-k Error:
  Analysis and Insights_. (2016)
- Eban, E., Schain, M., Mackey, A., Gordon, A., Rifkin, R., & Elidan, G.,
  _Scalable Learning of Non-Decomposable Objectives_. (2017)
