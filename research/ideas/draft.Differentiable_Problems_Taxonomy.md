# A Taxonomy for Differentiable Optimization of Discrete Problems

## Status
- **Status:**: draft
- **Complete Specs:**: 20%

## Core Idea

- The central hypothesis is that the many ad hoc tricks for making discrete
  problems gradient-friendly (Gumbel-Softmax, Sinkhorn, generic LP/box
  relaxation, reparameterization, the straight-through estimator, REINFORCE,
  surrogate losses) are not an unstructured grab-bag of hacks: they fall into
  a small number of families along two orthogonal axes. Axis 1 is *where the
  relaxation happens*: at the variable level (the decision itself is made
  continuous: Gumbel-Softmax, Sinkhorn, generic relaxation,
  reparameterization) or at the loss level (the decision stays discrete, but
  the objective evaluated on it is replaced by a smooth proxy: surrogate
  losses). The straight-through estimator straddles both: it performs a
  variable-level discrete op in the forward pass, exactly like a relaxation
  method would avoid doing, but patches the backward pass with a loss-level,
  crude gradient substitute rather than differentiating through a relaxed
  variable. Axis 2 is *whether the resulting gradient estimator is unbiased*:
  relaxation-based methods, STE, and surrogate losses are all biased (the
  quantity being differentiated is not the true discrete objective or not the
  true discrete variable), while REINFORCE/score-function estimators are
  unbiased but typically high-variance.
- This taxonomy is useful only if it is predictive: given a new discrete
  problem, it should narrow down which family of technique is worth trying
  first, based on answering three questions (where is the discreteness, is an
  unbiased estimator affordable, does the combinatorial structure have a
  known convex relaxation template). The non-obvious part is that this is an
  empirically testable claim, not just a organizing scheme: run the *same*
  small set of benchmark discrete problems through all 6 concrete techniques
  (relaxation, Gumbel-Softmax, STE, reparameterization, REINFORCE, Sinkhorn)
  plus surrogate losses, on a common scorecard (solution quality, gap to the
  discrete optimum, gradient variance, wall-clock, hyperparameter
  sensitivity), and check whether the taxonomy's cell assignment actually
  predicts which technique wins on which problem type. If it does not, the
  taxonomy (or the variable-vs-loss split specifically) is a false
  dichotomy and should be revised.
- This idea is the synthesis layer over 6 sibling research-idea files, each
  of which works out one concrete technique in isolation (5 known problems,
  1 worked example, discrete-vs-differentiable comparison). None of those
  files individually answers whether the taxonomy is real; only comparing
  across them, on shared problems and a shared scorecard, can.

## Formalization

- Setup: a discrete decision $x \in \mathcal{X}$ (categorical label,
  permutation, binary mask, subset, ...) parameterized by $\theta$, either
  deterministically as $x = g(\theta)$ (e.g. $\arg\max$, thresholding) or
  through a distribution $x \sim p_\theta(x)$. The objective is
  $L(\theta) = f(x(\theta))$ or $L(\theta) = \mathbb{E}_{x \sim p_\theta}[f(x)]$,
  and $f$ itself may be non-smooth (0/1 loss, accuracy, combinatorial cost).
  Because $\mathcal{X}$ is discrete (or $f$ is piecewise-constant), $\nabla_\theta L$
  is zero almost everywhere or undefined, so a *gradient estimator*
  $\hat g(\theta)$ is used as a stand-in for $\nabla_\theta L(\theta)$ in
  gradient descent.

- Bias-variance decomposition of a gradient estimator: for any estimator
  $\hat g$ of the true gradient $\nabla_\theta L(\theta)$,

```latex
\mathbb{E}\big[\lVert \hat g - \nabla_\theta L(\theta) \rVert^2\big]
  = \underbrace{\lVert \mathbb{E}[\hat g] - \nabla_\theta L(\theta) \rVert^2}_{\text{Bias}^2}
  + \underbrace{\mathbb{E}\big[\lVert \hat g - \mathbb{E}[\hat g] \rVert^2\big]}_{\text{Variance}}
```

  This is what "biased" vs "unbiased but high-variance" formally means:
  - **REINFORCE / score function**: $\hat g = f(x)\,\nabla_\theta \log
    p_\theta(x)$, with $x \sim p_\theta$. Since $\mathbb{E}_{x \sim
    p_\theta}[\nabla_\theta \log p_\theta(x)] = 0$ identically, $\mathbb{E}[\hat
    g] = \nabla_\theta L(\theta)$ exactly (bias term is 0), but the variance
    term is unbounded in general and grows with the magnitude/variance of
    $f(x)$, which is why baselines and control variates matter.
  - **Relaxation-based (Gumbel-Softmax, Sinkhorn, generic LP/box relaxation,
    reparameterization)**: replace $x$ with a continuous surrogate $\tilde
    x(\theta, \epsilon)$ (e.g. a softmax-weighted mixture, a doubly
    stochastic matrix) and compute $\hat g = \nabla_\theta f(\tilde
    x(\theta,\epsilon))$. Because $\tilde x \neq x$ in general (equality only
    holds in the zero-temperature / infinite-iteration limit),
    $\mathbb{E}[\hat g] \neq \nabla_\theta L(\theta)$: the bias term is
    nonzero but, since $\tilde x$ is a deterministic differentiable function
    of $\theta$ given $\epsilon$, the variance term is typically small
    (reparameterized samples correlate smoothly with $\theta$).
  - **Straight-Through Estimator**: forward pass uses the true discrete $x =
    g(\theta)$; backward pass substitutes $\nabla_\theta g(\theta) \approx
    I$ (or another hand-picked proxy Jacobian) in place of the true
    (zero-a.e.) derivative. $\hat g = \nabla_x f(x) \cdot I$. Bias is
    structural and unbounded in the worst case (no guarantee $\hat g$
    correlates with $\nabla_\theta L$ at all), but variance is 0 (the
    estimator is deterministic given $\theta$).
  - **Surrogate losses**: replace $f$ itself with a smooth $\tilde f$ (e.g.
    cross-entropy instead of 0/1 loss) and compute $\hat g = \nabla_\theta
    \tilde f(x(\theta))$ *exactly* (no stochastic estimation at all). Here
    "bias" is not sampling bias but objective mismatch:
    $\nabla_\theta \tilde f \neq \nabla_\theta f$ pointwise because $\tilde f
    \neq f$, even away from any discreteness. Variance (in the stochastic
    sense) is 0 for a fixed batch; the only source of estimator "error" is
    that the wrong function is being minimized.

- Two-axis taxonomy table:

  | | Variable-level relaxation | Loss-level relaxation |
  |---|---|---|
  | **Unbiased** | REINFORCE (applies to variable-level stochastic decisions; no known unbiased loss-level analogue) | (none in this survey) |
  | **Biased, low/no variance** | Gumbel-Softmax, Sinkhorn, generic relaxation, reparameterization | Surrogate losses |
  | **Straddles both** | Straight-Through Estimator (variable-level op, loss-level-style gradient patch) | |

- Decision tree for picking a family, given a new discrete problem:
  1. Is the discreteness in the *objective evaluated on an otherwise
     continuous/near-continuous decision* (e.g. accuracy, 0/1 loss, a
     step-function reward), or in the *decision variable itself* (a
     category, permutation, subset, structured object)?
     - Loss-level -> use a **surrogate loss** (cross-entropy, hinge,
       smoothed AUC, ...); use **STE** instead if the hard decision must be
       kept in the forward pass (e.g. because it feeds a discrete downstream
       system) but a usable backward signal is still needed.
     - Variable-level -> go to 2.
  2. Does $\mathcal{X}$ have a known convex-hull / relaxation template?
     - Categorical / simplex ($\mathcal{X} = \{1, \dots, K\}$) -> **Softmax
       / Gumbel-Softmax**; if exact sampling (not just a relaxed forward
       pass) is required, start from the **Gumbel-Max reparameterization**
       and relax only the $\arg\max$ inside it.
     - Permutations / doubly-stochastic matrices -> **Sinkhorn**.
     - Generic linear/box constraints (0/1 vectors, LP-representable sets)
       -> **generic continuous relaxation** + rounding.
     - No known template (trees, programs, graphs, arbitrary combinatorial
       objects) -> go to 3.
  3. Is an unbiased estimator affordable (enough samples per step, budget for
     variance reduction machinery: baselines, control variates,
     Rao-Blackwellization)?
     - Yes -> **REINFORCE / score-function**, with variance reduction.
     - No (need a cheap, simple, "good enough" gradient now) -> **STE** as a
       biased but zero-variance fallback, accepting the lack of formal
       descent guarantees.

## Key Examples

- **draft.Differentiable_Relaxation.md**: variable-level, biased/low-variance
  cell, generic LP/box relaxation template; worked example is expected to be
  a combinatorial 0/1 decision problem (e.g. a knapsack-style or set-cover-
  style integer program) relaxed to $[0,1]$ box constraints, solved
  continuously, then rounded, compared against the exact discrete optimum.
- **draft.Differentiable_Gumbel_Softmax.md**: variable-level, biased/low-
  variance cell, categorical/simplex template; worked example is expected to
  be a discrete-choice or discrete-latent-variable problem (e.g. a
  categorical VAE latent or a discrete architecture/gate selection),
  compared against hard-argmax training.
- **draft.Differentiable_Straight_Through_Estimator.md**: the straddling
  cell (variable-level discrete op, loss-level-style gradient patch);
  worked example is expected to be a quantization/binarization problem (e.g.
  training a binary or low-bit-width neural network), comparing STE-trained
  discrete weights/activations against a continuous-relaxation baseline.
- **draft.Differentiable_Reparameterization.md**: variable-level cell,
  categorical/simplex template via the Gumbel-Max trick, exact until further
  relaxed (at which point it converges with the Gumbel-Softmax cell); worked
  example is expected to be sampling-based, e.g. a discrete latent-variable
  generative model trained by pushing randomness outside the differentiable
  path.
- **draft.Differentiable_REINFORCE.md**: variable-level (or
  objective-through-stochastic-policy), unbiased/high-variance cell, no
  relaxation template required, so applicable to arbitrary discrete
  structure; worked example is expected to be a discrete-action or discrete-
  attention selection problem trained by policy gradient with a baseline.
- **draft.Differentiable_Sinkhorn.md**: variable-level, biased/low-variance
  cell, permutation/doubly-stochastic (Birkhoff polytope) template; worked
  example is expected to be a matching/assignment or differentiable-sorting
  problem, comparing the Sinkhorn-relaxed solution against the exact
  discrete matching.
- Note for completeness: the loss-level, biased/zero-variance cell
  (**surrogate losses**, cross-entropy vs 0/1 loss and similar) is covered
  by the separate sibling file draft.Differentiable_Surrogate_Losses.md,
  not by this synthesis file; it is the other half of the variable-vs-loss
  split this taxonomy is built around.

## Questions

1. Does the taxonomy's cell assignment correctly predict which technique
   wins empirically, per problem type, once all 6 techniques plus surrogate
   losses are run on the same benchmark problems?
2. Is there a principled way to choose the variance-reduction budget
   (REINFORCE + control variates) vs bias tolerance (relaxation/STE)
   tradeoff ahead of time, from problem properties alone, rather than by
   trial and error?
3. Do hybrid techniques (e.g. STE-refined Gumbel-Softmax, REINFORCE with a
   relaxation-based baseline/control variate, Sinkhorn followed by a
   surrogate loss on the soft assignment) dominate any single-family
   technique on the shared benchmark, and if so, does the taxonomy need a
   third "hybrid" category rather than treating hybrids as exceptions?
4. Is the variable-vs-loss split actually the right top-level axis, or does
   it collapse in practice because most interesting problems have
   discreteness in *both* the variables and the objective (e.g. training a
   classifier's discrete decision thresholds against accuracy), forcing a
   combination of techniques rather than a single cell lookup?

## Research Topics

- **Cross-technique benchmark suite design**: choosing a small set (roughly
  4-6) of discrete problems that are solvable by more than one technique
  family, so results are genuinely comparable rather than each sibling file
  reporting on a disjoint problem.
- **Common metric set**: defining gap-to-discrete-optimum, gradient-
  estimator variance (measured empirically over repeated draws/seeds),
  wall-clock per optimization step, and hyperparameter sensitivity
  (temperature, number of Sinkhorn iterations, number of REINFORCE samples,
  choice of surrogate) in a way that is measurable identically across all 6
  families.
- **Variable-vs-loss axis validity**: testing whether the variable-vs-loss
  split is a real structural distinction or a false dichotomy, by
  identifying problems where discreteness appears in both the variable and
  the objective and checking whether the taxonomy still gives a clean
  single-cell recommendation.
- **Hybrid-technique landscape**: cataloguing and benchmarking combinations
  across cells (STE-refined Gumbel-Softmax, relaxation-based baselines for
  REINFORCE, Sinkhorn-then-surrogate-loss pipelines) to see whether hybrids
  form a genuinely distinct, dominant region of the scorecard.

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1
  - Define the common benchmark problem set (roughly 4-6 discrete problems
    spanning categorical choice, permutation/matching, box-constrained
    combinatorial selection, and a loss-level accuracy-vs-surrogate case)
    and the shared scorecard metrics (gap to discrete optimum, gradient
    variance, wall-clock, hyperparameter sensitivity)
  - This is the result: a written benchmark-and-scorecard spec that every
    sibling research-idea file's worked example can be mapped onto or
    checked against

- Milestone 2
  - Once each of the 6 sibling research ideas has a working discrete-vs-
    differentiable implementation, aggregate their reported results
    (solution quality, gap to optimum, runtime, key hyperparameters) into a
    single table keyed by (technique, problem type, taxonomy cell)
  - This is the result: a consolidated results table spanning all 6
    techniques, ready for cross-technique comparison

- Milestone 3
  - Build a comparison table/plot across all 6 techniques (plus surrogate
    losses) on the shared benchmark problems from Milestone 1, highlighting
    which technique wins on which problem type and by how much
  - This is the result: a figure/table set showing empirical rankings per
    problem type, directly comparable to the taxonomy's predicted cell
    assignment from the Formalization section

- Milestone 4
  - Compare the empirical rankings from Milestone 3 against the taxonomy's
    predictions, quantify how often the taxonomy predicts the winning
    technique, and write up where it fails (e.g. hybrid techniques
    outperforming any single cell, or problems with discreteness in both
    variable and loss)
  - This is the result: a write-up of the taxonomy's predictive accuracy
    with proposed refinements (e.g. adding a hybrid category, splitting or
    merging axes) for a second iteration of the taxonomy

## References
- Mohamed, S., Rosca, M., Figurnov, M., and Mnih, A., _Monte Carlo Gradient
  Estimation in Machine Learning_. (2020)
- Williams, R. J., _Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning_. (1992)
- Jang, E., Gu, S., and Poole, B., _Categorical Reparameterization with
  Gumbel-Softmax_. (2017)
- Maddison, C. J., Mnih, A., and Teh, Y. W., _The Concrete Distribution: A
  Continuous Relaxation of Discrete Random Variables_. (2017)
- Mena, G., Belanger, D., Linderman, S., and Snoek, J., _Learning Latent
  Permutations with Gumbel-Sinkhorn Networks_. (2018)
