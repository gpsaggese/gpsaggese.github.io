# REINFORCE Gradient Estimation for Optimizing Through Discrete Choices

## Status
- **Status:**: draft
- **Complete Specs:**: 20%
- **Assignee:**: TBD

## Core Idea

- The score function / REINFORCE estimator sidesteps the "make the variable
  differentiable" problem entirely: instead of relaxing a discrete decision
  variable $x$ into a continuous surrogate (as Gumbel-Softmax, STE, or
  Sinkhorn relaxations do), it keeps $x$ exactly discrete and differentiates
  the *distribution* over $x$ instead. The log-derivative trick
  $\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] = \mathbb{E}_{x \sim
  p_\theta}[f(x) \nabla_\theta \log p_\theta(x)]$ turns an expectation over a
  discrete (and possibly non-differentiable, black-box) objective $f$ into an
  expectation of a differentiable quantity, which can be estimated by Monte
  Carlo sampling and optimized by ordinary gradient ascent on the policy
  parameters $\theta$. This is a fundamentally different move from every
  other approach in the parent brainstorm: there is no temperature to anneal,
  no straight-through bias, no rounding step at the end, because the discrete
  variable was never relaxed in the first place.
- The hypothesis to test: for a concrete discrete-decision problem where the
  reward function is non-differentiable (e.g., it contains a hard
  feasibility indicator), REINFORCE gives an unbiased gradient estimate of
  $J(\theta) = \mathbb{E}_{x \sim p_\theta}[f(x)]$, but that estimate is high
  variance enough to make naive training slow or unstable; and that a simple
  variance-reduction baseline meaningfully improves convergence speed and
  gradient signal-to-noise without introducing bias. This is worth
  demonstrating quantitatively (variance ratio, convergence speed, gap to
  the true optimum) rather than assumed, since the brainstorm's drawback list
  claims high variance but does not measure it.

## Formalization

### The log-derivative trick

- Goal: maximize $J(\theta) = \mathbb{E}_{x \sim p_\theta}[f(x)]$ over
  parameters $\theta$ of a distribution $p_\theta$ on a discrete space
  $\mathcal{X}$, where $f: \mathcal{X} \to \mathbb{R}$ may be non-smooth,
  non-differentiable, or a black box (only queryable, not analytically known)
- Direct differentiation is blocked because $x$ is discrete (no gradient
  w.r.t. $x$) and $f$ need not be differentiable even if $x$ were continuous
- Rewrite the expectation as a sum and differentiate through $p_\theta$
  instead of through $f$:

```latex
\nabla_\theta J(\theta) = \nabla_\theta \sum_{x} p_\theta(x) f(x)
  = \sum_{x} f(x) \nabla_\theta p_\theta(x)
  = \sum_{x} p_\theta(x) f(x) \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}
  = \mathbb{E}_{x \sim p_\theta}\big[f(x) \nabla_\theta \log p_\theta(x)\big]
```

- Monte Carlo estimator from $M$ samples $x^{(1)}, \dots, x^{(M)} \sim
  p_\theta$:
  $\widehat{\nabla_\theta J} = \frac{1}{M} \sum_{m=1}^M f(x^{(m)})
  \nabla_\theta \log p_\theta(x^{(m)})$
- This estimator is unbiased regardless of whether $f$ is differentiable,
  continuous, or even known in closed form: it only needs point evaluations
  $f(x^{(m)})$ and the score function $\nabla_\theta \log p_\theta(x^{(m)})$
  of the sampling distribution, which is chosen and controlled by the
  practitioner

### Baseline / control-variate variance reduction

- For any $b$ that does not depend on the sampled $x$ (a constant, a running
  average, or a function of $\theta$ alone):
  $\mathbb{E}_{x \sim p_\theta}[b \, \nabla_\theta \log p_\theta(x)] = b
  \nabla_\theta \sum_x p_\theta(x) = b \nabla_\theta 1 = 0$
- So subtracting $b$ leaves the estimator unbiased while changing its
  variance:
  $\nabla_\theta J = \mathbb{E}_{x \sim p_\theta}\big[(f(x) - b)
  \nabla_\theta \log p_\theta(x)\big]$
- The variance-minimizing baseline (per scalar parameter, approximately) is
  $b^\star = \dfrac{\mathbb{E}[f(x) \, \|\nabla_\theta \log
  p_\theta(x)\|^2]}{\mathbb{E}[\|\nabla_\theta \log p_\theta(x)\|^2]}$, a
  score-weighted average reward; in practice cheaper baselines work well:
  a running/exponential-moving-average of observed $f(x)$, a learned value
  estimate $V_\phi$, or a leave-one-out baseline computed from the other
  samples in the same minibatch (a simple form of Rao-Blackwellization)

### Concrete problem: black-box 0/1 knapsack via a Bernoulli policy

- $d$ items, weight $w_i > 0$ and value $v_i > 0$ for $i = 1, \dots, d$,
  capacity $C$
- Decision variable: $x \in \{0,1\}^d$, $x_i = 1$ means item $i$ is selected
- Reward (non-differentiable due to the hard feasibility indicator):
  $f(x) = \Big(\sum_{i=1}^d v_i x_i\Big) \cdot \mathbb{1}\Big[\sum_{i=1}^d
  w_i x_i \le C\Big]$
- Policy: independent Bernoulli per coordinate, logits $\theta \in
  \mathbb{R}^d$, $\sigma(\theta_i) = 1/(1 + e^{-\theta_i})$:
  $p_\theta(x) = \prod_{i=1}^d \sigma(\theta_i)^{x_i} (1 -
  \sigma(\theta_i))^{1-x_i}$
- Log-probability and its gradient (the score function) factorize
  coordinate-wise:
  $\log p_\theta(x) = \sum_{i=1}^d \big[x_i \log \sigma(\theta_i) + (1-x_i)
  \log(1 - \sigma(\theta_i))\big]$,
  $\dfrac{\partial \log p_\theta(x)}{\partial \theta_i} = x_i -
  \sigma(\theta_i)$
- REINFORCE gradient estimator with baseline $b$, from a batch of $M$ sampled
  subsets:
  $\widehat{\nabla_{\theta_i} J} = \frac{1}{M} \sum_{m=1}^M \big(f(x^{(m)}) -
  b\big) \big(x_i^{(m)} - \sigma(\theta_i)\big)$
- Update rule (gradient ascent): $\theta \leftarrow \theta + \eta
  \widehat{\nabla_\theta J}$
- As training converges, $\sigma(\theta_i) \to 1$ for items in the optimal
  subset $x^\star$ and $\sigma(\theta_i) \to 0$ otherwise, so $p_\theta$
  collapses onto $x^\star = \arg\max_{x \in \{0,1\}^d} f(x)$
- Note on framing: $x$ itself is never relaxed to $[0,1]^d$ at any point;
  $\sigma(\theta_i)$ is a Bernoulli *probability*, not a soft/relaxed
  surrogate for $x_i$ being evaluated inside $f$. Every call to $f$ uses a
  genuinely hard-sampled $x^{(m)} \in \{0,1\}^d$, which is exactly what
  distinguishes this approach from Gumbel-Softmax or Sinkhorn relaxation

## Key Examples

- **Known problems solved with REINFORCE**:
  - **Policy-gradient reinforcement learning**: discrete action spaces in
    control tasks, where the environment's reward is not a differentiable
    function of the action (Williams 1992; Sutton et al. 2000)
  - **Discrete latent-variable models**: variational inference with
    categorical or Bernoulli latent variables, where the ELBO's expectation
    over the latent is estimated with a score-function estimator and a
    learned baseline (Mnih & Gregor 2014, "NVIL")
  - **Neural architecture search**: a controller RNN samples discrete
    architecture descriptions (layer types, connections) and is trained with
    REINFORCE using validation accuracy as the (black-box, non-differentiable
    w.r.t. the architecture) reward (Zoph & Le 2017)
  - **Hard-attention models**: a glimpse location is sampled from a
    categorical/Gaussian policy over discrete image regions and trained with
    REINFORCE since the "look here" decision is not differentiable
    (Mnih et al. 2014, "Recurrent Models of Visual Attention")
  - **Black-box combinatorial optimization**: learning a sampling policy over
    combinatorial structures (tours, subsets, assignments) where the
    objective (tour length, knapsack value) is evaluated by a discrete
    routine rather than a differentiable formula (Bello et al. 2017, "Neural
    Combinatorial Optimization with RL")
- **Worked example: black-box 0/1 knapsack with a Bernoulli policy**: $d=20$
  items with random weights/values, capacity $C$ set to roughly 40% of total
  weight; the reward $f(x)$ is evaluated as a hard pass/fail-then-sum rule
  (no gradient through the indicator), and a Bernoulli policy $p_\theta$ is
  trained with REINFORCE, with and without a moving-average baseline, to
  concentrate its mass on the value-maximizing feasible subset
- **Edge case / failure mode: sparse feasibility**: if $C$ is set very small
  relative to $\sum_i w_i$, most randomly sampled subsets are infeasible and
  get $f(x) = 0$; the REINFORCE gradient then carries almost no signal
  (nearly every sample contributes the same "zero-reward" term), so variance
  swamps the mean and the policy barely moves until either the baseline is
  tuned aggressively or the batch size $M$ is increased substantially, a
  concrete illustration of the "sample-inefficient" drawback

## Questions

1. How does the variance-reduction benefit of a baseline scale with the
   dimension $d$ of the discrete decision, and does a constant/moving-average
   baseline remain adequate as $d$ grows, or does it need to become
   input-dependent (a learned value function per state)?
2. For the knapsack instance, how many REINFORCE samples are needed for the
   policy mode to match the brute-force optimum $x^\star$ with high
   probability, and how does that sample budget compare to just running
   exhaustive search directly while $d$ is still small enough for both to be
   feasible?
3. Provocative implication: since REINFORCE never relaxes $x$, could it serve
   as a drop-in, relaxation-free replacement anywhere a Gumbel-Softmax or STE
   is currently used, and if so, what specifically makes practitioners prefer
   the biased-but-lower-variance relaxation methods in practice?

## Research Topics

- **Baseline design comparison**: constant/moving-average baseline vs a
  learned value-function baseline $V_\phi$ vs a leave-one-out (per-batch)
  baseline, measured by resulting gradient variance and convergence speed on
  the same knapsack instance
- **REINFORCE vs relaxation on the same objective**: implement a
  Gumbel-Softmax (or continuous relaxation) version of the same knapsack
  reward and compare convergence speed, final solution quality, and
  hyperparameter sensitivity against REINFORCE with baseline
- **Reward sparsity and variance**: sweep the capacity $C$ (fraction of
  feasible subsets) and quantify how gradient variance and required sample
  budget degrade as feasible subsets become rare, to map out where REINFORCE
  breaks down in practice

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: survey known REINFORCE use cases and formalize the concrete
  problem
  - Confirm the 5 known-usage problems with primary references, and finalize
    the knapsack instance parameters ($d$, weight/value distributions,
    capacity $C$, targeted feasibility fraction)
  - Write out the score-function gradient and baseline formula specialized to
    the Bernoulli policy (as in Formalization)
  - This is the result: a fixed, reproducible problem instance and closed-form
    gradient expressions ready to implement

- Milestone 2: implement the discrete/exact baseline in Python
  - Implement brute-force enumeration of all $2^d$ subsets to compute the
    true optimum $x^\star$ and $f(x^\star)$ for small $d$ (e.g., $d \le 22$)
  - Implement a greedy value/weight-ratio heuristic as a scalable baseline for
    larger $d$ where brute force is infeasible
  - This is the result: a verified ground-truth optimum and a fast heuristic
    reference value to compare REINFORCE against

- Milestone 3: implement the REINFORCE estimator and gradient ascent
  - Implement the Bernoulli policy, sampling, and the score-function gradient
    estimator, run gradient ascent without a baseline
  - Add the moving-average baseline variant and re-run under identical
    seeds/sample budgets
  - This is the result: two trained policies (with/without baseline), their
    learning curves, and their converged modal subsets

- Milestone 4: compare quality, variance, and convergence
  - Measure value gap to the brute-force optimum for both REINFORCE variants
    and the greedy heuristic
  - Estimate per-update gradient variance empirically (repeated gradient
    estimates at fixed $\theta$) with vs without the baseline, and report the
    variance reduction ratio
  - Measure convergence speed (updates/samples to reach a fixed fraction of
    optimal value) with vs without the baseline
  - This is the result: a quantitative table/plot set showing the baseline's
    effect on variance and convergence, and how close REINFORCE gets to the
    exact/heuristic discrete solutions

## References
- Williams, R. J., _Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning_. (1992)
- Sutton, R. S., McAllester, D., Singh, S., Mansour, Y., _Policy Gradient
  Methods for Reinforcement Learning with Function Approximation_. (2000)
- Mnih, A., Gregor, K., _Neural Variational Inference and Learning in Belief
  Networks_. (2014)
- Mnih, V., Heess, N., Graves, A., Kavukcuoglu, K., _Recurrent Models of
  Visual Attention_. (2014)
