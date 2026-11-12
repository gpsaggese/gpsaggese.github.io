# Gumbel-Softmax Relaxation for Differentiable Categorical Decisions

## Status
- **Status:**: draft
- **Complete Specs:**: 0%

## Core Idea

- This idea studies the *variable-relaxation* branch of differentiable discrete
  optimization: instead of smoothing the loss/objective (as surrogate losses or
  the straight-through estimator do), Gumbel-Softmax relaxes the categorical
  *variable itself*. A hard one-hot choice $z = \text{onehot}(\arg\max_k
  \ell_k)$ is replaced by a temperature-controlled soft simplex vector $\tilde
  z(\tau)$ that is a smooth, reparameterizable function of the logits and an
  auxiliary noise source. As $\tau \to 0$, $\tilde z(\tau)$ converges to the
  discrete one-hot vector, so the relaxation is a continuous bridge between a
  fully differentiable object and the exact discrete decision.
- The hypothesis to test: for a concrete categorical-decision problem small
  enough to have a known discrete optimum (found by enumeration), training with
  Gumbel-Softmax relaxation plus gradient descent and a temperature-annealing
  schedule converges to that optimum (or close to it), while exposing a
  characteristic temperature/bias tradeoff: high $\tau$ gives low-variance but
  strongly biased (mushy) gradients, low $\tau$ gives near-unbiased but
  high-variance (vanishing, saturated) gradients, and the annealing schedule
  itself becomes a nontrivial hyperparameter that determines whether training
  reaches the discrete optimum or gets stuck in a soft, blended solution that
  never resolves to a hard decision. The interesting part is not that
  Gumbel-Softmax works (it is standard practice) but *quantifying* this
  tradeoff on a problem small enough to compare directly against ground truth.

## Formalization

### The Gumbel-Max trick (exact, non-differentiable)

- Let $\ell = (\ell_1, \ldots, \ell_K) \in \mathbb{R}^K$ be logits defining a
  categorical distribution via $p_k = \text{softmax}(\ell)_k$.
- Draw i.i.d. Gumbel noise $g_k = -\log(-\log(u_k))$, $u_k \sim
  \text{Uniform}(0,1)$, for $k = 1, \ldots, K$.
- Identity: $z = \arg\max_k (\ell_k + g_k)$ is an exact sample from
  $\text{Categorical}(p)$.
- This moves all randomness into $g$, external to the differentiable path, but
  $\arg\max$ itself still has zero gradient almost everywhere.

### The Gumbel-Softmax relaxation (differentiable)

- Replace $\arg\max$ with a temperature-$\tau$ softmax over the perturbed
  logits:

```latex
\tilde z_k(\tau) = \frac{\exp\big((\ell_k + g_k)/\tau\big)}
                          {\sum_{k'=1}^{K} \exp\big((\ell_{k'} + g_{k'})/\tau\big)},
\qquad k = 1, \ldots, K
```

- $\tilde z(\tau) \in \Delta^{K-1}$ (the probability simplex) is differentiable
  in $\ell$ for any $\tau > 0$, so gradients flow through it via the
  reparameterization trick.
- Limits: $\lim_{\tau \to 0} \tilde z(\tau) \to \text{onehot}(\arg\max_k(\ell_k
  + g_k))$ (recovers the discrete sample); $\tau \to \infty$ gives $\tilde
  z(\tau) \to$ uniform over $K$ categories (maximally smoothed, uninformative
  gradient).
- Straight-through variant (mentioned only for context, not the focus of this
  file): use $z_{hard} = \text{onehot}(\arg\max_k \tilde z_k)$ in the forward
  pass, but backpropagate through $\tilde z(\tau)$ as if it were the forward
  value.

### Concrete example: toy Mixture-of-Experts top-1 routing

- Dataset: $N$ fixed points $\{(x_i, y_i)\}_{i=1}^N$, $x_i \in \mathbb{R}$,
  generated from a piecewise ground-truth function with $K$ segments (e.g., a
  piecewise-linear function with $K = 3$ pieces), so that a $K$-expert model
  can fit it exactly if each point is routed to the correct expert.
- $K$ linear experts $f_k(x) = w_k x + b_k$, either fixed to the true
  per-segment coefficients (isolating the routing decision) or trained jointly
  (more realistic, adds a second differentiable component).
- Router: a small parametric function $\ell_i = g_\theta(x_i) \in
  \mathbb{R}^K$ producing per-point logits.
- Discrete (hard) formulation: $z_i = \text{onehot}(\arg\max_k \ell_{i,k})$,
  prediction $\hat y_i = \sum_k z_{i,k} f_k(x_i)$, loss
  $\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N (\hat y_i - y_i)^2$.
- The discrete assignment space is $K^N$; with $N$ small (e.g., $N=8$) and $K$
  small (e.g., $K=3$), $K^N$ is enumerable by brute force, giving a verifiable
  ground-truth optimum to compare against.
- Gumbel-Softmax formulation: $\hat y_i(\tau) = \sum_k \tilde z_{i,k}(\tau)
  f_k(x_i)$, with $\tilde z_i(\tau)$ as defined above using $\ell_i =
  g_\theta(x_i)$; $\theta$ is trained by gradient descent on
  $\mathcal{L}(\theta, \tau) = \frac{1}{N}\sum_i (\hat y_i(\tau) - y_i)^2$
  while $\tau$ is annealed toward 0 over training.
- Evaluation at deployment uses the hard router ($\tau \to 0$ or explicit
  $\arg\max$), so the gap between soft-training loss and hard-eval loss at
  intermediate $\tau$ is exactly the train/test mismatch this file is meant to
  quantify.

## Key Examples

- **Five known applications of Softmax / Gumbel-Softmax** (real, well-known
  instances of relaxing categorical/argmax variables):
  - **Differentiable neural architecture search**: DARTS relaxes the choice
    of operation on each computation-graph edge with a plain softmax mixture;
    GDAS (Searching for a Robust Neural Architecture in Four GPU Hours) uses
    Gumbel-Softmax specifically to sample a single, hard operation per edge
    while remaining end-to-end differentiable
  - **Discrete latent variables in VAEs**: Categorical VAEs replace a
    discrete latent code's non-differentiable sampling step with
    Gumbel-Softmax so the encoder can be trained by backpropagation (the
    original motivating example of Jang, Gu & Poole 2017)
  - **Categorical reparameterization for discrete choice models**: economic
    and behavioral choice models (multinomial-logit-style decisions) use
    Gumbel-Softmax to make gradient-based estimation of choice-model
    parameters possible instead of relying on likelihood-only fitting
  - **Learned discrete attention / expert routing**: Mixture-of-Experts and
    hard-attention mechanisms (e.g., in image captioning) use Gumbel-Softmax
    as an alternative to REINFORCE for training a stochastic routing/attention
    decision through backpropagation
  - **Differentiable decision trees**: internal split nodes that route an
    input left/right (or to one of $K$ children) use Gumbel-Softmax to keep
    the routing decision trainable by gradient descent while approaching a
    hard, interpretable split at low temperature
- **Concrete worked example (this file's testbed)**: the toy Mixture-of-Experts
  top-1 routing problem formalized above, $N=8$ points, $K=3$ linear experts,
  enumerable $K^N = 6561$ discrete assignments as ground truth, router trained
  via Gumbel-Softmax with annealed $\tau$
- **Failure mode to probe**: fixing $\tau$ too low from the start (e.g.,
  $\tau = 0.05$) should reproduce the "vanishing gradient, behaves like the
  discrete function" pathology, where the router logits barely move from
  initialization and training stalls at a poor, effectively random routing

## Questions

1. Does annealing $\tau$ over training beat any single fixed $\tau$ on this
   toy MoE problem, and which schedule (linear, exponential, step) is
   Pareto-best on final-loss-vs-training-stability for this problem size?
2. How does the Gumbel-Softmax estimator's bias (from a nonzero $\tau$) trade
   off against REINFORCE's unbiased-but-high-variance estimator in practice:
   does Gumbel-Softmax reach a lower final loss faster despite the bias, and
   does that advantage persist as $K$ grows?
3. Does the soft-forward/hard-deployment mismatch produce a measurable
   performance cliff between training loss (at intermediate $\tau$) and
   hard-eval loss, and does a straight-through finishing phase close that
   gap? If closing the gap requires switching to hard routing near the end
   anyway, does the relaxation actually save complexity over REINFORCE with
   proper variance reduction?

## Research Topics

- **Temperature annealing schedules**: compare fixed-low, fixed-high, linear
  decay, and exponential decay schedules for $\tau$ on convergence speed,
  final loss relative to the enumerated optimum, and run-to-run variance
- **Gradient bias/variance characterization**: empirically measure the bias
  (distance from the true gradient, estimated via enumeration) and variance
  of the Gumbel-Softmax gradient estimator versus the REINFORCE estimator,
  as a function of $\tau$ and number of Monte Carlo samples
- **Straight-through hybrid as mismatch fix**: test whether switching to a
  straight-through estimator (hard forward, soft backward) in the final
  training phase closes the soft-train/hard-eval performance gap without
  reintroducing REINFORCE-level variance
- **Scaling with the number of categories $K$**: characterize whether
  Gumbel-Softmax degrades as $K$ grows (higher-dimensional simplex, harder
  temperature tuning, more diluted soft blends at a given $\tau$)

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: survey known Gumbel-Softmax applications and formalize the
  concrete testbed problem
  - Confirm and document the 5 known-usage problems with citations (NAS, VAE
    discrete latents, discrete choice models, MoE/hard attention routing,
    differentiable decision trees)
  - Finalize the toy MoE routing problem: fix $N$, $K$, the ground-truth
    piecewise function, and whether experts are fixed or jointly trained
  - This is the result: a written formalization (logits, Gumbel-Max identity,
    softmax relaxation, loss) matching the Formalization section, plus a
    generated, versioned toy dataset

- Milestone 2: implement the discrete baseline solver in Python
  - Implement brute-force enumeration over the $K^N$ assignment space to find
    the exact global-optimum routing and its loss
  - Implement a REINFORCE (score-function estimator) baseline that trains the
    router's categorical policy via policy gradient with a moving-average
    baseline for variance reduction
  - This is the result: a verified ground-truth optimal loss from
    enumeration, plus a working REINFORCE trainer with logged gradient
    variance per step

- Milestone 3: implement the Gumbel-Softmax relaxation and annealed training
  loop
  - Implement the reparameterized sampler $\tilde z(\tau)$, the router
    forward/backward pass, and at least two temperature-annealing schedules
  - Train to convergence under each schedule and under fixed-$\tau$ controls
  - This is the result: trained routers under each schedule, with loss curves
    and final hard-eval loss logged per configuration

- Milestone 4: compare quality, gradient variance, and train/test mismatch
  - Compare final hard-eval loss (all three methods) against the enumerated
    optimum; compare gradient variance/bias for Gumbel-Softmax vs REINFORCE
    across $\tau$; measure the soft-loss/hard-loss gap across schedules
  - This is the result: a quantified temperature-sensitivity curve and a
    clear statement of when Gumbel-Softmax matches, beats, or underperforms
    REINFORCE and the enumerated optimum on this problem size

## References
- Jang, E., Gu, S., Poole, B., _Categorical Reparameterization with
  Gumbel-Softmax_. (2017)
- Maddison, C. J., Mnih, A., Teh, Y. W., _The Concrete Distribution: A
  Continuous Relaxation of Discrete Random Variables_. (2017)
- Dong, X., Yang, Y., _Searching for a Robust Neural Architecture in Four GPU
  Hours_. (2019)
- Williams, R. J., _Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning_. (1992)
