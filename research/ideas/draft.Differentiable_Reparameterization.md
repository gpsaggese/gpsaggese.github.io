# Reparameterization Tricks for Structured Discrete Sampling

## Status
- **Status:**: draft
- **Complete Specs:**: 20%
- **Assignee:**: TBD

## Core Idea

- The reparameterization trick splits a stochastic discrete operation into
  two pieces: a source of randomness with no free parameters (e.g., i.i.d.
  Gumbel noise), and a deterministic, parameterized transform of that noise
  (e.g., add the noise to logits and take argmax/top-k/argsort). The
  parameters (logits, a cost matrix) only ever appear inside the
  deterministic transform, so once that transform is made differentiable
  (or relaxed), gradients with respect to the parameters flow through it
  while the actual randomness stays outside the differentiable path
  entirely. This is a *variable*-side relaxation, not a *loss*-side one: the
  sampling/selection mechanism itself is what gets relaxed, as opposed to
  approaches that keep hard discrete variables and instead smooth the
  objective/loss around them (e.g. surrogate losses, straight-through
  estimators)
- The hypothesis under test: the Gumbel-Max identity is not just a curiosity
  about categorical sampling, it is a template ("noise outside, argmax/
  top-k/sort inside") that extends to structured combinatorial objects
  (subsets via Gumbel-Top-k, permutations via Gumbel-Sinkhorn, spanning
  trees via Gumbel perturbations on Matrix-Tree weights) whenever the
  discrete object can be written as the arg-optimum of a linear score over a
  combinatorial polytope. The interesting empirical question is how far this
  template travels before the "argmax-like" step becomes too expensive or
  too poorly approximated to relax cleanly, and how the resulting gradient
  bias/variance compares to simpler categorical Gumbel-Softmax
- This is non-obvious because most practitioners meet the Gumbel trick only
  in its categorical form (Gumbel-Softmax / Concrete distribution) and
  implicitly assume it does not generalize; but the same additive-noise
  identity underlies exact algorithms for weighted reservoir sampling
  (Gumbel-Top-k) and approximate algorithms for doubly-stochastic relaxation
  of permutations (Gumbel-Sinkhorn), suggesting a genuinely reusable
  variable-side design pattern rather than a one-off trick

## Formalization

### General Gumbel-Max identity (the reparameterization template)

- For unnormalized log-scores (logits) $\theta = (\theta_1, \dots, \theta_K)$
  over $K$ categories, define a categorical distribution
  $p_k = \exp(\theta_k) / \sum_j \exp(\theta_j)$. Sample i.i.d. Gumbel noise
  $g_k = -\log(-\log(u_k))$ with $u_k \sim \text{Uniform}(0,1)$. Then

```latex
\arg\max_k (\theta_k + g_k) \sim \text{Categorical}(p)
```

- This is an exact reparameterization: the only source of randomness ($g$)
  has a distribution independent of $\theta$, and $\theta$ only enters
  through a deterministic map ($\theta + g \mapsto \arg\max$). The gradient
  $\nabla_\theta \, \mathbb{E}_g[f(\arg\max_k(\theta_k+g_k))]$ is generally
  intractable to differentiate through the hard $\arg\max$, which is exactly
  why relaxations (Gumbel-Softmax, out of scope here) or structured
  generalizations (below) are needed
- The template generalizes whenever the discrete object $y$ can be written
  as $y^\star(\theta) = \arg\max_{y \in \mathcal{Y}} \langle \theta, y
  \rangle$ over a combinatorial set $\mathcal{Y}$ (categories, subsets,
  permutations, spanning trees, ...): perturb $\theta \to \theta + g$ with
  an appropriately-chosen noise family, then solve the (still discrete)
  arg-optimum; under the right noise family this again samples exactly from
  the Gibbs distribution $p(y) \propto \exp(\langle \theta, y\rangle)$
  (Perturb-and-MAP / Gumbel-Max generalization, Maddison et al. 2014;
  Papandreou & Yuille 2011)

### Concrete structured case: Gumbel-Top-k for subset sampling

- Task: sample a size-$m$ subset $S \subset \{1, \dots, K\}$ without
  replacement, where item $k$ has weight $\propto \exp(\theta_k)$ (weighted
  sampling without replacement, a.k.a. Plackett-Luce top-$m$)
- Exact reparameterization (Gumbel-Top-k, Kool et al. 2019): draw
  $g_k \sim \text{Gumbel}(0,1)$ i.i.d., form perturbed scores $\phi_k =
  \theta_k + g_k$, and take

```latex
S = \text{top-}m\text{-arg}(\phi_1, \dots, \phi_K)
```

  the indices of the $m$ largest $\phi_k$. This samples $S$ exactly
  according to sequential Plackett-Luce sampling-without-replacement, with
  the entire randomness pushed into the one-shot noise draw $g$ (no
  sequential renormalization needed at sample time)
- Differentiable relaxation of the same object: replace hard top-$m$
  selection with the relaxed indicator
  $\tilde{r}_k = \sigma\big((\phi_k - \tau_m(\phi)) / t\big)$, where $\tau_m
  (\phi)$ is the $m$-th largest perturbed score (the threshold) and $t$ is a
  temperature; as $t \to 0$, $\tilde{r}_k \to \mathbb{1}[k \in S]$. This
  gives a soft membership vector $\tilde{r} \in [0,1]^K$ usable inside a
  downstream differentiable loss, while $S$ itself was drawn by the exact
  (unbiased) reparameterization above
- Parameters $\theta$ only enter through $\phi = \theta + g$, so
  $\nabla_\theta \tilde{r}$ is well-defined via the softened threshold rule,
  and the exact discrete sample $S$ can still be recovered for
  evaluation/deployment by hardening $\tilde r$ at $t \to 0$

### Related structured extension: Gumbel-Sinkhorn for permutations

- For learning a permutation $P \in \{0,1\}^{K \times K}$ (doubly stochastic
  with 0/1 entries) scored by a matrix $\Theta \in \mathbb{R}^{K \times K}$,
  perturb $\Theta \to \Theta + G$ with i.i.d. Gumbel noise $G$ entrywise,
  then relax $\arg\max_P \langle \Theta + G, P \rangle$ (an assignment
  problem, solvable exactly via Hungarian algorithm as the discrete
  baseline) by the Sinkhorn normalization operator
  $S_t(X) = \lim_{l \to \infty} (\text{row-norm} \circ \text{col-norm})^l
  (X/t)$ applied to $\Theta + G$, producing a doubly-stochastic matrix that
  converges to a hard permutation as $t \to 0$ (Mena et al. 2018; Adams &
  Zemel 2011). Included here as a second structured instance to show the
  same noise-outside-transform-inside template applies beyond subset
  selection

## Key Examples

- **Five known applications of reparameterized discrete sampling**:
  - Gumbel-Max sampling for categorical variational inference / discrete
    latent-variable VAEs (Maddison, Mnih & Teh 2016; Jang, Gu & Poole 2017)
  - Gumbel-Top-k for differentiable subset selection / stochastic
    beam search and sampling-without-replacement (Kool, van Hoof &
    Welling 2019)
  - Gumbel-Sinkhorn for learning latent permutations / matching problems,
    e.g. jigsaw puzzle solving and object-to-slot assignment (Mena et al.
    2018)
  - A* Sampling: Gumbel-perturbed search over structured/continuous spaces
    generalizing Gumbel-Max to exponential-family and infinite spaces
    (Maddison, Tarlow & Minka 2014)
  - Differentiable ranking/sorting via reparameterized noise perturbations
    of scores, used for learning-to-rank losses that need soft top-$k$
    membership (Grover et al. 2019, NeuralSort; closely related to the
    Gumbel-Top-k threshold relaxation above)
- **Worked structured example (this idea's focus)**: a recommender-style
  top-$m$ item selection problem. $K=50$ items with latent utility logits
  $\theta_k$ (unknown, to be learned), and a downstream reward that depends
  on which $m=5$ items are shown together (e.g., diversity-adjusted click
  reward). Ground truth: items are supposed to be selected with probability
  proportional to a hidden Plackett-Luce weight vector $\theta^\star$. Apply
  Gumbel-Top-k: sample $\phi = \theta + g$, take the top-5 indices as the
  exact discrete slate; relax with the softened threshold indicator
  $\tilde r_k$ to get gradients of the expected reward with respect to
  $\theta$, and run gradient descent on $\theta$ to recover $\theta^\star$
- **Edge case / failure mode**: pushing the template to spanning trees
  (Gumbel perturbations of edge weights, then Matrix-Tree-Theorem-based
  differentiable relaxation of the arg-max spanning tree) requires an
  $O(K^3)$ determinant/matrix-inverse computation per gradient step instead
  of an $O(K \log K)$ sort; this is the point where "extend the trick to
  arbitrary combinatorial objects" runs into a genuine computational wall,
  illustrating the drawback that reparameterization is clean only for
  structures with a cheap arg-optimum oracle (sorting for top-$k$, Hungarian
  algorithm/Sinkhorn for assignment) and gets progressively harder and more
  approximate for richer structures (trees, general matroids, TSP tours)

## Questions

1. For Gumbel-Top-k, how does the bias/variance of the softened-threshold
   gradient estimator compare, as a function of temperature $t$ and subset
   size $m$, to a REINFORCE estimator on the same subset-selection problem?
2. Does the "noise outside, arg-optimum inside" template have a precise
   characterization of which combinatorial polytopes admit an exact
   (not just approximate) reparameterization, beyond the known cases
   (categorical, top-$k$, assignment via Sinkhorn is already approximate)?
3. If the true underlying object is a spanning tree or a TSP tour, is it
   better to force a Gumbel-style reparameterization through an expensive
   relaxation, or to fall back to a score-function estimator that treats the
   arg-optimum oracle as a black box, only relaxing the loss instead of the
   variable?

## Research Topics

- **Exact vs. approximate reparameterization**: characterize which
  combinatorial structures (categorical, top-$k$ subsets, permutations,
  spanning trees, matchings) admit an exact Gumbel-style reparameterization
  versus only an approximate one (e.g. Sinkhorn is a smooth relaxation of
  the discrete assignment, not an exact identity like Gumbel-Max)
- **Gradient bias/variance profiling**: empirically measure gradient bias
  and variance of the Gumbel-Top-k relaxation against ground-truth exact
  gradients (computable on small $K$ by brute-force enumeration) as
  temperature and subset size vary
- **Cost of extending beyond categoricals**: quantify the computational
  overhead (time per gradient step, and gradient quality) of moving from
  plain categorical Gumbel-Max to Gumbel-Top-k to Gumbel-Sinkhorn to
  Gumbel-perturbed spanning trees, to map out where the "reparameterize
  everything" strategy stops paying off
- **Interaction with downstream relaxations**: how sensitive is end-to-end
  training to the choice of relaxation applied after the exact
  reparameterized sample (softened threshold vs. Sinkhorn iterations vs.
  straight-through), holding the reparameterization step fixed

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: survey known reparameterized structured-sampling problems and
  formalize the concrete example
  - Collect and briefly document the 5 known-usage problems (categorical
    VAEs, Gumbel-Top-k subset selection, Gumbel-Sinkhorn permutations, A*
    Sampling, differentiable ranking) with their original references
  - Pick the top-$m$ subset-selection problem as the concrete worked
    example, and write out the exact Gumbel-Top-k identity and its softened
    relaxation in full (as in Formalization)
  - This is the result: a written formalization document plus a short
    literature table mapping each of the 5 problems to its noise family and
    arg-optimum structure

- Milestone 2: implement the discrete baseline in Python
  - Implement exact weighted sampling-without-replacement (Plackett-Luce
    top-$m$) via sequential normalization, and a brute-force enumeration
    baseline for small $K$ (e.g. $K \le 12$) to compute exact expected
    reward and exact gradients by finite differences
  - Implement a REINFORCE estimator on the same subset-selection objective
    as a second discrete-style baseline
  - This is the result: a tested reference implementation producing exact
    (small-$K$) gradients and sampling distributions to benchmark against

- Milestone 3: implement the reparameterized differentiable path
  - Implement Gumbel-Top-k sampling (noise draw + top-$m$-arg) in PyTorch/
    JAX with the softened-threshold relaxation $\tilde r_k$ and a
    temperature schedule
  - Wire it into a gradient-descent loop that optimizes $\theta$ against the
    diversity-adjusted reward objective, recovering $\theta^\star$ from
    simulated interaction data
  - This is the result: a working differentiable training loop that
    converges $\theta$ toward $\theta^\star$, with hard slates recoverable
    at evaluation time by hardening $\tilde r$
  - Note: this milestone relaxes the *sampling/selection variable* itself,
    kept in a differentiable path via the reparameterized noise, in
    contrast to loss-side relaxation approaches covered elsewhere in the
    parent brainstorm

- Milestone 4: compare and characterize generalization beyond categoricals
  - Compare sample quality, gradient bias/variance (vs. the brute-force and
    REINFORCE baselines from Milestone 2), and convergence speed of the
    reparameterized path across a sweep of $K$, $m$, and temperature
  - Extend the comparison qualitatively to a second structured case
    (implement a small Gumbel-Sinkhorn permutation-learning toy problem) to
    measure how the bias/variance and computational-cost profile change
    when moving from top-$k$ subsets to permutations
  - This is the result: a quantitative comparison table (bias, variance,
    wall-clock, convergence) plus a written characterization of how far the
    Gumbel-Max template extends before it becomes impractical

## References
- E. J. Gumbel, _Statistical Theory of Extreme Values and Some Practical
  Applications_. (1954)
- C. Maddison, D. Tarlow, T. Minka, _A* Sampling_. (2014)
- C. Maddison, A. Mnih, Y. W. Teh, _The Concrete Distribution: A Continuous
  Relaxation of Discrete Random Variables_. (2016)
- E. Jang, S. Gu, B. Poole, _Categorical Reparameterization with
  Gumbel-Softmax_. (2017)
- W. Kool, H. van Hoof, M. Welling, _Stochastic Beams and Where to Find
  Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement_.
  (2019)
- G. Mena, D. Belanger, S. Linderman, J. Snoek, _Learning Latent
  Permutations with Gumbel-Sinkhorn Networks_. (2018)
- A. Grover, E. Wang, A. Zweig, S. Ermon, _Stochastic Optimization of
  Sorting Networks via Continuous Relaxations_. (2019)
