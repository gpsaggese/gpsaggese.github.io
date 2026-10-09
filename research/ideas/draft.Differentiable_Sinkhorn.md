# Sinkhorn Relaxation for Differentiable Combinatorial Matching

## Status
- **Status:**: draft
- **Complete Specs:**: 20%

## Core Idea

- This idea relaxes the *variables* of a combinatorial problem, not the
  *loss/objective* (contrast with surrogate-loss approaches like
  cross-entropy-for-accuracy). A hard combinatorial structure, such as a
  permutation matrix or a matching, is replaced by a soft, continuous,
  probabilistic analogue (a doubly stochastic matrix), which is differentiable
  end to end and can sit inside a larger gradient-trained pipeline
- The hypothesis: for a small assignment/matching problem, projecting an
  arbitrary square matrix onto the Birkhoff polytope of doubly stochastic
  matrices via the Sinkhorn-Knopp normalization, combined with gradient
  descent on the entries and a temperature-annealing schedule, converges to a
  solution whose cost approaches the exact optimum found by the Hungarian
  algorithm, at additional computational cost from the iterative
  normalization
- This matters because many problems that look purely combinatorial (ranking,
  graph matching, clustering, tour construction) need to be embedded inside a
  differentiable model (e.g., a neural network trained end to end) where no
  exact discrete solver is available in the gradient path. Sinkhorn
  relaxation is the standard trick to keep gradients flowing through a
  "soft permutation" while still being able to anneal toward a hard, valid
  one at inference/deployment time
- It is non-obvious because the naive relaxation (allow any matrix, threshold
  at the end) does not respect the row/column-sum=1 constraints that make a
  matrix an actual permutation; Sinkhorn's alternating row/column
  normalization is the specific mechanism that keeps the relaxed object
  inside the feasible polytope of the discrete structure, rather than just
  inside a generic continuous box

## Formalization

### Discrete problem: linear assignment

- Given $n$ workers and $n$ tasks with a cost matrix $C \in \mathbb{R}^{n
  \times n}$, find a permutation $\sigma \in S_n$ (equivalently, a
  permutation matrix $P \in \{0,1\}^{n \times n}$ with $P_{i,\sigma(i)} = 1$)
  minimizing

```latex
\min_{P \in \mathcal{P}_n} \; \langle C, P \rangle = \sum_{i=1}^n C_{i, \sigma(i)}
```

- where $\mathcal{P}_n$ is the (discrete, finite) set of $n \times n$
  permutation matrices

### Continuous relaxation: the Birkhoff polytope

- Relax $\mathcal{P}_n$ to the Birkhoff polytope $\mathcal{B}_n$, the convex
  hull of permutation matrices, equal to the set of doubly stochastic
  matrices:

```latex
\mathcal{B}_n = \Big\{ P \in \mathbb{R}_{\geq 0}^{n \times n} \;:\;
  \sum_j P_{ij} = 1 \;\; \forall i, \quad \sum_i P_{ij} = 1 \;\; \forall j
  \Big\}
```

- By the Birkhoff-von Neumann theorem, $\mathcal{P}_n$ is exactly the set of
  vertices of $\mathcal{B}_n$: any doubly stochastic matrix is a convex
  combination of permutation matrices, so $\mathcal{B}_n$ is the "correct"
  convex relaxation, not just an arbitrary continuous superset

### Sinkhorn normalization operator

- Given any matrix with positive entries $X \in \mathbb{R}_{>0}^{n \times n}$
  (e.g., $X = \exp(-C/\tau)$ or the exponential of a learnable logit matrix
  $\theta$), define the Sinkhorn operator $S(X)$ as the limit of alternating
  row and column normalization:

```latex
S^0(X) = X, \qquad
S^{l}(X) = \mathcal{T}_c\big(\mathcal{T}_r(S^{l-1}(X))\big)
```

- where $\mathcal{T}_r$ divides every row by its row sum and $\mathcal{T}_c$
  divides every column by its column sum:

```latex
\mathcal{T}_r(X)_{ij} = \frac{X_{ij}}{\sum_k X_{ik}}, \qquad
\mathcal{T}_c(X)_{ij} = \frac{X_{ij}}{\sum_k X_{kj}}
```

- As $l \to \infty$, $S^l(X) \to S(X) \in \mathcal{B}_n$ (Sinkhorn's theorem:
  for any positive square matrix this iteration converges to a unique doubly
  stochastic matrix). In practice, $L$ = 10-50 iterations suffice for a
  reasonable approximation
- $S(\cdot)$ is differentiable (each row/column division is a smooth
  operation), so gradients flow from a downstream loss on $S(X)$ back to $X$
  or to $\theta$, unlike the discrete Hungarian solver

### Temperature and annealing toward hardness

- Parameterize $X_\theta = \exp(\theta / \tau)$ for a learnable logit matrix
  $\theta \in \mathbb{R}^{n \times n}$ and temperature $\tau > 0$
- As $\tau \to 0$, $S(X_\theta)$ concentrates mass on a single permutation
  (the Gumbel-Sinkhorn / soft-to-hard annealing limit): $\lim_{\tau \to 0}
  S(\exp(\theta/\tau))$ is (almost surely, for generic $\theta$) a
  permutation matrix, specifically $\arg\max_P \langle \theta, P \rangle$
- Training schedule: start at $\tau_0$ (soft, well-behaved gradients), decay
  $\tau_k = \tau_0 \cdot \gamma^k$ (or a fixed cosine/exponential schedule)
  over $K$ gradient steps, then round the final soft $P$ to a hard
  permutation matrix with the Hungarian algorithm applied to $-S(X_\theta)$
  (i.e. use the soft matrix's entries as a proxy cost for a final exact
  discrete projection)
- Optional: inject Gumbel noise $G_{ij} \sim \text{Gumbel}(0,1)$ into $\theta$
  before the Sinkhorn operator (Gumbel-Sinkhorn) to turn this into a
  reparameterized *sampler* over approximate permutations rather than a
  single deterministic point, useful when the downstream use case needs a
  distribution over matchings rather than one best matching

### Concrete worked example

- $n = 5$ workers, $n = 5$ tasks, cost matrix $C \in \mathbb{R}^{5 \times 5}$
  with entries drawn i.i.d. from $\text{Uniform}(0, 1)$ (fixed random seed for
  reproducibility)
- Objective: $\min_\theta \langle C, S(\exp(\theta/\tau)) \rangle$, optimized
  by gradient descent on $\theta$ with Adam, temperature annealed from
  $\tau_0 = 1.0$ down to $\tau_K = 0.05$ over $K = 500$ steps, $L = 20$
  Sinkhorn iterations per forward pass
- Final rounding: apply the Hungarian algorithm to $-S(X_\theta)$ at the last
  temperature to snap to a valid permutation, then report the true discrete
  cost $\langle C, P_{\text{final}} \rangle$ under the original $C$

## Key Examples

- **Five known uses of Sinkhorn-style relaxation in the literature**:
  - **Learning to rank / differentiable sorting**: replace the hard sort
    permutation with a soft one via Sinkhorn, so a ranking loss can be
    backpropagated through the sort operation (Adams & Zemel 2011; used in
    differentiable top-k and NeuralSort-style pipelines)
  - **Graph matching**: aligning nodes of two graphs (e.g., for image
    keypoint matching or molecule alignment) is a quadratic assignment
    problem; Sinkhorn layers are used inside neural graph-matching networks
    to produce soft correspondence matrices trainable end to end
  - **Linear assignment / permutation learning**: learning a latent
    permutation between two unordered sets (e.g., matching jigsaw puzzle
    pieces, or aligning unordered sensor readings to a canonical order) via
    the Gumbel-Sinkhorn reparameterization (Mena et al. 2018)
  - **Soft TSP tour construction**: relaxing the discrete tour (a cyclic
    permutation) to a doubly stochastic matrix to allow gradient-based or
    neural-network-guided construction heuristics for the traveling
    salesman problem, later rounded to a valid tour
  - **Differentiable clustering / soft assignment**: relaxing hard
    cluster-membership indicator matrices to soft assignment matrices
    normalized via Sinkhorn-like iterations, connecting to entropic optimal
    transport formulations of k-means (Cuturi 2013 Sinkhorn distances
    underlie this family)
- **Concrete worked example (this file's chosen problem)**: the $5 \times 5$
  random-cost linear assignment problem described in Formalization. Solved
  once exactly with `scipy.optimize.linear_sum_assignment` (Hungarian
  algorithm), and once via logits $\theta$, Sinkhorn normalization, Adam
  gradient descent, temperature annealing from $1.0 \to 0.05$, and a final
  Hungarian-rounding step; the two costs and wall-clock times are compared
  directly
- **Failure mode / edge case**: if $\tau$ is annealed too fast (e.g., $\tau_0
  = 0.05$ from the start), $S(X_\theta)$ saturates to near-binary values
  immediately, gradients vanish (rows/columns become nearly one-hot, so
  $\partial S / \partial \theta \approx 0$ almost everywhere), and gradient
  descent gets stuck near its random initialization; this reproduces, in
  miniature, the general "peaky softmax" failure mode shared with
  Gumbel-Softmax when temperature is mis-scheduled

## Questions

1. For this $5 \times 5$ instance, how does the optimality gap ($\langle C,
   P_{\text{Sinkhorn}} \rangle - \langle C, P_{\text{Hungarian}} \rangle$)
   scale as $n$ grows to 20, 50, 200? Is the relaxation's relative advantage
   (in a differentiable pipeline) worth an increasing absolute gap at larger
   $n$?
2. Since the Hungarian algorithm is exact, polynomial-time ($O(n^3)$), and
   has no hyperparameters, when is it ever preferable to use the Sinkhorn
   route for a *standalone* assignment problem rather than only when the
   assignment sits inside a larger differentiable model that needs
   end-to-end gradients through it?
3. Does the number of Sinkhorn iterations $L$ trade off against the number of
   outer gradient-descent steps $K$ for a fixed compute budget, i.e., is it
   better to run few Sinkhorn iterations per step with more outer steps, or
   many Sinkhorn iterations per step with fewer outer steps?

## Research Topics

- **Convergence rate of Sinkhorn iterations**: characterize how quickly
  $S^l(X)$ approaches $\mathcal{B}_n$ as a function of $n$ and the dynamic
  range of $X$'s entries, and how that interacts with the chosen $\tau$
- **Annealing schedule design**: compare fixed exponential decay,
  cosine decay, and adaptive schedules (e.g., anneal only once gradient norm
  stabilizes) for the effect on final optimality gap and training stability
- **Gumbel-Sinkhorn vs deterministic Sinkhorn**: investigate whether
  injecting Gumbel noise (turning the relaxation into a reparameterized
  sampler over approximate permutations) improves the quality of the final
  rounded solution compared to the deterministic version, at the cost of
  higher variance
- **Generalization beyond permutations**: survey what doubly-stochastic-like
  relaxations exist (or don't) for other combinatorial structures noted as a
  drawback (spanning trees, general subgraphs, programs), to map out where
  this template does and does not extend

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: survey known Sinkhorn-relaxation use cases and formalize the
  concrete problem
  - Confirm the five known-usage problems (ranking, graph matching,
    permutation learning, soft TSP, differentiable clustering) with one
    citation each
  - Write out the full formalization for the chosen $5 \times 5$ linear
    assignment instance: cost matrix, logit parameterization, Sinkhorn
    operator, annealing schedule
  - This is the result: a written formalization (this file) plus a fixed
    random-seed cost matrix ready to feed into both solvers

- Milestone 2: implement the exact discrete baseline
  - Implement the $5 \times 5$ (then parameterized to general $n$) linear
    assignment problem in Python using `scipy.optimize.linear_sum_assignment`
  - Record the optimal cost and wall-clock runtime
  - This is the result: a reference-correct, fast exact baseline solution and
    runtime to compare against

- Milestone 3: implement the Sinkhorn relaxation + gradient descent +
  annealing pipeline
  - Implement the Sinkhorn normalization operator (row/column normalization
    loop, $L$ iterations) in NumPy or PyTorch (PyTorch preferred, for
    autograd)
  - Implement the training loop: logits $\theta$, Adam optimizer,
    temperature annealing schedule, and the final Hungarian-rounding step
  - This is the result: a working differentiable pipeline that outputs a
    valid permutation matrix and its true discrete cost under $C$

- Milestone 4: compare quality, gap, and runtime; characterize when the
  extra cost is worth it
  - Run both solvers on the same instance across a range of $n$ (5, 10, 20,
    50), recording optimality gap, wall-clock time, and Sinkhorn iteration
    count needed for convergence
  - Run the mis-annealed (too-fast temperature decay) failure case to
    document the vanishing-gradient failure mode concretely
  - This is the result: a table/plot of gap vs $n$ and runtime vs $n$ for
    both methods, plus a concrete recommendation of when Sinkhorn relaxation
    is worth its overhead (i.e., only when embedded in a larger
    end-to-end-trained model, not for standalone assignment)

## References
- R. Sinkhorn, _A Relationship Between Arbitrary Positive Matrices and Doubly
  Stochastic Matrices_. (1964)
- R. P. Adams, R. S. Zemel, _Ranking via Sinkhorn Propagation_. (2011)
- G. Mena, D. Belanger, S. Linderman, J. Snoek, _Learning Latent Permutations
  with Gumbel-Sinkhorn Networks_. (2018)
- M. Cuturi, _Sinkhorn Distances: Lightspeed Computation of Optimal
  Transport_. (2013)
