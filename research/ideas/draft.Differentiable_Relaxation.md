# Differentiable Relaxation of Discrete Variables for Combinatorial Optimization

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- This idea studies the oldest member of the differentiable-discrete-optimization
  family: replace a discrete decision variable (0/1, categorical, permutation
  entry) with a continuous variable in a box $[0,1]$ or on a simplex, optimize
  the relaxed continuous problem (here via gradient descent rather than a
  generic LP/QP solver), and then round the continuous solution back to a
  discrete one. This is a relaxation of the *variables* of the problem, not of
  the *loss/objective* (surrogate losses, cross-entropy vs. 0/1 loss are a
  separate, sibling family of techniques, covered elsewhere)
- The hypothesis under test: for a chosen concrete combinatorial problem
  (0/1 knapsack), projected-gradient-descent optimization of the box
  relaxation, followed by a rounding/repair step, reaches a solution close to
  the true discrete optimum on typical random instances, but the achieved gap
  is not uniform: it depends on structural properties of the instance
  (how spread out the value/weight ratios are, how tight the capacity
  constraint is) and can be characterized rather than treated as
  unpredictable noise
- This matters because relaxation is the default first move whenever a
  discrete problem needs to sit inside a differentiable pipeline (e.g., a
  neural network with an internal combinatorial subproblem), but practitioners
  rarely quantify when the "optimize continuous, then round" recipe is safe vs.
  when it silently produces a poor or infeasible answer. Pairing an exact
  discrete solver with the continuous-relaxation-plus-gradient-descent
  pipeline on the same instances gives a direct, reproducible measurement of
  that gap

## Formalization

### General relaxation setup

- Discrete problem: a feasible set $S \subseteq \{0,1\}^n$ (or more generally
  a subset of a discrete lattice) and an objective $f: S \to \mathbb{R}$,
  solved as
  ```latex
  x^\star = \arg\max_{x \in S} f(x)
  ```
- Relaxation: replace $S$ with a continuous, convex superset. Two standard
  choices, which coincide when the linear constraints happen to be totally
  unimodular:
  - **Box relaxation**: $\hat{S} = [0,1]^n \cap \{Ax \le b\}$, i.e. drop the
    integrality constraint but keep the linear constraints
  - **Convex-hull relaxation**: $\hat{S} = \mathrm{conv}(S)$, the tightest
    possible convex relaxation
  - In general $\mathrm{conv}(S) \subseteq \hat{S}_{box}$, and the gap between
    the two is exactly the source of the "integrality gap": when the
    constraint matrix $A$ is totally unimodular (e.g. bipartite matching,
    assignment, transportation polytopes), $\hat{S}_{box} = \mathrm{conv}(S)$
    and every vertex of the relaxed polytope is already integral, so rounding
    is lossless. For general 0/1 problems (knapsack, vertex cover, set cover)
    this does not hold and $\hat{S}_{box} \supsetneq \mathrm{conv}(S)$
- Optimize the relaxed problem: $\hat{x}^\star = \arg\max_{x \in \hat{S}}
  f(x)$, here via (projected) gradient descent on a penalized Lagrangian
  rather than a simplex/interior-point LP solver, to stay inside the
  differentiable-programming framing
- Round: $x^\star_{round} = \mathrm{round}(\hat{x}^\star)$, via thresholding
  (e.g. $x_i \mapsto \mathbb{1}[\hat{x}_i > 0.5]$), greedy repair, or
  randomized rounding, then measure
  ```latex
  \text{gap} = f(x^\star) - f(x^\star_{round})
  ```
  and, separately, whether $x^\star_{round} \in S$ (feasibility) before repair

### Concrete problem: 0/1 Knapsack

- Instance: $n$ items, value $v_i > 0$, weight $w_i > 0$, capacity $W$
- Discrete formulation:
  ```latex
  \max_{x \in \{0,1\}^n} \sum_{i=1}^n v_i x_i
  \quad \text{s.t.} \quad \sum_{i=1}^n w_i x_i \le W
  ```
- Box relaxation: $x \in [0,1]^n$, same linear constraint. Since the
  constraint matrix here is a single dense row (not totally unimodular in
  general), $\hat{S}_{box} \supsetneq \mathrm{conv}(S)$ and a genuine
  integrality gap can appear; the classical fractional-knapsack LP optimum
  is known in closed form (greedy by ratio $v_i / w_i$, admitting exactly one
  fractional item), which gives a ready-made ground truth for the relaxed
  optimum independent of the gradient-descent implementation
- Differentiable surrogate solved by gradient descent: unconstrained penalized
  objective
  ```latex
  L(x) = -\sum_i v_i x_i + \lambda \cdot \mathrm{relu}\Big(\sum_i w_i x_i - W\Big)^2
  ```
  minimized by projected gradient descent, clipping $x_i$ back to $[0,1]$
  after every step; $\lambda$ is annealed upward across iterations to push
  feasibility violations toward zero without destabilizing early gradients
- Rounding/repair: threshold at $0.5$, then if infeasible, drop items in
  increasing order of $v_i / w_i$ until $\sum w_i x_i \le W$; if feasible with
  slack, greedily add back items in decreasing order of $v_i / w_i$

## Key Examples

- **Five known uses of variable relaxation in combinatorial optimization**:
  - **LP relaxation of general 0/1 Integer Linear Programs**: dropping
    integrality constraints and solving the resulting LP is the standard
    bounding step inside branch-and-bound/branch-and-cut solvers
  - **Vertex Cover**: relax $x_i \in \{0,1\}$ (vertex $i$ in cover or not) to
    $x_i \in [0,1]$ subject to $x_i + x_j \ge 1$ per edge; rounding at
    threshold $0.5$ gives the classic 2-approximation
  - **Set Cover**: relax set-selection indicators to $[0,1]$, solve the
    covering LP, then apply randomized rounding (include set $i$ with
    probability $x_i$, repeat) to get an $O(\log n)$-approximation
  - **0/1 Knapsack** (the concrete example worked below): relax item
    selection to $[0,1]$; the fractional LP optimum is solvable by a greedy
    ratio rule and is within one item's value of the discrete optimum
  - **Assignment / transportation problems**: relax permutation matrices to
    the Birkhoff polytope of doubly stochastic matrices (row/column sums
    equal to 1 simplex constraints); here total unimodularity makes the
    relaxation exact, an instructive contrast case to knapsack/vertex cover
- **Worked example: 0/1 Knapsack with the relaxation transform applied**:
  10-item instance with $v_i \in \{60, 100, 120, 80, 90, 70, 50, 110, 65, 95\}$,
  $w_i \in \{10, 20, 30, 15, 18, 12, 8, 25, 11, 19\}$, capacity $W = 60$; the
  DP/exhaustive solver returns the exact optimum, projected gradient descent
  on $L(x)$ converges to a fractional $\hat{x}^\star$ matching the greedy
  ratio-order LP solution (one item fractional at the capacity boundary), and
  threshold-plus-repair rounding is checked against the exact answer both in
  value and in which items are selected
- **Edge case / failure mode**: a "superincreasing" instance where one item
  has very high value and weight (e.g. $v_1 = 1000, w_1 = 59$, $W = 60$, with
  many small low-ratio items filling the rest of the weight budget); the
  fractional relaxation optimum is dominated by splitting the big item, so
  rounding down loses almost its entire value while rounding up violates
  capacity, producing the largest observed gap and stress-testing the
  repair heuristic

## Questions

1. For which constraint-matrix structures does the box relaxation coincide
   exactly with $\mathrm{conv}(S)$ (total unimodularity being the classic
   sufficient condition), and does gradient descent reliably land on an
   integral vertex in those cases without any rounding step at all?
2. Can instance-level features (spread of $v_i/w_i$ ratios, tightness of $W$
   relative to $\sum w_i$) predict the size of the optimality gap before
   solving both formulations, or is the gap only knowable after the fact?
3. Does solving the relaxed problem with projected gradient descent (instead
   of an exact LP solver reaching the true polytope vertex) introduce its own
   additional gap on top of the intrinsic integrality gap, and how does that
   extra gap scale with learning rate, penalty schedule, and iteration budget?

## Research Topics

- **Rounding scheme comparison**: measure the value gap and feasibility rate
  of deterministic threshold rounding vs. greedy repair vs. randomized
  rounding (Raghavan-Thompson style) across many random knapsack and vertex
  cover instances
- **Penalty/Lagrangian design for gradient-based relaxation solving**: study
  how the penalty coefficient $\lambda$ (fixed vs. annealed) and learning rate
  affect convergence to the true LP-relaxation optimum and constraint
  satisfaction
- **Instance hardness characterization**: identify structural predictors
  (ratio spread, capacity tightness, number of near-tied items) of when the
  relaxation is tight (small gap) vs. loose (large gap), and test the
  predictors against the superincreasing-instance failure mode

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: survey known relaxation-based problems, pick the concrete
  problem, and formalize it
  - Confirm and document the five known-usage problems (LP-relaxed ILPs,
    vertex cover, set cover, knapsack, assignment/transportation) with one
    reference each
  - Formalize the 0/1 knapsack problem, its box relaxation, and the penalized
    gradient-descent surrogate as in the Formalization section
  - This is the result: a written formalization plus the specific test
    instances (baseline instance and the superincreasing edge case) to use in
    later milestones

- Milestone 2: implement the exact/discrete baseline solver in Python
  - Implement both an exhaustive $2^n$ solver (for small $n$, as a
    ground-truth cross-check) and a pseudo-polynomial dynamic-programming
    knapsack solver (for larger $n$)
  - Validate the two agree on all test instances
  - This is the result: a tested exact solver returning optimal value and
    selected items for any knapsack instance used later

- Milestone 3: implement the continuous relaxation + gradient descent +
  rounding pipeline in Python
  - Implement projected gradient descent on $L(x)$ with clipping to $[0,1]^n$
    and an annealed penalty coefficient $\lambda$
  - Implement the threshold-plus-repair rounding step
  - Cross-check the converged fractional solution against the closed-form
    greedy ratio-order LP optimum
  - This is the result: a working relaxation pipeline that outputs a
    fractional solution, its rounded discrete solution, and per-run
    convergence diagnostics

- Milestone 4: compare quality, gap, and runtime, and characterize when the
  relaxation is tight vs. loose
  - Run both solvers across a batch of randomly generated instances plus the
    superincreasing edge case, recording optimality gap, feasibility rate
    after rounding, wall-clock runtime, and sensitivity to random seed
    (stability)
  - Regress the observed gap against instance features (ratio spread,
    capacity tightness) to test the hardness-prediction question
  - This is the result: a quantitative table/plot of gap vs. runtime vs.
    instance difficulty, plus a short characterization of when relaxation is
    safe to use unmonitored and when it is not

## References
- P. Raghavan and C. D. Thompson, _Randomized Rounding: A Technique for
  Provably Good Algorithms and Algorithmic Proofs_. (1987)
- V. V. Vazirani, _Approximation Algorithms_. (2001)
- D. B. Shmoys and D. P. Williamson, _The Design of Approximation
  Algorithms_. (2011)
- A. Schrijver, _Combinatorial Optimization: Polyhedra and Efficiency_. (2003)
