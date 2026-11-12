# Benchmarking Symbolic Regression Libraries

## Status

- **Status:**: draft
- **Complete Specs:**: 30%

## Core Idea

- Symbolic regression searches over closed-form expressions that fit data,
  returning a formula rather than an opaque predictor
  - The output is interpretable and extrapolates when the discovered form is
    the right one
  - The search space is combinatorial, so every library makes different
    trade-offs between search strategy, prior over operators, and complexity
    penalty
- Several mature implementations exist, each with a distinct strategy, and
  there is no single answer to which one to use on a new dataset
- Proposal: run a controlled comparison of the main libraries on a common
  benchmark, measuring not just fit but exact-formula recovery, robustness to
  noise, and sensitivity to the number of data points
- The non-obvious part is that goodness of fit is the wrong headline metric
  - A high $R^2$ from a 40-node expression is worse than a slightly lower
    $R^2$ from the true 5-node law
  - Recovery rate of the ground-truth expression is what actually
    distinguishes these methods, and it can only be measured on synthetic
    data where the truth is known

## Formalization

- Given data $\{(x_i, y_i)\}$, find an expression $g$ from a grammar of
  operators minimizing a complexity-penalized loss:
  ```
  g* = argmin_g  MSE(g, D) + lambda * complexity(g)
  ```
  - `complexity(g)` is typically the node count of the expression tree
  - The result is a Pareto front over (`accuracy`, `complexity`), not a single
    model
- Evaluation metrics, reported per method:
  - **Recovery rate**: fraction of problems where $g^*$ is symbolically
    equivalent to the ground truth, checked with a computer algebra system
    rather than by string match
  - **Pareto dominance**: area under the accuracy vs complexity front
  - **Extrapolation error**: test error outside the training support, which is
    where a wrong-but-fitting formula is exposed
  - **Cost**: wall-clock and evaluations to reach the solution
- Stress axes:
  - Noise level $\sigma$ added to $y$
  - Number of samples $n$
  - Number of input variables $d$ and presence of irrelevant variables
- Selection effect: running many methods, each with many hyperparameter
  settings, and reporting the best is the same search penalty described in
  `draft.Backtesting_Complexity.md`, so the trial count must be logged

## Key Examples

- **AI Feynman** (Max Tegmark's approach): exploits physics-motivated
  structure (symmetry, separability, dimensional analysis) to recursively
  simplify the problem
  - Install: `pip install aifeynman`
  - Docs say it is supported on Linux and macOS, not Windows-native
- **PySR**: Python interface to a fast backend, widely used in science for
  interpretable formulas
  - Available on PyPI and conda-forge, with continuing releases
- **gplearn**: classic genetic programming with a scikit-learn-style API via
  `SymbolicRegressor`
- **PhySO**: physics-oriented symbolic regression aimed explicitly at
  inferring analytical functions from data, strong on physics-style problems
- **Failure mode**: a method that recovers the Feynman equations (its own
  design target) but fails on an arbitrary algebraic expression, so the
  benchmark must include problems outside any method's home distribution
- **Failure mode**: expressions that fit in-sample and diverge outside the
  training support, which only the extrapolation metric catches

## Questions

1. Which method has the highest ground-truth recovery rate as a function of
   noise and sample count, and does the ranking change across those axes?
2. How much of AI Feynman's and PhySO's advantage comes from physics priors
   (dimensional analysis, symmetry) versus the search itself? An ablation
   removing units would answer this
3. Do the methods degrade gracefully with irrelevant input variables, or is
   variable selection a separate prerequisite step?
4. Where does an LLM-driven proposer fit against these baselines? This is the
   question in `draft.LLM_for_Symbolic_Regression.md` and
   `draft.Closed_Form_Formula_Discovery.md`, and the benchmark built here is
   what those ideas must be measured against

## Research Topics

- **Benchmark suites**: SRBench and the Feynman and Strogatz problem sets as
  the standard evaluation, plus synthetic expressions generated to a
  controlled complexity
- **Search strategies**: genetic programming, neural-guided search,
  brute-force with pruning, and physics-informed decomposition
- **Complexity penalties**: node count vs MDL-style description length, and
  the effect on the recovered expression, which connects to
  `draft.MDL_Extensions_with_Research_Process.md`
- **Symbolic equivalence checking**: using `sympy` to compare candidate and
  ground truth rather than comparing strings
- **Reproducibility**: pinning environments per library, since these packages
  have heavy and conflicting dependencies

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: environment and smoke test
  - Install each library in its own container or virtual environment, since
    dependencies conflict
  - Run one known problem per library end to end
  - This is the result: a reproducible harness with one runnable example per
    method

- Milestone 2: benchmark and scorer
  - Assemble the problem set (Feynman, Strogatz, plus generated expressions
    at controlled complexity)
  - Implement the scorer: symbolic equivalence via `sympy`, Pareto front,
    extrapolation error, and cost
  - This is the result: a single command that scores any method on the suite

- Milestone 3: the comparison
  - Run all methods across the noise and sample-count grid with a fixed
    compute budget per problem
  - This is the result: recovery-rate curves per method, plus a cost-matched
    ranking with the number of hyperparameter trials reported

- Milestone 4: ablations
  - Remove physics priors from the methods that use them and re-run
  - Add irrelevant input variables and re-run
  - This is the result: an attribution of performance to priors vs search,
    and a practical guide on which method to reach for on a new dataset

## References

- Udrescu and Tegmark, _AI Feynman: A Physics-Inspired Method for Symbolic
  Regression_. (2020)
- Cranmer, _Interpretable Machine Learning for Science with PySR and
  SymbolicRegression.jl_. (2023)
- Tenachi et al., _Deep Symbolic Regression for Physics Guided by Units
  Constraints (PhySO)_. (2023)
- La Cava et al., _Contemporary Symbolic Regression Methods and their
  Relative Performance (SRBench)_. (2021)
- https://arxiv.org/abs/2505.10762
