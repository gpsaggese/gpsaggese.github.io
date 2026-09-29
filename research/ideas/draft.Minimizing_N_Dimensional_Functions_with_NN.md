# Minimizing an N-Dimensional Function with a Neural Network

## Status

- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: TBD

## Core Idea

- Use a neural network to help find the minimum of an expensive or
  black-box function \(f: \mathbb{R}^{N} \to \mathbb{R}\)
- Two variants:
  - As a cheap surrogate that guides an iterative search
  - As a direct amortized mapping from a problem instance to its
    minimizer, skipping the iterative search entirely at inference time

## Formalization

- Given \(f: \mathbb{R}^{N} \to \mathbb{R}\), the goal is
  \(x^{*} = \arg\min_{x} f(x)\)
- **Surrogate optimization**: train a NN \(\hat{f}_{\theta} \approx f\)
  from sampled points, then use \(\hat{f}_{\theta}\)'s gradient or
  predictive uncertainty to choose the next query point
  (Bayesian-optimization style, with a NN surrogate instead of a Gaussian
  Process)
- **Amortized optimization**: train a NN \(h_{\phi}\) that maps a problem
  instance \(c\) (e.g., the parameters defining a family of functions
  \(f_{c}\)) directly to its minimizer:
  \[
  h_{\phi}(c) \approx \arg\min_{x} f_{c}(x)
  \]
  trained across many sampled instances \(c\), amortizing the search cost
  over the whole family instead of paying it once per instance

## Key Examples

- **Hyperparameter tuning**: Bayesian optimization with a NN surrogate
  (e.g., DNGO) instead of a Gaussian Process, to scale better with the
  number of observations
- **Amortized variational inference**: mapping distribution parameters
  directly to (near-)optimal variational parameters, instead of
  re-running gradient-based inference for every new input
- **Molecular/energy minimization**: NN potentials used to find low-energy
  molecular geometries, replacing expensive physics-based energy
  evaluations with a learned surrogate
- **Constrained portfolio choice**: worked out separately in
  [[draft.Mean_Variance_Optimization_with_NN]], as an instance of the
  amortized variant where the constraint set breaks the closed-form
  solution

## Questions

1. When is the upfront cost of training an amortized optimizer worth it
   compared to solving each instance independently with classical
   iterative methods?
2. Surrogate NNs typically lack the calibrated uncertainty estimates of a
   Gaussian Process: how much does this hurt the exploration/exploitation
   trade-off in a Bayesian-optimization-style loop?
3. Can amortized and iterative approaches be combined (NN gives a fast
   initial guess, then a few steps of classical local refinement)?

## Research Topics

- Benchmark both variants against CMA-ES, Bayesian optimization
  (GP-based), and plain gradient descent on standard test functions
  (Rosenbrock, Rastrigin, Ackley)
- Compare wall-clock time and function-evaluation budget needed to reach a
  target optimality gap
- Sensitivity of the amortized approach to distribution shift between
  training instances and test instances
- Differentiable convex optimization layers (`cvxpylayers`, OptNet) so a
  solver can sit inside the network and constraints are satisfied by
  construction

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
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References

- Snoek et al., _Scalable Bayesian Optimization Using Deep Neural
  Networks_. (2015), known as "DNGO"
- Chen et al., _Learning to Learn without Gradient Descent by Gradient
  Descent_. (2017)
- Amos, B., and Kolter, J. Z., _OptNet: Differentiable Optimization as a
  Layer in Neural Networks_. (2017)
- Derived from `draft.Misc_ML_ideas.md`
