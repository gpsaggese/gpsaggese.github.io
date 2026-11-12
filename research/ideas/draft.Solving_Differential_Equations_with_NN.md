# Solving Differential Equations with a Neural Network (Physics-Informed NN)

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- Parameterize the solution of an ODE/PDE as a neural network $u_{\theta}(x,
  t)$ instead of discretizing it on a mesh
- Automatic differentiation gives exact derivatives of $u_{\theta}$: the
  governing equation's residual plus its boundary/initial conditions can be
  added directly as loss terms
- Training pushes the network to satisfy the equation everywhere, without any
  grid (Physics-Informed Neural Networks, PINNs)

## Formalization

- For a differential operator $\mathcal{N}$ with source term $f$:
  $$
  \mathcal{N}[u](x, t) = f(x, t)
  $$
- Residual of the NN approximation $u_{\theta}$:
  $$
  R_{\theta}(x, t) = \mathcal{N}[u_{\theta}](x, t) - f(x, t)
  $$
- Training loss over collocation points, boundary points, and initial points:
  $$
  L(\theta) = w_{r} \, \underset{(x,t) \in \Omega}{\mathrm{mean}}\,
    R_{\theta}(x,t)^{2}
    + w_{b} \, \underset{(x,t) \in \partial\Omega}{\mathrm{mean}}\,
    (u_{\theta} - u_{b})^{2}
    + w_{i} \, \underset{x}{\mathrm{mean}}\,
    (u_{\theta}(x,0) - u_{0}(x))^{2}
  $$

## Key Examples

- **Burgers' equation**: the canonical PINN benchmark (Raissi et al., 2019),
  a nonlinear PDE with a known analytical/numerical reference solution
- **Heat/diffusion and Schrodinger equations**: standard test cases with
  known closed-form or high-accuracy numerical solutions to validate against
- **Inverse/parameter-estimation problems**: use a PINN to simultaneously
  solve the PDE and infer unknown coefficients (e.g., diffusivity) from
  sparse, noisy observations: mesh-based solvers cannot do this without a
  separate inversion loop

## Questions

1. Why do PINNs struggle with stiff, multi-scale, or high-frequency
   solutions (the "spectral bias" of neural networks toward low
   frequencies), and can architecture or loss-reweighting changes fix it?
2. How does PINN accuracy/compute compare to classical solvers
   (finite-difference, finite-element) at a fixed accuracy target. Does the
   comparison flip in high dimensions, where mesh-based methods suffer the
   curse of dimensionality but a NN's cost does not scale with grid size?
3. For inverse problems with sparse/noisy data, does the PINN approach
   outperform classical parameter-estimation methods, and by how much?

## Research Topics

- **Benchmark against classical solvers**: compare PINNs vs.
  finite-difference/finite-element solutions on 1D/2D PDEs with known
  solutions, quantifying accuracy vs. compute trade-offs
- **Adaptive collocation sampling**: adaptive collocation-point sampling to
  counter spectral bias
- **High-dimensional PDEs**: apply to problems where mesh methods become
  infeasible (e.g., Black-Scholes-type equations in many dimensions)

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: implement and validate the core PINN framework
  - Implement a generic PINN trainer: an NN $u_{\theta}(x, t)$, autodiff-based
    residual $R_{\theta}$, and collocation/boundary/initial-point sampling
    per the Formalization loss $L(\theta)$
  - Validate on Burgers' equation, the canonical PINN benchmark, against its
    known analytical/numerical reference solution
  - This is the result: a working PINN implementation that reproduces the
    Burgers' equation solution within an acceptable error vs. the reference

- Milestone 2: benchmark against classical solvers
  - Implement finite-difference/finite-element baseline solvers for the
    same 1D/2D PDEs (heat/diffusion and Schrodinger equations)
  - Compare accuracy vs. compute (wall-clock time, evaluation count) between
    the PINN and the classical solvers at matched accuracy targets
  - This is the result: an accuracy-vs-compute comparison across PDEs,
    directly addressing the Research Topics benchmarking question

- Milestone 3: investigate spectral bias and adaptive sampling
  - Construct a stiff, multi-scale, or high-frequency test case where
    spectral bias is expected to hurt PINN accuracy
  - Test architecture and loss-reweighting variants, plus an adaptive
    collocation-point sampling strategy, against uniform sampling
  - This is the result: a quantified improvement (or lack of one) from
    adaptive sampling and reweighting on the spectral-bias failure case

- Milestone 4: extend to inverse and high-dimensional problems
  - Implement the inverse/parameter-estimation variant: infer an unknown
    coefficient (e.g., diffusivity) from sparse, noisy observations, and
    compare against a classical parameter-estimation baseline
  - Attempt a high-dimensional PDE (e.g., a Black-Scholes-type equation)
    where mesh-based methods become infeasible
  - This is the result: inverse-problem accuracy vs. the classical baseline,
    plus a demonstration of PINN scaling to a high-dimensional case that
    mesh-based methods cannot handle

## References

- Raissi, Perdikaris, Karniadakis, _Physics-Informed Neural Networks: A Deep
  Learning Framework for Solving Forward and Inverse Problems Involving
  Nonlinear Partial Differential Equations_. (2019)
- Han, Jentzen, E, _Solving High-Dimensional Partial Differential Equations
  Using Deep Learning_. (2018)
- Derived from `draft.Misc_ML_ideas.md`
