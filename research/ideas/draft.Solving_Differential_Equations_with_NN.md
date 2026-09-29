# Solving Differential Equations with a Neural Network (Physics-Informed NN)

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: TBD

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

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References

- Raissi, Perdikaris, Karniadakis, _Physics-Informed Neural Networks: A Deep
  Learning Framework for Solving Forward and Inverse Problems Involving
  Nonlinear Partial Differential Equations_. (2019)
- Han, Jentzen, E, _Solving High-Dimensional Partial Differential Equations
  Using Deep Learning_. (2018)
- Derived from `draft.Misc_ML_ideas.md`
