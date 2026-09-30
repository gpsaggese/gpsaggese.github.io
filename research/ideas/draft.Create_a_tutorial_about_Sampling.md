# Create a Tutorial About Sampling

## Status
- **Status**: draft
- **Complete Specs**: TBD

## Core Idea
- Build a tutorial that teaches how to understand, implement, and apply
  sampling methods for probabilistic inference, bridging theory and practice
- Cover foundational methods (uniform sampling, rejection sampling, importance
  sampling) and modern approaches (Gibbs sampling, MCMC, Hamiltonian Monte
  Carlo)
- Explain the mathematical foundations of each technique alongside working
  Python implementations and visualizations of their behavior
- Show when and why to use each technique through real-world examples from
  Bayesian inference, generative models, and reinforcement learning
- Demonstrate convergence behavior, computational efficiency, and accuracy
  trade-offs through interactive code examples
- Target students learning probabilistic graphical models, Bayesian
  statistics, and advanced machine learning

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```text
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples
- **UCI Machine Learning Repository**: diverse datasets for Bayesian
  inference tasks (classification, clustering), free direct download, used to
  demonstrate MCMC sampling on real classification problems
- **Kaggle Datasets**: wide variety of datasets for probabilistic modeling and
  inference tasks, free with registration, used to apply importance sampling
  and Gibbs sampling to real-world prediction problems
- **PyMC Example Data**: datasets curated for Bayesian modeling tutorials,
  included free with the PyMC library, following established best practices
  from production Bayesian workflows
- **Synthetic toy datasets**: custom distributions with known ground truth,
  generated on demand to demonstrate sampling behavior and convergence on
  simple, interpretable problems

## Questions

1. [Open question 1: what remains unknown?]
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- [Topic 1]: [What to investigate]
- [Topic 2]: [What to investigate]
- [Topic 3]: [What to investigate]

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: core sampling techniques and comparison framework
  - Implement and visualize uniform random sampling, rejection sampling, and
    importance sampling with example distributions
  - Create interactive demonstrations showing convergence behavior of
    different MCMC methods (Metropolis-Hastings, Gibbs sampling)
  - Build a comparison framework measuring computational cost and accuracy
    across sampling techniques
  - Implement Hamiltonian Monte Carlo (HMC) and contrast with simpler MCMC
    approaches
  - Develop tutorials covering sampling in mixture models, latent variable
    models, and Bayesian regression
  - Create visualizations showing how proposal distributions affect MCMC
    acceptance rates and mixing

- Milestone 2: extensions
  - Implement variational inference methods and compare with sampling-based
    approaches
  - Add adaptive sampling techniques that learn proposal distributions during
    inference
  - Build a practical guide on diagnosing MCMC convergence using trace plots
    and diagnostics
  - Implement parallel tempering for sampling from multimodal distributions
  - Create a web-based interactive tool for visualizing sampling in 2D and 3D
    distributions
  - Develop benchmarks comparing sampling speed and accuracy on
    high-dimensional problems
  - Include modern methods like automatic differentiation variational
    inference (ADVI) or neural posterior estimation

## References
- Ermon, S. (2016). _CS 228: Probabilistic Graphical Models: Sampling_.
  https://ermongroup.github.io/cs228-notes/inference/sampling/
- Gelman, A., et al. (2011). _Bayesian Data Analysis_. Chapman and Hall/CRC
- Burda, Y., et al. (2013). _Importance Weighted Autoencoders_.
  https://arxiv.org/abs/1509.00519
- Kucukelbir, A., et al. (2014). _Automatic Variational Inference in Stan_.
  https://arxiv.org/abs/1506.03431
- Betancourt, M. (2017). _A Conceptual Introduction to Hamiltonian Monte
  Carlo_. https://arxiv.org/abs/1701.02434
- PyMC Documentation. _Sampling and Inference_. https://docs.pymc.io/
