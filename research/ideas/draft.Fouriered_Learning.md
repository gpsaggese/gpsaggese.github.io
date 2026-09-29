# Fouriered Learning

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- Instead of learning a mapping directly between inputs and outputs, apply a
  Fourier transform to both first, then learn the mapping between their
  frequency-domain coefficients
- Many real-world relationships (periodic, smooth, or band-limited signals)
  become far simpler (e.g., sparse or linear) once expressed in the frequency
  domain, so the learning problem itself may become easier even though no
  information is added

## Formalization

- Let $x, y$ be the original input/output, with Fourier transforms
  $\hat{x} = \mathcal{F}(x)$ and $\hat{y} = \mathcal{F}(y)$
- Instead of learning $f$ such that $y \approx f(x)$, learn $g$ such that:
  $$
  \hat{y} \approx g(\hat{x}), \qquad y = \mathcal{F}^{-1}(g(\hat{x}))
  $$
- The central question is when $g$ has lower complexity (sparser, more
  linear, lower VC dimension) than $f$ does in the original domain

## Key Examples

- **Seasonal time series forecasting**: periodic components (e.g., yearly
  seasonality) collapse into a few sparse spikes in the frequency domain, so
  a linear model on Fourier coefficients may outperform a nonlinear model
  fit directly on the raw series
- **Fourier Neural Operators (FNO)**: learn mappings between function spaces
  (e.g., PDE solution operators) via spectral convolutions, exploiting the
  same idea of computing in frequency space
- **Audio/image restoration**: denoising or super-resolution models that
  correct low- and high-frequency bands separately, rather than the raw
  signal as a whole

## Questions

1. Does the frequency-domain mapping $g$ provably have lower sample
   complexity (VC dimension, Rademacher complexity) than $f$ for classes
   of periodic or band-limited functions?
2. Which function classes admit a sparse or low-complexity representation in
   the Fourier domain, and can this be predicted ahead of training?
3. Does the idea generalize to other exchange-of-basis transforms (wavelets,
   learned/data-driven bases) beyond the Fourier basis?

## Research Topics

- Compare sample efficiency and error of learning in the Fourier domain vs.
  the original domain across forecasting/regression benchmarks
- Relationship to Fourier Neural Operators and spectral methods for PDEs
- Identify which application domains (finance, physics, audio) benefit most
  from a frequency-domain reformulation

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

- Li et al., _Fourier Neural Operator for Parametric Partial Differential
  Equations_ (2020)
- Derived from `draft.Misc_ML_ideas.md` (Section: Fouriered Learning)
