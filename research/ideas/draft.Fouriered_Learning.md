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

- Milestone 1: build synthetic benchmarks with controlled frequency-domain
  sparsity
  - Construct a dataset from a known band-limited/periodic function class
    (few dominant frequencies) so Fourier-domain sparsity is controllable
  - Construct a matched "dense" control dataset whose true mapping is not
    simpler in the Fourier domain, as a negative control
  - This is the result: two synthetic datasets (Fourier-sparse and
    Fourier-dense) with known ground-truth complexity in each domain

- Milestone 2: compare learning $f$ (raw domain) vs. $g$ (Fourier domain)
  - Implement the raw-domain baseline: learn $f$ with $y \approx f(x)$
    directly
  - Implement the Fourier-domain model: compute $\hat x = \mathcal F(x)$,
    $\hat y = \mathcal F(y)$, learn $g$ with $\hat y \approx g(\hat x)$,
    and invert with $\mathcal F^{-1}$
  - Measure sample complexity (error vs. training-set size) and model
    complexity (parameter count, sparsity) for both on both benchmarks
  - This is the result: sample-efficiency curves comparing $f$ and $g$ on
    the Fourier-sparse and Fourier-dense benchmarks

- Milestone 3: validate on a real seasonal forecasting dataset
  - Apply both approaches to a real seasonal time series (e.g.,
    electricity demand or retail sales with known yearly/weekly
    seasonality)
  - Compare forecast error and sample efficiency of the Fourier-domain
    model against the raw-domain baseline and a standard seasonal baseline
    (e.g., SARIMA)
  - This is the result: a forecast-error and sample-efficiency comparison
    table on a real dataset, benchmarked against a standard seasonal model

- Milestone 4: test generalization to other bases and to FNO-style
  spectral learning
  - Repeat the Milestone 2 comparison with a wavelet basis in place of the
    Fourier basis, to check whether the benefit is basis-specific
  - Implement a small Fourier Neural Operator baseline on a PDE-style task
    and compare it against the raw-domain and Fourier-domain regressions
  - This is the result: an assessment of whether the frequency-domain
    advantage generalizes beyond Fourier, and how it relates to FNO's
    spectral-convolution approach

## References

- Li et al., _Fourier Neural Operator for Parametric Partial Differential
  Equations_ (2020)
- Derived from `draft.Misc_ML_ideas.md` (Section: Fouriered Learning)
