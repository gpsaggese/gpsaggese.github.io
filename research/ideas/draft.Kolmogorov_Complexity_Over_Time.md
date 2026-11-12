# Kolmogorov Complexity Over Time

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- Use Kolmogorov complexity to measure the compressibility of a data
  stream over time
- Changes in compression rate can signal regime shifts, concept drift, or
  fundamental changes in the underlying process

## Formalization

- Measure compressibility of an entire data stream:
  \[
  K(x_{1:T})
  \]
- Or its incremental description length:
  \[
  L = \sum_{t=1}^{T} K(x_{t} \mid x_{<t})
  \]
- A sudden increase in \(K(x_{t} \mid x_{<t})\) indicates that past
  patterns no longer predict the present: a regime shift has occurred

## Key Examples

- **Financial time series**: a market that suddenly becomes less
  compressible (higher conditional Kolmogorov complexity) may indicate a
  regime change: the 2008 financial crisis or COVID-19 market turbulence
  are canonical examples where past patterns stopped predicting future
  behavior
- **Industrial sensor data**: predictive maintenance systems where
  machinery degradation causes the signal to become progressively less
  compressible before failure
- **Network intrusion detection**: normal traffic patterns have regular
  structure (low \(K\)); attacks introduce novel patterns, increasing
  \(K(x_{t} \mid x_{<t})\)

## Questions

1. Is there a fundamental limit to predictability based on Kolmogorov
   complexity? If \(K(x_{t} \mid x_{<t})\) approaches \(|x_{t}|\) (data is
   incompressible given past), does this imply the system has become
   random, or that the model class is too weak?

## Research Topics

- Using compression rate changes to detect regime shifts in real time
- Understanding the gap between Kolmogorov complexity (uncomputable) and
  practical compression-based approximations
- Bounding predictability from above using conditional Kolmogorov
  complexity
- Connecting compression-based change detection to PAC-style learning
  bounds

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build a practical compressibility estimator
  - Implement an estimator for $K(x_t \mid x_{<t})$ using an off-the-shelf
    compressor (e.g., gzip/LZ77 or PPM) and, separately, a small sequence
    model's negative log-likelihood as a compression proxy
  - Validate the estimator on synthetic streams with known, controlled
    regime shifts (e.g., switching the generating process at a known time)
  - This is the result: an estimator that correctly flags the known
    injected regime shifts in synthetic data

- Milestone 2: implement online change-point detection
  - Track the incremental description length
    $L = \sum_{t=1}^{T} K(x_t \mid x_{<t})$ online, and flag a regime shift
    when its rate jumps past a calibrated threshold
  - Benchmark detection accuracy and lead time against standard baselines
    (CUSUM, Bayesian online changepoint detection) on the synthetic streams
  - This is the result: a detector with measured precision/recall and
    detection lead time versus the two baselines

- Milestone 3: apply the detector to the key example domains
  - Run the detector on a financial time series spanning a known regime
    change (e.g., 2008 crisis or COVID-19 turbulence) and on a network
    intrusion detection dataset with labeled attack windows
  - Measure detection precision/recall and lead time against the labeled
    events in each dataset
  - This is the result: precision/recall and lead-time numbers on two real
    labeled domains, compared to the synthetic-data results

- Milestone 4: study the compression-approximation gap and predictability
  limit
  - Compare detection quality across compressor choices (gzip/LZ77 vs PPM
    vs neural log-likelihood) to characterize how the choice of
    approximation to the uncomputable $K$ affects detection
  - Empirically test whether $K(x_t \mid x_{<t})$ approaches $|x_t|$ (full
    incompressibility) near labeled regime boundaries, as a proxy for the
    predictability-limit question
  - This is the result: a comparison of compressor choices, plus an
    empirical answer to whether conditional complexity approaches the
    incompressibility limit at regime boundaries

## References

- Derived from `Research_plan/paper.tex` (Section: Quasi-Stationary
  Learning / Kolmogorov Complexity Over Time)
