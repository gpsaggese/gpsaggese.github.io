# Kolmogorov Complexity Over Time

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: TBD

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

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References

- Derived from `Research_plan/paper.tex` (Section: Quasi-Stationary
  Learning / Kolmogorov Complexity Over Time)
