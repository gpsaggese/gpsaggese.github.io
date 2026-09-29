# Optimal Training Holdout Split

## Status
- **Status:**: draft
- **Complete Specs:**: 0-100%
- **Assignee:**: ...

## Core Idea

- Common advice is to use a fixed train/holdout split (e.g., 60-40), but the
  optimal split likely depends on the problem
- Proposal: study the optimal holdout split as a function of:
  - Model complexity
  - Amount of noise in the data
  - Amount of data points
- Evaluate optimality along two axes:
  - How close the fitted model is to the true model
  - How close the estimated performance is to the actual performance

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. [Open question 1: what remains unknown?]
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- **Dataset selection**: pick several datasets (e.g., from Kaggle and
  synthetic datasets) to test the optimal split across different regimes

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build the synthetic data generators and curate real datasets
  - Implement synthetic data generators with a known true model (e.g.
    linear or polynomial with controllable degree) and controllable noise
    level $\sigma$ and sample size $N$
  - Select several Kaggle datasets spanning different sizes and apparent
    noise levels, to complement the synthetic grid with real-world data
  - This is the result: a synthetic generator covering a grid of $(N,
    \sigma, \text{complexity})$ settings, plus a curated set of Kaggle
    datasets for external validity checks

- Milestone 2: measure fit quality and estimate quality across split ratios
  - For each dataset and a grid of holdout fractions (e.g. 10% to 90%),
    train the model and measure, for synthetic data, distance between the
    fitted model and the true model (parameter error or true-risk gap)
  - For all datasets, measure the gap between holdout-estimated
    performance and actual performance (using a large held-out set or the
    known generating process for synthetic data)
  - This is the result: two metric surfaces, fit quality and estimate
    quality, each as a function of holdout fraction, for every $(N,
    \sigma, \text{complexity})$ setting

- Milestone 3: derive and validate a rule of thumb for the optimal split
  - Fit an empirical relationship predicting the optimal holdout fraction
    from $N$, $\sigma$, and model complexity, using the Milestone 2
    surfaces (e.g. via regression, or a bias-variance argument in the
    style of the $VC_{eff}$ formalization)
  - Validate the resulting rule out-of-sample on the Kaggle datasets held
    out from the fitting step
  - This is the result: a proposed formula for the optimal holdout
    fraction, with its prediction error measured against the brute-force
    optimal split on validation datasets

- Milestone 4: compare against cross-validation and bootstrap
  - Benchmark the derived single-split rule against k-fold cross-
    validation and bootstrap resampling on the same datasets, tracking
    both estimate quality and total compute cost
  - This is the result: a compute-versus-estimate-quality comparison
    showing where a well-chosen single split is competitive with, or
    dominated by, cross-validation

## References

- Author(s), _Title_. (Year)
