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

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References

- Author(s), _Title_. (Year)
