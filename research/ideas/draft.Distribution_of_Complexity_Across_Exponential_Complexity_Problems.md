# Distribution of Complexity Across Exponential-Complexity Problems

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- A problem can have exponential complexity, but how is that complexity
  distributed across the space of problem instances?

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

- Milestone 1: build an instrumented solver testbed
  - Pick 2-3 canonical exponential-complexity problems: random 3-SAT,
    subset-sum, and graph coloring (or TSP)
  - Wrap an exact solver for each (CDCL for SAT, branch-and-bound for
    subset-sum/TSP) that logs per-instance cost: wall-clock time, node count,
    backtrack count
  - Build an instance generator with a controllable parameter per problem
    (e.g. clause/variable ratio for 3-SAT, weight density for subset-sum)
  - This is the result: a pipeline that generates instances at a chosen
    parameter setting and returns a measured per-instance solve cost

- Milestone 2: measure the empirical hardness distribution
  - Sweep each problem's controlling parameter across a fine grid (e.g.
    clause/variable ratio 3.0 to 5.5 for 3-SAT), generating $N$ instances per
    grid point
  - Record the full solve-cost distribution per grid point, not only the
    mean, including percentiles and spread
  - This is the result: hardness-distribution plots per problem, showing
    where cost concentrates and how spread changes as the parameter moves

- Milestone 3: relate the distribution shape to phase transitions
  - Fit candidate distributions (log-normal, power-law/heavy-tailed) to the
    per-grid-point cost distribution and test goodness of fit
  - Check whether the hardest instances cluster near known phase-transition
    boundaries (e.g. the approximately 4.27 clause/variable ratio for random
    3-SAT)
  - Quantify the tail: the fraction of instances whose cost exceeds 10x and
    100x the median at each parameter setting
  - This is the result: a fitted distribution family per problem/parameter
    setting, with a quantified answer on whether hardness concentrates near a
    transition or spreads broadly

- Milestone 4: propose and test a distributed-complexity measure
  - Define a per-instance hardness score normalized against the problem's
    worst-case bound, and an aggregate measure (e.g. tail-index or entropy of
    the cost distribution) summarizing how uneven complexity is
  - Test whether the measure computed on one problem (e.g. 3-SAT) predicts
    the distribution shape found for another (e.g. subset-sum)
  - This is the result: a candidate distributed-complexity metric with a
    cross-problem comparison, answering whether hardness spread is a
    transferable property or problem-specific

## References
- Author(s), _Title_. (Year)
