# Number Representations that Speed Up Gradient Descent

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- Explore a special representation of numbers and operations that makes
  gradient descent converge faster
  - E.g., integer vs floating point, log representations

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

- Milestone 1: build a pluggable numeric backend and baseline
  - Implement gradient descent on a small suite of controlled losses
    (a convex quadratic with tunable condition number, logistic
    regression, and a small MLP) using a numeric backend that can be
    swapped without changing the optimizer logic
  - Record baseline convergence curves (loss versus iteration) in
    standard fp32, for each problem in the suite
  - This is the result: a working GD testbed with fp32 baseline
    convergence curves for each problem type

- Milestone 2: implement and verify alternative representations
  - Add fixed-point/integer, log-number-system (LNS), and reduced-width
    floating-point (fp16/bf16) representations for weights, gradients,
    and the update rule, matching each arithmetic op (add, multiply, the
    parameter update) to its fp32 counterpart within a stated tolerance
  - Write correctness tests comparing each representation's per-step
    update against the fp32 reference on the same inputs
  - This is the result: a set of drop-in numeric representations verified
    to compute the same GD update as fp32 up to their intrinsic precision

- Milestone 3: measure convergence speed across representations
  - Run GD to a fixed target loss on every problem/representation
    combination, sweeping the quadratic's condition number and the
    MLP's depth, and record iterations-to-target and a bit-budget-matched
    comparison across representations
  - This is the result: a table of iterations-to-target-loss per
    representation and problem, identifying which representation
    converges fastest at matched bit budget

- Milestone 4: explain the effect and test a hybrid scheme
  - Investigate a mechanistic explanation for any speedup found (e.g. the
    log representation turning multiplicative relationships into additive
    ones, changing effective step size behavior on ill-conditioned
    problems)
  - Implement a hybrid scheme that switches representation during
    training (e.g. log or integer early, fp32 near convergence) and
    compare it against the best single fixed representation from
    Milestone 3
  - This is the result: a hybrid switching scheme with measured speedup,
    or a documented null result, relative to the best fixed representation

## References

- Author(s), _Title_. (Year)
