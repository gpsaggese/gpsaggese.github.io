# Math Primitives for Neural Networks

## Status

- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- What are the different math primitives that can be used in a neural
  network
  - E.g., add, mul, relu

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```text
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

- Milestone 1: catalog the primitive set
  - Survey the operators used as building blocks in neural network layers
    (add, multiply, relu/max, min, comparison/step, division, exp/log) and
    group them into families (linear, piecewise-linear, multiplicative,
    saturating)
  - For each primitive, record its per-op computational cost (FLOPs,
    hardware cost proxy) and any known expressivity property (e.g.,
    universal-approximation role of relu, extrapolation role of
    multiplicative units)
  - This is the result: a taxonomy table of candidate primitives with cost
    and known expressivity notes

- Milestone 2: build a primitive-swappable architecture testbed
  - Implement a small MLP/CNN framework where each layer's core operator
    can be swapped: standard matmul-plus-add, an addition-only layer
    (L1-distance in place of multiplication), a multiplication-only layer,
    and an explicit add/multiply arithmetic unit
  - Train and validate each variant on toy tasks (synthetic arithmetic
    regression, MNIST classification) to confirm the swapped layers train
    stably
  - This is the result: a working testbed where each primitive variant
    trains to a sane baseline accuracy on the toy tasks

- Milestone 3: run an ablation across primitive sets
  - Systematically restrict which primitives a network may use (add+relu
    only, multiply only, add+multiply+relu, add+multiply+relu+comparison)
  - Measure accuracy, parameter efficiency, and FLOP cost for each
    restricted set across the toy tasks from Milestone 2
  - This is the result: an ablation table showing which minimal primitive
    sets suffice per task type, and the accuracy/cost trade-off of adding
    more primitives

- Milestone 4: test a targeted extrapolation task
  - Pick a task motivated by the ablation results where standard
    add+relu networks are expected to struggle at extrapolating beyond
    the training range (e.g., learning an arithmetic function on inputs
    outside the training interval)
  - Compare a standard network against one including the primitive the
    ablation flagged as relevant, on in-distribution and
    out-of-distribution accuracy
  - This is the result: evidence for or against the hypothesis that adding
    a targeted primitive beyond add/relu improves out-of-distribution
    extrapolation on that task

## References

- Author(s), _Title_. (Year)
