# How AlphaEvolve Works (and Reproducing It at Small Scale)

## Status
- **Status:**: draft
- **Complete Specs:**: 15%
- **Assignee:**: TBD

## Core Idea
- DeepMind's AlphaEvolve pairs an LLM (proposes code mutations) with an
  evolutionary search loop (maintains a population of candidate programs
  scored by an automatic evaluator) to discover new algorithms/heuristics that
  beat hand-designed baselines
- Understand the architecture well enough to reproduce a miniature version:
  LLM-driven mutation + selection + evaluator, applied to a small, cheaply
  scored problem (not matrix multiplication at DeepMind's scale, but something
  a single researcher can iterate on in days)
- Related to [[draft.Closed_Form_Formula_Discovery]] and
  [[draft.LLM_for_Symbolic_Regression]]: same "LLM proposes, evaluator
  scores, search selects" pattern applied to different discovery targets

## Formalization
- Population `P_t` of candidate programs at generation `t`
- Mutation: `p' = LLM(prompt(p, feedback))`
  - The LLM rewrites/edits a parent program conditioned on its score and
    (optionally) an execution trace/error
- Selection: keep top-k by evaluator score `f(p)`, with some diversity
  mechanism (e.g., island populations) to avoid premature convergence

## Key Examples
- **Toy target**: evolve a faster/shorter sorting-network or a better
  approximation constant for a known inequality, where `f(p)` is cheap to
  compute (runtime, correctness, approximation error)
- **Failure mode**: LLM mutations converge to trivial "cheat" solutions that
  exploit gaps in the evaluator rather than genuinely improving the
  algorithm (reward hacking)
  - Worth documenting explicitly

## Questions
1. How much of AlphaEvolve's benefit comes from the LLM's code-editing prior
   vs. from the evolutionary search structure itself (compare LLM-mutation
   vs. random-mutation baselines at matched compute)?
2. How do you design an evaluator robust to reward hacking on a small,
   cheaply-scored problem?
3. Does feeding execution traces/error messages back into the mutation prompt
   meaningfully speed up convergence vs. score-only feedback?

## Research Topics
- Evolutionary program search (genetic programming, island models)
- LLM-as-mutation-operator prompting strategies
- Reward hacking / evaluator robustness in open-ended search

## Next steps
- [ ] Read the AlphaEvolve paper and any public reproductions
- [ ] Pick a small, cheaply-scored target problem
- [ ] Implement a minimal LLM-mutation + selection loop
- [ ] Compare against a random-mutation baseline at matched compute budget

## Implementation plan

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References
- Novikov, A., et al. (2025). _AlphaEvolve: A coding agent for scientific and
  algorithmic discovery_ (DeepMind)
