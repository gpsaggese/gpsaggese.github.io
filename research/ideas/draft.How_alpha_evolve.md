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

- Milestone 1: build the evolutionary loop infrastructure
  - Implement population management (spawn, store, retire candidate
    programs) and a pluggable evaluator interface returning score $f(p)$
  - Implement selection with an island-population diversity mechanism to
    avoid premature convergence
  - Pick the toy target problem (e.g., a sorting network or an
    approximation constant for a known inequality) with a cheap,
    deterministic evaluator
  - This is the result: a working generation loop that evolves a
    population under a non-LLM (random) mutation operator on the toy
    problem

- Milestone 2: add the LLM mutation operator
  - Build the mutation prompt template: parent program, its score, and
    optional execution trace/error, producing an edited child program
  - Wire the LLM mutation operator into the Milestone 1 loop in place of
    random mutation
  - This is the result: an end-to-end LLM-mutation plus selection loop
    that improves the toy problem's best score over generations

- Milestone 3: compare against baselines and audit for reward hacking
  - Run LLM-mutation vs random-mutation at matched compute budget (same
    number of evaluator calls) and compare best-score convergence curves
  - Manually inspect top-scoring candidates each generation for evaluator
    exploits (reward hacking) rather than genuine improvement
  - This is the result: a convergence-curve comparison plus a documented
    list of any reward-hacking cases found and how the evaluator was
    tightened to close them

- Milestone 4: ablate feedback richness
  - Rerun the LLM-mutation loop with score-only feedback versus
    score-plus-execution-trace/error feedback in the mutation prompt
  - Measure convergence speed (generations or evaluator calls to reach a
    fixed score threshold) for each feedback condition
  - This is the result: an ablation showing whether trace/error feedback
    measurably speeds up convergence over score-only feedback

## References
- Novikov, A., et al. (2025). _AlphaEvolve: A coding agent for scientific and
  algorithmic discovery_ (DeepMind)
