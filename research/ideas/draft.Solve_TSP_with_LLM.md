# Solve the Traveling Salesman Problem with an LLM

## Status
- **Status:**: draft
- **Complete Specs:**: 15%
- **Assignee:**: TBD

## Core Idea
- Test whether an LLM (prompted directly, or fine-tuned on
  problem/solution pairs) can propose useful heuristics or construction
  moves for TSP instances, with a deliberately modest bar: **better than
  random**, not competitive with established solvers (Concorde, LKH)
- The interesting question isn't "can an LLM solve TSP optimally" (it can't,
  at any interesting scale) but **what kind of combinatorial-structure signal
  can an LLM extract from problem instances without a hand-coded algorithm**
- Related to `draft.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.md` and
  `draft.LLM_for_Symbolic_Regression.md`: same family of "can an LLM replace
  or augment a classical search/optimization algorithm" questions

## Formalization
- Baseline comparisons at instance size `n`:
  - Random tour: expected tour length `L_random`
  - Nearest-neighbor construction heuristic: `L_nn`
  - LLM-proposed tour or LLM-proposed next-city-to-visit policy: `L_llm`
- Success criterion: `L_llm < L_random` reliably, and ideally
  `L_llm` approaches `L_nn` as a first milestone before comparing against
  stronger heuristics (2-opt, Christofides)

## Key Examples
- **Direct prompting**: give the LLM city coordinates as text, ask for a
  visiting order; measure tour length across many random instances at small
  `n` (e.g., n=10-20)
- **Policy framing**: at each step, ask the LLM to pick the next city given
  current position and remaining cities (greedy-style), compare against
  nearest-neighbor greedy
- **Fine-tuned variant**: fine-tune on (instance, near-optimal tour) pairs
  from a solver, then test generalization to held-out instance sizes/
  distributions

## Questions
1. At what instance size does direct LLM prompting stop beating random (i.e.,
   where's the breakdown point as `n` grows)?
2. Does a fine-tuned LLM learn generalizable spatial-reasoning heuristics, or
   does it just memorize patterns from the training distribution of
   instances?
3. Is there value in using the LLM as a _heuristic proposer_ feeding into a
   classical local-search improver (2-opt), rather than as the end-to-end
   solver?

## Research Topics
- LLM spatial/combinatorial reasoning benchmarks
- Neural combinatorial optimization literature (pointer networks, graph
  neural nets for TSP) as a comparison point for what "learned heuristics"
  can achieve
- Hybrid LLM-proposer + classical-improver architectures

## Next steps
- [ ] Generate a benchmark set of small random TSP instances with known
  optimal/near-optimal tours
- [ ] Implement random and nearest-neighbor baselines
- [ ] Test direct LLM prompting at increasing instance sizes
- [ ] If promising, try the fine-tuned and hybrid variants

## References
- Vinyals, O., et al. (2015). _Pointer Networks_
- Kool, W., et al. (2019). _Attention, Learn to Solve Routing Problems!_
