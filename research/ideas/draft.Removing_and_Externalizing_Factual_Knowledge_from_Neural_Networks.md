# Removing and Externalizing Factual Knowledge from Neural Networks

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- Investigate how to remove information from a neural network
- Proposal: remove all facts from the network and replace them with a
  pluggable, external memory for facts
  - E.g., use a separate LLM purely for factual lookup

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

- Milestone 1: build a factual-knowledge probe and localize facts
  - Assemble a factual QA benchmark of subject-relation-object triples
    (e.g., LAMA/ParaRel-style, or a small synthetic fact set) to probe what
    a base model knows
  - Measure the base model's factual-recall accuracy on the probe
  - Apply an existing knowledge-localization/editing technique (e.g., ROME
    or MEMIT) to identify which weights encode which facts
  - This is the result: a baseline factual-recall accuracy plus a
    localization map linking specific facts to specific weights/layers

- Milestone 2: remove facts from the network
  - Apply model-editing/unlearning techniques to erase a target subset of
    facts from the localized weights
  - Measure the factual-recall drop on the removed facts (should fall to
    near-chance) and check for collateral damage: perplexity/accuracy on
    unrelated facts and general language tasks
  - This is the result: a "fact-stripped" model checkpoint with quantified
    removal effectiveness and collateral-damage measurements

- Milestone 3: build an external fact-lookup module
  - Implement an external memory store (key-value lookup or a small
    retrieval index) holding the removed facts, or route factual queries to
    a separate LLM dedicated purely to factual lookup
  - Build the interception/routing logic that detects a factual query and
    answers from the external store instead of the fact-stripped internal
    weights
  - This is the result: a working hybrid pipeline (fact-stripped base model
    plus external memory) that answers the Milestone 1 probe questions

- Milestone 4: evaluate the hybrid system end to end
  - Compare factual accuracy, inference latency, and answer quality of the
    hybrid system against the original, unmodified model
  - Test updatability specifically: measure the cost of editing a fact in
    the external store vs. the cost of re-finetuning/re-editing the
    original network for the same fact change
  - This is the result: a comparison table of accuracy, latency, and
    fact-update cost for the hybrid system vs. the unmodified baseline

## References

- Author(s), _Title_. (Year)
