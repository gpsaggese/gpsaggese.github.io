# Learned Confidence Scoring for Model Reasoning

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- Train a model that scores a machine's confidence in its own reasoning

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

- Milestone 1: build a labeled reasoning-trace dataset
  - Collect chain-of-thought traces from an LLM on a mix of QA and math
    benchmarks (e.g. GSM8K, a multiple-choice QA set), varying difficulty
  - Label each trace's final answer as correct or incorrect against ground
    truth
  - This is the result: a dataset of (question, reasoning trace, answer,
    correctness label) tuples spanning multiple task types

- Milestone 2: train the confidence-scoring model
  - Train a scorer that takes the reasoning trace (and question) as input
    and predicts the probability the final answer is correct, e.g. a probe
    on the model's hidden states or a classifier on the trace text
  - Compare it against baselines: verbalized self-reported confidence, and
    token-logprob-based confidence
  - This is the result: a trained scorer with calibration metrics (e.g.
    expected calibration error, AUROC for correct vs incorrect) that beats
    both baselines on a held-out split

- Milestone 3: validate generalization and downstream utility
  - Test the scorer's calibration when trained on one task type (e.g. math)
    and evaluated on another (e.g. QA)
  - Use the confidence score to drive an abstention policy (skip or escalate
    low-confidence answers) and measure the resulting selective accuracy
  - This is the result: a selective-accuracy curve showing whether
    abstaining on low-confidence answers raises accuracy on the remainder

- Milestone 4: analyze what drives the score
  - Correlate the scorer's output against interpretable trace features
    (trace length, hedging language, self-consistency across resamples)
  - This is the result: an analysis identifying which trace features the
    scorer relies on most, and whether they match human intuitions about
    reasoning quality

## References
- Author(s), _Title_. (Year)
