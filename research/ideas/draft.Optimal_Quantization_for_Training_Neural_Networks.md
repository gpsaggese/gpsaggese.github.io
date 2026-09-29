# Optimal Quantization for Training Neural Networks

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- Determine the right amount of quantization to use when training a neural
  network
  - Question: is it better to train first and quantize after, or to
    quantize during training?
  - E.g., ternary weights (-1, 0, 1)
  - E.g., fp4 precision
- Explore networks whose layers use different levels of bit precision

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

- Milestone 1: build a quantization-aware training testbed
  - Train a small-to-medium reference network (e.g. a CNN on CIFAR-10 or
    an MLP on MNIST) at full precision to establish the baseline accuracy
  - Implement pluggable quantizers covering post-training quantization
    (PTQ), quantization-aware training (QAT) with a straight-through
    gradient estimator, ternary weights ($-1, 0, 1$), and simulated fp4/
    fp8 precision
  - This is the result: a testbed reproducing the full-precision baseline
    plus working PTQ and QAT pipelines for each quantization scheme

- Milestone 2: compare train-then-quantize versus quantize-during-training
  - Run PTQ (train at full precision, then quantize) and QAT (quantize
    during training) at matched bit budgets (ternary, fp4, fp8), keeping
    architecture, data, and optimizer identical across runs
  - Record final accuracy, training-loss stability, and convergence speed
    for each bit budget and strategy
  - This is the result: an accuracy-versus-bit-budget table showing which
    strategy wins at each bit width, answering the train-then-quantize
    versus quantize-during-training question from Core Idea

- Milestone 3: search per-layer mixed precision
  - Probe each layer's sensitivity to quantization (e.g. accuracy drop
    from quantizing that layer alone) to rank layers by tolerance
  - Use a greedy or sensitivity-guided search to allocate a total bit
    budget unevenly across layers (mixing ternary, fp4, and fp8 layers)
    and train the resulting mixed-precision network with QAT
  - This is the result: a mixed-precision configuration that matches or
    beats the best uniform-precision network from Milestone 2 at the same
    average bit budget

- Milestone 4: analyze what drives the optimal precision per layer
  - Correlate each layer's optimal bit width (from Milestone 3) against
    candidate predictors such as layer depth, weight variance, and
    gradient-magnitude sensitivity
  - This is the result: a predictive heuristic linking a measurable
    per-layer statistic to its optimal precision, validated by comparing
    the heuristic's suggested allocation against the searched one

## References

- Author(s), _Title_. (Year)
