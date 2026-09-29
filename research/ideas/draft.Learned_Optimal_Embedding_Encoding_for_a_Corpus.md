# Learned Optimal Embedding Encoding for a Corpus

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- Learn the optimal embedding encoding for a corpus jointly with the
  model weights, and measure how much smaller the resulting network can
  be

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

1. Are the learned embeddings similar to standard (pretrained) embeddings?

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

- Milestone 1: establish the baseline
  - Pick a corpus and a compact model architecture (e.g. a small
    transformer LM) with a standard embedding table
  - Train the baseline and record its parameter count (embedding table plus
    the rest of the network) and task performance (e.g. perplexity)
  - This is the result: reference numbers (embedding parameters, total
    parameters, perplexity) for the standard-embedding baseline

- Milestone 2: learn the embedding encoding jointly with the weights
  - Design a compressed, learnable embedding encoding (e.g. a shared
    codebook, a low-rank factorization, or a hash-based encoding)
  - Train the encoding jointly with the model weights using a combined loss
    (task loss plus a regularizer on encoding size)
  - This is the result: a trained model variant with its embedding
    parameter count reported alongside task performance, at a size matched
    or smaller than the baseline

- Milestone 3: sweep the size/performance tradeoff
  - Vary the encoding budget (codebook size, rank, or hash width) and
    retrain across the range
  - This is the result: a Pareto curve of embedding parameter count vs
    perplexity, identifying the smallest encoding that matches baseline
    performance

- Milestone 4: compare learned embeddings to standard ones
  - Compare nearest-neighbor structure of the learned embeddings against
    standard pretrained embeddings (e.g. GloVe or the baseline's own table)
  - This is the result: a quantitative similarity comparison (e.g. neighbor
    overlap or correlation of pairwise-similarity matrices) that answers the
    Questions section's question directly

## References
- Author(s), _Title_. (Year)
