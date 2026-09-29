# Visualizing Model Weights and Gradients to Find Compressible Parts

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- Visualize weights from models
- Visualize the magnitude of gradients to understand which parts of the
  network are important or can be compressed
- Use models from `huggingface`

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

- Milestone 1: build a weight and gradient extraction and visualization
  pipeline
  - Pick 1-2 small pretrained models from `huggingface` (e.g., `distilbert`,
    `gpt2`) as the initial testbed
  - Extract per-layer weight tensors and per-layer gradient tensors from a
    backward pass on a small calibration dataset
  - Build visualizations: per-layer weight-magnitude heatmaps,
    weight-value histograms, and gradient-magnitude heatmaps
  - This is the result: a script or notebook that, given a `huggingface`
    checkpoint and a calibration batch, renders weight and gradient
    visualizations for every layer

- Milestone 2: define a compressibility score and rank model components
  - Define a saliency score combining weight magnitude and gradient
    magnitude per parameter, in the spirit of Optimal Brain Damage /
    movement pruning
  - Aggregate the score at multiple granularities: individual weight,
    neuron or channel, attention head, and full layer
  - Rank components by score and visualize the ranking as a heatmap or
    sorted bar chart overlaid on the model architecture
  - This is the result: a per-model, per-granularity compressibility map
    that flags the lowest-scoring (most removable) components

- Milestone 3: validate the score with real pruning experiments
  - Prune or zero out the lowest-scoring components at increasing
    compression ratios (e.g., 10%, 30%, 50%, 70%)
  - Measure downstream task performance (accuracy for classification,
    perplexity for language modeling) at each ratio
  - Compare against baselines: random pruning and magnitude-only pruning
    (no gradient term)
  - This is the result: a compression-ratio-vs-performance-drop curve
    showing whether the gradient-informed score removes components with
    less performance loss than the baselines

- Milestone 4: extend across architectures and build an interactive view
  - Apply the pipeline to additional `huggingface` architectures (e.g., an
    encoder-only, a decoder-only, and a vision transformer model)
  - Build an interactive dashboard (notebook widget or small web app) to
    browse weight, gradient, and compressibility maps across models and
    layers
  - This is the result: a reusable cross-architecture tool, plus a
    comparison of where compressible parts concentrate across
    architectures

## References
- Author(s), _Title_. (Year)
