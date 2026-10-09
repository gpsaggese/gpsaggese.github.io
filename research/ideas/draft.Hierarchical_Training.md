# Hierarchical Training

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- Train a neural network hierarchically: project the loss to each block,
  optimize each block, then do a single pass to connect all of them

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

- Milestone 1: define block partitioning and a per-block local loss
  - Choose a baseline architecture (e.g., a ResNet or Transformer) and
    partition it into sequential blocks
  - Define a local "projected loss" per block, e.g., an auxiliary head
    attached to each block's output, so a block can be optimized without
    gradients from downstream blocks
  - Implement the mechanism that feeds each block its input: the frozen
    previous block's forward output during block-local training
  - This is the result: an implemented per-block local loss and training
    loop that can train one block at a time

- Milestone 2: implement the full hierarchical procedure and benchmark it
  - Implement the sequential pipeline: train each block on its local
    loss, freeze it, then move to the next block
  - Implement the final joining pass: a single pass of end-to-end
    fine-tuning that connects all pretrained blocks on the true task loss
  - Train both the hierarchical procedure and a standard end-to-end
    backprop baseline on the same architecture and a common benchmark
    (e.g., image classification)
  - This is the result: matched accuracy/loss curves for hierarchical
    training vs. end-to-end backprop on the same benchmark

- Milestone 3: measure compute and memory efficiency
  - Measure peak memory and wall-clock/FLOPs for the hierarchical
    procedure vs. end-to-end backprop, since block-local training can
    avoid storing full-depth activations
  - Vary the number and depth of blocks to test how any accuracy gap to
    end-to-end training scales with hierarchy granularity
  - This is the result: a compute/memory efficiency comparison and a plot
    of accuracy gap vs. number of blocks

- Milestone 4: test block-parallel training enabled by the local losses
  - Train different blocks in parallel on separate workers using only
    their local losses, synchronizing only for the final joining pass
  - Measure the wall-clock speedup from block-parallel training against
    the sequential hierarchical and end-to-end baselines
  - This is the result: a measured wall-clock speedup (or lack of one)
    from parallelizing block-local training ahead of the final joining
    pass

## References

- Source discussion: https://claude.ai/share/000f840b-6a9f-4eb9-a5e9-3de6419bfe86
