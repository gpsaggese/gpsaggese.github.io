# Progressive Growing of Network Resolution and Structure

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- Learn a neural network using an approach similar to progressively
  increasing resolution for images
  - Change the number of gradient dimensions as training goes on
- Learn a neural network while also changing its structure as training goes
  on
  - Find the "optimal" number of parameters given the corpus
  - Make the architecture differentiable

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

- Milestone 1: build the fixed-architecture baseline
  - Pick a testbed domain where "resolution" has a clear meaning (e.g.,
    image classification/generation, analogous to Progressive GAN) and a
    fixed model architecture and input resolution as the control
  - Train the baseline to convergence and record final quality/accuracy,
    wall-clock training time, and parameter count
  - This is the result: baseline compute and quality numbers to compare all
    later growing variants against

- Milestone 2: implement progressive resolution growth
  - Design a schedule that increases the number of gradient dimensions
    (input/output resolution) over training steps, doubling or stepping up
    at fixed intervals
  - Implement a smooth blending (fade-in) between resolution stages so newly
    added dimensions do not destabilize already-trained weights
  - Train under the progressive schedule and compare final quality and
    wall-clock time against the Milestone 1 baseline
  - This is the result: a quantified compute/quality tradeoff of
    progressive-resolution training vs. the fixed-resolution baseline

- Milestone 3: implement differentiable structure growth
  - Parameterize the network's width/depth as differentiable quantities
    (e.g., learnable gates on channels/layers, DARTS-style relaxation) so
    the architecture itself can grow or shrink during training
  - Add a parameter-count regularization term to the loss so training
    searches for the "optimal" number of parameters for the corpus
  - Track parameter count and validation performance across training steps
  - This is the result: a differentiable-architecture training run whose
    parameter count converges to a steady state, with a performance vs.
    parameter-count curve

- Milestone 4: combine both mechanisms and evaluate
  - Train a model with progressive resolution growth and differentiable
    structure growth enabled simultaneously
  - Compare final architecture size, total training compute, and quality
    against the baseline and against each mechanism used alone
  - This is the result: an ablation table isolating the contribution of
    resolution growth vs. structure growth vs. the combined approach

## References

- Author(s), _Title_. (Year)
