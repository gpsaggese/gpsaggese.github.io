# Learned Image and Video Compression via Adaptive Regions

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- Compress images using a neural network that learns recurring patterns,
  since natural images share common characteristics
- Apply the same idea to video compression
- Explore compressing an image using variable-size regions (rectangles)
  instead of individual pixels
- Pose the region-approximation problem as a loss that gradient descent
  can optimize

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

1. Does compressing with variable-size regions instead of individual
   pixels outperform pixel-based approaches?

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

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References
- Author(s), _Title_. (Year)
