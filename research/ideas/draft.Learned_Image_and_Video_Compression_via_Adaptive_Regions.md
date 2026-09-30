# Learned Image and Video Compression via Adaptive Regions

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

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

- Milestone 1: differentiable region-based image representation
  - Parameterize an image as a set of variable-size rectangles (position,
    size, color/pattern), with a differentiable rasterizer
  - Pose region approximation as a loss (reconstruction error plus a rate
    penalty on the number of regions) optimized directly by gradient
    descent, per Formalization
  - This is the result: a prototype that reconstructs a single test image
    from a learned set of rectangles, with a visible quality/region-count
    tradeoff

- Milestone 2: amortize with a learned encoder
  - Train a neural network that predicts the region decomposition directly
    from an input image, replacing the per-image optimization of Milestone 1
  - Train the encoder-decoder end to end on a standard image dataset (e.g.
    Kodak or CLIC)
  - This is the result: a trained encoder-decoder that compresses and
    decompresses held-out images without per-image optimization

- Milestone 3: benchmark region-based vs pixel-based compression
  - Compare rate-distortion curves (bits per pixel vs PSNR/MS-SSIM) of the
    region-based codec against a pixel-based learned codec and a classical
    codec (e.g. JPEG or WebP)
  - This is the result: a rate-distortion comparison directly answering the
    Questions section's question of region-based vs pixel-based performance

- Milestone 4: extend to video
  - Add temporal prediction by tracking or warping regions across frames
    instead of re-encoding each frame independently
  - Evaluate on a small video benchmark against a per-frame (image-only)
    baseline
  - This is the result: a measurement of whether temporal region-reuse
    improves compression over the per-frame baseline

## References
- Author(s), _Title_. (Year)
