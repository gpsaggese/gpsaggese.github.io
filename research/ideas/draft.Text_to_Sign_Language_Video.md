# Text-to-Sign-Language Video with a Chosen Signer's Face

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: TBD

## Core Idea

- Given a text input, generate a video of a person signing it: text is
  translated into a sign-language gloss sequence, the gloss sequence drives a
  3D pose/skeleton track, and a rendering model turns the pose track into a
  photorealistic video with a chosen person's face and appearance
- The pipeline is deliberately factored into translation, pose synthesis, and
  rendering, so each stage can be evaluated on its own: end-to-end
  text-to-video hides which stage is failing
- The non-obvious part is that sign languages are not word-for-word encodings
  of spoken language: ASL has its own grammar, and much of the meaning is in
  non-manual markers (facial expression, eyebrow position, mouth morphemes),
  so a hand-only avatar is not merely lower quality, it is often
  unintelligible
- The identity/consent question is part of the research, not an afterthought:
  rendering an arbitrary person's face is a deepfake capability, so the
  design should assume an enrolled, consenting signer identity

## Formalization

- Stage 1, translation: `text -> gloss` sequence (a low-resource machine
  translation problem, with grammar reordering, not a lookup)
- Stage 2, pose synthesis: `gloss -> {p_t}` where `p_t` is a per-frame body,
  hand, and face keypoint set
- Stage 3, rendering: `{p_t} + identity -> video`
- Evaluation: BLEU-style scores on gloss translation, keypoint error (or DTW
  distance) on pose, and comprehension accuracy by fluent Deaf signers on
  held-out sentences (the only measure that matters)

## Key Examples

- **Announcement signing**: a fixed-domain corpus (weather, transit
  announcements) where vocabulary is bounded and a first system can actually
  reach intelligibility
- **Face-driven meaning**: two sentences with identical manual signs but
  different eyebrow/head position (statement vs. question). A pose-only
  model that ignores non-manual markers collapses them into one output
- **Failure mode**: fluent-looking but semantically wrong output, where the
  video is smooth and confident and the sentence means something else: the
  worst case for an accessibility tool
- **Failure mode**: rendering quality is judged by hearing evaluators who
  cannot read sign language, so the metric rewards realism instead of
  intelligibility

## Questions

1. How much of intelligibility comes from the non-manual channel, measured by
   ablating facial expression from an otherwise fixed pose track?
2. Is the factored pipeline better than end-to-end text-to-video at the data
   scales available for sign language, which are tiny compared to speech?
3. What consent and provenance mechanism should gate the identity-rendering
   stage, given the same model is a deepfake generator?

## Research Topics

- **Translation datasets**: sign language translation datasets
  (RWTH-PHOENIX-Weather 2014T, How2Sign) and their size/domain limits
- **Pose-to-video rendering**: pose-conditioned diffusion, neural avatars,
  face reenactment
- **Evaluation with Deaf signers**: automatic metrics are known to correlate
  poorly with comprehension
- **Provenance/watermarking**: watermarking of generated video, linking to
  the detection side of
  [draft.Detecting_and_Humanizing_LLM_Text](draft.Detecting_and_Humanizing_LLM_Text.md)

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Look for related research (SignGPT-style translation, Text2Sign, avatar
  rendering literature)
- Reproduce a gloss-translation baseline on PHOENIX-14T
- Get a pose-to-video renderer working on one enrolled identity
- Design the human comprehension evaluation with fluent signers

## References

- Camgoz, N. C., et al., _Neural Sign Language Translation_. (2018)
- Saunders, B., et al., _Progressive Transformers for End-to-End Sign
  Language Production_. (2020)
- Duarte, A., et al., _How2Sign: A Large-Scale Multimodal Dataset for
  Continuous American Sign Language_. (2021)
