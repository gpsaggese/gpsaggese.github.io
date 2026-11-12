# System for Cross-Posting and Measuring Blog Engagement

## Status
- **Status**: draft
- **Complete Specs**: 10%

## Core Idea
- Build a pipeline that takes one source post (Markdown in the repo) and
  publishes adapted versions across channels (LinkedIn, Substack, an MkDocs
  website, X), each with the format and length that channel rewards, then
  collects the engagement metrics back into one place
- The system side is plumbing (one source of truth, per-channel adapters,
  scheduled publishing); the research side is the feedback loop: with
  per-channel engagement logged against post features, one can ask what actually
  drives engagement rather than guessing
- The honest difficulty is causal, not technical: engagement is confounded by
  posting time, follower growth, topic, and platform algorithm changes, so
  naive A/B comparisons of "did the LLM-adapted version do better" are almost
  always wrong

## Formalization
- Source post `s`, per-channel adapter `a_c(s) -> post_c`, outcome `y_c` =
  impressions / reactions / clicks / subscriber delta
- Naive model: `y_c ~ f(features(post_c), time, follower_count)`
- Better: randomize the manipulable factor (e.g., headline variant, posting
  hour) so the comparison is an experiment, not an observational fit
- Minimum bar: report effect sizes with confidence intervals, and state the
  number of posts needed for the effect to be detectable at all: with a
  handful of posts per month, most claims are underpowered

## Key Examples
- **Format adaptation**: one technical article becomes a long-form Substack
  piece, a short LinkedIn post with a hook, and a docs page: same content,
  three shapes
- **Headline A/B**: two headline variants for the same post, randomized, with
  click-through as the outcome
- **Cross-channel funnel**: measure whether LinkedIn posts actually move
  Substack subscriptions, or whether the channels are independent audiences
- **Failure mode**: the pipeline optimizes for engagement and drifts toward
  clickbait, degrading the thing being promoted, so content quality has to be
  a constraint, not a free variable

## Questions
1. Which factors are actually manipulable and randomizable (headline, format,
   time) vs. merely observable (topic, audience), and is there enough post
   volume to detect any of them?
2. Does cross-posting cannibalize a channel (same reader, lower engagement per
   channel) or expand reach?
3. Can an LLM adapter match a hand-written per-channel version, judged blind by
   readers rather than by engagement metrics?

## Research Topics
- Channel APIs and their limits (LinkedIn and Substack posting/analytics access
  is restrictive and partly manual)
- One-source-many-targets content pipeline built on the repo's existing
  `.claude/skills/blog.*` skills
- Experiment design under low sample size: sequential testing, blocking on
  posting time
- Engagement metric definition: which outcome is the real goal (reach,
  subscribers, inbound contacts)

## Next steps
- [ ] Inventory each channel's publishing and analytics API, and what requires
      manual steps
- [ ] Build the source-of-truth Markdown format plus two channel adapters
- [ ] Add metric collection into a single table keyed by post and channel
- [ ] Run one randomized headline experiment and report whether it is powered
      enough to conclude anything

## References
- Existing repo skills: `.claude/skills/blog.create_from_notes`,
  `.claude/skills/blog.add_links`, `.claude/skills/blog.check_format`
