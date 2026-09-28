# Compression as a Proxy for Understanding

## Status

- **Status**: draft
- **Complete Specs**: 30%
- **Assignee**: TBD

## Core Idea

- The MDL and Kolmogorov view says understanding is the ability to compress:
  a model that grasps the structure of an artifact needs fewer bits to
  describe it
- Test this operationally for codebases and documents: use an LLM as the
  probability model inside an arithmetic coder, measure bits per token on a
  held-out file, and check whether that number predicts downstream competence
  on the same artifact (bug localization, refactor correctness, question
  answering)
- Raw compression is confounded by memorization, so the quantity of interest
  is the conditional gain: how many bits the rest of a repository saves when
  compressing a target file
  - This isolates the value of context from the value of prior exposure
- If the correlation holds, it gives a label-free metric with two uses:
  - Estimating whether a model or agent understands a specific codebase
    before letting it act
  - Selecting what to put in the context window by an MDL objective rather
    than by embedding similarity

## Formalization

- The code length of a token sequence $x$ under model $p_\theta$ is:
  $$
  L(x) = -\sum_{t} \log_2 p_\theta(x_t \mid x_{<t})
  $$
- The context gain from conditioning on a context set $C$ is:
  $$
  G(x \mid C) = L(x) - L(x \mid C)
  $$
  with the normalized form $g = G(x \mid C) / |x|$ in bits per token
- **Hypothesis**: downstream accuracy on tasks about $x$ is monotone in $g$,
  and $g$ is a better predictor than retrieval similarity scores
- Context selection becomes an MDL problem under a token budget $B$:
  $$
  C^{*} = \arg\max_{C : |C| \leq B} \; G(x \mid C)
  $$
- Memorization is controlled by comparing $L(x)$ for artifacts inside and
  outside the training window, and by applying semantics-preserving
  transformations that break surface memorization

## Key Examples

- **Context packing**: two retrieval strategies fill the same token budget;
  the one with higher context gain should give better bug localization, which
  is a directly falsifiable prediction
- **Memorization vs understanding**: a model compresses a popular open source
  repository extremely well, but after systematic renaming and refactoring the
  compression advantage disappears while a truly structural understanding
  should largely survive
- **Value of documentation**: measure the bits saved by adding a README or an
  architecture decision record to the context versus adding the equivalent
  number of tokens of raw code
  - This turns "is this documentation useful" into a measured number, feeding
    `draft.AI_Knowledge_Management_Development.md`
- **Failure mode**: highly repetitive boilerplate compresses well and predicts
  nothing about competence, so the metric must be normalized against a
  syntactic baseline compressor

## Questions

1. Does context gain predict task accuracy better than embedding similarity,
   BM25, or graph-based retrieval at a fixed token budget?
2. Is the relation between bits saved and accuracy smooth, or is there a
   threshold below which extra context does nothing?
3. Does context gain survive alpha-renaming and refactoring, i.e., does it
   measure understanding rather than recall?
4. Can an MDL context-packing objective be computed cheaply enough (small
   proxy model scoring, cached log-probabilities) to run inside an agent loop?
5. If compression predicts competence, does the converse hold: can a model be
   improved on a repository by explicitly training it to compress that
   repository?

## Research Topics

- **Measurement stack**: implement LLM-based arithmetic coding, or use
  log-probability sums directly, with careful tokenization and normalization
- **Contamination-controlled corpora**: private repositories, post-cutoff
  commits, and obfuscation transforms that preserve semantics
- **Correlation study**: pair compression measurements with downstream tasks
  on the same artifact and estimate predictive power against baselines
- **MDL retrieval**: greedy and submodular selection for the context budget
  problem, compared against standard retrieval-augmented generation
- **Cheap surrogates**: check whether a small model's context gain predicts a
  large model's competence, which would make the metric practical
- **Theory link**: relate the measured gain to the description length terms in
  `draft.MDL_Extensions_with_Research_Process.md` and to the drift signal in
  `draft.Kolmogorov_Complexity_Over_Time.md`

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build the measurement tool
  - Compute $L(x)$ and $L(x \mid C)$ for files in a repository using open
    models with accessible log-probabilities
  - Calibrate against `gzip` and a syntactic baseline to remove trivial
    redundancy effects
  - This is the result: a tool that reports context gain in bits per token for
    any (file, context) pair

- Milestone 2: correlation with competence
  - Build a task set on the same repositories: bug localization, patch
    correctness, and factual questions with known answers
  - Measure the correlation between context gain and task accuracy, against
    retrieval-similarity baselines
  - This is the result: evidence for or against compression as a competence
    proxy

- Milestone 3: memorization controls
  - Repeat on post-cutoff and privately held repositories, and on
    semantics-preserving obfuscations
  - This is the result: a decomposition of compression into recall and
    structure

- Milestone 4: MDL-based context packing
  - Implement greedy selection maximizing context gain under a token budget
    and compare downstream accuracy against standard retrieval
  - This is the result: a practical context-selection method with measured
    gains at equal cost

## References

- Deletang et al., _Language Modeling Is Compression_ (2023)
- Rissanen, _Modeling by Shortest Data Description_ (1978)
- Li and Vitanyi, _An Introduction to Kolmogorov Complexity and Its
  Applications_ (2008)
- `draft.Kolmogorov_Complexity_Over_Time.md`
- `draft.MDL_Extensions_with_Research_Process.md`
- `draft.AI_Knowledge_Management_Development.md`
