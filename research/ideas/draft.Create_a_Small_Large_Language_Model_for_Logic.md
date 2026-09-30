# Create a Small Large Language Model for Math and Logic

## Status

- **Status:**: draft
- **Complete Specs:**: 20%

## Core Idea

- Train a very small LM restricted to grade-school arithmetic, word problems, and
  elementary formal logic, and test how small a model can be while still reliably
  solving multi-step reasoning problems in that narrow domain
- Sibling ideas: [[draft.Create_a_Small_Large_Language_Model_for_Kids_Book]]
  (natural-language domain) and
  [[draft.Create_a_Small_Large_Language_Model_for_Python]] (code domain) apply the
  same "restrict the domain to shrink the model" methodology to different
  vocabularies
- Unlike the story-generation variant, success here is checkable automatically (is
  the arithmetic/logic answer correct?), which makes this domain a cleaner testbed
  for measuring reasoning capability vs. parameter count

## Formalization

### Training Data

- **TinyGSM / GSM8K-style synthetic problems**: grade-school arithmetic and word
  problems, easy to generate synthetically at any scale and difficulty
- **DeepMind Mathematics Dataset**: procedurally generated (algebra, arithmetic,
  calculus, probability), so difficulty and size are controlled precisely
- **Formal logic corpora**: propositional/first-order logic proof steps (e.g.,
  generated via a SAT/SMT solver or a proof assistant like Lean/Coq), to extend
  beyond arithmetic into symbolic deduction

## Key Examples

- **Model size sweep**: train 1M/10M/50M/125M parameter models on the same
  GSM8K-style corpus; measure at what size multi-step arithmetic accuracy emerges
  (vs. single-step)
- **Curriculum ablation**: compare training on a fixed difficulty distribution vs. a
  curriculum (easy -> hard) at matched parameter count
- **Failure mode**: small models get individual arithmetic steps right but lose track
  of the overall problem state across multi-step word problems

## Questions

1. What is the minimum parameter count for reliable 2-3 step arithmetic reasoning,
   and how does it compare to the parameter count needed for story coherence in the
   Kids_Book variant?
2. Does exposure to formal logic proofs improve arithmetic word-problem accuracy
   (transfer across sub-domains), or are they independent skills?
3. Is correctness (checkable) a more sample-efficient training signal than next-token
   prediction alone (i.e., does RL/verifier-guided training beat pure supervised
   pretraining at this scale)?

## Research Topics

- Scaling laws for narrow-domain reasoning LMs
- Verifier-guided or RL fine-tuning at small scale (use exact-match correctness as
  reward)
- Comparison against the Kids_Book and Python variants: does "restrict domain to
  shrink model" hold equally well for language, math, and code?

## Next steps

- [ ] Look for related research (TinyGSM, phi-1/phi-2 "textbooks are all you need"
      line of work, small-model math reasoning)
- [ ] Reproduce a GSM8K-style baseline at small scale as a sanity check
- [ ] Design the model-size sweep experiment with automatic correctness scoring
- [ ] Break the problem down into phases and milestones

## Implementation plan

- Milestone 1: data and baseline
  - Generate GSM8K-style and DeepMind Mathematics problems with difficulty
    controlled by the number of reasoning steps
  - Write the exact-match scorer for arithmetic answers
  - Train the smallest models (1M and 10M parameters) as a sanity-check
    baseline
  - This is the result: single-step and multi-step accuracy for the smallest
    models, and a validated scoring pipeline

- Milestone 2: size sweep and curriculum
  - Train 1M/10M/50M/125M parameter models on the same corpus, with and
    without an easy-to-hard curriculum
  - Add formal-logic proof steps to the corpus and measure the transfer to
    arithmetic word problems
  - This is the result: accuracy vs parameter count curves, the minimum size
    for reliable 2-3 step reasoning, and the curriculum and transfer effects

- Milestone 3: verifier-guided fine-tuning
  - Fine-tune the best small model with exact-match correctness as the reward
  - Compare it with supervised pretraining at the same compute
  - This is the result: evidence on whether a checkable signal is more
    sample-efficient than next-token prediction alone

## References

- Cobbe et al., _Training Verifiers to Solve Math Word Problems_, GSM8K. (2021)
- Saxton et al., _Analysing Mathematical Reasoning Abilities of Neural Models_,
  DeepMind Mathematics Dataset. (2019)
