# Create a Small Large Language Model for Children's Stories

## Status
- **Status**: draft
- **Complete Specs**: 20%
- **Assignee**: TBD

## Core Idea
- Train a very small LM (~10M-50M parameters) restricted to the vocabulary and
  narrative structure a 3-4 year old would understand, and test how small a
  model can be while still producing coherent short stories
- Sibling ideas: [[draft.Create_a_Small_Large_Language_Model_for_Logic]] (math/logic
  domain) and [[draft.Create_a_Small_Large_Language_Model_for_Python]] (code domain)
  apply the same "restrict the domain to shrink the model" methodology to
  different vocabularies

## Training Data
- **TinyStories** (Eldan & Li, Microsoft Research) — the canonical dataset for
  this: GPT-3.5/4-generated short stories using only words a 3-4 year old would
  know. Showed ~10M-parameter models can produce coherent text when trained on
  this restricted distribution
  - Paper: "TinyStories: How Small Can Language Models Be and Still Speak
    Coherent English?"
- Optionally augment with public-domain children's books (Project Gutenberg
  children's collection) filtered by Flesch-Kincaid reading level, to test
  whether real (vs. synthetic) restricted-vocabulary text changes the
  size/coherence tradeoff
- **LittleLearner / LittleCurriculum** ([littlelearner-ll.github.io](https://littlelearner-ll.github.io))
  — related prior work: trains 0.6B-5B param models on an 88B-token corpus
  filtered to K-5 Common Core standards, to get an "interpretable knowledge
  boundary" for studying acquired vs. elicited capabilities. Different goal
  (knowledge-boundary study, not model-size minimization) but same
  curriculum-restricted-domain methodology

## Key Examples
- **Model size sweep**: train 1M/10M/50M/125M parameter models on the same
  TinyStories corpus; measure at what size grammatical coherence, plot
  consistency, and simple causality ("the cat was hungry, so it ate") emerge
- **Vocabulary ablation**: shrink/grow the allowed vocabulary size and observe
  how it trades off against required model size for the same coherence bar
- **Failure mode**: models below some threshold produce locally fluent but
  globally inconsistent stories (character identity/name drift mid-story)

## Questions
1. What is the minimum parameter count for coherent multi-sentence
   children's-story generation, and how does it scale with vocabulary size?
2. Does restricting the domain (vocabulary + narrative simplicity) buy more
   parameter efficiency than restricting sequence length or training-set size?
3. Do the scaling laws found here (small vocab -> small model) transfer to the
   Logic and Python domain variants, or is language uniquely compressible?

## Research Topics
- Scaling laws for narrow-domain LMs (compare against Chinchilla-style scaling
  for general-domain LMs)
- Evaluation of "coherence" beyond perplexity (grammar checkers, GPT-4-as-judge,
  human eval on plot consistency)
- Synthetic-data generation pipelines (prompting a large model to produce
  vocabulary-constrained training data)

## Next steps
- [ ] Look for related research (TinyStories follow-ups, other constrained-domain LMs)
- [ ] Reproduce TinyStories baseline at small scale as a sanity check
- [ ] Design the model-size sweep experiment
- [ ] Break the problem down into phases and milestones

## References
- Eldan, R., & Li, Y. (2023). _TinyStories: How Small Can Language Models Be
  and Still Speak Coherent English?_
- LittleLearner project. _LittleCurriculum: an 88B-token K-5 Common Core
  corpus for studying knowledge acquisition boundaries._
  https://littlelearner-ll.github.io
