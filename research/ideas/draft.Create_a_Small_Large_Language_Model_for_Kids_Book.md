# Create a Small Large Language Model for Children's Stories

## Status
- **Status**: draft
- **Complete Specs**: 20%

## Core Idea
- Train a very small LM (~10M-50M parameters) restricted to the vocabulary and
  narrative structure a 3-4 year old would understand, and test how small a
  model can be while still producing coherent short stories
- Sibling ideas: [[draft.Create_a_Small_Large_Language_Model_for_Logic]] (math/logic
  domain) and [[draft.Create_a_Small_Large_Language_Model_for_Python]] (code domain)
  apply the same "restrict the domain to shrink the model" methodology to
  different vocabularies

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```text
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

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
- **Training corpus choice**: TinyStories (Eldan & Li), a synthetic corpus of
  GPT-3.5/4-generated short stories restricted to the vocabulary a 3-4 year
  old would know, versus public-domain children's books (Project Gutenberg)
  filtered by Flesch-Kincaid reading level, to test whether synthetic vs.
  real restricted-vocabulary text changes the size/coherence tradeoff
- **Comparison to LittleLearner/LittleCurriculum**: a related project that
  trains 0.6B-5B parameter models on an 88B-token K-5 Common Core corpus to
  study an "interpretable knowledge boundary" (acquired vs. elicited
  capabilities), a different goal from model-size minimization but the same
  curriculum-restricted-domain methodology

## Next steps
- [ ] Look for related research (TinyStories follow-ups, other
      constrained-domain LMs)
- [ ] Reproduce TinyStories baseline at small scale as a sanity check
- [ ] Design the model-size sweep experiment
- [ ] Break the problem down into phases and milestones

## Implementation plan

- Milestone 1: build training corpora and reproduce a baseline
  - Assemble the TinyStories corpus and a vocabulary-restricted tokenizer
    matching the target 3-4-year-old vocabulary
  - Reproduce a small TinyStories baseline model as a sanity check that the
    training pipeline yields coherent short stories
  - Build a parallel corpus of Project Gutenberg children's books filtered by
    Flesch-Kincaid reading level, for the synthetic-vs-real corpus ablation
  - This is the result: two comparable training corpora (synthetic
    TinyStories-style and filtered real books) and a working baseline model

- Milestone 2: run the model-size sweep
  - Train 1M/10M/50M/125M parameter models with a fixed architecture family
    on each corpus, at matched compute budget
  - Checkpoint periodically and track loss/perplexity curves per model size
  - This is the result: a size x corpus grid of trained checkpoints ready for
    coherence evaluation

- Milestone 3: evaluate coherence beyond perplexity
  - Build automatic checks: grammar-checker pass rate, a character
    name-consistency tracker, and a GPT-4-as-judge plot-consistency score
  - Run a small human eval to calibrate the automatic judge against human
    ratings of grammar, plot consistency, and causality
  - Plot coherence against parameter count to find the size threshold where
    name drift and plot inconsistency disappear
  - This is the result: a coherence-vs-size curve and the minimum parameter
    count for coherent multi-sentence story generation, per corpus

- Milestone 4: run the vocabulary ablation and cross-domain comparison
  - Sweep the allowed vocabulary size at fixed model size and measure the
    resulting coherence tradeoff
  - Compare the resulting scaling curve against the sibling Logic and Python
    domain ideas, if their results are available
  - This is the result: a vocabulary-size vs. required-model-size tradeoff
    curve, and a documented comparison against the sibling narrow-domain LMs

## References
- Eldan, R., & Li, Y. (2023). _TinyStories: How Small Can Language Models Be
  and Still Speak Coherent English?_
- LittleLearner project. _LittleCurriculum: an 88B-token K-5 Common Core
  corpus for studying knowledge acquisition boundaries._
  https://littlelearner-ll.github.io
