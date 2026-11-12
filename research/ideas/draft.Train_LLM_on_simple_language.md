# Training an LLM on a Compressed, Unambiguous Language

## Status

- **Status:**: draft
- **Complete Specs:**: 30%

## Core Idea

- Natural language is ambiguous and redundant, and an LLM spends capacity
  learning to resolve the ambiguity and to reproduce the redundancy
- Proposal: train an LLM on an intermediate language that is
  - Unambiguous: one surface form per meaning, no syntactic ambiguity
  - Non-redundant: function words carrying no information are dropped (e.g.,
    articles like "the", "a"), which is the "caveman" register already used
    for compressed communication in this repo
  - Then translate the model's output back into English as a final step
- Two claims are being tested at once and must be separated:
  - **Efficiency claim**: fewer tokens per unit of meaning implies more
    content per context window and cheaper training and inference
  - **Learnability claim**: a regular, low-ambiguity language is easier to
    learn, so a smaller model reaches the same task performance
- The non-obvious part is that these can pull in opposite directions
  - Redundancy in natural language is error-correcting
  - Removing it may make the model more brittle at exactly the points where
    the surface form was doing disambiguating work

## Formalization

- Let $E$ be English text and $S = \phi(E)$ its compressed form under an
  encoder $\phi$, with decoder $\psi$ mapping back to English
- Compression ratio in tokens under a fixed tokenizer:
  ```
  rho = |tokenize(S)| / |tokenize(E)|
  ```
- Round-trip fidelity, measured by semantic equivalence rather than exact
  match:
  ```
  fidelity = E_x [ sim(psi(phi(x)), x) ]
  ```
  - $sim$ is a semantic similarity or entailment-based score, plus a strict
    fact-preservation check
- The efficiency claim is only meaningful at fixed information content, so
  the comparison is done at equal bits, not equal tokens:
  ```
  bits_per_token(S) vs bits_per_token(E)
  ```
- Experimental design: train two models with the same architecture and the
  same compute budget
  - $M_E$ on English corpus $C$
  - $M_S$ on $\phi(C)$
  - Evaluate both on the same downstream tasks, decoding $M_S$ output through
    $\psi$ before scoring
- Ambiguity is the confound to control
  - If $\phi$ is lossy, $M_S$ is solving an easier problem and the comparison
    is invalid
  - Fidelity must be measured and reported before any capability claim

## Key Examples

- **Function-word deletion**: `the cat sat on the mat` maps to `cat sit mat`,
  roughly halving tokens while losing tense and definiteness
- **Ambiguity removal**: `I saw the man with the telescope` has two parses,
  and the compressed form must commit to one, which forces the encoder to
  disambiguate before the model ever sees the text
- **Existing controlled languages**: Attempto Controlled English and Basic
  English are prior attempts at a restricted, unambiguous subset, and give a
  baseline encoder that is not ad hoc
- **Failure mode**: compression that is not invertible, so the decoder
  hallucinates the dropped material (tense, plurality, negation scope) and
  fidelity collapses on exactly the semantically loaded cases
- **Failure mode**: a compressed corpus that no longer matches the tokenizer's
  training distribution, so token counts go up rather than down because words
  fragment into unfamiliar pieces

## Questions

1. Is the gain real once measured in bits rather than tokens, or is
   compression just moving work from the model into the encoder $\phi$?
2. Where does fidelity break? A prediction: negation, quantifier scope, and
   tense, i.e., exactly the places where English redundancy is load-bearing
3. At a fixed compute budget, does $M_S$ beat $M_E$ on downstream English
   tasks after decoding, or does the decode step give back the savings?
4. Does the model trained on the compressed language reason better, since the
   surface form is closer to a logical form? This connects to
   `draft.Create_a_Small_Large_Language_Model_for_Logic.md`
5. If a compressed register is strictly better for machine-to-machine text,
   agents should communicate in it and translate only at the human boundary

## Research Topics

- **Encoder design**: rule-based deletion (articles, copulas, inflection) vs a
  learned encoder vs an existing controlled language such as Attempto
  Controlled English
- **Tokenizer co-design**: training a tokenizer directly on the compressed
  corpus, since reusing an English tokenizer likely destroys the gain
- **Fidelity measurement**: round-trip semantic equivalence, with targeted
  probes for negation, tense, quantifiers, and coreference
- **Scaling comparison**: matched-compute training runs of $M_E$ and $M_S$ at
  small scale, e.g., a few hundred million parameters
- **Relation to compression-as-understanding**: this idea compresses the data,
  while `draft.Compression_as_Proxy_for_Understanding.md` uses compression as
  the measurement, so the two should share the same metrics

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build and measure the codec
  - Implement a rule-based encoder $\phi$ and a decoder $\psi$ (an LLM prompt
    is acceptable for $\psi$ at this stage)
  - Measure $\rho$ and round-trip fidelity on a held-out corpus, with the
    targeted probes for negation and tense
  - This is the result: a codec with a documented compression/fidelity
    trade-off curve

- Milestone 2: corpus and tokenizer
  - Compress a small pretraining corpus and train a tokenizer on it
  - Report bits per token for both corpora
  - This is the result: a fair, bits-normalized comparison of the two
    corpora, which decides whether the efficiency claim survives

- Milestone 3: matched-compute training runs
  - Train $M_E$ and $M_S$ with identical architecture, data budget, and
    compute
  - Evaluate on downstream tasks, decoding $M_S$ output before scoring
  - This is the result: the first capability comparison at equal compute

- Milestone 4: ablations
  - Vary the aggressiveness of $\phi$ (articles only, then inflection, then
    full caveman register)
  - This is the result: the point on the compression axis where fidelity or
    capability starts to degrade

## References

- Ogden, _Basic English: A General Introduction with Rules and Grammar_.
  (1930)
- Fuchs et al., _Attempto Controlled English for Knowledge Representation_.
  (2008)
- Deletang et al., _Language Modeling Is Compression_. (2024)
- Gage, _A New Algorithm for Data Compression_. (1994)
