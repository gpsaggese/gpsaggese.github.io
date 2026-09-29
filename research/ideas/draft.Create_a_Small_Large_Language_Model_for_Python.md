# Create a Small Large Language Model for Python Code

## Status
- **Status**: draft
- **Complete Specs**: 20%
- **Assignee**: TBD

## Core Idea
- Train a very small LM restricted to a single, narrow slice of Python (e.g.,
  short standalone functions with simple control flow) and test how small a
  model can be while still generating syntactically valid, executable code
- Sibling ideas: [[draft.Create_a_Small_Large_Language_Model_for_Kids_Book]]
  (natural-language domain) and
  [[draft.Create_a_Small_Large_Language_Model_for_Logic]] (math/logic domain)
  apply the same "restrict the domain to shrink the model" methodology to
  different vocabularies
- Correctness is directly checkable by running the generated code against
  unit tests, giving a much stricter success criterion than perplexity or
  human judgment

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```text
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples
- **Model size sweep**: train 1M/10M/50M/125M parameter models on the same
  narrow-Python corpus; measure at what size generated functions pass unit
  tests (pass@1) vs. merely being syntactically valid
- **Domain-width ablation**: compare "single-purpose functions only" vs.
  "functions + simple classes" at matched parameter count, to see how much
  restricting the domain further shrinks the model needed for a given pass
  rate
- **Failure mode**: small models produce syntactically valid code with
  off-by-one or wrong-operator bugs: plausible-looking but functionally
  incorrect, the code analogue of TinyStories' "locally fluent, globally
  inconsistent" failure

## Questions
1. What is the minimum parameter count for reliable single-function code
   generation with passing unit tests, and how does it compare to the
   parameter counts found for the Kids_Book and Logic variants?
2. Does execution feedback (run the generated code, use pass/fail as signal)
   let a small model reach a given pass rate with fewer parameters than
   next-token prediction alone?
3. Is "restrict domain to shrink model" as effective for code as for prose,
   given code has stricter syntactic/semantic constraints than natural
   language?

## Research Topics
- Scaling laws for narrow-domain code LMs (compare against general-purpose
  code-LM scaling, e.g., Codex/StarCoder scaling curves)
- Execution-guided training/fine-tuning at small scale (unit-test pass/fail as
  reward signal)
- Comparison against the Kids_Book and Logic variants: is there a shared
  "narrow-domain scaling law," or does each domain compress differently?
- **Training corpus choice**: a TinyStories-for-code analogue, synthetically
  generating short Python functions (single-purpose, under 15 lines,
  restricted standard-library subset) with a large model, following the same
  "generate simple examples with a big model, train a tiny model on them"
  recipe as TinyStories
- **Existing code corpora**: CodeParrot and The Stack (Python subset), filtered
  to short self-contained functions (drop files with wide imports, classes, or
  heavy dependencies) to match the "small vocabulary" spirit of the sibling
  ideas
- **Evaluation sets**: HumanEval and MBPP-style problem sets provide small,
  well-scoped function-level problems with unit tests, usable both as training
  signal and as held-out evaluation

## Next steps
- [ ] Look for related research (TinyStories, phi-1 "textbooks are all you
  need," small code-LM literature)
- [ ] Reproduce a HumanEval/MBPP-style baseline at small scale as a sanity check
- [ ] Design the model-size sweep experiment with automatic pass@1 scoring
- [ ] Break the problem down into phases and milestones

## Implementation plan

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References
- Eldan, R., & Li, Y. (2023). _TinyStories: How Small Can Language Models Be
  and Still Speak Coherent English?_
- Gunasekar, S., et al. (2023). _Textbooks Are All You Need_ (phi-1)
- Chen, M., et al. (2021). _Evaluating Large Language Models Trained on Code_
  (HumanEval)
