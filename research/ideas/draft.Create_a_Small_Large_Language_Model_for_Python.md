# Create a Small Large Language Model for Python Code

## Status
- **Status**: draft
- **Complete Specs**: 20%

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

- Milestone 1: build narrow-Python training corpora and an eval harness
  - Generate synthetic short (under 15 lines) single-purpose Python functions
    with a large model, following the TinyStories "big model generates
    simple examples" recipe, restricted to a narrow standard-library subset
  - Filter CodeParrot and The Stack Python subset to short, self-contained
    functions, dropping files with wide imports, classes, or heavy
    dependencies, as a second "real code" corpus
  - Set up HumanEval and MBPP as held-out eval sets with an execution
    sandbox for automatic pass@1 scoring
  - This is the result: two training corpora (synthetic and filtered real)
    and a working pass@1 evaluation harness

- Milestone 2: run the model-size sweep with automatic pass@1 scoring
  - Train 1M/10M/50M/125M parameter models with a fixed architecture family
    on each corpus, at matched compute budget
  - Score every checkpoint on HumanEval/MBPP-style problems for syntactic
    validity and pass@1
  - This is the result: a size x corpus grid with syntactic-validity and
    pass@1 curves, and the minimum parameter count for reliable
    single-function code generation

- Milestone 3: run the domain-width ablation and analyze failure modes
  - Train matched-size models on "single-purpose functions only" vs.
    "functions plus simple classes" corpora
  - Categorize failures at each size: syntax errors vs. plausible-looking
    but functionally wrong code (off-by-one, wrong operator)
  - This is the result: a quantified effect of domain width on required
    model size, and a taxonomy of small-model code failure modes

- Milestone 4: test execution-guided fine-tuning
  - Fine-tune the best small checkpoint using unit-test pass/fail as a
    reward signal (rejection sampling or RL-style fine-tuning)
  - Compare pass@1 at matched parameter count against the
    next-token-prediction-only baseline from Milestone 2
  - This is the result: a measured pass@1 gain (or lack of one) from
    execution feedback at fixed model size

## References
- Eldan, R., & Li, Y. (2023). _TinyStories: How Small Can Language Models Be
  and Still Speak Coherent English?_
- Gunasekar, S., et al. (2023). _Textbooks Are All You Need_ (phi-1)
- Chen, M., et al. (2021). _Evaluating Large Language Models Trained on Code_
  (HumanEval)
