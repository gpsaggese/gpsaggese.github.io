# Library for Embedding LLM in Code

## Status
- **Status:**: draft
- **Complete Specs:**: TBD
- **Assignee:**: TBD

## Core Idea

- Design a programming language or DSL where LLM calls and deterministic
  code are first-class citizens that interleave naturally, rather than
  LLMs being called via awkward API wrappers
- Formalize the semantics of LLM-as-interpolator: given a structured input
  and output type, the LLM fills in the gap between them, analogous to
  function application but with learned rather than programmed behavior
- Build a type system that tracks which values are LLM-produced
  (uncertain, non-deterministic) versus code-produced (deterministic),
  enabling static analysis of program reliability
- Investigate syntax designs: inline LLM blocks in Python-like syntax
  (e.g., `result = llm(prompt, input, output_type)`), or a fully new
  language with native `ask`, `infer`, and `generate` keywords
- Research compilation strategies: LLM calls can be cached, batched,
  retried, or replaced with fine-tuned models, with the compiler managing
  this transparently
- Explore how the interpolator view of LLMs (they interpolate between
  training examples) shapes language design: which operations are safe to
  delegate to interpolation versus requiring deterministic code
- Project objective: design and prototype a language where mixing LLM
  inference and deterministic computation is as natural as mixing
  functions and data structures in Python, with a type system that makes
  the uncertainty boundary explicit and a runtime that optimizes LLM call
  execution

## Formalization

- Decorator-based LLM function calls:
  ```python
  @llm(model=...)
  def func(a: ..., b: ...) -> int:
      """
      Sum with <a> and <b> and return
      """
  ```
- The decorator must:
  - Use `hdbg.dassert` to check that the LLM output has the right type
  - Use introspection on the function signature to infer the expected
    types
  - Use caching from `helpers/hcache_simple.py`
  - Automatically batch calls to the LLM
  - Support multi-shot prompting to fix incorrect behavior
  - Automatically create unit tests that show the input/output behavior
- A `compile` step converts LLM-backed code into plain code, so a function
  can evolve from all-LLM to all-code
- The LLM must be able to call Python code, not just be called by it

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. What is the right abstraction boundary between the LLM and the
   surrounding program?
2. How do you handle LLM non-determinism in a language with referential
   transparency?
3. Can a compiler statically bound the cost (tokens, latency) of an
   LLM-mixed program?
4. Is the interpolator framing (LLMs generalize between training points)
   useful for formal semantics?

## Research Topics

- **DSPy programs corpus** (https://github.com/stanfordnlp/dspy): real
  programs that mix LLM modules and Python code, from the DSPy community;
  useful as design inspiration and an evaluation corpus
- **LCEL (LangChain Expression Language) pipelines**
  (https://python.langchain.com/docs/expression_language/): examples of
  chaining LLM calls with tools, parsers, and code in a compositional
  style
- **Semantic Kernel programs**
  (https://github.com/microsoft/semantic-kernel): programs mixing
  "semantic functions" (LLM) and "native functions" (code) in a shared
  kernel, from Microsoft

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References

- DSPy: Python-embedded LLM programming with module composition and
  compilation
- LMQL (Beurer-Kellner et al., 2022): query language for LLMs with
  constraints and control flow
- Guidance (Microsoft, 2023): language for constraining and interleaving
  LLM generation with code
- Marvin: Python library for LLM-as-function with type annotations
- Outlines: structured generation library treating the LLM as a typed
  function
