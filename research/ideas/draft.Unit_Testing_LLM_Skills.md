# Unit Testing LLM Skills

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: TBD

## Core Idea

- Design a framework for writing unit tests for LLM skills (prompt-based
  agents), analogous to how pytest tests Python functions
- Define what "correct behavior" means for a skill: exact output match,
  semantic equivalence, structural constraints (e.g., output must be valid
  JSON), or behavioral invariants
- Build a test runner that executes a skill against a set of input fixtures
  and evaluates outputs against expected results using both rule-based and
  LLM-as-judge checks
- Goal: build a testing framework, `skilltest` or similar, that lets
  developers write deterministic, repeatable unit tests for LLM skills,
  enabling CI/CD pipelines to catch skill regressions before deployment,
  just as pytest catches code regressions

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples

- **PromptBench**: adversarial prompt benchmarks for robustness testing
  across NLP tasks, from Microsoft Research
  ([GitHub](https://github.com/microsoft/promptbench))
- **HELM (Holistic Evaluation of Language Models)**: standardized scenarios
  and metrics for LLM evaluation, from Stanford CRFM
  ([site](https://crfm.stanford.edu/helm/))
- **BIG-Bench**: 200+ diverse tasks for probing LLM capabilities, usable as
  skill test fixtures, from Google
  ([GitHub](https://github.com/google/BIG-bench))

## Questions

1. What is the minimal set of assertions needed to declare a skill
   "passing"?
2. How do you handle the non-determinism of LLM outputs in a deterministic
   test suite?
3. Should tests use a cheap fast model for CI and the real model for
   pre-release?
4. How do you version skill tests alongside skill prompts?

## Research Topics

- **Snapshot testing**: store a "golden" output for an LLM skill and flag
  when outputs drift beyond a threshold
- **Property-based testing**: generate inputs that probe boundary
  conditions (empty input, very long input, adversarial prompts) instead of
  fixed examples
- **Isolation from external dependencies**: how to isolate skills from API
  calls and the file system in tests, analogous to mocking in traditional
  unit tests

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

- LangChain evaluation module: basic input/output evaluation for chains
- DeepEval: pytest-style LLM evaluation framework
- Promptfoo: CLI tool for testing and comparing prompts
- Evals (OpenAI): framework for evaluating LLM outputs against rubrics
