# Unit Testing LLM Skills

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

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

- Milestone 1: design the test spec and assertion types
  - Define a fixture file format (e.g. YAML) with skill name, input, and
    one or more assertions per test case
  - Define the assertion types: exact match, structural (e.g. valid JSON
    against a schema), semantic equivalence, and behavioral invariants,
    answering what the minimal passing set looks like (Question 1)
  - This is the result: a documented `skilltest` fixture schema and
    assertion-type reference, with 2-3 example fixture files

- Milestone 2: build the runner core
  - Implement a CLI runner that loads fixtures, invokes the target skill,
    and evaluates outputs with rule-based checks first and an LLM-as-judge
    check for semantic assertions
  - Handle non-determinism by repeating each LLM-judged case a fixed number
    of times and requiring a majority pass, or by using a similarity
    threshold instead of exact match (Question 2)
  - Add an isolation layer that mocks external API calls and file system
    access during a test run, analogous to mocking in traditional unit
    tests (Research Topic)
  - This is the result: a working `skilltest` CLI that runs a small suite
    against a real skill and reports pass/fail with dependencies mocked out

- Milestone 3: add snapshot and property-based modes
  - Implement snapshot testing: store a golden output per test case and
    flag a run whose output drifts beyond a configurable threshold
  - Implement property-based input generators that probe boundary
    conditions (empty input, very long input, adversarial prompts) instead
    of only fixed examples
  - This is the result: snapshot and property-based tests running against
    2-3 example skills, catching an injected regression in each mode

- Milestone 4: pilot on real skills and wire into CI
  - Write `skilltest` suites for 3-5 existing skills in `.claude/skills/`
  - Run the suite with a cheap model in CI and the full model as a
    pre-release gate (Question 3), and measure the LLM-judge's false
    positive and false negative rate against manually labeled outcomes
  - This is the result: a CI job that runs skill regression tests on every
    commit for the piloted skills, plus a judge-accuracy report

## References

- LangChain evaluation module: basic input/output evaluation for chains
- DeepEval: pytest-style LLM evaluation framework
- Promptfoo: CLI tool for testing and comparing prompts
- Evals (OpenAI): framework for evaluating LLM outputs against rubrics
