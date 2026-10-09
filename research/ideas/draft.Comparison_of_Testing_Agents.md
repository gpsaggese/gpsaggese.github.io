# Comparison of AI Testing Agents

## Status
**Status:**: draft
**Complete Specs:**: 0-100%

## Core Idea

- AI testing agents are autonomous systems that generate, maintain, and
  validate unit, integration, and end-to-end tests with minimal manual
  intervention
- They span from lightweight test generators (e.g., `pytest-codemod` with
  LLM backends, Copilot for Tests) to full autonomous test orchestration
  platforms
- Key capabilities: test case generation from code, test maintenance when
  code evolves, flaky test detection, coverage optimization, and
  intelligent test data generation
- Agents differ in test type coverage (unit vs. integration), reproducibility
  across codebases, ability to detect and fix failing tests, and
  integration with CI/CD pipelines
- Most tools expose Python SDKs or CLI interfaces, making them accessible in
  standard test environments (pytest, unittest, Jest, etc.)
- This project teaches students to rigorously benchmark testing tools
  rather than relying on vendor marketing claims: good tests are hard to
  measure, and false claims about coverage can mask real gaps
- Project objective: design a controlled empirical study that benchmarks at
  least three AI testing agents across multiple Python codebases of varying
  complexity, comparing test coverage achieved, pass@1 (tests pass without
  modification), readability and maintainability of generated tests, and
  cost/time to generate

## Formalization

- Autonomy levels for testing agents:
  - **L0, suggest**: suggest the next test case, e.g., IDE autocomplete for
    assertions
  - **L1, generate one test**: generate a single test method, e.g.,
    "generate test for this function"
  - **L2, generate suite**: generate a full test class, e.g., auto-generate
    5+ test cases for a class
  - **L3, maintain**: update tests when code changes, e.g., refactor the
    test suite after an API change
  - **L4, autonomous tester**: plan a test strategy and fix failures, e.g.,
    achieve target coverage and detect flaky tests

## Key Examples

- Testing agents compared:
  - **Diffblue Cover** (code-based): AI-powered unit test generation for
    Java and C# with automatic coverage optimization; strength is high
    coverage in production (https://www.diffblue.com/cover)
  - **GitHub Copilot for Tests** (LLM-assisted): IDE-integrated test
    generation via natural language prompts; strength is being
    developer-friendly (https://github.com/features/copilot)
  - **TestMe** (IntelliJ, open-source): IDE plugin that generates unit
    tests using AST analysis and heuristics; strength is being local and
    lightweight (https://plugins.jetbrains.com/plugin/9471-testme)
  - **Symflower** (specialized): autonomous test generation for any
    programming language using symbolic execution; strength is being
    language-agnostic (https://www.symflower.com)
  - **Mintlify Test Generator** (CLI-based): API-based test generation from
    docstrings and type hints; strength is being documentation-driven
    (https://www.mintlify.com)
  - **pytest-codemod + LLM** (framework): open-source pytest refactoring
    with LLM-powered test suggestions; strength is being highly
    customizable (https://github.com/asottile/pyupgrade)
- Benchmark functions used to test the agents:
  - Simple functions (edge case detection): `extract_email_from_text(text:
    str) -> str | None`, `parse_date_string(date_str: str) -> datetime |
    None`, `clamp(value: float, min_val: float, max_val: float) -> float`
  - Complex logic (branching, loops): `calculate_shipping_cost(weight:
    float, zone: int, expedited: bool) -> float`,
    `merge_sorted_lists(list1: List[int], list2: List[int]) -> List[int]`,
    `validate_credit_card(card_number: str) -> bool`
  - Data structure operations (mutation, invariants):
    `insert_in_sorted_array(arr: List[int], value: int) -> List[int]`,
    `remove_duplicates(items: List[str]) -> List[str]`, class `LRUCache`
    with `get()`, `put()`, `evict()`
  - Concurrent / stateful code: `thread_safe_counter` class with
    increment/decrement, `async_fetch_and_cache(url: str) -> str`

## Questions

1. Which agents produce the most reliable tests, highest coverage, and most
   maintainable test code, and under what conditions?

## Research Topics

- **Mutation testing**: use a mutation testing framework (e.g., `mutmut`
  for Python) to inject bugs into the source code, and measure whether each
  agent's test suite detects the mutation; rank agents by mutation kill
  rate
- **Flaky test detection**: run each agent's generated tests 10+ times and
  measure the flakiness rate; document which agents produce deterministic
  vs. non-deterministic tests
- **Integration test generation**: challenge each agent to generate
  integration tests that span multiple functions/classes, and evaluate the
  correctness and complexity of the generated scenarios
- **Test readability scoring**: use an LLM-as-judge to score test code
  quality (clarity, naming conventions, documentation) independent of
  coverage metrics
- **Maintenance simulation**: create a version of the codebase 6-12 months
  into development (with refactoring, API changes, new features), and
  measure which agent's test suite requires the least modification to
  remain valid
- **Cost and performance analysis**: for cloud-based agents, estimate API
  call costs and wall-clock time to generate test suites, and discuss the
  ROI of automated testing vs. manual test authoring

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: agent setup and configuration
  - Install and configure at least three chosen testing agents (e.g.,
    Diffblue Cover, GitHub Copilot for Tests, Symflower) in isolated
    environments
  - Document version pinning, dependencies, and cost (cloud API calls vs.
    local execution)

- Milestone 2: test generation on benchmark functions
  - Select 10-15 Python functions of varying complexity (simple getters,
    complex logic, edge case handling, concurrency), and run each agent to
    generate test suites
  - Record the number of test cases generated, lines of test code,
    execution time, and whether tests execute without error (pass@1)

- Milestone 3: coverage comparison
  - Measure code coverage (branch, line, path coverage) achieved by each
    agent's test suite using `coverage.py`
  - Compare coverage depth and identify which agent produces the most
    thorough tests

- Milestone 4: test maintainability and flakiness
  - Intentionally modify source code (change function behavior, add edge
    cases, refactor)
  - Measure how well each agent's tests catch the regression, require
    modification to pass, and avoid flaky assertions (e.g., time-dependent
    checks, random output)

- Milestone 5: code quality review
  - Examine the generated test code for readability (naming, structure),
    modularity (DRY principle, helper functions), presence of
    comments/docstrings, and whether tests can be debugged or modified by
    humans independently of the agent

- Milestone 6: comparative scorecard
  - Build a rubric that weights coverage, pass@1, readability,
    maintainability, and runtime
  - Score each agent and discuss trade-offs (e.g., higher coverage at the
    cost of unreadable tests)

## References

- Diffblue Cover documentation: https://docs.diffblue.com
- GitHub Copilot API and examples:
  https://docs.github.com/en/copilot/quickstart
- pytest documentation: https://docs.pytest.org
- Coverage.py: https://coverage.readthedocs.io
- mutmut (mutation testing): https://mutmut.readthedocs.io
- OpenML test suites (real benchmark datasets for algorithm evaluation):
  https://www.openml.org
- LeetCode / HackerRank: curated coding problems with reference solutions
  for testing agent evaluation
