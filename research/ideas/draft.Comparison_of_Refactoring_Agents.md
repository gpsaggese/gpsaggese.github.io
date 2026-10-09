# Comparison of AI Code Refactoring Agents

## Status
**Status:**: draft
**Complete Specs:**: 0-100%

## Core Idea

- AI refactoring agents are systems that automatically improve code quality
  by eliminating duplication, reducing cyclomatic complexity, improving
  variable naming, restructuring modules, and applying design patterns with
  minimal manual intervention
- They range from rule-based linters with AI enhancement (e.g., `pylint` +
  LLM backends, Semgrep with explanations) to full autonomous refactoring
  engines
- Key capabilities: identifying code smells (dead code, long methods,
  complex conditionals), suggesting refactoring operations (extract method,
  consolidate duplicates, rename for clarity, replace magic numbers),
  estimating refactoring impact, preserving behavior (all tests pass after
  refactoring), and explaining the rationale for changes in natural
  language
- Agents differ in refactoring ambition (cosmetic cleanup vs. architectural
  redesign), safety (does refactored code have identical behavior?), code
  review compatibility, and runtime performance impact
- This project develops critical skills in measuring non-functional code
  quality improvements and trade-offs: good refactoring is hard to measure
  and easy to get wrong
- Project objective: design a controlled empirical study that benchmarks at
  least three AI refactoring agents on a curated set of Python codebases
  with intentional quality issues, comparing code smell detection accuracy,
  correctness of refactoring (behavior preservation), code quality
  improvements, review-ability of generated changes, and performance impact

## Formalization

- Refactoring agent capability ladder:
  - **L0, suggest**: highlight code smells, e.g., "this method is 150
    lines, consider extracting"
  - **L1, suggest fix**: propose a single refactoring, e.g., "extract
    method `_validate_input()` from line 20-45"
  - **L2, apply refactoring**: execute the refactoring operation, e.g.,
    automatically rename variable `x` to `user_id` across the file
  - **L3, multi-file refactoring**: coordinate refactoring across files,
    e.g., consolidate duplicate classes into a shared base class
  - **L4, autonomous redesign**: plan and execute architectural changes,
    e.g., redesign the module structure and move classes to new files
- Code quality metrics and tools used to measure before/after state:
  - **Cyclomatic complexity** (`radon cc`): number of decision points,
    lower is better; good target is under 10
  - **Maintainability index** (`radon mi`): composite metric based on
    complexity and lines of code; good target is above 80
  - **Code duplication** (`pylint` `dupdup`): percentage of duplicated
    code, lower is better; good target is under 5%
  - **Lines of code per function** (`radon metrics`): average function
    length, lower is better; good target is under 30 lines
  - **Code coverage** (`coverage.py`): percentage of code executed by
    tests, higher is better; good target is above 80%

## Key Examples

- Refactoring agents compared:
  - **GitHub Copilot** (IDE-integrated): AI-powered refactoring suggestions
    within IDEs with one-click application; strength is seamless workflow
    (https://github.com/features/copilot)
  - **Semgrep** (rule-based + LLM): static analysis with LLM-powered
    explanations and automated fix suggestions; strength is rule-driven
    precision (https://semgrep.dev)
  - **Devin** (autonomous agent): fully autonomous refactoring, including
    complex multi-file restructuring; strength is end-to-end transformation
    (https://cognition.ai)
  - **Rope** (Python-specific tool): Python refactoring library with
    AST-based transformations, LLM-enhanced discovery; strength is
    language-specific safety (https://github.com/python-rope/rope)
  - **Cursor** (IDE tool): AI IDE with refactoring suggestions and
    one-click application; strength is integrated debugging
    (https://www.cursor.com)
  - **Tuple** (Databricks, experimental): experimental multi-agent system
    for code improvement and refactoring; strength is research-grade
    autonomy (https://databricks.com)
- Intentionally seeded quality issues used to test the agents:
  - **Method too long** (more than 50 lines, multiple concerns):
    ```python
    def process_user_data(user_input):
        # Validation logic (10 lines)
        # Database query (5 lines)
        # Data transformation (15 lines)
        # Email notification (10 lines)
        # Logging (5 lines)
        # Ideal refactoring: extract into _validate(), _transform(),
        # _notify_user(), etc.
    ```
  - **Code duplication** (same logic in multiple functions):
    ```python
    def validate_email(email): ...
    def validate_email_v2(email): ...  # Almost identical
    # Ideal refactoring: consolidate to a single function
    ```
  - **High cyclomatic complexity** (deeply nested conditionals):
    ```python
    def check_access(user, resource, action):
        if user is not None:
            if user.is_active:
                if user.has_role('admin'):
                    return True
                elif user.has_role('editor'):
                    if resource.is_editable:
                        return True
        return False
        # Ideal refactoring: guard clauses, extract logic
    ```
  - **Poor naming** (unclear variable names):
    ```python
    def f(x, y):
        z = x * 0.15
        return y - z
        # Ideal refactoring: rename to calculate_discount(), apply_tax()
    ```
  - **Dead code** (unused imports, variables, functions):
    ```python
    import unused_module  # Ideal: remove
    def old_function():  # Ideal: remove or deprecate
        pass
    ```

## Questions

1. Which agents identify the most impactful refactoring opportunities,
   execute them safely (tests still pass), produce readable changes, and
   improve code quality metrics?

## Research Topics

- **Refactoring impact analysis**: for each refactoring, measure the impact
  on performance (runtime, memory), test execution time, build time, and
  deployment risk
- **Multi-layer refactoring**: create codebases with issues spanning
  multiple layers (API, business logic, database), and measure which
  agents can identify and refactor across layers
- **Style guide conformance**: test whether refactorings maintain project
  style (indentation, naming conventions, code organization) or introduce
  inconsistency
- **Incremental refactoring**: measure which agents prefer small,
  incremental refactorings (easier to review) vs. large rewrites (higher
  risk but more impact)
- **Performance optimization**: introduce intentionally slow code (O(n^2)
  algorithm, inefficient data structures), and measure whether agents
  detect and optimize it
- **Architectural patterns**: test whether agents can refactor code to
  follow design patterns (Factory, Observer, Strategy) without changing
  behavior
- **Automated refactoring chains**: test whether agents can chain multiple
  refactorings (e.g., extract method, then introduce strategy pattern, then
  consolidate classes)
- **Rollback analysis**: introduce refactorings, break functionality, and
  measure which agents can diagnose and revert the issue

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
  - Install and configure at least three chosen refactoring agents (e.g.,
    GitHub Copilot, Semgrep, Devin/Cursor) in isolated environments
  - Document version, dependencies, and cost/API limits

- Milestone 2: codebase selection and baseline
  - Select 3-5 Python codebases with known quality issues (high complexity,
    duplication, poor naming); small open-source projects work well, e.g.,
    `pallets/click`, `psf/requests`, `encode/httpx` (typically 5k-20k
    lines, well-tested, moderate complexity)
  - Measure baseline metrics: cyclomatic complexity, code duplication ratio
    (via `radon`, `pylint`, SonarQube), maintainability index (via
    `lizard`, `mi`), lines of code, number of violations

- Milestone 3: code smell detection
  - For each agent, identify and document all code smells detected (long
    methods, high complexity, duplication, dead code, unclear naming)
  - Measure sensitivity (did the agent find real smells?) and specificity
    (how many false positives?)

- Milestone 4: refactoring suggestion and execution
  - For each detected code smell, have the agent suggest and apply a
    refactoring
  - Record the type of refactoring, lines of code changed, and number of
    files affected

- Milestone 5: behavior preservation testing
  - After each refactoring, run the full test suite
  - Measure whether all tests pass, whether there are regressions, and
    whether performance degrades (runtime, memory)

- Milestone 6: code quality metrics
  - After refactoring, re-measure code quality metrics
  - Calculate the improvement in cyclomatic complexity, duplication ratio,
    maintainability index, and readability

- Milestone 7: change review-ability
  - Assess how easy the refactored code is to review: are changes logical
    and incremental, can a human understand the intent without seeing the
    original code, and are there unnecessary changes

- Milestone 8: correctness of reasoning
  - For each refactoring, extract the agent's explanation of why the change
    improves code quality
  - Score it on technical accuracy and clarity

- Milestone 9: comparative scorecard
  - Build a rubric weighing code smell detection accuracy, refactoring
    correctness, quality improvement, change review-ability, and
    performance impact
  - Rank agents and identify specialization

## References

- Code quality tools:
  - `radon` (cyclomatic complexity and maintainability index):
    https://radon.readthedocs.io
  - `pylint` (code analysis for Python): https://pylint.pycqa.org
  - SonarQube (full code quality platform): https://www.sonarqube.org
  - Code Climate (web-based quality metrics): https://codeclimate.com
- Refactoring reference:
  - _Refactoring_ by Martin Fowler: https://refactoring.com
  - Refactoring Guru patterns: https://refactoring.guru/refactoring
- AST and code transformation:
  - `libcst` (concrete syntax tree for Python):
    https://github.com/Instagram/LibCST
  - `ast` (Python's abstract syntax tree):
    https://docs.python.org/3/library/ast.html
  - `tree-sitter` (language-agnostic parser): https://tree-sitter.github.io
- Agent resources:
  - GitHub Copilot API: https://docs.github.com/en/copilot/quickstart
  - Semgrep documentation: https://semgrep.dev/docs
  - Devin/Cursor documentation: https://docs.cognition.ai
- Testing and behavior preservation:
  - pytest: https://docs.pytest.org
  - `coverage.py`: https://coverage.readthedocs.io
  - `pytest-benchmark` (performance testing):
    https://pytest-benchmark.readthedocs.io
