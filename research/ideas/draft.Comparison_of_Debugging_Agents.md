# Comparison of AI Code Debugging Agents

## Status
**Status:**: draft
**Complete Specs:**: 0-100%
**Assignee:**: ...

## Core Idea

- AI debugging agents are tools that automatically diagnose root causes of
  software failures, suggest fixes, and explain error patterns with minimal
  human intervention
- They range from lightweight error message analyzers (IDE integrations) to
  full autonomous debugging environments
- Key capabilities: stack trace interpretation, root cause localization
  across multiple files, hypothesis generation for bug fixes, code patch
  suggestion, reproduction of failure conditions, and natural-language
  explanation of failure modes
- Agents differ in inference latency, accuracy of root cause pinpointing,
  quality of proposed fixes, explainability of the debugging reasoning, and
  ability to handle complex bugs spanning multiple layers (UI, backend,
  database)
- This project develops critical skills in evaluating AI for
  safety-critical tasks, where incorrect debugging suggestions could mask
  real problems or introduce new bugs
- Project objective: design a controlled empirical study that benchmarks at
  least three AI debugging agents on a curated dataset of real and
  synthetic bugs, comparing root cause localization accuracy, correctness
  of proposed fixes, explanation quality, and false positive rate

## Formalization

- Debugging agent capability ladder:
  - **L0, suggest cause**: suggest a likely root cause, e.g., "undefined
    variable on line 42"
  - **L1, propose fix**: generate a patch, e.g., "add null check before
    accessing property"
  - **L2, localize and explain**: trace the root cause across files, e.g.,
    "variable mutated at line 20, leading to crash at line 50"
  - **L3, test and validate**: verify the fix works without regressions, by
    applying the fix, running tests, and measuring performance impact
  - **L4, autonomous debugging**: reproduce, debug, fix, and test end to
    end, shipping the bug fix
- Bug benchmark taxonomy used to construct the test set:
  - **Logic errors**: off-by-one in loops, incorrect conditional branches,
    wrong operator (e.g., `==` vs. `=`), incorrect return value or side
    effect
  - **Boundary conditions**: empty input handling, null/None dereference,
    integer overflow/underflow, array index out of bounds
  - **API misuse**: incorrect function call signature, forgetting required
    initialization, using a deprecated API, incorrect state machine
    transitions
  - **Concurrency**: race condition (shared mutable state), deadlock in
    locking, missing synchronization
  - **Resource management**: memory leak (object not freed), file handle
    not closed, connection pool exhaustion
  - **Integration bugs**: type mismatch between modules, data format
    incompatibility, timing dependency (events out of order)

## Key Examples

- Debugging agents compared:
  - **GitHub Copilot** (IDE-integrated): AI-powered code completion and
    diagnostics for debugging within IDEs; strength is developer workflow
    integration (https://github.com/features/copilot)
  - **Devin** (autonomous agent): fully autonomous software engineer that
    debugs, tests, and iterates on bug fixes; strength is end-to-end
    debugging (https://cognition.ai)
  - **OpenDevin** (open-source): open alternative to Devin with debugging,
    planning, and code execution capabilities; strength is being
    customizable and transparent (https://github.com/OpenDevin/OpenDevin)
  - **Cursor** (IDE tool): IDE with an integrated AI assistant for
    debugging and code improvement; strength is seamless editor integration
    (https://www.cursor.com)
  - **Semgrep + LLM** (static analysis): rule-based bug detection enhanced
    with LLM explanations and fix suggestions; strength is pattern-based
    reliability (https://semgrep.dev)
  - **Tabnine Debugger** (specialized): LLM-powered debugging suggestions in
    development environments; strength is being fast and lightweight
    (https://www.tabnine.com)
- **Off-by-one loop**:
  ```python
  def find_max(arr: List[int]) -> int:
      max_val = arr[0]
      for i in range(len(arr)):  # Should be range(1, len(arr))
          if arr[i] > max_val:
              max_val = arr[i]
      return max_val

  # Bug: arr[len(arr)] is out of bounds on first iteration when i == len(arr)
  ```
- **Null dereference**:
  ```python
  def get_user_email(user_id: int) -> str:
      user = database.find_user(user_id)
      return user.email  # Bug: user can be None if not found

  # Correct fix: check if user is None
  ```
- **Resource leak**:
  ```python
  def read_config(path: str) -> dict:
      file = open(path, 'r')
      data = json.load(file)
      return data
      # Bug: file handle not closed; should use 'with' statement
  ```

## Questions

1. Which agents accurately pinpoint root causes, propose working fixes, and
   explain failures in a way developers can understand?

## Research Topics

- **Multi-file bug tracing**: create bugs that require tracing across 3+
  files (e.g., data flows from API layer to business logic to database),
  and measure which agents can trace dependencies across module boundaries
- **Performance bug detection**: introduce intentional performance
  regressions (O(n^2) algorithm, memory leak causing GC pressure), and
  evaluate whether agents identify these as bugs and suggest optimization
- **Reproduction time**: measure how many agent turns (prompts/iterations)
  are needed to reach a correct diagnosis; shorter paths indicate more
  efficient reasoning
- **Adversarial tests**: introduce bugs that are intentionally obfuscated
  (e.g., triggered only under rare conditions), and evaluate robustness and
  false positive rate
- **Explanation clarity**: have human developers rate agent explanations
  independently, and measure whether explanation clarity correlates with
  fix correctness
- **Domain-specific bugs**: create bug sets for specific domains (web
  services, data pipelines, ML model training), and measure agent
  specialization
- **Fix quality rubric**: beyond "does it pass tests?", score fixes on
  elegance, performance impact, readability, and adherence to project style

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: bug benchmark creation
  - Curate or seed 20-30 reproducible bugs with known fixes, covering
    multiple categories (logic errors, boundary conditions, concurrency,
    memory, API misuse, performance)
  - Document the expected root cause, impact severity, and correct fix for
    each bug

- Milestone 2: agent setup and execution
  - Install and configure at least three debugging agents (e.g., Devin,
    OpenDevin, Cursor)
  - Run each agent with a bug reproduction script/test case and error
    message
  - Record time to identify the root cause, the proposed fix, and any
    partial or incorrect suggestions

- Milestone 3: root cause accuracy
  - For each bug, measure whether the agent correctly identified the file,
    the function/method, the line number or code expression, and the root
    cause category

- Milestone 4: fix correctness validation
  - For each proposed fix, apply it to the codebase and measure whether the
    test suite passes, whether there are regressions, and whether the fix
    addresses the root cause or just masks the symptom

- Milestone 5: explanation quality assessment
  - Extract the agent's explanation of the failure and root cause
  - Score it on clarity, technical accuracy, and usefulness for a developer
    to understand why the bug occurred, using LLM-as-judge or manual review
    by experienced developers

- Milestone 6: false positive and false negative analysis
  - Introduce code that appears buggy but is actually correct (false
    positive test) and code with subtle bugs the agent might miss (false
    negative test)
  - Measure precision and recall

- Milestone 7: comparative scorecard
  - Build a rubric weighing root cause accuracy, fix correctness,
    explanation quality, and false positive rate
  - Rank agents and identify which bug categories each agent handles
    best/worst

## References

- Real bug datasets:
  - Defects4J (Java bugs): https://github.com/rjust/defects4j
  - BugsInPy (Python bugs): https://github.com/soarsmu/BugsInPy
  - GitHub Issues (labeled bugs): search GitHub for `label:bug`
- Debugging tools and analysis:
  - `pdb` (Python debugger): https://docs.python.org/3/library/pdb.html
  - `gdb` (C/C++ debugger): https://www.gnu.org/software/gdb/
  - Valgrind (memory analysis): https://valgrind.org
  - `strace` / `ltrace` (system call tracing): for OS-level failures
- Test frameworks for validation:
  - pytest: https://docs.pytest.org
  - unittest: https://docs.python.org/3/library/unittest.html
  - Hypothesis (property-based testing): https://hypothesis.readthedocs.io
- Agent platforms:
  - Devin documentation: https://docs.cognition.ai
  - OpenDevin GitHub: https://github.com/OpenDevin/OpenDevin
  - Cursor docs: https://docs.cursor.com
- LLM-as-judge:
  - Claude API: https://anthropic.com/api
  - OpenAI API: https://platform.openai.com/docs
