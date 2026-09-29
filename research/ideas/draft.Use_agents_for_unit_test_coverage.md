# Use Agents for Unit Test Coverage

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: TBD

## Core Idea

- Use agents to help maintain high code coverage in the codebase

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. [Open question 1: what remains unknown?]
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- **Incremental mode**: on every commit, check what code changed and add or
  update tests for it
- **Non-incremental mode**: run the full coverage process across the whole
  codebase

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: measure baseline coverage
  - Run the existing coverage tooling (per
    `all.run_unit_tests.how_to_guide.md`) across the codebase
  - Rank files and functions by coverage gap to build the target list the
    agent will work through
  - This is the result: a baseline coverage report ranking the
    lowest-coverage files and functions

- Milestone 2: build incremental mode
  - Add a pre-commit or CI hook that diffs changed files in a commit or PR
  - For each changed function lacking coverage, dispatch an agent that
    follows the `testing.reach_coverage` skill and
    `all.write_unit_tests.how_to_guide.md` conventions to add or update
    tests
  - Run the updated test suite to confirm the new tests pass and coverage
    improves for the changed code
  - This is the result: a working incremental-mode hook that, on a sample
    PR, generates passing tests for the newly changed code

- Milestone 3: build non-incremental (batch) mode
  - Write a script that iterates over the lowest-coverage targets from
    Milestone 1 and dispatches one agent per file or function to write
    tests up to a target coverage threshold
  - Run the full test suite after each batch to catch regressions before
    accepting the generated tests
  - This is the result: a batch run that raises coverage for a sample set
    of target files to the chosen threshold, with all new tests passing

- Milestone 4: evaluate test quality and cost
  - Check that agent-written tests are meaningful rather than
    coverage-padding, e.g. via mutation testing or assert-density checks
  - Measure agent cost (tokens and time) per percentage point of coverage
    gained, for both modes
  - This is the result: a quality report cross-checking agent-written
    tests via mutation testing, plus a cost-per-coverage-point metric

## References

- `./helpers_root/helpers/test/docs/all.write_unit_tests.how_to_guide.md`
- `./helpers_root/helpers/test/docs/all.run_unit_tests.how_to_guide.md`
- `./website/docs/blog/posts/in_30_mins.Python_Code_Coverage.md`
- `./helpers_root/.claude/skills/testing.reach_coverage/SKILL.md`
- `./helpers_root/helpers/test/docs/all.unit_test_framework.explanation.md`
