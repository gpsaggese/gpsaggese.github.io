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

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References

- `./helpers_root/helpers/test/docs/all.write_unit_tests.how_to_guide.md`
- `./helpers_root/helpers/test/docs/all.run_unit_tests.how_to_guide.md`
- `./website/docs/blog/posts/in_30_mins.Python_Code_Coverage.md`
- `./helpers_root/.claude/skills/testing.reach_coverage/SKILL.md`
- `./helpers_root/helpers/test/docs/all.unit_test_framework.explanation.md`
