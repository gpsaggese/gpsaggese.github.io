# Use Agents to Update Repo Doc

## Status
- **Status:**: draft
- **Complete Specs:**: 0-100%

## Core Idea

- The repo has README files on a per-file basis and a per-dir basis (e.g.,
  `README.md`), each governed by its own rules (e.g., `file.README.md`)
- Idea: write a script that navigates the repo, finds which dir or file has
  been updated, and updates the corresponding README using an LLM agent
- The update should run incrementally, based on the last modification time of
  the source files and of the README itself

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```text
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

- **Prior art**: search for existing systems (e.g., context7) that already
  solve automatic README updates from code changes

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: detect what needs an update
  - Collect the specs of `README.md` and `exec.README.md`
  - Find the last modification of each file and of its README (using git or
    an explicit tag), so updates can run incrementally
  - Support a `--no_incremental` flag to force a full update
  - This is the result: a mechanism that knows which READMEs are stale

- Milestone 2: build the update command
  - Build `find_newer_files.py` to find files modified after a date, with
    `--file`, `--dir`, and `--type` filters
  - Add an `--update_file` flag to trigger the README update for a match
    - E.g., `research/ideas/prompt.update_README.md`
  - Use an LLM agent to regenerate the README content from the changed files
  - This is the result: a script that updates a README whenever its source
    files change

## References

- `helpers_root/.claude/skills/readme.create/SKILL.md`
- `helpers_root/.claude/skills/readme.write_architecture/SKILL.md`
- `helpers_root/.claude/skills/research_idea.update_readme/SKILL.md`
