# Automate Stacked-PR Splitting and Sequential Merging

## Status
**Status:**: draft
**Complete Specs:**: 15%
**Assignee:**: TBD

## Core Idea
- The `github.split_branch_in_PRs` skill already proposes how to split a
  large diff into small, cohesive PRs; this idea goes one step further and
  automates the lifecycle of the resulting stack: track inter-PR
  dependencies, keep each PR rebased on its parent as earlier PRs merge, and
  auto-merge each PR in order once its CI passes and it's approved
- Related to `draft.Codebase_autocleanup.md` (automated cleanup) as part of
  a broader "reduce manual PR-herding toil" theme
- Core research question: how much of the stacked-PR workflow (dependency
  tracking, rebase-on-merge, sequential auto-merge) can be automated safely
  without a human needing to babysit merge order?

## Formalization
- Represent the PR stack as a DAG: PR `i` depends on PR `j` if `i`'s branch
  is based on `j`'s branch
- On merge of `j`: rebase all direct dependents of `j` onto the new base,
  re-trigger CI, and merge the next PR in topological order once its checks
  pass

## Key Examples
- **Simple linear stack**: PR1 -> PR2 -> PR3, each small and independently
  reviewable; auto-merge should proceed PR1, PR2, PR3 as each passes CI +
  review, with automatic rebase in between
- **Conflict during auto-rebase**: merging PR1 causes a conflict when
  rebasing PR2: the system should halt and flag for human resolution rather
  than silently forcing the merge
- **Review-not-yet-approved**: PR2 is CI-green but not yet approved when PR1
  merges: the system rebases PR2 but does not merge it until approval lands

## Questions
1. What's the right failure mode when an automatic rebase produces a
   conflict: pause the whole stack, or skip just the conflicting PR?
2. How do you keep CI cost bounded when every merge in the stack triggers a
   rebase + re-run for all dependents?
3. Is there a safe default for "auto-merge" (e.g., only when review is
   already approved and CI green), or does it always need an explicit human
   go-ahead per PR?

## Research Topics
- **Prior art**: existing stacked-PR tooling, e.g., Graphite,
  `git-branchless`
- **Dependency tracking**: GitHub API/webhook design for tracking PR
  dependency DAGs
- **Rebase safety**: safe automatic-rebase strategies and conflict detection

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
  - Survey existing stacked-PR tools (Graphite, git-branchless, ghstack) for
    prior art before building anything new
  - Define the dependency-DAG representation for a PR stack
  - Prototype auto-rebase-on-merge for a simple linear stack
  - Decide and implement the auto-merge safety policy

## References
- Graphite: stacked PR workflow tool (prior art)
