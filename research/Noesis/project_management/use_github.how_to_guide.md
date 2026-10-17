# Use GitHub for Noesis

- The generic guide is
  [all.use_github.how_to_guide.md](../../../helpers_root/docs/work_organization/all.use_github.how_to_guide.md)
- This document has the Noesis settings and the deviations from the generic guide

## Start here

- [Noesis project board](https://github.com/users/gpsaggese/projects/2)
- [Repository issues](https://github.com/gpsaggese/gpsaggese.github.io/issues)
- [Repository pull requests](https://github.com/gpsaggese/gpsaggese.github.io/pulls)
- [Board and sprint process](github_projects_process.reference.md)
- Executable work belongs in an issue
  - Ideas can begin in planning notes before becoming scoped issues
  - See the [issue workflow](issue_workflow.explanation.md)

## Sprints and statuses

- Sprints run Thursday through Wednesday, with a Thursday meeting to check
  blockers
  - See the [sprint process](github_projects_process.reference.md#sprint-process)
- The board has four statuses: Todo, In Progress, Blocked, and Done
  - See
    [Lean board with four statuses](../../../helpers_root/docs/work_organization/all.github_projects_process.reference.md#lean-board-with-four-statuses)
  - Report a blocker using the template in
    [Updating an Issue](../../../helpers_root/docs/work_organization/all.use_github.how_to_guide.md#updating-an-issue)

## Filing an issue

- Follow
  [Filing a New Issue](../../../helpers_root/docs/work_organization/all.use_github.how_to_guide.md#filing-a-new-issue)
- Add the issue to the Noesis board and choose Todo
- Assign one accountable owner before implementation begins
  - If the ownership is unclear, ask the project maintainer
  - Name the collaborators separately
- Use the existing
  [repository labels](https://github.com/gpsaggese/gpsaggese.github.io/labels)
  - Do not assume that imported labels or label-sync scripts are configured here
  - Use priority labels only when the team agrees on their meaning

## Opening a pull request

- Follow
  [Filing a New PR](../../../helpers_root/docs/work_organization/all.use_github.how_to_guide.md#filing-a-new-pr)
- The PR title does not have to match the branch name
- Use the labels `PR_for_authors`, `PR_for_reviewers`, and `PR_for_integrators`
  only if they exist in the repository
  - See
    [PR Labels](../../../helpers_root/docs/work_organization/all.use_github.how_to_guide.md#pr-labels)

## Review and feedback

- Follow
  [Review](../../../helpers_root/docs/work_organization/all.use_github.how_to_guide.md#review)
  and
  [Addressing a Comment](../../../helpers_root/docs/work_organization/all.use_github.how_to_guide.md#addressing-a-comment)
- Keep the issue In Progress until the change is accepted and merged, unless it
  is genuinely blocked

## Closing an issue

- Follow
  [Closing an Issue](../../../helpers_root/docs/work_organization/all.use_github.how_to_guide.md#closing-an-issue)
- A maintainer or a designated reviewer confirms the acceptance, closes the
  issue, and sets the board status to Done

## GitHub Actions

- The repo workflows live under `.github/workflows`
  - See
    [GH Actions](../../../helpers_root/docs/work_organization/all.use_github.how_to_guide.md#gh-actions)
- Follow the [Noesis onboarding guide](../docs/onboarding.md) for local
  validation
