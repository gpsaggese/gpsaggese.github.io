# Noesis issue workflow

- The generic workflow (Idea -> Project -> Task) is in
  [Planning Hierarchy](../../../helpers_root/docs/planning/all.plan_work.how_to_guide.md#planning-hierarchy)
- This document has the Noesis specifics

## Idea

- Record the problem, the intended benefit, and the unknowns in a shared planning
  note or in a clearly marked exploratory issue
  - E.g., "Can we detect when a provider serves a weaker model than promised?"
- Link the idea from the relevant work so that the team can find it
- A Google Doc can be used if it is shared with the team
  - It is not a required separate system

## Project

- Noesis has one [project board](https://github.com/users/gpsaggese/projects/2)
  - Individual milestones and features do not each require a new board
- Use the [MVP plan](../docs/plan.md#mvp-execution-plan) to connect the tasks to
  the intended product outcome
- E.g., the verified-delivery milestone becomes separate issues for:
  - Calibration
  - Probe scheduling
  - Automatic evaluation
  - A reproducible fault experiment

## Task (issue)

- The anatomy of a task is described in
  [Filing a New Issue](../../../helpers_root/docs/work_organization/all.use_github.how_to_guide.md#filing-a-new-issue)
- Add the executable Noesis issues to the project board
  - Use the statuses in
    [Board structure](github_projects_process.reference.md#board-structure)
  - Select the work for one-week Thursday-Wednesday sprints
  - Review the blockers at the Thursday meeting
  - Report a blocker as it occurs
- E.g., "Run provider calibration" is complete when:
  - The selected configuration is documented
  - The results are documented
  - The costs are documented
  - The retained question set is documented
  - The documentation is reviewed
  - It is not complete merely when the script runs
- For filing, reviewing, and closing work, see the
  [GitHub guide](use_github.how_to_guide.md)
