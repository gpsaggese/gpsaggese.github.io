# Noesis GitHub Projects process

- This document has the Noesis settings of the generic process
  - Generic process:
    [all.github_projects_process.reference.md](../../../helpers_root/docs/work_organization/all.github_projects_process.reference.md)
  - Issue and PR steps: [GitHub guide](use_github.how_to_guide.md)
  - Collaboration: [team guide](organize_your_work.how_to_guide.md)

## Overview

- Track Noesis work in the
  [Noesis GitHub Project](https://github.com/users/gpsaggese/projects/2)
- Issues and pull requests live in the
  [repository](https://github.com/gpsaggese/gpsaggese.github.io)

## Board structure

- The board uses the lean board with four statuses: Todo, In Progress, Blocked,
  and Done
  - See
    [Lean board with four statuses](../../../helpers_root/docs/work_organization/all.github_projects_process.reference.md#lean-board-with-four-statuses)
- A card represents an issue and can show an assignee and a linked PR
- There is no separate In Review, QA, or Paused column

## Issue fields

- The fields are described in
  [Issue fields](../../../helpers_root/docs/work_organization/all.github_projects_process.reference.md#issue-fields)
- Noesis settings:
  - **Assignee**: One accountable owner, with the collaborators named in the issue
  - **Labels**: Use the labels in the
    [repository label list](https://github.com/gpsaggese/gpsaggese.github.io/labels)
  - **Sprint**: The Thursday-Wednesday week selected for the issue
  - **Pull request**: Reference the PR and the related issue in both directions
- Assignee and Status are required before work starts
- Sprint and Estimate are planning fields
  - Their existence on the live board still needs confirmation
  - If Sprint is not configured, record the dates in the weekly summary until it
    is available

## Sprint process

- A sprint lasts one week, from Thursday through Wednesday, inclusive
  - The next sprint begins on the following Thursday
  - E.g., October 15-21, 2026, followed by October 22-28
  - See
    [Project-specific cadence](../../../helpers_root/docs/work_organization/all.github_projects_process.reference.md#project-specific-cadence)
- The team meets every Thursday to check blockers and coordinate the new sprint
  - Agenda and rules:
    [Weekly blocker meeting](../../../helpers_root/docs/work_organization/all.scrum.explanation.md#weekly-blocker-meeting)
  - The meeting time, timezone, meeting URL, and facilitator remain to be agreed
- Write the weekly summary using
  [Lean weekly summary](../../../helpers_root/docs/work_organization/all.github_projects_process.reference.md#lean-weekly-summary)
  - Post it in the project status update
  - If that feature is unavailable to the team, post it in a linked issue
- Handle unfinished issues following
  [Carryover of unfinished issues](../../../helpers_root/docs/work_organization/all.github_projects_process.reference.md#carryover-of-unfinished-issues)

## Automation

- Board updates are manual unless a workflow has been verified on the Noesis
  project
  - Do not assume that automation from other organizations exists here
- An authorized project maintainer can configure a Sprint iteration that starts
  on a Thursday and lasts one week
  - See
    [Github Actions automations](../../../helpers_root/docs/work_organization/all.github_projects_process.reference.md#github-actions-automations)

## Larger work

- Follow the convention in
  [Parent and sub-issues](../../../helpers_root/docs/work_organization/all.github_projects_process.reference.md#parent-and-sub-issues)
