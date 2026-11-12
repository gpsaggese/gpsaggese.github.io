---
title: "My Agentic Engineering Flow"
draft: true
authors:
  - gpsaggese
date: 2026-09-08
description:
categories:
  - AI Coding
  - Developer Tools
---

TL;DR: How my agentic engineering flow has been shaping up

<!-- more -->

- This blog describes the `auto_task` flow: the convention I use to turn a problem
  into a spec, have an agent critique the spec, and then execute it as one or more
  branches and PRs
- It complements two companion posts:
  [A Queue of AI Coding Agents](/blog/a-queue-of-ai-coding-agents) covers how tasks
  get _into_ the queue, and
  [Stacked PRs for Agentic Development](/blog/stacked-prs-for-agentic-development)
  covers _how_ a multi-step task gets executed as a stack
- This post is the map that ties the pieces together
- The steps to follow are not here: they live in the how-to guides
  - Idea to merged PR: `docs/planning/all.plan_work.how_to_guide.md`
  - From task list to merged PR: `docs/tools/ai/all.auto_task.how_to_guide.md`

## The Unit of Work

- In my workflow, every unit of work maps to one GitHub issue
  - Most issues become a single Git branch and a single PR
  - A large refactor can span multiple branches / PRs for the same issue, e.g., split
    into chunks for safety and ease of review
- Branches and PRs are named after the issue, so I can trace any change from the
  issue to the merge
  - The naming pattern is in `docs/tools/ai/all.auto_task.how_to_guide.md`

## Spec Before Code

- Before any code is written, the problem and the solution are captured in a plan
  that follows a fixed template
- The template forces two things I care about most:
  - The problem is stated _before_ the solution
  - A multi-PR issue is broken into explicit, ordered `PR<N>` sections instead of one
    long unstructured list
- The same idea scales up to a whole project: a PRD, a design, and a plan come
  before the first task
  - See `docs/planning/all.plan_work.how_to_guide.md`

## Criticize First, Then Execute

- Four skills carry a task from a rough idea to a finished stack:
  - `/auto_task.create_specs_from_todos`
  - `/auto_task.criticize`
  - `/auto_task.execute_interactively`
  - `/auto_task.execute_with_stacked_prs`
- I always run `/auto_task.criticize` first
  - It stops the flow when the problem or the solution is not perfectly clear
  - Otherwise an agent guesses and builds downstream work on top of a guess
- Only after the plan is confirmed do I pick one of the two execution skills
  - Uninterrupted run: stacked PRs
  - Tighter feedback loop: interactive
  - See [Stacked PRs for Agentic Development](/blog/stacked-prs-for-agentic-development)
    for how I choose between them

## Keeping a PR Mergeable

- Once a branch exists, three narrower skills keep it moving toward a mergeable
  state, whether that happens on my laptop or on CI:
  - `/github.get_pr_to_pass_local_tests`
  - `/github.get_pr_to_pass_ci_tests`
  - `/github.get_pr_ready_to_merge`
- Each of these keeps its own progress file with a checklist of actions, so a run
  can be:
  - Monitored for completion
  - Resumed instead of restarted from scratch

## Agents Do Not Commit on Their Own

- In the interactive flow the agent never commits: I review the change first
- In the automated flows I authorize commits for one project at a time
  - The reasons and the options are in
    [TIL: How to Not Let Coding Agents Commit Code Automatically](/blog/til-how-to-not-let-coding-agents-commit-code-automatically)
  - The steps are in `docs/tools/ai/all.auto_task.how_to_guide.md`

## The Flow End to End

- The diagram and the table of skills, from idea to merged PR, are in the section
  "From Idea to Merged PR" of `docs/planning/all.plan_work.how_to_guide.md`
