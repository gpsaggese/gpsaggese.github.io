---
title: "A Queue of AI Coding Agents"
draft: true
authors:
  - gpsaggese
date: 2026-08-20
categories:
  - AI Coding
  - Developer Tools
---

TL;DR: My workflow to run a queue of AI agents that fix issues asynchronously.

<!-- more -->

- This blog is a short description of my workflow to organize and run a queue of AI
  agents fixing issues asynchronously

- My current coding set up is:
  - Source control: GitHub (GH) / Git
  - Agent harness: Claude Code (CC)
  - Model: Anthropic models
  - `helpers` repo

- None of this assumptions is strictly needed, and you can generalize to other code
  hosting / control systems, code harness, and models

# Overview

- The core components are:
  - **Queue**: A set of GitHub Issues that should be implemented asynchronously
    (identified by a specific label or assigned to a specific user, e.g., `claude`)
  - **Trigger**: a GitHub Action that runs when an issue is labeled assigned marking
    it as part of the Queue
  - **Agent**: Claude Code, running in the GH Actions, which
    - reads the issue
    - writes the code (running regressions, linting, etc)
    - opens a PR for human review and merge

- The actions are:
  - Create a list of issues with specs that can be executed asynchronously
  - Create programmatically GH issues from this list
  - Assign the GH issues to CC to generate a PR

## Getting Tasks in the Queue

- I maintain a markdown file called `ai_task_queue.md` with everything that I would
  like to be done over time

- The format of the task queue is simply a markdown file:
  - It has several sections (`Ready`, `Backlog`, `Done`) to represent the state of
    the tasks
  - Each section has a checklist of issues described in terms of title and specs

- `ai_task_queue.md` looks like:
  ```markdown
  # Ready

  ## [ ] <GitHub Issue Title>

  <Specs>

  ## [ ] Use System_to_one_line() in hgit.py

  <Specs>

  # Backlog

  ...

  # Done

  ...
  ```

- Each task has
  ```
  Title:
  Problem:
  Solution:
  Implementation complexity:
  Importance:
  ```

- Potential targets for asynchronous tasks are:
  - Todos in the codebase
    - E.g., marked with `TODO(ai_gp)` to communicate that are assigned by my user for
      AI execution
  - Ensure that files have a high enough code coverage by unit tests
  - Make sure all files follow the style rules of the repo
  - Make sure all files pass the linter stage
    - E.g., have no `pyright` warnings / errors
  - Triage and propose a solution for unit tests that are disabled
  - One-off refactoring and other code clean ups
  - Prototyping / "Tracer bullet" to one-shot implement something to see what is its
    impact and how it looks like
  - Explore research ideas

- Certain classes of tasks are run on a schedule rather than as `ai_task_queue.md`
  (which are mainly one-off tasks), e.g.,
  - Maintaining the documentation in sync with code
    - You want to run this every day or every commit
  - Making sure code coverage for the entire repo is high enough
    - You want to run this every day or every commit

## The Unit of Work

- In my workflow, the unit of work is something that corresponds to a GitHub issue

- Often it's implemented as a single Git branch + a GitHub PR
- Sometimes one GitHub issue can correspond to multiple Git branches / GitHub PR
  - E.g., the same refactoring can be done in "chunks" with several follow up PRs
    for both safety and ease of review

- In my workflow, a GitHub Issue is named using a fixed convention to keep it in sync
  with the corresponding branch(es) and PRs
  - E.g., For GitHub Issue `<num>` with title `<title>` (e.g., "Do this and that"),
    the Git branch and the PRs are named as `<Repo>_<Issue Number>_Do_this_and_that`,
    decorated with an `<id>` when there are multiple branches / PRs associated
    (e.g., `<Repo>_<Issue Number>_Do_this_and_that_<id>`

## Adding Tasks to the Queue

- My approach is to:
  - Add tasks to `Backlog` in `ai_task_queue.md` as they come up
  - Re-organize them over time
  - Triage their importance over time
  - Add / refine specs
  - Finally mark them as ready for execution when I feel they are clear enough

- Once in a while I grep the code base looking for TODOs that are suitable to be
  executed asynchronously
  - E.g., in `helpers` you can run:
    ```bash
    > rigtodo
    > rigtodo . py --todo ai_gp | tee cfile
    > vim -c "cfile cfile"
    ...
    ```
- Note that not all TODOs might be ready to go, so I manually review the list of
  potential tasks and add them manually to `Backlog` (and then `Ready`)

## Add Specs

- Besides prioritizing, I also want to make sure that the model has enough
  information to implement the code in the proper way, so I write enough specs to
  direct the model to do something in a way that would not surprise me at the
  time of the PR review

- I have several agent skills to help with managing the specs:
  - `/auto_task.create_specs`: create specs for a list of TODOs
  - `/auto_task.criticize`: to review and improve specs of tasks that will be
    executed asynchronously

## Feed Auto Tasks to Agent

- At this point each task conceptually has:
  - A description of the problem (e.g., title, bug description)
  - A solution (in terms of specs)
  - A complexity (low, medium, high)
  - An importance (low, medium, high)

- I keep improving `ai_task_queue.md` by reviewing and editing the specs, ranking
  issues by "complexity" and "importance"

- Once I see tasks that are in the `Ready` state, I pass it to the workers by running
  ```
  > git_create_issue_and_branch.py --gh_issue_title 'Implement TODOs' --gh_issue_body_file instr.md
  ```
  - This command automatically create a GitHub issue, a branch, and a PR all named
    following the convention, so that it's easy to relate one to the others
  - There is overlap with GitHub `gh` command to manage issues but personally I
    prefer to batch the issues in a GH, keep working / refining them, and then push
    everything to GitHub with a script

`./helpers_root/dev_scripts_helpers/ai/todo_janitor.template.md`

- This is the master list of what needs to be done

- The workflow is:
  - Maintain / replenish `ai_task_queue.md`
  - Create GitHub issues
  - Create a branch / PR for the agent to do the work
  - Let the agent fix the issues (running on GitHub infra, running regressions, etc)
    and creating a PR when ready
  - (Optional) Humans check out the branch to make changes manually
    - E.g., when explaining something to do to the agent takes longer than just doing
      it
  - Review PR and merge it
  - Close GH Issue
  - Update `ai_task_queue.md` with the ones that were done
    E.g., `./helpers_root/todo_janitor.prompt.update_plan.md`

- This is the same workflow I used to use with collaborators with the main difference
  that now AI agents are doing the work
  - Of course there is a bump in throughput and often in quality, at least for PR
    that are pure implementation and not design

- Other tasks can be executed on **GitHub**

  Attach the Git branch and the PR to the issue using a gh comment

  > gh pr comment

  Assign the task to Claude to let it go using the GH actions or use the Desktop

### An Alternative Local Flow

- Some tasks are best done **locally** and **interactively**
  - E.g., run the local regressions, more control and interactivity

- I use the same approach as above but working with AI agent in worktrees, which are
  managed automatically together with GH Issue, Git branch, GH PR using the same
  tool
  ```
  # Create a task, 
  > git_create_issue_and_branch.py --gh_issue_title 'Implement TODOs' --gh_issue_body_file instr.md --create_worktree
  > cd /Users/saggese/src/helpers1_worktree_1325; dev_scripts_helpers/thin_client/tmux.py --index 1325
  ```

- In some cases I want CC to commit directly in its branch (you know that I don't
like the agent do that while I am driving the development)

- Enable CC to commit (needed for local dev) using `control_cc_commit.py`

  ```
  claude> Implement the GitHub instructions from gh issue view 1325
  ```

- Commit after the first PR

- Review

- Run tests that were touched
  ```
  > i git_files --mode test_files --on-one-line --pbcopy
  > pytest_log $(pbpaste)
  #
  > pytest_log $(i git_files --mode test_files --on-one-line --only-print-files | remove_escape_chars.py -i -)
  ```

- .claude/todo_janitor.template.md

### Old instructions

//```
//# Create todo_janitor plan
//
//# Pick Issues to Fix
//
//## Make sure that the list is updated
//
//  ```
//  claude> Look at the last merged git PRs in master and in the current repo and mark the completed issues in plan.todo_janitor.md
//  ```
//
//## Pick an Issue and Create the Branch
//
//- Go in helpers1 tree (which is the one from which everything is orchestrated)
//
//- Pick an issue from `plan.todo_janitor.md` and create a `todo_janitor.issue.md`
//
//## Create CC Instructions
//
//- Create instructions for CC from `todo_janitor.instr.md`
//
//## Create the Branch / Worktree
//  ```
//  > create_git_worktree.py --gh_issue_title 'Clean up' --gh_issue_body_file todo_janitor.current_issue.md
//  > create_git_worktree.py --gh_issue_title "Rename invocations to sys_calls Throughout Codebase" --gh_issue_body body.txt --instr_file instr2.md
//
//  > create_git_worktree.py --gh_issue_id 1292 --instr_file instr2.md
//  ```
//
//> more instr2.md
//- Wait that all the checks are complete and passing
//  i gh_workflow_list
//
//- If the tests are passing, run all the tests locally
//  pytest_multi_build.py --target .
//
//- If there are not issues, then mark the PR as ready
//  gh pr  comment --body "All tests are passing"
//
//- Ask to review
//
//### Check that PR is ready to review
//
//> gh pr view
//gp_scratch_36 causify-ai/helpers#1330
//Draft • gpsaggese (GP Saggese) wants to merge 2 commits into master from gp_scratch_36 • about 4 minutes ago
//+2359 -836 • ✓ Checks passing
//
//### Update the CC task plan
//- Automate some of the work above
//  ```
//  orchestrate_task.py --plan ... --action
//
//  --action stage_todo calling create_git_worktree.py (
//      - create the body and instr.md
//      - update the todo
//  ```
//
//# Fix the issue
//- Go to helper...
//
//- git checkout HelpersTask1299_TODO_clean_up
//
//- Enable CC to commit
//  - Use .claude/cc_control
//```
//claude> Execute todo_janitor.template.md
//```
//
//## Commit the changes
//
//### [ ] Convert CC flow to script
//- Convert todo_janitor.template.md into a single script since CC doesn't
//  follow the directions
//
//````
