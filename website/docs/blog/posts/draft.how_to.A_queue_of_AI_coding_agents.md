---
title: "A Queue of AI Coding Agents"
draft: true
authors:
  - gpsaggese
date: 2026-08-20
description:
categories:
  - AI Coding
  - Developer Tools
---

TL;DR: My workflow to run a queue of AI agents that fix issues asynchronously

<!-- more -->

- This blog is a short description of my workflow to organize and run a queue of AI
  agents fixing issues asynchronously
- My current coding set up is:
  - Source control: GitHub (GH) / Git
  - Agent harness: Claude Code (CC)
  - Model: Anthropic models
  - `helpers` repo
- None of these assumptions are strictly needed: you can generalize to other code
  hosting / control systems, agent harnesses, and models

# Overview

- The core components are:
  - **Queue**: A set of GitHub Issues that should be implemented asynchronously
    (identified by a specific label or assigned to a specific user, e.g., `claude`)
  - **Trigger**: a GitHub Action that runs when an issue is labeled or assigned,
    marking it as part of the Queue
  - **Agent**: Claude Code, running in the GH Actions, which
    - Reads the issue
    - Writes the code (running regressions, linting, etc)
    - Opens a PR for human review and merge
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

- Each task has:

  ```
  Title:
  Problem:
  Solution:
  Implementation complexity:
  Importance:
  ```

- Potential targets for asynchronous tasks are:
  - Todos in the codebase
    - E.g., marked with `TODO(ai_gp)` to communicate that they are assigned by my
      user for AI execution
  - Ensure that files have a high enough code coverage by unit tests
  - Make sure all files follow the style rules of the repo
  - Make sure all files pass the linter stage
    - E.g., have no `pyright` warnings / errors
  - Triage and propose a solution for unit tests that are disabled
  - One-off refactoring and other code clean ups
  - Prototyping / "Tracer bullet" to one-shot implement something to see what its
    impact is and how it looks
  - Explore research ideas
- Certain classes of tasks are run on a schedule rather than through
  `ai_task_queue.md` (which is mainly for one-off tasks), e.g.,
  - Maintaining the documentation in sync with code
    - You want to run this every day or every commit
  - Making sure code coverage for the entire repo is high enough
    - You want to run this every day or every commit

## The Unit of Work

- In my workflow, the unit of work is something that corresponds to a GitHub issue
- Often it's implemented as a single Git branch + a GitHub PR
- Sometimes one GitHub issue can correspond to multiple Git branches / GitHub PRs
  - E.g., the same refactoring can be done in "chunks" with several follow-up PRs for
    both safety and ease of review
- In my workflow, a GitHub Issue is named using a fixed convention to keep it in sync
  with the corresponding branch(es) and PRs
  - E.g., for GitHub Issue `<num>` with title `<title>` (e.g., "Do this and that"),
    the Git branch and the PRs are named `<Repo>Task<num>_Do_this_and_that`,
    decorated with an `<id>` when there are multiple branches / PRs associated (e.g.,
    `<Repo>Task<num>_Do_this_and_that_<id>`)
  - See [My Agentic Engineering Flow](/blog/my-agentic-engineering-flow) for the full
    spec format and the skills that turn a queued task into a PR

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
    ```

- Note that not all TODOs are ready to go, so I manually review the list of potential
  tasks and add them to `Backlog` (and then `Ready`)

## Add Specs

- Besides prioritizing, I also want to make sure that the model has enough
  information to implement the code in the proper way, so I write enough specs to
  direct the model to do something in a way that would not surprise me at the time of
  the PR review
- I have several agent skills to help with managing the specs:
  - `/auto_task.create_specs_from_todos`: create specs for a list of TODOs
  - `/auto_task.criticize`: review and improve the specs of tasks that will be
    executed asynchronously

## Feed Auto Tasks to the Agent

- At this point each task conceptually has:
  - A description of the problem (e.g., title, bug description)
  - A solution (in terms of specs)
  - A complexity (low, medium, high)
  - An importance (low, medium, high)
- I keep improving `ai_task_queue.md` by reviewing and editing the specs, ranking
  issues by "complexity" and "importance"
- Once a task is in the `Ready` state, I pass it to the workers by running:

  ```
  > git_create_issue_and_branch.py --gh_issue_title 'Implement TODOs' --gh_issue_body_file instr.md
  ```

  - This command automatically creates a GitHub issue, a branch, and a PR all named
    following the convention, so that it's easy to relate one to the others
  - There is overlap with the GitHub `gh` command to manage issues, but I prefer to
    batch the issues, keep working / refining them, and then push everything to
    GitHub with a script
- The workflow is:
  - Maintain / replenish `ai_task_queue.md`
  - Create GitHub issues
  - Create a branch / PR for the agent to do the work
  - Let the agent fix the issues (running on GitHub infra, running regressions, etc)
    and create a PR when ready
  - (Optional) I check out the branch to make changes manually
    - E.g., when explaining something to the agent takes longer than just doing it
  - Review the PR and merge it
  - Close the GH Issue
  - Update `ai_task_queue.md` to move the finished tasks to `Done`
- This is the same workflow I used with human collaborators, with the main difference
  that now AI agents are doing the work
  - Of course there is a bump in throughput and often in quality, at least for PRs
    that are pure implementation and not design

### Running Tasks on GitHub

- Other tasks are executed entirely on **GitHub**, without a local checkout:
  - Attach the Git branch and the PR to the issue with a `gh pr comment`
  - Assign the task to Claude and let it run, either through GitHub Actions or the
    Claude Desktop app

### An Alternative Local Flow

- Some tasks are best done **locally** and **interactively**, e.g., to run the local
  regressions with more control and interactivity
- I use the same approach as above, but I work with the agent in a Git worktree that
  is managed automatically together with the GH Issue, Git branch, and GH PR, using
  the same tool:

  ```bash
  > git_create_issue_and_branch.py --gh_issue_title 'Implement TODOs' \
      --gh_issue_body_file instr.md --create_worktree
  > cd <worktree_path>; dev_scripts_helpers/thin_client/tmux.py --index <issue_num>
  ```

- By default I keep the agent from committing on its own while I am driving the
  development, since I want to review each change before it lands
  - When I do want CC to commit directly (useful for unattended local runs), I enable
    it explicitly with `control_cc_commit.py`
- Once the agent finishes a chunk of work, I:
  - Review the diff
  - Run the tests that were touched:

    ```bash
    > i git_files --mode test_files --on-one-line --pbcopy
    > pytest_log $(pbpaste)
    ```

  - Commit, push, and let the PR checks run
