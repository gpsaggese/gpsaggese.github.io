---
title: "How to Use the GitHub Stacked PRs Workflow"
authors:
  - gpsaggese
date: 2026-09-17
description:
draft: true
categories:
  - Developer Tools
---

TL;DR: Use `gh repo create` and the `gh stack` extension to turn a sequence of
dependent changes into a chain of linked, auto-retargeting pull requests

<!-- more -->

- This post covers the GitHub stacker PR workflow, which includes:
  - Building a stack of branches
  - Publishing it as linked PRs
  - Fixing an earlier layer
  - Cleaning up

## Prerequisites

- Install and authenticate the GitHub CLI:
  ```bash
  > brew install gh
  > gh auth login
  ```
- Install the `gh stack` extension, a one-time setup per machine:
  ```bash
  > gh extension install github/gh-stack
  ```
- Verify both are ready:
  ```bash
  > gh auth status
  > gh stack --help
  ```

## Creating a Scratch Repository

- If you want to use a temp repo instead of a real one you can follow the
  instructions below

- `gh repo create` can create an empty remote repository or turn an existing
  local one into a remote repository in the same call
- Create a fresh private repository and clone it locally:
  ```bash
  > gh repo create my-project --private --clone
  > cd my-project
  ```
- Or turn a local repository you already have into a new remote one:
  ```bash
  > gh repo create my-project --private --source=. --remote=origin --push
  ```
  - `--source=.`: use the current directory as the repository content
  - `--remote=origin`: name the git remote it creates
  - `--push`: push the current branch's commits right after creating the repo
- The `OWNER/` part of the name defaults to your own account when omitted, so
  `my-project` and `<your-user>/my-project` are equivalent

## Building a Stack of Branches

- `gh stack init` creates the bottom branch of a stack off the trunk (`main` by
  default, override with `--base`)
- `gh stack add` creates the next branch on top of whatever is currently
  checked out
- Example: a 3-layer feature split into schema, API, and UI changes:
  ```bash
  > gh stack init feature/step-1-schema
  # ... edit files ...
  > git add -A && git commit -m "Step 1: add database schema"

  > gh stack add feature/step-2-api
  # ... edit files ...
  > git add -A && git commit -m "Step 2: add API endpoint"

  > gh stack add feature/step-3-ui
  # ... edit files ...
  > git add -A && git commit -m "Step 3: add UI component"
  ```
- `gh stack init` also accepts every branch name at once, which creates or
  adopts all of them in a single command:
  ```bash
  > gh stack init feature/step-1-schema feature/step-2-api feature/step-3-ui
  ```

## Publishing the Stack as PRs

- `gh stack submit` pushes every branch in the stack and creates or updates its
  PR, wiring up the base branches automatically: Step 1 targets `main`, Step 2
  targets `feature/step-1-schema`, Step 3 targets `feature/step-2-api`
- In an interactive terminal it opens a single-screen editor to review titles,
  descriptions, and draft state before submitting
- Pass `--auto` to skip the editor and use auto-generated titles, which is what
  an agent-driven workflow typically wants:
  ```bash
  > gh stack submit --auto
  ```
- New PRs are created as drafts under `--auto` unless `--open` is also passed

## Viewing the Stack

- `gh stack view` lists every branch in the stack with its PR status:
  ```bash
  > gh stack view
  ```
  - `✓`: PR merged
  - `◎`: PR queued
  - `○`: PR open
  - `⚠`: needs rebase
- Add `--short` for a compact one-line-per-branch view, or `--json` for
  machine-readable output that a script or agent can parse

## Updating an Earlier PR and Cascading the Fix

- Review feedback often lands on a layer that other branches already build on
- Fix it in place, then let `gh stack` replay every branch above it instead of
  rebasing each one by hand:
  ```bash
  > gh stack checkout feature/step-1-schema
  # ... apply fix ...
  > git add -A && git commit -m "Fix: address review comment"

  > gh stack rebase --upstack
  > gh stack push
  ```
  - `gh stack checkout` accepts a stack number, a PR number, a PR URL, or a
    branch name
  - `gh stack rebase --upstack` rebases only the branches above the current
    one, from the current branch to the top of the stack
  - `gh stack push` pushes the stack's branches with a per-branch
    `--force-with-lease` check after the rebase
- `gh stack rebase` also takes `--downstack` (trunk to the current branch), or
  no flag at all to rebase the entire stack

## Keeping the Stack in Sync with the Trunk

- `gh stack sync` bundles the routine maintenance into a single command:
  ```bash
  > gh stack sync
  ```
- It fetches from the remote, fast-forwards the trunk, cascade-rebases every
  stack branch onto its updated parent, pushes everything atomically, and
  refreshes each PR's state from GitHub
- If a PR was added to the stack directly on GitHub, `sync` pulls its branch
  down and appends it to the local stack
- Pass `--prune` to delete local branches whose PR has already been merged

## Merging the Stack

- `gh stack merge` performs an atomic stack merge: every PR up to and
  including the one you target merges into the trunk in one all-or-nothing
  operation
- Merge only the bottom PR, which automatically retargets everything above it
  onto the trunk:
  ```bash
  > gh stack merge feature/step-1-schema --squash
  ```
- Merge the entire stack without an interactive prompt:
  ```bash
  > gh stack merge --yes --squash
  ```
- Choose the merge strategy with `--merge`, `--squash`, or `--rebase`
- Branch protection rules and required checks are enforced on every PR in the
  stack, not only the bottom one, and a stack merges cleanly into a merge queue
  when the repository uses one

## Navigating and Restructuring a Stack

- Jump between layers without typing branch names:
  ```bash
  > gh stack bottom
  > gh stack up
  > gh stack down
  > gh stack top
  > gh stack trunk
  ```
- `gh stack switch` opens an interactive picker of every branch in the stack
- `gh stack modify` opens a TUI to reorder, drop, fold, insert, or rename
  branches, applying every change together on save

## Adopting Existing Branches into a Stack

- Some workflows create branches before deciding to stack them, e.g., a helper
  script that creates one branch per GitHub issue
- `gh stack link` turns already-pushed branches, or already-open PRs, into a
  stack without needing local `gh stack` tracking history:
  ```bash
  > gh stack link feature/step-1-schema feature/step-2-api feature/step-3-ui
  ```
- Arguments are given bottom to top, and each one can be a branch name, a PR
  number, or a PR URL
- Branches without an existing PR get one created automatically with the
  correct base chaining

## Cleaning Up

- Remove a stack, both locally and on GitHub, once it is no longer needed:
  ```bash
  > gh stack unstack
  ```
  - Pass `--local` to only drop local tracking and leave the stack on GitHub
  - GitHub keeps any PR that is queued for merge or has auto-merge enabled, so
    an in-flight stack is left partially unstacked rather than broken
- Delete the whole scratch repository when the exercise is done:
  ```bash
  > gh repo delete my-project --yes
  ```
  - Deletion requires the `delete_repo` OAuth scope
    (`gh auth refresh -s delete_repo` grants it once)
  - Without an explicit repository argument, `--yes` is ignored and you are
    prompted for confirmation, since deletion cannot be undone

## Key Takeaways

- `gh stack init` / `add` / `submit` build and publish a stack in the same
  shape as the branches themselves: bottom to top
- `gh stack rebase --upstack` plus `gh stack push` is the whole fix for
  "I need to change something in an earlier PR without redoing the rest"
- `gh stack merge` treats the stack as one atomic unit, so a partial merge
  never leaves the trunk half-updated

| Command | Purpose |
| :------ | :------ |
| `gh repo create` | Create a new GitHub repository, optionally from a local one |
| `gh repo delete` | Permanently delete a GitHub repository |
| `gh extension install github/gh-stack` | Install the `gh stack` CLI extension |
| `gh stack init` | Start a new stack of branches off the trunk |
| `gh stack add` | Add a new branch on top of the current stack |
| `gh stack submit` | Push every branch and create or update its PR |
| `gh stack view` | Show the branches and PR status of the current stack |
| `gh stack checkout` | Switch to a branch by stack number, PR number, or name |
| `gh stack rebase` | Cascade-rebase the stack after a lower layer changes |
| `gh stack push` | Push the stack's active branches after a local rebase |
| `gh stack sync` | Fetch, rebase, push, and refresh PR state in one step |
| `gh stack merge` | Merge some or all of the stack, bottom-up, atomically |
| `gh stack top` / `bottom` | Jump to the top or bottom branch of the stack |
| `gh stack modify` | Reorder, drop, fold, or rename branches interactively |
| `gh stack link` | Turn already-pushed branches or PRs into a stack |
| `gh stack unstack` | Remove a stack, locally and/or on GitHub |
