---
title: "Merge, Rebase, or Squash: Choosing How to Catch Up with Master"
draft: true
authors:
  - gpsaggese
date: 2026-08-23
description:
categories:
  - Developer Tools
---

TL;DR: A short explanation of different ways of merging master into your feature
branch with Git and then an explanation of the `invoke git_merge_master` workflow
to automate the Git approach (merge!) that I tend to use, with the best tradeoff
for me

<!-- more -->

## The problem

- _"Merge master into my branch"_ is a common operation, but Git offers 3 different
  flows for it:
    - Merge
    - Rebase
    - Squash, then merge
- They produce different history shapes and a different number of conflicts to
  resolve for the exact same overlapping changes

- A feature branch with many small commits (a common pattern: commit early,
  commit often, squash later) is the worst case for one of these flows and a
  non-issue for the other two

## What `invoke git_merge_master` actually runs

- The task lives in
  [`helpers/lib_tasks/lib_tasks_git.py`](https://github.com/causify-ai/helpers/blob/master/helpers/lib_tasks/lib_tasks_git.py)
- Stripped to its essentials, it does 4 things:
    ```python
    hgit.is_client_clean(dir_name=".", abort_if_not_clean=abort_if_not_clean)
    if not skip_fetch:
        git_fetch_master(ctx, submodules=submodules)  # git fetch origin master:master
    cmd = "git merge master"
    if abort_if_not_ff:
        cmd += " --ff-only"
    hltltaut.run(ctx, cmd)
    if auto_merge:
        hltltaut.run(ctx, 'git commit -am "Merge master" && git push')
    ```
- In plain terms:
  - Check that the Git client is clean and refuse to run on a dirty tree
  - Fetch `origin/master` into a local `master` ref
  - Run a single `git merge master`
  - If that succeeds cleanly, commit and push automatically

- It is deliberately a one-shot merge: if the merge hits a conflict, the
  command exits non-zero and the task stops, leaving the conflict for you to
  resolve by hand

## Three flows for catching up with master

### Merge

- `git merge master` combines the two histories in one step and produces a
  single 3-way diff between the common ancestor, your branch, and `master`
- Every overlapping line is compared exactly once, regardless of how many
  commits produced it on either side
- Cost: 1 conflict resolution per overlapping region, no matter how many
  commits touch it
- Trade-off: history becomes non-linear (a merge commit with 2 parents), and
  every prior commit on the feature branch is preserved as-is

### Rebase

- `git rebase master` replays your branch's commits one at a time on top of
  `master`, as if you had written them after the fact
- Every replayed commit is diffed against `master` independently, so if 5
  commits all touch the same overlapping region, the rebase can stop 5
  times, not once
- `rerere` (reuse recorded resolution) can save you from retyping an
  identical fix, but only if the exact same conflict hunk reappears; commits
  that each change the line slightly differently still each need a fresh
  resolution
- Cost: up to 1 conflict resolution per commit that overlaps with upstream
  changes
- Trade-off: history stays linear, but every replayed commit gets a new
  hash, and the process is a lot more manual work when the branch has
  accumulated many small commits

### Squash, then merge

- Collapse the feature branch's own commits into 1 first
  (`git reset --soft <merge-base>` followed by a single commit, or an
  interactive rebase with `squash`/`fixup`), then merge `master`
- The subsequent merge behaves exactly like flow 1: 1 conflict resolution,
  because there is now only 1 commit's worth of changes to diff against
  `master`
- Cost: 1 conflict resolution, same as a plain merge
- Trade-off: rewrites the feature branch's own commit hashes (fine if it has
  not been shared, risky if others have already pulled it), and loses the
  fine-grained commit history in exchange for a clean one

## Side by side

| Flow               | Conflicts for N overlapping commits | Resulting history                            | Rewrites commits                   |
| :----------------- | :---------------------------------- | :------------------------------------------- | :--------------------------------- |
| Merge              | 1                                   | Non-linear                                   | No                                 |
| Rebase             | Up to N                             | Linear                                       | Yes, every replayed commit         |
| Squash, then merge | 1                                   | Non-linear, but branch collapses to 1 commit | Yes, only the branch's own commits |

## A reproducible example

- A full walkthrough and example are in
  [`data605/tutorials/tutorial_git_merge_flows/`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/data605/tutorials/tutorial_git_merge_flows/)
- `restart.sh` builds a scratch repo where
  - `main` advances by 1 commit that edits a comment line
  - `feature` branches off earlier and edits the _same_ line 5 times, in 5 separate
    small commits

- Running the 3 flows against that identical starting point:
    ```bash
    > ./demo_1_merge.sh          # 1 conflict
    > ./demo_2_rebase.sh         # 5 conflicts, one per replayed commit
    > ./demo_3_squash_then_merge.sh   # 1 conflict, feature collapses to 1 commit
    ```
- The rebase run stops on every single commit:
    ```text
    Rebasing (1/5)
    CONFLICT (content): Merge conflict in shared.py
    ... resolve, git add, git rebase --continue ...
    Rebasing (2/5)
    CONFLICT (content): Merge conflict in shared.py
    ... repeats through (5/5) ...
    ```
- The merge and squash-then-merge runs each stop exactly once, on the same
  file, with the same underlying overlap

## Which one to use

- `invoke git_merge_master` merges which is the right default for a shared feature
  branch
  - It gets 1 conflict resolution regardless of how many commits accumulated on
    either side
- Rebase pays off when the branch is short-lived, not yet pushed, or you
  specifically want linear history and are prepared for the conflict cost to
  scale with commit count
- Squash first when a branch has grown many "Update"-style commits that
  should not go into the permanent history anyway; it gets the low conflict
  cost of a merge and a cleaner log, at the price of rewriting the branch's
  own commits
