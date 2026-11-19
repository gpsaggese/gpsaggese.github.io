---
title: "Many Git Clients, One Branch: git_branch_copy and git_merge_master"
draft: true
authors:
    - gpsaggese
date: 2026-11-07
description:
categories:
    - Developer Tools
---

TL;DR: Squash-copying a shared branch into `master` and then merging `master`
back into every client causes rebase conflicts, and treating the branch as
disposable removes them

<!-- more -->

## The Setup

- I push from several Git clients to the same branch, e.g., `gp`
- Once in a while I land the work in `master`:
    - Run `invoke git_branch_copy` to copy `gp` into a PR branch
    - Merge the PR into `master`
    - Run `invoke git_merge_master` in every client to pick up the changes from
      `master`
- This looks fine on paper, but it produces frequent rebase conflicts
- This post explains:
    - What the two commands do
    - Why the combination conflicts
    - Three workflows to fix it, with pros and cons

<!--  rendered_images:begin -->
<!--  ```graphviz -->
<!--  digraph copy_merge_cycle { -->
<!--      bgcolor="transparent"; -->
<!--      pad="0.15"; -->
<!--      splines=spline; -->
<!--      nodesep=0.5; -->
<!--      ranksep=0.6; -->
<!--      rankdir=LR; -->
<!--   -->
<!--      node [shape=box, -->
<!--            style="rounded,filled", -->
<!--            penwidth=1.8, -->
<!--            fontname="Helvetica", -->
<!--            fontsize=12, -->
<!--            margin="0.22,0.14", -->
<!--            height=0.50]; -->
<!--   -->
<!--      edge [color="#A3B1C0", -->
<!--            penwidth=1.3, -->
<!--            arrowhead=vee, -->
<!--            arrowsize=0.75, -->
<!--            fontname="Helvetica", -->
<!--            fontsize=10, -->
<!--            fontcolor="#7B8794"]; -->
<!--   -->
<!--      // clients : green | branches : blue | commands : orange | problem : red -->
<!--      Clients [label="Client 1 .. N\nbranch gp", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"]; -->
<!--      Origin  [label="origin/gp\nN commits", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"]; -->
<!--      Copy    [label="git_branch_copy\nsquash", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"]; -->
<!--      Master  [label="master\n1 commit per copy", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"]; -->
<!--      Merge   [label="git_merge_master\nin every client", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"]; -->
<!--   -->
<!--      Clients -> Origin [label="  Push  "]; -->
<!--      Origin -> Copy [label="  Copy  "]; -->
<!--      Copy -> Master [label="  PR  "]; -->
<!--      Master -> Merge [label="  Fetch  "]; -->
<!--      Merge -> Clients [label="  N parallel merge commits  ", style=bold, penwidth=2.0, color="#D64545", fontcolor="#D64545", constraint=false]; -->
<!--  } -->
<!--  ``` -->
<!--  label=fig:copy-merge-cycle -->
<!--  caption=The copy and merge cycle: green is a client, blue is a branch, orange is a command, red is the step that creates the conflicts -->
<!--  rendered_images:end -->
<!--  render_images:begin -->

![The copy and merge cycle: green is a client, blue is a branch, orange is a command, red is the step that creates the conflicts](draft.how_to.Git_Branch_Copy_and_Merge_Master.md.figs/draft.how_to.Git_Branch_Copy_and_Merge_Master.1.png){#fig:copy-merge-cycle}

<!--  render_images:end -->

## What `git_branch_copy` Does

- The task lives in
  [`helpers/lib_tasks/lib_tasks_git.py`](https://github.com/causify-ai/helpers/blob/master/helpers/lib_tasks/lib_tasks_git.py)
- It copies the content of the current branch into a new branch without copying
  the commits, in 5 steps:
    1. Stage the untracked files, except the scratch files `tmp.*` and `*.log`
    2. Remove the remaining untracked files with `git clean -fd`
    3. Merge `master` into the current branch with
       `git_merge_master --abort-if-not-ff --no-auto-merge`, unless
       `--skip-git-merge-master` is passed
    4. Create the new branch from `master`, or switch to it if it exists
    5. Run `git merge --squash --ff <branch> && git reset HEAD`
- The result:
    - The new branch has the same files as the old branch
    - The changes are unstaged and uncommitted, so I choose what to commit
    - The history is not copied: N commits become 1
- Typical usage:
    ```bash
    > i git_branch_copy
    > i git_branch_copy --new-branch-name="CmTask5874_PR_flow_Images"
    > i git_branch_copy --skip-git-merge-master
    ```
- The original use case is landing a big branch as small PRs
    - Copy the branch, delete the files that do not belong in the PR, and commit
    - Removing code is easier than adding it chunk by chunk
- Step 3 uses `--abort-if-not-ff`, so the copy stops if merging `master` needs a
  real merge commit
    - After a squash copy lands in `master`, `gp` no longer fast-forwards to
      `master`
    - This is one reason I run `git_merge_master` in every client afterwards

## What `git_merge_master` Does

- The task wraps
  [`dev_scripts_helpers/git/git_merge_master.py`](https://github.com/causify-ai/helpers/blob/master/dev_scripts_helpers/git/git_merge_master.py)
- It runs 4 steps:
    1. Abort if the client is not clean
    2. Fetch `origin master` into the local `master` with
       `git fetch origin master:master`
    3. Run `git merge master`
    4. Commit if needed and run `git push`
- Useful flags:
    - `--abort-if-not-ff`: Fail unless the merge is a fast-forward
    - `--no-auto-merge`: Merge, but do not commit and push
    - `--dry-run`: Report which files would conflict, without touching the
      working directory
    - `--skip-fetch`: Do not fetch `master` first
- Preview the conflicts before merging:
    ```bash
    > i git_merge_master --dry-run
    ```
- The tradeoffs between merge, rebase, and squash are in
  `blog_posts/draft.how_to.i_git_merge_master.md`

## Why the Combination Conflicts

- Git tracks history by commit hash, not by content
    - The squash turns N commits on `gp` into 1 new commit with a new hash
    - `master` has the same changes as `gp`, but Git sees no link between them
    - E.g., `master` has `gp_14 (#1450)`, while `gp` still has the original
      `Update` commits
- Merging `master` into `gp` conflicts when `gp` moved after the copy
    - The squash side has the state at copy time
    - `gp` has that state plus later edits, so both sides changed the same lines
    - Git cannot know that the squash commit and the `Update` commits are the
      same work

<!--  rendered_images:begin -->
<!--  ```graphviz -->
<!--  digraph history_shape { -->
<!--      bgcolor="transparent"; -->
<!--      pad="0.15"; -->
<!--      splines=spline; -->
<!--      nodesep=0.4; -->
<!--      ranksep=0.6; -->
<!--      rankdir=LR; -->
<!--   -->
<!--      node [shape=box, -->
<!--            style="rounded,filled", -->
<!--            penwidth=1.8, -->
<!--            fontname="Helvetica", -->
<!--            fontsize=12, -->
<!--            margin="0.22,0.14", -->
<!--            height=0.50]; -->
<!--   -->
<!--      edge [color="#A3B1C0", -->
<!--            penwidth=1.3, -->
<!--            arrowhead=vee, -->
<!--            arrowsize=0.75, -->
<!--            fontname="Helvetica", -->
<!--            fontsize=10, -->
<!--            fontcolor="#7B8794"]; -->
<!--   -->
<!--      // gp commits : green | master commit : blue | merge : orange | link : red -->
<!--      Base [label="Base", fillcolor="#F7F9FB", color="#C7D0DA", fontcolor="#3A4A5C"]; -->
<!--      U1 [label="Update 1", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"]; -->
<!--      U2 [label="Update 2", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"]; -->
<!--      U3 [label="Update 3", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"]; -->
<!--      U4 [label="Update 4\n(after the copy)", fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"]; -->
<!--      S  [label="gp_14\n(squash on master)", fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"]; -->
<!--      M  [label="Merge master\nconflict", fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"]; -->
<!--   -->
<!--      Base -> U1 -> U2 -> U3 -> U4; -->
<!--      Base -> S; -->
<!--      U4 -> M; -->
<!--      S -> M; -->
<!--      U3 -> S [dir=none, style=dashed, color="#D64545", fontcolor="#D64545", constraint=false, label="  Same changes, different hash  "]; -->
<!--  } -->
<!--  ``` -->
<!--  label=fig:history-shape -->
<!--  caption=The squash commit on master and the Update commits on gp carry the same changes but share no hash: green is a gp commit, blue is the squash on master, orange is the merge that conflicts -->
<!--  rendered_images:end -->
<!--  render_images:begin -->

![The squash commit on master and the Update commits on gp carry the same changes but share no hash: green is a gp commit, blue is the squash on master, orange is the merge that conflicts](draft.how_to.Git_Branch_Copy_and_Merge_Master.md.figs/draft.how_to.Git_Branch_Copy_and_Merge_Master.2.png){#fig:history-shape}

<!--  render_images:end -->

- Every client repeats the merge on its own
    - Each client runs `git_merge_master` and creates its own merge commit
    - The merge commits have the same content but different hashes
    - The first push wins, and the other clients are rejected as
      non-fast-forward
- `pull.rebase=true` makes it worse
    - `git pull` rebases the local commits on top of `origin/gp`
    - A rebase drops merge commits and replays the commits they brought in
    - The replayed changes are already in `origin/gp` under other hashes, so the
      replay conflicts
- GitHub adds a second source of duplicates
    - A PR merged on GitHub lands in `master` as a new commit with a new hash
    - `gp` still has the original commits of the same PR
      <!-- TODO(gp): Check that the GitHub merge of the gp_N PRs uses squash, the log suggests it -->
- In short:
    - The squash hides the link between `gp` and `master`
    - N clients create N parallel merges of `master`
    - The rebase replays them one by one
      <!-- TODO(gp): Check the whole mechanism: it is inferred from the code, the history, and pull.rebase=true, and it was not reproduced in a scratch repo -->

## Three Workflows

| Workflow                               | History of `master`      | Conflicts in the clients | Main risk                         |
| :------------------------------------- | :----------------------- | :----------------------- | :-------------------------------- |
| Current: squash copy, merge everywhere | 1 commit per copy        | Frequent                 | Duplicate merges, rebase replays  |
| A: reset `gp` after each copy          | 1 commit per copy        | Expected to be rare      | `reset --hard` deletes local work |
| B: merge the real commits              | All the `Update` commits | Expected to be rare      | Noisy `master` history            |

### Current: Squash Copy, Merge Everywhere

- Pros:
    - `master` gets 1 clean commit per copy
    - `gp` keeps all the fine-grained commits
    - A big branch is easy to split into several PRs
- Cons:
    - `gp` and `master` share no history after the copy
    - Every client creates its own merge commit
    - `git pull` with `pull.rebase=true` replays commits that are already pushed

### A: Reset `gp` After Each Copy (Recommended)

- Core idea: `gp` is disposable
    - After the copy is merged into `master`, restart `gp` from `master`
    - Never merge `master` into a `gp` that still has the old history
- Steps:
    1. Commit, push, and pull in all clients, so nothing is lost
    2. Run `i git_branch_copy` in one client and merge the PR into `master`
    3. In the same client, restart `gp` from `master`:
        ```bash
        > git fetch origin
        > git checkout gp
        > git reset --hard origin/master
        > git push --force-with-lease origin gp
        ```
    4. In every other client, take the new `gp`:
        ```bash
        > git fetch origin
        > git checkout gp
        > git reset --hard origin/gp
        ```
- Pros:
    - There is no duplicated merge commit and no rebase over a merge
    - The squash is harmless, since `gp` restarts from the squash result
    - The `git_branch_copy` PR workflow stays the same
- Cons:
    - `reset --hard` deletes unpushed commits and edits
        - Push or `git stash` first, in every client
    - The force push rewrites `origin/gp`, so every client must reset
      <!-- TODO(gp): Check option A in a scratch repo with 2 clients before recommending it -->

### B: Merge the Real Commits

- Open the PR from `gp` itself and merge it with a merge commit, not a squash
    - `master` then contains the real `gp` commits
    - Merging `master` into `gp` is a fast-forward or a trivial merge
- Pros:
    - The histories of `gp` and `master` are linked
    - No copy step and no reset
- Cons:
    - `master` gets all the noisy `Update` commits

### Fix the Multi-Client Side in Any Case

- Let one designated client run `git_merge_master`
- The other clients only run `git pull --no-rebase`
- Set this in all the clients:
    ```bash
    > git config pull.rebase false
    > git config pull.ff only
    ```

    - `pull.ff only` fails loudly when the branches diverged, instead of
      creating a surprise rebase or merge
- Always pull before starting work in a client, and push often
- Pro: one merge commit instead of N parallel ones
- Con: it depends on discipline, and a diverged pull needs manual work

## Recovery from a Rebase in Conflict

- Run `git rebase --abort`
- Then run one of:
    ```bash
    > git pull --no-rebase
    > git reset --hard origin/gp
    ```

    - Use the second command only if nothing local is worth keeping
- Before pushing, check what would go to `origin/gp`:
    ```bash
    > git log origin/gp..gp
    ```

    - The commits ahead can include copies of `master` commits, and a push would
      put them on `origin/gp`
