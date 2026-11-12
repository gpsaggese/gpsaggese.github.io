# Summary

- This tutorial compares 3 ways to bring `main` (or `master`) into a feature branch
  that has accumulated several small commits:
  - Merge
  - Rebase
  - Squash, then merge
- All 3 scripts start from the same diverging history, built by `restart.sh`
- This tutorial focuses on why the same conflict can cost 1 resolution or 5,
  depending on the flow used

# The Setup

- ```
  > export GIT_ROOT=<your repo>
  # E.g.,
  # > export GIT_ROOT=/Users/saggese/src/umd_classes1

  > cd $GIT_ROOT
  ```

// TODO(ai_gp): Add a link to the .sh files in the repo
- `restart.sh` builds a scratch repo at `/tmp/merge_flows_demo` with a diverging
  history:
  - `main` advances by 1 commit that edits a comment line in `shared.py`
  - `feature` branches off before that commit, then edits the _same_ line 5 times, in
    5 separate small commits (a stand-in for a branch with many tiny "Update"
    commits)

  ```bash
  > cd $GIT_ROOT; source data605/tutorials/tutorial_git_merge_flows/restart.sh
  === phase: reset the scratch repo directory ===
  === phase: initialize the git repository and identity ===
  Initialized empty Git repository in /private/tmp/merge_flows_demo/.git/
  === phase: create the initial commit and branches ===
  [main (root-commit) e576906] Update #1
   1 file changed, 6 insertions(+)
   create mode 100644 shared.py
  Switched to a new branch 'feature'
  === phase: advance main with one commit ===
  Switched to branch 'main'
  [main e44c0a4] Update #2
   1 file changed, 1 insertion(+), 1 deletion(-)
  Switched to branch 'feature'
  === phase: advance feature with five commits on the same line ===
  [feature 3c34f2f] Update #3
   1 file changed, 1 insertion(+), 1 deletion(-)
  [feature 92eb76b] Update #4
   1 file changed, 1 insertion(+), 1 deletion(-)
  [feature 7c94e1c] Update #5
   1 file changed, 1 insertion(+), 1 deletion(-)
  [feature c194717] Update #6
   1 file changed, 1 insertion(+), 1 deletion(-)
  [feature 74ef576] Update #7
   1 file changed, 1 insertion(+), 1 deletion(-)
  ```
- The history looks like:
  ```
  === phase: show the two starting histories ===
  === main log ===
  e44c0a4 (main) Update #2
  e576906 Update #1
  === feature log ===
  74ef576 (HEAD -> feature) Update #7
  c194717 Update #6
  7c94e1c Update #5
  92eb76b Update #4
  3c34f2f Update #3
  e576906 Update #1
  ```

- Both branches touch the same 3 lines of `shared.py`, so every flow below hits a
  conflict; what differs is how many times

# Flow 1: Merge

- Run `demo_1_merge.sh`, or (better) execute the commands one at a time
- `git merge main` folds all of `main`'s and `feature`'s history together in one
  step, producing a single 3-way diff

// TODO(ai_gp): Explain which is HEAD and which is main when we merge
- Run script:
  ```
  > cd $GIT_ROOT; source data605/tutorials/tutorial_git_merge_flows/demo_1_merge.sh
  ```

- Or by hand
  ```bash
  > git checkout feature

  > git merge main -m "Merge main into feature"
  Auto-merging shared.py
  CONFLICT (content): Merge conflict in shared.py
  Automatic merge failed; fix conflicts and then commit the result.

  > git status -s
  UU shared.py

  > cat shared.py
  def process(rows):
      # Step 1: validate input
  <<<<<<< HEAD
      # Step 2: normalize (v5)
  =======
      # Step 2: normalize input rows
  >>>>>>> main
      # Step 3: dedupe
      # Step 4: write output
      return rows
  ```

- 1 conflict, in 1 file, resolved once

  ```bash
  # Edit shared.py by hand to resolve conflict and keeping the change in HEAD.
  ...

  > cat shared.py
  def process(rows):
      # Step 1: validate input
      # Step 2: normalize (v5)
      # Step 3: dedupe
      # Step 4: write output
      return rows

  > git add shared.py

  > git commit -m "Merge main into feature"

  > git log --oneline --graph
  *   fbced02 (HEAD -> feature) Merge main into feature
  |\
  | * e44c0a4 (main) Update #2
  * | 74ef576 Update #7
  * | c194717 Update #6
  * | 7c94e1c Update #5
  * | 92eb76b Update #4
  * | 3c34f2f Update #3
  |/
  * e576906 Update #1
  ```

- History is non-linear
  - The merge commit has 2 parents
  - `feature`'s 5 original commits are untouched

# Flow 2: Rebase

- Run `demo_2_rebase.sh`, or execute the commands one at a time
- `git rebase main` replays `feature`'s 5 commits one at a time on top of `main`,
  instead of merging the two histories in one step

- Run script:
  ```
  > cd $GIT_ROOT; source data605/tutorials/tutorial_git_merge_flows/demo_1_merge.sh
  ```

// TODO(ai_gp): Explain which is HEAD and which is main when we rebase

- Or by hand:

  ```bash
  > git checkout feature

  > git rebase main
    Rebasing (1/5)
    Auto-merging shared.py
    CONFLICT (content): Merge conflict in shared.py
    error: could not apply 81c6fa6... Update #3
  ```

- Because every one of the 5 commits edits the same line `main` also changed, the
  rebase stops on every single commit: 5 conflicts, not 1
  ```
  > git status
  interactive rebase in progress; onto f229123
  Last command done (1 command done):
     pick 352dbf5 Update #3
  Next commands to do (4 remaining commands):
     pick 54b5aae Update #4
     pick aa9314f Update #5
    (use "git rebase --edit-todo" to view and edit)
  You are currently rebasing branch 'feature' on 'f229123'.
    (fix conflicts and then run "git rebase --continue")
    (use "git rebase --skip" to skip this patch)
    (use "git rebase --abort" to check out the original branch)

  You are in a sparse checkout with 100% of tracked files present.

  Unmerged paths:
    (use "git restore --staged <file>..." to unstage)
    (use "git add <file>..." to mark resolution)
          both modified:   shared.py
  ```
- Note that Git says that there are 5 conflicts.
  ```
  Last command done (1 command done):
  ...
  Next commands to do (4 remaining commands):
  ```

- Fix the conflict
  ```bash
  > cat shared.py
    def process(rows):
        # Step 1: validate input
    <<<<<<< HEAD
        # Step 2: normalize input rows
    =======
        # Step 2: normalize (v1)
    >>>>>>> 81c6fa6 (Update #3)
        # Step 3: dedupe
        # Step 4: write output
        return rows
  ```
- Note that the updates from the branch are now not from `HEAD` like in the merge
  situation

  ```
  > git add shared.py

  > git rebase --continue
  Rebasing (2/5)
  Auto-merging shared.py
  CONFLICT (content): Merge conflict in shared.py
  error: could not apply c54b92f... Update #4
  ...
  ```

- Resolve choosing the branch, `git add`, `git rebase --continue`; repeat for commits
  3, 4, 5
- `rerere` (if enabled) does not help here: each of the 5 hunks has different
  content (`v1`, `v2`, `v3`, ...), so there is no repeated hunk to replay a cached
  resolution for

  ```bash
  > git log --oneline --graph
    * 0d4ef1e Update #7
    * 5226161 Update #6
    * 557d6e3 Update #5
    * 45f4c28 Update #4
    * b10d883 Update #3
    * f8da7eb Update #2
    * c21e97e Update #1
  ```

- History is linear, at the cost of 5 manual conflict resolutions instead of 1
  - This is the pattern behind "why does merging master take forever", when a branch
    has accumulated many small commits that overlap with upstream changes and gets
    rebased instead of merged

# Flow 3: Squash, Then Merge

- Run `demo_3_squash_then_merge.sh`, or execute the commands one at a time
- Collapse `feature`'s 5 commits into 1 before merging, using `git reset --soft` back
  to the branch point:

  ```bash
  > git checkout feature

  > BASE=$(git merge-base feature main)
  > git reset --soft "$BASE"
  > git commit -m "Update (squashed)"

  > git log --oneline feature
  ee206d6 Update (squashed)
  4c64853 Update #1
  ```

- Merging `main` now behaves exactly like flow 1: 1 conflict, resolved once

  ```bash
  > git merge main -m "Merge main into feature"
    Auto-merging shared.py
    CONFLICT (content): Merge conflict in shared.py

  > # Resolve, then:
  > git add shared.py
  > git commit -m "Merge main into feature"

  > git log --oneline --graph
  *   3f4f74c Merge main into feature
  |\
  | * c763b3e Update #2
  * | ee206d6 Update (squashed)
  |/
  * 4c64853 Update #1
  ```

- Same conflict count as a plain merge, and `feature`'s own history is now a single
  clean commit instead of 5 "Update" commits

# Comparison

| Flow               | Conflicts to resolve        | Resulting history                               | Rewrites existing commits                   |
| :----------------- | :-------------------------- | :---------------------------------------------- | :------------------------------------------ |
| Merge              | 1 (one 3-way diff)          | Non-linear (merge commit)                       | No                                          |
| Rebase             | Up to 1 per replayed commit | Linear                                          | Yes (every replayed commit gets a new hash) |
| Squash, then merge | 1                           | Non-linear, but `feature` collapses to 1 commit | Yes (`feature`'s own commits only)          |

- The takeaway: conflict cost under rebase scales with the number of commits that
  overlap with upstream changes, not with the size of the overlap itself
  - A feature branch with 1 commit and a feature branch with 50 "Update" commits
    touching the same lines cost the same to merge, but the second one can cost 50
    times more to rebase
