# Summary

- This tutorial compares 3 ways to bring `main` into a feature branch that
  has accumulated several small commits:
    - Merge
    - Rebase
    - Squash, then merge
- All 3 scripts start from the same diverging history, built by `restart.sh`
  This tutorial focuses on why the same conflict can cost 1 resolution or 5,
  depending on the flow used

# The setup

- `restart.sh` builds a scratch repo at `/tmp/merge_flows_demo` with a
  diverging history:
    - `main` advances by 1 commit that edits a comment line in `shared.py`
    - `feature` branches off before that commit, then edits the _same_ line 5
      times, in 5 separate small commits (a stand-in for a branch with many
      tiny "Update" commits)
    ```bash
    > source restart.sh
    === main log ===
    d03ac43 Update
    760eb9c Initial commit
    === feature log ===
    201623e Update
    7b96485 Update
    7fbb9c5 Update
    da4cec2 Update
    cead04d Update
    760eb9c Initial commit
    ```
- Both branches touch the same 3 lines of `shared.py`, so every flow below
  hits a conflict; what differs is how many times

# Flow 1: merge

- Run `demo_1_merge.sh`, or execute the commands one at a time
- `git merge main` folds all of `main`'s and `feature`'s history together in
  one step, producing a single 3-way diff

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
    > echo '    # Step 2: normalize input rows (v5)' # (edit shared.py by hand)
    > git add shared.py
    > git commit -m "Merge main into feature"

    > git log --oneline --graph
    *   46c7500 Merge main into feature
    |\
    | * 201623e Update
    * | cead04d Update
    * | 7b96485 Update
    * | 7fbb9c5 Update
    * | da4cec2 Update
    * | d03ac43 Update
    |/
    * 760eb9c Initial commit
    ```

- History stays non-linear: the merge commit has 2 parents, and `feature`'s
  5 original commits are untouched

# Flow 2: rebase

- Run `demo_2_rebase.sh`, or execute the commands one at a time
- `git rebase main` replays `feature`'s 5 commits one at a time on top of
  `main`, instead of merging the two histories in one step
    ```bash
    > git checkout feature
    > git rebase main
    Rebasing (1/5)
    Auto-merging shared.py
    CONFLICT (content): Merge conflict in shared.py
    error: could not apply ...: Update
    ```
- Because every one of the 5 commits edits the same line `main` also
  changed, the rebase stops on every single commit: 5 conflicts, not 1

    ```bash
    > cat shared.py
    def process(rows):
        # Step 1: validate input
    <<<<<<< HEAD
        # Step 2: normalize input rows (v1)
    =======
        # Step 2: normalize (v2)
    >>>>>>> ...: Update
        # Step 3: dedupe
        # Step 4: write output
        return rows

    > git add shared.py
    > git rebase --continue
    Rebasing (2/5)
    Auto-merging shared.py
    CONFLICT (content): Merge conflict in shared.py
    ...
    ```

    - Resolve, `git add`, `git rebase --continue`; repeat for commits 3, 4, 5
    - `rerere` (if enabled) does not help here: each of the 5 hunks has
      different content (`v1`, `v2`, `v3`, ...), so there is no repeated hunk
      to replay a cached resolution for

    ```bash
    > git log --oneline --graph
    * 6d24cfb Update
    * d6a1c5a Update
    * fe5a277 Update
    * fce5183 Update
    * 68b6833 Update
    * 71b1c4e Update
    * 4460fa9 Initial commit
    ```

- History is linear, at the cost of 5 manual conflict resolutions instead of
  1; this is the pattern behind "why does merging master take forever", when
  a branch has accumulated many small commits that overlap with upstream
  changes and gets rebased instead of merged

# Flow 3: squash, then merge

- Run `demo_3_squash_then_merge.sh`, or execute the commands one at a time
- Collapse `feature`'s 5 commits into 1 before merging, using
  `git reset --soft` back to the branch point:

    ```bash
    > git checkout feature
    > BASE=$(git merge-base feature main)
    > git reset --soft "$BASE"
    > git commit -m "Update (squashed)"

    > git log --oneline feature
    8355d64 Update (squashed)
    5513484 Initial commit
    ```

- Merging `main` now behaves exactly like flow 1: 1 conflict, resolved once

    ```bash
    > git merge main -m "Merge main into feature"
    Auto-merging shared.py
    CONFLICT (content): Merge conflict in shared.py

    > # resolve, then:
    > git add shared.py
    > git commit -m "Merge main into feature"

    > git log --oneline --graph
    *   682339e Merge main into feature
    |\
    | * 8355d64 Update
    * | 9e2ef14 Update (squashed)
    |/
    * 5513484 Initial commit
    ```

- Same conflict count as a plain merge, and `feature`'s own history is now a
  single clean commit instead of 5 "Update" commits

# Comparison

| Flow               | Conflicts to resolve        | Resulting history                               | Rewrites existing commits                   |
| :----------------- | :-------------------------- | :---------------------------------------------- | :------------------------------------------ |
| Merge              | 1 (one 3-way diff)          | Non-linear (merge commit)                       | No                                          |
| Rebase             | Up to 1 per replayed commit | Linear                                          | Yes (every replayed commit gets a new hash) |
| Squash, then merge | 1                           | Non-linear, but `feature` collapses to 1 commit | Yes (`feature`'s own commits only)          |

- The takeaway: conflict cost under rebase scales with the number of
  commits that overlap with upstream changes, not with the size of the
  overlap itself
    - A feature branch with 1 commit and a feature branch with 50 "Update"
      commits touching the same lines cost the same to merge, but the second
      one can cost 50 times more to rebase
