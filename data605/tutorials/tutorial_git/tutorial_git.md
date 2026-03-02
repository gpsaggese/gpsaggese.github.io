# Summary

- This tutorial covers Git basics including:
  - Creating and cloning repositories
  - Daily operations (adding, committing files)
  - Working with remotes
  - Branching and merging
  - Resolving merge conflicts

# Starting a Git project

## Create a repo from scratch

- Create a Git repo from scratch or from some local code
  ```bash
  > mkdir /tmp/git_test
  > cd /tmp/git_test
  > git init
  > ls -1 .git/
  HEAD
  config
  description
  hooks
  info
  objects
  refs
  ```
- You can experiment with Git in this scratch repo, but everything will be local
  - You need a remote repo (e.g., on GitHub) to make it more interesting and
    realistic

## Clone class project

- Clone a project, e.g., the class project from
  `https://github.com/gpsaggese/umd_classes`
  - The tutorial uses SSH (`git@github.com:...`): this requires an SSH key
    pair set up with GitHub; if you only have HTTPS access use
    `https://github.com/gpsaggese/umd_classes.git` instead
  ```bash
  > cd /tmp
  > git clone git@github.com:gpsaggese/umd_classes.git /tmp/umd_classes_tmp
  Cloning into '/tmp/umd_classes_tmp'...
  Warning: Permanently added 'github.com,140.82.114.4' (ECDSA) to the list of known hosts.
  remote: Enumerating objects: 157, done.
  remote: Counting objects: 100% (157/157), done.
  remote: Compressing objects: 100% (103/103), done.
  remote: Total 157 (delta 65), reused 132 (delta 43), pack-reused 0
  Receiving objects: 100% (157/157), 5.09 MiB | 33.43 MiB/s, done.
  Resolving deltas: 100% (65/65), done.
  ```

- `git` downloads the `.git` project and creates the "working tree" (a working
  copy of the project)
  ```bash
  > cd /tmp/umd_classes_tmp
  > ls -1
  Dockerfile
  LICENSE
  README.md
  dev_scripts
  project_template
  gp
  lectures
  projects

  > ls -1 .git
  HEAD
  config
  description
  hooks
  index
  info
  logs
  objects
  packed-refs
  refs
  ```

- The project is clean:
  ```bash
  > git status
  On branch main
  Your branch is up to date with 'origin/main'.

  You are in a sparse checkout with 100% of tracked files present.

  nothing to commit, working tree clean
  ```
  - Note: "sparse checkout" is an advanced Git feature not relevant here; ignore
    that line

- You can restore the repo to the initial state with:
  - First set `GIT_ROOT` to point to where you cloned this class repo, e.g.:
    ```bash
    > export GIT_ROOT=/path/to/umd_classes
    ```
  - Then run:
    ```bash
    > source $GIT_ROOT/tutorials/tutorial_git/restart.sh
    ```
  which in practice corresponds to:
  ```bash
  > rm -rf /tmp/umd_classes_tmp

  > git clone git@github.com:gpsaggese/umd_classes.git /tmp/umd_classes_tmp
  Cloning into '/tmp/umd_classes_tmp'...
  Warning: Permanently added 'github.com' (ED25519) to the list of known hosts.
  ```

# Daily use

## Adding a file

- Navigate into the cloned repo first:
  ```bash
  > cd /tmp/umd_classes_tmp
  ```

- The staging area (also called the index) is a buffer between your working
  directory and the repo history
  - Files must be explicitly staged with `git add` before they are included in
    a commit
  - This lets you craft commits precisely, even if multiple files were changed

- You can add a file
  ```bash
  > echo "print('hello')" >hello.py
  > python hello.py
  hello
  > git status
  On branch main
  Your branch is up to date with 'origin/main'.

  You are in a sparse checkout with 100% of tracked files present.

  Untracked files:
  (use "git add <file>..." to include in what will be committed)
  hello.py

  nothing added to commit but untracked files present (use "git add" to track)
  ```
- Now there is a file in the working directory that Git is not tracking
- Adding it to the staging area
  ```bash
  > git add hello.py
  > git status
  On branch main
  Your branch is up to date with 'origin/main'.

  You are in a sparse checkout with 100% of tracked files present.

  Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
  new file:   hello.py
  ```

## Commit a file

- Look at how the history changes (your output might change depending on where
  you are in the history of the repo):
  ```bash
  # Check history.
  > git log --graph --oneline -4
  * a349849 (HEAD -> main, origin/main, origin/HEAD) Checkpoint
  * f85c019 Checkpoint
  * 20734cf Checkpoint
  * e36d971 Checkpoint
  ```
  - `HEAD -> main`: the commit your working directory currently reflects,
    pointing at your local `main` branch
  - `origin/main`: where the remote (`origin`, i.e., GitHub) thinks `main` is
  - `origin/HEAD`: the default branch on the remote server
  - After a commit, `HEAD -> main` advances while `origin/main` stays behind
    until you push
  ```bash
  # Commit locally.
  # `-a` = stage all tracked modified files; `-m` = inline commit message.
  > git commit -am "Add hello.py"
  [main f919311] Add hello.py
  1 file changed, 1 insertion(+)
  create mode 100644 hello.py

  # Check history.
  > git log --graph --oneline -4
  * f919311 (HEAD -> main) Add hello.py
  * a349849 (origin/main, origin/HEAD) Checkpoint
  * f85c019 Checkpoint
  * 20734cf Checkpoint
  ```
- You see that a new node in the commit graph was added and the pointers are
  moved

# Git remote

- Check for the remote
  ```bash
  > git remote -v
  origin  git@github.com:gpsaggese/umd_classes.git (fetch)
  origin  git@github.com:gpsaggese/umd_classes.git (push)
  ```
  - `origin` is the default name Git gives to the remote you cloned from

- Get the data you don't have
  ```bash
  > git fetch

  # Fetch and integrate.
  > git pull

  # Fetch, stash any local changes, integrate, then restore local changes.
  > git pull --autostash
  ```
  - `git fetch` downloads new commits from the remote but does not touch your
    working directory
  - `git pull` = `git fetch` + `git merge` (or rebase, depending on config)
  - `--autostash`: temporarily stashes uncommitted local changes before pulling
    and restores them afterward, preventing conflicts with incoming changes
- This won't make a difference unless a commit went in between the time you
  cloned and fetched (which is very unlikely)

# Branching and merging

## Work on main

- You can execute the script `work_on_main.sh` or (better) execute the command
  line-by-line:
  - `git status -s`: `-s` = short format, shows one line per file instead of
    the verbose default output
  ```bash
  > ls
  Dockerfile LICENSE README.md dev_scripts project_template gp lectures projects

  > git status -s

  > touch work_main.py

  > git status -s
  ?? work_main.py

  > git add work_main.py

  > git status -s
  A  work_main.py

  > git log --graph --oneline -3
  * 38affbd (HEAD -> main, origin/main, origin/HEAD) Checkpoint
  * a78fa6a Checkpoint
  * 36b9526 Checkpoint

  > git commit -am 'Add work_main.py'
  [main 088344e] Add work_main.py
   1 file changed, 0 insertions(+), 0 deletions(-)
   create mode 100644 work_main.py

  > git log --graph --oneline -3
  * 088344e (HEAD -> main) Add work_main.py
  * 38affbd (origin/main, origin/HEAD) Checkpoint
  * a78fa6a Checkpoint
  ```

## Hot fix

- The script is `$GIT_ROOT/tutorials/tutorial_git/hot_fix.sh`

- Create a feature branch keeping history linear:
  ```bash
  > ls
  Dockerfile
  LICENSE
  README.md
  dev_scripts
  project_template
  gp
  lectures
  projects

  > git status
  On branch main
  Your branch is up to date with 'origin/main'.

  You are in a sparse checkout with 100% of tracked files present.

  nothing to commit, working tree clean

  > git log --graph --oneline -3
  * 68df32f (HEAD -> main, origin/main, origin/HEAD) Checkpoint
  * b495a2c Checkpoint
  * 38affbd Checkpoint
  ```

- Create a new branch `iss53`
  ```bash
  > git checkout -b iss53
  Switched to a new branch 'iss53'
  ```

- Create and commit a file in the branch `iss53`
  ```bash
  > touch feature.py

  > git add feature.py

  > git status -s
  A  feature.py

  > git commit -am 'Add feature.py'
  [iss53 dc84037] Add feature.py
   1 file changed, 0 insertions(+), 0 deletions(-)
   create mode 100644 feature.py

  > git log --graph --oneline -3
  * dc84037 (HEAD -> iss53) Add feature.py
  * 68df32f Checkpoint
  * b495a2c Checkpoint
  ```

- Create a new branch `hotfix` off `main`
  ```bash
  > git checkout main
  Switched to branch 'main'
  Your branch is up to date with 'origin/main'.

  > git checkout -b hotfix
  Switched to a new branch 'hotfix'

  > touch hot_fix.py

  > git add hot_fix.py

  > git status -s
  A  hot_fix.py

  > git commit -am 'Add hot_fix.py'
  [hotfix 402ed4f] Add hot_fix.py
   1 file changed, 0 insertions(+), 0 deletions(-)
   create mode 100644 hot_fix.py
  ```

- Go back to `main` and merge `hotfix`
  ```bash
  > git checkout main
  Switched to branch 'main'
  Your branch is up to date with 'origin/main'.

  > git merge hotfix -m 'Merge hot_fix.py'
  Merge made by the 'ort' strategy.
   hot_fix.py | 0
   1 file changed, 0 insertions(+), 0 deletions(-)
   create mode 100644 hot_fix.py

  > git log --graph --oneline -3
  *   b15d232 (HEAD -> main) Merge hot_fix.py
  |\
  | * 402ed4f (hotfix) Add hot_fix.py
  |/
  * 68df32f Checkpoint
  ```

- Go back to `iss53` branch and commit more changes
  ```bash
  > git checkout iss53
  Switched to branch 'iss53'

  > git log --graph --oneline -3
  * dc84037 (HEAD -> iss53) Add feature.py
  * 68df32f Checkpoint
  * b495a2c Checkpoint

  > touch feature2.py

  > git add feature2.py

  > git commit -am 'Add feature2.py'
  [iss53 49c2b96] Add feature2.py
   1 file changed, 0 insertions(+), 0 deletions(-)
   create mode 100644 feature2.py
  ```

- Merge `iss53` to main
  ```bash
  > git checkout main
  Switched to branch 'main'
  Your branch is ahead of 'origin/main' by 2 commits.
    (use "git push" to publish your local commits)

  > git merge iss53 -m 'Merge iss53'
  Merge made by the 'ort' strategy.
   feature.py  | 0
   feature2.py | 0
   2 files changed, 0 insertions(+), 0 deletions(-)
   create mode 100644 feature.py
   create mode 100644 feature2.py
   ```

## Merge conflicts

- A script running the entire flow is in `$GIT_ROOT/tutorials/tutorial_git/merge_conflict.sh`
  - You should execute each command one at a time

- Restore the repo to the initial state
  ```bash
  > source $GIT_ROOT/tutorials/tutorial_git/restart.sh
  ```
  which in practice corresponds to
  ```bash
  > rm -rf /tmp/umd_classes_tmp

  > git clone git@github.com:gpsaggese/umd_classes.git /tmp/umd_classes_tmp
  Cloning into '/tmp/umd_classes_tmp'...
  Warning: Permanently added 'github.com' (ED25519) to the list of known hosts.
  ```

- Create a branch `iss53` with some changes
  ```bash
  > cd /tmp/umd_classes_tmp

  > ls
  Dockerfile
  LICENSE
  README.md
  dev_scripts
  project_template
  gp
  lectures
  projects

  > git status
  On branch main
  You are in a sparse checkout with 100% of tracked files present.

  nothing to commit, working tree clean

  > git log --graph --oneline -3
  * c47a0b6 (HEAD -> main, origin/main, origin/HEAD) Checkpoint
  * 68df32f Checkpoint
  * b495a2c Checkpoint

  > git checkout -b iss53
  Switched to a new branch 'iss53'

  > echo 'hello from iss53' >feature.py

  > git add feature.py

  > git status -s
  A  feature.py

  > git commit -am 'Add feature.py'
  [iss53 f0517d8] Add feature.py
   1 file changed, 1 insertion(+)
   create mode 100644 feature.py

  > git log --graph --oneline -3
  * f0517d8 (HEAD -> iss53) Add feature.py
  * c47a0b6 Checkpoint
  * 68df32f Checkpoint
  ```

- Create a `hotfix` branch with some changes
  ```bash
  > git checkout main
  Switched to branch 'main'

  > git checkout -b hotfix
  Switched to a new branch 'hotfix'

  > echo 'hello from hotfix' >feature.py

  > git add feature.py

  > git status -s
  A  feature.py

  > git commit -am 'Add feature.py'
  [hotfix 299dc2e] Add feature.py
   1 file changed, 1 insertion(+)
   create mode 100644 feature.py
  ```

- Merge `hotfix` in `main`
  ```bash
  > git checkout main
  Switched to branch 'main'

  > git merge hotfix -m 'Merge hotfix'
  Merge made by the 'ort' strategy.
   feature.py | 1 +
   1 file changed, 1 insertion(+)
   create mode 100644 feature.py

  > git log --graph --oneline -3
  *   17b765b (HEAD -> main) Merge hotfix
  |\
  | * 299dc2e (hotfix) Add feature.py
  |/
  * c47a0b6 Checkpoint
  ```

- Merge `iss53` in `main` creating conflicts
  ```bash
  > git checkout main
  Already on 'main'

  > git merge iss53 -m 'Merge iss53'
  Auto-merging feature.py
  CONFLICT (add/add): Merge conflict in feature.py
  Recorded preimage for 'feature.py'
  Automatic merge failed; fix conflicts and then commit the result.

  # `AA` = file was added in both branches, creating a conflict.
  > git status -s
  AA feature.py

  > git diff
  diff --cc feature.py
  index 4f21d06,5b9abe9..0000000
  --- a/feature.py
  ++ b/feature.py
  @@@ -1,1 -1,1 +1,5 @@@
  +<<<<<<< HEAD
   +hello from hotfix
  +=======
  + hello from iss53
  +>>>>>>> iss53
  ```

- Solve the conflict and merge
  - The conflict markers mean:
    - `<<<<<<< HEAD`: start of the version from the current branch (`main`)
    - `=======`: separator between the two conflicting versions
    - `>>>>>>> iss53`: end of the version from the incoming branch (`iss53`)
  - Edit the file to keep the desired content, remove all marker lines, then
    stage and commit
  ```bash
  > echo 'hello from iss53 and hotfix' >feature.py

  > git add feature.py

  > git status -s
  M  feature.py

  > cat feature.py
  hello from iss53 and hotfix

  > git commit -m "Merge"
  Recorded resolution for 'feature.py'.
  [main cebc983] Merge

  > git status -s

  > git log --graph --oneline -5
  *   cebc983 (HEAD -> main) Merge
  |\
  | * f0517d8 (iss53) Add feature.py
  * |   17b765b Merge hotfix
  |\ \
  | |/
  |/|
  | * 299dc2e (hotfix) Add feature.py
  |/
  * c47a0b6 Checkpoint
  ```
