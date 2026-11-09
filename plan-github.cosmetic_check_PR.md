# Plan: Scripts for `github.cosmetic_check_PR`

- Goal:
  - Script 1: create a worktree from a PR URL
  - Scripts 2 to N: implement the deterministic checks of the skill
    `helpers_root/.claude/skills/github.cosmetic_check_PR/SKILL.md`
  - The same check scripts work on a PR and on the code in a dir
    - E.g., check that the files in a dir are smaller than 512KB
  - The skill and the command line both call the scripts
  - Each script points to a dir that holds a repo or a worktree
- Status: draft for review, no code written yet
- Edit this file, then ask to implement

# Architecture

## Idea: Separate the Scope From the Check

- A check never knows if it runs on a PR or on a dir
  - A provider builds the list of files to check: the scope
  - A check applies pure rules to that list and returns findings
- Flow:
  ```
  --dir, --scope, ... -> provider -> FileSet -> check -> findings -> report
  ```
- Tests need no git for the rules: build a `FileSet` by hand
- Tests of the providers use real temporary git repos

## FileSet

- `FileSet` holds:
  - `files`: a list of `FileInfo` for the files that exist in the scope
  - `deleted`: the paths removed by the PR, empty in `disk` scope
- `FileInfo` holds:
  - `path`: relative to the repo root
  - `size`: bytes
  - `mode`: git mode, e.g., `100644`, `160000` for a submodule pointer
  - `is_binary`: bool
  - `status`: `A`, `M`, or `-` when there is no diff

## Providers

- `diff`: the files changed between `--merge_base` and `--head`
  - Use `git diff --no-renames --raw --numstat` and `git cat-file --batch-check`
  - Read git objects only, so it works on a detached worktree
  - `--merge_base` default: `git merge-base origin/master HEAD`
  - `--head` default: `HEAD`
  - Why an explicit `--merge_base` for a PR
    - `origin/HEAD` is not set in this client
    - Remote `origin` may not point to the PR repo
    - The worktree script prints the exact `merge_base`, and the skill passes it on
- `disk`: all the files under `--dir` on disk, and the default scope
  - In a git repo, use `git ls-files --cached --others --exclude-standard`
    - It honors `.gitignore` and includes uncommitted files
  - Outside a git repo, use `os.walk`, without following symlinks and skipping `.git`
  - Size from `os.stat`
  - Binary from a NUL byte in the first 8KB
  - Mode from the stat bits
- Optional later, no change to the checks:
  - `staged`: `git diff --cached`, so the pre-commit hook can call the same checks
  - `tree`: the committed files at a ref, i.e., `diff` with the empty tree as
    `--merge_base`

## Common Flags

- Every check script takes the same flags, from one function in the shared module
  - `--dir <DIR>`: default `.`
  - `--scope {diff,disk}`: default `disk`
  - `--merge_base <SHA>` and `--head <SHA>`: used by `diff` only
- `--dir` is the place to run and also a filter
  - The repo root is `git -C <DIR> rev-parse --show-toplevel`
  - Only the files under `--dir` are checked
  - A `--dir` outside a git repo allows only `--scope disk`
  - A worktree and a repo are the same thing for the scripts

## Checks by Scope

- CHK1, size and binary files: both scopes, same rule
- CHK2, project directory:
  - `diff`: all the touched paths, deleted ones included, against `<PROJECT_PATH>`
    built from `--branch`
  - `disk`: `--dir` must match `class_project/<course>/<Term><Year>/projects/<name>`
    - Compare `<name>` with `--branch` only when it is given
- CHK3, PR title: two parts
  - Branch name rules: both scopes, from `--branch` or the branch of the repo
  - PR metadata rules (title, issue, body): only with `--pr_url`, otherwise `N/A`
- CHK4a, junk and secrets:
  - File rules: both scopes
  - `gitleaks`: `gitleaks git --log-opts <MERGE_BASE>..<HEAD>` in `diff`, and
    `gitleaks dir <DIR>` in `disk`
- CHK4b, class project layout: both scopes, run on the project dir
  - In a PR: `--dir <WORKTREE>/<PROJECT_PATH>`

## Output

- Text in the format of `helpers_root/.claude/templates/github_PR_check.template.md`,
  so the skill can paste it
  - The first line is `PASS`, `FAIL`, or `N/A`
  - Each finding has `Why` and `Fix`
- `--json` prints the same data for tools
- Exit code: 1 on `FAIL`, 0 otherwise

## Script Structure

- One script per check, with the `_parse()` / `_main(parser)` structure of
  `helpers_root/.claude/skills/script.rules.md`
- Each script module exposes one public function `check_<name>(file_set, ...)`
  - The runner imports it, and the tests call it directly
- Thresholds are flags, e.g., `--max_size_kb`, default 512

# Layout

- Scope-independent code and checks:
  `helpers_root/dev_scripts_helpers/repo_checks/`
  - `repo_check_utils.py`: `FileInfo`, `FileSet`, the providers, the common flags,
    `Finding`, the report formatter
  - `check_files.py`: CHK1
  - `check_risky.py`: CHK4a
- PR only: `helpers_root/dev_scripts_helpers/github/`
  - `gh_pr_worktree.py`: script 1
  - `gh_pr_check_title.py`: CHK3
- Class layout: `class_project/`
  - `check_project_dir.py`: CHK2
  - `check_project_layout.py`: CHK4b
- Names drop the `gh_pr_` prefix when the script is not PR specific

# Scripts

## Worktree

- `helpers_root/dev_scripts_helpers/github/gh_pr_worktree.py`
  - Input: `--pr_url` or `--pr_num`, and `--worktree_dir` (default `~/wt_pr<N>`)
  - Strip `/changes`, `/files`, and `/commits` from the URL
  - Fetch `pull/<N>/head` and the base branch from the PR repo URL
    - Use `--no-recurse-submodules`
    - Save each SHA right after its fetch
  - Compute the merge-base
  - Create the detached worktree, or reuse it when `HEAD` matches
  - Stop with a clear message when the dir exists with another `HEAD`: never delete
  - Print:
    - `worktree`, `head_sha`, `base_sha`, `merge_base`
    - `head_branch`, `base_branch`
    - the `git worktree remove` command
  - Option: `--dry_run`

## CHK1: File Size and Binary Files

- `helpers_root/dev_scripts_helpers/repo_checks/check_files.py`
  - Fail files over `--max_size_kb` (default 512, i.e., 524288 bytes)
  - Fail binaries: `is_binary`, the extension list, and `.DS_Store`
  - Count notebooks
  - Print only the failing files
  - Print a `git rm --cached` fix per file

## CHK2: Project Directory

- `class_project/check_project_dir.py`
  - Needs `--branch` in `diff` scope, since the worktree is detached
  - Detect a class PR
  - Build `<PROJECT_PATH>`
  - List the touched paths outside it
  - Print a `git mv` fix per path
  - Lives in `class_project/` because it encodes the class layout

## CHK3: PR Title

- `helpers_root/dev_scripts_helpers/github/gh_pr_check_title.py`
  - Get the PR from `--pr_url`, or from the PR of the branch checked out in `<DIR>`
    - Use `gh pr view` and `gh issue view`
  - Checks:
    - Title equal to the head branch
    - The `UmdTask<N>_` pattern and its allowed characters
    - The class pattern with the optional `_<N>` suffix
    - The issue exists
    - The issue title equals the project tag
    - The first body line is `Closes #N` or `Related to #N`, with the same `N`
  - Overrides for offline use and tests: `--title`, `--branch`, `--body_file`
  - Print a `gh pr edit` fix and never run it

## CHK4a: Junk and Secrets

- `helpers_root/dev_scripts_helpers/repo_checks/check_risky.py`
  - Find:
    - `tmp.*`, `*.log`, `__pycache__/`, `.ipynb_checkpoints/`, `.env`
    - A submodule pointer (mode `160000`), as a change in `diff` and as a file in
      `disk`
  - Run `gitleaks` with the repo rules file, as in `Checks by Scope`
  - Print `unknown` when `gitleaks` is missing
  - Option: `--skip_gitleaks`

## CHK4b: Class Project Layout

- `class_project/check_project_layout.py`
  - Compare the Docker files with `class_project/project_template`
  - Check that these files exist:
    - `<NAME>_utils.py`
    - The `<NAME>.API` and `<NAME>.example` notebooks, with their paired `.py` files
    - `README.md`
    - `test/test_<NAME>_utils.py`

## Runner (Optional)

- `class_project/check_all.py`
  - Run the checks above in order
  - Print the full report with the four verdicts
  - The skill adds only the CHK4 rule reading, which needs judgment
  - Lives in `class_project/` because it composes the class checks

# Usage Examples

- PR:
  ```bash
  > gh_pr_worktree.py --pr_url https://github.com/<OWNER>/<REPO>/pull/865
  > check_files.py --dir ~/wt_pr865 --scope diff --merge_base <MERGE_BASE>
  ```
- Code in a dir:
  ```bash
  > check_files.py --dir class_project/data605/Spring2026/projects/<PROJECT_DIR>
  ```
- Another size limit:
  ```bash
  > check_files.py --dir tutorials --max_size_kb 256
  ```

# Tests

- Use `hunitest.TestCase`, per `helpers_root/.claude/skills/testing.rules.md`
- Checks: call `check_<name>()` with a `FileSet` built by hand
  - No git, no disk
  - E.g., title parsing, expected path, binary detection, size filter
- Providers: build real temporary git repos in the scratch space, with `git init` and
  commits
  - Do not mock git
  - Test that `diff` and `disk` return the same `FileInfo` for the same files
- Keep `gh` calls behind one thin function
  - Tests never call `gh`
- Worktree script: use a local bare repo as the fake `pull/<N>/head` source

# Order of Work

- [ ] Step 1: `repo_check_utils.py` (`FileSet`, providers, report) and its tests
- [ ] Step 2: `check_files.py` and its tests
- [ ] Step 3: `gh_pr_worktree.py` and its tests
- [ ] Step 4: `gh_pr_check_title.py`, `class_project/check_project_dir.py`, and their
      tests
- [ ] Step 5: `check_risky.py`, `class_project/check_project_layout.py`, and their
      tests
- [ ] Step 6: `class_project/check_all.py`, the new `SKILL.md`, and the entries in
      `helpers_root/dev_scripts_helpers/README.md` for the new dir
- After each step:
  - Run the new tests
  - Run the new script on the existing worktree `~/wt_pr865` and on a project dir

# Defaults and Open Decisions

- Edit the choice if you disagree

## Default Scope

- Default: `disk`, since `--dir` alone checks the code in a dir
- Alternative: `diff` when `--merge_base` is given, with no `--scope` flag
  - Fewer flags, but the scope depends on a side effect

## Files Included in the `disk` Scope

- Default: tracked and untracked files, minus the ignored ones
- Alternative: tracked files only
  - This hides a large file that is not yet added

## Location and Names

- Default: new dir `helpers_root/dev_scripts_helpers/repo_checks/` for the scripts
  that are not PR specific
- Alternative: keep all of them in `helpers_root/dev_scripts_helpers/github/` with
  the `gh_pr_` prefix
  - Simpler, but the name says PR for a check that works on a dir

## Worktree Path

- Default: keep `~/wt_pr<N>`, which matches the existing `~/wt_pr865`
- Alternative: the repo convention for other worktrees, a sibling dir
  `<repo>_worktree_<N>`
- `--worktree_dir` overrides the default in both cases

## Location of the Class Checks

- Default: the class checks and the runner live in `class_project/`
- The generic checks live in `helpers_root/`, which other repos also use
- Alternative: all scripts in `helpers_root/`

## Docker Comparison

- Default: add a small public `compare_docker_files()` to
  `class_project/create_project.py` and call it from `check_project_layout.py`
  - No log file in the current client, no `uv` call
  - The function needs `pandas`, which only `create_project.py` loads today
- Alternative: leave `create_project.py` unchanged and copy its small `filecmp`
  logic
  - This duplicates the `_DOCKER_FILES` list

## Runner Script

- Default: write `check_all.py` in step 6
- Alternative: skip it, and let the skill call each script

## Pre-Commit Hook

- Default: leave `check_file_size()` of the pre-commit hook as it is
  - It skips notebooks, while CHK1 does not
- Alternative: later, add the `staged` scope and make the hook call `check_files.py`

## Commits

- I commit nothing: you decide when and what
