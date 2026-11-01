# Task: Merge `ci_state.py` and `invoke gh_workflow_list`

Both tools answer the question "is the latest run of each GH Actions workflow
on a branch green?". They use different code paths and give different answers.
Merge them into one code path, and fix the bugs listed below.

Do not commit. Do not push. Read and follow `.claude/rules.md` first.

## Files

All paths are under `helpers_root/`. `helpers_root` may be a Git submodule:
check with `git -C helpers_root status` before editing, and leave all changes
uncommitted.

- `helpers/lib_tasks/lib_tasks_gh.py`
  - `_get_workflow_table()`: fetches runs, parses tab-separated text
  - `gh_workflow_list()`: the invoke task
  - `_download_failed_run_log()`, `_colorize_status()`, `_print_table()`
  - `gh_watch()`: duplicate of the one in `lib_tasks_git.py`
- `dev_scripts_helpers/github/ci_state.py`: standalone script, JSON fetch
- `helpers/lib_tasks/test/test_lib_tasks_gh.py`
  - `Test__get_workflow_table` and `Test_gh_workflow_list` cover the old code
- `helpers/lib_tasks/lib_tasks_git.py`: second `gh_watch()` definition
- `tasks.py`: imports `gh_watch` (and `gh_workflow_list`)

## Current behavior (verified by reading the code)

| Aspect | `ci_state.py` | `gh_workflow_list` |
|---|---|---|
| Fetch | `gh run list --limit=200 --json=...`, no branch filter | `gh run list --limit 100 --branch X`, text output |
| Branch filter | Client-side on `headBranch` | Server-side, then client-side |
| Parsing | `json.loads` | Tab split, asserts 9 columns |
| Run chosen | Latest *completed* run per workflow | Newest run per workflow, no verdict if it is in progress, cancelled, skipped or `startup_failure` |
| Output | One markdown table plus failed/cancelled summary | "ok" / "broken" lines, or a raw table |
| Extras | `--max_col_width`, duration, age | `gh_login`, `--repo-short-name`, `--filter-by-completed`, `--daemon`, `pytest_repro` on failure |

The `ci_state.py` docstring says the server-side `--branch` filter of
`gh run list` can return stale runs. Treat that as a claim to confirm, not a
fact. Run `gh run list --branch master --limit 20` and
`gh run list --limit 200` and compare the newest `master` run in each output.
Record the result in a code comment.

## Bugs to fix

1. **Wrong branch name.** In `gh_workflow_list()`, the line
   `branch_name = hgit.get_branch_name()` (just before
   `workflows = table.unique("workflow")`) overwrites the branch chosen by
   `_get_branch_name(filter_by_branch)`. With `--filter-by-branch master` on a
   feature branch, the messages and the log file name use the feature branch.
   Keep the branch that was filtered on.
2. **Wrong loop bound.** `num_rows = table.size()[0]` should use `table_tmp`,
   the table for the current workflow.
3. **Fragile parsing.** `_get_workflow_table()` parses the text output of
   `gh run list`. The `TODO(Grisha)` above `gh_workflow_list` says the format
   changed. Use `--json` instead.
4. **No exit code.** Neither tool returns a non-zero exit code when a workflow
   failed. Add one, so a script can gate on it.
5. **Duplicate task.** `gh_watch` is defined twice: in `lib_tasks_gh.py` and in
   `lib_tasks_git.py`. Both carry
   `TODO(ai_gp): Merge this inside the other flow invoke gh_workflow_list --daemon`.
   `gh_workflow_list --daemon` already exists. See step 6.

## Design

1. Add one public function to `lib_tasks_gh.py`. Suggested name:
   `get_latest_runs_per_workflow(branch, *, repo_full_name_with_host="", limit=200)`.
   - Run `gh run list --limit <limit> --json=name,status,conclusion,databaseId,updatedAt,createdAt,headBranch`.
     Add `--repo <repo>` when `repo_full_name_with_host` is not empty.
   - Do not pass `--branch` to `gh`. Filter on `headBranch` in Python.
   - Return, per workflow name, the latest completed run. Keep the order
     newest-first, as `gh` returns it.
   - Return a second value (or a field) that says whether a newer run of the
     same workflow is still in progress.
   - Move the helpers `_parse_timestamp`, `_group_runs_by_workflow` and the
     table builder out of `ci_state.py` into the library. Make the helpers that
     both callers use public (no leading underscore), so `ci_state.py` no longer
     imports private names from another module.
2. Rule for in-progress, cancelled, skipped and `startup_failure` runs. This is
   a decision for the owner. Use this default and say so in the final report:
   - The verdict comes from the latest *completed* run.
   - Show a marker when a newer run is in progress.
   - `cancelled`, `skipped` and `startup_failure` appear in the table and the
     summary. They do not count as `failure` for the exit code.
3. Rewrite `ci_state.py` as a thin CLI over the library function.
   - Keep `--branch` (default `master`), `--max_col_width` and verbosity.
   - Add `--repo` (full name with host, default: the repo of the CWD).
   - Exit with code 1 if any workflow has `conclusion == "failure"`.
   - Keep the output format (markdown table, then the summary lines).
4. Rewrite the body of `gh_workflow_list()` over the same library function.
   - Keep: `gh_login`, `repo_short_name`, `filter_by_branch`,
     `filter_by_completed`, `show_stack_trace`, `print_table`, `daemon`,
     `interval`, and the `pytest_repro` call on failure.
   - Delete the tab parsing in `_get_workflow_table()`. If nothing else calls
     it after the rewrite, delete the function.
   - Keep `_download_failed_run_log()` and the status colors in one place. The
     log file name is `tmp.<prefix>.<workflow>.<branch>.txt`. Keep the two
     prefixes (`failure`, `ci_state`) only if a test or a doc relies on them.
     Otherwise use one prefix.
   - After a failure, `invoke` must also exit non-zero (raise or return a code,
     follow how other tasks in this file do it).
5. Fix the docstring comments that name the old functions
   (`_print_table`, `invoke gh_watch`, `_get_workflow_table`).
6. Remove the duplicate `gh_watch`.
   - Keep one definition. Make it call `gh_workflow_list(daemon=True, ...)`, or
     remove it and update `tasks.py` and the two docs that mention it
     (`.claude/skills/github.create_child_pr/SKILL.md`,
     `.claude/skills/todo.rules.md`).
   - Ask the owner before removing a task that users type by hand. If unsure,
     keep `gh_watch` as a one-line alias and delete only the duplicate copy.

## Tests

Run tests the way `helpers_root/CLAUDE.md` says (for example
`invoke docker_cmd --cmd "pytest <path> -v"`). Never run `uv run --active`.

- Update `Test__get_workflow_table` and `Test_gh_workflow_list` in
  `helpers/lib_tasks/test/test_lib_tasks_gh.py`. They patch
  `_get_workflow_table` and `_print_table`, so they must change with the code.
- Add unit tests for the new library function. Use a fixed JSON list of runs.
  Do not call GitHub. Cases:
  - runs of several branches: only the requested branch stays
  - several runs of one workflow: the latest completed run wins
  - newest run is in progress: the older completed run gives the verdict, and
    the marker is set
  - empty output from `gh`: empty result
  - a `failure` run: exit code 1 for the CLI
- Add a test for bug 1: with `filter_by_branch="master"` while the Git branch is
  a feature branch, the messages and log file name use `master`.
- Follow the test conventions in `.claude/rules.md` (naming, golden files,
  no mocking where a real object works).

## Manual check

1. Run `helpers_root/dev_scripts_helpers/github/ci_state.py` and
   `invoke gh_workflow_list --filter-by-branch master` from a feature branch.
2. Both must show the same verdict per workflow and the name `master`.
3. Both must exit non-zero when a workflow on `master` failed.

## Done when

- One code path fetches and groups the runs.
- The five bugs are fixed, each with a test.
- Fast tests for `test_lib_tasks_gh.py` pass.
- Lint passes on the touched files.
- Nothing is committed.
- The final report lists: files changed, the rule chosen for in-progress runs,
  the result of the stale-`--branch` check, and any step left undone.

## Plan

- [x] Read `.claude/rules.md`, `coding.rules.md`, `testing.rules.md`
- [x] Check `git -C helpers_root status`: submodule on branch `gp`, only
      untracked `*.py.log` files, leave all changes uncommitted
- [x] Confirm or refute the stale `--branch` claim with `gh`
- [x] Library in `lib_tasks_gh.py`
  - [x] Add `fetch_runs()` (JSON, no `--branch`, client-side filter)
  - [x] Add `get_latest_runs_per_workflow()` returning the latest completed run
        per workflow and the set of workflows with a newer run in progress
  - [x] Move `parse_timestamp`, `format_days_ago`, `group_runs_by_workflow`,
        table builder, `format_table`, `print_summary` out of `ci_state.py`
  - [x] Make `download_failed_run_log`, `colorize_status` public, one log prefix
        (`failure`, because `pytest.fix_ci_tests/SKILL.md` relies on it)
  - [x] Add `get_failed_workflows()` for the exit code
- [x] Rewrite `gh_workflow_list()` over the library (bugs 1, 2, 3, 4)
  - [x] Keep `branch_name` from `_get_branch_name(filter_by_branch)`
  - [x] Delete `_get_workflow_table()`, `_print_table()`, `_GH_RUN_LIST_LIMIT`
  - [x] Raise `invexc.Exit(code=1)` after a failure (not in daemon mode)
- [x] Rewrite `ci_state.py` as a thin CLI (`--branch`, `--repo`,
      `--max_col_width`, exit 1 on `failure`)
  - [x] Follow up: owner chose to remove `ci_state.py`. Use
        `invoke gh_workflow_list --filter-by-branch master` instead
- [x] Remove the duplicate `gh_watch` from `lib_tasks_git.py`, keep a one-line
      alias in `lib_tasks_gh.py`
- [x] Update docs that describe the changed behavior
- [x] Tests
  - [x] Replace `Test__get_workflow_table` with `Test_get_latest_runs_per_workflow`
  - [x] Update `Test_gh_workflow_list` (patch `fetch_runs`, not removed names)
  - [x] Add test for bug 1 (`filter_by_branch="master"` on a feature branch)
  - [x] Add `test_ci_state.py` (exit code 1 on a `failure` run). Removed with
        `ci_state.py`. `Test_gh_workflow_list` covers the same cases
  - [x] Run fast tests for `test_lib_tasks_gh.py` in the container
- [x] Lint touched files
- [x] `git add` new files, leave everything uncommitted
- [x] Write the Result section

## Result

### Done
- One code path in `helpers/lib_tasks/lib_tasks_gh.py`
  - `fetch_runs()` runs `gh run list --limit 200 --json=...`, no `--branch`
  - `get_latest_runs_per_workflow()` returns the latest completed run per
    workflow, newest first, plus the set of workflows with a newer run in
    progress
  - Moved from `ci_state.py` and made public: `parse_timestamp()`,
    `format_days_ago()`, `group_runs_by_workflow()`, `build_table_rows()`,
    `format_table()`, `print_summary()`. Also public now:
    `download_failed_run_log()`, `colorize_status()`
  - `ci_state.py` and `test_ci_state.py` are removed (`git rm`, owner decision).
    Run `invoke gh_workflow_list --filter-by-branch master` instead
  - `gh_workflow_list()` runs over the same function and raises
    `invexc.Exit(code=1)` after a `failure` (not in `--daemon` mode)
- Bugs
  - Bug 1: the branch from `_get_branch_name()` is kept (test
    `Test_gh_workflow_list.test4`)
  - Bug 2: latent, since the old loop always `break`s at `i == 0`. The loop is
    gone. Test `Test_get_latest_runs_per_workflow.test2` covers workflows with
    different run counts
  - Bug 3: `--json` replaces the tab parsing. `_get_workflow_table()`,
    `_print_table()` and `_GH_RUN_LIST_LIMIT` are deleted
  - Bug 4: exit code tested in `Test_gh_workflow_list.test3`
  - Bug 5: duplicate `gh_watch` removed from `lib_tasks_git.py`. One line alias
    stays in `lib_tasks_gh.py`. `tasks.py` and the skill docs need no change
- Rule for in-progress, cancelled, skipped and `startup_failure` runs: the
  default from the task
  - The verdict comes from the latest completed run
  - A newer run in progress shows `(newer run in progress)` in the table
  - `cancelled`, `skipped`, `startup_failure` show in the table and the summary.
    They do not count as `failure` for the exit code
- Stale `--branch` check, 2026-11-01, `master` of `causify-ai/helpers`: the claim
  is NOT reproduced. `gh run list --branch master --limit 20` and
  `gh run list --limit 200` gave the same newest run for every workflow. Only the
  `Scheduled` workflow was missing from the 20 run window. Result is in a code
  comment in `fetch_runs()`
- Tests: 196 passed, 59 skipped for `helpers/lib_tasks/test` and
  `dev_scripts_helpers/github/test/test_ci_state.py`, run with
  `CSFY_DOCKER_ENGINE=docker invoke docker_cmd`. This was before `ci_state.py`
  was removed
- Manual check on branch `gp`, before the removal: `ci_state.py` and
  `invoke gh_workflow_list --filter-by-branch master` showed the same verdict per
  workflow, said `master`, and exited 1 (`Publish MkDocs` and `Superslow tests`
  failed)
- Docs: `docs/tools/git/all.manual_flow.how_to_guide.md`
- `ci_state.py` and `test_ci_state.py` removed with `git rm`. The deletions are
  staged. Nothing is committed

### Not done
- Pre-existing `ruff` findings in the touched files are left alone
  - `PLR2044` (empty `#` comment) and `SIM905` in other functions, `UP017` in
    `get_workflow_run_ids()`, `ISC004` in other tests. My code adds none
  - `ruff check` has `fix = true` and rewrote them. I restored them to keep the
    diff scoped. Fixing the empty `#` lines conflicts with the rule "do not
    remove comments"
- Behavior changes for the owner to confirm
  - `--filter-by-completed` now matches the `status` or the `conclusion` of the
    latest completed run. Before, it matched the `completed` column only, so
    `failure` in the docs never matched
  - `--filter-by-branch all` and `--report-only-status=False` print a flat table
    of runs with no verdict and no exit code
  - `print_table` prints one table (the latest run per workflow) and not one
    table of all runs per workflow
  - The `# gh_workflow_list: ...` header prints once in `--daemon` mode, not
    on every refresh
  - The client side filter shares the 200 runs among all branches. The stale
    check suggests a server side `--branch` with `--limit 200` would reach more
    runs of a quiet branch
