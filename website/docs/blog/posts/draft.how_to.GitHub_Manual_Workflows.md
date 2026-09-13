# Goals

- Create small PRs
  - Allow agents to work in parallel with minimal overlap
  - Make it easy for humans to review the code and merge

- Automate the workflow
  - Specs -> Branch -> Work -> PR -> Checks -> Review -> Merge

# Tools

## Script Location

- All GitHub workflow scripts are:
  - **Invoke tasks**:
    - Defined in `helpers_root/helpers/lib_tasks/lib_tasks_git.py` and
      `helpers_root/helpers/lib_tasks/lib_tasks_gh.py`
    - Run via `invoke <task_name>` from the repository root
  - **Script files**:
    - Defined in `helpers_root/dev_scripts_helpers/git`
  // TODO(ai_gp): Add also agent skills like `github.`

- Invoke and scripts related to Git, start with `git_`, while ones related
  to GitHub start with `gh_`

## Summary
// TODO(ai_gp): Create summary table with cluster (e.g., Git Branch Management,
// GitHub ..., name, type (invoke, script, skill), short description

## Git Branch Management

### Create and Initialize Branches

#### `git_create_issue_and_branch.py`
- Creates a GitHub issue and corresponding git branch/worktree in one operation.

- Create issue and its initial branch
  ```bash
  > git_create_issue_and_branch.py --title "Fix bug in parser"
  ```

- Create with body, labels, and assignees
  ```bash
  > git_create_issue_and_branch.py \
    --title "Feature" \
    --body "Details" \
    --labels "enhancement" \
    --assignees "user1"
  ```

- Create stacked branches (e.g., `_1`, `_2`) for splitting one issue across
  multiple PRs
  ```bash
  > git_create_issue_and_branch.py --title "Large feature" --suffix 1
  ```

#### `git_branch_create`
- Create and push upstream branch, optionally creating a draft PR

- Create branch from GitHub issue
  ```bash
  > invoke git_branch_create --issue-id 123
  ```

- Create branch with explicit name
  ```bash
  > invoke git_branch_create --branch-name "MyTask456_Feature_description"
  ```

- Create branch without auto-creating draft PR
  ```bash
  > invoke git_branch_create --branch-name "MyTask456_Feature" --no-create-pr
  ```

### Copy and Rename Branches

#### `git_branch_copy`
- Create a new branch with the same content as the current branch (squash merge)

- Copy current branch to new auto-generated name
  ```bash
  > invoke git_branch_copy
  ```

- Copy with explicit new name
  ```bash
  > invoke git_branch_copy --new-branch-name "MyTask456_Feature_v2"
  ```

- Copy without merging master first
  ```bash
  > invoke git_branch_copy --skip-git-merge-master
  ```

#### `git_branch_rename`
- Rename current branch locally and remotely. If a PR exists, recreates it under the
  new name

- Rename current branch
  ```bash
  > invoke git_branch_rename --new-branch-name "MyTask456_Better_description"
  ```

### Query Branch Status

#### `git_branch_next_name`
- Generate a unique branch name derived from current branch to avoid conflicts

- Show next available name
  ```bash
  > invoke git_branch_next_name
  ```

- Generate name from specific branch
  ```bash
  > invoke git_branch_next_name --branch-name "MyTask456_Feature"
  ```

- Use GitHub API method (faster)
  ```bash
  > invoke git_branch_next_name --method github_api
  ```

#### `git_branch_is_merged`
- Check if current branch was merged into master using GitHub API and git

- Check merge status
  ```bash
  > invoke git_branch_is_merged
  ```

### Clean Up Branches

#### `git_branch_delete_merged`
- Remove (both local and remote) branches that have been merged into master

- Delete merged branches with user confirmation
  ```bash
  > invoke git_branch_delete_merged
  ```

- Delete without confirmation
  ```bash
  > invoke git_branch_delete_merged --no-confirm-delete
  ```

### Review Changes

#### `git_branch_diff`
- Diff files of current branch against a specified point (base, master, HEAD, or
  hash)

- Diff against branch point
  ```bash
  > invoke git_branch_diff --target base
  ```

- Diff against origin/master
  ```bash
  > invoke git_branch_diff --target master
  ```

- Diff uncommitted changes
  ```bash
  > invoke git_branch_diff --target head
  ```

- Diff against specific hash
  ```bash
  > invoke git_branch_diff --target hash --hash-value abc123def
  ```

- Diff last commit
  ```bash
  > invoke git_branch_diff --target last_commit
  ```

- Only Python files
  ```bash
  > invoke git_branch_diff --target base --file-types py
  ```

- Skip documentation files
  ```bash
  > invoke git_branch_diff --target base --skip-file-types txt,md
  ```

- Show files without opening vimdiff
  ```bash
  > invoke git_branch_diff --target base --only-print-files
  ```

#### `git_branch_files`
- Report detailed status of files changed in current branch (added, modified,
  deleted)

- Show file changes with status
  ```bash
  > invoke git_branch_files
  ```

## GitHub Workflow Management

### Authentication

#### `gh_login`
- Login to GitHub with configured SSH key and authentication token.

- Login with default account
  ```bash
  > invoke gh_login
  ```

- Login with specific account
  ```bash
  > invoke gh_login --account myorg
  ```

- Print auth status
  ```bash
  > invoke gh_login --print-status
  ```

### Issue Management

#### `gh_issue_create`
- Create a new GitHub issue with optional body, labels, assignees, and project.

- Simple issue
  ```bash
  > invoke gh_issue_create --title "Fix bug in parser"
  ```

- Issue with details
  ```bash
  > invoke gh_issue_create \
    --title "Add new feature" \
    --body "Description here" \
    --labels "enhancement,priority-high" \
    --assignees "user1,user2"
  ```

- Create issue and first branch (with suffix for stacking)
  ```bash
  > invoke gh_issue_create --title "Large feature" --suffix 1
  ```

#### `gh_issue_title`
- Print the branch-name-compatible title of a GitHub issue.

- Get and copy to clipboard
  ```bash
  > invoke gh_issue_title --issue-id 123
  ```

- Get without clipboard copy
  ```bash
  > invoke gh_issue_title --issue-id 123 --no-pbcopy
  ```

### Pull Request Management

#### `gh_create_pr`
- Create a draft PR for the current branch with optional body, labels, and reviewers.

- Create draft PR with branch name as title
  ```bash
  > invoke gh_create_pr
  ```

- Create PR with custom title
  ```bash
  > invoke gh_create_pr --title "My Custom PR Title" --draft
  ```

- Create ready-to-review PR
  ```bash
  > invoke gh_create_pr --draft False
  ```

- Create PR with reviewers and labels
  ```bash
  > invoke gh_create_pr --reviewer user1,user2 --labels "needs-review,enhancement"
  ```

- Create PR with auto-merge enabled
  ```bash
  > invoke gh_create_pr --auto-merge
  ```

### Workflow Management

#### `gh_workflow_list`
- Report the status of GitHub workflows with filtering and optional daemon mode for
  continuous monitoring.

- Check current branch workflows
  ```bash
  > invoke gh_workflow_list
  ```

- Check master branch workflows
  ```bash
  > invoke gh_workflow_list --filter-by-branch master
  ```

- Check all branches
  ```bash
  > invoke gh_workflow_list --filter-by-branch all
  ```

- Filter by completed status
  ```bash
  > invoke gh_workflow_list --filter-by-completed failure
  ```

- Monitor workflows continuously (updates every 60s)
  ```bash
  > invoke gh_workflow_list --daemon
  ```

- Monitor with custom interval (every 30s)
  ```bash
  > invoke gh_workflow_list --daemon --interval 30
  ```

#### `gh_workflow_run`
- Manually trigger GitHub workflows for a branch.

- Run all workflows on current branch
  ```bash
  > invoke gh_workflow_run --branch current_branch
  ```

- Run all workflows on master
  ```bash
  > invoke gh_workflow_run --branch master
  ```

- Run specific workflow
  ```bash
  > invoke gh_workflow_run --branch current_branch --workflows fast_tests
  ```

#### `gh_delete_workflow_runs`
- Delete workflow runs, optionally filtered by age.

- Delete all runs for a workflow (with confirmation)
  ```bash
  > invoke gh_delete_workflow_runs --workflow-name "Fast tests"
  ```

- Delete runs older than 30 days
  ```bash
  > invoke gh_delete_workflow_runs --workflow-name "Slow tests" --older-than-days 30
  ```

- Preview what would be deleted
  ```bash
  > invoke gh_delete_workflow_runs --workflow-name "Fast tests" --dry-run
  ```

- Delete without confirmation prompt
  ```bash
  > invoke gh_delete_workflow_runs --workflow-name "Fast tests" --no-confirmation
  ```

# Workflows

// TODO(ai_gp): Explain this
### How to Split large PRs
- /github.split_branch_in_PRs
- Review edit github_pr_plan.md
- /github.create_child_pr PR2

# References
- helpers_root/docs/work_organization/all.use_github.how_to_guide.md
