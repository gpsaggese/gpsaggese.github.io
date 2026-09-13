# General rules
- Find a workflow to make it easier to create smaller PRs
  - Instead of having lots of agents making changes to the same branch
  - Have a way to delegate to an agent to create a small PR, do the change,
    regress, review and merge
  - Always create PRs associated to each branch with a clear description
    - i git_branch_create -> i gh_create_pr --no-draft
  - Find an easy way to check which PR is still to merge, which one was merged

# Tools

// TODO(ai_gp): Add a section on where the scripts are (invoke,
// /Users/saggese/src/umd_classes2/helpers_root/dev_scripts_helpers/git/)

// TODO(ai_gp): Describe all the scripts and invoke and their function
git_create_issue_and_branch.py
git_branch_create
git_branch_copy
git_branch_next_name
git_branch_delete_merged
git_branch_is_merged
git_branch_rename

git_branch_diff
git_branch_files

gh_create_pr
gh_delete_workflow_runs
gh_issue_create
gh_issue_title
gh_login
gh_watch
gh_workflow_list
gh_workflow_run

# Workflows

### How to Split large PRs
- /github.split_branch_in_PRs
- Review edit github_pr_plan.md
- /github.create_child_pr PR2

# References
- helpers_root/docs/work_organization/all.use_github.how_to_guide.md
