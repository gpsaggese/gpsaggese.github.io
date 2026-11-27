# Good First Issues

- These tasks are small and are a good way to make your first contribution
- The tasks are in two repos:
  - `umd`: [gpsaggese.github.io](https://github.com/gpsaggese/gpsaggese.github.io)
  - `helpers`: [helpers](https://github.com/gpsaggese/helpers)

## How to Start

- Pick a task from the table below
  - The tasks are ordered from easiest to hardest, approximately
  - The size goes from `XS` (smallest) to `M` (largest)
- Check that no one else works on the task
- If the task has no GitHub issue, create one before you start
- Post in the issue that you take the task and when you expect to finish
- Follow [How to Create a PR With Claude Code](pr_process.how_to_guide.md)

## Tasks

| #   | Task                                                                     | Repo      | Issue                                                                   | Where to look and notes                                                                                                                                                                      | Size |
| :-- | :----------------------------------------------------------------------- | :-------- | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--- |
| 1   | Rename `purify_white_spaces()` to `purify_trailing_white_spaces()`       | `helpers` | [helpers #529](https://github.com/gpsaggese/helpers/issues/529)         | `helpers/hunit_test.py`, `helpers/hunit_test_purification.py`, `helpers/test/test_hunit_test_purification.py`                                                                                | XS   |
| 2   | List the pytest builds in order: `apple`, `dev_container`, `docker`      | `helpers` | Create one                                                              | `dev_scripts_helpers/testing/pytest_failed_multi_build.py`                                                                                                                                   | XS   |
| 3   | Remove the `./<dir>/` prefix from the `sync_gh_issue_labels.py` commands | `helpers` | Create one                                                              | `docs/tools/dev_system/all.synchronize_gh_issue_labels.how_to_guide.md`                                                                                                                      | XS   |
| 4   | Remove `:` from the names of files created by `invoke git_file_version`  | `helpers` | Create one                                                              | `helpers/lib_tasks/lib_tasks_git.py`. Add a unit test                                                                                                                                        | S    |
| 5   | Replace the old `cryptokaizen` org name in links                         | `helpers` | [helpers #515](https://github.com/gpsaggese/helpers/issues/515)         | Limit the first PR to the `.md` files. Ask which org name to use before you start                                                                                                            | S    |
| 6   | Add unit tests for `sort_config_string()`                                | `helpers` | [helpers #579](https://github.com/gpsaggese/helpers/issues/579)         | `config_root/config/config_utils.py` and `config_root/config/test/test_config_utils.py`. The function may have a bug: first write tests that show it                                         | S    |
| 7   | Create a check that finds dangling symlinks and empty files              | `helpers` | Create one                                                              | Related to `linters2/lint.py`. Add unit tests                                                                                                                                                | S    |
| 8   | Replace the copies of `docker_jupyter.sh` with symlinks                  | `umd`     | Create one                                                              | The template is [`docker_jupyter.sh`](../class_project/project_template/docker_jupyter.sh). Look at `create_links.py`. Test that Docker still works. Ask first: this task may be in progress | S    |
| 9   | Write a blog about our `md` workflow for Claude skills                   | `umd`     | [umd #423](https://github.com/gpsaggese/gpsaggese.github.io/issues/423) | No code. First read the [draft on Claude skills](../website/docs/blog/posts/draft.how_to.Claude_skills.md) and ask what the blog must cover                                                  | S    |
| 10  | Convert `ctags.sh` to `ctags.py`                                         | `helpers` | Create one                                                              | `dev_scripts_helpers/coding_tools/ctags.sh`                                                                                                                                                  | M    |