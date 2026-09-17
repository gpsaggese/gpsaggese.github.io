### [ ] Improve output

Creating GitHub issue via invoke: invoke gh_issue_create --title 'Improve research ideas' --assignees @me

```
> git_create_issue_and_branch.py --gh_issue_title "Improve research ideas" --no_abort_if_not_master --submodules 2>&1 | tee log.txt
14:27:40 - INFO  hdbg.py init_logger:1177                               > /Users/saggese/src/umd_classes3/helpers_root/dev_scripts_helpers/git/git_create_issue_and_branch.py --gh_issue_title Improve research ideas --no_abort_if_not_master --submodules
14:27:40 - WARN  hgit.py is_client_clean:2063                           Skipping 'helpers_root' in modified files
14:27:40 - INFO  git_create_issue_and_branch.py _main_workflow:528      Creating GitHub issue via invoke: invoke gh_issue_create --title 'Improve research ideas' --assignees @me
14:27:43 - INFO  git_create_issue_and_branch.py _main_workflow:537      Created issue #584
14:27:43 - INFO  git_create_issue_and_branch.py _create_branch_and_pr:230 Creating branch via invoke: invoke git_branch_create --issue-id 584 --no-abort-if-not-master
14:27:43 - INFO  hsystem.py _system:210             > (invoke git_branch_create --issue-id 584 --no-abort-if-not-master) 2>&1
14:28:03 - INFO  git_create_issue_and_branch.py _create_branch_and_pr:234 Branch created: UmdTask584_Improve_research_ideas_1
14:28:03 - INFO  git_create_issue_and_branch.py _main_workflow:547      Branch name: 'UmdTask584_Improve_research_ideas_1'
14:28:09 - INFO  git_create_issue_and_branch.py _create_branch_in_submodule:285 Creating branch in 'helpers_root' via invoke: cd helpers_root && invoke git_branch_create --branch-name UmdTask584_Improve_research_ideas_1 --no-abort-if-not-master
14:28:09 - INFO  hsystem.py _system:210             > (cd helpers_root && invoke git_branch_create --branch-name UmdTask584_Improve_research_ideas_1 --no-abort-if-not-master) 2>&1
14:28:24 - INFO  git_create_issue_and_branch.py _main:592               Returning to original branch: 'master'
```

### [ ] Improve git_create_issue_and_branch.py

Add an option to stay on the branches like git checkout -b 

### [ ]

What is --new-branch-name ? Is it with underscores? Should we allow to get also a
normal title?

Yes

i git_branch_copy --new-branch-name="HelpersTask1143_Skills_improvements_and_documentation_enhancements_with_transform_text_refactor"

git

```
> git log origin/master..master
commit e4332cefe67f8c9e5f6c432224635569f856cac1 (master)
Merge: 44a0214c 19d8cfcb
Author: GP Saggese <saggese@gmail.com>
Date:   Thu Sep 17 11:24:29 2026

    Merge branch 'master' of github.com:causify-ai/helpers

commit 44a0214cc8503716abb8c15c08bbf01a833f5e05
Merge: b827ae6a 2a07f93f
Author: GP Saggese <saggese@gmail.com>
Date:   Thu Sep 17 07:06:39 2026

    Merge branch 'master' of github.com:causify-ai/helpers
```

### [ ] git_branch_copy

The convention of the branch should be checked early

```
> i git_branch_copy --new-branch-name="Skills improvements and documentation enhancements with transform text refactor"
14:35:10 - INFO  hdbg.py init_logger:1177                               > /Users/saggese/src/venv/client_venv.helpers/bin/invoke git_branch_copy --new-branch-name=Skills improvements and documentation enhancements with transform text refactor
Removing dev_scripts_helpers/coding_tools/notify.py.log
Removing dev_scripts_helpers/git/git_merge_master.py.log
Removing tmp.system_cmd.sh
Removing tmp.system_output.txt
> git clean -fd
14:35:11 - INFO  hdbg.py init_logger:1177                               > /Users/saggese/src/venv/client_venv.helpers/bin/invoke git_merge_master --abort-if-not-ff --no-auto-merge --no-submodules
# git_merge_master: abort_if_not_ff=True, abort_if_not_clean=True, skip_fetch=False, auto_merge=False, submodules=False, dry_run=False
14:35:12 - INFO  hdbg.py init_logger:1167                               Saving log to file '/Users/saggese/src/umd_classes2/helpers_root/dev_scripts_helpers/git/git_merge_master.py.log'
14:35:12 - INFO  hdbg.py init_logger:1177                               > /Users/saggese/src/umd_classes2/helpers_root/dev_scripts_helpers/git/git_merge_master.py --abort_if_not_ff --no_auto_merge --no_submodules
  ... Warning: Permanently added 'github.com' (ED25519) to the list of known hosts.
  ... Already up to date.
> /Users/saggese/src/umd_classes2/helpers_root/dev_scripts_helpers/git/git_merge_master.py --abort_if_not_ff --no_auto_merge --no_submodules
> invoke git_merge_master --abort-if-not-ff --no-auto-merge --no-submodules
14:35:13 - INFO  lib_tasks_git.py git_branch_copy:878                   new_branch_name='Skills improvements and documentation enhancements with transform text refactor'
Switched to branch 'master'
Your branch is up to date with 'origin/master'.
14:35:20 - INFO  hdbg.py init_logger:1177                               > /Users/saggese/src/venv/client_venv.helpers/bin/invoke git_branch_create --branch-name Skills improvements and documentation enhancements with transform text refactor
# git_branch_create: branch_name='Skills improvements and documentation enhancements with transform text refactor', issue_id=0, repo_short_name='current', suffix='', only_branch_from_master=True, check_branch_name=True, create_pr=True, abort_if_not_clean=True, abort_if_not_master=True, dry_run=False
14:35:21 - INFO  hdbg.py init_logger:1167                               Saving log to file '/Users/saggese/src/umd_classes2/helpers_root/dev_scripts_helpers/git/git_branch_create.py.log'
14:35:21 - INFO  hdbg.py init_logger:1177                               > /Users/saggese/src/umd_classes2/helpers_root/dev_scripts_helpers/git/git_branch_create.py --branch_name Skills improvements and documentation enhancements with transform text refactor
14:35:21 - INFO  git_branch_create.py _create_branch:155                branch_name='Skills improvements and documentation enhancements with transform text refactor'
Traceback (most recent call last):
  File "/Users/saggese/src/umd_classes2/helpers_root/dev_scripts_helpers/git/git_branch_create.py", line 296, in <module>
    _main(_parse())
  File "/Users/saggese/src/umd_classes2/helpers_root/dev_scripts_helpers/git/git_branch_create.py", line 281, in _main
    _create_branch(
  File "/Users/saggese/src/umd_classes2/helpers_root/dev_scripts_helpers/git/git_branch_create.py", line 158, in _create_branch
    _dassert_valid_branch_name(branch_name)
  File "/Users/saggese/src/umd_classes2/helpers_root/dev_scripts_helpers/git/git_branch_create.py", line 107, in _dassert_valid_branch_name
    hdbg.dassert(
  File "/Users/saggese/src/umd_classes2/helpers_root/helpers/hdbg.py", line 172, in dassert
    _dfatal(txt, msg, *args, only_warning=only_warning)
  File "/Users/saggese/src/umd_classes2/helpers_root/helpers/hdbg.py", line 155, in _dfatal
    dfatal(dfatal_txt)
  File "/Users/saggese/src/umd_classes2/helpers_root/helpers/hdbg.py", line 84, in dfatal
    raise assertion_type(ret)
AssertionError:
################################################################################
* Failed assertion *
cond=None
Branch name must follow convention: '{RepoPrefix,Amp,...}TaskXYZ_...'
################################################################################
```

### [ ] Rename actions

/pytest.triage_github_unit_tests -> triage_ci (Improve)

/Users/saggese/src/umd_classes2/helpers_root/.claude/skills/github.get_pr_ready_to_merge/SKILL.md

/Users/saggese/src/umd_classes2/helpers_root/.claude/skills/github.get_pr_to_commit_state/SKILL.md
-> get_pr_to_pass_ci ?

/Users/saggese/src/umd_classes2/helpers_root/.claude/skills/github.get_pr_to_pass_local_tests/SKILL.md

