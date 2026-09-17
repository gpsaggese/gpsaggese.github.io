I want to create a document that explains the auto_task workflow, including the
different ways of executing tasks.

I want to do it without losing any information from the docs below. So if there is
content to be moved, we need to move it from the files

Let's write a technical reference in how_to.auto_task.md but I want to make sure the
information from the original source is not replicated,

How would you do that?

1) Read the following resources

- ./helpers_root/dev_scripts_helpers/ai/todo_janitor.template.md
- ./helpers_root/todo_janitor.prompt.update_plan.md
- ./helpers_root/todo_janitor.README.md

- website/docs/blog/posts/:
  - draft.how_to.My_agentic_engineering_flow.md
  - draft.how_to.Stacked_PRs_for_agentic_developent.md
  - draft.how_to.A_queue_of_AI_coding_agents.md

2) Read the skills
.claude/skills/auto_task.create_specs_from_todos/SKILL.md
.claude/skills/auto_task.criticize/SKILL.md
.claude/skills/auto_task.execute_interactively/SKILL.md
.claude/skills/auto_task.execute_remotely_with_single_pr/SKILL.md
.claude/skills/auto_task.execute_with_stacked_prs/SKILL.md

.claude/skills/auto_task.rules.md
.claude/templates/auto_task.template.md

3) Read the content below
```
### [ ] Document the flow

- Go to `master`

- Create a `tasks.md` (e.g., from `msml610/prompt.slides_and_book_flow.md`) in the
  auto_task format

- Review the task with
  ```
  claude> /auto_task.criticize tasks.md
  ```

- Then kick off the execution
  ```
  claude> /auto_task.execute_with_stacked_prs tasks.md
  claude> /auto_task.execute_interactively tasks.md
  ```

 /auto_task.execute_remotely_with_single_pr tasks.md

- // TODO(ai_gp): Describe the auto task rules

.claude/skills/auto_task.rules.md
.claude/templates/auto_task.template.md

- Describe the auto_task skills

> mdm skill l auto_task
auto_task.create_specs_from_todos
auto_task.criticize
auto_task.execute_interactively
auto_task.execute_with_stacked_prs

- Once everything is clear

git_create_issue_and_branch.py --gh_issue_title "Improve msml610/3.2 and 3.3 slides"

Close the wrong PR

gh pr close UmdTask557_Improve_msml6103_2_and_3_3_slides -c "Wrong" --delete-branch

- Explain the /pr.* skills

### Document the flow for 

/auto_task.execute_remotely_with_single_pr

This even creates automatically the issue and branch or one
can do it manually with:

> git_create_issue_and_branch.py --gh_issue_title "Replace pathlib.Path uses below with os.path" --gh_issue_body_file ./instr.md --submodules --no_abort_if_not_clean

### [ ] Create one or multiple PRs

- Make a decision based on `* Affected repos:` using --submodule depending

```

## Plan: Write `helpers_root/how_to.auto_task.md`

- [x] Draft outline for `helpers_root/how_to.auto_task.md` and get it approved
- [x] Write full document content following `.claude/skills/markdown.rules.md`
  and `.claude/skills/text.rules.md`
- [x] Verify formatting against both rule files
- [x] `git add` the new file (do not commit)

## Result
- Done: wrote `helpers_root/how_to.auto_task.md`, a narrative how-to covering
  the auto_task pipeline (create, criticize, three execute modes), pointing to
  `auto_task.rules.md` and each `SKILL.md` instead of duplicating conventions
  - `git add`-ed in the `helpers_root` repo, not committed
- Not done: nothing else from `instr.md`'s other scratch notes (pr.* skills,
  closing wrong PRs, etc.) was in scope for this sub-task
