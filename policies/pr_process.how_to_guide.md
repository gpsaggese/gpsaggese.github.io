# How to Create a PR With Claude Code

- A PR (pull request) asks to add your changes to the shared code; GP reviews it
  before it is merged
- Claude Code is an AI assistant that reads and writes code with you
- These are the steps, in order, from picking a task to a PR of the quality GP
  expects; each step links to the doc with the details
- Two PRs made this way, to use as examples:
  - [#619](https://github.com/gpsaggese/gpsaggese.github.io/pull/619): a new
    notebook, with five small commits and a full testing section
  - [#654](https://github.com/gpsaggese/gpsaggese.github.io/pull/654): fixes, with
    test counts before and after

## 1. Get Ready

- Do the onboarding checklist GP posts in your project's main issue, e.g., the
  [checklist on #590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590#issuecomment-6075690143):
  - Read the paper
  - Read the code in the project folder
  - Run `docker_jupyter.sh` and learn how the project is built and run
  - Read the Claude Code rules for code and notebooks:
    `.claude/skills/coding*` and `.claude/skills/notebook.*`
  - Read all the markdown files in the project folder
- Read GP's [AI policy](../website/docs/blog/posts/My_AI_Policy.md) and the
  [FAQ](FAQ.md)
  - Why: you must be able to explain every line you submit, and no AI slop

## 2. Pick a Task

- Take a task from the project's master plan (the file with the project's goals
  and tasks) that has a GitHub issue (a task page on GitHub)
  - Why: we only work on tasks that are on the plan and in GitHub
  - E.g., the task list in
    [`plan.racket_strategy.md`](../research/Optimal_strategy_for_racket_sports/plan.racket_strategy.md)
- Post in the issue when you expect to finish (ETA), and update it if it changes
- Post a short update in the issue every 2-3 days: done, next, blockers

## 3. Investigate Before Changing Anything

- Read the issue, its comments, and the code the task touches; change nothing yet
- Write down which files and functions the task affects, and any problem you find
- Ask GP in the issue if anything is unclear
  - Why: a wrong guess costs more time than a question

## 4. Create a Branch in a Worktree

- A branch is your own line of changes, separate from `master` (the shared
  version)
- Name it `UmdTask<issue>_<short_title>`, e.g.,
  `UmdTask623_Racket_sports_fix_private_renames_and_serve_direction`
- Create it in a worktree: a second folder of the repo (the project's code), on
  your branch
  - Why: your main folder stays as it is, e.g., for running Jupyter
  ```bash
  > git fetch origin master
  > git worktree add -b UmdTask123_Short_title ~/wt_123 origin/master
  ```
- Fill in the shared tools folder `helpers_root` in the worktree, over HTTPS:
  ```bash
  > cd ~/wt_123
  > git -c url."https://github.com/".insteadOf="git@github.com:" \
      submodule update --init helpers_root
  ```
- If you are not a collaborator on the repo, use the fork steps in
  [`how_to_contribute.md`](../class_project/how_to_contribute.md) instead

## 5. Work in Small Steps

- Split the task into small steps; finish and test each one before the next
  - Why: small steps are easy to test, to explain, and to review
- Make one commit per step: a commit saves a step with a short message
  - Add only the files of that step (`git add <file>`), never `git add .`
  - Say what changed and give the issue, e.g.,
    `Make draw_court() public again (#590)`
  - No extra lines at the end of the message (e.g., `Co-Authored-By`)

## 6. Follow the Rules for Each File Type

- `.claude/rules.md` lists the rules for each type of file; Claude Code follows
  them, and so do you:
  - **Python code**: `coding.rules.md`, template `coding.template.py`
  - **Unit tests**: `testing.rules.md`, template `testing.template.py`
  - **Notebooks**: `notebook.rules.md`, templates `notebook.template.ipynb` and
    `notebook_utils_template.py`
  - **Markdown**: `markdown.rules.md` and `text.rules.md`
  - **Master plan**: `auto_task.rules.md`, template `auto_task.template.md`
  - Rules are in `.claude/skills/`, templates in `.claude/templates/`
- Put all the code in a `_utils.py` file, with a notebook that shows it (one of
  GP's musts)
- Skills are saved instructions for common jobs, run in Claude Code as
  `/<name>`, e.g.:
  - `/notebook.refactor_to_utils`: move notebook code into the `_utils.py` file
  - `/testing.write_unit_tests`: plan and write the unit tests for a file

## 7. Understand Every Line

- Keep notes in plain words as you go: what each function does, and why you made
  each choice
  - Keep them in a personal file that you do not commit
- Treat what Claude Code writes as a first draft: read it, question it, and
  rewrite it (see
  [Edit Code/Text Written by AI](https://github.com/causify-ai/helpers/blob/master/docs/code_guidelines/all.submit_code_for_review.how_to_guide.md#edit-codetext-written-by-ai))
  - Why: GP may ask about any line, and you must answer

## 8. Test in Layers

- Run everything inside the project's Docker container (a ready-made environment
  with the right tools), e.g., with `./docker_bash.sh`
- From fast to thorough:
  - **Unit tests**: small automatic checks of each function, in
    `test/test_<file>.py`
  - **End-to-end check**: use the notebook or widget as a user would, and check
    what it shows
  - **Full test suite**: all the project's tests; write down the pass and fail
    counts before and after your change
  - **Linters**: tools that check style and simple mistakes; the repo's are in
    `.pre-commit-config.yaml`
  - **Notebook top to bottom**: restart and run all cells; the notebook's two
    files (`.ipynb` and `.py`) must match
  - **Fresh copy**: download your branch into a new folder and run the tests
    there, to catch files you forgot to commit
  - **Stress and mutation checks**, for important work: many random inputs, and
    small bugs planted on purpose to confirm the tests catch them
  - Why: each layer catches mistakes the others miss

## 9. Open the PR

- Push your branch (upload your commits) with a normal push; never force push,
  and never push to `master`
  ```bash
  > git push -u origin UmdTask123_Short_title
  ```
- Title: the branch name
- First line: `Closes #<issue>`, so the issue closes when the PR is merged
- Description sections, as in #619:
  - **Summary**: what the PR does
  - **Testing**: what you ran, with the results
  - **Design choices**: each decision that is not obvious, and why
  - **Limitations**: what the work does not do
  - **Pre-existing issues found**: problems you saw but did not fix here
- Last line: "I used Claude Code to help write and test this code, and I
  reviewed every change."
- Add GP as reviewer, and add the PR to the project's GitHub project
- GitHub runs automatic checks on each PR; if one fails, find out whether your
  change caused it and say so in the PR, as in #654

## 10. After the PR

- In the issue checklists, tick each finished item with a short link to the
  result, e.g., `- [x] Update the master plan: PR #671`
- Update the project's log (the list of what was done), the task list in the
  master plan, and the paper when the work changes them
- Answer review comments with new commits on the same branch
- When the PR is merged, remove the worktree; the branch stays:
  `git worktree remove ~/wt_123`

## Known Problems in the Docker Container

- `pytest` is not installed: run `pip install pytest` inside the container
- `git` reports "dubious ownership": run
  `git config --global --add safe.directory /git_root` inside the container
- Tests leave files owned by `root` (e.g., `tmp.pytest.log`): run the container as
  your user (`docker run -u $(id -u):$(id -g) ...`), and never commit those files
