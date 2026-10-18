# How We Run the Racket Sports Research Project

- How we work together on this project, written for a new student or RA joining it
- Based on GP's comments on
  [#625](https://github.com/gpsaggese/gpsaggese.github.io/issues/625#issuecomment-6185837906)
  and [#669](https://github.com/gpsaggese/gpsaggese.github.io/issues/669)
- Points to other docs instead of repeating them
- Keep this doc updated as we go: when the way we work changes, update it in a PR

## Project Links

- **Parent issue**:
  [#590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590)
- **GitHub project**:
  [Optimal_strategy_for_racket_sports](https://github.com/users/gpsaggese/projects/4)
- **Slack channel**: `#rl_for_racketsports`
- **Google Drive**:
  [root folder](https://drive.google.com/drive/folders/1NQB-LgY6Kn8Pw5Hbnp_1yCzN9HgoDIXe);
  there is no Google Doc yet
- **Paper**:
  [`paper.md`](../../papers/Optimal_strategy_for_racket_sports/paper.md)
- **Joining the research team**: Slack invite and weekly team sync in the
  [join doc](../../policies/how_to_join_the_research_team.md)

## Master Plan and Log

- [`plan.racket_strategy.md`](plan.racket_strategy.md): the master plan, with the
  roadmap, the milestones, the open points, and the task list
  - The milestones and the task list come with PR #671, not merged yet
- [`log.racket_strategy.md`](log.racket_strategy.md): what was done, newest entry
  first, with dates and links to issues, PRs, and milestones
  - Added by PR #660, not merged yet

## Tasks

- We only execute tasks that are on the master plan, after creating them in GH
  - For a new idea: add it to the plan, open an issue, then start working
- Each GH issue is in exactly one GH project: for this work,
  `Optimal_strategy_for_racket_sports`
- When you take a task, post an ETA in its issue, and update it if it changes
- One branch and one PR per issue, named `UmdTask<issue>_<short_title>`; the
  details are in the
  [#626](https://github.com/gpsaggese/gpsaggese.github.io/issues/626) doc
- A bulleted checklist in an issue or a comment is a list of tasks
  - Tick each box in GH when it is done, with a brief link to the result, e.g.,
    `- [x] Update the master plan with this discussion: PR #671`

## Claude Skills

- We use Claude Code with the repo's skills and rules (`.claude/rules.md`) to write
  code, tests, notebooks, and docs
- The step-by-step process for producing a PR is in
  [`pr_process.how_to_guide.md`](../../policies/pr_process.how_to_guide.md)

## Communication

- **GitHub**: technical discussion, in the issue or PR it belongs to
- **Slack**: quick pings
- **Office hours**: book a slot with GP through his
  [office hours form](https://forms.gle/FGcroUSrgRPbrBgK6), and join the weekly
  research team sync (see the join doc above)
- All decisions go in a non-ephemeral system: ideally markdown in the repo, or an
  issue or PR comment, never only Slack
  - A decision made on Slack or in a meeting is copied to its issue, e.g., the
    Milestone 2 discussion in
    [#624](https://github.com/gpsaggese/gpsaggese.github.io/issues/624)

## Paper

- Keep the paper in sync with the code, so that it represents the current state:
  about once a week, or when something new or important is found
- The paper's figures come from the notebooks
  ([#670](https://github.com/gpsaggese/gpsaggese.github.io/issues/670))

## Getting Started

- The onboarding items from GP's
  [checklist on #590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590#issuecomment-6075690143):
  - Read the paper
  - Read the code in `research/Optimal_strategy_for_racket_sports/`
  - Run `docker_jupyter.sh` and get familiar with the build and run system (as in
    DATA605 / MSML610)
  - Get familiar with the Claude skills in `.claude/skills/coding*` and
    `.claude/skills/notebook.*`
  - Read all the markdown files in `research/Optimal_strategy_for_racket_sports/`

## GP's Musts

- From the same checklist on #590:
  - All the code is in a `_utils.py` file, with a notebook to visualize it
  - Using AI is fine, but you must be able to explain exactly what every line of
    your code does and why
  - Absolutely no AI slop
