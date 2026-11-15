# How We Work on Agentic Data Science

- Goal: a new student or RA reads this file and knows how we work on the project
- Process: `docs/planning/all.plan_work.how_to_guide.md` in `helpers`
- Dev setup and coding rules: issue #879

## Resources

- GitHub project: [Agentic_Data_Science](https://github.com/users/gpsaggese/projects/9)
- Code: `research/agentic_data_science/` in this repo, and
  `agentic_eda/bakeoff/` in [`gpsaggese/tutorials`](https://github.com/gpsaggese/tutorials)
- Google Doc of the project:
  [Agentic Data Science: Vision and Plan](https://docs.google.com/document/d/1fAao18l5oqeGvmGq4V4S0RGvXggXAFtsiv9gMIt0UlU)
- Paper: `papers/RL_for_Automated_EDA/`

## Master Plan

- File: `plan.md`

## Log

- File: `progress.md`
- Format: history table, plus one row per merged PR

## Task Rule

- Execute only the tasks that are in the master plan
- Create the GitHub issue for a task before starting it, and point it to the PR id
  in `plan.md` (e.g., `PR_B2`)
- Post an ETA in the issue when you take the task, and update it when it changes
- Start from the code that exists: every entry in `plan.md` names the files it
  builds on
- Keep each PR to one entry of the plan; a large entry becomes a stack of small
  commits in one PR

## AI Rule

- Skills to use in each step of the loop:
  `docs/tools/ai/all.ai_workflows.how_to_guide.md`
- Using AI is fine, but the author must be able to explain every line of the
  AI-written code

## Communication Rule

- GitHub for technical discussion
- Slack for quick pings
- Office hours for the rest

## Paper Rule

- Update the paper when there is new or important research (e.g., once a week), so
  that it reflects the state of the code

## Records Rule

- Keep every decision and communication in a non-ephemeral system
- Prefer markdown in the repo
- Never keep a decision only in Slack

## Example

- `gpsaggese/tutorials#706`: one PR with four small commits, each with its own tests
