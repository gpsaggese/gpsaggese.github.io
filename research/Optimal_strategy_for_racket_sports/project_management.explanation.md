# How We Run the Racket Sports Research Project

- How we work together on this project, written for a new student or RA joining it
- Based on GP's comments on
  [#625](https://github.com/gpsaggese/gpsaggese.github.io/issues/625#issuecomment-6185837906)
  and [#669](https://github.com/gpsaggese/gpsaggese.github.io/issues/669), and on his
  rules for the research process on
  [#694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694#issuecomment-6712939277)
- Points to other docs instead of repeating them
- Keep this doc updated as we go: when the way we work changes, update it in a PR

## Project Links

- **Parent issue**:
  [#590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590)
- **GitHub project**:
  [Optimal_strategy_for_racket_sports](https://github.com/users/gpsaggese/projects/4)
- **Slack channel**: `#rl_for_racketsports`
- **Research Gdoc**:
  [Racket sports: research results](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit),
  with 2 tabs
  - `Research results`: one Heading 1 per experiment, with its setup, settings
    string, figures, decisions, and takeaways
  - `Project`: the agenda of our meetings and the
    [Questions for GP](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.xfydadrg3lvc)
- **Google Drive**:
  [root folder](https://drive.google.com/drive/folders/1NQB-LgY6Kn8Pw5Hbnp_1yCzN9HgoDIXe);
  the Gdoc is in its
  [`Optimal_strategy_for_racket_sports`](https://drive.google.com/drive/folders/1QPETTDAx9V2wo9PxCMrSV0R5KNyNTQRi)
  folder
  - Which folder holds the research Gdocs is pending GP:
    [Q2](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.21d3arlkot6c)
- **Paper**:
  [`paper.md`](../../papers/Optimal_strategy_for_racket_sports/paper.md)
- **Joining the research team**: Slack invite and weekly team sync in the
  [join doc](../../policies/how_to_join_the_research_team.md)

## Master Plan and Log

- [`plan.racket_strategy.md`](plan.racket_strategy.md): the master plan, with the
  roadmap, the milestones, the open points, and the task list
- [`log.racket_strategy.md`](log.racket_strategy.md): what was done, newest entry
  first, with dates and links to issues, PRs, and milestones
- Whether the experiments are logged in the log, in the Gdoc, or in both is pending
  GP:
  [Q1](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.21d3arlkot6c)

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
- A research question about a task is asked in the Gdoc, and the issue gets only a
  link to it, see [Research Questions](#research-questions)

## Claude Skills

- We use Claude Code with the repo's skills and rules (`.claude/rules.md`) to write
  code, tests, notebooks, and docs
- The step-by-step process for producing a PR is in
  [`pr_process.how_to_guide.md`](../../policies/pr_process.how_to_guide.md)
- Claude Code also drafts the research text: the Gdoc entries, the questions, and the
  log entries
  - The Drive connector reads the Gdoc and its comments, but cannot make an anchored
    comment or a suggestion, so a human reviews the text and posts it
  - An agent never answers a research question, accepts a suggestion, or resolves a
    thread

## Communication

- GP's rules
  ([#694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694#issuecomment-6712939277),
  2026-11-07):
  1. Stuff in GitHub and in Slack is meant to be lost
  2. GitHub is only for coding questions; research questions go in the Gdoc
  3. All the research discussions happen in the Gdoc, with comments and suggestion
     mode, and GitHub gets only a link to the question
- **Gdoc**: the research questions and their discussion, the results, and the
  research decisions, see [Research Questions](#research-questions)
- **GitHub**: tasks, coding questions, and code review, in the issue or PR they
  belong to, plus one line with the link to each research question
- **Slack**: quick pings, e.g., "I asked X in the Gdoc", with the link; never a
  question or a decision
- **Office hours**: book a slot with GP through his
  [office hours form](https://forms.gle/FGcroUSrgRPbrBgK6), and join the weekly
  research team sync (see the join doc above)
- Every decision is written where it lasts, never only in Slack, on GitHub, or in a
  meeting
  - A research decision goes in the Gdoc, as a "Decided" bullet under its experiment,
    with who decided and when
  - A decision of a meeting goes in the Gdoc too, e.g., the decisions of the
    2026-11-06 meeting under
    [Experiment 5](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.qrto9so35bnx)
  - A coding decision goes in the repo: the docs of the code, the README, or the log

## Research Questions

- The life of a research question:
  1. Ask it in the Gdoc, as a comment with an @mention of GP: on the result it is
     about, or on its line under
     [Questions for GP](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.xfydadrg3lvc)
     - One comment per question: the context in one line, the question, the options,
       and our recommendation
     - Propose a change of the text in suggestion mode
  2. Link it from GitHub: one line in the issue, with the link to the Gdoc heading or
     comment, and no discussion
  3. Answer it in the comment thread
  4. Decide: a "Decided" bullet under the experiment, with who and when, then resolve
     the thread
  5. Record it in the repo: the log entry of the PR that acts on it links to the
     decision, a new principle goes in
     [`design_principles.explanation.md`](design_principles.explanation.md) with the
     link as its source, and a change of scope goes in the plan
     - The log part is pending GP:
       [Q1](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.21d3arlkot6c)
- Do not write code that depends on an open research question
- When GP answers on GitHub or in a meeting, reply in the Gdoc thread with the answer
  and its link, record the decision, and resolve the thread
  - E.g., GP answered Q7 and Q8 on
    [#713](https://github.com/gpsaggese/gpsaggese.github.io/issues/713#issuecomment-6751589012),
    and both decisions are under
    [Experiment 6](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.kntokyu5w697),
    Decisions
- A coding question stays on GitHub, e.g., `inspect.getsource()` or
  `print_obj_info()` in the Gymnasium example, on
  [#846](https://github.com/gpsaggese/gpsaggese.github.io/pull/846)

## Paper

- Keep the paper in sync with the code, so that it represents the current state:
  about once a week, or when something new or important is found
- The paper's figures come from the notebooks
  ([#670](https://github.com/gpsaggese/gpsaggese.github.io/issues/670))
- Control the research through the paper
  ([GP on #694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694#issuecomment-6712939277)):
  a PR is hard to review by itself, so GP compares what it means with the paper or
  the Gdoc
  - Every PR that changes a result, a model, or an assumption names the paper section
    or the Gdoc heading that it changes, and updates it: the paper in the PR itself,
    the Gdoc when the PR is opened
  - So GP reviews the change of the paper or of the Gdoc, more than the code, as in
    [P14](design_principles.explanation.md#p14-the-plan-and-the-paper-control-the-code)
  - Such a PR has this section in its description, with each item ticked once it is
    answered:
    ```markdown
    ## Paper and Gdoc

    - [ ] Affected: <paper section and Gdoc heading, with links>
    - [ ] Results: <each result, figure, or number that changes, old -> new>, or none
    - [ ] Updated to match: <the paper in this PR; the Gdoc heading, with the date>
    - [ ] Assumptions: <each new or changed one, with its source (P16)>, or none
    ```
  - Any other PR says so in one line, e.g., "Paper and Gdoc: no change, tests only"
  - A PR that changes results with no change to the paper, the log, or the Gdoc is a
    red flag, and the checklist makes it visible
  - The checklist lives here for now: it can move to the research guide of helpers,
    or into a skill
    ([P15](design_principles.explanation.md#p15-put-the-smarts-in-the-skills)), if GP
    prefers:
    [Q3](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.21d3arlkot6c)

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
- Then read the research Gdoc: the `Research results` tab, then the Questions for GP
  in the `Project` tab

## GP's Musts

- From the same checklist on #590:
  - All the code is in a `_utils.py` file, with a notebook to visualize it
  - Using AI is fine, but you must be able to explain exactly what every line of
    your code does and why
  - Absolutely no AI slop
