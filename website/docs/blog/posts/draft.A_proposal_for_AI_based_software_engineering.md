---
title: "One Approach for Organizing Software Development for Agentic AI"
draft: true
authors:
  - gpsaggese
date: 2026-10-19
description:
categories:
  - AI Coding
  - Software Engineering
---

TL;DR: Treat human understanding as the scarce resource: cap PR size, assign owners,
enforce the architecture with tools, and make every PR something reviewers can run.

<!-- more -->

# One Approach for Organizing Software Development for Agentic AI

## Scope and Assumptions

- This playbook assumes:
  - Python services
  - GitHub (CODEOWNERS, branch protection, CI)
  - Claude Code (`CLAUDE.md`, `.claude/`)
- Other stacks and agents have equivalents: only the tool names change, not the ideas
  - E.g., `AGENTS.md` is the instruction file read by other coding agents

## The Problem

### The Pre-AI Way of Building Software

// TODO(gp): Check the basis for "100 lines", "10x" and "5-10x" : unsourced estimates
- Everybody understood the architecture and the specs very well since it was needed
  to write the code (and the unit tests)
- The code was written to be easy to understand for humans
- The time to write the code was 10x larger than the time to review it, so it was
  critical to have a common understanding of the system and it was easy to stay
  in sync through reviews
- PRs were very small and it was clear what each of them did and the trajectory

- Things were "slow" (say 100 lines of prod code per dev per day) but somehow under
  control

### The Interim-AI Way of Building Software

- When AI started to become good at writing code, coders trained in the old way would
  use AI to just accelerate the previous pre-AI flow
- The pace of delivering code did increase 5-10x (I've measured on my self)
- Review effort per line stayed the same, so total review time grew with the volume
  of code
  - It was still sustainable: reviewers read every change and understood it
- Humans were still in control

### The New-AI Way of "Building Software"

- AI has made writing code cheap and instantaneous
- But understanding and verifying code still costs what it always did
- Now generation of code is outpacing comprehension and we are getting:
  - AI-generated code that no human read in full
  - PRs too large to review properly
  - AI-generated code reviewed with AI, with nobody accountable for the result
  - Replies to AI comments with AI
  - Architecture that erodes one "reasonable-looking" change at a time
  - A system nobody can explain

// TODO(ai_gp)
- I am seeing:
  - Coders write code with AI
  - People use AI to review the code
  - Coders respond to questions from reviewers using AI
  - The PR is merged and nobody understands what the system does, how it does it, and
    so on

- In practice everyone approves changes without reading them and nobody is really
  paying attention

- Is this bad?

- The problem with this approach is that you quickly get into spaghetti code
  territory and it will take months to clean up what was generated in days and bring
  it back to something that makes sense to humans

- But why is spaghetti code bad?
  - Nobody can change it safely, since the effects of a change are unknown
  - Bugs are hard to find and debug
  - New people take a long time to onboard
  - Security problems hide in code that nobody understands

- One approach is to say "Who cares? machines will get so good that they will do the
  work of cleaning their own mess". This is possible, but unclear when, if, and how
- In the meantime humans need to come up with a process that keeps this mess under
  control

// TODO(gp): Check the mathematicians analogy and add a reference
- Many of these discussions are similar to what mathematicians are discussing, now
  that AI is proving to be able to automate some of the mathematician workflow
  - Generating a proof is getting cheap, but verifying it is not

## Principles

- Treat human understanding as the scarce resource, and organize the workflow around
  protecting it
  - Sections 1, 5, 6: small PRs, owners, and PRs that reviewers can run and
    understand
- Humans own intent, architecture and approval
  - Sections 2, 5: architecture map, ADRs, plans, and code owners
- Agents own implementation inside clear, machine-checked boundaries
  - Sections 3, 4: import-linter, automated gates, hooks, and shared configuration
- Section 7 measures whether these principles work

## 1. Stop the Bleeding (Week 1)

### 1.1 Cap PR Size

- Hard limit of **400 changed lines** per PR, excluding lockfiles, migrations,
  generated files and notebooks (`.ipynb`)
  - This is enforced in CI
- Aim for 300 lines or fewer: the local hook blocks above 300 (see 4.3)
- PRs adding more than ~100 lines must link a plan (see 5)
- Bigger work ships as **stacked PRs**, one concern each
- A tech lead can override this for genuine exceptions (e.g., mechanical renames)

### 1.2 A Human Owns and Understands Every PR

- The PR author must be able to walk a reviewer through the change **without the
  AI**
  - "Claude wrote it" is not an explanation
  - If they can't explain it, it doesn't merge
- The author writes the "what and why" in the PR description themselves, no AI
  accepted
  - CI checks that the section is filled in, since detecting AI-written text is
    unreliable
  - The author also answers reviewer questions in the PR thread themselves
  - Other sections can be agent-assisted (see the PR template in 5)

### 1.3 AI Review Is a First Pass, Never an Approval

- Use Claude to catch bugs, missing tests and style issues before human review
- Require at least one **human approval from the code owner** of every touched module
  - E.g., branch protection: "Require review from Code Owners"
  - Each module needs at least two code owners or a named fallback, since GitHub
    does not count the author's own approval
- The same tool or person must not both generate and approve a change

### 1.4 Limit Work in Progress

- For the initial few weeks, cap open PRs per engineer (e.g., 2) and merge on a
  steady cadence
- A temporary slowdown is far cheaper than months of untangling a system nobody
  understands

## 2. Recover the Architecture (Weeks 2–4)

- Agents can't respect an architecture that exists only in people's heads, or nowhere

### 2.1 Map What Exists

- Ask Claude Code to map the repo's modules, dependencies and main data flows into
  `docs/architecture.md` (e.g., a Mermaid diagram)
- **Have a senior engineer verify every line.** An unverified AI-generated map is
  worse than none

### 2.2 Decide the Target Structure

- A layered structure works well for most Python services:

| Layer          | Package                | Owns                                   | May import                         |
| -------------- | ---------------------- | -------------------------------------- | ---------------------------------- |
| API            | `myapp.api`            | HTTP routes, request/response schemas  | application                        |
| Application    | `myapp.application`    | use cases, orchestration, transactions | domain                             |
| Domain         | `myapp.domain`         | business rules, entities, pure logic   | stdlib, no framework or IO         |
| Infrastructure | `myapp.infrastructure` | DB, queues, external APIs              | domain (implements its interfaces) |

- The Domain layer must stay free of frameworks and IO
  - The contract in 3.1 forbids them by name: extend the list as new libraries
    appear
- Feature modules (`billing`, `accounts`, `pricing`...) must not import each other's
  internals
  - They talk only through the public interface in `<module>/__init__.py`
  - In the domain layer they do not talk at all: they are independent, and the
    application layer orchestrates them

### 2.3 Record Decisions as ADRs

- Keep short Architecture Decision Records in `docs/adr/`
- Write each decision so an agent can follow it literally:
  ```text
  All external HTTP calls go through `myapp.infrastructure.http.Client`; nothing else
  imports `httpx`
  ```
- Turn every decision that can be checked mechanically into an import-linter
  contract (see 3.1)
  - E.g., the decision above is enforced by the "Only the HTTP client imports
    `httpx`" contract

### 2.4 Assign Owners

- Use `.github/CODEOWNERS` so every module has a human who understands it and must
  approve changes:
- E.g.,
  ```text
  *                          @your-org/tech-leads
  /CLAUDE.md                 @your-org/tech-leads
  /.claude/                  @your-org/tech-leads
  /docs/adr/                 @your-org/tech-leads
  /src/myapp/**/billing/     @your-org/billing-team
  /src/myapp/**/accounts/    @your-org/accounts-team
  /src/myapp/api/            @your-org/tech-leads
  /migrations/               @your-org/tech-leads
  ```

## 3. Make the Architecture Machine-enforced (Weeks 2–4)

- Agents follow rules reliably when the rules are explicit **and** checked
  automatically

### 3.1 Architecture Tests with Import-linter

- Add to `pyproject.toml` and run `lint-imports` in CI:

  ```toml
  [tool.importlinter]
  root_package = "myapp"
  include_external_packages = true

  [[tool.importlinter.contracts]]
  name = "Layered architecture"
  type = "layers"
  layers = [
      "myapp.api",
      "myapp.application | myapp.infrastructure",
      "myapp.domain",
  ]

  [[tool.importlinter.contracts]]
  name = "API talks only to the application layer"
  type = "forbidden"
  source_modules = ["myapp.api"]
  forbidden_modules = ["myapp.domain", "myapp.infrastructure"]
  allow_indirect_imports = true

  [[tool.importlinter.contracts]]
  name = "Domain has no framework or IO dependencies"
  type = "forbidden"
  source_modules = ["myapp.domain"]
  forbidden_modules = ["fastapi", "sqlalchemy", "requests", "httpx", "boto3"]

  [[tool.importlinter.contracts]]
  name = "Only the HTTP client imports httpx"
  type = "forbidden"
  source_modules = ["myapp"]
  forbidden_modules = ["httpx"]
  ignore_imports = ["myapp.infrastructure.http -> httpx"]

  [[tool.importlinter.contracts]]
  name = "Feature modules are independent"
  type = "independence"
  modules = ["myapp.domain.billing", "myapp.domain.accounts", "myapp.domain.pricing"]
  ```

- The layers contract alone lets a higher layer import every lower layer, so the
  API contract enforces the "May import" column of the table in 2.2
- The independence contract covers the domain layer, where features must not know
  each other
  - Calls between features in the application layer go through their public
    interfaces and are checked in review by the code owners
- Your current code will probably fail. List today's violations under
  `ignore_imports`, burn them down over time, and let any **new** violation fail CI
  immediately
  - Add a CI check that the number of `ignore_imports` entries never increases, so
    the list only shrinks

### 3.2 Other Automated Gates

- **ruff** with complexity caps (`max-complexity = 10`, `max-args = 6`)
  - This pushes agents toward small functions
  - The caps do nothing unless the rules C901 and PLR0913 are selected:
    ```toml
    [tool.ruff.lint]
    extend-select = ["C901", "PLR0913"]

    [tool.ruff.lint.mccabe]
    max-complexity = 10

    [tool.ruff.lint.pylint]
    max-args = 6
    ```
- **mypy --strict** on the package
  - Legacy code will not pass: list the modules that are not yet clean under
    `[[tool.mypy.overrides]]` with `ignore_errors = true`, and burn the list down
    like `ignore_imports`
- **pytest + diff-cover**: at least 80% coverage on changed lines
- Contract tests on module public interfaces and API schemas

## 4. Configure Claude Code for the Team (Weeks 2–4)

- Commit the `.claude/` folder and `CLAUDE.md` files to the repo so every engineer's
  agent works under the same rules

### 4.1 Root `CLAUDE.md`

- Loaded into every session: keep it short and current
- It should contain:
  - What the system does (3–5 sentences)
  - The layer table and module rules above, plus "if a change seems to need a
    boundary violation, **stop and ask**"
  - The workflow: plan first, under 300 lines per change, tests first, never change
    a public interface without saying so
  - Commands: install, test, lint, type check, `lint-imports`
  - Don'ts: new dependencies without approval, editing merged migrations, silencing
    type or lint errors without a reason

### 4.2 Per-module `CLAUDE.md`

- One per significant module, written by its owner: purpose, public interface,
  **invariants that must never break**, and gotchas. Example:

- E.g.,
  ```markdown
  # billing
  Owner: @billing-team. Invoices, payments, refunds. NOT pricing.
  Public interface: create_invoice, apply_payment, refund.
  Invariants:
  - Invoice total is immutable once status is `issued`.
  - Money is an integer in minor units (cents) or a Decimal. Never float.
  - Every state change writes an InvoiceEvent in the same transaction.
  Gotcha: provider webhooks can arrive twice. Handlers must be idempotent.
  ```

### 4.3 Hooks: Catch Problems While the Agent Works, Not in CI

- In `.claude/settings.json`:
  ```json
  {
    "permissions": {
      "deny": ["Edit(vendor/**)", "Bash(git push --force*)"]
    },
    "hooks": {
      "PreToolUse": [
        {
          "matcher": "Edit|MultiEdit|Write",
          "hooks": [{ "type": "command",
                      "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/check_merged_migrations.py\"" }]
        }
      ],
      "PostToolUse": [
        {
          "matcher": "Edit|MultiEdit|Write",
          "hooks": [{ "type": "command",
                      "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/check_after_edit.py\"" }]
        }
      ],
      "Stop": [
        {
          "hooks": [{ "type": "command",
                      "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/check_diff_size.py\"" }]
        }
      ]
    }
  }
  ```

- **Before each edit:** block edits to migration files that already exist on
  `origin/main`, so merged migrations stay untouched while new migrations remain
  editable. Exit with code 2 so Claude sees the reason
- **After each edit:** run `ruff` on the file and `lint-imports`. On failure, exit
  with code 2 so the error is fed back to Claude and fixed immediately
- **When Claude finishes:** count changed lines against the PR base branch (the
  parent branch for a stacked PR, not `origin/main`; e.g., read it from
  `gh pr view --json baseRefName`), with the same exclusions as 1.1. Over 300,
  block and tell Claude to stop adding code and propose a split into stacked PRs
  - The hook must read `stop_hook_active` from its input and allow the stop when it
    is already true, otherwise it can loop forever

### 4.4 Custom Commands

- **`/make-plan <task>`** (`.claude/commands/make-plan.md`)
  - Avoid the name `/plan`, which can clash with the built-in plan mode command
  - Claude reads the relevant `CLAUDE.md` files and architecture doc, writes a plan
    to `docs/plans/<slug>.md`, and **stops without writing code**
  - Plan sections:
    - Problem, Approach (with rejected alternative)
    - Modules touched
    - Interface changes
    - New dependencies
    - Test plan
    - PR breakdown
    - Open questions
- **`/explain-pr`** (`.claude/commands/explain-pr.md`)
  - produces a _reviewer's guide_, not a summary
  - It covers
    - intent
    - the 3–5 places that need careful reading (file:line)
    - architecture impact
    - risks and untested paths
    - questions for the author
  - Instruct it to be skeptical and never say "looks good"

## 5. Change the Workflow: Spec First, Code Second (Weeks 2–4)

1. **Plan:** for anything beyond a small fix, run `/make-plan`
   - A human reviews the plan, which is about 30 lines instead of about 2,000 lines
     of diff across the stack
2. **Tests:** humans write or carefully review the acceptance tests
   - If the tests correctly capture the intended behavior, the implementation matters
     much less
3. **Implement:** Claude implements within the plan, with hooks enforcing boundaries
   and size
4. **Explain:** the author runs `/explain-pr`, edits the output, and puts it in the
   PR description
   - The author also adds the functionality summary, commands, notebook, and block
     diagram described in "Make PRs Demonstrable"
5. **Review:** the human code owner reviews, starting with the places the guide
   flags. Public interface changes also need a tech lead

- Enforce it in CI: PRs adding more than ~100 lines must link a plan
  (`docs/plans/...`) in the description
  - The plan is merged as its own small PR, reviewed by the code owner, before
    implementation starts
  - That merge is the plan approval counted in the metrics in 7

### Make PRs Demonstrable

- With giant AI-generated PRs, reading the diff is not enough for a human to judge
  a change
  - Reviewers should exercise the new behavior and see where it sits in the system
- Every PR description has four parts, adapted to the PR type (see below):
  - **Functionality added**: what the PR adds or changes from the user's point of
    view
    - Describe behavior, not a list of touched files
  - **Commands to see it in action**: a short list of copy-paste commands
    - E.g., run the new CLI on a sample input
    - E.g., start the service and call the new endpoint
    - The reviewer runs them from a clean checkout and sees the output
    - CI runs the commands as a smoke test and posts the real output on the PR, so
      the output is not written by the author or the AI
  - **A notebook**: lets reviewers "play" with the code
    - Commit it with the PR, with outputs stripped, and exclude it from the line
      count (see 1.1)
    - Reviewers change inputs and see how the behavior changes, instead of only
      reading code
    - CI executes it (e.g., `jupyter nbconvert --execute`), so it does not rot
  - **A block diagram**: the system's components and who does what
    - The author thinks hard about components and responsibilities, then refers to
      the diagram in the PR
    - AI tends to keep adding new objects, so each new class or module must have
      a place and a responsibility in the diagram
    - Use Mermaid: it diffs well and renders on GitHub and in MkDocs
    - The diagram is an excerpt of the architecture map in `docs/architecture.md`
      (see 2.1), with the nodes the PR adds or changes marked
    - A PR that adds a component also updates `docs/architecture.md`
    - Reviewers reject a new component whose responsibility overlaps an existing one
- What each PR type needs:
  - Feature: all four parts
  - Bug fix: commands that reproduce the bug before and after the fix
    - The notebook is optional
    - The diagram is needed only if components change
  - Refactor: no new functionality, so show that the tests are unchanged
    - The diagram is needed only if the structure changes
  - Docs, CI, dependency bumps: write "N/A: <reason>" for each part

### PR Template (`.github/pull_request_template.md`)

```markdown
## What and why
<!-- Human-written: your intent, not pasted from the agent. -->
Plan: docs/plans/xxx.md

## Functionality added
<!-- You edit, the agent may draft: observable behavior, not a list of files. -->

## See it in action
<!-- Copy-paste commands that run the new behavior. CI posts their output. -->
<!-- Write "N/A: <reason>" if the PR has no runnable behavior. -->
Notebook: docs/notebooks/xxx.ipynb

## Block diagram
<!-- Mermaid excerpt of docs/architecture.md. Mark what this PR adds or changes. -->

## Where reviewers should look
<!-- Edited /explain-pr output -->

## Author checklist
- [ ] I can explain every part of this change without the AI
- [ ] I ran it and checked the behavior myself, not just the tests
- [ ] The commands in "See it in action" work from a clean checkout
- [ ] Every new class or module appears in the block diagram
- [ ] I reviewed the tests, not only the code
- [ ] No new cross-module dependencies or public interface changes (or listed above)
- [ ] Docs / module CLAUDE.md updated if behavior or boundaries changed
```

## 6. Rebuild Shared Understanding as a Habit (Ongoing)

- **Weekly 30-minute architecture review:** new ADRs, structural changes, the
  `ignore_imports` count, the largest PRs, and reverts
- **Rotating walkthroughs:** an engineer presents a module they _didn't_ write
  - Gaps in understanding show up quickly
- **Docs live next to code:** a PR that changes behavior or boundaries updates
  `docs/architecture.md` or the module's `CLAUDE.md` in the same PR

## 7. Measure the Right Things (Ongoing)

- Stop celebrating output volume (PR count, lines written)
- Track:

| Metric                                           | Healthy direction             |
| ------------------------------------------------ | ----------------------------- |
| Median PR size                                   | Down, ideally under 300 lines |
| Time to first human review                       | Down                          |
| Revert / rollback rate                           | Down                          |
| Incidents traced to recent changes               | Down                          |
| Architecture violations (`ignore_imports` count) | Down                          |
| % of PRs with an approved plan                   | Up                            |

- Measure the baseline in week 1 and set the targets from it, since the table gives
  only directions
- If PR count rises while reverts and incidents also rise, you're not actually moving
  faster

## Costs and Trade-offs

- Slower first weeks: PR caps and WIP limits reduce output on purpose
- Senior time: verifying the architecture map, owning modules, and the weekly review
- Process overhead: a plan and demo material for each non-trivial PR
- Mitigation: start strict, loosen with evidence, and drop checks that do not catch
  real problems

## Rollout Plan

| When          | Actions                                                                                                                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Week 1**    | PR size check in CI; CODEOWNERS; PR template; branch protection requiring human code-owner approval; WIP limits                                                                                  |
| **Weeks 2–4** | Architecture map verified by seniors; root and per-module `CLAUDE.md`; import-linter with current violations ignored; commit `.claude/` with hooks and commands; start the `/make-plan` workflow |
| **Ongoing**   | Weekly architecture review; burn down ignored imports; module walkthroughs; track the metrics above                                                                                              |

- Start strict on limits
  - 300 lines locally
  - 400 in CI
  - 80% changed-line coverage

- Loosen only with evidence
  - E.g., median PR size, revert rate and incidents stay at their targets for
    several consecutive weeks (e.g., 4)

## Summary

- The bottleneck is human understanding, not code generation
- Protect it with small PRs, named owners, and PRs that reviewers can run and see
- Let tools enforce the architecture, so that agents work inside clear boundaries
- Measure revert rate, incidents, and architecture violations, not output volume
