---
title: "How to Run Claude on GitHub"
draft: true
authors:
    - gpsaggese
date: 2026-09-12
description:
categories:
    - AI Coding
    - Developer Tools
---

TL;DR: Wire up one GitHub Actions workflow and one secret, then trigger Claude
on any issue or PR by mentioning `@claude`.

<!-- more -->

- This post covers how to set up
  [`claude-code-action`](https://github.com/anthropics/claude-code-action) so
  Claude runs directly inside GitHub Actions on a repo, picking up issues and PR
  comments and pushing commits back
- It walks through the four steps needed to get from zero to a working setup:
    - Generate and store the `CLAUDE_CODE_OAUTH_TOKEN` secret
    - Add the `claude.yml` workflow to the repo
    - Trigger a run with an `@claude` mention
    - Check the run's progress

# Overview

- `claude-code-action` is a GitHub Action that:
    - Listens for `@claude` mentions in issues, PR comments, and PR reviews
    - Checks out the repo and runs Claude Code against the mentioned task
    - Pushes commits and comments back on the same PR
- Two things need to exist before any of this works:
    - A workflow file at `.github/workflows/claude.yml` in the repo, registered
      on the default branch
    - A `CLAUDE_CODE_OAUTH_TOKEN` secret on the repo
- The end-to-end flow, once both are in place:

    ```mermaid
    sequenceDiagram
        participant Dev as Developer
        participant GH as GitHub (issue / PR)
        participant Actions as GitHub Actions
        participant Claude as claude-code-action

        Dev->>GH: Comment "@claude ..." on an issue or PR
        GH->>Actions: Fires issue_comment event
        Actions->>Actions: Checks claude.yml on default branch
        Actions->>Claude: Runs job (checkout, incl. submodules)
        Claude->>Claude: Reads task, writes code
        Claude->>GH: Pushes commit(s) to the PR branch
        Claude->>GH: Comments with a summary
    ```

# Set Up the `CLAUDE_CODE_OAUTH_TOKEN` Secret

- Claude Code recognizes two different credentials, billed in two different
  ways:
    - `ANTHROPIC_API_KEY`: a static key from the Claude Console, billed
      pay-as-you-go per token against an API workspace
    - `CLAUDE_CODE_OAUTH_TOKEN`: a long-lived OAuth token tied to a Claude.ai
      subscription (Pro, Max, Team, or Enterprise), meant for non-interactive
      environments like CI
- The practical difference is the billing model:
    - An API key meters every token against the Console account
    - An OAuth token draws down the same rolling usage allowance as logging in
      at claude.ai
- Generate the OAuth token from a terminal where Claude Code is already logged
  in to the subscription that should be billed:
    ```bash
    > claude setup-token
    ```

    - The command opens a browser link for authorization
    - It prints a token starting with `sk-ant-oat01-`
- Store that token as a repo secret:
    ```bash
    > gh secret set CLAUDE_CODE_OAUTH_TOKEN --repo <owner>/<repo>
    ```

    - `gh` prompts for the value: paste the token and press enter
- Confirm the secret exists (this only lists names, never values):
    ```bash
    > gh secret list --repo <owner>/<repo>
    ```

# Set Up the Repo Workflow

- The [`helpers`](https://github.com/causify-ai/helpers) repo ships a reference
  [`claude.yml`](https://github.com/causify-ai/helpers/blob/master/.github/workflows/claude.yml)
  under `.github/workflows/`, meant to be reused by every repo that includes
  `helpers` as the `helpers_root` submodule
- That file only lives inside the submodule, so it never runs on its own:
    - GitHub Actions only registers workflows that sit in the outer repo's own
      `.github/workflows/` directory
    - A copy that exists only inside `helpers_root/.github/workflows/` is
      invisible to GitHub Actions
- Copy `claude.yml` into the outer repo's `.github/workflows/claude.yml`, with
  one addition if the repo uses `helpers_root` as a submodule: check out
  submodules recursively, since the repo's shared Claude Code config
  (`.claude/`, skills, rules) lives inside `helpers_root`
    ```yaml
    - name: Checkout repository
      uses: actions/checkout@v4
      with:
          fetch-depth: 1
          submodules: recursive
    ```
- Commit and push `claude.yml` to the default branch (`master`)
    - `issue_comment`, `issues`, and `pull_request_review*` events resolve the
      workflow definition from the default branch, not from the PR branch
    - A copy sitting only on a feature branch or an open PR never triggers, even
      with a matching `@claude` comment on that same PR
- Confirm GitHub picked it up:
    ```bash
    > gh workflow list --repo <owner>/<repo>
    ```

    - The output should show `Claude Code` with status `active`

# Trigger a Run

- Once the secret and the workflow are both in place, Claude starts on an
  explicit `@claude` mention in:
    - A comment on an issue or PR
    - A PR review comment
    - A PR review body
    - The title or body of a newly opened issue
- Assigning an issue or a PR to a user does not, by itself, trigger a run: the
  mention has to be explicit text somewhere the workflow reads
- The simplest trigger is a plain PR comment:
    ```bash
    > gh pr comment <PR_NUM> --repo <owner>/<repo> --body "@claude please implement this task"
    ```

# Check Progress

- List recent runs of the workflow:
    ```bash
    > gh run list --repo <owner>/<repo> --workflow=claude.yml --limit 5
    ```
- View a specific run:
    ```bash
    > gh run view <RUN_ID> --repo <owner>/<repo>
    ```
- Watch a run live until it finishes:
    ```bash
    > gh run watch <RUN_ID> --repo <owner>/<repo>
    ```
- On github.com, the same status shows up on the PR itself:
    - The Checks tab lists the `Claude Code` job and its status
    - The Conversation tab shows Claude's own comments as it works

# Common Pitfalls

- **Missing secret**: the run starts and fails immediately if
  `CLAUDE_CODE_OAUTH_TOKEN` was never set on the repo
- **Workflow only on the submodule**: `claude.yml` inside
  `helpers_root/.github/workflows/` is not enough; it has to be copied into the
  outer repo's own `.github/workflows/`
- **Workflow only on a branch**: pushing `claude.yml` to a feature branch, then
  commenting `@claude` on a PR from that same branch, does not trigger anything;
  the file needs to be on the default branch first
- **No `@claude` mention**: assigning an issue, or writing a comment that only
  references Claude indirectly (e.g., "let Claude handle this"), never triggers
  the action; the literal string `@claude` has to be present
