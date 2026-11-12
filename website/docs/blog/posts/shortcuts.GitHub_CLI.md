---
title: "GitHub CLI Shortcuts"
draft: false
authors:
  - gpsaggese
date: 2026-07-15
description: A guide to useful GitHub CLI commands
categories:
  - Developer Tools
---

TL;DR: A cheat sheet of `gh` (GitHub CLI) commands for repos, issues, PRs, and
workflows.

<!-- more -->

# GitHub CLI Shortcuts

A short guide to the most useful GitHub CLI commands for managing repositories,
issues, pull requests, and workflows.

// TODO(ai_gp): Convert into a table
```
  auth:          Authenticate gh and git with GitHub
  browse:        Open repositories, issues, pull requests, and more in the browser
  issue:         Manage issues
  pr:            Manage pull requests
  project:       Work with GitHub Projects.
  repo:          Manage repositories

  run:           View details about workflow runs
  workflow:      View details about GitHub Actions workflows
```

- `gh reference` to show a full reference of all the commands
  - E.g., `gh reference | vim -`

# Subcommands

## Authentication & Setup

- Check authentication status
  ```
  > gh auth status
  Logged in to github.com with oauth_token stored in /Users/saggese/.config/gh/hosts.yml
  ```

## Repository Management

- View repository details
  ```
  > gh repo view
  gpsaggese/gpsaggese.github.io
  No description provided
  GP's University of Maryland Machine Learning Classes
  ...
  ```

- List your repositories
  ```
  > gh repo list
  gpsaggese/umd_classes2              public
  gpsaggese/gpsaggese.github.io       public
  gpsaggese/ml-tutorials              private
  ```

## Issues

- List issues
  ```
  > gh issue list
  #42    Update documentation         open      about 2 days ago
  #41    Fix authentication bug       open      about 5 days ago
  #40    Add caching support          closed    about 1 week ago
  ```

- View a specific issue
  ```
  > gh issue view 42
  Update documentation
  #42 · OPEN
  opened by gpsaggese about 2 days ago
  This PR needs documentation updates.
  ```

- Create a new issue
  ```
  > gh issue create --title "Bug: offline mode" --body "App crashes when offline"
  Creating issue in gpsaggese/umd_classes2
  #43 (https://github.com/gpsaggese/umd_classes2/issues/43)
  ```

- Add a comment to an issue
  ```
  > gh issue comment 42 --body "I'm working on this"
  Commenting on issue #42 in gpsaggese/umd_classes2
  https://github.com/gpsaggese/umd_classes2/issues/42#issuecomment-1234567
  ```

- Close an issue
  ```
  > gh issue close 42
  Closed issue #42 in gpsaggese/umd_classes2
  ```

## Pull Requests

- List pull requests
  ```
  > gh pr list
  #38    Add feature flags               open      about 1 day ago
  #37    Refactor data pipeline          open      about 3 days ago
  #36    Update dependencies             merged    about 1 week ago
  ```

- View a PR
  ```
  > gh pr view 38
  Add feature flags
  #38 · OPEN
  opened by gpsaggese about 1 day ago
  Adds configurable feature toggles for A/B testing.
  ```

- Create a new PR
  ```
  > gh pr create --title "Add caching layer" --body "Improves query latency"
  Creating pull request for feature/caching into main
  #39 (https://github.com/gpsaggese/umd_classes2/pull/39)
  ```

- Check PR status
  ```
  > gh pr status
  Relevant pull requests in gpsaggese/umd_classes2
  Current branch
    #38  Add feature flags [OPEN]
         - Review by alice (approved)
         - Review by bob (changes-requested)
  ```

- Merge a PR
  ```
  > gh pr merge 36
  Merging PR #36 (Update dependencies)
  Merged via squash commit
  ```

## Workflows & Actions

- List GitHub Actions workflows
  ```
  > gh workflow list
  ID     Name               State  
  12345  Tests              active
  12346  Deploy             active
  12347  Lint Code          active
  ```

- List workflow runs
  ```
  > gh run list
  STATUS  TITLE                BRANCH  EVENT  ID        ELAPSED
  ✓       Tests                main    push   1234567   2m34s
  ✓       Tests                main    push   1234566   2m18s
  ✗       Tests                main    push   1234565   1m52s
  ```

- View a specific run
  ```
  > gh run view 1234567
  ID: 1234567
  Name: Tests
  Status: COMPLETED
  Conclusion: SUCCESS
  Branch: main
  Event: push
  ```

- Watch a run in progress
  ```
  > gh run watch 1234567
  Watching run 1234567...
  ✓ Test (3m2s)
  ✓ Build (2m45s)
  ✓ Deploy (1m30s)
  ```

## Useful Flags & Options

| Flag | Description |
|------|-------------|
| `-R, --repo <owner/repo>` | Specify repository (useful when not in a repo directory) |
| `--json` | Format output as JSON |
| `--limit <number>` | Limit the number of results |
| `-a, --assignee <user>` | Filter by assignee |
| `-L, --label <label>` | Filter by label |
| `-s, --state <state>` | Filter by state (open, closed, merged) |
| `--author <user>` | Filter by author |
| `--head <branch>` | Filter by head branch |
| `--base <branch>` | Filter by base branch |

## Working with Multiple Repositories

- List issues from a different repo
  ```
  > gh issue list -R owner/another-repo
  #5     Fix typo in README          open      about 2 hours ago
  #4     Update API docs             closed    about 1 week ago
  ```

- Check PRs across repos
  ```
  > gh pr list -R owner/repo1 && gh pr list -R owner/repo2
  repo1:
  #12    Feature X                   open
  repo2:
  #8     Bugfix Y                    open
  ```

# Useful One-Liners

- List all open PRs with JSON output
  ```
  > gh pr list --state open --json number,title,author
  [
    {"number":38,"title":"Add feature flags","author":{"login":"gpsaggese"}},
    {"number":37,"title":"Refactor data pipeline","author":{"login":"gpsaggese"}}
  ]
  ```

- Create an issue with a label
  ```
  > gh issue create --title "Bug Report" --label "bug"
  Creating issue in gpsaggese/umd_classes2
  #44 (https://github.com/gpsaggese/umd_classes2/issues/44)
  ```

- List PRs assigned to you
  ```
  > gh pr list --assignee @me
  #38    Add feature flags               open      about 1 day ago
  #37    Refactor data pipeline          open      about 3 days ago
  ```

# Common Workflows

- Finding current GitHub issue from the branch
  ```
  > git branch --show-current | sed 's/.*Task\([0-9]*\).*/\1/'
  1292
  ```

- Finding current GitHub PR
  ```
  > gh pr view --json number -q .number
  1293
  ```

- Creating a Pull Request
  ```
  > git checkout -b feature/my-feature
  Switched to a new branch 'feature/my-feature'
  
  > git add . && git commit -m "Add my feature" && git push origin feature/my-feature
  [feature/my-feature abc1234] Add my feature
  
  > gh pr create --title "Add my feature" --body "Description of changes"
  Creating pull request for feature/my-feature into main
  #39 (https://github.com/gpsaggese/umd_classes2/pull/39)
  ```

- Managing Issues
  ```
  > gh issue create --title "Bug: crashes on logout" --body "App crashes when user logs out" --label "bug,urgent"
  Creating issue in gpsaggese/umd_classes2
  #45 (https://github.com/gpsaggese/umd_classes2/issues/45)
  
  > gh issue comment 45 --body "I'm working on this"
  Commenting on issue #45 in gpsaggese/umd_classes2
  https://github.com/gpsaggese/umd_classes2/issues/45#issuecomment-1234567
  
  > gh issue close 45
  Closed issue #45 in gpsaggese/umd_classes2
  ```

# Additional Resources

- [Official GitHub CLI Documentation](https://cli.github.com/)
- [GitHub CLI Manual](https://cli.github.com/manual/)
- [GitHub API Reference](https://docs.github.com/en/rest)
