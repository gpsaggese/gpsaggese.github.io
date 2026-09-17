---
title: "Stacked PRs for Agentic Development"
draft: true
authors:
  - gpsaggese
date: 2026-09-04
description:
categories:
  - AI Coding
  - Developer Tools
---

TL;DR: Use stacked PRs to let AI agents work longer without frequent context
switches, balancing productivity gains against review complexity

<!-- more -->

- This is the execution half of
  [My Agentic Engineering Flow](/blog/my-agentic-engineering-flow): once a `tasks.md`
  has passed `/auto_task.criticize`, this post is about which of the two execution
  modes to pick and how each one is implemented under the hood

## The Challenge

- When working with AI agents, two competing goals create tension:
  - **Goal 1**: Let agents run longer on extended task sequences (ideally full-day
    runs)
  - **Goal 2**: Review agent work frequently without losing context or having the
    agent make expensive mistakes
- As humans, we avoid constant context switching and prefer continuous blocks of
  similar work
  - E.g., reviewing five related changes is more productive than alternating between
    running tasks, reviewing, running again, reviewing again

## Understanding Your Workflow

- The right approach to direct agents depends on how tasks relate to each other:
  - **Independent tasks**: Agents run in parallel on separate paths, and you review
    all results when complete
  - **Sequential tasks**: Each task must complete before the next begins, and
    feedback between steps shapes future work, forcing a linear progression

```mermaid
graph TB
  classDef execute fill:#2ecc71,stroke:#27ae60,color:#fff,font-weight:bold
  classDef review fill:#3498db,stroke:#2980b9,color:#fff,font-weight:bold
  
  subgraph independent["Independent Tasks (Parallel)"]
    A1["Task 1"]:::execute -->|run| A1r["Review 1"]:::review
    A2["Task 2"]:::execute -->|run| A2r["Review 2"]:::review
    A3["Task 3"]:::execute -->|run| A3r["Review 3"]:::review
    A1r -->|finish| end1["Done"]
    A2r -->|finish| end1
    A3r -->|finish| end1
  end

  subgraph sequential["Sequential Tasks"]
    B1["Task 1"]:::execute -->|run| B1r["Review 1"]:::review
    B1r -->|feedback| B2["Task 2"]:::execute
    B2 -->|run| B2r["Review 2"]:::review
    B2r -->|feedback| B3["Task 3"]:::execute
    B3 -->|run| B3r["Review 3"]:::review
  end
```

- Independent tasks run in parallel without affecting each other
  - The problem is that
    - It's difficult to keep track of N independent tasks
    - Often we would rather finish completely one task in 1 day, rather than
      completing $N$ tasks in $N$ days

## Strategy 1: Interactive Workflows

- An interactive workflow runs fast feedback loops:
  - Write all task specifications upfront (single spec session)
  - Assign one task to the agent
  - Review the result
  - Assign the next task based on what you learned
- **Pros**:
  - This approach minimizes divergence: feedback is frequent and specific so the
    agent and the user are in sync
- **Cons**:
  - User has constant context switching between review and dispatch
  - Sometimes user gets stuck waiting for the agent to finish

## Strategy 2: Precompiled Task List

- This approach is still interactive, but the idea is to remove spec-writing time
  from the loop, leaving only: run -> review -> repeat with faster feedback and lower
  cognitive load per cycle

```mermaid
graph LR
    subgraph spec["📋 Spec Phase"]
        S["Write complete spec list"]
    end
    
    subgraph loop1["🔄 Loop 1"]
        L1a["👤 You assign Task 1"]
        L1b["🤖 Agent runs"]
        L1c["👤 You review"]
        L1a --> L1b --> L1c
    end
    
    subgraph loop2["🔄 Loop 2"]
        L2a["👤 You assign Task 2"]
        L2b["🤖 Agent runs"]
        L2c["👤 You review"]
        L2a --> L2b --> L2c
    end
    
    subgraph loop3["🔄 Loop 3"]
        L3a["👤 You assign Task 3"]
        L3b["🤖 Agent runs"]
        L3c["👤 You review"]
        L3a --> L3b --> L3c
    end
    
    spec --> loop1 --> loop2 --> loop3
```

- A variation of the precompiled task list is to let the agent go ahead through the
  task list and commit multiple times
- **Cons**:
  - Reviewing the work becomes difficult since in Git / GitHub the unit of work is
    typically a PR as a group of commits and not a list of commits

## Strategy 3: Stacked PRs

- Stacked PRs reduce interruptions by letting the agent work longer:
  - Write all task specifications once, upfront
  - Agent completes multiple tasks in sequence
  - Agent creates multiple PRs automatically (one per task or logical change)
  - You review all PRs together when the agent finishes
  - All stacked PRs are merged together in sequence
- **Pros**:
  - This creates a single uninterrupted work block for the user and the agent
- **Cons**
  - _Divergence risk grows_: longer task sequences without feedback increase the
    chance the agent misunderstands requirements (or it was poorly specified)
  - _Compound complexity_: later tasks depend on earlier ones, so review becomes
    interconnected
  - _Expensive corrections_: fixing mistakes mid-sequence requires rebasing all
    downstream changes (aka the problem of "stacked PRs")

```mermaid
graph LR
    subgraph spec["📋 Spec Phase"]
        S["Write complete spec list"]
    end
    
    subgraph run["🤖 Run Phase"]
        R1["Agent Task 1"]
        R2["Agent Task 2"]
        R3["Agent Task 3"]
        R4["All PRs created"]
        R1 --> R2 --> R3 --> R4
    end
    
    subgraph review["👤 Review Phase"]
        V1["Review all PRs"]
        V2["Rebase if needed"]
        V1 --> V2
    end
    
    spec --> run --> review
```

## When to Use Each Strategy

- **Use interactive workflows** when:
  - Tasks are exploratory or uncertain
  - Feedback shapes future work
  - Costs of divergence are high
- **Use stacked PRs** when:
  - Tasks are well-defined and independent
  - Agent can work with clear, complete specifications
  - You prefer focused review sessions over frequent interruptions
  - Tasks fit a logical sequence (e.g., modular features or refactoring stages)

## Implementing Stacked PRs

- Both examples below implement the same feature split into three sequential tasks:
  - **Task 1**: Add the database schema
  - **Task 2**: Add the API endpoint (depends on Task 1)
  - **Task 3**: Add the UI component (depends on Task 2)

### GitHub Stacked PRs

- GitHub started offering native stacked PR support in summer 2026, allowing multiple
  PRs to stack on a single branch with automatic dependency tracking
  - Each PR links to the previous one, and merging happens in order
- **Creating the stack**: branch each task off the previous one and open a PR against
  that parent branch:

  - Create first PR `Step 1`
    ```bash
    # Create the branch.
    > git checkout -b feature/step-1-schema main
    # ... agent adds database schema ...
    > git add -A && git commit -m "Step 1: add database schema"
    > git push -u origin feature/step-1-schema
    # ... agent works ...

    # Create review on GitHub.
    > gh pr create --base main --head feature/step-1-schema \
        --title "Step 1: add database schema"
    ```

  - Create second PR `Step 2`
    ```bash
    > git checkout -b feature/step-2-api feature/step-1-schema
    # ... agent adds API endpoint ...
    > git add -A && git commit -m "Step 2: add API endpoint"
    > git push -u origin feature/step-2-api
    > gh pr create --base feature/step-1-schema --head feature/step-2-api \
        --title "Step 2: add API endpoint"
    ```

  - Create last PR `Step 3`
    ```bash
    > git checkout -b feature/step-3-ui feature/step-2-api
    # ... agent adds UI component ...
    > git add -A && git commit -m "Step 3: add UI component"
    > git push -u origin feature/step-3-ui
    > gh pr create --base feature/step-2-api --head feature/step-3-ui \
        --title "Step 3: add UI component"
    ```

- GitHub renders the three PRs as a linked stack. When `feature/step-1-schema` merges
  into `main`, GitHub automatically retargets Step 2's PR base to `main`
- **Updating an earlier PR**: if review feedback lands on Step 1, every downstream
  branch needs a manual rebase:

  ```bash
  > git checkout feature/step-1-schema
  # ... apply fix ...
  > git add -A && git commit -m "Fix: address review comment"
  > git push

  > git checkout feature/step-2-api
  > git rebase feature/step-1-schema
  > git push --force-with-lease

  > git checkout feature/step-3-ui
  > git rebase feature/step-2-api
  > git push --force-with-lease
  ```

- Refer to
  [GitHub's stacked PRs documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-stacked-pull-requests)
  for setup and workflow details

### GitHub Stacked PRs + Helpers

- The helpers framework provides several CLI tools to streamline the stacked PR
  workflow:

#### Creating Branches with Issues

- Use `git_create_issue_and_branch.py` to create a GitHub issue and corresponding
  branch simultaneously:

  ```bash
  # Create a new GitHub issue and branch for a feature
  > git_create_issue_and_branch.py \
      --gh_issue_title "Step 1: Add database schema" \
      --create_worktree

  # Or use an existing issue ID to create a branch
  > git_create_issue_and_branch.py \
      --gh_issue_id 123
  ```

- This script:
  - Creates the GitHub issue (or uses an existing one with `--gh_issue_id`)
  - Creates a git branch and PR named after the issue ID (e.g.,
    `IssueId123_Description`)
  - Optionally creates a git worktree for parallel work
  - Handles symmetric branch creation in submodules if present

#### Creating Branches Without Issues

- Use the `invoke git_branch_create` task to create a branch with a simpler workflow:

  ```bash
  # Create a branch directly
  > invoke git_branch_create --name "feature/step-1-schema"

  # Create a branch from an existing GitHub issue
  > invoke git_branch_create --issue-id 123

  # Optionally skip PR creation
  > invoke git_branch_create --name "feature/step-1-schema" --no-create-pr
  ```

#### Syncing with Master

- After creating your branch stack, use `invoke git_merge_master` to bring in changes
  from master without frequent context switches:

  ```bash
  # Merge master into the current branch
  > invoke git_merge_master

  # Skip automatic push (resolve conflicts manually first)
  > invoke git_merge_master --no-auto-merge
  ```

- This follows the merge-based approach (not rebase) to minimize conflict resolution
  when your branch has accumulated many small commits. See
  [Merge, Rebase, or Squash: Choosing How to Catch Up with Master](/blog/how-to-i-git-merge-master)
  for a deeper explanation of the merge strategy

#### Comparing Branches

- Use `git_branch_diff` to see differences between your branch and a target
- The `--target` options:
  - `base`: diff against your branch point (most useful for seeing what you changed)
  - `master`: diff against `origin/master`
  - `head`: diff only modified files
  - `hash`: diff against a specific commit hash with `--hash-value`
- E.g.,

  ```bash
  # See what changed between your branch and master
  > invoke git_branch_diff --target master

  # See files changed since branching point
  > invoke git_branch_diff --target base

  # Only print file names (useful for scripts)
  > invoke git_branch_diff --target master --only-print-files

  # Filter to specific file types
  > invoke git_branch_diff --target master --file-types py,md
  ```

## Key Takeaways

- **Balance runtime against interruption**: longer agent runs save context switches
  but increase divergence risk
- **Predefine all task specifications**: this is essential for both strategies
- **Stacked PRs work best for predictable sequences**: clear scope reduces surprises
- **Use tooling to manage complexity**: GitHub stacked PRs or `git-spice` handle the
  mechanical details
When in doubt, start with interactive workflows. Use stacked PRs once task sequences
are stable and well-understood
