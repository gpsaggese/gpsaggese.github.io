---
draft: true
title: "How to Use Claude Code"
authors:
  - gpsaggese
date: 2026-
description:
categories:
  - Causal AI
---

TL;DR: 

<!-- more -->

<!-- https://code.claude.com/docs/en -->

# Getting started

## Core concepts

### How CC works

- The agentic loop:
  - Gather context
  - Take action
  - Verify results

- It is powered by models (to reason) and tools (to act)
- CC is the harness around Claude (the model)
  - Provide tools
  - Manage context
  - Execution environment

- Multiple models `/model` or `claude --model <name>`

- Tools allow to act
  - E.g., read code, edit files, run commands, search the web
  - Return information feeding back into the loop

- Extensions of the agentic loop
  - Skills
  - MCP (connect to external services)
  - Hooks (automate workflows)
  - Subagents (offload / delegate tasks)

- Work with sessions
  - CC saves your conversation locally
  - Before CC makes changes, it takes a snapshot of files
  - Each session starts with a fresh context window
  - Each CC conversation is a session tied to the current dir

- Context window
  - Stores conversation history, file contents, skills, ...
  - You can use `/context` to see what's using space
  - When context reaches the limit. it's compacted, cleaned up
  - Skills are loaded on demand
  - Subagents are separated from main conversation

- Checkpoints
  - Every file edit is reversible
  - You can press `Esc` twice to rewind to a previous state

- Permission mode
  - `Shift + Tab` to cycle through permission modes
  - Asks before edits
  - Auto-accept edits
  - Plan mode (read-only tools)

- Ask CC for help
  - "How do I set up hooks?" ...
  - `/init` walks through creating a CLAUDE.md
  - `/agents` configures subagents
  - `/doctor`

- If CC is going to wrong path, just type your correction
  - CC will stop and adjust the approach

- CC performs best when it can check its own work
  - E.g., include test cases, paste expected UI, define the output you want

- Explore before implementing
  - Separate research from coding, using plan mode
    ```
    Read ... and understand. Then create a plan for ...
    ```
  - Then let CC implement

- Delegate, don't dictate
  - Think of CC as a capable colleague
  - Give context and direction and trust CC to figure out the details

### Extend CC

- Extensions for agentic loop
  - `CLAUDE.md` to add context for every sessions
  - Skills: on-demand / reusable knowledge and invocable workflows
  - MCP: connects to external services
  - Subagents: run in isolated context, returning summaries
  - Agent teams: coordinate independent sessions with shared tasks and
    peer-to-peer messaging
  - Hooks: run scripts outside the loop
  - Plugins and packages

- Skill
  - Markdown file containing knowledge and workflows
  - You can invoke skills with slash command or CC can load them when relevant
  - You can run skills in current conversation or via subagents

- Skill vs Subagents
  - Skills are reusable content you can load into any context
  - Subagents are isolated workers (e.g., parallel work and specialized workers)

- Skill vs `CLAUDE.md`
  - `CLAUDE.md` is for every session ("always do X" rule)
    - E.g., coding conventions, project structure, "never do X" rules
    - It should be less than 500 lines
  - Skills can be invoked for `/name`
  - Both can include paths with @path imports
    ```
    @path ./rules.md
    @path ./examples.md
    ```

- Subagent vs Agent team
  - Subagents run inside the same session and report back to main context
  - Agent teams are independent CC that communicate with each other
  - Use subagent when need a focused worked
  - Use agent team when agents need to coordinate independently

- MCP vs Skill
  - MCP connects to external services
  - Skills extend what CC knows

- You can nest `CLAUDE.md` files in subdirs

- Skill + MCP: MCP provides the connection, a skill teaches CC how to use it
- Skill + Subagent: A skill can spawn subagents for parallel work
- `CLAUDE.md` + Skills: CLAUDE.md is for always-on rules, skills are for
  on-demand knowledge
- Hook + MCP: a hook triggers external actions (e.g., "send a Slack notification"
  when CC modifies critical files)

- Features consume CC's context
  - Too much can fill context window and add noise

### Common workflows

- Quick codebase overview
  - Navigate to project root
  - "Give me an overview of this codebase"
  - "Explain the main architecture patterns"
  - "What are the key data models?"
  - ...

- Find relevant code
  - "Find the files that handle user authentication"
  - "How do these authentication files work together?"

- Fix bugs efficiently

- Refactor code
  - Do refactoring in small / testable increments

- Use specialized subagents
  - "Use the code-reviewer subagent ..."
  - "Have the debugger subagent investigate ..."

- Use Plan Mode for safe code analysis
  - E.g., `claude --permission-mode plan`

- Work with tests

- Create pull requests
  - Create a PR
  - E.g., `/commit-push-pr` skill

- Handle documentation

- Work with images
  - Drag-and-drop image into CC window
  - Copy-paste image with `CTRL+v` (not `CMD+v`)

- Reference files and directories
  - E.g., "What's the structure of @src/...?"

- Reference MCP resources
  - E.g., "Show me the data from @github:repos/owner/`

- Use extended thinking
  - The reasoning is visible in verbose mode, toggled with `CTRL+O`
  - Phrases like "think", "think hard", "ultrathink"

- Extended thinking controls how much internal reasoning CC perform before
  responding

- Resume conversations
  - `/rename` and `/resume`

- Git worktrees
  - When running on multiple tasks at once, you need to have different copies of
    the codebase
  - E.g., work on a feature in one worktree and fix a bug in another, without
    sessions interfering
  ```
  > git worktree add ../.. -b feature-XYZ
  > git worktree add ../.. bugfix-XYZ
  > cd ../.. && claude
  > git worktree list
  > git worktree remove ../...
  ```

- Subagents can use worktree to work in parallel
  - E.g., use `isolation: worktree` to the agent front matter

- Get notified when CC needs your attention
  - Use hooks in `.claude/settings.local.json`
  - Hooks communicate through stdin, stdout, stderr and exit codes

- Use Claude as a unix-style utility
  ```
  > claude -p "..." --output-format ...
  ```

### Best practices

- CC's context window fills up fast and then performance degrades (e.g.,
  forgetting earlier instructions)
  - Context window holds your entire conversation, every message, every file,
    every command output

- CC performs much better when it can verify its work
  - It needs a clear success criteria

- Explore first, then plan, then code
  - Enter plan mode to read files and answers questions without making changes
    - This is important when you are uncertain about the approach
    - If the task is simple (fix a typo, rename a variable) skip planning
  - Ask to create a detailed implementation plan (press `CTRL+G` in your text
    editor to edit directly)
  - Implement switch to normal mode and let CC code and verify against its plan
  - Commit

- Provide specific context to avoid corrections
  - Scope the task (e.g., specific file, testing preferences)
  - Point to sources
  - Reference existing patterns in your code base
  - Describe the symptom and what "fixed" looks like

- `CLAUDE.md`
  - Treat it like code: review it, prune it, test changes to see whether CC's
    behavior shifts
  - Use `@path/...` to import additional files

- CLI tools are the most context-efficient way to interact with external services
  - E.g., `gh`

- Create custom subagents
  - Stored in `.claude/agents`
  - Set of allowed tools
  - Invoke with "Use a subagent to review this code for security issues"

- Let CC interview you
  ```
  I want to build [brief description]. Interview me in detail using the AskUserQuestion tool.
  Ask about technical implementation, edge cases, concerns, and tradeoffs.
  Don't ask obvious questions, dig into the hard parts I might not have considered.
  Keep interviewing until we've covered everything, then write a complete spec to SPEC.md.
  ```

- Course-correct early and often
  - `ESC`: stop CC mid-action, context is preserved
  - `ESC + ESC`: rewind and restore code state
  - `/clear` to start fresh
  - `/rewind` to a checkpoint

- Manage context
  - `/clear`
  - `/compact <instructions>`

- Use subagents to investigate
  - When CC researches a codebase and reads lots of files, which consume context
  - A subagent explores the codebase and reports back with findings
  - E.g., "use a subagent to review this code for edge cases"

- Writer / Reviewer pattern
  - Writer: Implement XYZ
  - Reviewer: Review the implementation and look for edge cases, ...
  - Writer: Address the issues from the review ...

- Write code / Unit tests

- After two failed corrections, `/clear` and write a better initial prompt
  using what you have learned

# Build with CC

- CC include built-in subagents
  - Explore
    - Fast
    - Agent optimized for searching and analyzing codebases
    - Read-only tools
  - Plan
    - Gather context before presenting a plan
    - Read-only tools
  - General-purpose
    - Agent for complex, multi-step tasks, code modifications
    - All tools

- You can create your own subagent with
  - Custom prompts
  - Tool restrictions
  - Hooks
  - Skills

## Create custom subagents

- Subagents are specialized AI assistants that handle specific types of tasks
  - They have
    - Custom context window
    - Custom system prompt
    - Specific tool access

- When CC finds a task that matches a subagent description it delegates to
  subagent
  - Subagent works independently and returns results

- Subagents are defined in Markdown files with YAML frontmatter
  - Store in `.claude/agents`
    - Specific of a code base
    - User subagents for all projects
    - Specify name, description, tools, model, prompt
  - `/agents` command
  - You can pre-load skills into subagent at startup
  - You can enable memory to persist information across conversations (e.g.,
    build up knowledge over time)
  - Can run in foreground (blocking) or background (concurrent)

- Common patterns for subagents
  - Isolate operations that produce large amounts of output
    - E.g., fetching doc, processing logs
    - The work is self-contained and can return a summary
  - Run parallel research
  - Multi-step workflows to use subagents in sequence
    ```
    Use the code-reviewer subagent to find performance issues, then use the
    optimizer subagent to fix them
    ```

### Example subagents

- Subagents should excel at one specific task
  - Task descriptions should be clear

- Code reviewer
- Debugger
- Data scientist
- Database query validator

## Run agent teams

- Agent teams are disable by default

- Agent teams let you coordinate multiple CC instances working together
  - One session is the team lead
    - Coordinate work
    - Assign tasks
    - Synthesize results
  - Teammates
    - work independently
    - Communicate with each other

### When to use agent teams

- Use when parallel exploration adds real value
  - Research and review
  - New independent features
  - Debugging with competing hypotheses

- Subagent don't talk to each other but only to the main agent
- In agent teams, teammates share a task list and communicate with each other

## Create plugins

- Plugins allow to extend CC with skills, agents, hooks, and MCP servers

- Use plugins when you want to
  - share functionality with your team or community
  - use the same skills/agents across multiple projects
  - version control and update 
  - distribute through a marketplace

- Plugins use namespaced skills like `/plugin-name:hello`

- You can point CC to a plugin with `--plugin-dir`
  - Need a plugin manifest `.claude-plugin-name/plugin.json`
  - Then the various dirs `commands/`, `agents/`, `skills`

## Prebuilt plugins

- You can find and install plugins from marketplaces to install CC with new
  commands, agents, and capabilities

- Add the marketplace and browse catalog for plugins
- Install plugins

- External integrations
  - Github
  - Asana
  - Slack

## Extend CC with skills

- Commands and skills have been merged
  - `.claude/commands/review.md` and a skill at `.claude/skills/review/SKILL.md`
    both create `/review`

- `/simplify` reviews recently changed files for code, reuse, quality
  - It spawns 3 agents (code reuse, code quality, efficiency) and then applies
    fixes
  - E.g., `/simplify focus on memory efficiency`

- `/batch <instruction>` orchestrate large-scale changes in parallel
  - Decompose the work into 5 to 30 units
  - Present a plan for approval
  - Spawns agents in isolated `git worktree`
  - E.g., `/batch migrate src/ from Solid to React`

- Skills can be organized in nested `.claude/skills` directories
  - Useful in monorepos where each "package" can have their own skills

- Skills are like
  ```
  my-skill
    - SKILL.md
    - template.md   
    - reference.md  (detailed API docs, loaded when needed)
    - examples
      - sample.md   (usage examples, loaded when needed)
    - scripts/
      - helper.py   (utility scripts)
  ```
- Description can refer to more files
  ```
  - For complete API details, see [reference.md](reference.md)
  - For usage examples, see [examples.md](examples.md)
  ```

- Fields
  - `name`: name (if different from directory name)
  - `description`
  - `argument-hint`: hint shown during autocomplete
  - `disable-model-invocation`: prevent CC from triggering automatically
    - E.g., `/commit`, `send-slack-message`
  - `model`: model to use

- String substitution
  - `$ARGUMENTS`: all arguments when invoking the skill
  - `$N$, `$ARGUMENTS[n]`: n-th argument
  ```
  Fix GitHub issue $ARGUMENTS following our coding standards.
  ```
  - E.g., `/fix-issue 123`

- The `!command` runs shell commands before the skill content is sent to CC
  - E.g., to pull the content in `pr-summary`
    ```
    ## Pull request context
    - PR diff: !`gh pr diff`
    - PR comments: !`gh pr view --comments`
    - Changed files: !`gh pr diff --name-only`
    ```

## Output styles

- Control CC's system prompt
  - `Default`: normal behavior for software engineering
  - `Explanatory`: provide educational insights
  - `Learning`: collaborative, CC will add `TODO(human)` to implement

- `/output-style [style]`

- You can create new styles

## Automate with hooks

- Run shell commands automatically when CC
  - edits files
  - finishes tasks
  - need input
  - format code
  - send notification
  - enforce project rules

- Hooks are user-defined commands that execute at specific points
  - Provide deterministic control

- For decisions that require judgement, use prompt-based hooks

- Examples
  - Get notified when CC needs input
  - Autoformat code after edits
  - Audit config changes

- `Notification` event fires when CC is waiting for input or permission

- Hooks events fire at specific lifecycle points in CC
  - When an event fires, all matching hooks run in parallel
  - `Session Start`
  - `Notification`: when CC sends a notification
  - `Stop`: when CC finishes responding

- Hooks have a type
  - `prompt`: single turn LLM eval (i.e., Prompt-based hooks)
  - `agent`: multi-turn LLM eval (i.e., Agent-based hooks)

- Hooks communicate with CC through stdin, stdout, stderr, exit codes
  - Inputs are passed as JSON to `stdin`
  - Hooks can return a structure JSON output and exit code

- Hook matchers let specify which occurrence of an event
  - E.g., `Notification` has values `permission_prompt`, `idle_prompt`,
    `elicitation_dialog`

- Prompt based hooks
  - Use `type: "prompt"` and `model`
  - The return value should be `"ok": true / false`

## Programmatic usage

- Agent SDK and CLI (aka headless mode)
  ```bash
  > claude -p "..."
  ```
- Use `--output-format` for `text`, `json`
  - It is possible to pass `--json-schema`
  - Use `--allowedTools` to use certain tools without prompting
    - E.g., `Bash(git diff *)`, `Read,Edit`

## Model Context Protocol (MCP)

- MCP is an open source standard for AI-tool integrations

- E.g.,
  - Implement features from issue trackers
  - Analyze monitoring data
  - Query DBs
  - Automate workflows

- E.g.,
  - Notion
  - Slack
  - Asana
  - Zapier

- Use a remote HTTP server
- Use a local stdio server

## Troubleshooting

# Deployment

# Administration

# Configuration

# Reference

# Resources

- Official documentation for Claude Code: https://code.claude.com/docs/en

<!--

Add information about .claude/skills/

.claude/statusline.sh

> ls /Users/saggese/src/umd_classes1/helpers_root/dev_scripts_helpers/ai/
cc        ccc       ccp       instr     README.md se        sl
-->
