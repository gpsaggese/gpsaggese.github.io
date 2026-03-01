# Project Description

- AI coding agents = software tools powered by large language models (LLMs) that assist or
  autonomously perform software engineering tasks such as editing files, running tests, and
  fixing bugs
- They span a spectrum of autonomy levels, from single-line autocomplete (L0) to fully
  autonomous agents that plan, code, test, and ship pull requests end-to-end (L4)
- Key capabilities include multi-file awareness, git integration, terminal execution, and
  context-aware code reasoning across large codebases
- Tools range from lightweight CLI assistants (`Aider`) to full autonomous dev environments
  (`OpenDevin`) and self-hosted open-source model ecosystems (`Devstral`)
- They differ in architecture, cost, required infrastructure, and depth of integration with
  existing development workflows
- Evaluating these agents rigorously requires structured benchmarks, reproducible
  experimental setups, and ML-based analysis of performance patterns

# Comparison of Coding Agents

| Type | Name | Description | Website | Strength |
|---|---|---|---|---|
| Open-source | OpenDevin | Autonomous software engineer that plans tasks, writes code, runs tests, and fixes errors across repos | https://github.com/OpenDevin/OpenDevin | End-to-end development automation |
| Open-source | Aider | Terminal pair-programming agent that edits multiple files and integrates with git | https://github.com/Aider-AI/aider | Lightweight CLI productivity |
| Open-source | Devstral (Mistral) | Open coding model ecosystem used to power autonomous coding agents | https://mistral.ai | Strong local/self-hosted coding models |
| Commercial | Claude Code | Reasoning-focused coding agent for debugging and architecture tasks | https://www.anthropic.com/claude | Deep reasoning on large codebases |
| Open-source | Pi Coding Agent | Minimal extensible terminal coding harness with tools (read, write, edit, bash) and plugin skills/extensions | https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent | Highly customizable workflows |

## Autonomy Levels

> What percentage of the task can the agent complete without you touching the keyboard?

| Level | Capability | Example behaviors |
|------|------|------|
| L0 -- Autocomplete | Suggests next line | Tab completion |
| L1 -- Assistant | Writes functions when asked | "Write a parser" |
| L2 -- Repo-aware | Edits multiple files | Refactor feature |
| L3 -- Task agent | Implements ticket | Fix bug from issue |
| L4 -- Autonomous dev | Plans + codes + tests | Ships PR end-to-end |

# Project Blueprint

## Project Objective

Design and run an automated benchmarking pipeline that evaluates open-source AI coding agents
on a curated set of Python data science coding challenges. Use the collected evaluation data
to train an ML model that predicts agent success based on task complexity features, and use
the autonomy level framework to qualitatively compare agents across task categories.

## Tasks

- **Agent setup and evaluation**: Install and configure two open-source agents (e.g., `Aider`
  with a local Ollama model such as `codellama` or `mistral` for zero-cost operation); run
  each agent on the task subset; record pass@1 (whether the generated code passes the
  reference test cases), runtime, and number of agent turns needed

- **Autonomy level mapping**: Qualitatively assign each agent to an autonomy level (L0-L4)
  based on observed behavior during the evaluation; document specific task types where each
  agent escalates or fails to escalate to higher autonomy

- **Report**: Summarize which task characteristics most strongly predict agent success,
  compare agents on the autonomy scale, and recommend which agent is best suited for which
  class of data science task
