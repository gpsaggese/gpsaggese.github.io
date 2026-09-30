# Comparison of Coding Agents

## Status
- **Status:**: in_progress
- **Complete Specs:**: 0-100%

## Core Idea

- AI coding agents are software tools powered by LLMs that assist or
  autonomously perform software engineering tasks: editing files, running
  tests, fixing bugs
- They span a spectrum of autonomy levels, from single-line autocomplete (L0)
  to fully autonomous agents that plan, code, test, and ship pull requests
  end-to-end (L4)
- Key capabilities include multi-file awareness, git integration, terminal
  execution, and context-aware code reasoning across large codebases
- Tools range from lightweight CLI assistants (`Aider`) to full autonomous
  dev environments (`OpenDevin`) and self-hosted open-source model
  ecosystems (`Devstral`)
- They differ in architecture, cost, required infrastructure, and depth of
  integration with existing development workflows
- Evaluating these agents rigorously requires structured benchmarks,
  reproducible experimental setups, and ML-based analysis of performance
  patterns
- Proposal: design and run an automated benchmarking pipeline that evaluates
  open-source AI coding agents on a curated set of Python data science coding
  challenges
  - Use the collected evaluation data to train an ML model that predicts
    agent success from task complexity features
  - Use the autonomy level framework to qualitatively compare agents across
    task categories

## Formalization

- Autonomy levels: what percentage of the task the agent completes without a
  human touching the keyboard

  | Level               | Capability                   | Example behaviors  |
  | :------------------ | :---------------------------- | :------------------ |
  | L0: Autocomplete     | Suggests next line             | Tab completion       |
  | L1: Assistant        | Writes functions when asked    | "Write a parser"     |
  | L2: Repo-aware       | Edits multiple files           | Refactor feature     |
  | L3: Task agent       | Implements ticket              | Fix bug from issue   |
  | L4: Autonomous dev   | Plans, codes, tests            | Ships PR end-to-end  |

## Key Examples

- Representative coding agents surveyed:

  | Type        | Name | Description | Strength |
  | :---------- | :--- | :----------- | :-------- |
  | Open-source | [OpenDevin](https://github.com/OpenDevin/OpenDevin) | Autonomous software engineer that plans tasks, writes code, runs tests, and fixes errors across repos | End-to-end development automation |
  | Open-source | [Aider](https://github.com/Aider-AI/aider) | Terminal pair-programming agent that edits multiple files and integrates with git | Lightweight CLI productivity |
  | Open-source | [Devstral (Mistral)](https://mistral.ai) | Open coding model ecosystem used to power autonomous coding agents | Strong local/self-hosted coding models |
  | Commercial  | [Claude Code](https://www.anthropic.com/claude) | Reasoning-focused coding agent for debugging and architecture tasks | Deep reasoning on large codebases |
  | Open-source | [Pi Coding Agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent) | Minimal extensible terminal coding harness with tools (read, write, edit, bash) and plugin skills/extensions | Highly customizable workflows |

## Questions

1. [Open question 1: what remains unknown?]
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- [Topic 1]: [What to investigate]
- [Topic 2]: [What to investigate]
- [Topic 3]: [What to investigate]

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: agent setup and evaluation
  - Install and configure two open-source agents (e.g., `Aider` with a local
    Ollama model such as `codellama` or `mistral`, for zero-cost operation)
  - Run each agent on the task subset
  - Record pass@1 (whether the generated code passes the reference test
    cases), runtime, and number of agent turns needed
  - This is the result: a benchmark dataset of agent performance on the
    curated Python data science task subset

- Milestone 2: autonomy level mapping
  - Qualitatively assign each agent to an autonomy level (L0-L4) based on
    observed behavior during the evaluation
  - Document specific task types where each agent escalates, or fails to
    escalate, to a higher autonomy level
  - This is the result: an autonomy-level profile for each evaluated agent

- Milestone 3: report
  - Summarize which task characteristics most strongly predict agent success
  - Compare agents on the autonomy scale
  - Recommend which agent is best suited for which class of data science
    task
  - This is the result: a written comparison and recommendation report

## References
- Author(s), _Title_. (Year)
