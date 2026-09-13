# Summary

## Title
- Software Engineering with AI: Practical Workflows for AI-Assisted Development
- Building Software with AI: From Code Generation to Autonomous Agents

## Target Audience
- Advanced undergraduate/graduate CS students and working software engineers
  who already know git, testing, and CI/CD basics
- Assumes working familiarity with LLMs (prompting, context windows) but no
  prior AI/ML coursework

## Approach of the Book
- Focus on:
  - Practical workflows over deep ML theory
  - Concrete tool walkthroughs (Claude Code, Copilot, Cursor, CI integrations)
  - Worked examples: real prompts, real diffs, real PR/CI workflows
  - Making AI-assisted development operational in a real codebase
- Provide resources to go one level deep
  - Related classes (`msml610`, `data605`)
  - References to benchmarks (SWE-bench), papers, and vendor docs

## Short TOC
- The sequence of the parts in the book is:
  - Foundations of AI-Assisted Coding
    - 01, LLM-Assisted Code Generation
    - 02, AI Pair Programming Workflows
    - 03, Prompt Engineering for Code
    - 04, Retrieval-Augmented Generation for Codebases
  - AI Across the Development Lifecycle
    - 05, AI Code Review
    - 06, Test Generation and Coverage with AI
    - 07, Multi-Agent Systems for Software Tasks
    - 08, AI-Driven Refactoring and Legacy Code Migration
    - 09, Debugging with AI
  - Building, Evaluating, and Deploying AI Coding Systems
    - 10, Fine-Tuning and Adapting Models for Codebases
    - 11, Evaluation of AI Coding Tools
    - 12, AI in CI/CD
  - Trust, Collaboration, and Governance
    - 13, Trust, Verification, and Hallucination Risks
    - 14, Human-AI Collaboration Patterns
    - 15, Ethics, IP, and Security of AI-Generated Software

## All Lesson Materials
// No `generate_all_tocs.sh` exists yet for this book (see TODOs): list is
// hand-maintained until one is added

- `book_AI_Software_Engineering/lectures_notes/*.smd`

## Chapter Templates and Invariants
- Follow `.claude/skills/book.rules.md` for the Chapter Template (Goals,
  Topics, TODO, Slides, Lesson Materials, Notes) and Roadmap section
  conventions used throughout this file

# Roadmap

| Chap                                          | Slides                                                | Slides % | Criticize | Tutorial | Book |
| --------------------------------------------- | ------------------------------------------------------ | -------- | --------- | -------- | ---- |
|                                               |                                                        |          |           |          |      |
| **Foundations of AI-Assisted Coding**         |                                                        |          |           |          |      |
| 01. LLM-Assisted Code Generation              | Lesson01.1_LLM_Assisted_Code_Generation.txt           | 100%     |           |          |      |
| 02. AI Pair Programming Workflows             | Lesson02.1_AI_Pair_Programming_Workflows.txt          | 100%     |           |          |      |
| 03. Prompt Engineering for Code               | Lesson03.1_Prompt_Engineering_for_Code.txt            | 100%     |           |          |      |
| 04. Retrieval-Augmented Generation for Codebases | N/A                                                 |          |           |          |      |
| **AI Across the Development Lifecycle**       |                                                        |          |           |          |      |
| 05. AI Code Review                            | N/A                                                    |          |           |          |      |
| 06. Test Generation and Coverage with AI      | N/A                                                    |          |           |          |      |
| 07. Multi-Agent Systems for Software Tasks    | N/A                                                    |          |           |          |      |
| 08. AI-Driven Refactoring and Legacy Code Migration | N/A                                              |          |           |          |      |
| 09. Debugging with AI                         | N/A                                                    |          |           |          |      |
| **Building, Evaluating, and Deploying AI Coding Systems** |                                            |          |           |          |      |
| 10. Fine-Tuning and Adapting Models for Codebases | N/A                                                |          |           |          |      |
| 11. Evaluation of AI Coding Tools             | N/A                                                    |          |           |          |      |
| 12. AI in CI/CD                               | N/A                               | 5%       |           |          |      |
| **Trust, Collaboration, and Governance**      |                                                        |          |           |          |      |
| 13. Trust, Verification, and Hallucination Risks | N/A                                                  |          |           |          |      |
| 14. Human-AI Collaboration Patterns           | Lesson14.1_Human_AI_Collaboration_Patterns.txt        | 100%     |           |          |      |
| 15. Ethics, IP, and Security of AI-Generated Software | N/A                                             |          |           |          |      |

# Detailed TOC

# Part I: Foundations of AI-Assisted Coding

## 01: LLM-Assisted Code Generation

### Goals
- Explain how LLMs generate code from natural-language specs and context
- Show how context window size and retrieval shape generation quality
- Identify structural limits: hallucinated APIs, stale data, scale limits

### Topics
- How LLMs Generate Code
  - Autoregressive token prediction over code and docs
  - Training data: public repos, documentation, synthetic code
- Context and Prompting Patterns
  - System prompts, few-shot examples, instructions vs. completions
  - Context window size and its effect on multi-file generation
- Limits of Code Generation
  - Hallucinated APIs and outdated library versions
  - Token/context budget limits on large codebases
  - When generation degrades: long files, ambiguous specs

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson01.1_LLM_Assisted_Code_Generation.txt`

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic

### Books
- 2024, Alammar et al., "Hands-On Large Language Models: Language
  Understanding and Generation"
  (https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/):
  transformer architecture, autoregressive token prediction, and prompting mechanics
  underlying code generation
- 2024, Taulli, "AI-Assisted Programming: Better Planning, Coding, Testing,
  and Deployment"
  (https://www.oreilly.com/library/view/ai-assisted-programming/9781098164553/):
  planning-to-deployment workflows across ChatGPT, Claude, Copilot, and Cursor;
  modular prompting methodology for AI-generated code

### Papers
- 2026, Ashik et al., "When LLMs Lag Behind: Knowledge Conflicts from
  Evolving APIs in Code Generation" (https://arxiv.org/abs/2604.09515) —
  models default to memorized, pre-cutoff API patterns even when the prompt
  supplies current API info
- 2025, Huynh et al., "Large Language Models for Code Generation: A
  Comprehensive Survey of Challenges, Techniques, Evaluation, and
  Applications" (https://arxiv.org/abs/2503.01245): survey of
  architectures, training data, and evaluation for code-generating LLMs
- 2025, Lee et al., "Hallucination by Code Generation LLMs: Taxonomy,
  Benchmarks, Mitigation, and Challenges"
  (https://arxiv.org/abs/2504.20799): taxonomy of hallucinated APIs, logic
  errors, and outdated library usage in generated code
- 2024, Liu et al., "Exploring and Evaluating Hallucinations in
  LLM-Powered Code Generation" (https://arxiv.org/abs/2404.00971) —
  empirical categorization of hallucination types across mainstream code
  LLMs
- 2024, Tian et al., "CodeHalu: Investigating Code Hallucinations in LLMs
  via Execution-based Verification" (https://arxiv.org/abs/2405.00253) —
  execution-based benchmark for detecting hallucinated code
- 2024, Lozhkov et al., "StarCoder 2 and The Stack v2: The Next Generation"
  (https://arxiv.org/abs/2402.19173): training data and architecture
  behind an open code-generation model family
- 2023, Rozière et al., "Code Llama: Open Foundation Models for Code"
  (https://arxiv.org/abs/2308.12950): training data (85% code) and
  fine-tuning behind a widely used open code model
- 2023, Liu et al., "Lost in the Middle: How Language Models Use Long
  Contexts" (https://arxiv.org/abs/2307.03172): context window position
  effects on retrieval accuracy, directly relevant to multi-file generation
- 2022, Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models" (https://arxiv.org/abs/2201.11903): foundational
  prompting pattern for step-by-step code generation
- 2021, Chen et al., "Evaluating Large Language Models Trained on Code"
  (https://arxiv.org/abs/2107.03374): the Codex paper; introduces
  HumanEval and underlies GitHub Copilot
- 2021, Austin et al., "Program Synthesis with Large Language Models"
  (https://arxiv.org/abs/2108.07732): early large-scale study of LLM code
  synthesis from natural-language specs

### Internet resources
- Anthropic, "Prompt engineering overview"
  (https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview):
  official guidance on system prompts, few-shot examples, and context structuring
- GitHub Blog, "Inside GitHub: Working with the LLMs behind GitHub Copilot"
  (https://github.blog/ai-and-ml/github-copilot/inside-github-working-with-the-llms-behind-github-copilot/):
  practitioner account of prompt construction and context selection in a production
  code assistant
- OpenAI, "OpenAI Codex" (https://openai.com/index/openai-codex/) —
  background on the model that powered early Copilot-style code generation

## 02: AI Pair Programming Workflows

### Goals
- Compare pair-programming tools: Claude Code, Copilot, Cursor, and their models
- Map tool choice to workflow: CLI, IDE, desktop, web, mobile
- Show how integrations (CI, chat, MCP) extend pair programming past the editor

### Topics
- Pair Programming Tool Landscape
  - Claude Code, GitHub Copilot, Cursor: interaction models compared
  - Autocomplete vs. agentic (multi-step, tool-using) assistants
- Where to Run the Assistant
  - CLI, Desktop, VS Code, JetBrains, Web, Mobile
  - Trade-offs: terminal scripting vs. visual review vs. long-running tasks
- Extending the Workflow
  - GitHub/GitLab CI integration, automated PR review
  - Chat integrations (Slack), browser automation, MCP servers
- Working Asynchronously
  - Dispatch, remote control, scheduled tasks, background sessions

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson02.1_AI_Pair_Programming_Workflows.txt`

### Lesson Materials
- `book_AI_Software_Engineering/lectures_notes/claude_code.md`
  - [35%]: Where to run Claude Code (CLI/Desktop/VS Code/JetBrains/Web/
    Mobile), tool integrations (Chrome, GitHub Actions/GitLab CI, Slack,
    MCP), working away from the terminal (Dispatch, Remote Control,
    Channels, Scheduled tasks)
- `book_AI_Software_Engineering/lectures_notes/Lesson01-AI_workflows_for_coding.txt`
  - [10%]: 4-bullet stub outline only (Managing Code, Automating PR,
    Automating Documentation, Coding while you sleep): headers, no content
- Not covered
  - [55%]: Head-to-head comparison of Claude Code vs. Copilot vs. Cursor;
    autocomplete-vs-agentic distinction; no material at all on Copilot or
    Cursor specifically

### Books
- 2025, Laster, "Learning GitHub Copilot: Multiplying Your Coding
  Productivity Using AI"
  (https://www.oreilly.com/library/view/learning-github-copilot/9781098164645/):
  autocomplete vs. chat vs. agent modes, IDE/CLI setup, and workflow
  integration for a mainstream pair-programming tool
- 2002, Williams et al., "Pair Programming Illuminated"
  (https://dl.acm.org/doi/10.5555/548833): foundational practices of human
  pair programming that AI pair-programming tools now mirror and extend

### Papers
- 2026, Peralta et al., "Why Are Agentic Pull Requests Merged or Rejected?
  An Empirical Study" (https://arxiv.org/abs/2605.22534): outcomes of
  agent-authored PRs across Copilot, Devin, Codex, and Cursor in CI/review
  workflows
- 2026, Murphy-Hill et al., "Adoption and Impact of Command-Line AI Coding
  Agents: A Study of Microsoft's Early 2026 Rollout of Claude Code and
  GitHub Copilot CLI" (https://arxiv.org/abs/2607.01418): large-scale
  rollout data on adoption drivers and PR-throughput impact of CLI agents
- 2026, Chen et al., "Understanding How Enterprises Adopt the Model Context
  Protocol for LLM-Driven Software Engineering"
  (https://arxiv.org/abs/2606.09182): practitioner interviews on barriers
  to extending assistants with MCP-connected tools
- 2026, Geng et al., "Effective Strategies for Asynchronous Software
  Engineering Agents" (https://arxiv.org/abs/2603.21489): centralized
  delegation and isolated workspaces for background/asynchronous agent work
- 2026, Agarwal et al., "AI IDEs or Autonomous Agents? Measuring the Impact
  of Coding Agents on Software Development"
  (https://arxiv.org/abs/2601.13597): empirical comparison of IDE-embedded
  assistants vs. autonomous coding agents on code quality and productivity
- 2025, Hou et al., "Model Context Protocol (MCP): Landscape, Security
  Threats, and Future Research Directions"
  (https://arxiv.org/abs/2503.23278): protocol lifecycle and architecture
  behind MCP-based tool integrations
- 2025, Sapkota et al., "Vibe Coding vs. Agentic Coding: Fundamentals and
  Practical Implications of Agentic AI" (https://arxiv.org/abs/2505.19443)
 : contrasts conversational/autocomplete-style use with autonomous
  agentic workflows
- 2025, Wang et al., "AI Agentic Programming: A Survey of Techniques,
  Challenges, and Opportunities" (https://arxiv.org/abs/2508.11126) —
  taxonomy of agent architectures, planning, and tool integration
- 2022, Nguyen et al., "An Empirical Evaluation of GitHub Copilot's Code
  Suggestions" (https://sarahnadi.org/assets/pdf/pubs/NguyenMSR22.pdf) —
  correctness/understandability baseline for autocomplete-style suggestions
- 2000, Williams et al., "Strengthening the Case for Pair Programming"
  (http://sunnyday.mit.edu/16.355/williams.pdf): foundational empirical
  case for pair programming that AI pair-programming tools now extend

### Internet resources
- Cursor, "Cursor Docs" (https://cursor.com/docs): official docs for
  Agent mode, Rules, MCP, Skills, and CLI
- GitHub, "GitHub Copilot Documentation" (https://docs.github.com/copilot): official
  docs for autocomplete, chat, and agent modes across IDEs
- Model Context Protocol, "Specification"
  (https://modelcontextprotocol.io/specification/2025-11-25): official
  protocol spec for connecting assistants to external tools and data
- Anthropic, "Introducing the Model Context Protocol"
  (https://www.anthropic.com/news/model-context-protocol): announcement
  and rationale for the MCP standard
- Anthropic, "Where to run Claude Code"
  (https://code.claude.com/docs/en/platforms): official platform guide
  (CLI, Desktop, VS Code, JetBrains, Web, Mobile) underlying this chapter's
  local material

### Notes
- The two source files are stubs (56 lines total), not a full deck: treat
  the coverage numbers above as thin

## 03: Prompt Engineering for Code

### Goals
- Write specs an AI can implement correctly on the first attempt
- Use few-shot examples and chain-of-thought to steer code generation
- Structure prompts for multi-step and multi-file coding tasks

### Topics
- Writing Specs for AI
  - Precise requirements, constraints, and acceptance criteria
  - Examples of ambiguous vs. unambiguous specs
- Few-Shot and Pattern-Based Prompting
  - Providing in-context examples of desired code style
  - Style/convention transfer from an existing codebase
- Reasoning-Oriented Prompting
  - Chain-of-thought and step-by-step decomposition
  - Plan-then-implement prompting patterns
- Prompting for Multi-File Tasks
  - Decomposing large tasks into scoped, reviewable steps
  - Iterative refinement loops with the model

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson03.1_Prompt_Engineering_for_Code.txt`

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic

### Papers
- 2024, Schulhoff et al., "The Prompt Report: A Systematic Survey of
  Prompting Techniques" (https://arxiv.org/abs/2406.06608): taxonomy of
  prompting techniques, used to frame prompts as reusable, evaluable
  artifacts
- 2024, Ridnik et al., "Code Generation with AlphaCodium: From Prompt
  Engineering to Flow Engineering" (https://arxiv.org/abs/2401.08500) —
  test-anchored iterative flow that outperforms single-prompt generation
- 2023, Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World
  GitHub Issues?" (https://arxiv.org/abs/2310.06770): evidence that
  repository-level, multi-file tasks are far harder than function-level ones
- 2023, Li et al., "Structured Chain-of-Thought Prompting for Code
  Generation" (https://arxiv.org/abs/2305.06599): intermediate reasoning
  expressed as program structure beats prose reasoning for code
- 2023, Liu et al., "Is Your Code Generated by ChatGPT Really Correct?
  Rigorous Evaluation of Large Language Models for Code Generation"
  (https://arxiv.org/abs/2305.01210): weak test suites overstate
  correctness of generated code
- 2023, Chen et al., "Teaching Large Language Models to Self-Debug"
  (https://arxiv.org/abs/2304.05128): repair driven by execution feedback
  rather than unaided self-critique
- 2023, Madaan et al., "Self-Refine: Iterative Refinement with
  Self-Feedback" (https://arxiv.org/abs/2303.17651): iterative
  self-critique loop and its limits without an external signal
- 2023, Jiang et al., "Self-planning Code Generation with Large Language
  Models" (https://arxiv.org/abs/2303.06689): explicit planning phase
  before implementation
- 2023, White et al., "A Prompt Pattern Catalog to Enhance Prompt
  Engineering with ChatGPT" (https://arxiv.org/abs/2302.11382): named,
  reusable prompt structures usable as team assets
- 2022, Min et al., "Rethinking the Role of Demonstrations: What Makes
  In-Context Learning Work?" (https://arxiv.org/abs/2202.12837): format and
  output space of demonstrations drive most of the few-shot gain
- 2022, Wang et al., "Self-Consistency Improves Chain of Thought Reasoning
  in Language Models" (https://arxiv.org/abs/2203.11171): sampling several
  reasoning paths and voting, adapted to voting by test execution for code
- 2022, Kojima et al., "Large Language Models are Zero-Shot Reasoners"
  (https://arxiv.org/abs/2205.11916): instruction-only elicitation of
  step-by-step reasoning
- 2022, Zhou et al., "Least-to-Most Prompting Enables Complex Reasoning in
  Large Language Models" (https://arxiv.org/abs/2205.10625): ordered
  subproblem decomposition with answers fed forward
- 2022, Shrivastava et al., "Repository-Level Prompt Generation for Large
  Language Models of Code" (https://arxiv.org/abs/2206.12839): prompts
  assembled from repository context beat single-file prompts
- 2022, Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models" (https://arxiv.org/abs/2201.11903): foundational
  reasoning-before-answer prompting pattern
- 2021, Zhao et al., "Calibrate Before Use: Improving Few-Shot Performance
  of Language Models" (https://arxiv.org/abs/2102.09690): sensitivity of
  few-shot results to example choice and ordering
- 2021, Reynolds et al., "Prompt Programming for Large Language Models:
  Beyond the Few-Shot Paradigm" (https://arxiv.org/abs/2102.07350): prompts
  treated as programs rather than ad-hoc text
- 2021, Chen et al., "Evaluating Large Language Models Trained on Code"
  (https://arxiv.org/abs/2107.03374): correctness judged by executing unit
  tests, the model for executable acceptance criteria
- 2020, Brown et al., "Language Models are Few-Shot Learners"
  (https://arxiv.org/abs/2005.14165): in-context learning from a handful of
  examples

### Internet resources
- Anthropic, "Prompt engineering overview"
  (https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)
 : official guidance on structuring system prompts, context, instructions,
  and examples
- Anthropic, "Claude Code Best Practices"
  (https://www.anthropic.com/engineering/claude-code-best-practices) —
  practitioner guidance on specs, scoped steps, and reusable prompt assets
  in a real repository

## 04: Retrieval-Augmented Generation for Codebases

### Goals
- Explain how RAG grounds code generation in a repo's actual code and docs
- Compare retrieval strategies: embeddings, keyword search, symbol indexes
- Identify failure modes: stale indexes, irrelevant retrieval, retrieval gaps

### Topics
- Why RAG for Code
  - Limits of parametric knowledge for large/private codebases
  - Grounding generation in current repo state
- Retrieval Strategies
  - Embedding-based semantic search over code and docs
  - Keyword/AST/symbol-based retrieval; hybrid approaches
- Building a Code RAG Pipeline
  - Chunking strategies for code (function, file, module level)
  - Indexing, freshness, and incremental updates
- Failure Modes
  - Stale indexes after refactors
  - Irrelevant or noisy retrieval degrading generation quality

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic

# Part II: AI Across the Development Lifecycle

## 05: AI Code Review

### Goals
- Use AI to find correctness bugs, security issues, and style violations
- Distinguish automated linting from semantic AI-driven review
- Design review workflows that combine AI and human reviewers

### Topics
- What AI Code Review Catches
  - Correctness bugs, edge cases, logic errors
  - Security vulnerabilities and unsafe patterns
  - Style/convention violations and simplification opportunities
- Review Techniques
  - Diff-based review vs. whole-file/whole-repo review
  - Multi-pass and multi-agent adversarial review
- Integrating AI into Review Workflows
  - Inline PR comments, automated first-pass review
  - Human-in-the-loop: what to trust, what to verify
- Limits of AI Review
  - False positives/negatives, missing project-specific context

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic

## 06: Test Generation and Coverage with AI

### Goals
- Generate unit, property-based, and mutation tests with AI assistance
- Use AI to close coverage gaps without inflating low-value tests
- Evaluate generated tests for correctness and maintainability

### Topics
- Generating Unit Tests
  - From function signature/docstring to test cases
  - Edge cases, boundary conditions, error paths
- Property-Based and Mutation Testing
  - AI-generated property invariants
  - Using mutation testing to validate test-suite strength
- Coverage-Driven Generation
  - Targeting uncovered branches/lines with AI
  - Avoiding low-value, brittle, or redundant tests
- Evaluating Generated Tests
  - Do tests actually assert behavior, or just execute code?
  - Maintainability and readability of AI-written tests

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic

## 07: Multi-Agent Systems for Software Tasks

### Goals
- Design multi-agent workflows: orchestration, subagents, and tool use
- Decide when to fan out work vs. run a single sequential agent
- Apply patterns: pipelines, parallel review, judge panels, adversarial checks

### Topics
- Agent Architectures
  - Single agent with tools vs. orchestrator + subagents
  - Tool use: file access, shell, search, external APIs
- Orchestration Patterns
  - Pipelines, parallel fan-out, barriers, loop-until-dry
  - Judge panels and adversarial verification
- Coordination and Context
  - Passing context between agents; avoiding context bloat
  - Isolation: worktrees, sandboxing for parallel edits
- When Multi-Agent Helps
  - Comprehensive coverage, independent verification, scale
  - Costs: token overhead, coordination complexity

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic

## 08: AI-Driven Refactoring and Legacy Code Migration

### Goals
- Use AI to refactor code while preserving behavior
- Plan large-scale migrations (language, framework, API) with AI assistance
- Verify refactors and migrations with tests and diffing, not just review

### Topics
- Refactoring with AI
  - Renaming, extracting functions, simplifying control flow
  - Preserving behavior: test-before/test-after discipline
- Legacy Code Migration
  - Language/framework version upgrades
  - API migrations across large codebases
- Migration Strategy
  - Incremental vs. big-bang migration
  - Isolating migration units for independent verification
- Verifying Correctness
  - Golden-file/regression testing before and after
  - Diff review at scale; sampling strategies for large migrations

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic

## 09: Debugging with AI

### Goals
- Use AI to triage logs and reproduce failures from partial information
- Apply root-cause analysis techniques with AI assistance
- Know when AI debugging helps vs. when it guesses

### Topics
- Log and Error Triage
  - Parsing stack traces, error messages, and CI failure logs
  - Distinguishing symptom from root cause
- Reproducing Failures
  - Constructing minimal repro cases with AI help
  - Flaky test diagnosis
- Root-Cause Analysis
  - Bisection, hypothesis generation, targeted instrumentation
  - Reasoning across multiple files/services with AI
- Limits and Pitfalls
  - AI guessing plausible but wrong root causes
  - When to fall back to manual debugging

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic

# Part III: Building, Evaluating, and Deploying AI Coding Systems

## 10: Fine-Tuning and Adapting Models for Codebases

### Goals
- Compare fine-tuning to prompting/RAG for domain-specific code
- Understand data requirements and risks of fine-tuning on code
- Evaluate when fine-tuning is worth the cost

### Topics
- Why Fine-Tune
  - Domain-specific conventions, internal APIs, proprietary patterns
  - Fine-tuning vs. prompting vs. RAG: trade-offs
- Fine-Tuning Approaches
  - Full fine-tuning, LoRA/adapters, instruction tuning
  - Data collection: curated examples, synthetic data, code history
- Risks and Costs
  - Overfitting to stale patterns, catastrophic forgetting
  - Compute cost, maintenance burden, retraining cadence
- Alternatives
  - System prompts, RAG, and tool use as lighter-weight substitutes

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic

## 11: Evaluation of AI Coding Tools

### Goals
- Use benchmarks like SWE-bench to compare AI coding tool capability
- Define metrics beyond pass rate: cost, latency, human effort saved
- Design human evaluation studies for coding assistants

### Topics
- Benchmarks
  - SWE-bench and similar real-world issue-resolution benchmarks
  - HumanEval-style synthetic benchmarks and their limits
- Metrics Beyond Accuracy
  - Cost per task, latency, token usage
  - Human review time saved, edit distance from final merged code
- Human Evaluation
  - User studies, preference comparisons, longitudinal adoption studies
- Benchmark Pitfalls
  - Overfitting to benchmarks, data contamination, narrow task coverage

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic

## 12: AI in CI/CD

### Goals
- Use AI to automate PR creation, review, and merge conflict resolution
- Integrate AI agents into GitHub/GitLab CI pipelines safely
- Generate release notes and changelogs from commit/PR history with AI

### Topics
- Automated PR Workflows
  - AI-generated PRs from issues or specs
  - Automated first-pass PR review and inline comments
- Merge Conflict Resolution
  - AI-assisted conflict resolution strategies
  - When to trust automated resolution vs. escalate to a human
- CI Pipeline Integration
  - GitHub Actions/GitLab CI agent integration
  - Gating: what runs automatically vs. requires approval
- Release Automation
  - AI-generated release notes and changelogs
  - Versioning and rollback safety nets

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `book_AI_Software_Engineering/lectures_notes/claude_code.md`
  - [15%]: GitHub Actions/GitLab CI/CD integration, automated PR review
    mentioned as connected tools: no depth on merge conflicts, release
    notes, or gating policy
- Not covered
  - [85%]: Merge conflict resolution, release-note/changelog generation,
    approval gating design

# Part IV: Trust, Collaboration, and Governance

## 13: Trust, Verification, and Hallucination Risks

### Goals
- Identify when generated code hallucinates APIs, facts, or behavior
- Apply verification strategies: tests, sandboxing, provenance tracking
- Calibrate trust in AI output based on task type and evidence

### Topics
- Hallucination in Code
  - Invented APIs, incorrect library behavior, fabricated results
  - Why hallucination happens: pattern completion vs. grounded fact
- Verification Strategies
  - Test execution as ground truth
  - Sandboxed execution, golden-file comparison
- Provenance and Evidence
  - Tracking what evidence backs a generated claim/change
  - Evidence-carrying changes: tests, logs, diffs as proof
- Calibrating Trust
  - Task types where AI is reliable vs. unreliable
  - Confidence signals and when to require human verification

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic

## 14: Human-AI Collaboration Patterns

### Goals
- Compare review-loop, pair, and autonomous collaboration modes
- Design handoff points between human and AI in a workflow
- Match collaboration mode to task risk and reversibility

### Topics
- Collaboration Modes
  - Pair (synchronous) vs. autonomous (asynchronous) agent work
  - Review-loop: propose, review, revise cycles
- Designing Handoffs
  - What to delegate fully vs. what requires a human decision
  - Confirmation points for hard-to-reverse or outward-facing actions
- Task-Risk Matching
  - Low-risk (formatting, tests) vs. high-risk (schema, security) tasks
  - Escalation paths when AI is uncertain
- Team Workflows
  - Multiple engineers working alongside AI agents
  - Shared conventions and context (CLAUDE.md-style instructions)

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson14.1_Human_AI_Collaboration_Patterns.txt`

### Books
- 2022, National Academies of Sciences, Engineering, and Medicine,
  "Human-AI Teaming: State-of-the-Art and Research Needs"
  (https://www.nap.edu/catalog/26355/human-ai-teaming-state-of-the-art-and-research-needs):
  free full text; task allocation, trust, and team design across human-AI teams
- 2018, Daugherty et al., "Human + Machine: Reimagining Work in the Age
  of AI" (https://books.google.com/books?id=wpY4DwAAQBAJ): hybrid
  human+AI roles and handoff design in organizational workflows

### Papers
- 2026, Kang, "Governed AI-Assisted Engineering: Graduated Human Oversight for
  Agentic Code Generation in Regulated Domains"
  (https://arxiv.org/abs/2606.22484): three-tier oversight model
  (human-in-the-loop / human-over-the-loop / automated-with-monitoring) keyed to
  regulatory impact and reversibility
- 2026, Shukla et al., "Hedwig: Dynamic Autonomy for Coding Agents Under Local
  Oversight" (https://arxiv.org/abs/2605.11495): adaptive autonomy level per
  task for coding agents
- 2026, Heilman et al., "GitHub Copilot and Developer Productivity: An
  Observational Dose-Response Analysis" (https://arxiv.org/abs/2606.00438) —
  large-scale observational evidence on AI-assisted team workflows
- 2026, Raees et al., "From Trust to Appropriate Reliance: Measurement Constructs
  in Human-AI Decision-Making" (https://arxiv.org/abs/2604.23896): distinguishes
  trust, reliance, and appropriate reliance for calibrating escalation decisions
- 2025, Li et al., "The Rise of AI Teammates in Software Engineering (SE 3.0):
  How Autonomous Coding Agents Are Reshaping Software Engineering"
  (https://arxiv.org/abs/2507.15003): pair vs. autonomous collaboration modes
  for coding agents
- 2025, Mayer et al., "Human-AI Collaboration: Trade-offs Between Performance and
  Preferences" (https://arxiv.org/abs/2503.00248): agent consideration of human
  actions vs. team performance
- 2025, He et al., "Fine-Grained Appropriate Reliance: Human-AI Collaboration
  with a Multi-Step Transparent Decision Workflow for Complex Task Decomposition"
  (https://arxiv.org/abs/2501.10909): review-loop-style decomposition of
  multi-step tasks for reliance
- 2024, Vats et al., "A Survey on Human-AI Collaboration with Large Foundation
  Models" (https://arxiv.org/abs/2403.04931): collaborative design principles
  and governance frameworks
- 2023, Mehrotra et al., "A Systematic Review on Fostering Appropriate Trust in
  Human-AI Interaction" (https://arxiv.org/abs/2311.06305): practices and
  measures for calibrating trust, tied to task type
- 2023, Peng et al., "The Impact of AI on Developer Productivity: Evidence from
  GitHub Copilot" (https://arxiv.org/abs/2302.06590): controlled experiment
  quantifying pair-programming speedup
- 1999, Horvitz, "Principles of Mixed-Initiative User Interfaces"
  (https://doi.org/10.1145/302979.303030): foundational principles for when to
  hand off initiative between human and system

## 15: Ethics, IP, and Security of AI-Generated Software

### Goals
- Identify licensing and IP risks in AI-generated code
- Recognize vulnerability-injection and supply-chain risks from AI tools
- Apply governance practices for responsible AI-assisted development

### Topics
- Licensing and IP
  - Training-data provenance and license-contamination risk
  - Attribution and copyright questions for generated code
- Security Risks
  - AI-introduced vulnerabilities: injection, unsafe defaults
  - Supply-chain risk: AI-suggested dependencies and packages
- Responsible Use
  - Authorization boundaries for AI-driven security testing
  - Data privacy: what code/data should not reach external models
- Governance
  - Organizational policy for AI tool use in production code
  - Auditability and accountability for AI-assisted changes

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- Not covered
  - [100%]: No existing lecture material found in this repo for this topic
