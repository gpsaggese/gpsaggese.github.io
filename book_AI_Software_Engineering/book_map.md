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
  - Practical workflows over theory
  - Concrete tool walkthroughs (Claude Code, Copilot, Cursor, CI integrations)
  - Worked examples: real prompts, real diffs, real PR/CI workflows
  - Making AI-assisted development operational in a real codebase
- Provide resources to go one level deep
  - Related classes (`msml610`, `data605`)
  - References to benchmarks (SWE-bench), papers, and vendor docs

## Short TOC
- Foundations of AI-Assisted Coding
  - 01, Coding Models
  - 02, Skills Organization and Management
  - 03, AI Harness
  - 04, AI Pair Programming Workflows
  - 05, Prompt Engineering for Code
  - 06, Human-AI Collaboration Patterns
- Writing and Reviewing Code
  - 07, Code Cosmetics
  - 08, Documenting Code
  - 09, Reviewing Pull Requests
  - 10, Reviewing Existing Code
  - 11, Debugging with AI
- Planning and Architecture
  - 12, Planning
  - 13, Software Architecture Guidelines
- Delivery and Automation
  - 14, GitHub Projects and Workflows
  - 15, Releasing and Docker
  - 16, Setting CI/CD

## All Lesson Materials
- `helpers_root/docs/` (tools, coding, planning, work_organization,
  documentation_meta, and dev_system guides)
- `helpers_root/papers/AIgentic_Development_System/part*.tex`
- `book_AI_Software_Engineering/resources.md`
- `book_AI_Software_Engineering/lectures_source/claude_code.md`
- `book_Agentic_AI/lectures_source/*.smd`
- `data605/lectures_source/*.smd`
- The book's own decks (`book_AI_Software_Engineering/lectures_source/*.smd`)
  are chapter `### Slides`, not Lesson Materials

## Chapter Templates and Invariants

- Follow `.claude/skills/book_map.rules.md` for the Chapter Template (Goals,
  Topics, TODO, Slides, Lesson Materials, Notes) and Roadmap section conventions
  used throughout this file

# Roadmap

| Chap                                    | Slides                                          | Slides % | Criticize | Tutorial | Book |
| --------------------------------------- | ----------------------------------------------- | -------- | --------- | -------- | ---- |
| **Part I: Foundations of AI-Assisted Coding** |                                           |          |           |          |      |
| 01: Coding Models                       | Lesson01.1_LLM_Assisted_Code_Generation.smd     | 100%     |           |          |      |
| 02: Skills Organization and Management  | Lesson05.1_Tools.smd                            |          |           |          |      |
| 03: AI Harness                          | Lesson02.1_AI_Pair_Programming_Workflows.smd    |          |           |          |      |
| 04: AI Pair Programming Workflows       | Lesson02.1_AI_Pair_Programming_Workflows.smd    | 100%     |           |          |      |
| 05: Prompt Engineering for Code         | Lesson03.1_Prompt_Engineering_for_Code.smd      | 100%     |           |          |      |
| 06: Human-AI Collaboration Patterns     | Lesson14.1_Human_AI_Collaboration_Patterns.smd  | 100%     |           |          |      |
| **Part II: Writing and Reviewing Code** |                                                 |          |           |          |      |
| 07: Code Cosmetics                      | Lesson05.1_Tools.smd                            |          |           |          |      |
| 08: Documenting Code                    | Lesson05.1_Tools.smd                            |          |           |          |      |
| 09: Reviewing Pull Requests             | Lesson05.1_Tools.smd                            |          |           |          |      |
| 10: Reviewing Existing Code             |                                                 |          |           |          |      |
| 11: Debugging with AI                   | Lesson05.1_Tools.smd                            |          |           |          |      |
| **Part III: Planning and Architecture** |                                                 |          |           |          |      |
| 12: Planning                            | Lesson05.1_Tools.smd                            |          |           |          |      |
| 13: Software Architecture Guidelines    | Lesson05.1_Tools.smd                            |          |           |          |      |
| **Part IV: Delivery and Automation**    |                                                 |          |           |          |      |
| 14: GitHub Projects and Workflows       |                                                 |          |           |          |      |
| 15: Releasing and Docker                | Lesson05.1_Tools.smd                            |          |           |          |      |
| 16: Setting CI/CD                       | Lesson05.1_Tools.smd                            |          |           |          |      |

## TODOs
- Split the shared deck `Lesson05.1_Tools.smd` into per-chapter decks, or keep
  it as the shared deck for chapters 02, 07-09, 11-13, 15, and 16
- Write a dedicated deck for chapters 03, 10, and 14
- Confirm that Skills is chapter 02 and that Human-AI Collaboration Patterns
  belongs in Part I

# Detailed TOC

# Part I: Foundations of AI-Assisted Coding

- Assumptions
  - Human attention is limited
  - Avoid spaghetti code
  - The craft of Software Development
- Part I explains how models write code and how humans steer them.

## 01: Coding Models

### Goals
- Describe how LLMs generate code and where they fail
- Compare coding models by benchmark, such as HumanEval and SWE-bench
- Recognize context-window and hallucination limits in generated code

### Topics
- How LLMs Generate Code
  - Autoregressive token prediction
  - Training data: public repos, documentation, and synthetic code
- Context and Prompting Patterns
  - System prompts, few-shot examples, chain-of-thought, plan-then-implement
  - Context window size and "lost in the middle" effects
- Limits of Code Generation
  - Types of hallucination and knowledge conflicts
  - Degradation on long files and ambiguous specifications
- Coding Models and Benchmarks
  - Code-specific training and model families
  - Benchmarks: HumanEval, SWE-bench, and their limits

### TODO
- [ ] Add a slide on SWE-bench results

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson01.1_LLM_Assisted_Code_Generation.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [30%]: tokenization, attention, context window, KV cache, decoding
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [15%]: chain-of-thought as serial computation, limits of self-correction
- _Not covered_
  - [55%]: code model families, code-specific training data, HumanEval and
    SWE-bench benchmarks, hallucination taxonomy for code

## 02: Skills Organization and Management

### Goals
- Organize agent skills, rules, and templates in a repo
- Write a skill from a workflow that the team repeats
- Keep skills in sync with the rules and files they reference

### Topics
- Definitions: Topics, Skills, Rules, and Templates
  - Why each kind of file has its own role
  - Pointers from CLAUDE.md to rules and templates
- Agent Instruction Manifests
  - CLAUDE.md and AGENTS.md as machine-readable manuals
  - Repo contracts such as `repo_config.yaml`
- Writing a Skill
  - Frontmatter, goal, inputs, workflow, and verification sections
  - Running a skill from a slash command or the shell
- Maintaining Skills
  - Updating a skill from lessons learned in Git history
  - Checking references between skills and rules
- Testing Skills
  - Running a skill on a sample file and reviewing the diff
  - Detecting regressions after rule changes

### TODO
- [ ] Add a deck section on testing skills

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson05.1_Tools.smd`
  (section `# Rules, Skills, and Agent Instruction Manifests`)

### Lesson Materials
- `helpers_root/docs/tools/ai/all.ai_workflows.how_to_guide.md`
  - [70%]: topics, skills, rules, templates, running a skill, adding rules
- `book_AI_Software_Engineering/resources.md`
  - [30%]: agent instruction manifests, CLAUDE.md as a manual, self-documenting
    workflows
- `helpers_root/docs/tools/ai/all.ai_review.how_to_guide.md`
  - [15%]: tools that apply rules to files, improving the tools while reviewing
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [10%]: tool calls and their failure modes
- _Not covered_
  - [20%]: testing agent skills, skill evaluation, sharing skills across repos

## 03: AI Harness

### Goals
- Describe the parts of a coding harness: model, tool loop, and permissions
- Compare Claude Code, GitHub Copilot, and Cursor by surface and integration
- Choose a harness by work style and project location

### Topics
- Harness Anatomy
  - Model, tool loop, context management, and permissions
  - Instructions and skills loaded at session start
- Where the Harness Runs
  - CLI, desktop, IDE, web, and mobile surfaces
  - Local sessions versus cloud sessions
- Integrations
  - Browser automation, CI/CD, PR review, and chat channels
  - MCP servers and connectors
- Comparing Harnesses
  - Claude Code, GitHub Copilot, and Cursor
  - Trade-offs: autonomy, control, cost, and lock-in
- Choosing a Harness for a Team
  - Permission policies and audit of actions
  - Moving work between harnesses

### TODO
- [ ] Compare the Copilot and Cursor internals, not only their surfaces

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson02.1_AI_Pair_Programming_Workflows.smd`
  (sections `## Pair Programming Tool Landscape` and `## Where to Run the Assistant`)

### Lesson Materials
- `book_AI_Software_Engineering/lectures_source/claude_code.md`
  - [70%]: Claude Code surfaces, integrations, and working away from a terminal
- `helpers_root/docs/tools/ai/all.create_pr_with_claude_code.how_to_guide.md`
  - [25%]: a Claude Code PR workflow with worktrees and small steps
- `helpers_root/docs/tools/ai/all.auto_task.how_to_guide.md`
  - [20%]: running Claude Code tasks on GitHub and choosing an execution mode
- `book_Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`
  - [15%]: ReAct, acting on the world, early agent failures
- _Not covered_
  - [40%]: Copilot and Cursor internals, a harness comparison matrix,
    permission models, cost comparison

## 04: AI Pair Programming Workflows

### Goals
- Choose autocomplete, chat, or agentic assistance for a given task
- Run a pair-programming loop with an agent on a real change
- Use CI, PR review, and asynchronous sessions within one working day

### Topics
- Three Interaction Models
  - Autocomplete, chat, and agentic assistance
  - Vibe coding versus agentic coding
- Pair-Programming Loop
  - Investigate, plan, make small edits, and test in layers
  - Keeping a human in the loop at each step
- Working Surfaces
  - CLI, desktop and IDE, web and mobile
- Extending the Workflow
  - CI integration, automated PR review, and MCP
- Asynchronous Work
  - Dispatch, remote control, background sessions, and isolated workspaces

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson02.1_AI_Pair_Programming_Workflows.smd`

### Lesson Materials
- `helpers_root/docs/tools/ai/all.create_pr_with_claude_code.how_to_guide.md`
  - [60%]: investigate before changing, worktrees, small steps, layered tests
- `helpers_root/docs/tools/ai/all.ai_workflows.how_to_guide.md`
  - [40%]: coding workflows such as implementing TODOs and cleaning up a file
- `helpers_root/docs/tools/ai/all.auto_task.how_to_guide.md`
  - [30%]: task lists, criticizing the plan, execution modes
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [10%]: iterative refinement with grounded feedback
- _Not covered_
  - [35%]: pair-programming evidence and metrics, dispatch and remote control,
    channels and scheduled tasks

## 05: Prompt Engineering for Code

### Goals
- Write a code spec with acceptance criteria and explicit constraints
- Choose few-shot, chain-of-thought, or decomposition prompts per task
- Treat prompts as versioned artifacts that the team reviews

### Topics
- Writing Specs for AI
  - Anatomy of a code spec, acceptance criteria, and non-goals
  - Ambiguous versus unambiguous requests
- Few-Shot and Pattern-Based Prompting
  - Selecting examples that transfer to the task
  - Negative examples and few-shot failure modes
- Reasoning-Oriented Prompting
  - Structured chain-of-thought, self-planning, and least-to-most decomposition
  - When reasoning prompts do not help
- Multi-File Tasks
  - Decomposing work into scoped steps
  - Refinement loops and restart versus refine
- Prompts as Artifacts
  - Versioning prompts and reusable templates
  - Review and reuse across the team

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson03.1_Prompt_Engineering_for_Code.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [45%]: chain-of-thought variants, self-consistency, tree of thought,
    premise order
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [30%]: LLMs as optimizers, prompt optimization against a grounded score
- `book_Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`
  - [25%]: in-context learning, from prompting to acting, ReAct traces
- `helpers_root/docs/tools/ai/all.ai_workflows.how_to_guide.md`
  - [15%]: templates and prompt-driven rule workflows
- _Not covered_
  - [40%]: acceptance-criteria specs, negative examples, test-first prompting,
    style transfer from a codebase

## 06: Human-AI Collaboration Patterns

### Goals
- Pick pair, review-loop, or autonomous collaboration for each task
- Design handoffs with explicit confirmation points
- Match the level of oversight to the risk of the task

### Topics
- Collaboration Modes
  - Pair, review-loop, and autonomous collaboration
  - Comparing the three modes
- Designing Handoffs
  - What to delegate fully and what needs a human decision
  - Confirmation points for hard-to-reverse actions
- Task-Risk Matching
  - Two axes of risk and a risk matrix for delegation
  - Three-tier oversight and escalation when the AI is uncertain
- Team Workflows
  - Instruction manifests and shared conventions
  - Evidence and provenance for team trust

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson14.1_Human_AI_Collaboration_Patterns.smd`

### Lesson Materials
- `helpers_root/docs/tools/ai/all.ai_review.how_to_guide.md`
  - [35%]: reviewer workflow and human review of AI transforms
- `helpers_root/docs/tools/ai/all.auto_task.how_to_guide.md`
  - [30%]: the human criticizes the plan before execution
- `helpers_root/papers/AIgentic_Development_System/part4.AI_optimized_development.tex`
  - [20%]: guardrails and provenance for autonomous changes
- `helpers_root/docs/work_organization/all.use_github.how_to_guide.md`
  - [15%]: PR review and addressing review comments
- _Not covered_
  - [40%]: task-risk matrix, adaptive autonomy, calibrating trust, team-level
    metrics

# Part II: Writing and Reviewing Code

- Part II covers the code that humans and AI write, check, and document.

## 07: Code Cosmetics

### Goals
- Make generated code follow the repo style guide
- Write fast unit tests and check output with golden files
- Enforce coverage thresholds in CI

### Topics
- Style and Naming
  - Naming conventions, spelling, and avoiding code stutter
  - Comments and docstrings
- Linting
  - Linter framework: modifying and non-modifying checks
  - Pre-commit hooks as a quality gate
- Imports and Types
  - No `import *`, full-path imports, and import cycle detection
  - Type hints and common mypy errors
- Unit Testing
  - Test framework and golden file testing
  - Test categories: fast, slow, and superslow
- Code Coverage
  - Coverage measurement and CI thresholds
  - Mutation testing and quarantine of flaky tests

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson05.1_Tools.smd`
  (sections `# Automating Code Quality` and `# Automating Unit Testing`)

### Lesson Materials
- `helpers_root/papers/AIgentic_Development_System/part1.code_quality.tex`
  - [85%]: linter framework, pre-commit, coverage, unit test framework, golden
    files
- `helpers_root/docs/coding/all.coding_style.how_to_guide.md`
  - [80%]: naming, comments, docstrings, imports, logging, assertions, tests
- `helpers_root/docs/coding/all.coding_style_guidelines.reference.md`
  - [60%]: per-language rules for naming, docstrings, imports, and tests
- `helpers_root/docs/tools/linter/all.developing_linter.how_to_guide.md`
  - [60%]: linter scripts, modifying and non-modifying steps
- `helpers_root/docs/coding/all.type_hints.how_to_guide.md`
  - [50%]: type hints, mypy workarounds, typing conventions
- `book_AI_Software_Engineering/resources.md`
  - [40%]: code quality, unit testing, and coverage (section outline)
- _Not covered_
  - [10%]: mutation testing, flaky test handling, security gates

## 08: Documenting Code

### Goals
- Write docstrings and comments that serve both humans and AI tools
- Keep architecture diagrams in sync with the code they describe
- Update docs in the same PR as the code change

### Topics
- Docstrings and Comments
  - Descriptive versus imperative style, and what a docstring contains
  - Avoiding distracting comments and stale TODOs
- Architecture Diagrams
  - C4 levels: context, container, component, and code
  - PlantUML conventions and class diagrams
- Docs as Machine-Consumable Artifacts
  - Self-documenting executable workflows
  - CLAUDE.md as the agent's manual
- Maintaining Docs
  - Keeping docs in sync with the code
  - Generating and updating READMEs with AI
- Project Docs
  - Project docs: PRD, design, plan, progress, and decisions

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson05.1_Tools.smd`
  (section `# Automating Documentation`)

### Lesson Materials
- `helpers_root/docs/documentation_meta/all.architecture_diagrams.explanation.md`
  - [40%]: C4 levels, PlantUML conventions, class diagrams
- `helpers_root/docs/coding/all.coding_style.how_to_guide.md`
  - [35%]: comments and docstring conventions, stale TODOs
- `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
  - [35%]: keeping docs in sync, organizing project docs
- `helpers_root/papers/AIgentic_Development_System/part4.AI_optimized_development.tex`
  - [20%]: self-documenting executable workflows, agent instruction manifests
- `book_AI_Software_Engineering/resources.md`
  - [20%]: CLAUDE.md as a machine-consumable manual
- _Not covered_
  - [35%]: docstring generation pipelines, doc-drift checks in CI

## 09: Reviewing Pull Requests

### Goals
- Review a PR with an AI reviewer and then with a human reviewer
- Compare AI code review tools by features, price, and fit
- Address review comments with small follow-up commits

### Topics
- Review Workflow
  - Filing and reviewing a PR, labels, and review states
  - Addressing a review comment
- AI Reviewer Tools
  - The `ai_review` tool and its workflows
  - GitHub Copilot review, Graphite, CodeRabbit, and Deepcode
- Reviewer Rules and Prompts
  - Checking a PR against the repo rules
  - Improving the reviewer over time
- Automated Review in CI
  - Review as a CI step and its failure modes
- Human Judgment
  - What a human must still check in an AI-reviewed PR

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson05.1_Tools.smd`
  (section `# Automating the Pull Request Lifecycle`, slides on PR review)

### Lesson Materials
- `helpers_root/docs/tools/ai/all.ai_review.how_to_guide.md`
  - [75%]: the `ai_review` tool, reviewer workflow, improving the reviewer
- `helpers_root/docs/tools/ai/all.code_review_tools.reference.md`
  - [60%]: Copilot, Graphite, CodeRabbit, and Deepcode: features and prices
- `helpers_root/docs/work_organization/all.use_github.how_to_guide.md`
  - [50%]: PR filing, review, and addressing comments
- `helpers_root/docs/tools/ai/all.create_pr_with_claude_code.how_to_guide.md`
  - [25%]: opening and following up on a PR
- `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
  - [20%]: from idea to merged PR, review and merge
- _Not covered_
  - [25%]: review as a CI gate, measured review quality, reviewing AI-written
    PRs at scale

## 10: Reviewing Existing Code

### Goals
- Build a mental model of an unfamiliar module in one session
- Map dependencies and call paths before editing
- Record findings as notes that a teammate can reuse

### Topics
- Reading a Module
  - Public functions, invariants, and the role of each file
  - Pure functions and design philosophy
- Dependency Analysis
  - Import graphs and cycle detection
  - Package structure and the anatomy of a package
- Diagramming Existing Code
  - C4 component and code views
  - Class diagrams generated from code
- Using AI to Explain Code
  - Finding the rules and skills that apply to a file
  - Running read-only skills before editing
- Recording Findings
  - Research logs and reproducible notes

### TODO
- [ ] Write a deck on reading unfamiliar code with AI

### Slides
- N/A: no dedicated deck yet

### Lesson Materials
- `helpers_root/docs/coding/all.code_design.how_to_guide.md`
  - [35%]: design philosophy, invariants, pure functions, and roles
- `helpers_root/docs/documentation_meta/all.architecture_diagrams.explanation.md`
  - [35%]: C4 component and code views, class diagrams
- `helpers_root/docs/tools/ai/all.ai_workflows.how_to_guide.md`
  - [30%]: finding rules for a file, running a skill on a file
- `helpers_root/docs/planning/all.document_research.how_to_guide.md`
  - [15%]: research logs and reproducible notes
- `helpers_root/papers/AIgentic_Development_System/part2.runnable_directory.tex`
  - [10%]: runnable directories as bounded units of code
- _Not covered_
  - [60%]: reading unfamiliar code with AI, tracing call graphs, onboarding
    walkthroughs

## 11: Debugging with AI

### Goals
- Reproduce a failure before asking the AI to fix it
- Use test failures and runtime output as grounded feedback
- Stop refinement loops that do not converge

### Topics
- Reproducing Failures
  - Minimal reproducers and golden outputs
  - Logging and assertions as debugging aids
- Feedback Loops
  - Unit tests as grounded feedback
  - Self-judged feedback and its limits
- Self-Debugging
  - Generate, run, explain, and retry
  - Designing the verifier before the loop
- Root-Cause Analysis
  - Analyzing a failed gate
  - Feeding production incidents back to the agent
- Limits of Self-Correction
  - When models fix their own mistakes
  - When retries do not help

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson05.1_Tools.smd`
  (slide `Automating Root-Cause Analysis When a Gate Fails`)

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [55%]: self-debugging, grounded versus self-judged feedback, verifier design
- `helpers_root/docs/coding/all.coding_style.how_to_guide.md`
  - [25%]: logging levels, assertions, and reporting warnings
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [20%]: limits of self-correction, when self-correction helps
- _Not covered_
  - [50%]: reproducing production failures, debugging workflows in a repo,
    trace inspection

# Part III: Planning and Architecture

- Part III covers how work is planned and how the code is structured.

## 12: Planning

### Goals
- Turn an idea into a plan with milestones and PRs
- Keep a progress log and a decisions log for each project
- Criticize a task list before the AI executes it

### Topics
- Planning Hierarchy
  - Ideas, plans, PRs, and tasks
  - Capturing an idea
- Plan Documents
  - PRD, design, plan, progress, and decisions docs
  - Naming and organizing project docs
- End-to-End Plans
  - Rules for writing the plan
  - Applying architecture rules during planning
- Requirements and Decisions
  - Automating requirements capture
  - Architecture decision records
- Planning With an AI
  - Task lists, criticizing the plan, and executing in a loop

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson05.1_Tools.smd`
  (section `# Requirements and Planning`)

### Lesson Materials
- `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
  - [85%]: planning hierarchy, plan docs, end-to-end plan, progress log
- `helpers_root/docs/planning/all.document_research.how_to_guide.md`
  - [45%]: research log, reproducible research, daily entries
- `helpers_root/docs/tools/ai/all.auto_task.how_to_guide.md`
  - [45%]: task lists, criticizing the plan, execution modes
- `helpers_root/docs/work_organization/all.github_projects_process.reference.md`
  - [40%]: sprint process, weekly summary, carryover of unfinished issues
- `book_Agentic_AI/lectures_source/Lesson05.1-Reasoning_Memory_and_Planning.smd`
  - [15%]: planning as a loop, world models for planning
- _Not covered_
  - [15%]: effort estimation, roadmapping across quarters, requirements
    interviews

## 13: Software Architecture Guidelines

### Goals
- Apply design rules that keep modules independent and cohesive
- Choose a monorepo or multiple repos with explicit trade-offs
- Draw architecture diagrams that match the code structure

### Topics
- Design Philosophy
  - Keep it simple and separate what changes from what stays the same
  - Pure functions and invariants by default
- Repo Structure
  - Single repo versus multiple repos
  - Runnable directories and the helpers submodule
- Package Structure
  - Package hierarchy and the anatomy of a package
  - Import cycle prevention
- Architecture Diagrams
  - C4 context, container, component, and code levels
  - Class diagrams generated from code
- Architecture Decisions
  - Architecture decision records
  - Modularizing code with AI

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson05.1_Tools.smd`
  (section `# Code Modularization and Repo Structure`)

### Lesson Materials
- `helpers_root/papers/AIgentic_Development_System/part2.runnable_directory.tex`
  - [80%]: monorepo versus multi-repo, runnable directory architecture
- `helpers_root/docs/documentation_meta/all.architecture_diagrams.explanation.md`
  - [80%]: C4 levels, PlantUML conventions, class diagrams
- `helpers_root/docs/coding/all.code_design.how_to_guide.md`
  - [70%]: design philosophy, architecture, invariants, and roles
- `book_AI_Software_Engineering/resources.md`
  - [40%]: package structure, import cycle detection, single versus multiple repo
- `helpers_root/docs/coding/all.coding_style.how_to_guide.md`
  - [30%]: architecture and design patterns, import conventions
- _Not covered_
  - [20%]: service boundaries, scalability trade-offs

# Part IV: Delivery and Automation

- Part IV covers how code moves from a GitHub project to a released container.

## 14: GitHub Projects and Workflows

### Goals
- Set up a GitHub project with a lean four-status board
- Name, file, and close issues with consistent conventions
- Keep a clean PR history with Git branching and rebasing

### Topics
- GitHub Project Structure
  - Issue fields and the lean four-status board
  - Sprints, cadence, and carryover
- Issue and PR Workflows
  - Naming and filing issues, labels, and statuses
  - Filing a PR, review, and addressing comments
- Automation
  - Built-in workflows and GitHub Actions automations
  - Syncing project templates
- Git Branching
  - Branches, merges, rebase, and the golden rule of rebasing
  - Centralized and forking workflows
- Parent and Sub-Issues
  - Splitting work across issues

### TODO
- [ ] Write a deck on branch protection and code ownership

### Slides
- N/A: no dedicated deck yet (PR lifecycle is covered in chapter 09)

### Lesson Materials
- `helpers_root/docs/work_organization/all.github_projects_process.reference.md`
  - [85%]: project structure, board, sprints, automation, sub-issues
- `helpers_root/docs/work_organization/all.use_github.how_to_guide.md`
  - [85%]: concepts, issue and PR workflows, labels
- `helpers_root/docs/tools/dev_system/all.sync_gh_projects.how_to_guide.md`
  - [60%]: syncing project templates, previewing and applying changes
- `data605/lectures_source/Lesson02.2-Git_Branching.smd`
  - [45%]: branching, merging, rebase, centralized and forking workflows
- `data605/lectures_source/Lesson02.1-Git.smd`
  - [30%]: version control basics, Git remotes, GitHub
- _Not covered_
  - [15%]: branch protection, code ownership

## 15: Releasing and Docker

### Goals
- Build dev and prod images through one documented release flow
- Ship executables that run the same way in any container
- Release a multi-architecture image through a checklist

### Topics
- Container Stages
  - Local, dev, and prod stages
  - Pre-built versus build-on-the-fly containers
- Dockerized Executables
  - Thin environment and wrapper scripts
  - Sibling versus child containers
- Image Release
  - Build, tag, push, and QA of dev images
  - Multi-architecture builds
- Automated Release
  - Build and test on CI, then a manual review, then the release
  - PR labeling and team-based assignment
- Release Communication
  - Roll-out process and announcement email
  - Post-release checks

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson05.1_Tools.smd`
  (sections `# Dockerization and Dockerized Executables` and the container
  release slides in `# Automating Build, Release, and CI/CD`)

### Lesson Materials
- `helpers_root/docs/tools/dev_system/all.devops_docker_auto_release.explanation.md`
  - [90%]: automated build and test, manual review, automated release
- `helpers_root/papers/AIgentic_Development_System/part3.containerized_development.tex`
  - [90%]: dockerized executables, container release flow, ECS, Airflow
- `helpers_root/docs/tools/dev_system/all.devops_docker.how_to_guide.md`
  - [85%]: stages, release flows, invoke targets, QA
- `helpers_root/docs/tools/dev_system/all.devops_docker.reference.md`
  - [75%]: devops layout, build and run files, OS package install
- `helpers_root/docs/work_organization/all.rollout.how_to_guide.md`
  - [50%]: roll-out process and announcement email
- `data605/lectures_source/Lesson03.2-Docker.smd`
  - [40%]: Dockerfile, building and running containers
- `data605/lectures_source/Lesson03.3-Docker_Compose.smd`
  - [25%]: multi-container applications and compose commands
- _Not covered_
  - [15%]: production monitoring and rollback

## 16: Setting CI/CD

### Goals
- Set up a CI pipeline that runs fast and slow test tiers
- Make a failing gate report its root cause
- Release containers only after the gates pass

### Topics
- Build Automation
  - Centralized task registry with invoke
  - Build health monitoring and the buildmeister role
- CI Pipeline
  - Fast, slow, and superslow test tiers
  - CI integration and workflow behavior
- Quality Gates
  - Layered gates: lint, types, tests, coverage
  - Enforced thresholds and branch checks
- Release Pipeline
  - Container build and release on CI
  - Preproduction and production releases
- Monitoring After Release
  - Canary analysis and automated rollback
  - Feeding production incidents back to the agent

### Slides
- `book_AI_Software_Engineering/lectures_source/Lesson05.1_Tools.smd`
  (sections `# Automating Build, Release, and CI/CD` and
  `# Production Feedback and Release Monitoring`)

### Lesson Materials
- `helpers_root/papers/AIgentic_Development_System/part1.code_quality.tex`
  - [60%]: CI/CD automation section, CI integration, enforced thresholds
- `book_AI_Software_Engineering/resources.md`
  - [50%]: CI/CD automation and the buildmeister role, invoke workflows
- `helpers_root/docs/tools/dev_system/all.devops_docker_auto_release.explanation.md`
  - [40%]: automated build and test, release workflow file
- `helpers_root/papers/AIgentic_Development_System/part3.containerized_development.tex`
  - [30%]: integration with CI/CD and quality gates, buildmeister dashboard
- `helpers_root/docs/work_organization/all.github_projects_process.reference.md`
  - [15%]: GitHub Actions automations
- _Not covered_
  - [35%]: canary analysis, automated rollback, production incident loops
