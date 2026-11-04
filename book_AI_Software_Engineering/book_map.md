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
- `.claude/skills/*/SKILL.md` and `.claude/skills/*.rules.md` (skills and rules)
- `.claude/templates/`, `.claude/rules.md`, and `.claude/agents/`
- `website/docs/blog/posts/*.md` (blog posts and drafts)
- `research/ideas/*.md` (research ideas)
- `research/agentic_sw_eng/resources.md` (index of AI development resources)
- `.github/workflows/*.yml` (CI workflows of this repo)
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
- `website/docs/blog/posts/how_to.Compare_LLM_models.md`
  - [40%]: LLM evaluation landscape, reliable benchmarks, comparison by category
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [30%]: tokenization, attention, context window, KV cache, decoding
- `website/docs/blog/posts/how_to.LLM_effort.md`
  - [20%]: reasoning effort, token billing, a code generation example
- `research/ideas/draft.Create_LLM_Benchmark_for_thoroughness.md`
  - [15%]: benchmark design for multi-site tasks
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [15%]: chain-of-thought as serial computation, limits of self-correction
- `website/docs/blog/posts/draft.LLM_issues.md`
  - [10%]: LLMs as interpolators, correction loops
- `research/ideas/draft.Skill_vs_Luck_in_Agent_Benchmarks.md`
  - [10%]: noise in agent benchmark scores
- _Not covered_
  - [45%]: code model families, code-specific training data, SWE-bench results,
    hallucination taxonomy for code

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
- `.claude/skills/skill.rules.md`
  - [50%]: SKILL.md format, frontmatter, naming, directory structure, rule files
- `website/docs/blog/posts/draft.how_to.Claude_skills.md`
  - [45%]: what skills are, progressive disclosure, a first skill, guidelines
- `research/agentic_sw_eng/resources.md`
  - [35%]: index of rules, skills, templates, and agent configuration
- `book_AI_Software_Engineering/resources.md`
  - [30%]: agent instruction manifests, CLAUDE.md as a manual, self-documenting
    workflows
- `.claude/rules.md`
  - [30%]: mapping from file type to rules and templates
- `.claude/skills/skill.auto_add/SKILL.md`
  - [25%]: update a skill from lessons learned in Git history
- `research/ideas/draft.Unit_Testing_LLM_Skills.md`
  - [25%]: unit tests for LLM skills
- `.claude/skills/skill.check_refs/SKILL.md`
  - [20%]: check the references between skills and rules
- `research/ideas/draft.Measuring_Quality_of_Skills_and_Prompts.md`
  - [20%]: quality metrics for skills
- `website/docs/blog/posts/draft.how_to.Enforce_rules_in_Claude_Code.md`
  - [20%]: hooks that force the agent to load the rules for a file type
- `.claude/skills/skill.consolidate_skills/SKILL.md`
  - [15%]: merge and rename skills by topic
- `.claude/skills/skill.reorganize_rules/SKILL.md`
  - [15%]: reorganize the rules in a rules file
- `.claude/skills/skill.check_against_rules/SKILL.md`
  - [15%]: check a file against the rules that apply to it
- `.claude/skills/rules.add/SKILL.md`
  - [15%]: add a rule to the set of rules
- `helpers_root/docs/tools/ai/all.ai_review.how_to_guide.md`
  - [15%]: tools that apply rules to files, improving the tools while reviewing
- `website/docs/blog/posts/draft.ideas.Deterministic_skills.md`
  - [10%]: skills that call deterministic code
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [10%]: tool calls and their failure modes
- _Not covered_
  - [15%]: skill evaluation in production, sharing skills across repos

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
- `website/docs/blog/posts/draft.how_to.Coding_Agents.md`
  - [70%]: Claude Code, GitHub Copilot, ChatGPT, Cursor, Devin
- `book_AI_Software_Engineering/lectures_source/claude_code.md`
  - [70%]: Claude Code surfaces, integrations, and working away from a terminal
- `website/docs/blog/posts/draft.pidev_vs_claude_code_comparison.md`
  - [55%]: pi.dev versus Claude Code: core concepts and workflows
- `research/ideas/in_progress.Comparison_of_Coding_Agents.md`
  - [50%]: comparison of coding agents
- `website/docs/blog/posts/draft.survey.tools_to_generate_sw_with_agents.md`
  - [40%]: orchestrators, managed platforms, spec-driven tools
- `website/docs/blog/posts/draft.in_30_mins.pi_dev.md`
  - [35%]: a second harness: setup, concepts, examples
- `website/docs/blog/posts/draft.how_to.Use_Claude_Code.md`
  - [35%]: Claude Code features and usage
- `website/docs/blog/posts/draft.how_to.Automate_Coding_with_LLM.md`
  - [30%]: tool overview, when to use each tool, cost tracking
- `helpers_root/docs/tools/ai/all.create_pr_with_claude_code.how_to_guide.md`
  - [25%]: a Claude Code PR workflow with worktrees and small steps
- `research/ideas/draft.Tutorial_claude_agent_sdk.md`
  - [25%]: building agents with the Claude Agent SDK
- `helpers_root/docs/tools/ai/all.auto_task.how_to_guide.md`
  - [20%]: running Claude Code tasks on GitHub and choosing an execution mode
- `website/docs/blog/posts/how_to.Use_Claude_Code_with_Openrouter.md`
  - [20%]: running the Claude Code harness with other models
- `research/ideas/draft.Document_MCP.md`
  - [20%]: Model Context Protocol
- `book_Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`
  - [15%]: ReAct, acting on the world, early agent failures
- `website/docs/blog/posts/draft.how_to.Parse_Claude_logs.md`
  - [15%]: record types in a harness session log
- `website/docs/blog/posts/draft.in_5_mins.helpers_cc.md`
  - [15%]: the `cc` wrapper that switches the model behind Claude Code
- `website/docs/blog/posts/how_to.Count_cost.md`
  - [15%]: cost tracking of agent sessions with `ccusage` and OpenRouter
- _Not covered_
  - [25%]: permission models, Copilot and Cursor internals, a harness test on
    the same task

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
- `website/docs/blog/posts/draft.how_to.Use_Claude_Code.md`
  - [55%]: create and edit code, run and test, debug, larger projects, prompting
    tips
- `website/docs/blog/posts/draft.how_to.My_agentic_engineering_flow.md`
  - [50%]: unit of work, spec before code, criticize first, keep the PR
    mergeable
- `website/docs/blog/posts/draft.how_to.Automate_Coding_with_LLM.md`
  - [40%]: workflow patterns and advanced features
- `helpers_root/docs/tools/ai/all.ai_workflows.how_to_guide.md`
  - [40%]: coding workflows such as implementing TODOs and cleaning up a file
- `.claude/skills/auto_task.execute_interactively/SKILL.md`
  - [35%]: execute, review, commit loop over a task list
- `website/docs/blog/posts/draft.how_to.A_queue_of_AI_coding_agents.md`
  - [30%]: a task queue that feeds agents asynchronously
- `helpers_root/docs/tools/ai/all.auto_task.how_to_guide.md`
  - [30%]: task lists, criticizing the plan, execution modes
- `website/docs/blog/posts/draft.how_to.How_to_run_Claude_on_GitHub.md`
  - [30%]: `@claude` on issues and PRs as asynchronous help
- `website/docs/blog/posts/draft.how_to.Use_Claude_Code_Workflows.md`
  - [25%]: creating a script from scratch with Claude Code
- `website/docs/blog/posts/draft.how_to.Apply_Coding_AI_to_ML_Data_Science.md`
  - [20%]: coding assistants in ML and data science work
- `.claude/skills/coding.todoai_gp/SKILL.md`
  - [20%]: implement the TODO items left in a file
- `.claude/skills/coding.annotate_with_todoai/SKILL.md`
  - [15%]: let the rules write TODOs that the agent implements
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [10%]: iterative refinement with grounded feedback
- _Not covered_
  - [25%]: pair-programming metrics and evidence, dispatch and remote control,
    scheduled tasks

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
  - [45%]: chain-of-thought variants, self-consistency, tree of thought, premise
    order
- `research/ideas/draft.Prompt_fine_tuning.md`
  - [40%]: automated input/output loop to tune a prompt
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [30%]: LLMs as optimizers, prompt optimization against a grounded score
- `.claude/templates/ai.instruction_template.md`
  - [30%]: patterns for writing an AI instruction file
- `research/ideas/draft.Measuring_Quality_of_Skills_and_Prompts.md`
  - [30%]: measuring the quality of a prompt
- `website/docs/blog/posts/draft.how_to.Compress_LLM_in_out_tokens.md`
  - [25%]: terse prompts and terse output to cut tokens
- `research/ideas/draft.How_to_measure_LLM_compliance.md`
  - [25%]: measuring instruction-following compliance
- `book_Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`
  - [25%]: in-context learning, from prompting to acting, ReAct traces
- `website/docs/blog/posts/draft.in_30_mins.helpers_hllm_decorator.md`
  - [20%]: prompts as decorated functions: classification and extraction
- `helpers_root/docs/tools/helpers/all.hllm.explanation.md`
  - [20%]: LLM calls with caching, cost, and testing
- `research/ideas/draft.Create_script_to_apply_prompt_in_chunks.md`
  - [15%]: apply one prompt to a file in chunks
- `website/docs/blog/posts/in_30_mins.helpers_llm_cli.md`
  - [15%]: prompting from the command line
- `website/docs/blog/posts/in_30_mins.simonw_llm_cli.md`
  - [15%]: the `llm` CLI: prompts, keys, and templates
- `helpers_root/docs/tools/ai/all.ai_workflows.how_to_guide.md`
  - [15%]: templates and prompt-driven rule workflows
- `website/docs/blog/posts/draft.LLM_issues.md`
  - [10%]: correction loops for imprecise output
- _Not covered_
  - [30%]: acceptance-criteria specs, negative examples, test-first prompting,
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
- `helpers_root/docs/work_organization/all.team_collaboration.how_to_guide.md`
  - [35%]: rules of collaboration: asking for help, problem solving
- `website/docs/blog/posts/My_AI_Policy.md`
  - [35%]: LLMs as team members, no AI slop
- `website/docs/blog/posts/draft.A_proposal_for_AI_based_software_engineering.md`
  - [35%]: principles and team setup for agentic AI development
- `website/docs/blog/posts/draft.how_to.My_agentic_engineering_flow.md`
  - [30%]: criticize first, agents do not commit on their own
- `helpers_root/docs/tools/ai/all.auto_task.how_to_guide.md`
  - [30%]: the human criticizes the plan before execution
- `helpers_root/docs/coding/all.submit_code_for_review.how_to_guide.md`
  - [25%]: edit code written by AI, test in layers
- `research/ideas/draft.How_to_measure_LLM_compliance.md`
  - [20%]: measuring how well an LLM follows instructions
- `.claude/skills/auto_task.criticize/SKILL.md`
  - [20%]: criticize a plan before execution
- `helpers_root/papers/AIgentic_Development_System/part4.AI_optimized_development.tex`
  - [20%]: guardrails and provenance for autonomous changes
- `helpers_root/docs/coding/all.code_review.how_to_guide.md`
  - [20%]: author and reviewer points of view
- `website/docs/blog/posts/draft.article.2026.Ribeiro_et_al.Why_Should_I_Trust_You.md`
  - [15%]: when to trust a model prediction
- `research/ideas/draft.Counterfactual_Attribution_of_Agent_Failures.md`
  - [15%]: attributing failures to agent steps
- `helpers_root/docs/work_organization/all.use_github.how_to_guide.md`
  - [15%]: PR review and addressing review comments
- `.claude/agents/code.reviewer.md`
  - [10%]: a code reviewer agent
- _Not covered_
  - [30%]: task-risk matrix, adaptive autonomy, team-level metrics

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
- `.claude/skills/coding.rules.md`
  - [75%]: Python rules for naming, docstrings, imports, and functions
- `.claude/skills/testing.rules.md`
  - [65%]: unit test rules
- `helpers_root/docs/coding/all.coding_style_guidelines.reference.md`
  - [60%]: per-language rules for naming, docstrings, imports, and tests
- `helpers_root/docs/tools/linter/all.developing_linter.how_to_guide.md`
  - [60%]: linter scripts, modifying and non-modifying steps
- `helpers_root/docs/tools/testing_profiling/all.codecov.how_to_guide.md`
  - [55%]: Codecov setup, coverage configuration, GitHub Actions workflow
- `website/docs/blog/posts/in_30_mins.Python_Code_Coverage.md`
  - [55%]: coverage reports and `pytest` options
- `website/docs/blog/posts/draft.in_15_mins.helpers_hunit_test.md`
  - [50%]: golden file tests and `--update_outcomes`
- `helpers_root/docs/coding/all.type_hints.how_to_guide.md`
  - [50%]: type hints, mypy workarounds, typing conventions
- `helpers_root/docs/coding/all.submit_code_for_review.how_to_guide.md`
  - [45%]: style guide, linter, example code, layered tests
- `helpers_root/docs/tools/testing_profiling/all.code_coverage_subprocess.how_to_guide.md`
  - [40%]: coverage for subprocesses and Docker
- `.claude/skills/testing.write_unit_tests/SKILL.md`
  - [40%]: plan and write unit tests for a file
- `book_AI_Software_Engineering/resources.md`
  - [40%]: code quality, unit testing, and coverage (section outline)
- `.claude/skills/testing.reach_coverage/SKILL.md`
  - [35%]: raise unit test coverage toward 100 percent
- `research/ideas/draft.Use_agents_for_unit_test_coverage.md`
  - [35%]: agents that raise unit test coverage
- `helpers_root/docs/tools/git/all.git_hooks.how_to_guide.md`
  - [35%]: install pre-commit hooks and skip the checks
- `helpers_root/docs/build_system/all.linter_gh_workflow.explanation.md`
  - [30%]: the linter as a GitHub workflow
- `helpers_root/docs/tools/codebase/all.imports_and_packages.how_to_guide.md`
  - [30%]: package goals, hierarchy, cycle prevention
- `research/ideas/draft.Comparison_of_Testing_Agents.md`
  - [30%]: comparison of AI testing agents
- `.claude/skills/coding.fix_type_hints/SKILL.md`
  - [30%]: fix type hints to follow the typing conventions
- `.claude/skills/coding.fix_pyright/SKILL.md`
  - [25%]: run pyright and fix the reported lints
- `.claude/skills/testing.fix_unit_tests/SKILL.md`
  - [25%]: apply the unit test conventions
- `website/docs/blog/posts/draft.how_to.Create_Hook_To_Run_Ruff_In_Claude_Code.md`
  - [25%]: a hook that runs Ruff after every edit
- `.claude/skills/coding.fix_docstring/SKILL.md`
  - [20%]: fix Python docstrings
- `research/ideas/draft.Codebase_autocleanup.md`
  - [20%]: TODO-driven codebase cleanup by agents
- `website/docs/blog/posts/draft.how_to.VS_Code_Quick_Fix.md`
  - [15%]: linter output as quickfix entries
- _Not covered_
  - [10%]: mutation testing, flaky test handling

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
- `helpers_root/docs/documentation_meta/all.writing_docs.how_to_guide.md`
  - [55%]: conventions for writing docs
- `helpers_root/docs/documentation_meta/all.diataxis.explanation.md`
  - [50%]: tutorial, how-to guide, reference, explanation
- `research/ideas/draft.Comparison_of_Documentation_Agents.md`
  - [45%]: comparison of AI code documentation agents
- `helpers_root/docs/documentation_meta/all.architecture_diagrams.explanation.md`
  - [40%]: C4 levels, PlantUML conventions, class diagrams
- `research/ideas/draft.Use_agents_to_update_repo_doc.md`
  - [40%]: agents that keep repo docs up to date
- `.claude/skills/readme_dir.rules.md`
  - [40%]: README for a directory with executables
- `helpers_root/docs/documentation_meta/all.where_to_add_material.explanation.md`
  - [35%]: where new material belongs in the docs
- `helpers_root/docs/documentation_meta/all.google_technical_writing.how_to_guide.md`
  - [35%]: technical writing rules
- `helpers_root/docs/coding/all.coding_style.how_to_guide.md`
  - [35%]: comments and docstring conventions, stale TODOs
- `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
  - [35%]: keeping docs in sync, organizing project docs
- `.claude/skills/docs.add_content/SKILL.md`
  - [35%]: add content to the docs without redundancy
- `.claude/skills/readme_file.rules.md`
  - [35%]: README for a single script
- `.claude/skills/coding.fix_docstring/SKILL.md`
  - [30%]: fix Python docstrings
- `website/docs/blog/posts/draft.Writing_Books_For_Humans_and_AI.md`
  - [30%]: writing technical content for humans and AI
- `.claude/skills/readme_file.explain_code/SKILL.md`
  - [25%]: document classes, functions, and relationships
- `helpers_root/docs/documentation_meta/all.how_to_guide_template.md`
  - [25%]: doc templates
- `website/docs/blog/posts/in_30_mins.mdm_unified_markdown_manager.md`
  - [25%]: managing scattered content with `mdm`
- `.claude/skills/coding.fix_comments/SKILL.md`
  - [25%]: update docstrings and comments without changing logic
- `.claude/skills/markdown.rules.md`
  - [25%]: markdown conventions
- `helpers_root/papers/AIgentic_Development_System/part4.AI_optimized_development.tex`
  - [20%]: self-documenting executable workflows, agent instruction manifests
- `book_AI_Software_Engineering/resources.md`
  - [20%]: CLAUDE.md as a machine-consumable manual
- `helpers_root/docs/blogging/all.write_blog.how_to_guide.md`
  - [20%]: blogs in the docs ecosystem
- `.claude/skills/C4.rules.md`
  - [20%]: C4 diagrams with PlantUML
- `website/docs/blog/posts/draft.how_to.Codebase_local_kg.md`
  - [10%]: a local knowledge graph of the codebase
- _Not covered_
  - [25%]: doc-drift checks in CI, measured doc quality

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
- `helpers_root/docs/coding/all.code_review.how_to_guide.md`
  - [70%]: general rules, author and reviewer points of view, workflows
- `helpers_root/docs/tools/ai/all.code_review_tools.reference.md`
  - [60%]: Copilot, Graphite, CodeRabbit, and Deepcode: features and prices
- `.claude/skills/github.describe_pr_for_review/SKILL.md`
  - [50%]: describe a PR in a fixed format, score it, suggest a review order
- `helpers_root/docs/work_organization/all.use_github.how_to_guide.md`
  - [50%]: PR filing, review, and addressing comments
- `website/docs/blog/posts/draft.how_to.Github_Copilot_Review.md`
  - [45%]: request a Copilot review as a first line of defense
- `helpers_root/docs/coding/all.submit_code_for_review.how_to_guide.md`
  - [40%]: first review process: style, linter, example code
- `.claude/skills/coding_qa.review/SKILL.md`
  - [35%]: review Python files for bugs and suggest fixes
- `.claude/skills/skill.check_against_rules/SKILL.md`
  - [30%]: check a file against the rules that apply to it
- `.claude/agents/code.reviewer.md`
  - [30%]: a code reviewer agent
- `helpers_root/docs/tools/ai/all.create_pr_with_claude_code.how_to_guide.md`
  - [25%]: opening and following up on a PR
- `website/docs/blog/posts/draft.how_to.How_to_run_Claude_on_GitHub.md`
  - [25%]: `@claude` on PRs for review help
- `.claude/skills/github.get_pr_ready_to_merge/SKILL.md`
  - [20%]: get the PR to a state that can be reviewed and merged
- `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
  - [20%]: from idea to merged PR, review and merge
- `research/ideas/draft.Codebase_automerge.md`
  - [15%]: automated sequential merging of PRs
- `.github/workflows/claude.yml`
  - [15%]: the workflow that runs Claude on issues and PRs
- _Not covered_
  - [15%]: measured review quality, reviewing AI-written PRs at scale

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
- `website/docs/blog/posts/draft.how_to.Read_other_people_code.md`
  - [55%]: what to do and what not to do when reading unfamiliar code
- `.claude/skills/readme_file.explain_code/SKILL.md`
  - [45%]: document classes, functions, and relationships of a file
- `.claude/skills/coding.find_doc/SKILL.md`
  - [40%]: find and summarize docs for a dir, file, class, or function
- `research/ideas/draft.Code_Visualization.md`
  - [40%]: visualizing Python codebases
- `helpers_root/docs/coding/all.code_design.how_to_guide.md`
  - [35%]: design philosophy, invariants, pure functions, and roles
- `helpers_root/docs/documentation_meta/all.architecture_diagrams.explanation.md`
  - [35%]: C4 component and code views, class diagrams
- `.claude/skills/readme_file.write_architecture/SKILL.md`
  - [35%]: write architecture docs for a Python file
- `helpers_root/docs/tools/codebase/all.import_check.reference.md`
  - [35%]: visualize directory, external, and cyclic dependencies
- `helpers_root/docs/tools/ai/all.ai_workflows.how_to_guide.md`
  - [30%]: finding rules for a file, running a skill on a file
- `website/docs/blog/posts/draft.how_to.Codebase_local_kg.md`
  - [30%]: a local knowledge graph of the codebase for the agent
- `helpers_root/docs/tools/codebase/all.imports_and_packages.how_to_guide.md`
  - [25%]: package hierarchy and anatomy of a package
- `.claude/skills/coding_qa.review/SKILL.md`
  - [20%]: review Python files for bugs
- `helpers_root/docs/planning/all.document_research.how_to_guide.md`
  - [15%]: research logs and reproducible notes
- `.claude/skills/coding.improve_code_for_humans/SKILL.md`
  - [10%]: readability and debuggability changes
- `helpers_root/papers/AIgentic_Development_System/part2.runnable_directory.tex`
  - [10%]: runnable directories as bounded units of code
- _Not covered_
  - [35%]: onboarding walkthroughs, tracing call graphs with AI

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
- `.claude/skills/pytest.rules.md`
  - [45%]: root-cause diagnosis, fix plan, verification of a fix
- `research/ideas/draft.Comparison_of_Debugging_Agents.md`
  - [45%]: comparison of AI code debugging agents
- `.claude/skills/pytest.fix_local_tests/SKILL.md`
  - [40%]: analyze and fix failing local tests
- `.claude/skills/pytest.fix_ci_tests/SKILL.md`
  - [35%]: analyze and fix failing tests in GitHub CI
- `.claude/skills/github.get_pr_to_pass_ci_tests/SKILL.md`
  - [30%]: monitor CI checks and fix failures until they pass
- `helpers_root/docs/tools/testing_profiling/all.profiling.how_to_guide.md`
  - [30%]: profiling a command, Python code, and notebooks
- `website/docs/blog/posts/draft.debug.md`
  - [30%]: run tests inside the container, pytest logs
- `helpers_root/docs/build_system/all.pytest_allure.how_to_guide.md`
  - [25%]: test reports for finding failures
- `.claude/skills/github.get_pr_to_pass_local_tests/SKILL.md`
  - [25%]: make the local tests pass for a PR
- `helpers_root/docs/coding/all.coding_style.how_to_guide.md`
  - [25%]: logging levels, assertions, and reporting warnings
- `research/ideas/draft.Counterfactual_Attribution_of_Agent_Failures.md`
  - [25%]: attributing agent failures to steps
- `website/docs/blog/posts/draft.TIL.Apple_container_running_notebook.md`
  - [20%]: a worked debugging story: problem, root cause, fix
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [20%]: limits of self-correction, when self-correction helps
- `.claude/skills/testing.update_test_outcomes/SKILL.md`
  - [20%]: update the expected outcome of failing tests
- `.claude/skills/coding.fix_pyright/SKILL.md`
  - [15%]: fix type-checker findings
- `helpers_root/docs/work_organization/all.buildmeister.how_to_guide.md`
  - [15%]: build break triage and escalation
- `website/docs/blog/posts/draft.how_to.Parse_Claude_logs.md`
  - [15%]: reading agent session logs
- `.claude/skills/cfile.rules.md`
  - [10%]: a quickfix file to jump to errors
- _Not covered_
  - [30%]: reproducing production failures, trace inspection

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
- `.claude/skills/project.rules.md`
  - [70%]: rules for the PRD, design, plan, progress, and decisions docs
- `.claude/skills/project.create_plan/SKILL.md`
  - [60%]: end-to-end plan of a project from its design docs
- `.claude/skills/auto_task.create_plan_for_feature/SKILL.md`
  - [55%]: break a feature into an ordered PR plan
- `.claude/skills/project.write_prd/SKILL.md`
  - [50%]: turn an idea into a PRD
- `.claude/skills/auto_task.rules.md`
  - [50%]: plan format and executing a task list
- `helpers_root/docs/planning/all.document_research.how_to_guide.md`
  - [45%]: research log, reproducible research, daily entries
- `helpers_root/docs/tools/ai/all.auto_task.how_to_guide.md`
  - [45%]: task lists, criticizing the plan, execution modes
- `website/docs/blog/posts/draft.how_to.A_queue_of_AI_coding_agents.md`
  - [45%]: a task queue, the unit of work, specs
- `.claude/skills/architecture.write_design/SKILL.md`
  - [40%]: write design docs from the PRD, before the code
- `.claude/skills/project.update_progress/SKILL.md`
  - [40%]: update plan, log, and decisions after a merge
- `helpers_root/docs/work_organization/all.github_projects_process.reference.md`
  - [40%]: sprint process, weekly summary, carryover of unfinished issues
- `website/docs/blog/posts/draft.how_to.My_agentic_engineering_flow.md`
  - [40%]: spec before code, criticize first, then execute
- `.claude/skills/auto_task.create_specs_from_todos/SKILL.md`
  - [40%]: turn a list of TODOs into a plan
- `.claude/skills/auto_task.criticize/SKILL.md`
  - [35%]: review a plan or issue before execution
- `helpers_root/docs/work_organization/all.organize_your_work.how_to_guide.md`
  - [35%]: golden rules, roles, starting work, organizing research
- `website/docs/blog/posts/draft.survey.tools_to_generate_sw_with_agents.md`
  - [35%]: spec-driven planning and task tracking tools
- `helpers_root/docs/work_organization/all.scrum.explanation.md`
  - [30%]: Scrum roles and goals
- `website/docs/blog/posts/draft.A_proposal_for_AI_based_software_engineering.md`
  - [30%]: a staged plan for agentic AI development
- `book_Agentic_AI/lectures_source/Lesson05.1-Reasoning_Memory_and_Planning.smd`
  - [15%]: planning as a loop, world models for planning
- _Not covered_
  - [10%]: effort estimation, roadmapping across quarters

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
- `helpers_root/docs/documentation_meta/all.architecture_diagrams.explanation.md`
  - [80%]: C4 levels, PlantUML conventions, class diagrams
- `helpers_root/papers/AIgentic_Development_System/part2.runnable_directory.tex`
  - [80%]: monorepo versus multi-repo, runnable directory architecture
- `.claude/skills/architecture.rules.md`
  - [75%]: layers, dependency direction, data flow, interfaces
- `helpers_root/docs/coding/all.code_design.how_to_guide.md`
  - [70%]: design philosophy, architecture, invariants, and roles
- `helpers_root/docs/tools/dev_system/all.runnable_repo.reference.md`
  - [65%]: design goals, helper sub-repo, runnable dir
- `.claude/skills/C4.rules.md`
  - [50%]: C4 diagrams with PlantUML
- `helpers_root/docs/tools/codebase/all.imports_and_packages.how_to_guide.md`
  - [50%]: package goals, hierarchy, cycle prevention, anatomy of a package
- `.claude/skills/architecture.write_design/SKILL.md`
  - [45%]: write design docs before the code
- `research/ideas/draft.AI_Architecture_Decision_Support.md`
  - [45%]: AI support for architecture decisions
- `helpers_root/docs/tools/dev_system/all.create_a_runnable_dir.how_to_guide.md`
  - [40%]: create a runnable dir as a sub-directory under a super-repo
- `book_AI_Software_Engineering/resources.md`
  - [40%]: package structure, import cycle detection, single versus multiple
    repo
- `helpers_root/docs/tools/dev_system/all.create_a_super_repo_with_helpers.how_to_guide.md`
  - [35%]: create a super-repo with the helpers sub-repo
- `research/ideas/draft.AI_Technical_Debt_Management.md`
  - [35%]: quantifying and remediating technical debt with AI
- `website/docs/blog/posts/draft.monorepo_vs_multirepo.md`
  - [30%]: monorepo versus multirepo trade-offs
- `helpers_root/docs/coding/all.coding_style.how_to_guide.md`
  - [30%]: architecture and design patterns, import conventions
- `helpers_root/docs/tools/codebase/all.import_check.reference.md`
  - [30%]: dependency and cycle visualization
- `.claude/skills/coding.factor_common_code/SKILL.md`
  - [30%]: refactor duplicated code into shared functions
- `helpers_root/docs/tools/thin_environment/all.thin_environment.reference.md`
  - [25%]: thin environment shared across repos
- `.claude/skills/skill.add_architecture/SKILL.md`
  - [25%]: add a rule to the architecture rules
- `research/ideas/draft.Code_Visualization.md`
  - [20%]: visualizing Python codebases
- `.claude/skills/coding.make_function_private/SKILL.md`
  - [15%]: mark functions private or public
- _Not covered_
  - [15%]: service boundaries, scalability trade-offs

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
- `helpers_root/docs/tools/git/all.git.how_to_guide.md`
  - [70%]: Git workflow, best practices, troubleshooting
- `helpers_root/docs/tools/dev_system/all.sync_gh_projects.how_to_guide.md`
  - [60%]: syncing project templates, previewing and applying changes
- `helpers_root/docs/tools/dev_system/all.synchronize_gh_issue_labels.how_to_guide.md`
  - [50%]: synchronize issue labels across repos
- `website/docs/blog/posts/draft.how_to.Use_GitHub_Stacked_PR_workflow.md`
  - [50%]: build a stack of branches and publish it as PRs
- `helpers_root/docs/tools/git/all.manual_flow.how_to_guide.md`
  - [45%]: tools for Git branch and GitHub workflow management
- `data605/lectures_source/Lesson02.2-Git_Branching.smd`
  - [45%]: branching, merging, rebase, centralized and forking workflows
- `website/docs/blog/posts/draft.how_to.Stacked_PRs_for_agentic_developent.md`
  - [45%]: interactive flows, task lists, and stacked PRs
- `helpers_root/docs/tools/dev_system/all.synchronize_gh_settings.how_to_guide.md`
  - [40%]: synchronize repository settings
- `.claude/skills/github.split_branch_in_PRs/SKILL.md`
  - [40%]: split Git changes into small cohesive PRs
- `.claude/skills/auto_task.execute_with_stacked_prs/SKILL.md`
  - [40%]: execute tasks as a stack of branches and PRs
- `website/docs/blog/posts/draft.how_to.i_git_merge_master.md`
  - [35%]: three flows for catching up with master
- `research/ideas/draft.Codebase_automerge.md`
  - [35%]: automate stacked-PR splitting and sequential merging
- `.claude/skills/auto_task.restack/SKILL.md`
  - [30%]: propagate reviewed changes down a PR stack
- `data605/lectures_source/Lesson02.1-Git.smd`
  - [30%]: version control basics, Git remotes, GitHub
- `website/docs/blog/posts/draft.how_to.How_to_run_Claude_on_GitHub.md`
  - [25%]: `@claude` on issues and PRs
- `helpers_root/docs/tools/invoke/all.invoke_git_branch_copy.how_to_guide.md`
  - [25%]: copy a Git branch with invoke
- `.claude/skills/github.create_child_pr/SKILL.md`
  - [25%]: build a child PR from the current branch
- `.claude/skills/git.merge_conflicts/SKILL.md`
  - [20%]: resolve merge, rebase, and cherry-pick conflicts
- _Not covered_
  - [10%]: branch protection, code ownership

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
- `helpers_root/docs/tools/docker/all.dind_and_sibling_containers.how_to_guide.md`
  - [55%]: sibling containers versus Docker-in-Docker
- `helpers_root/docs/work_organization/all.rollout.how_to_guide.md`
  - [50%]: roll-out process and announcement email
- `helpers_root/docs/tools/thin_environment/all.thin_environment.reference.md`
  - [50%]: thin environment: concepts and files
- `helpers_root/docs/tools/docker/all.docker.tutorial.md`
  - [45%]: concepts, building an image, multi-container applications
- `data605/lectures_source/Lesson03.2-Docker.smd`
  - [40%]: Dockerfile, building and running containers
- `.claude/skills/docker.use_standard_style/SKILL.md`
  - [35%]: align Docker files to the standard project template
- `research/ideas/draft.Improve_dockerized_executables.md`
  - [35%]: make dockerized executables more transparent
- `.claude/skills/docker.shrink_container/SKILL.md`
  - [35%]: shrink and speed up a Docker container
- `website/docs/blog/posts/draft.how_to.Apple_Container.md`
  - [30%]: run containers with Apple Container: install, configure, list
- `helpers_root/docs/tools/dev_system/all.docker_optimizer_container.how_to_guide.md`
  - [30%]: the optimizer container: build, invariants, ECR flow
- `helpers_root/docs/tools/docker/all.dockerhub.how_to_guide.md`
  - [25%]: publishing images to a registry
- `data605/lectures_source/Lesson03.3-Docker_Compose.smd`
  - [25%]: multi-container applications and compose commands
- `.claude/skills/docker.shrink_requirements/SKILL.md`
  - [25%]: find unused packages in requirements.txt
- `website/docs/blog/posts/draft.TIL.Apple_container_running_notebook.md`
  - [20%]: running a notebook in a container: port forwarding fix
- `.claude/skills/tutorial_in_60_mins.propagate_docker_changes/SKILL.md`
  - [15%]: sync tutorial Docker files to the project template
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
- `helpers_root/docs/work_organization/all.buildmeister.how_to_guide.md`
  - [60%]: buildmeister process, notifications, daily reporting and handover
- `helpers_root/docs/tools/invoke/all.invoke_workflows.how_to_guide.md`
  - [55%]: task categories and common workflows with invoke
- `book_AI_Software_Engineering/resources.md`
  - [50%]: CI/CD automation and the buildmeister role, invoke workflows
- `.github/workflows/fast_tests.no_docker.yml`
  - [45%]: the fast-test CI job of this repo
- `helpers_root/docs/tools/dev_system/all.devops_docker_auto_release.explanation.md`
  - [40%]: automated build and test, release workflow file
- `helpers_root/docs/tools/testing_profiling/all.codecov.how_to_guide.md`
  - [40%]: coverage upload from the GitHub Actions workflow
- `.claude/skills/github.get_pr_to_pass_ci_tests/SKILL.md`
  - [40%]: monitor CI checks and fix failures until they pass
- `.claude/skills/pytest.fix_ci_tests/SKILL.md`
  - [35%]: analyze and fix failing tests in GitHub CI
- `helpers_root/docs/build_system/all.linter_gh_workflow.explanation.md`
  - [35%]: the linter as a GitHub workflow
- `helpers_root/docs/build_system/all.pytest_allure.explanation.md`
  - [35%]: test reports with historical trends
- `website/docs/blog/posts/draft.how_to.How_to_run_Claude_on_GitHub.md`
  - [35%]: an AI step in GitHub Actions: workflow and secret
- `.github/workflows/claude.yml`
  - [30%]: the workflow that runs Claude on issues and PRs
- `helpers_root/papers/AIgentic_Development_System/part3.containerized_development.tex`
  - [30%]: integration with CI/CD and quality gates, buildmeister dashboard
- `.claude/skills/github.get_pr_ready_to_merge/SKILL.md`
  - [30%]: get the PR to a state that can be reviewed and merged
- `helpers_root/docs/build_system/all.semgrep_workflow.explanation.md`
  - [25%]: Semgrep as a GitHub Actions security gate
- `helpers_root/docs/build_system/all.gitleaks_workflow.explanation.md`
  - [20%]: Gitleaks as a GitHub Actions secrets gate
- `.github/workflows/update_helpers_submodule.yml`
  - [15%]: the workflow that updates the helpers submodule
- `helpers_root/docs/work_organization/all.github_projects_process.reference.md`
  - [15%]: GitHub Actions automations
- _Not covered_
  - [25%]: canary analysis, automated rollback, production incident loops
