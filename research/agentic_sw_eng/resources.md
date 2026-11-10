# AI Development Resources

- This file lists resources in this repo about AI-assisted and agentic software
  development
- Each entry is title: `path`, with paths relative to the repo root
- Blog posts live in `website/docs/blog/posts/`, which `blog_posts/` links to
- The titles come from the first heading of each file, fixed by hand when the heading
  was not descriptive

# Paper: Causify AIgentic Development System

- Causify AIgentic Development System:
  `helpers_root/papers/AIgentic_Development_System/paper.tex`
- Causify AIgentic Development System (PDF):
  `helpers_root/papers/AIgentic_Development_System/Causify_AIgentic_Development_System.pdf`
- Part 1: Code Quality:
  `helpers_root/papers/AIgentic_Development_System/part1.code_quality.tex`
- Part 2: Runnable Directory:
  `helpers_root/papers/AIgentic_Development_System/part2.runnable_directory.tex`
- Part 3: Containerized Development:
  `helpers_root/papers/AIgentic_Development_System/part3.containerized_development.tex`
- Part 4: AI-Optimized Development:
  `helpers_root/papers/AIgentic_Development_System/part4.AI_optimized_development.tex`

# Blog posts: Claude Code and coding agents

- How to Use Claude Code:
  `website/docs/blog/posts/draft.how_to.Use_Claude_Code.md`
- How to Use Claude Code Workflows:
  `website/docs/blog/posts/draft.how_to.Use_Claude_Code_Workflows.md`
- How to Create and Use Claude Skills:
  `website/docs/blog/posts/draft.how_to.Claude_skills.md`
- How to Create a Claude Artifact:
  `website/docs/blog/posts/draft.how_to.Claude_Artifacts.md`
- How to Enforce Rules in Claude Code:
  `website/docs/blog/posts/draft.how_to.Enforce_rules_in_Claude_Code.md`
- A Queue of AI Coding Agents:
  `website/docs/blog/posts/draft.how_to.A_queue_of_AI_coding_agents.md`
- Automate Coding with LLM: Scripts and Tools for AI-Powered Code Transformation:
  `website/docs/blog/posts/draft.how_to.Automate_Coding_with_LLM.md`

# Blog posts: Agentic engineering workflows

- One Approach for Organizing Software Development for Agentic AI:
  `website/docs/blog/posts/draft.A_proposal_for_AI_based_software_engineering.md`
- My Agentic Engineering Flow:
  `website/docs/blog/posts/draft.how_to.My_agentic_engineering_flow.md`
- Stacked PRs for Agentic Development:
  `website/docs/blog/posts/draft.how_to.Stacked_PRs_for_agentic_developent.md`
- How to Use the GitHub Stacked PRs Workflow:
  `website/docs/blog/posts/draft.how_to.Use_GitHub_Stacked_PR_workflow.md`
- Codebase Local Knowledge Graph:
  `website/docs/blog/posts/draft.how_to.Codebase_local_kg.md`
- My AI Policy:
  `website/docs/blog/posts/My_AI_Policy.md`

# Helpers docs and tools for AI development

- AI Workflows: Topics, Skills, and Rules:
  `helpers_root/docs/tools/ai/all.ai_workflows.how_to_guide.md`
- Auto_task Workflow: From Task List to Merged PR:
  `helpers_root/docs/tools/ai/all.auto_task.how_to_guide.md`
- How to Create a PR With Claude Code:
  `helpers_root/docs/tools/ai/all.create_pr_with_claude_code.how_to_guide.md`
- Plan Work, with the sequence of skills from idea to merged PR:
  `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
- How to Document Research:
  `helpers_root/docs/planning/all.document_research.how_to_guide.md`
- Review Flows With LLMs:
  `helpers_root/docs/tools/ai/all.ai_review.how_to_guide.md`
- Code Review Tools:
  `helpers_root/docs/tools/ai/all.code_review_tools.reference.md`
- Use GitHub (Projects and Issues):
  `helpers_root/docs/work_organization/all.use_github.how_to_guide.md`
- Where to Add Material:
  `helpers_root/docs/documentation_meta/all.where_to_add_material.explanation.md`
- How to Create a PR With Claude Code (policy of this repo):
  `policies/pr_process.how_to_guide.md`
- AI Tools:
  `helpers_root/dev_scripts_helpers/ai/README.md`
- LLM Tools:
  `helpers_root/dev_scripts_helpers/llms/README.md`

# Rules, skills, and agent configuration

The skills directory holds one subdirectory per skill and one `*.rules.md` file per
topic.

- Rules Index: Mapping from File Type to Rules and Templates:
  `helpers_root/.claude/rules.md`
- Agent Skills and Rules (directory):
  `helpers_root/.claude/skills/`
- Rules for Agent Skills:
  `helpers_root/.claude/skills/skill.rules.md`
- Templates (directory):
  `helpers_root/.claude/templates/`
- Rules for auto_task Plans:
  `helpers_root/.claude/skills/auto_task.rules.md`
- Rules for Project Docs (PRD, design, plan, progress log, decisions):
  `helpers_root/.claude/skills/project.rules.md`
- Skills for Project Docs:
  `helpers_root/.claude/skills/project.write_prd/SKILL.md`,
  `helpers_root/.claude/skills/architecture.write_design/SKILL.md`,
  `helpers_root/.claude/skills/project.create_plan/SKILL.md`,
  `helpers_root/.claude/skills/project.update_progress/SKILL.md`
- Project Doc Templates:
  `helpers_root/.claude/templates/project_prd.template.md`,
  `helpers_root/.claude/templates/project_design.template.md`,
  `helpers_root/.claude/templates/project_plan.template.md`,
  `helpers_root/.claude/templates/project_progress.template.md`,
  `helpers_root/.claude/templates/project_decisions.template.md`,
  `helpers_root/.claude/templates/project_how_we_work.template.md`
- PR Description Template:
  `helpers_root/.claude/templates/github_PR_description.template.md`
- AI Instruction Template:
  `helpers_root/.claude/templates/ai.instruction_template.md`
- auto_task Template:
  `helpers_root/.claude/templates/auto_task.template.md`
- Implementation Spec Template:
  `helpers_root/.claude/templates/specs.template.md`
- Code Reviewer Agent:
  `helpers_root/.claude/agents/code.reviewer.md`
