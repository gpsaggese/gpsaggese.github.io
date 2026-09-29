# Main Guidelines

- `class_project/README.md`: Project rules for DATA605 and MSML610. Two project
  types, team size, deliverables. Start here
- `class_project/how_to_contribute.md`: Fork and PR workflow for student
  contributions

# Project Template and Creation

- `class_project/create_project.README.md`: How to use create_project.py to copy the
  template into a new project dir
- `class_project/project_template/docker_scripts.README.md`: Explains the Docker
  scripts in the template

# Project Descriptions

- `class_project/project_descriptions/README.md`: Tables of all tool projects for
  DATA605 and MSML610
- `class_project/project_descriptions/all_projects.md`: Full project list
- `class_project/project_descriptions/paper_candidates_analysis.md`: Analysis of
  candidate papers for research projects
- `class_project/project_descriptions/DATA605/<tool>_Project_Description.md`: One
  blueprint per DATA605 tool
- `class_project/project_descriptions/MSML610/<tool>_Project_Description.md`: One
  blueprint per MSML610 tool

# TA Files

- `class_project/ta/README.md`: TA workflow. Build tool list, then generate
  descriptions
- `class_project/ta/project_prompt.md`: LLM prompt to make a project blueprint for a
  tool
- `class_project/ta/research_prompt.md`: LLM prompt for research project
  descriptions
- `class_project/ta/DATA605_project_example.md`: Example description (TextBlob)
- `class_project/ta/generate_class_project_description.py`: Script that generates the
  descriptions

# Tutorials (Learn X in 60 Minutes)

## Conventions

- `.claude/skills/tutorials_in_60_mins.rules.md`: Main spec for a 60-minute
  tutorial
- `.claude/skills/tool_guide_in_30_mins.rules.md`: Single-markdown quick-reference
  guide for a tool (not a tutorial directory)

## Skills

- `.claude/skills/tutorials_in_60_mins.create/SKILL.md`: Create a new tutorial
  dir
- `.claude/skills/tutorials_in_60_mins.format/SKILL.md`: Format a dir to follow
  the conventions
- `.claude/skills/tutorials_in_60_mins.merge_markdown/SKILL.md`: Merge a
  markdown file into a notebook
- `.claude/skills/tutorials_in_60_mins.propagate_docker_changes/SKILL.md`:
  Sync the Docker files with `project_template`
- `helpers_root/how_to.ai_workflows.md`: Lists the `tutorials_in_60_mins` skill
  group

## Workflow

- `tutorials/README.gp.md`: Steps to create a tutorial, from
  `class_project/create_project.py` to `/blog.write_tutorial_readme`
- `tutorials/tutorials_checklist.md`: Onboarding checklist and quality principles
- `helpers_root/docs/blogging/all.write_blog.how_to_guide.md`: Blog guide that
  points to the tutorial conventions

## Templates and Examples

- `tutorials/project_template/`: Skeleton to copy (`template_utils.py`,
  `template.API.ipynb`, `template.example.ipynb`, Docker scripts)
- `tutorials/fastapi/`: Complete example of the three deliverables
- `tutorials/<tool>/`: One dir per tool (e.g., `shap`, `lime`, `tsfresh`,
  `LangChain_LangGraph`, `TorchRL_MAC`)
- `research/Causal_Analysis_of_Agent_Skill_And_Luck/all.learn_Causal_Analysis_of_Success_in_60_minutes.how_to_guide.md`:
  Research-side tutorial

## Blog Posts

- `website/docs/blog/posts/in_60_mins.<Tool>.md`: Published posts (`CausalML`,
  `Tensorflow`, `AutoGen`, `BambooAI`, `TorchRL_MAC`, `FastAPI`)
- `website/docs/blog/posts/draft.in_60_mins.GluonTS.md`: Draft post
- `website/README.blog.md`: Tracks blog posts and their status

# Research Projects

## Rules and Policies

- `class_project/README.md`: "Research Projects" section. Open question, teams of at
  most 3 students, good work becomes a blog post or paper
- `class_project/gp/email.project.md`: Email to students describing the small
  research project (notebooks, GitHub, blog or paper, video) and the links to pick
  an idea
- `class_project/gp/email.research.md`: Email on doing research while taking the
  class
- `policies/how_to_join_the_research_team.md`: Steps to join the research group
- `msml610/lectures_source/Class_Mechanics.aux.md`: Slides "Class Project: Small
  Research"
- `data605/gp/announcements.md`: Announcements with the project sign-up links

## Research Ideas

- `research/ideas/README.md`: Index of ideas
  - Table of active projects and assignees
  - Status prefixes and tracking table for all ideas
- `research/ideas/<STATUS>.<Idea_Name>.md`: One file per idea
  - `<STATUS>` is `draft`, `ready`, `in_progress`, or `done`
- `research/ideas/to_review/`: Draft ideas waiting for review
- `.claude/templates/research_idea.template.md`: Template for an idea file
- `.claude/skills/research_idea.rules.md`: Rules for file naming, template use, and
  the `README.md` tracking table

## Skills

- `.claude/skills/research_idea.add_from_file/SKILL.md`: Split a raw idea dump into
  idea files
- `.claude/skills/research_idea.brainstorm/SKILL.md`: Brainstorm 5 new ideas
- `.claude/skills/research_idea.check_redundant/SKILL.md`: Find and merge
  overlapping ideas
- `.claude/skills/research_idea.format/SKILL.md`: Format an idea file to the
  template
- `.claude/skills/research_idea.update_readme/SKILL.md`: Update the tracking table
  in `research/ideas/README.md`
- `.claude/skills/research_idea.write_draft_paper/SKILL.md`: Write a conference
  paper draft from an idea
- `.claude/skills/paper.*/SKILL.md`: Improve a paper (`paper.fix_figures`,
  `paper.improve_bibliography`, `paper.suggest_improvements`, `paper.use_style`)

## Code and Papers

- `research/<Project>/`: Code, notebooks, and notes for an active project
  - `research/Noesis/`: Noesis core platform
  - `research/agentic_data_science/`: RL for Automated EDA
  - `research/Causal_Analysis_of_Agent_Skill_And_Luck/`: Causal analysis of agent
    skill and luck
  - `research/Causal_Analysis_of_Financial_Tradability/`: Financial tradability
    analysis
  - `research/Implement_MonteCarlo_Tree_Search_and_Alpha_Zero/`: MCTS for discrete
    NP problems
  - `research/agentic_outreach/`: Agentic outreach
- `papers/<Paper_Name>/`: Paper for a project (e.g., `Noesis`,
  `RL_for_Automated_EDA`, `Optimal_strategy_for_racket_sports`,
  `AlphaZero_MCTS_for_TSP`)
- `papers/template/`: Paper template (`paper.md`, `Makefile`, `references.bib`,
  `ieee-template.typ`)
- `website/docs/06_research.md`: Research areas and publication lists

# Prompts

- `class_project/prompt.readme.md`: Rules for the project CSV (Team column, GroupId)
- `class_project/project_descriptions/prompt.analysis.md`: Prompt for project
  analysis
- `class_project/project_descriptions/prompt.update_README.md`: Prompt to update the
  descriptions README

# Per-term Data

- `class_project/data605/Spring2026/projects.csv`: DATA605 Spring 2026 team and
  project assignments
- `class_project/msml610/Fall2026/class_project.csv`: MSML610 Fall 2026 project
  choices
- `class_project/msml610/Fall2026/chosen_projects.csv`: MSML610 Fall 2026 final
  project list
- `class_project/data605/<Term>/projects/`: Student work per term
