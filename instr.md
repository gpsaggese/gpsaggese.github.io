I want to create two files similar to .claude/skills/research_idea.rules.md and
.claude/template/research_idea.template.md for describing the specs of a tutorial
"Learn X in 60 mins" like tutorial_specs.rules.md and tutorial_specs.template.md
about how to organize proposals in class_project/project_descriptions/MSML610/

Use the info below

# Tutorials (Learn X in 60 Minutes)

## Conventions

- `.claude/skills/tutorials_in_60_mins.rules.md`: Main spec for a 60-minute
  tutorial
  - Time split: setup, intro, API notebook, example notebook
  - Deliverables: `<project>_utils.py`, `<project>.API.ipynb`,
    `<project>.example.ipynb`
- `.claude/skills/tutorials_in_30_mins.rules.md`: Shorter 30-minute variant

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

## Plan

- [x] Survey the existing proposals in
  `class_project/project_descriptions/MSML610/` (117 files)
  - Legacy format: bold `**Description**`, `Technologies Used`, 3 ladder
    projects `### Project N` with `**Difficulty**: N (Easy|Medium|Hard)`
  - New Fall2026 format (10 files, e.g., `CrewAI`, `DoWhy`, `MLflow`, `Polars`):
    `### Fall2026` with `#### Tutorial` (specs of "Learn X in 60 mins") and
    `#### Project`, followed by the legacy ladder projects
- [x] Write `.claude/skills/tutorial_specs.rules.md`
  - Model on `.claude/skills/research_idea.rules.md`
  - Cover file naming, section structure (Description, session section with
    Tutorial and Project, project ladder), README tracking table
  - Include the tutorial conventions, skills, workflow, templates, and blog
    posts from the info above
- [x] Write `.claude/templates/tutorial_specs.template.md`
  - Model on `.claude/templates/research_idea.template.md`
  - Skeleton with `<VAR>` placeholders and no decision criteria
- [x] Check both files against `.claude/skills/markdown.rules.md`,
  `.claude/skills/text.rules.md`, and `.claude/skills/skill.rules.md`
  (`## Rules vs Templates`)
- [x] Run `git add` on the two new files (do not commit)

## Assumptions

- The template goes in `.claude/templates/` (the task says `.claude/template/`,
  but only `.claude/templates/` exists)
- `.claude` is a symlink to `helpers_root/.claude`, and `helpers_root` is a
  git submodule, so `git add` runs inside `helpers_root`
- `.claude/rules.md` is not changed, since `research_idea.rules.md` is not
  listed there either

## Result

### Done

- Created `.claude/skills/tutorial_specs.rules.md`
  - Layout: file naming, proposal sections, tutorial and project subsections,
    project ladder, README tracking table, tutorial ecosystem (info from above)
  - Based on the 117 existing proposals, mainly the new `### Fall2026` structure
    with `#### Tutorial` and `#### Project` (e.g., `CrewAI`, `DoWhy`, `MLflow`)
- Created `.claude/templates/tutorial_specs.template.md`
  - Skeleton with `<VAR>` placeholders, no decision criteria
- Ran `git add` on both files inside the `helpers_root` submodule (no commit)

### Not done

- `.claude/rules.md` is not updated with the new rules and template
  - `research_idea.rules.md` is not listed there, and the file has pending edits
- The existing 117 proposals are not reformatted to the new template
  - The task only asked for the rules and the template
- To confirm with the user, since these are my choices and not from the source:
  - `####` headings are kept, as an explicit exception to `markdown.rules.md`
  - Newest session goes first, and older session sections are kept
  - `Project N` entries are the project options next to the session section
