# How to Create a Tutorial

- This document is the entry point to create a tutorial in `tutorials/`
  - It collects the order of the steps and the skills to run at each step
  - It does not repeat the rules: it points to the file that owns each rule
- Work in progress: grow this file into the single guide for tutorials
  - Add a step here only after it worked on a real tutorial
  - Mark a doubt with `TODO(gp): Check ...`

## Quality Principles

- Read `tutorials/tutorials_checklist.md` before you start
- Avoid AI slop: a tutorial is not a prompt pasted into Claude Code
  - Read the sources, then synthesize them for a reader who has less time
- Use a professional tone: write for a peer
- State each fact once, in the right place
- Show examples: do not explain what the code already shows

## Pick the Kind of Tutorial

- Use the table to choose the deliverable and the file that owns its rules

| I want                                       | Deliverable                                       | Owner of the rules                                  |
| :------------------------------------------- | :------------------------------------------------ | :-------------------------------------------------- |
| "Learn X in 60 Minutes" for a package        | Dir `tutorials/<PROJ>/` with notebooks and Docker | `.claude/skills/tutorial_in_60_mins.rules.md`       |
| Teach the API of a package, one concept/cell | Notebook `<PROJ>.<ID>.API.<topic>.ipynb`          | `.claude/skills/notebook.rules.md`                  |
| Turn a lecture or a doc into a notebook      | Outline `.md`, then notebook and `*_utils.py`     | `.claude/skills/notebook.rules.md`                  |
| A quick-reference guide of a tool            | One file `in_30_mins.<tool>.md`                   | `.claude/skills/tool_guide_in_30_mins.rules.md`     |
| A blog post about the tutorial               | A post in `website/docs/blog/posts/`              | `.claude/skills/blog.rules.md`                      |
| A proposal of a tutorial for a class project | `class_project/project_descriptions/*/*.md`       | `.claude/skills/tutorial_specs.rules.md`            |

- The common case is the first row, and the notebooks come from rows 2 and 3

## Anatomy of a Tutorial

- A tutorial dir has these files
  ```text
  tutorials/<PROJ>/
  ├── <PROJ>_utils.py                 # Code behind the cells, no notebook logic
  ├── <PROJ>.01.API.<topic>.ipynb     # API walkthrough
  ├── <PROJ>.01.API.<topic>.py        # Jupytext percent mirror of the notebook
  ├── <PROJ>.example.ipynb            # End-to-end application (optional)
  ├── <PROJ>.example.py               # Jupytext percent mirror
  ├── Dockerfile, docker_*.sh         # Copied from the template
  ├── requirements.txt                # Pinned versions
  └── README.md                       # Quick start, less than 35 lines
  ```
- A notebook is a triplet kept in sync with `jupytext`: `.ipynb`, `.py`, and
  `*_utils.py`
- Two naming conventions exist
  - `.claude/skills/tutorial_in_60_mins.rules.md` uses `<PROJ>.API.ipynb`
  - `.claude/skills/notebook.create_api_intro/SKILL.md` and `tutorials/pgmpy/` use
    `<PROJ>.<ID>.API.<topic>.ipynb`, with the next `<ID>` for each new topic
  - TODO(gp): Reconcile the two conventions

## The Flow at a Glance

1. Create the dir from the template
2. Align Docker files and packages
3. Write the notebook
   - API intro of a package: `/notebook.create_api_intro`
   - Examples or lecture material: `/notebook.create_outline`, then
     `/notebook.implement_outline`
4. Polish the notebook and `*_utils.py`
5. Run the notebook in the container
6. Write the `README.md` and the blog post
7. Lint, then review the diff
8. Commit only after the user agrees

## Step 1: Create the Directory

- Create the skeleton
  ```bash
  > export PROJ=...
  > class_project/create_project.py --dst_dir tutorials/${PROJ}
  ```
  - See `class_project/create_project.README.md` for the other actions, e.g.,
    `--action copy_docker_files` and `--action compare_docker_files`
- Two templates exist
  - `class_project/project_template/`: the Docker structure to follow
  - `tutorials/project_template/`: a skeleton with `template.API.ipynb` and
    `template.example.ipynb`
- Commit the empty skeleton to have a clean base
  ```bash
  > git add tutorials/${PROJ}
  > git commit -am "Add template"
  ```

## Step 2: Align Docker Files and Packages

- Set the image name in `tutorials/${PROJ}/docker_name.sh`
  ```bash
  IMAGE_NAME=umd_project_${PROJ}
  ```
  - Use only lower case letters
- Install the packages from the notebook with `!pip install -q pkg==x.y.z`
  - Keep the same Docker image and `requirements.txt` as
    `class_project/project_template`
  - Edit `requirements.txt` only if the dependencies are hard to maintain, e.g.,
    packages that depend on each other
  - See "Python Dependencies Policy" in
    `class_project/project_template/docker_scripts.README.md`
- Run the skills
  ```text
  claude> /notebook.fix_packages tutorials/${PROJ}
  claude> /docker.use_standard_style tutorials/${PROJ}
  ```
- Build the container
  ```bash
  > cd tutorials/${PROJ}
  > docker_build.sh
  ```
- Pass a secret, e.g., an API key, into the container by name only
  - Read it from an environment variable in the notebook
  - Never print it, never write it in a file, never commit it
  - Check that `docker_bash.sh` and `docker_jupyter.sh` forward the variable

## Step 3: Write the Notebook

### Case A: Notebook for a Package API

- Use this to teach the API of a package, one primitive per cell
  ```text
  claude> /notebook.create_api_intro <PROJ> <doc URL> and save it in tutorials/<PROJ>/<PROJ>.01.API.<topic>.ipynb
  ```
- The skill is `.claude/skills/notebook.create_api_intro/SKILL.md`
  - It uses toy objects (2 to 5), one concept per cell, and a mental model table
    per primitive
  - It pairs the notebook with a `.py` file and writes `<PROJ>_utils.py`
- If the package is a REST API without a Python client
  - Use plain `requests` calls
  - Link the official docs of each endpoint right after the first call to it,
    instead of introspecting a class
  - Do not write live values (counts, search results) in markdown: describe the
    shape of the outcome, e.g., "an integer, larger than 0"

### Case B: Notebook for Examples or Lecture Material

- Brainstorm ideas (optional)
  ```text
  claude> /notebook.brainstorm_outline <lecture file>
  ```
  - It writes `notebook_ideas.<TAG>.md` with 5 ideas
- Create the outline
  ```text
  claude> /notebook.create_outline <lecture file, URL, or notebook_ideas.<TAG>.md>
  ```
  - It writes `notebook_outline.<TAG>.md`, with no code
  - Review and edit the outline: this is the cheap place to fix the design
- Implement the outline
  ```text
  claude> /notebook.implement_outline tutorials/${PROJ}/notebook_outline.<TAG>.md
  ```
  - It creates `<TAG>.ipynb` and `<TAG>_utils.py`
- Pair the notebook with a `.py` file
  ```bash
  > jupytext.py --action pair --files tutorials/${PROJ}/<TAG>.ipynb
  ```
- TODO(gp): Add a worked example of the review of an outline

## Step 4: Polish the Notebook and the Utils

- Run the skills in this order, and run `lint_numbered_cells` last because it
  renames the `cellN_*()` functions in the notebook and in `*_utils.py`
  1. `/notebook.fix_logging <notebook>`: standard logging init
  2. `/notebook.split_cells <notebook>`: one task per code cell
  3. `/notebook.split_header_cells <notebook>`: one header per markdown cell
  4. `/notebook.format_md_cells <notebook>`: nested bullets, width 85
  5. `/notebook.format_interactive_cells <notebook>`: only for widget notebooks
  6. `/notebook.lint_numbered_cells <notebook>`
- Clean `<PROJ>_utils.py`
  ```text
  claude> /coding.fix_from_imports tutorials/${PROJ}/${PROJ}_utils.py
  claude> /coding.fix_empty_lines tutorials/${PROJ}/${PROJ}_utils.py
  claude> /coding.fix_comments tutorials/${PROJ}/${PROJ}_utils.py
  claude> /coding.fix_type_hints tutorials/${PROJ}/${PROJ}_utils.py
  ```
- Fix a notebook that grew organically with
  - `/notebook.refactor_to_utils`, `/notebook.delete_dead_code`,
    `/notebook.split_cells`
- Merge a standalone markdown file into a notebook with
  `/tutorial_in_60_mins.merge_markdown`
- Apply all notebook rules in one call, instead of skill by skill
  ```bash
  > cc_lint.py --files "<notebook>" --topic notebook --mode session
  ```
  - Use the skills when you want to review each transformation
- TODO(gp): Confirm the order of the steps above on a second tutorial. It comes
  from `helpers_root/docs/tools/ai/all.ai_workflows.how_to_guide.md` and from the
  Semantic Scholar task in `instr.md`

## Step 5: Run the Notebook

- Run the paired Python file in the container
  ```bash
  > docker_cmd.sh "python /git_root/tutorials/${PROJ}/<name>.py"
  ```
  - A `# !pip install` cell is a comment in the `.py` file: chain
    `pip install -q <pkg>==<version> &&` before `python`
  - See `## Testing Notebook` in `.claude/skills/notebook.rules.md`
- Run the notebook after a kernel restart: it must finish without an error
- Test a notebook that needs a key with the key and without the key
- Edit interactively, then sync the pair
  ```bash
  > docker_jupyter.sh
  > docker_jupyter_port_forward.py umd_project_${PROJ}.jupyter <PORT>
  # Open http://localhost:8889/lab
  > jupytext.py --action sync --files tutorials/${PROJ}/<name>.ipynb
  ```

## Step 6: Write the README and the Blog Post

- Write `tutorials/${PROJ}/README.md`
  ```text
  claude> /readme_dir.create tutorials/${PROJ}
  ```
  - Follow `.claude/skills/tutorial_in_60_mins.rules.md`, section "Create README.md"
  - Imitate `tutorials/Autogen/README.md`, `tutorials/BambooAI/README.md`, or
    `tutorials/TensorFlow/README.md`
  - Format it with `lint_text.py -i`
- Write the blog post
  ```text
  claude> /blog.write_tutorial_readme ${PROJ}
  ```
  - For an ML library or a technique use `/blog.write_ml_tutorial`
  - For a single-file guide see `.claude/skills/tool_guide_in_30_mins.rules.md`

## Step 7: Lint and Review

- Lint the touched files
  ```bash
  > cc_lint.py --files "<file>" --topic <topic> --mode session
  ```
- Check the list "Validation Checklist" in
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - TODO(gp): That checklist cites `linters2/lint_branch.sh`, which does not
    exist in the repo. Fix the checklist
- Ask the user before you commit

## Run the Flow With `instr.md`

- Use an instruction file to hand the whole flow to Claude Code
  ```bash
  > create_instr
  ```
  - It creates `instr.md` from `.claude/templates/ai.instruction_template.md`
  - See `helpers_root/dev_scripts_helpers/ai/README.md` for `instr2.md` and
    `instr3.md`
- Run it with `/instr`, which executes `./instr.md` following
  `.claude/task_instructions.md`
- Structure of a good tutorial `instr.md`
  1. `# Task`: one sentence with the package and the kind of notebook
  2. Constraints: do not commit, do not push, read `.claude/rules.md` first
  3. `## Background to Read First`: local docs, a reference implementation, the
     official docs, and what to do if the docs are not reachable
  4. `## Decisions`: names, paths, scope, and the rules for secrets and for
     values that change over time. Write "already made, do not ask again"
  5. `## Skills to Execute`: the skills in order, each with its parameters and the
     content to cover
  6. `## Verification`: pair, run in the container, lint
  7. `## Out of Scope`: what to ask the user about before doing it
  8. `## Done When`: the list of checks and the content of the final report
  9. `## Plan`: a checkbox list that Claude Code updates while it works
- The worked example is the Semantic Scholar task in `instr.md`
  - It runs `notebook.create_api_intro`, `notebook.fix_packages`,
    `notebook.fix_logging`, and `notebook.split_cells` for the package
    `semantic_scholar`
  - Copy it to a new `instr.md` and edit the decisions, do not start from scratch
  - TODO(gp): Move the example to `.claude/templates/` once it has run once

## Skills Used by a Tutorial

- List the skills with `mdm skill list <topic>` and read one with
  `mdm skill describe <skill>`

- Directory
  - `tutorial_in_60_mins.create`, `tutorial_in_60_mins.format`
- Docker
  - `notebook.fix_packages`, `docker.use_standard_style`,
    `tutorial_in_60_mins.propagate_docker_changes`
- Design
  - `notebook.brainstorm_outline`, `notebook.create_outline`
- Notebook
  - `notebook.create_api_intro`, `notebook.implement_outline`,
    `notebook.create_animation`, `tutorial_in_60_mins.merge_markdown`
- Polish
  - `notebook.fix_logging`, `notebook.split_cells`, `notebook.split_header_cells`,
    `notebook.format_md_cells`, `notebook.format_interactive_cells`,
    `notebook.lint_numbered_cells`, `notebook.refactor_to_utils`,
    `notebook.delete_dead_code`
- Utils
  - `coding.fix_from_imports`, `coding.fix_empty_lines`, `coding.fix_comments`,
    `coding.fix_type_hints`
- Docs
  - `readme_dir.create`, `blog.write_tutorial_readme`, `blog.write_ml_tutorial`
- Link a tutorial to lecture slides
  - `slides.add_tutorial_links`

## Reference Files

- Rules and templates
  - `.claude/rules.md`: map from file type to rules and templates
  - `.claude/skills/tutorial_in_60_mins.rules.md`: layout and checklist of a tutorial
  - `.claude/skills/notebook.rules.md`: conventions of a notebook
  - `.claude/templates/API_notebook.template.ipynb` and
    `.claude/templates/API_notebook.py.template`: skeleton of an API notebook
  - `.claude/templates/notebook.template.ipynb`,
    `.claude/templates/notebook.template.py`, and
    `.claude/templates/notebook_utils_template.py`: skeleton of any notebook
  - `.claude/templates/ai.instruction_template.md`: skeleton of an `instr.md`
- Guides
  - `helpers_root/docs/tools/ai/all.ai_workflows.how_to_guide.md`: topics, skills,
    and the section "Notebook Workflows"
  - `class_project/create_project.README.md`: create a dir and sync Docker files
  - `class_project/README.md`: how class projects are organized
  - `tutorials/tutorials_checklist.md`: onboarding and quality principles
- Existing tutorials to imitate
  - `tutorials/pgmpy/`: numbered API notebooks and an outline
  - `tutorials/Autogen/`, `tutorials/BambooAI/`, `tutorials/TensorFlow/`: README and
    layout of a full tutorial
