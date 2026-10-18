- For all the actions below
  - Make reference to documents docs/ already existing
  - Do not replicate the content, but when factoring out common code add a pointer to
    the file

- Create a document in helpers_root/docs/planning/ about how we plan work

- Use content from research/Noesis/docs/how_we_work.md to general projects that are
  not Noesis
  - Factor out the common part and point

- In the doc there should be a part about creating a structure with an end-to-end
  plan using the format like
  ./research/Noesis/docs/plans/plan.Noesis.md

- In the doc there should be a part about creating a log of what was done similar to
  ./research/Noesis/docs/plans/mvp_progress.md (it's a log of everything that was
  done, including decisions)

- Use the style of documentation we use in helpers_root/docs

## Plan

- [x] Gather context
  - [x] Read `research/Noesis/docs/how_we_work.md`, `plans/plan.Noesis.md`,
        `plans/mvp_plan.md`, `plans/mvp_progress.md`
  - [x] Read the docs that already cover part of the topic, to point to them:
    - `helpers_root/docs/work_organization/all.issue_workflow.explanation.md`
    - `helpers_root/docs/work_organization/all.use_github.how_to_guide.md`
    - `helpers_root/docs/coding/all.submit_code_for_review.how_to_guide.md`
    - `helpers_root/docs/tools/ai/all.auto_task.how_to_guide.md`
    - `.claude/templates/auto_task.template.md`, `specs.template.md`,
      `github_PR_plan.template.md`
  - [x] Read the doc style: Diataxis file names, `all.writing_docs.how_to_guide.md`
- [x] Create `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
  - File name: `all.<verb_object>.how_to_guide.md` (Diataxis, goal oriented)
  - [x] Summary and the planning hierarchy (idea -> plan -> task -> PR), pointing to
        `all.issue_workflow.explanation.md`
  - [x] "The loop": factor out the generic steps of `how_we_work.md` (pick task,
        plan before code, issue, branch, code and tests, commit, PR, review, merge)
        and point to `all.use_github.how_to_guide.md` instead of copying it
  - [x] "Create an end-to-end plan": structure of a `plan.<Project>.md`, using
        `plan.Noesis.md` as the worked example (goal, plan hierarchy, roadmap with
        versions and checkboxes, component specs, PR list, conventions)
  - [x] "Keep a progress log": structure of a `<project>_progress.md`, using
        `mvp_progress.md` as the worked example (status, built, findings, next,
        history, decisions)
  - [x] "Keep docs in sync": when to update the plan, the log, and the decisions
  - [x] PR description template (the part marked `TODO(gp): Move out`)
  - [x] Resources
- [x] Update `research/Noesis/docs/how_we_work.md`
  - Keep only what is Noesis-specific (fork flow, `Noesis:` titles, `UmdTask` branch
    names, `mvp_plan.md` pointers, Noesis coding quirks, public-repo warning)
  - Replace the factored-out parts with pointers to the new doc
- [x] Update `helpers_root/docs/README.md`: add `docs/planning/` to the dir structure
- [x] Check
  - [x] Lines at most 85 columns (except unbreakable links and pre-existing lines),
        ASCII only, no em-dash, no trailing period in bullets
  - [x] Every path in a pointer exists, every anchor exists
  - [x] `git add` the new file (inside the `helpers_root` repo), no commit
- [x] Add the `Result` section below

## Result

### Done
- Created `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
  - Planning hierarchy and the set of plan documents (plan, execution plan, log,
    decisions, spec)
  - The loop (issue, branch, PR, review, merge) with pointers to
    `all.use_github.how_to_guide.md`, `all.git.how_to_guide.md`, and
    `all.submit_code_for_review.how_to_guide.md`
  - How to create an end-to-end plan: file structure, PR entry format, rules (from
    `plan.Noesis.md`)
  - How to keep a progress log with decisions: file structure, rules, decision format
    (from `mvp_progress.md` and `mvp_decisions.md`)
  - Docs-in-sync rules and the generic PR description template
- Factored out `research/Noesis/docs/how_we_work.md`
  - Generic loop, PR size rule, PR template, and sync rules moved to the new doc
  - Noesis keeps its fork flow, naming, `Fixes` line, conventions, and public-repo
    warning, plus pointers to the new doc
  - Removed the `TODO(gp): Move out` comment since the PR template moved
- Added `docs/planning/` to `helpers_root/docs/README.md`
- Ran `git add` on the new file (in the `helpers_root` repo), no commit

### Not done
- Did not run the linter (`i lint`): needs Docker, so the TOC was written by hand
  - Run `i lint --files="helpers_root/docs/planning/all.plan_work.how_to_guide.md"`
- Did not add a `Last review` tag: no one reviewed the doc yet
- Did not create a plan or log template under `.claude/templates/`: the doc embeds
  the PR entry and history skeletons and points to the Noesis files as examples
- Did not merge `all.issue_workflow.explanation.md` into the new doc, although it has
  a `TODO(Grisha)` suggesting a planning doc: out of scope, the new doc points to it
- Did not change how `Fixes` is used: `all.use_github.how_to_guide.md` says to avoid
  `Fixes`, while the Noesis fork flow requires it
  - `how_we_work.md` now states that Noesis overrides the generic advice
