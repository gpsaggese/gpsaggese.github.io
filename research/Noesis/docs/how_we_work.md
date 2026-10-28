# Noesis — How We Work

- The process for every change to `research/Noesis/`, so GP can review quickly and
  anyone can see why the code is the way it is

- The generic process (the loop, the plan, the progress log, the PR template) is in
  `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
  - This doc adds only the Noesis-specific parts

// TODO(javin): Update since for core controbutors we will just work on the main
// repo. We can still have the fork approach for external contributors.
- Contributions use the **fork and PR** method described in
  `class_project/how_to_contribute.md`

## The Loop

- The steps are the ones in the section "The Loop" of
  `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
- Noesis details for each step:
  - **Pick a task** from the Task Breakdown in `plan.md` (or propose one)
  - **Open a GitHub issue** in `gpsaggese/gpsaggese.github.io`, title
    `Noesis: <what>`; note the number, e.g., `#612`
  - **Sync your fork** with upstream, then **branch** in your fork from `master`,
    named after the issue: `UmdTask612_Noesis_<short_title>`

    ```bash
    > git fetch upstream && git checkout master && git merge upstream/master
    > git checkout -b UmdTask612_Noesis_<short_title>
    ```

  - **Code + tests** following the conventions below; run the Noesis tests locally
  - **Commit** with the issue reference, then push to your fork:

    ```bash
    > git add research/Noesis/<file> ...
    > git commit -m "<message> (gpsaggese/umd_classes#612)"
    > git push origin UmdTask612_Noesis_<short_title>
    ```

  - **Open a PR** from your fork's branch to `gpsaggese/gpsaggese.github.io`
    `master`, titled like the branch, using the PR description template below
    - Put `Fixes gpsaggese/umd_classes#612` at the start of a line so the issue
      closes on merge
      - This overrides the "Filing Process" advice in
        `helpers_root/docs/work_organization/all.use_github.how_to_guide.md`, which
        avoids `Fixes`
    - Add GP under **Reviewers** (not Assignees)
  - **Review**: a teammate first, then GP
  - **Merge** (GP), then sync your fork and mirror to the team backup repo

- `gpsaggese/umd_classes` is the repo's former name; GitHub redirects it, and it's
  the prefix `how_to_contribute.md` uses for issue references

## Conventions (GP's Rules)

- Coding rules: `helpers_root/.claude/skills/coding.rules.md`
- The ones that matter most here:
  - `hdbg.dassert_*` with a message instead of `assert` / `if ... raise`
    (exception: `RoutingError`, which is a client error mapped to HTTP)
  - `typing` hints (`Dict`, `Optional`), not `dict` / `X | None`
  - `os.path`, not `pathlib`
  - Optional parameters are keyword-only (after `*`) with real defaults
  - Three-line REST docstrings; `Import as:` in each module docstring with a
    short alias (e.g., `rnogaapi`)
  - `_LOG.debug(hprint.to_str("a b"))` at function entry, `_LOG.debug("return=%s", x)`
    at exit; single quotes around values in log messages, no trailing period
  - Section banners `# ####...` between classes / groups of functions

- Testing rules: `helpers_root/.claude/skills/testing.rules.md`
  - File `<component>/test/test_<module>.py`; class `Test_<function>` or `Test<Class>`; methods
    `test1`, `test2`, ... with a docstring saying what is tested
  - Sections `# Prepare inputs.` / `# Prepare outputs.` / `# Run test.` /
    `# Check outputs.`
  - Compare whole outputs with `self.assert_equal(str(actual), str(expected))`
  - No base test classes; share code through module functions
    (`gateway/test/gateway_test_utils.py`)
  - No network in tests: mock HTTP (`httpx.MockTransport`) or use `FakeProvider`

- Formatting: `black --line-length 82`, `isort --settings-path .isort.cfg` (repo
  root)
// TODO(javin): There is a linters2/lint.py that does even more complex checks
// and formatting. I will write a short intro.

## Never in This Repo

- This repository is **public**. Do not commit or paste into issues/PRs:
  - API keys or `.env` files (they're git-ignored; check `git status` anyway)
  - Business material: funding, equity, IP agreements, personal details about
    teammates, private correspondence
- Those live in the team's private backup repo / drive

## PR Description Template

- Use the template in the section "PR Description Template" of
  `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
- Noesis details for each section:
  - First line: `Fixes gpsaggese/umd_classes#<issue_number>`
  - What: name the plan task(s) the PR closes (e.g., `plan.md` T2.4)
  - Why: link the requirement (`plan.md`, MVP Execution Plan)
  - Decisions: add each one to `decisions.md` and link it

## Keeping Docs in Sync

- The rules are in the section "Keep Docs in Sync" of
  `helpers_root/docs/planning/all.plan_work.how_to_guide.md`
- Noesis files:
  - Change a module -> update its section in `design.gateway.md` (or the README of
    the component you touched)
  - Make a decision -> add it to `decisions.md`
  - Finish a task -> tick it in `plan.md` and add a line to
    `progress.md`
