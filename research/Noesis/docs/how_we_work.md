# Noesis — How We Work

- The process for every change to `research/Noesis/`, so GP can review quickly and
  anyone can see why the code is the way it is
- Contributions use the **fork and PR** method described in
  `class_project/how_to_contribute.md`; this doc adds the Noesis-specific parts

## The Loop

1. **Pick a task** from `mvp_plan.md` §3 (or propose one)
2. **Plan before code**: write a short design (what, files, interfaces, tests,
   open choices) and get a teammate's OK. Non-trivial designs go in the issue
3. **Open a GitHub issue** in `gpsaggese/gpsaggese.github.io`, title
   `Noesis: <what>`; note the number, e.g., `#612`
4. **Sync your fork** with upstream, then **branch** in your fork from `master`,
   named after the issue: `UmdTask612_Noesis_<short_title>`

   ```bash
   > git fetch upstream && git checkout master && git merge upstream/master
   > git checkout -b UmdTask612_Noesis_<short_title>
   ```

5. **Code + tests** following the conventions below; run the Noesis tests locally
6. **Commit** specific files (never `git add .`) with the issue reference, then
   push to your fork:

   ```bash
   > git add research/Noesis/<file> ...
   > git commit -m "<message> (gpsaggese/umd_classes#612)"
   > git push origin UmdTask612_Noesis_<short_title>
   ```

7. **Open a PR** from your fork's branch to `gpsaggese/gpsaggese.github.io`
   `master`, titled like the branch, using the PR description template below. Put
   `Fixes gpsaggese/umd_classes#612` at the start of a line so the issue closes on
   merge. Add GP under **Reviewers** (not Assignees)
8. **Review**: a teammate first, then GP. Address comments in new commits on the
   same branch
9. **Merge** (GP), then sync your fork and mirror to the team backup repo

- `gpsaggese/umd_classes` is the repo's former name; GitHub redirects it, and it's
  the prefix `how_to_contribute.md` uses for issue references

- Keep PRs reviewable: one component or one milestone task per PR. If a PR needs
  more than ~400 lines of non-test code, split it

## Conventions (GP's Rules)

- Coding rules: `helpers_root/.claude/skills/coding.rules.md`; testing rules:
  `helpers_root/.claude/skills/testing.rules.md`. The ones that matter most here:
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
- Tests:
  - File `test/test_<module>.py`; class `Test_<function>` or `Test<Class>`; methods
    `test1`, `test2`, ... with a docstring saying what is tested
  - Sections `# Prepare inputs.` / `# Prepare outputs.` / `# Run test.` /
    `# Check outputs.`
  - Compare whole outputs with `self.assert_equal(str(actual), str(expected))`
  - No base test classes; share code through module functions
    (`test/gateway_test_utils.py`)
  - No network in tests: mock HTTP (`httpx.MockTransport`) or use `FakeProvider`
- Formatting: `black --line-length 82`, `isort --settings-path .isort.cfg` (repo
  root)

## Never in This Repo

- This repository is **public**. Do not commit or paste into issues/PRs:
  - API keys or `.env` files (they're git-ignored; check `git status` anyway)
  - Business material: funding, equity, IP agreements, personal details about
    teammates, private correspondence
- Those live in the team's private backup repo / drive

## PR Description Template

Copy this into every PR:

```markdown
Fixes gpsaggese/umd_classes#<issue_number>

## What
- One paragraph: what this PR adds or changes, and which plan task(s) it closes
  (e.g., `mvp_plan.md` T2.4)

## Why
- The problem it solves; link the PRD section

## How It Works
- Main flow in a few bullets; point to the README/doc section updated

## Decisions
- Each non-obvious choice: decision, why, alternatives considered. Add it to
  `docs/mvp_decisions.md` and link it here

## Evidence
- Tests added (count, what they cover) and the command to run them
- Any live check (what was called, result, cost)

## Not in This PR
- Deliberately deferred items, and where they're tracked

## Questions for Reviewers
- Anything you want GP's opinion on
```

## Keeping Docs in Sync

- Change a module -> update its section in `gateway.README.md` (or the README of
  the component you touched)
- Make a decision -> add it to `mvp_decisions.md`
- Finish a task -> tick it in `mvp_plan.md` and add a line to `mvp_progress.md`
