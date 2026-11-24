# 00. Overview

**Status:** draft for review. No prototype code exists yet.

## Goal

Build a working end-to-end version of `workflow.drawio`: given a question about `eclipse-edc/Connector`, answer from its decision records with checked citations, warn when a record looks stale, and abstain when no rationale is recorded.

This is a prototype to show the idea works. It is not the study: one repository and draft labels give a smoke test, not evidence for RQ1–RQ4.

## Scope

**In:** 10 decision records from one repo, the offline pipeline, the online pipeline, the B0–B4 comparison, draft gold sets, a command-line interface, and a local web UI that shows what each step does to each decision (`07-ui.md`; *added during the build, at your request; a demo aid, not a deliverable*).

**Out:** other repositories, Docker and the course template layout (added after the prototype), Slack or chat sources, model training, a graph database, and documentation generation.

## Diagram mapping

Every box and edge in `workflow.drawio`. **Same** = built as drawn. **Changed** = built differently, see Deviations. **Reduced** = built with less data.

### Lane 1: offline

| Cell | Box | Status | Notes |
|---|---|---|---|
| `repo` | GitHub repos | Reduced | One repo, pinned SHA, no working tree (D9) |
| `parser` | ADR parser | Changed | Targets Decision / Rationale / Approach (D1) |
| `git` | Git history | Changed | Plain `git`, not PyDriller (D10) |
| `code` | Code tree | Changed | Read from git objects (D9) |
| `store` | Decision store | Changed | `status` always null (D2) |
| `linker` | Linker | Changed | Reference resolution, ADR-commit signal measured (D5) |
| `graph` | Decision graph | Changed | Adds a `references` edge; supersedes is partial (D3, D4) |
| `index` | Hybrid index | Same | Field chunks plus code chunks, embeddings plus BM25 |
| `stale` | Staleness scorer | Changed | Corrected window, rename-aware, null when unlinked (D6, D7) |
| `gap` | Gap ranker | Same | With 10 records most modules show as uncovered |
| `gapout` | Gap report | Same | |

### Lane 2: online

| Cell | Box | Status | Notes |
|---|---|---|---|
| `q` | Developer question | Same | |
| `router` | Query router | Same | |
| `r1` | Hybrid retrieval over ADR fields | Changed | Weights Rationale, as there is no Consequences section (D1) |
| `r2` | Code retrieval + git blame | Changed | Code to record goes through `governs` (D8) |
| `r3` | Graph lookup | Same | |
| `expand` | Graph expansion | Changed | Status filter does nothing here (D2) |
| `llm` | LLM answer | Changed | OpenAI API rather than a local model (D11) |
| `verify`, `vnote` | Citations supported? | Same | |
| `ans` | Answer | Same | |
| `abst` | Abstain | Same | |

### Lane 3: evaluation

| Cell | Box | Status | Notes |
|---|---|---|---|
| `b0` `b1` `b2` `b3` `b4` | Ablation ladder | Changed | B3 bundles more than one change (see `05-evaluation.md`) |
| `gold` | Gold sets | Reduced | 10 extraction, 10 linking, 16 QA (D12) |
| `metrics` | Metrics | Reduced | Smoke-test numbers, single annotator |

### Edges

| Cells | Status | Notes |
|---|---|---|
| `e1` `e2` `e3` `e4` `e7` `e8` `e9` `e10` `e11` `e12` `e15` | Same | |
| `e5` | Changed | "Files in ADR commit" is measured, and finds no code files that still exist (D5) |
| `e6` | Same | This is the linker's main signal |
| `e13` | Changed | Churn comes from rename-aware `git log` (D6) |
| `e14` | Same | |
| `o1` `o2` `o3` `o4` `o5` `o6` `o7` `o8` `o9` `o10` `o11` | Same | |
| `x1` `x2` `x3` | Same | |
| `be1` `be2` `be3` `be4` `m1` `m2` | Same | |

## Deviations from the diagram

| # | Deviation | Why |
|---|---|---|
| D1 | Parser targets Decision / Rationale / Approach. "Consequences weighted" becomes "Rationale weighted" | 111 of 113 records use this template, and none is MADR or Nygard |
| D2 | `status` stays null, and "accepted first" does nothing | None of the 113 records has a status. Inferring one would invent data. **So B3 only tests supersede handling** |
| D3 | `supersedes` can be partial and carries the replaced statement | 2026-09-22 replaces one sentence of 2025-08-26, so the older record must stay retrievable |
| D4 | New `references` edge between records | Records link by relative path, and multi-hop questions need it |
| D5 | Links come from resolving references to code. ADR-commit files are measured but add none | Across the 10, 9 adding commits are docs-only, and the 10th touched only `oauth2-daps` files that no longer exist |
| D6 | Churn is measured **from the record's first commit**, follows renames, and excludes shared config | See `01-corpus.md`. All three changes matter to the result |
| D7 | Unlinked records get `stale = null`, not 0 | 0 would read as "fresh" when it really means "nothing to measure" |
| D8 | Blame adds commit provenance only. Code to record goes through `governs` | Commits rarely name a record |
| D9 | Everything is read from git objects, with no checkout | About 100 paths exceed the Windows path limit, so a checkout silently loses them |
| D10 | Plain `git`, not PyDriller | Churn needs `--follow`, and this drops a dependency without losing anything |
| D11 | OpenAI API rather than local | Your choice. It departs from the course's local-only default, and the draft issue already asks GP about this. Only public repo content is sent |
| D12 | Smaller gold sets, one repo | Prototype scope |

`prototype/workflow_prototype.drawio` shows these changes. The original `workflow.drawio` stays as it is, because `Project_Plan.docx` and the draft issue describe the full study.

## Success criteria

The prototype is done when all of these hold:

1. `build` produces the decision store, graph, index, staleness scores and gap report from a fresh clone.
2. The staleness output matches `01-corpus.md`: 8 scored, 2 null, and `remove-atomikos` at 0.0.
3. `ask` answers each question type, and every citation points to something that was in the context.
4. S1 returns the current decision, T1 shows the dead-identifier warning, and U1–U3 can abstain under B4.
5. `eval` runs B0–B4 on all 16 questions and writes a results table, including where the system fails.

Accuracy targets are deliberately not set. With draft labels and 16 questions, a target would mean little. The value lies in what the comparison reveals.
