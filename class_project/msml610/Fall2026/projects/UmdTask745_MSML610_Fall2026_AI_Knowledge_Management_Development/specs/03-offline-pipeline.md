# 03. Offline pipeline

**Status:** draft for review. Lane 1 of `workflow.drawio`. Run once, by `python -m adrqa build`. Each step reads from git objects at the pinned SHA and writes JSON to `prototype/data/` (schemas in `02-data-model.md`).

> **Update (relation extractor).** Replacement detection now has a second source. `relations.py` can ask a chat model which records each record replaces (`supersedes` or `superseded by`, partial or full). A claim is kept only if its target is a record in the corpus and its evidence quote appears word for word in the record. Its edges are merged with the parser's rule-based ones, so the model cannot remove a rule edge, and a partial reading wins when the two disagree. Each edge in `records.json` then carries a `source` (`rule`, `model` or `rule+model`) and an `evidence` quote. It runs only when `OPENAI_API_KEY` and `OPENAI_CHAT_MODEL` are set. Without them the rule-based parser described below is the only source, and the "There is no LLM extraction" statement in section 1 holds. This section was written before this change, and `prototype/data/` above is now `data/` in this repository.

## 1. Parser (`parser`, `store`)

**Input:** the 10 `README.md` files. **Output:** `records.json`.

- Parse with `markdown-it-py`. `#` gives the title. `##` sections named Decision, Rationale and Approach give the fields, case-insensitively. Everything up to the next `##` belongs to that field, including `###` subsections.
- `date` comes from the folder name.
- `status` is taken only from an explicit status line. On this corpus it is always null, and it is never inferred.
- A section that cannot be found is recorded in `missing_fields`. There is no LLM extraction.
- `references`: relative links matching `../<record-id>/README.md`, keeping only IDs in the corpus.
- `supersedes`: a sentence with `supersed(es|ed)` that also links to a record. It is `partial` when the sentence quotes a statement, and that quote goes in `statement`. Otherwise it is `full`.
- **Known miss:** implicit replacement (2025-10-29 removes what 2025-07-25 added, without saying "supersede"). Not attempted; QA item S2 measures the effect.

## 2. Linker (`linker`, `e5`, `e6`)

**Input:** records and the tree at the SHA. **Output:** `links.json`. The rules are implemented in `specs/tools/measure_corpus.py`, and the build reuses them.

| Reference in the record | Rule | Signal |
|---|---|---|
| CamelCase word, in backticks or bare | A `.java` file with that exact name. Prefer non-test files, and use test files only if none exist | `explicit_class` |
| kebab-case word in backticks | A directory with that exact name. Links all its non-test `.java`, `.kts` and `.sql` files | `explicit_module` |
| Text with `/` in backticks | That exact path, file or directory, outside `docs/` | `explicit_path` |
| Filename in backticks (`.java`, `.kts`, `.sql`, `.toml`, `.yml`, `.yaml`, `.properties`) | A unique match in the tree | `explicit_file` |
| Commit that added the record | Files in that commit, excluding docs, that still exist at the SHA | `adr_commit` |

**Known weaknesses, kept on purpose so that evaluation can show them:**

- **It has no notion of negation.** "`dsp-2025` remain unchanged" still links that module's 27 files, and "the SPIs are not affected" still links them. Precision is low on these (0.35 on 2026-05-11).
- **Mentions count the same as governance.** `ContractDefinition` in 2025-05-28 is only an aside.
- **SQL tables are not resolved.** `edc_asset` does not map to its schema file, so recall suffers on 2026-09-22.

## 3. Decision graph (`graph`, `e8`, `e9`)

**Output:** `graph.json`, built with NetworkX.

- Nodes: 10 records, plus every linked file.
- Edges: `governs` (record to file), `supersedes` (newer to older, with scope and statement), `references` (record to record).
- Expected from the gold labels: 1 `supersedes` edge and 7 `references` edges.

## 4. Hybrid index (`index`, `e10`, `e11`)

**Output:** a Chroma collection and a BM25 index.

- **Record chunks:** one chunk per field, split further at `###` headings when over 400 tokens. The field name is kept as metadata. Config B1 instead uses whole-record fixed windows.
- **Code chunks:** 1,751 non-test Java files, one chunk per top-level type or method where possible, otherwise 300-token windows with 50 overlap.
- **README:** the root `README.md`, in one chunk per section.
- **Embeddings:** `text-embedding-3-large` (your choice, changed from `3-small`), batched and cached by content hash. About 1.3M tokens in total.
- **BM25:** `rank_bm25`. The tokenizer splits CamelCase and keeps the full token, so `AssetIndex` matches both as a whole and as `asset` and `index`.
- If `chromadb` does not install on Python 3.13, swap in NumPy cosine similarity. At this size, the results are the same.

## 5. Staleness scorer (`stale`, `e12`, `e13`, `x3`)

**Output:** `staleness.json`. Rules and expected values are in `01-corpus.md`.

```
churn_code = distinct commits in (first_commit, SHA] touching any linked file except shared config,
             following renames per file
stale      = churn_code / (1 + edits_after_first)
stale      = null with reason no_linked_files, if the record has no linked files
```

Dead identifier: the token existed in history (`git log -- ':(glob)**/<token>.java'` or the module path), and `git grep -w` finds it nowhere outside `docs/` at the SHA.

Warnings at answer time (config `STALE_WARN_THRESHOLD`, default 15, unvalidated):

| Condition | Warning |
|---|---|
| `stale` ≥ threshold | `stale_score`, with the number of commits |
| Any dead identifier | `dead_identifier`, naming it |
| `stale` is null | `no_linked_code`, not a score |

## 6. Gap ranker (`gap`, `gapout`, `e14`, `e15`)

**Output:** `gap_report.json`, the top 20 modules.

- **Module:** the directory above `src/main/`, 166 in total.
- **Complexity:** the sum of cyclomatic complexity across the module's non-test Java, using `lizard.analyze_file.analyze_source_code` on text from git objects.
- **Coverage:** share of the module's Java files with a `governs` edge.
- **Gap:** `complexity × (1 − coverage)`.

**Limit:** with 10 of 113 records indexed, most modules will show as uncovered. So the prototype demonstrates the mechanism, and its rankings say little about this repo's real documentation.

## Checks

Unit tests with small fixtures: parser on all 10 records (compared with `gold/extraction.json`), the linker's token rules, and staleness including the null case. An integration test compares the build output with `corpus_measurements.json`.
