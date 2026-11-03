# Estimate Paper Quality Implementation Plan

## Goal

- Build the pipeline, one executable per stage
  ([`resolve_papers.py`](/research/Estimate_academic_paper_quality/resolve_papers.py),
  [`download_papers.py`](/research/Estimate_academic_paper_quality/download_papers.py),
  [`extract_features.py`](/research/Estimate_academic_paper_quality/extract_features.py),
  and
  [`report_papers.py`](/research/Estimate_academic_paper_quality/report_papers.py))
  and the shell script
  [`estimate_paper_quality.sh`](/research/Estimate_academic_paper_quality/estimate_paper_quality.sh)
  that runs them in order, which turn a loose list of papers into:
  - A clean reference list
  - The PDF and the markdown of each paper
  - A CSV report of the features of each paper
- Background:
  - Requirements:
    [`docs/01-prd.md`](/research/Estimate_academic_paper_quality/docs/01-prd.md)
  - Design:
    [`docs/02-design.architecture.md`](/research/Estimate_academic_paper_quality/docs/02-design.architecture.md)
    and the component docs `02.<N>-design.<component>.md` that it links
  - Decisions:
    [`docs/03-decisions.md`](/research/Estimate_academic_paper_quality/docs/03-decisions.md)

## Plan Hierarchy

- This file: what to build and in which order
- [`07-progress.md`](/research/Estimate_academic_paper_quality/docs/07-progress.md):
  what is done and learned
- [`03-decisions.md`](/research/Estimate_academic_paper_quality/docs/03-decisions.md):
  why it is built this way
- `- [ ]` is an unmerged PR and `- [x]` is a merged PR
  - A PR entry has a `Status` line once code for it exists
  - `Status: written` means the code and the tests are committed on the branch `gp`
    and the PR is not merged into `main`, so its box stays unchecked
  - A `Status` line also lists what is missing from the entry (e.g., a live run)
  - Every scheduled PR is written, except `PR_Q2`, `PR_V1`, and `PR_V2`, and no
    unscheduled PR is started: see
    [`07-progress.md`](/research/Estimate_academic_paper_quality/docs/07-progress.md)
  - The code was written in one pass from the design docs, so no `06-spec.<PR_ID>.md`
    exists for the PRs marked `Spec: needed`: the design docs and
    [`D24`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d24-bibsources-keeps-its-own-disk-cache)
    and
    [`D25`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d25-the-parser-joins-the-lines-of-a-wrapped-bullet)
    hold what the specs would
- A PR marked `Spec: needed` gets `06-spec.<PR_ID>.md` before the work starts, with
  [`.claude/skills/auto_task.create_plan_for_feature/SKILL.md`](/.claude/skills/auto_task.create_plan_for_feature/SKILL.md)
- A PR id is `PR_<component letter><number>`:

  | Letter | Component                         | Design                                                                                                                    |
  | :----- | :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
  | `H`    | Download tool in `helpers_root`   | [`02-design.architecture.md`](/research/Estimate_academic_paper_quality/docs/02-design.architecture.md#external-code)     |
  | `T`    | Shared types, `epq_types.py`      | [`02-design.architecture.md`](/research/Estimate_academic_paper_quality/docs/02-design.architecture.md#data-model)        |
  | `P`    | `pipeline`                        | [`02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md)                       |
  | `R`    | `resolver`                        | [`02.2-design.resolver.md`](/research/Estimate_academic_paper_quality/docs/02.2-design.resolver.md)                       |
  | `B`    | `bib_sources`                     | [`02.3-design.bib_sources.md`](/research/Estimate_academic_paper_quality/docs/02.3-design.bib_sources.md)                 |
  | `F`    | `fetcher`                         | [`02.4-design.fetcher.md`](/research/Estimate_academic_paper_quality/docs/02.4-design.fetcher.md)                         |
  | `E`    | `extractor`                       | [`02.5-design.extractor.md`](/research/Estimate_academic_paper_quality/docs/02.5-design.extractor.md)                     |
  | `Q`    | `rater` (Paper quality)           | [`02.6-design.rater.md`](/research/Estimate_academic_paper_quality/docs/02.6-design.rater.md)                             |
  | `O`    | `report` (output files)           | [`02.7-design.report.md`](/research/Estimate_academic_paper_quality/docs/02.7-design.report.md)                           |
  | `V`    | Evaluation of the success metrics | [`01-prd.md`](/research/Estimate_academic_paper_quality/docs/01-prd.md#success-metrics)                                   |

## Roadmap

- `v0.1`: Normalized reference list
  - `resolve_papers.py`, then `report_papers.py --action report`, turns
    [`paper_list.csv`](/research/Estimate_academic_paper_quality/paper_list.csv) into
    `references.md` and into `issues.csv` with the unresolved and ambiguous lines
  - [ ] `PR_H1`: written
  - [ ] `PR_T1`: written
  - [ ] `PR_R1`: written
  - [ ] `PR_F1`: written
  - [ ] `PR_O1`: written
  - [ ] `PR_R2`: written
  - [ ] `PR_B1`: written
  - [ ] `PR_B2`: written
  - [ ] `PR_B3`: written
  - [ ] `PR_R3`: written
  - [ ] `PR_R4`: written
  - [ ] `PR_P1`: written
- `v0.2`: Papers on disk
  - The `download` stage writes `papers/<BASE>.pdf` and `papers/<BASE>.md`, adds the
    failed papers to `issues.csv`, and a rerun skips the converted papers
  - [ ] `PR_F2`: written
  - [ ] `PR_F3`: written
  - [ ] `PR_P2`: written
- `v0.3`: Report of candidate strategies
  - The `extract` stage writes `papers/<BASE>.features.json`, and `report.csv` gives
    the features and the Paper quality rating of each converted paper
  - [ ] `PR_Q1`: written
  - [ ] `PR_Q2`: not started
  - [ ] `PR_B4`: written
  - [ ] `PR_E1`: written
  - [ ] `PR_E2`: written
  - [ ] `PR_O2`: written
  - [ ] `PR_P3`: written
  - [ ] `PR_P5`: written
  - [ ] `PR_P6`: written
- `v0.4`: Measured pipeline
  - The three success metrics of the PRD are measured on labeled sets, the default
    LLM is picked, and the numbers are in `07-progress.md`
  - [ ] `PR_V1`: not started
  - [ ] `PR_V2`: not started
- Releases are cumulative: a version also needs the PRs of the versions before it

# Component Specs

## Pipeline

### Goal

- Run one executable per stage on one input file and one output dir, and run all of
  them in order with a shell script
- Turn the failure of one paper into an `Issue`, so the run goes on
- Skip the work that a previous run did
- Design:
  [`02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md)

### Open Questions

1. Should `extract_features.py` stop the run when the total LLM cost passes a cap, e.g.,
   `--max_cost_usd`? See `PR_P4`

## Resolver

### Goal

- Turn the lines of the input file into one deduped, sorted list of papers, or into
  issues with candidates
- Design:
  [`02.2-design.resolver.md`](/research/Estimate_academic_paper_quality/docs/02.2-design.resolver.md)

### Open Questions

1. Is `_MATCH_THRESHOLD = 0.9` right? Tune it on the labeled list of `PR_V1`
2. When a preprint and its journal version have different years, which year does the
   reference show? Today the canonical record of
   [`D1`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d1-semantic-scholar-record-is-the-canonical-record)
   decides
3. Settled in `PR_R4`: a line that is a path to a `.pdf` that does not exist gives an
   `unresolved` issue with the reason `the file '<path>' does not exist`
4. Where does the per-line catch of
   [`D21`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d21-the-loop-of-each-stage-catches-errors-per-paper)
   live for `resolve`?
   - Settled in `PR_R4`: `_resolve()` of `resolve_papers.py` (`run_resolve()` before
     `PR_P5`) loops on `_resolve_line()` with the catch,
     then calls `_finalize_papers()` for the merge and the sort
   - `_resolve_lines()` stays as a loop with no catch, for tests

## Bib Sources

### Goal

- Give the other components the records, the open-access links, and the citation
  counts of the lookup APIs, with a throttle, a retry, and a disk cache
- Design:
  [`02.3-design.bib_sources.md`](/research/Estimate_academic_paper_quality/docs/02.3-design.bib_sources.md)

### Open Questions

1. What are the current rate limits of Semantic Scholar and Crossref? Check them in
   `PR_B1`, before it sets the throttles

## Fetcher

### Goal

- Own the files of each paper in `<output_dir>/papers/`, and read the PDF for the
  other components
- Design:
  [`02.4-design.fetcher.md`](/research/Estimate_academic_paper_quality/docs/02.4-design.fetcher.md)

### Open Questions

1. Should `_check_pdf()` reject a PDF whose page-1 title does not match the title of
   the reference? See `PR_F4`
2. Should the download tool skip `lint`, so that the pipeline does not need Docker?
   See `PR_H2`

## Extractor

### Goal

- Read the markdown of one paper and return the features, the facts, and the
  affiliations, each with a quote checked against the PDF text
- Design:
  [`02.5-design.extractor.md`](/research/Estimate_academic_paper_quality/docs/02.5-design.extractor.md)

### Open Questions

1. What share of the quotes stays `verified=False` on the 20 labeled papers? If it is
   high, see `PR_E3`
2. Should `report.csv` mark a value whose quote is `verified=False`? See `PR_O3`

## Rater

### Goal

- Compute the Paper quality rating of one paper from three signals, with pure code
- Design:
  [`02.6-design.rater.md`](/research/Estimate_academic_paper_quality/docs/02.6-design.rater.md)

### Open Questions

1. Who fills
   [`affiliation_tiers.yaml`](/research/Estimate_academic_paper_quality/affiliation_tiers.yaml)
   from the QS site, and which entries are an "established industry research group"
   (score 4)? It blocks `PR_Q2`

## Report

### Goal

- Write and read every file that the pipeline keeps in `<output_dir>`
- Design:
  [`02.7-design.report.md`](/research/Estimate_academic_paper_quality/docs/02.7-design.report.md)

### Open Questions

- None

## Shared Types

### Goal

- Hold the frozen dataclasses that cross a module boundary, with no logic and no
  import of the project
- Design:
  [`02-design.architecture.md`](/research/Estimate_academic_paper_quality/docs/02-design.architecture.md#data-model)

### Open Questions

- None

## Download Tool

### Goal

- Change the shared download tool only where the pipeline needs it
- Design:
  [`02-design.architecture.md`](/research/Estimate_academic_paper_quality/docs/02-design.architecture.md#external-code)

### Open Questions

1. `PR_H1` changes the names of the new files that the tool writes for every user:
   does a user or a script depend on the old names (e.g., `Jr.` as the author, `.`
   and `,` in the title)?

## Evaluation

### Goal

- Measure the three success metrics of the PRD on labeled sets, and log the numbers
  in `07-progress.md`
- The architecture doc lists the evaluation harness as not built: update it before
  `PR_V1`

### Open Questions

1. Who labels the input list for Resolution accuracy and the 20 papers for
   Extraction accuracy, and where do the labels live? It blocks `v0.4`
2. Which LLM is the default? Pick it after the Extraction accuracy test, and log the
   cost per paper

# PRs

- The PRs are in roadmap order, then the unscheduled PRs by component
- Inside a version, a PR comes after every PR that it needs, then by increasing
  complexity

## v0.1: Normalized Reference List

### `PR_H1`: [ ] Public `format_base_filename()` in `download_utils`

- Status: written: `format_base_filename()` is in the helpers repo (commit `2f36c8ed`),
  with 7 tests in `Test_format_base_filename`. The `helpers_root` pointer of this
  repo is not moved yet, and the open question of the Download Tool section is
  not answered
- Background:
  - [`download_academic_paper_to_md.py`](/helpers_root/dev_scripts_helpers/download/download_academic_paper_to_md.py)
    names its files with the private `_format_base_filename(year, authors, title)`,
    called by `_get_output_base_path()`
  - It takes the last word of the first author (`Jackson, Jr.` gives `Jr.`), and it
    keeps `.` and `,` in the title, against
    [`.claude/skills/references.rules.md`](/.claude/skills/references.rules.md)
    `# Format to Use in File Names`
  - The pipeline must give the same name to the reference and to the files, and it
    cannot import a private function
- The PR goes to the helpers repo, then the `helpers_root` submodule pointer moves
- Changes:
  - Move the function to `format_base_filename()` in
    [`download_utils.py`](/helpers_root/dev_scripts_helpers/download/download_utils.py)
  - Take the family name: the part before the comma in `Last, First`, else the last
    word that is not a suffix
  - Drop the suffixes `Jr.`, `Sr.`, `II`, `III`, `IV`
  - Replace with `_` the chars that the rules list (space, `.`, `,`, `/`, `\`) and
    `'`, then join runs of `_`
  - Call the public function from `_get_output_base_path()`
- Requirements:
  [`R9`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r9-file-base-name)
- Decisions:
  [`D7`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d7-the-pipeline-computes-the-base-name-with-a-public-helper)
- Spec: needed: it changes the names of the new files of a tool with other users
- Unit tests, in
  [`test_download_utils.py`](/helpers_root/dev_scripts_helpers/download/test/test_download_utils.py):
  - `Paul Jackson Jr.` and `Jackson, Paul` both give `Jackson`
  - One author gives no `_et_al`, two authors give `_et_al`
  - A title with `?`, `:`, `.`, `,`, and `'` gives only letters, digits, and single
    `_`
  - No year gives `UnknownYear`
  - `Test__get_output_base_path` passes, with its outcomes updated to the new names
- Result: the pipeline and the download tool give the same base name, and it follows
  the file name rules

### `PR_T1`: [ ] Shared Types of the Resolve Stage and the Import Graph Test

- Status: written: `epq_types.py` has all the types of the architecture doc, not only the
  types of the resolve stage
- Background:
  - No code exists
  - The architecture doc lists the types in "Data Model" and the allowed imports in
    "Import Graph"
- Changes:
  - Add [`epq_types.py`](/research/Estimate_academic_paper_quality/epq_types.py) with
    the frozen dataclasses of the resolve stage: `InputLine`, `BibRecord`, `UrlMeta`,
    `Paper`, `Issue`, `ResolveResult`
  - Add each later type in the PR that first needs it, e.g., `FetchResult` in
    `PR_F3`
  - Add
    [`test/test_epq_import_graph.py`](/research/Estimate_academic_paper_quality/test/test_epq_import_graph.py):
    it reads the imports of each `epq_*.py` with `ast`, and fails on an import that
    the architecture doc does not allow
- Requirements: none alone: the types carry the data of
  [`R1`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r1-input-file-format)
  to
  [`R7`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r7-reverse-chronological-order)
- Decisions:
  [`D22`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d22-flat-modules-with-the-epq_-prefix)
- Spec: needed: the fields of each type are the contract between the modules, and
  the design gives only their names
- Unit tests:
  - Each type is frozen: setting a field raises `dataclasses.FrozenInstanceError`
  - The import graph test passes on the tree
  - The import graph test fails on a module in the scratch space where `epq_types`
    imports `report_papers`
- Result: the modules of `v0.1` share one data model, and a wrong import fails the
  tests

### `PR_R1`: [ ] `parse_line()`: Classify Each Input Line

- Status: written: `parse_lines()` also joins the lines of a wrapped bullet before
  `parse_line()` ([`D25`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d25-the-parser-joins-the-lines-of-a-wrapped-bullet))
- Background:
  - `download_utils.detect_doi()` and `download_utils.get_ssrn_id()` exist
  - The download tool checks a local file, then SSRN, then a DOI, then an arXiv id,
    then a URL
- Changes:
  - Add
    [`resolve_papers.py`](/research/Estimate_academic_paper_quality/resolve_papers.py)
    (`epq_resolver.py` before `PR_P6`) with `_parse_line(line_no, text)`, which
    follows the table of kinds in
    [`02.2-design.resolver.md`](/research/Estimate_academic_paper_quality/docs/02.2-design.resolver.md#one-request-end-to-end)
  - Read the author hints of a `title` line from its trailing `(...)`
  - Return `None` for an empty line and for a comment (`#`, `//`)
- Requirements:
  [`R1`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r1-input-file-format)
- Spec: not needed: the design gives the table of kinds and keys
- Unit tests:
  - One line per kind gives the right kind and key
  - A DOI that contains `\d{4}\.\d{4,5}` is a `doi`, not an `arxiv`
  - A local PDF in the scratch space is a `local_pdf`, with its absolute path
  - `Lazy Prices (Cohen; Malloy; Nguyen)` gives the title `Lazy Prices` and three
    author hints
  - A copy of
    [`paper_list.csv`](/research/Estimate_academic_paper_quality/paper_list.csv)
    parses with no exception, and each line gets a kind
- Result: each line of the input file has a kind and a key, with no network call

### `PR_F1`: [ ] PDF Readers of `epq_fetcher`

- Status: written
- Background:
  - `resolver` needs the title of a local PDF
    ([`D4`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d4-the-title-of-a-pdf-is-the-largest-text-of-page-1))
  - `extractor` and `rater` need the page texts and the page count in `v0.3`
- Changes:
  - Add
    [`epq_fetcher.py`](/research/Estimate_academic_paper_quality/epq_fetcher.py)
    with `read_pdf_title()`, `read_pdf_pages()`, `count_pages()`, and
    `read_markdown()`, based on `pymupdf`
  - `read_pdf_title()` returns the spans of page 1 with the largest font size,
    joined; else the `title` of the PDF metadata; else `""`
- Requirements:
  [`R1`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r1-input-file-format),
  [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating)
- Decisions:
  [`D4`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d4-the-title-of-a-pdf-is-the-largest-text-of-page-1),
  [`D12`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d12-the-page-of-a-quote-comes-from-the-pdf-text)
- Spec: not needed
- Unit tests, on PDFs that the test builds in the scratch space with `pymupdf`:
  - A title in 24 pt above body text in 10 pt gives the title
  - A page with no large text and a metadata title gives the metadata title
  - An empty page with no metadata gives `""`
  - A PDF of 3 pages gives `count_pages() == 3` and 3 page texts
- Result: the resolver reads the title of a local PDF with no download

### `PR_O1`: [ ] State Files, `references.md`, and `issues.csv`

- Status: written
- Background: each stage must run alone, and read the output of the stage before it
  from `state/`
  ([`D18`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d18-all-the-state-lives-in-the-output-dir))
- Changes: add
  [`epq_report.py`](/research/Estimate_academic_paper_quality/epq_report.py) with:
  - `write_papers()` and `read_papers()`, for `state/papers.json`
  - `write_issues()` and `read_issues()`, for `state/issues.<stage>.json`
  - `write_references()`: one bullet per paper, in the order of the list
  - `write_issues_csv()`: the columns of
    [`02.7-design.report.md`](/research/Estimate_academic_paper_quality/docs/02.7-design.report.md#one-request-end-to-end),
    by stage, then by line number, with the candidates joined by `|`
- Requirements:
  [`R2`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r2-reference-format),
  [`R7`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r7-reverse-chronological-order),
  [`R19`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r19-issues-csv)
- Decisions:
  [`D18`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d18-all-the-state-lives-in-the-output-dir),
  [`D20`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d20-the-normalized-list-also-goes-to-referencesmd)
- Spec: not needed
- Unit tests:
  - A write and a read of papers and of issues give back the same objects
  - `references.md` and `issues.csv` of 2 papers and 2 issues match the expected
    text, compared with `assert_equal()`
  - An input line with a comma and a quote stays one field of `issues.csv`
- Result: the outputs of a stage survive between runs, and the user reads the
  reference list and the issues

### `PR_R2`: [ ] `match_candidates()` and `paper_key()`

- Status: written
- Background:
  - A title lookup returns up to 5 candidates per source
  - [`R5`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r5-unresolved-or-ambiguous-line)
    needs a rule to pick one candidate, and a test must be able to check the rule
  - [`R6`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r6-duplicate-lines)
    needs a key to merge the lines of one paper
- Changes:
  - Add `normalize_title()`: ASCII, lowercase, no punctuation, single spaces
  - Add `paper_key(record)`, with the DOI, the arXiv id, and the Semantic Scholar id
    as aliases
  - Add `match_candidates(query, author_hints, candidates)`, with the rule of
    [`D3`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d3-a-line-matches-when-one-distinct-paper-passes-the-title-test)
    and `_MATCH_THRESHOLD = 0.9`
  - Return a match, or an `Issue` with status `unresolved` or `ambiguous` and up to 3
    candidates
- Requirements:
  [`R5`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r5-unresolved-or-ambiguous-line),
  [`R6`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r6-duplicate-lines)
- Decisions:
  [`D2`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d2-paper-identity-is-the-normalized-title-plus-the-first-author),
  [`D3`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d3-a-line-matches-when-one-distinct-paper-passes-the-title-test)
- Spec: not needed: `D3` states the rule
- Unit tests:
  - An equal title gives `1.0`, and a partial title gives `0.95` and matches
  - A candidate with a similarity below `0.9` is dropped
  - Two papers with the same title: the author hints pick the right one
  - Two groups left give `ambiguous` with both candidates
  - Two groups where exactly one has `1.0` give a match
  - No candidate gives `unresolved`
  - A preprint and its journal record with the same title and first author give one
    key
- Result: a line resolves to exactly one paper, or to an issue that says why

### `PR_B1`: [ ] Throttled, Retried, Cached GET of `epq_bib_sources`

- Status: written: the cache is in `BibSources`, per [`D24`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d24-bibsources-keeps-its-own-disk-cache). A host that
  fails after its retries is skipped for the rest of the run. The published rate
  limits (Bib Sources open question 1) were not checked: the `429` of Semantic
  Scholar without a key is in Findings 2 of [`07-progress.md`](/research/Estimate_academic_paper_quality/docs/07-progress.md)
- Background:
  - The private clients of the download tool retry, but have no disk cache
    ([`D23`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d23-new-api-clients-instead-of-the-private-ones-of-the-download-tool))
  - A rerun must make no network call for a line already looked up
- Changes:
  - Add
    [`epq_bib_sources.py`](/research/Estimate_academic_paper_quality/epq_bib_sources.py)
    with `_get(url, params)`:
    - Cache the entry `{"status", "body", "fetched_at"}` in a JSON file of
      `<cache_dir>/http/`, keyed by the URL and the params, with the modes of
      `hcache_simple`
      ([`D24`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d24-bibsources-keeps-its-own-disk-cache))
    - Throttle each host: 1 request per second for Semantic Scholar and Crossref, 1
      request per 3 seconds for arXiv
    - Retry on `429` and `5xx`, with a backoff of 2, 4, 8, 16 s
    - Cache a `404`, and raise after the last retry with no cache entry
  - Send `x-api-key` when `SEMANTIC_SCHOLAR_API_KEY` is set, and `mailto` to
    Crossref
  - Check the current rate limits of both APIs (Bib Sources open question 1), and
    log them in `07-progress.md`
- Requirements: none alone: the cache serves the rerun of
  [`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source)
  lookups
- Decisions:
  [`D18`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d18-all-the-state-lives-in-the-output-dir),
  [`D23`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d23-new-api-clients-instead-of-the-private-ones-of-the-download-tool)
- Spec: needed: the cache key, the throttle state, and the retry policy are new, and
  all the clients share them
- Unit tests, on a local `http.server` in a thread, with the backoff set to 0:
  - Two `429`, then a `200`: returns the body after 3 requests
  - A `404` is cached: the second call sends no request
  - Five `500`: raises, and the next call sends a request again
  - Two calls to one host are at least the throttle apart, with a short throttle
- Result: each client gets one safe GET, and a rerun sends no request for a URL
  already fetched

### `PR_B2`: [ ] Semantic Scholar, Crossref, and arXiv Clients

- Status: written
- Background: `resolver` needs the DOI, the arXiv id, the record type, the family
  names, and the open-access link; the clients of the tool give only the year, the
  authors, and the title
- Changes:
  - Add `BibSources` with `lookup_s2()`, `search_s2()`, `lookup_crossref()`,
    `search_crossref()`, and `lookup_arxiv()`
  - Add `_parse_s2()`, `_parse_crossref()`, and `_parse_arxiv()`: raw JSON or Atom
    to `BibRecord`, with the family names first
  - Ask Semantic Scholar for the fields of
    [`02.3-design.bib_sources.md`](/research/Estimate_academic_paper_quality/docs/02.3-design.bib_sources.md#one-request-end-to-end)
- Requirements:
  [`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source)
- Decisions:
  [`D1`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d1-semantic-scholar-record-is-the-canonical-record),
  [`D23`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d23-new-api-clients-instead-of-the-private-ones-of-the-download-tool)
- Spec: needed: the map of each API to `BibRecord`, e.g., family names, record type,
  SSRN id from the DOI
- Unit tests:
  - A local HTTP server answers with the JSON or the Atom of each API: a lookup hit,
    a search, a not found; the parsed records are compared with `assert_equal()`,
    with no internet
- Result: the resolver looks up a paper by id or by title in the three sources of
  [`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source)

### `PR_B3`: [ ] Unpaywall Link and Metadata of a URL Line

- Status: written
- Background: a `url` line needs the DOI or the title of its page, and a paper with
  no arXiv or Semantic Scholar link can have an Unpaywall link
- Changes:
  - Add `lookup_unpaywall(doi)`: the `url_for_pdf` of the best open-access location,
    only when `--email` is set
  - Add `fetch_url_meta(url)`:
    - A PDF answer goes to `state/cache/url_pdfs/<sha1>.pdf`, and the function
      returns its path
    - An HTML answer gives the `citation_doi`, `citation_title`, and
      `citation_author` meta tags, read with `bs4`
- Requirements:
  [`R1`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r1-input-file-format),
  [`R4`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r4-download-link)
- Spec: not needed
- Unit tests:
  - Parse a saved Unpaywall answer, with and without an open-access location
  - `fetch_url_meta()` on a local `http.server` that serves an HTML page with the
    meta tags, and a PDF
- Result: a publisher URL or a direct PDF URL resolves, and a paper gets one more
  open-access link

### `PR_R3`: [ ] `build_paper()`, `merge_papers()`, and `sort_papers()`

- Status: written
- Background: a matched group of records must give one reference with facts from a
  lookup source only, a base name, and the ordered download inputs
- Changes:
  - `build_paper(group, input_line)`:
    - Pick the canonical record in the order of
      [`D1`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d1-semantic-scholar-record-is-the-canonical-record)
    - Take the year, the first author, and the title from the canonical record only
    - List the download inputs in the order of
      [`D5`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d5-each-paper-keeps-an-ordered-list-of-download-inputs),
      each one only if known
    - Format the reference `<Year>, <Last name> [et al.], "<Title>" (<Link>)`, with
      the mark of
      [`D6`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d6-a-reference-with-no-download-link-ends-with--)
      when no link is public
    - Compute the base name with `download_utils.format_base_filename()` of `PR_H1`
  - `merge_papers()`: one `Paper` per `paper_key()` or alias, with all the line
    numbers and the union of the download inputs, in order
  - `sort_papers()`: year down, then the last name of the first author, then the
    title
- Requirements:
  [`R2`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r2-reference-format),
  [`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source),
  [`R4`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r4-download-link),
  [`R6`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r6-duplicate-lines),
  [`R7`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r7-reverse-chronological-order),
  [`R9`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r9-file-base-name)
- Decisions:
  [`D1`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d1-semantic-scholar-record-is-the-canonical-record),
  [`D5`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d5-each-paper-keeps-an-ordered-list-of-download-inputs),
  [`D6`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d6-a-reference-with-no-download-link-ends-with--),
  [`D7`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d7-the-pipeline-computes-the-base-name-with-a-public-helper)
- Spec: needed: the merge by key and aliases, and the order of the merged download
  inputs, are not obvious
- Unit tests:
  - Each case of the canonical record order picks the right record
  - A reference with a public link, one with `(<DOI URL> ???)`, and one with `(???)`
  - An arXiv line and a DOI line of the same paper merge into one `Paper` with both
    line numbers
  - A merge keeps the local PDF as the first download input
  - The sort breaks ties on the year with the author, then with the title
- Result: the matched records become one clean, deduped, sorted list of papers

### `PR_R4`: [ ] `resolve_lines()`: Lookups by Kind, End to End

- Status: written: `resolve` ran on
  [`paper_list.csv`](/research/Estimate_academic_paper_quality/paper_list.csv) with
  Crossref only, since Semantic Scholar answered `429`: 20 of 22 lines resolved,
  see Findings 2 of [`07-progress.md`](/research/Estimate_academic_paper_quality/docs/07-progress.md)
- Background: `PR_R1` to `PR_R3` are pure code; no code yet connects them to
  `BibSources` and to `read_pdf_title()`
- Changes:
  - Add `_lookup_candidates(input_line, sources)`, per the table of lookups in
    [`02.2-design.resolver.md`](/research/Estimate_academic_paper_quality/docs/02.2-design.resolver.md#one-request-end-to-end)
  - An id lookup (`arxiv`, `doi`, `ssrn`) that returns a record is a match with no
    similarity test
  - Add `resolve_line(input_line, sources)`: look up, match, and build one paper, or
    return one issue
  - Add `finalize_papers(papers)` (merge and sort) and `resolve_lines(lines,
    sources)`, which loops on the lines with no catch
  - Settle Resolver open question 3 and 4
- Requirements:
  [`R1`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r1-input-file-format),
  [`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source),
  [`R5`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r5-unresolved-or-ambiguous-line)
- Decisions:
  [`D3`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d3-a-line-matches-when-one-distinct-paper-passes-the-title-test),
  [`D4`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d4-the-title-of-a-pdf-is-the-largest-text-of-page-1),
  [`D21`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d21-the-loop-of-each-stage-catches-errors-per-paper)
- Spec: needed: the flows of the `url` and `local_pdf` lines, and the catch per line
- Unit tests, with a subclass of `BibSources` that returns fixed records and makes no
  network call:
  - An input with an arXiv line, a DOI line of the same paper, a title line, an
    ambiguous title, an unknown title, and a local PDF built in the scratch space
  - The result has the merged papers in the right order, and 2 issues with their
    candidates
- Result: a list of lines becomes papers and issues in one call

### `PR_P1`: [ ] CLI with the `resolve` and `report` Stages

- Status: written: `_VALID_ACTIONS` has the four stages, since the code of `PR_P2` and
  `PR_P3` is in the same files. The run on `paper_list.csv` is the one of `PR_R4`
- Superseded by `PR_P5`: the script `estimate_paper_quality.py`, the module
  `epq_pipeline.py`, and `PipelineConfig` named below are the design of that time, and
  are gone
- Background: `PR_R4` and `PR_O1` exist, but no entry point calls them
- Changes:
  - Add `estimate_paper_quality.py` with `_parse()` and `_main()`, per
    [`.claude/skills/script.rules.md`](/.claude/skills/script.rules.md)
    - Flags: `--input`, `--output_dir`, `--email`, `--cache_mode`, the stage flags of
      `hselect_action`, and `-v`
    - `_VALID_ACTIONS = ["resolve", "report"]`: `download` and `extract` come in
      `PR_P2` and `PR_P3`
  - Add `epq_pipeline.py` with `PipelineConfig`, `run_resolve()`, and `run_report()`
  - `_main()` checks `--input`, creates `papers/` and `state/`, and sets the cache
    dir of `hcache_simple` to `state/cache/`
  - `run_report()` writes `references.md` and `issues.csv`: `report.csv` comes in
    `PR_P3`
  - Run on
    [`paper_list.csv`](/research/Estimate_academic_paper_quality/paper_list.csv), and
    log the number of papers and of issues in `07-progress.md`
- Requirements:
  [`R2`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r2-reference-format),
  [`R5`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r5-unresolved-or-ambiguous-line),
  [`R7`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r7-reverse-chronological-order),
  [`R19`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r19-issues-csv)
- Decisions:
  [`D10`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d10-the-user-gives-the-output-dir),
  [`D18`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d18-all-the-state-lives-in-the-output-dir),
  [`D21`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d21-the-loop-of-each-stage-catches-errors-per-paper)
- Spec: needed: the CLI flags, the hand-off of the state between stages, and the
  offline end-to-end test
- Unit tests:
  - `_main()` with a patched `sys.argv` and a fake client of the APIs (the network is
    the external dependency): `references.md` is the expected text
  - `--action report` alone reads `state/` of the previous run and writes the same
    files
  - A missing `--input` stops the run before any network call
- Result: the user runs one command on a loose list, and gets one clean, sorted
  reference list and the lines to fix

## v0.2: Papers on Disk

### `PR_F2`: [ ] `is_fetched()` and the File Checks

- Status: written
- `is_fetched()`: true when `<BASE>.pdf` and `<BASE>.md` exist
- `_check_pdf()`: the file starts with `%PDF`
- `_check_markdown()`: the file has at least `_MIN_MARKDOWN_CHARS = 500` chars
- Requirements:
  [`R11`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r11-rerun-skips-converted-papers)
- Decisions:
  [`D9`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d9-check-the-pdf-header-after-a-download)
- Spec: not needed

### `PR_F3`: [ ] `fetch_paper()` Through the Download Tool

- Status: written: there is no `slow` test of the real tool. A manual run on the arXiv id
  `1706.03762` saved the PDF and the markdown in about 6 s (Findings 7 of
  [`07-progress.md`](/research/Estimate_academic_paper_quality/docs/07-progress.md)). A download of an SSRN paper did not run
- Run the tool as a subprocess, one download input at a time, per
  [`02.4-design.fetcher.md`](/research/Estimate_academic_paper_quality/docs/02.4-design.fetcher.md#one-request-end-to-end)
- Return a `FetchResult` with the status and the reason of each input
- Run only the conversion when a valid PDF exists with no markdown
- Unit tests use a stub tool in the scratch space; one `slow` test runs the real tool
  on a local PDF
- Requirements:
  [`R8`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r8-pdf-and-markdown-files),
  [`R9`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r9-file-base-name),
  [`R10`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r10-failed-download-or-conversion)
- Decisions:
  [`D5`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d5-each-paper-keeps-an-ordered-list-of-download-inputs),
  [`D8`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d8-run-the-download-tool-as-a-subprocess-without-summarize),
  [`D9`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d9-check-the-pdf-header-after-a-download)
- Spec: needed

### `PR_P2`: [ ] `download` Stage

- Status: written: it ran on one paper only (Findings 7 of [`07-progress.md`](/research/Estimate_academic_paper_quality/docs/07-progress.md)), not on
  the whole list, so the numbers of PDFs, markdown files, and failures are not
  logged
- `_download()` of `download_papers.py` (`run_download()` before `PR_P5`): skip a
  fetched paper, call `fetch_paper()`, turn a failed result into an `Issue`, and write
  `state/issues.download.json`
- `issues.csv` lists the download issues
- Run on `paper_list.csv`, and log the number of PDFs, markdown files, and failures in
  `07-progress.md`
- Requirements:
  [`R8`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r8-pdf-and-markdown-files),
  [`R10`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r10-failed-download-or-conversion),
  [`R11`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r11-rerun-skips-converted-papers),
  [`R19`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r19-issues-csv)
- Decisions:
  [`D21`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d21-the-loop-of-each-stage-catches-errors-per-paper)
- Spec: not needed

## v0.3: Report of Candidate Strategies

### `PR_Q1`: [ ] Score the Signals and Rate the Paper

- Status: written: the score tables are checked against `paper.rules.md` in
  `test_report_papers.py` (`test_epq_rater.py` before `PR_P6`)
- Superseded by `PR_P6`: the code is in `report_papers.py`, and `rate_paper()` is
  `_compute_rating()`
- Add `_score_page_length()`, `_score_affiliation()`, `_score_citations()`,
  `_compute_rating()`, and `_parse_tiers()`, per
  [`02.6-design.rater.md`](/research/Estimate_academic_paper_quality/docs/02.6-design.rater.md)
- A test checks the score tables against
  [`.claude/skills/paper.rules.md`](/.claude/skills/paper.rules.md)
- Requirements:
  [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating)
- Decisions:
  [`D16`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d16-citations-come-from-semantic-scholar-only),
  [`D17`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d17-round-half-up-with-mathfloormean--05)
- Spec: not needed: the design gives the tables and the rules

### `PR_Q2`: [ ] `affiliation_tiers.yaml` from the QS Site

- Fill
  [`affiliation_tiers.yaml`](/research/Estimate_academic_paper_quality/affiliation_tiers.yaml)
  by hand, with `as_of` and `source`
- Blocked by Rater open question 1: until then `tier_5` and `tier_4` are empty, and
  an affiliation scores by its kind, e.g., a university scores 3
- Requirements:
  [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating)
- Decisions:
  [`D15`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d15-affiliation-tiers-come-from-a-checked-in-dated-list)
- Spec: not needed

### `PR_B4`: [ ] `get_citation_count()`

- Status: written
- Look up Semantic Scholar by `paperId`, else by `DOI:` or `ARXIV:`, with
  `fetched_at` from the cache entry; return `None` when the paper is missing
- Requirements:
  [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating)
- Decisions:
  [`D16`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d16-citations-come-from-semantic-scholar-only)
- Spec: not needed

### `PR_E1`: [ ] `_prepare_text()` and `_locate_quote()`

- Status: written
- Pure functions of
  [`extract_features.py`](/research/Estimate_academic_paper_quality/extract_features.py)
  (`epq_extractor.py` before `PR_P6`):
  drop the reference list and cut a long paper; find a quote in the page texts and
  give its page
- Requirements:
  [`R14`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r14-evidence-for-each-value)
- Decisions:
  [`D12`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d12-the-page-of-a-quote-comes-from-the-pdf-text),
  [`D13`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d13-a-long-paper-loses-its-references-then-its-middle)
- Spec: not needed

### `PR_E2`: [ ] `_extract_features()` with One LLM Call

- Status: written: it ran only with a fake `llm_func`, not on a live LLM: no key, and the
  venv has no `openai`
- Add `_LlmExtraction`, the prompt
  [`extract_features.prompt.md`](/research/Estimate_academic_paper_quality/extract_features.prompt.md)
  (`epq_extractor.prompt.md` before `PR_P6`), and `_extract_features()` with an
  injected `llm_func`
- Unit tests use a fake `llm_func` that returns a fixed `_LlmExtraction`
- Requirements:
  [`R12`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r12-feature-list),
  [`R13`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r13-unknown-value),
  [`R14`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r14-evidence-for-each-value),
  [`R15`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r15-extracted-facts)
- Decisions:
  [`D11`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d11-one-structured-llm-call-per-paper)
- Spec: needed: the prompt and the conversion of the answer into `epq_types`

### `PR_O2`: [ ] Features File and `report.csv`

- Status: written
- Add `write_features_file()` and `read_features_file()` with `schema_version`, then
  `build_report_rows()` and `write_report()`
- Requirements:
  [`R17`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r17-rerun-skips-extraction),
  [`R18`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r18-report-csv),
  [`R20`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r20-features-file)
- Decisions:
  [`D14`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d14-the-features-file-is-json-and-is-the-rerun-cache-of-extract),
  [`D19`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d19-reportcsv-has-a-row-only-for-a-paper-with-a-features-file)
- Spec: needed: the JSON schema is the rerun cache, so it must stay stable

### `PR_P3`: [ ] `extract` Stage and `report.csv`

- Status: written: it did not run on `paper_list.csv`, so the LLM cost per paper is not
  logged
- `_extract()` of `extract_features.py` and the actions `rate` and `report` of
  `report_papers.py` (`run_extract()` and `run_report()` before `PR_P5`), per
  [`02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md#one-request-end-to-end),
  with the flags `--model`, `--force_extract`, and `--max_chars`
- The action `report` also writes `report.csv`
- Run on `paper_list.csv`, and log the LLM cost per paper in `07-progress.md`
- Requirements:
  [`R12`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r12-feature-list),
  [`R13`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r13-unknown-value),
  [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating),
  [`R17`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r17-rerun-skips-extraction),
  [`R18`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r18-report-csv),
  [`R19`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r19-issues-csv)
- Decisions:
  [`D21`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d21-the-loop-of-each-stage-catches-errors-per-paper)
- Spec: needed: the inputs of `rater`, the refresh of the citations, and the cost log

### `PR_P5`: [ ] One Executable per Stage and a Shell Script

- Status: written: the 196 tests of the project pass; the four scripts run `--help`
  through their `uv run` headers, and the shell script ran on stubs of the four
  scripts. No live run of the full chain yet
- Replace `estimate_paper_quality.py` and `epq_pipeline.py` with
  `download_papers.py`, `extract_features.py`, and `report_papers.py`, next to the
  existing `resolve_papers.py`; each script has its own `uv run` header, `--dry_run`,
  and the loop of its stage
- Add `epq_paths.py` (names of the shared files) and `epq_issues.py` (issue of a
  failed paper)
- Split the rating from the extraction: the features file holds `affiliations` and
  `n_pages` and no `quality`, the action `rate` writes `papers/<BASE>.rating.json`,
  and the schema version is 2
- Add the issue stage `rate` (statuses `rating_failed` and `rating_missing`) and
  `--refresh_citations`
- Add `estimate_paper_quality.sh`: runs the four executables in order, with `-i`, `-o`,
  `-m`, `-v`, `-f`, and `-r`
- Update the import graph test: the scripts import `epq_*` modules, never each other
- Split `test_epq_pipeline.py` into one test file per script, plus
  `test_epq_end_to_end.py` that runs the four stages in order and a rerun that costs
  nothing
- Requirements:
  [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating),
  [`R17`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r17-rerun-skips-extraction),
  [`R20`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r20-features-file)
- Decisions:
  [`D26`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d26-the-resolver-is-a-separate-executable-that-reads-and-writes-csv),
  [`D27`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d27-the-pipeline-is-split-into-one-executable-per-stage)
- Spec: not needed: the design is in
  [`02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md)
- Unit tests:
  - Each script has tests of its stage function and of its `_main()`
  - A new tier list changes the rating and leaves the features file unchanged
  - A stage that fails for a paper writes an issue and the next paper is done
  - `--dry_run` writes no file and makes no LLM call, no download, and no lookup
- Result: a stage runs alone on the files of an earlier run, and a new tier list needs
  no new LLM call

### `PR_P6`: [ ] Inline the Modules With One Caller

- Status: written: the 196 tests of the project pass
- Move the code of `epq_resolver.py`, `epq_extractor.py`, and `epq_rater.py` into
  `resolve_papers.py`, `extract_features.py`, and `report_papers.py`, and make the
  functions private
- Rename `epq_extractor.prompt.md` to `extract_features.prompt.md`
- Rename `rate_paper()` to `_compute_rating()`, since `report_papers.py` has
  `_rate_paper()`
- Move the tests: `test_epq_resolver.py`, `test_epq_extractor.py`, and
  `test_epq_rater.py` into `test_resolve_papers.py`, `test_extract_features.py`, and
  `test_report_papers.py`
- Update the test of the import graph, the README, and the design docs
- Decisions:
  [`D28`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d28-a-module-with-one-caller-lives-in-that-script)
- Spec: not needed: the code moves and the behavior stays
- Unit tests: the moved tests are the same, with the new names
- Result: a stage is one file, plus the modules that two files share

## v0.4: Measured Pipeline

### `PR_V1`: [ ] Resolution Accuracy and Download Coverage

- Label an input list: the first author, the year, and the title of each line, and
  whether a public PDF exists
- Compare a run with the labels, tune `_MATCH_THRESHOLD`, and log both metrics in
  `07-progress.md`
- Blocked by Evaluation open question 1
- Spec: needed

### `PR_V2`: [ ] Extraction Accuracy and the Default LLM

- Label the Sharpe ratios of 20 papers, as a researcher reads them
- Compare the extraction of a few models with the labels, pick the default model, and
  log the accuracy, the cost per paper, and the share of verified quotes
- Blocked by Evaluation open questions 1 and 2
- Spec: needed

## Unscheduled: Pipeline

### `PR_P4`: [ ] Cap on the LLM Cost

- Add `--max_cost_usd`: `_extract()` of `extract_features.py` stops when the total
  cost passes it
- Answers Pipeline open question 1

## Unscheduled: Fetcher

### `PR_F4`: [ ] Reject a PDF of the Wrong Paper

- `_check_pdf()` compares the title of page 1 with the title of the reference
- Answers Fetcher open question 1

## Unscheduled: Extractor

### `PR_E3`: [ ] Fuzzy Match in `_locate_quote()`

- Accept a quote that the PDF text breaks in an odd way, e.g., two columns
- Do it only if `PR_V2` shows a high share of unverified quotes

## Unscheduled: Report

### `PR_O3`: [ ] Mark an Unverified Value in `report.csv`

- Answers Extractor open question 2

## Unscheduled: Download Tool

### `PR_H2`: [ ] Conversion Without Docker

- The download tool passes `--skip_action lint` to the converter
- Answers Fetcher open question 2

### `PR_H3`: [ ] Content Type Check in the Download Tool

- The tool rejects a `200` answer that is not a PDF, for all its users
- The alternative of
  [`D9`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d9-check-the-pdf-header-after-a-download)

## Conventions

- Coding rules: [`.claude/skills/coding.rules.md`](/.claude/skills/coding.rules.md)
- Testing rules: [`.claude/skills/testing.rules.md`](/.claude/skills/testing.rules.md)
- Script rules: [`.claude/skills/script.rules.md`](/.claude/skills/script.rules.md)
- Layers and imports:
  [`.claude/skills/architecture.rules.md`](/.claude/skills/architecture.rules.md)
- Project docs: [`.claude/skills/project.rules.md`](/.claude/skills/project.rules.md)
