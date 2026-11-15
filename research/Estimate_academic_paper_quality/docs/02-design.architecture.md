# Design: Architecture

## Goal

- Turn a loose list of papers into a JSON report of candidate trading strategies, with
  the full text and a set of extracted features per paper
- Where it fits: text file of papers from the user -> this pipeline -> JSON report
  that the quant researcher sorts to pick the next papers to replicate
- Requirements:
  [`research/Estimate_academic_paper_quality/docs/01-prd.md`](/research/Estimate_academic_paper_quality/docs/01-prd.md)
- Decisions and alternatives, cited as `D<N>`:
  [`research/Estimate_academic_paper_quality/docs/03-decisions.md`](/research/Estimate_academic_paper_quality/docs/03-decisions.md)
- Status: implemented, see
  [`07-progress.md`](/research/Estimate_academic_paper_quality/docs/07-progress.md)
  for the status of the PRs

## Diagram

### C4 Refresher

- C4 has four levels: each one zooms into one box of the level above
  - C1 (Context): the system as one box, with its users and the external systems
    around it
    - Answers: who uses the system, and what does it depend on?
  - C2 (Container): the parts of the system that run or store data on their own
    (e.g., app, service, script, DB), and how they call each other
    - Answers: what is the system made of, and with which technology?
  - C3 (Component): the parts inside one container (e.g., modules, classes)
    - Answers: how is one container organized?
  - C4 (Code): the classes and functions inside one component
    - Not drawn: the code is the diagram
- Elements: `Person`, `System`, `Container`, `Component`
  - `_Ext` marks what the team does not build (e.g., `System_Ext`)
  - `Rel` is a labeled arrow: a call, an import, or a data flow
  - `Boundary` groups elements (e.g., a layer)
- This doc draws C1 and C2
  - C3 is in the doc of each component, `02.<N>-design.<component>.md`

### C1 (Context)

- The quant researcher runs four CLI scripts, one per stage, or the script
  `estimate_paper_quality.py` that runs them in order
  ([`D27`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d27-the-pipeline-is-split-into-one-executable-per-stage))
  - `resolve_papers.py` turns a CSV of papers into a papers file (JSON)
  - `download_papers.py` saves the PDF and the markdown of each paper
  - `extract_features.py` extracts the features of each paper with the LLM
  - `report_papers.py` rates the papers and writes the report
- The pipeline reads bibliographic facts and citation counts only from the lookup
  APIs, never from an LLM (PRD
  [`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source),
  [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating))
- The LLM reads the markdown of each paper and returns the features, the facts, and
  the affiliations, each with a quote of the paper
- Chrome downloads the SSRN PDFs, since SSRN blocks scripted clients

```plantuml
!pragma layout smetana
!include <C4/C4_Context>
LAYOUT_LEFT_RIGHT()
title Estimate Paper Quality: Context
Person(user, "Quant researcher", "Gives a list of papers, reads the JSON report")
System(epq, "Estimate Paper Quality pipeline", "Resolves, downloads, converts, extracts, rates, reports")
System_Ext(bib, "Bibliographic APIs", "Semantic Scholar, Crossref, arXiv, Unpaywall")
System_Ext(sites, "Paper hosts", "arXiv, SSRN, publisher sites, open-access mirrors")
System_Ext(chrome, "Chrome", "Started and closed by download_with_chrome.py for SSRN")
System_Ext(llm, "LLM provider", "Any provider of the llm library, OpenRouter by default, through helpers.hllm_cli")
Rel(user, epq, "Runs estimate_paper_quality.py, or the four scripts one by one", "CLI")
Rel(epq, bib, "Looks up metadata, links, citations", "HTTPS JSON")
Rel(epq, sites, "Downloads PDFs", "HTTPS")
Rel(epq, chrome, "Downloads SSRN PDFs", "CDP")
Rel(epq, llm, "Extracts features with quotes", "HTTPS JSON")
```

### C2 (Container)

- Four scripts, a script that chains them, three infrastructure modules, three
  shared leaf modules, and the existing download tool
- Arrows are imports or calls: each one points down a layer, never up
- The scripts do not import each other: they talk only through files in the
  `output_dir` ([`D26`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d26-the-resolver-is-a-separate-executable-that-reads-and-writes-csv),
  [`D27`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d27-the-pipeline-is-split-into-one-executable-per-stage))
- Every script also uses `epq_paths`, `epq_issues`, and `epq_summary`, which the diagram leaves out

```plantuml
!pragma layout smetana
!include <C4/C4_Container>
LAYOUT_TOP_DOWN()
title Estimate Paper Quality: Containers
Person(user, "Quant researcher")

Container_Boundary(script, "Script and orchestration layer") {
  Container(sh, "estimate_paper_quality.py", "", "Runs the four scripts in order, as subprocesses")
  Container(rcli, "resolve_papers.py", "", "Input CSV to papers file and issues file (JSON): parses lines, matches papers, dedupes, builds references")
  Container(dcli, "download_papers.py", "", "PDF and markdown of each paper")
  Container(ecli, "extract_features.py", "", "Features file of each paper: one LLM call, quotes checked against the PDF")
  Container(pcli, "report_papers.py", "", "Stages rate and report, selected with hselect_action: Paper quality rating from three signals")
}

Container_Boundary(infra, "Infrastructure layer") {
  Container(bib, "epq_bib_sources.py", "", "HTTP clients with throttle, retry, disk cache")
  Container(fetcher, "epq_fetcher.py", "", "Runs the download tool, checks and reads the PDF")
  Container(report, "epq_report.py", "", "Writes and reads all the output files")
}

Container_Boundary(ext, "Shared leaf modules, existing tool, outputs") {
  Container_Ext(tool, "download_academic_paper_to_md.py", "", "Downloads a PDF and converts it to markdown")
  ContainerDb(out, "output_dir", "Files", "NN-<stage>.* files, paper_*/ dirs, state/")
  Container(types, "epq_types.py", "", "Dataclasses shared by all modules")
  Container(paths, "epq_paths.py", "", "Paths of the files that the scripts share")
  Container(issues, "epq_issues.py", "", "Builds the issue of a failed paper")
}

Rel(user, sh, "~--input, ~--output_dir")
Rel(sh, rcli, "runs")
Rel(sh, dcli, "runs")
Rel(sh, ecli, "runs")
Rel(sh, pcli, "runs")
Rel(sh, paths, "get_report_path()")
Rel(rcli, bib, "BibSources, lookup_*(), search_title()")
Rel(rcli, fetcher, "read_pdf_title()")
Rel(rcli, report, "read_input_csv(), write_papers(), write_issues()")
Rel(dcli, fetcher, "is_fetched(), fetch_paper()")
Rel(dcli, report, "read_papers(), write_issues()")
Rel(ecli, fetcher, "read_markdown(), read_pdf_pages(), count_pages()")
Rel(ecli, report, "read_papers(), write_features_file(), write_issues()")
Rel(pcli, bib, "get_citation_count()")
Rel(pcli, report, "read_features_file(), write_rating_file(), write_*()")
Rel(fetcher, tool, "subprocess")
Rel(report, out, "JSON, markdown")
```

## Files

- The component `pipeline` is the four scripts and the script that chains them: there is no
  module between the scripts and the other components, since each script holds the
  loop of its own stage
  ([`02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md))
- The components `resolver`, `extractor`, and `rater` are not files: their code
  lives in the script of their stage, since each had one caller
  ([`D28`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d28-a-module-with-one-caller-lives-in-that-script))
- All code lives flat in
  [`research/Estimate_academic_paper_quality/`](/research/Estimate_academic_paper_quality/),
  tests in
  [`research/Estimate_academic_paper_quality/test/`](/research/Estimate_academic_paper_quality/test/)
  ([`D22`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d22-flat-modules-with-the-epq_-prefix))

  | File                                                                                                 | Component               | Layer                   |
  | :--------------------------------------------------------------------------------------------------- | :---------------------- | :---------------------- |
  | [`resolve_papers.py`](/research/Estimate_academic_paper_quality/resolve_papers.py)                   | `pipeline`, `resolver`  | Script, business logic  |
  | [`download_papers.py`](/research/Estimate_academic_paper_quality/download_papers.py)                 | `pipeline`              | Script, orchestration   |
  | [`extract_features.py`](/research/Estimate_academic_paper_quality/extract_features.py)               | `pipeline`, `extractor` | Script, business logic  |
  | [`extract_features.prompt.md`](/research/Estimate_academic_paper_quality/extract_features.prompt.md) | `extractor`             | Data: the system prompt |
  | [`report_papers.py`](/research/Estimate_academic_paper_quality/report_papers.py)                     | `pipeline`, `rater`     | Script, business logic  |
  | [`estimate_paper_quality.py`](/research/Estimate_academic_paper_quality/estimate_paper_quality.py)   | `pipeline`              | Script, orchestration   |
  | [`affiliation_tiers.yaml`](/research/Estimate_academic_paper_quality/affiliation_tiers.yaml)         | `rater`                 | Data: dated tier list   |
  | [`epq_bib_sources.py`](/research/Estimate_academic_paper_quality/epq_bib_sources.py)                 | `bib_sources`           | Infrastructure          |
  | [`epq_fetcher.py`](/research/Estimate_academic_paper_quality/epq_fetcher.py)                         | `fetcher`               | Infrastructure          |
  | [`epq_report.py`](/research/Estimate_academic_paper_quality/epq_report.py)                           | `report`                | Infrastructure          |
  | [`epq_types.py`](/research/Estimate_academic_paper_quality/epq_types.py)                             | shared                  | Data model, no logic    |
  | [`epq_paths.py`](/research/Estimate_academic_paper_quality/epq_paths.py)                             | shared                  | Names of the files      |
  | [`epq_issues.py`](/research/Estimate_academic_paper_quality/epq_issues.py)                           | shared                  | Issue of a failed paper |
  | [`epq_summary.py`](/research/Estimate_academic_paper_quality/epq_summary.py)                         | shared                  | Summary of a stage      |

## Import Graph

- Allowed imports, checked by a test of the import graph:
  - A script imports only `epq_*` modules, never another script
    - [`resolve_papers.py`](/research/Estimate_academic_paper_quality/resolve_papers.py):
      `epq_bib_sources`, `epq_fetcher`, `epq_report`, `epq_issues`, `epq_summary`,
      `epq_paths`, `epq_types`
    - [`download_papers.py`](/research/Estimate_academic_paper_quality/download_papers.py):
      `epq_fetcher`, `epq_report`, `epq_issues`, `epq_summary`, `epq_paths`, `epq_types`
    - [`extract_features.py`](/research/Estimate_academic_paper_quality/extract_features.py):
      `epq_fetcher`, `epq_report`, `epq_issues`, `epq_summary`, `epq_paths`, `epq_types`
    - [`report_papers.py`](/research/Estimate_academic_paper_quality/report_papers.py):
      `epq_bib_sources`, `epq_report`, `epq_issues`, `epq_summary`, `epq_paths`,
      `epq_types`
    - [`estimate_paper_quality.py`](/research/Estimate_academic_paper_quality/estimate_paper_quality.py):
      `epq_paths`, `epq_summary` (for `format_elapsed()`). It runs the four scripts as
      subprocesses and imports none of them
  - `epq_report`, `epq_issues` import only `epq_types`
  - `epq_summary` imports only `epq_paths` and `epq_types`
  - `epq_bib_sources`, `epq_fetcher` import only `epq_types` and `helpers`
  - `epq_types` and `epq_paths` import nothing of the project

## External Code

- [`helpers_root/dev_scripts_helpers/download/download_academic_paper_to_md.py`](/helpers_root/dev_scripts_helpers/download/download_academic_paper_to_md.py)
  downloads from arXiv, DOI, SSRN (through Chrome), a URL, or a local PDF, and
  converts the PDF to markdown with
  [`convert_pdf_to_md.py`](/helpers_root/dev_scripts_helpers/documentation/convert_pdf_to_md.py)
- [`helpers_root/dev_scripts_helpers/download/download_utils.py`](/helpers_root/dev_scripts_helpers/download/download_utils.py)
  has `detect_doi()` and `get_ssrn_id()`
- [`helpers_root/helpers/hselect_action.py`](/helpers_root/helpers/hselect_action.py)
  selects the stages
- [`helpers_root/helpers/hllm_cli.py`](/helpers_root/helpers/hllm_cli.py) has
  `apply_llm()`, which calls an LLM through the `llm` library and returns the text
  of the answer with the `TokenStats` of the call (tokens and cost), and
  `DEFAULT_MODEL`. The old `helpers.hllm` is obsolete and the pipeline does not use
  it
- [`helpers_root/helpers/hcache_simple.py`](/helpers_root/helpers/hcache_simple.py)
  has the `simple_cache()` decorator and the `--cache_mode` switch
  `add_cache_control_arg()`
- `download_utils.format_base_filename()` is the public name of the former private
  `_format_base_filename()` of the download tool, and `resolve_papers.py` calls it to
  compute the base name of a paper
  ([`D7`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d7-the-pipeline-computes-the-base-name-with-a-public-helper))

## Data Model

- [`epq_types.py`](/research/Estimate_academic_paper_quality/epq_types.py) holds the
  dataclasses that cross a module boundary: all are frozen
  - `InputLine`: line number, text, kind, key, author hints (`resolver`)
  - `BibRecord`: one record of one lookup source (`bib_sources`)
  - `Paper`: a resolved, deduped paper with its reference, base name, and download
    inputs (`resolver`)
  - `Issue`: an unresolved line or a failed paper, with stage, status, and reason
  - `Evidence`, `FeatureValue`, `LabeledFact`, `ExtractedFacts`: the output of
    `extractor`
  - `SignalScore`, `QualityRating`: the output of `rater`, written to the rating file
  - `PaperFeatures`: everything that goes into one features file: the features, the
    facts, the affiliations, the page count, and the provenance, but not the rating
    ([`D27`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d27-the-pipeline-is-split-into-one-executable-per-stage))
- Shape of the core types:

  ```python
  @dataclasses.dataclass(frozen=True)
  class Evidence:
      # Text of the paper, or the value given by an external source.
      quote: str
      # E.g., "Sec. 4.2, p. 17", "Table 3, p. 21", "Semantic Scholar".
      location: str
      # True if the quote was found in the text of the PDF.
      verified: bool

  @dataclasses.dataclass(frozen=True)
  class FeatureValue:
      # Free text as the paper states it, or "unknown".
      value: str
      evidence: Optional[Evidence]
  ```

## One Request End to End

- The user runs:

  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10 \
      --email me@umd.edu
  > research/Estimate_academic_paper_quality/download_papers.py \
      --output_dir ~/papers/run_2026_10
  > research/Estimate_academic_paper_quality/extract_features.py \
      --output_dir ~/papers/run_2026_10
  > research/Estimate_academic_paper_quality/report_papers.py \
      --output_dir ~/papers/run_2026_10
  ```

  - The three last commands read `01-resolve.paper_list.json` in the output dir, since
    `--input` defaults to it

- The script `estimate_paper_quality.py` runs the same four commands in order, and
  selects the stages with `--action`, `--skip_action`, and `--only_action`:

  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10
  ```

- `resolve_papers.py`: copies the input to `00-paper_list.csv`, then `resolver` parses
  22 rows, looks up each one, writes 20 `Paper` to `01-resolve.paper_list.json` and 2
  `Issue` to `01-resolve.issues.json`
  ([`D26`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d26-the-resolver-is-a-separate-executable-that-reads-and-writes-csv),
  superseded by
  [`D31`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d31-json-for-the-intermediate-files))
- `download_papers.py`: `fetcher` runs the download tool for each paper with no PDF
  and no markdown yet, and records the failures
- `extract_features.py`: for each converted paper with no features file yet,
  `extractor` calls the LLM once, and `report` writes
  `<output_dir>/paper_features/<BASE>.features.json` with the affiliations and the page count
- `report_papers.py`, stage `rate`: for each paper with a features file, `bib_sources`
  gives the citation count, `rater` rates the paper, and `report` writes
  `<output_dir>/paper_rating/<BASE>.rating.json`
- `report_papers.py`, stage `report`: `report` writes `04-report.references.md` and
  `04-report.report.json` from the issues files, the features files, and the rating
  files; the report has one row for each of the 20 papers and for each of the 2
  unresolved lines, with the issue of each failed row
- Rerun with 5 new rows: `resolve_papers.py` hits the HTTP cache for the 22 old rows,
  `download_papers.py` and `extract_features.py` skip the 20 old papers, so only the
  new papers cost network time and LLM tokens; `report_papers.py` rates all the
  papers again at no LLM cost, so a new tier list needs no new extraction
  ([`R11`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r11-rerun-skips-converted-papers),
  [`R17`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r17-rerun-skips-extraction))

## Requirements Map

| Req                                                                                                 | Component                         | How                                                            |
| :-------------------------------------------------------------------------------------------------- | :-------------------------------- | :------------------------------------------------------------- |
| [`R1`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r1-input-file-format)               | `resolver`                        | `_parse_line()` classifies the 6 kinds of line                  |
| [`R2`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r2-reference-format)                | `resolver`, `report`              | `_format_reference()`, written to `references.md`               |
| [`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source)            | `resolver`, `bib_sources`         | Canonical record from Semantic Scholar, Crossref, arXiv        |
| [`R4`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r4-download-link)                   | `resolver`, `bib_sources`         | Ordered download inputs, or the ` ???` mark                    |
| [`R5`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r5-unresolved-or-ambiguous-line)    | `resolver`, `pipeline`            | `Issue` with status `unresolved` or `ambiguous` and candidates |
| [`R6`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r6-duplicate-lines)                 | `resolver`                        | `_paper_key()` merges lines of the same paper                   |
| [`R7`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r7-reverse-chronological-order)     | `resolver`, `report`              | `_sort_papers()`: year down, then author, then title            |
| [`R8`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r8-pdf-and-markdown-files)          | `fetcher`                         | Runs the download tool into `<output_dir>/paper_pdf/` and `<output_dir>/paper_md/` |
| [`R9`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r9-file-base-name)                  | `resolver`, `fetcher`             | `format_base_filename()`, passed as `--output`                 |
| [`R10`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r10-failed-download-or-conversion) | `fetcher`, `pipeline`             | `FetchResult` with the reason, the loop goes on                |
| [`R11`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r11-rerun-skips-converted-papers)  | `fetcher`                         | `is_fetched()`: skip when `.pdf` and `.md` exist               |
| [`R12`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r12-feature-list)                  | `extractor`, `rater`              | Five features from the LLM and the record, plus the rating     |
| [`R13`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r13-unknown-value)                 | `extractor`                       | `unknown` value, no score except the rating                    |
| [`R14`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r14-evidence-for-each-value)       | `extractor`                       | Quote, section from the LLM, page from the PDF text            |
| [`R15`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r15-extracted-facts)               | `extractor`                       | `ExtractedFacts` with every labeled Sharpe ratio               |
| [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating)          | `rater`, `bib_sources`, `fetcher` | Page count, affiliation tier, Semantic Scholar citations       |
| [`R17`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r17-rerun-skips-extraction)        | `pipeline`, `report`              | Skip when the features file exists, unless `--force_extract`   |
| [`R18`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r18-report-file)                    | `report`                          | `report.json`, one row per paper or unresolved line            |
| [`R19`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r19-issues-file)                    | `report`                          | the `issue` of each failed row of `report.json`                |
| [`R20`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r20-features-file)                 | `report`                          | `<BASE>.features.json` next to the PDF, `<BASE>.rating.json`   |

## Configuration, Interfaces, and Known Limits

- Configuration: CLI flags of each script, listed in
  [`research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md),
  and described in the README of each script, e.g.,
  [`resolve_papers.py.README.md`](/research/Estimate_academic_paper_quality/resolve_papers.py.README.md)
- Interfaces: five CLI scripts, a papers file (JSON) between the
  first and the others, and one output dir
  ([`D30`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d30-output_dir-for-several-files-and-output-for-one))
  - Layout of the output dir:

    ```
    <output_dir>/
      00-paper_list.csv            copy of the input
      01-resolve.paper_list.json   the papers, R7
      01-resolve.issues.json       issues of resolve, R5
      01-resolve.summary.txt       counts and files of resolve
      02-download.issues.json      issues of download, R10
      02-download.summary.txt      counts and files of download
      03-extract.issues.json       issues of extract
      03-extract.summary.txt       counts and files of extract
      04-rate.issues.json          issues of rate
      04-rate.summary.txt          counts and files of rate
      04-report.report.json        R18, R19
      04-report.references.md      R2, R7
      04-report.summary.txt        counts and files of report
      paper_pdf/
        <BASE>.pdf                 R8
      paper_md/
        <BASE>.md                  R8
      paper_features/
        <BASE>.features.json       R20
      paper_rating/
        <BASE>.rating.json         R16, R20
      state/
        cache/                     HTTP response cache
        logs/                      logs of the download tool
    ```

  - The numbers are the order of the executables
    ([`D29`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d29-numbered-stage-files-in-the-output-dir)), and the full
    description is in
    [`research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md)

- Environment:
  - Network access, and Chrome for SSRN
    ([PRD constraints](/research/Estimate_academic_paper_quality/docs/01-prd.md#constraints-and-assumptions))
  - Docker, since the conversion runs a Dockerized prettier: the PRD does not list it
    (see
    [`research/Estimate_academic_paper_quality/docs/02.4-design.fetcher.md`](/research/Estimate_academic_paper_quality/docs/02.4-design.fetcher.md))
  - The key of the LLM provider, as the `llm` library reads it, e.g.,
    `OPENROUTER_KEY` for the default model of
    [`helpers.hllm_cli`](/helpers_root/helpers/hllm_cli.py)
  - Optional `SEMANTIC_SCHOLAR_API_KEY`, for a higher rate limit
- Known limits:
  - Stages run one paper at a time: about 100 papers take about an hour, mostly LLM
    and Chrome time
  - The user must pass the same `--output_dir` on a rerun, since the state lives
    there
    ([`D18`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d18-all-the-state-lives-in-the-output-dir))

## Deliberately Not Built

- Search for papers outside the input list, backtests, a total score, a ranking
  ([PRD Non-Goals](/research/Estimate_academic_paper_quality/docs/01-prd.md#non-goals))
- Parallel downloads or LLM calls: sequential is simple, and SSRN needs one Chrome
- An evaluation harness for the success metrics: it needs labeled sets, see Open
  Questions

## Open Questions

1. Who labels the input list for Resolution accuracy and the 20 papers for Extraction
   accuracy, and where do the labels live?
2. Which LLM is the default: pick it after the Extraction accuracy test on the 20
   labeled papers, and log the cost per paper
