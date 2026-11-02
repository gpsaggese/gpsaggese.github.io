# Estimate Academic Paper Quality

Code and docs of a pipeline that turns a list of academic papers into a CSV report
of candidate trading strategies.

- Input: a CSV with one paper per row, as a title, link, DOI, or local PDF
- Output: a papers CSV and an issues CSV from `resolve_papers.py`, then
  `references.md`, `report.csv`, and `issues.csv`, plus the PDF, markdown, and
  features file of each paper
- Users: quant researchers, who sort the CSV to pick the next papers to replicate

## Structure of the Dir

| Directory | Description                                                                 |
| :-------- | :-------------------------------------------------------------------------- |
| `docs/`   | Requirements, design, decisions, plan, and progress docs, in numbered order |
| `test/`   | Unit tests of the modules, and the fakes and helpers that they share        |

## Description of Files

Python and Markdown files, alphabetical by path. The status of the work is in
`research/Estimate_academic_paper_quality/docs/07-progress.md`.

| File                                  | Description                                                                           | Cluster      |
| :------------------------------------ | ------------------------------------------------------------------------------------- | ------------ |
| `docs/01-prd.md`                      | Product requirements: inputs, features, outputs, non-goals, and success metrics       | Requirements |
| `docs/02-design.architecture.md`      | Overall design: components, import graph, data model, and requirements map            | Design       |
| `docs/02.1-design.pipeline.md`        | CLI, the three stages, the per-paper error loop, and the skip of finished work        | Design       |
| `docs/02.2-design.resolver.md`        | Line parsing, paper matching, dedup, reference building, and sorting                  | Design       |
| `docs/02.3-design.bib_sources.md`     | Clients of the lookup APIs (Semantic Scholar, Crossref, arXiv, Unpaywall)             | Design       |
| `docs/02.4-design.fetcher.md`         | Download and convert each paper with the download tool, skipping finished papers      | Design       |
| `docs/02.5-design.extractor.md`       | One LLM call per paper that returns features, facts, affiliations, and quotes         | Design       |
| `docs/02.6-design.rater.md`           | Paper quality rating from page length, affiliation, and citation count                | Design       |
| `docs/02.7-design.report.md`          | Writers and readers of the output files: references, CSVs, and features files         | Design       |
| `docs/03-decisions.md`                | Log of each design decision, with its reason, alternatives, and status                | Decisions    |
| `docs/04-plan.md`                     | Versions and PRs to build, in order, with the open questions of each component        | Plans        |
| `docs/07-progress.md`                 | What is done and learned: milestone status, findings, next tasks, and history         | Plans        |
| `epq_bib_sources.py`                  | Clients of Semantic Scholar, Crossref, arXiv, and Unpaywall, with a throttle          | Lookup APIs  |
| `epq_extractor.prompt.md`             | System prompt of the LLM that extracts the features of a paper                        | Extract      |
| `epq_extractor.py`                    | One LLM call per paper, with each quote checked against the PDF text                  | Extract      |
| `epq_fetcher.py`                      | Run the download tool per paper, and read the PDF pages and the markdown              | Download     |
| `epq_pipeline.py`                     | Run the resolve step and the three stages, with a catch per row or paper              | Pipeline     |
| `epq_rater.py`                        | Rate a paper from its page count, its affiliations, and its citations                 | Rate         |
| `epq_report.py`                       | Write and read the issues, `references.md`, the CSVs, and the features                | Report       |
| `epq_resolver.py`                     | Parse each input line, match it to papers, and build the sorted references            | Resolve      |
| `epq_types.py`                        | Frozen dataclasses shared by the modules, with no logic and no project imports        | Shared types |
| `estimate_paper_quality.py`           | CLI that turns a papers CSV into PDFs, a reference list, and a CSV report             | Entry point  |
| `estimate_paper_quality.py.README.md` | Arguments, examples, outputs, and architecture of the CLI                             | Entry point  |
| `log.md`                              | Log of the steps that made the docs, and the original idea of the pipeline            | Log          |
| `paper_list.csv`                      | Input CSV of papers, one per row, with authors in parentheses                         | Input        |
| `paper_list.md`                       | Input list of papers in markdown, one bullet per paper: convert it to a CSV           | Input        |
| `README.md`                           | Summary, structure, files, executables, and architecture of this dir                  | Overview     |
| `resolve_papers.py`                   | CLI that turns an input CSV of papers into a papers CSV and an issues CSV             | Entry point  |
| `resolve_papers.py.README.md`         | Arguments, examples, outputs, and architecture of the resolver CLI                    | Entry point  |
| `test/epq_test_utils.py`              | Builders, a fake of the lookup APIs, a local HTTP server, and a stub tool             | Tests        |
| `test/test_epq_bib_sources.py`        | Lookup clients against a local HTTP server, with no internet                          | Tests        |
| `test/test_epq_extractor.py`          | Quote location, reference removal, text cut, and extraction with a fake LLM           | Tests        |
| `test/test_epq_fetcher.py`            | PDF readers, the skip of fetched papers, and the fetch through a stub tool            | Tests        |
| `test/test_epq_import_graph.py`       | Each module imports only the layers that the design allows                            | Tests        |
| `test/test_epq_pipeline.py`           | The resolve step and the three stages, on a local server, a stub tool, and a fake LLM | Tests        |
| `test/test_epq_rater.py`              | Score tables and rating, checked against `.claude/skills/paper.rules.md`              | Tests        |
| `test/test_epq_report.py`             | Writers and readers of the issues, the features file, and the CSVs                    | Tests        |
| `test/test_epq_resolver.py`           | Line parsing, title matching, paper building, merging, and resolving of lines         | Tests        |
| `test/test_resolve_papers.py`         | The command line of the resolver, with a fake of the lookup APIs                      | Tests        |

## Description of Executables

There are two executables. Run the first one, then the second one:
- `research/Estimate_academic_paper_quality/resolve_papers.py`: turns an input CSV
  into a papers CSV and an issues CSV. Its arguments, its output files, and its exit
  codes are in
  `research/Estimate_academic_paper_quality/resolve_papers.py.README.md`.
- `research/Estimate_academic_paper_quality/estimate_paper_quality.py`: turns the
  papers CSV into PDFs, a reference list, and a CSV report. Its arguments, its output
  files, and its exit codes are in
  `research/Estimate_academic_paper_quality/estimate_paper_quality.py.README.md`.

## `resolve_papers.py`

### What It Does

- Reads the column `input` of a CSV, one paper per row, and looks up each row in
  Semantic Scholar, Crossref, arXiv, and Unpaywall
- Writes the papers CSV, one row per paper from the newest to the oldest, and the
  issues CSV, one row per row that gave no paper
- Makes no LLM call, and needs no Docker and no Chrome

### Examples

- Resolve the list, and read the issues:
  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output ~/papers/run_2026_10/papers.csv
  ```

- Show how each row is classified, with no lookup:
  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output ~/papers/run_2026_10/papers.csv \
      --dry_run
  ```

## `estimate_paper_quality.py`

### What It Does

- Runs the stages `download`, `extract`, and `report` on the papers CSV of
  `--input`, and writes all the outputs to `--output_dir`
- Skips the work of a previous run on the same dir: the converted papers, the papers
  with a features file, and the cached answers of the APIs
- Calls an LLM only in the stage `extract`, and reads the Semantic Scholar key from
  `SEMANTIC_SCHOLAR_API_KEY`

### Examples

- Write the reference list only, with no download and no LLM call:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action report
  ```

- Run all the stages:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10
  ```

- Extract again the features of all the papers, with another model:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action extract --action report \
      --force_extract \
      --model <model>
  ```

## Description of Workflows

Check the list, then run the whole pipeline. The commands use the papers CSV and the
`--output_dir` of the examples above.

1. Resolve the list, and read the issues:
   ```bash
   > research/Estimate_academic_paper_quality/resolve_papers.py \
       --input research/Estimate_academic_paper_quality/paper_list.csv \
       --output ~/papers/run_2026_10/papers.csv \
       --email me@umd.edu
   ```
   Open `papers.issues.csv` next to the papers CSV. Each row is an input that was not
   resolved or that matched more than one paper, with its candidates.

2. Fix the rows that `papers.issues.csv` names in
   `research/Estimate_academic_paper_quality/paper_list.csv`, then run step 1 again.
   The rerun reuses the cached answers of the APIs. To add a link or the path of a
   local PDF to a paper, edit its `download_inputs` in the papers CSV.

3. Run all the stages. The stage `download` needs Docker, and the SSRN papers also
   need Chrome. The issues of step 1 go to `issues.csv`.
   ```bash
   > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
       --input ~/papers/run_2026_10/papers.csv \
       --output_dir ~/papers/run_2026_10
   ```

4. After a change of the LLM or of the prompt file, extract again and rewrite the
   reports:
   ```bash
   > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
       --input ~/papers/run_2026_10/papers.csv \
       --output_dir ~/papers/run_2026_10 \
       --clear_actions --action extract --action report \
       --force_extract \
       --model <model>
   ```

## Description of Architecture

- The design is in
  `research/Estimate_academic_paper_quality/docs/02-design.architecture.md`
- The layers, from the top down:
  - Scripts: `research/Estimate_academic_paper_quality/resolve_papers.py` resolves
    an input CSV, and `research/Estimate_academic_paper_quality/estimate_paper_quality.py`
    parses the flags and picks the stages
  - Orchestration: `research/Estimate_academic_paper_quality/epq_pipeline.py` runs
    the resolve step and each stage, with a catch per row or paper
  - Business logic:
    - `research/Estimate_academic_paper_quality/epq_resolver.py` parses each line and
      matches it to papers
    - `research/Estimate_academic_paper_quality/epq_extractor.py` makes one LLM call
      per paper
    - `research/Estimate_academic_paper_quality/epq_rater.py` rates a paper from its
      page count, affiliations, and citations
  - Infrastructure:
    - `research/Estimate_academic_paper_quality/epq_bib_sources.py` calls the lookup
      APIs, with a throttle, a retry, and a disk cache
    - `research/Estimate_academic_paper_quality/epq_fetcher.py` runs the download
      tool and reads the PDF pages and the markdown
    - `research/Estimate_academic_paper_quality/epq_report.py` writes and reads the
      CSV files of the resolver, the issues, and the outputs
  - Shared types: `research/Estimate_academic_paper_quality/epq_types.py`, with no
    project imports
- The imports point down the layers only, and
  `research/Estimate_academic_paper_quality/test/test_epq_import_graph.py` checks it
- Other files:
  - `research/Estimate_academic_paper_quality/epq_extractor.prompt.md`: the system
    prompt of the LLM
  - `research/Estimate_academic_paper_quality/affiliation_tiers.yaml`: the dated tier
    list of the affiliation signal, empty until a person fills it
- `docs/` is numbered in reading order: 01 requirements, 02 design, 03 decisions,
  04 plan, and 07 progress. A `06-spec.<PR_ID>.md` file is added before the work of a
  PR that needs a spec.
- `test/` has one `test_epq_<module>.py` file per module, with `hunitest` classes.
  `test/epq_test_utils.py` holds the fakes: the lookup APIs, a local HTTP server, and
  a stub download tool.
