# Estimate Academic Paper Quality

Code and docs of a pipeline that turns a list of academic papers into a CSV report
of candidate trading strategies.

- Input: a CSV with one paper per row, as a title, link, DOI, or local PDF
- Output: a papers CSV and an issues CSV from `resolve_papers.py`, then
  `references.md`, `report.csv`, and `issues.csv`, plus the PDF, markdown,
  features file, and rating file of each paper
- Users: quant researchers, who sort the CSV to pick the next papers to replicate

## Structure of the Dir

| Directory | Description                                                                 |
| :-------- | :-------------------------------------------------------------------------- |
| `docs/`   | Requirements, design, decisions, plan, and progress docs, in numbered order |
| `test/`   | Unit tests of the modules, and the fakes and helpers that they share        |

## Description of Files

Python and Markdown files, alphabetical by path. The status of the work is in
`research/Estimate_academic_paper_quality/docs/07-progress.md`.

| File                              | Description                                                                                           | Cluster      |
| :-------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------ |
| `docs/01-prd.md`                  | Product requirements: inputs, features, outputs, non-goals, and success metrics                       | Requirements |
| `docs/02-design.architecture.md`  | Overall design: components, import graph, data model, and requirements map                            | Design       |
| `docs/02.1-design.pipeline.md`    | The four executables, the files between them, the per-paper error loop, and the skip of finished work | Design       |
| `docs/02.2-design.resolver.md`    | Line parsing, paper matching, dedup, reference building, and sorting                                  | Design       |
| `docs/02.3-design.bib_sources.md` | Clients of the lookup APIs (Semantic Scholar, Crossref, arXiv, Unpaywall)                             | Design       |
| `docs/02.4-design.fetcher.md`     | Download and convert each paper with the download tool, skipping finished papers                      | Design       |
| `docs/02.5-design.extractor.md`   | One LLM call per paper that returns features, facts, affiliations, and quotes                         | Design       |
| `docs/02.6-design.rater.md`       | Paper quality rating from page length, affiliation, and citation count                                | Design       |
| `docs/02.7-design.report.md`      | Writers and readers of the output files: references, CSVs, and features files                         | Design       |
| `docs/03-decisions.md`            | Log of each design decision, with its reason, alternatives, and status                                | Decisions    |
| `docs/04-plan.md`                 | Versions and PRs to build, in order, with the open questions of each component                        | Plans        |
| `docs/07-progress.md`             | What is done and learned: milestone status, findings, next tasks, and history                         | Plans        |
| `download_papers.py`              | CLI that downloads the PDF of each paper of a papers CSV and converts it to markdown                  | Entry point  |
| `download_papers.py.README.md`    | Arguments, examples, outputs, and architecture of the download CLI                                    | Entry point  |
| `epq_bib_sources.py`              | Clients of Semantic Scholar, Crossref, arXiv, and Unpaywall, with a throttle                          | Lookup APIs  |
| `epq_fetcher.py`                  | Run the download tool per paper, and read the PDF pages and the markdown                              | Download     |
| `epq_issues.py`                   | Build the issue of a paper that failed in a stage, and keep the reason of an error                    | Shared       |
| `epq_paths.py`                    | Paths of the files that the executables share, with no project imports                                | Shared       |
| `epq_report.py`                   | Write and read the issues, `references.md`, the CSVs, and the features and rating files               | Report       |
| `epq_types.py`                    | Frozen dataclasses shared by the modules, with no logic and no project imports                        | Shared types |
| `estimate_paper_quality.sh`       | Shell script that runs the four executables in order on one input CSV and one output dir              | Entry point  |
| `extract_features.prompt.md`      | System prompt of the LLM that extracts the features of a paper                                        | Extract      |
| `extract_features.py`             | CLI that extracts the features of each downloaded paper with one LLM call and checks each quote       | Entry point  |
| `extract_features.py.README.md`   | Arguments, examples, outputs, and architecture of the extraction CLI                                  | Entry point  |
| `log.md`                          | Log of the steps that made the docs, and the original idea of the pipeline                            | Log          |
| `paper_list.csv`                  | Input CSV of papers, one per row, with authors in parentheses                                         | Input        |
| `README.md`                       | Summary, structure, files, executables, and architecture of this dir                                  | Overview     |
| `report_papers.py`                | CLI that rates the papers from three signals, then writes the report and the issues CSV               | Entry point  |
| `report_papers.py.README.md`      | Arguments, examples, outputs, and architecture of the rating and report CLI                           | Entry point  |
| `resolve_papers.py`               | CLI that parses and matches the rows of an input CSV, and writes a papers CSV and an issues CSV       | Entry point  |
| `resolve_papers.py.README.md`     | Arguments, examples, outputs, and architecture of the resolver CLI                                    | Entry point  |
| `test/epq_test_utils.py`          | Builders, a fake of the lookup APIs, a local HTTP server, and a stub tool                             | Tests        |
| `test/test_download_papers.py`    | The download stage with a stub tool, and the command line of the download CLI                         | Tests        |
| `test/test_epq_bib_sources.py`    | Lookup clients against a local HTTP server, with no internet                                          | Tests        |
| `test/test_epq_end_to_end.py`     | The four executables in order on a local server, a stub tool, and a fake LLM                          | Tests        |
| `test/test_epq_fetcher.py`        | PDF readers, the skip of fetched papers, and the fetch through a stub tool                            | Tests        |
| `test/test_epq_import_graph.py`   | Each module imports only the layers that the design allows                                            | Tests        |
| `test/test_epq_report.py`         | Writers and readers of the issues, the features and rating files, and the CSVs                        | Tests        |
| `test/test_extract_features.py`   | Quote location, text cut, extraction with a fake LLM, the extraction stage, and the command line      | Tests        |
| `test/test_report_papers.py`      | Score tables vs the rules file, rating, the stages `rate` and `report`, and the command line          | Tests        |
| `test/test_resolve_papers.py`     | Line parsing, title matching, paper building, merging, the resolve stage, and the command line        | Tests        |

## Description of Executables

There are four executables, one per stage. They talk to each other only through
files, so each one runs alone on the output of an earlier run. Run them in this
order:
- `research/Estimate_academic_paper_quality/resolve_papers.py`: turns an input CSV
  into a papers CSV and an issues CSV. Its arguments, its output files, and its exit
  codes are in
  `research/Estimate_academic_paper_quality/resolve_papers.py.README.md`.
- `research/Estimate_academic_paper_quality/download_papers.py`: downloads the PDF of
  each paper and converts it to markdown. Its arguments, its output files, and its
  exit codes are in
  `research/Estimate_academic_paper_quality/download_papers.py.README.md`.
- `research/Estimate_academic_paper_quality/extract_features.py`: extracts the
  features of each paper with one LLM call. Its arguments, its output files, and its
  exit codes are in
  `research/Estimate_academic_paper_quality/extract_features.py.README.md`.
- `research/Estimate_academic_paper_quality/report_papers.py`: rates each paper, and
  writes `references.md`, `report.csv`, and `issues.csv`. Its arguments, its output
  files, and its exit codes are in
  `research/Estimate_academic_paper_quality/report_papers.py.README.md`.

The shell script `research/Estimate_academic_paper_quality/estimate_paper_quality.sh`
runs the four executables in order.

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

## `download_papers.py`

### What It Does

- Downloads the PDF of each paper of the papers CSV, and converts it to markdown
- Skips the papers that are already converted, and writes the papers that failed to
  `state/issues.download.json`
- Makes no LLM call

### Examples

- Download the papers:
  ```bash
  > research/Estimate_academic_paper_quality/download_papers.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10
  ```

## `extract_features.py`

### What It Does

- Calls the LLM once per paper that has a markdown file and no features file, and
  writes `papers/<BASE>.features.json`
- Checks every quote of the answer against the text of the PDF
- Does not rate the paper, so a new tier list needs no new LLM call

### Examples

- Extract the features of the downloaded papers:
  ```bash
  > research/Estimate_academic_paper_quality/extract_features.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10
  ```

- Extract again the features of all the papers, with another model:
  ```bash
  > research/Estimate_academic_paper_quality/extract_features.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --force_extract \
      --model <model>
  ```

## `report_papers.py`

### What It Does

- Runs the stage `rate`: rates each paper from its page count, affiliations, and
  citation count, and writes `papers/<BASE>.rating.json`
- Runs the stage `report`: writes `references.md`, `report.csv`, and `issues.csv`
- Makes no LLM call, and reads the Semantic Scholar key from
  `SEMANTIC_SCHOLAR_API_KEY`

### Examples

- Write the reference list only, with no lookup:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action report
  ```

- Rate and report:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10
  ```

## `estimate_paper_quality.sh`

### What It Does

- Runs `resolve_papers.py`, `download_papers.py`, `extract_features.py`, and
  `report_papers.py` in order, and stops at the first stage that fails
- Writes the papers CSV and all the other outputs in the output dir
- A rerun on the same dir does not download, convert, or extract again a paper that
  is done

### Examples

- Run the whole pipeline:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.sh \
      -i research/Estimate_academic_paper_quality/paper_list.csv \
      -o ~/papers/run_2026_10
  ```

- Extract again with another model, and ask for fresh citation counts:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.sh \
      -i research/Estimate_academic_paper_quality/paper_list.csv \
      -o ~/papers/run_2026_10 \
      -m <model> -f -r
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
   > research/Estimate_academic_paper_quality/estimate_paper_quality.sh \
       -i research/Estimate_academic_paper_quality/paper_list.csv \
       -o ~/papers/run_2026_10
   ```
   To run one stage alone, call its executable on the papers CSV and the output dir
   of step 1, e.g., `download_papers.py --input ~/papers/run_2026_10/papers.csv
   --output_dir ~/papers/run_2026_10`.

4. After a change of the LLM or of the prompt file, extract again and rewrite the
   reports:
   ```bash
   > research/Estimate_academic_paper_quality/extract_features.py \
       --input ~/papers/run_2026_10/papers.csv \
       --output_dir ~/papers/run_2026_10 \
       --force_extract \
       --model <model>
   > research/Estimate_academic_paper_quality/report_papers.py \
       --input ~/papers/run_2026_10/papers.csv \
       --output_dir ~/papers/run_2026_10
   ```

5. After a change of `affiliation_tiers.yaml` or of the score bands, rate again and
   rewrite the reports. This makes no LLM call:
   ```bash
   > research/Estimate_academic_paper_quality/report_papers.py \
       --input ~/papers/run_2026_10/papers.csv \
       --output_dir ~/papers/run_2026_10
   ```

## Description of Architecture

- The design is in
  `research/Estimate_academic_paper_quality/docs/02-design.architecture.md`
- The layers, from the top down:
  - Scripts and orchestration: one executable per stage. Each one parses its flags
    in `_parse()` and runs its stage in `_main()`, with a catch per row or paper:
    - `research/Estimate_academic_paper_quality/resolve_papers.py` resolves an
      input CSV: it also parses each line, matches it to papers, and builds the
      references
    - `research/Estimate_academic_paper_quality/download_papers.py` downloads and
      converts the papers
    - `research/Estimate_academic_paper_quality/extract_features.py` extracts the
      features of the papers with one LLM call per paper
    - `research/Estimate_academic_paper_quality/report_papers.py` rates the papers
      from their page count, affiliations, and citations, and writes the report,
      with the stages `rate` and `report`
    - `research/Estimate_academic_paper_quality/estimate_paper_quality.sh` runs the
      four executables in order
  - Infrastructure:
    - `research/Estimate_academic_paper_quality/epq_bib_sources.py` calls the lookup
      APIs, with a throttle, a retry, and a disk cache
    - `research/Estimate_academic_paper_quality/epq_fetcher.py` runs the download
      tool and reads the PDF pages and the markdown
    - `research/Estimate_academic_paper_quality/epq_report.py` writes and reads the
      CSV files of the resolver, the issues, and the outputs
  - Shared types and utilities, with no project imports or only `epq_types.py`:
    - `research/Estimate_academic_paper_quality/epq_types.py` holds the dataclasses
    - `research/Estimate_academic_paper_quality/epq_paths.py` holds the paths of the
      files that the executables share
    - `research/Estimate_academic_paper_quality/epq_issues.py` builds the issue of a
      paper that failed
- The executables talk only through files on disk, and no executable imports another
- The imports point down the layers only, and
  `research/Estimate_academic_paper_quality/test/test_epq_import_graph.py` checks it
- Other files:
  - `research/Estimate_academic_paper_quality/extract_features.prompt.md`: the
    system prompt of the LLM
  - `research/Estimate_academic_paper_quality/affiliation_tiers.yaml`: the dated tier
    list of the affiliation signal, empty until a person fills it
- `docs/` is numbered in reading order: 01 requirements, 02 design, 03 decisions,
  04 plan, and 07 progress. A `06-spec.<PR_ID>.md` file is added before the work of a
  PR that needs a spec.
- `test/` has one test file per module or executable, with `hunitest` classes, and
  `test_epq_end_to_end.py` runs the four executables in order.
  `test/epq_test_utils.py` holds the fakes: the lookup APIs, a local HTTP server, and
  a stub download tool.
