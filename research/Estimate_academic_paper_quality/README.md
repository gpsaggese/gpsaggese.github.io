# Estimate Academic Paper Quality

- **Goal**: Code and docs of a pipeline that turns a list of academic papers into a
  CSV report of candidate trading strategies.

- **Input**: a CSV with one paper per row, as a title, link, DOI, or local PDF
- **Output**:
  - A papers file and an issues file from `resolve_papers.py`, both JSON
  - `04-report.references.md`, `04-report.report.csv`, and `04-report.issues.json`
  - The PDF, markdown, features file, and rating file of each paper
  - All in one output dir, see "Output Dir" below
- **Users**: quant researchers, who sort the CSV to pick the next papers to replicate

## Structure of the Dir

| Directory | Description                                                                 |
| :-------- | :-------------------------------------------------------------------------- |
| `docs/`   | Requirements, design, decisions, plan, and progress docs, in numbered order |
| `test/`   | Unit tests of the modules, and the fakes and helpers that they share        |

## Description of Files

- Python and Markdown files, alphabetical by path.

| File                                      | Description                                                                                                | Cluster      |
| :---------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------ |
| `affiliation_tiers.yaml.README.md`        | What the tier list contains, how to fill it from a QS export with the populate script, and how to check it | Entry point  |
| `docs/01-prd.md`                          | Product requirements: inputs, features, outputs, non-goals, and success metrics                            | Requirements |
| `docs/02-design.architecture.md`          | Overall design: components, import graph, data model, and requirements map                                 | Design       |
| `docs/02.1-design.pipeline.md`            | The four executables, the files between them, the per-paper error loop, and the skip of finished work      | Design       |
| `docs/02.2-design.resolver.md`            | Line parsing, paper matching, dedup, reference building, and sorting                                       | Design       |
| `docs/02.3-design.bib_sources.md`         | Clients of the lookup APIs (Semantic Scholar, Crossref, arXiv, Unpaywall)                                  | Design       |
| `docs/02.4-design.fetcher.md`             | Download and convert each paper with the download tool, skipping finished papers                           | Design       |
| `docs/02.5-design.extractor.md`           | One LLM call per paper that returns features, facts, affiliations, and quotes                              | Design       |
| `docs/02.6-design.rater.md`               | Paper quality rating from page length, affiliation, and citation count                                     | Design       |
| `docs/02.7-design.report.md`              | Writers and readers of the output files: references, CSVs, and features files                              | Design       |
| `docs/03-decisions.md`                    | Log of each design decision, with its reason, alternatives, and status                                     | Decisions    |
| `docs/04-plan.md`                         | Versions and PRs to build, in order, with the open questions of each component                             | Plans        |
| `docs/07-progress.md`                     | What is done and learned: milestone status, findings, next tasks, and history                              | Plans        |
| `download_papers.py`                      | CLI that downloads the PDF of each paper of a papers file and converts it to markdown                      | Entry point  |
| `download_papers.py.README.md`            | Arguments, examples, outputs, and architecture of the download CLI                                         | Entry point  |
| `epq_bib_sources.py`                      | Clients of Semantic Scholar, Crossref, arXiv, and Unpaywall, with a throttle                               | Lookup APIs  |
| `epq_fetcher.py`                          | Run the download tool per paper, and read the PDF pages and the markdown                                   | Download     |
| `epq_issues.py`                           | Build the issue of a paper that failed in a stage, and keep the reason of an error                         | Shared       |
| `epq_paths.py`                            | Names and paths of the files of the output dir, with no project imports                                    | Shared       |
| `epq_report.py`                           | Write and read the papers file, the issues, the CSVs, `references.md`, and the features and rating files   | Report       |
| `epq_summary.py`                          | Format and write the summary of a stage: counts, output files, and issues file                             | Shared       |
| `epq_types.py`                            | Frozen dataclasses shared by the modules, with no logic and no project imports                             | Shared types |
| `estimate_paper_quality.py`               | CLI that runs the four executables in order on one input CSV and one output dir, with stage selection      | Entry point  |
| `estimate_paper_quality.py.README.md`     | Arguments, examples, outputs, and architecture of the pipeline CLI                                         | Entry point  |
| `extract_features.prompt.md`              | System prompt of the LLM that extracts the features of a paper                                             | Extract      |
| `extract_features.py`                     | CLI that extracts the features of each downloaded paper with one LLM call and checks each quote            | Entry point  |
| `extract_features.py.README.md`           | Arguments, examples, outputs, and architecture of the extraction CLI                                       | Entry point  |
| `log.md`                                  | Log of the steps that made the docs, and the original idea of the pipeline                                 | Log          |
| `paper_list.csv`                          | Input CSV of papers, one per row, with authors in parentheses                                              | Input        |
| `populate_affiliation_tiers.py`           | CLI that fills `affiliation_tiers.yaml` from a CSV exported from the QS World University Rankings          | Entry point  |
| `README.md`                               | Summary, structure, files, executables, and architecture of this dir                                       | Overview     |
| `report_papers.py`                        | CLI that rates the papers from three signals, then writes the report and the issues file                   | Entry point  |
| `report_papers.py.README.md`              | Arguments, examples, outputs, and architecture of the rating and report CLI                                | Entry point  |
| `resolve_papers.py`                       | CLI that parses and matches the rows of an input CSV, and writes a papers file and an issues file          | Entry point  |
| `resolve_papers.py.README.md`             | Arguments, examples, outputs, and architecture of the resolver CLI                                         | Entry point  |
| `test/epq_test_utils.py`                  | Builders, a fake of the lookup APIs, a local HTTP server, and a stub tool                                  | Tests        |
| `test/test_download_papers.py`            | The download stage with a stub tool, and the command line of the download CLI                              | Tests        |
| `test/test_epq_bib_sources.py`            | Lookup clients against a local HTTP server, with no internet                                               | Tests        |
| `test/test_epq_end_to_end.py`             | The four executables in order on a local server, a stub tool, and a fake LLM                               | Tests        |
| `test/test_epq_fetcher.py`                | PDF readers, the skip of fetched papers, and the fetch through a stub tool                                 | Tests        |
| `test/test_epq_import_graph.py`           | Each module imports only the layers that the design allows                                                 | Tests        |
| `test/test_epq_paths.py`                  | The names of the files that each stage writes in the output dir                                            | Tests        |
| `test/test_epq_report.py`                 | Writers and readers of the issues, the features and rating files, and the CSVs                             | Tests        |
| `test/test_epq_summary.py`                | Count of issues, text of the summary, and the summary file                                                 | Tests        |
| `test/test_extract_features.py`           | Quote location, text cut, extraction with a fake LLM, the extraction stage, and the command line           | Tests        |
| `test/test_populate_affiliation_tiers.py` | The QS rank parsing, the column search, the merge with hand entries, and the read by `report_papers.py`    | Tests        |
| `test/test_report_papers.py`              | Score tables vs the rules file, rating, the stages `rate` and `report`, and the command line               | Tests        |
| `test/test_resolve_papers.py`             | Line parsing, title matching, paper building, merging, the resolve stage, and the command line             | Tests        |

## Description of Executables

- There are four executables, one per stage
  - They talk to each other only through files, so each one runs alone on the output
    of an earlier run

- Run them in this order:
  1. `research/Estimate_academic_paper_quality/resolve_papers.py`
      - Copies the input CSV, and turns it into a papers file and an issues file
      - Its arguments, its output files, and its exit codes are in
        `research/Estimate_academic_paper_quality/resolve_papers.py.README.md`.
  2. `research/Estimate_academic_paper_quality/download_papers.py`
      - Downloads the PDF of each paper and converts it to markdown
      - Its arguments, its output files, and its exit codes are in
        `research/Estimate_academic_paper_quality/download_papers.py.README.md`.
  3. `research/Estimate_academic_paper_quality/extract_features.py`
      - Extracts the features of each paper with one LLM call
      - Its arguments, its output files, and its exit codes are in
        `research/Estimate_academic_paper_quality/extract_features.py.README.md`.
  4. `research/Estimate_academic_paper_quality/report_papers.py`
      - Rates each paper, and writes `04-report.references.md`, `04-report.report.csv`,
        and `04-report.issues.json`
      - Its arguments, its output files, and its exit codes are in
        `research/Estimate_academic_paper_quality/report_papers.py.README.md`.

- The script `research/Estimate_academic_paper_quality/estimate_paper_quality.py`
  runs the four executables in order, as subprocesses. Its arguments and its stage
  selection are in
  `research/Estimate_academic_paper_quality/estimate_paper_quality.py.README.md`.

- The helper `research/Estimate_academic_paper_quality/populate_affiliation_tiers.py`
  is not a stage. It fills `affiliation_tiers.yaml` from a CSV exported from the QS
  site. What the YAML contains and how to fill it are in
  `research/Estimate_academic_paper_quality/affiliation_tiers.yaml.README.md`.

## Output Dir

- Every executable takes `--output_dir`, and the dir is the only place where files are
  written
  - The user gives the dir, and the executables infer the name of every file, so no
    file name is typed
  - A rerun on the same dir skips the work that is done
- A file that a stage writes is named `NN-<stage>.<what>.<ext>`, where `NN` is the
  order of the executable: `resolve` is `01`, `download` `02`, `extract` `03`, and
  `rate` and `report` `04`
  ```text
  <output_dir>/
      00-paper_list.csv
      01-resolve.paper_list.json
      01-resolve.issues.json
      01-resolve.summary.txt
      02-download.issues.json
      02-download.summary.txt
      03-extract.issues.json
      03-extract.summary.txt
      04-rate.issues.json
      04-rate.summary.txt
      04-report.report.csv
      04-report.issues.json
      04-report.references.md
      04-report.summary.txt
      papers/
          <BASE>.pdf
          <BASE>.md
          <BASE>.features.json
          <BASE>.rating.json
      state/
          cache/
          logs/
  ```
- The files:

  | File                          | Written by            | Content                                                     |
  | :---------------------------- | :-------------------- | :---------------------------------------------------------- |
  | `00-paper_list.csv`           | `resolve_papers.py`   | Copy of the input CSV, so the run keeps the list it used    |
  | `01-resolve.paper_list.json`  | `resolve_papers.py`   | The resolved papers, from the newest to the oldest          |
  | `01-resolve.issues.json`      | `resolve_papers.py`   | The input rows that gave no paper, or more than one         |
  | `02-download.issues.json`     | `download_papers.py`  | The papers that failed to download or to convert            |
  | `03-extract.issues.json`      | `extract_features.py` | The papers that failed to extract                           |
  | `04-rate.issues.json`         | `report_papers.py`    | The papers that failed to rate                              |
  | `NN-<stage>.summary.txt`      | Each stage            | Counts of items, output files, and issues file of the run   |
  | `04-report.report.csv`        | `report_papers.py`    | One row per rated paper: the table to sort in a spreadsheet |
  | `04-report.issues.json`        | `report_papers.py`    | The issues of all the stages                                |
  | `04-report.references.md`     | `report_papers.py`    | The references of all the resolved papers                   |
  | `papers/<BASE>.pdf`, `.md`    | `download_papers.py`  | The paper and its markdown                                  |
  | `papers/<BASE>.features.json` | `extract_features.py` | The features, facts, affiliations, and page count           |
  | `papers/<BASE>.rating.json`   | `report_papers.py`    | The rating and the three signals behind it                  |
  | `state/cache/`                | `resolve_papers.py`, `report_papers.py` | The cache of the lookup answers            |
  | `state/logs/`                 | `download_papers.py`  | The log of the download tool for each paper                 |

- Why this layout
  - The number gives the order of the pipeline, so a listing of the dir reads from
    the input to the report
    ([`D29`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d29-numbered-stage-files-in-the-output-dir))
  - The files that a program reads are JSON, and the file that a person sorts is
    CSV ([`D31`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d31-json-for-the-intermediate-files))
  - The files of one paper are in `papers/` with no number, since their suffix says
    the stage
  - `state/` holds only the cache and the logs, which are not results of a stage
  - In the docs, `report.csv`, `issues.json`, and `references.md` are the short names
    of the three `04-report.*` files
- The names are in `research/Estimate_academic_paper_quality/epq_paths.py`, and the
  rules are in `.claude/skills/architecture.rules.md` (`ARCH3.6`, `ARCH3.7`,
  `ARCH3.8`)

## `resolve_papers.py`

### What It Does

- Reads a CSV with one paper per row (no header needed), and looks up each row in
  Semantic Scholar, Crossref, arXiv, and Unpaywall
- Copies the input to `00-paper_list.csv` in the output dir
- Writes `01-resolve.paper_list.json`, one paper per item from the newest to the
  oldest, and `01-resolve.issues.json`, one issue per row that gave no paper
- Takes only `--output_dir`, since it writes several files, and infers their names
- Makes no LLM call, and needs no Docker and no Chrome

### Examples

- Resolve the list, and read the issues:
  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10
  ```

- Show how each row is classified, with no lookup:
  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10 \
      --dry_run
  ```

## `download_papers.py`

### What It Does

- Downloads the PDF of each paper of the papers file, and converts it to markdown
- Waits a random time between two downloads (`--min_delay_sec` and `--max_delay_sec`),
  so that the hosts do not take it for a bot
- Skips the papers that are already converted, and writes the papers that failed to
  `02-download.issues.json`
- Makes no LLM call

### Examples

- Download the papers that `resolve_papers.py` wrote in the dir:
  ```bash
  > research/Estimate_academic_paper_quality/download_papers.py \
      --output_dir ~/papers/run_2026_10
  ```

- Wait 5 to 20 seconds between two downloads:
  ```bash
  > research/Estimate_academic_paper_quality/download_papers.py \
      --output_dir ~/papers/run_2026_10 \
      --min_delay_sec 5 --max_delay_sec 20
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
      --output_dir ~/papers/run_2026_10
  ```

- Extract again the features of all the papers, with another model:
  ```bash
  > research/Estimate_academic_paper_quality/extract_features.py \
      --output_dir ~/papers/run_2026_10 \
      --force_extract \
      --model <model>
  ```

## `report_papers.py`

### What It Does

- Runs the stage `rate`: rates each paper from its page count, affiliations, and
  citation count, and writes `papers/<BASE>.rating.json`
- Runs the stage `report`: writes `04-report.references.md`,
  `04-report.report.csv`, and `04-report.issues.json`
- Makes no LLM call, and reads the Semantic Scholar key from
  `SEMANTIC_SCHOLAR_API_KEY`

### Examples

- Write the reference list only, with no lookup:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action report
  ```

- Rate and report:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --output_dir ~/papers/run_2026_10
  ```

## `estimate_paper_quality.py`

### What It Does

- Runs `resolve_papers.py`, `download_papers.py`, `extract_features.py`, and
  `report_papers.py` in order, and stops at the first stage that fails
- Selects the executables to run with `--action`, `--skip_action`, `--only_action`,
  `--clear_actions`, and `--all_actions`. There is one action per executable, named
  after it: `resolve_papers`, `download_papers`, `extract_features`, and
  `report_papers`
- Writes all the outputs in the output dir, with the names of "Output Dir"
- A rerun on the same dir does not download, convert, or extract again a paper that
  is done

### Examples

- Run the whole pipeline:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10
  ```

- Extract again with another model, and ask for fresh citation counts:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10 \
      --model <model> --force_extract --refresh_citations
  ```

- Run only `extract_features.py` and `report_papers.py` on the output of an earlier
  run:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --output_dir ~/papers/run_2026_10 \
      --only_action extract_features --only_action report_papers
  ```

- Show the command of each executable, with no execution:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10 \
      --dry_run
  ```

## Description of Workflows

Check the list, then run the whole pipeline. The commands use the `--output_dir` of
the examples above.

1. Resolve the list, and read the issues:
   ```bash
   > research/Estimate_academic_paper_quality/resolve_papers.py \
       --input research/Estimate_academic_paper_quality/paper_list.csv \
       --output_dir ~/papers/run_2026_10 \
       --email me@umd.edu
   ```
   Open `~/papers/run_2026_10/01-resolve.issues.json`. Each item is an input that was
   not resolved or that matched more than one paper, with its candidates.

2. Fix the rows that `01-resolve.issues.json` names in
   `research/Estimate_academic_paper_quality/paper_list.csv`, then run step 1 again.
   The rerun reuses the cached answers of the APIs. To add a link or the path of a
   local PDF to a paper, edit its `download_inputs` in `01-resolve.paper_list.json`,
   and do not run step 1 again, since it writes that file again.

3. Run all the stages. The stage `download` needs Docker, and the SSRN papers also
   need Chrome. The issues of step 1 go to `04-report.issues.json`.
   ```bash
   > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
       --input research/Estimate_academic_paper_quality/paper_list.csv \
       --output_dir ~/papers/run_2026_10
   ```
   To run one stage alone, select its action, e.g., `estimate_paper_quality.py
   --output_dir ~/papers/run_2026_10 --only_action download_papers`, or call its
   executable on the output dir of step 1, e.g., `download_papers.py --output_dir
   ~/papers/run_2026_10`. Both read `01-resolve.paper_list.json` in that dir.

4. After a change of the LLM or of the prompt file, extract again and rewrite the
   reports:
   ```bash
   > research/Estimate_academic_paper_quality/extract_features.py \
       --output_dir ~/papers/run_2026_10 \
       --force_extract \
       --model <model>
   > research/Estimate_academic_paper_quality/report_papers.py \
       --output_dir ~/papers/run_2026_10
   ```

5. Fill or refresh the affiliation tier list, or change the score bands, then rate
   again and rewrite the reports. The tier list needs a CSV that you export from the
   QS World University Rankings site. The rating makes no LLM call:
   ```bash
   > research/Estimate_academic_paper_quality/populate_affiliation_tiers.py \
       --qs_file ~/Downloads/qs_2026.csv \
       --edition "QS World University Rankings 2026" \
       --source "https://www.topuniversities.com/world-university-rankings"
   > research/Estimate_academic_paper_quality/report_papers.py \
       --output_dir ~/papers/run_2026_10
   ```
   The aliases, the hand entries (industry labs, NBER), the traps, and the checks are
   in `research/Estimate_academic_paper_quality/affiliation_tiers.yaml.README.md`.

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
    - `research/Estimate_academic_paper_quality/estimate_paper_quality.py` runs the
      four executables in order, as subprocesses with `hsystem`, and imports none of
      them
  - Infrastructure:
    - `research/Estimate_academic_paper_quality/epq_bib_sources.py` calls the lookup
      APIs, with a throttle, a retry, and a disk cache
    - `research/Estimate_academic_paper_quality/epq_fetcher.py` runs the download
      tool and reads the PDF pages and the markdown
    - `research/Estimate_academic_paper_quality/epq_report.py` writes and reads the
      papers file, the issues, the features and rating files, and the CSV outputs
  - Shared types and utilities, with no project imports or only `epq_types.py`:
    - `research/Estimate_academic_paper_quality/epq_types.py` holds the dataclasses
    - `research/Estimate_academic_paper_quality/epq_paths.py` holds the names and
      the paths of the files of the output dir
    - `research/Estimate_academic_paper_quality/epq_issues.py` builds the issue of a
      paper that failed
- The executables talk only through files on disk, and no executable imports another
- The imports point down the layers only, and
  `research/Estimate_academic_paper_quality/test/test_epq_import_graph.py` checks it
- Other files:
  - `research/Estimate_academic_paper_quality/extract_features.prompt.md`: the
    system prompt of the LLM
  - `research/Estimate_academic_paper_quality/affiliation_tiers.yaml`: the dated tier
    list of the affiliation signal, empty until a person fills it with
    `research/Estimate_academic_paper_quality/populate_affiliation_tiers.py`, as
    `research/Estimate_academic_paper_quality/affiliation_tiers.yaml.README.md`
    explains
- `docs/` is numbered in reading order: 01 requirements, 02 design, 03 decisions,
  04 plan, and 07 progress. A `06-spec.<PR_ID>.md` file is added before the work of a
  PR that needs a spec.
- `test/` has one test file per module or executable, with `hunitest` classes, and
  `test_epq_end_to_end.py` runs the four executables in order.
  `test/epq_test_utils.py` holds the fakes: the lookup APIs, a local HTTP server, and
  a stub download tool.
