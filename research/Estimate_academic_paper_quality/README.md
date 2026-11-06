# Estimate Academic Paper Quality

Code and docs of a pipeline that turns a list of academic papers into a CSV report
of candidate trading strategies.

- Input: a text file with one paper per line, as a title, link, DOI, or local PDF
- Output: `references.md`, `report.csv`, and `issues.csv`, plus the PDF, markdown,
  and features file of each paper
- Users: quant researchers, who sort the CSV to pick the next papers to replicate

## Structure of the Dir

| Directory | Description                                                                  |
| :-------- | :--------------------------------------------------------------------------- |
| `docs/`   | Requirements, design, decisions, plan, and progress docs, in numbered order  |
| `test/`   | Unit tests of the modules, and the fakes and helpers that they share         |

## Description of Files

Python and Markdown files, alphabetical by path. The status of the work is in
`research/Estimate_academic_paper_quality/docs/07-progress.md`.

| File                                  | Description                                                                      | Cluster      |
| :------------------------------------ | -------------------------------------------------------------------------------- | ------------ |
| `docs/01-prd.md`                      | Product requirements: inputs, features, outputs, non-goals, and success metrics  | Requirements |
| `docs/02-design.architecture.md`      | Overall design: components, import graph, data model, and requirements map       | Design       |
| `docs/02.1-design.pipeline.md`        | CLI, the four stages, the per-paper error loop, and the skip of finished work    | Design       |
| `docs/02.2-design.resolver.md`        | Line parsing, paper matching, dedup, reference building, and sorting             | Design       |
| `docs/02.3-design.bib_sources.md`     | Clients of the lookup APIs (Semantic Scholar, Crossref, arXiv, Unpaywall)        | Design       |
| `docs/02.4-design.fetcher.md`         | Download and convert each paper with the download tool, skipping finished papers | Design       |
| `docs/02.5-design.extractor.md`       | One LLM call per paper that returns features, facts, affiliations, and quotes    | Design       |
| `docs/02.6-design.rater.md`           | Paper quality rating from page length, affiliation, and citation count           | Design       |
| `docs/02.7-design.report.md`          | Writers and readers of the output files: references, CSVs, and features files    | Design       |
| `docs/03-decisions.md`                | Log of each design decision, with its reason, alternatives, and status           | Decisions    |
| `docs/04-plan.md`                     | Versions and PRs to build, in order, with the open questions of each component   | Plans        |
| `docs/07-progress.md`                 | What is done and learned: milestone status, findings, next tasks, and history    | Plans        |
| `epq_bib_sources.py`                  | Clients of Semantic Scholar, Crossref, arXiv, and Unpaywall, with a throttle     | Lookup APIs  |
| `epq_extractor.prompt.md`             | System prompt of the LLM that extracts the features of a paper                   | Extract      |
| `epq_extractor.py`                    | One LLM call per paper, with each quote checked against the PDF text             | Extract      |
| `epq_fetcher.py`                      | Run the download tool per paper, and read the PDF pages and the markdown         | Download     |
| `epq_pipeline.py`                     | Run the four stages on one input and one output dir, with a catch per paper      | Pipeline     |
| `epq_rater.py`                        | Rate a paper from its page count, its affiliations, and its citations            | Rate         |
| `epq_report.py`                       | Write and read the state files, `references.md`, the CSVs, and the features      | Report       |
| `epq_resolver.py`                     | Parse each input line, match it to papers, and build the sorted references       | Resolve      |
| `epq_types.py`                        | Frozen dataclasses shared by the modules, with no logic and no project imports   | Shared types |
| `estimate_paper_quality.py`           | CLI that turns a list of papers into a reference list and a CSV report           | Entry point  |
| `estimate_paper_quality.py.README.md` | Arguments, examples, outputs, and architecture of the CLI                        | Entry point  |
| `log.md`                              | Log of the steps that made the docs, and the original idea of the pipeline       | Log          |
| `paper_list.md`                       | Input list of papers, one per line, with authors in parentheses                  | Input        |
| `README.md`                           | Summary, structure, files, executables, and architecture of this dir             | Overview     |
| `test/epq_test_utils.py`              | Builders, a fake of the lookup APIs, a local HTTP server, and a stub tool        | Tests        |
| `test/test_epq_bib_sources.py`        | Lookup clients against a local HTTP server, with no internet                     | Tests        |
| `test/test_epq_extractor.py`          | Quote location, reference removal, text cut, and extraction with a fake LLM      | Tests        |
| `test/test_epq_fetcher.py`            | PDF readers, the skip of fetched papers, and the fetch through a stub tool       | Tests        |
| `test/test_epq_import_graph.py`       | Each module imports only the layers that the design allows                       | Tests        |
| `test/test_epq_pipeline.py`           | The four stages end to end, on a local server, a stub tool, and a fake LLM       | Tests        |
| `test/test_epq_rater.py`              | Score tables and rating, checked against `.claude/skills/paper.rules.md`         | Tests        |
| `test/test_epq_report.py`             | Writers and readers of the state, the features file, and the CSVs                | Tests        |
| `test/test_epq_resolver.py`           | Line parsing, title matching, paper building, merging, and resolving of lines    | Tests        |

## Description of Executables

The only executable is
`research/Estimate_academic_paper_quality/estimate_paper_quality.py`. Its arguments,
its output files, and its exit codes are in
`research/Estimate_academic_paper_quality/estimate_paper_quality.py.README.md`.

## `estimate_paper_quality.py`

### What It Does

- Runs the stages `resolve`, `download`, `extract`, and `report` on `--input`, and
  writes all the outputs to `--output_dir`
- Skips the work of a previous run on the same dir: the converted papers, the papers
  with a features file, and the cached answers of the APIs
- Calls an LLM only in the stage `extract`, and reads the Semantic Scholar key from
  `SEMANTIC_SCHOLAR_API_KEY`

### Examples

- Normalize the list only, with no download and no LLM call:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.md \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action resolve --action report
  ```

- Run all the stages, with Unpaywall links:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.md \
      --output_dir ~/papers/run_2026_10 \
      --email me@umd.edu
  ```

- Extract again the features of all the papers, with another model:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.md \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action extract --action report \
      --force_extract \
      --model <model>
  ```

## Description of Workflows

Check the list, then run the whole pipeline. The commands use the `--input` and the
`--output_dir` of the examples above.

1. Normalize the list, and read the issues:
   ```bash
   > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
       --input research/Estimate_academic_paper_quality/paper_list.md \
       --output_dir ~/papers/run_2026_10 \
       --clear_actions --action resolve --action report
   ```
   Open `issues.csv` in the output dir. Each row is a line that was not resolved or
   that matched more than one paper, with its candidates.

2. Fix the lines that `issues.csv` names in
   `research/Estimate_academic_paper_quality/paper_list.md`, then run step 1 again.
   The rerun reuses the cached answers of the APIs.

3. Run all the stages, with `--email` set. The stage `download` needs Docker, and the
   SSRN papers also need Chrome.
   ```bash
   > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
       --input research/Estimate_academic_paper_quality/paper_list.md \
       --output_dir ~/papers/run_2026_10 \
       --email me@umd.edu
   ```

4. After a change of the LLM or of the prompt file, extract again and rewrite the
   reports:
   ```bash
   > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
       --input research/Estimate_academic_paper_quality/paper_list.md \
       --output_dir ~/papers/run_2026_10 \
       --clear_actions --action extract --action report \
       --force_extract \
       --model <model>
   ```

## Description of Architecture

- The design is in
  `research/Estimate_academic_paper_quality/docs/02-design.architecture.md`
- The layers, from the top down:
  - Script: `research/Estimate_academic_paper_quality/estimate_paper_quality.py`
    parses the flags and picks the stages
  - Orchestration: `research/Estimate_academic_paper_quality/epq_pipeline.py` runs
    each stage, with a catch per paper
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
      state and the outputs
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
