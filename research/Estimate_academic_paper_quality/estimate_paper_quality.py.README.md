# `estimate_paper_quality.py`

- Turns a CSV of resolved papers into the PDF and markdown of each paper, and a CSV
  report of features and a Paper quality rating per paper
- Input: the papers CSV that `resolve_papers.py` writes from a loose list of papers
  (title, link, DOI, or local PDF)
- Output: `references.md`, `report.csv`, `issues.csv`, and `papers/`, in
  `--output_dir`
- Run it to pick the next papers to replicate, and rerun it on the same dir to pay
  only for the new or changed papers

## Examples

- Resolve the list of papers first, with the executable `resolve_papers.py`:
  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output ~/papers/run_2026_10/papers.csv
  ```

- Write the reference list only, with no download and no LLM call:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action report
  ```

- Run all the stages, with the Semantic Scholar key:
  ```bash
  > export SEMANTIC_SCHOLAR_API_KEY=<key>
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10
  ```

- Rewrite the CSV files from the state of a previous run, with no network call:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --only_action report
  ```

- Extract again the features of all the papers with another LLM:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action extract --action report \
      --force_extract \
      --model <model>
  ```

- Ask again for the citation counts, ignoring the cached answers of the APIs:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action extract --action report \
      --cache_mode REFRESH_CACHE
  ```

## Configuration & Inputs

### Command-line Arguments

| Argument                                | Type | Default              | Description                                                                                                |
| :-------------------------------------- | :--- | :------------------- | :--------------------------------------------------------------------------------------------------------- |
| `-i`, `--input`                         | str  | required             | Papers CSV that `resolve_papers.py` wrote                                                                  |
| `--resolve_issues`                      | str  | `""`                 | Issues CSV that `resolve_papers.py` wrote: empty is the papers CSV with `.issues.csv`, if that file exists |
| `--output_dir`                          | str  | required             | Dir of all the outputs and of the state of the run                                                         |
| `--model`                               | str  | `""`                 | LLM of the stage `extract`: empty is the default model of `helpers.hllm`                                   |
| `--force_extract`, `--no_force_extract` | flag | `--no_force_extract` | Extract again the papers that have a features file, and refresh their citation count                       |
| `--max_chars`                           | int  | `300000`             | Cap on the characters of markdown sent to the LLM                                                          |
| `--cache_mode`                          | str  | `""`                 | One of `REFRESH_CACHE`, `DISABLE_CACHE`, `HIT_CACHE_OR_ABORT`: applies to API answers                      |
| `--cache_debug`                         | flag | false                | Log at WARNING level each cache hit, miss, and refresh                                                     |
| `--action`                              | str  | all three stages     | Add a stage to the list of stages to run (repeatable)                                                      |
| `--skip_action`                         | str  | none                 | Remove a stage from the list of stages to run (repeatable)                                                 |
| `--only_action`                         | str  | none                 | Run only this stage: incompatible with the other action flags (repeatable)                                 |
| `--all_actions`                         | flag | false                | Start from all the stages                                                                                  |
| `--clear_actions`                       | flag | false                | Start from no stage, then add the stages with `--action`                                                   |
| `-v`                                    | str  | `INFO`               | Logging level: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`                                 |
| `--no_report_command_line`              | flag | false                | Do not log the command line                                                                                |

- The stages are `download`, `extract`, and `report`, run in this order
- The stage `resolve` is the separate executable `resolve_papers.py`: run it once
  before this script
- A stage that does not run reads the output of the stage before it from
  `<output_dir>/`, e.g., `report` reads the features files of `extract`

### Input File Format

- `--input` is the papers CSV that `resolve_papers.py` writes, with the columns
  `reference`, `year`, `authors`, `title`, `doi`, `arxiv_id`, `s2_paper_id`, `source`,
  `base_name`, `download_inputs`, `line_nos`, `key`, and `aliases`: see
  [`resolve_papers.py.README.md`](/research/Estimate_academic_paper_quality/resolve_papers.py.README.md)
- A person can fix or delete a row of the papers CSV before the run, e.g., to add a
  `download_inputs` link or the path of a local PDF
- The issues CSV of `resolve_papers.py` goes to `issues.csv`: it lists the inputs that
  gave no paper. Without it, `issues.csv` has no unresolved line

### Environment Variables

- `SEMANTIC_SCHOLAR_API_KEY`: key of Semantic Scholar, sent as `x-api-key`
  - Default: empty, so the requests use the shared pool, which answers `429` often
- `OPENAI_API_KEY`: key of the default provider of `helpers.hllm`, needed by the
  stage `extract`
- `OPENROUTER_API_KEY`: key for a model that `helpers.hllm` serves through OpenRouter

### Other Requirements

- The stage `download` runs the download tool in `helpers_root`, which needs Docker
  and, for SSRN papers, Chrome
- The first line of the file selects its dependencies with `uv run`
- The affiliation tier list
  [`affiliation_tiers.yaml`](/research/Estimate_academic_paper_quality/affiliation_tiers.yaml)
  is empty until a person fills it: an affiliation then scores by its kind

## Output & Side Effects

### Files Created

- Final outputs, in `<output_dir>`:
  - `references.md`: one bullet per paper, from the newest to the oldest
  - `report.csv`: one row per paper with a features file, with the columns
// TODO(ai_gp): Make into a bullet list with an example per field
    `reference`, `paper_quality`, `performance`, `recency`, `asset_class`,
    `theoretical_domain`, `mechanism`, `page_length_score`, `affiliation_score`,
    `citations_score`, and `base_name`
  - `issues.csv`: one row per line or paper to fix, with the columns `line_no`,
    `input`, `base_name`, `stage`, `status`, `reason`, and `candidates`
- Files of each paper, in `<output_dir>/papers/`:
  - `<BASE>.pdf` and `<BASE>.md`, where `<BASE>` is `<Year>.<Author>.<Title>`
  - `<BASE>.features.json`: features, facts, quotes, and rating, with a
    `schema_version`
- State, in `<output_dir>/state/`:
  - `issues.download.json` and `issues.extract.json`: the issues of each stage
  - `cache/http/<sha1>.json`: the cached answer of each API request
  - `cache/url_pdfs/<sha1>.pdf`: PDFs that a `url` line pointed to
  - `logs/<BASE>.download.log`: the log of the download tool for each paper
- A rerun skips a paper that has its PDF and markdown, and a paper that has a
  features file, unless `--force_extract` is set

### Exit Codes

- `0`: Success. Papers that fail do not change it: they are rows of `issues.csv`
- `1`: An assertion or an exception stopped the run, e.g.:
  - `--input` is not an existing file, or `--resolve_issues` is set and is not one
  - `--input` is not a papers CSV
- `2`: A missing required argument or an invalid flag

### Logging

- Logs go to the console, at the level of `-v`, and to a file
  `research/Estimate_academic_paper_quality/estimate_paper_quality.py.log`, which
  each run overwrites
- The run logs the stages to run, a progress bar for each stage, a warning for each
  failed line or paper, and the LLM cost of each paper and of the run

## Software Architecture

- The overall design is in
  [`02-design.architecture.md`](/research/Estimate_academic_paper_quality/docs/02-design.architecture.md)
- This file is only the entry point: the work is in `epq_pipeline.py`

### Data Flow

1. **Input**: `_parse()` reads the flags, and `_main()` builds a `PipelineConfig`,
   checks that `--input` exists, creates `papers/` and `state/cache/`, and reads the
   `Paper` objects of the papers CSV
2. **Stage selection**: `hselect_action` gives the list of stages, and the loop with
   `mark_action()` runs them in order
3. **`download`**: for each paper not on disk, the download tool saves
   `papers/<BASE>.pdf` and `papers/<BASE>.md`; a failure becomes an issue
4. **`extract`**: for each markdown file without a features file, one LLM call
   returns the features, facts, and affiliations, each with a quote checked
   against the PDF text; the rater adds the Paper quality rating from the page
   count, the affiliation, and the citation count
5. **`report`**: `references.md`, `report.csv`, and `issues.csv` are written from
   the papers CSV, the features files, the issues CSV of `resolve_papers.py`, and
   the issues of the other stages

### Key Functions

- `_parse() -> argparse.ArgumentParser`
  - Defines the flags, with the helpers for the cache mode, the stages, and `-v`
- `_main(parser) -> None`
  - Builds the config, the API clients, and the stage list, then calls the `run_*`
    function of each stage
- `epq_pipeline.run_download(config, papers, *, tool_path, log_level) -> List[Issue]`
  - Skips the converted papers, and turns each failed fetch into an issue
- `epq_pipeline.run_extract(config, papers, *, llm_func, cost_tracker, sources,
  tiers) -> List[Issue]`
  - Extracts, rates, and writes the features file of each paper, with a catch per
    paper
- `epq_pipeline.run_report(config, papers) -> None`
  - Reads the issues CSV of the resolver and the state of the other stages, and writes
    the three final files
- `epq_pipeline.read_papers(config) -> List[Paper]`
  - Reads the papers CSV that `resolve_papers.py` wrote

### Design Patterns

- **CSV between the resolver and the pipeline**: the papers CSV and the issues CSV
  are the interface between the two executables (decision `D26`)
- **Stages with a shared state dir**: every stage reads and writes files in
  `--output_dir`, so any stage can run alone and a rerun resumes
- **Catch per item**: a failed paper becomes a row of `issues.csv` and the run goes
  on (decision `D21`)
- **Injected dependencies**: `BibSources` and `llm_func` are arguments of the
  stages, so the tests use a fake API and a fake LLM
- **Lazy import**: `helpers.hllm` is imported only when `extract` runs, since the
  other stages do not need the LLM client libraries
- **Disk caches**: the API answers of the citation counts are cached in
  `state/cache/`, and the features files are the cache of `extract`
