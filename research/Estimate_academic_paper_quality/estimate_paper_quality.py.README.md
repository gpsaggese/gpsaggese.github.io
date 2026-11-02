# `estimate_paper_quality.py`

- Turns a loose list of papers into a clean reference list, the PDF and markdown of
  each paper, and a CSV report of features and a Paper quality rating per paper
- Input: a text file with one paper per line (title, link, DOI, or local PDF)
- Output: `references.md`, `report.csv`, `issues.csv`, and `papers/`, in
  `--output_dir`
- Run it to pick the next papers to replicate, and rerun it on the same dir to pay
  only for the new or changed papers

## Examples

- Normalize the list only, with no download and no LLM call:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.md \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action resolve --action report
  ```

- Run all the stages, with Unpaywall links and the Semantic Scholar key:
  ```bash
  > export SEMANTIC_SCHOLAR_API_KEY=<key>
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.md \
      --output_dir ~/papers/run_2026_10 \
      --email me@umd.edu
  ```

- Rewrite the CSV files from the state of a previous run, with no network call:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.md \
      --output_dir ~/papers/run_2026_10 \
      --only_action report
  ```

- Extract again the features of all the papers with another LLM:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.md \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action extract --action report \
      --force_extract \
      --model <model>
  ```

- Look up all the papers again, ignoring the cached answers of the APIs:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.md \
      --output_dir ~/papers/run_2026_10 \
      --clear_actions --action resolve --action report \
      --cache_mode REFRESH_CACHE
  ```

## Configuration & Inputs

### Command-line Arguments

| Argument                                | Type | Default              | Description                                                                           |
| :-------------------------------------- | :--- | :------------------- | :------------------------------------------------------------------------------------ |
| `--input`                               | str  | required             | Text file with one paper per line                                                     |
| `--output_dir`                          | str  | required             | Dir of all the outputs and of the state of the run                                    |
| `--email`                               | str  | `""`                 | Contact email for Unpaywall: empty skips Unpaywall                                    |
| `--model`                               | str  | `""`                 | LLM of the stage `extract`: empty is the default model of `helpers.hllm`              |
| `--force_extract`, `--no_force_extract` | flag | `--no_force_extract` | Extract again the papers that have a features file, and refresh their citation count  |
| `--max_chars`                           | int  | `300000`             | Cap on the characters of markdown sent to the LLM                                     |
| `--cache_mode`                          | str  | `""`                 | One of `REFRESH_CACHE`, `DISABLE_CACHE`, `HIT_CACHE_OR_ABORT`: applies to API answers |
| `--cache_debug`                         | flag | false                | Log at WARNING level each cache hit, miss, and refresh                                |
| `--action`                              | str  | all four stages      | Add a stage to the list of stages to run (repeatable)                                 |
| `--skip_action`                         | str  | none                 | Remove a stage from the list of stages to run (repeatable)                            |
| `--only_action`                         | str  | none                 | Run only this stage: incompatible with the other action flags (repeatable)            |
| `--all_actions`                         | flag | false                | Start from all the stages                                                             |
| `--clear_actions`                       | flag | false                | Start from no stage, then add the stages with `--action`                              |
| `-v`                                    | str  | `INFO`               | Logging level: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`            |
| `--no_report_command_line`              | flag | false                | Do not log the command line                                                           |

- The stages are `resolve`, `download`, `extract`, and `report`, run in this order
- A stage that does not run reads the output of the stage before it from
  `<output_dir>/state/`: run `resolve` once before the other stages

### Input File Format

- One paper per line: a title with the authors in parentheses, e.g.,
  `Lazy Prices (Cohen; Malloy; Nguyen)`, an arXiv or SSRN link, a DOI, a publisher
  link, or the path of a local PDF
- A bullet that is wrapped on several lines counts as one entry
- An empty line and a comment line (`#`, `//`) are skipped
- Example: [`paper_list.md`](/research/Estimate_academic_paper_quality/paper_list.md)

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
  - `papers.json`: the papers of the stage `resolve`
  - `issues.resolve.json`, `issues.download.json`, and `issues.extract.json`: the
    issues of each stage
  - `cache/http/<sha1>.json`: the cached answer of each API request
  - `cache/url_pdfs/<sha1>.pdf`: PDFs that a `url` line pointed to
  - `logs/<BASE>.download.log`: the log of the download tool for each paper
- A rerun skips a paper that has its PDF and markdown, and a paper that has a
  features file, unless `--force_extract` is set

### Exit Codes

- `0`: Success. Papers that fail do not change it: they are rows of `issues.csv`
- `1`: An assertion or an exception stopped the run, e.g.:
  - `--input` is not an existing file
  - A stage other than `resolve` ran with no `state/papers.json`
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
   checks that `--input` exists, and creates `papers/` and `state/cache/`
2. **Stage selection**: `hselect_action` gives the list of stages, and the loop with
   `mark_action()` runs them in order
3. **`resolve`**: the lines of `--input` become `Paper` objects and `Issue` objects,
   from Semantic Scholar, Crossref, arXiv, and Unpaywall, then are deduped and sorted
   by year; the result goes to `state/papers.json` and `state/issues.resolve.json`
4. **`download`**: for each paper not on disk, the download tool saves
   `papers/<BASE>.pdf` and `papers/<BASE>.md`; a failure becomes an issue
5. **`extract`**: for each markdown file without a features file, one LLM call
   returns the features, facts, and affiliations, each with a quote checked
   against the PDF text; the rater adds the Paper quality rating from the page
   count, the affiliation, and the citation count
6. **`report`**: `references.md`, `report.csv`, and `issues.csv` are written from
   `papers.json`, the features files, and the issues of every stage

### Key Functions

- `_parse() -> argparse.ArgumentParser`
  - Defines the flags, with the helpers for the cache mode, the stages, and `-v`
- `_main(parser) -> None`
  - Builds the config, the API clients, and the stage list, then calls the `run_*`
    function of each stage
- `epq_pipeline.run_resolve(config, sources) -> ResolveResult`
  - Resolves each line with a catch per line, then merges and sorts the papers
- `epq_pipeline.run_download(config, papers, *, tool_path, log_level) -> List[Issue]`
  - Skips the converted papers, and turns each failed fetch into an issue
- `epq_pipeline.run_extract(config, papers, *, llm_func, cost_tracker, sources,
  tiers) -> List[Issue]`
  - Extracts, rates, and writes the features file of each paper, with a catch per
    paper
- `epq_pipeline.run_report(config, papers) -> None`
  - Reads the state of the other stages and writes the three final files
- `epq_pipeline.read_state_papers(config) -> List[Paper]`
  - Gives the papers to a stage that runs without `resolve`

### Design Patterns

- **Stages with a shared state dir**: every stage reads and writes files in
  `--output_dir`, so any stage can run alone and a rerun resumes
- **Catch per item**: a failed line or paper becomes a row of `issues.csv` and the
  run goes on (decision `D21`)
- **Injected dependencies**: `BibSources` and `llm_func` are arguments of the
  stages, so the tests use a fake API and a fake LLM
- **Lazy import**: `helpers.hllm` is imported only when `extract` runs, since the
  other stages do not need the LLM client libraries
- **Disk caches**: the API answers are cached in `state/cache/`, and the features
  files are the cache of `extract`
