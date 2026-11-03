# `report_papers.py`

- Rates the quality of each extracted paper, and writes the final report
- Input: the papers CSV that `resolve_papers.py` writes, and the features files that
  `extract_features.py` writes
- Output: `papers/<BASE>.rating.json` for each paper, and `report.csv`,
  `references.md`, and `issues.csv` in `--output_dir`
- Run it last. It has two stages, `rate` and `report`. The stage `rate` is free: it
  makes no LLM call, so a new tier list or a fresh citation count needs only a rerun
  of this script

## Examples

- Rate the papers and write the report:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10
  ```

- Write the report from the existing rating files, with no lookup:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --only_action report
  ```

- Rate the papers again after a change of `affiliation_tiers.yaml`:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10
  ```

- Ask again for the citation counts, with the Semantic Scholar key:
  ```bash
  > export SEMANTIC_SCHOLAR_API_KEY=<key>
  > research/Estimate_academic_paper_quality/report_papers.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --refresh_citations
  ```

- Show the papers to rate and the files to write, with no lookup and no output file:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --dry_run
  ```

## Configuration & Inputs

### Command-line Arguments

| Argument                                          | Type | Default                  | Description                                                                                                |
| :------------------------------------------------ | :--- | :----------------------- | :--------------------------------------------------------------------------------------------------------- |
| `-i`, `--input`                                   | str  | required                 | Papers CSV that `resolve_papers.py` wrote                                                                  |
| `--resolve_issues`                                | str  | `""`                     | Issues CSV that `resolve_papers.py` wrote: empty is the papers CSV with `.issues.csv`, if that file exists |
| `--output_dir`                                    | str  | required                 | Dir of all the outputs and of the state of the run                                                         |
| `--refresh_citations`, `--no_refresh_citations`   | flag | `--no_refresh_citations` | Ask for fresh citation counts in the stage `rate`                                                          |
| `--dry_run`                                       | flag | false                    | Show the papers to rate and the files to write, with no lookup and no output file                          |
| `--cache_mode`                                    | str  | `""`                     | One of `REFRESH_CACHE`, `DISABLE_CACHE`, `HIT_CACHE_OR_ABORT`: applies to API answers                      |
| `--cache_debug`                                   | flag | false                    | Log at WARNING level each cache hit, miss, and refresh                                                     |
| `--action`, `--skip_action`, `--only_action`      | str  | `rate`, `report`         | Select the stages to run (repeatable)                                                                      |
| `--all_actions`, `--clear_actions`                | flag | false                    | Start from all the stages, or from none                                                                    |
| `-v`                                              | str  | `INFO`                   | Logging level: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                                    |

- The stages are `rate` and `report`, run in this order. A stage that does not run
  reads the output of the other from `<output_dir>/`

### Environment Variables

- `SEMANTIC_SCHOLAR_API_KEY`: key of Semantic Scholar, sent as `x-api-key`
  - Default: empty, so the requests use the shared pool, which answers `429` often

### Other Requirements

- The affiliation tier list
  [`affiliation_tiers.yaml`](/research/Estimate_academic_paper_quality/affiliation_tiers.yaml)
  is empty until a person fills it: an affiliation then scores by its kind
- The first line of the file selects its dependencies with `uv run`
- The script does not use an LLM, Docker, or Chrome

## Output & Side Effects

### Files Created

- In `<output_dir>/papers/`: `<BASE>.rating.json` for each paper with a features file:
  the rating from 1 to 5 and the three signals behind it (`page_length`, `affiliation`,
  `citations`), each with its value, score, and note
- In `<output_dir>/state/`: `issues.rate.json`, the papers that failed to rate
- In `<output_dir>`:
  - `references.md`: one bullet per paper, from the newest to the oldest
  - `report.csv`: one row per paper with a features file and a rating file, with the
    columns:
    - `reference`: the reference of the paper, e.g.,
      `2020, Cohen et al., "Lazy Prices" (https://arxiv.org/pdf/2001.00001)`
    - `paper_quality`: the rating from 1 to 5, or `unknown`
    - `performance`: the reported performance, e.g., `Sharpe 1.4`
    - `recency`: the year of the paper, e.g., `2020`
    - `asset_class`: the asset class, e.g., `US equities`
    - `theoretical_domain`: the theoretical domain, or `unknown`
    - `mechanism`: the mechanism, e.g., `associational`
    - `page_length_score`, `affiliation_score`, `citations_score`: the three signals,
      each from 1 to 5, or `unknown`
    - `base_name`: the base name of the paper files, e.g.,
      `2020.Cohen_et_al.Lazy_Prices`
  - `issues.csv`: one row per line or paper to fix, with the columns `line_no`,
    `input`, `base_name`, `stage`, `status`, `reason`, and `candidates`. The stages
    are `resolve`, `download`, `extract`, and `rate`
- The stage `rate` rates all the papers at each run, since the tier list and the
  citation counts can change. The HTTP answers of the citation lookups are cached in
  `state/cache/`
- With `--dry_run` the script writes nothing

### Exit Codes

- `0`: Success. Papers that fail do not change it: they are rows of `issues.csv`
- `1`: An assertion or an exception stopped the run, e.g., `--input` is not an
  existing file, or `--resolve_issues` is set and is not one
- `2`: A missing required argument or an invalid flag

### Logging

- Logs go to the console, at the level of `-v`, and to a file
  `research/Estimate_academic_paper_quality/report_papers.py.log`, which each run
  overwrites
- The run logs the stages to run, a progress bar for the stage `rate`, a warning for
  each failed paper, and the numbers of papers, rows, and issues

## Software Architecture

- The design of the rater is in
  [`02.6-design.rater.md`](/research/Estimate_academic_paper_quality/docs/02.6-design.rater.md),
  and its code is in this file, with no module of its own (decision `D28`). The design
  of the report is in
  [`02.7-design.report.md`](/research/Estimate_academic_paper_quality/docs/02.7-design.report.md)

### Data Flow

1. **Input**: `_main()` checks the inputs, finds the issues CSV of the resolver, and
   reads the `Paper` objects of the papers CSV
2. **Stage selection**: `hselect_action` gives the list of stages, and the loop with
   `mark_action()` runs them in order
3. **`rate`**: for each paper with a features file, the script asks for the citation
   count, and `_compute_rating()` scores the page count, the affiliations, and
   the citations. A failure becomes an issue
4. **`report`**: the script reads the features and rating files and the issues of
   the other stages, then writes `references.md`, `report.csv`, and `issues.csv`

### Key Functions

- `_parse() -> argparse.ArgumentParser`
  - Defines the flags, with the helpers for the cache mode, the stages, and `-v`
- `_main(parser) -> None`
  - Checks the inputs, builds the API client and the tier list for `rate`, and runs
    the stages
- `_rate(papers, output_dir, *, sources, tiers, today, refresh_citations, dry_run)
  -> List[Issue]`
  - Rates each paper with a catch per paper, and writes the issues of the stage
- `_report(papers, output_dir, resolve_issues_file, *, dry_run) -> None`
  - Collects the issues, reads the files of each paper, and writes the three outputs
- `_load_tiers(path) -> AffiliationTiers`
  - Loads the dated tier list from YAML with `_parse_tiers()`
- `_compute_rating(inputs, tiers) -> QualityRating`
  - Rates a paper with the mean of the known scores of `_score_page_length()`,
    `_score_affiliation()`, and `_score_citations()`, rounded half up

### Design Patterns

- **Files between stages**: `rate` writes the rating files and `report` reads them,
  so each stage runs alone. The script imports no other executable (decision `D27`)
- **Rating apart from extraction**: the features file holds the inputs of the rating
  (affiliations and page count), so a new tier list or band needs no new LLM call
- **Catch per item**: a failed paper becomes an issue and the run goes on (decision
  `D21`)
- **Injected dependencies**: `BibSources` is an argument of `_rate()`, so the tests use a
  fake API
