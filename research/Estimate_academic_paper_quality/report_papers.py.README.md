# `report_papers.py`

- Rates the quality of each extracted paper, and writes the final report
- Input: the papers file that
  `research/Estimate_academic_paper_quality/resolve_papers.py` writes, by default
  `01-resolve.paper_list.json` in `--output_dir`, and the features files that
  `research/Estimate_academic_paper_quality/extract_features.py` writes
- Output: `paper_rating/<BASE>.rating.json` for each paper, and `04-rate.issues.json`,
  `04-rate.summary.txt`, `04-report.report.json`, `04-report.references.md`, and
  `04-report.summary.txt` in `--output_dir`
- Run it last. It has two stages, `rate` and `report`. The stage `rate` is free: it
  makes no LLM call, so a new tier list or a fresh citation count needs only a rerun
  of this script

## Examples

- Rate the papers and write the report:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --output_dir ~/papers/run_2026_10
  ```

- Write the report from the existing rating files, with no lookup:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --output_dir ~/papers/run_2026_10 \
      --only_action report
  ```

- Rate the papers again after a change of
  `research/Estimate_academic_paper_quality/affiliation_tiers.yaml`:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --output_dir ~/papers/run_2026_10
  ```

- Ask again for the citation counts, with the Semantic Scholar key:
  ```bash
  > export SEMANTIC_SCHOLAR_API_KEY=<key>
  > research/Estimate_academic_paper_quality/report_papers.py \
      --output_dir ~/papers/run_2026_10 \
      --refresh_citations
  ```

- Show the papers to rate and the files to write, with no lookup and no output file:
  ```bash
  > research/Estimate_academic_paper_quality/report_papers.py \
      --output_dir ~/papers/run_2026_10 \
      --dry_run
  ```

## Configuration & Inputs

### Command-line Arguments

| Argument                                        | Type | Default                  | Description                                                                               |
| :---------------------------------------------- | :--- | :----------------------- | :---------------------------------------------------------------------------------------- |
| `-i`, `--input`                                 | str  | `""`                     | Papers file of the stage resolve: empty is `01-resolve.paper_list.json` in `--output_dir` |
| `--output_dir`                                  | str  | required                 | Dir of all the outputs and of the state of the run                                        |
| `--refresh_citations`, `--no_refresh_citations` | flag | `--no_refresh_citations` | Ask for fresh citation counts in the stage `rate`                                         |
| `--dry_run`                                     | flag | false                    | Show the papers to rate and the files to write, with no lookup and no output file         |
| `--cache_mode`                                  | str  | `""`                     | One of `REFRESH_CACHE`, `DISABLE_CACHE`, `HIT_CACHE_OR_ABORT`: applies to API answers     |
| `--cache_debug`                                 | flag | false                    | Log at WARNING level each cache hit, miss, and refresh                                    |
| `--action`, `--skip_action`, `--only_action`    | str  | `rate`, `report`         | Select the stages to run (repeatable)                                                     |
| `--all_actions`, `--clear_actions`              | flag | false                    | Start from all the stages, or from none                                                   |
| `-v`                                            | str  | `INFO`                   | Logging level: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                   |

- The stages are `rate` and `report`, run in this order. A stage that does not run
  reads the output of the other from `<output_dir>/`
- The stage `report` reads the issues of the other stages from
  `01-resolve.issues.json`, `02-download.issues.json`, `03-extract.issues.json`, and
  `04-rate.issues.json` in `--output_dir`. A missing file means that the stage did
  not run: it gives no issue, and for `01-resolve.issues.json` it also logs a warning

### Environment Variables

- `SEMANTIC_SCHOLAR_API_KEY`: key of Semantic Scholar, sent as `x-api-key`
  - Default: empty, so the requests use the shared pool, which answers `429` often

### Other Requirements

- The affiliation tier list
  [`research/Estimate_academic_paper_quality/affiliation_tiers.yaml`](/research/Estimate_academic_paper_quality/affiliation_tiers.yaml)
  is empty until a person fills it: an affiliation then scores by its kind
- The first line of the file selects its dependencies with `uv run`
- The script does not use an LLM, Docker, or Chrome

## Output & Side Effects

### Files Created

- The layout of the whole output dir is in the section "Output Dir" of
  [`research/Estimate_academic_paper_quality/README.md`](/research/Estimate_academic_paper_quality/README.md#output-dir)
- In `<output_dir>/paper_rating/`: `<BASE>.rating.json` for each paper with a features file:
  the rating from 1 to 5 and the three signals behind it (`page_length`, `affiliation`,
  `citations`), each with its value, score, and note
- `<output_dir>/04-rate.issues.json`: the papers that failed to rate
- `<output_dir>/04-rate.summary.txt` and `<output_dir>/04-report.summary.txt`: the
  summary of each stage, with the same text as the end of the log (the counts of
  papers, the time of the stage, the output files, and the path of the issues file). A rerun overwrites
  them. In the summary of `report`, the unit is the row of `04-report.report.json`:
  `Completed successfully` is the number of `ok` rows, `Failed` is the number of
  `failed` rows, and the issues are counted as `<stage>.<status>`. See the section "Summary of a Stage" in
  [`research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md)
- In `<output_dir>`:
  - `04-report.references.md`: one bullet per paper, from the newest to the oldest
  - `04-report.report.json`: the key `rows` is a list with one row per paper and one row
    per input line that gave no paper, and the file has a `schema_version`. The rows
    of the papers come first, from the newest to the oldest, then the other rows by
    line number. A row has the keys:
    - `line_nos`: the lines of the input list that gave the row, e.g., `[3, 17]` for a
      paper that two lines name
    - `status`: `ok` for a paper with a features file, a rating file, and no issue,
      and `failed` for any other row
    - `reference`: the reference of the paper with no link, e.g.,
      `2020, Cohen et al., "Lazy Prices"`, or the text of the line that gave no paper
    - `links`: the link of the reference, in parentheses, e.g.,
      `(https://arxiv.org/pdf/2001.00001)`, or `null` for a line that gave no paper
    - `paper_quality`: the rating from 1 to 5, or `null` if it is unknown
    - `performance`: the reported performance, e.g., `Sharpe 1.4`
    - `recency`: the year of the paper, e.g., `2020`
    - `asset_class`: the asset class, e.g., `US equities`
    - `theoretical_domain`: the theoretical domain, or `unknown`
    - `mechanism`: the mechanism, e.g., `associational`
    - `page_length_score`, `affiliation_score`, `citations_score`: the three signals,
      each a number from 1 to 5, or `null` if it is unknown
    - `base_name`: the base name of the paper files, e.g.,
      `2020.Cohen_et_al.Lazy_Prices`, or `null` for a line that gave no paper
    - `issue`: `null` if the row has none, or an object with the keys `stage`
      (`resolve`, `download`, `extract`, or `rate`), `status`, `reason`, and
      `candidates` (a list). It comes from the files `NN-<stage>.issues.json`. A paper
      with many issues has the one of the earliest stage, and the log has a warning
    - The values from `paper_quality` to `citations_score` are `null` if the paper has
      no features file or no rating file. A `failed` row with `issue` `null` and no
      values means that an earlier stage did not run for the paper
- The stage `rate` rates all the papers at each run, since the tier list and the
  citation counts can change. The HTTP answers of the citation lookups are cached in
  `state/cache/`
- With `--dry_run` the script writes nothing

### Exit Codes

- `0`: Success. Papers that fail do not change it: they are rows of
  `04-report.report.json` with `status` `failed`. A missing `01-resolve.issues.json`
  does not change it
- `1`: An assertion or an exception stopped the run, e.g., the papers file is not an
  existing file
- `2`: A missing required argument or an invalid flag

### Logging

- Logs go to the console, at the level of `-v`, and to a file
  `research/Estimate_academic_paper_quality/report_papers.py.log`, which each run
  overwrites
- The run logs the stages to run, a progress bar for the stage `rate`, a warning for
  each failed paper, a warning if `01-resolve.issues.json` is missing, and the
  summary of each stage

## Software Architecture

- The design of the rater is in
  [`research/Estimate_academic_paper_quality/docs/02.6-design.rater.md`](/research/Estimate_academic_paper_quality/docs/02.6-design.rater.md),
  and its code is in this file, with no module of its own (decision `D28`). The design
  of the report is in
  [`research/Estimate_academic_paper_quality/docs/02.7-design.report.md`](/research/Estimate_academic_paper_quality/docs/02.7-design.report.md)

### Data Flow

1. **Input**: `_main()` checks the inputs and reads the `Paper` objects of the papers
   file with `epqrepor.read_papers()`. The papers file is `--input`, or
   `epqpaths.get_papers_path()` if `--input` is empty
2. **Stage selection**: `hselect_action` gives the list of stages, and the loop with
   `mark_action()` runs them in order
3. **`rate`**: for each paper with a features file, the script asks for the citation
   count, and `_compute_rating()` scores the page count, the affiliations, and
   the citations. A failure becomes an issue
4. **`report`**: the script reads the features and rating files and the issues files
   `NN-<stage>.issues.json` of the other stages, then writes
   `04-report.references.md` and `04-report.report.json`, with one row for each
   paper and for each unresolved line

### Key Functions

- `_parse() -> argparse.ArgumentParser`
  - Defines the flags, with the helpers for the cache mode, the stages, and `-v`
- `_main(parser) -> None`
  - Checks the inputs, builds the API client and the tier list for `rate`, and runs
    the stages
- `_rate(papers, output_dir, *, sources, tiers, today, refresh_citations, dry_run)
  -> List[Issue]`
  - Rates each paper with a catch per paper, and writes the issues of the stage to
    `04-rate.issues.json`
- `_read_state_issues(output_dir, stage) -> List[Issue]`
  - Reads the issues file of a stage with `epqrepor.read_issues()`: empty if the
    stage did not run
- `_report(papers, output_dir, *, dry_run) -> None`
  - Collects the issues of the four stages with `_read_state_issues()`, reads the
    files of each paper, builds the rows with `epqrepor.build_report_rows()`, and
    writes the two outputs
- `epqpaths.get_issues_path()`, `get_references_path()`, `get_report_path()`
  - Name the files of the stages, so that the writer and the reader of a file agree
    on the path
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
