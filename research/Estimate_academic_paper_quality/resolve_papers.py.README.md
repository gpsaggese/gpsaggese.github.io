# `resolve_papers.py`

- Turns a CSV of loose paper names into a CSV of resolved, deduped papers
- Input: a CSV with a column `input`, one paper per row: a title with the authors in
  parentheses, an arXiv or SSRN link, a DOI, a publisher link, or the path of a local
  PDF
- Output: a papers CSV, one row per paper from the newest to the oldest, and an issues
  CSV, one row per input that gave no paper
- Run it first: its papers CSV is the input of `download_papers.py`,
  `extract_features.py`, and `report_papers.py`. The script `estimate_paper_quality.sh`
  runs all the stages in order

## Examples

- Resolve a list of papers:
  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output ~/papers/run_2026_10/papers.csv
  ```

- Show how each row is classified, with no lookup and no output file:
  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output ~/papers/run_2026_10/papers.csv \
      --dry_run
  ```

- Add Unpaywall links, and use the Semantic Scholar key for a higher rate limit:
  ```bash
  > export SEMANTIC_SCHOLAR_API_KEY=<key>
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output ~/papers/run_2026_10/papers.csv \
      --email me@umd.edu
  ```

- Look up all the papers again, ignoring the cached answers of the APIs:
  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output ~/papers/run_2026_10/papers.csv \
      --cache_mode REFRESH_CACHE
  ```

## Configuration & Inputs

### Command-line Arguments

| Argument          | Type | Default                         | Description                                                                           |
| :---------------- | :--- | :------------------------------ | :------------------------------------------------------------------------------------ |
| `-i`, `--input`   | str  | required                        | Input CSV with a column `input`, one paper per row                                    |
| `-o`, `--output`  | str  | required                        | Papers CSV to write                                                                   |
| `--issues_output` | str  | `<output>` with `.issues.csv`   | Issues CSV to write, e.g., `papers.csv` gives `papers.issues.csv`                     |
| `--cache_dir`     | str  | `cache/` next to the papers CSV | Dir of the cached answers of the APIs                                                 |
| `--email`         | str  | `""`                            | Contact email for Unpaywall: empty skips Unpaywall                                    |
| `--dry_run`       | flag | false                           | Show how each row is classified, with no lookup and no output file                    |
| `--cache_mode`    | str  | `""`                            | One of `REFRESH_CACHE`, `DISABLE_CACHE`, `HIT_CACHE_OR_ABORT`: applies to API answers |
| `--cache_debug`   | flag | false                           | Log at WARNING level each cache hit, miss, and refresh                                |
| `-v`              | str  | `INFO`                          | Logging level: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`            |

### Input CSV Format

- The column `input` is required, and the other columns are ignored
- One paper per row: a title with the authors in parentheses, e.g.,
  `Lazy Prices (Cohen; Malloy; Nguyen)`, an arXiv or SSRN link, a DOI, a publisher
  link, or the path of a local PDF
- The number of a row is its place after the header, counted from 1: it is the
  `line_no` of the issues
- An empty row and a row that starts with `#` or `//` are skipped
- A CSV from a spreadsheet works: a BOM at the start of the file is skipped
- A list in markdown, one bullet per paper, must become a CSV first
- Example: [`paper_list.csv`](/research/Estimate_academic_paper_quality/paper_list.csv)

### Environment Variables

- `SEMANTIC_SCHOLAR_API_KEY`: key of Semantic Scholar, sent as `x-api-key`
  - Default: empty, so the requests use the shared pool, which answers `429` often

### Other Requirements

- The first line of the file selects its dependencies with `uv run`
- The resolver does not use an LLM, Docker, or Chrome

## Output & Side Effects

### Files Created

- The papers CSV (`--output`): one row per paper, newest first, with the columns:
  - `reference`: the reference in the format of `.claude/skills/references.rules.md`
  - `year`: publication year, `0` if unknown
  - `authors`: authors as `Family, Given`, joined by `|`
  - `title`: title of the canonical record
  - `doi`, `arxiv_id`, `s2_paper_id`: identifiers of the paper, empty if unknown
  - `source`: source of the canonical record, one of `semantic_scholar`, `crossref`,
    `arxiv`
  - `base_name`: base name of the PDF and markdown files, `<Year>.<Author>.<Title>`
  - `download_inputs`: ordered inputs for the download tool, joined by `|`
  - `line_nos`: rows of the input CSV that gave this paper, joined by `|`
  - `key`, `aliases`: identity of the paper, used to merge the rows of one paper
- The issues CSV (`--issues_output`): one row per input that gave no paper, with the
  columns `line_no`, `input`, `base_name`, `stage`, `status`, `reason`, and
  `candidates` (joined by `|`)
  - `stage` is `resolve`, and `status` is `unresolved` or `ambiguous`
- The cache of the API answers, in `--cache_dir`
- Both CSV files are written again on each run, and a paper that is not in the input
  CSV anymore leaves them

### Exit Codes

- `0`: Success. Rows that do not resolve do not change it: they are rows of the issues
  CSV
- `1`: An assertion or an exception stopped the run, e.g.:
  - `--input` is not an existing file
  - `--input` has no column `input`
- `2`: A missing required argument or an invalid flag

### Logging

- Logs go to the console, at the level of `-v`, and to a file
  `research/Estimate_academic_paper_quality/resolve_papers.py.log`, which each run
  overwrites
- The run logs a progress bar, a warning for each failed row or failed API, and the
  numbers of papers and issues

## Software Architecture

- The design of the resolver is in
  [`02.2-design.resolver.md`](/research/Estimate_academic_paper_quality/docs/02.2-design.resolver.md)
- This file is the entry point and the orchestration, and it also holds the matching:
  no module of the project has the resolver code (decision `D28`)

### Data Flow

1. **Input**: `_parse()` reads the flags, and `_main()` resolves the paths, then reads
   the rows of the input CSV
2. **Classification**: `_read_input_lines()` turns each row into an `InputLine`, with a
   kind of `local_pdf`, `ssrn`, `doi`, `arxiv`, `url`, or `title`
3. **Lookup and match**: `_resolve_line()` looks the row up in Semantic
   Scholar, Crossref, arXiv, and Unpaywall, and keeps one paper or makes an issue
4. **Merge**: `_finalize_papers()` merges the rows of one paper and sorts the papers
   from the newest to the oldest
5. **Output**: `epq_report` writes the papers CSV and the issues CSV

### Key Functions

- `_parse() -> argparse.ArgumentParser`
  - Defines the flags, with the helpers for the cache mode and `-v`
- `_main(parser) -> None`
  - Resolves the paths, builds the API clients, and calls `_resolve()`: with
    `--dry_run` it only logs the kind of each row
- `_read_input_lines(input_csv) -> List[InputLine]`
  - Reads the column `input` and classifies each row with `_parse_line()`
- `_resolve_line(input_line, sources) -> ResolveResult`
  - Looks one row up, matches it with `_match_candidates()`, and builds the `Paper`
    with `_build_paper()`, or an `Issue`
- `_finalize_papers(papers) -> List[Paper]`
  - Merges the rows of one paper with `_merge_papers()` and sorts them with
    `_sort_papers()`
- `_resolve(input_csv, papers_csv, issues_csv, sources) -> ResolveResult`
  - Resolves each row with a catch per row, then merges and sorts the papers, and
    writes the two CSV files
- `epq_report.write_papers_csv()`, `read_papers_csv()`, `read_issues_csv()`
  - Write and read the two CSV files, so that the next executables read them

### Design Patterns

- **CSV in, CSV out**: the resolver shares no state dir with the other stages, so a
  person can fix the papers CSV by hand before the download (decision `D26`)
- **Catch per item**: a failed row becomes a row of the issues CSV and the run goes on
  (decision `D26`)
- **Injected dependencies**: `BibSources` is an argument of `_resolve()`, so the
  tests use a fake API
- **Disk cache**: the API answers are cached in `--cache_dir`, so a rerun on the same
  list makes no request
