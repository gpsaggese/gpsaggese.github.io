# `resolve_papers.py`

- Turns a CSV of loose paper names into a JSON list of resolved, deduped papers
- Input: a CSV with one paper per row, with no header: a title with the authors in
  parentheses, an arXiv or SSRN link, a DOI, a publisher link, or the path of a local
  PDF
- Output: a papers file, one item per paper from the newest to the oldest, and an
  issues file, one item per input that gave no paper. The script also copies the
  input CSV, so that `--output_dir` keeps the list that the run used
- Run it first: its papers file is the input of
  `research/Estimate_academic_paper_quality/download_papers.py`,
  `research/Estimate_academic_paper_quality/extract_features.py`, and
  `research/Estimate_academic_paper_quality/report_papers.py`. The script
  `research/Estimate_academic_paper_quality/estimate_paper_quality.py` runs all the
  stages in order

## Examples

- Resolve a list of papers:
  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10
  ```

- Show how each row is classified, with no lookup and no output file:
  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10 \
      --dry_run
  ```

- Add Unpaywall links, and use the Semantic Scholar key for a higher rate limit:
  ```bash
  > export SEMANTIC_SCHOLAR_API_KEY=<key>
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10 \
      --email me@umd.edu
  ```

- Look up all the papers again, ignoring the cached answers of the APIs:
  ```bash
  > research/Estimate_academic_paper_quality/resolve_papers.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10 \
      --cache_mode REFRESH_CACHE
  ```

## Configuration & Inputs

### Command-line Arguments

| Argument        | Type | Default  | Description                                                                           |
| :-------------- | :--- | :------- | :------------------------------------------------------------------------------------ |
| `-i`, `--input` | str  | required | Input CSV with one paper per row, no header needed                                    |
| `--output_dir`  | str  | required | Dir of all the outputs and of the state of the run                                    |
| `--cache_dir`   | str  | `""`     | Dir of the cache of the API answers: empty is `state/cache/` in `--output_dir`        |
| `--email`       | str  | `""`     | Contact email for Unpaywall: empty skips Unpaywall                                    |
| `--dry_run`     | flag | false    | Show how each row is classified, with no lookup and no output file                    |
| `--cache_mode`  | str  | `""`     | One of `REFRESH_CACHE`, `DISABLE_CACHE`, `HIT_CACHE_OR_ABORT`: applies to API answers |
| `--cache_debug` | flag | false    | Log at WARNING level each cache hit, miss, and refresh                                |
| `-v`            | str  | `INFO`   | Logging level: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`            |

- The script takes `--output_dir` only: it writes several files, so it infers their
  names (see "Files Created")
- Flag abbreviations are off, so `--output` is an error and is not read as
  `--output_dir`

### Input CSV Format

- The header is optional. With no header, the text of each row is one entry: the
  cells of the row are joined back with `,`, so a title with a comma stays whole
- A header with a column `input` is also accepted. Then the entry is that column, and
  the other columns are ignored
- One paper per row: a title with the authors in parentheses, e.g.,
  `Lazy Prices (Cohen; Malloy; Nguyen)`, an arXiv or SSRN link, a DOI, a publisher
  link, or the path of a local PDF
- The number of a row is its place after the header (or in the file, with no header),
  counted from 1: it is the `line_no` of the issues
- An empty row and a row that starts with `#` or `//` are skipped
- A CSV from a spreadsheet works: a BOM at the start of the file is skipped
- A list in markdown, one bullet per paper, must become a CSV first
- Example: [`research/Estimate_academic_paper_quality/paper_list.csv`](/research/Estimate_academic_paper_quality/paper_list.csv)

### Environment Variables

- `SEMANTIC_SCHOLAR_API_KEY`: key of Semantic Scholar, sent as `x-api-key`
  - Default: empty, so the requests use the shared pool, which answers `429` often

### Other Requirements

- The first line of the file selects its dependencies with `uv run`
- The resolver does not use an LLM, Docker, or Chrome

## Output & Side Effects

### Files Created

- The layout of the whole output dir is in the section "Output Dir" of
  [`research/Estimate_academic_paper_quality/README.md`](/research/Estimate_academic_paper_quality/README.md#output-dir)
- `<output_dir>/00-paper_list.csv`: a copy of the input CSV
  - An earlier copy is overwritten, since a rerun can use a new list
  - The copy is skipped if `--input` is already that file, e.g., a rerun on the copy
- `<output_dir>/01-resolve.paper_list.json`: the papers file, with a `schema_version`
  and a list `papers`, one object per paper, newest first, with the fields:
  - `reference`: the reference in the format of `.claude/skills/references.rules.md`
  - `year`: publication year, `0` if unknown
  - `authors`: authors as `Family, Given`, in a list
  - `title`: title of the canonical record
  - `doi`, `arxiv_id`, `s2_paper_id`: identifiers of the paper, empty if unknown
  - `source`: source of the canonical record, one of `semantic_scholar`, `crossref`,
    `arxiv`
  - `base_name`: base name of the PDF and markdown files, `<Year>.<Author>.<Title>`
  - `download_inputs`: ordered inputs for the download tool, in a list
  - `line_nos`: rows of the input CSV that gave this paper, in a list
  - `key`, `aliases`: identity of the paper, used to merge the rows of one paper
- `<output_dir>/01-resolve.issues.json`: the issues file, with a `schema_version` and
  a list `issues`, one object per input that gave no paper, with the fields
  `line_no`, `input`, `base_name`, `stage`, `status`, `reason`, and `candidates` (in
  a list)
  - `stage` is `resolve`, and `status` is `unresolved` or `ambiguous`
- `<output_dir>/01-resolve.summary.txt`: the summary of the stage, with the same text
  as the end of the log (the counts of items, the output files, and the path of the
  issues file). A rerun overwrites it. See the section "Summary of a Stage" in
  [`research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md)
- The cache of the API answers, in `--cache_dir`
- Both JSON files are written again on each run, and a paper that is not in the input
  CSV anymore leaves them
- With `--dry_run` the script writes nothing

### Exit Codes

- `0`: Success. Rows that do not resolve do not change it: they are items of the
  issues file
- `1`: An assertion or an exception stopped the run, e.g.:
  - `--input` is not an existing file
- `2`: A missing required argument, an invalid flag, or an abbreviated flag

### Logging

- Logs go to the console, at the level of `-v`, and to a file
  `research/Estimate_academic_paper_quality/resolve_papers.py.log`, which each run
  overwrites
- The run logs a progress bar, a warning for each failed row or failed API, and the
  summary of the stage, with the counts of lines, issues, and papers

## Software Architecture

- The design of the resolver is in
  [`research/Estimate_academic_paper_quality/docs/02.2-design.resolver.md`](/research/Estimate_academic_paper_quality/docs/02.2-design.resolver.md)
- This file is the entry point and the orchestration, and it also holds the matching:
  no module of the project has the resolver code (decision `D28`)

### Data Flow

1. **Input**: `_parse()` reads the flags, and `_main()` resolves the paths with
   `epqpaths`, checks that `--input` exists, and copies it with `_copy_input()`.
   Then `_resolve()` reads the rows of the input CSV
2. **Classification**: `_read_input_lines()` turns each row into an `InputLine`, with a
   kind of `local_pdf`, `ssrn`, `doi`, `arxiv`, `url`, or `title`
3. **Lookup and match**: `_resolve_line()` looks the row up in Semantic
   Scholar, Crossref, arXiv, and Unpaywall, and keeps one paper or makes an issue
4. **Merge**: `_finalize_papers()` merges the rows of one paper and sorts the papers
   from the newest to the oldest
5. **Output**: `epqrepor.write_papers()` and `epqrepor.write_issues()` write the
   papers file and the issues file

### Key Functions

- `_parse() -> argparse.ArgumentParser`
  - Defines the flags, with the helpers for the cache mode and `-v`
- `_main(parser) -> None`
  - Resolves the paths with `epqpaths.get_*`, copies the input, builds the API
    clients, and calls `_resolve()`: with `--dry_run` it only logs the kind of each row
- `_copy_input(input_csv, output_dir, *, dry_run) -> None`
  - Copies the input CSV to `00-paper_list.csv` in the output dir, and skips the copy
    if the input is already that file
- `_read_input_lines(input_csv) -> List[InputLine]`
  - Reads the rows of the CSV and classifies each row with `_parse_line()`
- `_resolve_line(input_line, sources) -> ResolveResult`
  - Looks one row up, matches it with `_match_candidates()`, and builds the `Paper`
    with `_build_paper()`, or an `Issue`
- `_finalize_papers(papers) -> List[Paper]`
  - Merges the rows of one paper with `_merge_papers()` and sorts them with
    `_sort_papers()`
- `_resolve(input_csv, papers_file, issues_file, sources) -> ResolveResult`
  - Resolves each row with a catch per row, then merges and sorts the papers, and
    writes the two JSON files
- `epqrepor.write_papers()`, `read_papers()`, `write_issues()`, `read_issues()`
  - Write and read the two JSON files, so that the next executables read them
- `epqpaths.get_papers_path()`, `get_issues_path()`, `get_input_copy_path()`
  - Name the three files, so that the writer and the reader of a file agree on the
    path

### Design Patterns

- **Files between stages**: the resolver shares only files with the other stages, so
  a person can fix the papers file by hand before the download (decision `D26`)
- **Copy of the input**: `00-paper_list.csv` keeps the list that made the papers file
- **Catch per item**: a failed row becomes an item of the issues file and the run
  goes on (decision `D26`)
- **Injected dependencies**: `BibSources` is an argument of `_resolve()`, so the
  tests use a fake API
- **Disk cache**: the API answers are cached in `--cache_dir`, so a rerun on the same
  list makes no request
