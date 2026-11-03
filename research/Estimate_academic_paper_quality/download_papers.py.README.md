# `download_papers.py`

- Downloads the PDF of each paper of a papers CSV and converts it to markdown
- Input: the papers CSV that `resolve_papers.py` writes
- Output: `papers/<BASE>.pdf` and `papers/<BASE>.md` for each paper, and the issues of
  the papers that failed, in `--output_dir`
- Run it after `resolve_papers.py` and before `extract_features.py`. A rerun skips the
  papers that are already converted

## Examples

- Download the papers of a papers CSV:
  ```bash
  > research/Estimate_academic_paper_quality/download_papers.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10
  ```

- List the papers to download, with no download and no output file:
  ```bash
  > research/Estimate_academic_paper_quality/download_papers.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --dry_run
  ```

- Debug a failed download: the level is forwarded to the download tool:
  ```bash
  > research/Estimate_academic_paper_quality/download_papers.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      -v DEBUG
  ```

## Configuration & Inputs

### Command-line Arguments

| Argument       | Type | Default  | Description                                                           |
| :------------- | :--- | :------- | :-------------------------------------------------------------------- |
| `-i`, `--input` | str  | required | Papers CSV that `resolve_papers.py` wrote                            |
| `--output_dir` | str  | required | Dir of all the outputs and of the state of the run                    |
| `--dry_run`    | flag | false    | Show the papers to download, with no download and no output file      |
| `-v`           | str  | `INFO`   | Logging level: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

- A person can fix or delete a row of the papers CSV before the run, e.g., to add a
  `download_inputs` link or the path of a local PDF
- The format of the papers CSV is in
  [`resolve_papers.py.README.md`](/research/Estimate_academic_paper_quality/resolve_papers.py.README.md)

### Other Requirements

- The script runs the download tool
  [`download_academic_paper_to_md.py`](/helpers_root/dev_scripts_helpers/download/download_academic_paper_to_md.py),
  which needs Docker and, for SSRN papers, Chrome
- The first line of the file selects its dependencies with `uv run`
- The script does not use an LLM

## Output & Side Effects

### Files Created

- In `<output_dir>/papers/`: `<BASE>.pdf` and `<BASE>.md`, where `<BASE>` is
  `<Year>.<Author>.<Title>`
- In `<output_dir>/state/`:
  - `issues.download.json`: the papers that failed, read by `report_papers.py`
  - `logs/<BASE>.download.log`: the log of the download tool for each paper
- A paper that has a valid PDF and a markdown file is not downloaded or converted
  again. A paper that has only a valid PDF is only converted
- With `--dry_run` the script writes nothing

### Exit Codes

- `0`: Success. Papers that fail do not change it: they are issues for `report_papers.py`
- `1`: An assertion or an exception stopped the run, e.g., `--input` is not an
  existing file
- `2`: A missing required argument or an invalid flag

### Logging

- Logs go to the console, at the level of `-v`, and to a file
  `research/Estimate_academic_paper_quality/download_papers.py.log`, which each run
  overwrites
- The run logs a progress bar, and a warning for each failed paper

## Software Architecture

- The overall design is in
  [`02-design.architecture.md`](/research/Estimate_academic_paper_quality/docs/02-design.architecture.md)

### Data Flow

1. **Input**: `_main()` checks that `--input` exists, then reads the `Paper` objects of
   the papers CSV
2. **Download**: for each paper not on disk, `epq_fetcher.fetch_paper()` tries the
   download inputs in order and keeps the first valid PDF
3. **Conversion**: the download tool converts the PDF to markdown. A failure at any
   step becomes an `Issue`
4. **Output**: `epq_report.write_issues()` writes `state/issues.download.json`

### Key Functions

- `_parse() -> argparse.ArgumentParser`
  - Defines the flags, with the helper for `-v`
- `_main(parser) -> None`
  - Checks the input, then calls `_download()`
- `_download(papers, output_dir, *, tool_path, log_level, dry_run) -> List[Issue]`
  - Skips the converted papers, turns each failed fetch into an issue, and writes
    the issues of the stage

### Design Patterns

- **Files between stages**: the script reads the papers CSV and writes only files in
  `--output_dir`. It imports no other executable (decision `D27`)
- **Catch per item**: a failed paper becomes an issue and the run goes on (decision
  `D21`)
- **Idempotent**: a rerun does the work only for the papers that are not converted
