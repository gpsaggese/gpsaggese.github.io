# `download_papers.py`

- Downloads the PDF of each paper of a papers file and converts it to markdown
- Input: the papers file that
  `research/Estimate_academic_paper_quality/resolve_papers.py` writes, by default
  `01-resolve.paper_list.json` in `--output_dir`
- Output: `paper_pdf/<BASE>.pdf` and `paper_md/<BASE>.md` for each paper, and
  `02-download.issues.json` with the papers that failed, in `--output_dir`
- Run it after `research/Estimate_academic_paper_quality/resolve_papers.py` and
  before `research/Estimate_academic_paper_quality/extract_features.py`. A rerun
  skips the papers that are already converted
- It waits a random time between two downloads, so that the hosts do not take it for
  a bot

## Examples

- Download the papers of `01-resolve.paper_list.json` in the output dir:
  ```bash
  > research/Estimate_academic_paper_quality/download_papers.py \
      --output_dir ~/papers/run_2026_10
  ```

- Wait 5 to 20 seconds between two downloads:
  ```bash
  > research/Estimate_academic_paper_quality/download_papers.py \
      --output_dir ~/papers/run_2026_10 \
      --min_delay_sec 5 \
      --max_delay_sec 20
  ```

- List the papers to download, with no download and no output file:
  ```bash
  > research/Estimate_academic_paper_quality/download_papers.py \
      --output_dir ~/papers/run_2026_10 \
      --dry_run
  ```

- Debug a failed download: the level is forwarded to the download tool:
  ```bash
  > research/Estimate_academic_paper_quality/download_papers.py \
      --output_dir ~/papers/run_2026_10 \
      -v DEBUG
  ```

## Configuration & Inputs

### Command-line Arguments

| Argument          | Type  | Default  | Description                                                                               |
| :---------------- | :---- | :------- | :---------------------------------------------------------------------------------------- |
| `-i`, `--input`   | str   | `""`     | Papers file of the stage resolve: empty is `01-resolve.paper_list.json` in `--output_dir` |
| `--output_dir`    | str   | required | Dir of all the outputs and of the state of the run                                        |
| `--min_delay_sec` | float | `3.0`    | Shortest random wait between two downloads, in seconds                                    |
| `--max_delay_sec` | float | `10.0`   | Longest random wait between two downloads, in seconds                                     |
| `--dry_run`       | flag  | false    | Show the papers to download, with no download and no output file                          |
| `-v`              | str   | `INFO`   | Logging level: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                   |

- A person can fix or delete an item of the papers file before the run, e.g., to add
  a `download_inputs` link or the path of a local PDF
- The format of the papers file is in
  [`research/Estimate_academic_paper_quality/resolve_papers.py.README.md`](/research/Estimate_academic_paper_quality/resolve_papers.py.README.md)

### Random Delay

- The script waits a random time between two downloads: the time is drawn uniformly
  from `--min_delay_sec` to `--max_delay_sec`
- Why: a client that downloads at a steady pace looks like a bot, and a host can
  throttle or block it
- When it waits:
  - Before each download, except the first of the run
  - After a download that failed, as after one that worked
  - Never for a paper that is skipped because it is already converted
  - Never with `--dry_run`
- To turn the wait off, set `--min_delay_sec 0 --max_delay_sec 0`
- The values must satisfy `0 <= --min_delay_sec <= --max_delay_sec`: otherwise the
  run stops

### Other Requirements

- The script runs the download tool
  [`helpers_root/dev_scripts_helpers/download/download_academic_paper_to_md.py`](/helpers_root/dev_scripts_helpers/download/download_academic_paper_to_md.py),
  which needs Docker and, for SSRN papers, Chrome
- The first line of the file selects its dependencies with `uv run`
- The script does not use an LLM

## Output & Side Effects

### Files Created

- The layout of the whole output dir is in the section "Output Dir" of
  [`research/Estimate_academic_paper_quality/README.md`](/research/Estimate_academic_paper_quality/README.md#output-dir)
- In `<output_dir>/paper_pdf/`: `<BASE>.pdf`, where `<BASE>` is
  `<Year>.<Author>.<Title>`
- In `<output_dir>/paper_md/`: `<BASE>.md`, with the same `<BASE>`
- `<output_dir>/02-download.issues.json`: the papers that failed, read by
  `research/Estimate_academic_paper_quality/report_papers.py`
- `<output_dir>/state/logs/<BASE>.download.log`: the log of the download tool for
  each paper
- `<output_dir>/02-download.summary.txt`: the summary of the stage, with the same text
  as the end of the log (the counts of items, the time of the stage, the output files,
  and the path of the
  issues file). A rerun overwrites it. See the section "Summary of a Stage" in
  [`research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md)
- A paper that has a valid PDF and a markdown file is not downloaded or converted
  again. A paper that has only a valid PDF is only converted
- With `--dry_run` the script writes nothing

### Exit Codes

- `0`: Success. Papers that fail do not change it: they are issues for
  `research/Estimate_academic_paper_quality/report_papers.py`
- `1`: An assertion or an exception stopped the run, e.g.:
  - The papers file is not an existing file
  - `--min_delay_sec` is negative or greater than `--max_delay_sec`
- `2`: A missing required argument or an invalid flag

### Logging

- Logs go to the console, at the level of `-v`, and to a file
  `research/Estimate_academic_paper_quality/download_papers.py.log`, which each run
  overwrites
- The run logs a progress bar, a warning for each failed paper, and the summary of
  the stage
- With `-v DEBUG` the run also logs each wait

## Software Architecture

- The overall design is in
  [`research/Estimate_academic_paper_quality/docs/02-design.architecture.md`](/research/Estimate_academic_paper_quality/docs/02-design.architecture.md)

### Data Flow

1. **Input**: `_main()` checks that the papers file exists and that the delays are
   valid, then `epqrepor.read_papers()` reads the `Paper` objects. The papers file is
   `--input`, or `epqpaths.get_papers_path()` if `--input` is empty
2. **Download**: for each paper not on disk, `epq_fetcher.fetch_paper()` tries the
   download inputs in order and keeps the first valid PDF. Before each download but
   the first, `_download()` waits a random time
3. **Conversion**: the download tool converts the PDF to markdown. A failure at any
   step becomes an `Issue`
4. **Output**: `epqrepor.write_issues()` writes `02-download.issues.json`, at the
   path of `epqpaths.get_issues_path()`, then `epqsumm.emit_stage_summary()` logs the
   summary of the stage and writes `02-download.summary.txt`

### Key Functions

- `_parse() -> argparse.ArgumentParser`
  - Defines the flags, with the helper for `-v`
- `_main(parser) -> None`
  - Checks the input and the delays, then calls `_download()`
- `_download(papers, output_dir, *, tool_path, log_level, min_delay_sec,
  max_delay_sec, dry_run, sleep_func=time.sleep) -> List[Issue]`
  - Skips the converted papers, waits `random.uniform(min_delay_sec, max_delay_sec)`
    seconds with `sleep_func` before each download but the first, turns each failed
    fetch into an issue, and writes the issues of the stage

### Design Patterns

- **Files between stages**: the script reads the papers file and writes only files in
  `--output_dir`. It imports no other executable (decision `D27`)
- **Catch per item**: a failed paper becomes an issue and the run goes on (decision
  `D21`)
- **Idempotent**: a rerun does the work only for the papers that are not converted
- **Injected dependencies**: `sleep_func` is an argument of `_download()`, so the
  tests do not wait
