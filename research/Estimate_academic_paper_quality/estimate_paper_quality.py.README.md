# `estimate_paper_quality.py`

- Runs the four stages of the pipeline in order: `resolve_papers.py`,
  `download_papers.py`, `extract_features.py`, and `report_papers.py`
- Input: a CSV with one paper per row, as a title, link, DOI, or local PDF
- Output: all the files of the stages, in one `--output_dir`. The final result is
  `04-report.report.json`
- Use it to run the whole pipeline with one command, or to rerun a part of it. Each
  executable is an action, named after it, so `--only_action` and `--skip_action`
  select the executables to run

## Examples

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
      --model <model> \
      --force_extract \
      --refresh_citations
  ```

- Run only `extract_features.py` and `report_papers.py` on the output of an earlier
  run:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --output_dir ~/papers/run_2026_10 \
      --only_action extract_features \
      --only_action report_papers
  ```

- Run all the executables except `download_papers.py`, e.g., when the PDFs are in
  place:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10 \
      --skip_action download_papers
  ```

- Show the command of each executable, with no execution, at the level `DEBUG`:
  ```bash
  > research/Estimate_academic_paper_quality/estimate_paper_quality.py \
      --input research/Estimate_academic_paper_quality/paper_list.csv \
      --output_dir ~/papers/run_2026_10 \
      --dry_run \
      -v DEBUG
  ```

## Configuration & Inputs

### Command-line Arguments

| Argument                                        | Type | Default                  | Description                                                                          |
| :---------------------------------------------- | :--- | :----------------------- | :----------------------------------------------------------------------------------- |
| `-i`, `--input`                                 | str  | `""`                     | CSV with one paper per row: the input of `resolve_papers`, required if it runs       |
| `--output_dir`                                  | str  | required                 | Dir of all the outputs and of the state of the run                                   |
| `--model`                                       | str  | `""`                     | LLM of `extract_features.py`: empty is the default model of that script              |
| `--force_extract`, `--no_force_extract`         | flag | `--no_force_extract`     | Extract again the papers that have a features file                                   |
| `--refresh_citations`, `--no_refresh_citations` | flag | `--no_refresh_citations` | Ask for fresh citation counts in `report_papers`                                     |
| `--dry_run`                                     | flag | false                    | Show the command of each executable, with no execution                               |
| `--action`, `--skip_action`, `--only_action`    | str  | all the actions          | Select the actions to run (repeatable)                                               |
| `--all_actions`, `--clear_actions`              | flag | false                    | Start from all the actions, or from none                                             |
| `-v`                                            | str  | `INFO`                   | Logging level, forwarded to every stage                                              |

- The actions are `resolve_papers`, `download_papers`, `extract_features`, and
  `report_papers`, always run in this order. An executable that does not run reads the
  output of the earlier runs from `--output_dir`
- Only `resolve_papers.py` gets `--input`. The other executables find
  `01-resolve.paper_list.json` in `--output_dir`. `--input` is ignored, with a
  warning, if `resolve_papers` is not selected
- `report_papers.py` runs with its two own actions, `rate` and `report`
- A flag of a single executable that the script does not forward, e.g., `--email` of
  `resolve_papers.py` or `--min_delay_sec` of `download_papers.py`, needs a call of
  that executable. The other executables can then run through this script with
  `--skip_action resolve_papers` or `--skip_action download_papers`

### Environment Variables

- `PYTHONPATH`: the script puts the repo root and `helpers_root` in front of it for
  the executables
- The variables of the stages, e.g., `SEMANTIC_SCHOLAR_API_KEY`, pass to them
  unchanged

### Other Requirements

- `download_papers.py` needs Docker, and the SSRN papers also need Chrome
- Each executable selects its own dependencies with a `uv run` line. This script needs
  only the `helpers` package

## Output & Side Effects

### Files Created

- None of its own: every file comes from a stage. The layout of the output dir is in
  the section "Output Dir" of
  [`research/Estimate_academic_paper_quality/README.md`](/research/Estimate_academic_paper_quality/README.md#output-dir)
- Each stage writes its own `NN-<stage>.summary.txt`: the counts of items, the time
  of the stage, the output files, and the path of the issues file of the stage
- With `--dry_run` the script runs no stage, so it writes no file and makes no
  lookup or LLM call
- A rerun on the same `--output_dir` does not download, convert, or extract again a
  paper that is done. `report_papers.py` rates all the papers at each run

### Exit Codes

- `0`: Success. A paper that fails does not change it: it is a row of
  `04-report.report.json` with `status` `failed`
- `1`: An assertion stopped the run, e.g., `--input` is missing while the action
  `resolve_papers` runs, or a stage failed. The script stops at the first stage that fails
  and does not run the later ones
- `2`: A missing required argument or an invalid flag

### Logging

- Logs go to the console, at the level of `-v`, and to a file
  `research/Estimate_academic_paper_quality/estimate_paper_quality.py.log`, which
  each run overwrites
- The run logs the stages to run, a header for each stage, and the path of the report
- At the end the run logs the time of each action and the total, e.g.,
  ```text
  Time of each action
    resolve_papers:   12.3 sec
    download_papers:  1 min 5.0 sec
    Total:            1 min 17.3 sec
  ```
  The table is not logged with `--dry_run`, or when an action fails
- The output of a stage goes to the console as it runs, and each stage ends with its
  summary. The command of a stage shows
  at the level `DEBUG`, or with `--dry_run`

## Software Architecture

- The design is in
  [`research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md).
  The choice of a Python script that runs the stages as subprocesses is decision
  `D32`

### Data Flow

1. **Input**: `_main()` selects the actions with `hselect_action`, and checks that
   `--input` is an existing file if `resolve_papers` runs
2. **Environment**: `_add_repo_to_python_path()` puts the repo root and `helpers_root`
   in `PYTHONPATH`, which the subprocesses inherit
3. **Commands**: for each selected action, `hgit.find_file_in_git_tree()` locates
   `<action>.py`, and `_build_command()` builds its command line with `--output_dir`, the
   flags of that stage, and `-v <log level>`
4. **Run**: `_run_action()` calls `hsystem.system()`, which raises on a failed stage and
   so stops the pipeline. With `--dry_run` it only logs the command
5. **Output**: the executables write their files in `--output_dir`. The script logs
   the path of `04-report.report.json`

### Key Functions

- `_parse() -> argparse.ArgumentParser`
  - Defines the flags, with the helpers for the boolean flags, the stages, and `-v`
- `_main(parser) -> None`
  - Selects the stages, checks the inputs, and runs the stages in order
- `_add_repo_to_python_path() -> None`
  - Makes the modules of the repo visible to the executables
- `_build_command(action, executable, *, input_csv, output_dir, model, force_extract,
  refresh_citations, log_level) -> str`
  - Builds the command line of one stage. It is pure, so the tests check it with no
    subprocess
- `_run_action(action, cmd, *, dry_run) -> None`
  - Runs the command of a stage, or logs it with `[DRY_RUN]`

### Design Patterns

- **Subprocess, not import**: the script runs each executable with `hsystem`, so the
  stages keep their own dependencies, and share no process (decisions `D27` and
  `D32`). The import graph test allows only `epq_paths` here
- **One action per executable, named after it**: `hselect_action` gives the selection
  with the same flags as `report_papers.py` (rule `ARCH3.5`)
- **Forward `-v`**: the log level goes to every stage, so `-v DEBUG` applies to all
  of them
- **Fail fast**: a failed stage raises, so the later stages do not run on a missing
  file
