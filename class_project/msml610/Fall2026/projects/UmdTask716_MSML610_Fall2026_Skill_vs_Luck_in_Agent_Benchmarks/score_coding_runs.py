#!/usr/bin/env python
"""
Score coding-benchmark solutions on the EvalPlus tests.

The solutions come from either:
- A file written by `run_coding_agents.py`: 1 row per cell and scaffold
- A public EvalPlus sample zip: 1 row per sample; a sample that is a raw
  chat answer goes through the EvalPlus `sanitize()` first

Each failure gets a label (e.g., `wrong_answer` or `timeout`), and with
`--n_checks 2` or more each solution is tested again, to find tests that pass
or fail at random.

The solutions are untrusted code, so run this script in Docker without
network access:
> ./docker_cmd_sandbox.sh "python score_coding_runs.py ..."

# Usage Example

- Score the answers of a model:
> score_coding_runs.py \
    --runs_file data/raw/coding_runs/humaneval/deepseek-coder_1.3b_t0.8.jsonl

- Score the first 50 samples per task of a public zip, twice each:
> score_coding_runs.py \
    --zip_file data/raw/evalplus_samples/v0.1.0/code-llama-7b_temp_0.8.zip \
    --max_samples 50 --n_checks 2

Import as:

import score_coding_runs as scocorun
"""

import argparse
import logging
import os
import re
from typing import Tuple

import pandas as pd

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser

import skill_vs_luck_runs_utils as rutils

_LOG = logging.getLogger(__name__)


# #############################################################################


def _get_run_rows(runs_file: str) -> Tuple[str, str, pd.DataFrame]:
    """
    Read the cells saved by `run_coding_agents.py`, 1 row per scaffold.

    :param runs_file: e.g.,
        `data/raw/coding_runs/humaneval/deepseek-coder_1.3b_t0.8.jsonl`
    :return: output name (e.g., `local_humaneval_deepseek-coder_1.3b_t0.8`),
        benchmark, and the rows with the code
    """
    records = rutils.read_runs_file(runs_file)
    hdbg.dassert_lt(0, len(records), "No cells in '%s'", runs_file)
    settings = hio.from_json(runs_file.replace(".jsonl", ".settings.json"))
    rows = pd.DataFrame(
        [row for record in records for row in rutils.get_scaffold_rows(record)]
    )
    rows.insert(0, "model", settings["model"])
    rows.insert(1, "temperature", settings["temperature"])
    # The benchmark is the name of the dir that holds the file.
    benchmark = os.path.basename(os.path.dirname(runs_file))
    stem = os.path.basename(runs_file)[: -len(".jsonl")]
    name = f"local_{benchmark}_{stem}"
    return name, benchmark, rows


def _get_zip_rows(
    zip_file: str, max_samples: int
) -> Tuple[str, str, pd.DataFrame]:
    """
    Read the samples of a public EvalPlus zip, 1 row per sample.

    :param zip_file: e.g., `.../v0.1.0/code-llama-7b_temp_0.8.zip`
    :param max_samples: samples to keep per task, `0` for all of them
    :return: output name (e.g., `evalplus_humaneval_code-llama-7b_t0.8`),
        benchmark, and the rows with the code
    """
    # The file name has the model and the temperature.
    match = re.match(
        r"^(?P<model>.+)_temp_(?P<temperature>[0-9.]+)\.zip$",
        os.path.basename(zip_file),
    )
    match = hdbg.dassert_re_match(match, "Unknown zip name '%s'", zip_file)
    model = match.group("model")
    temperature = float(match.group("temperature"))
    samples = rutils.iter_zip_samples(zip_file, max_samples=max_samples or None)
    rows = pd.DataFrame(
        [
            {"task_id": task_id, "run": run, "code": code}
            for task_id, run, code in samples
        ]
    )
    rows.insert(0, "model", model)
    rows.insert(1, "temperature", temperature)
    rows.insert(2, "scaffold", "one_shot")
    # The task ids tell the benchmark, e.g., `HumanEval/0` or `Mbpp/2`.
    is_humaneval = rows["task_id"].str.startswith("HumanEval/")
    hdbg.dassert(is_humaneval.all() or not is_humaneval.any())
    benchmark = "humaneval" if is_humaneval.all() else "mbpp"
    name = f"evalplus_{benchmark}_{model}_t{temperature:g}"
    return name, benchmark, rows


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--runs_file",
        action="store",
        default="",
        help="Answers saved by `run_coding_agents.py`",
    )
    parser.add_argument(
        "--zip_file",
        action="store",
        default="",
        help="Public EvalPlus sample zip",
    )
    parser.add_argument(
        "--max_samples",
        action="store",
        type=int,
        default=0,
        help="Samples to keep per task of a zip, 0 for all of them",
    )
    parser.add_argument(
        "--n_checks",
        action="store",
        type=int,
        default=1,
        help="Times to run the tests of each solution",
    )
    parser.add_argument(
        "--num_workers",
        action="store",
        type=int,
        default=0,
        help="Processes to use, 0 for half of the CPUs",
    )
    parser.add_argument(
        "--dst_dir",
        action="store",
        default="data/coding_runs",
        help="Where to save the scores",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    hdbg.dassert_ne(
        bool(args.runs_file),
        bool(args.zip_file),
        "Pass exactly 1 of --runs_file and --zip_file",
    )
    # Read the solutions.
    if args.runs_file:
        name, benchmark, rows = _get_run_rows(args.runs_file)
    else:
        name, benchmark, rows = _get_zip_rows(args.zip_file, args.max_samples)
    # Keep the tasks of the pinned dataset version (e.g., the 2023 MBPP
    # samples cover 399 tasks, MBPP+ v0.2.0 has 378).
    problems = rutils.load_problems(benchmark)
    known = rows["task_id"].isin(list(problems))
    if not known.all():
        _LOG.warning(
            "Dropping '%s' rows of tasks that are not in the dataset",
            int((~known).sum()),
        )
    rows = rows[known].reset_index(drop=True)
    if args.zip_file:
        # Some public samples are raw chat answers (prose around a code block).
        rows["code"] = [
            rutils.clean_sample(code, problems[task_id]["entry_point"])
            for task_id, code in zip(rows["task_id"], rows["code"])
        ]
    # Score the solutions.
    results = rutils.score_solutions(
        benchmark,
        list(zip(rows["task_id"], rows["code"])),
        n_checks=args.n_checks,
        num_workers=args.num_workers,
    )
    scores = pd.DataFrame(results)
    scores["n_checks"] = args.n_checks
    rows = pd.concat([rows.drop(columns="code"), scores], axis=1)
    # Save the scores.
    hio.create_dir(args.dst_dir, incremental=True)
    path = os.path.join(args.dst_dir, f"{name}.csv.gz")
    rows.to_csv(path, index=False)
    _LOG.info(
        "Saved '%s' scores in '%s', plus pass rate '%.3f'",
        len(rows),
        path,
        rows["plus_pass"].mean(),
    )


if __name__ == "__main__":
    _main(_parse())
