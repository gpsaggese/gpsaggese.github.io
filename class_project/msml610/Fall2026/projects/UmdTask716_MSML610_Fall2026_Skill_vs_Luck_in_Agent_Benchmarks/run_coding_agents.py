#!/usr/bin/env python
"""
Generate solutions to HumanEval+ or MBPP+ tasks with a model served by Ollama.

1 cell of the outcome tensor is 1 task and 1 run. In each cell the model
answers once (the one-shot scaffold); if the answer fails an example shown in
the prompt, the model sees the error and tries again, up to `--max_repairs`
times (the repair scaffold). Both scaffolds share the first attempt.

- Each cell is saved as soon as it is done, so an interrupted job continues
  where it stopped
- The seed of a cell depends only on the task and the run, so a cell can be
  generated again; the answer is often the same, but not always, since
  Ollama reuses its prompt cache between requests
- The model version and the settings are saved next to the answers, and a
  restart with a different model version stops with an error

Run it in Docker, with access to Ollama on the host:
> ./docker_cmd_sandbox.sh --ollama "python run_coding_agents.py ..."

# Usage Example

- Generate 5 runs at temperature 0.8 for the first 10 HumanEval+ tasks:
> run_coding_agents.py --benchmark humaneval --model deepseek-coder:1.3b \
    --temperature 0.8 --runs 5 --num_tasks 10

- Generate 1 greedy run for all the MBPP+ tasks:
> run_coding_agents.py --benchmark mbpp --model codeqwen:7b --temperature 0 \
    --runs 1

Import as:

import run_coding_agents as rucoag
"""

import argparse
import json
import logging
import os

import tqdm

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser

import skill_vs_luck_runs_utils as rutils

_LOG = logging.getLogger(__name__)


# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        action="store",
        choices=rutils.BENCHMARKS,
        required=True,
        help="Benchmark to run",
    )
    parser.add_argument(
        "--model",
        action="store",
        required=True,
        help="Ollama model, e.g., `deepseek-coder:1.3b`",
    )
    parser.add_argument(
        "--temperature",
        action="store",
        type=float,
        default=0.8,
        help="Sampling temperature, 0 for greedy decoding",
    )
    parser.add_argument(
        "--runs", action="store", type=int, default=5, help="Runs per task"
    )
    parser.add_argument(
        "--max_repairs",
        action="store",
        type=int,
        default=1,
        help="Attempts allowed after the first one",
    )
    parser.add_argument(
        "--num_tasks",
        action="store",
        type=int,
        default=0,
        help="Use only the first tasks, 0 for all of them",
    )
    parser.add_argument(
        "--dst_dir",
        action="store",
        default="data/raw/coding_runs",
        help="Where to save the answers",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    hdbg.dassert_lte(1, args.runs)
    if args.temperature == 0:
        # Greedy decoding gives the same answer for every seed.
        hdbg.dassert_eq(args.runs, 1, "Use 1 run with greedy decoding")
    # Select the tasks.
    problems = rutils.load_problems(args.benchmark)
    task_ids = rutils.sort_task_ids(list(problems))
    if args.num_tasks > 0:
        task_ids = task_ids[: args.num_tasks]
    # Collect the model version and the settings.
    info = rutils.get_model_info(args.model)
    stop = info["stop"] + [t for t in rutils.STOP_TEXTS if t not in info["stop"]]
    settings = {
        **info,
        "stop": stop,
        "temperature": args.temperature,
        "max_repairs": args.max_repairs,
        "options": rutils.GENERATION_OPTIONS,
        "instruction_prefix": rutils.INSTRUCTION_PREFIX,
        "response_prefix": rutils.RESPONSE_PREFIX,
    }
    # Save the settings, or check them against the saved ones on a restart.
    path = rutils.get_runs_file(
        args.dst_dir, args.benchmark, args.model, args.temperature
    )
    hio.create_dir(os.path.dirname(path), incremental=True)
    settings_path = path.replace(".jsonl", ".settings.json")
    if os.path.exists(settings_path):
        hdbg.dassert_eq(
            hio.from_json(settings_path),
            settings,
            "The saved answers come from a different model or settings",
        )
    else:
        hio.to_json(settings_path, settings)
    # Skip the cells that are already done.
    done = {
        (record["task_id"], record["run"])
        for record in rutils.read_runs_file(path)
    }
    cells = [
        (run, task_id)
        for run in range(args.runs)
        for task_id in task_ids
        if (task_id, run) not in done
    ]
    _LOG.info("Generating '%s' cells ('%s' already done)", len(cells), len(done))
    # Go run by run, so that a stopped job leaves complete runs.
    with open(path, "a") as file:
        for run, task_id in tqdm.tqdm(cells, mininterval=30, ascii=True):
            record = rutils.generate_cell(
                problems[task_id],
                args.benchmark,
                args.model,
                run,
                temperature=args.temperature,
                max_repairs=args.max_repairs,
                stop=stop,
            )
            file.write(json.dumps(record) + "\n")
            file.flush()
    _LOG.info("Saved the answers in '%s'", path)


if __name__ == "__main__":
    _main(_parse())
