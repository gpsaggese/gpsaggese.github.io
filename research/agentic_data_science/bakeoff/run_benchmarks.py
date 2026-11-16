#!/usr/bin/env python3
"""
Run agents on the benchmark sample and score their answers.

Usage:
> run_benchmarks.py --agents oracle --limit 3
> run_benchmarks.py --agents claude_code,smolagents
> run_benchmarks.py --agents claude_code --retry-errors
> run_benchmarks.py --summary

Every (agent, benchmark, task) run gets a fresh working directory with copies of
the task's data files, and appends one JSON line to `results/benchmark_runs.jsonl`
(answer, score, tokens, cost, time, error). Runs already in that file are
skipped, so an interrupted run resumes where it stopped. `--retry-errors` reruns
the tasks whose latest run failed. A run stops after 5 failures in a row, which
usually means a usage limit, not hard tasks.

`--summary` prints accuracy per agent and benchmark, with a 95% bootstrap
confidence interval, plus cost and time.
"""

import argparse
import dataclasses
import json
import logging
import os
import random
import shutil
from typing import Dict, List, Optional, Set, Tuple

import helpers.hdbg as hdbg
import helpers.hparser as hparser

import research.agentic_data_science.bakeoff.benchmarks as radsbben
import research.agentic_data_science.bakeoff.task_agents as radsbtaag

_LOG = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RUNS_FILE = os.path.join(RESULTS_DIR, "benchmark_runs.jsonl")


def make_workdir(task: radsbben.Task, root: str, agent: str) -> str:
    """
    Create a fresh working directory holding copies of the task's files.
    """
    workdir = os.path.join(root, agent, task.benchmark, task.task_id)
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    for src in task.files:
        dst = os.path.join(workdir, os.path.basename(src))
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return workdir


def latest_runs(path: str) -> List[Dict]:
    """
    Return the last run of each (agent, benchmark, task_id) in the runs file.

    A task that was retried keeps only its latest result.
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    latest = {(r["agent"], r["benchmark"], r["task_id"]): r for r in rows}
    return list(latest.values())


def done_runs(path: str, retry_errors: bool = False) -> Set[Tuple[str, str, str]]:
    """
    Return the (agent, benchmark, task_id) triples that need no new run.

    :param retry_errors: if True, a task whose latest run failed is not done
    """
    return {
        (r["agent"], r["benchmark"], r["task_id"])
        for r in latest_runs(path)
        if not (retry_errors and r.get("error"))
    }


class TooManyErrors(RuntimeError):
    """
    Raised when an agent fails many tasks in a row (e.g., a usage limit).
    """


def run(
    agents: List[str],
    benchmarks: List[str],
    limit: Optional[int] = None,
    runs_file: str = RUNS_FILE,
    bench_root: str = radsbben.BENCHMARK_DIR,
    retry_errors: bool = False,
    max_consecutive_errors: int = 5,
) -> int:
    """
    Run each agent on each sampled task not yet in `runs_file`.

    :param limit: run at most this many tasks per benchmark
    :param retry_errors: also rerun tasks whose latest run failed
    :param max_consecutive_errors: stop after this many failures in a row, since
        that usually means a usage limit or an outage, not hard tasks
    :return: number of new runs
    """
    os.makedirs(os.path.dirname(runs_file), exist_ok=True)
    work_root = os.path.join(os.path.dirname(runs_file), "work")
    done = done_runs(runs_file, retry_errors)
    n_new = 0
    errors_in_a_row = 0
    for agent_name in agents:
        agent = radsbtaag.get_task_agent(agent_name)
        for bench in benchmarks:
            tasks = radsbben.load_sample(bench, bench_root)[:limit]
            for task in tasks:
                key = (agent_name, bench, task.task_id)
                if key in done:
                    continue
                workdir = make_workdir(task, work_root, agent_name)
                res = agent.timed_answer(task, workdir)
                score = radsbben.score(task, res.answer) if res.error is None else 0.0
                row = {
                    "agent": agent_name,
                    "benchmark": bench,
                    "task_id": task.task_id,
                    "score": score,
                    **dataclasses.asdict(res),
                }
                with open(runs_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                n_new += 1
                errors_in_a_row = errors_in_a_row + 1 if res.error else 0
                if errors_in_a_row >= max_consecutive_errors:
                    raise TooManyErrors(
                        f"{agent_name}: {errors_in_a_row} failures in a row, last: "
                        f"{(res.error or '').strip().splitlines()[-1][:200]}. "
                        "Check the account limits, then rerun with --retry-errors"
                    )
                _LOG.info(
                    "%s %s %s score=%s time=%ss%s",
                    agent_name, bench, task.task_id, score, res.runtime_s,
                    " ERROR" if res.error else "",
                )
    return n_new


def bootstrap_ci(
    scores: List[float], n_boot: int = 2000, seed: int = 0
) -> Tuple[float, float]:
    """
    Return the 95% bootstrap confidence interval of the mean.
    """
    if not scores:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def summarize(runs_file: str = RUNS_FILE) -> List[Dict]:
    """
    Aggregate the runs file into one row per (agent, benchmark).
    """
    rows = latest_runs(runs_file)
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        groups.setdefault((r["agent"], r["benchmark"]), []).append(r)
    out = []
    for (agent, bench), rs in sorted(groups.items()):
        scores = [r["score"] for r in rs]
        lo, hi = bootstrap_ci(scores)
        costs = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
        out.append(
            {
                "agent": agent,
                "benchmark": bench,
                "n": len(rs),
                "accuracy": sum(scores) / len(scores),
                "ci_low": lo,
                "ci_high": hi,
                "errors": sum(1 for r in rs if r.get("error")),
                "cost_usd": round(sum(costs), 4) if costs else None,
                "mean_time_s": round(sum(r["runtime_s"] or 0 for r in rs) / len(rs), 1),
            }
        )
    return out


def _print_summary(rows: List[Dict]) -> None:
    print(f"{'agent':12s} {'benchmark':9s} {'n':>3s} {'acc':>6s} {'95% CI':>15s} "
          f"{'err':>4s} {'cost$':>8s} {'time_s':>7s}")
    for r in rows:
        cost = f"{r['cost_usd']:.2f}" if r["cost_usd"] is not None else "-"
        print(f"{r['agent']:12s} {r['benchmark']:9s} {r['n']:3d} {r['accuracy']:6.2f} "
              f"  [{r['ci_low']:.2f}, {r['ci_high']:.2f}] {r['errors']:4d} "
              f"{cost:>8s} {r['mean_time_s']:7.1f}")


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agents", help="Comma-separated agent names")
    parser.add_argument(
        "--benchmarks",
        default=",".join(radsbben.LOADERS),
        help="Comma-separated benchmark names (default: all)",
    )
    parser.add_argument("--limit", type=int, help="Max tasks per benchmark")
    parser.add_argument(
        "--retry-errors", action="store_true", help="Rerun tasks whose last run failed"
    )
    parser.add_argument("--runs-file", default=RUNS_FILE, help="Output JSONL")
    parser.add_argument("--summary", action="store_true", help="Print accuracy table")
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser, argv: Optional[List[str]] = None) -> None:
    args = parser.parse_args(argv)
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    if args.agents:
        try:
            n = run(
                args.agents.split(","),
                args.benchmarks.split(","),
                args.limit,
                args.runs_file,
                retry_errors=args.retry_errors,
            )
        except TooManyErrors as e:
            raise SystemExit(f"Stopped: {e}")
        print(f"{n} new runs written to {args.runs_file}")
    if args.summary:
        _print_summary(summarize(args.runs_file))
    if not (args.agents or args.summary):
        parser.print_help()


if __name__ == "__main__":
    _main(_parse())
