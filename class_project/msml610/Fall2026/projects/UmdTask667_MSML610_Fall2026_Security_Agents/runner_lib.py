"""
Shared steps of the agent runners in `runners/`.

Every runner:
- Writes anonymized copies of the snippets to a work dir
- Runs one agent on those copies and keeps its raw output in the work dir
- Maps the findings back to snippets and writes them with `findings_lib`

Import as:

import runner_lib as runlib
"""

import argparse
import datetime
import logging
import os
from typing import Any, Dict, List

import pandas as pd

import helpers.hdbg as hdbg
import helpers.hio as hio

import benchmark_lib as benlib
import findings_lib as finlib

_LOG = logging.getLogger(__name__)


def add_runner_args(parser: argparse.ArgumentParser) -> None:
    """
    Add the command line options shared by all runners.
    """
    parser.add_argument(
        "--benchmark_dir",
        action="store",
        default="benchmark",
        help="Dir containing the snippets",
    )
    parser.add_argument(
        "--run_id",
        action="store",
        required=True,
        help="Id of the run, e.g., `r1`",
    )
    parser.add_argument(
        "--raw_dir",
        action="store",
        default="results/raw",
        help="Dir for anonymized snippets and raw agent output",
    )
    parser.add_argument(
        "--results_dir",
        action="store",
        default="results/normalized",
        help="Dir for normalized findings",
    )


def prepare_cases(
    args: argparse.Namespace, agent: str, *, keep_cache: bool = False
) -> Dict[str, Any]:
    """
    Create the work dir of a run and anonymize the snippets into it.

    :param args: parsed runner options
    :param agent: name of the agent
    :param keep_cache: keep files from previous attempts of the same run (e.g.,
        cached LLM responses) and only rebuild the anonymized snippets
    :return: dict with:
        - `work_dir`: dir for the raw output of this run
        - `cases_dir`: dir with the anonymized snippets
        - `case_map`: table from `benchmark_lib.anonymize_benchmark()`
    """
    work_dir = os.path.join(args.raw_dir, agent, args.run_id)
    cases_dir = os.path.join(work_dir, "cases")
    # Start from an empty dir so that old cases are not scanned.
    if keep_cache:
        hio.create_dir(work_dir, incremental=True)
        hio.create_dir(cases_dir, incremental=False)
    else:
        hio.create_dir(work_dir, incremental=False)
    case_map = benlib.anonymize_benchmark(args.benchmark_dir, cases_dir)
    case_map.to_csv(os.path.join(work_dir, "case_map.csv"), index=False)
    paths = {"work_dir": work_dir, "cases_dir": cases_dir, "case_map": case_map}
    return paths


def get_timestamp() -> str:
    """
    Return the current UTC time in ISO format.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    return timestamp


def finish_run(
    findings: List[Dict[str, Any]],
    meta: Dict[str, Any],
    *,
    args: argparse.Namespace,
    agent: str,
    case_map: pd.DataFrame,
) -> None:
    """
    Map raw findings to snippets and write the normalized run.

    :param findings: findings on case files with stripped line numbers
    :param meta: run metadata to store next to the findings
    """
    findings = finlib.attach_snippet_ids(findings, case_map)
    for finding in findings:
        finding["agent"] = agent
        finding["run_id"] = args.run_id
        finding["model"] = meta.get("model", "")
    # Record which snippets were reviewed, so that a partial run (e.g., a
    # smoke test) is scored only on them.
    meta = dict(
        meta,
        n_snippets=len(case_map),
        snippet_ids=sorted(case_map["snippet_id"].tolist()),
    )
    finlib.write_run(
        findings, meta, results_dir=args.results_dir, agent=agent, run_id=args.run_id
    )
    n_flagged = len({f["snippet_id"] for f in findings})
    _LOG.info(
        "%s run '%s': %s findings on %s of %s snippets",
        agent,
        args.run_id,
        len(findings),
        n_flagged,
        len(case_map),
    )
    hdbg.dassert_lte(n_flagged, len(case_map))
