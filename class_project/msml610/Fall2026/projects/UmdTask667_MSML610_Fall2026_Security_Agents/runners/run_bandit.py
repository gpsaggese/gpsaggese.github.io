#!/usr/bin/env python

"""
Run Bandit on the anonymized benchmark and store normalized findings.

Uses the default Bandit checks. Raw JSON goes to
`results/raw/bandit/<run_id>/`.

# Run on the benchmark
> runners/run_bandit.py --run_id r1
"""

import argparse
import logging
import os
import time

import helpers.hdbg as hdbg
import helpers.hparser as hparser
import helpers.hsystem as hsystem

import findings_lib as finlib
import runner_lib as runlib

_LOG = logging.getLogger(__name__)

_AGENT = "bandit"

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    runlib.add_runner_args(parser)
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    paths = runlib.prepare_cases(args, _AGENT)
    json_file = os.path.join(paths["work_dir"], "bandit.json")
    cmd = f"bandit -r {paths['cases_dir']} -f json -o {json_file} -q"
    started_at = runlib.get_timestamp()
    start = time.time()
    # Bandit exits with 1 when it reports issues.
    rc = hsystem.system(cmd, abort_on_error=False, suppress_output=False)
    seconds = time.time() - start
    hdbg.dassert_in(rc, (0, 1), "Bandit failed with exit code %s", rc)
    # `bandit --version` also prints the Python version on a second line.
    _, version = hsystem.system_to_string("bandit --version")
    version = version.split("\n")[0].strip()
    findings = finlib.read_bandit_json(json_file, agent=_AGENT)
    meta = {
        "tool_version": version,
        "model": "",
        "configs": ["default"],
        "started_at": started_at,
        "seconds": round(seconds, 2),
        "cost_usd": 0.0,
    }
    runlib.finish_run(
        findings, meta, args=args, agent=_AGENT, case_map=paths["case_map"]
    )


if __name__ == "__main__":
    _main(_parse())
