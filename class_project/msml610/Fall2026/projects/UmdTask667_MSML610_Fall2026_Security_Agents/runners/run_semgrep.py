#!/usr/bin/env python

"""
Run Semgrep CE on the anonymized benchmark and store normalized findings.

Uses the `p/python` and `p/owasp-top-ten` registry rule sets with default
settings. Raw SARIF goes to `results/raw/semgrep/<run_id>/`.

# Run on the benchmark
> runners/run_semgrep.py --run_id r1
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

_AGENT = "semgrep"
_CONFIGS = ("p/python", "p/owasp-top-ten")

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
    sarif_file = os.path.join(paths["work_dir"], "semgrep.sarif")
    # `--no-git-ignore` is needed since the work dir is git-ignored.
    configs = " ".join(f"--config {config}" for config in _CONFIGS)
    cmd = (
        f"semgrep scan {configs} --sarif --metrics off --no-git-ignore "
        f"--output {sarif_file} {paths['cases_dir']}"
    )
    started_at = runlib.get_timestamp()
    start = time.time()
    hsystem.system(cmd, suppress_output=False)
    seconds = time.time() - start
    _, version = hsystem.system_to_one_line("semgrep --version")
    findings = finlib.read_sarif(sarif_file, agent=_AGENT)
    meta = {
        "tool_version": version,
        "model": "",
        "configs": list(_CONFIGS),
        "started_at": started_at,
        "seconds": round(seconds, 2),
        "cost_usd": 0.0,
    }
    runlib.finish_run(
        findings, meta, args=args, agent=_AGENT, case_map=paths["case_map"]
    )


if __name__ == "__main__":
    _main(_parse())
