#!/usr/bin/env python

"""
Score all agent runs against the benchmark labels and print a metrics table.

# Score every run in `results/normalized/` and save `results/scores.csv`
> score_results.py
"""

import argparse
import logging
import os

import pandas as pd

import helpers.hdbg as hdbg
import helpers.hparser as hparser

import benchmark_lib as benlib
import findings_lib as finlib
import scoring_lib as scolib

_LOG = logging.getLogger(__name__)

# Metrics to print, in order.
_COLUMNS = (
    "agent",
    "run_id",
    "n_snippets",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "fpr",
    "mcc",
    "recall_line",
    "recall_strict",
    "cwe_accuracy",
    "paired_accuracy",
)

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--benchmark_dir",
        action="store",
        default="benchmark",
        help="Dir containing the snippets and `labels.csv`",
    )
    parser.add_argument(
        "--results_dir",
        action="store",
        default="results/normalized",
        help="Dir with normalized findings",
    )
    parser.add_argument(
        "--output_file",
        action="store",
        default="results/scores.csv",
        help="CSV to write all metrics to",
    )
    parser.add_argument(
        "--line_window",
        action="store",
        type=int,
        default=scolib.LINE_WINDOW,
        help="Max distance between a finding and a bug line",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Refuse to score against labels that are out of sync with the snippets.
    labels_file = os.path.join(args.benchmark_dir, "labels.csv")
    errors = benlib.validate_benchmark(args.benchmark_dir, labels_file=labels_file)
    hdbg.dassert_eq(len(errors), 0, "Benchmark is invalid: %s", errors)
    labels = benlib.load_labels(labels_file)
    findings = finlib.read_findings(args.results_dir)
    runs = finlib.read_runs(args.results_dir)
    scores = scolib.score_runs(
        labels, findings, runs, line_window=args.line_window
    )
    scores.to_csv(args.output_file, index=False)
    _LOG.info("Wrote scores to '%s'", args.output_file)
    n_vuln = int(labels["vulnerable"].sum())
    print(
        f"Benchmark: {len(labels)} snippets "
        f"({n_vuln} vulnerable, {len(labels) - n_vuln} safe)"
    )
    with pd.option_context("display.width", 200, "display.precision", 2):
        print(scores[list(_COLUMNS)].to_string(index=False))


if __name__ == "__main__":
    _main(_parse())
