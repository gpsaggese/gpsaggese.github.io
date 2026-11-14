#!/usr/bin/env python

"""
Validate the benchmark and regenerate `labels.csv` from the snippet headers.

# Validate and rewrite `benchmark/labels.csv`
> update_labels.py

# Only check that the snippets are valid and `labels.csv` is in sync
> update_labels.py --check
"""

import argparse
import logging
import os

import helpers.hdbg as hdbg
import helpers.hparser as hparser

import benchmark_lib as benlib

_LOG = logging.getLogger(__name__)

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
        help="Dir containing the snippets",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write `labels.csv`, only check it is in sync",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    labels_file = os.path.join(args.benchmark_dir, "labels.csv")
    # Validate the snippets, and `labels.csv` too in check mode.
    errors = benlib.validate_benchmark(
        args.benchmark_dir, labels_file=labels_file if args.check else ""
    )
    for error in errors:
        _LOG.error("%s", error)
    hdbg.dassert_eq(len(errors), 0, "Found %s errors in the benchmark", len(errors))
    if args.check:
        _LOG.info("Benchmark is valid and '%s' is in sync", labels_file)
    else:
        benlib.write_labels(args.benchmark_dir, labels_file)


if __name__ == "__main__":
    _main(_parse())
