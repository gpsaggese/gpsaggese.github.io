#!/usr/bin/env python
"""
Download the SWE-bench Verified per-task results and save them as CSV files.

The data comes from:
- `SWE-bench/experiments`, at a fixed commit: results and metadata of every
  submission
- The Hugging Face dataset `SWE-bench/SWE-bench_Verified`: the 500 tasks

Only this script needs internet. The notebooks read the saved files.

# Usage Example

- Rebuild the data in `data`, replacing the current files:
> fetch_swebench_verified.py --dst_dir data --overwrite

- Use a different commit of `SWE-bench/experiments`:
> fetch_swebench_verified.py --dst_dir data --overwrite --sha <commit>

Import as:

import fetch_swebench_verified as fesweve
"""

import argparse
import logging

import helpers.hdbg as hdbg
import helpers.hparser as hparser
import helpers.hselect_input_output as hseinout

import skill_vs_luck_utils as utils

_LOG = logging.getLogger(__name__)


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    hseinout.add_dst_dir_arg(
        parser, dst_dir_required=False, dst_dir_default="data"
    )
    parser.add_argument(
        "--sha",
        action="store",
        default=utils.SWEBENCH_EXPERIMENTS_SHA,
        help="Commit of `SWE-bench/experiments` to read the results from",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    dst_dir = hseinout.parse_dst_dir_arg(args)
    # Download and parse all the submissions.
    dataset = utils.build_swebench_dataset(args.sha)
    # Save the tables and where they came from.
    utils.save_swebench_dataset(dataset, dst_dir, args.sha)


if __name__ == "__main__":
    _main(_parse())
