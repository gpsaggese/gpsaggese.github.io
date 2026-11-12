#!/usr/bin/env python
"""
Run the deterministic offline checker experiment and write review artifacts.

# Usage Example

    > research/Noesis/scripts/run_checker_replay.py

    > research/Noesis/scripts/run_checker_replay.py \
        --seeds 0,1,2 --contracts_per_seed 1000 --output_dir /tmp/replay

Import as:

import research.Noesis.scripts.run_checker_replay as rnscchre
"""

import argparse
import dataclasses
import logging
import pathlib
import sys
from typing import Optional, Sequence, Tuple

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.checker.checker_replay_report as rnochrerp

_LOG = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "research/Noesis/docs/findings/checker_replay"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def parse_seeds(value: str) -> Tuple[int, ...]:
    """
    Parse a comma-separated, unique integer seed list for ``argparse``.
    """
    try:
        seeds = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds or any(not item.strip() for item in value.split(",")):
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated deterministic seeds (default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--contracts_per_seed",
        type=int,
        default=1000,
        help="Contracts per scenario and seed (default: 1000)",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, CSV, and Markdown artifacts",
    )
    return parser.parse_args(argv)


def _main(args: argparse.Namespace) -> int:
    """
    Run the experiment, write artifacts, and return acceptance as an exit code.
    """
    _LOG.debug(hprint.to_str("args"))
    if args.contracts_per_seed < 1:
        raise ValueError("contracts_per_seed must be positive")
    result = rnochrerp.run_replay_experiment(
        args.seeds,
        args.contracts_per_seed,
    )
    paths = rnochrerp.write_replay_artifacts(result, args.output_dir)
    print(paths.markdown_path.read_text(encoding="utf-8"))
    print("Artifacts:")
    for path in dataclasses.astuple(paths):
        print("- %s" % path)
    return 0 if result.passed else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(_main(_parse_args()))
