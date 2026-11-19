#!/usr/bin/env python
"""
Run the no-spend behavioral-fingerprint control experiment.

Set ``NOESIS_MODEL_EQUALITY_REPO`` to a checkout of the pinned Model Equality
Testing repository. The runner verifies its Git commit before importing it.
"""

import argparse
import dataclasses

import research.Noesis.monitor.monitor_fingerprint_experiment as rnmonfiex
import research.Noesis.monitor.monitor_fingerprint_report as rnmonfire


DEFAULT_DST_DIR = "research/Noesis/docs/findings/model_drift_fingerprint"


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials_per_scenario",
        type=int,
        default=100,
        help="Seeded trials for each scenario",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Permutation samples for each two-sample test",
    )
    parser.add_argument(
        "--character_length",
        type=int,
        default=1000,
        help="Unicode characters retained per completion",
    )
    parser.add_argument(
        "--dst_dir",
        default=DEFAULT_DST_DIR,
        help="Artifact output directory",
    )
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    protocol = rnmonfiex.FingerprintProtocol(
        trials_per_scenario=args.trials_per_scenario,
        permutations=args.permutations,
        character_length=args.character_length,
    )
    result = rnmonfiex.run_fingerprint_experiment(protocol)
    paths = rnmonfire.write_fingerprint_artifacts(result, args.dst_dir)
    print(rnmonfire.render_fingerprint_markdown(result))
    print("Artifacts:")
    for path in dataclasses.astuple(paths):
        print("- %s" % path)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    _main(_parse())
