#!/usr/bin/env python
"""Analyze complete saved Lane 2 pools without additional provider calls."""

import argparse
import os
from typing import Optional, Sequence

import research.Noesis.monitor.monitor_live_fingerprint as rnmonlifp
import research.Noesis.monitor.monitor_live_fingerprint_analysis as rnmonlifa


DEFAULT_DST_DIR = (
    "research/Noesis/docs/findings/model_drift_live_fingerprint"
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations_path",
        action="append",
        required=True,
        help="Repeat in chronological order for a parent and any resume runs.",
    )
    parser.add_argument("--dst_dir", default=DEFAULT_DST_DIR)
    parser.add_argument("--partial_trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint_path")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    protocol = rnmonlifp.LiveFingerprintProtocol().make_detector_protocol()
    rows = rnmonlifa.load_paid_observation_rows(args.observations_path)
    records = rnmonlifa.build_live_completion_triples(rows, protocol)
    checkpoint_path = args.checkpoint_path or os.path.join(
        args.dst_dir, "fingerprint_checkpoint.jsonl"
    )
    trials = rnmonlifa.run_live_fingerprint_tests(
        records,
        protocol,
        partial_trials=args.partial_trials,
        workers=args.workers,
        checkpoint_path=checkpoint_path,
    )
    diagnostics = rnmonlifa.pool_diagnostics(records)
    rnmonlifa.write_live_fingerprint_artifacts(
        args.dst_dir, protocol, trials, diagnostics
    )
    summaries = rnmonlifa.summarize_live_fingerprint_tests(trials)
    for item in summaries:
        print(
            "%s detection=%.3f median_p=%.4f"
            % (
                item["scenario"],
                item["detection_rate"],
                item["median_p_value"],
            )
        )
    print("Findings: %s/fingerprint_findings.md" % args.dst_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
