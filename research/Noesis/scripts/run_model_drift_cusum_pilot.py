#!/usr/bin/env python
"""Run the small offline CUSUM-versus-Fisher pilot with resource reporting."""

import argparse
import dataclasses
import logging
import platform
import resource
import time

import research.Noesis.monitor.monitor_cusum_pilot as rnmoncupi
import research.Noesis.monitor.monitor_cusum_report as rnmoncure

import helpers.hdbg as hdbg
import helpers.hparser as hparser
import helpers.hselect_input_output as hselsio


_LOG = logging.getLogger(__name__)
DEFAULT_DST_DIR = "research/Noesis/docs/findings/model_drift_cusum_pilot"


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--trials_per_scenario",
        type=int,
        default=250,
        help="Seeded paired trials per planted scenario",
    )
    parser.add_argument(
        "--calibration_trials",
        type=int,
        default=1000,
        help="Independent healthy streams used to calibrate the threshold",
    )
    parser.add_argument(
        "--max_rss_mb",
        type=float,
        default=512.0,
        help="Fail after writing artifacts if process RSS exceeds this cap",
    )
    hselsio.add_dst_dir_arg(
        parser,
        dst_dir_required=False,
        dst_dir_default=DEFAULT_DST_DIR,
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    if args.max_rss_mb <= 0:
        raise ValueError("max_rss_mb must be positive")
    protocol = rnmoncupi.CusumPilotProtocol(
        trials_per_scenario=args.trials_per_scenario,
        calibration_trials=args.calibration_trials,
    )
    dst_dir = hselsio.parse_dst_dir_arg(args)
    _LOG.info(
        "Running one-process offline pilot: %s trials per scenario",
        protocol.trials_per_scenario,
    )
    start = time.perf_counter()
    result = rnmoncupi.run_cusum_pilot(protocol)
    metrics = rnmoncure.CusumExecutionMetrics(
        wall_time_seconds=time.perf_counter() - start,
        peak_rss_mb=_peak_rss_mb(),
    )
    paths = rnmoncure.write_cusum_artifacts(result, metrics, dst_dir)
    print(rnmoncure.render_findings(result, metrics))
    print("Artifacts:")
    for path in dataclasses.astuple(paths):
        print("- %s" % path)
    if metrics.peak_rss_mb > args.max_rss_mb:
        raise SystemExit(
            "Peak RSS %.1f MB exceeded %.1f MB cap"
            % (metrics.peak_rss_mb, args.max_rss_mb)
        )


if __name__ == "__main__":
    _main(_parse())
