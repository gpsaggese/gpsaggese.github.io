#!/usr/bin/env python
"""
Run the offline capability-drift experiment and write review artifacts.

# Usage Example

- Run the full Phase 1 protocol:
> run_model_drift_capability_experiment.py

- Run a short deterministic rehearsal:
> run_model_drift_capability_experiment.py \
    --trials_per_scenario 10 \
    --power_trials 5 \
    --dst_dir /tmp/noesis_capability_rehearsal
"""

import argparse
import dataclasses
import logging

import helpers.hdbg as hdbg
import helpers.hparser as hparser
import helpers.hselect_input_output as hselsio

import research.Noesis.monitor.monitor_capability_experiment as rnmoncaex
import research.Noesis.monitor.monitor_capability_report as rnmoncare

_LOG = logging.getLogger(__name__)

DEFAULT_DST_DIR = "research/Noesis/docs/findings/model_drift_capability"


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--trials_per_scenario",
        type=int,
        default=1000,
        help="Seeded trials for each primary scenario",
    )
    parser.add_argument(
        "--power_trials",
        type=int,
        default=100,
        help="Seeded trials for each power-analysis point",
    )
    hselsio.add_dst_dir_arg(
        parser,
        dst_dir_required=False,
        dst_dir_default=DEFAULT_DST_DIR,
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    if args.trials_per_scenario < 1:
        raise ValueError("trials_per_scenario must be positive")
    if args.power_trials < 1:
        raise ValueError("power_trials must be positive")
    dst_dir = hselsio.parse_dst_dir_arg(args)
    protocol = rnmoncaex.CapabilityExperimentProtocol(
        trials_per_scenario=args.trials_per_scenario
    )
    _LOG.info(
        "Running %s primary trials per scenario",
        protocol.trials_per_scenario,
    )
    result = rnmoncaex.run_capability_experiment(protocol)
    _LOG.info("Running %s trials per power point", args.power_trials)
    power_points = rnmoncaex.run_capability_power_analysis(
        protocol,
        args.power_trials,
    )
    paths = rnmoncare.write_capability_artifacts(
        result,
        power_points,
        dst_dir,
    )
    print(rnmoncare.render_capability_markdown(result, power_points))
    print("Artifacts:")
    for path in dataclasses.astuple(paths):
        print("- %s" % path)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    _main(_parse())
