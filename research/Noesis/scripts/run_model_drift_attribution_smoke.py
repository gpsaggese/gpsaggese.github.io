#!/usr/bin/env python
"""
Validate or execute the 15-call provider-attribution smoke test.

Validation is the default and makes no provider calls. Live execution requires
``OPENROUTER_API_KEY``, ``--execute``, and an approved cap of at least $0.25.
"""

import argparse
import asyncio
import os
from typing import Optional, Sequence

import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
import research.Noesis.monitor.monitor_paid_smoke as rnmonpasm
import research.Noesis.monitor.monitor_paid_smoke_report as rnmonpasr


DEFAULT_DST_DIR = "research/Noesis/docs/findings/model_drift_attribution_smoke"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst_dir", default=DEFAULT_DST_DIR)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="make the paid provider calls",
    )
    parser.add_argument(
        "--approved_cap_usd",
        type=float,
        default=0.0,
        help="explicitly approved maximum spend",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    campaign = rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST
    calls = rnmonpasm.build_default_smoke_calls(campaign)
    policy = rnmonpasm.SmokePolicy()
    rnmonpasm.validate_smoke_plan(calls, policy)
    paths = rnmonpasr.make_artifact_paths(args.dst_dir)
    rnmonpasr.write_smoke_plan(paths, calls, policy, campaign)
    route_counts = {}
    for call in calls:
        route_counts[call.route_name] = route_counts.get(call.route_name, 0) + 1
    print("Attribution-smoke plan: 5-call gate, then 10 calls")
    for route_name in sorted(route_counts):
        print("- %s: %s calls" % (route_name, route_counts[route_name]))
    print("Maximum spend: $%.2f" % policy.maximum_cost_usd)
    print("Plan: %s" % paths.plan_path)
    if not args.execute:
        print("Validation only: no provider calls were made.")
        return 0
    if args.approved_cap_usd < policy.maximum_cost_usd:
        raise ValueError(
            "--approved_cap_usd must be at least %.2f"
            % policy.maximum_cost_usd
        )
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required; export it locally, not in chat"
        )
    import research.Noesis.gateway.gateway_openrouter as rnogaope

    store = rnmonpasr.JsonlObservationStore(paths.observations_path)
    provider = rnogaope.OpenRouterProvider(api_key)
    try:
        try:
            execution = await rnmonpasm.execute_paid_smoke(
                provider, calls, policy, store.record
            )
        except Exception:  # pylint: disable=W0718
            rnmonpasr.write_smoke_failure(
                paths.failure_path, "PERSISTENCE_OR_ORCHESTRATION_FAILURE"
            )
            raise
    finally:
        await provider.aclose()
    rnmonpasr.write_smoke_summary(paths.summary_path, execution)
    print(
        "State=%s calls=%s cost_usd=%.8f"
        % (
            execution.state,
            len(execution.observations),
            execution.total_cost_usd,
        )
    )
    if execution.halt_reason is not None:
        print("Halt reason: %s" % execution.halt_reason)
    print("Observations: %s" % paths.observations_path)
    print("Summary: %s" % paths.summary_path)
    return 0 if execution.state == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_parse_args())))
