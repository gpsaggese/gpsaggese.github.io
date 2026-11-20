#!/usr/bin/env python
"""
Validate or execute the $10 four-route controlled warm-up.

The bank must be a frozen 60-row JSONL file with 12 items in each approved
family. Validation is the default. Live execution additionally requires
``OPENROUTER_API_KEY``, ``--execute``, and ``--approved_cap_usd 10``.
"""

import argparse
import asyncio
import os
from typing import Optional, Sequence

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_controlled_warmup as rnmoncowar
import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
import research.Noesis.monitor.monitor_paid_execution as rnmonpaex
import research.Noesis.monitor.monitor_paid_report as rnmonpare


DEFAULT_DST_DIR = "research/Noesis/docs/findings/model_drift_controlled_warmup"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank_path", required=True)
    parser.add_argument("--dst_dir", default=DEFAULT_DST_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved_cap_usd", type=float, default=0.0)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    manifest = rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST
    protocol = rnmoncowar.WarmupProtocol()
    items = rnmoncowar.load_warmup_bank(args.bank_path)
    calls = rnmoncowar.build_warmup_calls(items, manifest, protocol)
    policy = protocol.make_execution_policy()
    paths = rnmonpare.make_paid_artifact_paths(args.dst_dir)
    plan_summary = rnmoncowar.summarize_warmup_plan(calls)
    rnmonpare.write_paid_plan(
        paths.plan_path,
        "noesis-controlled-warmup-v1",
        calls,
        policy,
        {
            "campaign_id": manifest.campaign_id,
            "campaign_manifest_hash": rnmonexma.phase_zero_manifest_hash(
                manifest
            ),
            "bank_hash": rnmoncowar.warmup_bank_hash(items),
            **plan_summary,
        },
    )
    print("Controlled warm-up: %s calls" % len(calls))
    print("Maximum spend: $%.2f" % policy.maximum_cost_usd)
    print("Plan: %s" % paths.plan_path)
    if not args.execute:
        print("Validation only: no provider calls were made.")
        return 0
    if args.approved_cap_usd < policy.maximum_cost_usd:
        raise ValueError("--approved_cap_usd must be at least 10.00")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required; export it locally, not in chat"
        )
    import research.Noesis.gateway.gateway_openrouter as rnogaope

    store = rnmonpare.JsonlPaidObservationStore(paths.observations_path)
    provider: rnogapro.Provider = rnogaope.OpenRouterProvider(api_key)
    try:
        try:
            execution = await rnmonpaex.execute_paid_plan(
                provider, calls, policy, store.record
            )
        except Exception:  # pylint: disable=W0718
            rnmonpare.write_paid_failure(
                paths.failure_path, "PERSISTENCE_OR_ORCHESTRATION_FAILURE"
            )
            raise
    finally:
        await provider.aclose()
    rnmonpare.write_paid_summary(paths.summary_path, execution)
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
    return 0 if execution.state == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_parse_args())))
