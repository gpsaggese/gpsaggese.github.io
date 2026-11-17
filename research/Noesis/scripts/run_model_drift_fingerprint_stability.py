#!/usr/bin/env python
"""Validate or execute the compressed 180-call 70B stability campaign."""

import argparse
import asyncio
import getpass
import os
from typing import Optional, Sequence

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
import research.Noesis.monitor.monitor_fingerprint_stability as rnmonfist
import research.Noesis.monitor.monitor_live_fingerprint as rnmonlifp
import research.Noesis.monitor.monitor_paid_execution as rnmonpaex
import research.Noesis.monitor.monitor_paid_report as rnmonpare


DEFAULT_BANK_PATH = "research/Noesis/data/model_drift_fingerprint_bank_v1.jsonl"
DEFAULT_DST_DIR = (
    "research/Noesis/docs/findings/model_drift_fingerprint_stability_live"
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank_path", default=DEFAULT_BANK_PATH)
    parser.add_argument("--dst_dir", default=DEFAULT_DST_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved_cap_usd", type=float, default=0.0)
    parser.add_argument("--prompt_for_key", action="store_true")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    protocol = rnmonfist.FingerprintStabilityProtocol()
    prompts = rnmonlifp.load_live_fingerprint_bank(args.bank_path)
    calls = rnmonfist.build_stability_calls(prompts, protocol)
    policy = protocol.make_execution_policy()
    paths = rnmonpare.make_paid_artifact_paths(args.dst_dir)
    rnmonpare.write_paid_plan(
        paths.plan_path,
        "noesis-live-fingerprint-stability-v1",
        calls,
        policy,
        {
            "campaign_id": "noesis-qos-phase0-stability-compressed-v1",
            "parent_campaign_manifest_hash": rnmonexma.phase_zero_manifest_hash(
                rnmonlifp.LIVE_FINGERPRINT_MANIFEST
            ),
            "bank_hash": rnmonlifp.live_fingerprint_bank_hash(prompts),
            "sampling_temperature": protocol.sampling_temperature,
            "analysis_plan": "five 70B pools, ten pairwise A/A comparisons",
            **rnmonfist.summarize_stability_plan(calls),
        },
    )
    print("Compressed fingerprint stability campaign: %s calls" % len(calls))
    print("- Three new Cloudflare 70B pools: 60 calls each")
    print("Sampling temperature: %.1f" % protocol.sampling_temperature)
    print("Maximum spend: $%.2f" % policy.maximum_cost_usd)
    print("Plan: %s" % paths.plan_path)
    if not args.execute:
        print("Validation only: no provider calls were made.")
        return 0
    if args.approved_cap_usd < policy.maximum_cost_usd:
        raise ValueError("--approved_cap_usd must be at least 0.50")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and args.prompt_for_key:
        api_key = getpass.getpass(
            "Paste the OpenRouter API key (hidden), then press Enter: "
        )
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required")
    import research.Noesis.gateway.gateway_openrouter as rnogaope

    store = rnmonpare.JsonlPaidObservationStore(paths.observations_path)
    provider: rnogapro.Provider = rnogaope.OpenRouterProvider(api_key)
    try:
        execution = await rnmonpaex.execute_paid_plan(
            provider, calls, policy, store.record
        )
    finally:
        await provider.aclose()
    rnmonpare.write_paid_summary(paths.summary_path, execution)
    print(
        "State=%s calls=%s cost_usd=%.8f"
        % (execution.state, len(execution.observations), execution.total_cost_usd)
    )
    if execution.halt_reason:
        print("Halt reason: %s" % execution.halt_reason)
    return 0 if execution.state == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_parse_args())))
