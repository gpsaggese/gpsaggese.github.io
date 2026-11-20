#!/usr/bin/env python
"""Validate or execute the paid 180-call Lane 2 fingerprint pilot."""

import argparse
import asyncio
import dataclasses
import getpass
import os
from typing import Optional, Sequence

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
import research.Noesis.monitor.monitor_live_fingerprint as rnmonlifp
import research.Noesis.monitor.monitor_paid_execution as rnmonpaex
import research.Noesis.monitor.monitor_paid_report as rnmonpare


DEFAULT_BANK_PATH = (
    "research/Noesis/data/model_drift_fingerprint_bank_v1.jsonl"
)
DEFAULT_DST_DIR = (
    "research/Noesis/docs/findings/model_drift_live_fingerprint"
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank_path", default=DEFAULT_BANK_PATH)
    parser.add_argument("--dst_dir", default=DEFAULT_DST_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved_cap_usd", type=float, default=0.0)
    parser.add_argument(
        "--prompt_for_key",
        action="store_true",
        help="Securely prompt for the API key without saving it to history.",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    manifest = rnmonlifp.LIVE_FINGERPRINT_MANIFEST
    protocol = rnmonlifp.LiveFingerprintProtocol()
    prompts = rnmonlifp.load_live_fingerprint_bank(args.bank_path)
    calls = rnmonlifp.build_live_fingerprint_calls(
        prompts, manifest, protocol
    )
    policy = protocol.make_execution_policy()
    detector_protocol = protocol.make_detector_protocol()
    paths = rnmonpare.make_paid_artifact_paths(args.dst_dir)
    plan_summary = rnmonlifp.summarize_live_fingerprint_plan(calls)
    rnmonpare.write_paid_plan(
        paths.plan_path,
        "noesis-live-fingerprint-collection-v1",
        calls,
        policy,
        {
            "campaign_id": manifest.campaign_id,
            "campaign_manifest_hash": rnmonexma.phase_zero_manifest_hash(
                manifest
            ),
            "bank_hash": rnmonlifp.live_fingerprint_bank_hash(prompts),
            "detector_protocol": dataclasses.asdict(detector_protocol),
            "sampling_temperature": protocol.sampling_temperature,
            "max_tokens": protocol.max_tokens,
            "analysis_scope": (
                "single live A/A test, full swap test, and conditional "
                "partial-mixture sensitivity"
            ),
            **plan_summary,
        },
    )
    print("Live fingerprint pilot: %s calls" % len(calls))
    print("- Cloudflare 70B reference-a: 60 calls")
    print("- Cloudflare 70B reference-b: 60 calls")
    print("- Cloudflare 8B alternative: 60 calls")
    print("Sampling temperature: %.1f" % protocol.sampling_temperature)
    print("Maximum spend: $%.2f" % policy.maximum_cost_usd)
    print("Plan: %s" % paths.plan_path)
    if not args.execute:
        print("Validation only: no provider calls were made.")
        return 0
    if args.approved_cap_usd < policy.maximum_cost_usd:
        raise ValueError("--approved_cap_usd must be at least 2.00")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and args.prompt_for_key:
        api_key = getpass.getpass(
            "Paste the OpenRouter API key (hidden), then press Enter: "
        )
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
