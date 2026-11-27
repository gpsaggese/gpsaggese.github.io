#!/usr/bin/env python
"""Collect replacements for unusable Lane 2 calls without hiding failures."""

import argparse
import asyncio
import dataclasses
import getpass
import json
import os
from typing import Mapping, Optional, Sequence

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_live_fingerprint as rnmonlifp
import research.Noesis.monitor.monitor_paid_execution as rnmonpaex
import research.Noesis.monitor.monitor_paid_report as rnmonpare


DEFAULT_BANK_PATH = (
    "research/Noesis/data/model_drift_fingerprint_bank_v1.jsonl"
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank_path", default=DEFAULT_BANK_PATH)
    parser.add_argument("--prior_observations_path", required=True)
    parser.add_argument("--dst_dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved_cap_usd", type=float, default=0.0)
    parser.add_argument("--prompt_for_key", action="store_true")
    return parser.parse_args(argv)


def _load_rows(path: str) -> Sequence[Mapping[str, object]]:
    rows = []
    with open(path, encoding="utf-8") as file_:
        for line in file_:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_top_up_calls(
    original_calls: Sequence[rnmonpaex.PaidCallSpec],
    rows: Sequence[Mapping[str, object]],
) -> Sequence[rnmonpaex.PaidCallSpec]:
    by_id = {call.call_id: call for call in original_calls}
    replacements = []
    for row in rows:
        usable = (
            row.get("provider_status") == "ok"
            and row.get("attribution_matches") is True
            and row.get("usage_complete") is True
            and isinstance(row.get("response"), str)
            and bool(row.get("response"))
        )
        if usable:
            continue
        call_id = str(row.get("call_id", ""))
        if call_id not in by_id:
            raise ValueError("prior observation is not in the frozen plan")
        original = by_id[call_id]
        replacements.append(
            dataclasses.replace(
                original,
                call_id="topup-01-%s" % original.call_id,
                batch=1,
                metadata=original.metadata + (("replaces_call_id", call_id),),
            )
        )
    if not replacements:
        raise ValueError("prior collection has no unusable calls to top up")
    return tuple(replacements)


async def _run(args: argparse.Namespace) -> int:
    protocol = rnmonlifp.LiveFingerprintProtocol()
    prompts = rnmonlifp.load_live_fingerprint_bank(args.bank_path)
    original_calls = rnmonlifp.build_live_fingerprint_calls(prompts)
    prior_rows = _load_rows(args.prior_observations_path)
    calls = _build_top_up_calls(original_calls, prior_rows)
    maximum_cost = 0.10
    policy = rnmonpaex.PaidExecutionPolicy(
        maximum_calls=len(calls),
        maximum_cost_usd=maximum_cost,
        per_call_reserve_usd=0.01,
    )
    paths = rnmonpare.make_paid_artifact_paths(args.dst_dir)
    rnmonpare.write_paid_plan(
        paths.plan_path,
        "noesis-live-fingerprint-topup-v1",
        calls,
        policy,
        {
            "bank_hash": rnmonlifp.live_fingerprint_bank_hash(prompts),
            "prior_observations_path": args.prior_observations_path,
            "replacement_calls": len(calls),
            "original_failures_preserved": True,
        },
    )
    print("Fingerprint top-up: %s call(s)" % len(calls))
    print("Maximum additional spend: $%.2f" % maximum_cost)
    print("Plan: %s" % paths.plan_path)
    if not args.execute:
        print("Validation only: no provider calls were made.")
        return 0
    if args.approved_cap_usd < maximum_cost:
        raise ValueError("--approved_cap_usd must be at least 0.10")
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
    return 0 if execution.state == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_parse_args())))
