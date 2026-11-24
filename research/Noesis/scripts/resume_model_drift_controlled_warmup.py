#!/usr/bin/env python
"""Resume only unattempted calls after a reviewed warm-up safety halt."""

import argparse
import asyncio
import decimal
import hashlib
import json
import os
from typing import Mapping, Optional, Sequence

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_controlled_warmup as rnmoncowar
import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
import research.Noesis.monitor.monitor_paid_execution as rnmonpaex
import research.Noesis.monitor.monitor_paid_report as rnmonpare


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank_path", required=True)
    parser.add_argument("--prior_run_dir", required=True)
    parser.add_argument("--dst_dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved_total_cap_usd", type=float, default=0.0)
    return parser.parse_args(argv)


def _load_jsonl(path: str) -> Sequence[Mapping[str, object]]:
    rows = []
    with open(path, encoding="utf-8") as file_:
        for line_number, line in enumerate(file_, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    "invalid prior observation row %s: %s"
                    % (line_number, error)
                ) from error
    return rows


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_:
        for chunk in iter(lambda: file_.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _run(args: argparse.Namespace) -> int:
    manifest = rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST
    protocol = rnmoncowar.WarmupProtocol()
    items = rnmoncowar.load_warmup_bank(args.bank_path)
    calls = rnmoncowar.build_warmup_calls(items, manifest, protocol)
    observations_path = os.path.join(
        args.prior_run_dir, "observations.jsonl"
    )
    summary_path = os.path.join(args.prior_run_dir, "summary.json")
    prior = _load_jsonl(observations_path)
    with open(summary_path, encoding="utf-8") as file_:
        prior_summary = json.load(file_)
    if prior_summary.get("state") != "HALTED":
        raise ValueError("resume requires a reviewed HALTED parent run")
    if len(prior) != prior_summary.get("completed_calls"):
        raise ValueError("parent summary does not match persisted observations")
    remaining = rnmoncowar.build_unattempted_calls(calls, prior)
    prior_cost_decimal = sum(
        (decimal.Decimal(str(row["cost_usd"])) for row in prior),
        decimal.Decimal("0"),
    )
    remaining_cap_decimal = (
        decimal.Decimal(str(protocol.maximum_cost_usd)) - prior_cost_decimal
    )
    prior_cost = float(prior_cost_decimal)
    remaining_cap = float(remaining_cap_decimal)
    policy = rnmonpaex.PaidExecutionPolicy(
        maximum_calls=len(remaining),
        maximum_cost_usd=remaining_cap,
        per_call_reserve_usd=protocol.per_call_reserve_usd,
    )
    paths = rnmonpare.make_paid_artifact_paths(args.dst_dir)
    rnmonpare.write_paid_plan(
        paths.plan_path,
        "noesis-controlled-warmup-resume-v1",
        remaining,
        policy,
        {
            "campaign_manifest_hash": rnmonexma.phase_zero_manifest_hash(
                manifest
            ),
            "bank_hash": rnmoncowar.warmup_bank_hash(items),
            "parent_observations_sha256": _file_hash(observations_path),
            "parent_halt_reason": prior_summary.get("halt_reason"),
            "prior_attempted_calls": len(prior),
            "prior_cost_usd": str(prior_cost_decimal),
            "remaining_total_cap_usd": str(remaining_cap_decimal),
            "unattempted_calls": len(remaining),
            "retry_failed_calls": False,
        },
    )
    print("Reviewed parent halt: %s" % prior_summary.get("halt_reason"))
    print("Preserved attempts: %s" % len(prior))
    print("Unattempted calls: %s" % len(remaining))
    print("Remaining total cap: $%.8f" % remaining_cap)
    print("Resume plan: %s" % paths.plan_path)
    if not args.execute:
        print("Validation only: no provider calls were made.")
        return 0
    if args.approved_total_cap_usd < protocol.maximum_cost_usd:
        raise ValueError("--approved_total_cap_usd must be at least 10.00")
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
                provider, remaining, policy, store.record
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
        "State=%s new_calls=%s combined_calls=%s combined_cost_usd=%.8f"
        % (
            execution.state,
            len(execution.observations),
            len(prior) + len(execution.observations),
            prior_cost + execution.total_cost_usd,
        )
    )
    if execution.halt_reason is not None:
        print("Halt reason: %s" % execution.halt_reason)
    return 0 if execution.state == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_parse_args())))
