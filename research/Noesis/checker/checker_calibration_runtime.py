"""
Execute, summarize, and persist live-provider canary calibration.

Import as:

import research.Noesis.checker.checker_calibration_runtime as rnochcaru
"""

import asyncio
import collections
import math
from typing import Dict, Sequence, Tuple

import psycopg_pool

import research.Noesis.checker.checker_calibration as rnochcal
import research.Noesis.checker.checker_capability as rnochcap
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_store as rnochsto
import research.Noesis.gateway.gateway_providers as rnogapro


def validate_calibration_inputs(
    canaries: Sequence[rnochmod.Canary],
    targets: Sequence[rnochmod.CalibrationTarget],
) -> str:
    """
    Validate inputs before spending money on provider calls.
    """
    if not canaries:
        raise ValueError("calibration needs at least one canary")
    canary_ids = [canary.canary_id for canary in canaries]
    if len(canary_ids) != len(set(canary_ids)):
        raise ValueError("calibration contains duplicate canary ids")
    bank_versions = {canary.bank_version for canary in canaries}
    if len(bank_versions) != 1:
        raise ValueError("calibration must use exactly one bank version")
    if len(targets) != len(set(targets)):
        raise ValueError("calibration contains duplicate targets")
    healthy_count = sum(
        target.role == rnochmod.CalibrationRole.HEALTHY for target in targets
    )
    weak_count = sum(
        target.role == rnochmod.CalibrationRole.WEAK for target in targets
    )
    if healthy_count < 1:
        raise ValueError("calibration needs at least one healthy target")
    if weak_count != 1:
        raise ValueError("calibration needs exactly one weak target")
    return next(iter(bank_versions))


async def _run_attempt(
    semaphore: asyncio.Semaphore,
    provider: rnogapro.Provider,
    canary: rnochmod.Canary,
    target: rnochmod.CalibrationTarget,
    attempt: int,
    policy: rnochmod.CalibrationPolicy,
    measurer: rnochcap.CapabilityMeasurer,
) -> rnochmod.CalibrationObservation:
    """
    Call one target once and deterministically grade a successful response.
    """
    call = rnogapro.ProviderCall(
        model=target.model_id,
        messages=[{"role": "user", "content": canary.prompt}],
        max_tokens=policy.max_tokens,
        timeout_s=policy.timeout_s,
        temperature=0.0,
    )
    async with semaphore:
        result = await provider.call(target.provider_slug, call)
    if result.status == rnogapro.OK:
        grade = measurer.measure(canary, result.text)
        correct = grade.correct
    else:
        correct = None
    return rnochmod.CalibrationObservation(
        canary_id=canary.canary_id,
        target=target,
        attempt=attempt,
        provider_status=result.status,
        correct=correct,
        latency_ms=result.latency_s * 1000,
        cost_usd=result.cost_usd,
        response=result.text,
    )


def _percentile_95(values: Sequence[float]) -> float:
    """
    Return the nearest-rank 95th percentile of a non-empty sample.
    """
    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1
    return ordered[index]


def summarize_calibration(
    targets: Sequence[rnochmod.CalibrationTarget],
    observations: Sequence[rnochmod.CalibrationObservation],
    decisions: Sequence[rnochmod.CanaryCalibration],
) -> Dict[str, object]:
    """
    Build a JSON-compatible calibration report.
    """
    accepted_ids = {
        decision.canary_id for decision in decisions if decision.accepted
    }
    target_stats = []
    for target in targets:
        rows = [row for row in observations if row.target == target]
        successful = [row for row in rows if row.provider_status == rnogapro.OK]
        gradable = [row for row in successful if row.correct is not None]
        accuracy = None
        if gradable:
            accuracy = sum(row.correct is True for row in gradable) / len(
                gradable
            )
        latencies = [row.latency_ms for row in successful]
        survivor_rows = [
            row for row in gradable if row.canary_id in accepted_ids
        ]
        survivor_accuracy = None
        if survivor_rows:
            survivor_accuracy = sum(
                row.correct is True for row in survivor_rows
            ) / len(survivor_rows)
        target_stats.append(
            {
                "role": target.role.value,
                "provider_slug": target.provider_slug,
                "model_id": target.model_id,
                "calls": len(rows),
                "ok_calls": len(successful),
                "gradable_calls": len(gradable),
                "accuracy": accuracy,
                "survivor_calls": len(survivor_rows),
                "survivor_accuracy": survivor_accuracy,
                "latency_p95_ms": (
                    _percentile_95(latencies) if latencies else None
                ),
                "cost_usd": sum(row.cost_usd for row in rows),
            }
        )
    reason_counts = collections.Counter(
        decision.reason for decision in decisions
    )
    return {
        "candidate_count": len(decisions),
        "accepted_count": sum(decision.accepted for decision in decisions),
        "rejected_count": sum(not decision.accepted for decision in decisions),
        "reason_counts": dict(sorted(reason_counts.items())),
        "total_calls": len(observations),
        "total_cost_usd": sum(row.cost_usd for row in observations),
        "target_stats": target_stats,
    }


async def execute_calibration(
    provider: rnogapro.Provider,
    canaries: Sequence[rnochmod.Canary],
    targets: Sequence[rnochmod.CalibrationTarget],
    policy: rnochmod.CalibrationPolicy,
    *,
    measurer: rnochcap.CapabilityMeasurer = rnochcap.DEFAULT_MEASURER,
) -> rnochmod.CalibrationExecution:
    """
    Collect and evaluate all required provider observations.
    """
    validate_calibration_inputs(canaries, targets)
    semaphore = asyncio.Semaphore(policy.concurrency)
    attempts = [
        _run_attempt(
            semaphore,
            provider,
            canary,
            target,
            attempt,
            policy,
            measurer,
        )
        for canary in canaries
        for target in targets
        for attempt in range(1, policy.repetitions + 1)
    ]
    observations = tuple(await asyncio.gather(*attempts))
    decisions = rnochcal.evaluate_canaries(
        canaries,
        targets,
        observations,
        policy.repetitions,
        healthy_required_accuracy=policy.healthy_required_accuracy,
        weak_maximum_accuracy=policy.weak_maximum_accuracy,
    )
    summary = summarize_calibration(targets, observations, decisions)
    return rnochmod.CalibrationExecution(observations, decisions, summary)


async def run_calibration(
    pool: psycopg_pool.AsyncConnectionPool,
    provider: rnogapro.Provider,
    canaries: Sequence[rnochmod.Canary],
    targets: Sequence[rnochmod.CalibrationTarget],
    policy: rnochmod.CalibrationPolicy,
    *,
    measurer: rnochcap.CapabilityMeasurer = rnochcap.DEFAULT_MEASURER,
) -> Tuple[int, rnochmod.CalibrationExecution]:
    """
    Register candidates, execute calibration, and persist the complete run.
    """
    bank_version = validate_calibration_inputs(canaries, targets)
    await rnochsto.register_canary_bank(pool, canaries)
    configuration = {
        "grader_version": measurer.version,
        "max_tokens": policy.max_tokens,
        "timeout_s": policy.timeout_s,
        "concurrency": policy.concurrency,
        "healthy_required_accuracy": policy.healthy_required_accuracy,
        "weak_maximum_accuracy": policy.weak_maximum_accuracy,
    }
    run_id = await rnochsto.start_calibration_run(
        pool,
        bank_version,
        targets,
        policy.repetitions,
        configuration,
    )
    try:
        execution = await execute_calibration(
            provider,
            canaries,
            targets,
            policy,
            measurer=measurer,
        )
        await rnochsto.record_calibration_observations(
            pool, run_id, execution.observations
        )
        completed = await rnochsto.complete_calibration_run(
            pool, run_id, execution.decisions, execution.summary
        )
        if not completed:
            raise RuntimeError(
                "calibration run %s was no longer RUNNING" % run_id
            )
    except Exception as error:
        await rnochsto.fail_calibration_run(pool, run_id, str(error))
        raise
    return run_id, execution
