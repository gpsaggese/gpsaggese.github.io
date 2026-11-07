"""
Execute one bounded, standalone model-monitoring pass.

Import as:

import research.Noesis.monitor.monitor_runtime as rnmonrun
"""

import asyncio
import collections
import time
from typing import Awaitable, Callable, Dict, Optional, Sequence, Tuple

import psycopg_pool

import research.Noesis.checker.checker_canaries as rnochcan
import research.Noesis.checker.checker_capability as rnochcap
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_store as rnochsto
import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_models as rnmonmod
import research.Noesis.monitor.monitor_store as rnmonsto

INTERNAL_ERROR = "internal_error"
PROVIDER_EXCEPTION = "PROVIDER_EXCEPTION"
GRADER_EXCEPTION = "GRADER_EXCEPTION"

ObservationCallback = Callable[[rnmonmod.MonitoringObservation], Awaitable[None]]


def validate_monitoring_inputs(
    canaries: Sequence[rnochmod.Canary],
    target: rnmonmod.MonitoringTarget,
    policy: rnmonmod.MonitoringPolicy,
) -> None:
    """
    Validate a monitoring plan before making provider calls.
    """
    _ = target
    if not canaries:
        raise ValueError("monitoring needs at least one canary")
    canary_ids = [canary.canary_id for canary in canaries]
    if len(canary_ids) != len(set(canary_ids)):
        raise ValueError("monitoring contains duplicate canary ids")
    bank_versions = {canary.bank_version for canary in canaries}
    if bank_versions != {policy.bank_version}:
        raise ValueError(
            "monitoring canaries must use bank version %r" % policy.bank_version
        )
    if policy.canary_count > len(canaries):
        raise ValueError(
            "monitoring requested %s canaries from a bank of %s"
            % (policy.canary_count, len(canaries))
        )


def select_monitoring_canaries(
    canaries: Sequence[rnochmod.Canary],
    run_id: int,
    policy: rnmonmod.MonitoringPolicy,
    selection_secret: str,
) -> Tuple[rnochmod.Canary, ...]:
    """
    Select the secret, reproducible canaries for one monitoring run.
    """
    if run_id <= 0:
        raise ValueError("run_id must be positive")
    selection_key = "monitoring-run:%s" % run_id
    return rnochcan.select_canaries_for_key(
        canaries,
        selection_key,
        policy.canary_count,
        selection_secret,
    )


def build_monitor_call(
    canary: rnochmod.Canary,
    target: rnmonmod.MonitoringTarget,
    policy: rnmonmod.MonitoringPolicy,
) -> rnogapro.ProviderCall:
    """
    Build an ordinary user request without exposing canary metadata.
    """
    return rnogapro.ProviderCall(
        model=target.model_id,
        messages=[{"role": "user", "content": canary.prompt}],
        max_tokens=policy.max_tokens,
        timeout_s=policy.timeout_s,
        temperature=0.0,
    )


def _failure_observation(
    canary: rnochmod.Canary,
    target: rnmonmod.MonitoringTarget,
    provider_status: str,
    grader_version: str,
    reason: str,
    error: Exception,
    latency_ms: float,
) -> rnmonmod.MonitoringObservation:
    """
    Convert one unexpected per-canary exception into durable evidence.
    """
    return rnmonmod.MonitoringObservation(
        canary_id=canary.canary_id,
        target=target,
        provider_status=provider_status,
        correct=None,
        grader_reason=reason,
        grader_version=grader_version,
        latency_ms=latency_ms,
        cost_usd=0.0,
        error=str(error)[:500],
    )


async def observe_canary(
    provider: rnogapro.Provider,
    canary: rnochmod.Canary,
    target: rnmonmod.MonitoringTarget,
    policy: rnmonmod.MonitoringPolicy,
    *,
    measurer: rnochcap.CapabilityMeasurer = rnochcap.DEFAULT_MEASURER,
    clock_func: Callable[[], float] = time.perf_counter,
) -> rnmonmod.MonitoringObservation:
    """
    Call and grade one canary without propagating per-canary failures.
    """
    call = build_monitor_call(canary, target, policy)
    started_at = clock_func()
    try:
        result = await asyncio.wait_for(
            provider.call(target.provider_slug, call),
            timeout=policy.timeout_s,
        )
    except asyncio.TimeoutError:
        latency_ms = max(0.0, (clock_func() - started_at) * 1000)
        error = RuntimeError(
            "runtime timeout after %s seconds" % policy.timeout_s
        )
        return _failure_observation(
            canary,
            target,
            rnogapro.TIMEOUT,
            measurer.version,
            "PROVIDER_STATUS_TIMEOUT",
            error,
            latency_ms,
        )
    except Exception as error:  # pylint: disable=W0718
        latency_ms = max(0.0, (clock_func() - started_at) * 1000)
        return _failure_observation(
            canary,
            target,
            INTERNAL_ERROR,
            measurer.version,
            PROVIDER_EXCEPTION,
            error,
            latency_ms,
        )
    correct: Optional[bool] = None
    grader_reason = "PROVIDER_STATUS_%s" % result.status.upper()
    error = result.error
    if result.status == rnogapro.OK:
        try:
            grade = measurer.measure(canary, result.text)
            correct = grade.correct
            grader_reason = grade.reason
        except Exception as exception:  # pylint: disable=W0718
            grader_reason = GRADER_EXCEPTION
            error = str(exception)[:500]
    return rnmonmod.MonitoringObservation(
        canary_id=canary.canary_id,
        target=target,
        provider_status=result.status,
        correct=correct,
        grader_reason=grader_reason,
        grader_version=measurer.version,
        latency_ms=result.latency_s * 1000,
        cost_usd=result.cost_usd,
        response=result.text,
        served_by=result.served_by,
        model_served=result.model_served,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        http_status=result.http_status,
        error=error,
    )


def summarize_monitoring(
    observations: Sequence[rnmonmod.MonitoringObservation],
) -> Dict[str, object]:
    """
    Build a JSON-compatible summary without making a nerfing decision.
    """
    statuses = collections.Counter(
        observation.provider_status for observation in observations
    )
    gradable = [
        observation
        for observation in observations
        if observation.correct is not None
    ]
    correct_calls = sum(observation.correct is True for observation in gradable)
    accuracy = correct_calls / len(gradable) if gradable else None
    return {
        "total_calls": len(observations),
        "ok_calls": statuses.get(rnogapro.OK, 0),
        "gradable_calls": len(gradable),
        "correct_calls": correct_calls,
        "incorrect_calls": len(gradable) - correct_calls,
        "ungradable_calls": len(observations) - len(gradable),
        "accuracy": accuracy,
        "provider_status_counts": dict(sorted(statuses.items())),
        "total_cost_usd": sum(
            observation.cost_usd for observation in observations
        ),
    }


async def execute_monitoring(
    provider: rnogapro.Provider,
    canaries: Sequence[rnochmod.Canary],
    target: rnmonmod.MonitoringTarget,
    policy: rnmonmod.MonitoringPolicy,
    *,
    measurer: rnochcap.CapabilityMeasurer = rnochcap.DEFAULT_MEASURER,
    on_observation: Optional[ObservationCallback] = None,
) -> rnmonmod.MonitoringExecution:
    """
    Execute selected canaries sequentially and report each immediately.
    """
    validate_monitoring_inputs(canaries, target, policy)
    if len(canaries) != policy.canary_count:
        raise ValueError("execute_monitoring requires the selected canary count")
    observations = []
    for canary in canaries:
        observation = await observe_canary(
            provider,
            canary,
            target,
            policy,
            measurer=measurer,
        )
        if on_observation is not None:
            await on_observation(observation)
        observations.append(observation)
    frozen_observations = tuple(observations)
    return rnmonmod.MonitoringExecution(
        frozen_observations,
        summarize_monitoring(frozen_observations),
    )


async def run_monitoring(
    pool: psycopg_pool.AsyncConnectionPool,
    provider: rnogapro.Provider,
    target: rnmonmod.MonitoringTarget,
    policy: rnmonmod.MonitoringPolicy,
    selection_secret: str,
    *,
    measurer: rnochcap.CapabilityMeasurer = rnochcap.DEFAULT_MEASURER,
) -> Tuple[int, rnmonmod.MonitoringExecution]:
    """
    Select, execute, and persist one standalone monitoring run.
    """
    canaries = await rnochsto.load_active_canaries(pool, policy.bank_version)
    validate_monitoring_inputs(canaries, target, policy)
    run_id = await rnmonsto.start_monitoring_run(
        pool, target, policy, measurer.version
    )
    try:
        selected = select_monitoring_canaries(
            canaries, run_id, policy, selection_secret
        )

        async def _persist(
            observation: rnmonmod.MonitoringObservation,
        ) -> None:
            await rnmonsto.record_monitoring_observation(
                pool, run_id, observation
            )

        execution = await execute_monitoring(
            provider,
            selected,
            target,
            policy,
            measurer=measurer,
            on_observation=_persist,
        )
        completed = await rnmonsto.complete_monitoring_run(
            pool, run_id, execution.summary
        )
        if not completed:
            raise RuntimeError(
                "monitoring run %s was no longer RUNNING" % run_id
            )
    except Exception as error:
        await rnmonsto.fail_monitoring_run(pool, run_id, str(error))
        raise
    return run_id, execution
