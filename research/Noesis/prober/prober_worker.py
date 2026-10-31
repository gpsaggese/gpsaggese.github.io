"""
Run scheduled probing repeatedly until a caller requests shutdown.

Import as:

import research.Noesis.prober.prober_worker as rnoprwor
"""

import asyncio
import dataclasses
import logging
from datetime import datetime, timezone
from typing import Callable

import psycopg_pool

import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_runtime as rnochrun
import research.Noesis.checker.checker_sink as rnochsink
import research.Noesis.gateway.gateway_faults as rnogafau
import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.prober.prober_runtime as rnoprrun

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ProbeWorkerResult:
    """
    Summarize work completed before graceful shutdown.
    """

    cycles: int
    executions: int
    errors: int
    finalized_contracts: int = 0
    published_reports: int = 0


@dataclasses.dataclass(frozen=True)
class ProbeCheckerCycleResult:
    """
    Represent probe and checker outcomes from one shared-clock cycle.
    """

    probes: rnoprrun.ProbeCycleResult
    checker: rnochrun.CheckerPassResult


async def run_probe_cycle(
    pool: psycopg_pool.AsyncConnectionPool,
    provider: rnogapro.Provider,
    faults: rnogafau.FaultRegistry,
    policy: rnochmod.ProbePolicy,
    schedule_secret: str,
    now: datetime,
    *,
    sink: rnochsink.VerdictSink,
    checker_policy: rnochmod.WindowPolicy = rnochmod.WindowPolicy(),
    checker_version: str = "m2-v1",
) -> ProbeCheckerCycleResult:
    """
    Run due probes before evaluating contract windows at the same timestamp.
    """
    probes = await rnoprrun.run_due_probes(
        pool,
        provider,
        faults,
        policy,
        schedule_secret,
        now,
    )
    checker = await rnochrun.run_checker_pass(
        pool,
        sink,
        checker_policy,
        now,
        checker_version=checker_version,
        bank_version=policy.bank_version,
    )
    return ProbeCheckerCycleResult(probes, checker)


async def _wait_for_stop(
    stop_event: asyncio.Event, poll_interval_s: float
) -> None:
    """
    Wait for shutdown or return when the next cycle is due.
    """
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_s)
    except asyncio.TimeoutError:
        pass


async def run_probe_worker(
    pool: psycopg_pool.AsyncConnectionPool,
    provider: rnogapro.Provider,
    faults: rnogafau.FaultRegistry,
    policy: rnochmod.ProbePolicy,
    schedule_secret: str,
    stop_event: asyncio.Event,
    *,
    sink: rnochsink.VerdictSink,
    checker_policy: rnochmod.WindowPolicy = rnochmod.WindowPolicy(),
    checker_version: str = "m2-v1",
    poll_interval_s: float = 1.0,
    clock_func: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> ProbeWorkerResult:
    """
    Run probe cycles until `stop_event` is set.

    A failed database-level cycle is logged and retried. Per-probe and
    per-contract failures are isolated inside the combined cycle.
    """
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    if not schedule_secret:
        raise ValueError("schedule_secret must not be empty")
    if not checker_version:
        raise ValueError("checker_version must not be empty")
    cycles = 0
    execution_count = 0
    error_count = 0
    finalized_count = 0
    published_count = 0
    while not stop_event.is_set():
        cycles += 1
        try:
            result = await run_probe_cycle(
                pool,
                provider,
                faults,
                policy,
                schedule_secret,
                clock_func(),
                sink=sink,
                checker_policy=checker_policy,
                checker_version=checker_version,
            )
            execution_count += len(result.probes.executions)
            cycle_finalized = sum(
                evaluation.finalized for evaluation in result.checker.evaluations
            )
            finalized_count += cycle_finalized
            published_count += len(result.checker.published_reports)
            cycle_errors = len(result.probes.errors) + len(result.checker.errors)
            error_count += cycle_errors
            if (
                result.probes.executions
                or result.checker.evaluations
                or cycle_errors
            ):
                _LOG.info(
                    "Probe cycle %s: executions=%s finalized=%s"
                    " published=%s errors=%s",
                    cycles,
                    len(result.probes.executions),
                    cycle_finalized,
                    len(result.checker.published_reports),
                    cycle_errors,
                )
        except Exception:
            error_count += 1
            _LOG.exception("Probe cycle %s failed; retrying", cycles)
        await _wait_for_stop(stop_event, poll_interval_s)
    return ProbeWorkerResult(
        cycles,
        execution_count,
        error_count,
        finalized_count,
        published_count,
    )
