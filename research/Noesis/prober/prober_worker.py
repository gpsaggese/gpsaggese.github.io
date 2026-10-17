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
    poll_interval_s: float = 1.0,
    clock_func: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> ProbeWorkerResult:
    """
    Run probe cycles until `stop_event` is set.

    A failed database-level cycle is logged and retried. Per-contract and
    per-probe failures are already isolated by `run_due_probes()`.
    """
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    if not schedule_secret:
        raise ValueError("schedule_secret must not be empty")
    cycles = 0
    execution_count = 0
    error_count = 0
    while not stop_event.is_set():
        cycles += 1
        try:
            result = await rnoprrun.run_due_probes(
                pool,
                provider,
                faults,
                policy,
                schedule_secret,
                clock_func(),
            )
            execution_count += len(result.executions)
            error_count += len(result.errors)
            if result.executions or result.errors:
                _LOG.info(
                    "Probe cycle %s: executions=%s errors=%s",
                    cycles,
                    len(result.executions),
                    len(result.errors),
                )
        except Exception:
            error_count += 1
            _LOG.exception("Probe cycle %s failed; retrying", cycles)
        await _wait_for_stop(stop_event, poll_interval_s)
    return ProbeWorkerResult(cycles, execution_count, error_count)
