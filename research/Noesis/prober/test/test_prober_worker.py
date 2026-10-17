"""
Test the stoppable scheduled-probe worker.
"""

import asyncio
import datetime
from unittest import mock

import helpers.hunit_test as hunitest

import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.gateway.gateway_faults as rnogafau
import research.Noesis.prober.prober_runtime as rnoprrun
import research.Noesis.prober.prober_worker as rnoprwor


class Test_run_probe_worker(hunitest.TestCase):
    """
    Test repeated cycles and failure recovery.
    """

    def test1(self) -> None:
        """
        Test execution counts are accumulated until graceful shutdown.
        """
        execution = rnochmod.ProbeExecution(1, 2, "ok", True, "MATCH")
        results = (
            rnoprrun.ProbeCycleResult((execution,), ()),
            rnoprrun.ProbeCycleResult((), ("probe 2: failed",)),
        )

        async def _body() -> rnoprwor.ProbeWorkerResult:
            stop_event = asyncio.Event()

            async def _run_due(*args, **kwargs):
                _ = (args, kwargs)
                result = results[_run_due.call_count]
                _run_due.call_count += 1
                if _run_due.call_count == len(results):
                    stop_event.set()
                return result

            _run_due.call_count = 0
            with mock.patch.object(rnoprrun, "run_due_probes", new=_run_due):
                return await rnoprwor.run_probe_worker(
                    mock.sentinel.pool,
                    mock.sentinel.provider,
                    rnogafau.FaultRegistry(),
                    rnochmod.ProbePolicy("v1", "model"),
                    "secret",
                    stop_event,
                    poll_interval_s=0.01,
                )

        actual = asyncio.run(_body())
        self.assertEqual(actual, rnoprwor.ProbeWorkerResult(2, 1, 1))

    def test2(self) -> None:
        """
        Test a whole-cycle exception is counted and retried.
        """

        async def _body() -> rnoprwor.ProbeWorkerResult:
            stop_event = asyncio.Event()

            async def _run_due(*args, **kwargs):
                _ = (args, kwargs)
                _run_due.call_count += 1
                if _run_due.call_count == 1:
                    raise RuntimeError("database unavailable")
                stop_event.set()
                return rnoprrun.ProbeCycleResult((), ())

            _run_due.call_count = 0
            with mock.patch.object(rnoprrun, "run_due_probes", new=_run_due):
                return await rnoprwor.run_probe_worker(
                    mock.sentinel.pool,
                    mock.sentinel.provider,
                    rnogafau.FaultRegistry(),
                    rnochmod.ProbePolicy("v1", "model"),
                    "secret",
                    stop_event,
                    poll_interval_s=0.001,
                    clock_func=lambda: datetime.datetime.now(
                        datetime.timezone.utc
                    ),
                )

        actual = asyncio.run(_body())
        self.assertEqual(actual, rnoprwor.ProbeWorkerResult(2, 0, 1))

    def test3(self) -> None:
        """
        Test unsafe worker configuration fails before any cycle.
        """
        with self.assertRaises(ValueError):
            asyncio.run(
                rnoprwor.run_probe_worker(
                    mock.sentinel.pool,
                    mock.sentinel.provider,
                    rnogafau.FaultRegistry(),
                    rnochmod.ProbePolicy("v1", "model"),
                    "",
                    asyncio.Event(),
                )
            )
