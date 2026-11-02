"""
Test the stoppable scheduled-probe worker.
"""

import asyncio
import datetime
from unittest import mock

import helpers.hunit_test as hunitest
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_runtime as rnochrun
import research.Noesis.checker.checker_sink as rnochsink
import research.Noesis.gateway.gateway_faults as rnogafau
import research.Noesis.prober.prober_runtime as rnoprrun
import research.Noesis.prober.prober_worker as rnoprwor

_NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _report() -> rnochmod.VerdictReport:
    assessment = rnochmod.MetricAssessment(
        successes=1,
        total=1,
        rate=1.0,
        upper_bound=1.0,
        promised_rate=0.9,
        sufficient=True,
        conclusive_failure=False,
    )
    return rnochmod.VerdictReport(
        contract_id=42,
        seller_id="seller_1",
        verdict=rnochmod.Verdict.PASSED,
        disposition=rnochmod.CheckerDisposition.VALID,
        latency=assessment,
        quality=assessment,
        reason_codes=("COMPLIANT",),
        excluded_counts={},
        window_start=_NOW,
        window_end=_NOW + datetime.timedelta(seconds=30),
        checker_version="m2-v1",
        bank_version="v1",
    )


def _evaluation(report: rnochmod.VerdictReport) -> rnochmod.ContractEvaluation:
    readiness = rnochmod.WindowReadiness(
        ready=True,
        reason=rnochmod.WindowCloseReason.TARGETS_REACHED,
        counts=rnochmod.WindowEvidenceCounts(1, 1),
        deadline=_NOW + datetime.timedelta(seconds=60),
    )
    return rnochmod.ContractEvaluation(42, readiness, True, report)


# #############################################################################
# Test_run_probe_cycle
# #############################################################################


class Test_run_probe_cycle(hunitest.TestCase):
    """
    Test probe-before-checker ordering and sink injection.
    """

    def test1(self) -> None:
        """
        Test one timestamp drives probes before a complete checker report.
        """
        events = []
        report = _report()
        evaluation = _evaluation(report)
        sink = rnochsink.RecordingVerdictSink()

        async def _run_due(*args, **kwargs):
            self.assertEqual(args[-1], _NOW)
            self.assertEqual(kwargs, {})
            events.append("probes")
            return rnoprrun.ProbeCycleResult((), ())

        async def _run_checker(pool, sink_arg, checker_policy, now, **kwargs):
            _ = (pool, checker_policy)
            self.assertIs(sink_arg, sink)
            self.assertEqual(now, _NOW)
            self.assertEqual(kwargs["checker_version"], "m2-v1")
            self.assertEqual(kwargs["bank_version"], "v1")
            events.append("checker")
            await sink_arg.publish(report)
            return rnochrun.CheckerPassResult(
                (evaluation,),
                (report,),
                (),
            )

        async def _body() -> rnoprwor.ProbeCheckerCycleResult:
            with (
                mock.patch.object(rnoprrun, "run_due_probes", new=_run_due),
                mock.patch.object(
                    rnochrun,
                    "run_checker_pass",
                    new=_run_checker,
                ),
            ):
                return await rnoprwor.run_probe_cycle(
                    mock.sentinel.pool,
                    mock.sentinel.provider,
                    rnogafau.FaultRegistry(),
                    rnochmod.ProbePolicy("v1", "model"),
                    "secret",
                    _NOW,
                    sink=sink,
                )

        actual = asyncio.run(_body())
        self.assertEqual(events, ["probes", "checker"])
        self.assertEqual(actual.checker.published_reports, (report,))
        self.assertEqual(sink.reports, (report,))


class Test_run_probe_worker(hunitest.TestCase):
    """
    Test repeated cycles and failure recovery.
    """

    def test1(self) -> None:
        """
        Test execution counts are accumulated until graceful shutdown.
        """
        execution = rnochmod.ProbeExecution(1, 2, "ok", True, "MATCH")
        report = _report()
        results = (
            rnoprwor.ProbeCheckerCycleResult(
                rnoprrun.ProbeCycleResult((execution,), ()),
                rnochrun.CheckerPassResult(
                    (_evaluation(report),),
                    (report,),
                    (),
                ),
            ),
            rnoprwor.ProbeCheckerCycleResult(
                rnoprrun.ProbeCycleResult((), ("probe 2: failed",)),
                rnochrun.CheckerPassResult(
                    (),
                    (),
                    ("contract 2 evaluation: failed",),
                ),
            ),
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
            with mock.patch.object(rnoprwor, "run_probe_cycle", new=_run_due):
                return await rnoprwor.run_probe_worker(
                    mock.sentinel.pool,
                    mock.sentinel.provider,
                    rnogafau.FaultRegistry(),
                    rnochmod.ProbePolicy("v1", "model"),
                    "secret",
                    stop_event,
                    sink=rnochsink.RecordingVerdictSink(),
                    poll_interval_s=0.01,
                )

        actual = asyncio.run(_body())
        self.assertEqual(actual, rnoprwor.ProbeWorkerResult(2, 1, 2, 1, 1))

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
                return rnoprwor.ProbeCheckerCycleResult(
                    rnoprrun.ProbeCycleResult((), ()),
                    rnochrun.CheckerPassResult((), (), ()),
                )

            _run_due.call_count = 0
            with mock.patch.object(rnoprwor, "run_probe_cycle", new=_run_due):
                return await rnoprwor.run_probe_worker(
                    mock.sentinel.pool,
                    mock.sentinel.provider,
                    rnogafau.FaultRegistry(),
                    rnochmod.ProbePolicy("v1", "model"),
                    "secret",
                    stop_event,
                    sink=rnochsink.RecordingVerdictSink(),
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
                    sink=rnochsink.RecordingVerdictSink(),
                )
            )
