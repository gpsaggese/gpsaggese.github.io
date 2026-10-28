"""
Test verdict-report construction and the mocked market-facing sink.
"""

import asyncio
import dataclasses
import datetime

import helpers.hunit_test as hunitest
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_sink as rnochsink

_START = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
_END = _START + datetime.timedelta(seconds=60)


def _assessment() -> rnochmod.MetricAssessment:
    return rnochmod.MetricAssessment(
        successes=18,
        total=20,
        rate=0.9,
        upper_bound=0.95,
        promised_rate=0.9,
        sufficient=True,
        conclusive_failure=False,
    )


def _decision(
    *,
    verdict: rnochmod.Verdict = rnochmod.Verdict.PASSED,
    disposition: rnochmod.CheckerDisposition = (
        rnochmod.CheckerDisposition.VALID
    ),
) -> rnochmod.VerdictDecision:
    return rnochmod.VerdictDecision(
        verdict=verdict,
        disposition=disposition,
        latency=_assessment(),
        quality=_assessment(),
        reason_codes=("COMPLIANT",),
        excluded_counts={"latency:EXCLUDED_BUYER_ERROR": 1},
    )


def _report(
    *,
    verdict: rnochmod.Verdict = rnochmod.Verdict.PASSED,
    disposition: rnochmod.CheckerDisposition = (
        rnochmod.CheckerDisposition.VALID
    ),
) -> rnochmod.VerdictReport:
    return rnochsink.build_verdict_report(
        42,
        "groq",
        _decision(verdict=verdict, disposition=disposition),
        _START,
        _END,
        checker_version="m2-v1",
        bank_version="v1",
    )


# #############################################################################
# Test_build_verdict_report
# #############################################################################


class Test_build_verdict_report(hunitest.TestCase):
    """
    Test stable report construction and boundary validation.
    """

    def test1(self) -> None:
        """
        Test every decision and identity field is preserved.
        """
        # Prepare inputs.
        decision = _decision()
        # Run test.
        actual = rnochsink.build_verdict_report(
            42,
            "groq",
            decision,
            _START,
            _END,
            checker_version="m2-v1",
            bank_version="v1",
        )
        # Prepare outputs.
        expected = rnochmod.VerdictReport(
            contract_id=42,
            seller_id="groq",
            verdict=decision.verdict,
            disposition=decision.disposition,
            latency=decision.latency,
            quality=decision.quality,
            reason_codes=decision.reason_codes,
            excluded_counts=decision.excluded_counts,
            window_start=_START,
            window_end=_END,
            checker_version="m2-v1",
            bank_version="v1",
        )
        # Check outputs.
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test invalid report identity, version, and time fields are rejected.
        """
        # Prepare inputs.
        report = _report()
        naive = datetime.datetime(2026, 1, 1)
        cases = [
            ({"contract_id": 0}, "contract_id must be positive"),
            ({"seller_id": ""}, "seller_id must not be empty"),
            ({"window_start": naive}, "window_start must be timezone-aware"),
            ({"window_end": naive}, "window_end must be timezone-aware"),
            (
                {"window_end": _START - datetime.timedelta(seconds=1)},
                "window_end must not precede window_start",
            ),
            ({"checker_version": ""}, "checker_version must not be empty"),
            ({"bank_version": ""}, "bank_version must not be empty"),
        ]
        # Run test and check outputs.
        for changes, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                dataclasses.replace(report, **changes)


# #############################################################################
# Test_is_actionable_report
# #############################################################################


class Test_is_actionable_report(hunitest.TestCase):
    """
    Test that only safe seller outcomes may reach a market consumer.
    """

    def test1(self) -> None:
        """
        Test valid pass and failure reports are actionable.
        """
        for verdict in [rnochmod.Verdict.PASSED, rnochmod.Verdict.FAILED]:
            self.assertTrue(
                rnochsink.is_actionable_report(_report(verdict=verdict))
            )

    def test2(self) -> None:
        """
        Test insufficient and checker-unhealthy reports are not actionable.
        """
        cases = [
            _report(verdict=rnochmod.Verdict.INSUFFICIENT),
            _report(
                verdict=rnochmod.Verdict.FAILED,
                disposition=rnochmod.CheckerDisposition.CHECKER_UNHEALTHY,
            ),
        ]
        for report in cases:
            self.assertFalse(rnochsink.is_actionable_report(report))


# #############################################################################
# TestRecordingVerdictSink
# #############################################################################


class TestRecordingVerdictSink(hunitest.TestCase):
    """
    Test the in-memory market-facing stand-in.
    """

    def test1(self) -> None:
        """
        Test reports are preserved in publication order through a tuple view.
        """
        # Prepare inputs.
        sink = rnochsink.RecordingVerdictSink()
        first = _report()
        second = _report(verdict=rnochmod.Verdict.FAILED)

        async def _publish() -> None:
            await sink.publish(first)
            await sink.publish(second)

        # Run test.
        asyncio.run(_publish())
        # Check outputs.
        self.assertEqual(sink.reports, (first, second))
