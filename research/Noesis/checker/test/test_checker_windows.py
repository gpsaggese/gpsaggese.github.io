"""
Test pure collection-target and deadline readiness logic.
"""

import dataclasses
import datetime
from typing import Iterable, Optional

import helpers.hunit_test as hunitest
import research.Noesis.checker.checker_evidence as rnochevi
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_windows as rnochwin
import research.Noesis.gateway.gateway_providers as rnogapro

_START = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
_TERMS = rnochmod.ContractTerms(42, 2.0, 0.9, 0.85)
_POLICY = rnochmod.WindowPolicy()


def _contract(
    *,
    state: str = "ACTIVE",
    target_requests: int = 2,
    target_canaries: int = 2,
    maximum_seconds: int = 60,
    window_start: Optional[datetime.datetime] = _START,
) -> rnochmod.ContractWindow:
    return rnochmod.ContractWindow(
        terms=_TERMS,
        state=state,
        target_requests=target_requests,
        target_canaries=target_canaries,
        maximum_seconds=maximum_seconds,
        checker_policy="m2-v1",
        window_start=window_start,
        window_end=None,
    )


def _buyer(
    request_id: int,
    *,
    status: str = rnogapro.OK,
    latency_ms: float = 100.0,
    attribution_mismatch: bool = False,
) -> rnochmod.RequestObservation:
    return rnochmod.RequestObservation(
        request_id,
        status,
        latency_ms,
        attribution_mismatch=attribution_mismatch,
    )


def _canary(
    request_id: int,
    *,
    status: str = rnogapro.OK,
    correct: Optional[bool] = True,
) -> rnochmod.RequestObservation:
    return rnochmod.RequestObservation(
        request_id,
        status,
        100.0,
        is_canary=True,
        canary_correct=correct,
    )


def _evidence(
    requests: Iterable[rnochmod.RequestObservation],
) -> rnochmod.EvidenceSet:
    return rnochevi.build_evidence(requests, _TERMS, _POLICY)


# #############################################################################
# Test_count_window_evidence
# #############################################################################


class Test_count_window_evidence(hunitest.TestCase):
    """
    Test that readiness counts only usable seller evidence.
    """

    def test1(self) -> None:
        """
        Test seller outcomes count while non-seller and checker errors do not.
        """
        # Prepare inputs.
        requests = [
            _buyer(1),
            _buyer(2, latency_ms=3000.0),
            _buyer(3, status=rnogapro.TIMEOUT),
            _buyer(4, status=rnogapro.BUYER_ERROR),
            _buyer(5, status=rnogapro.GATEWAY_ERROR),
            _buyer(6, attribution_mismatch=True),
            _canary(7),
            _canary(8, correct=False),
            _canary(9, status=rnogapro.UPSTREAM_ERROR, correct=None),
            _canary(10, correct=None),
        ]
        evidence = _evidence(requests)
        # Run test.
        actual = rnochwin.count_window_evidence(evidence)
        # Check outputs.
        expected = rnochmod.WindowEvidenceCounts(3, 3)
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test canary latency never satisfies the buyer collection target.
        """
        # Prepare inputs.
        evidence = _evidence([_canary(1), _canary(2)])
        # Run test.
        actual = rnochwin.count_window_evidence(evidence)
        # Check outputs.
        self.assertEqual(actual, rnochmod.WindowEvidenceCounts(0, 2))


# #############################################################################
# Test_assess_window_readiness
# #############################################################################


class Test_assess_window_readiness(hunitest.TestCase):
    """
    Test collection targets, deadlines, and invalid runtime inputs.
    """

    def test1(self) -> None:
        """
        Test reaching only one collection target keeps the window open.
        """
        # Prepare inputs.
        contract = _contract()
        evidence = _evidence([_buyer(1), _buyer(2), _canary(3)])
        now = _START + datetime.timedelta(seconds=30)
        # Run test.
        actual = rnochwin.assess_window_readiness(contract, evidence, now)
        # Check outputs.
        self.assertFalse(actual.ready)
        self.assertEqual(actual.reason, rnochmod.WindowCloseReason.NOT_READY)
        self.assertEqual(actual.counts, rnochmod.WindowEvidenceCounts(2, 1))

    def test2(self) -> None:
        """
        Test both collection targets close a window before its deadline.
        """
        # Prepare inputs.
        contract = _contract()
        evidence = _evidence([_buyer(1), _buyer(2), _canary(3), _canary(4)])
        now = _START + datetime.timedelta(seconds=30)
        # Run test.
        actual = rnochwin.assess_window_readiness(contract, evidence, now)
        # Check outputs.
        self.assertTrue(actual.ready)
        self.assertEqual(
            actual.reason, rnochmod.WindowCloseReason.TARGETS_REACHED
        )

    def test3(self) -> None:
        """
        Test the exact deadline and later timestamps close a sparse window.
        """
        # Prepare inputs.
        contract = _contract()
        evidence = _evidence([])
        deadline = _START + datetime.timedelta(seconds=60)
        # Run test and check outputs.
        for now in [deadline, deadline + datetime.timedelta(seconds=1)]:
            actual = rnochwin.assess_window_readiness(contract, evidence, now)
            self.assertTrue(actual.ready)
            self.assertEqual(
                actual.reason, rnochmod.WindowCloseReason.DEADLINE_REACHED
            )
            self.assertEqual(actual.deadline, deadline)

    def test4(self) -> None:
        """
        Test one microsecond before the deadline remains open.
        """
        # Prepare inputs.
        contract = _contract()
        evidence = _evidence([])
        now = _START + datetime.timedelta(seconds=60, microseconds=-1)
        # Run test.
        actual = rnochwin.assess_window_readiness(contract, evidence, now)
        # Check outputs.
        self.assertFalse(actual.ready)
        self.assertEqual(actual.reason, rnochmod.WindowCloseReason.NOT_READY)

    def test5(self) -> None:
        """
        Test completed targets take precedence when the deadline also elapsed.
        """
        # Prepare inputs.
        contract = _contract()
        evidence = _evidence([_buyer(1), _buyer(2), _canary(3), _canary(4)])
        now = _START + datetime.timedelta(seconds=61)
        # Run test.
        actual = rnochwin.assess_window_readiness(contract, evidence, now)
        # Check outputs.
        self.assertEqual(
            actual.reason, rnochmod.WindowCloseReason.TARGETS_REACHED
        )

    def test6(self) -> None:
        """
        Test invalid contract and clock inputs are rejected.
        """
        # Prepare inputs.
        valid_contract = _contract()
        naive = datetime.datetime(2026, 1, 1)
        cases = [
            (
                dataclasses.replace(valid_contract, state="PASSED"),
                _START,
                "ACTIVE",
            ),
            (
                dataclasses.replace(valid_contract, window_start=None),
                _START,
                "window_start is required",
            ),
            (
                dataclasses.replace(valid_contract, window_start=naive),
                _START,
                "window_start must be timezone-aware",
            ),
            (valid_contract, naive, "now must be timezone-aware"),
            (
                dataclasses.replace(valid_contract, target_requests=0),
                _START,
                "target_requests must be positive",
            ),
            (
                dataclasses.replace(valid_contract, target_canaries=0),
                _START,
                "target_canaries must be positive",
            ),
            (
                dataclasses.replace(valid_contract, maximum_seconds=0),
                _START,
                "maximum_seconds must be positive",
            ),
        ]
        # Run test and check outputs.
        for contract, now, message in cases:
            with self.assertRaisesRegex(rnochwin.WindowReadinessError, message):
                rnochwin.assess_window_readiness(contract, _evidence([]), now)
