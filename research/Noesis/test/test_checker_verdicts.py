"""
Test evidence attribution and explainable verdict decisions.
"""

from typing import List, Optional

import helpers.hunit_test as hunitest

import research.Noesis.checker_evidence as rnochevi
import research.Noesis.checker_models as rnochmod
import research.Noesis.checker_verdicts as rnochver
import research.Noesis.gateway_providers as rnogapro

_CONTRACT = rnochmod.ContractTerms(1, 2.0, 0.9, 0.85)
_POLICY = rnochmod.WindowPolicy()


def _buyer(request_id: int, on_time: bool = True) -> rnochmod.RequestObservation:
    return rnochmod.RequestObservation(
        request_id,
        rnogapro.OK,
        1000.0 if on_time else 3000.0,
    )


def _canary(
    request_id: int, correct: Optional[bool] = True
) -> rnochmod.RequestObservation:
    return rnochmod.RequestObservation(
        request_id,
        rnogapro.OK,
        500.0,
        is_canary=True,
        canary_correct=correct,
    )


def _decision(
    requests: List[rnochmod.RequestObservation],
) -> rnochmod.VerdictDecision:
    evidence = rnochevi.build_evidence(requests, _CONTRACT, _POLICY)
    return rnochver.decide_verdict(evidence, _CONTRACT, _POLICY)


# #############################################################################
# Test_classify_latency
# #############################################################################


class Test_classify_latency(hunitest.TestCase):
    """
    Test seller attribution and latency policy.
    """

    def test1(self) -> None:
        """
        Test the latency boundary is inclusive.
        """
        for latency, success in [(1999.0, True), (2000.0, True), (2000.1, False)]:
            request = rnochmod.RequestObservation(1, rnogapro.OK, latency)
            result = rnochevi.classify_latency(request, _CONTRACT, _POLICY)
            self.assertTrue(result.eligible)
            self.assertEqual(result.success, success)

    def test2(self) -> None:
        """
        Test seller failures count and Noesis/buyer failures do not.
        """
        seller_statuses = [rnogapro.TIMEOUT, rnogapro.UPSTREAM_ERROR]
        for status in seller_statuses:
            request = rnochmod.RequestObservation(1, status, 100.0)
            result = rnochevi.classify_latency(request, _CONTRACT, _POLICY)
            self.assertEqual((result.eligible, result.success), (True, False))
        excluded_statuses = [
            rnogapro.BUYER_ERROR,
            rnogapro.GATEWAY_ERROR,
            "refused_cap",
        ]
        for status in excluded_statuses:
            request = rnochmod.RequestObservation(1, status, 100.0)
            result = rnochevi.classify_latency(request, _CONTRACT, _POLICY)
            self.assertFalse(result.eligible)
            self.assertFalse(result.checker_error)

    def test3(self) -> None:
        """
        Test canary latency is excluded unless the policy opts in.
        """
        request = _canary(1)
        excluded = rnochevi.classify_latency(request, _CONTRACT, _POLICY)
        included = rnochevi.classify_latency(
            request,
            _CONTRACT,
            rnochmod.WindowPolicy(include_canaries_in_latency=True),
        )
        self.assertFalse(excluded.eligible)
        self.assertEqual((included.eligible, included.success), (True, True))


# #############################################################################
# Test_classify_quality
# #############################################################################


class Test_classify_quality(hunitest.TestCase):
    """
    Test deterministic quality evidence and checker-health errors.
    """

    def test1(self) -> None:
        """
        Test correct and incorrect canary grades.
        """
        for correct in [True, False]:
            result = rnochevi.classify_quality(_canary(1, correct))
            self.assertEqual((result.eligible, result.success), (True, correct))

    def test2(self) -> None:
        """
        Test a missing grade is a checker error, not a seller failure.
        """
        evidence = rnochevi.build_evidence([_canary(1, None)], _CONTRACT, _POLICY)
        self.assertFalse(evidence.checker_healthy)
        self.assertEqual(
            evidence.requests[0].quality.reason,
            rnochevi.MISSING_CANARY_GRADE,
        )


# #############################################################################
# Test_decide_verdict
# #############################################################################


class Test_decide_verdict(hunitest.TestCase):
    """
    Test evidence minima, failure precedence, and checker protection.
    """

    def test1(self) -> None:
        """
        Test a sufficiently sampled healthy window passes.
        """
        requests = [_buyer(i) for i in range(1, 21)]
        requests += [_canary(i) for i in range(101, 109)]
        decision = _decision(requests)
        self.assertEqual(decision.verdict, rnochmod.Verdict.PASSED)
        self.assertEqual(decision.reason_codes, (rnochver.COMPLIANT,))

    def test2(self) -> None:
        """
        Test either sufficiently sampled metric can fail the contract.
        """
        cases = [
            [_buyer(i, False) for i in range(1, 21)]
            + [_canary(i) for i in range(101, 109)],
            [_buyer(i) for i in range(1, 21)]
            + [_canary(i, False) for i in range(101, 109)],
        ]
        reasons = [
            rnochver.LATENCY_BELOW_PROMISE,
            rnochver.QUALITY_BELOW_PROMISE,
        ]
        for requests, reason in zip(cases, reasons):
            decision = _decision(requests)
            self.assertEqual(decision.verdict, rnochmod.Verdict.FAILED)
            self.assertEqual(decision.reason_codes, (reason,))

    def test3(self) -> None:
        """
        Test each metric must meet its own minimum before it can fail.
        """
        requests = [_buyer(1, False), _buyer(2, False)]
        requests += [_canary(i) for i in range(101, 109)]
        decision = _decision(requests)
        self.assertEqual(decision.verdict, rnochmod.Verdict.INSUFFICIENT)
        self.assertFalse(decision.latency.conclusive_failure)

    def test4(self) -> None:
        """
        Test a conclusive failure overrides the other metric's shortage.
        """
        requests = [_buyer(i, False) for i in range(1, 21)]
        requests += [_canary(101), _canary(102)]
        decision = _decision(requests)
        self.assertEqual(decision.verdict, rnochmod.Verdict.FAILED)
        self.assertEqual(decision.reason_codes, (rnochver.LATENCY_BELOW_PROMISE,))

    def test5(self) -> None:
        """
        Test checker errors protect the seller.
        """
        requests = [_buyer(i, False) for i in range(1, 21)]
        requests += [_canary(i) for i in range(101, 108)]
        requests.append(_canary(108, None))
        decision = _decision(requests)
        self.assertEqual(decision.verdict, rnochmod.Verdict.INSUFFICIENT)
        self.assertEqual(
            decision.disposition,
            rnochmod.CheckerDisposition.CHECKER_UNHEALTHY,
        )
