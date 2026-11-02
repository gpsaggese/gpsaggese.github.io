"""
Test contract-window evaluation, finalization, and verdict publication.
"""

import asyncio
import datetime
from typing import Optional
from unittest import mock

import helpers.hunit_test as hunitest
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_runtime as rnochrun
import research.Noesis.checker.checker_sink as rnochsink
import research.Noesis.checker.checker_store as rnochsto
import research.Noesis.common.test.noesis_test_utils as rnctnteut
import research.Noesis.gateway.gateway_providers as rnogapro

_START = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
_POLICY = rnochmod.WindowPolicy(
    minimum_requests=1,
    minimum_canaries=1,
)


def _contract(
    contract_id: int = 42,
    *,
    seller_id: str = "seller_1",
    target_requests: int = 1,
    target_canaries: int = 1,
    maximum_seconds: int = 60,
) -> rnochmod.ContractWindow:
    return rnochmod.ContractWindow(
        terms=rnochmod.ContractTerms(contract_id, 2.0, 0.9, 0.85),
        state="ACTIVE",
        target_requests=target_requests,
        target_canaries=target_canaries,
        maximum_seconds=maximum_seconds,
        checker_policy="m2-v1",
        window_start=_START,
        window_end=None,
        seller_id=seller_id,
    )


def _buyer(request_id: int) -> rnochmod.RequestObservation:
    return rnochmod.RequestObservation(request_id, rnogapro.OK, 100.0)


def _canary(
    request_id: int,
    *,
    correct: Optional[bool] = True,
) -> rnochmod.RequestObservation:
    return rnochmod.RequestObservation(
        request_id,
        rnogapro.OK,
        100.0,
        is_canary=True,
        canary_correct=correct,
    )


def _readiness() -> rnochmod.WindowReadiness:
    return rnochmod.WindowReadiness(
        True,
        rnochmod.WindowCloseReason.TARGETS_REACHED,
        rnochmod.WindowEvidenceCounts(1, 1),
        _START + datetime.timedelta(seconds=60),
    )


def _assessment() -> rnochmod.MetricAssessment:
    return rnochmod.MetricAssessment(1, 1, 1.0, 1.0, 0.9, True, False)


def _report(
    contract_id: int,
    *,
    seller_id: str,
    verdict: rnochmod.Verdict = rnochmod.Verdict.PASSED,
) -> rnochmod.VerdictReport:
    return rnochmod.VerdictReport(
        contract_id=contract_id,
        seller_id=seller_id,
        verdict=verdict,
        disposition=rnochmod.CheckerDisposition.VALID,
        latency=_assessment(),
        quality=_assessment(),
        reason_codes=("COMPLIANT",),
        excluded_counts={},
        window_start=_START,
        window_end=_START + datetime.timedelta(seconds=30),
        checker_version="m2-v1",
        bank_version="v1",
    )


# #############################################################################
# Test_evaluate_contract
# #############################################################################


class Test_evaluate_contract(hunitest.TestCase):
    """
    Test readiness, finalization ownership, and report construction.
    """

    def test1(self) -> None:
        """
        Test a non-ready window remains unchanged.
        """

        async def _body() -> rnochmod.ContractEvaluation:
            finalize = mock.AsyncMock()
            with (
                mock.patch.object(
                    rnochsto,
                    "list_request_observations",
                    new=mock.AsyncMock(return_value=(_buyer(1),)),
                ),
                mock.patch.object(
                    rnochsto,
                    "finalize_contract",
                    new=finalize,
                ),
            ):
                result = await rnochrun.evaluate_contract(
                    mock.sentinel.pool,
                    _contract(),
                    _POLICY,
                    _START + datetime.timedelta(seconds=30),
                    checker_version="m2-v1",
                    bank_version="v1",
                )
            finalize.assert_not_awaited()
            return result

        actual = asyncio.run(_body())
        self.assertFalse(actual.readiness.ready)
        self.assertFalse(actual.finalized)
        self.assertIsNone(actual.report)

    def test2(self) -> None:
        """
        Test the finalization winner receives a complete actionable report.
        """
        now = _START + datetime.timedelta(seconds=30)
        finalize = mock.AsyncMock(return_value=True)

        async def _body() -> rnochmod.ContractEvaluation:
            with (
                mock.patch.object(
                    rnochsto,
                    "list_request_observations",
                    new=mock.AsyncMock(return_value=(_buyer(1), _canary(2))),
                ),
                mock.patch.object(
                    rnochsto,
                    "finalize_contract",
                    new=finalize,
                ),
            ):
                return await rnochrun.evaluate_contract(
                    mock.sentinel.pool,
                    _contract(),
                    _POLICY,
                    now,
                    checker_version="m2-v1",
                    bank_version="v1",
                )

        actual = asyncio.run(_body())
        self.assertTrue(actual.finalized)
        self.assertEqual(actual.report.verdict, rnochmod.Verdict.PASSED)
        self.assertEqual(actual.report.seller_id, "seller_1")
        self.assertEqual(actual.report.window_end, now)
        self.assertTrue(rnochsink.is_actionable_report(actual.report))
        self.assertEqual(finalize.await_args.kwargs["window_end"], now)

    def test3(self) -> None:
        """
        Test a concurrent finalization loser cannot produce a report.
        """

        async def _body() -> rnochmod.ContractEvaluation:
            with (
                mock.patch.object(
                    rnochsto,
                    "list_request_observations",
                    new=mock.AsyncMock(return_value=(_buyer(1), _canary(2))),
                ),
                mock.patch.object(
                    rnochsto,
                    "finalize_contract",
                    new=mock.AsyncMock(return_value=False),
                ),
            ):
                return await rnochrun.evaluate_contract(
                    mock.sentinel.pool,
                    _contract(),
                    _POLICY,
                    _START + datetime.timedelta(seconds=30),
                    checker_version="m2-v1",
                    bank_version="v1",
                )

        actual = asyncio.run(_body())
        self.assertTrue(actual.readiness.ready)
        self.assertFalse(actual.finalized)
        self.assertIsNone(actual.report)

    def test4(self) -> None:
        """
        Test the deadline persists insufficient evidence without actionability.
        """

        async def _body() -> rnochmod.ContractEvaluation:
            with (
                mock.patch.object(
                    rnochsto,
                    "list_request_observations",
                    new=mock.AsyncMock(return_value=()),
                ),
                mock.patch.object(
                    rnochsto,
                    "finalize_contract",
                    new=mock.AsyncMock(return_value=True),
                ),
            ):
                return await rnochrun.evaluate_contract(
                    mock.sentinel.pool,
                    _contract(),
                    _POLICY,
                    _START + datetime.timedelta(seconds=60),
                    checker_version="m2-v1",
                    bank_version="v1",
                )

        actual = asyncio.run(_body())
        self.assertTrue(actual.finalized)
        self.assertEqual(actual.report.verdict, rnochmod.Verdict.INSUFFICIENT)
        self.assertFalse(rnochsink.is_actionable_report(actual.report))

    def test5(self) -> None:
        """
        Test unsafe clock and version configurations fail before database I/O.
        """
        valid_contract = _contract()
        cases = [
            (
                valid_contract,
                datetime.datetime(2026, 1, 1),
                "m2-v1",
                "v1",
                "timezone-aware",
            ),
            (valid_contract, _START, "", "v1", "checker_version"),
            (valid_contract, _START, "m2-v1", "", "bank_version"),
            (valid_contract, _START, "m2-v2", "v1", "does not match"),
        ]
        for contract, now, checker_version, bank_version, message in cases:
            with self.assertRaisesRegex(rnochrun.CheckerRuntimeError, message):
                asyncio.run(
                    rnochrun.evaluate_contract(
                        mock.sentinel.pool,
                        contract,
                        _POLICY,
                        now,
                        checker_version=checker_version,
                        bank_version=bank_version,
                    )
                )


# #############################################################################
# Test_run_checker_pass
# #############################################################################


class Test_run_checker_pass(hunitest.TestCase):
    """
    Test publication policy and per-contract failure isolation.
    """

    def test1(self) -> None:
        """
        Test only actionable reports from finalization winners publish.
        """
        contracts = (_contract(1), _contract(2), _contract(3))
        actionable = _report(1, seller_id="seller_1")
        insufficient = _report(
            2,
            seller_id="seller_2",
            verdict=rnochmod.Verdict.INSUFFICIENT,
        )
        evaluations = (
            rnochmod.ContractEvaluation(1, _readiness(), True, actionable),
            rnochmod.ContractEvaluation(2, _readiness(), True, insufficient),
            rnochmod.ContractEvaluation(3, _readiness(), False, None),
        )
        sink = rnochsink.RecordingVerdictSink()

        async def _body() -> rnochrun.CheckerPassResult:
            with (
                mock.patch.object(
                    rnochsto,
                    "list_active_contracts",
                    new=mock.AsyncMock(return_value=contracts),
                ),
                mock.patch.object(
                    rnochrun,
                    "evaluate_contract",
                    new=mock.AsyncMock(side_effect=evaluations),
                ),
            ):
                return await rnochrun.run_checker_pass(
                    mock.sentinel.pool,
                    sink,
                    _POLICY,
                    _START,
                    checker_version="m2-v1",
                    bank_version="v1",
                )

        actual = asyncio.run(_body())
        self.assertEqual(actual.evaluations, evaluations)
        self.assertEqual(actual.published_reports, (actionable,))
        self.assertEqual(actual.errors, ())
        self.assertEqual(sink.reports, (actionable,))

    def test2(self) -> None:
        """
        Test evaluation and sink failures do not stop later contracts.
        """
        contracts = (_contract(1), _contract(2), _contract(3))
        second_report = _report(2, seller_id="seller_2")
        third_report = _report(3, seller_id="seller_3")
        second = rnochmod.ContractEvaluation(2, _readiness(), True, second_report)
        third = rnochmod.ContractEvaluation(3, _readiness(), True, third_report)
        sink = mock.Mock()
        sink.publish = mock.AsyncMock(
            side_effect=(RuntimeError("sink unavailable"), None)
        )

        async def _body() -> rnochrun.CheckerPassResult:
            with (
                mock.patch.object(
                    rnochsto,
                    "list_active_contracts",
                    new=mock.AsyncMock(return_value=contracts),
                ),
                mock.patch.object(
                    rnochrun,
                    "evaluate_contract",
                    new=mock.AsyncMock(
                        side_effect=(RuntimeError("bad evidence"), second, third)
                    ),
                ),
            ):
                return await rnochrun.run_checker_pass(
                    mock.sentinel.pool,
                    sink,
                    _POLICY,
                    _START,
                    checker_version="m2-v1",
                    bank_version="v1",
                )

        actual = asyncio.run(_body())
        self.assertEqual(actual.evaluations, (second, third))
        self.assertEqual(actual.published_reports, (third_report,))
        self.assertEqual(
            actual.errors,
            (
                "contract 1 evaluation: bad evidence",
                "contract 2 sink: sink unavailable",
            ),
        )
        self.assertEqual(sink.publish.await_count, 2)


# #############################################################################
# Test_run_checker_pass_postgres
# #############################################################################


class Test_run_checker_pass_postgres(hunitest.TestCase):
    """
    Test the complete active-contract finalization path against PostgreSQL.
    """

    def test1(self) -> None:
        """
        Test one ready contract persists and publishes exactly once.
        """

        async def _body(pool) -> None:
            await rnctnteut.add_account(pool, "buyer_1", "buyer", "buyer-key")
            await rnctnteut.add_seller(pool, "seller_1", "groq", "Groq")
            contract_id = await rnctnteut.add_contract(
                pool,
                "buyer_1",
                "seller_1",
            )
            now = datetime.datetime.now(datetime.timezone.utc)
            async with pool.connection() as conn:
                await conn.execute(
                    "UPDATE contracts SET n_tasks = 1, canaries_target = 1,"
                    " window_start = %s WHERE contract_id = %s",
                    (now - datetime.timedelta(seconds=30), contract_id),
                )
                await conn.execute(
                    "INSERT INTO canaries (canary_id, prompt, answer,"
                    " normalizer, bank_version, family, status) VALUES"
                    " ('math-v1-001', 'What is 1+1?', '2', 'integer',"
                    " 'v1', 'math', 'ACTIVE')"
                )
                await conn.execute(
                    "INSERT INTO requests (contract_id, seller_id,"
                    " pinned_provider, model_requested, is_canary, canary_id,"
                    " canary_correct, prompt, latency_ms, status) VALUES"
                    " (%s, 'seller_1', 'Groq', 'test-model', FALSE, NULL,"
                    " NULL, '[]', 100, 'ok'),"
                    " (%s, 'seller_1', 'Groq', 'test-model', TRUE,"
                    " 'math-v1-001', TRUE, '[]', 100, 'ok')",
                    (contract_id, contract_id),
                )
            sink = rnochsink.RecordingVerdictSink()
            first = await rnochrun.run_checker_pass(
                pool,
                sink,
                _POLICY,
                now,
                checker_version="m2-v1",
                bank_version="v1",
            )
            second = await rnochrun.run_checker_pass(
                pool,
                sink,
                _POLICY,
                now,
                checker_version="m2-v1",
                bank_version="v1",
            )
            self.assertEqual(len(first.published_reports), 1)
            self.assertEqual(first.published_reports[0].seller_id, "seller_1")
            self.assertEqual(second.evaluations, ())
            self.assertEqual(len(sink.reports), 1)
            rows = await rnctnteut.fetch_all(
                pool,
                "SELECT state FROM contracts WHERE contract_id = %s",
                contract_id,
            )
            self.assertEqual(rows[0][0], "PASSED")
            rows = await rnctnteut.fetch_all(
                pool,
                "SELECT count(*) FROM contract_verdicts"
                " WHERE contract_id = %s",
                contract_id,
            )
            self.assertEqual(rows[0][0], 1)

        rnctnteut.run_with_clean_db(self, _body)
