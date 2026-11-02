"""
Test scheduled probe execution through checker and gateway boundaries.
"""

import asyncio
import dataclasses
import datetime
from unittest import mock

import helpers.hunit_test as hunitest
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_store as rnochsto
import research.Noesis.common.test.noesis_test_utils as rnctnteut
import research.Noesis.gateway.gateway_fake_provider as rnogafapr
import research.Noesis.gateway.gateway_faults as rnogafau
import research.Noesis.gateway.gateway_request_log as rnogarelo
import research.Noesis.gateway.gateway_routing as rnogarou
import research.Noesis.prober.prober_runtime as rnoprrun


def _canary() -> rnochmod.Canary:
    return rnochmod.Canary(
        "math-v1-001",
        "v1",
        "math",
        "What is one plus one? Return only the integer.",
        "2",
        "integer",
        1,
    )


def _route(contract_id: int = 42) -> rnogarou.Route:
    return rnogarou.Route(
        buyer_id="buyer_1",
        contract_id=contract_id,
        seller_id="seller_1",
        provider_slug="groq",
        provider_name="Groq",
        l_max=2.0,
    )


def _policy() -> rnochmod.ProbePolicy:
    return rnochmod.ProbePolicy("v1", "healthy-model")


def _contract(
    now: datetime.datetime,
    target_canaries: int,
) -> rnochmod.ContractWindow:
    return rnochmod.ContractWindow(
        rnochmod.ContractTerms(42, 2.0, 0.9, 0.85),
        "ACTIVE",
        20,
        target_canaries,
        60,
        "m2-v1",
        now - datetime.timedelta(seconds=61),
        None,
        "seller_1",
    )


# #############################################################################
# Test_execute_probe
# #############################################################################


class Test_execute_probe(hunitest.TestCase):
    """
    Test one claimed probe through routing, faults, grading, and logging.
    """

    def test1(self) -> None:
        """
        Test a correct answer is server-marked and completed.
        """
        provider = rnogafapr.FakeProvider(
            {"groq": rnogafapr.FakeBehavior("Groq", answer=lambda messages: "2")}
        )
        faults = rnogafau.FaultRegistry()

        async def _body() -> rnochmod.ProbeExecution:
            with (
                mock.patch.object(
                    rnochsto,
                    "mark_probe_running",
                    new=mock.AsyncMock(return_value=True),
                ),
                mock.patch.object(
                    rnogarou,
                    "resolve_internal_route",
                    new=mock.AsyncMock(return_value=_route()),
                ),
                mock.patch.object(
                    rnogarelo,
                    "log_request",
                    new=mock.AsyncMock(return_value=101),
                ) as log_request,
                mock.patch.object(
                    rnochsto,
                    "complete_probe",
                    new=mock.AsyncMock(return_value=True),
                ),
            ):
                execution = await rnoprrun.execute_probe(
                    mock.sentinel.pool,
                    provider,
                    faults,
                    7,
                    42,
                    _canary(),
                    _policy(),
                )
                kwargs = log_request.await_args.kwargs
                self.assertEqual(
                    (
                        kwargs["is_canary"],
                        kwargs["canary_id"],
                        kwargs["canary_correct"],
                        kwargs["grader_version"],
                    ),
                    (True, "math-v1-001", True, "m2-v1"),
                )
                return execution

        execution = asyncio.run(_body())
        self.assertEqual(execution.request_id, 101)
        self.assertTrue(execution.correct)
        call = provider.calls[0][1]
        self.assertEqual(
            call.messages,
            [{"role": "user", "content": _canary().prompt}],
        )
        self.assertNotIn("canary", str(call.messages).lower())

    def test2(self) -> None:
        """
        Test a model-swap fault is applied, graded, and loudly logged.
        """
        provider = rnogafapr.FakeProvider(
            {"groq": rnogafapr.FakeBehavior("Groq", answer=lambda messages: "3")}
        )
        faults = rnogafau.FaultRegistry()
        faults.set(
            "seller_1",
            rnogafau.Fault(model_override="weak-model"),
        )

        async def _body() -> rnochmod.ProbeExecution:
            with (
                mock.patch.object(
                    rnochsto,
                    "mark_probe_running",
                    new=mock.AsyncMock(return_value=True),
                ),
                mock.patch.object(
                    rnogarou,
                    "resolve_internal_route",
                    new=mock.AsyncMock(return_value=_route()),
                ),
                mock.patch.object(
                    rnogarelo,
                    "log_request",
                    new=mock.AsyncMock(return_value=102),
                ) as log_request,
                mock.patch.object(
                    rnochsto,
                    "complete_probe",
                    new=mock.AsyncMock(return_value=True),
                ),
            ):
                execution = await rnoprrun.execute_probe(
                    mock.sentinel.pool,
                    provider,
                    faults,
                    8,
                    42,
                    _canary(),
                    _policy(),
                )
                self.assertTrue(log_request.await_args.kwargs["fault_injected"])
                return execution

        execution = asyncio.run(_body())
        self.assertFalse(execution.correct)
        self.assertEqual(provider.calls[0][1].model, "weak-model")


# #############################################################################
# Test_run_due_probes
# #############################################################################


class Test_run_due_probes(hunitest.TestCase):
    """
    Test one complete fake-provider cycle against PostgreSQL.
    """

    def test1(self) -> None:
        """
        Test one internal failure does not stop a later due probe.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        second_canary = dataclasses.replace(_canary(), canary_id="math-v1-002")
        successful = rnochmod.ProbeExecution(
            probe_id=8,
            request_id=102,
            provider_status="ok",
            correct=True,
            grader_reason="MATCH",
        )

        async def _body() -> rnoprrun.ProbeCycleResult:
            with (
                mock.patch.object(
                    rnochsto,
                    "list_active_contracts",
                    new=mock.AsyncMock(return_value=(_contract(now, 2),)),
                ),
                mock.patch.object(
                    rnochsto,
                    "load_active_canaries",
                    new=mock.AsyncMock(return_value=(_canary(), second_canary)),
                ),
                mock.patch.object(
                    rnochsto,
                    "list_claimed_probe_slots",
                    new=mock.AsyncMock(return_value=()),
                ),
                mock.patch.object(
                    rnochsto,
                    "claim_probe_slot",
                    new=mock.AsyncMock(side_effect=(7, 8)),
                ),
                mock.patch.object(
                    rnoprrun,
                    "execute_probe",
                    new=mock.AsyncMock(
                        side_effect=(RuntimeError("boom"), successful)
                    ),
                ),
            ):
                return await rnoprrun.run_due_probes(
                    mock.sentinel.pool,
                    mock.sentinel.provider,
                    rnogafau.FaultRegistry(),
                    _policy(),
                    "schedule-secret",
                    now,
                )

        cycle = asyncio.run(_body())
        self.assertEqual(cycle.executions, (successful,))
        self.assertEqual(cycle.errors, ("probe 7: boom",))

    def test2(self) -> None:
        """
        Test a due slot becomes one completed, graded request.
        """

        async def _body(pool) -> None:
            await rnctnteut.add_account(pool, "buyer_1", "buyer", "buyer-key")
            await rnctnteut.add_seller(pool, "seller_1", "groq", "Groq")
            contract_id = await rnctnteut.add_contract(
                pool, "buyer_1", "seller_1"
            )
            now = datetime.datetime.now(datetime.timezone.utc)
            async with pool.connection() as conn:
                await conn.execute(
                    "UPDATE contracts SET window_start = %s,"
                    " canaries_target = 1, window_max_seconds = 60"
                    " WHERE contract_id = %s",
                    (now - datetime.timedelta(seconds=61), contract_id),
                )
                await conn.execute(
                    "INSERT INTO canaries ("
                    " canary_id, prompt, answer, normalizer, bank_version,"
                    " family, generator_seed, calibrated, status"
                    ") VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, 'ACTIVE')",
                    (
                        _canary().canary_id,
                        _canary().prompt,
                        _canary().answer,
                        _canary().normalizer,
                        _canary().bank_version,
                        _canary().family,
                        _canary().generator_seed,
                    ),
                )
            provider = rnogafapr.FakeProvider(
                {
                    "groq": rnogafapr.FakeBehavior(
                        "Groq", answer=lambda messages: "2"
                    )
                }
            )
            cycle = await rnoprrun.run_due_probes(
                pool,
                provider,
                rnogafau.FaultRegistry(),
                _policy(),
                "schedule-secret",
                now,
            )
            self.assertEqual(len(cycle.executions), 1)
            self.assertEqual(cycle.errors, ())
            rows = await rnctnteut.fetch_all(
                pool,
                "SELECT p.state, r.is_canary, r.canary_id,"
                " r.canary_correct, r.grader_version"
                " FROM canary_probes p"
                " JOIN requests r ON r.request_id = p.request_id"
                " WHERE p.contract_id = %s",
                contract_id,
            )
            self.assertEqual(
                rows[0],
                ("COMPLETED", True, "math-v1-001", True, "m2-v1"),
            )

        rnctnteut.run_with_clean_db(self, _body)
