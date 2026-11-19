"""
Test one bounded standalone monitoring pass.
"""

import asyncio
from unittest import mock

import helpers.hunit_test as hunitest

import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.common.test.noesis_test_utils as rnctnteut
import research.Noesis.gateway.gateway_fake_provider as rnogafapr
import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_models as rnmonmod
import research.Noesis.monitor.monitor_runtime as rnmonrun


def _canary(number: int, answer: str = "2") -> rnochmod.Canary:
    return rnochmod.Canary(
        "math-v1-%03d" % number,
        "v1",
        "math",
        "What is one plus one? Return only the integer.",
        answer,
        "integer",
        number,
    )


def _target() -> rnmonmod.MonitoringTarget:
    return rnmonmod.MonitoringTarget("groq", "test-model")


def _provider(answer: str = "2") -> rnogafapr.FakeProvider:
    return rnogafapr.FakeProvider(
        {
            "groq": rnogafapr.FakeBehavior(
                "Groq", latency_s=0.2, answer=lambda messages: answer
            )
        }
    )


class _RaisingProvider:
    async def call(self, slug, request):
        _ = (slug, request)
        raise RuntimeError("provider bug")


class _HangingProvider:
    async def call(self, slug, request):
        _ = (slug, request)
        await asyncio.sleep(1.0)


class _RaisingMeasurer:
    @property
    def version(self) -> str:
        return "raising-v1"

    def measure(self, canary, response):
        _ = (canary, response)
        raise RuntimeError("grader bug")


class Test_execute_monitoring(hunitest.TestCase):
    """
    Test sequential provider calls, grading, and per-canary isolation.
    """

    def test1(self) -> None:
        """
        Test successful calls are graded and reported immediately.
        """
        provider = _provider()
        policy = rnmonmod.MonitoringPolicy(canary_count=2)
        recorded = []

        async def _record(observation) -> None:
            recorded.append(observation)

        execution = asyncio.run(
            rnmonrun.execute_monitoring(
                provider,
                (_canary(1), _canary(2)),
                _target(),
                policy,
                on_observation=_record,
            )
        )
        self.assertEqual(tuple(recorded), execution.observations)
        self.assertEqual(execution.summary["correct_calls"], 2)
        self.assertEqual(execution.summary["accuracy"], 1.0)
        self.assertEqual(len(provider.calls), 2)
        first_call = provider.calls[0][1]
        self.assertEqual(first_call.temperature, 0.0)
        self.assertEqual(first_call.timeout_s, 30.0)
        self.assertEqual(
            first_call.messages,
            [{"role": "user", "content": _canary(1).prompt}],
        )

    def test2(self) -> None:
        """
        Test provider timeouts are ungradable observations, not run failures.
        """
        provider = _provider()
        provider.behaviors["groq"].latency_s = 31.0
        policy = rnmonmod.MonitoringPolicy(canary_count=1)
        execution = asyncio.run(
            rnmonrun.execute_monitoring(
                provider, (_canary(1),), _target(), policy
            )
        )
        observation = execution.observations[0]
        self.assertEqual(observation.provider_status, rnogapro.TIMEOUT)
        self.assertIsNone(observation.correct)
        self.assertEqual(observation.grader_reason, "PROVIDER_STATUS_TIMEOUT")
        self.assertIn("runtime timeout", observation.error)
        self.assertEqual(execution.summary["ungradable_calls"], 1)

    def test3(self) -> None:
        """
        Test unexpected provider and grader exceptions stay per-canary.
        """
        policy = rnmonmod.MonitoringPolicy(canary_count=1)
        provider_failure = asyncio.run(
            rnmonrun.execute_monitoring(
                _RaisingProvider(), (_canary(1),), _target(), policy
            )
        ).observations[0]
        self.assertEqual(provider_failure.provider_status, "internal_error")
        self.assertEqual(
            provider_failure.grader_reason, rnmonrun.PROVIDER_EXCEPTION
        )
        grader_failure = asyncio.run(
            rnmonrun.execute_monitoring(
                _provider(),
                (_canary(1),),
                _target(),
                policy,
                measurer=_RaisingMeasurer(),
            )
        ).observations[0]
        self.assertEqual(grader_failure.provider_status, rnogapro.OK)
        self.assertIsNone(grader_failure.correct)
        self.assertEqual(grader_failure.grader_reason, rnmonrun.GRADER_EXCEPTION)

    def test4(self) -> None:
        """
        Test the runtime enforces its timeout around any provider adapter.
        """
        policy = rnmonmod.MonitoringPolicy(canary_count=1, timeout_s=0.01)
        execution = asyncio.run(
            rnmonrun.execute_monitoring(
                _HangingProvider(), (_canary(1),), _target(), policy
            )
        )
        observation = execution.observations[0]
        self.assertEqual(observation.provider_status, rnogapro.TIMEOUT)
        self.assertIsNone(observation.correct)
        self.assertEqual(observation.grader_reason, "PROVIDER_STATUS_TIMEOUT")


class Test_run_monitoring(hunitest.TestCase):
    """
    Test persisted orchestration and fatal store failure handling.
    """

    def test1(self) -> None:
        """
        Test a persistence failure marks the run FAILED.
        """
        policy = rnmonmod.MonitoringPolicy(canary_count=1)
        with (
            mock.patch.object(
                rnmonrun.rnochsto,
                "load_active_canaries",
                new=mock.AsyncMock(return_value=(_canary(1),)),
            ),
            mock.patch.object(
                rnmonrun.rnmonsto,
                "start_monitoring_run",
                new=mock.AsyncMock(return_value=17),
            ),
            mock.patch.object(
                rnmonrun.rnmonsto,
                "record_monitoring_observation",
                new=mock.AsyncMock(side_effect=RuntimeError("database down")),
            ),
            mock.patch.object(
                rnmonrun.rnmonsto,
                "fail_monitoring_run",
                new=mock.AsyncMock(return_value=True),
            ) as fail,
            self.assertRaisesRegex(RuntimeError, "database down"),
        ):
            asyncio.run(
                rnmonrun.run_monitoring(
                    mock.sentinel.pool,
                    _provider(),
                    _target(),
                    policy,
                    "selection-secret",
                )
            )
        fail.assert_awaited_once_with(mock.sentinel.pool, 17, "database down")

    def test2(self) -> None:
        """
        Test a fake-provider run persists its grade and provider measurements.
        """

        async def _body(pool) -> None:
            async with pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO canaries ("
                    " canary_id, prompt, answer, normalizer, bank_version,"
                    " family, generator_seed, status"
                    ") VALUES (%s, %s, %s, %s, %s, %s, %s, 'ACTIVE')",
                    (
                        _canary(1).canary_id,
                        _canary(1).prompt,
                        _canary(1).answer,
                        _canary(1).normalizer,
                        _canary(1).bank_version,
                        _canary(1).family,
                        _canary(1).generator_seed,
                    ),
                )
            policy = rnmonmod.MonitoringPolicy(canary_count=1)
            run_id, execution = await rnmonrun.run_monitoring(
                pool,
                _provider(),
                _target(),
                policy,
                "selection-secret",
            )
            self.assertEqual(execution.summary["correct_calls"], 1)
            rows = await rnctnteut.fetch_all(
                pool,
                "SELECT r.state, o.correct, o.latency_ms, o.prompt_tokens,"
                " o.completion_tokens, o.cost_usd FROM monitoring_runs r"
                " JOIN monitoring_observations o USING (run_id)"
                " WHERE r.run_id = %s",
                run_id,
            )
            self.assertEqual(rows[0], ("COMPLETED", True, 200.0, 10, 5, 0.00001))

        rnctnteut.run_with_clean_db(self, _body)
