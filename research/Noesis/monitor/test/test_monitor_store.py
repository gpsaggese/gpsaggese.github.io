"""
Test standalone-monitoring PostgreSQL persistence.
"""

import helpers.hunit_test as hunitest

import research.Noesis.common.test.noesis_test_utils as rnctnteut
import research.Noesis.monitor.monitor_models as rnmonmod
import research.Noesis.monitor.monitor_store as rnmonsto


async def _add_active_canary(pool) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO canaries (canary_id, prompt, answer, normalizer,"
            " bank_version, family, status) VALUES"
            " ('math-v1-001', 'One plus one?', '2', 'integer',"
            " 'v1', 'math', 'ACTIVE')"
        )


def _observation() -> rnmonmod.MonitoringObservation:
    return rnmonmod.MonitoringObservation(
        canary_id="math-v1-001",
        target=rnmonmod.MonitoringTarget("groq", "test-model"),
        provider_status="ok",
        correct=True,
        grader_reason="MATCH",
        grader_version="m2-v1",
        latency_ms=200.0,
        cost_usd=0.00001,
        response="2",
        served_by="Groq",
        model_served="test-model",
        prompt_tokens=10,
        completion_tokens=1,
        http_status=200,
    )


class Test_monitor_store(hunitest.TestCase):
    """
    Test the RUNNING-to-COMPLETED lifecycle and stored evidence.
    """

    def test1(self) -> None:
        """
        Test one observation is durable and completes its run.
        """

        async def _body(pool) -> None:
            await _add_active_canary(pool)
            target = rnmonmod.MonitoringTarget("groq", "test-model")
            policy = rnmonmod.MonitoringPolicy(canary_count=1)
            run_id = await rnmonsto.start_monitoring_run(
                pool, target, policy, "m2-v1"
            )
            await rnmonsto.record_monitoring_observation(
                pool, run_id, _observation()
            )
            completed = await rnmonsto.complete_monitoring_run(
                pool, run_id, {"total_calls": 1, "correct_calls": 1}
            )
            self.assertTrue(completed)
            self.assertFalse(
                await rnmonsto.complete_monitoring_run(pool, run_id, {})
            )
            rows = await rnctnteut.fetch_all(
                pool,
                "SELECT state, provider_slug, model_id,"
                " summary->>'correct_calls' FROM monitoring_runs"
                " WHERE run_id = %s",
                run_id,
            )
            self.assertEqual(rows[0], ("COMPLETED", "groq", "test-model", "1"))
            rows = await rnctnteut.fetch_all(
                pool,
                "SELECT correct, grader_reason, latency_ms, prompt_tokens,"
                " completion_tokens, cost_usd, served_by, model_served"
                " FROM monitoring_observations WHERE run_id = %s",
                run_id,
            )
            self.assertEqual(
                rows[0],
                (
                    True,
                    "MATCH",
                    200.0,
                    10,
                    1,
                    0.00001,
                    "Groq",
                    "test-model",
                ),
            )

        rnctnteut.run_with_clean_db(self, _body)

    def test2(self) -> None:
        """
        Test failure preserves an observation already committed.
        """

        async def _body(pool) -> None:
            await _add_active_canary(pool)
            target = rnmonmod.MonitoringTarget("groq", "test-model")
            policy = rnmonmod.MonitoringPolicy(canary_count=1)
            run_id = await rnmonsto.start_monitoring_run(
                pool, target, policy, "m2-v1"
            )
            await rnmonsto.record_monitoring_observation(
                pool, run_id, _observation()
            )
            self.assertTrue(
                await rnmonsto.fail_monitoring_run(pool, run_id, "stopped")
            )
            rows = await rnctnteut.fetch_all(
                pool,
                "SELECT state, summary->>'error' FROM monitoring_runs"
                " WHERE run_id = %s",
                run_id,
            )
            self.assertEqual(rows[0], ("FAILED", "stopped"))
            rows = await rnctnteut.fetch_all(
                pool,
                "SELECT count(*) FROM monitoring_observations WHERE run_id = %s",
                run_id,
            )
            self.assertEqual(rows[0][0], 1)

        rnctnteut.run_with_clean_db(self, _body)
