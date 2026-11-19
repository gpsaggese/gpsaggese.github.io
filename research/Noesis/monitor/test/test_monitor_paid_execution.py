"""Test the shared safety core for paid research campaigns."""

import unittest

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_paid_execution as rnmonpaex
import research.Noesis.monitor.monitor_paid_smoke as rnmonpasm


class _SequenceProvider:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def call(self, slug, request):
        self.calls.append((slug, request))
        return self.results.pop(0)


def _success_result(provider_name, model_id, cost_usd=0.0001):
    return rnogapro.ProviderResult(
        status=rnogapro.OK,
        text="OK",
        served_by=provider_name,
        model_served=model_id,
        prompt_tokens=12,
        completion_tokens=1,
        cost_usd=cost_usd,
        latency_s=0.1,
        http_status=200,
        error=None,
    )


class TestExecutePaidSmoke(unittest.IsolatedAsyncioTestCase):
    """Test completion and immediate safety halts."""

    async def test1(self):
        calls = rnmonpasm.build_default_smoke_calls()
        provider = _SequenceProvider(
            [
                _success_result(call.expected_provider_name, call.model_id)
                for call in calls
            ]
        )
        recorded = []

        async def _record(observation):
            recorded.append(observation)

        execution = await rnmonpasm.execute_paid_smoke(
            provider, calls, rnmonpasm.SmokePolicy(), _record
        )
        self.assertEqual(execution.state, "COMPLETED")
        self.assertEqual(len(recorded), 15)
        self.assertEqual(len(provider.calls), 15)

    async def test2(self):
        calls = rnmonpasm.build_default_smoke_calls()
        first = calls[0]
        provider = _SequenceProvider(
            [_success_result("Unexpected Provider", first.model_id)]
        )

        async def _record(observation):
            _ = observation

        execution = await rnmonpasm.execute_paid_smoke(
            provider, calls, rnmonpasm.SmokePolicy(), _record
        )
        self.assertEqual(execution.state, "HALTED")
        self.assertEqual(execution.halt_reason, "ATTRIBUTION_MISMATCH")
        self.assertEqual(len(provider.calls), 1)

    async def test3(self):
        calls = rnmonpasm.build_default_smoke_calls()
        first = calls[0]
        provider = _SequenceProvider(
            [_success_result(first.expected_provider_name, "wrong-model")]
        )

        async def _record(observation):
            _ = observation

        execution = await rnmonpasm.execute_paid_smoke(
            provider, calls, rnmonpasm.SmokePolicy(), _record
        )
        self.assertEqual(execution.halt_reason, "ATTRIBUTION_MISMATCH")
        self.assertEqual(len(provider.calls), 1)

    async def test4(self):
        calls = rnmonpasm.build_default_smoke_calls()
        first = calls[0]
        provider = _SequenceProvider(
            [_success_result(first.expected_provider_name, first.model_id)]
        )

        async def _record(_observation):
            raise OSError("simulated persistence failure")

        with self.assertRaisesRegex(OSError, "persistence failure"):
            await rnmonpasm.execute_paid_smoke(
                provider, calls, rnmonpasm.SmokePolicy(), _record
            )
        self.assertEqual(len(provider.calls), 1)


class TestPaidExecutionPolicy(unittest.TestCase):
    """Test that reserved worst-case spend fits inside the cap."""

    def test1(self):
        with self.assertRaises(ValueError):
            rnmonpaex.PaidExecutionPolicy(
                maximum_calls=240,
                maximum_cost_usd=10.0,
                per_call_reserve_usd=0.05,
            )

    def test2(self):
        call = rnmonpaex.PaidCallSpec(
            call_id="fingerprint-call",
            batch=1,
            route_name="route",
            provider_slug="provider",
            expected_provider_name="Provider",
            model_id="model",
            prompt="Prompt",
            max_tokens=10,
            timeout_s=5.0,
            metadata=(("sampling_temperature", "1.0"),),
        )
        self.assertEqual(call.sampling_temperature, 1.0)

    def test3(self):
        with self.assertRaisesRegex(ValueError, "between zero and two"):
            rnmonpaex.PaidCallSpec(
                call_id="fingerprint-call",
                batch=1,
                route_name="route",
                provider_slug="provider",
                expected_provider_name="Provider",
                model_id="model",
                prompt="Prompt",
                max_tokens=10,
                timeout_s=5.0,
                metadata=(("sampling_temperature", "2.1"),),
            )
