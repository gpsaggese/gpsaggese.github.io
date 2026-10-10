import asyncio
import inspect
import logging
import time

import helpers.hunit_test as hunitest
import research.Noesis.gateway_fake_provider as rnogafapr
import research.Noesis.gateway_openrouter as rnogaope
import research.Noesis.gateway_providers as rnogapro

_LOG = logging.getLogger(__name__)


def _request(
    *, timeout_s: float = 2.0, max_tokens: int = 16
) -> rnogapro.ProviderCall:
    """
    Return a minimal provider call.
    """
    request = rnogapro.ProviderCall(
        model="m",
        messages=[{"role": "user", "content": "What is 17 * 23?"}],
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    return request


def _run(
    fake: rnogafapr.FakeProvider, slug: str, request: rnogapro.ProviderCall
) -> rnogapro.ProviderResult:
    """
    Run one fake call synchronously.
    """
    return asyncio.run(fake.call(slug, request))


# #############################################################################
# TestFakeProvider
# #############################################################################


class TestFakeProvider(hunitest.TestCase):
    """
    Test the scripted provider used by the gateway tests.
    """

    def test1(self) -> None:
        """
        Test a scripted success and that the call is recorded.
        """
        # Prepare inputs.
        fake = rnogafapr.FakeProvider(
            {
                "groq": rnogafapr.FakeBehavior(
                    "Groq", latency_s=0.2, answer=lambda m: "391"
                )
            }
        )
        # Run test.
        actual = _run(fake, "groq", _request())
        # Check outputs.
        self.assertEqual(
            (actual.status, actual.text, actual.served_by, actual.latency_s),
            (rnogapro.OK, "391", "Groq", 0.2),
        )
        self.assertEqual(fake.calls[0][0], "groq")

    def test2(self) -> None:
        """
        Test that latency above the timeout becomes a capped TIMEOUT.
        """
        # Prepare inputs.
        fake = rnogafapr.FakeProvider(
            {"novita": rnogafapr.FakeBehavior("Novita", latency_s=9.0)}
        )
        # Run test.
        actual = _run(fake, "novita", _request(timeout_s=2.0))
        # Check outputs.
        self.assertEqual(
            (actual.status, actual.latency_s), (rnogapro.TIMEOUT, 2.0)
        )

    def test3(self) -> None:
        """
        Test that an unknown slug is a gateway (our) error.
        """
        # Run test.
        actual = _run(rnogafapr.FakeProvider({}), "google-vertex", _request())
        # Check outputs.
        self.assertEqual(
            (actual.status, actual.http_status), (rnogapro.GATEWAY_ERROR, 404)
        )

    def test4(self) -> None:
        """
        Test that random failures are repeatable for a given seed.
        """

        def _statuses(seed: int) -> list:
            fake = rnogafapr.FakeProvider(
                {"p": rnogafapr.FakeBehavior("P", fail_rate=0.5)}, seed=seed
            )
            return [_run(fake, "p", _request()).status for _ in range(20)]

        # Run test.
        first = _statuses(7)
        # Check outputs.
        self.assertEqual(first, _statuses(7))
        self.assertIn(rnogapro.OK, first)
        self.assertIn(rnogapro.UPSTREAM_ERROR, first)

    def test5(self) -> None:
        """
        Test impersonation and the `max_tokens` cap on completion tokens.
        """
        # Prepare inputs.
        fake = rnogafapr.FakeProvider(
            {"groq": rnogafapr.FakeBehavior("Groq", impersonate="Together")}
        )
        # Run test.
        actual = _run(fake, "groq", _request(max_tokens=2))
        # Check outputs.
        self.assertEqual(
            (actual.served_by, actual.completion_tokens), ("Together", 2)
        )

    def test6(self) -> None:
        """
        Test that `real_sleep` actually waits.
        """
        # Prepare inputs.
        fake = rnogafapr.FakeProvider(
            {"p": rnogafapr.FakeBehavior("P", latency_s=0.15)}, real_sleep=True
        )
        start = time.perf_counter()
        # Run test.
        _run(fake, "p", _request())
        # Check outputs.
        self.assertGreaterEqual(time.perf_counter() - start, 0.14)


# #############################################################################
# TestProviderResult
# #############################################################################


class TestProviderResult(hunitest.TestCase):
    """
    Test the shared provider interface types.
    """

    def test1(self) -> None:
        """
        Test that only TIMEOUT and UPSTREAM_ERROR count against the seller.
        """
        # Prepare outputs.
        expected = {
            rnogapro.OK: False,
            rnogapro.TIMEOUT: True,
            rnogapro.UPSTREAM_ERROR: True,
            rnogapro.BUYER_ERROR: False,
            rnogapro.GATEWAY_ERROR: False,
        }
        # Run test.
        actual = {}
        for status in expected:
            result = rnogapro.ProviderResult(
                status=status,
                text=None,
                served_by=None,
                model_served=None,
                prompt_tokens=None,
                completion_tokens=None,
                cost_usd=0.0,
                latency_s=0.0,
                http_status=None,
                error=None,
            )
            actual[status] = result.counts_against_seller
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test that both adapters implement the `Provider.call` signature.
        """
        # Prepare outputs.
        expected = ["self", "slug", "request"]
        # Run test and check outputs.
        for cls in [rnogaope.OpenRouterProvider, rnogafapr.FakeProvider]:
            params = list(inspect.signature(cls.call).parameters)
            self.assertEqual(params, expected)
            self.assertTrue(inspect.iscoroutinefunction(cls.call))
