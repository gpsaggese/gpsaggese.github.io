import asyncio
import logging
import time
from typing import Optional, Tuple

import helpers.hunit_test as hunitest
import research.Noesis.gateway_fake_provider as rnogafapr
import research.Noesis.gateway_faults as rnogafau
import research.Noesis.gateway_providers as rnogapro

_LOG = logging.getLogger(__name__)

_70B = "meta-llama/llama-3.3-70b-instruct"
_8B = "meta-llama/llama-3.1-8b-instruct"


def _call(*, timeout_s: float = 2.0) -> rnogapro.ProviderCall:
    """
    Return a minimal provider call for the 70B model.
    """
    call = rnogapro.ProviderCall(
        model=_70B,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        timeout_s=timeout_s,
    )
    return call


def _fake(*, latency_s: float = 0.1) -> rnogafapr.FakeProvider:
    """
    Return a fake with Groq and DeepInfra's 8B tag.
    """
    fake = rnogafapr.FakeProvider(
        {
            "groq": rnogafapr.FakeBehavior("Groq", latency_s=latency_s),
            "deepinfra/fp8": rnogafapr.FakeBehavior(
                "DeepInfra", latency_s=latency_s
            ),
        }
    )
    return fake


def _run(
    fake: rnogafapr.FakeProvider,
    fault: Optional[rnogafau.Fault],
    call: rnogapro.ProviderCall,
    *,
    slug: str = "groq",
) -> Tuple[rnogapro.ProviderResult, bool]:
    """
    Run `call_with_fault()` synchronously.
    """
    return asyncio.run(rnogafau.call_with_fault(fake, slug, call, fault))


# #############################################################################
# TestFault
# #############################################################################


class TestFault(hunitest.TestCase):
    """
    Test fault validation and activity.
    """

    def test1(self) -> None:
        """
        Test that invalid faults are rejected.
        """
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnogafau.Fault(extra_latency_s=-1)
        with self.assertRaises(AssertionError):
            rnogafau.Fault(slug_override="deepinfra/fp8")

    def test2(self) -> None:
        """
        Test `is_active`.
        """
        # Run test and check outputs.
        self.assertFalse(rnogafau.Fault().is_active)
        self.assertTrue(rnogafau.Fault(extra_latency_s=0.5).is_active)
        self.assertTrue(rnogafau.Fault(model_override=_8B).is_active)


# #############################################################################
# TestFaultRegistry
# #############################################################################


class TestFaultRegistry(hunitest.TestCase):
    """
    Test the in-memory registry of active faults.
    """

    def test1(self) -> None:
        """
        Test set/get/all, that an inactive fault clears, and clear().
        """
        # Prepare inputs.
        reg = rnogafau.FaultRegistry()
        slow = rnogafau.Fault(extra_latency_s=3)
        # Run test and check outputs.
        reg.set("groq", slow)
        self.assertEqual(reg.get("groq"), slow)
        self.assertEqual(reg.all(), {"groq": slow})
        reg.set("groq", rnogafau.Fault())
        self.assertIsNone(reg.get("groq"))
        reg.set("groq", slow)
        reg.clear("groq")
        reg.clear("never-set")
        self.assertEqual(reg.all(), {})


# #############################################################################
# Test_call_with_fault
# #############################################################################


class Test_call_with_fault(hunitest.TestCase):
    """
    Test applying slowdowns and model swaps to provider calls.
    """

    def test1(self) -> None:
        """
        Test that no fault (or an inactive one) passes the call through.
        """
        # Run test and check outputs.
        for fault in [None, rnogafau.Fault()]:
            fake = _fake()
            result, injected = _run(fake, fault, _call())
            slug, sent = fake.calls[-1]
            self.assertFalse(injected)
            self.assertEqual(
                (result.status, result.latency_s), (rnogapro.OK, 0.1)
            )
            self.assertEqual(
                (slug, sent.model, sent.timeout_s), ("groq", _70B, 2.0)
            )

    def test2(self) -> None:
        """
        Test that a slowdown really delays and shrinks the provider timeout.
        """
        # Prepare inputs.
        fake = _fake(latency_s=0.1)
        start = time.perf_counter()
        # Run test.
        result, injected = _run(
            fake, rnogafau.Fault(extra_latency_s=0.2), _call()
        )
        # Check outputs.
        self.assertTrue(injected)
        self.assertGreaterEqual(time.perf_counter() - start, 0.19)
        self.assertEqual(result.status, rnogapro.OK)
        self.assertAlmostEqual(result.latency_s, 0.3)
        self.assertAlmostEqual(fake.calls[-1][1].timeout_s, 1.8)

    def test3(self) -> None:
        """
        Test that delay + provider time past the limit is a TIMEOUT.
        """
        # Run test.
        result, injected = _run(
            _fake(latency_s=0.2),
            rnogafau.Fault(extra_latency_s=0.15),
            _call(timeout_s=0.3),
        )
        # Check outputs.
        self.assertTrue(injected)
        self.assertEqual(result.status, rnogapro.TIMEOUT)

    def test4(self) -> None:
        """
        Test that a delay at or over the timeout never calls the provider.
        """
        # Prepare inputs.
        fake = _fake()
        # Run test.
        result, injected = _run(
            fake, rnogafau.Fault(extra_latency_s=0.2), _call(timeout_s=0.1)
        )
        # Check outputs.
        self.assertTrue(injected)
        self.assertEqual(
            (result.status, result.latency_s), (rnogapro.TIMEOUT, 0.1)
        )
        self.assertEqual(fake.calls, [])

    def test5(self) -> None:
        """
        Test a model swap on the same provider, with and without a tag
        override.
        """
        # Prepare inputs.
        fake = _fake()
        # Run test.
        result, injected = _run(fake, rnogafau.Fault(model_override=_8B), _call())
        _run(
            fake,
            rnogafau.Fault(model_override=_8B, slug_override="deepinfra/fp8"),
            _call(),
            slug="deepinfra/turbo",
        )
        # Check outputs.
        self.assertTrue(injected)
        self.assertEqual(
            (fake.calls[0][0], fake.calls[0][1].model), ("groq", _8B)
        )
        self.assertEqual((result.served_by, result.model_served), ("Groq", _8B))
        self.assertEqual(fake.calls[1][0], "deepinfra/fp8")
