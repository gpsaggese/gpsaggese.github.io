import asyncio
import json
import logging
from typing import Any, Callable, List, Optional

import httpx

import helpers.hunit_test as hunitest
import research.Noesis.gateway_openrouter as rnogaope
import research.Noesis.gateway_providers as rnogapro

_LOG = logging.getLogger(__name__)

_MODEL = "meta-llama/llama-3.3-70b-instruct"
_REQUEST = rnogapro.ProviderCall(
    model=_MODEL,
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
    max_tokens=16,
    timeout_s=5.0,
    temperature=0,
)
# Shape of the real Parasail 429 seen in `docs/findings/openrouter_spike.md`.
_PROVIDER_ERROR_BODY = {
    "error": {
        "message": "Provider returned error",
        "code": 429,
        "metadata": {"raw": "rate-limited upstream", "provider_name": "Parasail"},
    }
}


def _ok_body(provider: str) -> dict:
    """
    Return a successful OpenRouter response body.
    """
    body = {
        "id": "gen-1",
        "provider": provider,
        "model": _MODEL,
        "choices": [{"message": {"role": "assistant", "content": "391"}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2, "cost": 0.0000134},
    }
    return body


class _FakeClock:
    """
    Advance 0.5 s per read, so every call measures a latency of 0.5 s.
    """

    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        self.t += 0.5
        return self.t


def _call(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    slug: str = "groq",
    sent: Optional[List[httpx.Request]] = None,
) -> rnogapro.ProviderResult:
    """
    Run one `OpenRouterProvider.call()` against a mocked HTTP transport.

    :param handler: returns the mocked OpenRouter response
    :param slug: provider tag to pin
    :param sent: if given, collects the requests sent
    """

    def _recording(request: httpx.Request) -> httpx.Response:
        if sent is not None:
            sent.append(request)
        return handler(request)

    async def _go() -> rnogapro.ProviderResult:
        http = httpx.AsyncClient(transport=httpx.MockTransport(_recording))
        provider = rnogaope.OpenRouterProvider(
            "sk-test", http=http, clock_func=_FakeClock()
        )
        result = await provider.call(slug, _REQUEST)
        await http.aclose()
        return result

    return asyncio.run(_go())


# #############################################################################
# Test_classify_error
# #############################################################################


class Test_classify_error(hunitest.TestCase):
    """
    Test whose fault a failed OpenRouter response is.
    """

    def helper(self, http_status: int, body: Any, expected: str) -> None:
        """
        Classify and compare with `expected`.
        """
        # Run test.
        actual = rnogaope.classify_error(http_status, body)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test that an error OpenRouter attributes to the provider is the
        seller's fault (the real Parasail case).
        """
        self.helper(429, _PROVIDER_ERROR_BODY, rnogapro.UPSTREAM_ERROR)

    def test2(self) -> None:
        """
        Test that naming the provider in metadata alone is enough.
        """
        body = {"error": {"message": "x", "metadata": {"provider_name": "Groq"}}}
        self.helper(502, body, rnogapro.UPSTREAM_ERROR)

    def test3(self) -> None:
        """
        Test that OpenRouter's own 5xx/429 with no provider named is not the
        seller's fault (the blame fix).
        """
        self.helper(
            503, {"error": {"message": "Unavailable"}}, rnogapro.GATEWAY_ERROR
        )
        self.helper(
            429, {"error": {"message": "Rate limit"}}, rnogapro.GATEWAY_ERROR
        )
        self.helper(500, None, rnogapro.GATEWAY_ERROR)

    def test4(self) -> None:
        """
        Test that our key / credits / pin problems are our fault.
        """
        for code in [401, 402, 403, 404]:
            self.helper(code, {"error": {"message": "x"}}, rnogapro.GATEWAY_ERROR)

    def test5(self) -> None:
        """
        Test that malformed requests are the buyer's fault.
        """
        for code in [400, 413, 422]:
            self.helper(code, {"error": {"message": "x"}}, rnogapro.BUYER_ERROR)

    def test6(self) -> None:
        """
        Test that non-dict bodies don't crash and aren't blamed on the seller.
        """
        self.helper(502, "oops", rnogapro.GATEWAY_ERROR)
        self.helper(502, {"error": "flat string"}, rnogapro.GATEWAY_ERROR)


# #############################################################################
# TestOpenRouterProvider
# #############################################################################


class TestOpenRouterProvider(hunitest.TestCase):
    """
    Test the OpenRouter adapter end to end over a mocked HTTP transport.
    """

    def test1(self) -> None:
        """
        Test that a success is parsed into answer, attribution, usage, cost.
        """
        # Prepare outputs.
        expected = rnogapro.ProviderResult(
            status=rnogapro.OK,
            text="391",
            served_by="Groq",
            model_served=_MODEL,
            prompt_tokens=20,
            completion_tokens=2,
            cost_usd=0.0000134,
            latency_s=0.5,
            http_status=200,
            error=None,
        )
        # Run test.
        actual = _call(lambda req: httpx.Response(200, json=_ok_body("Groq")))
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test that the request pins the provider and disables fallbacks.
        """
        # Prepare inputs.
        sent: List[httpx.Request] = []
        # Prepare outputs.
        expected = {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "What is 17 * 23?"}],
            "max_tokens": 16,
            "provider": {"only": ["deepinfra/turbo"], "allow_fallbacks": False},
            "usage": {"include": True},
            "temperature": 0,
        }
        # Run test.
        _call(
            lambda req: httpx.Response(200, json=_ok_body("DeepInfra")),
            slug="deepinfra/turbo",
            sent=sent,
        )
        # Check outputs.
        (request,) = sent
        self.assertEqual(request.headers["Authorization"], "Bearer sk-test")
        self.assert_equal(str(json.loads(request.content)), str(expected))

    def test3(self) -> None:
        """
        Test that a garbage answer is still `OK` for the gateway (the checker
        grades content later).
        """
        # Prepare inputs.
        body = _ok_body("Novita")
        body["choices"][0]["message"]["content"] = "!!!!!!!!"
        # Run test.
        actual = _call(lambda req: httpx.Response(200, json=body))
        # Check outputs.
        self.assertEqual((actual.status, actual.text), (rnogapro.OK, "!!!!!!!!"))

    def test4(self) -> None:
        """
        Test that HTTP errors are classified, with zero cost.
        """
        # Run test.
        actual = _call(lambda req: httpx.Response(429, json=_PROVIDER_ERROR_BODY))
        # Check outputs.
        self.assertEqual(actual.status, rnogapro.UPSTREAM_ERROR)
        self.assertEqual((actual.http_status, actual.cost_usd), (429, 0.0))
        self.assertTrue(actual.counts_against_seller)

    def test5(self) -> None:
        """
        Test that a timeout counts against the seller.
        """

        def _handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=req)

        # Run test.
        actual = _call(_handler)
        # Check outputs.
        self.assertEqual(actual.status, rnogapro.TIMEOUT)
        self.assertTrue(actual.counts_against_seller)

    def test6(self) -> None:
        """
        Test that failing to reach OpenRouter is our fault.
        """

        def _handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns", request=req)

        # Run test.
        actual = _call(_handler)
        # Check outputs.
        self.assertEqual(actual.status, rnogapro.GATEWAY_ERROR)
        self.assertFalse(actual.counts_against_seller)

    def test7(self) -> None:
        """
        Test errors reported inside a 200: blamed on the seller only if a
        provider is named.
        """
        # Prepare inputs.
        named = {"provider": "Groq", "error": {"code": 502, "message": "died"}}
        unnamed = {"error": {"code": 500, "message": "internal"}}
        # Run test.
        actual_named = _call(lambda req: httpx.Response(200, json=named))
        actual_unnamed = _call(lambda req: httpx.Response(200, json=unnamed))
        # Check outputs.
        self.assertEqual(actual_named.status, rnogapro.UPSTREAM_ERROR)
        self.assertEqual(actual_named.served_by, "Groq")
        self.assertEqual(actual_unnamed.status, rnogapro.GATEWAY_ERROR)

    def test8(self) -> None:
        """
        Test that a 200 without an answer, or not JSON, is the seller's fault.
        """
        # Prepare inputs.
        bodies = [
            {"provider": "Groq", "choices": []},
            {"provider": "Groq", "choices": [{"message": {"role": "assistant"}}]},
        ]
        # Run test and check outputs.
        for body in bodies:
            actual = _call(lambda req, b=body: httpx.Response(200, json=b))
            self.assertEqual(actual.status, rnogapro.UPSTREAM_ERROR)
        actual = _call(lambda req: httpx.Response(200, text="<html>bad</html>"))
        self.assertEqual(actual.status, rnogapro.UPSTREAM_ERROR)

    def test9(self) -> None:
        """
        Test that an empty API key is rejected.
        """
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnogaope.OpenRouterProvider("")
