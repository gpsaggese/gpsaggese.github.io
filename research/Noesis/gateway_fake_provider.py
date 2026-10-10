"""
In-memory `Provider` for tests: scripted behavior per slug, no network.

Import as:

import research.Noesis.gateway_fake_provider as rnogafapr
"""

import asyncio
import dataclasses
import logging
import random
from typing import Callable, Dict, List, Optional, Tuple

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.gateway_providers as rnogapro

_LOG = logging.getLogger(__name__)

# Turns the request's messages into the answer text.
AnswerFunc = Callable[[List[Dict[str, str]]], str]


def _default_answer(messages: List[Dict[str, str]]) -> str:
    """
    Answer "ok" to anything.
    """
    _ = messages
    return "ok"


# #############################################################################
# FakeBehavior
# #############################################################################


@dataclasses.dataclass
class FakeBehavior:
    """
    How one fake upstream behaves.

    E.g., `FakeBehavior("Novita", latency_s=9.0)` always times out under a
    4 s timeout; `FakeBehavior("Groq", impersonate="Together")` answers but
    claims to be Together (an attribution mismatch).
    """

    # Name reported back, e.g., "Groq".
    served_by: str
    # Reported latency, in seconds.
    latency_s: float = 0.3
    answer: AnswerFunc = _default_answer
    # Chance in [0, 1] that a call fails with `fail_status`.
    fail_rate: float = 0.0
    fail_status: str = rnogapro.UPSTREAM_ERROR
    cost_usd: float = 0.00001
    # If not "", report this provider instead of `served_by`.
    impersonate: str = ""


# #############################################################################
# FakeProvider
# #############################################################################


class FakeProvider:
    """
    `Provider` whose behavior is configured per slug.

    - Latency above the request's `timeout_s` becomes a `TIMEOUT`, like a real
      provider
    - `real_sleep=True` actually waits `latency_s` (for timing tests)
    - Every call is recorded in `calls` so tests can inspect what was sent
    """

    def __init__(
        self,
        behaviors: Dict[str, FakeBehavior],
        *,
        seed: int = 0,
        real_sleep: bool = False,
    ) -> None:
        """
        Initialize the fake provider.

        :param behaviors: one script per slug
        :param seed: seed for random failures, so they are repeatable
        :param real_sleep: whether to actually wait `latency_s`
        """
        self.behaviors = behaviors
        self.calls: List[Tuple[str, rnogapro.ProviderCall]] = []
        self._rng = random.Random(seed)
        self._real_sleep = real_sleep

    async def call(
        self, slug: str, request: rnogapro.ProviderCall
    ) -> rnogapro.ProviderResult:
        """
        Serve `request` according to the script for `slug`.

        :param slug: which scripted upstream to use
        :param request: what to ask
        :return: the scripted outcome
        """
        _LOG.debug(hprint.to_str("slug"))
        self.calls.append((slug, request))
        behavior = self.behaviors.get(slug)
        if behavior is None:
            # Unknown pin: same as OpenRouter's 404 "no endpoints" -> our fault.
            return _failure(
                rnogapro.GATEWAY_ERROR,
                0.0,
                http_status=404,
                error=f"unknown slug '{slug}'",
            )
        hdbg.dassert_lte(0.0, behavior.fail_rate, "fail_rate must be in [0, 1]")
        latency_s = min(behavior.latency_s, request.timeout_s)
        if self._real_sleep:
            await asyncio.sleep(latency_s)
        if behavior.latency_s > request.timeout_s:
            return _failure(rnogapro.TIMEOUT, latency_s, error="timeout")
        if self._rng.random() < behavior.fail_rate:
            return _failure(
                behavior.fail_status,
                latency_s,
                http_status=502,
                served_by=behavior.served_by,
                error="injected failure",
            )
        result = rnogapro.ProviderResult(
            status=rnogapro.OK,
            text=behavior.answer(request.messages),
            served_by=behavior.impersonate or behavior.served_by,
            model_served=request.model,
            prompt_tokens=10,
            completion_tokens=min(5, request.max_tokens),
            cost_usd=behavior.cost_usd,
            latency_s=latency_s,
            http_status=200,
            error=None,
        )
        return result


def _failure(
    status: str,
    latency_s: float,
    *,
    http_status: Optional[int] = None,
    served_by: Optional[str] = None,
    error: str = "",
) -> rnogapro.ProviderResult:
    """
    Build a failure result with no answer and zero cost.
    """
    result = rnogapro.ProviderResult(
        status=status,
        text=None,
        served_by=served_by,
        model_served=None,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=0.0,
        latency_s=latency_s,
        http_status=http_status,
        error=error,
    )
    return result
