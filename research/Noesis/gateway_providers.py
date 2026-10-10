"""
The provider interface: one standard request, one standard result.

Every way of reaching a model (OpenRouter today, direct provider APIs later, a
fake in tests) implements `Provider`. The rest of the gateway only sees
`ProviderCall` and `ProviderResult`.

For the overall design see `research/Noesis/docs/gateway.README.md`.

Import as:

import research.Noesis.gateway_providers as rnogapro
"""

import dataclasses
from typing import Dict, List, Optional, Protocol

# #############################################################################
# Statuses
# #############################################################################

# Result statuses. Only `TIMEOUT` and `UPSTREAM_ERROR` count against the seller.
OK = "ok"
# Provider too slow: seller's fault.
TIMEOUT = "timeout"
# Provider returned an error: seller's fault.
UPSTREAM_ERROR = "upstream_error"
# Malformed request: buyer's fault.
BUYER_ERROR = "buyer_error"
# Our key/credits/config, or OpenRouter itself: our fault.
GATEWAY_ERROR = "gateway_error"

SELLER_FAULT_STATUSES = frozenset({TIMEOUT, UPSTREAM_ERROR})


# #############################################################################
# ProviderCall
# #############################################################################


@dataclasses.dataclass(frozen=True)
class ProviderCall:
    """
    What we ask a provider to do.

    E.g., `ProviderCall("meta-llama/llama-3.3-70b-instruct", [{"role": "user",
    "content": "Hi"}], 256, 4.0)` asks for at most 256 tokens within 4 s.
    """

    model: str
    # OpenAI format: `[{"role": "user", "content": "..."}]`.
    messages: List[Dict[str, str]]
    max_tokens: int
    timeout_s: float
    # `None` means "don't send a temperature" (provider default), which is
    # different from any numeric value.
    temperature: Optional[float] = None


# #############################################################################
# ProviderResult
# #############################################################################


@dataclasses.dataclass(frozen=True)
class ProviderResult:
    """
    What came back from one call.

    Always returned, never raised: failures are expressed as a `status`, so
    every call can be logged and graded.
    """

    status: str
    # The answer, if any.
    text: Optional[str]
    # Provider that actually answered, e.g., "Groq".
    served_by: Optional[str]
    model_served: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    # Real cost in USD; 0.0 on failure.
    cost_usd: float
    # Measured by us, in seconds.
    latency_s: float
    http_status: Optional[int]
    # Short error text for the log.
    error: Optional[str]

    @property
    def counts_against_seller(self) -> bool:
        """
        Return whether this outcome should hurt the seller's reputation.
        """
        return self.status in SELLER_FAULT_STATUSES


# #############################################################################
# Provider
# #############################################################################


class Provider(Protocol):
    """
    Anything that can serve a `ProviderCall` through one pinned upstream.
    """

    async def call(self, slug: str, request: ProviderCall) -> ProviderResult:
        """
        Serve `request` through the upstream identified by `slug`.

        :param slug: which upstream to pin, e.g., "groq" (an OpenRouter tag)
        :param request: what to ask
        :return: the outcome; never raises for upstream failures
        """
        ...
