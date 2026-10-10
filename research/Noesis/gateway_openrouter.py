"""
OpenRouter implementation of `Provider`: one pinned upstream per call.

Every call is pinned (`provider.only=[slug]`, `allow_fallbacks=false`), so the
answer can only come from the seller's provider. Failures are classified so
that only real provider failures count against the seller.

This replaces the `PR_S7` placeholder in `plan.Noesis.md`.

Import as:

import research.Noesis.gateway_openrouter as rnogaope
"""

import logging
import time
from typing import Any, Callable, Dict, Optional

import httpx

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.gateway_providers as rnogapro

_LOG = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter's message when the upstream provider (not OpenRouter) failed.
_PROVIDER_ERROR_MESSAGE = "Provider returned error"
# The request itself was invalid.
_BUYER_HTTP_CODES = [400, 413, 422]


# #############################################################################
# Blame
# #############################################################################


def classify_error(http_status: int, body: Any) -> str:
    """
    Decide whose fault a failed OpenRouter response is.

    - OpenRouter says the provider failed      -> `UPSTREAM_ERROR` (seller)
    - Our key / credits / routing config       -> `GATEWAY_ERROR` (us)
    - Malformed request                        -> `BUYER_ERROR` (buyer)
    - Anything else (OpenRouter's own 5xx/429) -> `GATEWAY_ERROR` (not the
      seller)

    E.g., the real Parasail rate-limit error seen in
    `docs/findings/openrouter_spike.md` is
    `{"error": {"message": "Provider returned error", "code": 429, ...}}`
    and is classified as `UPSTREAM_ERROR`.

    :param http_status: HTTP status returned by OpenRouter
    :param body: parsed JSON body, or `None` if it wasn't JSON
    :return: one of the statuses in `gateway_providers`
    """
    # Default: our key / credits / pin (401/402/403/404) or OpenRouter itself.
    status = rnogapro.GATEWAY_ERROR
    error = body.get("error") if isinstance(body, dict) else None
    metadata = error.get("metadata") if isinstance(error, dict) else None
    names_provider = isinstance(error, dict) and (
        error.get("message") == _PROVIDER_ERROR_MESSAGE
        or (isinstance(metadata, dict) and bool(metadata.get("provider_name")))
    )
    if names_provider:
        status = rnogapro.UPSTREAM_ERROR
    elif http_status in _BUYER_HTTP_CODES:
        status = rnogapro.BUYER_ERROR
    return status


# #############################################################################
# OpenRouterProvider
# #############################################################################


class OpenRouterProvider:
    """
    `Provider` backed by OpenRouter's `/chat/completions`.

    `http` and `clock_func` are injectable so tests need no network.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        http: Optional[httpx.AsyncClient] = None,
        clock_func: Callable[[], float] = time.perf_counter,
    ) -> None:
        """
        Initialize the provider.

        :param api_key: OpenRouter API key
        :param base_url: OpenRouter API root
        :param http: client to send requests with
            - Default: a new client owned (and closed) by this object
        :param clock_func: monotonic clock used to measure latency
        """
        hdbg.dassert_ne(api_key, "", "OpenRouterProvider needs an API key")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = http or httpx.AsyncClient()
        self._owns_http = http is None
        self._clock_func = clock_func

    async def aclose(self) -> None:
        """
        Close the HTTP client if this object created it.
        """
        if self._owns_http:
            await self._http.aclose()

    async def call(
        self, slug: str, request: rnogapro.ProviderCall
    ) -> rnogapro.ProviderResult:
        """
        Send `request` to the upstream `slug` only, with no fallbacks.

        :param slug: OpenRouter endpoint tag to pin, e.g., "groq"
        :param request: what to ask
        :return: the outcome; never raises for network or upstream failures
        """
        _LOG.debug(hprint.to_str("slug"))
        body: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            # Pin the seller's provider so attribution is unambiguous.
            "provider": {"only": [slug], "allow_fallbacks": False},
            # Ask OpenRouter to report the real cost.
            "usage": {"include": True},
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        start = self._clock_func()
        # Network failures are expected outcomes to grade, not bugs, so they
        # are converted into statuses instead of propagating.
        try:
            resp = await self._http.post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=request.timeout_s,
            )
        except httpx.TimeoutException:
            return self._failure(slug, rnogapro.TIMEOUT, start, None, "timeout")
        except httpx.TransportError as e:
            # Couldn't reach OpenRouter at all: our side, not the seller's.
            return self._failure(
                slug, rnogapro.GATEWAY_ERROR, start, None, repr(e)
            )
        latency_s = self._clock_func() - start
        data = _parse_json(resp)
        if resp.status_code != 200:
            return self._failure(
                slug,
                classify_error(resp.status_code, data),
                start,
                resp.status_code,
                resp.text[:500],
                latency_s=latency_s,
            )
        if not isinstance(data, dict):
            return self._failure(
                slug,
                rnogapro.UPSTREAM_ERROR,
                start,
                200,
                "non-JSON body",
                latency_s=latency_s,
            )
        if "error" in data:
            # 200 with an error inside: OpenRouter's mid-request failure report.
            status = (
                rnogapro.UPSTREAM_ERROR
                if data.get("provider")
                else classify_error(200, data)
            )
            return self._failure(
                slug,
                status,
                start,
                200,
                str(data["error"])[:500],
                latency_s=latency_s,
                served_by=data.get("provider"),
            )
        choices = data.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        if "content" not in message:
            return self._failure(
                slug,
                rnogapro.UPSTREAM_ERROR,
                start,
                200,
                "no completion in response",
                latency_s=latency_s,
                served_by=data.get("provider"),
            )
        usage = data.get("usage") or {}
        result = rnogapro.ProviderResult(
            status=rnogapro.OK,
            text=message["content"],
            served_by=data.get("provider"),
            model_served=data.get("model"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost_usd=float(usage.get("cost") or 0.0),
            latency_s=latency_s,
            http_status=200,
            error=None,
        )
        _LOG.debug("return=%s", result)
        return result

    def _failure(
        self,
        slug: str,
        status: str,
        start: float,
        http_status: Optional[int],
        error: str,
        *,
        latency_s: float = -1.0,
        served_by: Optional[str] = None,
    ) -> rnogapro.ProviderResult:
        """
        Build a failure result with no answer and zero cost.

        :param latency_s: measured latency; -1.0 means "measure now"
        """
        if latency_s < 0:
            latency_s = self._clock_func() - start
        _LOG.info("Call '%s' via '%s': '%s'", status, slug, error[:200])
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


def _parse_json(resp: httpx.Response) -> Any:
    """
    Return the JSON body, or `None` if the body isn't JSON.
    """
    # A provider can return HTML or plain text on failure; that's an outcome to
    # classify, not a crash.
    try:
        data = resp.json()
    except ValueError:
        data = None
    return data
