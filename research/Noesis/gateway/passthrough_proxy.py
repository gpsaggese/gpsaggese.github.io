"""
Passthrough proxy for the Noesis Server.

`Gateway` exposes one `call()` API that dispatches a prompt to a registered
provider and logs the raw request/response pair together with provider,
model, latency, and cost metadata.

Import as:

import research.Noesis.gateway.passthrough_proxy as rnopapro
"""

import abc
import dataclasses
import logging
import time
from typing import Callable, Dict, List, Optional

import helpers.hdbg as hdbg
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)


# #############################################################################
# ProviderConfig
# #############################################################################


# A provider's call function stands in for the real network call to a
# provider's SDK/API (e.g., OpenAI, Anthropic). It takes `(model, prompt)`
# and returns the raw response text.
ProviderCallFunc = Callable[[str, str], str]


# #############################################################################
# ProviderConfig
# #############################################################################


@dataclasses.dataclass
class ProviderConfig:
    """
    Register one LLM provider behind the gateway's single `call()` API.

    E.g., `ProviderConfig("openai_mock", echo_func, 0.01)` registers a provider
    named "openai_mock" that charges 0.01 per character of prompt+response
    text.
    """

    # Short name callers pass to `Gateway.call()` to select this provider.
    name: str
    # Stand-in for the real provider SDK call.
    call_func: ProviderCallFunc
    # Crude placeholder pricing model: $ per character of prompt+response
    # text. This can be swapped for real per-token provider pricing once
    # one is needed.
    cost_per_char: float

    def __init__(
        self, name: str, call_func: ProviderCallFunc, cost_per_char: float
    ) -> None:
        hdbg.dassert_ne(name, "", "ProviderConfig needs a non-empty name")
        self.name = name
        self.call_func = call_func
        hdbg.dassert_lte(
            0.0,
            cost_per_char,
            "ProviderConfig.cost_per_char must be non-negative: %s",
            cost_per_char,
        )
        self.cost_per_char = cost_per_char


# #############################################################################
# RequestLogEntry
# #############################################################################


@dataclasses.dataclass
class RequestLogEntry:
    """
    Storage schema for one logged prompt/response pair.

    One entry is appended per `Gateway.call()`, matched or not: every request
    is logged regardless of provider outcome.
    """

    # Monotonically increasing id, unique within one `Gateway` instance.
    request_id: int
    # Name of the provider that served the request.
    provider: str
    # Model identifier forwarded to the provider.
    model: str
    # Raw, unscrubbed prompt text sent to the provider.
    prompt: str
    # Raw, unscrubbed response text returned by the provider.
    response: str
    # Wall-clock time the provider call took, in seconds.
    latency_in_secs: float
    # Estimated cost of the request, in dollars (see `ProviderConfig`).
    cost: float


# #############################################################################
# RequestLogStore
# #############################################################################


class RequestLogStore(abc.ABC):
    """
    Pluggable storage backend for `Gateway`'s request/response log.
    """

    @abc.abstractmethod
    def append(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        latency_in_secs: float,
        cost: float,
    ) -> RequestLogEntry:
        """
        Persist one logged request/response pair.

        :return: the persisted `RequestLogEntry`, with `request_id` assigned
            by the store
        """
        ...

    @abc.abstractmethod
    def get_all(self) -> List[RequestLogEntry]:
        """
        :return: every logged entry, in call order
        """
        ...

    @abc.abstractmethod
    def query(
        self, *, provider: str = "", model: str = ""
    ) -> List[RequestLogEntry]: ...


# #############################################################################
# _InMemoryRequestLogStore
# #############################################################################


# TODO(ai_gp): Make it public.
class _InMemoryRequestLogStore(RequestLogStore):
    """
    Default `RequestLogStore`: today's `_log`/`_next_request_id`, extracted
    unchanged; `append()` builds the same `RequestLogEntry` `Gateway.call()`
    builds today and keeps its own counter.
    """

    def __init__(self) -> None:
        self._log: List[RequestLogEntry] = []
        self._next_request_id = 0

    def append(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        latency_in_secs: float,
        cost: float,
    ) -> RequestLogEntry:
        entry = RequestLogEntry(
            self._next_request_id,
            provider,
            model,
            prompt,
            response,
            latency_in_secs,
            cost,
        )
        self._next_request_id += 1
        self._log.append(entry)
        return entry

    def get_all(self) -> List[RequestLogEntry]:
        return list(self._log)

    def query(
        self, *, provider: str = "", model: str = ""
    ) -> List[RequestLogEntry]:
        entries = self._log
        if provider:
            entries = [entry for entry in entries if entry.provider == provider]
        if model:
            entries = [entry for entry in entries if entry.model == model]
        return list(entries)


# #############################################################################
# Gateway
# #############################################################################


class Gateway:
    """
    Passthrough proxy.

    - One `call()` API dispatches a prompt to a registered provider and logs
      the raw request/response pair
    - `get_log()` / `query_log()` make every logged pair queryable
    - The log lives in a pluggable `RequestLogStore`, not directly on this
      class
    """

    def __init__(
        self,
        *,
        clock_func: Callable[[], float] = time.perf_counter,
        store: Optional[RequestLogStore] = None,
    ) -> None:
        """
        :param clock_func: monotonic clock used to measure call latency
            - Default: `time.perf_counter`
            - Tests inject a deterministic fake clock instead
        :param store: pluggable storage backend for the request/response log
        """
        self._providers: Dict[str, ProviderConfig] = {}
        # TODO(ai_gp): Enforce callers to pass.
        if store is None:
            store = _InMemoryRequestLogStore()
        self._store = store
        self._clock_func = clock_func

    def register_provider(self, provider_config: ProviderConfig) -> None:
        """
        Register a provider so `call()` can route requests to it.

        :param provider_config: provider to add behind the gateway's API
        """
        _LOG.debug(hprint.to_str("provider_config"))
        hdbg.dassert_not_in(
            provider_config.name,
            self._providers,
            "Provider '%s' is already registered",
            provider_config.name,
        )
        self._providers[provider_config.name] = provider_config

    def call(self, provider_name: str, model: str, prompt: str) -> str:
        """
        Route `prompt` to `provider_name`/`model` and log the exchange.

        :param provider_name: name of a provider registered via
        `register_provider()`
        :param model: model identifier to forward to the provider's
            `call_func`
        :param prompt: raw prompt text to send
        :return: raw response text returned by the provider
        """
        _LOG.debug(hprint.to_str("provider_name model"))
        hdbg.dassert_in(
            provider_name,
            self._providers,
            "Unknown provider '%s'; registered providers: %s",
            provider_name,
            sorted(self._providers.keys()),
        )
        provider_config = self._providers[provider_name]
        # Time only the provider call itself, not the logging bookkeeping.
        start_time = self._clock_func()
        response = provider_config.call_func(model, prompt)
        latency_in_secs = self._clock_func() - start_time
        cost = round(
            provider_config.cost_per_char * (len(prompt) + len(response)), 6
        )
        self._store.append(
            provider_name, model, prompt, response, latency_in_secs, cost
        )
        _LOG.debug("return=%s", response)
        return response

    def get_log(self) -> List[RequestLogEntry]:
        """
        :return: every logged request/response pair, in call order
        """
        return self._store.get_all()

    def query_log(
        self, *, provider: str = "", model: str = ""
    ) -> List[RequestLogEntry]:
        """
        Return logged entries filtered by provider and/or model.

        :param provider: keep only entries logged for this provider
            - Default: "" (no provider filter)
        :param model: keep only entries logged for this model
            - Default: "" (no model filter)
        :return: matching entries, in call order
        """
        return self._store.query(provider=provider, model=model)
