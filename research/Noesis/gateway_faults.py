"""
Demo fault injection: deliberately degrade one seller to prove the checker
catches it.

Off by default; every affected request is logged with `fault_injected = true`.
Faults live in memory (lost on restart), which is fine for a demo run.

Import as:

import research.Noesis.gateway_faults as rnogafau
"""

import asyncio
import dataclasses
import logging
from typing import Dict, Optional, Tuple

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.gateway_providers as rnogapro

_LOG = logging.getLogger(__name__)


# #############################################################################
# Fault
# #############################################################################


@dataclasses.dataclass(frozen=True)
class Fault:
    """
    How to degrade one seller.

    E.g., `Fault(extra_latency_s=3.0)` adds 3 s to every answer;
    `Fault(model_override="meta-llama/llama-3.1-8b-instruct")` silently serves
    the 8B model on the same provider.
    """

    # Seconds added before the provider call (a real delay the buyer feels).
    extra_latency_s: float = 0.0
    # Serve this model instead of the contracted one; "" = no swap.
    model_override: str = ""
    # Pin this OpenRouter tag for the override model if the seller's normal
    # tag doesn't serve it (e.g., DeepInfra: "deepinfra/fp8"). Same provider,
    # so attribution still matches. "" = keep the seller's tag.
    slug_override: str = ""

    def __post_init__(self) -> None:
        hdbg.dassert_lte(
            0.0, self.extra_latency_s, "extra_latency_s must be non-negative"
        )
        hdbg.dassert(
            not self.slug_override or self.model_override,
            "slug_override needs model_override",
        )

    @property
    def is_active(self) -> bool:
        """
        Return whether this fault changes anything.
        """
        return self.extra_latency_s > 0 or self.model_override != ""


# #############################################################################
# FaultRegistry
# #############################################################################


class FaultRegistry:
    """
    Active faults by `seller_id`.
    """

    def __init__(self) -> None:
        self._faults: Dict[str, Fault] = {}

    def set(self, seller_id: str, fault: Fault) -> None:
        """
        Activate `fault` for `seller_id`; an inactive fault clears it.
        """
        if fault.is_active:
            self._faults[seller_id] = fault
        else:
            self._faults.pop(seller_id, None)

    def clear(self, seller_id: str) -> None:
        """
        Remove any fault for `seller_id`.
        """
        self._faults.pop(seller_id, None)

    def get(self, seller_id: str) -> Optional[Fault]:
        """
        Return the active fault for `seller_id`, if any.
        """
        return self._faults.get(seller_id)

    def all(self) -> Dict[str, Fault]:
        """
        Return a copy of all active faults.
        """
        return dict(self._faults)


# #############################################################################
# Applying faults
# #############################################################################


async def call_with_fault(
    provider: rnogapro.Provider,
    slug: str,
    call: rnogapro.ProviderCall,
    fault: Optional[Fault],
) -> Tuple[rnogapro.ProviderResult, bool]:
    """
    Call the provider, applying `fault` if one is active.

    - Slowdown: sleep first, then give the provider only the time left; if the
      delay alone uses up the timeout, it's a `TIMEOUT` (seller's fault)
    - Model swap: send the override model (and tag) instead

    :param provider: real or fake provider
    :param slug: the seller's normal tag
    :param call: what the buyer asked for
    :param fault: active fault for this seller, `None` if none
    :return: result as the buyer sees it, and whether a fault was applied
    """
    _LOG.debug(hprint.to_str("slug fault"))
    if fault is None or not fault.is_active:
        result = await provider.call(slug, call)
        return result, False
    delay = fault.extra_latency_s
    if delay >= call.timeout_s:
        # The delay alone exceeds the limit: the buyer waits until timeout.
        await asyncio.sleep(call.timeout_s)
        result = rnogapro.ProviderResult(
            status=rnogapro.TIMEOUT,
            text=None,
            served_by=None,
            model_served=None,
            prompt_tokens=None,
            completion_tokens=None,
            cost_usd=0.0,
            latency_s=call.timeout_s,
            http_status=None,
            error="timeout (injected delay)",
        )
        return result, True
    if delay > 0:
        await asyncio.sleep(delay)
    faulty_call = dataclasses.replace(
        call,
        model=fault.model_override or call.model,
        timeout_s=call.timeout_s - delay,
    )
    result = await provider.call(fault.slug_override or slug, faulty_call)
    # The buyer waited for the delay too.
    result = dataclasses.replace(result, latency_s=result.latency_s + delay)
    return result, True
