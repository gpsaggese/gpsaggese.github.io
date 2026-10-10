"""
One `requests` row per provider call, success or failure.

The checker grades contracts only from these rows, so every call must be
logged, including timeouts and errors. Extends `PR_S1`'s request log with the
contract field required by `PR_S4`.

Import as:

import research.Noesis.gateway_request_log as rnogarelo
"""

import json
import logging
from typing import Optional

import psycopg_pool

import helpers.hprint as hprint
import research.Noesis.gateway_providers as rnogapro
import research.Noesis.gateway_routing as rnogarou

_LOG = logging.getLogger(__name__)


def is_attribution_mismatch(
    route: rnogarou.Route, result: rnogapro.ProviderResult
) -> bool:
    """
    Return whether we can't confirm the pinned provider produced this answer.

    - Answered, but by a different provider        -> mismatch
    - Answered, but OpenRouter didn't say by whom  -> mismatch (unverifiable)
    - No answer (failure)                          -> not a mismatch

    Mismatched rows are excluded from the seller's score.

    :param route: who should have answered
    :param result: what came back
    :return: whether attribution is unconfirmed
    """
    mismatch = (
        result.status == rnogapro.OK and result.served_by != route.provider_name
    )
    return mismatch


async def log_request(
    pool: psycopg_pool.AsyncConnectionPool,
    route: rnogarou.Route,
    call: rnogapro.ProviderCall,
    result: rnogapro.ProviderResult,
    *,
    is_canary: bool = False,
    canary_id: Optional[str] = None,
    canary_correct: Optional[bool] = None,
    fault_injected: bool = False,
) -> int:
    """
    Insert one `requests` row.

    :param pool: open connection pool
    :param route: contract/seller the call was made under
    :param call: what was sent
    :param result: what came back
    :param is_canary: set by the prober (milestone 2) together with
        `canary_id` and `canary_correct`
    :param fault_injected: set when a demo fault altered this call
    :return: the new `request_id`
    """
    _LOG.debug(hprint.to_str("route is_canary fault_injected"))
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO requests ("
            " contract_id, seller_id, pinned_provider, actual_provider,"
            " attribution_mismatch, model_requested, model_served,"
            " is_canary, canary_id, canary_correct,"
            " prompt, completion, prompt_tokens, completion_tokens,"
            " latency_ms, status, cost_usd, fault_injected"
            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,"
            "          %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING request_id",
            (
                route.contract_id,
                route.seller_id,
                route.provider_name,
                result.served_by,
                is_attribution_mismatch(route, result),
                call.model,
                result.model_served,
                is_canary,
                canary_id,
                canary_correct,
                json.dumps(call.messages),
                result.text,
                result.prompt_tokens,
                result.completion_tokens,
                result.latency_s * 1000.0,
                result.status,
                result.cost_usd,
                fault_injected,
            ),
        )
        (request_id,) = await cursor.fetchone()
    _LOG.debug("return=%s", request_id)
    return request_id
