"""
HTTP endpoints for the Noesis gateway: OpenAI-compatible chat completions,
model listing, health, and operator admin.

Request flow for `POST /v1/chat/completions`:
auth -> contract -> route -> clamp tokens -> (fault) -> provider -> log ->
OpenAI-shaped response.

`RoutingError`s raised by any endpoint are turned into OpenAI-style error
responses once, by the handler registered in `gateway_app.create_app()`.

Import as:

import research.Noesis.gateway_api as rnogaapi
"""

import dataclasses
import logging
import time
from typing import Any, Dict, List, Optional

import fastapi
import psycopg
import psycopg_pool
import pydantic
from fastapi import Header
from fastapi.responses import JSONResponse

import research.Noesis.gateway_admin as rnogaadm
import research.Noesis.gateway_faults as rnogafau
import research.Noesis.gateway_providers as rnogapro
import research.Noesis.gateway_request_log as rnogarelo
import research.Noesis.gateway_routing as rnogarou

_LOG = logging.getLogger(__name__)

# HTTP status returned to the buyer for each failed provider status.
_FAILURE_HTTP_STATUS = {
    rnogapro.TIMEOUT: 504,
    rnogapro.UPSTREAM_ERROR: 502,
    rnogapro.GATEWAY_ERROR: 503,
    rnogapro.BUYER_ERROR: 400,
}


# #############################################################################
# Request bodies
# #############################################################################


class ChatMessage(pydantic.BaseModel):
    """
    One OpenAI-style chat message.
    """

    role: str
    content: str


class ChatCompletionRequest(pydantic.BaseModel):
    """
    Subset of OpenAI's chat-completions request we support.

    Unknown fields are ignored, so standard SDKs work unchanged.
    """

    model: str
    messages: List[ChatMessage] = pydantic.Field(min_length=1)
    max_tokens: Optional[int] = pydantic.Field(default=None, gt=0)
    temperature: Optional[float] = pydantic.Field(default=None, ge=0, le=2)
    stream: bool = False


class DevContractRequest(pydantic.BaseModel):
    """
    Body of `POST /admin/dev-contract`.
    """

    buyer_id: str
    seller_id: str
    n_tasks: int = pydantic.Field(default=50, gt=0)
    l_max: float = pydantic.Field(default=2.0, gt=0)
    r_min: float = pydantic.Field(default=0.9, ge=0, le=1)
    price: float = pydantic.Field(default=0.0, ge=0)


class FaultRequest(pydantic.BaseModel):
    """
    Body of `PUT /admin/faults/{seller_id}`.
    """

    extra_latency_s: float = pydantic.Field(default=0.0, ge=0)
    model_override: str = ""
    slug_override: str = ""


# #############################################################################
# Helpers
# #############################################################################


def error_response(
    http_status: int,
    code: str,
    message: str,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """
    Build an OpenAI-style error: `{"error": {"message", "type", "code"}}`.

    :param http_status: e.g., 404
    :param code: short machine-readable code, e.g., "contract_not_found"
    :param message: human-readable explanation
    :param headers: extra response headers
    :return: the response
    """
    error_type = "invalid_request_error" if http_status < 500 else "api_error"
    response = JSONResponse(
        status_code=http_status,
        content={"error": {"message": message, "type": error_type, "code": code}},
        headers=headers,
    )
    return response


def _completion_body(
    request_id: int,
    model: str,
    max_tokens: int,
    result: rnogapro.ProviderResult,
) -> Dict[str, Any]:
    """
    Build an OpenAI chat-completion body from a successful result.
    """
    prompt_tokens = result.prompt_tokens or 0
    completion_tokens = result.completion_tokens or 0
    finish_reason = "length" if completion_tokens >= max_tokens else "stop"
    body = {
        "id": f"noesis-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return body


async def _require_operator(
    request: fastapi.Request, authorization: Optional[str]
) -> None:
    """
    Raise `RoutingError` unless the caller holds an operator key.
    """
    api_key = rnogarou.parse_bearer(authorization)
    await rnogarou.authenticate(request.app.state.pool, api_key, "operator")


async def _require_seller(request: fastapi.Request, seller_id: str) -> None:
    """
    Raise `RoutingError` 404 unless `seller_id` exists.
    """
    async with request.app.state.pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT 1 FROM sellers WHERE seller_id = %s", (seller_id,)
        )
        row = await cursor.fetchone()
    if row is None:
        raise rnogarou.RoutingError(
            404, "seller_not_found", f"Seller '{seller_id}' not found"
        )


# #############################################################################
# Router
# #############################################################################


def build_router() -> fastapi.APIRouter:
    """
    Build the gateway's routes.

    Endpoints read shared state from `request.app.state`: `pool`, `provider`,
    `settings`, `faults` (set up by `gateway_app.create_app()`).

    :return: router to include in the app
    """
    router = fastapi.APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        request: fastapi.Request,
        authorization: Optional[str] = Header(default=None),
        x_noesis_contract: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        state = request.app.state
        settings = state.settings
        # Who is asking, and which seller serves them.
        api_key = rnogarou.parse_bearer(authorization)
        contract_id = rnogarou.parse_contract_id(x_noesis_contract)
        buyer = await rnogarou.authenticate(state.pool, api_key, "buyer")
        route = await rnogarou.resolve_route(
            state.pool, buyer.account_id, contract_id
        )
        if body.stream:
            return error_response(
                400, "stream_not_supported", "Streaming is not supported"
            )
        if body.model != settings.model_id:
            return error_response(
                404,
                "model_not_found",
                f"Model '{body.model}' not served; use '{settings.model_id}'",
            )
        # Clamp to the task definition (PRD D1).
        max_tokens = min(
            body.max_tokens or settings.max_tokens_cap, settings.max_tokens_cap
        )
        call = rnogapro.ProviderCall(
            model=settings.model_id,
            messages=[m.model_dump() for m in body.messages],
            max_tokens=max_tokens,
            timeout_s=settings.timeout_multiplier * route.l_max,
            temperature=body.temperature,
        )
        # Serve, applying any demo fault for this seller.
        fault = state.faults.get(route.seller_id)
        result, injected = await rnogafau.call_with_fault(
            state.provider, route.provider_slug, call, fault
        )
        # Log every call, success or failure: the checker grades from these.
        request_id = await rnogarelo.log_request(
            state.pool, route, call, result, fault_injected=injected
        )
        headers = {
            "X-Noesis-Request-Id": str(request_id),
            "X-Noesis-Contract": str(route.contract_id),
            "X-Noesis-Provider": result.served_by or "",
        }
        if result.status != rnogapro.OK:
            return error_response(
                _FAILURE_HTTP_STATUS[result.status],
                result.status,
                f"Provider call failed ({result.status})",
                headers=headers,
            )
        return JSONResponse(
            content=_completion_body(
                request_id, settings.model_id, max_tokens, result
            ),
            headers=headers,
        )

    @router.get("/v1/models")
    async def list_models(
        request: fastapi.Request,
        authorization: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        state = request.app.state
        api_key = rnogarou.parse_bearer(authorization)
        await rnogarou.authenticate(state.pool, api_key, "buyer")
        model = {
            "id": state.settings.model_id,
            "object": "model",
            "owned_by": "noesis",
        }
        return JSONResponse(content={"object": "list", "data": [model]})

    @router.get("/health")
    async def health(request: fastapi.Request) -> JSONResponse:
        # An unreachable DB is the condition this endpoint reports, so it is
        # caught and turned into a 503 rather than propagated.
        try:
            async with request.app.state.pool.connection() as conn:
                await conn.execute("SELECT 1")
        except (psycopg.Error, psycopg_pool.PoolTimeout) as e:
            _LOG.warning("Health check failed: '%s'", e)
            return JSONResponse(
                status_code=503, content={"status": "error", "db": "down"}
            )
        return JSONResponse(content={"status": "ok", "db": "ok"})

    @router.post("/admin/dev-contract", status_code=201)
    async def dev_contract(
        body: DevContractRequest,
        request: fastapi.Request,
        authorization: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        await _require_operator(request, authorization)
        contract = await rnogaadm.create_dev_contract(
            request.app.state.pool,
            body.buyer_id,
            body.seller_id,
            n_tasks=body.n_tasks,
            l_max=body.l_max,
            r_min=body.r_min,
            price=body.price,
        )
        return JSONResponse(status_code=201, content=dataclasses.asdict(contract))

    @router.get("/admin/faults")
    async def list_faults(
        request: fastapi.Request,
        authorization: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        await _require_operator(request, authorization)
        faults = request.app.state.faults.all()
        content = {k: dataclasses.asdict(v) for k, v in faults.items()}
        return JSONResponse(content=content)

    @router.put("/admin/faults/{seller_id}")
    async def set_fault(
        seller_id: str,
        body: FaultRequest,
        request: fastapi.Request,
        authorization: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        await _require_operator(request, authorization)
        await _require_seller(request, seller_id)
        if body.slug_override and not body.model_override:
            return error_response(
                400, "invalid_fault", "slug_override needs model_override"
            )
        fault = rnogafau.Fault(**body.model_dump())
        request.app.state.faults.set(seller_id, fault)
        _LOG.warning("FAULT INJECTION '%s': %s", seller_id, fault)
        content = {
            "seller_id": seller_id,
            **dataclasses.asdict(fault),
            "active": fault.is_active,
        }
        return JSONResponse(content=content)

    @router.delete("/admin/faults/{seller_id}")
    async def clear_fault(
        seller_id: str,
        request: fastapi.Request,
        authorization: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        await _require_operator(request, authorization)
        await _require_seller(request, seller_id)
        request.app.state.faults.clear(seller_id)
        _LOG.warning("FAULT INJECTION '%s': cleared", seller_id)
        return JSONResponse(content={"seller_id": seller_id, "active": False})

    return router
