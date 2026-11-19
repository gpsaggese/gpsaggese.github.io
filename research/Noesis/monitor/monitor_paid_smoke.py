"""
Plan and execute the bounded Phase 0 provider-attribution smoke test.

Import as:

import research.Noesis.monitor.monitor_paid_smoke as rnmonpasm
"""

import dataclasses
from typing import Awaitable, Callable, Dict, Mapping, Optional, Sequence, Tuple

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
import research.Noesis.monitor.monitor_paid_execution as rnmonpaex


ObservationCallback = Callable[["SmokeObservation"], Awaitable[None]]


@dataclasses.dataclass(frozen=True)
class SmokeCallSpec:
    """
    Freeze one low-token call in the attribution-smoke manifest.
    """

    call_id: str
    batch: int
    route_name: str
    provider_slug: str
    expected_provider_name: str
    model_id: str
    prompt: str
    max_tokens: int = 8
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "call_id",
            "route_name",
            "provider_slug",
            "expected_provider_name",
            "model_id",
            "prompt",
        ):
            if not getattr(self, name).strip():
                raise ValueError("%s must not be empty" % name)
        if self.batch not in (1, 2):
            raise ValueError("batch must be 1 or 2")
        if self.max_tokens < 1 or self.timeout_s <= 0:
            raise ValueError("call bounds must be positive")


@dataclasses.dataclass(frozen=True)
class SmokePolicy:
    """
    Bound spend and failure tolerance for the paid smoke test.
    """

    maximum_calls: int = 15
    maximum_cost_usd: float = 0.25
    per_call_reserve_usd: float = 0.01
    timeout_count: int = 2
    consecutive_provider_errors: int = 2
    maximum_error_rate: float = 0.20

    def __post_init__(self) -> None:
        if self.maximum_calls < 1:
            raise ValueError("maximum_calls must be positive")
        if self.maximum_cost_usd <= 0 or self.per_call_reserve_usd <= 0:
            raise ValueError("cost limits must be positive")
        if self.per_call_reserve_usd > self.maximum_cost_usd:
            raise ValueError("per-call reserve exceeds the campaign cap")
        if self.timeout_count < 1 or self.consecutive_provider_errors < 1:
            raise ValueError("failure counts must be positive")
        if not 0 <= self.maximum_error_rate <= 1:
            raise ValueError("maximum_error_rate must be between zero and one")


@dataclasses.dataclass(frozen=True)
class SmokeObservation:
    """
    Store one paid call result and its attribution checks.
    """

    call_id: str
    batch: int
    route_name: str
    requested_provider: str
    expected_provider_name: str
    requested_model: str
    provider_status: str
    served_by: Optional[str]
    model_served: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cost_usd: float
    latency_ms: float
    http_status: Optional[int]
    response: Optional[str]
    error: Optional[str]
    attribution_matches: bool
    usage_complete: bool


@dataclasses.dataclass(frozen=True)
class SmokeExecution:
    """
    Store a complete or halted attribution-smoke execution.
    """

    state: str
    halt_reason: Optional[str]
    observations: Tuple[SmokeObservation, ...]
    total_cost_usd: float


def _route_map(
    manifest: rnmonexma.PhaseZeroManifest,
) -> Mapping[str, rnmonexma.RouteSpec]:
    return {route.name: route for route in manifest.routes}


def build_default_smoke_calls(
    manifest: rnmonexma.PhaseZeroManifest = rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST,
) -> Tuple[SmokeCallSpec, ...]:
    """
    Build the reviewed five-call gate followed by ten additional calls.

    Totals are five Cloudflare 70B calls, four Cloudflare 8B calls, and three
    calls each to Cloudflare 3B and CoreWeave 70B.
    """
    route_names = (
        "cloudflare-llama-3.3-70b",
        "cloudflare-llama-3.1-8b",
        "cloudflare-llama-3.2-3b",
        "coreweave-llama-3.3-70b",
        "cloudflare-llama-3.3-70b",
        "cloudflare-llama-3.3-70b",
        "cloudflare-llama-3.1-8b",
        "cloudflare-llama-3.2-3b",
        "coreweave-llama-3.3-70b",
        "cloudflare-llama-3.3-70b",
        "cloudflare-llama-3.1-8b",
        "cloudflare-llama-3.2-3b",
        "coreweave-llama-3.3-70b",
        "cloudflare-llama-3.3-70b",
        "cloudflare-llama-3.1-8b",
    )
    routes = _route_map(manifest)
    calls = []
    for index, route_name in enumerate(route_names, start=1):
        route = routes.get(route_name)
        if route is None:
            raise ValueError("smoke route %r is absent from manifest" % route_name)
        call_id = "attribution-smoke-%02d" % index
        calls.append(
            SmokeCallSpec(
                call_id=call_id,
                batch=1 if index <= 5 else 2,
                route_name=route.name,
                provider_slug=route.provider_slug,
                expected_provider_name=route.expected_provider_name,
                model_id=route.model_id,
                prompt=(
                    "Reply with exactly OK. This is attribution test %s."
                    % call_id
                ),
            )
        )
    return tuple(calls)


def validate_smoke_plan(
    calls: Sequence[SmokeCallSpec], policy: SmokePolicy
) -> None:
    """
    Reject an expanded or malformed plan before any paid call.
    """
    if len(calls) != policy.maximum_calls:
        raise ValueError(
            "smoke plan must contain exactly %s calls" % policy.maximum_calls
        )
    call_ids = [call.call_id for call in calls]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("smoke call ids must be unique")
    if [call.batch for call in calls[:5]] != [1] * 5:
        raise ValueError("the first five calls must be batch 1")
    if [call.batch for call in calls[5:]] != [2] * 10:
        raise ValueError("the remaining ten calls must be batch 2")
    if any(call.max_tokens > 8 for call in calls):
        raise ValueError("attribution smoke calls are capped at eight tokens")


async def execute_paid_smoke(
    provider: rnogapro.Provider,
    calls: Sequence[SmokeCallSpec],
    policy: SmokePolicy,
    on_observation: ObservationCallback,
) -> SmokeExecution:
    """
    Execute through the shared paid-campaign safety and persistence core.
    """
    validate_smoke_plan(calls, policy)
    paid_calls = tuple(
        rnmonpaex.PaidCallSpec(
            call_id=call.call_id,
            batch=call.batch,
            route_name=call.route_name,
            provider_slug=call.provider_slug,
            expected_provider_name=call.expected_provider_name,
            model_id=call.model_id,
            prompt=call.prompt,
            max_tokens=call.max_tokens,
            timeout_s=call.timeout_s,
            metadata=(("stage", "attribution-smoke"),),
        )
        for call in calls
    )
    paid_policy = rnmonpaex.PaidExecutionPolicy(
        maximum_calls=policy.maximum_calls,
        maximum_cost_usd=policy.maximum_cost_usd,
        per_call_reserve_usd=policy.per_call_reserve_usd,
        timeout_count=policy.timeout_count,
        consecutive_provider_errors=policy.consecutive_provider_errors,
        maximum_error_rate=policy.maximum_error_rate,
    )
    smoke_observations = []

    async def _record(observation: rnmonpaex.PaidObservation) -> None:
        smoke_observation = SmokeObservation(
            **{
                field.name: getattr(observation, field.name)
                for field in dataclasses.fields(SmokeObservation)
            }
        )
        await on_observation(smoke_observation)
        smoke_observations.append(smoke_observation)

    execution = await rnmonpaex.execute_paid_plan(
        provider, paid_calls, paid_policy, _record
    )
    return SmokeExecution(
        execution.state,
        execution.halt_reason,
        tuple(smoke_observations),
        execution.total_cost_usd,
    )


def summarize_smoke(execution: SmokeExecution) -> Dict[str, object]:
    """
    Build a JSON-compatible execution summary.
    """
    return {
        "state": execution.state,
        "halt_reason": execution.halt_reason,
        "completed_calls": len(execution.observations),
        "successful_calls": sum(
            item.provider_status == rnogapro.OK
            for item in execution.observations
        ),
        "attribution_matches": sum(
            item.attribution_matches for item in execution.observations
        ),
        "total_cost_usd": execution.total_cost_usd,
    }
