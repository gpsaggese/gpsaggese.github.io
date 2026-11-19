"""
Execute bounded, provider-pinned research campaigns with durable callbacks.

Import as:

import research.Noesis.monitor.monitor_paid_execution as rnmonpaex
"""

import dataclasses
from typing import Awaitable, Callable, Dict, Optional, Sequence, Tuple

import research.Noesis.gateway.gateway_providers as rnogapro


@dataclasses.dataclass(frozen=True)
class PaidCallSpec:
    """
    Freeze one provider call in a paid research plan.
    """

    call_id: str
    batch: int
    route_name: str
    provider_slug: str
    expected_provider_name: str
    model_id: str
    prompt: str
    max_tokens: int
    timeout_s: float
    metadata: Tuple[Tuple[str, str], ...] = ()

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
        if self.batch < 1 or self.max_tokens < 1 or self.timeout_s <= 0:
            raise ValueError("call bounds must be positive")
        keys = [key for key, _ in self.metadata]
        if len(keys) != len(set(keys)):
            raise ValueError("metadata keys must be unique")
        temperature = dict(self.metadata).get("sampling_temperature")
        if temperature is not None:
            try:
                numeric_temperature = float(temperature)
            except ValueError as error:
                raise ValueError(
                    "sampling_temperature metadata must be numeric"
                ) from error
            if not 0 <= numeric_temperature <= 2:
                raise ValueError(
                    "sampling_temperature must be between zero and two"
                )

    @property
    def sampling_temperature(self) -> float:
        """
        Return the explicitly frozen sampling temperature for this call.

        Older paid plans predate temperature metadata and intentionally retain
        their original deterministic value of zero.
        """
        value = dict(self.metadata).get("sampling_temperature", "0")
        return float(value)


@dataclasses.dataclass(frozen=True)
class PaidExecutionPolicy:
    """
    Bound calls, cost, and operational failures for a paid plan.
    """

    maximum_calls: int
    maximum_cost_usd: float
    per_call_reserve_usd: float
    timeout_count: int = 2
    consecutive_provider_errors: int = 2
    maximum_error_rate: float = 0.20

    def __post_init__(self) -> None:
        if self.maximum_calls < 1:
            raise ValueError("maximum_calls must be positive")
        if self.maximum_cost_usd <= 0 or self.per_call_reserve_usd <= 0:
            raise ValueError("cost limits must be positive")
        if self.maximum_calls * self.per_call_reserve_usd > self.maximum_cost_usd:
            raise ValueError("call reserves exceed the campaign cap")
        if self.timeout_count < 1 or self.consecutive_provider_errors < 1:
            raise ValueError("failure counts must be positive")
        if not 0 <= self.maximum_error_rate <= 1:
            raise ValueError("maximum_error_rate must be between zero and one")


@dataclasses.dataclass(frozen=True)
class PaidObservation:
    """
    Store one result and the evidence required by the safety gate.
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
    metadata: Tuple[Tuple[str, str], ...]


@dataclasses.dataclass(frozen=True)
class PaidExecution:
    """
    Store a complete or halted paid research execution.
    """

    state: str
    halt_reason: Optional[str]
    observations: Tuple[PaidObservation, ...]
    total_cost_usd: float


ObservationCallback = Callable[[PaidObservation], Awaitable[None]]


def validate_paid_plan(
    calls: Sequence[PaidCallSpec], policy: PaidExecutionPolicy
) -> None:
    """
    Reject expanded or malformed plans before any paid call.
    """
    if len(calls) != policy.maximum_calls:
        raise ValueError(
            "plan must contain exactly %s calls" % policy.maximum_calls
        )
    call_ids = [call.call_id for call in calls]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("call ids must be unique")


def _to_provider_call(call: PaidCallSpec) -> rnogapro.ProviderCall:
    return rnogapro.ProviderCall(
        model=call.model_id,
        messages=[{"role": "user", "content": call.prompt}],
        max_tokens=call.max_tokens,
        timeout_s=call.timeout_s,
        temperature=call.sampling_temperature,
    )


def _to_observation(
    call: PaidCallSpec, result: rnogapro.ProviderResult
) -> PaidObservation:
    successful = result.status == rnogapro.OK
    attribution_matches = (
        successful
        and result.served_by == call.expected_provider_name
        and result.model_served == call.model_id
    )
    usage_complete = (
        successful
        and result.prompt_tokens is not None
        and result.completion_tokens is not None
        and result.cost_usd > 0
    )
    return PaidObservation(
        call_id=call.call_id,
        batch=call.batch,
        route_name=call.route_name,
        requested_provider=call.provider_slug,
        expected_provider_name=call.expected_provider_name,
        requested_model=call.model_id,
        provider_status=result.status,
        served_by=result.served_by,
        model_served=result.model_served,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_s * 1000,
        http_status=result.http_status,
        response=result.text,
        error=result.error,
        attribution_matches=attribution_matches,
        usage_complete=usage_complete,
        metadata=call.metadata,
    )


def _halt_reason(
    observations: Sequence[PaidObservation], policy: PaidExecutionPolicy
) -> Optional[str]:
    latest = observations[-1]
    if latest.provider_status == rnogapro.OK and not latest.attribution_matches:
        return "ATTRIBUTION_MISMATCH"
    if latest.provider_status == rnogapro.OK and not latest.usage_complete:
        return "MISSING_USAGE_METADATA"
    timeouts = sum(
        item.provider_status == rnogapro.TIMEOUT for item in observations
    )
    if timeouts >= policy.timeout_count:
        return "TIMEOUT_LIMIT"
    consecutive_errors = 0
    for item in reversed(observations):
        if item.provider_status != rnogapro.UPSTREAM_ERROR:
            break
        consecutive_errors += 1
    if consecutive_errors >= policy.consecutive_provider_errors:
        return "CONSECUTIVE_PROVIDER_ERRORS"
    errors = sum(item.provider_status != rnogapro.OK for item in observations)
    if errors / len(observations) > policy.maximum_error_rate:
        return "ERROR_RATE_LIMIT"
    return None


async def execute_paid_plan(
    provider: rnogapro.Provider,
    calls: Sequence[PaidCallSpec],
    policy: PaidExecutionPolicy,
    on_observation: ObservationCallback,
) -> PaidExecution:
    """
    Execute sequentially and persist each result before applying stop rules.
    """
    validate_paid_plan(calls, policy)
    observations = []
    total_cost_usd = 0.0
    for call in calls:
        if total_cost_usd + policy.per_call_reserve_usd > policy.maximum_cost_usd:
            return PaidExecution(
                "HALTED", "COST_LIMIT", tuple(observations), total_cost_usd
            )
        try:
            result = await provider.call(
                call.provider_slug, _to_provider_call(call)
            )
        except Exception as error:  # pylint: disable=W0718
            result = rnogapro.ProviderResult(
                status=rnogapro.GATEWAY_ERROR,
                text=None,
                served_by=None,
                model_served=None,
                prompt_tokens=None,
                completion_tokens=None,
                cost_usd=0.0,
                latency_s=0.0,
                http_status=None,
                error=str(error)[:500],
            )
        observation = _to_observation(call, result)
        await on_observation(observation)
        observations.append(observation)
        total_cost_usd += observation.cost_usd
        if total_cost_usd > policy.maximum_cost_usd:
            return PaidExecution(
                "HALTED",
                "COST_LIMIT_EXCEEDED",
                tuple(observations),
                total_cost_usd,
            )
        reason = _halt_reason(observations, policy)
        if reason is not None:
            return PaidExecution(
                "HALTED", reason, tuple(observations), total_cost_usd
            )
    return PaidExecution(
        "COMPLETED", None, tuple(observations), total_cost_usd
    )


def summarize_paid_execution(execution: PaidExecution) -> Dict[str, object]:
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
