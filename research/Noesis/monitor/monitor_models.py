"""
Define immutable values used by standalone model monitoring.

Import as:

import research.Noesis.monitor.monitor_models as rnmonmod
"""

import dataclasses
from typing import Mapping, Optional, Tuple

MAX_CANARIES_PER_RUN = 100


@dataclasses.dataclass(frozen=True)
class MonitoringTarget:
    """
    Identify one provider/model pair without a market contract.
    """

    provider_slug: str
    model_id: str

    def __post_init__(self) -> None:
        if not self.provider_slug.strip():
            raise ValueError("provider_slug must not be empty")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")


@dataclasses.dataclass(frozen=True)
class MonitoringPolicy:
    """
    Represent the bounded settings for one standalone monitoring run.
    """

    bank_version: str = "v1"
    canary_count: int = 10
    max_tokens: int = 64
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.bank_version.strip():
            raise ValueError("bank_version must not be empty")
        if not 1 <= self.canary_count <= MAX_CANARIES_PER_RUN:
            raise ValueError(
                "canary_count must be between 1 and %s" % MAX_CANARIES_PER_RUN
            )
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")


@dataclasses.dataclass(frozen=True)
class MonitoringObservation:
    """
    Represent one provider call and its deterministic grade.
    """

    canary_id: str
    target: MonitoringTarget
    provider_status: str
    correct: Optional[bool]
    grader_reason: str
    grader_version: str
    latency_ms: float
    cost_usd: float
    response: Optional[str] = None
    served_by: Optional[str] = None
    model_served: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    http_status: Optional[int] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.canary_id:
            raise ValueError("canary_id must not be empty")
        if not self.provider_status:
            raise ValueError("provider_status must not be empty")
        if not self.grader_reason:
            raise ValueError("grader_reason must not be empty")
        if not self.grader_version:
            raise ValueError("grader_version must not be empty")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must not be negative")
        if self.cost_usd < 0:
            raise ValueError("cost_usd must not be negative")
        for name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if value is not None and value < 0:
                raise ValueError("%s must not be negative" % name)


@dataclasses.dataclass(frozen=True)
class MonitoringExecution:
    """
    Represent the observations and summary from one monitoring run.
    """

    observations: Tuple[MonitoringObservation, ...]
    summary: Mapping[str, object]
