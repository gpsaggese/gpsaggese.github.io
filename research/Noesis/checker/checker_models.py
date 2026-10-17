"""
Define immutable values shared by the Noesis checker modules.

Import as:

import research.Noesis.checker.checker_models as rnochmod
"""

import dataclasses
import enum
from datetime import datetime
from typing import Dict, Mapping, Optional, Tuple


class Verdict(str, enum.Enum):
    """
    Represent contract outcomes persisted by the checker.
    """

    PASSED = "PASSED"
    FAILED = "FAILED"
    INSUFFICIENT = "INSUFFICIENT"


class CheckerDisposition(str, enum.Enum):
    """
    Represent whether evidence can safely be attributed to the seller.
    """

    VALID = "VALID"
    CHECKER_UNHEALTHY = "CHECKER_UNHEALTHY"


class CalibrationRole(str, enum.Enum):
    """
    Represent how a model/provider target is used during calibration.
    """

    HEALTHY = "HEALTHY"
    WEAK = "WEAK"


@dataclasses.dataclass(frozen=True)
class Canary:
    """
    Represent one versioned, deterministically graded test question.
    """

    canary_id: str
    bank_version: str
    family: str
    prompt: str
    answer: str
    normalizer: str
    generator_seed: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class GradeResult:
    """
    Represent the result of applying one deterministic grader.
    """

    # None means the checker, rather than the seller, could not grade the answer.
    correct: Optional[bool]
    reason: str
    normalized_expected: Optional[str]
    normalized_actual: Optional[str]
    grader_version: str


@dataclasses.dataclass(frozen=True)
class CalibrationTarget:
    """
    Represent one provider/model target used during calibration.
    """

    role: CalibrationRole
    provider_slug: str
    model_id: str

    def __post_init__(self) -> None:
        if not self.provider_slug:
            raise ValueError("provider_slug must not be empty")
        if not self.model_id:
            raise ValueError("model_id must not be empty")


@dataclasses.dataclass(frozen=True)
class CalibrationObservation:
    """
    Represent one graded calibration attempt against one target.
    """

    canary_id: str
    target: CalibrationTarget
    attempt: int
    provider_status: str
    correct: Optional[bool]
    latency_ms: float
    cost_usd: float
    response: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.canary_id:
            raise ValueError("canary_id must not be empty")
        if self.attempt < 1:
            raise ValueError("attempt must be positive")
        if not self.provider_status:
            raise ValueError("provider_status must not be empty")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must not be negative")
        if self.cost_usd < 0:
            raise ValueError("cost_usd must not be negative")


@dataclasses.dataclass(frozen=True)
class CanaryCalibration:
    """
    Represent the calibration decision for one candidate canary.
    """

    canary_id: str
    healthy_accuracy: Optional[float]
    weak_accuracy: Optional[float]
    accepted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class CalibrationPolicy:
    """
    Represent the reproducible settings for one calibration run.
    """

    repetitions: int = 3
    max_tokens: int = 64
    timeout_s: float = 15.0
    concurrency: int = 5
    healthy_required_accuracy: float = 1.0
    weak_maximum_accuracy: float = 0.6

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if not 0 <= self.healthy_required_accuracy <= 1:
            raise ValueError(
                "healthy_required_accuracy must be between zero and one"
            )
        if not 0 <= self.weak_maximum_accuracy <= 1:
            raise ValueError(
                "weak_maximum_accuracy must be between zero and one"
            )


@dataclasses.dataclass(frozen=True)
class CalibrationExecution:
    """
    Represent all reproducible outputs of one calibration execution.
    """

    observations: Tuple[CalibrationObservation, ...]
    decisions: Tuple[CanaryCalibration, ...]
    summary: Mapping[str, object]


@dataclasses.dataclass(frozen=True)
class ContractTerms:
    """
    Represent contract promises consumed by the checker.
    """

    contract_id: int
    l_max_s: float
    r_min: float
    q_min: float

    def __post_init__(self) -> None:
        if self.contract_id <= 0:
            raise ValueError("contract_id must be positive")
        if self.l_max_s <= 0:
            raise ValueError("l_max_s must be positive")
        if not 0 <= self.r_min <= 1:
            raise ValueError("r_min must be between zero and one")
        if not 0 <= self.q_min <= 1:
            raise ValueError("q_min must be between zero and one")


@dataclasses.dataclass(frozen=True)
class ContractWindow:
    """
    Represent persisted contract fields needed by checker orchestration.
    """

    terms: ContractTerms
    state: str
    target_requests: int
    target_canaries: int
    maximum_seconds: int
    checker_policy: str
    window_start: Optional[datetime]
    window_end: Optional[datetime]


@dataclasses.dataclass(frozen=True)
class ProbeScheduleItem:
    """
    Represent one deterministic canary slot in a contract window.
    """

    contract_id: int
    probe_slot: int
    canary_id: str
    due_at: datetime

    def __post_init__(self) -> None:
        if self.contract_id <= 0:
            raise ValueError("contract_id must be positive")
        if self.probe_slot < 1:
            raise ValueError("probe_slot must be positive")
        if not self.canary_id:
            raise ValueError("canary_id must not be empty")
        if self.due_at.utcoffset() is None:
            raise ValueError("due_at must be timezone-aware")


@dataclasses.dataclass(frozen=True)
class ProbePolicy:
    """
    Represent provider-call settings used by the scheduled prober.
    """

    bank_version: str
    model_id: str
    max_tokens: int = 64
    timeout_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if not self.bank_version:
            raise ValueError("bank_version must not be empty")
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout_multiplier <= 0:
            raise ValueError("timeout_multiplier must be positive")


@dataclasses.dataclass(frozen=True)
class ProbeExecution:
    """
    Represent the persisted outcome of one claimed probe.
    """

    probe_id: int
    request_id: int
    provider_status: str
    correct: Optional[bool]
    grader_reason: str


@dataclasses.dataclass(frozen=True)
class WindowPolicy:
    """
    Represent checker policy independent of a seller's promises.
    """

    minimum_requests: int = 20
    minimum_canaries: int = 8
    confidence: float = 0.95
    # Canary prompts are short and can bias buyer latency downward.
    include_canaries_in_latency: bool = False
    exact_bound_below: int = 10

    def __post_init__(self) -> None:
        if self.minimum_requests < 1:
            raise ValueError("minimum_requests must be positive")
        if self.minimum_canaries < 1:
            raise ValueError("minimum_canaries must be positive")
        if not 0.5 < self.confidence < 1:
            raise ValueError("confidence must be between 0.5 and 1")
        if self.exact_bound_below < 1:
            raise ValueError("exact_bound_below must be positive")


@dataclasses.dataclass(frozen=True)
class RequestObservation:
    """
    Represent the request fields used to construct contract evidence.
    """

    request_id: int
    status: str
    latency_ms: float
    is_canary: bool = False
    canary_correct: Optional[bool] = None
    attribution_mismatch: bool = False


@dataclasses.dataclass(frozen=True)
class EvidenceClassification:
    """
    Represent how one request contributes to one metric.
    """

    eligible: bool
    success: Optional[bool]
    reason: str
    checker_error: bool = False


@dataclasses.dataclass(frozen=True)
class ClassifiedRequest:
    """
    Represent latency and quality classifications for one request.
    """

    request: RequestObservation
    latency: EvidenceClassification
    quality: EvidenceClassification


@dataclasses.dataclass(frozen=True)
class EvidenceSet:
    """
    Represent all classified requests in a contract evaluation window.
    """

    requests: Tuple[ClassifiedRequest, ...]

    @property
    def checker_healthy(self) -> bool:
        """
        Return whether every observation was classified safely.
        """
        return not any(
            row.latency.checker_error or row.quality.checker_error
            for row in self.requests
        )


@dataclasses.dataclass(frozen=True)
class MetricAssessment:
    """
    Represent observed results and a promised metric's confidence bound.
    """

    successes: int
    total: int
    rate: Optional[float]
    upper_bound: Optional[float]
    promised_rate: float
    sufficient: bool
    conclusive_failure: bool


@dataclasses.dataclass(frozen=True)
class VerdictDecision:
    """
    Represent the complete explainable output of the verdict engine.
    """

    verdict: Verdict
    disposition: CheckerDisposition
    latency: MetricAssessment
    quality: MetricAssessment
    reason_codes: Tuple[str, ...]
    excluded_counts: Dict[str, int]
