"""
Define immutable values shared by the Noesis checker modules.

Import as:

import research.Noesis.checker_models as rnochmod
"""

import dataclasses
import enum
from datetime import datetime
from typing import Dict, Optional, Tuple


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
