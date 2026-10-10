"""
Classify request rows as seller evidence or excluded observations.

Import as:

import research.Noesis.checker_evidence as rnochevi
"""

import collections
from typing import Dict, Iterable

import research.Noesis.checker_models as rnochmod
import research.Noesis.gateway_providers as rnogapro

ELIGIBLE_ON_TIME = "ELIGIBLE_ON_TIME"
ELIGIBLE_TOO_SLOW = "ELIGIBLE_TOO_SLOW"
ELIGIBLE_CORRECT = "ELIGIBLE_CORRECT"
ELIGIBLE_INCORRECT = "ELIGIBLE_INCORRECT"
ELIGIBLE_SELLER_FAILURE = "ELIGIBLE_SELLER_FAILURE"

EXCLUDED_CANARY_LATENCY = "EXCLUDED_CANARY_LATENCY"
EXCLUDED_ATTRIBUTION_MISMATCH = "EXCLUDED_ATTRIBUTION_MISMATCH"
EXCLUDED_BUYER_ERROR = "EXCLUDED_BUYER_ERROR"
EXCLUDED_GATEWAY_ERROR = "EXCLUDED_GATEWAY_ERROR"
EXCLUDED_REFUSED_CAP = "EXCLUDED_REFUSED_CAP"
NOT_APPLICABLE = "NOT_APPLICABLE"

INVALID_LATENCY = "INVALID_LATENCY"
MISSING_CANARY_GRADE = "MISSING_CANARY_GRADE"
UNEXPECTED_CANARY_GRADE = "UNEXPECTED_CANARY_GRADE"
UNKNOWN_REQUEST_STATUS = "UNKNOWN_REQUEST_STATUS"

_NON_SELLER_EXCLUSIONS = {
    rnogapro.BUYER_ERROR: EXCLUDED_BUYER_ERROR,
    rnogapro.GATEWAY_ERROR: EXCLUDED_GATEWAY_ERROR,
    "refused_cap": EXCLUDED_REFUSED_CAP,
}


def _excluded(reason: str) -> rnochmod.EvidenceClassification:
    return rnochmod.EvidenceClassification(False, None, reason)


def _checker_error(reason: str) -> rnochmod.EvidenceClassification:
    return rnochmod.EvidenceClassification(False, None, reason, checker_error=True)


def _eligible(success: bool, reason: str) -> rnochmod.EvidenceClassification:
    return rnochmod.EvidenceClassification(True, success, reason)


def classify_latency(
    request: rnochmod.RequestObservation,
    contract: rnochmod.ContractTerms,
    policy: rnochmod.WindowPolicy,
) -> rnochmod.EvidenceClassification:
    """
    Classify one request for buyer-experienced latency reliability.
    """
    if request.attribution_mismatch:
        return _excluded(EXCLUDED_ATTRIBUTION_MISMATCH)
    if request.is_canary and not policy.include_canaries_in_latency:
        return _excluded(EXCLUDED_CANARY_LATENCY)
    if request.status == rnogapro.OK:
        if request.latency_ms < 0:
            return _checker_error(INVALID_LATENCY)
        on_time = request.latency_ms <= contract.l_max_s * 1000
        reason = ELIGIBLE_ON_TIME if on_time else ELIGIBLE_TOO_SLOW
        return _eligible(on_time, reason)
    if request.status in rnogapro.SELLER_FAULT_STATUSES:
        return _eligible(False, ELIGIBLE_SELLER_FAILURE)
    if request.status in _NON_SELLER_EXCLUSIONS:
        return _excluded(_NON_SELLER_EXCLUSIONS[request.status])
    return _checker_error(UNKNOWN_REQUEST_STATUS)


def classify_quality(
    request: rnochmod.RequestObservation,
) -> rnochmod.EvidenceClassification:
    """
    Classify one request for deterministic canary quality.
    """
    if not request.is_canary:
        if request.canary_correct is not None:
            return _checker_error(UNEXPECTED_CANARY_GRADE)
        return _excluded(NOT_APPLICABLE)
    if request.attribution_mismatch:
        return _excluded(EXCLUDED_ATTRIBUTION_MISMATCH)
    if request.status in rnogapro.SELLER_FAULT_STATUSES:
        return _eligible(False, ELIGIBLE_SELLER_FAILURE)
    if request.status in _NON_SELLER_EXCLUSIONS:
        return _excluded(_NON_SELLER_EXCLUSIONS[request.status])
    if request.status != rnogapro.OK:
        return _checker_error(UNKNOWN_REQUEST_STATUS)
    if request.canary_correct is None:
        return _checker_error(MISSING_CANARY_GRADE)
    reason = ELIGIBLE_CORRECT if request.canary_correct else ELIGIBLE_INCORRECT
    return _eligible(request.canary_correct, reason)


def build_evidence(
    requests: Iterable[rnochmod.RequestObservation],
    contract: rnochmod.ContractTerms,
    policy: rnochmod.WindowPolicy,
) -> rnochmod.EvidenceSet:
    """
    Classify all observations in a contract window.
    """
    rows = tuple(
        rnochmod.ClassifiedRequest(
            request=request,
            latency=classify_latency(request, contract, policy),
            quality=classify_quality(request),
        )
        for request in requests
    )
    return rnochmod.EvidenceSet(rows)


def summarize_exclusions(evidence: rnochmod.EvidenceSet) -> Dict[str, int]:
    """
    Count meaningful exclusions separately for latency and quality.
    """
    counts = collections.Counter()
    for row in evidence.requests:
        if not row.latency.eligible:
            counts["latency:%s" % row.latency.reason] += 1
        if not row.quality.eligible and row.quality.reason != NOT_APPLICABLE:
            counts["quality:%s" % row.quality.reason] += 1
    return dict(sorted(counts.items()))
