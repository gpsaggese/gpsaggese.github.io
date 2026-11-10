"""
Decide when a contract window has finished collecting seller evidence.

Import as:

import research.Noesis.checker.checker_windows as rnochwin
"""

from datetime import datetime, timedelta

import research.Noesis.checker.checker_models as rnochmod


class WindowReadinessError(ValueError):
    """
    Signal contract or clock input that cannot support window evaluation.
    """


def count_window_evidence(
    evidence: rnochmod.EvidenceSet,
) -> rnochmod.WindowEvidenceCounts:
    """
    Count usable buyer-latency and canary-quality observations.
    """
    eligible_buyer_requests = sum(
        not row.request.is_canary and row.latency.eligible
        for row in evidence.requests
    )
    eligible_canaries = sum(
        row.request.is_canary and row.quality.eligible
        for row in evidence.requests
    )
    return rnochmod.WindowEvidenceCounts(
        eligible_buyer_requests,
        eligible_canaries,
    )


def assess_window_readiness(
    contract: rnochmod.ContractWindow,
    evidence: rnochmod.EvidenceSet,
    now: datetime,
) -> rnochmod.WindowReadiness:
    """
    Return whether collection targets or the wall-clock deadline are reached.
    """
    if contract.state != "ACTIVE":
        raise WindowReadinessError("contract must be ACTIVE")
    if contract.window_start is None:
        raise WindowReadinessError("contract window_start is required")
    if contract.window_start.utcoffset() is None:
        raise WindowReadinessError("contract window_start must be timezone-aware")
    if now.utcoffset() is None:
        raise WindowReadinessError("now must be timezone-aware")
    if contract.target_requests < 1:
        raise WindowReadinessError("target_requests must be positive")
    if contract.target_canaries < 1:
        raise WindowReadinessError("target_canaries must be positive")
    if contract.maximum_seconds < 1:
        raise WindowReadinessError("maximum_seconds must be positive")
    counts = count_window_evidence(evidence)
    targets_reached = (
        counts.eligible_buyer_requests >= contract.target_requests
        and counts.eligible_canaries >= contract.target_canaries
    )
    deadline = contract.window_start + timedelta(seconds=contract.maximum_seconds)
    if targets_reached:
        reason = rnochmod.WindowCloseReason.TARGETS_REACHED
    elif now >= deadline:
        reason = rnochmod.WindowCloseReason.DEADLINE_REACHED
    else:
        reason = rnochmod.WindowCloseReason.NOT_READY
    return rnochmod.WindowReadiness(
        ready=reason != rnochmod.WindowCloseReason.NOT_READY,
        reason=reason,
        counts=counts,
        deadline=deadline,
    )
