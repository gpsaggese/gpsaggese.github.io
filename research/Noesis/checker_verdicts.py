"""
Compute explainable contract verdicts from classified evidence.

Import as:

import research.Noesis.checker_verdicts as rnochver
"""

from typing import Iterable

import research.Noesis.checker_evidence as rnochevi
import research.Noesis.checker_models as rnochmod
import research.Noesis.checker_statistics as rnochsta

COMPLIANT = "COMPLIANT"
LATENCY_BELOW_PROMISE = "LATENCY_BELOW_PROMISE"
QUALITY_BELOW_PROMISE = "QUALITY_BELOW_PROMISE"
INSUFFICIENT_LATENCY_EVIDENCE = "INSUFFICIENT_LATENCY_EVIDENCE"
INSUFFICIENT_QUALITY_EVIDENCE = "INSUFFICIENT_QUALITY_EVIDENCE"
CHECKER_UNHEALTHY = "CHECKER_UNHEALTHY"


def _assess_metric(
    classifications: Iterable[rnochmod.EvidenceClassification],
    promised_rate: float,
    minimum_evidence: int,
    policy: rnochmod.WindowPolicy,
) -> rnochmod.MetricAssessment:
    eligible = tuple(item for item in classifications if item.eligible)
    total = len(eligible)
    successes = sum(item.success is True for item in eligible)
    rate = successes / total if total else None
    upper_bound = rnochsta.upper_confidence_bound(
        successes,
        total,
        confidence=policy.confidence,
        exact_below=policy.exact_bound_below,
    )
    sufficient = total >= minimum_evidence
    conclusive_failure = sufficient and rnochsta.is_conclusive_failure(
        upper_bound, promised_rate
    )
    return rnochmod.MetricAssessment(
        successes,
        total,
        rate,
        upper_bound,
        promised_rate,
        sufficient,
        conclusive_failure,
    )


def assess_latency(
    evidence: rnochmod.EvidenceSet,
    contract: rnochmod.ContractTerms,
    policy: rnochmod.WindowPolicy,
) -> rnochmod.MetricAssessment:
    """
    Assess the contract's latency reliability promise.
    """
    return _assess_metric(
        (row.latency for row in evidence.requests),
        contract.r_min,
        policy.minimum_requests,
        policy,
    )


def assess_quality(
    evidence: rnochmod.EvidenceSet,
    contract: rnochmod.ContractTerms,
    policy: rnochmod.WindowPolicy,
) -> rnochmod.MetricAssessment:
    """
    Assess the contract's deterministic canary-quality promise.
    """
    return _assess_metric(
        (row.quality for row in evidence.requests),
        contract.q_min,
        policy.minimum_canaries,
        policy,
    )


def decide_verdict(
    evidence: rnochmod.EvidenceSet,
    contract: rnochmod.ContractTerms,
    policy: rnochmod.WindowPolicy,
) -> rnochmod.VerdictDecision:
    """
    Produce one seller verdict from classified window evidence.
    """
    latency = assess_latency(evidence, contract, policy)
    quality = assess_quality(evidence, contract, policy)
    excluded = rnochevi.summarize_exclusions(evidence)
    if not evidence.checker_healthy:
        return rnochmod.VerdictDecision(
            rnochmod.Verdict.INSUFFICIENT,
            rnochmod.CheckerDisposition.CHECKER_UNHEALTHY,
            latency,
            quality,
            (CHECKER_UNHEALTHY,),
            excluded,
        )
    failures = []
    if latency.conclusive_failure:
        failures.append(LATENCY_BELOW_PROMISE)
    if quality.conclusive_failure:
        failures.append(QUALITY_BELOW_PROMISE)
    if failures:
        return rnochmod.VerdictDecision(
            rnochmod.Verdict.FAILED,
            rnochmod.CheckerDisposition.VALID,
            latency,
            quality,
            tuple(failures),
            excluded,
        )
    insufficient = []
    if not latency.sufficient:
        insufficient.append(INSUFFICIENT_LATENCY_EVIDENCE)
    if not quality.sufficient:
        insufficient.append(INSUFFICIENT_QUALITY_EVIDENCE)
    if insufficient:
        return rnochmod.VerdictDecision(
            rnochmod.Verdict.INSUFFICIENT,
            rnochmod.CheckerDisposition.VALID,
            latency,
            quality,
            tuple(insufficient),
            excluded,
        )
    return rnochmod.VerdictDecision(
        rnochmod.Verdict.PASSED,
        rnochmod.CheckerDisposition.VALID,
        latency,
        quality,
        (COMPLIANT,),
        excluded,
    )


def explain_verdict(decision: rnochmod.VerdictDecision) -> str:
    """
    Return a compact human-readable summary for logs and a future CLI.
    """
    return "%s: %s; latency=%s/%s, quality=%s/%s, disposition=%s" % (
        decision.verdict.value,
        ", ".join(decision.reason_codes),
        decision.latency.successes,
        decision.latency.total,
        decision.quality.successes,
        decision.quality.total,
        decision.disposition.value,
    )
