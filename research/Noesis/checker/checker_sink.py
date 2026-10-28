"""
Define the boundary for publishing actionable checker verdicts.

Import as:

import research.Noesis.checker.checker_sink as rnochsink
"""

from datetime import datetime
from typing import List, Protocol, Tuple

import research.Noesis.checker.checker_models as rnochmod


class VerdictSink(Protocol):
    """
    Receive one immutable checker report without exposing market internals.
    """

    async def publish(self, report: rnochmod.VerdictReport) -> None:
        """
        Publish one actionable contract report.

        :param report: immutable checker output for one finalized contract
        """
        ...


def build_verdict_report(
    contract_id: int,
    seller_id: str,
    decision: rnochmod.VerdictDecision,
    window_start: datetime,
    window_end: datetime,
    *,
    checker_version: str,
    bank_version: str,
) -> rnochmod.VerdictReport:
    """
    Convert an internal verdict decision into the stable sink payload.
    """
    return rnochmod.VerdictReport(
        contract_id=contract_id,
        seller_id=seller_id,
        verdict=decision.verdict,
        disposition=decision.disposition,
        latency=decision.latency,
        quality=decision.quality,
        reason_codes=decision.reason_codes,
        excluded_counts=dict(decision.excluded_counts),
        window_start=window_start,
        window_end=window_end,
        checker_version=checker_version,
        bank_version=bank_version,
    )


def is_actionable_report(report: rnochmod.VerdictReport) -> bool:
    """
    Return whether the market may safely consume this seller result.
    """
    return (
        report.disposition == rnochmod.CheckerDisposition.VALID
        and report.verdict
        in (
            rnochmod.Verdict.PASSED,
            rnochmod.Verdict.FAILED,
        )
    )


class RecordingVerdictSink:
    """
    Record reports in memory for tests and mocked market demonstrations.
    """

    def __init__(self) -> None:
        self._reports: List[rnochmod.VerdictReport] = []

    @property
    def reports(self) -> Tuple[rnochmod.VerdictReport, ...]:
        """
        Return an immutable view of reports in publication order.
        """
        return tuple(self._reports)

    async def publish(self, report: rnochmod.VerdictReport) -> None:
        """
        Record one report.

        :param report: immutable checker output for one finalized contract
        """
        self._reports.append(report)
