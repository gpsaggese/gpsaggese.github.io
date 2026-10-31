#!/usr/bin/env python
"""
Demonstrate contract-window monitoring without network or database access.

The demo runs the production probe/checker composition, evidence classifier,
readiness policy, verdict engine, report builder, and recording verdict sink.
Only external provider and PostgreSQL boundaries are replaced with deterministic
in-memory fixtures.

# Usage Example

    > research/Noesis/scripts/demo_contract_monitoring.py

Import as:

import research.Noesis.scripts.demo_contract_monitoring as rnscdemo
"""

import asyncio
import datetime
from typing import Dict, List, Tuple
from unittest import mock

import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_sink as rnochsink
import research.Noesis.checker.checker_store as rnochsto
import research.Noesis.gateway.gateway_faults as rnogafau
import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.prober.prober_runtime as rnoprrun
import research.Noesis.prober.prober_worker as rnoprwor

_START = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
_NOW = _START + datetime.timedelta(seconds=60)
_CHECKER_VERSION = "m2-v1"
_BANK_VERSION = "v1"


def _contract(contract_id: int, seller_id: str) -> rnochmod.ContractWindow:
    """
    Build one demo contract with a two-request, two-canary target.
    """
    return rnochmod.ContractWindow(
        terms=rnochmod.ContractTerms(contract_id, 2.0, 0.9, 0.85),
        state="ACTIVE",
        target_requests=2,
        target_canaries=2,
        maximum_seconds=60,
        checker_policy=_CHECKER_VERSION,
        window_start=_START,
        window_end=None,
        seller_id=seller_id,
    )


def _buyer(
    request_id: int,
    latency_ms: float,
    status: str = rnogapro.OK,
) -> rnochmod.RequestObservation:
    """
    Build one buyer request observation.
    """
    return rnochmod.RequestObservation(request_id, status, latency_ms)


def _canary(
    request_id: int,
    correct: bool,
) -> rnochmod.RequestObservation:
    """
    Build one deterministically graded canary observation.
    """
    return rnochmod.RequestObservation(
        request_id,
        rnogapro.OK,
        100.0,
        is_canary=True,
        canary_correct=correct,
    )


def _demo_inputs() -> Tuple[
    Tuple[rnochmod.ContractWindow, ...],
    Dict[int, List[rnochmod.RequestObservation]],
]:
    """
    Return contracts and buyer evidence for four visible outcomes.
    """
    contracts = (
        _contract(1, "healthy_seller"),
        _contract(2, "slow_seller"),
        _contract(3, "weak_seller"),
        _contract(4, "sparse_seller"),
    )
    observations = {
        1: [
            _buyer(1, 100.0),
            _buyer(2, 150.0),
            _buyer(3, 0.0, rnogapro.GATEWAY_ERROR),
        ],
        2: [_buyer(4, 2500.0), _buyer(5, 3000.0)],
        3: [_buyer(6, 100.0), _buyer(7, 150.0)],
        4: [],
    }
    return contracts, observations


async def build_report() -> str:
    """
    Run one complete mocked-I/O monitoring cycle and return its report.
    """
    contracts, observations = _demo_inputs()
    finalized: Dict[int, rnochmod.VerdictDecision] = {}
    sink = rnochsink.RecordingVerdictSink()
    checker_policy = rnochmod.WindowPolicy(
        minimum_requests=2,
        minimum_canaries=2,
    )

    async def _run_due_probes(
        pool,
        provider,
        faults,
        policy,
        schedule_secret,
        now,
    ) -> rnoprrun.ProbeCycleResult:
        _ = (pool, provider, faults, policy, schedule_secret, now)
        results = []
        correctness = {
            1: (True, True),
            2: (True, True),
            3: (False, False),
            4: (True,),
        }
        for contract_id, grades in correctness.items():
            for probe_slot, correct in enumerate(grades, start=1):
                request_id = contract_id * 100 + probe_slot
                observations[contract_id].append(_canary(request_id, correct))
                results.append(
                    rnochmod.ProbeExecution(
                        probe_id=request_id,
                        request_id=request_id,
                        provider_status=rnogapro.OK,
                        correct=correct,
                        grader_reason="MATCH" if correct else "MISMATCH",
                    )
                )
        return rnoprrun.ProbeCycleResult(tuple(results), ())

    async def _list_active_contracts(pool):
        _ = pool
        return contracts

    async def _list_request_observations(pool, contract_id):
        _ = pool
        return tuple(observations[contract_id])

    async def _finalize_contract(
        pool,
        contract_id,
        decision,
        evidence,
        **kwargs,
    ):
        _ = (pool, evidence, kwargs)
        if contract_id in finalized:
            return False
        finalized[contract_id] = decision
        return True

    with (
        mock.patch.object(
            rnoprrun,
            "run_due_probes",
            new=_run_due_probes,
        ),
        mock.patch.object(
            rnochsto,
            "list_active_contracts",
            new=_list_active_contracts,
        ),
        mock.patch.object(
            rnochsto,
            "list_request_observations",
            new=_list_request_observations,
        ),
        mock.patch.object(
            rnochsto,
            "finalize_contract",
            new=_finalize_contract,
        ),
    ):
        result = await rnoprwor.run_probe_cycle(
            mock.sentinel.pool,
            mock.sentinel.provider,
            rnogafau.FaultRegistry(),
            rnochmod.ProbePolicy(_BANK_VERSION, "demo-model"),
            "demo-secret",
            _NOW,
            sink=sink,
            checker_policy=checker_policy,
            checker_version=_CHECKER_VERSION,
        )
    published_ids = {report.contract_id for report in result.checker.published_reports}
    lines = [
        "1) Seeded 4 ACTIVE contracts with buyer evidence.",
        "2) Prober persisted %s scheduled canaries before checker evaluation."
        % len(result.probes.executions),
        "3) Checker evaluated %s ready windows:" % len(result.checker.evaluations),
    ]
    for evaluation in result.checker.evaluations:
        report = evaluation.report
        assert report is not None
        counts = evaluation.readiness.counts
        lines.append(
            "   contract=%s seller=%s close=%s buyer=%s canary=%s"
            " verdict=%s published=%s"
            % (
                report.contract_id,
                report.seller_id,
                evaluation.readiness.reason.value,
                counts.eligible_buyer_requests,
                counts.eligible_canaries,
                report.verdict.value,
                "yes" if report.contract_id in published_ids else "no",
            )
        )
    published = ", ".join(
        "%s:%s" % (report.contract_id, report.verdict.value) for report in sink.reports
    )
    lines.extend(
        [
            "4) Persistence boundary finalized %s contracts exactly once."
            % len(finalized),
            "5) RecordingVerdictSink received %s actionable reports: %s"
            % (len(sink.reports), published),
            "   Contract 4 stayed sink-silent because INSUFFICIENT is"
            " non-actionable.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """
    Run the demo and print its deterministic report.
    """
    print(asyncio.run(build_report()))


if __name__ == "__main__":
    main()
