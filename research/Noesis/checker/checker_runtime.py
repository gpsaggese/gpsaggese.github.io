"""
Evaluate ready contract windows and publish actionable checker reports.

Import as:

import research.Noesis.checker.checker_runtime as rnochrun
"""

import dataclasses
import logging
from datetime import datetime
from typing import List, Tuple

import psycopg_pool

import research.Noesis.checker.checker_evidence as rnochevi
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_sink as rnochsink
import research.Noesis.checker.checker_store as rnochsto
import research.Noesis.checker.checker_verdicts as rnochver
import research.Noesis.checker.checker_windows as rnochwin

_LOG = logging.getLogger(__name__)


class CheckerRuntimeError(RuntimeError):
    """
    Signal a contract or runtime configuration that cannot be evaluated.
    """


@dataclasses.dataclass(frozen=True)
class CheckerPassResult:
    """
    Summarize one non-blocking pass over the active contracts.
    """

    evaluations: Tuple[rnochmod.ContractEvaluation, ...]
    published_reports: Tuple[rnochmod.VerdictReport, ...]
    errors: Tuple[str, ...]


def _validate_runtime_configuration(
    contract: rnochmod.ContractWindow,
    now: datetime,
    checker_version: str,
    bank_version: str,
) -> None:
    """
    Reject ambiguous clocks and unsupported version combinations.
    """
    if now.utcoffset() is None:
        raise CheckerRuntimeError("now must be timezone-aware")
    if not checker_version:
        raise CheckerRuntimeError("checker_version must not be empty")
    if not bank_version:
        raise CheckerRuntimeError("bank_version must not be empty")
    if contract.checker_policy != checker_version:
        raise CheckerRuntimeError(
            "contract checker_policy %r does not match runtime version %r"
            % (contract.checker_policy, checker_version)
        )


async def evaluate_contract(
    pool: psycopg_pool.AsyncConnectionPool,
    contract: rnochmod.ContractWindow,
    policy: rnochmod.WindowPolicy,
    now: datetime,
    *,
    checker_version: str,
    bank_version: str,
) -> rnochmod.ContractEvaluation:
    """
    Evaluate and atomically finalize one contract when its window is ready.

    The database decides whether this worker won finalization. A losing worker
    receives no report and therefore cannot publish a duplicate outcome.
    """
    _validate_runtime_configuration(
        contract,
        now,
        checker_version,
        bank_version,
    )
    contract_id = contract.terms.contract_id
    observations = await rnochsto.list_request_observations(pool, contract_id)
    evidence = rnochevi.build_evidence(
        observations,
        contract.terms,
        policy,
    )
    readiness = rnochwin.assess_window_readiness(contract, evidence, now)
    if not readiness.ready:
        return rnochmod.ContractEvaluation(
            contract_id,
            readiness,
            False,
            None,
        )
    decision = rnochver.decide_verdict(
        evidence,
        contract.terms,
        policy,
    )
    finalized = await rnochsto.finalize_contract(
        pool,
        contract_id,
        decision,
        evidence,
        checker_version=checker_version,
        bank_version=bank_version,
        window_end=now,
    )
    if not finalized:
        return rnochmod.ContractEvaluation(
            contract_id,
            readiness,
            False,
            None,
        )
    report = rnochsink.build_verdict_report(
        contract_id,
        contract.seller_id,
        decision,
        contract.window_start,
        now,
        checker_version=checker_version,
        bank_version=bank_version,
    )
    return rnochmod.ContractEvaluation(
        contract_id,
        readiness,
        True,
        report,
    )


async def run_checker_pass(
    pool: psycopg_pool.AsyncConnectionPool,
    sink: rnochsink.VerdictSink,
    policy: rnochmod.WindowPolicy,
    now: datetime,
    *,
    checker_version: str,
    bank_version: str,
) -> CheckerPassResult:
    """
    Evaluate every active contract once without sleeping or polling.

    Evaluation and sink failures are isolated per contract. Failure to list the
    active contracts remains a pass-level error for the worker to retry.
    """
    contracts = await rnochsto.list_active_contracts(pool)
    evaluations: List[rnochmod.ContractEvaluation] = []
    published_reports: List[rnochmod.VerdictReport] = []
    errors = []
    for contract in contracts:
        contract_id = contract.terms.contract_id
        try:
            evaluation = await evaluate_contract(
                pool,
                contract,
                policy,
                now,
                checker_version=checker_version,
                bank_version=bank_version,
            )
        except Exception as error:
            message = "contract %s evaluation: %s" % (contract_id, error)
            _LOG.exception("Could not evaluate %s", message)
            errors.append(message)
            continue
        evaluations.append(evaluation)
        report = evaluation.report
        if report is None or not rnochsink.is_actionable_report(report):
            continue
        try:
            await sink.publish(report)
        except Exception as error:
            message = "contract %s sink: %s" % (contract_id, error)
            _LOG.exception("Could not publish %s", message)
            errors.append(message)
            continue
        published_reports.append(report)
    return CheckerPassResult(
        tuple(evaluations),
        tuple(published_reports),
        tuple(errors),
    )
