"""
Claim, execute, grade, and persist due canary probes.

Import as:

import research.Noesis.prober.prober_runtime as rnoprrun
"""

import dataclasses
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import psycopg_pool

import research.Noesis.checker.checker_capability as rnochcap
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_probe_schedule as rnochprsc
import research.Noesis.checker.checker_store as rnochsto
import research.Noesis.gateway.gateway_faults as rnogafau
import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.gateway.gateway_request_log as rnogarelo
import research.Noesis.gateway.gateway_routing as rnogarou

_LOG = logging.getLogger(__name__)


class ProbeExecutionError(RuntimeError):
    """
    Signal an internal probe-state failure.
    """


@dataclasses.dataclass(frozen=True)
class ProbeCycleResult:
    """
    Represent all outcomes from one non-blocking scheduler pass.
    """

    executions: Tuple[rnochmod.ProbeExecution, ...]
    errors: Tuple[str, ...]


def build_probe_call(
    route: rnogarou.Route,
    canary: rnochmod.Canary,
    policy: rnochmod.ProbePolicy,
) -> rnogapro.ProviderCall:
    """
    Build a buyer-shaped provider request with no canary marker.
    """
    return rnogapro.ProviderCall(
        model=policy.model_id,
        messages=[{"role": "user", "content": canary.prompt}],
        max_tokens=policy.max_tokens,
        timeout_s=policy.timeout_multiplier * route.l_max,
        temperature=0.0,
    )


async def execute_probe(
    pool: psycopg_pool.AsyncConnectionPool,
    provider: rnogapro.Provider,
    faults: rnogafau.FaultRegistry,
    probe_id: int,
    contract_id: int,
    canary: rnochmod.Canary,
    policy: rnochmod.ProbePolicy,
    *,
    measurer: rnochcap.CapabilityMeasurer = rnochcap.DEFAULT_MEASURER,
) -> rnochmod.ProbeExecution:
    """
    Execute one claimed slot through the normal gateway serving path.
    """
    started = await rnochsto.mark_probe_running(pool, probe_id)
    if not started:
        raise ProbeExecutionError(
            "probe %s was not in SCHEDULED state" % probe_id
        )
    try:
        route = await rnogarou.resolve_internal_route(pool, contract_id)
        call = build_probe_call(route, canary, policy)
        fault = faults.get(route.seller_id)
        result, injected = await rnogafau.call_with_fault(
            provider, route.provider_slug, call, fault
        )
        if result.status == rnogapro.OK:
            grade = measurer.measure(canary, result.text)
            correct = grade.correct
            grader_reason = grade.reason
        else:
            correct = None
            grader_reason = "PROVIDER_%s" % result.status.upper()
        request_id = await rnogarelo.log_request(
            pool,
            route,
            call,
            result,
            is_canary=True,
            canary_id=canary.canary_id,
            canary_correct=correct,
            grader_version=measurer.version,
            fault_injected=injected,
        )
        completed = await rnochsto.complete_probe(
            pool, probe_id, request_id, measurer.version
        )
        if not completed:
            raise ProbeExecutionError(
                "probe %s was no longer RUNNING" % probe_id
            )
    except Exception as error:
        try:
            await rnochsto.fail_probe(pool, probe_id, str(error))
        except Exception:
            _LOG.exception("Could not persist failure for probe %s", probe_id)
        raise
    return rnochmod.ProbeExecution(
        probe_id=probe_id,
        request_id=request_id,
        provider_status=result.status,
        correct=correct,
        grader_reason=grader_reason,
    )


async def run_due_probes(
    pool: psycopg_pool.AsyncConnectionPool,
    provider: rnogapro.Provider,
    faults: rnogafau.FaultRegistry,
    policy: rnochmod.ProbePolicy,
    schedule_secret: str,
    now: datetime,
) -> ProbeCycleResult:
    """
    Run one scheduler pass without sleeping or polling.

    Each due slot is first claimed with a database uniqueness constraint.
    Internal failure of one probe is recorded and does not stop later probes.
    """
    contracts = await rnochsto.list_active_contracts(pool)
    canaries = await rnochsto.load_active_canaries(pool, policy.bank_version)
    canaries_by_id: Dict[str, rnochmod.Canary] = {
        canary.canary_id: canary for canary in canaries
    }
    executions: List[rnochmod.ProbeExecution] = []
    errors = []
    for contract in contracts:
        if contract.window_start is None:
            continue
        try:
            schedule = rnochprsc.build_probe_schedule(
                contract, canaries, schedule_secret
            )
            claimed_slots = await rnochsto.list_claimed_probe_slots(
                pool, contract.terms.contract_id
            )
            due = rnochprsc.get_due_probes(schedule, now, claimed_slots)
        except Exception as error:
            message = "contract %s: %s" % (
                contract.terms.contract_id,
                error,
            )
            _LOG.exception("Could not schedule probes for %s", message)
            errors.append(message)
            continue
        for item in due:
            probe_id = await rnochsto.claim_probe_slot(
                pool,
                item.contract_id,
                item.probe_slot,
                item.canary_id,
                scheduled_at=item.due_at,
            )
            if probe_id is None:
                continue
            try:
                execution = await execute_probe(
                    pool,
                    provider,
                    faults,
                    probe_id,
                    item.contract_id,
                    canaries_by_id[item.canary_id],
                    policy,
                )
                executions.append(execution)
            except Exception as error:
                message = "probe %s: %s" % (probe_id, error)
                _LOG.exception("Could not execute %s", message)
                errors.append(message)
    return ProbeCycleResult(tuple(executions), tuple(errors))
