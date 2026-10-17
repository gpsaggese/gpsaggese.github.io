"""
Persist checker orchestration state and final verdict evidence in PostgreSQL.

Import as:

import research.Noesis.checker.checker_store as rnochsto
"""

import json
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence, Tuple

import psycopg_pool

import research.Noesis.checker.checker_models as rnochmod


def _contract_window(row: Tuple[Any, ...]) -> rnochmod.ContractWindow:
    return rnochmod.ContractWindow(
        terms=rnochmod.ContractTerms(
            contract_id=row[0],
            l_max_s=row[7],
            r_min=row[8],
            q_min=row[9],
        ),
        state=row[1],
        target_requests=row[2],
        target_canaries=row[3],
        maximum_seconds=row[4],
        checker_policy=row[5],
        window_start=row[6],
        window_end=row[10],
    )


async def list_active_contracts(
    pool: psycopg_pool.AsyncConnectionPool,
) -> Tuple[rnochmod.ContractWindow, ...]:
    """
    Return every active contract in stable id order.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT contract_id, state, n_tasks, canaries_target,"
            " window_max_seconds, checker_policy, window_start, l_max, r_min,"
            " q_min, window_end FROM contracts WHERE state = 'ACTIVE'"
            " ORDER BY contract_id"
        )
        rows = await cursor.fetchall()
    return tuple(_contract_window(row) for row in rows)


async def get_contract(
    pool: psycopg_pool.AsyncConnectionPool, contract_id: int
) -> Optional[rnochmod.ContractWindow]:
    """
    Return one contract snapshot, or None when it does not exist.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT contract_id, state, n_tasks, canaries_target,"
            " window_max_seconds, checker_policy, window_start, l_max, r_min,"
            " q_min, window_end FROM contracts WHERE contract_id = %s",
            (contract_id,),
        )
        row = await cursor.fetchone()
    return _contract_window(row) if row is not None else None


async def load_active_canaries(
    pool: psycopg_pool.AsyncConnectionPool, bank_version: str
) -> Tuple[rnochmod.Canary, ...]:
    """
    Load enabled canaries from one immutable bank version.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT canary_id, bank_version, family, prompt, answer,"
            " normalizer, generator_seed FROM canaries"
            " WHERE bank_version = %s AND status = 'ACTIVE'"
            " ORDER BY canary_id",
            (bank_version,),
        )
        rows = await cursor.fetchall()
    return tuple(
        rnochmod.Canary(
            canary_id=row[0],
            bank_version=row[1],
            family=row[2],
            prompt=row[3],
            answer=row[4],
            normalizer=row[5],
            generator_seed=row[6],
        )
        for row in rows
    )


async def register_canary_bank(
    pool: psycopg_pool.AsyncConnectionPool,
    canaries: Sequence[rnochmod.Canary],
) -> None:
    """
    Insert new candidates and reject conflicting definitions of existing IDs.
    """
    if not canaries:
        raise ValueError("canary bank must not be empty")
    canary_ids = [canary.canary_id for canary in canaries]
    if len(canary_ids) != len(set(canary_ids)):
        raise ValueError("canary bank contains duplicate ids")
    bank_versions = {canary.bank_version for canary in canaries}
    if len(bank_versions) != 1:
        raise ValueError("canary bank must contain exactly one version")
    bank_version = next(iter(bank_versions))
    async with pool.connection() as conn, conn.transaction():
        inserted_ids = []
        for canary in canaries:
            cursor = await conn.execute(
                "INSERT INTO canaries ("
                " canary_id, prompt, answer, normalizer, bank_version, family,"
                " generator_seed, status, calibrated"
                ") VALUES (%s, %s, %s, %s, %s, %s, %s, 'DRAFT', FALSE)"
                " ON CONFLICT (canary_id) DO NOTHING RETURNING canary_id",
                (
                    canary.canary_id,
                    canary.prompt,
                    canary.answer,
                    canary.normalizer,
                    canary.bank_version,
                    canary.family,
                    canary.generator_seed,
                ),
            )
            row = await cursor.fetchone()
            if row is not None:
                inserted_ids.append(row[0])
        if inserted_ids:
            cursor = await conn.execute(
                "SELECT run_id FROM canary_calibration_runs"
                " WHERE bank_version = %s LIMIT 1",
                (bank_version,),
            )
            if await cursor.fetchone() is not None:
                raise ValueError(
                    "cannot add canaries to calibrated bank version %r"
                    % bank_version
                )
        cursor = await conn.execute(
            "SELECT canary_id, bank_version, family, prompt, answer,"
            " normalizer, generator_seed FROM canaries"
            " WHERE canary_id = ANY(%s) ORDER BY canary_id",
            (canary_ids,),
        )
        stored = {
            row[0]: rnochmod.Canary(
                canary_id=row[0],
                bank_version=row[1],
                family=row[2],
                prompt=row[3],
                answer=row[4],
                normalizer=row[5],
                generator_seed=row[6],
            )
            for row in await cursor.fetchall()
        }
        for canary in canaries:
            if stored.get(canary.canary_id) != canary:
                raise ValueError(
                    "canary id %r has a conflicting stored definition"
                    % canary.canary_id
                )


async def start_calibration_run(
    pool: psycopg_pool.AsyncConnectionPool,
    bank_version: str,
    targets: Sequence[rnochmod.CalibrationTarget],
    repetitions: int,
    configuration: Mapping[str, object],
) -> int:
    """
    Create a RUNNING calibration record and return its identifier.
    """
    healthy_models = sorted(
        {
            target.model_id
            for target in targets
            if target.role == rnochmod.CalibrationRole.HEALTHY
        }
    )
    weak_targets = [
        target
        for target in targets
        if target.role == rnochmod.CalibrationRole.WEAK
    ]
    if not healthy_models:
        raise ValueError("calibration needs at least one healthy target")
    if len(weak_targets) != 1:
        raise ValueError("calibration needs exactly one weak target")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    stored_configuration = dict(configuration)
    stored_configuration["targets"] = [
        {
            "role": target.role.value,
            "provider_slug": target.provider_slug,
            "model_id": target.model_id,
        }
        for target in targets
    ]
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO canary_calibration_runs ("
            " bank_version, healthy_model, weak_model, repetitions,"
            " configuration"
            ") VALUES (%s, %s, %s, %s, %s::jsonb) RETURNING run_id",
            (
                bank_version,
                ",".join(healthy_models),
                weak_targets[0].model_id,
                repetitions,
                json.dumps(stored_configuration),
            ),
        )
        (run_id,) = await cursor.fetchone()
    return run_id


async def record_calibration_observations(
    pool: psycopg_pool.AsyncConnectionPool,
    run_id: int,
    observations: Sequence[rnochmod.CalibrationObservation],
) -> None:
    """
    Persist every raw response and deterministic grade for one run.
    """
    async with pool.connection() as conn, conn.transaction():
        cursor = await conn.execute(
            "SELECT state FROM canary_calibration_runs"
            " WHERE run_id = %s FOR UPDATE",
            (run_id,),
        )
        run = await cursor.fetchone()
        if run is None:
            raise ValueError("calibration run %s does not exist" % run_id)
        if run[0] != "RUNNING":
            raise ValueError("calibration run %s is not RUNNING" % run_id)
        for observation in observations:
            await conn.execute(
                "INSERT INTO canary_calibration_results ("
                " run_id, canary_id, target_role, provider_slug, model_id,"
                " attempt, provider_status, response, correct, latency_ms,"
                " cost_usd"
                ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    run_id,
                    observation.canary_id,
                    observation.target.role.value,
                    observation.target.provider_slug,
                    observation.target.model_id,
                    observation.attempt,
                    observation.provider_status,
                    observation.response,
                    observation.correct,
                    observation.latency_ms,
                    observation.cost_usd,
                ),
            )


async def complete_calibration_run(
    pool: psycopg_pool.AsyncConnectionPool,
    run_id: int,
    decisions: Sequence[rnochmod.CanaryCalibration],
    summary: Mapping[str, object],
) -> bool:
    """
    Atomically persist decisions and mark one RUNNING run COMPLETED.
    """
    decision_ids = [decision.canary_id for decision in decisions]
    if not decision_ids:
        raise ValueError("calibration decisions must not be empty")
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("calibration decisions contain duplicate ids")
    async with pool.connection() as conn, conn.transaction():
        cursor = await conn.execute(
            "SELECT state, bank_version FROM canary_calibration_runs"
            " WHERE run_id = %s FOR UPDATE",
            (run_id,),
        )
        run = await cursor.fetchone()
        if run is None:
            raise ValueError("calibration run %s does not exist" % run_id)
        if run[0] != "RUNNING":
            return False
        bank_version = run[1]
        for decision in decisions:
            metadata = {
                "last_calibration_run_id": run_id,
                "calibration_reason": decision.reason,
            }
            cursor = await conn.execute(
                "UPDATE canaries SET calibrated = %s, healthy_acc = %s,"
                " weak_acc = %s,"
                " status = CASE WHEN %s THEN 'CALIBRATED' ELSE 'DRAFT' END,"
                " metadata = metadata || %s::jsonb"
                " WHERE canary_id = %s AND bank_version = %s"
                " AND status IN ('DRAFT', 'CALIBRATED')"
                " RETURNING canary_id",
                (
                    decision.accepted,
                    decision.healthy_accuracy,
                    decision.weak_accuracy,
                    decision.accepted,
                    json.dumps(metadata),
                    decision.canary_id,
                    bank_version,
                ),
            )
            if await cursor.fetchone() is None:
                raise ValueError(
                    "canary %r is missing, has another version, or is immutable"
                    % decision.canary_id
                )
        await conn.execute(
            "UPDATE canary_calibration_runs SET state = 'COMPLETED',"
            " summary = %s::jsonb, completed_at = now() WHERE run_id = %s",
            (json.dumps(dict(summary)), run_id),
        )
    return True


async def fail_calibration_run(
    pool: psycopg_pool.AsyncConnectionPool,
    run_id: int,
    error: str,
) -> bool:
    """
    Mark a RUNNING calibration as failed without changing canary status.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "UPDATE canary_calibration_runs SET state = 'FAILED',"
            " summary = %s::jsonb, completed_at = now()"
            " WHERE run_id = %s AND state = 'RUNNING' RETURNING run_id",
            (json.dumps({"error": error}), run_id),
        )
        row = await cursor.fetchone()
    return row is not None


async def activate_calibrated_bank(
    pool: psycopg_pool.AsyncConnectionPool,
    run_id: int,
    bank_version: str,
    minimum_count: int,
) -> int:
    """
    Activate one completed run only when enough canaries survived.

    The run id ties activation to one immutable set of calibration decisions.
    Repeating the call for an already-active run is safe.
    """
    if minimum_count < 1:
        raise ValueError("minimum_count must be positive")
    async with pool.connection() as conn, conn.transaction():
        cursor = await conn.execute(
            "SELECT state, bank_version FROM canary_calibration_runs"
            " WHERE run_id = %s FOR UPDATE",
            (run_id,),
        )
        run = await cursor.fetchone()
        if run is None:
            raise ValueError("calibration run %s does not exist" % run_id)
        if run != ("COMPLETED", bank_version):
            raise ValueError(
                "calibration run %s is not a completed run for bank %r"
                % (run_id, bank_version)
            )
        cursor = await conn.execute(
            "SELECT count(*) FROM canaries WHERE bank_version = %s"
            " AND calibrated = TRUE AND status IN ('CALIBRATED', 'ACTIVE')"
            " AND (metadata->>'last_calibration_run_id')::BIGINT = %s",
            (bank_version, run_id),
        )
        (survivor_count,) = await cursor.fetchone()
        if survivor_count < minimum_count:
            raise ValueError(
                "bank %r has %s calibrated canaries; at least %s required"
                % (bank_version, survivor_count, minimum_count)
            )
        await conn.execute(
            "UPDATE canaries SET status = 'ACTIVE'"
            " WHERE bank_version = %s AND calibrated = TRUE"
            " AND status = 'CALIBRATED'"
            " AND (metadata->>'last_calibration_run_id')::BIGINT = %s",
            (bank_version, run_id),
        )
    return survivor_count


async def list_request_observations(
    pool: psycopg_pool.AsyncConnectionPool, contract_id: int
) -> Tuple[rnochmod.RequestObservation, ...]:
    """
    Load the request fields consumed by the evidence classifier.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT request_id, status, latency_ms, is_canary, canary_correct,"
            " attribution_mismatch FROM requests WHERE contract_id = %s"
            " ORDER BY request_id",
            (contract_id,),
        )
        rows = await cursor.fetchall()
    return tuple(
        rnochmod.RequestObservation(
            request_id=row[0],
            status=row[1],
            latency_ms=row[2],
            is_canary=row[3],
            canary_correct=row[4],
            attribution_mismatch=row[5],
        )
        for row in rows
    )


async def claim_probe_slot(
    pool: psycopg_pool.AsyncConnectionPool,
    contract_id: int,
    probe_slot: int,
    canary_id: str,
    *,
    scheduled_at: Optional[datetime] = None,
) -> Optional[int]:
    """
    Claim one contract slot, or return None if it is already claimed.
    """
    if scheduled_at is not None and scheduled_at.utcoffset() is None:
        raise ValueError("scheduled_at must be timezone-aware")
    columns = "contract_id, probe_slot, canary_id"
    placeholders = "%s, %s, %s"
    values = [contract_id, probe_slot, canary_id]
    if scheduled_at is not None:
        columns += ", scheduled_at"
        placeholders += ", %s"
        values.append(scheduled_at)
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO canary_probes (%s) VALUES (%s)"
            " ON CONFLICT (contract_id, probe_slot) DO NOTHING"
            " RETURNING probe_id" % (columns, placeholders),
            values,
        )
        row = await cursor.fetchone()
    return row[0] if row is not None else None


async def list_claimed_probe_slots(
    pool: psycopg_pool.AsyncConnectionPool,
    contract_id: int,
) -> Tuple[int, ...]:
    """
    Return every already-persisted probe slot for one contract.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT probe_slot FROM canary_probes"
            " WHERE contract_id = %s ORDER BY probe_slot",
            (contract_id,),
        )
        rows = await cursor.fetchall()
    return tuple(row[0] for row in rows)


async def mark_probe_running(
    pool: psycopg_pool.AsyncConnectionPool, probe_id: int
) -> bool:
    """
    Move a scheduled probe to RUNNING exactly once.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "UPDATE canary_probes SET state = 'RUNNING', started_at = now()"
            " WHERE probe_id = %s AND state = 'SCHEDULED' RETURNING probe_id",
            (probe_id,),
        )
        row = await cursor.fetchone()
    return row is not None


async def complete_probe(
    pool: psycopg_pool.AsyncConnectionPool,
    probe_id: int,
    request_id: int,
    grader_version: str,
) -> bool:
    """
    Attach the logged request and complete a running probe.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "UPDATE canary_probes SET state = 'COMPLETED', request_id = %s,"
            " grader_version = %s, completed_at = now()"
            " WHERE probe_id = %s AND state = 'RUNNING' RETURNING probe_id",
            (request_id, grader_version, probe_id),
        )
        row = await cursor.fetchone()
    return row is not None


async def fail_probe(
    pool: psycopg_pool.AsyncConnectionPool, probe_id: int, error: str
) -> bool:
    """
    Record an internal probe failure without blaming the seller.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "UPDATE canary_probes SET state = 'FAILED_INTERNAL', error = %s,"
            " completed_at = now() WHERE probe_id = %s"
            " AND state IN ('SCHEDULED', 'RUNNING') RETURNING probe_id",
            (error, probe_id),
        )
        row = await cursor.fetchone()
    return row is not None


async def finalize_contract(
    pool: psycopg_pool.AsyncConnectionPool,
    contract_id: int,
    decision: rnochmod.VerdictDecision,
    evidence: rnochmod.EvidenceSet,
    *,
    checker_version: str,
    bank_version: str,
    window_end: datetime,
) -> bool:
    """
    Atomically freeze evidence, write the verdict, and close a contract.
    """
    request_ids = [row.request.request_id for row in evidence.requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("evidence contains duplicate request ids")
    async with pool.connection() as conn, conn.transaction():
        cursor = await conn.execute(
            "SELECT state, window_start FROM contracts"
            " WHERE contract_id = %s FOR UPDATE",
            (contract_id,),
        )
        contract = await cursor.fetchone()
        if contract is None:
            raise ValueError("contract %s does not exist" % contract_id)
        if contract[0] not in ("ACTIVE", "CLOSED"):
            return False
        if request_ids:
            cursor = await conn.execute(
                "SELECT request_id FROM requests"
                " WHERE contract_id = %s AND request_id = ANY(%s)",
                (contract_id, request_ids),
            )
            stored_ids = {row[0] for row in await cursor.fetchall()}
            if stored_ids != set(request_ids):
                raise ValueError(
                    "evidence contains requests from another contract"
                )
        await conn.execute(
            "INSERT INTO contract_verdicts ("
            " contract_id, n, r_lat, r_lat_upper, n_canary, q, q_upper,"
            " verdict, reason, latency_successes, quality_successes,"
            " checker_version, bank_version, checker_disposition, reason_codes,"
            " excluded_counts, window_start, window_end"
            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,"
            " %s, %s, %s::jsonb, %s::jsonb, %s, %s)",
            (
                contract_id,
                decision.latency.total,
                decision.latency.rate,
                decision.latency.upper_bound,
                decision.quality.total,
                decision.quality.rate,
                decision.quality.upper_bound,
                decision.verdict.value,
                ",".join(decision.reason_codes),
                decision.latency.successes,
                decision.quality.successes,
                checker_version,
                bank_version,
                decision.disposition.value,
                json.dumps(decision.reason_codes),
                json.dumps(decision.excluded_counts),
                contract[1],
                window_end,
            ),
        )
        for row in evidence.requests:
            await conn.execute(
                "INSERT INTO verdict_evidence ("
                " contract_id, request_id, latency_eligible, latency_success,"
                " latency_reason, quality_eligible, quality_success,"
                " quality_reason) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    contract_id,
                    row.request.request_id,
                    row.latency.eligible,
                    row.latency.success,
                    row.latency.reason,
                    row.quality.eligible,
                    row.quality.success,
                    row.quality.reason,
                ),
            )
        await conn.execute(
            "UPDATE contracts SET state = %s, window_end = %s WHERE contract_id = %s",
            (decision.verdict.value, window_end, contract_id),
        )
    return True
