"""
Persist checker orchestration state and final verdict evidence in PostgreSQL.

Import as:

import research.Noesis.checker_store as rnochsto
"""

import json
from datetime import datetime
from typing import Any, Optional, Tuple

import psycopg_pool

import research.Noesis.checker_models as rnochmod


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
) -> Optional[int]:
    """
    Claim one contract slot, or return None if it is already claimed.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO canary_probes (contract_id, probe_slot, canary_id)"
            " VALUES (%s, %s, %s)"
            " ON CONFLICT (contract_id, probe_slot) DO NOTHING"
            " RETURNING probe_id",
            (contract_id, probe_slot, canary_id),
        )
        row = await cursor.fetchone()
    return row[0] if row is not None else None


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
                raise ValueError("evidence contains requests from another contract")
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
