"""
Persist standalone monitoring runs and observations in PostgreSQL.

Import as:

import research.Noesis.monitor.monitor_store as rnmonsto
"""

import json
from typing import Mapping

import psycopg_pool

import research.Noesis.monitor.monitor_models as rnmonmod


async def start_monitoring_run(
    pool: psycopg_pool.AsyncConnectionPool,
    target: rnmonmod.MonitoringTarget,
    policy: rnmonmod.MonitoringPolicy,
    grader_version: str,
) -> int:
    """
    Create a RUNNING monitoring record and return its identifier.
    """
    if not grader_version:
        raise ValueError("grader_version must not be empty")
    configuration = {
        "canary_count": policy.canary_count,
        "max_tokens": policy.max_tokens,
        "timeout_s": policy.timeout_s,
        "grader_version": grader_version,
        "selection_scheme": "hmac-family-balanced-v1",
    }
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO monitoring_runs ("
            " bank_version, provider_slug, model_id, configuration"
            ") VALUES (%s, %s, %s, %s::jsonb) RETURNING run_id",
            (
                policy.bank_version,
                target.provider_slug,
                target.model_id,
                json.dumps(configuration),
            ),
        )
        (run_id,) = await cursor.fetchone()
    return run_id


async def record_monitoring_observation(
    pool: psycopg_pool.AsyncConnectionPool,
    run_id: int,
    observation: rnmonmod.MonitoringObservation,
) -> None:
    """
    Persist one observation while its monitoring run is RUNNING.

    Each call commits independently so earlier observations survive a later
    process or database failure.
    """
    async with pool.connection() as conn, conn.transaction():
        cursor = await conn.execute(
            "SELECT provider_slug, model_id, state FROM monitoring_runs"
            " WHERE run_id = %s FOR UPDATE",
            (run_id,),
        )
        run = await cursor.fetchone()
        if run is None:
            raise ValueError("monitoring run %s does not exist" % run_id)
        if run[2] != "RUNNING":
            raise ValueError("monitoring run %s is not RUNNING" % run_id)
        expected_target = (run[0], run[1])
        actual_target = (
            observation.target.provider_slug,
            observation.target.model_id,
        )
        if actual_target != expected_target:
            raise ValueError(
                "observation target %r does not match run target %r"
                % (actual_target, expected_target)
            )
        await conn.execute(
            "INSERT INTO monitoring_observations ("
            " run_id, canary_id, provider_status, response, correct,"
            " grader_reason, grader_version, requested_provider, served_by,"
            " requested_model, model_served, latency_ms, prompt_tokens,"
            " completion_tokens, cost_usd, http_status, error"
            ") VALUES ("
            " %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,"
            " %s, %s, %s, %s"
            ")",
            (
                run_id,
                observation.canary_id,
                observation.provider_status,
                observation.response,
                observation.correct,
                observation.grader_reason,
                observation.grader_version,
                observation.target.provider_slug,
                observation.served_by,
                observation.target.model_id,
                observation.model_served,
                observation.latency_ms,
                observation.prompt_tokens,
                observation.completion_tokens,
                observation.cost_usd,
                observation.http_status,
                observation.error,
            ),
        )


async def complete_monitoring_run(
    pool: psycopg_pool.AsyncConnectionPool,
    run_id: int,
    summary: Mapping[str, object],
) -> bool:
    """
    Mark one RUNNING monitoring run COMPLETED and store its summary.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "UPDATE monitoring_runs SET state = 'COMPLETED',"
            " summary = %s::jsonb, completed_at = now()"
            " WHERE run_id = %s AND state = 'RUNNING' RETURNING run_id",
            (json.dumps(dict(summary)), run_id),
        )
        row = await cursor.fetchone()
    return row is not None


async def fail_monitoring_run(
    pool: psycopg_pool.AsyncConnectionPool,
    run_id: int,
    error: str,
) -> bool:
    """
    Mark one RUNNING monitoring run FAILED without deleting observations.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "UPDATE monitoring_runs SET state = 'FAILED',"
            " summary = %s::jsonb, completed_at = now()"
            " WHERE run_id = %s AND state = 'RUNNING' RETURNING run_id",
            (json.dumps({"error": error}), run_id),
        )
        row = await cursor.fetchone()
    return row is not None
