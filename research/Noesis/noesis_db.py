"""
Async Postgres access for the Noesis MVP: a shared connection pool and a
migrations runner.

Uses `psycopg` 3 (async) rather than `helpers.hsql` (`psycopg2`, sync): the
gateway serves many concurrent requests from one event loop, and blocking DB
calls would serialize them.

Import as:

import research.Noesis.noesis_db as rnonodb
"""

import logging
import os
from typing import List, Tuple

import psycopg_pool

import research.Noesis.noesis_settings as rnonoset

_LOG = logging.getLogger(__name__)

MIGRATIONS_DIR = os.path.join(rnonoset.NOESIS_DIR, "migrations")


async def connect(
    database_url: str, *, min_size: int = 1, max_size: int = 10
) -> psycopg_pool.AsyncConnectionPool:
    """
    Open a connection pool and wait until Postgres accepts connections.

    Connections are autocommit; use `conn.transaction()` for multi-statement
    work.

    :param database_url: e.g., "postgresql://user:pwd@host:5433/db"
    :param min_size: connections kept open when idle
    :param max_size: upper bound on concurrent connections
    :return: an open pool
    """
    _LOG.debug("min_size=%s max_size=%s", min_size, max_size)
    pool = psycopg_pool.AsyncConnectionPool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        open=False,
        kwargs={"autocommit": True},
    )
    # Fail fast with a clear timeout instead of on the first request.
    await pool.open(wait=True, timeout=10)
    return pool


def _read_migrations(migrations_dir: str) -> List[Tuple[str, str]]:
    """
    Return `(filename, sql)` for every `*.sql` in `migrations_dir`, sorted.
    """
    filenames = sorted(
        f for f in os.listdir(migrations_dir) if f.endswith(".sql")
    )
    migrations = []
    for filename in filenames:
        with open(os.path.join(migrations_dir, filename), "r") as f:
            migrations.append((filename, f.read()))
    return migrations


async def apply_migrations(
    pool: psycopg_pool.AsyncConnectionPool,
    *,
    migrations_dir: str = MIGRATIONS_DIR,
) -> List[str]:
    """
    Run every `*.sql` in `migrations_dir` not yet recorded, in filename order.

    Each file runs in its own transaction together with its bookkeeping row in
    `schema_migrations`, so a failing file leaves no partial change and is
    retried next time.

    :param pool: open connection pool
    :param migrations_dir: directory of `NNN_name.sql` files
    :return: filenames applied by this call (empty if already up to date)
    """
    _LOG.debug("migrations_dir='%s'", migrations_dir)
    migrations = _read_migrations(migrations_dir)
    applied: List[str] = []
    async with pool.connection() as conn:
        # Remember which migrations already ran.
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " filename TEXT PRIMARY KEY,"
            " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        cursor = await conn.execute("SELECT filename FROM schema_migrations")
        done = {row[0] for row in await cursor.fetchall()}
        for filename, sql in migrations:
            if filename in done:
                continue
            # All-or-nothing: the schema change and its record commit together.
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (filename,),
                )
            _LOG.info("Applied migration '%s'", filename)
            applied.append(filename)
    _LOG.debug("return=%s", applied)
    return applied
