"""
Load demo sellers and accounts from `config/seed.yaml` into Postgres.

Safe to run on every startup: provider info is refreshed, but a seller's
reputation and state are never reset. API keys come from env vars named in the
YAML and are stored only as SHA-256 hashes.

Import as:

import research.Noesis.noesis_seed as rnonosee
"""

import dataclasses
import hashlib
import logging
import secrets
from typing import Any, Dict, Mapping

import psycopg_pool
import yaml

import helpers.hdbg as hdbg
import research.Noesis.noesis_settings as rnonoset

_LOG = logging.getLogger(__name__)

_ACCOUNT_KINDS = ["buyer", "seller", "operator"]


# #############################################################################
# SeedResult
# #############################################################################


@dataclasses.dataclass(frozen=True)
class SeedResult:
    """
    How many rows of each kind the seed file declared.
    """

    accounts: int
    sellers: int


# #############################################################################
# Seeding
# #############################################################################


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for storage and lookup.

    Keys are long random tokens, so a plain SHA-256 (no salt, no KDF) is enough.

    :param api_key: raw key
    :return: hex digest
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def _read_seed(path: str) -> Dict[str, Any]:
    """
    Read and sanity-check the seed YAML.

    :return: dict with `sellers` and `accounts` lists
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f.read()) or {}
    for key in ["sellers", "accounts"]:
        hdbg.dassert_isinstance(
            data.get(key, []), list, "'%s' in '%s' must be a list", key, path
        )
    return data


async def load_seed(
    pool: psycopg_pool.AsyncConnectionPool,
    settings: rnonoset.Settings,
    environ: Mapping[str, str],
) -> SeedResult:
    """
    Upsert accounts and sellers from `settings.seed_path`.

    :param pool: open connection pool
    :param settings: provides `seed_path` and `model_id`
    :param environ: where the `api_key_env` names are looked up (merged
        environment + dotenv)
    :return: counts of declared accounts and sellers
    """
    _LOG.debug("seed_path='%s'", settings.seed_path)
    data = _read_seed(settings.seed_path)
    accounts = data.get("accounts", [])
    sellers = data.get("sellers", [])
    async with pool.connection() as conn, conn.transaction():
        for acct in accounts:
            hdbg.dassert_in(
                acct.get("kind"),
                _ACCOUNT_KINDS,
                "Bad kind for account '%s'",
                acct.get("account_id"),
            )
            key = environ.get(acct.get("api_key_env", ""), "")
            hdbg.dassert_ne(
                key,
                "",
                "Env var '%s' for account '%s' is not set",
                acct.get("api_key_env"),
                acct["account_id"],
            )
            # Re-seeding with a new key replaces the stored hash (key rotation).
            await conn.execute(
                "INSERT INTO accounts (account_id, kind, api_key_hash) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (account_id) DO UPDATE "
                "SET api_key_hash = EXCLUDED.api_key_hash",
                (acct["account_id"], acct["kind"], hash_api_key(key)),
            )
        for seller in sellers:
            seller_id = seller["seller_id"]
            account_id = f"seller_{seller_id}"
            # Sellers don't call our API in the MVP: give them an unusable key.
            await conn.execute(
                "INSERT INTO accounts (account_id, kind, api_key_hash) "
                "VALUES (%s, 'seller', %s) ON CONFLICT (account_id) DO NOTHING",
                (account_id, hash_api_key(secrets.token_urlsafe(32))),
            )
            # Refresh provider info only; reputation/state/probation untouched.
            await conn.execute(
                "INSERT INTO sellers "
                "(seller_id, account_id, provider_slug, provider_name, model_id) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (seller_id) DO UPDATE SET "
                " provider_slug = EXCLUDED.provider_slug,"
                " provider_name = EXCLUDED.provider_name,"
                " model_id = EXCLUDED.model_id",
                (
                    seller_id,
                    account_id,
                    seller["provider_slug"],
                    seller["provider_name"],
                    settings.model_id,
                ),
            )
    result = SeedResult(accounts=len(accounts), sellers=len(sellers))
    _LOG.debug("return=%s", result)
    return result
