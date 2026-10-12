"""
Entrypoint for the `NoesisPlatform` HTTP API, run via:

> uvicorn research.Noesis.noesis_platform.main:app --host 0.0.0.0 --port 8000

Builds a module-level `app` (`platform_api.create_app()`) over an
`OrderBook`/`Gateway`/`ContractStore` selected by the `NOESIS_DB_BACKEND` env
var: `"memory"` or `"postgres"`

Import as:

import research.Noesis.noesis_platform.main as rnoemain
"""

import logging
import os
from typing import Dict

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.gateway.passthrough_proxy as rnopapro
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.noesis_platform.platform_api as rnoplapi

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)


# #############################################################################
# Config parsing
# #############################################################################


def _parse_api_keys(raw: str) -> Dict[str, str]:
    """
    Parse `NOESIS_API_KEYS` into the `Dict[str, str]` `create_app()` expects.

    :param raw: comma-separated `key:account` pairs
        - E.g. `"key1:account1,key2:account2"`
        - Empty string: no keys configured (every write endpoint then
          rejects every caller with `401`, a safe default)
    :return: map from API key to account id
        - E.g., `{"key1": "account1", "key2": "account2"}`.
    """
    _LOG.debug(hprint.to_str("raw"))
    api_keys: Dict[str, str] = {}
    if not raw:
        return api_keys
    for entry in raw.split(","):
        hdbg.dassert_in(
            ":", entry, "Malformed NOESIS_API_KEYS entry '%s'", entry
        )
        key, account_id = entry.split(":", 1)
        api_keys[key] = account_id
    _LOG.debug("return=%s", api_keys)
    return api_keys


def _get_db_backend() -> str:
    """
    Resolve `NOESIS_DB_BACKEND` into the backend type.

    :return: `"memory"` (default) or `"postgres"`
    """
    db_backend = os.environ.get("NOESIS_DB_BACKEND", "memory")
    return db_backend


# #############################################################################
# App construction
# #############################################################################


_DB_BACKEND = _get_db_backend()

_LOG.info("NOESIS_DB_BACKEND=%s", _DB_BACKEND)

if _DB_BACKEND == "postgres":
    # Deferred imports.
    import helpers.hsql_implementation as hsqlimpl
    import research.Noesis.noesis_platform.postgres_store as rnoposto

    # Fixed env var names, same as `hsqlimpl.get_connection_from_env_vars()`
    # already expects.
    _host = os.environ["POSTGRES_HOST"]
    _dbname = os.environ["POSTGRES_DB"]
    _port = int(os.environ["POSTGRES_PORT"])
    _user = os.environ["POSTGRES_USER"]
    _password = os.environ["POSTGRES_PASSWORD"]
    # Fail fast with a clear timeout instead of the first request hitting a
    # connection error if the `noesis_postgres` container is still starting.
    hsqlimpl.wait_db_connection(_host, _dbname, _port, _user, _password)
    _connection = hsqlimpl.get_connection_from_env_vars()
    # Create schema.
    rnoposto.init_schema(_connection)
    # Build tables.
    order_book = rnbacaau.OrderBook(
        store=rnoposto.PostgresOrderBookStore(_connection)
    )
    gateway = rnopapro.Gateway(store=rnoposto.PostgresRequestLogStore(_connection))
    contract_store = rnoposto.PostgresContractStore(_connection)
else:
    # In-memory backedn.
    order_book = rnbacaau.OrderBook()
    gateway = rnopapro.Gateway()
    contract_store = None

api_keys = _parse_api_keys(os.environ.get("NOESIS_API_KEYS", ""))
app = rnoplapi.create_app(
    order_book, gateway, api_keys, contract_store=contract_store
)
