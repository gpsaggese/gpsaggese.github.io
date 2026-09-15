# `PR_P2b`: Use Postgres Backend - Implementation Spec

- This document specifies the Python code, schema, and Docker changes needed to
  implement `PR_P2b` from `research/Noesis/plan.Noesis.md`
- Scope, as stated in the plan:
  - Externalize the in-memory state of `NoesisMarket` and `NoesisServer` (order book,
    contract log, request log) to a real datastore (e.g. Postgres or Redis)
  - Result: a `NoesisMarket`/`NoesisServer` instance backed by persistent storage
    instead of the current in-process `List`/`Dict` state
- Roadmap position: `PR_P2b` is not listed under any `v0.x` roadmap bullet in
  `plan.Noesis.md`; it is sequenced directly after `PR_P2` (Cloud Deployment) in the
  `NoesisPlatform` PR list, and its own `TODO(gp)` note points at how to "inject a
  Postgres instance in the container", i.e. `PR_P2`'s Docker/devops scaffold. This
  spec therefore assumes `PR_P2` lands first and describes its own Docker/config
  deltas on top of `spec.PR_P2.md`'s design (`main.py`, `devops/`,
  `docker-compose.noesis.yml`, `GET /health`) rather than on top of the code actually
  on disk today, since `spec.PR_P2.md` is itself still a specification only, per its
  own header, not implemented code
- This is a specification only: no code in this document has been implemented; it
  describes what `PR_P2b` needs to add on top of the current `research/Noesis/*.py`
  code

## Design Decisions

### Postgres Via `helpers.hsql`

- Decision: back the storage layer with Postgres through `helpers.hsql` /
  `helpers.hsql_implementation`, not a new database client
- Justification:
  - The plan says "e.g. Postgres or Redis"; this PR's own title ("Use Postgres
    Backend") already picks Postgres
  - `helpers.hsql` / `helpers.hsql_implementation` (`psycopg2`-based, function style:
    `get_connection_from_env_vars()`, `wait_db_connection()`,
    `execute_query_to_df()`, `execute_insert_query()`, etc.) is already used across
    the wider Causify ecosystem this repo's dev-system is modeled on, gated so
    importing `helpers.hsql` does not require `psycopg2` unless it is actually
    installed (`hsql.py`'s `hmodule.has_module("psycopg2")` check)
  - `helpers.hsql_test.TestDbHelper` / `TestImOmsDbHelper` provides a test harness
    that spins up an ephemeral Postgres via `docker-compose` for tests that need a
    real DB (see "Unit Test Plan" below)
  - A concrete precedent already wires a Postgres-backed service into this
    ecosystem's `invoke`/Docker flow: `datapull/im_lib_tasks.py` (per-stage env file
    at `devops/env/{stage}.im_db_config.env`, an `im_postgres` `docker-compose`
    service, `im_docker_up`/`im_docker_down` invoke tasks)

### Raw SQL Via `helpers.hsql_implementation`, Not an ORM

- Decision: call `helpers.hsql_implementation`'s `DataFrame`/raw-SQL functions
  directly; do not introduce an ORM (e.g. SQLAlchemy)
- Justification: `helpers.hsql_implementation`'s functions operate on
  `pandas.DataFrame`s and raw SQL strings, not mapped classes, which is a smaller
  diff on top of this codebase's existing plain dataclasses (`Bid`, `Ask`,
  `Contract`, `RequestLogEntry`) than introducing a second object-mapping layer on
  top of them

## Trade-off and Alternative Design

- Redis: the plan allows "Postgres or Redis". Not chosen because no comparable Redis
  client/test-harness module exists anywhere in `helpers_root` (unlike Postgres's
  `helpers.hsql` / `helpers.hsql_test.TestDbHelper`), and the PR's own title already
  commits to Postgres. Trade-off: Redis could be a good fit for the ephemeral pending
  bid/ask queues, at the cost of building new client and test infrastructure this
  repo does not already have
- ORM (e.g. SQLAlchemy): not chosen, per "Raw SQL via `helpers.hsql_implementation`,
  Not an ORM" above. Trade-off: an ORM would give mapped-class ergonomics and
  migration tooling, at the cost of a second object-mapping layer on top of the
  existing dataclasses and a bigger diff

## Out of Scope

- Redis as an alternative backend: this spec picks Postgres per the PR's own title
  (see "Design Decisions" above)
- A schema migration/versioning framework (e.g. Alembic): `init_schema()`'s
  `CREATE TABLE IF NOT EXISTS` DDL (below) is enough for this prototype's first
  schema; a real migration tool is a follow-up if the schema changes later
- Connection pooling: one shared `psycopg2` connection per process, matching how
  `main.py` already builds one process-lifetime `OrderBook`/`Gateway` singleton
  (`spec.PR_P2.md`); revisit if load testing shows a single connection is a
  bottleneck
- Provisioning a managed Postgres instance (e.g. AWS RDS) for the `PR_P2` ECS
  deployment path: one-time AWS infra setup, out of scope for this PR's Python/schema
  code, same caveat `spec.PR_P2.md`'s "Cloud Target" section raises for the ECS
  cluster itself
- Any change to the matching algorithm, contract schema shape, or HTTP surface: this
  PR only changes _where_ state lives, not what the state means or how
  `_match_orders_in_tier()` clears a tier
- `PR_M8`'s real fulfillment wiring: dispatch still calls `mock_fulfill()`;
  persisting a `Contract.fulfilled` value does not make the value itself real
- A `/ready` endpoint distinct from `PR_P2`'s `GET /health`: this PR keeps `/health`
  DB-agnostic (a liveness check, not a readiness check per "Risks and Limitations to
  Call Out" below), since adding a DB-probing endpoint is an orthogonal concern
  `PR_P2` did not scope either

## Current State

- Three separate in-memory state surfaces, matching the plan's "order book, contract
  log, request log" list exactly, all lost on process exit (`architecture.md`
  Weakness 6):
  - `batch_call_auction.OrderBook.__init__` (`batch_call_auction.py:317-319`):
    `self._bids: List[Bid] = []` / `self._asks: List[Ask] = []`, appended to by
    `submit_bid()`/`submit_ask()` and unconditionally emptied by `clear_round()`
    (`batch_call_auction.py:371-373`)
  - `platform_api._MarketState.__init__` (`platform_api.py:206-211`):
    `self._contracts_by_id: Dict[int, Contract] = {}`, `self._next_contract_id = 0`,
    `self._latest_round_by_tier: Dict[str, RoundClearResponse] = {}`,
    `self._next_round_id = 0`; this is the "contract log" plus the per-tier "latest
    cleared round" cache standing in for `NoesisMarket`'s pricing feed (`PR_M4`, not
    implemented yet)
  - `passthrough_proxy.Gateway.__init__` (`passthrough_proxy.py:128-130`):
    `self._log: List[RequestLogEntry] = []`, `self._next_request_id = 0`
- All three ids (`contract_id`, `round_id`, `request_id`) are assigned by a plain
  Python counter starting at `0` on every process start; a naive persistence layer
  that keeps these counters as-is would start reassigning colliding ids after every
  restart, silently corrupting the very persistence this PR is meant to add (see
  "Risks and Limitations to Call Out" below)
- No Postgres/SQL dependency anywhere in `research/Noesis` today: `helpers.hsql` is
  available repo-wide but unused by any of the four existing modules

## Implementation

- Follows this codebase's existing dependency-injection idiom (`architecture.md`'s
  "Key design decisions": "every side effect that would be non-deterministic in a
  test ... is injected as a callable"), extended from injected _callables_
  (`FulfillmentFunc`, `ProviderCallFunc`) to injected _storage objects_: one small
  `abc.ABC` per state surface, colocated with the class that owns it, plus an
  `_InMemory*Store` default that is today's plain `List`/`Dict` code extracted
  unchanged, so every existing test keeps passing with no behavior change
- A new module, `research/Noesis/postgres_store.py` (import alias `rnpost`), holds
  only the Postgres-specific pieces: the schema DDL, `init_schema()`, and one
  `Postgres*Store` class per `ABC`. It imports the three owning modules to implement
  their `ABC`s; none of the three owning modules import it, so there is no import
  cycle and no new dependency on `psycopg2` for a caller that never touches the
  Postgres backend (matches `helpers.hsql`'s own optional-import gating)

### `batch_call_auction.py`: `OrderBookStore`

```python
import abc

class OrderBookStore(abc.ABC):
    """
    Pluggable storage backend for `OrderBook`'s pending `Bid`/`Ask` queues.
    """

    @abc.abstractmethod
    def add_bid(self, bid: Bid) -> None:
        ...

    @abc.abstractmethod
    def add_ask(self, ask: Ask) -> None:
        ...

    @abc.abstractmethod
    def get_bids(self) -> List[Bid]:
        """
        :return: pending bids, in submission order
        """
        ...

    @abc.abstractmethod
    def get_asks(self) -> List[Ask]:
        """
        :return: pending asks, in submission order
        """
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        """
        Drop every stored bid/ask (`OrderBook.clear_round()`'s existing
        drop-everything semantics; see `architecture.md` Weakness 4).
        """
        ...


class _InMemoryOrderBookStore(OrderBookStore):
    """
    Default `OrderBookStore`: today's plain `List[Bid]`/`List[Ask]`, extracted
    unchanged.
    """
    # `add_bid()`/`add_ask()`/`get_bids()`/`get_asks()`/`clear()` reproduce
    # exactly what `OrderBook.submit_bid()`/`submit_ask()`/
    # `get_pending_bids()`/`get_pending_asks()`/`clear_round()`'s
    # `self._bids = []` do today.
```

- `OrderBook.__init__(self, *, store: Optional[OrderBookStore] = None)`:
  `if store is None: store = _InMemoryOrderBookStore()`. The `Optional[...] = None`
  default here is a deliberate exception to `.claude/skills/coding.rules.md`'s
  "Minimize Default Values of None": a stateful default
  (`store: OrderBookStore = _InMemoryOrderBookStore()`) would create **one** store
  instance at function-definition time, shared and mutated by every `OrderBook()`
  call, exactly the mutable-default-argument pitfall
  `contract_dispatch.mock_fulfill()`'s existing
  `rng: Optional[ random.Random] = None` parameter already works around the same way
  in this codebase

### `platform_api.py`: `ContractStore`

```python
class ContractStore(abc.ABC):
    """
    Pluggable storage backend for `_MarketState`'s contract log and per-tier
    "latest cleared round" cache.
    """

    @abc.abstractmethod
    def save_contract(self, contract: rnocodis.Contract) -> int:
        """
        :return: the `contract_id` assigned to `contract`
        """
        ...

    @abc.abstractmethod
    def get_contract(self, contract_id: int) -> rnocodis.Contract:
        ...

    @abc.abstractmethod
    def next_round_id(self) -> int:
        """
        Assign one new `round_id`, shared by every tier cleared in the same
        `clear_round()` call (see the note on `round_id` below).
        """
        ...

    @abc.abstractmethod
    def save_round(self, round_response: RoundClearResponse) -> None:
        ...

    @abc.abstractmethod
    def get_latest_round(self, tier: str) -> RoundClearResponse:
        ...


class _InMemoryContractStore(ContractStore):
    """
    Default `ContractStore`: today's `_contracts_by_id`/`_next_contract_id`/
    `_latest_round_by_tier`/`_next_round_id`, extracted unchanged.
    """
```

- `_MarketState.__init__(self, order_book, *, fulfillment_fn=..., store: Optional[ContractStore] = None)`:
  same `None`-default rationale as `OrderBook.store` above

### `passthrough_proxy.py`: `RequestLogStore`

```python
class RequestLogStore(abc.ABC):
    """
    Pluggable storage backend for `Gateway`'s request/response log.
    """

    @abc.abstractmethod
    def append(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        latency_in_secs: float,
        cost: float,
    ) -> RequestLogEntry:
        """
        Persist one logged request/response pair.

        :return: the persisted `RequestLogEntry`, with `request_id` assigned
            by the store
        """
        ...

    @abc.abstractmethod
    def get_all(self) -> List[RequestLogEntry]:
        """
        :return: every logged entry, in call order
        """
        ...

    @abc.abstractmethod
    def query(
        self, *, provider: str = "", model: str = ""
    ) -> List[RequestLogEntry]:
        ...


class _InMemoryRequestLogStore(RequestLogStore):
    """
    Default `RequestLogStore`: today's `_log`/`_next_request_id`, extracted
    unchanged; `append()` builds the same `RequestLogEntry` `Gateway.call()`
    builds today and keeps its own counter.
    """
```

- `Gateway.__init__(self, *, clock_fn=time.perf_counter, store: Optional[ RequestLogStore] = None)`:
  same `None`-default rationale as above

### `research/Noesis/postgres_store.py` (new)

- Module docstring: points back to this file (`spec.PR_P2b.md`) and to the three
  `ABC`s it implements
- Owns the DDL, run by `init_schema(connection)` (idempotent: every statement is
  `CREATE TABLE IF NOT EXISTS`, safe to call on every `main.py` startup, no migration
  framework needed for this prototype's first schema per "Out of Scope" above)
- `noesis_` prefix on every table, since this may run against a shared Postgres
  instance alongside other projects' tables (matches the `im_postgres_db_local` style
  naming already used elsewhere in this ecosystem)

```sql
CREATE TABLE IF NOT EXISTS noesis_bids (
    id BIGSERIAL PRIMARY KEY,
    buyer_id TEXT NOT NULL,
    n_tasks INTEGER NOT NULL,
    c_level_min TEXT NOT NULL,
    l_max DOUBLE PRECISION NOT NULL,
    r_min DOUBLE PRECISION NOT NULL,
    p_max DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS noesis_asks (
    id BIGSERIAL PRIMARY KEY,
    seller_id TEXT NOT NULL,
    n_tasks INTEGER NOT NULL,
    c_level TEXT NOT NULL,
    l_typical DOUBLE PRECISION NOT NULL,
    r_typical DOUBLE PRECISION NOT NULL,
    p_min DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS noesis_contracts (
    contract_id BIGSERIAL PRIMARY KEY,
    buyer_id TEXT NOT NULL,
    seller_id TEXT NOT NULL,
    n_tasks INTEGER NOT NULL,
    c_level TEXT NOT NULL,
    l_max DOUBLE PRECISION NOT NULL,
    r_min DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    fulfilled BOOLEAN
);

-- `round_id` is NOT `SERIAL` on this table: one `clear_round()` call clears
-- every tier under the SAME round_id (see `next_round_id()` below), so the id
-- is generated once per round, not once per row.
CREATE SEQUENCE IF NOT EXISTS noesis_round_id_seq;

CREATE TABLE IF NOT EXISTS noesis_tier_rounds (
    tier TEXT NOT NULL,
    round_id BIGINT NOT NULL,
    clearing_price DOUBLE PRECISION,
    matched_volume INTEGER NOT NULL,
    cleared_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tier, round_id)
);

CREATE TABLE IF NOT EXISTS noesis_request_log (
    request_id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    latency_in_secs DOUBLE PRECISION NOT NULL,
    cost DOUBLE PRECISION NOT NULL
);
```

- `contract_id`/`request_id` are `BIGSERIAL` (one id per row, matching today's 1:1
  counter-per-record); `round_id` is a separate `noesis_round_id_seq`, fetched once
  per `clear_round()` call via `next_round_id()`
  (`SELECT nextval('noesis_round_id_seq')`) and reused for every tier's row in that
  round, replicating `platform_api.py:238-239`'s current "`round_id` assigned once,
  before the per-tier loop" behavior exactly
- `init_schema(connection: hsql.DbConnection) -> None`: runs the DDL above via
  `connection.cursor().execute(...)`, one statement at a time, then
  `connection. commit()` (or relies on the `autocommit=True` connections
  `hsql. get_connection*()` already returns by default)
- `PostgresOrderBookStore(connection)`:
  - `add_bid(bid)` / `add_ask(ask)`:
    `hsqlimpl.execute_insert_query(connection, pd.DataFrame([dataclasses.asdict(bid)]), "noesis_bids")`
    (and the `asks` equivalent); `execute_insert_query()` is a bulk-row helper built
    on `psycopg2.extras.execute_values`, which fits a single-row insert fine and
    needs no `RETURNING` here since bid/ask ids are never read back
  - `get_bids()` / `get_asks()`:
    `hsqlimpl.execute_query_to_df(connection, "SELECT buyer_id, n_tasks, c_level_min, l_max, r_min, p_max FROM noesis_bids ORDER BY id")`,
    then one `Bid(**row)` per DataFrame row. **The `ORDER BY id` is required, not
    cosmetic**: `_match_orders_in_tier()`'s docstring states "`sorted()` is stable,
    so ties fall back to submission order" - without an explicit `ORDER BY`, Postgres
    does not guarantee row order, which would make tie-breaking within a price-tied
    tier nondeterministic and silently diverge from `_InMemoryOrderBookStore`'s
    list-append order
  - `clear()`: `DELETE FROM noesis_bids; DELETE FROM noesis_asks;` (matches
    `OrderBook.clear_round()`'s "drop everything" semantics exactly; a `TRUNCATE`
    would work too but `DELETE` keeps the `BIGSERIAL` sequence monotonically
    increasing across rounds, which is harmless here since bid/ask ids are never
    surfaced)
- `PostgresContractStore(connection)`:
  - `save_contract(contract)`: a raw
    `cursor.execute("INSERT INTO noesis_contracts(...) VALUES (...) RETURNING contract_id", (...)); return cursor.fetchone()[0]`,
    **not** `hsqlimpl.execute_insert_query()`, since that helper is a bulk multi-row
    insert built on `execute_values()` with no `RETURNING` support, and this call
    specifically needs the generated `contract_id` back
  - `get_contract(contract_id)`:
    `hsqlimpl.execute_query_to_df(connection, f"SELECT ... FROM noesis_contracts WHERE contract_id = {contract_id}")`;
    `hdbg.dassert_lt(0, len(df), "Unknown contract_id '%s'", contract_id)` (same
    "unknown id" contract `_MarketState.get_contract()` already documents, still
    surfaced as an HTTP 400 by `platform_api.py`'s existing `AssertionError` handler)
    then `Contract(**df.iloc[0].to_dict())`
  - `next_round_id()`:
    `cursor.execute("SELECT nextval('noesis_round_id_seq')") ; return cursor.fetchone()[0]`
  - `save_round(round_response)`: raw insert into `noesis_tier_rounds` (needs no
    `RETURNING`; `round_id` is already known from `next_round_id()`)
  - `get_latest_round(tier)`:
    `SELECT ... FROM noesis_tier_rounds WHERE tier = %s ORDER BY round_id DESC LIMIT 1`,
    `hdbg.dassert_lt(0, len(df), "No cleared round yet for tier '%s'", tier)`,
    matching `_MarketState. get_latest_round()`'s existing assertion
- `PostgresRequestLogStore(connection)`:
  - `append(...)`: raw
    `INSERT INTO noesis_request_log(...) VALUES (...) RETURNING request_id`, same
    `RETURNING`-needs-raw-SQL reasoning as `save_contract()`; builds and returns the
    `RequestLogEntry` with the returned id
  - `get_all()`: `SELECT ... FROM noesis_request_log ORDER BY request_id` (the same
    "in call order" invariant as `get_bids()`/`get_asks()` above, and for the same
    reason: `Gateway.get_log()`'s docstring promises call order)
  - `query(provider="", model="")`: builds the `WHERE` clause conditionally (skip a
    clause for each empty filter), matching `Gateway.query_log()`'s existing
    semantics exactly, still `ORDER BY request_id`

### `research/Noesis/main.py` (extends `PR_P2`'s Spec)

- New env var `NOESIS_DB_BACKEND` (`os.environ.get("NOESIS_DB_BACKEND", "memory")`):
  `"memory"` (default, today's behavior, no Postgres dependency) or `"postgres"`
- When `"postgres"`:
  - Read the same five `POSTGRES_HOST`/`POSTGRES_DB`/`POSTGRES_PORT`/
    `POSTGRES_USER`/`POSTGRES_PASSWORD` env vars
    `hsql_implementation. get_connection_from_env_vars()` already expects (fixed
    names, not `research/Noesis`-specific, since they come from the official Postgres
    Docker image's own env var convention that
    `hsql_implementation. get_connection_info_from_env_file()`'s comment already
    cites)
  - `hsqlimpl.wait_db_connection(host, dbname, port, user, password)` before
    proceeding, so `main.py` fails fast with a clear timeout instead of the first
    request hitting a connection error if the `noesis_postgres` container is still
    starting
  - `connection = hsqlimpl.get_connection_from_env_vars()`
  - `rnpost.init_schema(connection)`
  - `order_book = rnbacaau.OrderBook(store=rnpost.PostgresOrderBookStore( connection))`
  - `gateway = rnopapro.Gateway(store=rnpost.PostgresRequestLogStore( connection))`
  - `contract_store = rnpost.PostgresContractStore(connection)`
- When `"memory"` (default): `order_book = rnbacaau.OrderBook()`,
  `gateway = rnopapro.Gateway()`, `contract_store = None` - byte-for-byte `PR_P2`'s
  existing spec, unchanged
- `app = rnoplapi.create_app(order_book, gateway, api_keys, contract_store= contract_store)`
- `_LOG.info("NOESIS_DB_BACKEND=%s", _DB_BACKEND)` at startup, so a deployment's logs
  make the active backend obvious without inspecting env vars directly

### `research/Noesis/devops/compose/docker-compose.noesis.yml` (extends `PR_P2`'s Spec)

- Extends `spec.PR_P2.md`'s compose file with one new service and one new volume,
  modeled on `datapull/im_lib_tasks.py`'s `im_postgres` service and the ecosystem's
  standard `postgres:<version>` image usage
  (`helpers_root/dev_scripts_helpers/update_devops_packages/test/db_example/ docker-compose.yml`):

  ```yaml
  services:
    noesis_api:
      # ... existing PR_P2 fields (extends, command, ports) ...
      environment:
        - POSTGRES_HOST=noesis_postgres
        - POSTGRES_DB=${POSTGRES_DB}
        - POSTGRES_PORT=5432
        - POSTGRES_USER=${POSTGRES_USER}
        - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
        - NOESIS_DB_BACKEND=${NOESIS_DB_BACKEND}
      depends_on:
        noesis_postgres:
          condition: service_healthy

    noesis_postgres:
      image: postgres:16
      restart: "no"
      environment:
        - POSTGRES_DB=${POSTGRES_DB}
        - POSTGRES_USER=${POSTGRES_USER}
        - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      volumes:
        - noesis_postgres_data:/var/lib/postgresql/data
      healthcheck:
        test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
        interval: 5s
        timeout: 5s
        retries: 5

  volumes:
    noesis_postgres_data: {}
  ```

- Local persistence: `noesis_postgres_data` is a plain Docker volume, so
  `docker compose down` (no `-v`) preserves it across a restart, but it is still
  node-local, not replicated; a host disk failure loses it the same as today's
  in-memory state would on a crash (see "Risks and Limitations to Call Out" below)
- Production (the `PR_P2` "Cloud Target" ECS path): do **not** run `noesis_postgres`
  as a sidecar container in the same ECS task; use AWS RDS (managed Postgres)
  instead, with the five `POSTGRES_*` values set as ECS task definition `secrets`
  entries, the same treatment `PR_P2` already specifies for `NOESIS_API_KEYS`
  Provisioning the RDS instance itself is one-time AWS infra setup, out of scope here
  (see "Out of Scope" above)

## Interaction with Existing Code

- `batch_call_auction.py`:
  - `submit_bid()`/`submit_ask()`/`get_pending_bids()`/`get_pending_asks()` become
    one-line delegations to `self._store.add_bid()` / `self._store.get_bids()` / etc
  - `clear_round()` changes minimally: replace `self._bids`/`self._asks` reads with
    `self._store.get_bids()`/`get_asks()`, and the trailing
    `self._bids = []; self._asks = []` with `self._store.clear()`;
    `_match_orders_in_tier()` itself (the pure matching algorithm) is untouched
- `platform_api.py`:
  - `clear_round()` changes:
    `round_id = self._next_round_id; self. _next_round_id += 1` becomes
    `round_id = self._store.next_round_id()` (called once, before the per-tier loop,
    exactly where today's counter increment happens); the per-contract
    `self._contracts_by_id[self. _next_contract_id] = contract; self._next_contract_id += 1`
    loop becomes `contract_id = self._store.save_contract(contract)` (the store
    assigns the id); `self._latest_round_by_tier[c_level] = round_response` becomes
    `self. _store.save_round(round_response)`
  - `get_contract()`/`get_latest_round()` delegate to
    `self._store. get_contract()`/`get_latest_round()`, raising the same
    `hdbg. dassert_in(...)`-driven `AssertionError` on an unknown id/tier as today
    (the `_InMemoryContractStore` keeps the existing `hdbg.dassert_in` checks;
    `PostgresContractStore` raises the same way on an empty query result, per
    "Implementation" above)
  - `create_app()`'s signature grows one new keyword parameter,
    `contract_store: Optional[ContractStore] = None`, threaded through to
    `_MarketState(order_book, fulfillment_fn=fulfillment_fn, store=contract_store)`
    so `main.py` can inject `PostgresContractStore` without `_MarketState` needing to
    be constructed outside `create_app()`
- `passthrough_proxy.py`:
  - `call()` changes: the
    `entry = RequestLogEntry(self._next_request_id, ...); self._next_request_id += 1; self._log.append(entry)`
    block becomes
    `entry = self._store.append(provider_name, model, prompt, response, latency_in_secs, cost)`
  - `get_log()`/`query_log()` delegate to
    `self._store.get_all()`/`query( provider=provider, model=model)`
- Data flow: `main.py` picks `NOESIS_DB_BACKEND`, builds `OrderBook`/`Gateway`/
  `contract_store` from either the in-memory or Postgres store, and `create_app()`
  wires `contract_store` into `_MarketState`
- Backward compatibility: `OrderBook()`, `Gateway()`, and `_MarketState( order_book)`
  called with no `store=` argument default to the extracted `_InMemory*Store`s,
  byte-for-byte the same list/dict behavior as today; every existing call site and
  test keeps passing unchanged (see "Unit Test Plan" below)

## Configuration and Secrets

- New required env vars when `NOESIS_DB_BACKEND=postgres`: `POSTGRES_HOST`,
  `POSTGRES_DB`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD` (names fixed
  by `hsql_implementation.get_connection_from_env_vars()`, not renamable without also
  changing that shared helper)
- `NOESIS_DB_BACKEND` itself defaults to `"memory"`, so an existing `PR_P2`
  deployment that never sets it keeps running exactly as before this PR lands; this
  is the rollback path if the Postgres backend misbehaves in production
- `research/Noesis/devops/env/default.env` (`PR_P2` created a placeholder) gets
  `POSTGRES_DB`/`POSTGRES_USER`/`NOESIS_DB_BACKEND` added for local dev defaults
- `POSTGRES_PASSWORD`: stays out of the committed `default.env`, same treatment
  `PR_P2` already gives `NOESIS_API_KEYS`; local dev via `docker compose run -e` /
  shell env (never committed); production via ECS `secrets` backed by AWS Secrets
  Manager or Parameter Store

## Unit Test Plan

- New file `research/Noesis/test/test_postgres_store.py`, naming per
  `.claude/skills/testing.rules.md`
- Uses `helpers.hsql_test.TestDbHelper` (or its `TestImOmsDbHelper` concrete
  subclass, adapted with a `noesis_postgres` service name/env file) rather than
  mocking Postgres: this is a deliberate call worth flagging against
  `.claude/skills/testing.rules.md`'s general "Mock Only External Dependencies"
  guidance, which lists databases as something to mock. `TestDbHelper` is this
  ecosystem's own established, already-maintained mechanism specifically for testing
  code that talks to a real Postgres (spins up an ephemeral `docker-compose` Postgres
  per test class, torn down after), and exercising the actual DDL/SQL in
  `postgres_store.py` against a mock would not catch a real SQL error; if this
  tradeoff is rejected at implementation time, mocking `psycopg2` at the call site is
  the fallback, at the cost of not testing the DDL/SQL itself
  - `@pytest.mark.requires_docker_in_docker` (inherited from `TestDbHelper`)
  - `TestPostgresOrderBookStore`:
    - `test1`: `add_bid()`/`add_ask()` then `get_bids()`/`get_asks()` round-trip the
      same `Bid`/`Ask` fields back
    - `test2`: three bids added in a known order come back from `get_bids()` in that
      same order (guards the `ORDER BY id` requirement above)
    - `test3`: `clear()` empties both tables; a subsequent `get_bids()`/ `get_asks()`
      returns `[]`
  - `TestPostgresContractStore`:
    - `test1`: `save_contract()` returns an id that `get_contract()` resolves back to
      an equal `Contract`
    - `test2`: a fresh `PostgresContractStore(connection)` instance (same connection,
      new Python object, simulating a process restart) still resolves a `contract_id`
      saved by a prior instance, the behavior `_InMemoryContractStore` cannot offer
      and the whole point of this PR
    - `test3`: `next_round_id()` called twice returns two different, increasing ids;
      `save_round()`/`get_latest_round()` round-trip a `RoundClearResponse`
    - `test4`: `get_contract()` on an unknown id raises `AssertionError`
  - `TestPostgresRequestLogStore`:
    - `test1`: `append()` then `get_all()` round-trips the logged fields,
      `request_id` assigned by the store
    - `test2`: `query(provider=...)`/`query(model=...)` filter as
      `Gateway. query_log()` already does today
  - `Test_init_schema`:
    - `test1`: calling `init_schema()` twice on the same connection does not raise
      (idempotency of `CREATE TABLE IF NOT EXISTS`)
- `research/Noesis/test/test_batch_call_auction.py`,
  `test/test_contract_dispatch.py`, `test/test_passthrough_proxy.py`, and
  `test/test_platform_api.py`: no changes expected; this is the regression signal
  that the default backend truly did not change, not just an assumption (see
  "Backward compatibility" in "Interaction with Existing Code" above)
- Extend `research/Noesis/test/test_main.py` (from `PR_P2`'s spec): a case covering
  `NOESIS_DB_BACKEND` defaulting to `"memory"` when unset, exercised the same way
  `PR_P2`'s `Test__parse_api_keys` isolates a pure env-parsing function from
  `main.py`'s import-time side effects

## Risks and Limitations to Call Out

- **Id-counter correctness is the main risk of a shallow implementation**: if
  `contract_id`/`round_id`/`request_id` are kept as Python-side counters seeded at
  `0` on every process start (today's behavior) instead of moved to DB-generated ids
  (`BIGSERIAL`/`nextval()` as specced above), persisted rows from a prior process
  would collide with ids reassigned after a restart; this spec's
  `save_contract()`/`next_round_id()`/`append()` all return a DB-assigned id
  specifically to avoid this
- `round_id` must be assigned once per `clear_round()` call and shared across every
  tier's row in that round, not once per row; using a per-row `SERIAL` on
  `noesis_tier_rounds` (an easy mistake, since `contract_id`/`request_id` _are_ fine
  as per-row `SERIAL`s) would silently change `round_id`'s meaning
- `ORDER BY` is required, not optional, on every `get_bids()`/`get_asks()`/
  `get_all()`/`get_latest_round()` query: Postgres row order is otherwise
  unspecified, which would make the auction's price-tie-breaking and the request
  log's "in call order" contract nondeterministic in a way the in-memory list never
  was
- `GET /health` stays DB-agnostic (a liveness check per `spec.PR_P2.md`'s design, not
  extended into a readiness check here): a container can report `{"status": "ok"}`
  while `NOESIS_DB_BACKEND=postgres` and the DB connection is actually down, since
  `main.py` only calls `wait_db_connection()` once at startup, not on every health
  check
- Local dev persistence is a single Docker volume, not a replicated/managed
  datastore; production should use RDS, not the compose `noesis_postgres` service,
  per "Implementation" above
- No connection pooling: a single shared `psycopg2` connection serves every request
  for the process lifetime; acceptable for this prototype's expected load, a
  bottleneck if concurrent request volume grows (see "Out of Scope")
- This PR does not address `architecture.md`'s other open weaknesses (exact-tier-only
  matching, at-most-one-bid-per-buyer assumption, no reputation filtering, no rate
  limiting): it only changes where the three state surfaces live, not the business
  logic operating on them

## Result

- Implemented as specced:
  - `OrderBookStore`/`_InMemoryOrderBookStore` (`batch_call_auction.py`),
    `ContractStore`/`_InMemoryContractStore` (`platform_api.py`),
    `RequestLogStore`/`_InMemoryRequestLogStore` (`passthrough_proxy.py`): each
    owning class (`OrderBook`, `_MarketState`, `Gateway`) now takes a keyword-only
    `store: Optional[...] = None`, defaulting to the extracted in-memory store; every
    existing call site and unit test (`test/test_ batch_call_auction.py`,
    `test/test_contract_dispatch.py`, `test/test_ passthrough_proxy.py`,
    `test/test_platform_api.py`) passes unchanged (53 passed, run locally),
    confirming the default-backend behavior did not change
  - `research/Noesis/postgres_store.py`: `init_schema()` and the DDL above,
    `PostgresOrderBookStore`, `PostgresContractStore`, `PostgresRequestLogStore`, all
    implemented exactly as specced
    (`BIGSERIAL`/`nextval('noesis_round_id_seq')`-assigned ids, not Python counters;
    explicit `ORDER BY` on every list query)
  - One deviation from the literal spec text, for correctness:
    `get_latest_ round(tier)` and `query(provider=..., model=...)` bind
    `tier`/`provider`/ `model` as query parameters via a raw parameterized
    `cursor.execute()` instead of the
    `hsqlimpl.execute_query_to_df(connection, f"... WHERE tier = '{tier}'")` f-string
    form the spec's prose suggests: `execute_query_to_df()` has no parameter-binding
    support, and `tier`/ `provider`/`model` are caller-controlled HTTP inputs
    (`GET /rounds/ {tier}/latest`, `GET /logs?provider=&model=`), so f-string
    interpolation there would be a SQL-injection vector. `get_contract(contract_id)`
    kept the spec's literal f-string form since `contract_id` is FastAPI-coerced to
    `int` before reaching this code
  - `main.py`: written from scratch (did not exist on disk), combining
    `spec.PR_P2.md`'s baseline (`_parse_api_keys()`, `GET /health`, module-level
    `app`) with this spec's `NOESIS_DB_BACKEND` switch; the Postgres branch does a
    deferred (`if`-scoped)
    `import research.Noesis. postgres_store`/`helpers.hsql_implementation`, so the
    default `memory` backend has no `psycopg2` import-time dependency, matching
    `helpers. hsql`'s own optional-import gating
  - `platform_api.create_app()` gained `GET /health` and the
    `contract_ store: Optional[ContractStore] = None` keyword parameter, threaded to
    `_MarketState(..., store=contract_store)`
  - `devops/compose/docker-compose.noesis.yml` and `devops/env/default.env`: added
    under `research/Noesis/devops/`, combining `spec.PR_P2.md`'s `noesis_api` service
    with this spec's `noesis_postgres` service/volume and env additions; validated as
    syntactically-valid YAML
  - Tests: `test/test_postgres_store.py` (`TestPostgresOrderBookStore`,
    `TestPostgresContractStore`, `TestPostgresRequestLogStore`, `Test_init_ schema`,
    all against `helpers.hsql_test.TestImOmsDbHelper`, not mocks, per the spec's own
    tradeoff call) and `test/test_main.py` (`Test__parse_api_keys`,
    `Test__get_db_backend`); `test/test_platform_ api.py` gained `Test_health`
- Verified against a real Postgres, beyond what collection/import checks alone would
  catch: with `psycopg2-binary` installed locally and a throwaway `postgres:16`
  container, every `postgres_store.py` class (`init_schema()` idempotency, all three
  stores' round-trip/ordering/ unknown-id behavior) and the full `main.py` app with
  `NOESIS_DB_BACKEND= postgres` (`GET /health`, `POST /bids`/`/asks`,
  `POST /rounds/clear`, `GET /contracts/{id}`, `GET /rounds/{tier}/latest`, including
  the 400 on an unknown `contract_id`) were exercised end-to-end and passed; the
  container was removed afterward. `test/test_postgres_store.py` itself was only
  _collected_ successfully in this sandbox (10 tests), not run via
  `TestImOmsDbHelper`'s own `docker-compose`, since
  `helpers.hserver. can_run_docker_from_docker()` is `False` here (no
  docker-in-docker); it should be re-run in an environment where that helper returns
  `True` before this PR is considered fully verified by its own test suite
- Deferred, deliberately out of this PR's scope (see "Out of Scope" above): no ECS
  task definition/service or AWS RDS instance was provisioned;
  `NOESIS_DB_BACKEND=postgres` was exercised only against a local ad hoc container as
  described above, never against a managed/cloud Postgres (RDS or otherwise)
  `research/Noesis` making the full dev-system "runnable dir" scaffolding
  (`changelog.txt`, `repo_config.yaml`, `.dockerignore`, `devops/docker_build/`,
  `devops/docker_run/`, the `tasks.py`/`conftest.py`/ `invoke.yaml`/`pytest.ini`
  symlinks) was `spec.PR_P2.md`'s own scope, not this PR's, and has since landed
  there (see `spec.PR_P2.md`'s "Result"); `docker-compose.noesis.yml`'s
  `extends: tmp.docker-compose.yml` now resolves against that scaffolding, though the
  ECS/RDS provisioning above remains outstanding