"""
Import as:

import research.Noesis.noesis_platform.test.test_postgres_store as rnttpost
"""

import logging

import pytest

# `psycopg2` (via `helpers.hsql_implementation`) is only needed to actually
# talk to Postgres; skip cleanly instead of failing collection where it is not
# installed.
# `TestImOmsDbHelper` itself further gates every test class below with
# `@pytest.mark.requires_docker_in_docker`.
pytest.importorskip("psycopg2")

import helpers.hsql_test as hsqltest  # noqa: E402 # pylint: disable=wrong-import-position
import research.Noesis.market.batch_call_auction as rnbacaau  # noqa: E402 # pylint: disable=wrong-import-position
import research.Noesis.market.contract_dispatch as rnocodis  # noqa: E402 # pylint: disable=wrong-import-position
import research.Noesis.market.round_runner as rnmaroru  # noqa: E402 # pylint: disable=wrong-import-position
import research.Noesis.noesis_platform.postgres_store as rnoposto  # noqa: E402 # pylint: disable=wrong-import-position

_LOG = logging.getLogger(__name__)


# Every table `init_schema()` creates; emptied before each test so tests
# don't depend on execution order (`.claude/skills/testing.rules.md`'s "Keep
# Tests Self-Contained"), per `TestDbHelper`'s "don't assume the DB is
# clean" invariant.
_NOESIS_TABLES = [
    "noesis_bids",
    "noesis_asks",
    "noesis_contracts",
    "noesis_tier_rounds",
    "noesis_request_log",
]


# #############################################################################
# _NoesisDbHelper
# #############################################################################


class _NoesisDbHelper(hsqltest.TestImOmsDbHelper):
    """
    `TestImOmsDbHelper` wired to a `noesis_postgres` service/env file,
    shared by every `postgres_store.py` test class below.

    Uses a real ephemeral Postgres (spun up by `TestDbHelper`'s
    `docker-compose`) rather than mocking `psycopg2`; a deliberate call
    against `.claude/skills/testing.rules.md`'s general "Mock Only External
    Dependencies" guidance, flagged in `spec.PR_P2b.md`'s "Unit Test Plan":
    exercising the actual DDL/SQL in `postgres_store.py` against a mock
    would not catch a real SQL error.
    """

    @classmethod
    def get_id(cls) -> int:
        """
        :return: a per-class id, so concurrently-run test classes (e.g.
            under `pytest-xdist`) don't collide on the same host port
        """
        return abs(hash(cls.__name__)) % 1000

    @classmethod
    def _get_compose_file(cls) -> str:
        return f"tmp.docker-compose.{cls.__name__}.yml"

    @classmethod
    def _get_service_name(cls) -> str:
        return "noesis_postgres"

    @classmethod
    def _get_db_env_path(cls) -> str:
        return f"tmp.{cls.__name__}.env"

    @classmethod
    def _get_postgres_db(cls) -> str:
        return "noesis_db_local"

    @pytest.fixture(autouse=True)
    def setup_teardown_test(self):
        """
        Setup and teardown for each test.
        """
        # Run before each test.
        self.set_up_test()
        yield

    def set_up_test(self) -> None:
        """
        Create the `noesis_*` schema (idempotent) and empty every table.
        """
        rnoposto.init_schema(self.connection)
        cursor = self.connection.cursor()
        for table_name in _NOESIS_TABLES:
            cursor.execute(f"DELETE FROM {table_name}")
        self.connection.commit()


# #############################################################################
# TestPostgresOrderBookStore
# #############################################################################


class TestPostgresOrderBookStore(_NoesisDbHelper):
    """
    Test `postgres_store.PostgresOrderBookStore` against a real Postgres.
    """

    def test1(self) -> None:
        """
        Test that `add_bid()`/`add_ask()` then `get_bids()`/`get_asks()`
        round-trip the same fields back.
        """
        # Prepare inputs.
        store = rnoposto.PostgresOrderBookStore(self.connection)
        bid = rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.99, 20.0)
        ask = rnbacaau.Ask("seller_1", 100, "frontier", 2.5, 0.98, 16.0)
        # Run test.
        store.add_bid(bid)
        store.add_ask(ask)
        # Check outputs.
        self.assert_equal(str(store.get_bids()), str([bid]))
        self.assert_equal(str(store.get_asks()), str([ask]))

    def test2(self) -> None:
        """
        Test that three bids added in a known order come back from
        `get_bids()` in that same order.
        """
        # Prepare inputs.
        store = rnoposto.PostgresOrderBookStore(self.connection)
        bids = [
            rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.99, 20.0),
            rnbacaau.Bid("buyer_2", 100, "frontier", 2.0, 0.99, 19.0),
            rnbacaau.Bid("buyer_3", 100, "frontier", 2.0, 0.99, 18.0),
        ]
        # Run test.
        for bid in bids:
            store.add_bid(bid)
        # Check outputs.
        self.assert_equal(str(store.get_bids()), str(bids))

    def test3(self) -> None:
        """
        Test that `clear()` empties both tables.
        """
        # Prepare inputs.
        store = rnoposto.PostgresOrderBookStore(self.connection)
        store.add_bid(rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.99, 20.0))
        store.add_ask(rnbacaau.Ask("seller_1", 100, "frontier", 2.5, 0.98, 16.0))
        # Run test.
        store.clear()
        # Check outputs.
        self.assert_equal(str(store.get_bids()), str([]))
        self.assert_equal(str(store.get_asks()), str([]))


# #############################################################################
# TestPostgresContractStore
# #############################################################################


class TestPostgresContractStore(_NoesisDbHelper):
    """
    Test `postgres_store.PostgresContractStore` against a real Postgres.
    """

    def test1(self) -> None:
        """
        Test that `save_contract()` returns an id that `get_contract()`
        resolves back to an equal `Contract`.
        """
        # Prepare inputs.
        store = rnoposto.PostgresContractStore(self.connection)
        contract = rnocodis.Contract(
            "buyer_1", "seller_1", 100, "frontier", 2.0, 0.999, 18.0, True
        )
        # Run test.
        contract_id = store.save_contract(contract)
        actual = store.get_contract(contract_id)
        # Check outputs.
        self.assert_equal(str(actual), str(contract))

    def test2(self) -> None:
        """
        Test that a fresh `PostgresContractStore` instance (same
        connection, new Python object, simulating a process restart) still
        resolves a `contract_id` saved by a prior instance.
        """
        # Prepare inputs.
        store1 = rnoposto.PostgresContractStore(self.connection)
        contract = rnocodis.Contract(
            "buyer_1", "seller_1", 100, "frontier", 2.0, 0.999, 18.0, True
        )
        contract_id = store1.save_contract(contract)
        # Run test.
        store2 = rnoposto.PostgresContractStore(self.connection)
        actual = store2.get_contract(contract_id)
        # Check outputs.
        self.assert_equal(str(actual), str(contract))

    def test3(self) -> None:
        """
        Test that `next_round_id()` called twice returns two different,
        increasing ids, and that `save_round()`/`get_latest_round()`
        round-trip a `RoundSummary`.
        """
        # Prepare inputs.
        store = rnoposto.PostgresContractStore(self.connection)
        # Run test.
        round_id_1 = store.next_round_id()
        round_id_2 = store.next_round_id()
        round_response = rnmaroru.RoundSummary(
            tier="frontier",
            round_id=round_id_2,
            clearing_price=18.0,
            matched_volume=100,
        )
        store.save_round(round_response)
        actual = store.get_latest_round("frontier")
        # Check outputs.
        self.assertLess(round_id_1, round_id_2)
        self.assert_equal(str(actual), str(round_response))

    def test4(self) -> None:
        """
        Test that `get_contract()` on an unknown id raises `AssertionError`.
        """
        # Prepare inputs.
        store = rnoposto.PostgresContractStore(self.connection)
        unknown_contract_id = 999999
        # Run test and check output.
        with self.assertRaises(AssertionError):
            store.get_contract(unknown_contract_id)


# #############################################################################
# TestPostgresRequestLogStore
# #############################################################################


class TestPostgresRequestLogStore(_NoesisDbHelper):
    """
    Test `postgres_store.PostgresRequestLogStore` against a real Postgres.
    """

    def test1(self) -> None:
        """
        Test that `append()` then `get_all()` round-trips the logged
        fields, with `request_id` assigned by the store.
        """
        # Prepare inputs.
        store = rnoposto.PostgresRequestLogStore(self.connection)
        # Run test.
        entry = store.append(
            "openai_mock", "gpt-mock", "hello", "world", 0.5, 0.01
        )
        actual = store.get_all()
        # Check outputs.
        self.assert_equal(str(actual), str([entry]))

    def test2(self) -> None:
        """
        Test that `query(provider=...)`/`query(model=...)` filter entries,
        matching `Gateway.query_log()`'s existing semantics.
        """
        # Prepare inputs.
        store = rnoposto.PostgresRequestLogStore(self.connection)
        store.append("provider_a", "model_a", "p1", "r1", 0.1, 0.01)
        entry_b = store.append("provider_b", "model_b", "p2", "r2", 0.2, 0.02)
        # Run test.
        actual_by_provider = store.query(provider="provider_b")
        actual_by_model = store.query(model="model_b")
        # Check outputs.
        self.assert_equal(str(actual_by_provider), str([entry_b]))
        self.assert_equal(str(actual_by_model), str([entry_b]))


# #############################################################################
# Test_init_schema
# #############################################################################


class Test_init_schema(_NoesisDbHelper):
    """
    Test `postgres_store.init_schema()`.
    """

    def test1(self) -> None:
        """
        Test that calling `init_schema()` twice on the same connection does
        not raise.
        """
        # Run test and check output: no exception raised.
        rnoposto.init_schema(self.connection)
        rnoposto.init_schema(self.connection)
