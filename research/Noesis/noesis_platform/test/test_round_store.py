"""
Test the market-owned round summary at the PostgreSQL adapter boundary.

Import as:

import research.Noesis.noesis_platform.test.test_round_store as rnplterost
"""

import dataclasses
import logging
import unittest.mock as umock

import pytest

pytest.importorskip("psycopg2")

import helpers.hunit_test as hunitest
import research.Noesis.market.round_runner as rnmaroru
import research.Noesis.noesis_platform.postgres_store as rnoposto

_LOG = logging.getLogger(__name__)


# #############################################################################
# TestPostgresContractStore
# #############################################################################


class TestPostgresContractStore(hunitest.TestCase):
    """
    Test plain round summaries using a mocked external database connection.
    """

    def test1(self) -> None:
        """
        Save and read a market summary without constructing any HTTP models.
        """
        # Prepare inputs: fake only the database boundary, not internal helpers.
        connection = umock.Mock(spec=["cursor", "commit"])
        cursor = connection.cursor.return_value
        summary = rnmaroru.RoundSummary("standard", 7, 3.5, 12)
        cursor.fetchone.return_value = ("standard", 7, 3.5, 12)
        store = rnoposto.PostgresContractStore(connection)
        # Prepare outputs.
        expected_parameters = [("standard", 7, 3.5, 12), ("standard",)]
        # Run test.
        store.save_round(summary)
        actual = store.get_latest_round(summary.tier)
        # Check outputs: persistence accepts and returns the library value type.
        self.assertIsInstance(store, rnmaroru.ContractStore)
        self.assertIs(type(actual), rnmaroru.RoundSummary)
        self.assert_equal(
            str(dataclasses.asdict(actual)), str(dataclasses.asdict(summary))
        )
        parameters = [call.args[1] for call in cursor.execute.call_args_list]
        self.assert_equal(str(parameters), str(expected_parameters))
        self.assertEqual(connection.commit.call_count, 1)
