import logging
import os

import helpers.hunit_test as hunitest
import research.Noesis.noesis_seed as rnonosee
import research.Noesis.test.gateway_test_utils as rnteteut

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_hash_api_key
# #############################################################################


class Test_hash_api_key(hunitest.TestCase):
    """
    Test API-key hashing.
    """

    def test1(self) -> None:
        """
        Test that hashing is deterministic and never returns the raw key.
        """
        # Run test.
        actual = rnonosee.hash_api_key("secret")
        # Check outputs.
        self.assertEqual(actual, rnonosee.hash_api_key("secret"))
        self.assertNotEqual(actual, "secret")
        self.assertEqual(len(actual), 64)


# #############################################################################
# Test_load_seed
# #############################################################################


class Test_load_seed(hunitest.TestCase):
    """
    Test loading sellers and accounts from the seed file.
    """

    def test1(self) -> None:
        """
        Test that the real seed file loads 5 sellers and hashed keys.
        """
        # Prepare outputs.
        expected_sellers = [
            ("coreweave", "coreweave/fp16", "CoreWeave", "eligible", 0.8),
            ("deepinfra", "deepinfra/turbo", "DeepInfra", "eligible", 0.8),
            ("groq", "groq", "Groq", "eligible", 0.8),
            ("novita", "novita/bf16", "Novita", "eligible", 0.8),
            ("together", "together", "Together", "eligible", 0.8),
        ]

        async def _body(pool) -> None:
            # Run test.
            result = await rnonosee.load_seed(
                pool, rnteteut.get_test_settings(), rnteteut.SEED_ENV
            )
            sellers = await rnteteut.fetch_all(
                pool,
                "SELECT seller_id, provider_slug, provider_name, state,"
                " reputation FROM sellers ORDER BY seller_id",
            )
            keys = dict(
                await rnteteut.fetch_all(
                    pool, "SELECT account_id, api_key_hash FROM accounts"
                )
            )
            # Check outputs.
            self.assertEqual((result.accounts, result.sellers), (2, 5))
            self.assert_equal(str(sellers), str(expected_sellers))
            self.assertEqual(
                keys["buyer_1"], rnonosee.hash_api_key(rnteteut.BUYER_KEY)
            )
            self.assertNotIn(rnteteut.BUYER_KEY, keys.values())
            self.assertIn("seller_groq", keys)

        rnteteut.run_with_clean_db(self, _body)

    def test2(self) -> None:
        """
        Test that re-seeding keeps reputation/state and rotates keys.
        """

        async def _body(pool) -> None:
            # Prepare inputs.
            settings = rnteteut.get_test_settings()
            await rnonosee.load_seed(pool, settings, rnteteut.SEED_ENV)
            async with pool.connection() as conn:
                await conn.execute(
                    "UPDATE sellers SET reputation = 0.3, state = 'blocked' "
                    "WHERE seller_id = 'groq'"
                )
            new_env = {**rnteteut.SEED_ENV, "NOESIS_BUYER_1_KEY": "new-secret"}
            # Run test.
            await rnonosee.load_seed(pool, settings, new_env)
            seller = await rnteteut.fetch_all(
                pool,
                "SELECT reputation, state FROM sellers WHERE seller_id = 'groq'",
            )
            key = await rnteteut.fetch_all(
                pool,
                "SELECT api_key_hash FROM accounts WHERE account_id = 'buyer_1'",
            )
            # Check outputs.
            self.assertEqual(seller, [(0.3, "blocked")])
            self.assertEqual(key, [(rnonosee.hash_api_key("new-secret"),)])

        rnteteut.run_with_clean_db(self, _body)

    def test3(self) -> None:
        """
        Test that a missing key env var fails loudly, naming it.
        """

        async def _body(pool) -> None:
            # Run test.
            with self.assertRaises(AssertionError) as cm:
                await rnonosee.load_seed(
                    pool,
                    rnteteut.get_test_settings(),
                    {"NOESIS_BUYER_1_KEY": "x"},
                )
            # Check outputs.
            self.assertIn("NOESIS_OPERATOR_KEY", str(cm.exception))

        rnteteut.run_with_clean_db(self, _body)

    def test4(self) -> None:
        """
        Test that an unknown account kind is rejected.
        """
        # Prepare inputs.
        seed_path = os.path.join(self.get_scratch_space(), "seed.yaml")
        with open(seed_path, "w") as f:
            f.write(
                "sellers: []\naccounts:\n"
                "  - {account_id: x, kind: admin,"
                " api_key_env: NOESIS_BUYER_1_KEY}\n"
            )

        async def _body(pool) -> None:
            # Run test and check outputs.
            with self.assertRaises(AssertionError):
                await rnonosee.load_seed(
                    pool,
                    rnteteut.get_test_settings(seed_path=seed_path),
                    rnteteut.SEED_ENV,
                )

        rnteteut.run_with_clean_db(self, _body)
