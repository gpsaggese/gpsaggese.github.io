import logging
from typing import Dict, Optional

import helpers.hunit_test as hunitest
import research.Noesis.gateway_routing as rnogarou
import research.Noesis.test.gateway_test_utils as rnteteut

_LOG = logging.getLogger(__name__)


def _expect_error(
    self_: hunitest.TestCase,
    exc: rnogarou.RoutingError,
    http_status: int,
    code: str,
) -> None:
    """
    Check a `RoutingError`'s HTTP status and code.
    """
    self_.assertEqual((exc.http_status, exc.code), (http_status, code))


async def _world(pool) -> Dict[str, int]:
    """
    Create two buyers, an operator, one seller, and contracts in several states.

    :return: contract ids by name
    """
    await rnteteut.add_account(pool, "buyer_1", "buyer", "key-buyer-1")
    await rnteteut.add_account(pool, "buyer_2", "buyer", "key-buyer-2")
    await rnteteut.add_account(pool, "operator", "operator", "key-op")
    await rnteteut.add_seller(pool, "groq", "groq", "Groq")
    ids = {
        "active": await rnteteut.add_contract(pool, "buyer_1", "groq", l_max=1.5),
        "pending": await rnteteut.add_contract(
            pool, "buyer_1", "groq", state="PENDING"
        ),
        "passed": await rnteteut.add_contract(
            pool, "buyer_1", "groq", state="PASSED"
        ),
        "other_buyer": await rnteteut.add_contract(pool, "buyer_2", "groq"),
    }
    return ids


# #############################################################################
# Test_parse_bearer
# #############################################################################


class Test_parse_bearer(hunitest.TestCase):
    """
    Test extracting the API key from the `Authorization` header.
    """

    def test1(self) -> None:
        """
        Test well-formed headers.
        """
        # Run test and check outputs.
        self.assertEqual(rnogarou.parse_bearer("Bearer abc123"), "abc123")
        self.assertEqual(rnogarou.parse_bearer("Bearer   padded  "), "padded")

    def test2(self) -> None:
        """
        Test that missing or malformed headers are 401.
        """
        # Prepare inputs.
        headers = [None, "", "abc123", "Basic abc123", "Bearer ", "bearer abc"]
        # Run test and check outputs.
        for header in headers:
            with self.assertRaises(rnogarou.RoutingError) as cm:
                rnogarou.parse_bearer(header)
            _expect_error(self, cm.exception, 401, "invalid_api_key")


# #############################################################################
# Test_parse_contract_id
# #############################################################################


class Test_parse_contract_id(hunitest.TestCase):
    """
    Test parsing the `X-Noesis-Contract` header.
    """

    def test1(self) -> None:
        """
        Test well-formed ids.
        """
        # Run test and check outputs.
        self.assertEqual(rnogarou.parse_contract_id("42"), 42)
        self.assertEqual(rnogarou.parse_contract_id(" 7 "), 7)

    def test2(self) -> None:
        """
        Test that missing or invalid ids are 400 with the right code.
        """
        # Prepare inputs.
        cases: Dict[Optional[str], str] = {
            None: "missing_contract",
            "": "missing_contract",
            "   ": "missing_contract",
            "abc": "invalid_contract",
            "4.2": "invalid_contract",
            "0": "invalid_contract",
            "-5": "invalid_contract",
        }
        # Run test and check outputs.
        for header, code in cases.items():
            with self.assertRaises(rnogarou.RoutingError) as cm:
                rnogarou.parse_contract_id(header)
            _expect_error(self, cm.exception, 400, code)


# #############################################################################
# Test_authenticate
# #############################################################################


class Test_authenticate(hunitest.TestCase):
    """
    Test API-key lookup and account-kind checks.
    """

    def test1(self) -> None:
        """
        Test the right key and kind, a wrong key, and a wrong kind.
        """

        async def _body(pool) -> None:
            # Prepare inputs.
            await _world(pool)
            # Run test.
            buyer = await rnogarou.authenticate(pool, "key-buyer-1", "buyer")
            operator = await rnogarou.authenticate(pool, "key-op", "operator")
            # Check outputs.
            self.assertEqual(buyer, rnogarou.Account("buyer_1", "buyer"))
            self.assertEqual(operator.account_id, "operator")
            cases = [
                ("wrong-key", "buyer", 401, "invalid_api_key"),
                ("key-buyer-1", "operator", 403, "wrong_account_kind"),
                ("key-op", "buyer", 403, "wrong_account_kind"),
            ]
            for key, kind, http_status, code in cases:
                with self.assertRaises(rnogarou.RoutingError) as cm:
                    await rnogarou.authenticate(pool, key, kind)
                _expect_error(self, cm.exception, http_status, code)

        rnteteut.run_with_clean_db(self, _body)


# #############################################################################
# Test_resolve_route
# #############################################################################


class Test_resolve_route(hunitest.TestCase):
    """
    Test contract -> seller -> provider resolution.
    """

    def test1(self) -> None:
        """
        Test that the buyer's ACTIVE contract resolves to the seller's route.
        """

        async def _body(pool) -> None:
            # Prepare inputs.
            ids = await _world(pool)
            # Prepare outputs.
            expected = rnogarou.Route(
                buyer_id="buyer_1",
                contract_id=ids["active"],
                seller_id="groq",
                provider_slug="groq",
                provider_name="Groq",
                l_max=1.5,
            )
            # Run test.
            actual = await rnogarou.resolve_route(pool, "buyer_1", ids["active"])
            # Check outputs.
            self.assert_equal(str(actual), str(expected))

        rnteteut.run_with_clean_db(self, _body)

    def test2(self) -> None:
        """
        Test unknown (404), someone else's (404, same as unknown), and
        non-ACTIVE (409) contracts.
        """

        async def _body(pool) -> None:
            # Prepare inputs.
            ids = await _world(pool)
            cases = [
                (ids["active"] + 1000, 404, "contract_not_found"),
                (ids["other_buyer"], 404, "contract_not_found"),
                (ids["pending"], 409, "contract_not_active"),
                (ids["passed"], 409, "contract_not_active"),
            ]
            # Run test and check outputs.
            for contract_id, http_status, code in cases:
                with self.assertRaises(rnogarou.RoutingError) as cm:
                    await rnogarou.resolve_route(pool, "buyer_1", contract_id)
                _expect_error(self, cm.exception, http_status, code)

        rnteteut.run_with_clean_db(self, _body)

    def test3(self) -> None:
        """
        Test that the not-found message doesn't reveal the real owner.
        """

        async def _body(pool) -> None:
            # Prepare inputs.
            ids = await _world(pool)
            # Prepare outputs.
            expected = f"Contract {ids['other_buyer']} not found"
            # Run test.
            with self.assertRaises(rnogarou.RoutingError) as cm:
                await rnogarou.resolve_route(pool, "buyer_1", ids["other_buyer"])
            # Check outputs.
            self.assert_equal(cm.exception.message, expected)

        rnteteut.run_with_clean_db(self, _body)
