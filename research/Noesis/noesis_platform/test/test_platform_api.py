"""
Import as:

import research.Noesis.noesis_platform.test.test_platform_api as rnttplap
"""

import logging
from typing import Optional, Tuple

import pytest

# `fastapi` is declared in `research/Noesis/requirements.txt`, not the repo's
# pinned `pip_list.txt`; skip cleanly instead of failing collection where
# it is not installed.
pytest.importorskip("fastapi")

import fastapi.testclient  # noqa: E402 # pylint: disable=wrong-import-position

import helpers.hunit_test as hunitest
import research.Noesis.gateway.passthrough_proxy as rnopapro
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.contract_dispatch as rnocodis
import research.Noesis.market.credit_ledger as rnmacrle
import research.Noesis.noesis_platform.platform_api as rnoplapi

_LOG = logging.getLogger(__name__)


# Fixed API key / account id shared across every test in this file.
_TEST_API_KEY = "test-api-key"
_TEST_ACCOUNT_ID = "account_1"


def _echo_call_func(model: str, prompt: str) -> str:
    """
    Stand-in provider `call_func` that echoes `model` and `prompt` back.

    :param model: model identifier passed to the provider
    :param prompt: prompt text passed to the provider
    :return: deterministic response text derived from `model`/`prompt`
    """
    return f"[{model}] {prompt}"


def _build_test_client(
    *,
    fulfillment_func: rnocodis.FulfillmentFunc = lambda _contract: True,
    credit_ledger: Optional[rnmacrle.CreditLedger] = None,
) -> Tuple[fastapi.testclient.TestClient, rnbacaau.OrderBook, rnopapro.Gateway]:
    """
    Build a `fastapi.testclient.TestClient` over a fresh `create_app()`.

    :param fulfillment_func: injected into `create_app()` so `POST
        /rounds/clear` dispatches deterministically
        - Default: report every contract as fulfilled
    :param credit_ledger: injected into `create_app()` to gate `POST /bids`
        - Default: `None`, no gating
    :return: `(test_client, order_book, gateway)`; `order_book`/`gateway`
        are returned so a test can inspect or register a provider on them
        directly, alongside driving the API
    """
    order_book = rnbacaau.OrderBook()
    gateway = rnopapro.Gateway(clock_func=iter([0.0, 1.0] * 20).__next__)
    api_keys = {_TEST_API_KEY: _TEST_ACCOUNT_ID}
    app = rnoplapi.create_app(
        order_book,
        gateway,
        api_keys,
        fulfillment_func=fulfillment_func,
        credit_ledger=credit_ledger,
    )
    test_client = fastapi.testclient.TestClient(app)
    return test_client, order_book, gateway


# #############################################################################
# Test_submit_bid
# #############################################################################


class Test_submit_bid(hunitest.TestCase):
    """
    Test `platform_api.create_app()`'s `POST /bids` endpoint.
    """

    def test1(self) -> None:
        """
        Test that a valid bid is accepted and queued on the order book.
        """
        # Prepare inputs.
        test_client, order_book, _ = _build_test_client()
        bid_payload = {
            "buyer_id": "buyer_1",
            "n_tasks": 100,
            "c_level_min": "frontier",
            "l_max": 2.0,
            "r_min": 0.999,
            "p_max": 20.0,
        }
        # Prepare outputs.
        expected_status_code = 201
        expected_pending_bids = [
            rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.999, 20.0)
        ]
        # Run test.
        response = test_client.post(
            "/bids",
            json=bid_payload,
            headers={rnoplapi.API_KEY_HEADER: _TEST_API_KEY},
        )
        # Check outputs.
        self.assertEqual(response.status_code, expected_status_code)
        self.assert_equal(str(response.json()), str(bid_payload))
        self.assert_equal(
            str(order_book.get_pending_bids()), str(expected_pending_bids)
        )

    def test2(self) -> None:
        """
        Test that an invalid bid (non-positive `n_tasks`) is rejected with
        an HTTP 400 instead of being queued.
        """
        # Prepare inputs.
        test_client, order_book, _ = _build_test_client()
        bid_payload = {
            "buyer_id": "buyer_1",
            "n_tasks": 0,
            "c_level_min": "frontier",
            "l_max": 2.0,
            "r_min": 0.999,
            "p_max": 20.0,
        }
        # Prepare outputs.
        expected_status_code = 400
        expected_pending_bids = []
        # Run test.
        response = test_client.post(
            "/bids",
            json=bid_payload,
            headers={rnoplapi.API_KEY_HEADER: _TEST_API_KEY},
        )
        # Check outputs.
        self.assertEqual(response.status_code, expected_status_code)
        self.assert_equal(
            str(order_book.get_pending_bids()), str(expected_pending_bids)
        )


# #############################################################################
# Test_submit_ask
# #############################################################################


class Test_submit_ask(hunitest.TestCase):
    """
    Test `platform_api.create_app()`'s `POST /asks` endpoint.
    """

    def test1(self) -> None:
        """
        Test that a valid ask is accepted and queued on the order book.
        """
        # Prepare inputs.
        test_client, order_book, _ = _build_test_client()
        ask_payload = {
            "seller_id": "seller_1",
            "n_tasks": 100,
            "c_level": "frontier",
            "l_typical": 2.5,
            "r_typical": 0.98,
            "p_min": 16.0,
        }
        # Prepare outputs.
        expected_status_code = 201
        expected_pending_asks = [
            rnbacaau.Ask("seller_1", 100, "frontier", 2.5, 0.98, 16.0)
        ]
        # Run test.
        response = test_client.post(
            "/asks",
            json=ask_payload,
            headers={rnoplapi.API_KEY_HEADER: _TEST_API_KEY},
        )
        # Check outputs.
        self.assertEqual(response.status_code, expected_status_code)
        self.assert_equal(str(response.json()), str(ask_payload))
        self.assert_equal(
            str(order_book.get_pending_asks()), str(expected_pending_asks)
        )

    def test2(self) -> None:
        """
        Test that an invalid ask (`r_typical` outside `[0, 1]`) is rejected
        with an HTTP 400 instead of being queued.
        """
        # Prepare inputs.
        test_client, order_book, _ = _build_test_client()
        ask_payload = {
            "seller_id": "seller_1",
            "n_tasks": 100,
            "c_level": "frontier",
            "l_typical": 2.5,
            "r_typical": 1.5,
            "p_min": 16.0,
        }
        # Prepare outputs.
        expected_status_code = 400
        expected_pending_asks = []
        # Run test.
        response = test_client.post(
            "/asks",
            json=ask_payload,
            headers={rnoplapi.API_KEY_HEADER: _TEST_API_KEY},
        )
        # Check outputs.
        self.assertEqual(response.status_code, expected_status_code)
        self.assert_equal(
            str(order_book.get_pending_asks()), str(expected_pending_asks)
        )


# #############################################################################
# Test_clear_round
# #############################################################################


class Test_clear_round(hunitest.TestCase):
    """
    Test `platform_api.create_app()`'s `POST /rounds/clear` and `GET
    /rounds/{tier}/latest` endpoints.
    """

    def test1(self) -> None:
        """
        Test that clearing a round with one crossing bid/ask returns the
        tier's clearing price and matched volume.
        """
        # Prepare inputs.
        test_client, _, _ = _build_test_client()
        headers = {rnoplapi.API_KEY_HEADER: _TEST_API_KEY}
        test_client.post(
            "/bids",
            json={
                "buyer_id": "buyer_1",
                "n_tasks": 100,
                "c_level_min": "frontier",
                "l_max": 2.0,
                "r_min": 0.999,
                "p_max": 20.0,
            },
            headers=headers,
        )
        test_client.post(
            "/asks",
            json={
                "seller_id": "seller_1",
                "n_tasks": 100,
                "c_level": "frontier",
                "l_typical": 2.5,
                "r_typical": 0.98,
                "p_min": 16.0,
            },
            headers=headers,
        )
        # Prepare outputs.
        expected = [
            {
                "tier": "frontier",
                "round_id": 0,
                "clearing_price": 18.0,
                "matched_volume": 100,
            }
        ]
        # Run test.
        response = test_client.post("/rounds/clear")
        # Check outputs.
        self.assertEqual(response.status_code, 200)
        self.assert_equal(str(response.json()), str(expected))

    def test2(self) -> None:
        """
        Test that `GET /rounds/{tier}/latest` returns an HTTP 400 before any
        round has cleared that tier.
        """
        # Prepare inputs.
        test_client, _, _ = _build_test_client()
        # Prepare outputs.
        expected_status_code = 400
        # Run test.
        response = test_client.get("/rounds/frontier/latest")
        # Check outputs.
        self.assertEqual(response.status_code, expected_status_code)

    def test3(self) -> None:
        """
        Test that `GET /rounds/{tier}/latest` returns the same outcome
        `POST /rounds/clear` reported for that tier.
        """
        # Prepare inputs.
        test_client, _, _ = _build_test_client()
        headers = {rnoplapi.API_KEY_HEADER: _TEST_API_KEY}
        test_client.post(
            "/bids",
            json={
                "buyer_id": "buyer_1",
                "n_tasks": 100,
                "c_level_min": "frontier",
                "l_max": 2.0,
                "r_min": 0.999,
                "p_max": 20.0,
            },
            headers=headers,
        )
        test_client.post(
            "/asks",
            json={
                "seller_id": "seller_1",
                "n_tasks": 100,
                "c_level": "frontier",
                "l_typical": 2.5,
                "r_typical": 0.98,
                "p_min": 16.0,
            },
            headers=headers,
        )
        clear_response = test_client.post("/rounds/clear")
        # Prepare outputs.
        expected = clear_response.json()[0]
        # Run test.
        response = test_client.get("/rounds/frontier/latest")
        # Check outputs.
        self.assertEqual(response.status_code, 200)
        self.assert_equal(str(response.json()), str(expected))

    def test4(self) -> None:
        """
        Test that a second `POST /rounds/clear` increments `round_id`, even
        though `clear_round()` empties the book between rounds.
        """
        # Prepare inputs.
        test_client, _, _ = _build_test_client()
        headers = {rnoplapi.API_KEY_HEADER: _TEST_API_KEY}
        bid_payload = {
            "buyer_id": "buyer_1",
            "n_tasks": 100,
            "c_level_min": "frontier",
            "l_max": 2.0,
            "r_min": 0.999,
            "p_max": 20.0,
        }
        ask_payload = {
            "seller_id": "seller_1",
            "n_tasks": 100,
            "c_level": "frontier",
            "l_typical": 2.5,
            "r_typical": 0.98,
            "p_min": 16.0,
        }
        # Prepare outputs.
        expected_first_round_id = 0
        expected_second_round_id = 1
        # Run test: one bid/ask, then a clear, per round.
        test_client.post("/bids", json=bid_payload, headers=headers)
        test_client.post("/asks", json=ask_payload, headers=headers)
        first_response = test_client.post("/rounds/clear")
        test_client.post("/bids", json=bid_payload, headers=headers)
        test_client.post("/asks", json=ask_payload, headers=headers)
        second_response = test_client.post("/rounds/clear")
        # Check outputs.
        self.assertEqual(
            first_response.json()[0]["round_id"], expected_first_round_id
        )
        self.assertEqual(
            second_response.json()[0]["round_id"], expected_second_round_id
        )


# #############################################################################
# Test_get_contract
# #############################################################################


class Test_get_contract(hunitest.TestCase):
    """
    Test `platform_api.create_app()`'s `GET /contracts/{contract_id}`
    endpoint.
    """

    def test1(self) -> None:
        """
        Test that a contract produced by `POST /rounds/clear` is readable
        by the `contract_id` assigned to it.
        """
        # Prepare inputs.
        test_client, _, _ = _build_test_client()
        headers = {rnoplapi.API_KEY_HEADER: _TEST_API_KEY}
        test_client.post(
            "/bids",
            json={
                "buyer_id": "buyer_1",
                "n_tasks": 100,
                "c_level_min": "frontier",
                "l_max": 2.0,
                "r_min": 0.999,
                "p_max": 20.0,
            },
            headers=headers,
        )
        test_client.post(
            "/asks",
            json={
                "seller_id": "seller_1",
                "n_tasks": 100,
                "c_level": "frontier",
                "l_typical": 2.5,
                "r_typical": 0.98,
                "p_min": 16.0,
            },
            headers=headers,
        )
        test_client.post("/rounds/clear")
        # Prepare outputs.
        expected = {
            "contract_id": 0,
            "buyer_id": "buyer_1",
            "seller_id": "seller_1",
            "n_tasks": 100,
            "c_level": "frontier",
            "l_max": 2.0,
            "r_min": 0.999,
            "price": 18.0,
            "fulfilled": True,
        }
        # Run test.
        response = test_client.get("/contracts/0")
        # Check outputs.
        self.assertEqual(response.status_code, 200)
        self.assert_equal(str(response.json()), str(expected))

    def test2(self) -> None:
        """
        Test that looking up an unknown `contract_id` returns an HTTP 400.
        """
        # Prepare inputs.
        test_client, _, _ = _build_test_client()
        # Prepare outputs.
        expected_status_code = 400
        # Run test.
        response = test_client.get("/contracts/999")
        # Check outputs.
        self.assertEqual(response.status_code, expected_status_code)

    def test3(self) -> None:
        """
        Test that a contract reported as unfulfilled by `fulfillment_func`
        round-trips `fulfilled=False`.
        """
        # Prepare inputs.
        test_client, _, _ = _build_test_client(
            fulfillment_func=lambda _contract: False
        )
        headers = {rnoplapi.API_KEY_HEADER: _TEST_API_KEY}
        test_client.post(
            "/bids",
            json={
                "buyer_id": "buyer_1",
                "n_tasks": 100,
                "c_level_min": "frontier",
                "l_max": 2.0,
                "r_min": 0.999,
                "p_max": 20.0,
            },
            headers=headers,
        )
        test_client.post(
            "/asks",
            json={
                "seller_id": "seller_1",
                "n_tasks": 100,
                "c_level": "frontier",
                "l_typical": 2.5,
                "r_typical": 0.98,
                "p_min": 16.0,
            },
            headers=headers,
        )
        test_client.post("/rounds/clear")
        # Prepare outputs.
        expected_fulfilled = False
        # Run test.
        response = test_client.get("/contracts/0")
        # Check outputs.
        self.assertEqual(response.json()["fulfilled"], expected_fulfilled)


# #############################################################################
# Test_create_completion
# #############################################################################


class Test_create_completion(hunitest.TestCase):
    """
    Test `platform_api.create_app()`'s `POST /completions` endpoint.
    """

    def test1(self) -> None:
        """
        Test that a completion request is routed to the registered provider
        and logged on the underlying `Gateway`.
        """
        # Prepare inputs.
        test_client, _, gateway = _build_test_client()
        gateway.register_provider(
            rnopapro.ProviderConfig("openai_mock", _echo_call_func, 0.01)
        )
        completion_payload = {
            "provider": "openai_mock",
            "model": "gpt-4o-mini",
            "prompt": "2+2=",
        }
        # Prepare outputs.
        expected = {
            "request_id": 0,
            "provider": "openai_mock",
            "model": "gpt-4o-mini",
            "response": "[gpt-4o-mini] 2+2=",
            "latency_in_secs": 1.0,
            "cost": 0.22,
        }
        # Run test.
        response = test_client.post(
            "/completions",
            json=completion_payload,
            headers={rnoplapi.API_KEY_HEADER: _TEST_API_KEY},
        )
        # Check outputs.
        self.assertEqual(response.status_code, 200)
        self.assert_equal(str(response.json()), str(expected))
        self.assertEqual(len(gateway.get_log()), 1)

    def test2(self) -> None:
        """
        Test that a completion request to an unregistered provider returns
        an HTTP 400 instead of a stack trace.
        """
        # Prepare inputs.
        test_client, _, _ = _build_test_client()
        completion_payload = {
            "provider": "unknown_provider",
            "model": "gpt-4o-mini",
            "prompt": "2+2=",
        }
        # Prepare outputs.
        expected_status_code = 400
        # Run test.
        response = test_client.post(
            "/completions",
            json=completion_payload,
            headers={rnoplapi.API_KEY_HEADER: _TEST_API_KEY},
        )
        # Check outputs.
        self.assertEqual(response.status_code, expected_status_code)


# #############################################################################
# Test_get_logs
# #############################################################################


class Test_get_logs(hunitest.TestCase):
    """
    Test `platform_api.create_app()`'s `GET /logs` endpoint.
    """

    def test1(self) -> None:
        """
        Test that `GET /logs` returns every completion logged so far, with
        no auth required.
        """
        # Prepare inputs.
        test_client, _, gateway = _build_test_client()
        gateway.register_provider(
            rnopapro.ProviderConfig("openai_mock", _echo_call_func, 0.0)
        )
        test_client.post(
            "/completions",
            json={
                "provider": "openai_mock",
                "model": "gpt-4o-mini",
                "prompt": "2+2=",
            },
            headers={rnoplapi.API_KEY_HEADER: _TEST_API_KEY},
        )
        # Prepare outputs.
        expected = [
            {
                "request_id": 0,
                "provider": "openai_mock",
                "model": "gpt-4o-mini",
                "prompt": "2+2=",
                "response": "[gpt-4o-mini] 2+2=",
                "latency_in_secs": 1.0,
                "cost": 0.0,
            }
        ]
        # Run test.
        response = test_client.get("/logs")
        # Check outputs.
        self.assertEqual(response.status_code, 200)
        self.assert_equal(str(response.json()), str(expected))

    def test2(self) -> None:
        """
        Test that `GET /logs?provider=...` filters logged entries by
        provider.
        """
        # Prepare inputs.
        test_client, _, gateway = _build_test_client()
        gateway.register_provider(
            rnopapro.ProviderConfig("openai_mock", _echo_call_func, 0.0)
        )
        gateway.register_provider(
            rnopapro.ProviderConfig("anthropic_mock", _echo_call_func, 0.0)
        )
        headers = {rnoplapi.API_KEY_HEADER: _TEST_API_KEY}
        test_client.post(
            "/completions",
            json={
                "provider": "openai_mock",
                "model": "gpt-4o-mini",
                "prompt": "a",
            },
            headers=headers,
        )
        test_client.post(
            "/completions",
            json={
                "provider": "anthropic_mock",
                "model": "claude-3-haiku",
                "prompt": "b",
            },
            headers=headers,
        )
        # Prepare outputs.
        expected_num_entries = 1
        expected_provider = "openai_mock"
        # Run test.
        response = test_client.get("/logs", params={"provider": "openai_mock"})
        # Check outputs.
        self.assertEqual(response.status_code, 200)
        entries = response.json()
        self.assertEqual(len(entries), expected_num_entries)
        self.assertEqual(entries[0]["provider"], expected_provider)


# #############################################################################
# Test_require_api_key
# #############################################################################


class Test_require_api_key(hunitest.TestCase):
    """
    Test the `X-API-Key` gate shared by `POST /bids`, `POST /asks`, and
    `POST /completions`.
    """

    def helper(self, path: str, payload: dict) -> None:
        """
        Assert that `path` rejects a missing and an invalid API key with an
        HTTP 401, without reaching the underlying library call.

        :param path: endpoint path to `POST` to
        :param payload: JSON body to send with the request
        """
        # Prepare inputs.
        test_client, order_book, gateway = _build_test_client()
        expected_status_code = 401
        # Run test: no `X-API-Key` header at all.
        response_missing_key = test_client.post(path, json=payload)
        # Run test: an `X-API-Key` header that is not a registered key.
        response_wrong_key = test_client.post(
            path, json=payload, headers={rnoplapi.API_KEY_HEADER: "wrong-key"}
        )
        # Check outputs.
        self.assertEqual(response_missing_key.status_code, expected_status_code)
        self.assertEqual(response_wrong_key.status_code, expected_status_code)
        self.assertEqual(order_book.get_pending_bids(), [])
        self.assertEqual(order_book.get_pending_asks(), [])
        self.assertEqual(gateway.get_log(), [])

    def test1(self) -> None:
        """
        Test that `POST /bids` is rejected without a valid API key.
        """
        # Prepare inputs.
        path = "/bids"
        payload = {
            "buyer_id": "buyer_1",
            "n_tasks": 100,
            "c_level_min": "frontier",
            "l_max": 2.0,
            "r_min": 0.999,
            "p_max": 20.0,
        }
        # Run test.
        self.helper(path, payload)

    def test2(self) -> None:
        """
        Test that `POST /asks` is rejected without a valid API key.
        """
        # Prepare inputs.
        path = "/asks"
        payload = {
            "seller_id": "seller_1",
            "n_tasks": 100,
            "c_level": "frontier",
            "l_typical": 2.5,
            "r_typical": 0.98,
            "p_min": 16.0,
        }
        # Run test.
        self.helper(path, payload)

    def test3(self) -> None:
        """
        Test that `POST /completions` is rejected without a valid API key.
        """
        # Prepare inputs.
        path = "/completions"
        payload = {
            "provider": "openai_mock",
            "model": "gpt-4o-mini",
            "prompt": "2+2=",
        }
        # Run test.
        self.helper(path, payload)


# #############################################################################
# Test_credit_gate
# #############################################################################


class Test_credit_gate(hunitest.TestCase):
    """
    Test `POST /bids` and `POST /rounds/clear` with a `CreditLedger`.
    """

    def helper(
        self, ledger: rnmacrle.CreditLedger, n_tasks: int, p_max: float
    ) -> Tuple[int, rnbacaau.OrderBook]:
        """
        Post one bid for `_TEST_ACCOUNT_ID` through an app gated by `ledger`.

        :param ledger: credit ledger injected into the app
        :param n_tasks: tasks in the bid
        :param p_max: most the buyer pays per task
        :return: `(status_code, order_book)`
        """
        test_client, order_book, _ = _build_test_client(credit_ledger=ledger)
        bid_payload = {
            "buyer_id": "buyer_1",
            "n_tasks": n_tasks,
            "c_level_min": "frontier",
            "l_max": 2.0,
            "r_min": 0.999,
            "p_max": p_max,
        }
        response = test_client.post(
            "/bids",
            json=bid_payload,
            headers={rnoplapi.API_KEY_HEADER: _TEST_API_KEY},
        )
        return response.status_code, order_book

    def test1(self) -> None:
        """
        Test that a bid the account can afford is accepted and the balance is
        untouched until the round clears.
        """
        # Prepare inputs.
        ledger = rnmacrle.CreditLedger(free_credit=100.0)
        n_tasks = 4
        p_max = 20.0
        # Prepare outputs.
        expected = "status=201 balance=100.0"
        # Run test.
        status_code, _ = self.helper(ledger, n_tasks, p_max)
        # Check outputs.
        balance = ledger.get_balance(_TEST_ACCOUNT_ID)
        actual = f"status={status_code} balance={balance}"
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test that a bid over the balance is rejected with an HTTP 400 before
        reaching the order book.
        """
        # Prepare inputs.
        ledger = rnmacrle.CreditLedger(free_credit=100.0)
        n_tasks = 100
        p_max = 20.0
        # Prepare outputs.
        expected = "status=400 pending_bids=[]"
        # Run test.
        status_code, order_book = self.helper(ledger, n_tasks, p_max)
        # Check outputs.
        pending = order_book.get_pending_bids()
        actual = f"status={status_code} pending_bids={pending}"
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Test that clearing debits only the filled tasks, at the clearing
        price, and that a later empty clear leaves the balance unchanged.
        """
        # Prepare inputs.
        ledger = rnmacrle.CreditLedger(free_credit=100.0)
        test_client, _, _ = _build_test_client(credit_ledger=ledger)
        headers = {rnoplapi.API_KEY_HEADER: _TEST_API_KEY}
        test_client.post(
            "/bids",
            json={
                "buyer_id": "buyer_1",
                "n_tasks": 4,
                "c_level_min": "frontier",
                "l_max": 2.0,
                "r_min": 0.999,
                "p_max": 20.0,
            },
            headers=headers,
        )
        test_client.post(
            "/asks",
            json={
                "seller_id": "seller_1",
                "n_tasks": 3,
                "c_level": "frontier",
                "l_typical": 2.5,
                "r_typical": 0.98,
                "p_min": 16.0,
            },
            headers=headers,
        )
        # Prepare outputs.
        # 3 of 4 tasks fill at the midpoint price 18.0: debit 54.0.
        expected = "after_clear=46.0 after_second_clear=46.0"
        # Run test.
        test_client.post("/rounds/clear")
        after_clear = ledger.get_balance(_TEST_ACCOUNT_ID)
        test_client.post("/rounds/clear")
        after_second_clear = ledger.get_balance(_TEST_ACCOUNT_ID)
        # Check outputs.
        actual = (
            f"after_clear={after_clear} after_second_clear={after_second_clear}"
        )
        self.assert_equal(actual, expected)

    def test4(self) -> None:
        """
        Test that a `buyer_id` already bound to another account in the round
        is rejected with an HTTP 400.
        """
        # Prepare inputs.
        ledger = rnmacrle.CreditLedger(free_credit=100.0)
        order_book = rnbacaau.OrderBook()
        gateway = rnopapro.Gateway()
        api_keys = {"key_a": "acct_a", "key_b": "acct_b"}
        app = rnoplapi.create_app(
            order_book, gateway, api_keys, credit_ledger=ledger
        )
        test_client = fastapi.testclient.TestClient(app)
        bid_payload = {
            "buyer_id": "buyer_1",
            "n_tasks": 1,
            "c_level_min": "frontier",
            "l_max": 2.0,
            "r_min": 0.999,
            "p_max": 1.0,
        }
        # Prepare outputs.
        expected = "first=201 second=400 pending_bids=1"
        # Run test.
        first = test_client.post(
            "/bids",
            json=bid_payload,
            headers={rnoplapi.API_KEY_HEADER: "key_a"},
        )
        second = test_client.post(
            "/bids",
            json=bid_payload,
            headers={rnoplapi.API_KEY_HEADER: "key_b"},
        )
        # Check outputs.
        pending = len(order_book.get_pending_bids())
        actual = (
            f"first={first.status_code} second={second.status_code}"
            f" pending_bids={pending}"
        )
        self.assert_equal(actual, expected)


# #############################################################################
# Test_health
# #############################################################################


class Test_health(hunitest.TestCase):
    """
    Test `platform_api.create_app()`'s `GET /health` endpoint.
    """

    def test1(self) -> None:
        """
        Test that `GET /health` returns `200` and `{"status": "ok"}` without
        an `X-API-Key` header.
        """
        # Prepare inputs.
        test_client, _, _ = _build_test_client()
        # Prepare outputs.
        expected_status_code = 200
        expected_body = {"status": "ok"}
        # Run test.
        response = test_client.get("/health")
        # Check outputs.
        self.assertEqual(response.status_code, expected_status_code)
        self.assertEqual(response.json(), expected_body)
