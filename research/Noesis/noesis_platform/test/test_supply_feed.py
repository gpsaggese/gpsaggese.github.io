"""
Import as:

import research.Noesis.noesis_platform.test.test_supply_feed as rnptsufe
"""

import logging
import os
import pprint
from typing import Any, Dict, List

import httpx

import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hunit_test as hunitest
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.noesis_platform.supply_feed as rnplsufe

_LOG = logging.getLogger(__name__)

_MODEL = "meta-llama/llama-3.3-70b-instruct"


def _get_catalog() -> Dict[str, Any]:
    """
    Return a two-endpoint OpenRouter catalog for the served model.
    """
    catalog = {
        "data": {
            "id": _MODEL,
            "endpoints": [
                {
                    "tag": "groq",
                    "pricing": {
                        "prompt": "0.00000059",
                        "completion": "0.00000079",
                    },
                },
                {
                    "tag": "deepinfra/turbo",
                    "pricing": {
                        "prompt": "0.0000001",
                        "completion": "0.00000032",
                    },
                },
            ],
        }
    }
    return catalog


def _get_mock_client(
    sent: List[httpx.Request], body: Dict[str, Any], *, status_code: int = 200
) -> httpx.Client:
    """
    Return a client that records each request in `sent` and answers `body`.

    :param sent: list the handled requests are appended to
    :param body: JSON body every response carries
    :param status_code: HTTP status of every response
    :return: client over `httpx.MockTransport`; no network
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status_code, json=body)

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    return client


# #############################################################################
# Test_load_seller_supply
# #############################################################################


class Test_load_seller_supply(hunitest.TestCase):
    """
    Test `supply_feed.load_seller_supply()`.
    """

    def test1(self) -> None:
        """
        Test that the real seed file yields the 5 demo sellers with their
        typical latency and reliability.
        """
        # Prepare inputs.
        seed_path = rnplsufe.DEFAULT_SEED_PATH
        # Prepare outputs.
        expected = """
        [SellerSupply(seller_id='coreweave',
                      provider_slug='coreweave/fp16',
                      l_typical=0.26,
                      r_typical=1.0),
         SellerSupply(seller_id='groq',
                      provider_slug='groq',
                      l_typical=0.57,
                      r_typical=1.0),
         SellerSupply(seller_id='deepinfra',
                      provider_slug='deepinfra/turbo',
                      l_typical=1.04,
                      r_typical=1.0),
         SellerSupply(seller_id='together',
                      provider_slug='together',
                      l_typical=1.3,
                      r_typical=1.0),
         SellerSupply(seller_id='novita',
                      provider_slug='novita/bf16',
                      l_typical=2.39,
                      r_typical=1.0)]
        """
        # Run test.
        actual = pprint.pformat(rnplsufe.load_seller_supply(seed_path))
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test that a seller entry without `l_typical` raises `AssertionError`.
        """
        # Prepare inputs.
        seed_path = os.path.join(self.get_scratch_space(), "seed.yaml")
        content = """
        sellers:
          - {seller_id: groq, provider_slug: groq, r_typical: 1.0}
        """
        hio.to_file(seed_path, hprint.dedent(content))
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnplsufe.load_seller_supply(seed_path)


# #############################################################################
# Test_fetch_endpoint_catalog
# #############################################################################


class Test_fetch_endpoint_catalog(hunitest.TestCase):
    """
    Test `supply_feed.fetch_endpoint_catalog()`.
    """

    def helper(self, api_key: str, expected: str) -> None:
        """
        Fetch the catalog through a mock client and compare the request that
        was sent and the endpoints that came back with `expected`.

        :param api_key: OpenRouter API key passed to the fetch
        :param expected: expected `url=`, `authorization=`, `tags=` lines
        """
        # Prepare inputs.
        sent: List[httpx.Request] = []
        http = _get_mock_client(sent, _get_catalog())
        # Run test.
        catalog = rnplsufe.fetch_endpoint_catalog(http, _MODEL, api_key=api_key)
        http.close()
        # Check outputs.
        (request,) = sent
        tags = [endpoint["tag"] for endpoint in catalog["data"]["endpoints"]]
        actual = "\n".join(
            [
                f"url={request.url}",
                f"authorization={request.headers.get('Authorization')}",
                f"tags={tags}",
            ]
        )
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test that the catalog is fetched from the model's endpoints URL with
        the key as a bearer token.
        """
        # Prepare inputs.
        api_key = "sk-test"
        # Prepare outputs.
        expected = f"""
        url=https://openrouter.ai/api/v1/models/{_MODEL}/endpoints
        authorization=Bearer sk-test
        tags=['groq', 'deepinfra/turbo']
        """
        # Run test and check outputs.
        self.helper(api_key, expected)

    def test2(self) -> None:
        """
        Test that no `Authorization` header is sent without a key.
        """
        # Prepare inputs.
        api_key = ""
        # Prepare outputs.
        expected = f"""
        url=https://openrouter.ai/api/v1/models/{_MODEL}/endpoints
        authorization=None
        tags=['groq', 'deepinfra/turbo']
        """
        # Run test and check outputs.
        self.helper(api_key, expected)

    def test3(self) -> None:
        """
        Test that an OpenRouter error response raises instead of returning a
        catalog, so no asks are built from it.
        """
        # Prepare inputs.
        sent: List[httpx.Request] = []
        body = {"error": {"message": "service unavailable"}}
        http = _get_mock_client(sent, body, status_code=503)
        # Run test and check outputs.
        with self.assertRaises(httpx.HTTPStatusError):
            rnplsufe.fetch_endpoint_catalog(http, _MODEL)
        http.close()
        self.assertEqual(len(sent), 1)

    def test4(self) -> None:
        """
        Test that a network failure reaching OpenRouter propagates as an
        `httpx.TransportError`.
        """

        # Prepare inputs.
        def _handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        http = httpx.Client(transport=httpx.MockTransport(_handler))
        # Run test and check outputs.
        with self.assertRaises(httpx.TransportError):
            rnplsufe.fetch_endpoint_catalog(http, _MODEL)
        http.close()


# #############################################################################
# Test_refresh_market_supply
# #############################################################################


class Test_refresh_market_supply(hunitest.TestCase):
    """
    Test `supply_feed.refresh_market_supply()`.
    """

    def test1(self) -> None:
        """
        Test that the listed sellers of the seed file are queued as asks with
        the default tier and task count.
        """
        # Prepare inputs.
        order_book = rnbacaau.OrderBook()
        catalog = _get_catalog()
        sellers = rnplsufe.load_seller_supply(rnplsufe.DEFAULT_SEED_PATH)
        # Prepare outputs.
        expected = """
        [Ask(seller_id='groq',
             n_tasks=60,
             c_level='standard',
             l_typical=0.57,
             r_typical=1.0,
             p_min=0.00079224),
         Ask(seller_id='deepinfra',
             n_tasks=60,
             c_level='standard',
             l_typical=1.04,
             r_typical=1.0,
             p_min=0.00018192)]
        """
        # Run test.
        submitted = rnplsufe.refresh_market_supply(order_book, catalog, sellers)
        pending = order_book.get_pending_asks()
        # Check outputs.
        actual = pprint.pformat(submitted)
        self.assert_equal(actual, expected, dedent=True)
        # The queued asks are exactly the ones returned.
        self.assert_equal(str(pending), str(submitted))
