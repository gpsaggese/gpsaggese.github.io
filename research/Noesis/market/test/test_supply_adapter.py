"""
Import as:

import research.Noesis.market.test.test_supply_adapter as rnmtsuad
"""

import logging
import pprint
from typing import Any, Dict, List

import helpers.hunit_test as hunitest
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.supply_adapter as rnmasuad

_LOG = logging.getLogger(__name__)


def _get_catalog() -> Dict[str, Any]:
    """
    Return a small OpenRouter endpoint catalog in the shape of
    `GET /api/v1/models/{model}/endpoints`.

    Prices are the ones OpenRouter quoted for Llama 3.3 70B; `cloudflare/fp8`
    belongs to no seller and `google-vertex` has no prices.
    """
    catalog = {
        "data": {
            "id": "meta-llama/llama-3.3-70b-instruct",
            "endpoints": [
                {
                    "tag": "groq",
                    "provider_name": "Groq",
                    "pricing": {
                        "prompt": "0.00000059",
                        "completion": "0.00000079",
                    },
                },
                {
                    "tag": "coreweave/fp16",
                    "provider_name": "CoreWeave",
                    "pricing": {
                        "prompt": "0.00000071",
                        "completion": "0.00000071",
                    },
                },
                {
                    "tag": "cloudflare/fp8",
                    "provider_name": "Cloudflare",
                    "pricing": {
                        "prompt": "0.000000293",
                        "completion": "0.000002253",
                    },
                },
                {
                    "tag": "google-vertex",
                    "provider_name": "Google",
                    "pricing": {},
                },
            ],
        }
    }
    return catalog


def _get_sellers() -> List[rnmasuad.SellerSupply]:
    """
    Return three sellers: two listed in the catalog, one (`novita`) not.
    """
    sellers = [
        rnmasuad.SellerSupply("groq", "groq", 0.57, 1.0),
        rnmasuad.SellerSupply("coreweave", "coreweave/fp16", 0.26, 1.0),
        rnmasuad.SellerSupply("novita", "novita/bf16", 2.39, 1.0),
    ]
    return sellers


def _build_asks(
    catalog: Dict[str, Any], sellers: List[rnmasuad.SellerSupply]
) -> List[rnbacaau.Ask]:
    """
    Build the asks of `sellers` from `catalog` with the MVP tier and task
    count.

    :param catalog: OpenRouter endpoint catalog
    :param sellers: sellers of the market
    :return: asks from `build_asks()`
    """
    quotes = rnmasuad.parse_endpoint_quotes(catalog)
    n_tasks = 60
    c_level = "standard"
    asks = rnmasuad.build_asks(quotes, sellers, n_tasks=n_tasks, c_level=c_level)
    return asks


# #############################################################################
# Test_parse_endpoint_quotes
# #############################################################################


class Test_parse_endpoint_quotes(hunitest.TestCase):
    """
    Test `supply_adapter.parse_endpoint_quotes()`.
    """

    def helper(self, catalog: Dict[str, Any], expected: str) -> None:
        """
        Parse the quotes of `catalog` and compare the pretty-printed result
        with `expected`.

        :param catalog: OpenRouter endpoint catalog
        :param expected: expected pretty-printed quotes
        """
        # Run test.
        quotes = rnmasuad.parse_endpoint_quotes(catalog)
        # Check outputs.
        actual = pprint.pformat(quotes)
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test that priced endpoints become quotes and unpriced ones are skipped.
        """
        # Prepare inputs.
        catalog = _get_catalog()
        # Prepare outputs.
        expected = """
        [EndpointQuote(provider_slug='groq',
                       prompt_price=5.9e-07,
                       completion_price=7.9e-07),
         EndpointQuote(provider_slug='coreweave/fp16',
                       prompt_price=7.1e-07,
                       completion_price=7.1e-07),
         EndpointQuote(provider_slug='cloudflare/fp8',
                       prompt_price=2.93e-07,
                       completion_price=2.253e-06)]
        """
        # Run test and check outputs.
        self.helper(catalog, expected)

    def test2(self) -> None:
        """
        Test that a catalog with no endpoints yields no quotes.
        """
        # Prepare inputs.
        catalog = {"data": {"endpoints": []}}
        # Prepare outputs.
        expected = "[]"
        # Run test and check outputs.
        self.helper(catalog, expected)


# #############################################################################
# Test_compute_task_price
# #############################################################################


class Test_compute_task_price(hunitest.TestCase):
    """
    Test `supply_adapter.compute_task_price()`.
    """

    def helper(
        self,
        prompt_price: float,
        completion_price: float,
        margin: float,
        expected: float,
    ) -> None:
        """
        Price one task and compare with `expected`.

        :param prompt_price: price of one prompt token
        :param completion_price: price of one completion token
        :param margin: fraction added on top of the token cost
        :param expected: expected price per task
        """
        # Run test.
        actual = rnmasuad.compute_task_price(
            prompt_price, completion_price, margin=margin
        )
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test1(self) -> None:
        """
        Test the worst-case task cost at the PRD D1 caps: 1,000 prompt and 256
        completion tokens.
        """
        # Prepare inputs.
        prompt_price = 5.9e-07
        completion_price = 7.9e-07
        margin = 0.0
        # Prepare outputs.
        expected = 0.00079224
        # Run test and check outputs.
        self.helper(prompt_price, completion_price, margin, expected)

    def test2(self) -> None:
        """
        Test that the margin scales the token cost.
        """
        # Prepare inputs.
        prompt_price = 1e-07
        completion_price = 1e-07
        margin = 0.5
        # Prepare outputs.
        expected = 0.0001884
        # Run test and check outputs.
        self.helper(prompt_price, completion_price, margin, expected)


# #############################################################################
# Test_build_asks
# #############################################################################


class Test_build_asks(hunitest.TestCase):
    """
    Test `supply_adapter.build_asks()`.
    """

    def helper(
        self,
        catalog: Dict[str, Any],
        sellers: List[rnmasuad.SellerSupply],
        expected: str,
    ) -> None:
        """
        Build the asks of `sellers` from `catalog` and compare the
        pretty-printed result with `expected`.

        :param catalog: OpenRouter endpoint catalog
        :param sellers: sellers of the market
        :param expected: expected pretty-printed asks
        """
        # Run test.
        asks = _build_asks(catalog, sellers)
        # Check outputs.
        actual = pprint.pformat(asks)
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test that each listed seller gets one ask with the expected tier and
        price, and that the endpoint of no seller is ignored.
        """
        # Prepare inputs.
        catalog = _get_catalog()
        sellers = _get_sellers()[:2]
        # Prepare outputs.
        expected = """
        [Ask(seller_id='groq',
             n_tasks=60,
             c_level='standard',
             l_typical=0.57,
             r_typical=1.0,
             p_min=0.00079224),
         Ask(seller_id='coreweave',
             n_tasks=60,
             c_level='standard',
             l_typical=0.26,
             r_typical=1.0,
             p_min=0.00089176)]
        """
        # Run test and check outputs.
        self.helper(catalog, sellers, expected)

    def test2(self) -> None:
        """
        Test that a seller whose endpoint is not in the catalog gets no ask,
        so a removed model's supply is not renewed.
        """
        # Prepare inputs.
        catalog = _get_catalog()
        sellers = _get_sellers()[2:]
        # Prepare outputs.
        expected = "[]"
        # Run test and check outputs.
        self.helper(catalog, sellers, expected)

    def test3(self) -> None:
        """
        Test that a tag listed twice in the catalog prices one ask, from its
        first quote.
        """
        # Prepare inputs.
        catalog = _get_catalog()
        duplicate = dict(catalog["data"]["endpoints"][0])
        duplicate["pricing"] = {"prompt": "0.000001", "completion": "0.000001"}
        catalog["data"]["endpoints"].append(duplicate)
        sellers = _get_sellers()[:1]
        # Prepare outputs.
        expected = """
        [Ask(seller_id='groq',
             n_tasks=60,
             c_level='standard',
             l_typical=0.57,
             r_typical=1.0,
             p_min=0.00079224)]
        """
        # Run test and check outputs.
        self.helper(catalog, sellers, expected)


# #############################################################################
# Test_submit_supply
# #############################################################################


class Test_submit_supply(hunitest.TestCase):
    """
    Test `supply_adapter.submit_supply()`.
    """

    def test1(self) -> None:
        """
        Test that refreshing twice in one round does not accumulate duplicate
        supply: the second refresh queues nothing.
        """
        # Prepare inputs.
        order_book = rnbacaau.OrderBook()
        asks = _build_asks(_get_catalog(), _get_sellers())
        # Prepare outputs.
        expected = """
        first=['groq', 'coreweave']
        second=[]
        pending=['groq', 'coreweave']
        """
        # Run test.
        first = rnmasuad.submit_supply(order_book, asks)
        second = rnmasuad.submit_supply(order_book, asks)
        pending = order_book.get_pending_asks()
        # Check outputs.
        actual = "\n".join(
            [
                f"first={[ask.seller_id for ask in first]}",
                f"second={[ask.seller_id for ask in second]}",
                f"pending={[ask.seller_id for ask in pending]}",
            ]
        )
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test that a buyer bid clears against adapter-created supply: the
        cheapest seller fills the bid, and the next round starts empty.
        """
        # Prepare inputs.
        order_book = rnbacaau.OrderBook()
        asks = _build_asks(_get_catalog(), _get_sellers())
        rnmasuad.submit_supply(order_book, asks)
        buyer_id = "buyer_1"
        n_tasks = 50
        c_level_min = "standard"
        l_max = 2.0
        r_min = 0.9
        p_max = 0.001
        bid = rnbacaau.Bid(buyer_id, n_tasks, c_level_min, l_max, r_min, p_max)
        order_book.submit_bid(bid)
        # Prepare outputs.
        expected = """
        {'standard': TierClearResult(c_level='standard',
                                     clearing_price=0.00089612,
                                     fills=[Fill(buyer_id='buyer_1',
                                                 seller_id='groq',
                                                 c_level='standard',
                                                 n_tasks=50,
                                                 price=0.00089612)],
                                     unfilled_bid_n_tasks=0,
                                     unfilled_ask_n_tasks=70)}
        pending=[]
        """
        # Run test.
        results = order_book.clear_round()
        pending = order_book.get_pending_asks()
        # Check outputs.
        actual = "\n".join(
            [
                pprint.pformat(results),
                f"pending={pending}",
            ]
        )
        self.assert_equal(actual, expected, dedent=True)
