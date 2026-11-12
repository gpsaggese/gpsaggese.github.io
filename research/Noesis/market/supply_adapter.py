"""
Turn OpenRouter endpoint prices into market asks for the demo sellers.

The adapter is pure: it takes the parsed JSON of OpenRouter's
`GET /api/v1/models/{model}/endpoints` and the sellers of the market, and
returns one `Ask` per seller whose endpoint is in the catalog. Fetching the
catalog and reading the seller file are done by
`noesis_platform/supply_feed.py`, since `market/` imports no other Noesis
component.

Import as:

import research.Noesis.market.supply_adapter as rnmasuad
"""

import dataclasses
import logging
from typing import Any, Dict, List, Mapping, Sequence

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.market.batch_call_auction as rnbacaau

_LOG = logging.getLogger(__name__)


# #############################################################################
# Constants
# #############################################################################


# Task definition from `docs/specs/mvp_prd.md` D1: one chat completion with at
# most 1,000 prompt tokens and 256 completion tokens. An ask is priced at this
# worst case, so a seller never sells a task below what it can cost to serve.
PROMPT_TOKENS_PER_TASK = 1000
COMPLETION_TOKENS_PER_TASK = 256

# Task prices are ~1e-4 USD; rounding to 1e-9 keeps float noise out of the book.
_PRICE_DECIMALS = 9


# #############################################################################
# SellerSupply
# #############################################################################


@dataclasses.dataclass(frozen=True)
class SellerSupply:
    """
    Store what the market knows about one seller's supply.

    E.g., `SellerSupply("groq", "groq", 0.57, 1.0)` is the seller `groq`,
    reached through the OpenRouter endpoint tag `groq`, with a typical latency
    of 0.57 s and a typical reliability of 1.0 (`common/seed.yaml`).
    """

    seller_id: str
    provider_slug: str
    l_typical: float
    r_typical: float


# #############################################################################
# EndpointQuote
# #############################################################################


@dataclasses.dataclass(frozen=True)
class EndpointQuote:
    """
    Store the per-token prices OpenRouter quotes for one endpoint.

    E.g., `EndpointQuote("groq", 5.9e-07, 7.9e-07)` means the `groq` endpoint
    charges 0.59 USD per million prompt tokens and 0.79 USD per million
    completion tokens.
    """

    provider_slug: str
    prompt_price: float
    completion_price: float


# #############################################################################
# Catalog parsing
# #############################################################################


def parse_endpoint_quotes(catalog: Mapping[str, Any]) -> List[EndpointQuote]:
    """
    Extract one price quote per endpoint from an OpenRouter endpoint catalog.

    Endpoints without a `tag` or without both prices are skipped, since the
    market cannot price an ask from them.

    :param catalog: parsed JSON body of
        `GET /api/v1/models/{model}/endpoints`, e.g.,
        ```
        {"data": {"endpoints": [{"tag": "groq",
                                 "pricing": {"prompt": "0.00000059",
                                             "completion": "0.00000079"}}]}}
        ```
    :return: quotes in catalog order, e.g.,
        ```
        [EndpointQuote(provider_slug='groq', prompt_price=5.9e-07,
                       completion_price=7.9e-07)]
        ```
    """
    data = catalog.get("data")
    hdbg.dassert_isinstance(data, dict, "Catalog has no 'data' object")
    endpoints = data.get("endpoints")
    hdbg.dassert_isinstance(endpoints, list, "Catalog has no 'endpoints' list")
    num_endpoints = len(endpoints)
    _LOG.debug("num_endpoints=%s", num_endpoints)
    quotes: List[EndpointQuote] = []
    for endpoint in endpoints:
        # OpenRouter reports prices as decimal strings.
        tag = endpoint.get("tag", "")
        pricing = endpoint.get("pricing") or {}
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")
        if not tag or prompt_price is None or completion_price is None:
            _LOG.debug("Skipping endpoint without tag or prices: '%s'", tag)
            continue
        quotes.append(
            EndpointQuote(tag, float(prompt_price), float(completion_price))
        )
    _LOG.debug("return=%s", quotes)
    return quotes


# #############################################################################
# Pricing
# #############################################################################


def compute_task_price(
    prompt_price: float, completion_price: float, *, margin: float = 0.0
) -> float:
    """
    Price one task from per-token prices, as `docs/specs/mvp_prd.md` S2 says.

    The price is the worst-case token cost of one task (PRD D1 caps) times
    `1 + margin`.

    :param prompt_price: price of one prompt token, in USD
    :param completion_price: price of one completion token, in USD
    :param margin: fraction added on top of the token cost
        - E.g., `0.1` sells a task at 110% of its token cost
    :return: minimum price per task the seller accepts, in USD, rounded to
        1e-9
    """
    _LOG.debug(hprint.to_str("prompt_price completion_price margin"))
    hdbg.dassert_lte(0.0, prompt_price, "prompt_price must be non-negative")
    hdbg.dassert_lte(
        0.0, completion_price, "completion_price must be non-negative"
    )
    hdbg.dassert_lte(0.0, margin, "margin must be non-negative")
    # Worst-case token cost of one task at the PRD D1 caps.
    token_cost = (
        prompt_price * PROMPT_TOKENS_PER_TASK
        + completion_price * COMPLETION_TOKENS_PER_TASK
    )
    _LOG.debug(hprint.to_str("token_cost"))
    price = round(token_cost * (1.0 + margin), _PRICE_DECIMALS)
    _LOG.debug("return=%s", price)
    return price


# #############################################################################
# Ask building
# #############################################################################


def build_asks(
    quotes: Sequence[EndpointQuote],
    sellers: Sequence[SellerSupply],
    *,
    n_tasks: int,
    c_level: str,
    margin: float = 0.0,
) -> List[rnbacaau.Ask]:
    """
    Build one ask per seller whose endpoint has a quote in the catalog.

    A seller whose endpoint is missing from `quotes` (e.g., the model was
    removed from OpenRouter) gets no ask, so its supply is not renewed. Endpoints
    that belong to no seller are ignored. When a tag appears twice in `quotes`,
    the first quote wins.

    :param quotes: per-endpoint prices from `parse_endpoint_quotes()`
    :param sellers: the sellers of the market and their endpoint tags
    :param n_tasks: tasks offered by each ask
    :param c_level: capability tier of every ask, e.g., "standard"
    :param margin: fraction added on top of the token cost
    :return: asks in `sellers` order, e.g.,
        ```
        [Ask(seller_id='groq', n_tasks=60, c_level='standard',
             l_typical=0.57, r_typical=1.0, p_min=0.00079224)]
        ```
    """
    _LOG.debug(hprint.to_str("n_tasks c_level margin"))
    # Index the catalog by endpoint tag, keeping the first quote of a tag.
    # Match sellers to endpoints by tag, not by display name, since tags are
    # what the gateway pins to.
    quote_by_slug: Dict[str, EndpointQuote] = {}
    for quote in quotes:
        quote_by_slug.setdefault(quote.provider_slug, quote)
    _LOG.debug("num_tags=%s num_sellers=%s", len(quote_by_slug), len(sellers))
    asks: List[rnbacaau.Ask] = []
    for seller in sellers:
        # A seller whose endpoint is gone gets no ask, so its supply lapses.
        quote = quote_by_slug.get(seller.provider_slug)
        if quote is None:
            _LOG.info(
                "No OpenRouter quote for seller '%s' ('%s'): no ask",
                seller.seller_id,
                seller.provider_slug,
            )
            continue
        p_min = compute_task_price(
            quote.prompt_price, quote.completion_price, margin=margin
        )
        _LOG.debug("Seller '%s' has p_min=%s", seller.seller_id, p_min)
        asks.append(
            rnbacaau.Ask(
                seller.seller_id,
                n_tasks,
                c_level,
                seller.l_typical,
                seller.r_typical,
                p_min,
            )
        )
    _LOG.debug("return=%s", asks)
    return asks


# #############################################################################
# Order book submission
# #############################################################################


def submit_supply(
    order_book: rnbacaau.OrderBook, asks: Sequence[rnbacaau.Ask]
) -> List[rnbacaau.Ask]:
    """
    Queue `asks` on the order book, one ask per seller per round.

    `OrderBook.clear_round()` drops every order it processes, so the supply is
    re-submitted before each round. A seller that already has a pending ask in
    the current round is skipped, so refreshing twice in one round does not
    accumulate duplicate supply.

    :param order_book: order book of the next round
    :param asks: asks from `build_asks()`
    :return: the asks that were actually queued
    """
    num_asks = len(asks)
    _LOG.debug("num_asks=%s", num_asks)
    # Sellers that already have supply queued in this round are not re-posted.
    pending_sellers = {ask.seller_id for ask in order_book.get_pending_asks()}
    submitted: List[rnbacaau.Ask] = []
    for ask in asks:
        # One ask per seller per round; a second refresh is a no-op.
        if ask.seller_id in pending_sellers:
            _LOG.debug("Seller '%s' already has a pending ask", ask.seller_id)
            continue
        order_book.submit_ask(ask)
        pending_sellers.add(ask.seller_id)
        submitted.append(ask)
    _LOG.debug("return=%s", submitted)
    return submitted
