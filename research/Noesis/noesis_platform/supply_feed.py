"""
Fetch OpenRouter endpoint prices and post them to the market as asks.

This is the network and file side of `PR_S9`: it reads the sellers from
`common/seed.yaml`, fetches the endpoint catalog of the served model from
OpenRouter, and hands both to the pure adapter in `market/supply_adapter.py`.
Run `refresh_market_supply()` once before each auction round (e.g., from the
`PR_M7` scheduler once it exists) so OpenRouter's capacity shows up as supply
without a manual `POST /asks` per seller.

Import as:

import research.Noesis.noesis_platform.supply_feed as rnplsufe
"""

import logging
import os
from typing import Any, Dict, List

import httpx
import yaml

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.supply_adapter as rnmasuad

_LOG = logging.getLogger(__name__)


# #############################################################################
# Constants
# #############################################################################


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Same as `common/noesis_settings.DEFAULT_MODEL_ID`; `noesis_platform/` does not
# import `common/` (see `test/test_component_imports.py`).
DEFAULT_MODEL_ID = "meta-llama/llama-3.3-70b-instruct"

DEFAULT_SEED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "common",
    "seed.yaml",
)

# The MVP has one tier (`docs/specs/mvp_prd.md` D2).
DEFAULT_C_LEVEL = "standard"

# Tasks offered per ask: the largest bid the simulator posts in one round
# (`docs/specs/mvp_prd.md` S1: `n_tasks` 20-60), so one seller can fill any
# single bid. OpenRouter publishes no capacity figure to derive this from.
DEFAULT_N_TASKS = 60


# #############################################################################
# Sellers
# #############################################################################


def load_seller_supply(seed_path: str) -> List[rnmasuad.SellerSupply]:
    """
    Read the sellers and their typical latency and reliability from the seed.

    :param seed_path: YAML file with a `sellers` list, e.g., `common/seed.yaml`
    :return: one `SellerSupply` per seller, in file order
    """
    _LOG.debug(hprint.to_str("seed_path"))
    # Same YAML layout `common/noesis_seed.py` loads into Postgres.
    with open(seed_path, "r") as f:
        data = yaml.safe_load(f.read()) or {}
    sellers_data = data.get("sellers", [])
    hdbg.dassert_isinstance(
        sellers_data, list, "'sellers' in '%s' must be a list", seed_path
    )
    _LOG.debug("num_sellers=%s", len(sellers_data))
    sellers = []
    for seller in sellers_data:
        for key in ["seller_id", "provider_slug", "l_typical", "r_typical"]:
            hdbg.dassert_in(
                key, seller, "Seller entry in '%s' is missing a field", seed_path
            )
        # Cast the numbers to `float`, since YAML may parse `1` as an int.
        sellers.append(
            rnmasuad.SellerSupply(
                seller["seller_id"],
                seller["provider_slug"],
                float(seller["l_typical"]),
                float(seller["r_typical"]),
            )
        )
    _LOG.debug("return=%s", sellers)
    return sellers


# #############################################################################
# OpenRouter catalog
# #############################################################################


def fetch_endpoint_catalog(
    http: httpx.Client,
    model_id: str,
    *,
    api_key: str = "",
    base_url: str = DEFAULT_BASE_URL,
) -> Dict[str, Any]:
    """
    Fetch the endpoints that serve `model_id` on OpenRouter.

    The endpoint list and its prices are public; the key is sent when given so
    OpenRouter also fills in the authenticated-only fields.

    :param http: client to send the request with
        - Tests pass a client over `httpx.MockTransport`
    :param model_id: OpenRouter model id, e.g.,
        "meta-llama/llama-3.3-70b-instruct"
    :param api_key: OpenRouter API key; empty sends no `Authorization` header
    :param base_url: OpenRouter API root
    :return: parsed JSON body of `GET /models/{model_id}/endpoints`
    """
    _LOG.debug(hprint.to_str("model_id base_url"))
    # Never log `headers`, since they carry the API key.
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = http.get(
        f"{base_url.rstrip('/')}/models/{model_id}/endpoints", headers=headers
    )
    _LOG.debug("status_code=%s", resp.status_code)
    resp.raise_for_status()
    catalog = resp.json()
    num_endpoints = len(catalog.get("data", {}).get("endpoints", []))
    _LOG.debug("num_endpoints=%s", num_endpoints)
    return catalog


# #############################################################################
# Market refresh
# #############################################################################


def refresh_market_supply(
    order_book: rnbacaau.OrderBook,
    catalog: Dict[str, Any],
    sellers: List[rnmasuad.SellerSupply],
    *,
    n_tasks: int = DEFAULT_N_TASKS,
    c_level: str = DEFAULT_C_LEVEL,
    margin: float = 0.0,
) -> List[rnbacaau.Ask]:
    """
    Turn one OpenRouter catalog into asks and queue them for the next round.

    :param order_book: order book of the next round
    :param catalog: parsed catalog from `fetch_endpoint_catalog()`
    :param sellers: sellers from `load_seller_supply()`
    :param n_tasks: tasks offered by each ask
    :param c_level: capability tier of every ask
    :param margin: fraction added on top of the token cost
    :return: the asks that were queued
    """
    _LOG.debug(hprint.to_str("n_tasks c_level margin"))
    quotes = rnmasuad.parse_endpoint_quotes(catalog)
    asks = rnmasuad.build_asks(
        quotes, sellers, n_tasks=n_tasks, c_level=c_level, margin=margin
    )
    _LOG.debug("num_quotes=%s num_asks=%s", len(quotes), len(asks))
    submitted = rnmasuad.submit_supply(order_book, asks)
    _LOG.info("Queued %d asks from %d endpoints", len(submitted), len(quotes))
    _LOG.debug("return=%s", submitted)
    return submitted
