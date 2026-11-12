#!/usr/bin/env python
"""
Show OpenRouter endpoint prices becoming market asks and clearing against a bid.

Walks the `PR_S9` path end to end: load the sellers from `common/seed.yaml`,
read an OpenRouter endpoint catalog, build one ask per seller, queue the asks on
an `OrderBook`, then submit one buyer bid and clear the round.

# Usage Example

Run on the checked-in catalog sample (no key, no network):

    > research/Noesis/scripts/demo_supply_asks.py

Fetch the live catalog (reads `OPENROUTER_API_KEY` from the env or
`research/Noesis/.env`; a read-only call, no spend):

    > research/Noesis/scripts/demo_supply_asks.py --live

Offer 20 tasks per seller and bid 50 tasks at up to 0.0005 USD each:

    > research/Noesis/scripts/demo_supply_asks.py --n_tasks 20 --bid_p_max 0.0005

Import as:

import research.Noesis.scripts.demo_supply_asks as rnscdesa
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import httpx

import helpers.hdbg as hdbg
import helpers.hparser as hparser
import helpers.hprint as hprint
import research.Noesis.common.noesis_settings as rnonoset
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.supply_adapter as rnmasuad
import research.Noesis.noesis_platform.supply_feed as rnplsufe

_LOG = logging.getLogger(__name__)

# `research/Noesis`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Checked-in OpenRouter endpoint catalog, used when `--live` is not set.
_SAMPLE_CATALOG = os.path.join(
    _ROOT, "data", "openrouter", "endpoints_sample.json"
)


# #############################################################################
# Catalog
# #############################################################################


def _load_catalog(*, live: bool, model_id: str) -> Dict[str, Any]:
    """
    Return the endpoint catalog, from OpenRouter or from the checked-in sample.

    :param live: fetch from OpenRouter instead of reading the sample
    :param model_id: OpenRouter model id
    :return: parsed catalog as `fetch_endpoint_catalog()` returns it
    """
    _LOG.debug(hprint.to_str("live model_id"))
    if live:
        # An exported `OPENROUTER_API_KEY` takes precedence over the `.env` file.
        env = rnonoset.read_dotenv(os.path.join(_ROOT, ".env"))
        api_key = os.environ.get("OPENROUTER_API_KEY") or env.get(
            "OPENROUTER_API_KEY", ""
        )
        # Read-only call to OpenRouter: no spend.
        with httpx.Client(timeout=20.0) as http:
            catalog = rnplsufe.fetch_endpoint_catalog(
                http, model_id, api_key=api_key
            )
        source = "live"
    else:
        # Offline path: the checked-in sample, so the demo runs without a key.
        with open(_SAMPLE_CATALOG, "r") as f:
            catalog = json.load(f)
        source = _SAMPLE_CATALOG
    _LOG.info("Catalog source: '%s'", source)
    return catalog


# #############################################################################
# Report
# #############################################################################


def build_report(
    catalog: Dict[str, Any],
    sellers: List[rnmasuad.SellerSupply],
    *,
    n_tasks: int,
    bid_n_tasks: int,
    bid_p_max: float,
) -> str:
    """
    Run the supply path on `catalog` and describe each step as text.

    :param catalog: OpenRouter endpoint catalog
    :param sellers: sellers of the market
    :param n_tasks: tasks offered by each ask
    :param bid_n_tasks: tasks the sample buyer bids for
    :param bid_p_max: most the sample buyer pays per task, in USD
    :return: multi-line report
    """
    _LOG.debug(hprint.to_str("n_tasks bid_n_tasks bid_p_max len(sellers)"))
    lines: List[str] = []
    # 1) What OpenRouter quotes.
    quotes = rnmasuad.parse_endpoint_quotes(catalog)
    lines.append(f"1) OpenRouter quotes {len(quotes)} endpoints for the model:")
    for quote in quotes:
        task_price = rnmasuad.compute_task_price(
            quote.prompt_price, quote.completion_price
        )
        lines.append(
            f"   {quote.provider_slug:<28} prompt={quote.prompt_price:.3e}"
            f" completion={quote.completion_price:.3e}"
            f" -> task={task_price:.8f} USD"
        )
    # 2) Which quotes belong to our sellers.
    # A seller whose endpoint tag is not quoted has no supply.
    quoted = {quote.provider_slug for quote in quotes}
    lines.append(f"2) Seed sellers ({len(sellers)}) matched by endpoint tag:")
    for seller in sellers:
        status = "quoted" if seller.provider_slug in quoted else "NO QUOTE"
        lines.append(
            f"   {seller.seller_id:<10} tag={seller.provider_slug:<16}"
            f" l_typical={seller.l_typical}s r_typical={seller.r_typical}"
            f"  {status}"
        )
    # 3) The asks the adapter queues.
    order_book = rnbacaau.OrderBook()
    # Turn the catalog into asks and queue them on the order book.
    asks = rnplsufe.refresh_market_supply(
        order_book, catalog, sellers, n_tasks=n_tasks
    )
    lines.append(f"3) Asks queued on the order book ({len(asks)}):")
    for ask in asks:
        lines.append(
            f"   {ask.seller_id:<10} n_tasks={ask.n_tasks} c_level={ask.c_level}"
            f" p_min={ask.p_min:.8f}"
        )
    # A second refresh in the same round is a no-op, see `submit_supply()`.
    again = rnplsufe.refresh_market_supply(
        order_book, catalog, sellers, n_tasks=n_tasks
    )
    lines.append(f"   second refresh in the same round queued {len(again)} asks")
    # 4) One bid clears against that supply.
    buyer_id = "buyer_demo"
    # Standard tier, latency under 2.0 s, reliability at least 0.9, and at most
    # `bid_p_max` per task (see `rnbacaau.Bid`).
    bid = rnbacaau.Bid(buyer_id, bid_n_tasks, "standard", 2.0, 0.9, bid_p_max)
    order_book.submit_bid(bid)
    # `clear_round()` matches the bid against the queued asks.
    results = order_book.clear_round()
    _LOG.debug(hprint.to_str("len(results)"))
    lines.append(
        f"4) Bid {bid_n_tasks} tasks at p_max={bid_p_max} -> clear_round():"
    )
    for tier, result in results.items():
        lines.append(
            f"   tier={tier} clearing_price={result.clearing_price}"
            f" unfilled_bid={result.unfilled_bid_n_tasks}"
            f" unfilled_ask={result.unfilled_ask_n_tasks}"
        )
        for fill in result.fills:
            lines.append(
                f"   fill: {fill.buyer_id} <- {fill.seller_id}"
                f" n_tasks={fill.n_tasks} price={fill.price}"
            )
    # `clear_round()` drops the orders it processed, so no asks stay pending.
    lines.append(
        f"   pending asks after the round: {len(order_book.get_pending_asks())}"
    )
    report = "\n".join(lines)
    _LOG.debug(hprint.to_str("len(report)"))
    return report


# #############################################################################
# Script
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch the catalog from OpenRouter instead of the sample",
    )
    parser.add_argument(
        "--model", default=rnplsufe.DEFAULT_MODEL_ID, help="OpenRouter model id"
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=rnplsufe.DEFAULT_N_TASKS,
        help="Tasks offered by each ask",
    )
    # Sample buyer.
    parser.add_argument(
        "--bid_n_tasks", type=int, default=50, help="Tasks the sample buyer bids"
    )
    parser.add_argument(
        "--bid_p_max",
        type=float,
        default=0.001,
        help="Most the sample buyer pays per task, in USD",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level)
    _LOG.debug(hprint.to_str("args"))
    # Load the inputs: the catalog and the seed sellers.
    catalog = _load_catalog(live=args.live, model_id=args.model)
    sellers = rnplsufe.load_seller_supply(rnplsufe.DEFAULT_SEED_PATH)
    # Run the supply path and build the report.
    report = build_report(
        catalog,
        sellers,
        n_tasks=args.n_tasks,
        bid_n_tasks=args.bid_n_tasks,
        bid_p_max=args.bid_p_max,
    )
    # The report is the output of the demo, so print it.
    print(report)


if __name__ == "__main__":
    _main(_parse())
