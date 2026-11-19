"""
Run and display the in-memory market demonstration with mocked fulfillment.

Import as:

import research.Noesis.notebooks.market_in_memory_utils as utils
"""

import logging
from typing import List

import helpers.hnotebook as hnotebo
from IPython.display import display

import research.Noesis.market.auction_plot as rnmaaupl
import research.Noesis.market.auction_report as rnmaaurp
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.round_runner as rnmaroru

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utility logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


def run_auction(
    bids: List[rnbacaau.Bid], asks: List[rnbacaau.Ask]
) -> rnmaroru.RoundResult:
    """
    Submit a fresh in-memory book and clear it with mocked successful delivery.

    :param bids: original buyer orders
    :param asks: original seller orders
    :return: one round; each invocation has its own store and starts at round 0
    """
    _LOG.debug("Submitting %d bids and %d asks", len(bids), len(asks))
    book = rnbacaau.OrderBook()
    for bid in bids:
        book.submit_bid(bid)
    for ask in asks:
        book.submit_ask(ask)
    # True records a mock outcome; no provider, payment, or delivery is invoked.
    result = rnmaroru.run_round(
        book, rnmaroru.InMemoryContractStore(), lambda contract: True
    )
    return result


def display_result(
    bids: List[rnbacaau.Bid],
    asks: List[rnbacaau.Ask],
    result: rnmaroru.RoundResult,
) -> None:
    """
    Display M12 tables and a supply/demand curve for every cleared tier.

    :param bids: original orders, needed to plot submitted demand
    :param asks: original orders, needed to plot submitted supply
    :param result: cleared round with mocked fulfillment outcomes
    """
    print(f"round_id={result.round_id}; fulfillment=MOCKED (always True)")
    # Keep names beside tables, including empty contracts and residual orders.
    for name, table in [
        ("tier_summary", rnmaaurp.tier_summary_df(result)),
        (
            "contracts (fulfilled is MOCKED)",
            rnmaaurp.contracts_to_df(result.contracts),
        ),
        ("unmatched_orders", rnmaaurp.unmatched_to_df(result)),
    ]:
        print(f"{name}=")
        display(table)
    # Use submitted quantities, not residual quantities, for the curves.
    for tier, summary in sorted(result.tier_results.items()):
        print(f"tier={tier}; clearing_price={summary.clearing_price}")
        figure = rnmaaupl.plot_supply_demand(
            tier, bids, asks, summary.clearing_price
        )
        display(figure)
