"""
Build supply/demand figures without displaying or saving them.

Import as:

import research.Noesis.market.auction_plot as rnmaaupl
"""

import itertools
import logging
from typing import List, Optional, Tuple

import matplotlib.axes as maxes
import matplotlib.figure as mfigure

import research.Noesis.market.batch_call_auction as rnbacaau

_LOG = logging.getLogger(__name__)


def _add_curve(
    ax: maxes.Axes, orders: List[Tuple[int, float]], label: str, color: str
) -> None:
    """
    Draw price bands across cumulative quantities; omit empty sides.

    :param ax: figure axes
    :param orders: quantity/price pairs already sorted in auction priority
    :param label: legend entry
    :param color: curve color
    """
    _LOG.debug("label='%s' num_orders='%s'", label, len(orders))
    if orders:
        quantities, prices = zip(*orders)
        # Post steps span [previous cumulative quantity, next quantity].
        x_values = [0] + list(itertools.accumulate(quantities))
        y_values = list(prices) + [prices[-1]]
        ax.step(
            x_values,
            y_values,
            where="post",
            label=label,
            color=color,
            linewidth=2.5,
        )


def plot_supply_demand(
    tier: str,
    bids: List[rnbacaau.Bid],
    asks: List[rnbacaau.Ask],
    clearing_price: Optional[float],
) -> mfigure.Figure:
    """
    Return one tier's step curves with its clearing price and quantity marked.

    Quantities are tasks; prices use the same per-task unit as the input orders.
    The marker reconstructs volume affordable at the supplied clearing price;
    pass the corresponding M10 summary's price, or `None` for a no-trade round.

    :param tier: exact capability tier to display
    :param bids: original submitted bids, not unmatched residuals
    :param asks: original submitted asks
    :param clearing_price: actual round price, or no price when no trade occurred
    :return: standalone figure; the caller owns display, saving, and cleanup
    """
    _LOG.debug("tier='%s' clearing_price='%s'", tier, clearing_price)
    # Sort copies and filter tiers so callers can reuse the complete input book.
    demand = sorted(
        [(b.n_tasks, b.p_max) for b in bids if b.c_level_min == tier],
        key=lambda order: order[1],
        reverse=True,
    )
    supply = sorted(
        [(a.n_tasks, a.p_min) for a in asks if a.c_level == tier],
        key=lambda order: order[1],
    )
    figure = mfigure.Figure(figsize=(7, 4), layout="constrained")
    ax = figure.subplots()
    _add_curve(ax, demand, "Demand", "#A6C8F4")
    _add_curve(ax, supply, "Supply", "#FFD1A6")
    # No-cross and empty tiers have no clearing point; zero is a valid price.
    if clearing_price is not None:
        volume = min(
            sum(q for q, p in demand if p >= clearing_price),
            sum(q for q, p in supply if p <= clearing_price),
        )
        ax.axhline(
            clearing_price, label="Clearing price", color="black", linestyle="--"
        )
        ax.scatter(
            [volume], [clearing_price], label="Clearing point", color="black"
        )
    ax.set(
        title=f"Supply and demand: {tier}",
        xlabel="Tasks",
        ylabel="Price per task",
    )
    if demand or supply or clearing_price is not None:
        ax.legend()
    _LOG.debug("num_curves='%s'", len(ax.lines))
    return figure
