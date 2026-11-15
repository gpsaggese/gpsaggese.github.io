#!/usr/bin/env python
"""
Show a seeded auction as tables; fulfillment is mocked and no network is used.

# Usage Example
> python -m research.Noesis.scripts.demo_auction_report --scenario excess_demand

Import as:

import research.Noesis.scripts.demo_auction_report as rnscdear
"""

import argparse

import helpers.hdbg as hdbg
import helpers.hparser as hparser

import research.Noesis.market.auction_plot as rnmaaupl
import research.Noesis.market.auction_report as rnmaaurp
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.round_runner as rnmaroru
import research.Noesis.market.scenarios as rnmascen


def _parse() -> argparse.ArgumentParser:
    """
    Build the offline demonstration's command-line interface.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=hparser.CustomHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "balanced",
            "excess_demand",
            "excess_supply",
            "multi_tier",
            "no_cross",
        ),
        default="excess_demand",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default="", help="Optional output PNG path")
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Run a scenario and display its inputs and mocked contracts.

    :param parser: parser returned by `_parse()`
    """
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Reuse M11 fixtures and M10 orchestration instead of simulating a new auction.
    factory = getattr(rnmascen, f"make_{args.scenario}_scenario")
    bids, asks = factory(seed=args.seed)
    book = rnbacaau.OrderBook()
    for bid in bids:
        book.submit_bid(bid)
    for ask in asks:
        book.submit_ask(ask)
    result = rnmaroru.run_round(
        book, rnmaroru.InMemoryContractStore(), lambda c: True
    )
    # This CLI owns presentation; library helpers neither print nor write files.
    print("Fulfillment is MOCKED; no provider calls or payments.")
    for name, table in [
        ("orders", rnmaaurp.orders_to_df(bids, asks)),
        ("contracts", rnmaaurp.contracts_to_df(result.contracts)),
        ("tiers", rnmaaurp.tier_summary_df(result)),
        ("unmatched", rnmaaurp.unmatched_to_df(result)),
    ]:
        print(f"{name}=\n{table.to_string(index=False)}")
    # Optional export stays outside the pure report and plotting helpers.
    if args.plot:
        # Export only in the demo; plotting helpers return in-memory figures.
        tier = next(iter(result.tier_results))
        figure = rnmaaupl.plot_supply_demand(
            tier, bids, asks, result.tier_results[tier].clearing_price
        )
        figure.savefig(args.plot)
        print(f"plot={args.plot}")


if __name__ == "__main__":
    _main(_parse())
