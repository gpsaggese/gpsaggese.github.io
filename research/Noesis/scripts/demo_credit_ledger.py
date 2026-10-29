#!/usr/bin/env python
"""
Show the free credit ledger gating bids and charging cleared contracts.

Walks the `PR_P4` path end to end through the HTTP API: one account receives
the free grant, a bid within the balance is accepted, a bid over it is
rejected, an ask clears against the accepted bid, and the balance drops by the
filled tasks at the clearing price.

# Usage Example

Run with the defaults (free credit 100, bid 4 tasks at 20, ask 3 tasks at 16):

    > research/Noesis/scripts/demo_credit_ledger.py

Grant less credit, so the first bid is already rejected:

    > research/Noesis/scripts/demo_credit_ledger.py --free_credit 50

Import as:

import research.Noesis.scripts.demo_credit_ledger as rnscdecl
"""

import argparse
import logging
from typing import Any, Dict, List

import fastapi.testclient
import httpx

import helpers.hdbg as hdbg
import helpers.hparser as hparser
import helpers.hprint as hprint
import research.Noesis.gateway.passthrough_proxy as rnopapro
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.credit_ledger as rnmacrle
import research.Noesis.noesis_platform.platform_api as rnoplapi

_LOG = logging.getLogger(__name__)

_API_KEY = "demo_key"
_ACCOUNT_ID = "acct_demo"
_HEADERS = {rnoplapi.API_KEY_HEADER: _API_KEY}


# #############################################################################
# Report
# #############################################################################


def _bid_payload(buyer_id: str, n_tasks: int, p_max: float) -> Dict[str, Any]:
    payload = {
        "buyer_id": buyer_id,
        "n_tasks": n_tasks,
        "c_level_min": "standard",
        "l_max": 2.0,
        "r_min": 0.9,
        "p_max": p_max,
    }
    return payload


def _describe_bid(response: httpx.Response) -> str:
    """
    :return: one line with the status and, on a rejection, its reason
    """
    if response.status_code == 201:
        text = "201 accepted"
    else:
        # The 400 detail is the assertion text; keep its message line.
        detail_lines = [
            line for line in response.json()["detail"].splitlines() if line
        ]
        reason = detail_lines[-2]
        text = f"{response.status_code} rejected: {reason}"
    return text


def build_report(
    *,
    free_credit: float,
    bid_n_tasks: int,
    bid_p_max: float,
    ask_n_tasks: int,
    ask_p_min: float,
) -> str:
    """
    Run the credit path over an in-process app and describe each step as text.

    :param free_credit: credit granted to the account on first use
    :param bid_n_tasks: tasks the buyer bids for
    :param bid_p_max: most the buyer pays per task
    :param ask_n_tasks: tasks the seller offers
    :param ask_p_min: least the seller accepts per task
    :return: multi-line report
    """
    _LOG.debug(
        hprint.to_str("free_credit bid_n_tasks bid_p_max ask_n_tasks ask_p_min")
    )
    ledger = rnmacrle.CreditLedger(free_credit=free_credit)
    app = rnoplapi.create_app(
        rnbacaau.OrderBook(),
        rnopapro.Gateway(),
        {_API_KEY: _ACCOUNT_ID},
        credit_ledger=ledger,
    )
    client = fastapi.testclient.TestClient(app)
    lines: List[str] = []
    # 1) The grant.
    balance = ledger.get_balance(_ACCOUNT_ID)
    lines.append(f"1) Account '{_ACCOUNT_ID}' starts with balance={balance}")
    # 2) A bid within the balance.
    cost = bid_n_tasks * bid_p_max
    response = client.post(
        "/bids",
        json=_bid_payload("buyer_1", bid_n_tasks, bid_p_max),
        headers=_HEADERS,
    )
    lines.append(
        f"2) Bid {bid_n_tasks} tasks at p_max={bid_p_max} (cost {cost})"
        f" -> {_describe_bid(response)}"
    )
    lines.append(
        f"   balance after the bid: {ledger.get_balance(_ACCOUNT_ID)}"
        " (nothing is held)"
    )
    # 3) A bid over the balance.
    over_n_tasks = int(balance // bid_p_max) + 1
    over_cost = over_n_tasks * bid_p_max
    response = client.post(
        "/bids",
        json=_bid_payload("buyer_2", over_n_tasks, bid_p_max),
        headers=_HEADERS,
    )
    lines.append(
        f"3) Bid {over_n_tasks} tasks at p_max={bid_p_max} (cost {over_cost})"
        f" -> {_describe_bid(response)}"
    )
    # 4) An ask clears against the accepted bid.
    client.post(
        "/asks",
        json={
            "seller_id": "seller_1",
            "n_tasks": ask_n_tasks,
            "c_level": "standard",
            "l_typical": 1.0,
            "r_typical": 0.95,
            "p_min": ask_p_min,
        },
        headers=_HEADERS,
    )
    results = client.post("/rounds/clear").json()
    lines.append(
        f"4) Ask {ask_n_tasks} tasks at p_min={ask_p_min} -> clear_round():"
    )
    for result in results:
        lines.append(
            f"   tier={result['tier']} clearing_price={result['clearing_price']}"
            f" matched_volume={result['matched_volume']}"
        )
    # 5) The debit.
    balance = ledger.get_balance(_ACCOUNT_ID)
    lines.append(f"5) Balance after clearing: {balance}")
    client.post("/rounds/clear")
    lines.append(
        f"   balance after a second clear: {ledger.get_balance(_ACCOUNT_ID)}"
    )
    # 6) The new balance gates the next round.
    edge_n_tasks = int(balance // bid_p_max)
    for n_tasks in (edge_n_tasks, edge_n_tasks + 1):
        response = client.post(
            "/bids",
            json=_bid_payload("buyer_1", n_tasks, bid_p_max),
            headers=_HEADERS,
        )
        lines.append(
            f"6) Bid {n_tasks} tasks at p_max={bid_p_max}"
            f" (cost {n_tasks * bid_p_max}) -> {_describe_bid(response)}"
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
        "--free_credit",
        type=float,
        default=100.0,
        help="Credit granted to the account on first use",
    )
    parser.add_argument(
        "--bid_n_tasks", type=int, default=4, help="Tasks the buyer bids for"
    )
    parser.add_argument(
        "--bid_p_max",
        type=float,
        default=20.0,
        help="Most the buyer pays per task",
    )
    parser.add_argument(
        "--ask_n_tasks", type=int, default=3, help="Tasks the seller offers"
    )
    parser.add_argument(
        "--ask_p_min",
        type=float,
        default=16.0,
        help="Least the seller accepts per task",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level)
    _LOG.debug(hprint.to_str("args"))
    report = build_report(
        free_credit=args.free_credit,
        bid_n_tasks=args.bid_n_tasks,
        bid_p_max=args.bid_p_max,
        ask_n_tasks=args.ask_n_tasks,
        ask_p_min=args.ask_p_min,
    )
    print(report)


if __name__ == "__main__":
    _main(_parse())
