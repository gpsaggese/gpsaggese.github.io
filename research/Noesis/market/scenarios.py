"""
Reproducible order generators for five auction regimes and unrestricted books.

Named generators vary quantities and prices while guaranteeing their regime.
Generation returns fresh data only; it does not run auctions or access storage.

Import as:

import research.Noesis.market.scenarios as rnmascen
"""

import collections.abc
import logging
import random
from typing import List, Sequence, Tuple

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.market.batch_call_auction as rnbacaau

_LOG = logging.getLogger(__name__)


# #############################################################################
# Scenario type
# #############################################################################

# TODO(giles): Need a short comment to explain what it is, what it does
Scenario = Tuple[List[rnbacaau.Bid], List[rnbacaau.Ask]]


# #############################################################################
# Regime construction
# #############################################################################


# TODO(giles): You can run /coding.improve_code_for_humans which makes the
# code easier for humans (adding comments, etc).
# The rule is that each "chunk" of code of 5-10 lines should have a comment
# explaining what to do.
def _make_regime(
    seed: int,
    n_bids: int,
    n_asks: int,
    tiers: Sequence[str],
    regime: str,
    quantity_scale: int,
    price_scale: int,
) -> Scenario:
    """
    Construct compatible books by quantity balancing and separated price bands.

    :param seed: local random seed
    :param n_bids: positive buyer order count per tier
    :param n_asks: positive seller order count per tier
    :param tiers: unique nonblank tier names
    :param regime: balanced, demand, supply, or no_cross
    :param quantity_scale: positive integer task multiplier
    :param price_scale: positive integer price multiplier
    :return: fresh bids and asks with unique participant ids
    """
    # TODO(giles): FYI there is a also a function to print the entire func signature.
    _LOG.debug(hprint.to_str("seed n_bids n_asks tiers regime"))
    for name, value in (
        ("seed", seed),
        ("n_bids", n_bids),
        ("n_asks", n_asks),
        ("quantity_scale", quantity_scale),
        ("price_scale", price_scale),
    ):
        hdbg.dassert_isinstance(value, int, "'%s' must be an integer", name)
        hdbg.dassert_ne(type(value), bool, "'%s' must not be boolean", name)
        if name != "seed":
            hdbg.dassert_lt(0, value, "%s must be positive", name)
    hdbg.dassert_isinstance(
        tiers, collections.abc.Sequence, "tiers must be an ordered sequence"
    )
    hdbg.dassert_not_in(
        type(tiers), (str, bytes), "tiers must not be a bare string or bytes"
    )
    hdbg.dassert_lt(0, len(tiers), "tiers must not be empty")
    for tier in tiers:
        hdbg.dassert_isinstance(tier, str, "tier names must be strings")
        hdbg.dassert_ne(tier.strip(), "", "tier names must not be blank")
    hdbg.dassert_eq(len(tiers), len(set(tiers)), "tiers must be distinct")
    # TODO(giles): This needs a comment otherwise one needs to read the entire
    # code to understand what it does. You want code to have "multiple
    # level of abstractions". The skill will fix this, but FYI
    rng = random.Random(seed)
    bids: List[rnbacaau.Bid] = []
    asks: List[rnbacaau.Ask] = []
    for tier_index, tier in enumerate(tiers):
        bid_quantities = [rng.randint(1, 100) for _ in range(n_bids)]
        ask_quantities = [rng.randint(1, 100) for _ in range(n_asks)]
        # Raise the smaller total to guarantee full matching before imbalance.
        difference = sum(bid_quantities) - sum(ask_quantities)
        if difference > 0:
            ask_quantities[-1] += difference
        else:
            bid_quantities[-1] -= difference
        if regime == "demand":
            bid_quantities[rng.randrange(n_bids)] += rng.randint(1, 100)
        elif regime == "supply":
            ask_quantities[rng.randrange(n_asks)] += rng.randint(1, 100)
        # Every bid crosses every ask, except in the explicit no-cross regime.
        # Disjoint tier bands guarantee distinct multi-tier clearing prices.
        base = tier_index * 20
        bid_low, ask_low = (1, 6) if regime == "no_cross" else (6, 1)
        for quantity in bid_quantities:
            price = float(
                (base + rng.randint(bid_low, bid_low + 3)) * price_scale
            )
            bids.append(
                rnbacaau.Bid(
                    f"buyer_{len(bids)}",
                    quantity * quantity_scale,
                    tier,
                    2.0,
                    0.90,
                    price,
                )
            )
        for quantity in ask_quantities:
            price = float(
                (base + rng.randint(ask_low, ask_low + 3)) * price_scale
            )
            asks.append(
                rnbacaau.Ask(
                    f"seller_{len(asks)}",
                    quantity * quantity_scale,
                    tier,
                    1.0,
                    0.95,
                    price,
                )
            )
    scenario = (bids, asks)
    _LOG.debug("return=%s", (len(bids), len(asks)))
    return scenario


# #############################################################################
# Named scenario families
# #############################################################################


def make_balanced_scenario(
    *,
    seed: int = 0,
    n_bids: int = 2,
    n_asks: int = 2,
    tier: str = "standard",
    quantity_scale: int = 1,
    price_scale: int = 1,
) -> Scenario:
    # TODO(giles): you can remove the default from the docstring because
    # of DRY and the info is in the interface. You can improve the skills
    # with this rule.
    # Same everywhere.
    """
    Generate fully matched demand and supply.

    :param seed: integer seed for reproducible varied inputs
        - Default: `0`
    :param n_bids: positive buyer count in the book
        - Default: `2`
    :param n_asks: positive seller count in the book
        - Default: `2`
    :param tier: nonblank capability tier name
        - Default: `"standard"`
    :param quantity_scale: positive integer multiplier for task quantities
        - Default: `1`
    :param price_scale: positive integer multiplier for illustrative prices
        - Default: `1`
    :return: fresh (bids, asks) lists satisfying the named regime
    """
    _LOG.debug(
        hprint.to_str("seed n_bids n_asks tier quantity_scale price_scale")
    )
    scenario = _make_regime(
        seed, n_bids, n_asks, [tier], "balanced", quantity_scale, price_scale
    )
    _LOG.debug("return=%s", (len(scenario[0]), len(scenario[1])))
    return scenario


def make_excess_demand_scenario(
    *,
    seed: int = 0,
    n_bids: int = 2,
    n_asks: int = 2,
    tier: str = "standard",
    quantity_scale: int = 1,
    price_scale: int = 1,
) -> Scenario:
    """
    Generate unmatched buyer quantity and fully matched supply.

    :param seed: integer seed for reproducible varied inputs
        - Default: `0`
    :param n_bids: positive buyer count in the book
        - Default: `2`
    :param n_asks: positive seller count in the book
        - Default: `2`
    :param tier: nonblank capability tier name
        - Default: `"standard"`
    :param quantity_scale: positive integer multiplier for task quantities
        - Default: `1`
    :param price_scale: positive integer multiplier for illustrative prices
        - Default: `1`
    :return: fresh (bids, asks) lists satisfying the named regime
    """
    _LOG.debug(
        hprint.to_str("seed n_bids n_asks tier quantity_scale price_scale")
    )
    scenario = _make_regime(
        seed, n_bids, n_asks, [tier], "demand", quantity_scale, price_scale
    )
    _LOG.debug("return=%s", (len(scenario[0]), len(scenario[1])))
    return scenario


def make_excess_supply_scenario(
    *,
    seed: int = 0,
    n_bids: int = 2,
    n_asks: int = 2,
    tier: str = "standard",
    quantity_scale: int = 1,
    price_scale: int = 1,
) -> Scenario:
    """
    Generate unmatched seller quantity and fully matched demand.

    :param seed: integer seed for reproducible varied inputs
        - Default: `0`
    :param n_bids: positive buyer count in the book
        - Default: `2`
    :param n_asks: positive seller count in the book
        - Default: `2`
    :param tier: nonblank capability tier name
        - Default: `"standard"`
    :param quantity_scale: positive integer multiplier for task quantities
        - Default: `1`
    :param price_scale: positive integer multiplier for illustrative prices
        - Default: `1`
    :return: fresh (bids, asks) lists satisfying the named regime
    """
    _LOG.debug(
        hprint.to_str("seed n_bids n_asks tier quantity_scale price_scale")
    )
    scenario = _make_regime(
        seed, n_bids, n_asks, [tier], "supply", quantity_scale, price_scale
    )
    _LOG.debug("return=%s", (len(scenario[0]), len(scenario[1])))
    return scenario


def make_multi_tier_scenario(
    *,
    seed: int = 0,
    n_bids: int = 2,
    n_asks: int = 2,
    tiers: Sequence[str] = ("standard", "frontier"),
    quantity_scale: int = 1,
    price_scale: int = 1,
) -> Scenario:
    """
    Generate fully matched tiers with distinct clearing prices.

    :param seed: integer seed for reproducible varied inputs
        - Default: `0`
    :param n_bids: positive buyer count per tier
        - Default: `2`
    :param n_asks: positive seller count per tier
        - Default: `2`
    :param tiers: at least two distinct tier names
        - Default: `("standard", "frontier")`
    :param quantity_scale: positive integer multiplier for task quantities
        - Default: `1`
    :param price_scale: positive integer multiplier for illustrative prices
        - Default: `1`
    :return: fresh (bids, asks) lists satisfying the named regime
    """
    _LOG.debug(
        hprint.to_str("seed n_bids n_asks tiers quantity_scale price_scale")
    )
    hdbg.dassert_isinstance(
        tiers, collections.abc.Sequence, "tiers must be an ordered sequence"
    )
    hdbg.dassert_not_in(
        type(tiers), (str, bytes), "tiers must not be a bare string or bytes"
    )
    hdbg.dassert_lte(2, len(tiers), "multi-tier requires at least two tiers")
    scenario = _make_regime(
        seed, n_bids, n_asks, tiers, "balanced", quantity_scale, price_scale
    )
    _LOG.debug("return=%s", (len(scenario[0]), len(scenario[1])))
    return scenario


def make_no_cross_scenario(
    *,
    seed: int = 0,
    n_bids: int = 2,
    n_asks: int = 2,
    tier: str = "standard",
    quantity_scale: int = 1,
    price_scale: int = 1,
) -> Scenario:
    """
    Generate zero matches because all bid limits are below ask limits.

    :param seed: integer seed for reproducible varied inputs
        - Default: `0`
    :param n_bids: positive buyer count in the book
        - Default: `2`
    :param n_asks: positive seller count in the book
        - Default: `2`
    :param tier: nonblank capability tier name
        - Default: `"standard"`
    :param quantity_scale: positive integer multiplier for task quantities
        - Default: `1`
    :param price_scale: positive integer multiplier for illustrative prices
        - Default: `1`
    :return: fresh (bids, asks) lists satisfying the named regime
    """
    _LOG.debug(
        hprint.to_str("seed n_bids n_asks tier quantity_scale price_scale")
    )
    scenario = _make_regime(
        seed, n_bids, n_asks, [tier], "no_cross", quantity_scale, price_scale
    )
    _LOG.debug("return=%s", (len(scenario[0]), len(scenario[1])))
    return scenario


# #############################################################################
# Unrestricted random books
# #############################################################################


# TODO(giles): Some of the comments are not in the interface. The interface
# only describes what it does, not how. E.g., the types checking, etc. is
# redundant with the code. The random.Random(seed) is implementation detail,
# etc.
# TODO(giles): Add more comments for the chunks of code (manually or with the
# skill).
def make_random_scenario(
    seed: int,
    n_bids: int,
    n_asks: int,
    tiers: Sequence[str],
) -> Scenario:
    """
    Return reproducible random orders; a particular matching regime is not promised.

    Use a local random.Random(seed), never global random.seed(). Identical
    arguments (including tier order) produce equal values and fresh objects.
    Each order draws its tier from the supplied sequence, its task quantity
    uniformly from integers 1 through 100, and its price from integers 1
    through 20, represented as float. Generate bids before asks. Within each
    order draw tier, quantity, then price. Ids are buyer_0, buyer_1, ... and
    seller_0, seller_1, ... in list order.

    All scenarios use buyer l_max=2.0 and r_min=0.90, and seller
    l_typical=1.0 and r_typical=0.95. These compatible fixed performance
    values isolate tier/price matching rather than performance variation.

    Input validation follows the project's assertion conventions:
    seed is an integer; counts are nonnegative integers (booleans excluded);
    tiers is a nonempty ordered sequence of distinct nonempty,
    non-whitespace strings, not a bare string. Do not restrict tier names to
    a global registry. Zero counts are allowed; even an empty result requires
    valid arguments. Invalid arguments raise AssertionError.

    :param seed: integer seed, including zero or negative integers; not bool
    :param n_bids: number of buyer orders, including zero
    :param n_asks: number of seller orders, including zero
    :param tiers: ordered unique tier names; list or tuple is recommended
    :return: fresh (bids, asks) lists with the requested lengths
    """
    _LOG.debug(hprint.to_str("seed n_bids n_asks tiers"))
    for name, value in (("seed", seed), ("n_bids", n_bids), ("n_asks", n_asks)):
        hdbg.dassert_isinstance(value, int, "'%s' must be an integer", name)
        hdbg.dassert_ne(type(value), bool, "'%s' must not be boolean", name)
    hdbg.dassert_lte(0, n_bids, "n_bids must be nonnegative")
    hdbg.dassert_lte(0, n_asks, "n_asks must be nonnegative")
    hdbg.dassert_isinstance(
        tiers, collections.abc.Sequence, "tiers must be an ordered sequence"
    )
    hdbg.dassert_not_in(
        type(tiers), (str, bytes), "tiers must not be a bare string or bytes"
    )
    hdbg.dassert_lt(0, len(tiers), "tiers must not be empty")
    for tier in tiers:
        hdbg.dassert_isinstance(tier, str, "tier names must be strings")
        hdbg.dassert_ne(tier.strip(), "", "tier names must not be blank")
    hdbg.dassert_eq(len(tiers), len(set(tiers)), "tiers must be distinct")
    rng = random.Random(seed)
    bids = []
    for index in range(n_bids):
        tier = rng.choice(tiers)
        quantity = rng.randint(1, 100)
        price = float(rng.randint(1, 20))
        bids.append(
            rnbacaau.Bid(f"buyer_{index}", quantity, tier, 2.0, 0.90, price)
        )
    asks = []
    for index in range(n_asks):
        tier = rng.choice(tiers)
        quantity = rng.randint(1, 100)
        price = float(rng.randint(1, 20))
        asks.append(
            rnbacaau.Ask(f"seller_{index}", quantity, tier, 1.0, 0.95, price)
        )
    scenario = (bids, asks)
    _LOG.debug("return=%s", (len(bids), len(asks)))
    return scenario
