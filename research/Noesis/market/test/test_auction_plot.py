"""
Test figure structure and step coordinates without pixel comparisons.

Import as:

import research.Noesis.market.test.test_auction_plot as rnmatempl
"""

from typing import Optional, Tuple

import helpers.hunit_test as hunitest

import research.Noesis.market.auction_plot as rnmaaupl
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.scenarios as rnmascen

# #############################################################################
# Test_plot_supply_demand
# #############################################################################


class Test_plot_supply_demand(hunitest.TestCase):
    """
    Check curves, tier isolation, clearing markers, and boundary cases.
    """

    def helper(
        self,
        scenario: str,
        tier: str,
        price: Optional[float],
        expected: Tuple[object, ...],
    ) -> None:
        """
        Compare one seeded figure's complete structural description.

        :param scenario: named M11 family
        :param tier: tier to display
        :param price: clearing price from the fixed scenario
        :param expected: curve coordinates and clearing offsets
        """
        # Prepare inputs.
        factory = getattr(rnmascen, f"make_{scenario}_scenario")
        bids, asks = factory(seed=0, n_bids=1, n_asks=1)
        before = repr((bids, asks))
        # Run test.
        figure = rnmaaupl.plot_supply_demand(tier, bids, asks, price)
        # Check outputs: curve geometry and markers, never rendered pixels.
        ax = figure.axes[0]
        curves = [
            (
                line.get_label(),
                line.get_drawstyle(),
                line.get_xdata().tolist(),
                line.get_ydata().tolist(),
            )
            for line in ax.lines[:2]
        ]
        if price is not None:
            self.assertEqual(list(ax.lines[-1].get_ydata()), [price, price])
        markers = [
            collection.get_offsets().tolist() for collection in ax.collections
        ]
        actual = (curves, markers, len(ax.lines), len(figure.axes))
        self.assert_equal(str(actual), str(expected))
        self.assertEqual(
            (ax.get_xlabel(), ax.get_ylabel()), ("Tasks", "Price per task")
        )
        self.assertEqual(repr((bids, asks)), before)

    def test1(self) -> None:
        """
        Check the balanced plot against fixed coordinates.
        """
        # Prepare inputs and expected outputs.
        scenario, tier, price = "balanced", "standard", 5.0
        expected = (
            [
                ("Demand", "steps-post", [0, 98], [9.0, 9.0]),
                ("Supply", "steps-post", [0, 98], [1.0, 1.0]),
            ],
            [[[98.0, 5.0]]],
            3,
            1,
        )
        # Run test and check outputs.
        self.helper(scenario, tier, price, expected)

    def test2(self) -> None:
        """
        Check the excess_demand plot against fixed coordinates.
        """
        # Prepare inputs and expected outputs.
        scenario, tier, price = "excess_demand", "standard", 6.0
        expected = (
            [
                ("Demand", "steps-post", [0, 104], [8.0, 8.0]),
                ("Supply", "steps-post", [0, 98], [4.0, 4.0]),
            ],
            [[[98.0, 6.0]]],
            3,
            1,
        )
        # Run test and check outputs.
        self.helper(scenario, tier, price, expected)

    def test3(self) -> None:
        """
        Check the excess_supply plot against fixed coordinates.
        """
        # Prepare inputs and expected outputs.
        scenario, tier, price = "excess_supply", "standard", 6.0
        expected = (
            [
                ("Demand", "steps-post", [0, 98], [8.0, 8.0]),
                ("Supply", "steps-post", [0, 104], [4.0, 4.0]),
            ],
            [[[98.0, 6.0]]],
            3,
            1,
        )
        # Run test and check outputs.
        self.helper(scenario, tier, price, expected)

    def test4(self) -> None:
        """
        Check the multi_tier plot against fixed coordinates.
        """
        # Prepare inputs and expected outputs.
        scenario, tier, price = "multi_tier", "frontier", 26.5
        expected = (
            [
                ("Demand", "steps-post", [0, 66], [29.0, 29.0]),
                ("Supply", "steps-post", [0, 66], [24.0, 24.0]),
            ],
            [[[66.0, 26.5]]],
            3,
            1,
        )
        # Run test and check outputs.
        self.helper(scenario, tier, price, expected)

    def test5(self) -> None:
        """
        Check the no_cross plot against fixed coordinates.
        """
        # Prepare inputs and expected outputs.
        scenario, tier, price = "no_cross", "standard", None
        expected = (
            [
                ("Demand", "steps-post", [0, 98], [4.0, 4.0]),
                ("Supply", "steps-post", [0, 98], [6.0, 6.0]),
            ],
            [],
            2,
            1,
        )
        # Run test and check outputs.
        self.helper(scenario, tier, price, expected)

    def test6(self) -> None:
        """
        Return an empty, labeled figure for a tier with no orders.
        """
        # Prepare inputs.
        tier, bids, asks, price = "missing", [], [], None
        # Run test.
        figure = rnmaaupl.plot_supply_demand(tier, bids, asks, price)
        # Check outputs.
        self.assertEqual((len(figure.axes), len(figure.axes[0].lines)), (1, 0))

    def test7(self) -> None:
        """
        Place price jumps after each order's full quantity, not one step early.
        """
        # Prepare inputs: unequal quantities expose incorrect step alignment.
        tier, price = "standard", None
        bids = [
            rnbacaau.Bid("b1", 3, tier, 2, 0.9, 9),
            rnbacaau.Bid("b2", 7, tier, 2, 0.9, 5),
        ]
        asks = []
        # Run test.
        figure = rnmaaupl.plot_supply_demand(tier, bids, asks, price)
        # Check outputs.
        line = figure.axes[0].lines[0]
        self.assertEqual(
            (line.get_xdata().tolist(), line.get_ydata().tolist()),
            ([0, 3, 10], [9, 5, 5]),
        )
