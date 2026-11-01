"""Check fixed ticket splits and the action-cost routing policy."""

import collections
import logging

import helpers.hunit_test as hunitest
import research.Implement_System_One_Decision_Models.ticket_routing as troute

_LOG = logging.getLogger(__name__)


class Test_load_cases(hunitest.TestCase):
    """Test `ticket_routing.load_cases` on the frozen synthetic data."""

    def test1(self) -> None:
        """Keep disjoint balanced development and test splits."""
        dev = troute.load_cases("dev")
        test = troute.load_cases("test")
        self.assertEqual(len(dev), 36)
        self.assertEqual(len(test), 36)
        self.assertEqual(
            collections.Counter(row["label"] for row in dev),
            {"billing": 12, "shipping": 12, "technical": 12},
        )
        self.assertEqual(
            collections.Counter(row["label"] for row in test),
            {"billing": 12, "shipping": 12, "technical": 12},
        )
        self.assertFalse({row["id"] for row in dev} & {row["id"] for row in test})


class Test_choose_min_cost_action(hunitest.TestCase):
    """Test `ticket_routing.choose_min_cost_action` on known distributions."""

    def test1(self) -> None:
        """Route a certain billing ticket automatically."""
        probabilities = {"billing": 1.0, "shipping": 0.0, "technical": 0.0}
        self.assertEqual(troute.choose_min_cost_action(probabilities), "billing")
        self.assertEqual(troute.expected_costs(probabilities)["billing"], 0.0)

    def test2(self) -> None:
        """Send a uniform distribution to human review."""
        probabilities = {
            "billing": 1 / 3,
            "shipping": 1 / 3,
            "technical": 1 / 3,
        }
        self.assertEqual(troute.choose_min_cost_action(probabilities), "review")
        self.assertEqual(troute.expected_costs(probabilities)["review"], 1.5)
