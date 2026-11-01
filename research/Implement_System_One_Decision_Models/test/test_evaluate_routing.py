"""Check routing metrics against small hand-computable predictions."""

import logging
import math

import helpers.hunit_test as hunitest
import research.Implement_System_One_Decision_Models.evaluate_routing as eroute

_LOG = logging.getLogger(__name__)


class Test_classification_metrics(hunitest.TestCase):
    """Test `evaluate_routing.classification_metrics` for three-class scores."""

    def test1(self) -> None:
        """Compute accuracy, Brier, and log loss from two correct predictions."""
        records = [
            {
                "true_route": "billing",
                "top_route": "billing",
                "probabilities": {
                    "billing": 0.8,
                    "shipping": 0.1,
                    "technical": 0.1,
                },
            },
            {
                "true_route": "shipping",
                "top_route": "shipping",
                "probabilities": {
                    "billing": 0.2,
                    "shipping": 0.7,
                    "technical": 0.1,
                },
            },
        ]
        result = eroute.classification_metrics(records)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["multiclass_brier"], 0.1)
        self.assertAlmostEqual(result["nll"], -(math.log(0.8) + math.log(0.7)) / 2)


class Test_reliability_bins(hunitest.TestCase):
    """Test `evaluate_routing.reliability_bins` for top-label calibration."""

    def test1(self) -> None:
        """Weight one occupied bin by its fraction of all predictions."""
        records = [
            {
                "true_route": "billing",
                "top_route": "billing",
                "probabilities": {"billing": 0.8, "shipping": 0.1, "technical": 0.1},
            },
            {
                "true_route": "shipping",
                "top_route": "shipping",
                "probabilities": {"billing": 0.2, "shipping": 0.7, "technical": 0.1},
            },
        ]
        bins, ece = eroute.reliability_bins(records, n_bins=2)
        self.assertEqual(bins[0]["count"], 0)
        self.assertEqual(bins[1]["count"], 2)
        self.assertAlmostEqual(bins[1]["mean_confidence"], 0.75)
        self.assertAlmostEqual(ece, 0.25)
