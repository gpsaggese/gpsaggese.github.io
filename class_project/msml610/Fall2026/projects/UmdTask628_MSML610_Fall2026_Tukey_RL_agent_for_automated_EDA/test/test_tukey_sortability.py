"""
Test the sortability wrapper on graphs whose values are known by hand.

Import as:

import test.test_tukey_sortability as ttetusor
"""

import logging

import CausalDisco.analytics as cdanaly
import numpy as np

import helpers.hunit_test as hunitest
import tukey_env as tenv
import tukey_sortability as tsortab

_LOG = logging.getLogger(__name__)


class Test_sortabilities(hunitest.TestCase):
    """
    Check the definitions on small graphs and simulated data.
    """

    def test1(self) -> None:
        """
        Relatives: a chain ties everywhere (1/2), a collider is sorted (1).
        """
        chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        collider = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        lines = []
        for name, B in (("chain", chain), ("collider", collider)):
            counts = cdanaly.count_relatives(B).tolist()
            edges = cdanaly.true_relatives_sortability(
                B, measure="adjacent_pairs"
            )
            lines.append(f"{name} counts {counts}, edges {edges}")
        actual = "\n".join(lines)
        expected = """
        chain counts [3, 3, 3], edges 0.5
        collider counts [2, 2, 3], edges 1.0
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        A negative weight flips CausalDisco's edge measure; binary does not.
        """
        scores = np.array([1.0, 2.0])
        weighted = np.array([[0.0, -0.8], [0.0, 0.0]])
        binary = (weighted != 0).astype(float)
        flipped = cdanaly.order_alignment_adjacent_pairs(weighted, scores)
        correct = cdanaly.order_alignment_adjacent_pairs(binary, scores)
        self.assertEqual((flipped, correct), (0.0, 1.0))

    def test3(self) -> None:
        """
        Prototype data: raw variance sorts the order; standardizing removes it.
        """
        inst = tenv.sample_instance(tenv.PROTO_CONFIG, 0)
        x = inst.d_train.to_numpy()
        raw = tsortab.sortabilities(x, inst.W)
        z = (x - x.mean(axis=0)) / x.std(axis=0)
        standardized = tsortab.sortabilities(z, inst.W)
        self.assertGreater(raw["var_edges"], 0.8)
        self.assertEqual(standardized["var_edges"], 0.5)
        # Relatives and R2 depend on correlations only.
        for key in ("r2_edges", "rel_emp_edges", "rel_oracle_edges"):
            self.assertAlmostEqual(raw[key], standardized[key], places=6)
        self.assertGreaterEqual(raw["rel_oracle_edges"], 0.5)
