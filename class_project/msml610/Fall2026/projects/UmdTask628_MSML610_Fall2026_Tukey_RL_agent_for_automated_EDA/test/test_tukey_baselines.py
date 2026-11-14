"""
Test the baselines: output conventions, PDAG conversion and scoring.

A transposed adjacency would reverse every edge, so each library's convention
is checked on data where the baseline is known to recover the graph.

Import as:

import test.test_tukey_baselines as ttetubas
"""

import logging
import unittest.mock as umock

import numpy as np

import helpers.hunit_test as hunitest
import tukey_baselines as tbasel
import tukey_env as tenv
import tukey_scoring as tscore

_LOG = logging.getLogger(__name__)


class Test_causallearn_to_pdag(hunitest.TestCase):
    """
    Convert causal-learn's edge encoding.
    """

    def test1(self) -> None:
        """
        Directed both ways, undirected, bidirected and absent pairs.
        """
        graph = np.zeros((5, 5), dtype=int)
        # 0 -> 1; 3 -> 2; 1 - 2; 3 <-> 4.
        graph[0, 1], graph[1, 0] = -1, 1
        graph[2, 3], graph[3, 2] = 1, -1
        graph[1, 2], graph[2, 1] = -1, -1
        graph[3, 4], graph[4, 3] = 1, 1
        A, bidirected = tbasel.causallearn_to_pdag(graph)
        edges = [(int(i), int(j)) for i, j in np.argwhere(A == 1)]
        pairs = [(int(i), int(j)) for i, j in np.argwhere(bidirected)]
        actual = f"edges {edges}\nbidirected {pairs}"
        expected = """
        edges [(0, 1), (1, 2), (2, 1), (3, 2), (3, 4), (4, 3)]
        bidirected [(3, 4), (4, 3)]
        """
        self.assert_equal(actual, expected, dedent=True)


class Test_run_baseline(hunitest.TestCase):
    """
    Check each library's orientation convention on easy data.
    """

    def test1(self) -> None:
        """
        DirectLiNGAM recovers non-Gaussian 4-variable graphs (no reversal).
        """
        cfg = tenv.GraphConfig(family="linear_exp", n_nodes=4, n_edges=4)
        shd = []
        for seed in range(5):
            inst = tenv.sample_instance(cfg, seed)
            output = tbasel.run_baseline(
                "direct_lingam", inst.d_train.to_numpy()
            )
            scores = tbasel.score_output(output, inst.truth)
            shd.append(scores["shd_dag"])
        self.assertLessEqual(sum(shd), 2)

    def test2(self) -> None:
        """
        On raw prototype data sorting by variance recovers most edges.
        """
        shd = []
        n_edges = []
        for seed in range(5):
            inst = tenv.sample_instance(tenv.PROTO_CONFIG, seed)
            output = tbasel.run_baseline(
                "var_sort_regress", inst.d_train.to_numpy()
            )
            shd.append(tbasel.score_output(output, inst.truth)["shd_dag"])
            n_edges.append(int(inst.truth.B.sum()))
        # A transposed output would reverse every edge (SHD near the edge count).
        self.assertLess(sum(shd), 0.5 * sum(n_edges))

    def test3(self) -> None:
        """
        PC on a clean collider returns its CPDAG; empty scores |E|.
        """
        B = np.zeros((3, 3), dtype=int)
        B[0, 2] = B[1, 2] = 1
        rng = np.random.default_rng(1)
        x = np.zeros((6000, 3))
        x[:, 0] = rng.normal(size=6000)
        x[:, 1] = rng.normal(size=6000)
        x[:, 2] = 0.8 * x[:, 0] + 0.8 * x[:, 1] + rng.normal(size=6000)
        truth = tscore.make_truth(
            B, tscore.TARGET_CPDAG, np.zeros(3, bool), x[:4000], x[4000:]
        )
        pc = tbasel.score_output(tbasel.run_baseline("pc", x[:4000]), truth)
        empty = tbasel.score_output(
            tbasel.run_baseline("empty", x[:4000]), truth
        )
        actual = (
            f"pc shd_target {pc['shd_target']}, "
            f"bidirected {pc['n_bidirected']}\n"
            f"empty shd_target {empty['shd_target']}, score {empty['score']}"
        )
        expected = """
        pc shd_target 0, bidirected 0
        empty shd_target 2, score -1.0
        """
        self.assert_equal(actual, expected, dedent=True)

    def test4(self) -> None:
        """
        PC on two variables runs one CI test; the cache holds 3 more entries.
        """
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 2))
        x[:, 1] += x[:, 0]
        output = tbasel.run_baseline("pc", x)
        actual = [output.n_ci_tests, output.n_ci_tests_legacy]
        self.assert_equal(str(actual), "[1, 4]")

    def test5(self) -> None:
        """
        GES on a clean collider returns its CPDAG (no reversed edge).
        """
        rng = np.random.default_rng(1)
        x = np.zeros((6000, 3))
        x[:, 0] = rng.normal(size=6000)
        x[:, 1] = rng.normal(size=6000)
        x[:, 2] = 0.8 * x[:, 0] + 0.8 * x[:, 1] + rng.normal(size=6000)
        B = np.zeros((3, 3), dtype=int)
        B[0, 2] = B[1, 2] = 1
        truth = tscore.make_truth(
            B, tscore.TARGET_CPDAG, np.zeros(3, bool), x[:4000], x[4000:]
        )
        scores = tbasel.score_output(tbasel.run_baseline("ges", x[:4000]), truth)
        self.assertEqual((scores["shd_target"], scores["score"]), (0, 0.0))

    def test6(self) -> None:
        """
        Every baseline returns a well-formed graph that scores on a main graph.
        """
        cfg = tenv.GraphConfig(family="linear_exp", n_nodes=5, n_edges=6)
        inst = tenv.sample_instance(cfg, 3)
        x = inst.d_train.to_numpy()
        for name in tbasel.BASELINES:
            output = tbasel.run_baseline(name, x)
            self.assertEqual(output.adjacency.shape, (5, 5), msg=name)
            if output.kind == "dag":
                self.assertTrue(tscore.is_dag(output.adjacency), msg=name)
            scores = tbasel.score_output(output, inst.truth)
            self.assertIn("shd_target", scores, msg=name)

    def test7(self) -> None:
        """
        On raw prototype data the R2 and relatives sorts recover most edges.
        """
        for name in ("r2_sort_regress", "rel_sort_regress"):
            shd, n_edges = [], []
            for seed in range(5):
                inst = tenv.sample_instance(tenv.PROTO_CONFIG, seed)
                output = tbasel.run_baseline(name, inst.d_train.to_numpy())
                shd.append(tbasel.score_output(output, inst.truth)["shd_dag"])
                n_edges.append(int(inst.truth.B.sum()))
            # A transposed output would reverse every edge.
            self.assertLess(sum(shd), sum(n_edges), msg=name)

    def test8(self) -> None:
        """
        Variables that tie on the relatives count are ordered by index.
        """
        cfg = tenv.GraphConfig(family="linear_exp", n_nodes=5, n_edges=6)
        x = tenv.sample_instance(cfg, 3).d_train.to_numpy()
        # Each row: the counts, and the rank each variable must get. The ranks
        # have no ties, so the sort inside CausalDisco cannot reorder them.
        cases = (
            ([0, 0, 0, 0, 0], [0, 1, 2, 3, 4]),
            ([1, 0, 1, 0, 0], [3, 0, 4, 1, 2]),
        )
        for counts, ranks in cases:
            with (
                umock.patch.object(
                    tbasel.cdanaly,
                    "rel_count_emp_hard",
                    return_value=np.array(counts),
                ),
                umock.patch.object(
                    tbasel.cdbasel,
                    "sort_regress",
                    wraps=tbasel.cdbasel.sort_regress,
                ) as sort_regress,
            ):
                output = tbasel.run_baseline("rel_sort_regress", x)
            passed = sort_regress.call_args.args[1]
            self.assertEqual(passed.tolist(), ranks, msg=str(counts))
            # An edge can only go from a lower to a higher rank.
            for i, j in zip(*np.nonzero(output.adjacency)):
                self.assertLess(ranks[i], ranks[j], msg=str(counts))


class Test_score_output(hunitest.TestCase):
    """
    A bidirected conflict is its own mark.
    """

    def test1(self) -> None:
        """
        A bidirected pair matches neither direction nor an undirected edge.
        """
        B = np.zeros((3, 3), dtype=int)
        B[0, 1] = 1
        rng = np.random.default_rng(2)
        x = rng.normal(size=(3000, 3))
        x[:, 1] += 0.8 * x[:, 0]
        truth = tscore.make_truth(
            B, tscore.TARGET_CPDAG, np.zeros(3, bool), x[:2000], x[2000:]
        )
        A = np.zeros((3, 3), dtype=int)
        A[0, 1] = A[1, 0] = 1
        bidirected = np.zeros((3, 3), dtype=bool)
        plain = tscore.score_pdag(A, truth)
        bidirected[0, 1] = bidirected[1, 0] = True
        conflict = tscore.score_pdag(A, truth, bidirected=bidirected)
        actual = (
            f"undirected: shd_target {plain['shd_target']}\n"
            f"bidirected: shd_target {conflict['shd_target']}, "
            f"n {conflict['n_bidirected']}"
        )
        expected = """
        undirected: shd_target 0
        bidirected: shd_target 1, n 1
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        A PDAG without extension on an uninformative instance: structure only.

        The truth has no edge, so its held-out gain is 0 and normal scoring is
        structural only (`-shd_norm`); the fallback now agrees, where the
        legacy fallback also subtracted the prediction penalty.
        """
        B = np.zeros((4, 4), dtype=int)
        rng = np.random.default_rng(3)
        x = rng.normal(size=(3000, 4))
        truth = tscore.make_truth(
            B, tscore.TARGET_DAG, np.zeros(4, bool), x[:2000], x[2000:]
        )
        # An undirected 4-cycle has no consistent DAG extension.
        A = np.zeros((4, 4), dtype=int)
        for i, j in ((0, 1), (1, 2), (2, 3), (0, 3)):
            A[i, j] = A[j, i] = 1
        output = tbasel.BaselineOutput(
            name="pc", kind="pdag", adjacency=A, bidirected=None, seconds=0.0
        )
        scores = tbasel.score_output(output, truth)
        legacy = tbasel.fallback_score_legacy(
            scores["shd_target"], scores["n_true_edges"], lam=0.5
        )
        actual = [
            scores["score_fallback"],
            scores["prediction_informative"],
            scores["shd_target"],
            scores["score"],
            legacy,
        ]
        self.assert_equal(str(actual), "[True, False, 4, -4.0, -2.5]")

    def test3(self) -> None:
        """
        On an informative instance the fallback keeps the prediction penalty.
        """
        actual = [
            tbasel.fallback_score(2, 4, informative=True, lam=0.5),
            tbasel.fallback_score_legacy(2, 4, lam=0.5),
            tbasel.fallback_score(2, 4, informative=False, lam=0.5),
        ]
        self.assert_equal(str(actual), "[-0.75, -0.75, -0.5]")


class Test_fci(hunitest.TestCase):
    """
    Check FCI's output convention on a hidden common cause it can identify.
    """

    def test1(self) -> None:
        """
        FCI finds T1 <-> T2 with private causes A -> T1 and B -> T2.
        """
        rng = np.random.default_rng(0)
        a, b, u = rng.normal(size=(3, 2000))
        t1 = a + u + rng.normal(size=2000)
        t2 = b + u + rng.normal(size=2000)
        # Columns: T2, A, T1, B.
        x = np.column_stack([t2, a, t1, b])
        output = tbasel.run_baseline("fci", x)
        self.assertEqual(output.kind, "pag")
        P = output.adjacency
        self.assertEqual((P[0, 2], P[2, 0]), (1, 1))
        self.assertEqual((P[1, 2], P[2, 1]), (2, 1))
        self.assertEqual((P[3, 0], P[0, 3]), (2, 1))
        self.assertEqual(int((P != 0).sum()), 6)
