"""
Test the benchmark's statistics and one end-to-end task.

Import as:

import test.test_tukey_benchmark as ttetuben
"""

import json
import logging
import os
import unittest.mock as umock

import numpy as np

import helpers.hunit_test as hunitest
import tukey_benchmark as tbench
import tukey_sets as tsets

_LOG = logging.getLogger(__name__)

# The committed manifests, found from this file so the tests run from any
# directory.
_MANIFEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs/sets",
)


class Test_statistics(hunitest.TestCase):
    """
    Check the intervals and tests against hand-computed values.
    """

    def test1(self) -> None:
        """
        Wilson interval of 8/10 and exact McNemar of 7 vs 1 discordant pairs.
        """
        # Run test.
        low, high = tbench.wilson_interval(8, 10)
        actual = (
            f"wilson {low:.4f} {high:.4f}\n"
            f"mcnemar {tbench.mcnemar_exact(7, 1):.4f} "
            f"{tbench.mcnemar_exact(0, 0):.4f}"
        )
        # Check outputs.
        # Wilson 8/10: 0.4902-0.9433; McNemar: 2 * P(Bin(8, 1/2) <= 1) = 0.0703.
        expected = """
        wilson 0.4902 0.9433
        mcnemar 0.0703 1.0000
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        A constant sample has a degenerate bootstrap interval.
        """
        # Run test and check outputs.
        self.assertEqual(tbench.bootstrap_ci(np.full(20, 0.25)), (0.25, 0.25))

    def test3(self) -> None:
        """
        Paired summary: differences, exact recovery and win/tie/loss.
        """
        # Prepare inputs.
        reference = [
            {
                "seed": s,
                "score": -0.5,
                "shd_target": 2,
                "shd_dag": 2,
                "skeleton_f1_emitted": 0.8,
                "orientation_accuracy_definite": 0.5,
                "orientation_coverage": 1.0,
                "calls": 10,
                "seconds": 0.1,
            }
            for s in range(4)
        ]
        results = [dict(r) for r in reference]
        results[0].update(score=0.0, shd_target=0)
        results[1].update(score=-0.75, shd_target=3)
        # Run test.
        summary = tbench.summarize(results, reference)
        # Check outputs.
        actual = (
            f"score_minus_proto {summary['score_minus_proto']}\n"
            f"exact_recovery {summary['exact_recovery']}\n"
            f"win_tie_loss {summary['win_tie_loss_vs_proto']}"
        )
        expected = """
        score_minus_proto 0.0625
        exact_recovery 0.25
        win_tie_loss (1, 2, 1)
        """
        self.assert_equal(actual, expected, dedent=True)


class Test_run_method(hunitest.TestCase):
    """
    One task of each kind runs end to end.
    """

    def test1(self) -> None:
        """
        Scripted and baseline methods return the same keys on a prototype graph.
        """
        # Prepare inputs.
        eval_set = tsets.EVAL_SETS["V_m2_A0"]
        seed = eval_set.seeds[0]
        cfg = tsets.TASKS[eval_set.task](seed)
        keys = None
        # Run test and check outputs.
        for method, params in (("proto_scripted", {}), ("pc", {"alpha": 0.05})):
            result = tbench.run_method(method, params, cfg, seed)
            keys = keys or sorted(result)
            self.assertEqual(sorted(result), keys)
            self.assertTrue(np.isfinite(result["score"]))

    def test2(self) -> None:
        """
        Both paths refuse an instance whose hash differs from the manifest.
        """
        # Prepare inputs.
        eval_set = tsets.EVAL_SETS["V_m2_A0"]
        seed = eval_set.seeds[0]
        cfg = tsets.TASKS[eval_set.task](seed)
        # Run test and check outputs.
        # The check is strict on the machine type that wrote the manifests.
        with umock.patch("platform.machine", return_value="aarch64"):
            for method, params in (
                ("proto_scripted", {}),
                ("pc", {"alpha": 0.05}),
            ):
                with self.assertRaises(AssertionError, msg=method):
                    tbench.run_method(
                        method, params, cfg, seed, expected_hash="0"
                    )

    def test3(self) -> None:
        """
        Checked tasks give the same results as unchecked runs.
        """
        # Prepare inputs.
        tasks = tbench.make_tasks(
            "V_m2_A0", "pc", {"alpha": 0.05}, _MANIFEST_DIR
        )
        # Run test.
        results = tbench.run_tasks(tasks[:3], n_workers=1)
        # Check outputs.
        for task, result in zip(tasks[:3], results):
            set_name, method, params, seed, _ = task
            cfg = tsets.TASKS[tsets.EVAL_SETS[set_name].task](seed)
            unchecked = tbench.run_method(method, params, cfg, seed)
            self.assertEqual(result["score"], unchecked["score"])
            self.assertEqual(result["edge_marks"], unchecked["edge_marks"])

    def test4(self) -> None:
        """
        Tuning and the table stop on a tampered manifest before running.
        """
        # Prepare inputs.
        manifest_dir = self.get_scratch_space()
        # Each set's own manifest with its first seed changed.
        for name in ("V_m2_A0", "B_m2_A0"):
            with open(tsets.manifest_path(_MANIFEST_DIR, name)) as file:
                manifest = json.load(file)
            manifest["instances"][0]["seed"] += 1
            with open(tsets.manifest_path(manifest_dir, name), "w") as file:
                json.dump(manifest, file)
        # Run test and check outputs.
        with umock.patch.object(tbench, "SETTINGS", ("A0",)):
            with self.assertRaises(AssertionError):
                tbench.tune(n_workers=1, manifest_dir=manifest_dir)
            tuned = {"A0": {m: {"params": {}} for m in tbench.GRIDS}}
            with self.assertRaises(AssertionError):
                tbench.benchmark(tuned, n_workers=1, manifest_dir=manifest_dir)

    def test5(self) -> None:
        """
        A hash mismatch inside a worker stops the run.
        """
        # Prepare inputs.
        tasks = tbench.make_tasks(
            "V_m2_A0", "pc", {"alpha": 0.05}, _MANIFEST_DIR
        )
        set_name, method, params, seed, _ = tasks[0]
        # Run test and check outputs.
        # The check is strict on the machine type that wrote the manifests.
        # The worker sees the patch because the pool forks this process.
        with umock.patch("platform.machine", return_value="aarch64"):
            with self.assertRaises(AssertionError):
                tbench.run_tasks(
                    [(set_name, method, params, seed, "0")], n_workers=1
                )

    def test6(self) -> None:
        """
        The stored Milestone 2 rows of two setting-D graphs are reproduced.
        """
        # Prepare inputs.
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(project_dir, "results/m2/per_graph.json")) as file:
            stored = json.load(file)["settings"]["D"]
        with open(
            os.path.join(project_dir, "results/m2/table_summary.json")
        ) as file:
            tuned = json.load(file)["tuned"]["D"]
        items = tsets.load_set("B_m2_D", _MANIFEST_DIR)[:2]
        mismatches = []
        # Run test and check outputs.
        for method, values in sorted(stored["methods"].items()):
            if (
                method == "var_sort_regress"
                and not tsets.is_reference_machine()
            ):
                # On standardized data every variance is 1 up to rounding
                # noise, so this method's order depends on the CPU.
                continue
            params = tbench.FIXED.get(method, None)
            if params is None:
                params = tuned[method]["params"]
            for k, (seed, cfg, expected_hash) in enumerate(items):
                self.assertEqual(seed, stored["seeds"][k])
                result = tbench.run_method(
                    method, params, cfg, seed, expected_hash=expected_hash
                )
                actual = (round(result["score"], 4), int(result["shd_target"]))
                wanted = (values["score"][k], values["shd_target"][k])
                if actual != wanted:
                    mismatches.append((method, seed, actual, wanted))
        self.assertEqual(mismatches, [])
