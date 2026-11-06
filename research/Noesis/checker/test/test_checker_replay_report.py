"""
Test aggregation and artifact rendering for offline checker replay.
"""

import csv
import io
import json
import pathlib
import tempfile

import helpers.hunit_test as hunitest
import research.Noesis.checker.checker_replay as rnochrep
import research.Noesis.checker.checker_replay_report as rnochrerp


class Test_run_replay_experiment(hunitest.TestCase):
    """
    Test aggregation across scenarios and seeds.
    """

    def test1(self) -> None:
        """
        Test deterministic all-success runs aggregate without false failures.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("healthy", 50, 10, 1.0, 1.0)
        # Run test.
        actual = rnochrerp.run_replay_experiment(
            (0, 1),
            5,
            scenarios=(scenario,),
        )
        # Check outputs.
        self.assertTrue(actual.passed)
        self.assertEqual(actual.seeds, (0, 1))
        self.assertEqual(actual.scenarios[0].total_contracts, 10)
        self.assertEqual(actual.scenarios[0].verdict_counts["PASSED"], 10)
        self.assertEqual(actual.scenarios[0].verdict_counts["FAILED"], 0)

    def test2(self) -> None:
        """
        Test latency degradation stays separately attributable after aggregation.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("slow", 50, 10, 0.0, 1.0)
        # Run test.
        actual = rnochrerp.run_replay_experiment(
            (0, 1),
            5,
            scenarios=(scenario,),
        ).scenarios[0]
        # Check outputs.
        self.assertEqual(actual.latency_failure_count, 10)
        self.assertEqual(actual.quality_failure_count, 0)
        self.assertEqual(actual.both_failure_count, 0)
        self.assertTrue(actual.passed)


class Test_render_replay_artifacts(hunitest.TestCase):
    """
    Test stable JSON, CSV, and Markdown replay artifacts.
    """

    @staticmethod
    def _result() -> rnochrerp.ReplayExperimentResult:
        scenario = rnochrep.ReplayScenario("healthy", 50, 10, 1.0, 1.0)
        return rnochrerp.run_replay_experiment(
            (0, 1),
            3,
            scenarios=(scenario,),
        )

    def test1(self) -> None:
        """
        Test JSON preserves the protocol, aggregate, and per-seed runs.
        """
        # Run test.
        actual = json.loads(rnochrerp.render_replay_json(self._result()))
        # Check outputs.
        self.assertEqual(actual["schema_version"], "noesis-checker-replay-v1")
        self.assertEqual(actual["seeds"], [0, 1])
        self.assertEqual(len(actual["scenarios"][0]["runs"]), 2)
        self.assertEqual(
            actual["scenarios"][0]["aggregate"]["total_contracts"],
            6,
        )
        self.assertEqual(
            actual["scenarios"][0]["aggregate"]["analytic_latency_failure_rate"],
            0.0,
        )
        self.assertTrue(actual["passed"])

    def test2(self) -> None:
        """
        Test CSV contains two seed rows and one aggregate row.
        """
        # Run test.
        rows = list(
            csv.DictReader(
                io.StringIO(rnochrerp.render_replay_csv(self._result()))
            )
        )
        # Check outputs.
        self.assertEqual(len(rows), 3)
        self.assert_equal(
            str([row["seed"] for row in rows]),
            str(["0", "1", "aggregate"]),
        )
        self.assertEqual(rows[-1]["passed_acceptance"], "True")

    def test3(self) -> None:
        """
        Test Markdown makes acceptance and the synthetic boundary explicit.
        """
        # Run test.
        actual = rnochrerp.render_replay_markdown(self._result())
        # Check outputs.
        self.assertIn("Overall acceptance: **PASS**", actual)
        self.assertIn("| healthy | 6 | 6 | 0 | 0 |", actual)
        self.assertIn("## Failure attribution", actual)
        self.assertIn("| healthy | 0 | 0 | 0 |", actual)
        self.assertIn("deterministic synthetic replay results", actual)
        self.assertIn("claims about a live provider or prove", actual)

    def test4(self) -> None:
        """
        Test the writer creates all three parseable artifacts.
        """
        with tempfile.TemporaryDirectory() as directory:
            # Run test.
            paths = rnochrerp.write_replay_artifacts(
                self._result(),
                pathlib.Path(directory),
            )
            # Check outputs.
            self.assertTrue(paths.json_path.is_file())
            self.assertTrue(paths.csv_path.is_file())
            self.assertTrue(paths.markdown_path.is_file())
            json.loads(paths.json_path.read_text(encoding="utf-8"))
            self.assertIn(
                "Overall acceptance: **PASS**",
                paths.markdown_path.read_text(encoding="utf-8"),
            )
