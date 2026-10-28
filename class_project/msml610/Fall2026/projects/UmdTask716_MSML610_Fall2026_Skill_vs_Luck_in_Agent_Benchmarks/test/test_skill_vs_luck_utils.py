"""
Import as:

import test.test_skill_vs_luck_utils as ttsvlut
"""

import logging
import pprint

import numpy as np
import pandas as pd

import helpers.hunit_test as hunitest

import skill_vs_luck_utils as utils

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_parse_results_json
# #############################################################################


class Test_parse_results_json(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.parse_results_json()`.
    """

    def test1(self) -> None:
        """
        Test that the solved tasks become 1 and the other tasks 0.
        """
        # Prepare inputs.
        results = {"resolved": ["task_b"], "no_generation": ["task_c"]}
        task_ids = ["task_a", "task_b", "task_c"]
        # Prepare outputs.
        expected = "{'task_a': 0, 'task_b': 1, 'task_c': 0}"
        # Run test.
        outcomes = utils.parse_results_json(results, task_ids)
        # Check outputs.
        actual = str(outcomes.to_dict())
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test a submission that solved no task.
        """
        # Prepare inputs.
        results = {}
        task_ids = ["task_a", "task_b"]
        # Prepare outputs.
        expected = "{'task_a': 0, 'task_b': 0}"
        # Run test.
        outcomes = utils.parse_results_json(results, task_ids)
        # Check outputs.
        actual = str(outcomes.to_dict())
        self.assert_equal(actual, expected)


# #############################################################################
# Test_parse_per_instance_details
# #############################################################################


class Test_parse_per_instance_details(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.parse_per_instance_details()`.
    """

    def test1(self) -> None:
        """
        Test that the records follow the order of the task ids.
        """
        # Prepare inputs.
        details = {
            "task_b": {"cost": 0.5, "api_calls": 30, "resolved": False},
            "task_a": {"cost": 0.25, "api_calls": 12, "resolved": True},
        }
        task_ids = ["task_a", "task_b"]
        # Prepare outputs.
        expected = """
                resolved  cost  api_calls
        task_a         1  0.25         12
        task_b         0  0.50         30
        """
        # Run test.
        records = utils.parse_per_instance_details(details, task_ids)
        # Check outputs.
        actual = records.to_string()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_parse_metadata
# #############################################################################


class Test_parse_metadata(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.parse_metadata()`.
    """

    def test1(self) -> None:
        """
        Test a full `metadata.yaml` where some fields are lists.
        """
        # Prepare inputs.
        metadata = {
            "info": {"name": "GPT 5 mini", "resolved": 56.2},
            "tags": {
                "agent": "mini-SWE-agent",
                "model": ["gpt-5-mini-2025-08-07"],
                "org": ["OpenAI"],
                "system": {"attempts": 1},
            },
        }
        folder = "20260217_mini-v2.0.0_gpt-5-mini"
        # Prepare outputs.
        expected = """
        {'agent': 'mini-SWE-agent',
         'attempts': '1',
         'date': '2026-02-17',
         'model': 'gpt-5-mini-2025-08-07',
         'name': 'GPT 5 mini',
         'org': 'OpenAI',
         'reported_score': 56.2,
         'submission': '20260217_mini-v2.0.0_gpt-5-mini'}
        """
        # Run test.
        fields = utils.parse_metadata(metadata, folder)
        # Check outputs.
        actual = pprint.pformat(fields)
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test a submission without `metadata.yaml`.
        """
        # Prepare inputs.
        metadata = {}
        folder = "20250728_zai_glm4-5"
        # Prepare outputs.
        expected = """
        {'agent': '',
         'attempts': '',
         'date': '2025-07-28',
         'model': '',
         'name': '20250728_zai_glm4-5',
         'org': '',
         'reported_score': nan,
         'submission': '20250728_zai_glm4-5'}
        """
        # Run test.
        fields = utils.parse_metadata(metadata, folder)
        # Check outputs.
        actual = pprint.pformat(fields)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_compute_leaderboard
# #############################################################################


class Test_compute_leaderboard(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compute_leaderboard()`.
    """

    def test1(self) -> None:
        """
        Test the ranking, a tie, and 2 agents with the same name.
        """
        # Prepare inputs.
        outcomes = pd.DataFrame(
            [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0]],
            index=["sub_a", "sub_b", "sub_c"],
            columns=["t1", "t2", "t3", "t4"],
        )
        submissions = pd.DataFrame(
            {
                "name": ["Model X", "Model Y", "Model X"],
                "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
                "agent": ["", "", ""],
                "attempts": ["1", "1", "2+"],
            },
            index=["sub_a", "sub_b", "sub_c"],
        )
        # Prepare outputs.
        expected = """
               rank                 label  score  n_resolved     name        date agent attempts
        sub_b     1               Model Y   0.75           3  Model Y  2025-02-01              1
        sub_a     2  Model X (2025-01-01)   0.50           2  Model X  2025-01-01              1
        sub_c     2  Model X (2025-03-01)   0.50           2  Model X  2025-03-01             2+
        """
        # Run test.
        leaderboard = utils.compute_leaderboard(outcomes, submissions)
        # Check outputs.
        actual = leaderboard.to_string()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_compute_score_cis
# #############################################################################


class Test_compute_score_cis(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compute_score_cis()`.
    """

    def test1(self) -> None:
        """
        Test the scores, the binomial standard errors, and that each interval
        contains the score.
        """
        # Prepare inputs.
        outcomes = pd.DataFrame(
            [[1, 1, 1, 0] * 25, [0, 1, 0, 0] * 25],
            index=["sub_a", "sub_b"],
        )
        # Prepare outputs.
        expected = """
               score  se_binomial  contains_score
        sub_a   0.75       0.0433            True
        sub_b   0.25       0.0433            True
        """
        # Run test.
        cis = utils.compute_score_cis(outcomes, n_boot=500, seed=1)
        # Check outputs.
        cis["contains_score"] = (cis["ci_low"] <= cis["score"]) & (
            cis["score"] <= cis["ci_high"]
        )
        actual = cis[["score", "se_binomial", "contains_score"]].round(4)
        actual = actual.to_string()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_compare_pair
# #############################################################################


class Test_compare_pair(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compare_pair()`.
    """

    def test1(self) -> None:
        """
        Test 2 agents that disagree on 2 tasks, both solved only by A.
        """
        # Prepare inputs.
        outcomes_a = pd.Series([1, 1, 1, 0])
        outcomes_b = pd.Series([0, 0, 1, 0])
        # Prepare outputs.
        # A wins both disagreements, so the exact McNemar p-value is
        # 2 * 0.5^2 = 0.5.
        expected = "{'gap': 0.5, 'only_a': 2, 'only_b': 0, 'p_value': 0.5, 'sd_diff': 0.5774}"
        # Run test.
        comparison = utils.compare_pair(outcomes_a, outcomes_b, n_boot=200)
        # Check outputs.
        keys = ["gap", "only_a", "only_b", "p_value", "sd_diff"]
        actual = {key: round(comparison[key], 4) for key in keys}
        actual = pprint.pformat(actual)
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test 2 agents with identical outcomes.
        """
        # Prepare inputs.
        outcomes_a = pd.Series([1, 0, 1, 0])
        outcomes_b = pd.Series([1, 0, 1, 0])
        # Prepare outputs.
        expected = """
        {'ci_high': 0.0,
         'ci_low': 0.0,
         'gap': 0.0,
         'only_a': 0,
         'only_b': 0,
         'p_value': 1.0,
         'sd_diff': 0.0}
        """
        # Run test.
        comparison = utils.compare_pair(outcomes_a, outcomes_b, n_boot=200)
        # Check outputs.
        actual = pprint.pformat(comparison)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_holm_adjust
# #############################################################################


class Test_holm_adjust(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.holm_adjust()`.
    """

    def test1(self) -> None:
        """
        Test 3 p-values where 2 adjusted values end up equal.
        """
        # Prepare inputs.
        p_values = [0.01, 0.04, 0.03]
        # Prepare outputs.
        expected = "[0.03, 0.06, 0.06]"
        # Run test.
        adjusted = utils.holm_adjust(p_values)
        # Check outputs.
        actual = str([round(float(value), 4) for value in adjusted])
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test that adjusted p-values never exceed 1.
        """
        # Prepare inputs.
        p_values = [0.6, 0.7]
        # Prepare outputs.
        expected = "[1.0, 1.0]"
        # Run test.
        adjusted = utils.holm_adjust(p_values)
        # Check outputs.
        actual = str([round(float(value), 4) for value in adjusted])
        self.assert_equal(actual, expected)


# #############################################################################
# Test_compute_min_tasks
# #############################################################################


class Test_compute_min_tasks(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compute_min_tasks()`.
    """

    def test1(self) -> None:
        """
        Test the default 80% power.
        """
        # Prepare inputs.
        sd_diff = 0.5
        gap = 0.05
        # Prepare outputs.
        # (1.96 + 0.8416)^2 * 0.5^2 / 0.05^2 = 784.89.
        expected = 784.89
        # Run test.
        n_tasks = utils.compute_min_tasks(sd_diff, gap)
        # Check outputs.
        self.assertAlmostEqual(n_tasks, expected, places=2)

    def test2(self) -> None:
        """
        Test 50% power, which matches the formula in the spec.
        """
        # Prepare inputs.
        sd_diff = 0.5
        gap = 0.05
        # Prepare outputs.
        # 1.96^2 * 0.5^2 / 0.05^2 = 384.15.
        expected = 384.15
        # Run test.
        n_tasks = utils.compute_min_tasks(sd_diff, gap, power=0.5)
        # Check outputs.
        self.assertAlmostEqual(n_tasks, expected, places=2)


# #############################################################################
# Test_compute_min_detectable_gap
# #############################################################################


class Test_compute_min_detectable_gap(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compute_min_detectable_gap()`.
    """

    def test1(self) -> None:
        """
        Test a benchmark of 400 tasks at the default 80% power.
        """
        # Prepare inputs.
        sd_diff = 0.5
        n_tasks = 400
        # Prepare outputs.
        # (1.96 + 0.8416) * 0.5 / sqrt(400) = 0.0700.
        expected = 0.0700
        # Run test.
        gap = utils.compute_min_detectable_gap(sd_diff, n_tasks)
        # Check outputs.
        self.assertAlmostEqual(gap, expected, places=4)

    def test2(self) -> None:
        """
        Test that it inverts `compute_min_tasks()`.
        """
        # Prepare inputs.
        sd_diff = 0.43
        gap = 0.03
        # Run test.
        n_tasks = utils.compute_min_tasks(sd_diff, gap)
        actual = utils.compute_min_detectable_gap(sd_diff, n_tasks)
        # Check outputs.
        self.assertAlmostEqual(actual, gap, places=10)


# #############################################################################
# Test_compute_expected_scores
# #############################################################################


class Test_compute_expected_scores(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.compute_expected_scores()`.
    """

    def test1(self) -> None:
        """
        Test an agent with ability 0, whose score is 0.5 by symmetry.
        """
        # Prepare inputs.
        abilities = [0.0]
        # Prepare outputs.
        expected = 0.5
        # Run test.
        scores = utils.compute_expected_scores(abilities, difficulty_sd=1.5)
        # Check outputs.
        self.assertAlmostEqual(scores["agent_0"], expected, places=6)

    def test2(self) -> None:
        """
        Test tasks that are all equally hard, so the score is the sigmoid of
        the ability.
        """
        # Prepare inputs.
        abilities = [1.0]
        # Prepare outputs.
        expected = 1 / (1 + np.exp(-1.0))
        # Run test.
        scores = utils.compute_expected_scores(abilities, difficulty_sd=0.0)
        # Check outputs.
        self.assertAlmostEqual(scores["agent_0"], expected, places=6)


# #############################################################################
# Test_simulate_outcomes
# #############################################################################


class Test_simulate_outcomes(hunitest.TestCase):
    """
    Test `skill_vs_luck_utils.simulate_outcomes()`.
    """

    def test1(self) -> None:
        """
        Test the shape and the labels of the simulated matrix.
        """
        # Prepare inputs.
        abilities = [-1.0, 0.0, 1.0]
        # Prepare outputs.
        expected = """
        (3, 4)
        ['agent_0', 'agent_1', 'agent_2']
        ['task_0', 'task_1', 'task_2', 'task_3']
        """
        # Run test.
        outcomes = utils.simulate_outcomes(abilities, 4, seed=3)
        # Check outputs.
        actual = "\n".join(
            [
                str(outcomes.shape),
                str(outcomes.index.tolist()),
                str(outcomes.columns.tolist()),
            ]
        )
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test that the same seed gives the same benchmark.
        """
        # Prepare inputs.
        abilities = [0.0, 0.5]
        # Run test.
        outcomes_1 = utils.simulate_outcomes(abilities, 50, seed=7)
        outcomes_2 = utils.simulate_outcomes(abilities, 50, seed=7)
        # Check outputs.
        self.assertTrue(outcomes_1.equals(outcomes_2))
