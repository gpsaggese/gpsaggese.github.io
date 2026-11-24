"""
Import as:

import test.test_skill_vs_luck_runs_utils as ttsvlrut
"""

import logging
import os
import pprint
import zipfile
from typing import Any, Dict, Tuple

import pandas as pd
import pytest

import helpers.hunit_test as hunitest

import skill_vs_luck_runs_utils as rutils

_LOG = logging.getLogger(__name__)


def _get_add_problem() -> dict:
    """
    Build a tiny task, `add(a, b)`, in the format of `load_problems()`.

    :return: task dict
    """
    problem = {
        "task_id": "HumanEval/999",
        "prompt": 'def add(a, b):\n    """\n    >>> add(1, 2)\n    3\n    """\n',
        "entry_point": "add",
        "base_input": [[1, 2], [3, 4]],
        "plus_input": [[10, 20], [-1, 1]],
        "atol": 0,
    }
    return problem


def _get_add_expected() -> dict:
    """
    Build the expected outputs of `_get_add_problem()`.

    :return: expected outputs, in the format of `get_expected_outputs()`
    """
    expected = {
        "base": [3, 7],
        "base_time": [0.001, 0.001],
        "plus": [30, 0],
        "plus_time": [0.001, 0.001],
    }
    return expected


# #############################################################################
# Test_get_cell_seed
# #############################################################################


class Test_get_cell_seed(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_cell_seed()`.
    """

    def test1(self) -> None:
        """
        Test that a cell always gets the same seed.
        """
        # Run test.
        seed_1 = rutils.get_cell_seed("HumanEval/2", 3)
        seed_2 = rutils.get_cell_seed("HumanEval/2", 3)
        # Check outputs.
        self.assertEqual(seed_1, seed_2)
        self.assertTrue(0 <= seed_1 < 2**31)

    def test2(self) -> None:
        """
        Test that different runs and tasks get different seeds.
        """
        # Run test.
        seeds = {
            rutils.get_cell_seed(task_id, run)
            for task_id in ("HumanEval/2", "HumanEval/3")
            for run in range(5)
        }
        # Check outputs.
        self.assertEqual(len(seeds), 10)


# #############################################################################
# Test_build_messages
# #############################################################################


class Test_build_messages(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.build_messages()`.
    """

    def test1(self) -> None:
        """
        Test the EvalPlus prompt, which ends with the start of the answer.
        """
        # Prepare inputs.
        prompt = "\ndef f(x):\n    pass\n"
        # Prepare outputs.
        expected = """
        [{'content': 'Please provide a self-contained Python script that solves the '
                     'following problem in a markdown code block:\\n'
                     '```\\n'
                     'def f(x):\\n'
                     '    pass\\n'
                     '```\\n',
          'role': 'user'},
         {'content': 'Below is a Python script with a self-contained function that '
                     'solves the problem and passes corresponding tests:\\n'
                     '```python\\n',
          'role': 'assistant'}]
        """
        # Run test.
        messages = rutils.build_messages(prompt)
        # Check outputs.
        actual = pprint.pformat(messages)
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_build_repair_messages
# #############################################################################


class Test_build_repair_messages(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.build_repair_messages()`.
    """

    def test1(self) -> None:
        """
        Test that the failed code and the feedback are added, and that the
        conversation ends with the start of a new answer.
        """
        # Prepare inputs.
        messages = rutils.build_messages("def f(x):\n    pass")
        # Run test.
        new_messages = rutils.build_repair_messages(
            messages, "def f(x):\n    return 0", "assert f(1) == 1\nfailed"
        )
        # Check outputs.
        roles = [message["role"] for message in new_messages]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
        self.assertTrue(new_messages[1]["content"].endswith("    return 0\n```"))
        self.assertIn("assert f(1) == 1", new_messages[2]["content"])
        self.assertTrue(new_messages[3]["content"].endswith("```python\n"))


# #############################################################################
# Test_get_visible_tests
# #############################################################################


class Test_get_visible_tests(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_visible_tests()`.
    """

    def test1(self) -> None:
        """
        Test an MBPP+ prompt, which shows 1 `assert` line.
        """
        # Prepare inputs.
        problem = {
            "prompt": '"""\nWrite a function.\nassert f(1) == 2\n"""\n',
            "entry_point": "f",
        }
        # Prepare outputs.
        expected = "['assert f(1) == 2']"
        # Run test.
        tests = rutils.get_visible_tests(problem, "mbpp")
        # Check outputs.
        self.assert_equal(str(tests), expected)

    def test2(self) -> None:
        """
        Test a doctest example and an arrow example, with a float tolerance.
        """
        # Prepare inputs.
        problem = {
            "prompt": (
                "def f(x):\n"
                '    """\n'
                "    >>> f(3)\n"
                "    0.333\n"
                "    f(1) ➞ [1, 2]\n"
                '    """\n'
            ),
            "entry_point": "f",
        }
        # Prepare outputs.
        expected = (
            "['assert abs(f(3) - 0.333) <= 0.0005', 'assert f(1) == [1, 2]']"
        )
        # Run test.
        tests = rutils.get_visible_tests(problem, "humaneval")
        # Check outputs.
        self.assert_equal(str(tests), expected)

    def test3(self) -> None:
        """
        Test that the closing quotes of the docstring are not read as part of
        the result.
        """
        # Prepare inputs.
        problem = {
            "prompt": 'def f(x):\n    """\n    >>> f(3.5)\n    0.5\n    """\n',
            "entry_point": "f",
        }
        # Prepare outputs.
        expected = "['assert abs(f(3.5) - 0.5) <= 0.05']"
        # Run test.
        tests = rutils.get_visible_tests(problem, "humaneval")
        # Check outputs.
        self.assert_equal(str(tests), expected)

    def test4(self) -> None:
        """
        Test that examples that are not a call with a literal are skipped.
        """
        # Prepare inputs.
        problem = {
            "prompt": (
                "def f(x):\n"
                '    """\n'
                "    >>> g(3)\n"
                "    1\n"
                "    f(2) == a list of 2 items\n"
                '    """\n'
            ),
            "entry_point": "f",
        }
        # Run test.
        tests = rutils.get_visible_tests(problem, "humaneval")
        # Check outputs.
        self.assertEqual(tests, [])


# #############################################################################
# Test_get_task_label
# #############################################################################


class Test_get_task_label(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_task_label()`.
    """

    def test1(self) -> None:
        """
        Test the folder names of both benchmarks.
        """
        # Run test.
        labels = [
            rutils.get_task_label(name) for name in ("HumanEval_81", "Mbpp_468")
        ]
        # Check outputs.
        self.assertEqual(labels, ["HumanEval/81", "Mbpp/468"])

    def test2(self) -> None:
        """
        Test that an unknown folder name raises an error.
        """
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rutils.get_task_label("HumanEval-81")


# #############################################################################
# Test_sort_task_ids
# #############################################################################


class Test_sort_task_ids(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.sort_task_ids()`.
    """

    def test1(self) -> None:
        """
        Test that the ids are sorted by number, not as text.
        """
        # Prepare inputs.
        task_ids = ["HumanEval/10", "HumanEval/2", "HumanEval/1"]
        # Run test.
        sorted_ids = rutils.sort_task_ids(task_ids)
        # Check outputs.
        self.assertEqual(
            sorted_ids, ["HumanEval/1", "HumanEval/2", "HumanEval/10"]
        )


# #############################################################################
# Test_get_runs_file
# #############################################################################


class Test_get_runs_file(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_runs_file()`.
    """

    def test1(self) -> None:
        """
        Test the file name of a model and a temperature.
        """
        # Prepare outputs.
        expected = "data/raw/coding_runs/mbpp/deepseek-coder_1.3b_t0.8.jsonl"
        # Run test.
        path = rutils.get_runs_file(
            "data/raw/coding_runs", "mbpp", "deepseek-coder:1.3b", 0.8
        )
        # Check outputs.
        self.assert_equal(path, expected)


# #############################################################################
# Test_get_scaffold_rows
# #############################################################################


class Test_get_scaffold_rows(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_scaffold_rows()`.
    """

    def test1(self) -> None:
        """
        Test a cell with a repair: one-shot uses the first attempt, repair the
        last one, and the costs add up over the attempts.
        """
        # Prepare inputs.
        attempt = {"visible_pass": False, "tokens_in": 100, "tokens_out": 20}
        record = {
            "task_id": "HumanEval/0",
            "run": 1,
            "n_visible_tests": 2,
            "attempts": [
                dict(attempt, code="v1", seconds=1.0),
                dict(attempt, code="v2", seconds=2.0, visible_pass=True),
            ],
        }
        # Prepare outputs.
        expected = """
        [('one_shot', 'v1', 1, 0, 100, 1.0), ('repair', 'v2', 2, 1, 200, 3.0)]
        """
        # Run test.
        rows = rutils.get_scaffold_rows(record)
        # Check outputs.
        actual = str(
            [
                (
                    row["scaffold"],
                    row["code"],
                    row["n_attempts"],
                    row["visible_pass"],
                    row["tokens_in"],
                    row["gen_seconds"],
                )
                for row in rows
            ]
        )
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_clean_sample
# #############################################################################


class Test_clean_sample(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.clean_sample()`.
    """

    def test1(self) -> None:
        """
        Test that a sample that is already code is scored as is.
        """
        # Prepare inputs.
        code = "def f(x):\n    return x\n\n\nprint(f(1))\n"
        # Run test.
        cleaned = rutils.clean_sample(code, "f")
        # Check outputs.
        self.assertEqual(cleaned, code)

    def test2(self) -> None:
        """
        Test that the code is extracted from a raw chat answer.
        """
        # Prepare inputs.
        code = "Here it is:\n\n```python\ndef f(x):\n    return x\n```\nDone.\n"
        # Prepare outputs.
        expected = "def f(x):\n    return x"
        # Run test.
        cleaned = rutils.clean_sample(code, "f")
        # Check outputs.
        self.assert_equal(cleaned, expected)


# #############################################################################
# Test_iter_zip_samples
# #############################################################################


class Test_iter_zip_samples(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.iter_zip_samples()`.
    """

    def test1(self) -> None:
        """
        Test that the samples are read in order and `max_samples` is applied.
        """
        # Prepare inputs.
        zip_path = os.path.join(self.get_scratch_space(), "samples.zip")
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for name in (
                "HumanEval_1/0.py",
                "HumanEval_1/1.py",
                "HumanEval_0/0.py",
            ):
                zip_file.writestr(f"model_temp_0.8/{name}", f"# {name}\n")
        # Prepare outputs.
        expected = (
            "[('HumanEval/0', 0, '# HumanEval_0/0.py\\n'), "
            "('HumanEval/1', 0, '# HumanEval_1/0.py\\n')]"
        )
        # Run test.
        samples = list(rutils.iter_zip_samples(zip_path, max_samples=1))
        # Check outputs.
        self.assert_equal(str(samples), expected)


# #############################################################################
# Test_run_visible_tests
# #############################################################################


class Test_run_visible_tests(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.run_visible_tests()`.
    """

    def test1(self) -> None:
        """
        Test a solution that passes its examples.
        """
        # Run test.
        passed, feedback = rutils.run_visible_tests(
            "def f(x):\n    return x + 1", ["assert f(1) == 2"]
        )
        # Check outputs.
        self.assertTrue(passed)
        self.assertEqual(feedback, "")

    def test2(self) -> None:
        """
        Test that a wrong solution gets feedback with the value it returned.
        """
        # Run test.
        passed, feedback = rutils.run_visible_tests(
            "def f(x):\n    return x", ["assert f(1) == 2"]
        )
        # Check outputs.
        self.assertFalse(passed)
        self.assert_equal(
            feedback, "assert f(1) == 2\nfailed: the left side is 1"
        )

    def test3(self) -> None:
        """
        Test that an answer without code fails without running anything.
        """
        # Run test.
        passed, feedback = rutils.run_visible_tests("  ", ["assert f(1) == 2"])
        # Check outputs.
        self.assertFalse(passed)
        self.assert_equal(feedback, "The answer contains no code.")

    def test4(self) -> None:
        """
        Test an MBPP+ example that uses `math`, with a solution that does not
        import it.
        """
        # Prepare inputs.
        code = "def area(r):\n    return 3.14159265 * r * r"
        tests = ["assert math.isclose(area(1), 3.14159265, rel_tol=0.001)"]
        # Run test.
        passed, feedback = rutils.run_visible_tests(code, tests)
        # Check outputs.
        self.assertTrue(passed)
        self.assertEqual(feedback, "")

    def test5(self) -> None:
        """
        Test that a solution that prints a lot is still checked.
        """
        # Prepare inputs.
        code = (
            "def f(x):\n"
            "    for _ in range(200000):\n"
            "        print('noise')\n"
            "    return x + 1"
        )
        # Run test.
        passed, feedback = rutils.run_visible_tests(code, ["assert f(1) == 2"])
        # Check outputs.
        self.assertTrue(passed)
        self.assertEqual(feedback, "")

    def test6(self) -> None:
        """
        Test that a solution that never returns is stopped.
        """
        # Prepare inputs.
        code = "def f(x):\n    while True:\n        pass"
        # Run test.
        passed, feedback = rutils.run_visible_tests(
            code, ["assert f(1) == 2"], timeout=2
        )
        # Check outputs.
        self.assertFalse(passed)
        self.assert_equal(feedback, "Running the examples took more than 2 s.")


# #############################################################################
# Test_score_solution
# #############################################################################


class Test_score_solution(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.score_solution()`.
    """

    def test1(self) -> None:
        """
        Test a correct solution.
        """
        # Prepare outputs.
        expected = (
            "{'base_pass': 1, 'failed_input': -1, 'failed_split': '', "
            "'plus_pass': 1}"
        )
        # Run test.
        score = rutils.score_solution(
            "humaneval",
            _get_add_problem(),
            "def add(a, b):\n    return a + b",
            _get_add_expected(),
        )
        # Check outputs.
        self.assert_equal(pprint.pformat(score), expected)

    def test2(self) -> None:
        """
        Test a solution that passes the base tests but fails a plus input.
        """
        # Prepare outputs.
        expected = (
            "{'base_pass': 1, 'failed_input': 1, 'failed_split': 'plus', "
            "'plus_pass': 0}"
        )
        # Run test.
        score = rutils.score_solution(
            "humaneval",
            _get_add_problem(),
            "def add(a, b):\n    return a + b if a >= 0 else 99",
            _get_add_expected(),
        )
        # Check outputs.
        self.assert_equal(pprint.pformat(score), expected)


# #############################################################################
# Test_diagnose_failure
# #############################################################################


class Test_diagnose_failure(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.diagnose_failure()`.
    """

    def _diagnose(self, solution: str) -> str:
        """
        Score a solution of `add()` and label its failure.

        :param solution: code of the solution
        :return: failure label
        """
        problem = _get_add_problem()
        expected = _get_add_expected()
        score = rutils.score_solution("humaneval", problem, solution, expected)
        label = rutils.diagnose_failure(problem, solution, expected, score)
        return label

    def test1(self) -> None:
        """
        Test the labels of failures that do not hang.
        """
        # Prepare inputs.
        solutions = {
            "pass": "def add(a, b):\n    return a + b",
            "wrong": "def add(a, b):\n    return a - b",
            "raises": "def add(a, b):\n    return a / 0",
            "syntax": "def add(a, b)\n    return a + b",
            "missing": "def plus(a, b):\n    return a + b",
            "empty": "",
        }
        # Prepare outputs.
        expected = """
        {'empty': 'no_code',
         'missing': 'missing_function',
         'pass': '',
         'raises': 'runtime_error',
         'syntax': 'syntax_error',
         'wrong': 'wrong_answer'}
        """
        # Run test.
        labels = {
            name: self._diagnose(solution)
            for name, solution in solutions.items()
        }
        # Check outputs.
        self.assert_equal(pprint.pformat(labels), expected, dedent=True)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that a solution that never returns is labeled as a timeout.
        """
        # Run test.
        label = self._diagnose("def add(a, b):\n    while True:\n        pass")
        # Check outputs.
        self.assert_equal(label, "timeout")


# #############################################################################
# Test_get_outcome_matrix
# #############################################################################


class Test_get_outcome_matrix(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_outcome_matrix()`.
    """

    def test1(self) -> None:
        """
        Test that the scores of 1 agent become a tasks x runs matrix.
        """
        # Prepare inputs.
        scores = pd.DataFrame(
            {
                "source": "local",
                "benchmark": "humaneval",
                "model": ["m", "m", "m", "m", "other"],
                "scaffold": "one_shot",
                "temperature": 0.8,
                "task_id": [
                    "HumanEval/10",
                    "HumanEval/10",
                    "HumanEval/2",
                    "HumanEval/2",
                    "HumanEval/2",
                ],
                "run": [0, 1, 0, 1, 0],
                "plus_pass": [1, 0, 0, 0, 1],
            }
        )
        # Prepare outputs.
        expected = """
        run           0  1
        task_id
        HumanEval/2   0  0
        HumanEval/10  1  0
        """
        # Run test.
        matrix = rutils.get_outcome_matrix(
            scores,
            source="local",
            benchmark="humaneval",
            model="m",
            scaffold="one_shot",
            temperature=0.8,
        )
        # Check outputs.
        self.assert_equal(matrix.to_string(), expected, dedent=True)


# #############################################################################
# Test_compute_variance_components
# #############################################################################


class Test_compute_variance_components(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_variance_components()`.
    """

    def test1(self) -> None:
        """
        Test 4 tasks and 2 runs, computed by hand.

        Per-task rates are 1, 0, 0.5, 0.5, so the run variance is
        (0 + 0 + 0.5 + 0.5) / 4 = 0.25 and the task variance is
        1/6 - 0.25 / 2 = 0.0417.
        """
        # Prepare inputs.
        outcomes = pd.DataFrame([[1, 1], [0, 0], [1, 0], [1, 0]])
        # Prepare outputs.
        expected = """
        score          0.5000
        sigma2_task    0.0417
        sigma2_run     0.2500
        run_share      0.8571
        se_1_run       0.2700
        se_all_runs    0.2041
        """
        # Run test.
        components = rutils.compute_variance_components(outcomes)
        # Check outputs.
        self.assert_equal(components.round(4).to_string(), expected, dedent=True)

    def test2(self) -> None:
        """
        Test that runs that always agree have no run variance.
        """
        # Prepare inputs.
        outcomes = pd.DataFrame([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
        # Run test.
        components = rutils.compute_variance_components(outcomes)
        # Check outputs.
        self.assertEqual(components["sigma2_run"], 0.0)
        self.assertEqual(components["run_share"], 0.0)


# #############################################################################
# Test_compare_agents_over_runs
# #############################################################################


class Test_compare_agents_over_runs(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compare_agents_over_runs()`.
    """

    def test1(self) -> None:
        """
        Test 2 agents with the same outcomes.
        """
        # Prepare inputs.
        outcomes = pd.DataFrame([[1, 0], [1, 1], [0, 0]])
        # Prepare outputs.
        expected = (
            "{'ci_high': 0.0, 'ci_low': 0.0, 'gap': 0.0, 'gap_run_0': 0.0, "
            "'p_value': 1.0}"
        )
        # Run test.
        comparison = rutils.compare_agents_over_runs(
            outcomes, outcomes, n_boot=100
        )
        # Check outputs.
        self.assert_equal(pprint.pformat(comparison), expected)

    def test2(self) -> None:
        """
        Test an agent that solves every task against one that solves none.
        """
        # Prepare inputs.
        best = pd.DataFrame([[1, 1]] * 10)
        worst = pd.DataFrame([[0, 0]] * 10)
        # Run test.
        comparison = rutils.compare_agents_over_runs(best, worst, n_boot=100)
        # Check outputs.
        self.assertEqual(comparison["gap"], 1.0)
        self.assertEqual(comparison["ci_low"], 1.0)


# #############################################################################
# Test_compute_ranking_stability
# #############################################################################


class Test_compute_ranking_stability(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_ranking_stability()`.
    """

    def test1(self) -> None:
        """
        Test an agent that beats the other in every run: the order never
        changes.
        """
        # Prepare inputs.
        matrices = {
            "a": pd.DataFrame([[1, 1], [1, 0]]),
            "b": pd.DataFrame([[0, 0], [0, 0]]),
        }
        # Run test.
        summary, pairs = rutils.compute_ranking_stability(matrices, n_draws=100)
        # Check outputs.
        self.assertEqual(summary["p_same_order"], 1.0)
        self.assertEqual(pairs.loc[0, "upper"], "a")
        self.assertEqual(pairs.loc[0, "p_flip"], 0.0)

    def test3(self) -> None:
        """
        Test that 2 agents with equal means do not count against the order.
        """
        # Prepare inputs.
        matrices = {
            "a": pd.DataFrame([[1, 1], [1, 1]]),
            "b": pd.DataFrame([[1, 0], [1, 0]]),
            "c": pd.DataFrame([[0, 1], [0, 1]]),
        }
        # Run test.
        summary, _ = rutils.compute_ranking_stability(matrices, n_draws=100)
        # Check outputs: no draw puts `b` or `c` above `a`, and the order of `b`
        # and `c`, whose means are equal, does not count.
        self.assertEqual(summary["p_same_order"], 1.0)

    def test2(self) -> None:
        """
        Test 2 agents whose runs alternate: half the draws flip the order.
        """
        # Prepare inputs.
        matrices = {
            "a": pd.DataFrame([[1, 0], [1, 0]]),
            "b": pd.DataFrame([[0, 1], [0, 1]]),
        }
        # Run test.
        _, pairs = rutils.compute_ranking_stability(matrices, n_draws=4000)
        # Check outputs.
        self.assertAlmostEqual(pairs.loc[0, "p_flip"], 0.5, delta=0.05)


# #############################################################################
# Test_compute_min_tasks_with_runs
# #############################################################################


class Test_compute_min_tasks_with_runs(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_min_tasks_with_runs()`.
    """

    def test1(self) -> None:
        """
        Test 1 and 2 runs per task, computed by hand.

        (1.96 + 0.8416)^2 = 7.849, so 1 run needs 7.849 * 0.3 / 0.05^2 =
        941.9 tasks and 2 runs need 7.849 * 0.2 / 0.05^2 = 627.9.
        """
        # Run test.
        n_1 = rutils.compute_min_tasks_with_runs(0.1, 0.2, 0.05, 1)
        n_2 = rutils.compute_min_tasks_with_runs(0.1, 0.2, 0.05, 2)
        # Check outputs.
        self.assertAlmostEqual(n_1, 941.9, places=1)
        self.assertAlmostEqual(n_2, 627.9, places=1)


# #############################################################################
# Test_compute_flaky_share
# #############################################################################


class Test_compute_flaky_share(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_flaky_share()`.
    """

    def test1(self) -> None:
        """
        Test 4 solutions scored twice, 1 of them flaky.
        """
        # Prepare inputs.
        scores = pd.DataFrame({"n_checks": 2, "plus_pass_count": [0, 1, 2, 2]})
        # Run test.
        share = rutils.compute_flaky_share(scores)
        # Check outputs.
        self.assertEqual(share, 0.25)


def _get_two_agent_scores() -> pd.DataFrame:
    """
    Build the scores of 2 agents on 3 tasks and 2 runs.

    Agent `a:1` solves every cell; agent `b:1` solves only task 0.

    :return: scores in the format of `load_scores()`
    """
    rows = []
    for model, solved_tasks in (("a:1", {0, 1, 2}), ("b:1", {0})):
        for task in range(3):
            for run in range(2):
                rows.append(
                    {
                        "source": "local",
                        "benchmark": "humaneval",
                        "model": model,
                        "temperature": 0.8,
                        "scaffold": "one_shot",
                        "task_id": f"HumanEval/{task}",
                        "run": run,
                        "base_pass": int(task in solved_tasks),
                        "plus_pass": int(task in solved_tasks),
                    }
                )
    scores = pd.DataFrame(rows)
    return scores


def _get_score_row(
    model: str,
    temperature: float,
    scaffold: str,
    task: int,
    run: int,
    passed: int,
    *,
    source: str = "local",
    failure: str = "",
) -> Dict[str, Any]:
    """
    Build 1 row of scores, in the format of `load_scores()`.

    :param model: model of the agent
    :param temperature: sampling temperature
    :param scaffold: `one_shot` or `repair`
    :param task: number of the HumanEval task
    :param run: run of the cell
    :param passed: `plus_pass` of the solution, also used as `base_pass`
    :param source: `local` or `evalplus`
    :param failure: failure label of the solution
    :return: 1 row of scores
    """
    row = {
        "source": source,
        "benchmark": "humaneval",
        "model": model,
        "temperature": temperature,
        "scaffold": scaffold,
        "task_id": f"HumanEval/{task}",
        "run": run,
        "base_pass": passed,
        "plus_pass": passed,
        "failure": failure,
    }
    return row


# #############################################################################
# Test_get_agent_label
# #############################################################################


class Test_get_agent_label(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_agent_label()`.
    """

    def test1(self) -> None:
        """
        Test that the tag of the model is dropped.
        """
        # Run test.
        label = rutils.get_agent_label("codeqwen:7b-chat-v1.5-q4_K_M", "repair")
        # Check outputs.
        self.assert_equal(label, "codeqwen / repair")


# #############################################################################
# Test_compute_score_table
# #############################################################################


class Test_compute_score_table(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_score_table()`.
    """

    def test1(self) -> None:
        """
        Test the pass rates and the counts of 2 agents.
        """
        # Prepare outputs.
        expected = """
                                                     n_tasks  n_runs  base_pass  plus_pass
        source benchmark model temperature scaffold
        local  humaneval a:1   0.8         one_shot        3       2   1.000000   1.000000
                         b:1   0.8         one_shot        3       2   0.333333   0.333333
        """
        # Run test.
        table = rutils.compute_score_table(_get_two_agent_scores())
        # Check outputs.
        actual = "\n".join(
            line.rstrip() for line in table.to_string().splitlines()
        )
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_compute_validation_table
# #############################################################################


class Test_compute_validation_table(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_validation_table()`.
    """

    def test1(self) -> None:
        """
        Test that only the greedy one-shot rows are compared.
        """
        # Prepare inputs: 3 of 4 greedy one-shot answers pass, and the other
        # rows must be ignored.
        rows = [
            _get_score_row("opencoder:8b", 0.0, "one_shot", task, 0, passed)
            for task, passed in enumerate([1, 1, 1, 0])
        ]
        rows.append(_get_score_row("opencoder:8b", 0.0, "repair", 3, 0, 1))
        rows.append(_get_score_row("opencoder:8b", 0.8, "one_shot", 3, 0, 1))
        scores = pd.DataFrame(rows)
        # Run test.
        table = rutils.compute_validation_table(scores)
        # Check outputs.
        row = table.loc[("opencoder:8b", "humaneval")]
        self.assertEqual(len(table), 1)
        self.assertEqual(row["n_tasks"], 4)
        self.assertAlmostEqual(row["plus_pass"], 0.75)
        self.assertAlmostEqual(row["published_plus"], 0.774)
        self.assertAlmostEqual(row["plus_diff"], 0.75 - 0.774)

    def test2(self) -> None:
        """
        Test that a model without a published score is rejected.
        """
        # Prepare inputs.
        scores = pd.DataFrame(
            [_get_score_row("unknown:1b", 0.0, "one_shot", 0, 0, 1)]
        )
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rutils.compute_validation_table(scores)


# #############################################################################
# Test_compute_failure_modes
# #############################################################################


class Test_compute_failure_modes(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_failure_modes()`.
    """

    def test1(self) -> None:
        """
        Test the split between a task that flips and a task that always
        fails.
        """
        # Prepare inputs: task 0 passes once and fails once, task 1 fails in
        # both runs, and the greedy row must be ignored.
        rows = [
            _get_score_row("a:1", 0.8, "one_shot", 0, 0, 1),
            _get_score_row(
                "a:1", 0.8, "one_shot", 0, 1, 0, failure="wrong_answer"
            ),
            _get_score_row("a:1", 0.8, "one_shot", 1, 0, 0, failure="timeout"),
            _get_score_row(
                "a:1", 0.8, "one_shot", 1, 1, 0, failure="runtime_error"
            ),
            _get_score_row(
                "a:1", 0.0, "one_shot", 0, 0, 0, failure="syntax_error"
            ),
        ]
        scores = pd.DataFrame(rows)
        # Run test.
        table = rutils.compute_failure_modes(
            scores, source="local", benchmark="humaneval", temperature=0.8
        )
        # Check outputs.
        self.assertEqual(
            list(table.index), ["runtime_error", "timeout", "wrong_answer"]
        )
        self.assertEqual(table.loc["wrong_answer", ("one_shot", "flips")], 1)
        self.assertEqual(table.loc["timeout", ("one_shot", "always_fails")], 1)
        self.assertEqual(int(table.to_numpy().sum()), 3)


# #############################################################################
# Test_compute_variance_table
# #############################################################################


class Test_compute_variance_table(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_variance_table()`.
    """

    def test1(self) -> None:
        """
        Test 2 agents whose runs always agree: no run variance.
        """
        # Run test.
        table = rutils.compute_variance_table(
            _get_two_agent_scores(),
            source="local",
            benchmark="humaneval",
            temperature=0.8,
        )
        # Check outputs.
        self.assertEqual(list(table.index), ["a / one_shot", "b / one_shot"])
        self.assertEqual(list(table["sigma2_run"]), [0.0, 0.0])


# #############################################################################
# Test_compare_agent_pairs
# #############################################################################


class Test_compare_agent_pairs(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compare_agent_pairs()`.
    """

    def test1(self) -> None:
        """
        Test the gap of 1 pair and the Holm column.
        """
        # Prepare inputs.
        pairs = [(("a:1", "one_shot"), ("b:1", "one_shot"))]
        # Run test.
        table = rutils.compare_agent_pairs(
            _get_two_agent_scores(),
            pairs,
            source="local",
            benchmarks=["humaneval"],
            temperature=0.8,
        )
        # Check outputs.
        self.assertEqual(table.loc[0, "agent_a"], "a / one_shot")
        self.assertAlmostEqual(table.loc[0, "gap"], 2 / 3)
        self.assertEqual(table.loc[0, "p_holm"], table.loc[0, "p_value"])


# #############################################################################
# Test_compute_temperature_table
# #############################################################################


class Test_compute_temperature_table(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_temperature_table()`.
    """

    def test1(self) -> None:
        """
        Test the run variance of 1 public model at 2 temperatures.
        """
        # Prepare inputs: the outcomes of 2 tasks over 4 samples at each
        # temperature of model `m`; model `n` has only 1 temperature.
        outcomes = {
            0.2: {0: [1, 1, 1, 0], 1: [0, 0, 0, 0]},
            0.8: {0: [1, 1, 0, 0], 1: [1, 0, 0, 0]},
        }
        rows = [
            _get_score_row(
                "m",
                temperature,
                "one_shot",
                task,
                run,
                passed,
                source="evalplus",
            )
            for temperature, tasks in outcomes.items()
            for task, runs in tasks.items()
            for run, passed in enumerate(runs)
        ]
        rows.append(
            _get_score_row("n", 0.8, "one_shot", 0, 0, 1, source="evalplus")
        )
        scores = pd.DataFrame(rows)
        # Run test.
        table = rutils.compute_temperature_table(scores, benchmark="humaneval")
        # Check outputs: the run variances are (1/4 + 0) / 2 at 0.2 and
        # (1/3 + 1/4) / 2 at 0.8.
        self.assertEqual(list(table.index), ["m"])
        self.assertAlmostEqual(table.loc["m", "score_low"], 3 / 8)
        self.assertAlmostEqual(table.loc["m", "sigma2_run_low"], 1 / 8)
        self.assertAlmostEqual(table.loc["m", "sigma2_run_high"], 7 / 24)
        self.assertAlmostEqual(table.loc["m", "run_ratio"], 7 / 3)


def _get_comparison_row(
    benchmark: str,
    agent_a: Tuple[str, str],
    agent_b: Tuple[str, str],
    gap: float,
    p_holm: float,
) -> Dict[str, Any]:
    """
    Build 1 row of the output of `compare_agent_pairs()`.

    :param benchmark: `humaneval` or `mbpp`
    :param agent_a: `(model, scaffold)` of agent A
    :param agent_b: `(model, scaffold)` of agent B
    :param gap: score of A minus B
    :param p_holm: Holm-adjusted p-value
    :return: row with the columns that `check_predictions()` reads
    """
    row = {
        "benchmark": benchmark,
        "agent_a": rutils.get_agent_label(*agent_a),
        "agent_b": rutils.get_agent_label(*agent_b),
        "gap": gap,
        "p_holm": p_holm,
        "distinguishable": p_holm < 0.05,
    }
    return row


def _get_prediction_inputs() -> Dict[str, Any]:
    """
    Build the inputs of `check_predictions()`, with every prediction holding.

    :return: keyword arguments of `check_predictions()`
    """
    weak, model_a, model_b = "w:1b", "a:7b", "b:8b"
    models = [weak, model_a, model_b]
    benchmarks = ["humaneval", "mbpp"]
    scaffolds = ["one_shot", "repair"]
    # Greedy scores 2 points above the published ones.
    validation = pd.DataFrame(
        {"plus_diff": 0.02},
        index=pd.MultiIndex.from_product(
            [models, benchmarks], names=["model", "benchmark"]
        ),
    )
    # Repair gains 2 points on HumanEval+ and 4 points on MBPP+.
    rows = [
        {
            "source": "local",
            "benchmark": benchmark,
            "model": model,
            "temperature": 0.8,
            "scaffold": scaffold,
            "plus_pass": 0.5 + (gain if scaffold == "repair" else 0.0),
        }
        for benchmark, gain in (("humaneval", 0.02), ("mbpp", 0.04))
        for model in models
        for scaffold in scaffolds
    ]
    keys = ["source", "benchmark", "model", "temperature", "scaffold"]
    score_table = pd.DataFrame(rows).set_index(keys).sort_index()
    # Runs are 25% of the variance of every agent.
    labels = [rutils.get_agent_label(m, s) for m in models for s in scaffolds]
    variance_tables = {
        benchmark: pd.DataFrame({"run_share": 0.25}, index=labels)
        for benchmark in benchmarks
    }
    # The pre-registered family: the close pair is never distinguishable,
    # and the weaker model is behind both models of the pair.
    rows = []
    for benchmark in benchmarks:
        for scaffold in scaffolds:
            rows.append(
                _get_comparison_row(
                    benchmark,
                    (model_a, scaffold),
                    (model_b, scaffold),
                    0.01,
                    0.5,
                )
            )
            for model in (model_a, model_b):
                rows.append(
                    _get_comparison_row(
                        benchmark,
                        (model, scaffold),
                        (weak, scaffold),
                        0.2,
                        0.001,
                    )
                )
        for model in models:
            rows.append(
                _get_comparison_row(
                    benchmark, (model, "repair"), (model, "one_shot"), 0.03, 0.04
                )
            )
    comparisons = pd.DataFrame(rows)
    temperature_table = pd.DataFrame(
        {"run_ratio": [2.5, 3.0]}, index=["m1", "m2"]
    )
    inputs = {
        "validation": validation,
        "score_table": score_table,
        "variance_tables": variance_tables,
        "comparisons": comparisons,
        "flaky_share": 0.0,
        "temperature_table": temperature_table,
        "weak_model": weak,
        "pair_models": (model_a, model_b),
    }
    return inputs


# #############################################################################
# Test_check_predictions
# #############################################################################


class Test_check_predictions(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.check_predictions()`.
    """

    def test1(self) -> None:
        """
        Test results where every prediction holds.
        """
        # Prepare inputs: list 1 comparison of the weaker model the other way
        # around, which reverses its gap.
        inputs = _get_prediction_inputs()
        comparisons = inputs["comparisons"]
        comparisons.loc[1, ["agent_a", "agent_b"]] = comparisons.loc[
            1, ["agent_b", "agent_a"]
        ].to_numpy()
        comparisons.loc[1, "gap"] = -0.2
        # Run test.
        table = rutils.check_predictions(**inputs)
        # Check outputs.
        self.assertEqual(list(table.index), [f"P{i}" for i in range(1, 8)])
        self.assertEqual(table["holds"].tolist(), [True] * 7)
        self.assertEqual(
            table.loc["P1", "observed"],
            "0 of 4 comparisons distinguishable, smallest Holm p 0.5",
        )
        self.assertEqual(
            table.loc["P5", "observed"],
            "gains from +2.0 to +4.0 points, larger on MBPP+ for 3 of 3 models",
        )

    def test2(self) -> None:
        """
        Test results where every prediction fails.
        """
        # Prepare inputs: break each prediction once.
        inputs = _get_prediction_inputs()
        comparisons = inputs["comparisons"]
        # P1: a comparison of the close pair is distinguishable.
        comparisons.loc[0, "p_holm"] = 0.01
        comparisons.loc[0, "distinguishable"] = True
        # P2: the weaker model is ahead in a comparison, which still counts
        # as distinguishable.
        comparisons.loc[1, "gap"] = -0.2
        # P3: half of the variance of an agent comes from the runs.
        inputs["variance_tables"]["mbpp"].iloc[0, 0] = 0.5
        # P4: a greedy score is 6 points below the published one.
        inputs["validation"].iloc[0, 0] = -0.06
        # P5: a model gains less on MBPP+ than on HumanEval+.
        key = ("local", "mbpp", "a:7b", 0.8, "repair")
        inputs["score_table"].loc[key, "plus_pass"] = 0.51
        # P6 and P7.
        inputs["flaky_share"] = 0.02
        inputs["temperature_table"].iloc[0, 0] = 1.5
        # Run test.
        table = rutils.check_predictions(**inputs)
        # Check outputs.
        self.assertEqual(table["holds"].tolist(), [False] * 7)
        self.assertEqual(
            table.loc["P4", "observed"],
            "5 of 6 within 5 points, largest gap -6.0 points (w, humaneval)",
        )
