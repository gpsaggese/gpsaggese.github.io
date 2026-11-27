"""
Import as:

import test.test_skill_vs_luck_runs_utils as ttsvlrut
"""

import logging
import os
import pprint
import unittest.mock as umock
import zipfile
from typing import Any, Dict, Tuple

import numpy as np
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
# Test_call_ollama
# #############################################################################


class Test_call_ollama(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.call_ollama()` with a fake Ollama server.
    """

    def test1(self) -> None:
        """
        Test that an answer Ollama stops for repeating a token comes back
        empty, without retrying.
        """
        # Prepare inputs: the error that Ollama returns.
        response = umock.Mock(
            status_code=500,
            text='{"error":"prediction aborted, token repeat limit reached"}',
        )
        # Run test.
        with umock.patch.object(
            rutils.requests, "post", return_value=response
        ) as post:
            reply = rutils.call_ollama(
                "m", [], temperature=0, seed=1, stop=[], url="http://fake"
            )
        # Check outputs.
        self.assertEqual(post.call_count, 1)
        self.assertEqual(reply["text"], "")
        self.assertEqual(reply["done_reason"], "repeat_limit")

    def test2(self) -> None:
        """
        Test that a normal answer comes back with its counts.
        """
        # Prepare inputs.
        response = umock.Mock(status_code=200, text="")
        response.json.return_value = {
            "message": {"content": "\treturn 1"},
            "prompt_eval_count": 5,
            "eval_count": 3,
            "total_duration": 2e9,
            "done_reason": "stop",
        }
        # Run test.
        with umock.patch.object(rutils.requests, "post", return_value=response):
            reply = rutils.call_ollama(
                "m", [], temperature=0, seed=1, stop=[], url="http://fake"
            )
        # Check outputs: tabs become 4 spaces, as in EvalPlus.
        expected = {
            "text": "    return 1",
            "tokens_in": 5,
            "tokens_out": 3,
            "seconds": 2.0,
            "done_reason": "stop",
        }
        self.assertEqual(reply, expected)


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

    def test4(self) -> None:
        """
        Test the exact shares, with every combination of runs taken once.
        """
        # Prepare inputs: `a` is ahead of `b` in 2 of its 3 runs.
        matrices = {
            "a": pd.DataFrame([[1, 1, 0], [1, 1, 0]]),
            "b": pd.DataFrame([[1, 1, 1], [0, 0, 0]]),
        }
        # Run test.
        summary, pairs = rutils.compute_ranking_stability(matrices, n_draws=None)
        # Check outputs.
        self.assertAlmostEqual(summary["p_same_order"], 2 / 3)
        self.assertAlmostEqual(pairs.loc[0, "p_flip"], 1 / 3)


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
# Test_get_model_label
# #############################################################################


class Test_get_model_label(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_model_label()`.
    """

    def test1(self) -> None:
        """
        Test that the 2 DeepSeek models get different labels.
        """
        # Run test.
        labels = [
            rutils.get_model_label(model)
            for model in (
                "deepseek-coder:1.3b-instruct-q4_K_M",
                "deepseek-coder:6.7b-instruct-q4_K_M",
            )
        ]
        # Check outputs.
        self.assertEqual(labels, ["deepseek-coder 1.3b", "deepseek-coder 6.7b"])

    def test2(self) -> None:
        """
        Test the name of a public model, which has no tag.
        """
        # Run test.
        label = rutils.get_model_label("code-llama-7b")
        # Check outputs.
        self.assert_equal(label, "code-llama-7b")


# #############################################################################
# Test_get_agent_label
# #############################################################################


class Test_get_agent_label(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_agent_label()`.
    """

    def test1(self) -> None:
        """
        Test that the label keeps the name and the size of the model.
        """
        # Run test.
        label = rutils.get_agent_label("codeqwen:7b-chat-v1.5-q4_K_M", "repair")
        # Check outputs.
        self.assert_equal(label, "codeqwen 7b / repair")


# #############################################################################
# Test_get_agent_matrices
# #############################################################################


class Test_get_agent_matrices(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_agent_matrices()`.
    """

    def test1(self) -> None:
        """
        Test the matrices of 2 agents.
        """
        # Run test.
        matrices = rutils.get_agent_matrices(
            _get_two_agent_scores(),
            source="local",
            benchmark="humaneval",
            temperature=0.8,
        )
        # Check outputs.
        self.assertEqual(list(matrices), ["a 1 / one_shot", "b 1 / one_shot"])
        self.assertEqual(matrices["b 1 / one_shot"].sum().tolist(), [1, 1])

    def test2(self) -> None:
        """
        Test that 2 models with the same label are refused.
        """
        # Prepare inputs: 2 tags that keep the same name and size.
        scores = _get_two_agent_scores()
        scores["model"] = scores["model"].map({"a:1": "a:1-x", "b:1": "a:1-y"})
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rutils.get_agent_matrices(
                scores, source="local", benchmark="humaneval", temperature=0.8
            )


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
        self.assertEqual(list(table.index), ["a 1 / one_shot", "b 1 / one_shot"])
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
        self.assertEqual(table.loc[0, "agent_a"], "a 1 / one_shot")
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


# #############################################################################
# Test_get_public_greedy_agents
# #############################################################################


def _get_one_run_scores() -> pd.DataFrame:
    """
    Build the local and public scores of the Study B tests, on 4 tasks.

    - Local models `a:1`, `b:2`, and `deepseek-coder:6.7b-instruct-q4_K_M` at
      0.8, with both scaffolds and 2 runs
    - Public greedy agents `g1` and `deepseek-coder-6.7b-instruct`, and a
      public sampled model `s` at 0.8

    :return: scores in the format of `load_scores()`
    """
    deepseek = "deepseek-coder:6.7b-instruct-q4_K_M"
    # Model -> outcomes of each task in each of the 2 runs.
    local = {
        "a:1": [[1, 1], [1, 0], [0, 0], [0, 1]],
        "b:2": [[1, 1], [0, 0], [1, 0], [0, 0]],
        deepseek: [[1, 1], [1, 1], [0, 1], [0, 0]],
    }
    rows = []
    for model, tasks in local.items():
        for task, runs in enumerate(tasks):
            for run, passed in enumerate(runs):
                rows.append(
                    _get_score_row(model, 0.8, "one_shot", task, run, passed)
                )
                # The repair agent also solves every task the one-shot
                # agent misses in run 0.
                repaired = max(passed, int(run == 0))
                rows.append(
                    _get_score_row(model, 0.8, "repair", task, run, repaired)
                )
    public = {
        "g1": [1, 0, 0, 0],
        "deepseek-coder-6.7b-instruct": [1, 1, 1, 0],
    }
    for model, outcomes in public.items():
        for task, passed in enumerate(outcomes):
            rows.append(
                _get_score_row(
                    model, 0.0, "one_shot", task, 0, passed, source="evalplus"
                )
            )
    rows.append(_get_score_row("s", 0.8, "one_shot", 0, 0, 1, source="evalplus"))
    scores = pd.DataFrame(rows)
    return scores


class Test_get_public_greedy_agents(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_public_greedy_agents()`.
    """

    def test1(self) -> None:
        """
        Test that only the public greedy samples become extra agents.
        """
        # Prepare inputs.
        scores = _get_one_run_scores()
        # Prepare outputs.
        expected = """
        task_id                       HumanEval/0  HumanEval/1  HumanEval/2  HumanEval/3
        deepseek-coder-6.7b-instruct            1            1            1            0
        g1                                      1            0            0            0
        """
        # Run test.
        agents = rutils.get_public_greedy_agents(scores, "humaneval")
        # Check outputs.
        self.assert_equal(agents.to_string(), expected, dedent=True)


# #############################################################################
# Test_get_one_run_settings
# #############################################################################


class Test_get_one_run_settings(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_one_run_settings()`.
    """

    def test1(self) -> None:
        """
        Test the names and the order of a grid of 2 x 2 x 2 settings.
        """
        # Prepare outputs.
        expected = """
        rasch 0.0001 alone
        rasch 0.0001 extras
        rasch 1 alone
        rasch 1 extras
        2pl 0.0001 alone
        2pl 0.0001 extras
        2pl 1 alone
        2pl 1 extras
        """
        # Run test.
        settings = rutils.get_one_run_settings(
            ["rasch", "2pl"], [0.0001, 1.0], [False, True]
        )
        # Check outputs.
        self.assert_equal("\n".join(settings), expected, dedent=True)
        self.assertEqual(
            settings["2pl 1 extras"],
            {"irt_model": "2pl", "penalty": 1.0, "extras": True},
        )


# #############################################################################
# Test_evaluate_one_run_estimator
# #############################################################################


class Test_evaluate_one_run_estimator(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.evaluate_one_run_estimator()`.
    """

    def test1(self) -> None:
        """
        Test that every setting sees the same draws and that the truth comes
        from all the samples.
        """
        # Prepare inputs: 3 sampled models on 6 tasks with 4 samples each.
        rng = np.random.default_rng(1)
        samples = {
            model: pd.DataFrame(
                (rng.random((6, 4)) < 0.5).astype(int),
                index=[f"HumanEval/{task}" for task in range(6)],
            )
            for model in ("m1", "m2", "m3")
        }
        extras = pd.DataFrame(
            [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0]],
            index=["g1", "m2"],
            columns=[f"HumanEval/{task}" for task in range(6)],
        )
        rasch = {"irt_model": "rasch", "penalty": 0.01, "extras": False}
        settings = {
            "a": rasch,
            "b": dict(rasch),
            "c": {**rasch, "extras": True},
        }
        # Run test.
        results = rutils.evaluate_one_run_estimator(
            samples, extras, settings, 1, n_draws=2
        )
        # Check outputs.
        self.assertEqual(len(results), 3 * 3 * 2)
        by_setting = results.set_index(["setting", "model", "draw"])
        self.assertTrue(
            np.allclose(
                by_setting.loc["a", "estimate"],
                by_setting.loc["b", "estimate"],
                equal_nan=True,
            )
        )
        truth = rutils.compute_variance_components(samples["m1"])["run_share"]
        m1 = results[results["model"] == "m1"]
        self.assertTrue(np.allclose(m1["truth"], truth))


# #############################################################################
# Test_fit_calibration
# #############################################################################


class Test_fit_calibration(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.fit_calibration()`.
    """

    def test1(self) -> None:
        """
        Test estimates that are all twice the truth.
        """
        # Prepare inputs.
        estimates = pd.Series([0.2, 0.4])
        truths = pd.Series([0.1, 0.2])
        # Run test.
        factor = rutils.fit_calibration(estimates, truths)
        # Check outputs.
        self.assertAlmostEqual(factor, 0.5)

    def test2(self) -> None:
        """
        Test that the factor follows the large estimates, which carry most of
        the absolute error.
        """
        # Prepare inputs: the ratios truth / estimate are 1, 2, and 0.5.
        estimates = pd.Series([0.1, 0.1, 0.8])
        truths = pd.Series([0.1, 0.2, 0.4])
        # Run test.
        factor = rutils.fit_calibration(estimates, truths)
        # Check outputs: the errors add up to 0.2 with 0.5 and to 0.5 with 1.
        self.assertAlmostEqual(factor, 0.5)


# #############################################################################
# Test_summarize_one_run_errors
# #############################################################################


class Test_summarize_one_run_errors(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.summarize_one_run_errors()`.
    """

    def test1(self) -> None:
        """
        Test the errors with 1 factor for all the models and with the factor
        of the other models.
        """
        # Prepare inputs: `m1` is estimated at twice its truth and `m2`
        # exactly, in 2 draws each.
        results = pd.DataFrame(
            {
                "setting": "s",
                "model": ["m1", "m1", "m2", "m2"],
                "estimate": [0.4, 0.4, 0.5, 0.5],
                "truth": [0.2, 0.2, 0.5, 0.5],
            }
        )
        # Prepare outputs: `m2` has the larger weight, so the factor is 1;
        # without `m1` the factor is 1 and without `m2` it is 0.5.
        expected = """
        {'above': 0.5,
         'cv_error': 0.225,
         'cv_within': 0.0,
         'error': 0.1,
         'error_raw': 0.1,
         'factor': 1.0,
         'within': 0.5}
        """
        # Run test.
        summary = rutils.summarize_one_run_errors(results)
        # Check outputs.
        values = summary.loc["s"].items()
        actual = {key: round(value, 6) for key, value in values}
        self.assert_equal(pprint.pformat(actual), expected, dedent=True)


# #############################################################################
# Test_summarize_one_run_cases
# #############################################################################


class Test_summarize_one_run_cases(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.summarize_one_run_cases()`.
    """

    def test1(self) -> None:
        """
        Test the calibrated estimates of 1 model at 2 temperatures.
        """
        # Prepare inputs: with the factor 2, the draws are 0.15 and 0.35 at
        # 0.2, and 0.45 and 0.75 at 0.8.
        results = pd.DataFrame(
            {
                "setting": "s",
                "temperature": [0.2, 0.2, 0.8, 0.8],
                "model": "m",
                "estimate": [0.075, 0.175, 0.225, 0.375],
                "truth": [0.2, 0.2, 0.5, 0.5],
            }
        )
        # Prepare outputs.
        expected = """
        {(0.2, 'm'): {'estimate': 0.25,
                      'high': 0.33,
                      'low': 0.17,
                      'truth': 0.2,
                      'within': 0.5},
         (0.8, 'm'): {'estimate': 0.6,
                      'high': 0.72,
                      'low': 0.48,
                      'truth': 0.5,
                      'within': 0.5}}
        """
        # Run test.
        cases = rutils.summarize_one_run_cases(results, "s", 2.0)
        # Check outputs.
        actual = pprint.pformat(cases.round(6).to_dict(orient="index"))
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_compute_second_run_tasks
# #############################################################################


class Test_compute_second_run_tasks(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_second_run_tasks()`.
    """

    def test1(self) -> None:
        """
        Test an agent with no run luck, which 1 second run already shows.
        """
        # Prepare inputs: 5 tasks always pass and 5 always fail.
        outcomes = pd.DataFrame([[1, 1, 1]] * 5 + [[0, 0, 0]] * 5)
        # Run test.
        n_needed = rutils.compute_second_run_tasks(outcomes)
        # Check outputs.
        self.assertEqual(n_needed, 1.0)

    def test2(self) -> None:
        """
        Test runs whose first 2 disagree more than the other runs, so even a
        second run of every task is not enough.
        """
        # Prepare inputs: 20 tasks always pass and 20 alternate 0, 1, 0, 1, so
        # the first 2 runs always differ on them.
        outcomes = pd.DataFrame([[1, 1, 1, 1]] * 20 + [[0, 1, 0, 1]] * 20)
        # Run test.
        n_needed = rutils.compute_second_run_tasks(outcomes)
        # Check outputs: the 2 runs give a run share of 1.0, the 4 runs 0.88.
        self.assertTrue(np.isnan(n_needed))

    def test3(self) -> None:
        """
        Test an agent with run luck on half of 200 tasks.
        """
        # Prepare inputs.
        rng = np.random.default_rng(0)
        chances = np.repeat([0.0, 0.5, 1.0], [50, 100, 50])
        outcomes = pd.DataFrame(
            (rng.random((200, 5)) < chances[:, None]).astype(int)
        )
        # Run test.
        n_needed = rutils.compute_second_run_tasks(outcomes, n_draws=200)
        # Check outputs: 2 runs of a task with chance 0.5 differ only half the
        # time, so about 3 in 4 tasks need a second run.
        self.assertEqual(n_needed, 151.0)


# #############################################################################
# Test_get_one_run_cases
# #############################################################################


class Test_get_one_run_cases(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_one_run_cases()`.
    """

    def test1(self) -> None:
        """
        Test the number of cases of each phase, benchmark, and temperature.
        """
        # Prepare outputs.
        expected = """
        phase  benchmark  temperature
        1      humaneval  0.8            6
               mbpp       0.8            6
        2      humaneval  0.2            6
                          0.8            6
               mbpp       0.8            6
        """
        # Run test.
        cases = rutils.get_one_run_cases()
        # Check outputs.
        counts = cases.groupby(["phase", "benchmark", "temperature"]).size()
        self.assert_equal(counts.to_string(), expected, dedent=True)


# #############################################################################
# Test_build_one_run_matrix
# #############################################################################


class Test_build_one_run_matrix(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.build_one_run_matrix()`.
    """

    def test1(self) -> None:
        """
        Test that a repair agent comes with the one-shot agents of the other
        models only, and without the public agent of the same model.
        """
        # Prepare inputs.
        scores = _get_one_run_scores()
        # Prepare outputs.
        expected = """
        task_id                                         HumanEval/0  HumanEval/1  HumanEval/2  HumanEval/3
        a:1 / repair                                              1            1            1            1
        b:2 / one_shot                                            1            0            1            0
        deepseek-coder:6.7b-instruct-q4_K_M / one_shot            1            1            0            0
        g1                                                        1            0            0            0
        """
        # Run test.
        matrix = rutils.build_one_run_matrix(
            scores, "humaneval", 0.8, "a:1", "repair", extras=True
        )
        # Check outputs.
        self.assert_equal(matrix.to_string(), expected, dedent=True)

    def test2(self) -> None:
        """
        Test the matrix without extra agents, with the first run of each
        agent.
        """
        # Prepare inputs.
        scores = _get_one_run_scores()
        # Prepare outputs.
        expected = """
        task_id                                         HumanEval/0  HumanEval/1  HumanEval/2  HumanEval/3
        b:2 / one_shot                                            1            0            1            0
        a:1 / one_shot                                            1            1            0            0
        deepseek-coder:6.7b-instruct-q4_K_M / one_shot            1            1            0            0
        """
        # Run test.
        matrix = rutils.build_one_run_matrix(
            scores, "humaneval", 0.8, "b:2", "one_shot", extras=False
        )
        # Check outputs.
        self.assert_equal(matrix.to_string(), expected, dedent=True)


# #############################################################################
# Test_estimate_one_run_cases
# #############################################################################


class Test_estimate_one_run_cases(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.estimate_one_run_cases()`.
    """

    def test1(self) -> None:
        """
        Test that each case gets its truth from all the runs and 1 estimate
        per setting.
        """
        # Prepare inputs.
        scores = _get_one_run_scores()
        cases = pd.DataFrame(
            [(2, "humaneval", 0.8, "a:1", "one_shot")],
            columns=["phase", "benchmark", "temperature", "model", "scaffold"],
        )
        settings = {
            "untuned": rutils.UNTUNED_ONE_RUN_SETTING,
            "tuned": {
                **rutils.UNTUNED_ONE_RUN_SETTING,
                "calibration": 0.5,
                "extras": True,
            },
        }
        # Run test.
        results = rutils.estimate_one_run_cases(scores, cases, settings)
        # Check outputs.
        outcomes = rutils.get_outcome_matrix(
            scores,
            source="local",
            benchmark="humaneval",
            model="a:1",
            scaffold="one_shot",
            temperature=0.8,
        )
        truth = rutils.compute_variance_components(outcomes)["run_share"]
        row = results.iloc[0]
        self.assertEqual(row["n_runs"], 2)
        self.assertAlmostEqual(row["truth"], truth)
        self.assertTrue(0 < row["tuned"] < row["untuned"])


# #############################################################################
# Test_summarize_one_run_test
# #############################################################################


class Test_summarize_one_run_test(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.summarize_one_run_test()`.
    """

    def test1(self) -> None:
        """
        Test the means, the errors, and the shares of B1 and B2 in each group
        of cases.
        """
        # Prepare inputs: in the last group, the tuned estimates are off by
        # 0.20 and 0.05, so 1 of the 2 is within 10 points, and no estimate
        # before tuning is above the truth.
        results = pd.DataFrame(
            {
                "phase": [1, 2, 2, 2],
                "temperature": [0.8, 0.2, 0.8, 0.8],
                "truth": [0.4, 0.2, 0.5, 0.3],
                "untuned": [0.5, 0.35, 0.45, 0.2],
                "tuned": [0.45, 0.25, 0.3, 0.35],
            }
        )
        # Prepare outputs.
        expected = """
           phase  temperature  cases  truth  untuned  tuned  untuned_error  tuned_error  within  above
        0      1          0.8      1    0.4    0.500  0.450          0.100        0.050     1.0    1.0
        1      2          0.2      1    0.2    0.350  0.250          0.150        0.050     1.0    1.0
        2      2          0.8      2    0.4    0.325  0.325          0.075        0.125     0.5    0.0
        """
        # Run test.
        summary = rutils.summarize_one_run_test(
            results, tuned="tuned", untuned="untuned"
        )
        # Check outputs.
        actual = summary.reset_index().round(3).to_string()
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_check_one_run_predictions
# #############################################################################


def _get_one_run_results(n_within: int, n_above: int) -> pd.DataFrame:
    """
    Build the results of the 30 test cases with a known count for B1 and B2.

    The 12 phase-1 cases are far off with both settings, so they would change
    the counts if they were used.

    :param n_within: phase-2 cases with the tuned estimate within 10 points
    :param n_above: phase-2 cases with the estimate before tuning above the
        truth
    :return: results in the format of `estimate_one_run_cases()`
    """
    results = pd.DataFrame(
        {"phase": [1] * 12 + [2] * 18, "truth": 0.3, "tuned": 0.9}
    )
    results["untuned"] = 0.9
    phase_2 = results.index[results["phase"] == 2]
    results.loc[phase_2, "tuned"] = [0.35] * n_within + [0.5] * (18 - n_within)
    results.loc[phase_2, "untuned"] = [0.4] * n_above + [0.2] * (18 - n_above)
    return results


class Test_check_one_run_predictions(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.check_one_run_predictions()`.
    """

    def test1(self) -> None:
        """
        Test results where both predictions hold.
        """
        # Prepare inputs.
        results = _get_one_run_results(12, 10)
        # Prepare outputs.
        expected = """
        B1: True, 12 of 18 within 10 points
        B2: True, 10 of 18 above the truth
        """
        # Run test.
        table = rutils.check_one_run_predictions(
            results, tuned="tuned", untuned="untuned"
        )
        # Check outputs.
        actual = "\n".join(
            f"{name}: {row['holds']}, {row['observed']}"
            for name, row in table.iterrows()
        )
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test results just short of both predictions: 9 of 18 is not more than
        half.
        """
        # Prepare inputs.
        results = _get_one_run_results(11, 9)
        # Prepare outputs.
        expected = """
        B1: False, 11 of 18 within 10 points
        B2: False, 9 of 18 above the truth
        """
        # Run test.
        table = rutils.check_one_run_predictions(
            results, tuned="tuned", untuned="untuned"
        )
        # Check outputs.
        actual = "\n".join(
            f"{name}: {row['holds']}, {row['observed']}"
            for name, row in table.iterrows()
        )
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_compute_second_run_cases
# #############################################################################


class Test_compute_second_run_cases(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_second_run_cases()`.
    """

    def test1(self) -> None:
        """
        Test that each case gets the count of its own runs.
        """
        # Prepare inputs: `x:1` has no run luck, so 1 second run is enough,
        # while the first 2 runs of `y:1` always differ on 2 of its 4 tasks,
        # so even a second run of every task gives 1.0 against 0.8.
        runs = {
            "x:1": [[1, 1], [1, 1], [0, 0], [0, 0]],
            "y:1": [[1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1]],
        }
        rows = [
            _get_score_row(model, 0.8, "one_shot", task, run, passed)
            for model, tasks in runs.items()
            for task, outcomes in enumerate(tasks)
            for run, passed in enumerate(outcomes)
        ]
        scores = pd.DataFrame(rows)
        cases = pd.DataFrame(
            [
                (1, "humaneval", 0.8, "x:1", "one_shot"),
                (2, "humaneval", 0.8, "y:1", "one_shot"),
            ],
            columns=["phase", "benchmark", "temperature", "model", "scaffold"],
        )
        # Prepare outputs.
        expected = """
           phase  benchmark  temperature model  scaffold  n_tasks  second_run_tasks
        0      1  humaneval          0.8   x:1  one_shot        4               1.0
        1      2  humaneval          0.8   y:1  one_shot        4               NaN
        """
        # Run test.
        results = rutils.compute_second_run_cases(scores, cases)
        # Check outputs.
        self.assert_equal(results.to_string(), expected, dedent=True)


# #############################################################################
# Test_combine_scores
# #############################################################################


def _get_phase_scores() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the scores of the 2 phases of runs, on 2 tasks.

    - Phase 1: model `a:1` has 1 greedy run, which passes both tasks, and 2
      runs at 0.8
    - Phase 2: the greedy file of `a:1` has a second run and was scored
      again, and this time run 0 fails task 1; model `b:2` has 1 greedy run

    :return: the scores of phase 1 and of phase 2
    """
    old = [_get_score_row("a:1", 0.0, "one_shot", task, 0, 1) for task in (0, 1)]
    old += [
        _get_score_row("a:1", 0.8, "one_shot", task, run, task)
        for task in (0, 1)
        for run in (0, 1)
    ]
    new = [
        _get_score_row("a:1", 0.0, "one_shot", 0, 0, 1),
        _get_score_row("a:1", 0.0, "one_shot", 1, 0, 0),
        _get_score_row("a:1", 0.0, "one_shot", 0, 1, 1),
        _get_score_row("a:1", 0.0, "one_shot", 1, 1, 1),
        _get_score_row("b:2", 0.0, "one_shot", 0, 0, 1),
        _get_score_row("b:2", 0.0, "one_shot", 1, 0, 0),
    ]
    return pd.DataFrame(old), pd.DataFrame(new)


class Test_combine_scores(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.combine_scores()`.
    """

    def test1(self) -> None:
        """
        Test that a file scored in both phases keeps only its phase-2 scores.
        """
        # Prepare inputs.
        old, new = _get_phase_scores()
        # Run test.
        scores = rutils.combine_scores(old, new)
        # Check outputs.
        counts = scores.groupby(["model", "temperature"]).size().to_dict()
        self.assertEqual(
            counts, {("a:1", 0.0): 4, ("a:1", 0.8): 4, ("b:2", 0.0): 2}
        )
        greedy = rutils.get_outcome_matrix(
            scores,
            source="local",
            benchmark="humaneval",
            model="a:1",
            scaffold="one_shot",
            temperature=0.0,
        )
        self.assertEqual(greedy[0].tolist(), [1, 0])

    def test2(self) -> None:
        """
        Test that phase-2 scores missing a run of phase 1 are refused.
        """
        # Prepare inputs: drop run 0 of the phase-2 greedy file.
        old, new = _get_phase_scores()
        new = new[new["run"] == 1]
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rutils.combine_scores(old, new)


# #############################################################################
# Test_compute_rescored_agreement
# #############################################################################


class Test_compute_rescored_agreement(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_rescored_agreement()`.
    """

    def test1(self) -> None:
        """
        Test 2 answers scored in both phases, 1 with a different result.
        """
        # Prepare inputs.
        old, new = _get_phase_scores()
        # Run test.
        agreement = rutils.compute_rescored_agreement(old, new)
        # Check outputs.
        self.assertEqual(agreement, {"n_answers": 2, "same_share": 0.5})


# #############################################################################
# Test_compute_greedy_changes
# #############################################################################


class Test_compute_greedy_changes(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_greedy_changes()`.
    """

    def test1(self) -> None:
        """
        Test a model whose 2 greedy runs differ on 1 of 2 tasks.
        """
        # Prepare inputs.
        _, new = _get_phase_scores()
        # Run test.
        table = rutils.compute_greedy_changes(
            new, models=["a:1"], benchmark="humaneval"
        )
        # Check outputs.
        actual = pprint.pformat(table.loc["a:1"].to_dict())
        expected = r"""
        {'changed_share': 0.5,
         'n_changed': 1.0,
         'n_tasks': 2.0,
         'score_run_0': 0.5,
         'score_run_1': 1.0}
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test that a model with 1 greedy run is refused.
        """
        # Prepare inputs.
        _, new = _get_phase_scores()
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rutils.compute_greedy_changes(
                new, models=["b:2"], benchmark="humaneval"
            )


# #############################################################################
# Test_get_first_runs_matrices
# #############################################################################


def _get_stability_scores() -> pd.DataFrame:
    """
    Build the scores of 2 models on 2 tasks, with 3 runs at 2 temperatures.

    - At 0.2, `a:1` passes both tasks and `b:2` passes task 0, in every run
    - At 0.8, `a:1` passes both tasks in runs 0 and 1 and none in run 2,
      while `b:2` passes task 0 in every run, so a draw of run 2 for `a:1`
      reverses the order
    - The repair scaffold has the same results as the one-shot scaffold

    :return: scores in the format of `load_scores()`
    """
    rows = []
    for scaffold in ("one_shot", "repair"):
        for task in (0, 1):
            for run in (0, 1, 2):
                rows.append(_get_score_row("a:1", 0.2, scaffold, task, run, 1))
                rows.append(
                    _get_score_row(
                        "b:2", 0.2, scaffold, task, run, int(task == 0)
                    )
                )
                rows.append(
                    _get_score_row("a:1", 0.8, scaffold, task, run, int(run < 2))
                )
                rows.append(
                    _get_score_row(
                        "b:2", 0.8, scaffold, task, run, int(task == 0)
                    )
                )
    scores = pd.DataFrame(rows)
    return scores


class Test_get_first_runs_matrices(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.get_first_runs_matrices()`.
    """

    def test1(self) -> None:
        """
        Test that only the first 2 of the 3 runs are kept.
        """
        # Run test.
        matrices = rutils.get_first_runs_matrices(
            _get_stability_scores(),
            models=["a:1", "b:2"],
            scaffold="one_shot",
            benchmark="humaneval",
            temperature=0.8,
            n_runs=2,
        )
        # Check outputs.
        self.assertEqual(list(matrices), ["a 1 / one_shot", "b 2 / one_shot"])
        self.assertEqual(list(matrices["a 1 / one_shot"].columns), [0, 1])
        self.assertEqual(matrices["a 1 / one_shot"].sum().sum(), 4)

    def test2(self) -> None:
        """
        Test that asking for more runs than an agent has is refused.
        """
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rutils.get_first_runs_matrices(
                _get_stability_scores(),
                models=["a:1", "b:2"],
                scaffold="one_shot",
                benchmark="humaneval",
                temperature=0.8,
                n_runs=4,
            )


# #############################################################################
# Test_compute_stability_table
# #############################################################################


class Test_compute_stability_table(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.compute_stability_table()`.
    """

    def test1(self) -> None:
        """
        Test an order that always holds at 0.2 and holds in 2 of the 3
        combinations of runs at 0.8.
        """
        # Run test.
        table = rutils.compute_stability_table(
            _get_stability_scores(),
            models=["a:1", "b:2"],
            benchmark="humaneval",
            temperatures=[0.2, 0.8],
            n_runs=3,
        )
        # Check outputs.
        self.assertEqual(
            list(table.index),
            [
                ("one_shot", 0.2),
                ("one_shot", 0.8),
                ("repair", 0.2),
                ("repair", 0.8),
            ],
        )
        self.assertEqual(table.loc[("one_shot", 0.2), "p_same_order"], 1.0)
        self.assertAlmostEqual(table.loc[("repair", 0.8), "p_same_order"], 2 / 3)


# #############################################################################
# Test_check_temperature_predictions
# #############################################################################


def _get_temperature_prediction_inputs() -> Dict[str, Any]:
    """
    Build the inputs of `check_temperature_predictions()`, with every
    prediction holding.

    :return: keyword arguments of `check_temperature_predictions()`
    """
    models = ["w:1b", "a:7b", "b:8b"]
    scaffolds = ["one_shot", "repair"]
    labels = [rutils.get_agent_label(m, s) for m in models for s in scaffolds]
    # Runs are 20% of the variance at 0.2 and 40% at 0.8, listed in another
    # order.
    variance_low = pd.DataFrame({"run_share": 0.2}, index=labels)
    variance_high = pd.DataFrame({"run_share": 0.4}, index=labels[::-1])
    greedy_changes = pd.DataFrame(
        {"changed_share": [0.0, 0.006, 0.012]}, index=models
    )
    comparisons = pd.DataFrame(
        {"p_holm": [0.4, 0.8], "distinguishable": [False, False]}
    )
    index = pd.MultiIndex.from_product(
        [scaffolds, [0.2, 0.8]], names=["scaffold", "temperature"]
    )
    stability = pd.DataFrame(
        {"p_same_order": [0.9, 0.8, 0.95, 0.85]}, index=index
    )
    inputs = {
        "variance_low": variance_low,
        "variance_high": variance_high,
        "greedy_changes": greedy_changes,
        "comparisons": comparisons,
        "stability": stability,
    }
    return inputs


class Test_check_temperature_predictions(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.check_temperature_predictions()`.
    """

    def test1(self) -> None:
        """
        Test results where every prediction holds.
        """
        # Run test.
        table = rutils.check_temperature_predictions(
            **_get_temperature_prediction_inputs()
        )
        # Check outputs.
        self.assertEqual(list(table.index), [f"C{i}" for i in range(1, 6)])
        self.assertEqual(table["holds"].tolist(), [True] * 5)
        self.assertEqual(
            table.loc["C2", "observed"], "6 of 6 lower, by 20 to 20 points"
        )
        self.assertEqual(
            table.loc["C5", "observed"],
            "2 of 2 scaffolds more often (one_shot 90.0% vs 80.0%, repair "
            "95.0% vs 85.0%)",
        )

    def test2(self) -> None:
        """
        Test results where every prediction fails.
        """
        # Prepare inputs: break each prediction once.
        inputs = _get_temperature_prediction_inputs()
        # C1 and C2: an agent has more run luck at 0.2 than at 0.8.
        inputs["variance_low"].iloc[0, 0] = 0.45
        # C3: 3% of the tasks of a model change between the greedy runs.
        inputs["greedy_changes"].iloc[2, 0] = 0.03
        # C4: the close pair is distinguishable with 1 scaffold.
        inputs["comparisons"].loc[0, ["p_holm", "distinguishable"]] = [
            0.01,
            True,
        ]
        # C5: with repair, the order holds less often at 0.2.
        inputs["stability"].loc[("repair", 0.2), "p_same_order"] = 0.8
        # Run test.
        table = rutils.check_temperature_predictions(**inputs)
        # Check outputs.
        self.assertEqual(table["holds"].tolist(), [False] * 5)
        self.assertEqual(
            table.loc["C1", "observed"], "5 of 6 inside, from 20% to 45%"
        )
        self.assertEqual(
            table.loc["C3", "observed"],
            "2 of 3 models below 2%, from 0.0% to 3.0%",
        )


# #############################################################################
# Test_check_new_model_predictions
# #############################################################################


def _get_new_model_prediction_inputs() -> Dict[str, Any]:
    """
    Build the inputs of `check_new_model_predictions()`, with every
    prediction holding.

    :return: keyword arguments of `check_new_model_predictions()`
    """
    models = ["d:6.7b-x", "l:8b-x", "g:7b-x"]
    validation = pd.DataFrame(
        {"plus_diff": [0.01, -0.02, 0.03, 0.0, -0.04, 0.02]},
        index=pd.MultiIndex.from_product(
            [models, ["humaneval", "mbpp"]], names=["model", "benchmark"]
        ),
    )
    comparisons = pd.DataFrame(
        {"p_holm": [0.3, 0.6, 1.0], "distinguishable": [False] * 3}
    )
    stability = pd.Series({"p_same_order": 0.3, "mean_kendall_tau": 0.8})
    inputs = {
        "validation": validation,
        "comparisons": comparisons,
        "stability": stability,
    }
    return inputs


class Test_check_new_model_predictions(hunitest.TestCase):
    """
    Test `skill_vs_luck_runs_utils.check_new_model_predictions()`.
    """

    def test1(self) -> None:
        """
        Test results where every prediction holds.
        """
        # Run test.
        table = rutils.check_new_model_predictions(
            **_get_new_model_prediction_inputs()
        )
        # Check outputs.
        self.assertEqual(list(table.index), ["D1", "D2", "D3"])
        self.assertEqual(table["holds"].tolist(), [True] * 3)
        self.assertEqual(
            table.loc["D1", "observed"],
            "6 of 6 within 5 points, largest gap -4.0 points (g 7b, humaneval)",
        )

    def test2(self) -> None:
        """
        Test results where every prediction fails.
        """
        # Prepare inputs: break each prediction once.
        inputs = _get_new_model_prediction_inputs()
        # D1: a greedy score is 6 points below the published one.
        inputs["validation"].iloc[5, 0] = -0.06
        # D2: a pair of the cluster is distinguishable.
        inputs["comparisons"].loc[2, ["p_holm", "distinguishable"]] = [
            0.02,
            True,
        ]
        # D3: most 1-run leaderboards keep the order.
        inputs["stability"]["p_same_order"] = 0.6
        # Run test.
        table = rutils.check_new_model_predictions(**inputs)
        # Check outputs.
        self.assertEqual(table["holds"].tolist(), [False] * 3)
        self.assertEqual(
            table.loc["D2", "observed"],
            "1 of 3 pairs distinguishable, smallest Holm p 0.02",
        )
        self.assertEqual(
            table.loc["D3", "observed"], "60.0% keep the order of the 6 models"
        )
