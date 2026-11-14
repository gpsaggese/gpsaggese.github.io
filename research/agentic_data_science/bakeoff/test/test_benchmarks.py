import json
import os

import helpers.hunit_test as hunitest

import research.agentic_data_science.bakeoff.benchmarks as aebbench


def _task(benchmark: str, gold, **meta) -> aebbench.Task:
    return aebbench.Task(benchmark, "0", "q", [], gold, meta)


def _write_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# #############################################################################
# Test_scorers
# #############################################################################


class Test_scorers(hunitest.TestCase):
    def test_dabench(self) -> None:
        """
        Every `@name[value]` must match; numbers within 1e-6.
        """
        t = _task("dabench", [["mean_fare", "34.65"], ["count", "3"]])
        self.assertEqual(aebbench.score(t, "@mean_fare[34.65] @count[3]"), 1.0)
        self.assertEqual(aebbench.score(t, "@count[3.0] then @mean_fare[34.65]"), 1.0)
        self.assertEqual(aebbench.score(t, "@mean_fare[34.65]"), 0.0)
        self.assertEqual(aebbench.score(t, "@mean_fare[34.66] @count[3]"), 0.0)

    def test_qrdata_numerical(self) -> None:
        """
        The first number counts and must be within 3% of the gold.
        """
        t = _task("qrdata", "0.20", question_type="numerical")
        self.assertEqual(aebbench.score(t, "The proportion is 0.204"), 1.0)
        self.assertEqual(aebbench.score(t, "20%"), 1.0)
        self.assertEqual(aebbench.score(t, "0.21"), 0.0)
        self.assertEqual(aebbench.score(t, "no idea"), 0.0)

    def test_qrdata_choice(self) -> None:
        t = _task("qrdata", "B", question_type="multiple choice")
        self.assertEqual(aebbench.score(t, "b) because"), 1.0)
        self.assertEqual(aebbench.score(t, "the answer is B"), 0.0)

    def test_dacode(self) -> None:
        """
        JSON fields must match; keys are case-insensitive, floats within 0.01.
        """
        gold = [{"highest country": ["Uruguay"], "Agricultural Land %": [82.6]}]
        t = _task("dacode", gold, ignore_order=False)
        pred = 'Answer: {"Highest Country": "uruguay", "agricultural land %": 82.605}'
        self.assertEqual(aebbench.score(t, pred), 1.0)
        self.assertEqual(
            aebbench.score(t, '{"highest country": ["Chile"], "agricultural land %": [82.6]}'),
            0.0,
        )
        self.assertEqual(aebbench.score(t, '{"highest country": ["Uruguay"]}'), 0.0)
        self.assertEqual(aebbench.score(t, "Uruguay, 82.6"), 0.0)

    def test_dacode_ignore_order(self) -> None:
        gold = [{"countries": ["A", "B"]}]
        ordered = _task("dacode", gold, ignore_order=False)
        unordered = _task("dacode", gold, ignore_order=True)
        self.assertEqual(aebbench.score(ordered, '{"countries": ["B", "A"]}'), 0.0)
        self.assertEqual(aebbench.score(unordered, '{"countries": ["B", "A"]}'), 1.0)

    def test_dabstep(self) -> None:
        """
        Official scorer: lists ignore order, numbers ignore separators.
        """
        self.assertEqual(aebbench.score(_task("dabstep", "NL, BE"), "be; nl"), 1.0)
        self.assertEqual(aebbench.score(_task("dabstep", "1234.5"), "1,234.50"), 1.0)
        self.assertEqual(aebbench.score(_task("dabstep", "64"), "65"), 0.0)

    def test_gold_scores_one(self) -> None:
        """
        The gold answer, written as an agent would, gets full marks.
        """
        tasks = [
            _task("dabench", [["a", "1.5"], ["b", "x"]]),
            _task("qrdata", "12%", question_type="numerical"),
            _task("dacode", [{"k": [1, 2]}], ignore_order=False),
            _task("dabstep", "Ecommerce"),
        ]
        for t in tasks:
            self.assertEqual(aebbench.score(t, aebbench.gold_as_prediction(t)), 1.0)
            self.assertEqual(aebbench.score(t, ""), 0.0)


# #############################################################################
# Test_loaders
# #############################################################################


class Test_loaders(hunitest.TestCase):
    def test_dabench_skips_duplicate_labels(self) -> None:
        """
        A label that repeats an answer name is skipped.
        """
        root = self.get_scratch_space()
        d = os.path.join(root, "InfiAgent", "examples", "DA-Agent", "data")
        q = {"constraints": "c", "format": "@x[v]", "file_name": "t.csv",
             "level": "easy", "concepts": []}
        _write_jsonl(
            os.path.join(d, "da-dev-questions.jsonl"),
            [dict(q, id=1, question="ok"), dict(q, id=2, question="bad")],
        )
        _write_jsonl(
            os.path.join(d, "da-dev-labels.jsonl"),
            [{"id": 1, "common_answers": [["x", "1"]]},
             {"id": 2, "common_answers": [["x", "1"], ["x", "2"]]}],
        )
        tasks = aebbench.load_tasks("dabench", root)
        self.assertEqual([t.task_id for t in tasks], ["1"])
        self.assertTrue(tasks[0].files[0].endswith("da-dev-tables/t.csv"))
        self.assertIn("Answer format: @x[v]", tasks[0].question)

    def test_dacode_keeps_text_tasks(self) -> None:
        """
        Only tasks scored by `compare_text` are loaded.
        """
        root = self.get_scratch_space()
        d = os.path.join(root, "da-code", "da_code")
        _write_jsonl(
            os.path.join(d, "configs", "task", "all.jsonl"),
            [{"id": i, "type": "t", "instruction": "do", "hardness": "Easy"}
             for i in ("di-text-001", "di-csv-001")],
        )
        _write_jsonl(
            os.path.join(d, "configs", "eval", "eval_all.jsonl"),
            [{"id": "di-text-001", "func": ["compare_text"],
              "result": [{"number": [{"k": 1}]}], "options": [{}]},
             {"id": "di-csv-001", "func": ["compare_csv"],
              "result": [{"file": ["r.csv"]}], "options": [{}]}],
        )
        os.makedirs(os.path.join(d, "source", "di-text-001"))
        open(os.path.join(d, "source", "di-text-001", "data.csv"), "w").close()
        tasks = aebbench.load_tasks("dacode", root)
        self.assertEqual([t.task_id for t in tasks], ["di-text-001"])
        self.assertEqual(tasks[0].gold, [{"k": 1}])
        self.assertEqual(len(tasks[0].files), 1)

    def test_dabstep_hidden_answers(self) -> None:
        """
        Test tasks load with `gold=None` and cannot be scored locally.
        """
        root = self.get_scratch_space()
        d = os.path.join(root, "DABstep", "data", "tasks")
        row = {"task_id": 5, "question": "q", "guidelines": "g", "level": "easy"}
        _write_jsonl(os.path.join(d, "dev.jsonl"), [dict(row, answer="NL")])
        _write_jsonl(os.path.join(d, "all.jsonl"), [dict(row, answer="")])
        dev = aebbench.load_dabstep(root, "dev")
        test = aebbench.load_dabstep(root, "default")
        self.assertEqual(dev[0].gold, "NL")
        self.assertIsNone(test[0].gold)
        with self.assertRaises(ValueError):
            aebbench.score(test[0], "NL")

    def test_sample_ids(self) -> None:
        """
        Sampling is fixed by the seed, and small benchmarks are taken whole.
        """
        tasks = [_task("qrdata", "1") for _ in range(100)]
        for i, t in enumerate(tasks):
            t.task_id = str(i)
        a = aebbench.sample_ids(tasks, 10, seed=0)
        self.assertEqual(a, aebbench.sample_ids(tasks, 10, seed=0))
        self.assertNotEqual(a, aebbench.sample_ids(tasks, 10, seed=1))
        self.assertEqual(len(a), 10)
        self.assertEqual(aebbench.sample_ids(tasks[:5], 10), "all")
