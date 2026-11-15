import json
import os
import unittest.mock as umock

import helpers.hunit_test as hunitest

import research.agentic_data_science.bakeoff.benchmarks as radsbben
import research.agentic_data_science.bakeoff.run_benchmarks as radsbrube
import research.agentic_data_science.bakeoff.task_agents as radsbtaag


def _task(task_id: str = "1", gold: str = "0.5") -> radsbben.Task:
    return radsbben.Task(
        "qrdata", task_id, "What is the rate?", [], gold, {"question_type": "numerical"}
    )


class _FixedAgent(radsbtaag.TaskAgent):
    """
    Answer every task with the same text; crash on task id "boom".
    """

    name = "fixed"

    def __init__(self, text: str) -> None:
        self.text = text

    def answer(self, task, workdir):
        if task.task_id == "boom":
            raise RuntimeError("agent crashed")
        return radsbtaag.TaskResult(answer=self.text, cost_usd=0.01)


# #############################################################################
# Test_prompt
# #############################################################################


class Test_prompt(hunitest.TestCase):
    def test_extract_final_answer(self) -> None:
        """
        The last `FINAL ANSWER:` line wins; no marker gives "".
        """
        text = "draft FINAL ANSWER: 1\nchecked again\nFinal Answer: `0.52`"
        self.assertEqual(radsbtaag.extract_final_answer(text), "0.52")
        self.assertEqual(radsbtaag.extract_final_answer("no marker"), "")

    def test_build_prompt_lists_files(self) -> None:
        """
        The prompt lists the working-directory files by relative path.
        """
        workdir = self.get_scratch_space()
        open(os.path.join(workdir, "data.csv"), "w").close()
        prompt = radsbtaag.build_prompt(_task(), workdir)
        self.assertIn("- data.csv", prompt)
        self.assertIn("What is the rate?", prompt)
        self.assertIn("FINAL ANSWER:", prompt)
        tool = radsbtaag.build_prompt(_task(), workdir, radsbtaag.FINISH_TOOL)
        self.assertIn("final_answer", tool)
        self.assertNotIn("FINAL ANSWER:", tool)

    def test_timed_answer_records_crash(self) -> None:
        res = _FixedAgent("x").timed_answer(_task("boom"), self.get_scratch_space())
        self.assertIn("agent crashed", res.error)
        self.assertIsNotNone(res.runtime_s)


# #############################################################################
# Test_agents
# #############################################################################


class Test_agents(hunitest.TestCase):
    def test_claude_code_reads_result(self) -> None:
        """
        The Claude Code wrapper takes answer, cost and tokens from the result.
        """
        import claude_agent_sdk as cas

        result = cas.ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
            num_turns=3, session_id="s", total_cost_usd=0.12,
            usage={"input_tokens": 100, "cache_read_input_tokens": 50,
                   "output_tokens": 20},
            result="The rate is about half.\nFINAL ANSWER: 0.5",
        )

        async def fake_query(prompt, options):
            self.assertEqual(options.setting_sources, [])
            yield result

        with umock.patch.object(cas, "query", fake_query):
            res = radsbtaag.ClaudeCodeAgent().answer(_task(), self.get_scratch_space())
        self.assertEqual(res.answer, "0.5")
        self.assertEqual(res.cost_usd, 0.12)
        self.assertEqual(res.tokens_in, 150)
        self.assertEqual(res.tokens_out, 20)
        self.assertIsNone(res.error)

    def test_smolagents_needs_model(self) -> None:
        with umock.patch.dict(os.environ, {radsbtaag.MODEL_ENV: ""}):
            res = radsbtaag.SmolagentsAgent().answer(_task(), self.get_scratch_space())
        self.assertIn(radsbtaag.MODEL_ENV, res.error)

    def test_oracle(self) -> None:
        res = radsbtaag.OracleAgent().answer(_task(gold="0.5"), "")
        self.assertEqual(radsbben.score(_task(gold="0.5"), res.answer), 1.0)


# #############################################################################
# Test_run_benchmarks
# #############################################################################


class Test_run_benchmarks(hunitest.TestCase):
    def _run(self, runs_file: str, agent: radsbtaag.TaskAgent, tasks) -> int:
        with umock.patch.object(
            radsbtaag, "get_task_agent", return_value=agent
        ), umock.patch.object(radsbben, "load_sample", return_value=tasks):
            return radsbrube.run(["fixed"], ["qrdata"], runs_file=runs_file)

    def test_run_scores_and_resumes(self) -> None:
        """
        Runs are scored and written once; a crash scores 0 and is kept.
        """
        runs = os.path.join(self.get_scratch_space(), "runs.jsonl")
        tasks = [_task("1", "0.5"), _task("2", "0.9"), _task("boom", "0.5")]
        self.assertEqual(self._run(runs, _FixedAgent("FINAL 0.5"), tasks), 3)
        self.assertEqual(self._run(runs, _FixedAgent("FINAL 0.5"), tasks), 0)
        with open(runs, encoding="utf-8") as f:
            rows = {r["task_id"]: r for r in map(json.loads, f)}
        self.assertEqual(rows["1"]["score"], 1.0)
        self.assertEqual(rows["2"]["score"], 0.0)
        self.assertEqual(rows["boom"]["score"], 0.0)
        self.assertIn("agent crashed", rows["boom"]["error"])
        summary = radsbrube.summarize(runs)[0]
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(summary["errors"], 1)
        self.assertAlmostEqual(summary["cost_usd"], 0.02)

    def test_workdir_copies_files(self) -> None:
        scratch = self.get_scratch_space()
        src = os.path.join(scratch, "in.csv")
        with open(src, "w", encoding="utf-8") as f:
            f.write("a\n1\n")
        task = radsbben.Task("qrdata", "7", "q", [src], "1", {})
        workdir = radsbrube.make_workdir(task, os.path.join(scratch, "work"), "fixed")
        self.assertEqual(os.listdir(workdir), ["in.csv"])

    def test_bootstrap_ci(self) -> None:
        lo, hi = radsbrube.bootstrap_ci([1.0] * 5 + [0.0] * 5)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertEqual(radsbrube.bootstrap_ci([1.0, 1.0]), (1.0, 1.0))
