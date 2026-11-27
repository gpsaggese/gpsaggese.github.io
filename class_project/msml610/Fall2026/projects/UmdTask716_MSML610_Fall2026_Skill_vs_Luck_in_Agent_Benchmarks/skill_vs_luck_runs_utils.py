"""
Helper functions for the coding-benchmark runs of the Skill vs Luck project.

Used by `run_coding_agents.py`, `score_coding_runs.py`, and the
`skill_vs_luck.runs`, `skill_vs_luck.one_run`,
`skill_vs_luck.one_run_test_set`, `skill_vs_luck.temperatures`, and
`skill_vs_luck.more_models` notebooks:
- Load HumanEval+ and MBPP+ with EvalPlus
- Ask a model served by Ollama for a solution, with the EvalPlus prompt
- Check a solution on the examples shown in the prompt (repair scaffold)
- Score solutions on the EvalPlus tests and label each failure
- Read the public EvalPlus samples
- Split the variance of a score into its task and run parts, compare agents
  over tasks and runs, and check how stable a 1-run leaderboard is
- Tune and test the estimator of run luck from 1 run (Study B)
- Check the predictions of the lower temperatures (Study C) and of the 3
  more models (Study D)

Import as:

import skill_vs_luck_runs_utils as rutils
"""

import ast
import concurrent.futures
import doctest
import gc
import glob
import hashlib
import itertools
import json
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from typing import Any, Dict, Iterator, List, Optional, Tuple

import evalplus.config
import evalplus.data
import evalplus.eval
import evalplus.eval._special_oracle
import evalplus.eval.utils
import evalplus.evaluate
import evalplus.sanitize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.stats
import tqdm

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint
import helpers.hretry as hretry

import skill_vs_luck_utils as svlutils

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


# Benchmarks, with the dataset files that `evalplus==0.3.1` pins.
BENCHMARKS = ("humaneval", "mbpp")
_DATASET_FILES = {
    "humaneval": "HumanEvalPlus-v0.1.10.jsonl",
    "mbpp": "MbppPlus-v0.2.0.jsonl",
}
# Prompt that EvalPlus uses for chat models (`evalplus/codegen.py`).
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)
# Sampling of the EvalPlus Hugging Face backend: `top_p` 0.95, the default
# `top_k` of 50, no repeat penalty, and at most 768 new tokens. Ollama's own
# defaults (e.g., repeat penalty 1.1) would change the outputs.
GENERATION_OPTIONS = {
    "top_p": 0.95,
    "top_k": 50,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "num_predict": 768,
    "num_ctx": 4096,
}
# Texts where EvalPlus cuts a generation (`evalplus/provider/utility.py`),
# plus the end of the code block, after which only prose follows.
STOP_TEXTS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n```",
]
# Ollama server, as seen from the Docker container.
DEFAULT_OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", "http://host.docker.internal:11434"
)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Connect the notebook logger to this module's logger.

    :param notebook_log: logger of the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Data
# #############################################################################


def load_problems(benchmark: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the tasks of a benchmark with the EvalPlus loader.

    The loader also converts the MBPP+ inputs that JSON cannot store (e.g.,
    tuples), so scoring sees the same inputs as EvalPlus.

    :param benchmark: `humaneval` or `mbpp`
    :return: task id -> task, with `prompt`, `entry_point`, `base_input`,
        `plus_input`, `atol`, and `canonical_solution`
    """
    hdbg.dassert_in(benchmark, BENCHMARKS)
    # Make sure that EvalPlus reads the local file instead of downloading it.
    path = os.path.join(evalplus.data.utils.CACHE_DIR, _DATASET_FILES[benchmark])
    hdbg.dassert_file_exists(
        path, "Set XDG_CACHE_HOME (see `docker_cmd_sandbox.sh`)"
    )
    if benchmark == "humaneval":
        problems = evalplus.data.get_human_eval_plus()
    else:
        problems = evalplus.data.get_mbpp_plus()
    return problems


def get_expected_outputs(
    benchmark: str, problems: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute the expected outputs by running the reference solutions.

    EvalPlus caches the result, so only the first call is slow.

    :param benchmark: `humaneval` or `mbpp`
    :param problems: output of `load_problems()`
    :return: task id -> `base`, `plus` outputs and their run times
    """
    if benchmark == "humaneval":
        dataset_hash = evalplus.data.get_human_eval_plus_hash()
        not_none_tasks = []
    else:
        dataset_hash = evalplus.data.get_mbpp_plus_hash()
        # Tasks where only "returns None or not" is checked.
        not_none_tasks = evalplus.eval._special_oracle.MBPP_OUTPUT_NOT_NONE_TASKS
    expected = evalplus.evaluate.get_groundtruth(
        problems, dataset_hash, not_none_tasks
    )
    return expected


def get_task_label(name: str) -> str:
    """
    Convert a task folder name of the EvalPlus samples into a task id.

    :param name: e.g., `HumanEval_81` or `Mbpp_468`
    :return: e.g., `HumanEval/81` or `Mbpp/468`
    """
    match = re.match(r"^(HumanEval|Mbpp)_(\d+)$", name)
    match = hdbg.dassert_re_match(match, "Unknown task folder '%s'", name)
    task_id = f"{match.group(1)}/{match.group(2)}"
    return task_id


def clean_sample(code: str, entry_point: str) -> str:
    """
    Get the code to score from a public EvalPlus sample.

    A sample that is already code (it parses and defines the function) is
    scored as is, as EvalPlus did for its leaderboard. A raw chat answer
    (prose around a code block) goes through `extract_code()`.

    :param code: content of the sample file
    :param entry_point: name of the function to solve
    :return: code to score
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    defines = tree is not None and any(
        isinstance(node, ast.FunctionDef) and node.name == entry_point
        for node in tree.body
    )
    if not defines:
        code = extract_code(code, entry_point)
    return code


def iter_zip_samples(
    zip_path: str, *, max_samples: Optional[int] = None
) -> Iterator[Tuple[str, int, str]]:
    """
    Read the solutions in a public EvalPlus sample zip, without extracting it.

    Each zip has 1 folder per task and 1 file per sample, e.g.,
    `.../HumanEval_81/3.py`.

    :param zip_path: path to the zip
    :param max_samples: keep only samples `0`, ..., `max_samples - 1` of each
        task, or all of them if `None`
    :return: iterator over `(task_id, sample_index, code)`, sorted
    """
    with zipfile.ZipFile(zip_path) as zip_file:
        names = [n for n in zip_file.namelist() if n.endswith(".py")]
        entries = []
        for name in names:
            folder, file_name = name.split("/")[-2:]
            index = int(file_name[: -len(".py")])
            if max_samples is None or index < max_samples:
                entries.append((get_task_label(folder), index, name))
        for task_id, index, name in sorted(entries):
            code = zip_file.read(name).decode("utf-8")
            yield task_id, index, code


# #############################################################################
# Generation with Ollama
# #############################################################################


def get_cell_seed(task_id: str, run: int) -> int:
    """
    Get the sampling seed of 1 cell of the outcome tensor.

    The seed depends only on the task and the run, so any cell can be
    generated again on its own.

    :param task_id: e.g., `HumanEval/2`
    :param run: run index, e.g., `0`
    :return: seed in `[0, 2^31)`
    """
    digest = hashlib.sha256(f"{task_id}|{run}".encode()).hexdigest()
    seed = int(digest[:8], 16) % 2**31
    return seed


def build_messages(prompt: str) -> List[Dict[str, str]]:
    """
    Build the EvalPlus chat prompt for a task.

    The last message is the start of the answer, so the model continues
    directly with the code, as in EvalPlus.

    :param prompt: task prompt, i.e., a signature with a docstring
    :return: messages for the Ollama chat API
    """
    messages = [
        {
            "role": "user",
            "content": f"{INSTRUCTION_PREFIX}\n```\n{prompt.strip()}\n```\n",
        },
        {"role": "assistant", "content": f"{RESPONSE_PREFIX}\n```python\n"},
    ]
    return messages


def build_repair_messages(
    messages: List[Dict[str, str]], code: str, feedback: str
) -> List[Dict[str, str]]:
    """
    Add a failed attempt and its feedback to the conversation.

    :param messages: messages of the last attempt, ending with the start of
        the answer
    :param code: code of the last attempt
    :param feedback: what went wrong, from `run_visible_tests()`
    :return: messages for the next attempt, ending with the start of the
        answer again
    """
    # Close the last answer with the code it produced.
    new_messages = messages[:-1] + [
        {
            "role": "assistant",
            "content": messages[-1]["content"] + code + "\n```",
        },
        {
            "role": "user",
            "content": (
                "The code fails on an example from the problem:\n"
                f"```\n{feedback}\n```\n"
                "Please fix it and provide the full corrected script in a "
                "markdown code block."
            ),
        },
        {
            "role": "assistant",
            "content": "Below is the corrected Python script:\n```python\n",
        },
    ]
    return new_messages


def get_model_info(
    model: str, *, url: str = DEFAULT_OLLAMA_URL
) -> Dict[str, Any]:
    """
    Get the version and the stop texts of a model served by Ollama.

    :param model: e.g., `deepseek-coder:1.3b`
    :param url: Ollama server
    :return: dict with `model`, `digest`, `ollama_version`, and `stop` (the
        model's own stop texts)
    """
    tags = requests.get(f"{url}/api/tags", timeout=60).json()["models"]
    digests = {tag["name"]: tag["digest"] for tag in tags}
    hdbg.dassert_in(model, digests, "Pull the model with `ollama pull` first")
    show = requests.post(
        f"{url}/api/show", json={"model": model}, timeout=60
    ).json()
    # The parameters are lines like `stop "<|im_end|>"`.
    stop = []
    for line in show.get("parameters", "").splitlines():
        key, _, value = line.partition(" ")
        if key == "stop":
            stop.append(json.loads(value.strip()))
    version = requests.get(f"{url}/api/version", timeout=60).json()["version"]
    info = {
        "model": model,
        "digest": digests[model],
        "ollama_version": version,
        "stop": stop,
    }
    return info


# Retry for up to 10 minutes, e.g., while the Ollama app restarts. A retry
# gives the same answer, since the seed fixes the output.
@hretry.sync_retry(
    (requests.RequestException,), num_attempts=10, retry_delay_in_sec=60
)
def call_ollama(
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    seed: int,
    stop: List[str],
    url: str = DEFAULT_OLLAMA_URL,
) -> Dict[str, Any]:
    """
    Ask a model served by Ollama to continue a conversation.

    :param model: e.g., `deepseek-coder:1.3b`
    :param messages: output of `build_messages()`
    :param temperature: sampling temperature, `0` for greedy decoding
    :param seed: sampling seed
    :param stop: texts that end the generation
    :param url: Ollama server
    :return: dict with `text`, `tokens_in`, `tokens_out`, `seconds`
        (generation time on the server), and `done_reason` (`stop`,
        `length` if the answer hit the token limit, or `repeat_limit` if
        Ollama stopped an answer stuck repeating a token)
    """
    _LOG.debug(hprint.to_str("model temperature seed"))
    options = dict(GENERATION_OPTIONS, temperature=temperature, seed=seed)
    options["stop"] = stop
    response = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "options": options,
            "stream": False,
        },
        timeout=600,
    )
    if response.status_code == 500 and "token repeat limit" in response.text:
        # Ollama stops an answer that repeats the same token too many times,
        # e.g., "db0000000...", and returns an error. The answer has failed
        # and a retry repeats it with the same seed, so return no code.
        _LOG.warning("Ollama stopped a repeating answer: %s", response.text)
        reply = {
            "text": "",
            "tokens_in": 0,
            "tokens_out": 0,
            "seconds": 0.0,
            "done_reason": "repeat_limit",
        }
    else:
        response.raise_for_status()
        body = response.json()
        reply = {
            # EvalPlus replaces tabs with 4 spaces.
            "text": body["message"]["content"].replace("\t", "    "),
            "tokens_in": body.get("prompt_eval_count", 0),
            "tokens_out": body.get("eval_count", 0),
            "seconds": body.get("total_duration", 0) / 1e9,
            "done_reason": body.get("done_reason", ""),
        }
    return reply


def extract_code(text: str, entry_point: str) -> str:
    """
    Extract the solution from a model answer, as EvalPlus does.

    :param text: answer of the model
    :param entry_point: name of the function to solve, e.g., `truncate_number`
    :return: the function and what it depends on, or `""` if no code is found
    """
    code = evalplus.sanitize.sanitize(text, entrypoint=entry_point)
    return code


# #############################################################################
# Examples shown in the prompt
# #############################################################################


def _is_call_to(source: str, entry_point: str) -> bool:
    """
    Check whether an expression is a call to the function under test.

    :param source: e.g., `truncate_number(3.5)`
    :param entry_point: e.g., `truncate_number`
    :return: whether `source` parses as `entry_point(...)`
    """
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError:
        return False
    node = tree.body
    is_call = (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == entry_point
    )
    return is_call


def _get_literal(text: str) -> Optional[str]:
    """
    Normalize a value written in a prompt, if it is a Python literal.

    :param text: e.g., `0.5`, `'abc'`, `[1, 2]`, or `True.`
    :return: the literal as Python source, or `None` if it is not one
    """
    text = text.strip().rstrip(".,;")
    try:
        ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return text


def get_visible_tests(problem: Dict[str, Any], benchmark: str) -> List[str]:
    """
    Get the examples shown in a task prompt, as `assert` statements.

    - MBPP+ prompts show 1 `assert` line
    - HumanEval+ prompts show doctest examples, or lines like `f(x) == y` or
      `f(x) ➞ y`; only a direct call with a literal result is kept, so an
      ambiguous example gives no feedback instead of wrong feedback
    - A float result is compared up to the precision it is written with

    :param problem: task, from `load_problems()`
    :param benchmark: `humaneval` or `mbpp`
    :return: e.g., `["assert truncate_number(3.5) == 0.5"]`
    """
    prompt = problem["prompt"]
    entry_point = problem["entry_point"]
    if benchmark == "mbpp":
        tests = [
            line.strip()
            for line in prompt.splitlines()
            if line.strip().startswith("assert ")
        ]
        return tests
    # Collect `(call, result)` pairs from doctest examples. The parser sees the
    # whole function, so the closing quotes of the docstring can end up in the
    # result: drop them.
    pairs = []
    for example in doctest.DocTestParser().get_examples(prompt):
        want = "\n".join(
            line
            for line in example.want.splitlines()
            if line.strip() not in ('"""', "'''")
        )
        pairs.append((example.source.strip(), want.strip()))
    # Collect pairs from lines like `f(x) == y`, `f(x) ➞ y`, or `f(x) => y`.
    pattern = re.compile(
        r"^\s*(?P<call>" + re.escape(entry_point) + r"\(.*\))\s*"
        r"(?:==>|==|➞|=>|->|returns|should return)\s*(?P<want>.+?)\s*$"
    )
    for line in prompt.splitlines():
        match = pattern.match(line)
        if match:
            pairs.append((match.group("call"), match.group("want")))
    # Keep the pairs that are clearly a call with a literal result.
    tests = []
    for call, want in pairs:
        literal = _get_literal(want)
        if literal is None and "\n" in want:
            # A line of prose right after the result is part of `want`.
            literal = _get_literal(want.splitlines()[0])
        if not _is_call_to(call, entry_point) or literal is None:
            continue
        value = ast.literal_eval(literal)
        if isinstance(value, float):
            # E.g., `0.333` matches anything within 0.0005.
            decimals = len(literal.split(".")[-1]) if "." in literal else 0
            tolerance = 0.5 * 10 ** (-decimals)
            test = f"assert abs({call} - {literal}) <= {tolerance}"
        else:
            test = f"assert {call} == {literal}"
        if test not in tests:
            tests.append(test)
    return tests


# Script that runs the visible tests of a solution in a separate process. It
# reads the code and the tests as JSON from stdin, and writes 1 JSON result to
# `result.json`, since the solution can print anything to stdout.
_VISIBLE_TESTS_RUNNER = r"""
import ast
import json
import math
import sys

import evalplus.eval.utils

data = json.load(sys.stdin)
# Open the result file first, since the guard disables `open()`.
result_file = open("result.json", "w")
# Disable functions that could damage the machine (e.g., `os.system`).
evalplus.eval.utils.reliability_guard(maximum_memory_bytes=4 * 2**30)
namespace = {}
result = {"passed": True, "feedback": ""}
try:
    exec(data["code"], namespace)
except BaseException as e:
    feedback = f"Loading the code raised {type(e).__name__}: {e}"
    result = {"passed": False, "feedback": feedback}
else:
    # A few MBPP+ examples call `math.isclose()` or `sys.getsizeof()`, so give
    # them these modules even if the solution does not import them.
    namespace.setdefault("math", math)
    namespace.setdefault("sys", sys)
    for test in data["tests"]:
        try:
            exec(test, namespace)
        except AssertionError:
            # Show what the left side returned, when the test compares a value.
            node = ast.parse(test).body[0].test
            shown = ""
            if isinstance(node, ast.Compare):
                try:
                    shown = repr(eval(ast.unparse(node.left), namespace))
                except BaseException:
                    shown = ""
            feedback = f"{test}\nfailed"
            if shown:
                feedback += f": the left side is {shown[:300]}"
            result = {"passed": False, "feedback": feedback}
            break
        except BaseException as e:
            error = f"{type(e).__name__}: {str(e)[:300]}"
            result = {"passed": False, "feedback": f"{test}\nraised {error}"}
            break
json.dump(result, result_file)
result_file.close()
"""


def run_visible_tests(
    code: str, tests: List[str], *, timeout: float = 10.0
) -> Tuple[bool, str]:
    """
    Run a solution on the examples shown in its prompt.

    The code runs in a separate process with a time limit, with the
    dangerous functions disabled as in EvalPlus.

    :param code: solution, from `extract_code()`
    :param tests: output of `get_visible_tests()`
    :param timeout: seconds before the run is stopped
    :return: whether all the tests pass, and the feedback for the model, e.g.,
        `assert f(1) == 2\nfailed: the left side is 3`
    """
    if not code.strip():
        return False, "The answer contains no code."
    if not tests:
        # Nothing to check against: accept the solution as it is.
        return True, ""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Drop what the solution prints, so a print loop cannot fill memory.
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _VISIBLE_TESTS_RUNNER],
                input=json.dumps({"code": code, "tests": tests}),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                cwd=tmp_dir,
            )
        except subprocess.TimeoutExpired:
            return False, f"Running the examples took more than {timeout:.0f} s."
        result_path = os.path.join(tmp_dir, "result.json")
        if completed.returncode != 0 or not os.path.exists(result_path):
            # The process died, e.g., because the code ran out of memory.
            error = completed.stderr[-300:]
            feedback = f"Running the code stopped the program: {error}"
            return False, feedback
        with open(result_path) as file:
            result = json.load(file)
    return result["passed"], result["feedback"]


# #############################################################################
# Generating the outcome tensor
# #############################################################################


def sort_task_ids(task_ids: List[str]) -> List[str]:
    """
    Sort task ids by their number.

    :param task_ids: e.g., `["HumanEval/10", "HumanEval/2"]`
    :return: e.g., `["HumanEval/2", "HumanEval/10"]`
    """
    sorted_ids = sorted(task_ids, key=lambda task_id: int(task_id.split("/")[1]))
    return sorted_ids


def get_runs_file(
    dst_dir: str, benchmark: str, model: str, temperature: float
) -> str:
    """
    Get the file that stores the answers of 1 model at 1 temperature.

    :param dst_dir: e.g., `data/raw/coding_runs`
    :param benchmark: `humaneval` or `mbpp`
    :param model: e.g., `deepseek-coder:1.3b`
    :param temperature: e.g., `0.8`
    :return: e.g.,
        `data/raw/coding_runs/humaneval/deepseek-coder_1.3b_t0.8.jsonl`
    """
    model_slug = model.replace(":", "_").replace("/", "_")
    path = os.path.join(
        dst_dir, benchmark, f"{model_slug}_t{temperature:g}.jsonl"
    )
    return path


def read_runs_file(path: str) -> List[Dict[str, Any]]:
    """
    Read the cells saved by `run_coding_agents.py`.

    :param path: output of `get_runs_file()`
    :return: 1 dict per generated cell, or `[]` if the file does not exist
    """
    records = []
    if os.path.exists(path):
        with open(path) as file:
            records = [json.loads(line) for line in file if line.strip()]
    return records


def generate_cell(
    problem: Dict[str, Any],
    benchmark: str,
    model: str,
    run: int,
    *,
    temperature: float,
    max_repairs: int,
    stop: List[str],
    url: str = DEFAULT_OLLAMA_URL,
) -> Dict[str, Any]:
    """
    Generate 1 cell of the outcome tensor: 1 task, 1 run, both scaffolds.

    The first attempt is the one-shot answer. If it fails an example shown in
    the prompt, the model sees the error and tries again, up to
    `max_repairs` times: the last attempt is the repair-scaffold answer. So
    the 2 scaffolds share the same first attempt.

    :param problem: task, from `load_problems()`
    :param benchmark: `humaneval` or `mbpp`
    :param model: e.g., `deepseek-coder:1.3b`
    :param run: run index, which sets the seed with the task id
    :param temperature: sampling temperature, `0` for greedy decoding
    :param max_repairs: attempts allowed after the first one
    :param stop: texts that end a generation
    :param url: Ollama server
    :return: dict with `task_id`, `run`, `seed`, `temperature`,
        `n_visible_tests`, and `attempts` (each with `text`, `code`,
        `visible_pass`, `feedback`, `tokens_in`, `tokens_out`, `seconds`,
        `done_reason`)
    """
    task_id = problem["task_id"]
    seed = get_cell_seed(task_id, run)
    tests = get_visible_tests(problem, benchmark)
    messages = build_messages(problem["prompt"])
    attempts = []
    for _ in range(max_repairs + 1):
        reply = call_ollama(
            model,
            messages,
            temperature=temperature,
            seed=seed,
            stop=stop,
            url=url,
        )
        code = extract_code(reply["text"], problem["entry_point"])
        visible_pass, feedback = run_visible_tests(code, tests)
        attempts.append(
            {
                "text": reply["text"],
                "code": code,
                "visible_pass": visible_pass,
                "feedback": feedback,
                "tokens_in": reply["tokens_in"],
                "tokens_out": reply["tokens_out"],
                "seconds": reply["seconds"],
                "done_reason": reply["done_reason"],
            }
        )
        if visible_pass:
            break
        # Give the model the error and ask for a fix.
        messages = build_repair_messages(messages, code, feedback)
    record = {
        "task_id": task_id,
        "run": run,
        "seed": seed,
        "temperature": temperature,
        "n_visible_tests": len(tests),
        "attempts": attempts,
    }
    return record


# #############################################################################
# Scoring with the EvalPlus tests
# #############################################################################


def score_solution(
    benchmark: str,
    problem: Dict[str, Any],
    solution: str,
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Score a solution on the base and the plus tests, as EvalPlus does.

    :param benchmark: `humaneval` or `mbpp`
    :param problem: task, from `load_problems()`
    :param solution: code of the solution
    :param expected: expected outputs of the task, from `get_expected_outputs()`
    :return: dict with `base_pass` and `plus_pass` (1 if the base tests, and
        both the base and the plus tests, pass), and `failed_split` and
        `failed_input` (where the first failure happened, or `""` and `-1`)
    """
    result = evalplus.evaluate.check_correctness(
        benchmark,
        0,
        problem,
        solution,
        expected,
        base_only=False,
        fast_check=True,
    )
    base_status, base_details = result["base"]
    plus_status, plus_details = result["plus"]
    base_pass = int(base_status == evalplus.eval.PASS)
    plus_pass = int(base_pass == 1 and plus_status == evalplus.eval.PASS)
    # With `fast_check`, the last checked input is the one that failed.
    failed_split, failed_input = "", -1
    if not base_pass:
        failed_split, failed_input = "base", len(base_details) - 1
    elif not plus_pass:
        failed_split, failed_input = "plus", len(plus_details) - 1
    score = {
        "base_pass": base_pass,
        "plus_pass": plus_pass,
        "failed_split": failed_split,
        "failed_input": failed_input,
    }
    return score


# Codes reported by `_diagnose_in_process()`.
_DIAGNOSIS_LABELS = {
    0: "wrong_answer",
    1: "runtime_error",
    2: "timeout",
    3: "missing_function",
    4: "load_error",
}


def _diagnose_in_process(
    code: str,
    entry_point: str,
    inputs: List[Any],
    time_limit: float,
    label: "multiprocessing.sharedctypes.Synchronized",
) -> None:
    """
    Run a solution on 1 input and record how it fails.

    This runs in a child process, with the dangerous functions disabled.

    :param code: solution
    :param entry_point: name of the function
    :param inputs: arguments of the failing call
    :param time_limit: seconds allowed for the call
    :param label: shared value that receives a key of `_DIAGNOSIS_LABELS`
    """
    evalplus.eval.utils.reliability_guard(
        maximum_memory_bytes=evalplus.eval.query_maximum_memory_bytes()
    )
    namespace: Dict[str, Any] = {}
    try:
        with evalplus.eval.utils.swallow_io():
            exec(code, namespace)
    except BaseException:
        label.value = 4
        return
    if entry_point not in namespace:
        label.value = 3
        return
    try:
        with evalplus.eval.utils.time_limit(time_limit):
            with evalplus.eval.utils.swallow_io():
                namespace[entry_point](*inputs)
    except evalplus.eval.utils.TimeoutException:
        label.value = 2
        return
    except BaseException:
        label.value = 1
        return
    # The call returned a value, which EvalPlus found wrong.
    label.value = 0


def diagnose_failure(
    problem: Dict[str, Any],
    solution: str,
    expected: Dict[str, Any],
    score: Dict[str, Any],
) -> str:
    """
    Label why a solution failed the EvalPlus tests.

    EvalPlus reports only pass or fail, so the first failing input is run
    again to tell the failure modes apart.

    :param problem: task, from `load_problems()`
    :param solution: code of the solution
    :param expected: expected outputs of the task
    :param score: output of `score_solution()`
    :return: `""` if the solution passed, or one of `no_code`,
        `syntax_error`, `load_error`, `missing_function`, `timeout`,
        `runtime_error`, `wrong_answer`
    """
    if score["plus_pass"]:
        return ""
    # Checks that do not run the code.
    if not solution.strip():
        return "no_code"
    try:
        ast.parse(solution)
    except SyntaxError:
        return "syntax_error"
    # Run the first failing input again, with the time limit of EvalPlus.
    split = score["failed_split"]
    index = max(score["failed_input"], 0)
    inputs = problem[f"{split}_input"][index]
    reference_time = expected[f"{split}_time"][index]
    time_limit = max(
        evalplus.config.DEFAULT_MIN_TIME_LIMIT,
        evalplus.config.DEFAULT_GT_TIME_LIMIT_FACTOR * reference_time,
    )
    label = multiprocessing.Value("i", 1)
    process = multiprocessing.Process(
        target=_diagnose_in_process,
        args=(solution, problem["entry_point"], inputs, time_limit, label),
    )
    process.start()
    process.join(timeout=time_limit + 5)
    if process.is_alive():
        # The process hung even past the time limit.
        process.kill()
        process.join()
        return "timeout"
    return _DIAGNOSIS_LABELS[label.value]


# #############################################################################
# Scoring many solutions
# #############################################################################


# Tasks and expected outputs shared with the scoring worker processes.
_WORKER_DATA: Dict[str, Any] = {}


def _load_worker_data(benchmark: str) -> None:
    """
    Load the tasks and the expected outputs before the workers start.

    The workers are forked, so they share this memory instead of loading
    their own copy: the expected outputs of MBPP+ take a few GB, and 1 copy
    per worker does not fit in the Docker VM.

    :param benchmark: `humaneval` or `mbpp`
    """
    if _WORKER_DATA.get("benchmark") != benchmark:
        problems = load_problems(benchmark)
        _WORKER_DATA["benchmark"] = benchmark
        _WORKER_DATA["problems"] = problems
        _WORKER_DATA["expected"] = get_expected_outputs(benchmark, problems)
    # Keep the garbage collector away from these objects, otherwise each
    # worker copies the memory pages it touches.
    gc.freeze()


def _score_in_worker(task_id: str, code: str, n_checks: int) -> Dict[str, Any]:
    """
    Score 1 solution `n_checks` times, in a scoring worker process.

    :param task_id: task of the solution
    :param code: solution
    :param n_checks: times to run the tests
    :return: dict with `base_pass` and `plus_pass` of the first check,
        `plus_pass_count` over all the checks, and `failure` (see
        `diagnose_failure()`)
    """
    benchmark = _WORKER_DATA["benchmark"]
    problem = _WORKER_DATA["problems"][task_id]
    expected = _WORKER_DATA["expected"][task_id]
    scores = [
        score_solution(benchmark, problem, code, expected)
        for _ in range(n_checks)
    ]
    result = {
        "base_pass": scores[0]["base_pass"],
        "plus_pass": scores[0]["plus_pass"],
        "plus_pass_count": sum(score["plus_pass"] for score in scores),
        "failure": diagnose_failure(problem, code, expected, scores[0]),
    }
    return result


def score_solutions(
    benchmark: str,
    solutions: List[Tuple[str, str]],
    *,
    n_checks: int = 1,
    num_workers: int = 0,
) -> List[Dict[str, Any]]:
    """
    Score many solutions in parallel, as EvalPlus does.

    Identical solutions to the same task are scored once, e.g., a repair
    answer that is the same as the one-shot answer.

    :param benchmark: `humaneval` or `mbpp`
    :param solutions: `(task_id, code)` pairs
    :param n_checks: times to run the tests of each solution, to find flaky
        tests
    :param num_workers: processes to use, `0` for half of the CPUs (the
        EvalPlus default)
    :return: 1 dict per pair, in the same order (see `_score_in_worker()`)
    """
    hdbg.dassert_lte(1, n_checks)
    _load_worker_data(benchmark)
    if num_workers <= 0:
        num_workers = max(1, (os.cpu_count() or 2) // 2)
    unique = sorted(set(solutions))
    _LOG.info(
        "Scoring '%s' distinct solutions of '%s' with '%s' workers",
        len(unique),
        len(solutions),
        num_workers,
    )
    # Fork the workers explicitly, so they inherit the loaded data.
    context = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        num_workers, mp_context=context
    ) as executor:
        futures = {
            pair: executor.submit(_score_in_worker, pair[0], pair[1], n_checks)
            for pair in unique
        }
        # Wait for all the solutions, with a progress bar.
        for _ in tqdm.tqdm(
            concurrent.futures.as_completed(futures.values()),
            total=len(futures),
            desc="Scoring",
            mininterval=30,
            ascii=True,
        ):
            pass
    results = [futures[pair].result() for pair in solutions]
    return results


def get_scaffold_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turn 1 cell from `generate_cell()` into 1 row per scaffold.

    :param record: 1 cell
    :return: 2 dicts, for `one_shot` (first attempt) and `repair` (last
        attempt), with `task_id`, `run`, `scaffold`, `code`, `n_attempts`,
        `visible_pass`, `truncated` (the answer hit the token limit),
        `tokens_in`, `tokens_out`, and `gen_seconds`
    """
    attempts = record["attempts"]
    rows = []
    for scaffold, used in (("one_shot", attempts[:1]), ("repair", attempts)):
        rows.append(
            {
                "task_id": record["task_id"],
                "run": record["run"],
                "scaffold": scaffold,
                "code": used[-1]["code"],
                "n_attempts": len(used),
                "n_visible_tests": record["n_visible_tests"],
                "visible_pass": int(used[-1]["visible_pass"]),
                # Cells made before `done_reason` was saved have no value.
                "truncated": int(used[-1].get("done_reason") == "length"),
                "tokens_in": sum(a["tokens_in"] for a in used),
                "tokens_out": sum(a["tokens_out"] for a in used),
                "gen_seconds": sum(a["seconds"] for a in used),
            }
        )
    return rows


# #############################################################################
# Analysis of the outcome tensor
# #############################################################################


def load_scores(scores_dir: str, *, pattern: str = "*.csv.gz") -> pd.DataFrame:
    """
    Load the scores saved by `score_coding_runs.py`.

    :param scores_dir: e.g., `data/coding_runs`
    :param pattern: files to load, e.g., `evalplus_*.csv.gz` for the public
        samples only
    :return: 1 row per scored solution, with `source` (`local` or
        `evalplus`) and `benchmark` taken from the file name
    """
    frames = []
    for path in sorted(glob.glob(os.path.join(scores_dir, pattern))):
        source, benchmark = os.path.basename(path).split("_")[:2]
        # Keep an empty `failure` as `""` instead of NaN.
        frame = pd.read_csv(path, keep_default_na=False)
        frame.insert(0, "source", source)
        frame.insert(1, "benchmark", benchmark)
        frames.append(frame)
    hdbg.dassert_lt(0, len(frames), "No scores in '%s'", scores_dir)
    scores = pd.concat(frames, ignore_index=True)
    return scores


def get_outcome_matrix(
    scores: pd.DataFrame,
    *,
    source: str,
    benchmark: str,
    model: str,
    scaffold: str,
    temperature: float,
) -> pd.DataFrame:
    """
    Get the tasks x runs matrix of 1 agent.

    :param scores: output of `load_scores()`
    :param source: `local` (our runs) or `evalplus` (public samples)
    :param benchmark: `humaneval` or `mbpp`
    :param model: e.g., `opencoder:8b`
    :param scaffold: `one_shot` or `repair`
    :param temperature: e.g., `0.8`
    :return: tasks x runs matrix of `plus_pass` (0/1), with sorted task ids
    """
    mask = (
        (scores["source"] == source)
        & (scores["benchmark"] == benchmark)
        & (scores["model"] == model)
        & (scores["scaffold"] == scaffold)
        & (scores["temperature"] == temperature)
    )
    rows = scores[mask]
    hdbg.dassert_lt(0, len(rows), "No scores for '%s'", model)
    matrix = rows.pivot(index="task_id", columns="run", values="plus_pass")
    hdbg.dassert(not matrix.isna().any().any(), "Some cells have no score")
    matrix = matrix.loc[sort_task_ids(list(matrix.index))].astype(int)
    return matrix


def compute_variance_components(outcomes: pd.DataFrame) -> pd.Series:
    r"""
    Split the variance of a score into its task and run parts.

    With per-task success rates $p_i$, the score over $N$ tasks and $R$ runs
    has variance (the decomposition of the spec)
    $\sigma^2_{task} / N + \sigma^2_{run} / (N R)$, where:
    - $\sigma^2_{run} = E[p_i (1 - p_i)]$ is the run-to-run variance of 1 task
    - $\sigma^2_{task} = \mathrm{Var}(p_i)$ is how much the tasks differ
    Both are estimated without bias, which needs at least 2 runs.

    :param outcomes: tasks x runs matrix of 0/1
    :return: `score`, `sigma2_task`, `sigma2_run`, `run_share` (share of
        the variance of 1 run that is run-to-run luck), and `se_1_run` and
        `se_all_runs` (standard error of the score with 1 run and with all
        the runs)
    """
    values = outcomes.to_numpy(dtype=float)
    n_tasks, n_runs = values.shape
    hdbg.dassert_lte(2, n_runs, "Need at least 2 runs per task")
    # The sample variance over the runs of a task estimates p (1 - p).
    sigma2_run = float(values.var(axis=1, ddof=1).mean())
    # The rates measured over R runs vary by Var(p) + E[p (1 - p)] / R.
    p_hat = values.mean(axis=1)
    sigma2_task = max(float(p_hat.var(ddof=1)) - sigma2_run / n_runs, 0.0)
    total = sigma2_task + sigma2_run
    components = pd.Series(
        {
            "score": float(values.mean()),
            "sigma2_task": sigma2_task,
            "sigma2_run": sigma2_run,
            "run_share": sigma2_run / total if total > 0 else 0.0,
            "se_1_run": float(np.sqrt(total / n_tasks)),
            "se_all_runs": float(
                np.sqrt((sigma2_task + sigma2_run / n_runs) / n_tasks)
            ),
        }
    )
    return components


def compare_agents_over_runs(
    outcomes_a: pd.DataFrame,
    outcomes_b: pd.DataFrame,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Compare 2 agents on the same tasks, with the luck of the tasks and of
    the runs.

    The success rate of each task, averaged over its runs, already carries
    the run-to-run noise. So resampling the tasks, with the 2 agents' rates
    kept together, covers both sources (the "paired bootstrap over tasks and
    seeds" of the spec): resampling the runs as well would count the run
    noise twice. The p-value is a paired t-test on the same per-task rates.

    :param outcomes_a: tasks x runs matrix of 0/1 of agent A
    :param outcomes_b: tasks x runs matrix of 0/1 of agent B, same tasks
    :param n_boot: number of bootstrap samples
    :param alpha: the interval covers `1 - alpha`
    :param seed: random seed
    :return: dict with `gap` (score of A minus B), `ci_low`, `ci_high`,
        `p_value`, and `gap_run_0` (the gap if each agent had only its first
        run)
    """
    hdbg.dassert(
        outcomes_a.index.equals(outcomes_b.index),
        "The 2 agents must be evaluated on the same tasks",
    )
    a = outcomes_a.to_numpy(dtype=float)
    b = outcomes_b.to_numpy(dtype=float)
    n_tasks = a.shape[0]
    # Resample the per-task gaps, with multinomial counts as in
    # `skill_vs_luck_utils.compute_score_cis()`.
    rates_a, rates_b = a.mean(axis=1), b.mean(axis=1)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n_tasks, np.full(n_tasks, 1 / n_tasks), size=n_boot
    )
    gaps = weights @ (rates_a - rates_b) / n_tasks
    p_value = 1.0
    if np.any(rates_a != rates_b):
        p_value = float(scipy.stats.ttest_rel(rates_a, rates_b).pvalue)
    comparison = {
        "gap": float(rates_a.mean() - rates_b.mean()),
        "ci_low": float(np.quantile(gaps, alpha / 2)),
        "ci_high": float(np.quantile(gaps, 1 - alpha / 2)),
        "p_value": p_value,
        "gap_run_0": float(a[:, 0].mean() - b[:, 0].mean()),
    }
    return comparison


def compute_ranking_stability(
    matrices: Dict[str, pd.DataFrame],
    *,
    n_draws: Optional[int] = 10000,
    seed: int = 0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Check how often a leaderboard from 1 run per agent matches the
    leaderboard from all the runs.

    Each draw picks 1 run at random for each agent, as a leaderboard that
    runs every agent once would. With few agents and runs, every combination
    of 1 run per agent can be taken once instead, which gives the exact
    shares: random draws could split 2 equal shares by chance.

    :param matrices: agent label -> tasks x runs matrix of 0/1, same tasks
    :param n_draws: number of random leaderboards, or `None` to take every
        combination of runs once
    :param seed: random seed, unused with `n_draws=None`
    :return:
        - summary with `p_same_order` (share of draws that reverse no pair
          whose means over the runs differ) and `mean_kendall_tau`
        - 1 row per pair of agents with `gap` (mean over runs) and `p_flip`
          (share of draws where the pair swaps places, ties count half)
    """
    labels = list(matrices)
    # Score of every agent in every run, as a agents x runs array.
    run_scores = np.array(
        [matrices[label].mean(axis=0).to_numpy() for label in labels]
    )
    mean_scores = run_scores.mean(axis=1)
    if n_draws is None:
        # Each combination of 1 run per agent is as likely as any other.
        n_runs = run_scores.shape[1]
        hdbg.dassert_lte(n_runs ** len(labels), 10**6, "Too many combinations")
        picks = np.array(
            list(itertools.product(range(n_runs), repeat=len(labels)))
        )
    else:
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, run_scores.shape[1], size=(n_draws, len(labels)))
    drawn = run_scores[np.arange(len(labels)), picks]
    # A draw keeps the order if it reverses no pair with different means: a
    # tie in the draw is not a reversal, and pairs with equal means have no
    # order to keep.
    strict_pairs = [
        (i, j)
        for i, j in itertools.permutations(range(len(labels)), 2)
        if mean_scores[i] > mean_scores[j]
    ]
    kept = np.ones(len(picks), dtype=bool)
    for i, j in strict_pairs:
        kept &= drawn[:, i] >= drawn[:, j]
    same_order = float(kept.mean())
    taus = [scipy.stats.kendalltau(row, mean_scores).statistic for row in drawn]
    summary = pd.Series(
        {
            "p_same_order": float(same_order),
            "mean_kendall_tau": float(np.nanmean(taus)),
        }
    )
    # Count how often each pair swaps places.
    rows = []
    for i, j in itertools.combinations(range(len(labels)), 2):
        upper, lower = (i, j) if mean_scores[i] >= mean_scores[j] else (j, i)
        diff = drawn[:, upper] - drawn[:, lower]
        p_flip = float(np.mean(diff < 0) + 0.5 * np.mean(diff == 0))
        rows.append(
            {
                "upper": labels[upper],
                "lower": labels[lower],
                "gap": float(mean_scores[upper] - mean_scores[lower]),
                "p_flip": p_flip,
            }
        )
    pairs = pd.DataFrame(rows).sort_values("gap").reset_index(drop=True)
    return summary, pairs


def compute_pair_variance_components(
    outcomes_a: pd.DataFrame, outcomes_b: pd.DataFrame
) -> Dict[str, float]:
    """
    Split the variance of the per-task gap between 2 agents into its task
    and run parts.

    :param outcomes_a: tasks x runs matrix of 0/1 of agent A
    :param outcomes_b: tasks x runs matrix of 0/1 of agent B, same tasks
    :return: dict with `sigma2_task_diff` (how much the true gap varies
        across tasks) and `sigma2_run_diff` (run-to-run variance of the gap
        on 1 task, with 1 run of each agent)
    """
    a = outcomes_a.to_numpy(dtype=float)
    b = outcomes_b.to_numpy(dtype=float)
    hdbg.dassert_eq(a.shape, b.shape, "The 2 agents need the same runs")
    n_runs = a.shape[1]
    sigma2_run_diff = float(
        a.var(axis=1, ddof=1).mean() + b.var(axis=1, ddof=1).mean()
    )
    diff = a.mean(axis=1) - b.mean(axis=1)
    sigma2_task_diff = max(
        float(diff.var(ddof=1)) - sigma2_run_diff / n_runs, 0.0
    )
    components = {
        "sigma2_task_diff": sigma2_task_diff,
        "sigma2_run_diff": sigma2_run_diff,
    }
    return components


def compute_min_tasks_with_runs(
    sigma2_task_diff: float,
    sigma2_run_diff: float,
    gap: float,
    n_runs: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    r"""
    Compute the tasks needed to detect a gap when every task is run
    `n_runs` times.

    $N = (z_{1 - \alpha / 2} + z_{power})^2
    (\sigma^2_{task,d} + \sigma^2_{run,d} / R) / \Delta^2$: more runs only
    shrink the run part, so for a fixed budget of `N R` evaluations, more
    tasks always beat more runs.

    :param sigma2_task_diff: output of `compute_pair_variance_components()`
    :param sigma2_run_diff: output of `compute_pair_variance_components()`
    :param gap: gap to detect, e.g., `0.02` for 2 points
    :param n_runs: runs per task
    :param alpha: significance level
    :param power: chance of detecting a real gap
    :return: number of tasks
    """
    hdbg.dassert_lt(0, gap, "The gap must be positive")
    hdbg.dassert_lte(1, n_runs)
    z_alpha = scipy.stats.norm.ppf(1 - alpha / 2)
    z_power = scipy.stats.norm.ppf(power)
    variance = sigma2_task_diff + sigma2_run_diff / n_runs
    n_tasks = (z_alpha + z_power) ** 2 * variance / gap**2
    return float(n_tasks)


def compute_flaky_share(scores: pd.DataFrame) -> float:
    """
    Compute the share of solutions whose tests gave different results when
    run again.

    :param scores: output of `load_scores()`, scored with `n_checks >= 2`
    :return: share of rows with `0 < plus_pass_count < n_checks`
    """
    checked = scores[scores["n_checks"] >= 2]
    hdbg.dassert_lt(0, len(checked), "No solution was scored twice")
    flaky = (checked["plus_pass_count"] > 0) & (
        checked["plus_pass_count"] < checked["n_checks"]
    )
    return float(flaky.mean())


# #############################################################################
# Tables and plots for the notebook
# #############################################################################


def get_model_label(model: str) -> str:
    """
    Get a short name for a model, for tables and plots.

    The name keeps the size of a local model, since 2 local models can share
    a name (DeepSeek-Coder 1.3B and 6.7B).

    :param model: e.g., `codeqwen:7b-chat-v1.5-q4_K_M` (local) or
        `code-llama-7b` (public samples)
    :return: e.g., `codeqwen 7b`
    """
    name, _, tag = model.partition(":")
    if tag:
        name = f"{name} {tag.split('-')[0]}"
    return name


def get_agent_label(model: str, scaffold: str) -> str:
    """
    Get a short name for an agent, for tables and plots.

    :param model: e.g., `codeqwen:7b-chat-v1.5-q4_K_M`
    :param scaffold: `one_shot` or `repair`
    :return: e.g., `codeqwen 7b / repair`
    """
    label = f"{get_model_label(model)} / {scaffold}"
    return label


def get_agent_matrices(
    scores: pd.DataFrame, *, source: str, benchmark: str, temperature: float
) -> Dict[str, pd.DataFrame]:
    """
    Get the tasks x runs matrix of every agent of a source and benchmark.

    :param scores: output of `load_scores()`
    :param source: `local` or `evalplus`
    :param benchmark: `humaneval` or `mbpp`
    :param temperature: e.g., `0.8`
    :return: agent label -> tasks x runs matrix of 0/1
    """
    mask = (
        (scores["source"] == source)
        & (scores["benchmark"] == benchmark)
        & (scores["temperature"] == temperature)
    )
    rows = scores.loc[mask, ["model", "scaffold"]].drop_duplicates()
    agents = list(rows.itertuples(index=False))
    labels = [get_agent_label(model, scaffold) for model, scaffold in agents]
    hdbg.dassert_no_duplicates(labels, "2 agents have the same label")
    matrices = {
        label: get_outcome_matrix(
            scores,
            source=source,
            benchmark=benchmark,
            model=model,
            scaffold=scaffold,
            temperature=temperature,
        )
        for label, (model, scaffold) in zip(labels, agents)
    }
    return matrices


def compute_score_table(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the scores of every agent.

    :param scores: output of `load_scores()`
    :return: 1 row per source, benchmark, model, temperature, and scaffold,
        with `n_tasks`, `n_runs`, `base_pass`, `plus_pass`, `mean_attempts`,
        and `mean_seconds` (generation time per cell)
    """
    keys = ["source", "benchmark", "model", "temperature", "scaffold"]
    table = scores.groupby(keys).agg(
        n_tasks=("task_id", "nunique"),
        n_runs=("run", "nunique"),
        base_pass=("base_pass", "mean"),
        plus_pass=("plus_pass", "mean"),
    )
    # The generation columns exist only for the local runs.
    if "n_attempts" in scores.columns:
        local = scores.groupby(keys).agg(
            mean_attempts=("n_attempts", "mean"),
            mean_seconds=("gen_seconds", "mean"),
        )
        table = table.join(local)
    return table


# Published greedy pass@1 (%) of the local models on the EvalPlus leaderboard
# (https://evalplus.github.io/leaderboard.html), with full-precision weights:
# model, benchmark, score on the base tests, score on the base and plus tests.
_PUBLISHED_SCORES = [
    ("deepseek-coder:1.3b-instruct-q4_K_M", "humaneval", 65.9, 60.4),
    ("deepseek-coder:1.3b-instruct-q4_K_M", "mbpp", 65.3, 54.8),
    ("codeqwen:7b-chat-v1.5-q4_K_M", "humaneval", 83.5, 78.7),
    ("codeqwen:7b-chat-v1.5-q4_K_M", "mbpp", 79.4, 69.0),
    ("opencoder:8b", "humaneval", 81.7, 77.4),
    ("opencoder:8b", "mbpp", 82.0, 71.4),
    # The models of phase 2 (`preregistration_2.md`, Study D).
    ("deepseek-coder:6.7b-instruct-q4_K_M", "humaneval", 74.4, 71.3),
    ("deepseek-coder:6.7b-instruct-q4_K_M", "mbpp", 74.9, 65.6),
    ("llama3.1:8b-instruct-q4_K_M", "humaneval", 69.5, 62.8),
    ("llama3.1:8b-instruct-q4_K_M", "mbpp", 68.3, 55.6),
    ("codegemma:7b-instruct-q4_K_M", "humaneval", 60.4, 51.8),
    ("codegemma:7b-instruct-q4_K_M", "mbpp", 70.4, 56.9),
]


def compute_validation_table(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Compare the greedy one-shot scores of the local models with their
    published EvalPlus scores.

    The published scores use full-precision weights, while the local models
    run with 4-bit weights, so a few points of difference are expected.

    :param scores: output of `load_scores()`
    :return: 1 row per model and benchmark with `n_tasks`, `base_pass`,
        `plus_pass`, `published_base`, `published_plus`, and `plus_diff`
        (`plus_pass` minus `published_plus`), all as shares
    """
    mask = (
        (scores["source"] == "local")
        & (scores["temperature"] == 0)
        & (scores["scaffold"] == "one_shot")
    )
    hdbg.dassert_lt(0, mask.sum(), "No greedy scores")
    table = (
        scores[mask]
        .groupby(["model", "benchmark"])
        .agg(
            n_tasks=("task_id", "nunique"),
            base_pass=("base_pass", "mean"),
            plus_pass=("plus_pass", "mean"),
        )
    )
    # Add the published scores, as shares like the local ones.
    published = pd.DataFrame(
        _PUBLISHED_SCORES,
        columns=["model", "benchmark", "published_base", "published_plus"],
    ).set_index(["model", "benchmark"])
    table = table.join(published / 100)
    hdbg.dassert(
        not table["published_plus"].isna().any(),
        "A model has no published score",
    )
    table["plus_diff"] = table["plus_pass"] - table["published_plus"]
    return table


def compute_failure_modes(
    scores: pd.DataFrame, *, source: str, benchmark: str, temperature: float
) -> pd.DataFrame:
    """
    Count the failure modes on the tasks that flip between runs and on the
    tasks that always fail.

    A task flips for an agent when some of its runs pass and some fail, so
    those failures are run-to-run luck. A task that fails in every run is
    beyond the agent.

    :param scores: output of `load_scores()`
    :param source: `local` or `evalplus`
    :param benchmark: `humaneval` or `mbpp`
    :param temperature: e.g., `0.8`
    :return: number of failed solutions, with 1 row per failure mode and 1
        column per scaffold and kind of task (`always_fails` or `flips`)
    """
    mask = (
        (scores["source"] == source)
        & (scores["benchmark"] == benchmark)
        & (scores["temperature"] == temperature)
    )
    rows = scores[mask]
    hdbg.dassert_lt(0, len(rows), "No scores for '%s'", benchmark)
    # Count the runs that pass, for every agent and task.
    keys = ["model", "scaffold", "task_id"]
    n_passed = rows.groupby(keys)["plus_pass"].transform("sum")
    failed = rows[rows["plus_pass"] == 0].copy()
    failed["task"] = np.where(
        n_passed[failed.index] > 0, "flips", "always_fails"
    )
    table = pd.crosstab(failed["failure"], [failed["scaffold"], failed["task"]])
    return table


def compute_variance_table(
    scores: pd.DataFrame, *, source: str, benchmark: str, temperature: float
) -> pd.DataFrame:
    """
    Split the variance of the score of every agent into its task and run
    parts.

    :param scores: output of `load_scores()`
    :param source: `local` or `evalplus`
    :param benchmark: `humaneval` or `mbpp`
    :param temperature: e.g., `0.8`
    :return: 1 row per agent label, with the fields of
        `compute_variance_components()`
    """
    matrices = get_agent_matrices(
        scores, source=source, benchmark=benchmark, temperature=temperature
    )
    rows = {
        label: compute_variance_components(matrix)
        for label, matrix in matrices.items()
    }
    table = pd.DataFrame(rows).T.sort_values("score", ascending=False)
    return table


def plot_variance_parts(
    variance_table: pd.DataFrame,
    *,
    title: str,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the task part and the run part of the variance of 1 run.

    :param variance_table: output of `compute_variance_table()`
    :param title: plot title
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    data = variance_table.iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(data.index, data["sigma2_task"], color="tab:blue", label="tasks")
    ax.barh(
        data.index,
        data["sigma2_run"],
        left=data["sigma2_task"],
        color="tab:orange",
        label="runs",
    )
    # Write the run share next to each bar.
    for i, (_, row) in enumerate(data.iterrows()):
        total = row["sigma2_task"] + row["sigma2_run"]
        ax.text(total, i, f"  {100 * row['run_share']:.0f}% runs", va="center")
    # Leave room for the labels and keep the legend away from the bars.
    totals = data["sigma2_task"] + data["sigma2_run"]
    ax.set_xlim(0, 1.3 * totals.max())
    ax.set_xlabel("Variance of a 1-run outcome on a task")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    plt.show()


def compare_agent_pairs(
    scores: pd.DataFrame,
    pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]],
    *,
    source: str,
    benchmarks: List[str],
    temperature: float,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compare pairs of agents over tasks and runs, on several benchmarks, with
    1 Holm adjustment over all the comparisons.

    :param scores: output of `load_scores()`
    :param pairs: `((model_a, scaffold_a), (model_b, scaffold_b))` pairs
    :param source: `local` or `evalplus`
    :param benchmarks: e.g., `["humaneval", "mbpp"]`
    :param temperature: e.g., `0.8`
    :param alpha: significance level, after the Holm adjustment
    :return: 1 row per benchmark and pair with `benchmark`, `agent_a`,
        `agent_b`, the fields of `compare_agents_over_runs()`, `p_holm`, and
        `distinguishable`
    """
    rows = []
    for benchmark in benchmarks:
        for agent_a, agent_b in pairs:
            matrices = [
                get_outcome_matrix(
                    scores,
                    source=source,
                    benchmark=benchmark,
                    model=model,
                    scaffold=scaffold,
                    temperature=temperature,
                )
                for model, scaffold in (agent_a, agent_b)
            ]
            row = compare_agents_over_runs(*matrices, alpha=alpha)
            row["benchmark"] = benchmark
            row["agent_a"] = get_agent_label(*agent_a)
            row["agent_b"] = get_agent_label(*agent_b)
            rows.append(row)
    table = pd.DataFrame(rows)
    table["p_holm"] = svlutils.holm_adjust(table["p_value"])
    table["distinguishable"] = table["p_holm"] < alpha
    columns = ["benchmark", "agent_a", "agent_b", "gap", "ci_low", "ci_high"]
    columns += ["gap_run_0", "p_value", "p_holm", "distinguishable"]
    table = table[columns]
    return table


def plot_flip_rates(
    pairs: pd.DataFrame,
    *,
    title: str,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot how often each pair of agents swaps places in a 1-run leaderboard,
    against the gap between them.

    Only the pairs that swap in some leaderboards get a label: the others all
    sit at 0, where their labels would cover each other.

    :param pairs: second output of `compute_ranking_stability()`
    :param title: plot title
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(100 * pairs["gap"], 100 * pairs["p_flip"], color="tab:blue")
    for _, row in pairs[pairs["p_flip"] > 0].iterrows():
        ax.annotate(
            f"{row['upper']} > {row['lower']}",
            (100 * row["gap"], 100 * row["p_flip"]),
            fontsize=7,
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.axhline(50, color="tab:gray", linestyle=":", label="coin flip")
    ax.set_xlabel("Gap over all the runs (points)")
    ax.set_ylabel("Chance that 1 run swaps the pair (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_min_tasks_vs_runs(
    sigma2_task_diff: float,
    sigma2_run_diff: float,
    *,
    n_tasks: int,
    title: str,
    runs_list: Tuple[int, ...] = (1, 2, 5, 20),
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the tasks needed to detect a gap, for a few numbers of runs per
    task.

    :param sigma2_task_diff: output of `compute_pair_variance_components()`
    :param sigma2_run_diff: output of `compute_pair_variance_components()`
    :param n_tasks: tasks in the benchmark, drawn as a horizontal line
    :param title: plot title
    :param runs_list: runs per task to draw
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    gaps = np.linspace(0.01, 0.15, 100)
    fig, ax = plt.subplots(figsize=figsize)
    for n_runs in runs_list:
        needed = [
            compute_min_tasks_with_runs(
                sigma2_task_diff, sigma2_run_diff, gap, n_runs
            )
            for gap in gaps
        ]
        ax.plot(100 * gaps, needed, label=f"{n_runs} run(s) per task")
    ax.axhline(n_tasks, color="tab:red", linestyle="--", label="benchmark size")
    ax.set_yscale("log")
    ax.set_xlabel("Gap to detect (points)")
    ax.set_ylabel("Tasks needed (80% power)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()


def compute_temperature_table(
    scores: pd.DataFrame,
    *,
    benchmark: str,
    low: float = 0.2,
    high: float = 0.8,
) -> pd.DataFrame:
    """
    Compare the run-to-run luck of the public samples at 2 temperatures.

    Each public sample of a task plays the role of 1 run.

    :param scores: output of `load_scores()`
    :param benchmark: `humaneval` or `mbpp`
    :param low: the lower temperature
    :param high: the higher temperature
    :return: 1 row per model sampled at both temperatures, with `score`,
        `sigma2_run`, and `run_share` at each temperature (e.g.,
        `sigma2_run_low`), and `run_ratio` (`sigma2_run_high` over
        `sigma2_run_low`)
    """
    public = scores[
        (scores["source"] == "evalplus") & (scores["benchmark"] == benchmark)
    ]
    # Keep the models sampled at both temperatures.
    temperatures = public.groupby("model")["temperature"].agg(set)
    models = [
        model for model, temps in temperatures.items() if {low, high} <= temps
    ]
    hdbg.dassert_lt(0, len(models), "No model has both temperatures")
    rows = []
    for model in models:
        row: Dict[str, Any] = {"model": model}
        for name, temperature in (("low", low), ("high", high)):
            matrix = get_outcome_matrix(
                scores,
                source="evalplus",
                benchmark=benchmark,
                model=model,
                scaffold="one_shot",
                temperature=temperature,
            )
            components = compute_variance_components(matrix)
            for field in ("score", "sigma2_run", "run_share"):
                row[f"{field}_{name}"] = components[field]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    table["run_ratio"] = table["sigma2_run_high"] / table["sigma2_run_low"]
    table = table.sort_values("score_low", ascending=False)
    return table


def plot_temperature_effect(
    temperature_table: pd.DataFrame,
    *,
    title: str,
    low: float = 0.2,
    high: float = 0.8,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the run-to-run variance of every model at the 2 temperatures.

    :param temperature_table: output of `compute_temperature_table()`
    :param title: plot title
    :param low: the lower temperature, for the axis label
    :param high: the higher temperature, for the axis label
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=figsize)
    x = temperature_table["sigma2_run_low"]
    y = temperature_table["sigma2_run_high"]
    ax.scatter(x, y, color="tab:blue")
    for model in temperature_table.index:
        ax.annotate(
            model,
            (x[model], y[model]),
            fontsize=7,
            xytext=(4, 2),
            textcoords="offset points",
        )
    # Draw the lines of equal variance and of twice the variance (P7).
    limit = np.linspace(0, float(x.max()) * 1.1, 2)
    ax.plot(limit, limit, color="tab:gray", linestyle=":", label="same")
    ax.plot(limit, 2 * limit, color="tab:red", linestyle="--", label="2 times")
    ax.set_xlabel(f"Run-to-run variance at temperature {low:g}")
    ax.set_ylabel(f"Run-to-run variance at temperature {high:g}")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()


def check_predictions(
    *,
    validation: pd.DataFrame,
    score_table: pd.DataFrame,
    variance_tables: Dict[str, pd.DataFrame],
    comparisons: pd.DataFrame,
    flaky_share: float,
    temperature_table: pd.DataFrame,
    weak_model: str,
    pair_models: Tuple[str, str],
    temperature: float = 0.8,
) -> pd.DataFrame:
    """
    Check each prediction of `preregistration.md` against the results.

    The rules were written before any of the full runs was scored:
    - P1: no comparison of the close pair is distinguishable
    - P2: every comparison of the weaker model with a model of the pair is
      distinguishable, with the weaker model behind
    - P3: the run share is between 10% and 40% for every agent on every
      benchmark
    - P4: every greedy plus score is within 5 points of the published one
    - P5: in the sampled runs, repair beats one-shot for every model on every
      benchmark, and every model gains more on MBPP+ than on HumanEval+
    - P6: less than 1% of the answers change when scored again
    - P7: `run_ratio` is at least 2 for every public model

    :param validation: output of `compute_validation_table()`
    :param score_table: output of `compute_score_table()`
    :param variance_tables: benchmark -> output of `compute_variance_table()`
        for the sampled runs
    :param comparisons: output of `compare_agent_pairs()` for the
        pre-registered family, on both benchmarks
    :param flaky_share: output of `compute_flaky_share()`
    :param temperature_table: output of `compute_temperature_table()`
    :param weak_model: e.g., `deepseek-coder:1.3b-instruct-q4_K_M`
    :param pair_models: the 2 models of the close pair
    :param temperature: temperature of the sampled runs
    :return: 1 row per prediction, `P1` to `P7`, with `prediction`,
        `observed`, and `holds`
    """
    rows = {}
    # Find the comparisons of the close pair and of the weaker model, with
    # either scaffold.
    scaffolds = ("one_shot", "repair")
    close_pairs = {
        frozenset(get_agent_label(model, scaffold) for model in pair_models)
        for scaffold in scaffolds
    }
    weak_labels = {
        get_agent_label(weak_model, scaffold) for scaffold in scaffolds
    }
    weak_pairs = {
        frozenset(
            (
                get_agent_label(model, scaffold),
                get_agent_label(weak_model, scaffold),
            )
        )
        for model in pair_models
        for scaffold in scaffolds
    }
    keys = [
        frozenset(pair)
        for pair in zip(comparisons["agent_a"], comparisons["agent_b"])
    ]
    close = comparisons[[key in close_pairs for key in keys]]
    weak = comparisons[[key in weak_pairs for key in keys]]
    n_benchmarks = comparisons["benchmark"].nunique()
    hdbg.dassert_eq(len(close), len(scaffolds) * n_benchmarks)
    hdbg.dassert_eq(len(weak), 2 * len(scaffolds) * n_benchmarks)
    # P1: the close pair is never distinguishable.
    n_close = int(close["distinguishable"].sum())
    rows["P1"] = {
        "prediction": "The close pair is not distinguishable",
        "observed": (
            f"{n_close} of {len(close)} comparisons distinguishable, "
            f"smallest Holm p {close['p_holm'].min():.3g}"
        ),
        "holds": n_close == 0,
    }
    # P2: the weaker model is always distinguishable, and behind.
    sign = np.where(weak["agent_b"].isin(weak_labels), 1, -1)
    n_weak = int((weak["distinguishable"] & (sign * weak["gap"] > 0)).sum())
    rows["P2"] = {
        "prediction": "The weaker model is distinguishable from the pair",
        "observed": (
            f"{n_weak} of {len(weak)} comparisons distinguishable with the "
            f"weaker model behind, largest Holm p {weak['p_holm'].max():.3g}"
        ),
        "holds": n_weak == len(weak),
    }
    # P3: the run share of every agent on every benchmark.
    shares = pd.concat(
        {name: table["run_share"] for name, table in variance_tables.items()}
    )
    inside = shares.between(0.1, 0.4)
    rows["P3"] = {
        "prediction": "Runs are 10% to 40% of the variance of 1 run",
        "observed": (
            f"{int(inside.sum())} of {len(shares)} inside, from "
            f"{100 * shares.min():.0f}% to {100 * shares.max():.0f}%"
        ),
        "holds": bool(inside.all()),
    }
    # P4: the greedy scores against the published ones.
    diff = validation["plus_diff"]
    within = diff.abs() <= 0.05
    model, benchmark = diff.abs().idxmax()
    rows["P4"] = {
        "prediction": "Greedy scores are within 5 points of the published",
        "observed": (
            f"{int(within.sum())} of {len(diff)} within 5 points, largest gap "
            f"{100 * diff[(model, benchmark)]:+.1f} points "
            f"({model.split(':')[0]}, {benchmark})"
        ),
        "holds": bool(within.all()),
    }
    # P5: the gain of repair over one-shot in the sampled runs.
    plus = score_table.xs(
        ("local", temperature), level=("source", "temperature")
    )["plus_pass"]
    gains = plus.xs("repair", level="scaffold") - plus.xs(
        "one_shot", level="scaffold"
    )
    by_model = gains.unstack("benchmark")
    hdbg.dassert_eq(sorted(by_model.columns), ["humaneval", "mbpp"])
    more_on_mbpp = by_model["mbpp"] > by_model["humaneval"]
    rows["P5"] = {
        "prediction": "Repair helps every model, more on MBPP+",
        "observed": (
            f"gains from {100 * gains.min():+.1f} to {100 * gains.max():+.1f} "
            f"points, larger on MBPP+ for {int(more_on_mbpp.sum())} of "
            f"{len(more_on_mbpp)} models"
        ),
        "holds": bool((gains > 0).all() and more_on_mbpp.all()),
    }
    # P6: the flaky tests.
    rows["P6"] = {
        "prediction": "Less than 1% of the answers are flaky",
        "observed": f"{100 * flaky_share:.2f}% of the answers changed",
        "holds": flaky_share < 0.01,
    }
    # P7: the run variance of the public samples at the 2 temperatures.
    ratio = temperature_table["run_ratio"]
    rows["P7"] = {
        "prediction": "Run variance at 0.8 is at least 2 times that at 0.2",
        "observed": (
            f"{int((ratio >= 2).sum())} of {len(ratio)} models at least 2 "
            f"times, ratio from {ratio.min():.2f} to {ratio.max():.2f}"
        ),
        "holds": bool((ratio >= 2).all()),
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table


# #############################################################################
# Study B: run luck from 1 run
# #############################################################################


# Local models of the 2 phases of runs.
PHASE_1_MODELS = (
    "deepseek-coder:1.3b-instruct-q4_K_M",
    "codeqwen:7b-chat-v1.5-q4_K_M",
    "opencoder:8b",
)
PHASE_2_MODELS = (
    "deepseek-coder:6.7b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
    "codegemma:7b-instruct-q4_K_M",
)
# Local models that also have public EvalPlus samples (on MBPP+). Those
# samples never go into a matrix with the local model, since they show its
# strengths on each task.
_PUBLIC_NAMES = {
    "deepseek-coder:1.3b-instruct-q4_K_M": "deepseek-coder-1.3b-instruct",
    "deepseek-coder:6.7b-instruct-q4_K_M": "deepseek-coder-6.7b-instruct",
}


def get_public_greedy_agents(
    scores: pd.DataFrame, benchmark: str
) -> pd.DataFrame:
    """
    Get the public greedy samples of a benchmark, the extra agents of Study B.

    :param scores: output of `load_scores()`
    :param benchmark: `humaneval` or `mbpp`
    :return: 1 row per public model with a greedy sample and 1 column per
        task, with 0/1
    """
    mask = (
        (scores["source"] == "evalplus")
        & (scores["benchmark"] == benchmark)
        & (scores["temperature"] == 0)
    )
    models = sorted(scores.loc[mask, "model"].unique())
    hdbg.dassert_lt(0, len(models), "No public greedy samples")
    rows = {}
    for model in models:
        matrix = get_outcome_matrix(
            scores,
            source="evalplus",
            benchmark=benchmark,
            model=model,
            scaffold="one_shot",
            temperature=0.0,
        )
        hdbg.dassert_eq(matrix.shape[1], 1, "Greedy samples have 1 run")
        rows[model] = matrix[0]
    agents = pd.DataFrame(rows).T
    return agents


def get_one_run_settings(
    irt_models: List[str], penalties: List[float], extras: List[bool]
) -> Dict[str, Dict[str, Any]]:
    """
    Build a grid of settings of the 1-run estimator.

    :param irt_models: e.g., `["rasch", "2pl"]`
    :param penalties: e.g., `[0.01, 0.1]`
    :param extras: e.g., `[False, True]`, whether the matrices hold the
        public greedy agents
    :return: name (e.g., `rasch 0.01 alone`) -> `irt_model`, `penalty`, and
        `extras`
    """
    settings = {}
    for irt_model, penalty, use_extras in itertools.product(
        irt_models, penalties, extras
    ):
        name = f"{irt_model} {penalty:g} {'extras' if use_extras else 'alone'}"
        settings[name] = {
            "irt_model": irt_model,
            "penalty": penalty,
            "extras": use_extras,
        }
    return settings


def _draw_one_run(outcomes: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """
    Draw 1 run at random for each task.

    :param outcomes: tasks x runs matrix of 0/1
    :param rng: random generator
    :return: 0/1 of each task
    """
    values = outcomes.to_numpy()
    picks = rng.integers(0, values.shape[1], size=values.shape[0])
    one_run = pd.Series(
        values[np.arange(values.shape[0]), picks], index=outcomes.index
    )
    return one_run


def evaluate_one_run_estimator(
    samples: Dict[str, pd.DataFrame],
    extras: pd.DataFrame,
    settings: Dict[str, Dict[str, Any]],
    n_others: int,
    *,
    n_draws: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Run the 1-run estimator on the public samples of 1 temperature.

    This is the development set of Study B. Each model in `samples` is a
    case. In each draw, the matrix of a case holds the model and `n_others`
    other models of `samples` picked at random, each with 1 sample drawn at
    random per task as its 1 run, like the test matrices that hold 3 or 6 of
    our agents. With the `extras` setting, the matrix also holds the public
    greedy agents of the models that are not in it. All the settings see the
    same draws, so their errors can be compared.

    :param samples: model -> tasks x samples matrix of 0/1, at 1 temperature
    :param extras: output of `get_public_greedy_agents()`
    :param settings: name -> `irt_model`, `penalty`, and `extras` (whether
        the matrix holds the extra agents)
    :param n_others: number of other sampled models in each matrix
    :param n_draws: draws for each case
    :param seed: random seed
    :return: 1 row per setting, case, and draw with `setting`, `model`,
        `draw`, `score` (of the 1 run), `estimate` (the run share before
        any calibration), and `truth` (the run share from all the samples)
    """
    models = sorted(samples)
    hdbg.dassert_lte(n_others, len(models) - 1)
    rows = []
    for case_index, model in enumerate(tqdm.tqdm(models, desc="Cases")):
        truth = compute_variance_components(samples[model])["run_share"]
        others = [other for other in models if other != model]
        for draw in range(n_draws):
            # The draw depends only on the seed, the case, and the draw index,
            # so every setting sees the same matrices.
            rng = np.random.default_rng([seed, case_index, draw])
            picked = rng.choice(others, size=n_others, replace=False)
            members = [model] + list(picked)
            one_runs = pd.DataFrame(
                {
                    member: _draw_one_run(samples[member], rng)
                    for member in members
                }
            ).T
            # The extra agents never include the models of the matrix.
            others_extras = extras.drop(index=members, errors="ignore")
            for name, setting in settings.items():
                if setting["extras"]:
                    matrix = pd.concat([one_runs, others_extras])
                else:
                    matrix = one_runs
                estimates = svlutils.estimate_run_share(
                    matrix,
                    irt_model=setting["irt_model"],
                    penalty=setting["penalty"],
                )
                rows.append(
                    {
                        "setting": name,
                        "model": model,
                        "draw": draw,
                        "score": estimates.loc[model, "score"],
                        "estimate": estimates.loc[model, "run_share"],
                        "truth": truth,
                    }
                )
    results = pd.DataFrame(rows)
    return results


def fit_calibration(estimates: pd.Series, truths: pd.Series) -> float:
    """
    Find the calibration factor with the smallest mean absolute error.

    The factor `c` that minimizes the sum of |truth - c x estimate| is the
    median of truth / estimate, weighted by the estimate.

    :param estimates: run shares before calibration
    :param truths: measured run shares
    :return: the factor
    """
    hdbg.dassert_eq(len(estimates), len(truths))
    hdbg.dassert_lt(0, estimates.min(), "The estimates must be positive")
    ratios = truths.to_numpy() / estimates.to_numpy()
    order = np.argsort(ratios)
    cumulative = np.cumsum(estimates.to_numpy()[order])
    # The weighted median is the first ratio with half of the weight below it.
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2))
    factor = float(ratios[order][index])
    return factor


def summarize_one_run_errors(
    results: pd.DataFrame, *, tolerance: float = 0.10
) -> pd.DataFrame:
    """
    Compare the settings of the 1-run estimator on the development set.

    :param results: output of `evaluate_one_run_estimator()` for every
        temperature, with a `temperature` column
    :param tolerance: largest error that counts as close, e.g., `0.10` for
        10 points
    :return: 1 row per setting with:
        - `above`: share of the draws where the estimate before
          calibration is above the truth
        - `error_raw`: mean absolute error before calibration
        - `factor`: calibration factor fit on all the cases
        - `error`, `within`: mean absolute error after calibration, and
          share of the draws within `tolerance` of the truth
        - `cv_error`, `cv_within`: the same when the factor of each model
          comes from the other models only, which estimates the error on
          new models
    """
    rows = {}
    for name, group in results.groupby("setting", sort=False):
        factor = fit_calibration(group["estimate"], group["truth"])
        errors = (factor * group["estimate"] - group["truth"]).abs()
        # Fit the factor without each model, then check it on that model.
        cv_errors = []
        for model in group["model"].unique():
            train = group[group["model"] != model]
            test = group[group["model"] == model]
            cv_factor = fit_calibration(train["estimate"], train["truth"])
            cv_error = (cv_factor * test["estimate"] - test["truth"]).abs()
            cv_errors.append(cv_error)
        cv_errors = pd.concat(cv_errors)
        raw_errors = (group["estimate"] - group["truth"]).abs()
        rows[name] = {
            "above": float((group["estimate"] > group["truth"]).mean()),
            "error_raw": float(raw_errors.mean()),
            "factor": factor,
            "error": float(errors.mean()),
            "within": float((errors <= tolerance).mean()),
            "cv_error": float(cv_errors.mean()),
            "cv_within": float((cv_errors <= tolerance).mean()),
        }
    summary = pd.DataFrame.from_dict(rows, orient="index")
    return summary


def summarize_one_run_cases(
    results: pd.DataFrame,
    setting: str,
    factor: float,
    *,
    tolerance: float = 0.10,
) -> pd.DataFrame:
    """
    Summarize the draws of each development case for 1 setting.

    :param results: output of `evaluate_one_run_estimator()` for every
        temperature, with a `temperature` column
    :param setting: name of the setting
    :param factor: calibration factor applied to the estimates
    :param tolerance: largest error that counts as close
    :return: 1 row per temperature and model with `truth`, `estimate` (mean
        of the calibrated estimates over the draws), `low` and `high` (10th
        and 90th percentiles), and `within` (share of the draws within
        `tolerance` of the truth)
    """
    group = results[results["setting"] == setting]
    hdbg.dassert_lt(0, len(group), "No results for setting '%s'", setting)
    calibrated = factor * group["estimate"]
    close = (calibrated - group["truth"]).abs() <= tolerance
    keys = [group["temperature"], group["model"]]
    by_case = calibrated.groupby(keys)
    cases = pd.DataFrame(
        {
            "truth": group["truth"].groupby(keys).first(),
            "estimate": by_case.mean(),
            "low": by_case.quantile(0.1),
            "high": by_case.quantile(0.9),
            "within": close.groupby(keys).mean(),
        }
    )
    return cases


def plot_one_run_estimates(
    results: pd.DataFrame,
    factors: Dict[str, float],
    *,
    tolerance: float = 0.10,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the estimated run share of each development case against the truth.

    There is 1 panel per setting. Each point is 1 model at 1 temperature, at
    the mean of its draws, with a bar from the 10th to the 90th percentile
    of the draws. The gray band holds the estimates within `tolerance` of
    the truth.

    :param results: output of `evaluate_one_run_estimator()` for every
        temperature, with a `temperature` column
    :param factors: setting -> calibration factor of its estimates
    :param tolerance: half width of the band, e.g., `0.10` for 10 points
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(
        1, len(factors), figsize=figsize, sharex=True, sharey=True, squeeze=False
    )
    limit = np.array([0.0, 1.0])
    colors = {0.2: "tab:blue", 0.8: "tab:orange"}
    for ax, (setting, factor) in zip(axes[0], factors.items()):
        cases = summarize_one_run_cases(
            results, setting, factor, tolerance=tolerance
        )
        # Shade the estimates within the tolerance of the truth.
        ax.fill_between(
            limit,
            limit - tolerance,
            limit + tolerance,
            color="tab:gray",
            alpha=0.2,
            linewidth=0,
            label=f"within {100 * tolerance:.0f} points",
        )
        ax.plot(limit, limit, color="tab:gray", linestyle=":", linewidth=1)
        for temperature, group in cases.groupby(level="temperature"):
            ax.errorbar(
                group["truth"],
                group["estimate"],
                yerr=[
                    group["estimate"] - group["low"],
                    group["high"] - group["estimate"],
                ],
                fmt="o",
                markersize=5,
                linewidth=1,
                color=colors.get(temperature, "tab:gray"),
                label=f"temperature {temperature:g}",
            )
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)
        ax.set_title(f"{setting}, factor {factor:.2f}")
        ax.set_xlabel("Run share from all 50 samples")
    axes[0][0].set_ylabel("Run share estimated from 1 sample")
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def compute_second_run_tasks(
    outcomes: pd.DataFrame,
    *,
    tolerance: float = 0.10,
    coverage: float = 0.95,
    n_draws: int = 1000,
    seed: int = 0,
) -> float:
    r"""
    Find how many tasks need a second run to measure the run share of an
    agent.

    The first run covers every task and a second run covers only `n` tasks
    picked at random. On those tasks, half the squared difference of the 2
    runs estimates $p (1 - p)$ without bias, and its average divided by
    score x (1 - score) of the first run is the run share.

    :param outcomes: tasks x runs matrix of 0/1 of 1 agent; the first 2 runs
        are used, and the run share from all the runs is the truth
    :param tolerance: largest error allowed, e.g., `0.10` for 10 points
    :param coverage: share of the random picks that must be within
        `tolerance`
    :param n_draws: random picks of the tasks
    :param seed: random seed
    :return: the smallest `n` such that `coverage` of the picks are within
        `tolerance` for `n` and for every larger `n`, or NaN if even all the
        tasks are not enough
    """
    values = outcomes.to_numpy(dtype=float)
    n_tasks = values.shape[0]
    hdbg.dassert_lte(2, values.shape[1], "Need at least 2 runs per task")
    truth = compute_variance_components(outcomes)["run_share"]
    score = values[:, 0].mean()
    # Half the squared difference of 2 runs of a task is 0 or 1/2.
    halves = (values[:, 0] - values[:, 1]) ** 2 / 2
    # Each pick is the first `n` tasks of a random order, for all `n` at once.
    rng = np.random.default_rng(seed)
    orders = np.argsort(rng.random((n_draws, n_tasks)), axis=1)
    sums = np.cumsum(halves[orders], axis=1)
    estimates = sums / np.arange(1, n_tasks + 1) / (score * (1 - score))
    within = (np.abs(estimates - truth) <= tolerance).mean(axis=0)
    # Keep the `n` where this `n` and every larger one reach the coverage.
    enough = np.flip(np.logical_and.accumulate(np.flip(within >= coverage)))
    n_needed = float(np.argmax(enough) + 1) if enough.any() else np.nan
    return n_needed


# Settings of the 1-run estimator before any tuning (`preregistration_2.md`).
UNTUNED_ONE_RUN_SETTING = {
    "irt_model": "rasch",
    "penalty": 0.01,
    "calibration": 1.0,
    "extras": False,
}
# Settings picked on the development set in the `skill_vs_luck.one_run`
# notebook: of 21 settings, the one with the smallest error when the factor
# of each model comes from the other models. Committed before the test set
# was used.
TUNED_ONE_RUN_SETTING = {
    "irt_model": "2pl",
    "penalty": 0.1,
    "calibration": 1.344,
    "extras": False,
}


def get_one_run_cases() -> pd.DataFrame:
    """
    List the 30 test cases of Study B in `preregistration_2.md`.

    A case is 1 local agent on 1 benchmark at 1 temperature:
    - Phase 1: the agents of the old models at 0.8 on both benchmarks
    - Phase 2: the agents of the new models at 0.8 on both benchmarks, and
      the agents of the old models at 0.2 on HumanEval+

    :return: 1 row per case with `phase`, `benchmark`, `temperature`,
        `model`, and `scaffold`
    """
    rows = []
    for phase, models in ((1, PHASE_1_MODELS), (2, PHASE_2_MODELS)):
        for benchmark in BENCHMARKS:
            for model in models:
                for scaffold in ("one_shot", "repair"):
                    rows.append((phase, benchmark, 0.8, model, scaffold))
    for model in PHASE_1_MODELS:
        for scaffold in ("one_shot", "repair"):
            rows.append((2, "humaneval", 0.2, model, scaffold))
    cases = pd.DataFrame(
        rows, columns=["phase", "benchmark", "temperature", "model", "scaffold"]
    )
    return cases


def build_one_run_matrix(
    scores: pd.DataFrame,
    benchmark: str,
    temperature: float,
    model: str,
    scaffold: str,
    *,
    extras: bool,
) -> pd.DataFrame:
    """
    Build the 1-run matrix of a test case of Study B.

    The matrix holds the agent of the case and the one-shot agents of the
    other local models with the same benchmark and temperature, each with its
    first run. With `extras`, it also holds the public greedy agents of the
    benchmark, except the ones of the local models in the matrix. The
    one-shot and repair agents of a model are never in the same matrix, since
    the repair answer starts from the one-shot answer.

    :param scores: output of `load_scores()` with the local and the public
        scores
    :param benchmark: benchmark of the case
    :param temperature: temperature of the case
    :param model: model of the case
    :param scaffold: scaffold of the case
    :param extras: whether the matrix holds the public greedy agents
    :return: agents x tasks matrix of 0/1, with the agent of the case first,
        e.g., `opencoder:8b / repair`
    """
    local = scores[scores["source"] == "local"]
    mask = (
        (local["benchmark"] == benchmark)
        & (local["temperature"] == temperature)
        & (local["scaffold"] == "one_shot")
    )
    models = sorted(local.loc[mask, "model"].unique())
    hdbg.dassert_in(model, models)
    # The case comes first, then the one-shot agents of the other models.
    agents = [(model, scaffold)]
    agents += [(other, "one_shot") for other in models if other != model]
    one_runs = {}
    for agent_model, agent_scaffold in agents:
        outcomes = get_outcome_matrix(
            scores,
            source="local",
            benchmark=benchmark,
            model=agent_model,
            scaffold=agent_scaffold,
            temperature=temperature,
        )
        one_runs[f"{agent_model} / {agent_scaffold}"] = outcomes[0]
    matrix = pd.DataFrame(one_runs).T
    if extras:
        # The extra agents never include a local model of the matrix.
        public = get_public_greedy_agents(scores, benchmark)
        same = [_PUBLIC_NAMES[name] for name in models if name in _PUBLIC_NAMES]
        matrix = pd.concat([matrix, public.drop(index=same, errors="ignore")])
    return matrix


def estimate_one_run_cases(
    scores: pd.DataFrame,
    cases: pd.DataFrame,
    settings: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Estimate the run share of each test case of Study B from its first run.

    :param scores: output of `load_scores()` with the local scores of both
        phases and the public scores
    :param cases: output of `get_one_run_cases()`
    :param settings: name -> `irt_model`, `penalty`, `calibration`, and
        `extras` (see `build_one_run_matrix()`)
    :return: `cases` with `n_runs`, `truth` (the run share from all the
        runs), and 1 column per setting with its estimate
    """
    rows = []
    for case in tqdm.tqdm(cases.itertuples(index=False), total=len(cases)):
        # Find the truth from all the runs of the case.
        outcomes = get_outcome_matrix(
            scores,
            source="local",
            benchmark=case.benchmark,
            model=case.model,
            scaffold=case.scaffold,
            temperature=case.temperature,
        )
        truth = compute_variance_components(outcomes)["run_share"]
        row = {**case._asdict(), "n_runs": outcomes.shape[1], "truth": truth}
        # Estimate the run share from the first runs, with each setting.
        agent = f"{case.model} / {case.scaffold}"
        for name, setting in settings.items():
            matrix = build_one_run_matrix(
                scores,
                case.benchmark,
                case.temperature,
                case.model,
                case.scaffold,
                extras=setting["extras"],
            )
            estimates = svlutils.estimate_run_share(
                matrix,
                irt_model=setting["irt_model"],
                penalty=setting["penalty"],
                calibration=setting["calibration"],
            )
            row[name] = estimates.loc[agent, "run_share"]
        rows.append(row)
    results = pd.DataFrame(rows)
    return results


def summarize_one_run_test(
    results: pd.DataFrame,
    *,
    tuned: str,
    untuned: str,
    tolerance: float = 0.10,
) -> pd.DataFrame:
    """
    Summarize the test cases of Study B by phase and temperature.

    :param results: output of `estimate_one_run_cases()`
    :param tuned: name of the setting picked on the development set
    :param untuned: name of the setting before any tuning
    :param tolerance: largest error that counts as close, e.g., `0.10` for
        10 points
    :return: 1 row per phase and temperature with:
        - `cases`: number of cases
        - `truth`, `untuned`, `tuned`: mean run share, measured and
          estimated by each setting
        - `untuned_error`, `tuned_error`: mean absolute error of each
          setting, to compare with the errors on the development set
        - `within`: share of the cases whose tuned estimate is within
          `tolerance` of the truth, the rule of B1
        - `above`: share of the cases whose estimate before tuning is above
          the truth, the rule of B2
    """
    keys = [results["phase"], results["temperature"]]
    untuned_errors = (results[untuned] - results["truth"]).abs()
    tuned_errors = (results[tuned] - results["truth"]).abs()
    columns = pd.DataFrame(
        {
            "truth": results["truth"],
            "untuned": results[untuned],
            "tuned": results[tuned],
            "untuned_error": untuned_errors,
            "tuned_error": tuned_errors,
            "within": tuned_errors <= tolerance,
            "above": results[untuned] > results["truth"],
        }
    )
    summary = columns.groupby(keys).mean()
    summary.insert(0, "cases", columns.groupby(keys).size())
    return summary


def plot_one_run_test(
    results: pd.DataFrame,
    settings: List[str],
    *,
    tolerance: float = 0.10,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the estimated run share of each test case of Study B against the
    truth.

    There is 1 panel per setting. Each point is 1 case, colored by its
    temperature as in `plot_one_run_estimates()`; the phase-1 cases, whose
    run shares were known when the estimator was tuned, are hollow. The gray
    band holds the estimates within `tolerance` of the truth.

    :param results: output of `estimate_one_run_cases()`
    :param settings: names of the settings to plot, e.g., `["untuned",
        "tuned"]`
    :param tolerance: half width of the band, e.g., `0.10` for 10 points
    :param figsize: figure size, default `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(
        1,
        len(settings),
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    # Leave room above the largest run share, in steps of 0.1.
    largest = float(results[["truth", *settings]].to_numpy().max())
    top = min(1.0, np.ceil(10 * largest + 0.5) / 10)
    limit = np.array([0.0, top])
    colors = {0.2: "tab:blue", 0.8: "tab:orange"}
    for ax, setting in zip(axes[0], settings):
        # Shade the estimates within the tolerance of the truth.
        ax.fill_between(
            limit,
            limit - tolerance,
            limit + tolerance,
            color="tab:gray",
            alpha=0.2,
            linewidth=0,
            label=f"within {100 * tolerance:.0f} points",
        )
        ax.plot(limit, limit, color="tab:gray", linestyle=":", linewidth=1)
        groups = results.groupby(["phase", "temperature"])
        for (phase, temperature), group in groups:
            color = colors.get(temperature, "tab:gray")
            ax.scatter(
                group["truth"],
                group[setting],
                s=40,
                facecolors=color if phase == 2 else "none",
                edgecolors=color,
                label=f"phase {phase}, temperature {temperature:g}",
            )
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_title(setting)
        ax.set_xlabel("Run share from all the runs")
    axes[0][0].set_ylabel("Run share estimated from run 1")
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def check_one_run_predictions(
    results: pd.DataFrame, *, tuned: str, untuned: str
) -> pd.DataFrame:
    """
    Check the predictions B1 and B2 of `preregistration_2.md`.

    Both are decided on the 18 phase-2 cases, since the run shares of the
    phase-1 cases were known when the estimator was tuned.

    :param results: output of `estimate_one_run_cases()`
    :param tuned: name of the setting picked on the development set
    :param untuned: name of the setting before any tuning
    :return: 1 row per prediction with `prediction`, `observed`, and `holds`
    """
    phase_2 = results[results["phase"] == 2]
    hdbg.dassert_eq(len(phase_2), 18, "Study B has 18 phase-2 cases")
    within = (phase_2[tuned] - phase_2["truth"]).abs() <= 0.10
    above = phase_2[untuned] > phase_2["truth"]
    rows = {
        "B1": {
            "prediction": (
                "The tuned estimate is within 10 points for at least 12 of "
                "the 18 cases"
            ),
            "observed": f"{int(within.sum())} of 18 within 10 points",
            "holds": bool(within.sum() >= 12),
        },
        "B2": {
            "prediction": (
                "The estimate before tuning is above the truth in more than "
                "half of the 18 cases"
            ),
            "observed": f"{int(above.sum())} of 18 above the truth",
            "holds": bool(above.sum() > 9),
        },
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table


def compute_second_run_cases(
    scores: pd.DataFrame, cases: pd.DataFrame
) -> pd.DataFrame:
    """
    Find how many tasks need a second run in each test case of Study B.

    :param scores: output of `load_scores()` with the local scores of both
        phases
    :param cases: output of `get_one_run_cases()`
    :return: `cases` with `n_tasks` and `second_run_tasks`, the output of
        `compute_second_run_tasks()` on all the runs of the case (NaN when
        even a second run of every task is not enough)
    """
    rows = []
    for case in cases.itertuples(index=False):
        outcomes = get_outcome_matrix(
            scores,
            source="local",
            benchmark=case.benchmark,
            model=case.model,
            scaffold=case.scaffold,
            temperature=case.temperature,
        )
        rows.append(
            {
                **case._asdict(),
                "n_tasks": outcomes.shape[0],
                "second_run_tasks": compute_second_run_tasks(outcomes),
            }
        )
    results = pd.DataFrame(rows)
    return results


# #############################################################################
# Studies C and D: lower temperatures and more models
# #############################################################################


def combine_scores(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the scores of the 2 phases of runs.

    A file of runs that got more runs in phase 2 was scored again with all its
    runs, e.g., the greedy runs of the phase-1 models, which got a second
    greedy run. Those agents keep only their phase-2 scores.

    :param old: output of `load_scores()` for phase 1
    :param new: output of `load_scores()` for phase 2
    :return: the scores of both phases, with each answer once
    """
    keys = ["source", "benchmark", "model", "temperature"]
    old_keys = pd.MultiIndex.from_frame(old[keys])
    rescored = old_keys.isin(pd.MultiIndex.from_frame(new[keys]))
    # The new scores must cover every run of the old ones.
    old_runs = old[rescored].groupby(keys)["run"].agg(set)
    new_runs = new.groupby(keys)["run"].agg(set)
    for key, runs in old_runs.items():
        hdbg.dassert_is_subset(runs, new_runs[key], "'%s' lost runs", key)
    scores = pd.concat([old[~rescored], new], ignore_index=True)
    return scores


def compute_rescored_agreement(
    old: pd.DataFrame, new: pd.DataFrame
) -> Dict[str, float]:
    """
    Check that the answers scored in both phases got the same results.

    :param old: output of `load_scores()` for phase 1
    :param new: output of `load_scores()` for phase 2
    :return: dict with `n_answers` (answers scored in both phases) and
        `same_share` (share of them with the same `plus_pass`)
    """
    keys = ["source", "benchmark", "model", "temperature", "scaffold"]
    keys += ["task_id", "run"]
    both = old[keys + ["plus_pass"]].merge(
        new[keys + ["plus_pass"]], on=keys, suffixes=("_old", "_new")
    )
    hdbg.dassert_lt(0, len(both), "No answer was scored in both phases")
    same = both["plus_pass_old"] == both["plus_pass_new"]
    agreement = {"n_answers": len(both), "same_share": float(same.mean())}
    return agreement


def compute_greedy_changes(
    scores: pd.DataFrame, *, models: List[str], benchmark: str
) -> pd.DataFrame:
    """
    Count the tasks whose one-shot result changes between 2 greedy runs.

    Greedy decoding has no randomness, so a change comes from Ollama itself,
    e.g., from its prompt cache changing the arithmetic.

    :param scores: output of `load_scores()`, with 2 greedy runs of each
        model
    :param models: e.g., `PHASE_1_MODELS`
    :param benchmark: `humaneval` or `mbpp`
    :return: 1 row per model with `n_tasks`, `n_changed` (tasks whose result
        differs between the 2 runs), `changed_share`, and the score of each
        run, `score_run_0` and `score_run_1`
    """
    rows = {}
    for model in models:
        outcomes = get_outcome_matrix(
            scores,
            source="local",
            benchmark=benchmark,
            model=model,
            scaffold="one_shot",
            temperature=0.0,
        )
        hdbg.dassert_eq(list(outcomes.columns), [0, 1], "Need 2 greedy runs")
        changed = outcomes[0] != outcomes[1]
        rows[model] = {
            "n_tasks": len(outcomes),
            "n_changed": int(changed.sum()),
            "changed_share": float(changed.mean()),
            "score_run_0": float(outcomes[0].mean()),
            "score_run_1": float(outcomes[1].mean()),
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table


def get_first_runs_matrices(
    scores: pd.DataFrame,
    *,
    models: List[str],
    scaffold: str,
    benchmark: str,
    temperature: float,
    n_runs: int,
) -> Dict[str, pd.DataFrame]:
    """
    Get the tasks x runs matrices of some local agents, with only their first
    runs.

    Leaderboards that are compared must draw from the same number of runs
    (`preregistration_2.md`): with fewer runs, the drawn run is a bigger part
    of the mean it is checked against.

    :param scores: output of `load_scores()`
    :param models: e.g., `PHASE_1_MODELS`
    :param scaffold: `one_shot` or `repair`
    :param benchmark: `humaneval` or `mbpp`
    :param temperature: e.g., `0.8`
    :param n_runs: number of first runs to keep, e.g., `3`
    :return: agent label -> tasks x runs matrix of 0/1 with the runs 0 to
        `n_runs - 1`
    """
    runs = list(range(n_runs))
    matrices = {}
    for model in models:
        outcomes = get_outcome_matrix(
            scores,
            source="local",
            benchmark=benchmark,
            model=model,
            scaffold=scaffold,
            temperature=temperature,
        )
        hdbg.dassert_is_subset(runs, outcomes.columns, "'%s' lacks runs", model)
        matrices[get_agent_label(model, scaffold)] = outcomes[runs]
    hdbg.dassert_eq(len(matrices), len(models), "2 models have the same label")
    return matrices


def compute_stability_table(
    scores: pd.DataFrame,
    *,
    models: List[str],
    benchmark: str,
    temperatures: List[float],
    n_runs: int,
    n_draws: Optional[int] = None,
) -> pd.DataFrame:
    """
    Check how often 1-run leaderboards of some models keep their order, for
    each scaffold and temperature.

    :param scores: output of `load_scores()`
    :param models: e.g., `PHASE_1_MODELS`
    :param benchmark: `humaneval` or `mbpp`
    :param temperatures: e.g., `[0.2, 0.8]`
    :param n_runs: first runs of each agent to draw from (see
        `get_first_runs_matrices()`)
    :param n_draws: random leaderboards per scaffold and temperature, or
        `None` to take every combination of runs once (see
        `compute_ranking_stability()`)
    :return: 1 row per scaffold and temperature with the summary of
        `compute_ranking_stability()`
    """
    rows = {}
    for scaffold in ("one_shot", "repair"):
        for temperature in temperatures:
            matrices = get_first_runs_matrices(
                scores,
                models=models,
                scaffold=scaffold,
                benchmark=benchmark,
                temperature=temperature,
                n_runs=n_runs,
            )
            summary, _ = compute_ranking_stability(matrices, n_draws=n_draws)
            rows[(scaffold, temperature)] = summary
    table = pd.DataFrame(rows).T
    table.index.names = ["scaffold", "temperature"]
    return table


def check_temperature_predictions(
    *,
    variance_low: pd.DataFrame,
    variance_high: pd.DataFrame,
    greedy_changes: pd.DataFrame,
    comparisons: pd.DataFrame,
    stability: pd.DataFrame,
    low: float = 0.2,
    high: float = 0.8,
) -> pd.DataFrame:
    """
    Check the predictions C1 to C5 of `preregistration_2.md` (Study C).

    - C1: the run share at 0.2 is between 5% and 30% for each of the 6 agents
    - C2: the run share at 0.2 is below the one at 0.8 for each agent
    - C3: fewer than 2% of the tasks change their one-shot result between the
      2 greedy runs, for each model
    - C4: the close pair is not distinguishable at 0.2, with either scaffold
    - C5: 1-run leaderboards of the 3 models keep the order more often at 0.2
      than at 0.8, for each scaffold

    :param variance_low: output of `compute_variance_table()` at 0.2
    :param variance_high: output of `compute_variance_table()` at 0.8, with
        all the runs of phase 1
    :param greedy_changes: output of `compute_greedy_changes()`
    :param comparisons: output of `compare_agent_pairs()` for the close pair
        at 0.2, 1 row per scaffold
    :param stability: output of `compute_stability_table()` at both
        temperatures
    :param low: the lower temperature
    :param high: the higher temperature
    :return: 1 row per prediction, `C1` to `C5`, with `prediction`,
        `observed`, and `holds`
    """
    rows = {}
    # C1 and C2: the run share of each agent at the 2 temperatures.
    share_low = variance_low["run_share"]
    share_high = variance_high["run_share"].reindex(share_low.index)
    hdbg.dassert_eq(len(share_low), 6, "Study C has 6 agents")
    hdbg.dassert(not share_high.isna().any(), "An agent misses its run share")
    inside = share_low.between(0.05, 0.30)
    rows["C1"] = {
        "prediction": f"Runs are 5% to 30% of the variance at {low:g}",
        "observed": (
            f"{int(inside.sum())} of {len(inside)} inside, from "
            f"{100 * share_low.min():.0f}% to {100 * share_low.max():.0f}%"
        ),
        "holds": bool(inside.all()),
    }
    drop = share_high - share_low
    rows["C2"] = {
        "prediction": f"The run share is lower at {low:g} than at {high:g}",
        "observed": (
            f"{int((drop > 0).sum())} of {len(drop)} lower, by "
            f"{100 * drop.min():.0f} to {100 * drop.max():.0f} points"
        ),
        "holds": bool((drop > 0).all()),
    }
    # C3: the tasks that change between the 2 greedy runs.
    changed = greedy_changes["changed_share"]
    hdbg.dassert_eq(len(changed), 3, "Study C has 3 models")
    rows["C3"] = {
        "prediction": "Fewer than 2% of the tasks change between greedy runs",
        "observed": (
            f"{int((changed < 0.02).sum())} of {len(changed)} models below "
            f"2%, from {100 * changed.min():.1f}% to "
            f"{100 * changed.max():.1f}%"
        ),
        "holds": bool((changed < 0.02).all()),
    }
    # C4: the close pair at the lower temperature.
    hdbg.dassert_eq(len(comparisons), 2, "Study C has 1 test per scaffold")
    n_distinguishable = int(comparisons["distinguishable"].sum())
    rows["C4"] = {
        "prediction": f"The close pair is not distinguishable at {low:g}",
        "observed": (
            f"{n_distinguishable} of {len(comparisons)} comparisons "
            f"distinguishable, smallest Holm p "
            f"{comparisons['p_holm'].min():.3g}"
        ),
        "holds": n_distinguishable == 0,
    }
    # C5: the 1-run leaderboards at the 2 temperatures.
    kept_low = stability.xs(low, level="temperature")["p_same_order"]
    kept_high = stability.xs(high, level="temperature")["p_same_order"]
    more = kept_low > kept_high
    shares = ", ".join(
        f"{scaffold} {100 * kept_low[scaffold]:.1f}% vs "
        f"{100 * kept_high[scaffold]:.1f}%"
        for scaffold in kept_low.index
    )
    rows["C5"] = {
        "prediction": f"1-run leaderboards keep the order more often at {low:g}",
        "observed": (
            f"{int(more.sum())} of {len(more)} scaffolds more often ({shares})"
        ),
        "holds": bool(more.all()),
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table


def check_new_model_predictions(
    *,
    validation: pd.DataFrame,
    comparisons: pd.DataFrame,
    stability: pd.Series,
) -> pd.DataFrame:
    """
    Check the predictions D1 to D3 of `preregistration_2.md` (Study D).

    - D1: the greedy one-shot plus score of each new model is within 5 points
      of its published score, on both benchmarks
    - D2: no pair of the MBPP+ cluster is distinguishable
    - D3: on MBPP+, fewer than half of the random 1-run leaderboards keep the
      order of all 6 models

    :param validation: output of `compute_validation_table()` for the new
        models
    :param comparisons: output of `compare_agent_pairs()` for the 3 pairs of
        the cluster, with the first 3 runs at 0.8
    :param stability: summary of `compute_ranking_stability()` for the
        one-shot agents of the 6 models on MBPP+, with the first 3 runs at
        0.8
    :return: 1 row per prediction, `D1` to `D3`, with `prediction`,
        `observed`, and `holds`
    """
    rows = {}
    # D1: the greedy scores against the published ones.
    diff = validation["plus_diff"]
    hdbg.dassert_eq(len(diff), 6, "Study D has 3 models x 2 benchmarks")
    within = diff.abs() <= 0.05
    model, benchmark = diff.abs().idxmax()
    rows["D1"] = {
        "prediction": "Greedy scores are within 5 points of the published",
        "observed": (
            f"{int(within.sum())} of {len(diff)} within 5 points, largest gap "
            f"{100 * diff[(model, benchmark)]:+.1f} points "
            f"({get_model_label(model)}, {benchmark})"
        ),
        "holds": bool(within.all()),
    }
    # D2: the 3 pairs of the MBPP+ cluster.
    hdbg.dassert_eq(len(comparisons), 3, "The cluster has 3 pairs")
    n_distinguishable = int(comparisons["distinguishable"].sum())
    rows["D2"] = {
        "prediction": "No pair of the MBPP+ cluster is distinguishable",
        "observed": (
            f"{n_distinguishable} of {len(comparisons)} pairs "
            f"distinguishable, smallest Holm p "
            f"{comparisons['p_holm'].min():.3g}"
        ),
        "holds": n_distinguishable == 0,
    }
    # D3: the 1-run leaderboards of the 6 models.
    kept = float(stability["p_same_order"])
    rows["D3"] = {
        "prediction": "Fewer than half of the 1-run leaderboards keep the order",
        "observed": f"{100 * kept:.1f}% keep the order of the 6 models",
        "holds": kept < 0.5,
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table
