"""
Benchmarks with known answers for comparing data-science agents.

Each benchmark turns into a list of `Task` objects: a question, the data files the
agent may read, and the gold answer. `score(task, prediction)` returns 1.0 or 0.0
using the benchmark's own scoring rules, so results are comparable with the
published numbers.

| Name    | Source                         | Tasks used            | Scoring rule                          |
| :------ | :----------------------------- | :-------------------- | :------------------------------------ |
| dabench | InfiAgent-DABench (ICML 2024)  | 257 dev questions [1] | every `@name[value]` must match       |
| qrdata  | QRData (ACL 2024)              | 411 questions         | numbers within 3%, choices by prefix  |
| dacode  | DA-Code (EMNLP 2024)           | 71 text-answer tasks  | JSON fields, floats within 0.01       |
| dabstep | DABstep (Adyen, 2025)          | 10 dev tasks          | official DABstep scorer               |

Notes:
- [1] Questions whose label repeats an answer name are skipped: no answer can
  match them
- DA-Code has 500 tasks; only the 71 scored by `compare_text` give a text
  answer. The others are scored on files the agent writes (CSV, plots, models)
  and are out of scope here
- DABstep hides the answers of its 450 test tasks; only the 10 dev tasks can be
  scored locally. The test tasks load with `gold=None` and are scored by
  submitting to the leaderboard

The raw data is downloaded into `data/benchmarks/` by `download_benchmarks.py`.
"""

import ast
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import research.agentic_data_science.bakeoff.dabstep_scorer as aebdasco

_LOG = logging.getLogger(__name__)

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "data", "benchmarks")
SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "benchmark_sample.json")


@dataclass
class Task:
    """
    One question of a benchmark.

    :param benchmark: benchmark name, e.g. "dabench"
    :param task_id: id unique within the benchmark
    :param question: full prompt for the agent, including the answer format
    :param files: absolute paths of the data files the agent may read
    :param gold: gold answer in the benchmark's own format; None if hidden
    :param meta: extra fields (difficulty, category, scoring options)
    """

    benchmark: str
    task_id: str
    question: str
    files: List[str]
    gold: Any
    meta: Dict[str, Any] = field(default_factory=dict)


# #############################################################################
# InfiAgent-DABench
# #############################################################################


def _dabench_dir(root: str) -> str:
    return os.path.join(root, "InfiAgent", "examples", "DA-Agent", "data")


def load_dabench(root: str) -> List[Task]:
    """
    Load the 257 InfiAgent-DABench dev questions.
    """
    d = _dabench_dir(root)
    questions = _read_jsonl(os.path.join(d, "da-dev-questions.jsonl"))
    labels = {r["id"]: r for r in _read_jsonl(os.path.join(d, "da-dev-labels.jsonl"))}
    tasks = []
    for q in questions:
        names = [name for name, _ in labels[q["id"]]["common_answers"]]
        if len(names) != len(set(names)):
            # The label repeats an answer name with different values, so no
            # answer can match it (e.g., id 734).
            _LOG.warning("Skipping DABench %s: duplicate answer names", q["id"])
            continue
        prompt = (
            f"{q['question']}\n\nConstraints: {q['constraints']}\n\n"
            f"Answer format: {q['format']}"
        )
        tasks.append(
            Task(
                benchmark="dabench",
                task_id=str(q["id"]),
                question=prompt,
                files=[os.path.join(d, "da-dev-tables", q["file_name"])],
                gold=labels[q["id"]]["common_answers"],
                meta={"level": q["level"], "concepts": q["concepts"]},
            )
        )
    return tasks


def score_dabench(task: Task, prediction: str) -> float:
    """
    Score like `eval_closed_form.py`: every gold `@name[value]` must be present.

    Values match if equal as strings or within 1e-6 as numbers.
    """
    found = dict(re.findall(r"@(\w+)\[(.*?)\]", prediction or ""))
    for name, value in task.gold:
        pred = found.get(name)
        if pred is None:
            return 0.0
        if pred != value:
            try:
                if abs(float(pred) - float(value)) >= 1e-6:
                    return 0.0
            except ValueError:
                return 0.0
    return 1.0


# #############################################################################
# QRData
# #############################################################################


def load_qrdata(root: str) -> List[Task]:
    """
    Load the 411 QRData questions; data files come from the unzipped `data.zip`.
    """
    d = os.path.join(root, "QRData", "benchmark")
    rows = json.load(open(os.path.join(d, "QRData.json")))
    tasks = []
    for i, r in enumerate(rows):
        meta = _as_obj(r["meta_data"])
        files = _as_obj(r["data_files"])
        qtype = meta["question_type"]
        if qtype == "numerical":
            fmt = "Give the final answer as a single number."
        else:
            fmt = "Give the final answer as the letter of the correct option."
        tasks.append(
            Task(
                benchmark="qrdata",
                task_id=str(i),
                question=f"{r['data_description']}\n\n{r['question']}\n\n{fmt}",
                files=[os.path.join(d, "data", f) for f in files],
                gold=r["answer"],
                meta={"question_type": qtype, "keywords": meta.get("keywords", [])},
            )
        )
    return tasks


def score_qrdata(task: Task, prediction: str) -> float:
    """
    Score like QRData's `eval.py`.

    Numerical: the first number in the prediction is within 3% of the gold.
    Multiple choice: the prediction starts with the gold option.
    """
    pred = (prediction or "").lower()
    gold = task.gold.lower()
    if task.meta["question_type"] != "numerical":
        return float(gold == pred[: len(gold)])
    gold_val = float(gold[:-1]) / 100 if gold.endswith("%") else float(gold)
    m = re.search(r"-?\d+\.?\d*%?", pred)
    if not m:
        return 0.0
    s = m.group()
    pred_val = float(s[:-1]) / 100 if s.endswith("%") else float(s)
    lo, hi = sorted([gold_val * 0.97, gold_val * 1.03])
    return float(lo < pred_val < hi)


# #############################################################################
# DA-Code
# #############################################################################


def load_dacode(root: str) -> List[Task]:
    """
    Load the 71 DA-Code tasks whose answer is text (`compare_text`).
    """
    d = os.path.join(root, "da-code", "da_code")
    evals = {
        r["id"]: r
        for r in _read_jsonl(os.path.join(d, "configs", "eval", "eval_all.jsonl"))
    }
    tasks = []
    for t in _read_jsonl(os.path.join(d, "configs", "task", "all.jsonl")):
        ev = evals.get(t["id"])
        if ev is None or ev["func"] != ["compare_text"]:
            continue
        src = os.path.join(d, "source", t["id"])
        files = (
            sorted(os.path.join(src, f) for f in os.listdir(src))
            if os.path.isdir(src)
            else []
        )
        tasks.append(
            Task(
                benchmark="dacode",
                task_id=t["id"],
                question=t["instruction"],
                files=files,
                gold=ev["result"][0]["number"],
                meta={
                    "type": t["type"],
                    "hardness": t["hardness"],
                    "ignore_order": ev["options"][0].get("ignore_order", False),
                },
            )
        )
    return tasks


def score_dacode(task: Task, prediction: str) -> float:
    """
    Score like DA-Code's `compare_text` with `score_rule="all"`.

    The first `{...}` in the prediction is parsed as JSON. Every gold key must be
    present (case-insensitive); floats match within 0.01, other values exactly
    (strings case-insensitive). Several gold dicts may be accepted.
    """
    pred = _parse_json_object(prediction or "")
    if pred is None:
        return 0.0
    ignore_order = task.meta.get("ignore_order", False)
    return max(_dacode_dict_score(g, pred, ignore_order) for g in task.gold)


def _dacode_dict_score(gold: Dict, pred: Dict, ignore_order: bool) -> float:
    pred_l = {str(k).lower().strip(): v for k, v in pred.items()}
    for key, g in ((str(k).lower().strip(), v) for k, v in gold.items()):
        if key not in pred_l:
            return 0.0
        p = pred_l[key]
        # A one-element list and its element compare equal.
        g1 = g[0] if isinstance(g, list) and len(g) == 1 else g
        p1 = p[0] if isinstance(p, list) and len(p) == 1 else p
        if type(g1) == type(p1):
            g, p = g1, p1
        elif isinstance(g, list) and not isinstance(p, list):
            p = [p]
        elif not isinstance(g, list) and isinstance(p, list):
            g = [g]
        try:
            p = type(g)(p)
        except (TypeError, ValueError):
            return 0.0
        if not _dacode_equal(g, p, ignore_order):
            return 0.0
    return 1.0


def _dacode_equal(g: Any, p: Any, ignore_order: bool) -> bool:
    if isinstance(g, float):
        return abs(g - p) <= 1e-2
    if isinstance(g, str):
        return g.lower().strip() == p.lower().strip()
    if isinstance(g, list):
        if len(g) != len(p):
            return False
        if ignore_order:
            key = lambda x: (x is None, str(x))
            g, p = sorted(g, key=key), sorted(p, key=key)
        return all(
            type(a) == type(b) and _dacode_equal(a, b, ignore_order)
            for a, b in zip(g, p)
        )
    if isinstance(g, dict):
        return g.keys() == p.keys() and all(
            type(g[k]) == type(p[k]) and _dacode_equal(g[k], p[k], ignore_order)
            for k in g
        )
    return g == p


# #############################################################################
# DABstep
# #############################################################################


def load_dabstep(root: str, split: str = "dev") -> List[Task]:
    """
    Load DABstep tasks.

    :param split: "dev" (10 tasks with answers) or "default" (450 tasks,
        answers hidden, so `gold=None`)
    """
    d = os.path.join(root, "DABstep", "data")
    name = "dev.jsonl" if split == "dev" else "all.jsonl"
    context = os.path.join(d, "context")
    files = (
        sorted(os.path.join(context, f) for f in os.listdir(context))
        if os.path.isdir(context)
        else []
    )
    tasks = []
    for r in _read_jsonl(os.path.join(d, "tasks", name)):
        gold = r.get("answer") or None
        tasks.append(
            Task(
                benchmark="dabstep",
                task_id=str(r["task_id"]),
                question=f"{r['question']}\n\nGuidelines: {r['guidelines']}",
                files=files,
                gold=gold,
                meta={"level": r.get("level"), "split": split},
            )
        )
    return tasks


def score_dabstep(task: Task, prediction: str) -> float:
    """
    Score with the official DABstep scorer.
    """
    if task.gold is None:
        raise ValueError(f"DABstep task {task.task_id} has a hidden answer")
    return float(aebdasco.question_scorer(prediction or "", str(task.gold)))


# #############################################################################
# Registry, sampling, and gold check
# #############################################################################


LOADERS: Dict[str, Callable[[str], List[Task]]] = {
    "dabench": load_dabench,
    "qrdata": load_qrdata,
    "dacode": load_dacode,
    "dabstep": load_dabstep,
}

SCORERS: Dict[str, Callable[[Task, str], float]] = {
    "dabench": score_dabench,
    "qrdata": score_qrdata,
    "dacode": score_dacode,
    "dabstep": score_dabstep,
}


def load_tasks(name: str, root: str = BENCHMARK_DIR) -> List[Task]:
    """
    Load all scorable tasks of a benchmark.
    """
    if name not in LOADERS:
        raise KeyError(f"Unknown benchmark '{name}'. Known: {sorted(LOADERS)}")
    return LOADERS[name](root)


def score(task: Task, prediction: Optional[str]) -> float:
    """
    Score a prediction with the rules of the task's benchmark.
    """
    return SCORERS[task.benchmark](task, prediction or "")


def gold_as_prediction(task: Task) -> str:
    """
    Write the gold answer the way a correct agent would answer.

    Used to check that each scorer gives full marks to the gold answer.
    """
    if task.benchmark == "dabench":
        return " ".join(f"@{n}[{v}]" for n, v in task.gold)
    if task.benchmark == "dacode":
        return json.dumps(task.gold[0])
    return str(task.gold)


def sample_ids(tasks: List[Task], n: int, seed: int = 0) -> Union[List[str], str]:
    """
    Pick a fixed random sample of task ids.

    :return: the sampled ids, or "all" if there are at most `n` tasks
    """
    ids = [t.task_id for t in tasks]
    if len(ids) <= n:
        return "all"
    return sorted(random.Random(seed).sample(ids, n), key=ids.index)


def load_sample(name: str, root: str = BENCHMARK_DIR) -> List[Task]:
    """
    Load the tasks of a benchmark that are in the committed sample file.
    """
    ids = json.load(open(SAMPLE_FILE))[name]
    tasks = load_tasks(name, root)
    if ids == "all":
        return tasks
    by_id = {t.task_id: t for t in tasks}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"{name}: sampled ids not found: {missing[:5]}")
    return [by_id[i] for i in ids]


# #############################################################################
# Helpers
# #############################################################################


def _read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _as_obj(value: Any) -> Any:
    """
    Return `value` parsed if it is a string holding a Python literal.
    """
    return ast.literal_eval(value) if isinstance(value, str) else value


def _parse_json_object(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    s = m.group()
    for candidate in (s, s.replace("'", '"')):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None
