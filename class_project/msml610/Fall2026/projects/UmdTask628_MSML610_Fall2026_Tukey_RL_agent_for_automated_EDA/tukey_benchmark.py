"""
Milestone 2 benchmark: every non-learned method on every setting.

Methods (all see only D_train; scripted agents run through the tool API with
an unlimited budget and emit DAGs, cycles removed by `project_to_dag()`):
- `proto_scripted`: the prototype's rule
- `scripted_tuned`: the same rule with its threshold tuned on validation
- `scripted_plus`: `ScriptedPlus`, Anderson-Darling threshold tuned
- `pc` (alpha 0.05) and `pc_tuned`; `ges`; `direct_lingam`
- `var_sort_regress`, `r2_sort_regress`, `rel_sort_regress` (alpha 0.05) and
  `rel_sort_regress_tuned`, `random_sort_regress`; `empty`

Tuning picks, per setting, the parameter with the best mean combined score on
the setting's validation twin `V_m2_<setting>`; the table is computed once on
`B_m2_<setting>` with those parameters.

Import as:

import tukey_benchmark as tbench
"""

import concurrent.futures
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats

import helpers.hdbg as hdbg
import tukey_baselines as tbasel
import tukey_env as tenv
import tukey_scoring as tscore
import tukey_scripted as tscrip
import tukey_sets as tsets

_LOG = logging.getLogger(__name__)

SETTINGS = ("A0", "A1", "B", "Braw", "C", "D", "E", "F", "G", "H")
# Parameter grids searched on validation.
GRIDS: Dict[str, Dict[str, List[float]]] = {
    "scripted_tuned": {"threshold": [0.05, 0.1, 0.15, 0.2, 0.25]},
    "scripted_plus": {"ad_threshold": [0.576, 0.787, 1.092]},
    "pc_tuned": {"alpha": [0.001, 0.01, 0.05, 0.1]},
    "rel_sort_regress_tuned": {"alpha": [0.01, 0.05, 0.1]},
}
# Methods with fixed parameters.
FIXED: Dict[str, Dict[str, float]] = {
    "proto_scripted": {},
    "pc": {"alpha": 0.05},
    "ges": {},
    "direct_lingam": {},
    "var_sort_regress": {},
    "r2_sort_regress": {},
    "rel_sort_regress": {"alpha": 0.05},
    "random_sort_regress": {},
    "empty": {},
}
_KEYS = (
    "score",
    "shd_target",
    "shd_dag",
    "skeleton_f1_emitted",
    "orientation_accuracy_definite",
    "orientation_coverage",
)
_UNLIMITED_BUDGET = 1_000_000


# #############################################################################
# One method on one instance
# #############################################################################


def run_method(
    method: str,
    params: Dict[str, float],
    cfg: tenv.GraphConfig,
    seed: int,
    *,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one method on one instance and score it.

    :param method: method name (`FIXED` or `GRIDS` key)
    :param params: method parameters
    :param cfg: generator settings
    :param seed: instance seed
    :param expected_hash: the instance's hash in its set manifest; the
        instance the method actually sees must match it
        - Default: None checks nothing (instances outside any set)
    :return: selected score keys, tool calls, seconds and the emitted edges
        with their marks (`tukey_scoring.edge_marks()`)
    """
    if method in ("proto_scripted", "scripted_tuned", "scripted_plus"):
        if method == "scripted_plus":
            agent = tscrip.ScriptedPlus(**params)
        else:
            agent = tscrip.ScriptedExpert(**params)
        env = tenv.TukeyEnv(cfg, budget=_UNLIMITED_BUDGET)
        evidence, _ = env.reset(seed=seed)
        if expected_hash is not None:
            tsets.check_instance(env.instance, expected_hash)
        while True:
            action = agent.next_action(evidence)
            evidence, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        edges = agent.emit_graph(evidence)
        B_hat, _ = tscore.edges_to_adjacency(edges, cfg.n_nodes)
        hdbg.dassert(tscore.is_dag(B_hat), "Agents must emit a DAG")
        scores = env.submit_graph(edges)
        scores["calls"] = len(evidence.history)
        scores["seconds"] = float("nan")
        marks = tscore.edge_marks(B_hat)
    else:
        name = method.removesuffix("_tuned")
        inst = tenv.sample_instance(cfg, seed)
        if expected_hash is not None:
            tsets.check_instance(inst, expected_hash)
        output = tbasel.run_baseline(
            name, inst.d_train.to_numpy(), seed=seed, params=params
        )
        scores = tbasel.score_output(output, inst.truth)
        scores["calls"] = scores["n_ci_tests"]
        marks = (
            []
            if output.kind == "pag"
            else tscore.edge_marks(output.adjacency, output.bidirected)
        )
    result = {key: scores.get(key, float("nan")) for key in _KEYS}
    result["calls"] = scores["calls"]
    result["seconds"] = scores["seconds"]
    result["edge_marks"] = marks
    return result


# A task: set name, method, params, instance seed, manifest hash.
Task = Tuple[str, str, Dict[str, float], int, str]


def make_tasks(
    set_name: str,
    method: str,
    params: Dict[str, float],
    manifest_dir: str,
    *,
    allow_sealed: bool = False,
) -> List[Task]:
    """
    Build one method's tasks on every instance of a set, from its manifest.

    :param set_name: evaluation set
    :param method: method name
    :param params: method parameters
    :param manifest_dir: directory of the set manifests
    :param allow_sealed: True only in the confirmatory evaluation
    :return: tasks in manifest order
    """
    items = tsets.load_set(set_name, manifest_dir, allow_sealed=allow_sealed)
    tasks = [
        (set_name, method, params, seed, sha256) for seed, _, sha256 in items
    ]
    return tasks


def _run_task(task: Task) -> Dict[str, Any]:
    """
    Run one task in a worker.

    :param task: set name, method, params, seed, manifest hash
    :return: result with the task fields
    """
    set_name, method, params, seed, sha256 = task
    cfg = tsets.TASKS[tsets.EVAL_SETS[set_name].task](seed)
    result = run_method(method, params, cfg, seed, expected_hash=sha256)
    result.update({"set": set_name, "method": method, "seed": seed})
    result["params"] = params
    return result


def run_tasks(tasks: List[Task], *, n_workers: int) -> List[Dict[str, Any]]:
    """
    Run tasks in parallel processes.

    :param tasks: tasks from `make_tasks()`
    :param n_workers: worker processes
    :return: results in task order
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_run_task, tasks, chunksize=8))
    return results


# #############################################################################
# Statistics
# #############################################################################


def bootstrap_ci(
    values: np.ndarray, *, n_boot: int = 10_000, seed: int = 0
) -> Tuple[float, float]:
    """
    Compute a percentile bootstrap 95% interval of a mean.

    :param values: per-graph values
    :param n_boot: resamples
    :param seed: random seed
    :return: lower and upper bounds
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson_interval(successes: int, n: int) -> Tuple[float, float]:
    """
    Compute the Wilson 95% interval of a proportion.

    :param successes: number of successes
    :param n: number of trials
    :return: lower and upper bounds
    """
    z = 1.959964
    p = successes / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(center - half), float(center + half)


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """
    Compute the exact two-sided McNemar p-value from discordant counts.

    :param a_only: graphs only method A recovers exactly
    :param b_only: graphs only method B recovers exactly
    :return: p-value
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    p_value = scipy.stats.binomtest(min(a_only, b_only), n, 0.5).pvalue
    return float(p_value)


def summarize(
    results: List[Dict[str, Any]], reference: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Summarize one method on one setting, paired with the reference method.

    :param results: per-graph results of the method, in seed order
    :param reference: per-graph results of `proto_scripted`, same seeds
    :return: means with intervals, exact recovery, paired comparisons
    """
    hdbg.dassert_eq([r["seed"] for r in results], [r["seed"] for r in reference])
    summary: Dict[str, Any] = {"n_graphs": len(results)}
    for key in _KEYS + ("calls", "seconds"):
        values = np.array([r[key] for r in results], dtype=float)
        summary[key] = (
            float(np.nanmean(values)) if np.any(np.isfinite(values)) else None
        )
    for key in ("score", "shd_target"):
        values = np.array([r[key] for r in results], dtype=float)
        summary[f"{key}_ci"] = bootstrap_ci(values)
        ref = np.array([r[key] for r in reference], dtype=float)
        diff = values - ref
        summary[f"{key}_minus_proto"] = float(diff.mean())
        summary[f"{key}_minus_proto_ci"] = bootstrap_ci(diff)
    exact = np.array([r["shd_target"] == 0 for r in results])
    exact_ref = np.array([r["shd_target"] == 0 for r in reference])
    summary["exact_recovery"] = float(exact.mean())
    summary["exact_recovery_ci"] = wilson_interval(int(exact.sum()), len(exact))
    summary["exact_mcnemar_p_vs_proto"] = mcnemar_exact(
        int((exact & ~exact_ref).sum()), int((~exact & exact_ref).sum())
    )
    score = np.array([r["score"] for r in results])
    score_ref = np.array([r["score"] for r in reference])
    summary["win_tie_loss_vs_proto"] = (
        int((score > score_ref + 1e-12).sum()),
        int((np.abs(score - score_ref) <= 1e-12).sum()),
        int((score < score_ref - 1e-12).sum()),
    )
    return summary


# #############################################################################
# Tuning and table
# #############################################################################


def _grid(method: str) -> List[Dict[str, float]]:
    """
    List the parameter settings of a tuned method.

    :param method: key of `GRIDS`
    :return: parameter dictionaries
    """
    ((name, values),) = GRIDS[method].items()
    return [{name: value} for value in values]


def tune(
    *, n_workers: int, manifest_dir: str
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Pick each tuned method's parameter per setting on validation graphs.

    :param n_workers: worker processes
    :param manifest_dir: directory of the set manifests
    :return: chosen parameters and validation means per setting and method
    """
    tasks = []
    for setting in SETTINGS:
        for method in GRIDS:
            for params in _grid(method):
                tasks.extend(
                    make_tasks(f"V_m2_{setting}", method, params, manifest_dir)
                )
    results = run_tasks(tasks, n_workers=n_workers)
    tuned: Dict[str, Dict[str, Any]] = {}
    for setting in SETTINGS:
        tuned[setting] = {}
        for method in GRIDS:
            means = {}
            for params in _grid(method):
                scores = [
                    r["score"]
                    for r in results
                    if r["set"] == f"V_m2_{setting}"
                    and r["method"] == method
                    and r["params"] == params
                ]
                means[str(params)] = float(np.mean(scores))
            best = max(_grid(method), key=lambda p: means[str(p)])
            tuned[setting][method] = {"params": best, "validation": means}
    return tuned


def benchmark(
    tuned: Dict[str, Dict[str, Any]], *, n_workers: int, manifest_dir: str
) -> Dict[str, Any]:
    """
    Run every method on every benchmark setting and summarize.

    :param tuned: output of `tune()`
    :param n_workers: worker processes
    :param manifest_dir: directory of the set manifests
    :return: summaries per setting and method, and the per-graph scores and
        emitted edges
    """
    tasks = []
    for setting in SETTINGS:
        methods = dict(FIXED)
        methods.update({m: tuned[setting][m]["params"] for m in GRIDS})
        for method, params in methods.items():
            tasks.extend(
                make_tasks(f"B_m2_{setting}", method, params, manifest_dir)
            )
    results = run_tasks(tasks, n_workers=n_workers)
    table: Dict[str, Any] = {}
    per_graph: Dict[str, Any] = {}
    for setting in SETTINGS:
        set_name = f"B_m2_{setting}"
        by_method: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            if r["set"] == set_name:
                by_method.setdefault(r["method"], []).append(r)
        for rows in by_method.values():
            rows.sort(key=lambda r: r["seed"])
        reference = by_method["proto_scripted"]
        table[setting] = {
            method: summarize(rows, reference)
            for method, rows in sorted(by_method.items())
        }
        per_graph[setting] = {
            method: {
                "seeds": [r["seed"] for r in rows],
                "score": [r["score"] for r in rows],
                "shd_target": [r["shd_target"] for r in rows],
                "edge_marks": [r["edge_marks"] for r in rows],
            }
            for method, rows in sorted(by_method.items())
        }
    return {"table": table, "per_graph": per_graph}
