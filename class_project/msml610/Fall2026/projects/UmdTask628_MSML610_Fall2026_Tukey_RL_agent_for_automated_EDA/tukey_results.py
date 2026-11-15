#!/usr/bin/env python
"""
Regenerate the set manifests and the stored Milestone 1-2 results.

Every file under `configs/sets/`, `results/m1/` and `results/m2/` comes from
one action below. Run them inside the project's Docker image from the project
directory as `./tukey_results.py <action> ...`, e.g. through
`./docker_cmd.sh "cd /git_root/<project dir> && ./tukey_results.py ..."`.
With `--dry_run` an action computes its results and logs the files it would
write, without writing them.

# Usage Example
- Write the manifests of the sets this milestone uses (never overwrites):
> tukey_results.py write_manifests --manifest_dir configs/sets \
    --names V_main V_proto
- Describe each generator family (redraws, targets, difficulty) (M1):
> tukey_results.py generators --out results/m1/generators.json
- Report var-, R2- and relatives-sortability per generator family (M1):
> tukey_results.py sortability --out results/m1/sortability.json
- Reproduce the Oct 10 numbers within 0.1 SHD on the prototype's seeds 0-19
  (M2):
> tukey_results.py proto_check --out results/m2/proto_check.json
- Tune the non-learned methods on validation and run the M2 table, then split
  it into the two committed files (about 1.5 minutes with 8-10 workers):
> tukey_results.py m2_table --out table.json --n_workers 8
> tukey_results.py m2_split --table table.json --out results/m2
- Run an action and log the files it would write, without writing them:
> tukey_results.py proto_check --out results/m2/proto_check.json --dry_run

The committed `results/m2` files come from the Oct 2026 run of this code
before the scoring audit. Regenerating them (checked Nov 12, 2026) gives the
same per-graph scores and structural distances, the same tuned parameters and
the same summary scores; the summary differs only in two renamed metrics
(`skeleton_f1` and `orientation_accuracy` became `skeleton_f1_emitted`,
`orientation_accuracy_definite` and `orientation_coverage`), in PC's call
counts (3 lower, the correction in `results/post_audit/pc_ci_counts.json`)
and in timings.

Import as:

import tukey_results as tresul
"""

import argparse
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List

import CausalDisco.baselines as cdbasel
import numpy as np
import scipy.stats

import helpers.hdbg as hdbg
import helpers.hparser as hparser
import research.agentic_data_science.RL_for_auto_EDA_prototype as rlproto
import tukey_baselines as tbasel
import tukey_benchmark as tbench
import tukey_env as tenv
import tukey_scoring as tscore
import tukey_scripted as tscrip
import tukey_sets as tsets
import tukey_sortability as tsortab

_LOG = logging.getLogger(__name__)


# #############################################################################
# Shared helpers
# #############################################################################


def _write_json(
    obj: Dict[str, Any],
    out: str,
    *,
    compact: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Write a JSON file with a final newline.

    :param obj: content
    :param out: output path
    :param compact: write without whitespace (large per-graph files)
    :param dry_run: log the path instead of writing the file
    """
    if dry_run:
        _LOG.warning("[DRY_RUN] Would write '%s'", out)
    else:
        _LOG.info("Writing '%s'", out)
        with open(out, "w") as file:
            if compact:
                json.dump(obj, file, separators=(",", ":"))
            else:
                json.dump(obj, file, indent=1)
            file.write("\n")


def _shd_dag(B_hat: np.ndarray, B_true: np.ndarray) -> int:
    """
    Compute the structural Hamming distance between two DAGs.

    :param B_hat: estimated DAG (nonzero entries are edges)
    :param B_true: true DAG
    :return: number of node pairs whose marks differ
    """
    shd = tscore.shd_marks(
        tscore.marks_from_pdag((B_hat != 0).astype(int)),
        tscore.marks_from_pdag(B_true),
    )
    return shd


def _paired_bootstrap(
    diff: np.ndarray, *, n_boot: int = 10_000, seed: int = 0
) -> Dict[str, float]:
    """
    Compute the mean of paired differences with a percentile bootstrap CI.

    :param diff: per-graph differences
    :param n_boot: bootstrap resamples
    :param seed: random seed
    :return: mean and 95% interval
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    summary = {
        "mean": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }
    return summary


def _wilcoxon_p(diff: np.ndarray) -> float:
    """
    Compute the two-sided Wilcoxon signed-rank p-value of paired differences.

    :param diff: per-graph differences
    :return: p-value; 1.0 when every difference is zero
    """
    p_value = float(scipy.stats.wilcoxon(diff).pvalue) if np.any(diff) else 1.0
    return p_value


def _holm(p_values: Dict[str, float]) -> Dict[str, float]:
    """
    Adjust p-values for multiple comparisons with Holm's step-down method.

    :param p_values: raw p-value per test
    :return: adjusted p-value per test
    """
    names = sorted(p_values, key=lambda name: p_values[name])
    adjusted = {}
    running = 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


# #############################################################################
# Manifests
# #############################################################################


def write_manifests(
    manifest_dir: str, names: List[str], *, dry_run: bool = False
) -> None:
    """
    Write the manifest of each set, refusing to overwrite an existing one.

    :param manifest_dir: output directory
    :param names: set names
    :param dry_run: build the manifests but write nothing
    """
    hdbg.dassert_lt(0, len(names), "Pass --names")
    if dry_run:
        _LOG.warning("[DRY_RUN] Would create '%s' if missing", manifest_dir)
    else:
        os.makedirs(manifest_dir, exist_ok=True)
    for name in names:
        path = tsets.manifest_path(manifest_dir, name)
        hdbg.dassert(not os.path.exists(path), "Manifest exists: %s", path)
        manifest = tsets.build_manifest(tsets.EVAL_SETS[name])
        _LOG.info("%s: %s instances", name, manifest["n_instances"])
        _write_json(manifest, path, dry_run=dry_run)


# #############################################################################
# Milestone 1: generator families
# #############################################################################


def generator_report(out: str, *, dry_run: bool = False) -> Dict[str, Any]:
    """
    Describe each generator family on its `V_sort_*` graphs (M1).

    Reports redraws per instance, sampling time, target classes, mechanisms,
    the smallest noise share (linear mechanisms), and the difficulty
    covariate `min_partial_corr` (smallest population |partial correlation| of
    a true edge over conditioning sets of size <= 2; linear mechanisms), with
    the share of graphs that would pass the original margin "every such
    |pcorr| >= 0.1", which was replaced by the per-edge margin.

    :param out: output JSON path
    :param dry_run: compute the report but write nothing
    :return: report per family
    """
    report: Dict[str, Any] = {}
    for name in sorted(tsets.EVAL_SETS):
        if not name.startswith("V_sort_"):
            continue
        eval_set = tsets.EVAL_SETS[name]
        rejected, seconds, min_pcorr, min_share = [], [], [], []
        targets: Dict[str, int] = {}
        mechanisms: Dict[str, int] = {}
        for seed in eval_set.seeds:
            cfg = tsets.TASKS[eval_set.task](seed)
            start = time.perf_counter()
            inst = tenv.sample_instance(cfg, seed)
            seconds.append(time.perf_counter() - start)
            rejected.append(inst.n_rejected)
            target = inst.truth.target_class
            targets[target] = targets.get(target, 0) + 1
            mechanisms[inst.mechanism] = mechanisms.get(inst.mechanism, 0) + 1
            if inst.mechanism == "linear":
                noise_var = inst.noise_std**2
                cov = tenv.population_covariance(inst.W, noise_var)
                min_share.append(float((noise_var / np.diag(cov)).min()))
                min_pcorr.append(tenv.min_partial_corr(inst.W, noise_var))
        summary: Dict[str, Any] = {
            "n_graphs": len(eval_set.seeds),
            "redraws_mean": float(np.mean(rejected)),
            "redraws_max": int(np.max(rejected)),
            "ms_per_instance": 1000 * float(np.mean(seconds)),
            "targets": targets,
            "mechanisms": mechanisms,
        }
        if min_pcorr:
            summary["min_noise_share"] = float(np.min(min_share))
            summary["min_partial_corr_median"] = float(np.median(min_pcorr))
            summary["frac_min_partial_corr_below_0.05"] = float(
                np.mean(np.array(min_pcorr) < 0.05)
            )
            summary["frac_passing_original_margin"] = float(
                np.mean(np.array(min_pcorr) >= 0.1)
            )
        report[name.removeprefix("V_sort_")] = summary
        _LOG.info("%s: %s", name, summary)
    _write_json(report, out, dry_run=dry_run)
    return report


def sortability_report(out: str, *, dry_run: bool = False) -> Dict[str, Any]:
    """
    Measure how far simple scores reveal the causal order, per family (M1).

    For each `V_sort_*` set (100 raw-scale validation graphs per family) it
    computes var-, R2- and relatives-sortability under the `paths` and `edges`
    definitions on the raw training rows and on their standardized version,
    and compares var-sort-regress with random-order regress on standardized
    data (paired SHD differences: bootstrap CI, Wilcoxon signed-rank p-value,
    Holm-adjusted across families): on standardized data sorting by variance
    should be no better than a random order.

    :param out: output JSON path
    :param dry_run: compute the report but write nothing
    :return: report per family
    """
    report: Dict[str, Any] = {}
    for name in sorted(tsets.EVAL_SETS):
        if not name.startswith("V_sort_"):
            continue
        eval_set = tsets.EVAL_SETS[name]
        values: Dict[str, List[float]] = {}
        shd_var, shd_random = [], []
        for seed in eval_set.seeds:
            inst = tenv.sample_instance(tsets.TASKS[eval_set.task](seed), seed)
            x = inst.d_train.to_numpy()
            z = (x - x.mean(axis=0)) / x.std(axis=0)
            for scale, data in (("raw", x), ("std", z)):
                for key, value in tsortab.sortabilities(data, inst.W).items():
                    values.setdefault(f"{scale}_{key}", []).append(value)
            B_true = inst.truth.B
            shd_var.append(_shd_dag(cdbasel.var_sort_regress(z), B_true))
            shd_random.append(
                _shd_dag(cdbasel.random_sort_regress(z, seed=seed), B_true)
            )
        summary: Dict[str, Any] = {
            key: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
            for key, v in sorted(values.items())
        }
        summary["n_graphs"] = len(eval_set.seeds)
        summary["shd_var_sort_regress_std"] = float(np.mean(shd_var))
        summary["shd_random_sort_regress_std"] = float(np.mean(shd_random))
        diff = np.array(shd_var, dtype=float) - np.array(shd_random, dtype=float)
        summary["var_minus_random_shd"] = _paired_bootstrap(diff)
        summary["var_minus_random_shd"]["wilcoxon_p"] = _wilcoxon_p(diff)
        report[name.removeprefix("V_sort_")] = summary
        _LOG.info("%s: %s", name, summary)
    # Holm adjustment across the families.
    holm = _holm(
        {
            family: summary["var_minus_random_shd"]["wilcoxon_p"]
            for family, summary in report.items()
        }
    )
    for family, p_value in holm.items():
        report[family]["var_minus_random_shd"]["holm_p"] = p_value
    _write_json(report, out, dry_run=dry_run)
    return report


# #############################################################################
# Milestone 2: prototype check and baseline table
# #############################################################################


def proto_check(out: str, *, dry_run: bool = False) -> Dict[str, Any]:
    """
    Reproduce the Oct 10 measurements within 0.1 SHD on seeds 0-19.

    Mean DAG SHD on the prototype's raw data of: its scripted rule (edges as
    emitted, possibly cyclic, with the prototype's `shd`), DirectLiNGAM, and
    var-sort-regress on raw and on standardized rows. The Oct 10 values were
    1.80, 0.10, 0.65 and 5.30.

    :param out: output JSON path
    :param dry_run: compute the check but write nothing
    :return: means and per-seed values
    """
    values: Dict[str, List[float]] = {
        "proto_scripted": [],
        "direct_lingam": [],
        "var_sort_regress_raw": [],
        "var_sort_regress_std": [],
    }
    for seed in range(20):
        env = rlproto.Environment(seed=seed)
        x = env.X_train
        edges = tscrip.proto_faithful_edges(x)
        B_hat, _ = tscore.edges_to_adjacency(edges, env.n_nodes)
        values["proto_scripted"].append(rlproto.shd(B_hat, env.B_true))
        lingam_out = tbasel.run_baseline("direct_lingam", x)
        values["direct_lingam"].append(
            _shd_dag(lingam_out.adjacency, env.B_true)
        )
        values["var_sort_regress_raw"].append(
            _shd_dag(cdbasel.var_sort_regress(x), env.B_true)
        )
        z = (x - x.mean(axis=0)) / x.std(axis=0)
        values["var_sort_regress_std"].append(
            _shd_dag(cdbasel.var_sort_regress(z), env.B_true)
        )
    reference = {
        "proto_scripted": 1.80,
        "direct_lingam": 0.10,
        "var_sort_regress_raw": 0.65,
        "var_sort_regress_std": 5.30,
    }
    report = {
        name: {
            "mean": float(np.mean(v)),
            "oct10": reference[name],
            "within_0.1": bool(abs(np.mean(v) - reference[name]) <= 0.1 + 1e-12),
            "per_seed": [int(x) for x in v],
        }
        for name, v in values.items()
    }
    _write_json(report, out, dry_run=dry_run)
    _LOG.info("Prototype check: %s", {k: v["mean"] for k, v in report.items()})
    return report


def m2_table(
    out: str, n_workers: int, manifest_dir: str, *, dry_run: bool = False
) -> Dict[str, Any]:
    """
    Tune the non-learned methods on validation, then run the M2 table.

    Every instance is checked against its set manifest before a method sees
    it.

    :param out: output JSON path
    :param n_workers: worker processes
    :param manifest_dir: directory of the set manifests
    :param dry_run: compute the table but write nothing
    :return: tuned parameters, summaries and per-graph scores
    """
    tuned = tbench.tune(n_workers=n_workers, manifest_dir=manifest_dir)
    _LOG.info(
        "Tuned parameters: %s",
        {s: {m: v["params"] for m, v in t.items()} for s, t in tuned.items()},
    )
    result = tbench.benchmark(
        tuned, n_workers=n_workers, manifest_dir=manifest_dir
    )
    result["tuned"] = tuned
    _write_json(result, out, dry_run=dry_run)
    return result


def m2_split(table_path: str, out_dir: str, *, dry_run: bool = False) -> None:
    """
    Split an M2 table into a summary file and a compact per-graph file.

    The full table keeps per-graph arrays for every method; the split lists
    the seeds once per setting and rounds scores to 4 decimals, so both files
    fit the repository's 500 KB limit.

    :param table_path: `table.json` written by `m2_table`
    :param out_dir: directory for `table_summary.json` and `per_graph.json`
    :param dry_run: split the table but write nothing
    """
    with open(table_path, "rb") as file:
        raw = file.read()
    table = json.loads(raw)
    source = {
        "file": table_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    summary = {
        "table": table["table"],
        "tuned": table["tuned"],
        "source": source,
    }
    per_graph: Dict[str, Any] = {"source": source, "settings": {}}
    for setting, methods in table["per_graph"].items():
        seeds = next(iter(methods.values()))["seeds"]
        entry: Dict[str, Any] = {"seeds": seeds, "methods": {}}
        for method, values in methods.items():
            hdbg.dassert_eq(
                values["seeds"],
                seeds,
                "The per-graph file lists the seeds once per setting",
            )
            entry["methods"][method] = {
                "score": [round(x, 4) for x in values["score"]],
                "shd_target": [int(x) for x in values["shd_target"]],
            }
        per_graph["settings"][setting] = entry
    if dry_run:
        _LOG.warning("[DRY_RUN] Would create '%s' if missing", out_dir)
    else:
        os.makedirs(out_dir, exist_ok=True)
    _write_json(
        summary, os.path.join(out_dir, "table_summary.json"), dry_run=dry_run
    )
    _write_json(
        per_graph,
        os.path.join(out_dir, "per_graph.json"),
        compact=True,
        dry_run=dry_run,
    )


# #############################################################################
# Command line
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=[
            "write_manifests",
            "generators",
            "sortability",
            "proto_check",
            "m2_table",
            "m2_split",
        ],
        help="What to regenerate",
    )
    parser.add_argument("--out", action="store", default="", help="Output")
    parser.add_argument(
        "--n_workers", action="store", type=int, default=8, help="Processes"
    )
    parser.add_argument(
        "--table", action="store", default="", help="M2 table to split"
    )
    parser.add_argument(
        "--manifest_dir",
        action="store",
        default="configs/sets",
        help="Directory of the set manifests",
    )
    parser.add_argument(
        "--names",
        action="store",
        nargs="*",
        default=[],
        help="Sets whose manifests to write",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute the results and log the files, without writing them",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    if args.action == "write_manifests":
        write_manifests(args.manifest_dir, args.names, dry_run=args.dry_run)
        return
    hdbg.dassert_ne(args.out, "", "Pass --out")
    if args.action == "generators":
        generator_report(args.out, dry_run=args.dry_run)
    elif args.action == "sortability":
        sortability_report(args.out, dry_run=args.dry_run)
    elif args.action == "proto_check":
        proto_check(args.out, dry_run=args.dry_run)
    elif args.action == "m2_table":
        m2_table(
            args.out, args.n_workers, args.manifest_dir, dry_run=args.dry_run
        )
    elif args.action == "m2_split":
        hdbg.dassert_ne(args.table, "", "Pass --table")
        m2_split(args.table, args.out, dry_run=args.dry_run)
    else:
        raise ValueError(f"Unknown action '{args.action}'")


if __name__ == "__main__":
    _main(_parse())
