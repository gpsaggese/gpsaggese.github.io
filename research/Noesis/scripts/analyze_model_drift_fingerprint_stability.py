#!/usr/bin/env python
"""Run ten serial A/A fingerprint comparisons with an RSS safety watchdog."""

import argparse
import csv
import itertools
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import research.Noesis.monitor.monitor_fingerprint_experiment as rnmonfiex


POOL_NAMES = (
    "reference-a",
    "reference-b",
    "reference-c",
    "reference-d",
    "reference-e",
)
DEFAULT_DST_DIR = (
    "research/Noesis/docs/findings/model_drift_fingerprint_stability_live"
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations_path", action="append", required=True)
    parser.add_argument("--dst_dir", default=DEFAULT_DST_DIR)
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--memory_limit_mb", type=int, default=2048)
    parser.add_argument("--worker_pair", nargs=2, choices=POOL_NAMES)
    parser.add_argument("--worker_result")
    return parser.parse_args(argv)


def _metadata(row: Mapping[str, object]) -> Dict[str, str]:
    return {str(key): str(value) for key, value in row.get("metadata", [])}


def _load_pools(paths: Sequence[str]) -> Dict[str, rnmonfiex.CompletionPool]:
    values = {name: {} for name in POOL_NAMES}
    bank_hashes = set()
    for path in paths:
        with open(path, encoding="utf-8") as file_:
            for line in file_:
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = _metadata(row)
                pool = metadata.get("pool")
                if pool not in values:
                    continue
                bank_hashes.add(metadata.get("bank_hash", ""))
                key = (int(metadata["prompt_index"]), int(metadata["sample_index"]))
                response = row.get("response")
                usable = (
                    row.get("provider_status") == "ok"
                    and row.get("attribution_matches") is True
                    and row.get("usage_complete") is True
                    and isinstance(response, str)
                    and bool(response)
                )
                if usable:
                    if key in values[pool]:
                        raise ValueError("duplicate usable row for %s %s" % (pool, key))
                    values[pool][key] = response
    if len(bank_hashes) != 1 or not next(iter(bank_hashes), ""):
        raise ValueError("all five pools must share one non-empty prompt-bank hash")
    pools = {}
    expected = {(prompt, sample) for prompt in range(5) for sample in range(12)}
    for name, by_key in values.items():
        if set(by_key) != expected:
            missing = sorted(expected - set(by_key))
            raise ValueError("pool %s is incomplete; missing %s" % (name, missing))
        keys = sorted(by_key)
        pools[name] = rnmonfiex.CompletionPool(
            label=name,
            prompt_ids=tuple(prompt for prompt, _ in keys),
            completions=tuple(by_key[key] for key in keys),
        )
    return pools


def _run_worker(args: argparse.Namespace) -> int:
    if not args.worker_result:
        raise ValueError("--worker_result is required with --worker_pair")
    pools = _load_pools(args.observations_path)
    left, right = args.worker_pair
    pairs = list(itertools.combinations(POOL_NAMES, 2))
    seed = pairs.index((left, right))
    protocol = rnmonfiex.FingerprintProtocol()
    result = rnmonfiex.run_official_hamming_mmd(
        pools[left], pools[right], protocol, seed
    )
    record = {
        "left_pool": left,
        "right_pool": right,
        "seed": seed,
        "p_value": result.p_value,
        "statistic": result.statistic,
        "detected_raw": result.p_value < protocol.alpha,
    }
    path = Path(args.worker_result)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _rss_kb(pid: int) -> int:
    result = subprocess.run(
        ["/bin/ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        check=False,
        text=True,
    )
    value = result.stdout.strip()
    return int(value) if value else 0


def _load_checkpoint(path: Path) -> Dict[Tuple[str, str], dict]:
    completed = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["left_pool"], row["right_pool"])
        if key in completed:
            raise ValueError("duplicate stability checkpoint row")
        completed[key] = row
    return completed


def _append_checkpoint(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_:
        file_.write(json.dumps(record, sort_keys=True) + "\n")
        file_.flush()
        os.fsync(file_.fileno())


def _run_guarded_pair(
    args: argparse.Namespace,
    pair: Tuple[str, str],
    result_path: Path,
    log_path: Path,
) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
    ]
    for path in args.observations_path:
        command.extend(["--observations_path", path])
    command.extend(
        [
            "--worker_pair",
            pair[0],
            pair[1],
            "--worker_result",
            str(result_path),
        ]
    )
    maximum_rss_kb = 0
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            maximum_rss_kb = max(maximum_rss_kb, _rss_kb(process.pid))
            if maximum_rss_kb > args.memory_limit_mb * 1024:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise MemoryError(
                    "analysis exceeded the %s MB RSS safety limit"
                    % args.memory_limit_mb
                )
            time.sleep(0.25)
    if process.returncode:
        raise RuntimeError("pair %s failed; see %s" % (pair, log_path))
    record = json.loads(result_path.read_text(encoding="utf-8"))
    record["maximum_worker_rss_mb"] = round(maximum_rss_kb / 1024.0, 1)
    return record


def _holm_adjust(records: Sequence[dict]) -> None:
    ordered = sorted(records, key=lambda row: row["p_value"])
    running = 0.0
    total = len(ordered)
    for index, row in enumerate(ordered):
        running = max(running, (total - index) * row["p_value"])
        row["holm_adjusted_p_value"] = min(1.0, running)
        row["detected_holm"] = row["holm_adjusted_p_value"] < 0.05


def _write_artifacts(dst: Path, pools: Mapping[str, rnmonfiex.CompletionPool], records: Sequence[dict]) -> None:
    _holm_adjust(records)
    summary = {
        "schema_version": "noesis-fingerprint-stability-analysis-v1",
        "comparisons": list(records),
        "comparison_count": len(records),
        "raw_alerts": sum(row["detected_raw"] for row in records),
        "holm_alerts": sum(row["detected_holm"] for row in records),
        "maximum_worker_rss_mb": max(row["maximum_worker_rss_mb"] for row in records),
    }
    (dst / "stability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (dst / "stability_comparisons.csv").open("w", encoding="utf-8", newline="") as file_:
        writer = csv.DictWriter(file_, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# Compressed Fingerprint Stability Findings",
        "",
        "Five same-day Cloudflare 70B pools were compared pairwise using the frozen Hamming-MMD detector (1,000 permutations).",
        "",
        "| Pools | p value | Holm-adjusted p | Raw alert | Holm alert | Peak worker RAM |",
        "| :---- | ------: | --------------: | :-------: | :--------: | --------------: |",
    ]
    for row in records:
        lines.append(
            "| `%s` / `%s` | %.4f | %.4f | %s | %s | %.1f MB |"
            % (
                row["left_pool"], row["right_pool"], row["p_value"],
                row["holm_adjusted_p_value"], "yes" if row["detected_raw"] else "no",
                "yes" if row["detected_holm"] else "no", row["maximum_worker_rss_mb"],
            )
        )
    lines.extend([
        "",
        "Raw alerts: `%s/10`; Holm-familywise alerts: `%s/10`." % (summary["raw_alerts"], summary["holm_alerts"]),
        "Peak observed worker RSS: `%.1f MB` (hard cutoff: `2048 MB`)." % summary["maximum_worker_rss_mb"],
        "",
        "These pools were collected in one compressed session, so this measures same-session stochastic stability rather than multi-day provider drift.",
        "",
    ])
    (dst / "stability_findings.md").write_text("\n".join(lines), encoding="utf-8")


def _run(args: argparse.Namespace) -> int:
    if args.memory_limit_mb != 2048:
        raise ValueError("this campaign freezes the RSS safety limit at 2048 MB")
    pools = _load_pools(args.observations_path)
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint_path or dst / "stability_checkpoint.jsonl")
    log_path = dst / "stability_analysis.log"
    completed = _load_checkpoint(checkpoint)
    pairs = list(itertools.combinations(POOL_NAMES, 2))
    with tempfile.TemporaryDirectory(prefix="noesis-stability-") as temp_dir:
        for index, pair in enumerate(pairs, start=1):
            if pair in completed:
                print("[%s/10] %s vs %s already checkpointed" % (index, *pair), flush=True)
                continue
            print("[%s/10] testing %s vs %s" % (index, *pair), flush=True)
            result_path = Path(temp_dir) / ("pair-%02d.json" % index)
            record = _run_guarded_pair(args, pair, result_path, log_path)
            _append_checkpoint(checkpoint, record)
            completed[pair] = record
            print(
                "        p=%.4f peak_ram=%.1fMB"
                % (record["p_value"], record["maximum_worker_rss_mb"]),
                flush=True,
            )
    records = [completed[pair] for pair in pairs]
    _write_artifacts(dst, pools, records)
    print("Analysis completed: %s" % (dst / "stability_findings.md"), flush=True)
    return 0


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.worker_pair:
        raise SystemExit(_run_worker(parsed))
    raise SystemExit(_run(parsed))
