"""Analyze saved Lane 2 pools with the frozen Hamming-MMD detector."""

import csv
import concurrent.futures
import contextlib
import dataclasses
import io
import json
import os
import statistics
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import research.Noesis.monitor.monitor_fingerprint_experiment as rnmonfiex


@dataclasses.dataclass(frozen=True)
class LiveFingerprintTrial:
    """Store one test on the fixed live response pools."""

    scenario: str
    seed: int
    substitution_fraction: float
    substituted_count: int
    p_value: float
    statistic: float
    detected: bool


def load_paid_observation_rows(
    paths: Sequence[str],
) -> Tuple[Mapping[str, object], ...]:
    """Load and combine immutable observation files in execution order."""
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as file_:
            for line_number, line in enumerate(file_, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "invalid observation row %s in %s: %s"
                        % (line_number, path, error)
                    ) from error
    call_ids = [str(row.get("call_id", "")) for row in rows]
    if any(not call_id for call_id in call_ids):
        raise ValueError("every observation must have a call_id")
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("combined observations contain duplicate call ids")
    return tuple(rows)


def _metadata(row: Mapping[str, object]) -> Dict[str, str]:
    try:
        return {str(key): str(value) for key, value in row["metadata"]}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("observation has malformed fingerprint metadata") from error


def build_live_completion_triples(
    rows: Sequence[Mapping[str, object]],
    protocol: rnmonfiex.FingerprintProtocol,
) -> Tuple[rnmonfiex.CompletionTriple, ...]:
    """Build complete prompt-aligned triples and reject missing evidence."""
    expected_pools = {"reference-a", "reference-b", "alternative"}
    by_key: Dict[Tuple[str, int, int], str] = {}
    failures = []
    bank_hashes = set()
    for row in rows:
        metadata = _metadata(row)
        pool = metadata.get("pool", "")
        if pool not in expected_pools:
            raise ValueError("observation has an unexpected fingerprint pool")
        try:
            prompt_index = int(metadata["prompt_index"])
            sample_index = int(metadata["sample_index"])
        except (KeyError, ValueError) as error:
            raise ValueError("observation has invalid prompt/sample indices") from error
        if not 0 <= prompt_index < protocol.prompt_count:
            raise ValueError("fingerprint prompt index is outside the protocol")
        if not 0 <= sample_index < protocol.samples_per_prompt:
            raise ValueError("fingerprint sample index is outside the protocol")
        bank_hashes.add(metadata.get("bank_hash", ""))
        key = (pool, prompt_index, sample_index)
        response = row.get("response")
        usable = (
            row.get("provider_status") == "ok"
            and row.get("attribution_matches") is True
            and row.get("usage_complete") is True
            and isinstance(response, str)
            and bool(response)
        )
        if usable:
            if key in by_key:
                raise ValueError("duplicate usable fingerprint pool/prompt/sample row")
            by_key[key] = response
        else:
            failures.append(str(row.get("call_id", "unknown")))
    if len(bank_hashes) != 1 or not next(iter(bank_hashes), ""):
        raise ValueError("observations do not share one non-empty bank hash")
    expected_rows = (
        protocol.prompt_count * protocol.samples_per_prompt * len(expected_pools)
    )
    if len(rows) < expected_rows:
        raise ValueError(
            "analysis requires at least %s attempted calls; found %s"
            % (expected_rows, len(rows))
        )
    if len(by_key) != expected_rows:
        raise ValueError(
            "analysis requires complete successful pools; unusable calls: %s"
            % ", ".join(failures)
        )
    records = []
    for prompt_index in range(protocol.prompt_count):
        for sample_index in range(protocol.samples_per_prompt):
            records.append(
                rnmonfiex.CompletionTriple(
                    record_id="p%03d-s%03d" % (prompt_index, sample_index),
                    prompt_id=prompt_index,
                    reference_a=by_key[("reference-a", prompt_index, sample_index)],
                    reference_b=by_key[("reference-b", prompt_index, sample_index)],
                    alternative=by_key[("alternative", prompt_index, sample_index)],
                )
            )
    return tuple(records)


def run_live_fingerprint_tests(
    records: Sequence[rnmonfiex.CompletionTriple],
    protocol: rnmonfiex.FingerprintProtocol,
    test_func: rnmonfiex.FingerprintTest = rnmonfiex.run_official_hamming_mmd,
    partial_trials: int = 100,
    workers: int = 1,
    checkpoint_path: Optional[str] = None,
) -> Tuple[LiveFingerprintTrial, ...]:
    """Run single A/A and full-swap tests plus conditional mixture trials."""
    if partial_trials < 1:
        raise ValueError("partial_trials must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    scenario_specs = (
        ("healthy_a_a", 0.0, 1),
        ("full_model_swap", 1.0, 1),
        ("partial_substitution_10", 0.10, partial_trials),
        ("partial_substitution_25", 0.25, partial_trials),
        ("partial_substitution_50", 0.50, partial_trials),
    )
    tasks = []
    for scenario, fraction, trial_count in scenario_specs:
        for seed in range(trial_count):
            tasks.append((scenario, fraction, seed))
    completed = _load_trial_checkpoint(checkpoint_path)
    task_keys = {(scenario, seed) for scenario, _, seed in tasks}
    if not set(completed).issubset(task_keys):
        raise ValueError("checkpoint contains trials outside this protocol")
    pending = [
        task for task in tasks if (task[0], task[2]) not in completed
    ]
    if workers > 1:
        if test_func is not rnmonfiex.run_official_hamming_mmd:
            raise ValueError("parallel analysis supports only the official detector")
        payloads = [
            (tuple(records), protocol, scenario, fraction, seed)
            for scenario, fraction, seed in pending
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers
        ) as executor:
            futures = {
                executor.submit(_run_official_task, payload): payload
                for payload in payloads
            }
            for future in concurrent.futures.as_completed(futures):
                trial = future.result()
                completed[(trial.scenario, trial.seed)] = trial
                _append_trial_checkpoint(checkpoint_path, trial)
        return tuple(completed[(scenario, seed)] for scenario, _, seed in tasks)
    for scenario, fraction, seed in pending:
        reference, candidate, substituted = rnmonfiex.build_comparison_pools(
            records, fraction, seed
        )
        result = test_func(reference, candidate, protocol, seed)
        trial = LiveFingerprintTrial(
                scenario=scenario,
                seed=seed,
                substitution_fraction=fraction,
                substituted_count=len(substituted),
                p_value=result.p_value,
                statistic=result.statistic,
                detected=result.p_value < protocol.alpha,
            )
        completed[(scenario, seed)] = trial
        _append_trial_checkpoint(checkpoint_path, trial)
    return tuple(completed[(scenario, seed)] for scenario, _, seed in tasks)


def _load_trial_checkpoint(
    path: Optional[str],
) -> Dict[Tuple[str, int], LiveFingerprintTrial]:
    """Load completed trials from an append-only checkpoint."""
    completed: Dict[Tuple[str, int], LiveFingerprintTrial] = {}
    if path is None or not os.path.exists(path):
        return completed
    with open(path, encoding="utf-8") as file_:
        for line_number, line in enumerate(file_, start=1):
            if not line.strip():
                continue
            try:
                trial = LiveFingerprintTrial(**json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    "invalid checkpoint row %s: %s" % (line_number, error)
                ) from error
            key = (trial.scenario, trial.seed)
            if key in completed:
                raise ValueError("checkpoint contains duplicate trials")
            completed[key] = trial
    return completed


def _append_trial_checkpoint(
    path: Optional[str], trial: LiveFingerprintTrial
) -> None:
    """Durably append one completed local test before scheduling progress is lost."""
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as file_:
        file_.write(json.dumps(dataclasses.asdict(trial), sort_keys=True) + "\n")
        file_.flush()
        os.fsync(file_.fileno())


def _run_official_task(payload: tuple) -> LiveFingerprintTrial:
    """Run one official detector task in an isolated worker process."""
    records, protocol, scenario, fraction, seed = payload
    reference, candidate, substituted = rnmonfiex.build_comparison_pools(
        records, fraction, seed
    )
    # The pinned package emits one progress bar per permutation test. Suppress
    # only that display; exceptions still propagate through the process pool.
    with contextlib.redirect_stderr(io.StringIO()):
        result = rnmonfiex.run_official_hamming_mmd(
            reference, candidate, protocol, seed
        )
    return LiveFingerprintTrial(
        scenario=scenario,
        seed=seed,
        substitution_fraction=fraction,
        substituted_count=len(substituted),
        p_value=result.p_value,
        statistic=result.statistic,
        detected=result.p_value < protocol.alpha,
    )


def summarize_live_fingerprint_tests(
    trials: Sequence[LiveFingerprintTrial],
) -> Tuple[Mapping[str, object], ...]:
    """Aggregate results without treating correlated mixtures as new data."""
    scenario_names = []
    for trial in trials:
        if trial.scenario not in scenario_names:
            scenario_names.append(trial.scenario)
    summaries = []
    for name in scenario_names:
        selected = [trial for trial in trials if trial.scenario == name]
        summaries.append(
            {
                "scenario": name,
                "substitution_fraction": selected[0].substitution_fraction,
                "substituted_count": selected[0].substituted_count,
                "trials": len(selected),
                "detection_rate": sum(item.detected for item in selected)
                / len(selected),
                "median_p_value": statistics.median(
                    item.p_value for item in selected
                ),
                "minimum_p_value": min(item.p_value for item in selected),
                "maximum_p_value": max(item.p_value for item in selected),
            }
        )
    return tuple(summaries)


def pool_diagnostics(
    records: Sequence[rnmonfiex.CompletionTriple],
) -> Mapping[str, Mapping[str, object]]:
    """Report completion-length diagnostics for each live pool."""
    pools = {
        "reference-a": [record.reference_a for record in records],
        "reference-b": [record.reference_b for record in records],
        "alternative": [record.alternative for record in records],
    }
    return {
        name: {
            "responses": len(values),
            "median_characters": statistics.median(map(len, values)),
            "responses_at_least_1000_characters": sum(
                len(value) >= 1000 for value in values
            ),
        }
        for name, values in pools.items()
    }


def render_live_findings(
    protocol: rnmonfiex.FingerprintProtocol,
    summaries: Sequence[Mapping[str, object]],
    diagnostics: Mapping[str, Mapping[str, object]],
) -> str:
    """Render a reviewable Markdown findings report."""
    lines = [
        "# Live Behavioral-fingerprint Pilot Findings",
        "",
        "## Scope",
        "",
        "- Live pools: two Cloudflare Llama 3.3 70B pools and one Cloudflare Llama 3.1 8B pool",
        "- Pool size: `%s` responses (`%s` prompts x `%s` samples)"
        % (protocol.sample_size, protocol.prompt_count, protocol.samples_per_prompt),
        "- Detector: `mmd_hamming`, `%s` permutations, alpha `%.2f`"
        % (protocol.permutations, protocol.alpha),
        "- The A/A and full-swap rows are single live tests.",
        "- Partial-mixture rates are conditional sensitivity checks on these fixed saved pools; they are not independent estimates of statistical power.",
        "",
        "## Results",
        "",
        "| Scenario | Swap | Count | Trials | Detection | Median p | p range |",
        "| :------- | ---: | ----: | -----: | --------: | -------: | :------ |",
    ]
    for item in summaries:
        lines.append(
            "| `%s` | %.0f%% | %s | %s | %.3f | %.4f | %.4f–%.4f |"
            % (
                item["scenario"],
                float(item["substitution_fraction"]) * 100,
                item["substituted_count"],
                item["trials"],
                item["detection_rate"],
                item["median_p_value"],
                item["minimum_p_value"],
                item["maximum_p_value"],
            )
        )
    lines.extend(
        [
            "",
            "## Pool diagnostics",
            "",
            "| Pool | Responses | Median characters | At least 1,000 characters |",
            "| :--- | --------: | ----------------: | ------------------------: |",
        ]
    )
    for name, values in diagnostics.items():
        lines.append(
            "| `%s` | %s | %s | %s |"
            % (
                name,
                values["responses"],
                values["median_characters"],
                values["responses_at_least_1000_characters"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- A non-alerting A/A result is necessary but does not estimate the long-run false-positive rate from one comparison.",
            "- A detected full swap shows separation for 70B versus 8B on this prompt distribution; it does not prove weight identity.",
            "- Longitudinal drift claims require additional independent windows collected later under the same frozen plan.",
            "",
        ]
    )
    return "\n".join(lines)


def write_live_fingerprint_artifacts(
    output_dir: str,
    protocol: rnmonfiex.FingerprintProtocol,
    trials: Sequence[LiveFingerprintTrial],
    diagnostics: Mapping[str, Mapping[str, object]],
) -> None:
    """Write JSON, CSV, and Markdown analysis artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    summaries = summarize_live_fingerprint_tests(trials)
    summary_record = {
        "schema_version": "noesis-live-fingerprint-analysis-v1",
        "protocol": dataclasses.asdict(protocol),
        "scenario_results": list(summaries),
        "pool_diagnostics": diagnostics,
        "limitations": [
            "one A/A comparison does not estimate a false-positive rate",
            "partial trials reuse fixed live pools and are correlated",
            "behavioral difference does not identify exact model weights",
        ],
    }
    with open(
        os.path.join(output_dir, "fingerprint_summary.json"),
        "w",
        encoding="utf-8",
    ) as file_:
        json.dump(summary_record, file_, indent=2, sort_keys=True)
        file_.write("\n")
    with open(
        os.path.join(output_dir, "fingerprint_trials.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as file_:
        writer = csv.DictWriter(
            file_, fieldnames=[field.name for field in dataclasses.fields(LiveFingerprintTrial)]
        )
        writer.writeheader()
        for trial in trials:
            writer.writerow(dataclasses.asdict(trial))
    with open(
        os.path.join(output_dir, "fingerprint_findings.md"),
        "w",
        encoding="utf-8",
    ) as file_:
        file_.write(render_live_findings(protocol, summaries, diagnostics))
