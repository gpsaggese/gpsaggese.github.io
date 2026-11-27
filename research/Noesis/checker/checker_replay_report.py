"""
Aggregate offline replay runs and render reproducible experiment artifacts.

Import as:

import research.Noesis.checker.checker_replay_report as rnochrerp
"""

import csv
import dataclasses
import io
import json
import logging
import pathlib
from typing import Dict, Mapping, Sequence, Tuple

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_replay as rnochrep
import research.Noesis.checker.checker_replay_runner as rnochrer

_LOG = logging.getLogger(__name__)

_SCHEMA_VERSION = "noesis-checker-replay-v1"
_PLAN_ACCEPTANCE_TOLERANCE = 0.05


# #############################################################################
# Experiment results
# #############################################################################


@dataclasses.dataclass(frozen=True)
class ReplayScenarioResult:
    """
    Store per-seed runs and aggregate agreement for one scenario.
    """

    scenario: rnochrep.ReplayScenario
    runs: Tuple[rnochrer.ReplayRunSummary, ...]
    verdict_counts: Mapping[str, int]
    latency_failure_count: int
    quality_failure_count: int
    both_failure_count: int
    total_contracts: int
    observed_failure_rate: float
    analytic_failure_rate: float
    absolute_difference: float
    statistical_tolerance: float
    statistical_agreement: bool
    within_five_points: bool

    @property
    def passed(self) -> bool:
        """
        Return whether statistical and roadmap acceptance checks both pass.
        """
        return self.statistical_agreement and self.within_five_points


@dataclasses.dataclass(frozen=True)
class ReplayExperimentResult:
    """
    Store the complete deterministic replay experiment.
    """

    contracts_per_seed: int
    seeds: Tuple[int, ...]
    config: rnochrer.ReplayExperimentConfig
    scenarios: Tuple[ReplayScenarioResult, ...]

    @property
    def passed(self) -> bool:
        """
        Return whether every scenario meets its analytic prediction.
        """
        return all(scenario.passed for scenario in self.scenarios)


@dataclasses.dataclass(frozen=True)
class ReplayArtifactPaths:
    """
    Store paths written for one replay experiment.
    """

    json_path: pathlib.Path
    csv_path: pathlib.Path
    markdown_path: pathlib.Path


def _aggregate_scenario(
    scenario: rnochrep.ReplayScenario,
    runs: Sequence[rnochrer.ReplayRunSummary],
) -> ReplayScenarioResult:
    """
    Aggregate per-seed replay summaries for one scenario.
    """
    hdbg.dassert(runs, "runs must not be empty")
    verdict_counts = {verdict.value: 0 for verdict in rnochmod.Verdict}
    latency_failure_count = 0
    quality_failure_count = 0
    both_failure_count = 0
    total_contracts = 0
    analytic_failure_rate = runs[0].analytic_rates.failed
    for run in runs:
        hdbg.dassert_eq(run.scenario_name, scenario.name)
        hdbg.dassert_eq(run.analytic_rates.failed, analytic_failure_rate)
        for verdict, count in run.verdict_counts.items():
            verdict_counts[verdict] += count
        latency_failure_count += run.latency_failure_count
        quality_failure_count += run.quality_failure_count
        both_failure_count += run.both_failure_count
        total_contracts += run.contracts
    observed_failure_rate = (
        verdict_counts[rnochmod.Verdict.FAILED.value] / total_contracts
    )
    absolute_difference = abs(observed_failure_rate - analytic_failure_rate)
    statistical_tolerance = rnochrer.replay_acceptance_tolerance(
        analytic_failure_rate,
        total_contracts,
    )
    return ReplayScenarioResult(
        scenario=scenario,
        runs=tuple(runs),
        verdict_counts=verdict_counts,
        latency_failure_count=latency_failure_count,
        quality_failure_count=quality_failure_count,
        both_failure_count=both_failure_count,
        total_contracts=total_contracts,
        observed_failure_rate=observed_failure_rate,
        analytic_failure_rate=analytic_failure_rate,
        absolute_difference=absolute_difference,
        statistical_tolerance=statistical_tolerance,
        statistical_agreement=absolute_difference <= statistical_tolerance,
        within_five_points=(absolute_difference <= _PLAN_ACCEPTANCE_TOLERANCE),
    )


def run_replay_experiment(
    seeds: Sequence[int],
    contracts_per_seed: int,
    *,
    scenarios: Sequence[
        rnochrep.ReplayScenario
    ] = rnochrep.DEFAULT_REPLAY_SCENARIOS,
    config: rnochrer.ReplayExperimentConfig = rnochrer.DEFAULT_REPLAY_CONFIG,
) -> ReplayExperimentResult:
    """
    Run every scenario and seed through the production checker replay.
    """
    _LOG.debug(hprint.to_str("seeds contracts_per_seed"))
    hdbg.dassert(seeds, "seeds must not be empty")
    hdbg.dassert_eq(len(seeds), len(set(seeds)), "seeds must be unique")
    for seed in seeds:
        hdbg.dassert_isinstance(seed, int)
        hdbg.dassert(not isinstance(seed, bool), "seeds must not be Boolean")
    hdbg.dassert_lte(1, contracts_per_seed, "contracts must be positive")
    hdbg.dassert(scenarios, "scenarios must not be empty")
    results = []
    for scenario in scenarios:
        runs = tuple(
            rnochrer.run_replay_scenario(
                scenario,
                seed,
                contracts_per_seed,
                config=config,
            )
            for seed in seeds
        )
        results.append(_aggregate_scenario(scenario, runs))
    result = ReplayExperimentResult(
        contracts_per_seed=contracts_per_seed,
        seeds=tuple(seeds),
        config=config,
        scenarios=tuple(results),
    )
    _LOG.debug("return=%s", result)
    return result


# #############################################################################
# Artifact rendering
# #############################################################################


def _run_record(run: rnochrer.ReplayRunSummary) -> Dict[str, object]:
    """
    Convert one seed run to a JSON-compatible record.
    """
    return {
        "seed": run.seed,
        "contracts": run.contracts,
        "verdict_counts": dict(run.verdict_counts),
        "latency_failure_count": run.latency_failure_count,
        "quality_failure_count": run.quality_failure_count,
        "both_failure_count": run.both_failure_count,
        "observed_failure_rate": run.observed_failure_rate,
    }


def experiment_to_dict(
    result: ReplayExperimentResult,
) -> Dict[str, object]:
    """
    Convert an experiment result to a stable JSON-compatible mapping.
    """
    scenarios = []
    for scenario_result in result.scenarios:
        scenario = scenario_result.scenario
        scenarios.append(
            {
                "name": scenario.name,
                "inputs": dataclasses.asdict(scenario),
                "runs": [_run_record(run) for run in scenario_result.runs],
                "aggregate": {
                    "total_contracts": scenario_result.total_contracts,
                    "verdict_counts": dict(scenario_result.verdict_counts),
                    "latency_failure_count": (
                        scenario_result.latency_failure_count
                    ),
                    "quality_failure_count": (
                        scenario_result.quality_failure_count
                    ),
                    "both_failure_count": scenario_result.both_failure_count,
                    "observed_failure_rate": (
                        scenario_result.observed_failure_rate
                    ),
                    "analytic_failure_rate": (
                        scenario_result.analytic_failure_rate
                    ),
                    "analytic_latency_failure_rate": (
                        scenario_result.runs[0].analytic_rates.latency_failed
                    ),
                    "analytic_quality_failure_rate": (
                        scenario_result.runs[0].analytic_rates.quality_failed
                    ),
                    "absolute_difference": (scenario_result.absolute_difference),
                    "statistical_tolerance": (
                        scenario_result.statistical_tolerance
                    ),
                    "statistical_agreement": (
                        scenario_result.statistical_agreement
                    ),
                    "within_five_points": (scenario_result.within_five_points),
                    "passed": scenario_result.passed,
                },
            }
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "passed": result.passed,
        "contracts_per_seed": result.contracts_per_seed,
        "seeds": list(result.seeds),
        "checker_config": dataclasses.asdict(result.config),
        "scenarios": scenarios,
    }


def render_replay_json(result: ReplayExperimentResult) -> str:
    """
    Render the complete experiment as deterministic JSON.
    """
    return json.dumps(experiment_to_dict(result), indent=2, sort_keys=True) + "\n"


def _csv_rows(result: ReplayExperimentResult) -> Sequence[Dict[str, object]]:
    """
    Return per-seed and aggregate flat records for CSV output.
    """
    rows = []
    for scenario_result in result.scenarios:
        for run in scenario_result.runs:
            absolute_difference = abs(
                run.observed_failure_rate - run.analytic_rates.failed
            )
            statistical_tolerance = rnochrer.replay_acceptance_tolerance(
                run.analytic_rates.failed,
                run.contracts,
            )
            rows.append(
                {
                    "scenario": scenario_result.scenario.name,
                    "seed": run.seed,
                    "contracts": run.contracts,
                    "passed": run.verdict_counts["PASSED"],
                    "failed": run.verdict_counts["FAILED"],
                    "insufficient": run.verdict_counts["INSUFFICIENT"],
                    "latency_failures": run.latency_failure_count,
                    "quality_failures": run.quality_failure_count,
                    "both_failures": run.both_failure_count,
                    "observed_failure_rate": run.observed_failure_rate,
                    "analytic_failure_rate": run.analytic_rates.failed,
                    "analytic_latency_failure_rate": (
                        run.analytic_rates.latency_failed
                    ),
                    "analytic_quality_failure_rate": (
                        run.analytic_rates.quality_failed
                    ),
                    "absolute_difference": absolute_difference,
                    "statistical_tolerance": statistical_tolerance,
                    "passed_acceptance": (
                        absolute_difference <= statistical_tolerance
                        and absolute_difference <= _PLAN_ACCEPTANCE_TOLERANCE
                    ),
                }
            )
        rows.append(
            {
                "scenario": scenario_result.scenario.name,
                "seed": "aggregate",
                "contracts": scenario_result.total_contracts,
                "passed": scenario_result.verdict_counts["PASSED"],
                "failed": scenario_result.verdict_counts["FAILED"],
                "insufficient": scenario_result.verdict_counts["INSUFFICIENT"],
                "latency_failures": scenario_result.latency_failure_count,
                "quality_failures": scenario_result.quality_failure_count,
                "both_failures": scenario_result.both_failure_count,
                "observed_failure_rate": (scenario_result.observed_failure_rate),
                "analytic_failure_rate": (scenario_result.analytic_failure_rate),
                "analytic_latency_failure_rate": (
                    scenario_result.runs[0].analytic_rates.latency_failed
                ),
                "analytic_quality_failure_rate": (
                    scenario_result.runs[0].analytic_rates.quality_failed
                ),
                "absolute_difference": scenario_result.absolute_difference,
                "statistical_tolerance": (scenario_result.statistical_tolerance),
                "passed_acceptance": scenario_result.passed,
            }
        )
    return rows


def render_replay_csv(result: ReplayExperimentResult) -> str:
    """
    Render per-seed and aggregate experiment rows as CSV.
    """
    fieldnames = [
        "scenario",
        "seed",
        "contracts",
        "passed",
        "failed",
        "insufficient",
        "latency_failures",
        "quality_failures",
        "both_failures",
        "observed_failure_rate",
        "analytic_failure_rate",
        "analytic_latency_failure_rate",
        "analytic_quality_failure_rate",
        "absolute_difference",
        "statistical_tolerance",
        "passed_acceptance",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_csv_rows(result))
    return stream.getvalue()


def render_replay_markdown(result: ReplayExperimentResult) -> str:
    """
    Render a concise human-reviewable replay findings report.
    """
    status = "PASS" if result.passed else "FAIL"
    lines = [
        "# Checker offline replay findings",
        "",
        "- Overall acceptance: **%s**" % status,
        "- Seeds: `%s`" % ",".join(str(seed) for seed in result.seeds),
        "- Contracts per seed and scenario: %s" % result.contracts_per_seed,
        "- Total simulated contracts: %s"
        % sum(item.total_contracts for item in result.scenarios),
        "- Checker promises: latency compliance %.1f%%; canary quality %.1f%%"
        % (100 * result.config.r_min, 100 * result.config.q_min),
        "- Decision rule: one-sided %.1f%% confidence upper bound"
        % (100 * result.config.confidence),
        "- Acceptance: analytic agreement and no more than 5 percentage points",
        "",
        "## Aggregate results",
        "",
        "| Scenario | Contracts | Passed | Failed | Insufficient | Observed fail"
        " | Analytic fail | Difference | Result |",
        "| :------- | --------: | -----: | -----: | -----------: | ------------:"
        " | ------------: | ---------: | :----- |",
    ]
    for item in result.scenarios:
        lines.append(
            "| %s | %s | %s | %s | %s | %.4f | %.4f | %.4f | **%s** |"
            % (
                item.scenario.name,
                item.total_contracts,
                item.verdict_counts["PASSED"],
                item.verdict_counts["FAILED"],
                item.verdict_counts["INSUFFICIENT"],
                item.observed_failure_rate,
                item.analytic_failure_rate,
                item.absolute_difference,
                "PASS" if item.passed else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Failure attribution",
            "",
            "| Scenario | Latency failures | Quality failures | Both |",
            "| :------- | ---------------: | ---------------: | ---: |",
        ]
    )
    for item in result.scenarios:
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                item.scenario.name,
                item.latency_failure_count,
                item.quality_failure_count,
                item.both_failure_count,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The healthy scenario estimates false alarms. The slow-only and",
            "weak-model-only scenarios verify that latency and quality failures",
            "remain separately attributable. Borderline scenarios exercise",
            "uncertain evidence near the promises. The insufficient-evidence",
            "scenario verifies that a deadline does not manufacture a pass or",
            "failure when the minimum evidence is missing.",
            "",
            "These are deterministic synthetic replay results. They validate the",
            "checker implementation against its analytic model; they do not make",
            "claims about a live provider or prove that a deployed model was",
            "nerfed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_replay_artifacts(
    result: ReplayExperimentResult,
    output_dir: pathlib.Path,
) -> ReplayArtifactPaths:
    """
    Write JSON, CSV, and Markdown artifacts to ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ReplayArtifactPaths(
        json_path=output_dir / "checker_replay.json",
        csv_path=output_dir / "checker_replay.csv",
        markdown_path=output_dir / "checker_replay.md",
    )
    paths.json_path.write_text(render_replay_json(result), encoding="utf-8")
    paths.csv_path.write_text(render_replay_csv(result), encoding="utf-8")
    paths.markdown_path.write_text(
        render_replay_markdown(result),
        encoding="utf-8",
    )
    return paths
