"""
Run a small, deterministic Bernoulli-CUSUM pilot against Fisher windows.

Import as:

import research.Noesis.monitor.monitor_cusum_pilot as rnmoncupi
"""

import dataclasses
import hashlib
import math
import random
import statistics
from typing import Optional, Sequence, Tuple

import research.Noesis.monitor.monitor_capability_experiment as rnmoncaex


_SEED_VERSION = "cusum-pilot-v1"
_SCENARIO_NAMES = (
    "healthy_a_a",
    "full_weak_model_swap",
    "threshold_level_drop",
    "gradual_degradation",
    "partial_substitution_25",
    "partial_substitution_50",
)


@dataclasses.dataclass(frozen=True)
class CusumPilotProtocol:
    """Freeze the small offline comparison before execution."""

    trials_per_scenario: int = 250
    calibration_trials: int = 1000
    baseline_windows: int = 5
    post_windows: int = 15
    observations_per_window: int = 60
    healthy_accuracy: float = 0.95
    alternative_accuracy: float = 0.85
    target_false_alarm_rate: float = 0.01

    def __post_init__(self) -> None:
        for name, value in (
            ("trials_per_scenario", self.trials_per_scenario),
            ("calibration_trials", self.calibration_trials),
            ("baseline_windows", self.baseline_windows),
            ("post_windows", self.post_windows),
            ("observations_per_window", self.observations_per_window),
        ):
            if value < 1:
                raise ValueError("%s must be positive" % name)
        if not 0 < self.alternative_accuracy < self.healthy_accuracy < 1:
            raise ValueError(
                "accuracies must satisfy 0 < alternative < healthy < 1"
            )
        if not 0 < self.target_false_alarm_rate < 1:
            raise ValueError("target_false_alarm_rate must be between 0 and 1")

    def make_capability_protocol(
        self,
    ) -> rnmoncaex.CapabilityExperimentProtocol:
        """Build the existing Fisher protocol on the same evidence scale."""
        return rnmoncaex.CapabilityExperimentProtocol(
            trials_per_scenario=self.trials_per_scenario,
            baseline_windows=self.baseline_windows,
            post_windows=self.post_windows,
            observations_per_window=self.observations_per_window,
            healthy_accuracy=self.healthy_accuracy,
            minimum_quality_drop=(
                self.healthy_accuracy - self.alternative_accuracy
            ),
        )


@dataclasses.dataclass(frozen=True)
class CusumTrialResult:
    """Store paired detector outcomes for one seeded evidence stream."""

    scenario_name: str
    seed: int
    cusum_detected: bool
    cusum_query: Optional[int]
    fisher_detected: bool
    fisher_query: Optional[int]


@dataclasses.dataclass(frozen=True)
class CusumScenarioResult:
    """Aggregate the paired results for one planted scenario."""

    scenario_name: str
    trials: int
    cusum_detection_rate: float
    fisher_detection_rate: float
    cusum_median_query: Optional[float]
    fisher_median_query: Optional[float]
    cusum_p90_query: Optional[int]
    fisher_p90_query: Optional[int]


@dataclasses.dataclass(frozen=True)
class CusumPilotResult:
    """Store the complete preliminary offline comparison."""

    protocol: CusumPilotProtocol
    threshold: float
    calibration_false_alarm_rate: float
    scenarios: Tuple[CusumScenarioResult, ...]
    trials: Tuple[CusumTrialResult, ...]


def _stream_seed(stream: str, seed: int) -> int:
    material = "%s|%s|%s" % (_SEED_VERSION, stream, seed)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def bernoulli_log_likelihood_increment(
    correct: bool,
    healthy_accuracy: float,
    alternative_accuracy: float,
) -> float:
    """Return evidence favoring the lower-quality alternative."""
    if correct:
        return math.log(alternative_accuracy / healthy_accuracy)
    return math.log(
        (1 - alternative_accuracy) / (1 - healthy_accuracy)
    )


def cusum_alarm_query(
    outcomes: Sequence[bool],
    healthy_accuracy: float,
    alternative_accuracy: float,
    threshold: float,
) -> Optional[int]:
    """Return the one-based query at which the one-sided CUSUM alarms."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    statistic = 0.0
    for query, correct in enumerate(outcomes, start=1):
        statistic = max(
            0.0,
            statistic
            + bernoulli_log_likelihood_increment(
                correct,
                healthy_accuracy,
                alternative_accuracy,
            ),
        )
        if statistic >= threshold:
            return query
    return None


def _maximum_cusum(
    outcomes: Sequence[bool],
    healthy_accuracy: float,
    alternative_accuracy: float,
) -> float:
    statistic = 0.0
    maximum = 0.0
    for correct in outcomes:
        statistic = max(
            0.0,
            statistic
            + bernoulli_log_likelihood_increment(
                correct,
                healthy_accuracy,
                alternative_accuracy,
            ),
        )
        maximum = max(maximum, statistic)
    return maximum


def _healthy_outcomes(protocol: CusumPilotProtocol, seed: int) -> Tuple[bool, ...]:
    rng = random.Random(_stream_seed("calibration", seed))
    count = protocol.post_windows * protocol.observations_per_window
    return tuple(rng.random() < protocol.healthy_accuracy for _ in range(count))


def calibrate_cusum_threshold(
    protocol: CusumPilotProtocol,
) -> Tuple[float, float]:
    """Calibrate the smallest empirical threshold within the alarm budget."""
    maxima = tuple(
        _maximum_cusum(
            _healthy_outcomes(protocol, seed),
            protocol.healthy_accuracy,
            protocol.alternative_accuracy,
        )
        for seed in range(protocol.calibration_trials)
    )
    epsilon = 1e-12
    candidates = tuple(sorted(set(maxima))) + (max(maxima) + epsilon,)
    for threshold in candidates:
        false_alarm_rate = sum(
            maximum >= threshold for maximum in maxima
        ) / len(maxima)
        if false_alarm_rate <= protocol.target_false_alarm_rate:
            return threshold, false_alarm_rate
    raise AssertionError("CUSUM threshold calibration failed")


def _select_scenarios() -> Tuple[rnmoncaex.CapabilityScenario, ...]:
    by_name = {
        scenario.name: scenario
        for scenario in rnmoncaex.DEFAULT_CAPABILITY_SCENARIOS
    }
    return tuple(by_name[name] for name in _SCENARIO_NAMES)


def _flatten_correct_outcomes(
    generated: rnmoncaex.GeneratedTrial,
) -> Tuple[bool, ...]:
    outcomes = []
    for window in generated.post_windows:
        for observation in window:
            if observation.correct is not None:
                outcomes.append(observation.correct)
    return tuple(outcomes)


def _nearest_rank(values: Sequence[int], percentile: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _summarize_scenario(
    scenario_name: str,
    trials: Sequence[CusumTrialResult],
) -> CusumScenarioResult:
    cusum_queries = [
        trial.cusum_query
        for trial in trials
        if trial.cusum_query is not None
    ]
    fisher_queries = [
        trial.fisher_query
        for trial in trials
        if trial.fisher_query is not None
    ]
    return CusumScenarioResult(
        scenario_name=scenario_name,
        trials=len(trials),
        cusum_detection_rate=(
            sum(trial.cusum_detected for trial in trials) / len(trials)
        ),
        fisher_detection_rate=(
            sum(trial.fisher_detected for trial in trials) / len(trials)
        ),
        cusum_median_query=(
            statistics.median(cusum_queries) if cusum_queries else None
        ),
        fisher_median_query=(
            statistics.median(fisher_queries) if fisher_queries else None
        ),
        cusum_p90_query=_nearest_rank(cusum_queries, 0.90),
        fisher_p90_query=_nearest_rank(fisher_queries, 0.90),
    )


def run_cusum_pilot(
    protocol: CusumPilotProtocol,
    scenarios: Sequence[
        rnmoncaex.CapabilityScenario
    ] = _select_scenarios(),
) -> CusumPilotResult:
    """Run the paired, deterministic CUSUM-versus-Fisher pilot."""
    threshold, calibration_false_alarm_rate = calibrate_cusum_threshold(
        protocol
    )
    capability_protocol = protocol.make_capability_protocol()
    trial_results = []
    scenario_results = []
    for scenario in scenarios:
        current_trials = []
        for seed in range(protocol.trials_per_scenario):
            generated = rnmoncaex.generate_capability_trial(
                capability_protocol,
                scenario,
                seed,
            )
            outcomes = _flatten_correct_outcomes(generated)
            cusum_query = cusum_alarm_query(
                outcomes,
                protocol.healthy_accuracy,
                protocol.alternative_accuracy,
                threshold,
            )
            fisher = rnmoncaex.run_capability_trial(
                capability_protocol,
                scenario,
                seed,
            )
            trial = CusumTrialResult(
                scenario_name=scenario.name,
                seed=seed,
                cusum_detected=cusum_query is not None,
                cusum_query=cusum_query,
                fisher_detected=fisher.detected,
                fisher_query=fisher.queries_to_detection,
            )
            trial_results.append(trial)
            current_trials.append(trial)
        scenario_results.append(
            _summarize_scenario(scenario.name, current_trials)
        )
    return CusumPilotResult(
        protocol=protocol,
        threshold=threshold,
        calibration_false_alarm_rate=calibration_false_alarm_rate,
        scenarios=tuple(scenario_results),
        trials=tuple(trial_results),
    )
