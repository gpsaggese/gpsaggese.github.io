"""
Run blinded, deterministic capability-drift experiments offline.

Import as:

import research.Noesis.monitor.monitor_capability_experiment as rnmoncaex
"""

import dataclasses
import enum
import hashlib
import math
import random
import statistics
from typing import Mapping, Optional, Sequence, Tuple

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_detection as rnmondet
import research.Noesis.monitor.monitor_metrics as rnmonmet
import research.Noesis.monitor.monitor_models as rnmonmod

import helpers.hdbg as hdbg

_SEED_VERSION = "capability-drift-v1"
_BASE_LATENCY_MS = 100.0
_BASE_COST_USD = 0.00001
_TARGET = rnmonmod.MonitoringTarget("offline-provider", "offline-model")

CANARY_FAMILIES = (
    "arithmetic",
    "code-trace",
    "modular",
    "ordering",
    "string",
    "table",
)


# #############################################################################
# Experiment models
# #############################################################################


class ScenarioExpectation(str, enum.Enum):
    """
    Describe the acceptance rule for one planted scenario.
    """

    NO_ALERT = "NO_ALERT"
    DETECT = "DETECT"
    INSUFFICIENT = "INSUFFICIENT"
    MEASURE = "MEASURE"


@dataclasses.dataclass(frozen=True)
class CapabilityScenario:
    """
    Describe one hidden post-baseline intervention.
    """

    name: str
    post_accuracy: Optional[float]
    expectation: ScenarioExpectation
    substitution_probability: float = 0.0
    accuracy_drop_per_window: float = 0.0
    availability_probability: float = 1.0
    gradable_probability: float = 1.0
    latency_multiplier: float = 1.0
    cost_multiplier: float = 1.0
    maximum_detection_window: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must not be empty")
        for name, value in (
            ("substitution_probability", self.substitution_probability),
            ("availability_probability", self.availability_probability),
            ("gradable_probability", self.gradable_probability),
        ):
            if not 0 <= value <= 1:
                raise ValueError("%s must be between zero and one" % name)
        if self.post_accuracy is not None and not 0 <= self.post_accuracy <= 1:
            raise ValueError("post_accuracy must be between zero and one")
        if self.accuracy_drop_per_window < 0:
            raise ValueError("accuracy_drop_per_window must not be negative")
        if self.latency_multiplier <= 0 or self.cost_multiplier < 0:
            raise ValueError("metric multipliers are invalid")


@dataclasses.dataclass(frozen=True)
class CapabilityExperimentProtocol:
    """
    Freeze one reproducible capability-drift experiment.
    """

    trials_per_scenario: int = 1000
    baseline_windows: int = 5
    post_windows: int = 15
    observations_per_window: int = 60
    healthy_accuracy: float = 0.95
    weak_accuracy: float = 0.60
    alpha: float = 0.01
    minimum_quality_drop: float = 0.10
    confirmation_windows: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            ("trials_per_scenario", self.trials_per_scenario),
            ("baseline_windows", self.baseline_windows),
            ("post_windows", self.post_windows),
            ("observations_per_window", self.observations_per_window),
            ("confirmation_windows", self.confirmation_windows),
        ):
            if value < 1:
                raise ValueError("%s must be positive" % name)
        for name, value in (
            ("healthy_accuracy", self.healthy_accuracy),
            ("weak_accuracy", self.weak_accuracy),
            ("minimum_quality_drop", self.minimum_quality_drop),
        ):
            if not 0 <= value <= 1:
                raise ValueError("%s must be between zero and one" % name)
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if self.weak_accuracy >= self.healthy_accuracy:
            raise ValueError("weak_accuracy must be below healthy_accuracy")

    def make_detection_policy(self) -> rnmondet.CapabilityDetectionPolicy:
        """
        Build the detector policy represented by this protocol.
        """
        return rnmondet.CapabilityDetectionPolicy(
            alpha=self.alpha,
            minimum_quality_drop=self.minimum_quality_drop,
            minimum_graded_per_window=self.observations_per_window,
            minimum_baseline_graded=(
                self.baseline_windows * self.observations_per_window
            ),
            confirmation_windows=self.confirmation_windows,
        )


@dataclasses.dataclass(frozen=True)
class TrialTruth:
    """
    Store planted facts that are visible only to the evaluator.
    """

    scenario_name: str
    post_accuracies: Tuple[float, ...]
    expectation: ScenarioExpectation


@dataclasses.dataclass(frozen=True)
class GeneratedTrial:
    """
    Keep detector inputs separate from the planted intervention truth.
    """

    baseline_windows: Tuple[Tuple[rnmonmod.MonitoringObservation, ...], ...]
    post_windows: Tuple[Tuple[rnmonmod.MonitoringObservation, ...], ...]
    canary_families: Mapping[str, str]
    truth: TrialTruth


@dataclasses.dataclass(frozen=True)
class CapabilityTrialResult:
    """
    Store the blinded detector outcome for one seeded trial.
    """

    scenario_name: str
    seed: int
    detected: bool
    detection_window: Optional[int]
    queries_to_detection: Optional[int]
    final_state: str
    false_alarm: bool
    baseline_accuracy: float


@dataclasses.dataclass(frozen=True)
class CapabilityScenarioResult:
    """
    Aggregate repeated trials for one scenario.
    """

    scenario: CapabilityScenario
    total_trials: int
    detected_trials: int
    detection_rate: float
    false_alarm_rate: float
    insufficient_rate: float
    median_detection_window: Optional[float]
    p90_detection_window: Optional[int]
    accepted: Optional[bool]


@dataclasses.dataclass(frozen=True)
class CapabilityExperimentResult:
    """
    Store the primary Phase 1 experiment and every trial result.
    """

    protocol: CapabilityExperimentProtocol
    scenarios: Tuple[CapabilityScenarioResult, ...]
    trials: Tuple[CapabilityTrialResult, ...]

    @property
    def passed(self) -> bool:
        """
        Return whether every scenario with an acceptance gate passed.
        """
        gated = [
            scenario.accepted
            for scenario in self.scenarios
            if scenario.accepted is not None
        ]
        return bool(gated) and all(gated)


@dataclasses.dataclass(frozen=True)
class CapabilityPowerPoint:
    """
    Store one sample-size and baseline-length sensitivity result.
    """

    scenario_name: str
    observations_per_window: int
    baseline_windows: int
    trials: int
    detection_rate: float
    false_alarm_rate: float
    median_detection_window: Optional[float]


# #############################################################################
# Default scenarios
# #############################################################################


DEFAULT_CAPABILITY_SCENARIOS = (
    CapabilityScenario(
        "healthy_a_a",
        0.95,
        ScenarioExpectation.NO_ALERT,
    ),
    CapabilityScenario(
        "full_weak_model_swap",
        0.60,
        ScenarioExpectation.DETECT,
        maximum_detection_window=3,
    ),
    CapabilityScenario(
        "small_immaterial_drop",
        0.92,
        ScenarioExpectation.NO_ALERT,
    ),
    CapabilityScenario(
        "threshold_level_drop",
        0.85,
        ScenarioExpectation.MEASURE,
    ),
    CapabilityScenario(
        "gradual_degradation",
        None,
        ScenarioExpectation.DETECT,
        accuracy_drop_per_window=0.03,
        maximum_detection_window=8,
    ),
    CapabilityScenario(
        "partial_substitution_10",
        None,
        ScenarioExpectation.MEASURE,
        substitution_probability=0.10,
    ),
    CapabilityScenario(
        "partial_substitution_25",
        None,
        ScenarioExpectation.MEASURE,
        substitution_probability=0.25,
    ),
    CapabilityScenario(
        "partial_substitution_50",
        None,
        ScenarioExpectation.DETECT,
        substitution_probability=0.50,
        maximum_detection_window=5,
    ),
    CapabilityScenario(
        "latency_only",
        0.95,
        ScenarioExpectation.NO_ALERT,
        latency_multiplier=2.0,
    ),
    CapabilityScenario(
        "price_only",
        0.95,
        ScenarioExpectation.NO_ALERT,
        cost_multiplier=2.0,
    ),
    CapabilityScenario(
        "availability_failure",
        0.95,
        ScenarioExpectation.INSUFFICIENT,
        availability_probability=0.0,
    ),
    CapabilityScenario(
        "ungradable_evidence",
        0.95,
        ScenarioExpectation.INSUFFICIENT,
        gradable_probability=0.0,
    ),
)


# #############################################################################
# Deterministic generation
# #############################################################################


def _stream_seed(
    seed: int,
    scenario_name: str,
    phase: str,
    window_index: int,
    metric: str,
) -> int:
    material = "%s|%s|%s|%s|%s|%s" % (
        _SEED_VERSION,
        seed,
        scenario_name,
        phase,
        window_index,
        metric,
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _canary_layout(count: int) -> Tuple[Tuple[str, ...], Mapping[str, str]]:
    canary_ids = []
    canary_families = {}
    family_numbers = {family: 0 for family in CANARY_FAMILIES}
    for index in range(count):
        family = CANARY_FAMILIES[index % len(CANARY_FAMILIES)]
        family_numbers[family] += 1
        canary_id = "%s-v1-%03d" % (family, family_numbers[family])
        canary_ids.append(canary_id)
        canary_families[canary_id] = family
    return tuple(canary_ids), canary_families


def _scenario_accuracy(
    scenario: CapabilityScenario,
    protocol: CapabilityExperimentProtocol,
    post_window_index: int,
) -> float:
    if scenario.substitution_probability > 0:
        probability = scenario.substitution_probability
        return (
            1 - probability
        ) * protocol.healthy_accuracy + probability * protocol.weak_accuracy
    if scenario.post_accuracy is not None:
        return scenario.post_accuracy
    return max(
        protocol.weak_accuracy,
        protocol.healthy_accuracy
        - scenario.accuracy_drop_per_window * post_window_index,
    )


def _generate_window(
    canary_ids: Sequence[str],
    scenario: CapabilityScenario,
    accuracy: float,
    seed: int,
    phase: str,
    window_index: int,
) -> Tuple[rnmonmod.MonitoringObservation, ...]:
    availability_rng = random.Random(
        _stream_seed(
            seed,
            scenario.name,
            phase,
            window_index,
            "availability",
        )
    )
    gradable_rng = random.Random(
        _stream_seed(
            seed,
            scenario.name,
            phase,
            window_index,
            "gradable",
        )
    )
    quality_rng = random.Random(
        _stream_seed(
            seed,
            scenario.name,
            phase,
            window_index,
            "quality",
        )
    )
    observations = []
    for canary_id in canary_ids:
        available = availability_rng.random() < scenario.availability_probability
        if not available:
            observations.append(
                rnmonmod.MonitoringObservation(
                    canary_id=canary_id,
                    target=_TARGET,
                    provider_status=rnogapro.UPSTREAM_ERROR,
                    correct=None,
                    grader_reason="SIMULATED_PROVIDER_ERROR",
                    grader_version="offline-exact-match-v1",
                    latency_ms=_BASE_LATENCY_MS,
                    cost_usd=0.0,
                    error="simulated provider error",
                )
            )
            continue
        gradable = gradable_rng.random() < scenario.gradable_probability
        correct = quality_rng.random() < accuracy if gradable else None
        observations.append(
            rnmonmod.MonitoringObservation(
                canary_id=canary_id,
                target=_TARGET,
                provider_status=rnogapro.OK,
                correct=correct,
                grader_reason=(
                    "SIMULATED_MATCH"
                    if correct is True
                    else (
                        "SIMULATED_MISMATCH"
                        if correct is False
                        else "SIMULATED_UNGRADABLE"
                    )
                ),
                grader_version="offline-exact-match-v1",
                latency_ms=(_BASE_LATENCY_MS * scenario.latency_multiplier),
                cost_usd=_BASE_COST_USD * scenario.cost_multiplier,
                response="simulated response",
                served_by=_TARGET.provider_slug,
                model_served=_TARGET.model_id,
                prompt_tokens=10,
                completion_tokens=5,
                http_status=200,
            )
        )
    return tuple(observations)


def generate_capability_trial(
    protocol: CapabilityExperimentProtocol,
    scenario: CapabilityScenario,
    seed: int,
) -> GeneratedTrial:
    """
    Generate raw windows while keeping planted truth in a separate object.
    """
    canary_ids, canary_families = _canary_layout(
        protocol.observations_per_window
    )
    baseline_scenario = dataclasses.replace(
        scenario,
        post_accuracy=protocol.healthy_accuracy,
        substitution_probability=0.0,
        accuracy_drop_per_window=0.0,
        availability_probability=1.0,
        gradable_probability=1.0,
        latency_multiplier=1.0,
        cost_multiplier=1.0,
    )
    baseline_windows = tuple(
        _generate_window(
            canary_ids,
            baseline_scenario,
            protocol.healthy_accuracy,
            seed,
            "baseline",
            index,
        )
        for index in range(1, protocol.baseline_windows + 1)
    )
    post_accuracies = tuple(
        _scenario_accuracy(scenario, protocol, index)
        for index in range(1, protocol.post_windows + 1)
    )
    post_windows = tuple(
        _generate_window(
            canary_ids,
            scenario,
            accuracy,
            seed,
            "post",
            index,
        )
        for index, accuracy in enumerate(post_accuracies, start=1)
    )
    return GeneratedTrial(
        baseline_windows=baseline_windows,
        post_windows=post_windows,
        canary_families=canary_families,
        truth=TrialTruth(
            scenario_name=scenario.name,
            post_accuracies=post_accuracies,
            expectation=scenario.expectation,
        ),
    )


# #############################################################################
# Blinded execution and evaluation
# #############################################################################


def _series_identity(
    protocol: CapabilityExperimentProtocol,
) -> rnmondet.MonitoringSeriesIdentity:
    configuration = dataclasses.asdict(protocol)
    configuration_hash = hashlib.sha256(
        repr(sorted(configuration.items())).encode("utf-8")
    ).hexdigest()
    return rnmondet.MonitoringSeriesIdentity(
        provider_slug=_TARGET.provider_slug,
        model_id=_TARGET.model_id,
        bank_version="v1",
        grader_version="offline-exact-match-v1",
        configuration_hash=configuration_hash,
    )


def run_capability_trial(
    protocol: CapabilityExperimentProtocol,
    scenario: CapabilityScenario,
    seed: int,
) -> CapabilityTrialResult:
    """
    Run one generated trial without passing scenario truth to the detector.
    """
    generated = generate_capability_trial(protocol, scenario, seed)
    baseline_metrics = tuple(
        rnmonmet.aggregate_monitoring_window(
            observations,
            generated.canary_families,
            index,
        )
        for index, observations in enumerate(generated.baseline_windows, start=1)
    )
    policy = protocol.make_detection_policy()
    baseline = rnmondet.build_capability_baseline(
        _series_identity(protocol), baseline_metrics, policy
    )
    consecutive_flags = 0
    final_state = rnmondet.CapabilityState.OK
    detection_window = None
    for index, observations in enumerate(generated.post_windows, start=1):
        metrics = rnmonmet.aggregate_monitoring_window(
            observations,
            generated.canary_families,
            index,
        )
        detection = rnmondet.evaluate_capability_window(
            baseline,
            metrics,
            policy,
            previous_consecutive_flags=consecutive_flags,
        )
        final_state = detection.state
        consecutive_flags = detection.consecutive_flags
        if detection.state == rnmondet.CapabilityState.NERFED:
            detection_window = index
            break
    detected = detection_window is not None
    false_alarm = detected and generated.truth.expectation in (
        ScenarioExpectation.NO_ALERT,
        ScenarioExpectation.INSUFFICIENT,
    )
    return CapabilityTrialResult(
        scenario_name=scenario.name,
        seed=seed,
        detected=detected,
        detection_window=detection_window,
        queries_to_detection=(
            detection_window * protocol.observations_per_window
            if detection_window is not None
            else None
        ),
        final_state=final_state.value,
        false_alarm=false_alarm,
        baseline_accuracy=baseline.accuracy,
    )


def _nearest_rank(values: Sequence[int], percentile: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _scenario_acceptance(
    scenario: CapabilityScenario,
    detection_rate: float,
    false_alarm_rate: float,
    insufficient_rate: float,
    p90_detection_window: Optional[int],
) -> Optional[bool]:
    if scenario.expectation == ScenarioExpectation.MEASURE:
        return None
    if scenario.expectation == ScenarioExpectation.NO_ALERT:
        return false_alarm_rate <= 0.05
    if scenario.expectation == ScenarioExpectation.INSUFFICIENT:
        return detection_rate == 0 and insufficient_rate >= 0.95
    hdbg.dassert_eq(scenario.expectation, ScenarioExpectation.DETECT)
    within_delay = True
    if scenario.maximum_detection_window is not None:
        within_delay = (
            p90_detection_window is not None
            and p90_detection_window <= scenario.maximum_detection_window
        )
    return detection_rate >= 0.80 and within_delay


def _aggregate_scenario(
    scenario: CapabilityScenario,
    trials: Sequence[CapabilityTrialResult],
) -> CapabilityScenarioResult:
    hdbg.dassert(trials, "Scenario trials must not be empty")
    detected = [trial for trial in trials if trial.detected]
    detection_windows = [
        trial.detection_window
        for trial in detected
        if trial.detection_window is not None
    ]
    total = len(trials)
    detection_rate = len(detected) / total
    false_alarm_rate = sum(trial.false_alarm for trial in trials) / total
    insufficient_rate = (
        sum(
            trial.final_state == rnmondet.CapabilityState.INSUFFICIENT.value
            for trial in trials
        )
        / total
    )
    p90_detection_window = _nearest_rank(detection_windows, 0.90)
    return CapabilityScenarioResult(
        scenario=scenario,
        total_trials=total,
        detected_trials=len(detected),
        detection_rate=detection_rate,
        false_alarm_rate=false_alarm_rate,
        insufficient_rate=insufficient_rate,
        median_detection_window=(
            statistics.median(detection_windows) if detection_windows else None
        ),
        p90_detection_window=p90_detection_window,
        accepted=_scenario_acceptance(
            scenario,
            detection_rate,
            false_alarm_rate,
            insufficient_rate,
            p90_detection_window,
        ),
    )


def run_capability_experiment(
    protocol: CapabilityExperimentProtocol,
    scenarios: Sequence[CapabilityScenario] = DEFAULT_CAPABILITY_SCENARIOS,
) -> CapabilityExperimentResult:
    """
    Run every scenario across deterministic seeds and aggregate results.
    """
    hdbg.dassert(scenarios, "Scenarios must not be empty")
    all_trials = []
    scenario_results = []
    for scenario in scenarios:
        trials = tuple(
            run_capability_trial(protocol, scenario, seed)
            for seed in range(protocol.trials_per_scenario)
        )
        all_trials.extend(trials)
        scenario_results.append(_aggregate_scenario(scenario, trials))
    return CapabilityExperimentResult(
        protocol=protocol,
        scenarios=tuple(scenario_results),
        trials=tuple(all_trials),
    )


def run_capability_power_analysis(
    protocol: CapabilityExperimentProtocol,
    trials: int,
    *,
    window_sizes: Sequence[int] = (20, 40, 60, 80, 100),
    baseline_window_counts: Sequence[int] = (3, 5, 10),
) -> Tuple[CapabilityPowerPoint, ...]:
    """
    Measure detector sensitivity across query and baseline sizes.
    """
    if trials < 1:
        raise ValueError("Power-analysis trials must be positive")
    selected_names = {
        "healthy_a_a",
        "full_weak_model_swap",
        "threshold_level_drop",
        "partial_substitution_50",
    }
    selected_scenarios = [
        scenario
        for scenario in DEFAULT_CAPABILITY_SCENARIOS
        if scenario.name in selected_names
    ]
    points = []
    for window_size in window_sizes:
        for baseline_windows in baseline_window_counts:
            varied = dataclasses.replace(
                protocol,
                trials_per_scenario=trials,
                observations_per_window=window_size,
                baseline_windows=baseline_windows,
            )
            result = run_capability_experiment(varied, selected_scenarios)
            for scenario_result in result.scenarios:
                points.append(
                    CapabilityPowerPoint(
                        scenario_name=scenario_result.scenario.name,
                        observations_per_window=window_size,
                        baseline_windows=baseline_windows,
                        trials=trials,
                        detection_rate=scenario_result.detection_rate,
                        false_alarm_rate=scenario_result.false_alarm_rate,
                        median_detection_window=(
                            scenario_result.median_detection_window
                        ),
                    )
                )
    return tuple(points)
