"""
Detect material capability degradation against a frozen monitoring baseline.

Import as:

import research.Noesis.monitor.monitor_detection as rnmondet
"""

import dataclasses
import enum
import functools
import math
from typing import Optional, Sequence, Tuple

import helpers.hdbg as hdbg

import research.Noesis.monitor.monitor_metrics as rnmonmet

# #############################################################################
# Detection models
# #############################################################################


class CapabilityState(str, enum.Enum):
    """
    Represent the result of one capability-window comparison.
    """

    OK = "OK"
    WARNING = "WARNING"
    NERFED = "NERFED"
    INSUFFICIENT = "INSUFFICIENT"


@dataclasses.dataclass(frozen=True)
class MonitoringSeriesIdentity:
    """
    Identify the exact series to which one baseline belongs.
    """

    provider_slug: str
    model_id: str
    bank_version: str
    grader_version: str
    configuration_hash: str

    def __post_init__(self) -> None:
        for name, value in dataclasses.asdict(self).items():
            if not value.strip():
                raise ValueError("%s must not be empty" % name)


@dataclasses.dataclass(frozen=True)
class CapabilityDetectionPolicy:
    """
    Freeze statistical and practical capability-degradation thresholds.
    """

    alpha: float = 0.01
    minimum_quality_drop: float = 0.10
    minimum_graded_per_window: int = 60
    minimum_baseline_graded: int = 300
    confirmation_windows: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if not 0 <= self.minimum_quality_drop <= 1:
            raise ValueError("minimum_quality_drop must be between zero and one")
        if self.minimum_graded_per_window < 1:
            raise ValueError("minimum_graded_per_window must be positive")
        if self.minimum_baseline_graded < 1:
            raise ValueError("minimum_baseline_graded must be positive")
        if self.confirmation_windows < 1:
            raise ValueError("confirmation_windows must be positive")


@dataclasses.dataclass(frozen=True)
class BaselineFamilyCounts:
    """
    Store frozen baseline counts for one canary family.
    """

    family: str
    gradable: int
    correct: int
    incorrect: int


@dataclasses.dataclass(frozen=True)
class CapabilityBaseline:
    """
    Store the frozen healthy reference for one monitoring series.
    """

    identity: MonitoringSeriesIdentity
    window_count: int
    gradable: int
    correct: int
    incorrect: int
    accuracy: float
    families: Tuple[BaselineFamilyCounts, ...]


@dataclasses.dataclass(frozen=True)
class CapabilityDetection:
    """
    Store one detector decision and the evidence supporting it.
    """

    state: CapabilityState
    window_index: int
    baseline_accuracy: float
    current_accuracy: Optional[float]
    quality_drop: Optional[float]
    p_value: Optional[float]
    statistically_significant: bool
    practically_significant: bool
    consecutive_flags: int
    reason: str


# #############################################################################
# Exact statistics
# #############################################################################


@functools.cache
def one_sided_fisher_quality_p_value(
    baseline_correct: int,
    baseline_incorrect: int,
    current_correct: int,
    current_incorrect: int,
) -> float:
    """
    Return the exact probability of current quality this low or lower.

    The calculation conditions on both group sizes and the total number of
    correct responses. It is the lower tail of Fisher's exact test.
    """
    counts = (
        baseline_correct,
        baseline_incorrect,
        current_correct,
        current_incorrect,
    )
    if any(count < 0 for count in counts):
        raise ValueError("Fisher counts must not be negative")
    baseline_total = baseline_correct + baseline_incorrect
    current_total = current_correct + current_incorrect
    if baseline_total == 0 or current_total == 0:
        raise ValueError("Fisher test needs two non-empty groups")
    population = baseline_total + current_total
    total_correct = baseline_correct + current_correct
    minimum_current_correct = max(
        0, current_total - (population - total_correct)
    )
    maximum_current_correct = min(current_total, total_correct)
    if not minimum_current_correct <= current_correct <= maximum_current_correct:
        raise ValueError("Observed Fisher table is impossible")
    denominator = math.comb(population, current_total)
    probability = 0.0
    for value in range(minimum_current_correct, current_correct + 1):
        probability += (
            math.comb(total_correct, value)
            * math.comb(
                population - total_correct,
                current_total - value,
            )
            / denominator
        )
    return min(1.0, probability)


# #############################################################################
# Baselines and decisions
# #############################################################################


def build_capability_baseline(
    identity: MonitoringSeriesIdentity,
    windows: Sequence[rnmonmet.MonitoringWindowMetrics],
    policy: CapabilityDetectionPolicy,
) -> CapabilityBaseline:
    """
    Freeze sufficient healthy window metrics into one capability baseline.
    """
    hdbg.dassert(windows, "Baseline windows must not be empty")
    gradable = sum(window.gradable for window in windows)
    if gradable < policy.minimum_baseline_graded:
        raise ValueError(
            "Baseline needs at least %s graded responses, found %s"
            % (policy.minimum_baseline_graded, gradable)
        )
    correct = sum(window.correct for window in windows)
    family_counts = {}
    for window in windows:
        for family in window.families:
            counts = family_counts.setdefault(family.family, [0, 0])
            counts[0] += family.gradable
            counts[1] += family.correct
    families = tuple(
        BaselineFamilyCounts(
            family=family,
            gradable=family_counts[family][0],
            correct=family_counts[family][1],
            incorrect=(family_counts[family][0] - family_counts[family][1]),
        )
        for family in sorted(family_counts)
    )
    return CapabilityBaseline(
        identity=identity,
        window_count=len(windows),
        gradable=gradable,
        correct=correct,
        incorrect=gradable - correct,
        accuracy=correct / gradable,
        families=families,
    )


def evaluate_capability_window(
    baseline: CapabilityBaseline,
    current: rnmonmet.MonitoringWindowMetrics,
    policy: CapabilityDetectionPolicy,
    previous_consecutive_flags: int = 0,
) -> CapabilityDetection:
    """
    Compare one current window with the baseline without mutating state.
    """
    if previous_consecutive_flags < 0:
        raise ValueError("previous_consecutive_flags must not be negative")
    if current.gradable < policy.minimum_graded_per_window:
        return CapabilityDetection(
            state=CapabilityState.INSUFFICIENT,
            window_index=current.window_index,
            baseline_accuracy=baseline.accuracy,
            current_accuracy=current.accuracy,
            quality_drop=None,
            p_value=None,
            statistically_significant=False,
            practically_significant=False,
            consecutive_flags=0,
            reason="Fewer than %s graded responses"
            % policy.minimum_graded_per_window,
        )
    hdbg.dassert_is_not(current.accuracy, None)
    quality_drop = baseline.accuracy - current.accuracy
    p_value = one_sided_fisher_quality_p_value(
        baseline.correct,
        baseline.incorrect,
        current.correct,
        current.incorrect,
    )
    statistically_significant = p_value < policy.alpha
    practically_significant = quality_drop >= policy.minimum_quality_drop
    flagged = statistically_significant and practically_significant
    consecutive_flags = previous_consecutive_flags + 1 if flagged else 0
    if not flagged:
        state = CapabilityState.OK
        reason = "Window did not meet both detection thresholds"
    elif consecutive_flags < policy.confirmation_windows:
        state = CapabilityState.WARNING
        reason = "Material degradation needs an independent confirmation"
    else:
        state = CapabilityState.NERFED
        reason = "Material degradation was independently confirmed"
    return CapabilityDetection(
        state=state,
        window_index=current.window_index,
        baseline_accuracy=baseline.accuracy,
        current_accuracy=current.accuracy,
        quality_drop=quality_drop,
        p_value=p_value,
        statistically_significant=statistically_significant,
        practically_significant=practically_significant,
        consecutive_flags=consecutive_flags,
        reason=reason,
    )
