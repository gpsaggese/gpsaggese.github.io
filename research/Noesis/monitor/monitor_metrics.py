"""
Aggregate standalone monitoring observations into immutable window metrics.

Import as:

import research.Noesis.monitor.monitor_metrics as rnmonmet
"""

import dataclasses
import math
import statistics
from typing import Mapping, Optional, Sequence, Tuple

import helpers.hdbg as hdbg

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_models as rnmonmod

# #############################################################################
# Metric models
# #############################################################################


@dataclasses.dataclass(frozen=True)
class FamilyCapabilityMetrics:
    """
    Store capability counts for one canary family in one window.
    """

    family: str
    attempted: int
    gradable: int
    correct: int
    incorrect: int
    ungradable: int
    accuracy: Optional[float]


@dataclasses.dataclass(frozen=True)
class MonitoringWindowMetrics:
    """
    Store source-independent metrics for one monitoring window.
    """

    window_index: int
    attempted: int
    returned: int
    gradable: int
    correct: int
    incorrect: int
    ungradable: int
    accuracy: Optional[float]
    availability: Optional[float]
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    total_cost_usd: float
    cost_per_task_usd: Optional[float]
    cost_per_correct_task_usd: Optional[float]
    families: Tuple[FamilyCapabilityMetrics, ...]


# #############################################################################
# Metric helpers
# #############################################################################


def _accuracy(correct: int, gradable: int) -> Optional[float]:
    """
    Return an accuracy or `None` when no response was gradable.
    """
    if gradable == 0:
        return None
    return correct / gradable


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    """
    Return the nearest-rank percentile for a non-empty sequence.
    """
    if not values:
        return None
    hdbg.dassert_lte(0.0, percentile)
    hdbg.dassert_lte(percentile, 1.0)
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _family_metrics(
    family: str,
    observations: Sequence[rnmonmod.MonitoringObservation],
) -> FamilyCapabilityMetrics:
    """
    Aggregate one family of monitoring observations.
    """
    attempted = len(observations)
    gradable = sum(item.correct is not None for item in observations)
    correct = sum(item.correct is True for item in observations)
    incorrect = gradable - correct
    return FamilyCapabilityMetrics(
        family=family,
        attempted=attempted,
        gradable=gradable,
        correct=correct,
        incorrect=incorrect,
        ungradable=attempted - gradable,
        accuracy=_accuracy(correct, gradable),
    )


# #############################################################################
# Window aggregation
# #############################################################################


def aggregate_monitoring_window(
    observations: Sequence[rnmonmod.MonitoringObservation],
    canary_families: Mapping[str, str],
    window_index: int,
) -> MonitoringWindowMetrics:
    """
    Aggregate raw monitoring observations without making a verdict.

    :param observations: Raw observations from simulation or live monitoring
    :param canary_families: Mapping from canary identifier to family
    :param window_index: One-based position in the monitoring series
    :return: Immutable metrics for the window
    """
    hdbg.dassert_lte(1, window_index, "window_index must be positive")
    hdbg.dassert(observations, "observations must not be empty")
    family_observations = {}
    for observation in observations:
        hdbg.dassert_in(
            observation.canary_id,
            canary_families,
            "Every observation must have a canary family",
        )
        family = canary_families[observation.canary_id]
        family_observations.setdefault(family, []).append(observation)
    attempted = len(observations)
    returned_observations = [
        item for item in observations if item.provider_status == rnogapro.OK
    ]
    gradable = sum(item.correct is not None for item in observations)
    correct = sum(item.correct is True for item in observations)
    incorrect = gradable - correct
    total_cost_usd = sum(item.cost_usd for item in observations)
    cost_per_correct_task_usd = None
    if correct > 0:
        cost_per_correct_task_usd = total_cost_usd / correct
    return MonitoringWindowMetrics(
        window_index=window_index,
        attempted=attempted,
        returned=len(returned_observations),
        gradable=gradable,
        correct=correct,
        incorrect=incorrect,
        ungradable=attempted - gradable,
        accuracy=_accuracy(correct, gradable),
        availability=len(returned_observations) / attempted,
        latency_p50_ms=(
            statistics.median(
                [item.latency_ms for item in returned_observations]
            )
            if returned_observations
            else None
        ),
        latency_p95_ms=_percentile(
            [item.latency_ms for item in returned_observations], 0.95
        ),
        total_cost_usd=total_cost_usd,
        cost_per_task_usd=total_cost_usd / attempted,
        cost_per_correct_task_usd=cost_per_correct_task_usd,
        families=tuple(
            _family_metrics(family, family_observations[family])
            for family in sorted(family_observations)
        ),
    )
