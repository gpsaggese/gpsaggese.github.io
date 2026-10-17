"""
Select reliable canaries from graded healthy- and weak-model observations.

Import as:

import research.Noesis.checker.checker_calibration as rnochcal
"""

import collections
from typing import Dict, Iterable, List, Sequence, Tuple

import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.gateway.gateway_providers as rnogapro

CALIBRATED = "CALIBRATED"
INCOMPLETE_OBSERVATIONS = "INCOMPLETE_OBSERVATIONS"
HEALTHY_ACCURACY_TOO_LOW = "HEALTHY_ACCURACY_TOO_LOW"
WEAK_ACCURACY_TOO_HIGH = "WEAK_ACCURACY_TOO_HIGH"


class CalibrationError(ValueError):
    """
    Signal inconsistent calibration inputs rather than a model-quality result.
    """


def _validate_inputs(
    canaries: Sequence[rnochmod.Canary],
    targets: Sequence[rnochmod.CalibrationTarget],
    observations: Sequence[rnochmod.CalibrationObservation],
    repetitions: int,
    healthy_required_accuracy: float,
    weak_maximum_accuracy: float,
) -> None:
    """
    Validate one complete calibration-analysis input set.
    """
    if repetitions < 1:
        raise CalibrationError("repetitions must be positive")
    for name, value in (
        ("healthy_required_accuracy", healthy_required_accuracy),
        ("weak_maximum_accuracy", weak_maximum_accuracy),
    ):
        if not 0 <= value <= 1:
            raise CalibrationError("%s must be between zero and one" % name)
    canary_ids = [canary.canary_id for canary in canaries]
    if not canary_ids:
        raise CalibrationError("calibration needs at least one canary")
    if len(canary_ids) != len(set(canary_ids)):
        raise CalibrationError("calibration contains duplicate canary ids")
    if not targets:
        raise CalibrationError("calibration needs at least one target")
    if len(targets) != len(set(targets)):
        raise CalibrationError("calibration contains duplicate targets")
    healthy_targets = [
        target
        for target in targets
        if target.role == rnochmod.CalibrationRole.HEALTHY
    ]
    weak_targets = [
        target
        for target in targets
        if target.role == rnochmod.CalibrationRole.WEAK
    ]
    if not healthy_targets:
        raise CalibrationError("calibration needs at least one healthy target")
    if len(weak_targets) != 1:
        raise CalibrationError("calibration needs exactly one weak target")
    expected_canaries = set(canary_ids)
    expected_targets = set(targets)
    seen = set()
    for observation in observations:
        if observation.canary_id not in expected_canaries:
            raise CalibrationError(
                "observation references unknown canary %r"
                % observation.canary_id
            )
        if observation.target not in expected_targets:
            raise CalibrationError("observation references an unknown target")
        if observation.attempt > repetitions:
            raise CalibrationError(
                "observation attempt %s exceeds repetitions %s"
                % (observation.attempt, repetitions)
            )
        key = (observation.canary_id, observation.target, observation.attempt)
        if key in seen:
            raise CalibrationError(
                "calibration contains a duplicate observation"
            )
        seen.add(key)


def _accuracy(
    observations: Sequence[rnochmod.CalibrationObservation],
) -> float:
    """
    Return accuracy for a complete set of usable observations.
    """
    correct = sum(observation.correct is True for observation in observations)
    return correct / len(observations)


def evaluate_canaries(
    canaries: Sequence[rnochmod.Canary],
    targets: Sequence[rnochmod.CalibrationTarget],
    observations: Iterable[rnochmod.CalibrationObservation],
    repetitions: int,
    *,
    healthy_required_accuracy: float = 1.0,
    weak_maximum_accuracy: float = 0.6,
) -> Tuple[rnochmod.CanaryCalibration, ...]:
    """
    Evaluate each candidate against complete healthy and weak observations.

    A provider failure, grader failure, or missing attempt makes the candidate
    incomplete. It is not interpreted as evidence that the target model lacks
    capability.

    :param canaries: candidate bank records
    :param targets: healthy provider/model targets and one weak target
    :param observations: already-graded provider attempts
    :param repetitions: required attempts for every canary-target pair
    :param healthy_required_accuracy: minimum combined healthy accuracy
    :param weak_maximum_accuracy: maximum weak-model accuracy
    :return: one stable, canary-id-ordered decision per candidate
    """
    observations = tuple(observations)
    _validate_inputs(
        canaries,
        targets,
        observations,
        repetitions,
        healthy_required_accuracy,
        weak_maximum_accuracy,
    )
    grouped: Dict[
        Tuple[str, rnochmod.CalibrationTarget],
        List[rnochmod.CalibrationObservation],
    ] = collections.defaultdict(list)
    for observation in observations:
        grouped[(observation.canary_id, observation.target)].append(observation)
    healthy_targets = tuple(
        target
        for target in targets
        if target.role == rnochmod.CalibrationRole.HEALTHY
    )
    weak_target = next(
        target
        for target in targets
        if target.role == rnochmod.CalibrationRole.WEAK
    )
    decisions = []
    for canary in sorted(canaries, key=lambda item: item.canary_id):
        healthy = tuple(
            observation
            for target in healthy_targets
            for observation in grouped[(canary.canary_id, target)]
        )
        weak = tuple(grouped[(canary.canary_id, weak_target)])
        expected_healthy = repetitions * len(healthy_targets)
        complete = len(healthy) == expected_healthy and len(weak) == repetitions
        usable = complete and all(
            observation.provider_status == rnogapro.OK
            and observation.correct is not None
            for observation in healthy + weak
        )
        if not usable:
            decision = rnochmod.CanaryCalibration(
                canary.canary_id,
                None,
                None,
                False,
                INCOMPLETE_OBSERVATIONS,
            )
            decisions.append(decision)
            continue
        healthy_accuracy = _accuracy(healthy)
        weak_accuracy = _accuracy(weak)
        if healthy_accuracy < healthy_required_accuracy:
            accepted = False
            reason = HEALTHY_ACCURACY_TOO_LOW
        elif weak_accuracy > weak_maximum_accuracy:
            accepted = False
            reason = WEAK_ACCURACY_TOO_HIGH
        else:
            accepted = True
            reason = CALIBRATED
        decision = rnochmod.CanaryCalibration(
            canary.canary_id,
            healthy_accuracy,
            weak_accuracy,
            accepted,
            reason,
        )
        decisions.append(decision)
    return tuple(decisions)
