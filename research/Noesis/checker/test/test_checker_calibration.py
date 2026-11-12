"""
Test deterministic canary calibration decisions.
"""

from typing import Dict, Tuple

import helpers.hunit_test as hunitest

import research.Noesis.checker.checker_calibration as rnochcal
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.gateway.gateway_providers as rnogapro


def _canary(identifier: str) -> rnochmod.Canary:
    return rnochmod.Canary(
        identifier,
        "v1",
        "math",
        "Return one integer.",
        "1",
        "integer",
    )


def _targets() -> Tuple[rnochmod.CalibrationTarget, ...]:
    return (
        rnochmod.CalibrationTarget(
            rnochmod.CalibrationRole.HEALTHY,
            "groq",
            "healthy-model",
        ),
        rnochmod.CalibrationTarget(
            rnochmod.CalibrationRole.HEALTHY,
            "together",
            "healthy-model",
        ),
        rnochmod.CalibrationTarget(
            rnochmod.CalibrationRole.WEAK,
            "groq",
            "weak-model",
        ),
    )


def _observations(
    canary: rnochmod.Canary,
    target_answers: Dict[rnochmod.CalibrationTarget, Tuple[bool, ...]],
) -> Tuple[rnochmod.CalibrationObservation, ...]:
    observations = []
    for target, answers in target_answers.items():
        for attempt, correct in enumerate(answers, start=1):
            observation = rnochmod.CalibrationObservation(
                canary.canary_id,
                target,
                attempt,
                rnogapro.OK,
                correct,
                100.0,
                0.001,
            )
            observations.append(observation)
    return tuple(observations)


# #############################################################################
# Test_evaluate_canaries
# #############################################################################


class Test_evaluate_canaries(hunitest.TestCase):
    """
    Test `checker_calibration.evaluate_canaries()`.
    """

    def test1(self) -> None:
        """
        Accept a stable healthy question that discriminates the weak model.
        """
        # Prepare inputs.
        canary = _canary("math-v1-001")
        targets = _targets()
        answers = {
            targets[0]: (True, True, True),
            targets[1]: (True, True, True),
            targets[2]: (True, False, False),
        }
        observations = _observations(canary, answers)
        # Prepare outputs.
        expected = rnochmod.CanaryCalibration(
            "math-v1-001",
            1.0,
            1 / 3,
            True,
            rnochcal.CALIBRATED,
        )
        # Run test.
        actual = rnochcal.evaluate_canaries((canary,), targets, observations, 3)
        # Check outputs.
        self.assertEqual(actual, (expected,))

    def test2(self) -> None:
        """
        Reject a question that the healthy targets do not always answer.
        """
        # Prepare inputs.
        canary = _canary("math-v1-001")
        targets = _targets()
        answers = {
            targets[0]: (True, True, False),
            targets[1]: (True, True, True),
            targets[2]: (False, False, False),
        }
        observations = _observations(canary, answers)
        # Run test.
        actual = rnochcal.evaluate_canaries((canary,), targets, observations, 3)
        # Check outputs.
        self.assertFalse(actual[0].accepted)
        self.assertEqual(actual[0].reason, rnochcal.HEALTHY_ACCURACY_TOO_LOW)
        self.assertAlmostEqual(actual[0].healthy_accuracy, 5 / 6)

    def test3(self) -> None:
        """
        Reject a question that is too easy for the weak model.
        """
        # Prepare inputs.
        canary = _canary("math-v1-001")
        targets = _targets()
        answers = {
            targets[0]: (True, True, True),
            targets[1]: (True, True, True),
            targets[2]: (True, True, False),
        }
        observations = _observations(canary, answers)
        # Run test.
        actual = rnochcal.evaluate_canaries((canary,), targets, observations, 3)
        # Check outputs.
        self.assertFalse(actual[0].accepted)
        self.assertEqual(actual[0].reason, rnochcal.WEAK_ACCURACY_TOO_HIGH)
        self.assertAlmostEqual(actual[0].weak_accuracy, 2 / 3)

    def test4(self) -> None:
        """
        Reject incomplete observations without treating them as wrong answers.
        """
        # Prepare inputs.
        canary = _canary("math-v1-001")
        targets = _targets()
        answers = {
            targets[0]: (True, True, True),
            targets[1]: (True, True, True),
            targets[2]: (False, False),
        }
        observations = _observations(canary, answers)
        # Run test.
        actual = rnochcal.evaluate_canaries((canary,), targets, observations, 3)
        # Check outputs.
        self.assertFalse(actual[0].accepted)
        self.assertEqual(actual[0].reason, rnochcal.INCOMPLETE_OBSERVATIONS)
        self.assertIsNone(actual[0].healthy_accuracy)
        self.assertIsNone(actual[0].weak_accuracy)

    def test5(self) -> None:
        """
        Reject a provider failure without treating it as a weak-model miss.
        """
        # Prepare inputs.
        canary = _canary("math-v1-001")
        targets = _targets()
        answers = {
            targets[0]: (True, True, True),
            targets[1]: (True, True, True),
            targets[2]: (False, False, False),
        }
        observations = list(_observations(canary, answers))
        failed = observations[-1]
        observations[-1] = rnochmod.CalibrationObservation(
            failed.canary_id,
            failed.target,
            failed.attempt,
            rnogapro.GATEWAY_ERROR,
            None,
            failed.latency_ms,
            failed.cost_usd,
        )
        # Run test.
        actual = rnochcal.evaluate_canaries((canary,), targets, observations, 3)
        # Check outputs.
        self.assertFalse(actual[0].accepted)
        self.assertEqual(actual[0].reason, rnochcal.INCOMPLETE_OBSERVATIONS)
        self.assertIsNone(actual[0].healthy_accuracy)
        self.assertIsNone(actual[0].weak_accuracy)
