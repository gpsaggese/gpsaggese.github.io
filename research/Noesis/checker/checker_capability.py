"""
Define the pluggable capability-measurement boundary.

Import as:

import research.Noesis.checker.checker_capability as rnochcap
"""

from typing import Protocol

import research.Noesis.checker.checker_grading as rnochgra
import research.Noesis.checker.checker_models as rnochmod


class CapabilityMeasurer(Protocol):
    """
    Measure one response using an interchangeable implementation.

    A future verifier model or hosted evaluator can satisfy this protocol
    without changing calibration, probing, routing, or persistence.
    """

    @property
    def version(self) -> str:
        """
        Return the version persisted with every measurement.
        """
        ...

    def measure(
        self, canary: rnochmod.Canary, response: str
    ) -> rnochmod.GradeResult:
        """
        Measure one prompt/response pair against its canary contract.
        """
        ...


class DeterministicBenchmarkMeasurer:
    """
    Implement capability measurement with the existing exact-match grader.
    """

    @property
    def version(self) -> str:
        """
        Return the deterministic grader version.
        """
        return rnochgra.GRADER_VERSION

    def measure(
        self, canary: rnochmod.Canary, response: str
    ) -> rnochmod.GradeResult:
        """
        Normalize and grade one deterministic canary answer.
        """
        return rnochgra.grade_response(canary, response)


DEFAULT_MEASURER = DeterministicBenchmarkMeasurer()
