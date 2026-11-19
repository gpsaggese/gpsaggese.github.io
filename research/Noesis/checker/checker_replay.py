"""
Generate deterministic synthetic observations for offline checker replay.

Import as:

import research.Noesis.checker.checker_replay as rnochrep
"""

import dataclasses
import hashlib
import logging
import random
from typing import Tuple

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.gateway.gateway_providers as rnogapro

_LOG = logging.getLogger(__name__)

_SEED_VERSION = "pr-s6-v1"
_ON_TIME_LATENCY_MULTIPLIER = 0.8
_SLOW_LATENCY_MULTIPLIER = 1.2


# #############################################################################
# Replay scenarios
# #############################################################################


@dataclasses.dataclass(frozen=True)
class ReplayScenario:
    """
    Describe one controlled population used by the replay experiment.
    """

    name: str
    buyer_requests: int
    canaries: int
    latency_compliance_probability: float
    quality_probability: float

    def __post_init__(self) -> None:
        hdbg.dassert(self.name, "Scenario name must not be empty")
        hdbg.dassert_lte(
            0,
            self.buyer_requests,
            "Buyer request count must not be negative",
        )
        hdbg.dassert_lte(
            0,
            self.canaries,
            "Canary count must not be negative",
        )
        hdbg.dassert_is_proportion(
            self.latency_compliance_probability,
            "Latency compliance probability must be in [0, 1]",
        )
        hdbg.dassert_is_proportion(
            self.quality_probability,
            "Quality probability must be in [0, 1]",
        )


DEFAULT_REPLAY_SCENARIOS = (
    ReplayScenario("healthy", 50, 10, 0.97, 0.98),
    ReplayScenario("slow_only", 50, 10, 0.60, 0.98),
    ReplayScenario("weak_model_only", 50, 10, 0.97, 0.50),
    ReplayScenario("borderline_quality", 50, 10, 0.97, 0.80),
    ReplayScenario("borderline_latency", 50, 10, 0.85, 0.98),
    ReplayScenario("insufficient_evidence", 19, 7, 1.00, 1.00),
)


# #############################################################################
# Deterministic observation generation
# #############################################################################


def _stream_seed(base_seed: int, scenario_name: str, metric: str) -> int:
    """
    Derive a stable random seed without Python's randomized ``hash()``.
    """
    material = "%s|%s|%s|%s" % (
        _SEED_VERSION,
        base_seed,
        scenario_name,
        metric,
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class ReplayObservationGenerator:
    """
    Generate reproducible buyer and canary rows from one replay scenario.

    Latency and quality use separate pseudo-random streams, so adding draws to
    one metric cannot silently alter the other metric's observations.
    """

    def __init__(
        self,
        scenario: ReplayScenario,
        base_seed: int,
        *,
        latency_threshold_ms: float = 1500.0,
        first_request_id: int = 1,
    ) -> None:
        hdbg.dassert_isinstance(
            scenario,
            ReplayScenario,
            "scenario must be a ReplayScenario",
        )
        hdbg.dassert_isinstance(base_seed, int, "base_seed must be an integer")
        hdbg.dassert(
            not isinstance(base_seed, bool),
            "base_seed must not be a Boolean",
        )
        hdbg.dassert_lt(
            0,
            latency_threshold_ms,
            "Latency threshold must be positive",
        )
        hdbg.dassert_lte(
            1,
            first_request_id,
            "First request ID must be positive",
        )
        self._scenario = scenario
        self._latency_threshold_ms = latency_threshold_ms
        self._next_request_id = first_request_id
        self._latency_rng = random.Random(
            _stream_seed(base_seed, scenario.name, "latency")
        )
        self._quality_rng = random.Random(
            _stream_seed(base_seed, scenario.name, "quality")
        )

    def _allocate_request_id(self) -> int:
        """
        Return the next deterministic request ID.
        """
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def generate(self) -> Tuple[rnochmod.RequestObservation, ...]:
        """
        Generate one contract's raw buyer and canary observations.
        """
        _LOG.debug(hprint.to_str("self._scenario.name self._next_request_id"))
        observations = []
        for _ in range(self._scenario.buyer_requests):
            on_time = (
                self._latency_rng.random()
                < self._scenario.latency_compliance_probability
            )
            multiplier = (
                _ON_TIME_LATENCY_MULTIPLIER
                if on_time
                else _SLOW_LATENCY_MULTIPLIER
            )
            observations.append(
                rnochmod.RequestObservation(
                    request_id=self._allocate_request_id(),
                    status=rnogapro.OK,
                    latency_ms=self._latency_threshold_ms * multiplier,
                )
            )
        canary_latency_ms = (
            self._latency_threshold_ms * _ON_TIME_LATENCY_MULTIPLIER
        )
        for _ in range(self._scenario.canaries):
            correct = (
                self._quality_rng.random() < self._scenario.quality_probability
            )
            observations.append(
                rnochmod.RequestObservation(
                    request_id=self._allocate_request_id(),
                    status=rnogapro.OK,
                    latency_ms=canary_latency_ms,
                    is_canary=True,
                    canary_correct=correct,
                )
            )
        result = tuple(observations)
        _LOG.debug("return=%s", result)
        return result
