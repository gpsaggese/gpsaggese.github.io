"""
Run deterministic offline observations through the production checker logic.

Import as:

import research.Noesis.checker.checker_replay_runner as rnochrer
"""

import dataclasses
import datetime
import logging
import math
from typing import Dict

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.checker.checker_evidence as rnochevi
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_replay as rnochrep
import research.Noesis.checker.checker_statistics as rnochsta
import research.Noesis.checker.checker_verdicts as rnochver
import research.Noesis.checker.checker_windows as rnochwin

_LOG = logging.getLogger(__name__)

_WINDOW_START = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
_TARGET_REACHED_SECONDS = 30


# #############################################################################
# Replay configuration and results
# #############################################################################


@dataclasses.dataclass(frozen=True)
class ReplayExperimentConfig:
    """
    Freeze contract and checker settings used by offline replay.
    """

    l_max_s: float = 1.5
    r_min: float = 0.90
    q_min: float = 0.85
    target_requests: int = 50
    target_canaries: int = 10
    maximum_seconds: int = 60
    minimum_requests: int = 20
    minimum_canaries: int = 8
    confidence: float = 0.95
    exact_bound_below: int = 10

    def __post_init__(self) -> None:
        hdbg.dassert_lt(0, self.l_max_s, "l_max_s must be positive")
        hdbg.dassert_is_proportion(self.r_min, "r_min must be in [0, 1]")
        hdbg.dassert_is_proportion(self.q_min, "q_min must be in [0, 1]")
        hdbg.dassert_lte(
            1, self.target_requests, "target_requests must be positive"
        )
        hdbg.dassert_lte(
            1, self.target_canaries, "target_canaries must be positive"
        )
        hdbg.dassert_lte(
            1, self.maximum_seconds, "maximum_seconds must be positive"
        )
        hdbg.dassert_lte(
            1, self.minimum_requests, "minimum_requests must be positive"
        )
        hdbg.dassert_lte(
            1, self.minimum_canaries, "minimum_canaries must be positive"
        )
        hdbg.dassert_lt(0.5, self.confidence, "confidence must exceed 0.5")
        hdbg.dassert_lt(self.confidence, 1, "confidence must be below 1")
        hdbg.dassert_lte(
            1, self.exact_bound_below, "exact_bound_below must be positive"
        )

    def make_policy(self) -> rnochmod.WindowPolicy:
        """
        Build the production checker policy represented by this configuration.
        """
        return rnochmod.WindowPolicy(
            minimum_requests=self.minimum_requests,
            minimum_canaries=self.minimum_canaries,
            confidence=self.confidence,
            include_canaries_in_latency=False,
            exact_bound_below=self.exact_bound_below,
        )

    def make_contract(self, contract_id: int) -> rnochmod.ContractWindow:
        """
        Build one production contract window for offline evaluation.
        """
        return rnochmod.ContractWindow(
            terms=rnochmod.ContractTerms(
                contract_id,
                self.l_max_s,
                self.r_min,
                self.q_min,
            ),
            state="ACTIVE",
            target_requests=self.target_requests,
            target_canaries=self.target_canaries,
            maximum_seconds=self.maximum_seconds,
            checker_policy="m2-replay-v1",
            window_start=_WINDOW_START,
            window_end=None,
            seller_id="offline-replay-seller",
        )


DEFAULT_REPLAY_CONFIG = ReplayExperimentConfig()


@dataclasses.dataclass(frozen=True)
class ReplayContractOutcome:
    """
    Store one synthetic contract's production readiness and verdict outputs.
    """

    contract_id: int
    readiness: rnochmod.WindowReadiness
    decision: rnochmod.VerdictDecision


@dataclasses.dataclass(frozen=True)
class AnalyticVerdictRates:
    """
    Store exact binomial verdict probabilities for one replay scenario.
    """

    passed: float
    failed: float
    insufficient: float
    latency_failed: float
    quality_failed: float


@dataclasses.dataclass(frozen=True)
class ReplayRunSummary:
    """
    Store aggregate observed and analytic results for one scenario and seed.
    """

    scenario_name: str
    seed: int
    contracts: int
    verdict_counts: Dict[str, int]
    latency_failure_count: int
    quality_failure_count: int
    both_failure_count: int
    observed_failure_rate: float
    analytic_rates: AnalyticVerdictRates


_MINIMUM_ACCEPTANCE_TOLERANCE = 0.005
_STANDARD_ERROR_MULTIPLIER = 4


# #############################################################################
# Analytic predictions
# #############################################################################


def _binomial_probability(
    successes: int,
    total: int,
    probability: float,
) -> float:
    """
    Return the probability of one binomial success count.
    """
    return (
        math.comb(total, successes)
        * probability**successes
        * (1 - probability) ** (total - successes)
    )


def metric_failure_probability(
    total: int,
    true_rate: float,
    promised_rate: float,
    minimum_evidence: int,
    policy: rnochmod.WindowPolicy,
) -> float:
    """
    Return the exact probability that the production metric rule fails.
    """
    hdbg.dassert_lte(0, total, "total must not be negative")
    hdbg.dassert_is_proportion(true_rate, "true_rate must be in [0, 1]")
    hdbg.dassert_is_proportion(
        promised_rate,
        "promised_rate must be in [0, 1]",
    )
    hdbg.dassert_lte(1, minimum_evidence, "minimum_evidence must be positive")
    if total < minimum_evidence:
        return 0.0
    probability = 0.0
    for successes in range(total + 1):
        upper_bound = rnochsta.upper_confidence_bound(
            successes,
            total,
            confidence=policy.confidence,
            exact_below=policy.exact_bound_below,
        )
        if rnochsta.is_conclusive_failure(upper_bound, promised_rate):
            probability += _binomial_probability(successes, total, true_rate)
    return probability


def analytic_verdict_rates(
    scenario: rnochrep.ReplayScenario,
    *,
    config: ReplayExperimentConfig = DEFAULT_REPLAY_CONFIG,
) -> AnalyticVerdictRates:
    """
    Return exact expected verdict rates for one independent replay scenario.
    """
    policy = config.make_policy()
    latency_failed = metric_failure_probability(
        scenario.buyer_requests,
        scenario.latency_compliance_probability,
        config.r_min,
        config.minimum_requests,
        policy,
    )
    quality_failed = metric_failure_probability(
        scenario.canaries,
        scenario.quality_probability,
        config.q_min,
        config.minimum_canaries,
        policy,
    )
    no_failure = (1 - latency_failed) * (1 - quality_failed)
    failed = 1 - no_failure
    evidence_sufficient = (
        scenario.buyer_requests >= config.minimum_requests
        and scenario.canaries >= config.minimum_canaries
    )
    passed = no_failure if evidence_sufficient else 0.0
    insufficient = 0.0 if evidence_sufficient else no_failure
    return AnalyticVerdictRates(
        passed=passed,
        failed=failed,
        insufficient=insufficient,
        latency_failed=latency_failed,
        quality_failed=quality_failed,
    )


def replay_acceptance_tolerance(
    expected_rate: float,
    contracts: int,
) -> float:
    """
    Return a conservative tolerance for a finite replay experiment.

    The tolerance is four binomial standard errors with a 0.5 percentage
    point floor.  Deterministic 0% and 100% scenarios therefore still have a
    clear non-zero numerical tolerance, while stochastic scenarios scale with
    their Monte Carlo uncertainty.
    """
    hdbg.dassert_is_proportion(
        expected_rate,
        "expected_rate must be in [0, 1]",
    )
    hdbg.dassert_lte(1, contracts, "contracts must be positive")
    standard_error = math.sqrt(expected_rate * (1 - expected_rate) / contracts)
    return max(
        _MINIMUM_ACCEPTANCE_TOLERANCE,
        _STANDARD_ERROR_MULTIPLIER * standard_error,
    )


def agrees_with_analytic_prediction(summary: ReplayRunSummary) -> bool:
    """
    Return whether observed failures agree with the analytic prediction.
    """
    difference = abs(
        summary.observed_failure_rate - summary.analytic_rates.failed
    )
    tolerance = replay_acceptance_tolerance(
        summary.analytic_rates.failed,
        summary.contracts,
    )
    return difference <= tolerance


# #############################################################################
# Production checker replay
# #############################################################################


def evaluate_replay_contract(
    scenario: rnochrep.ReplayScenario,
    generator: rnochrep.ReplayObservationGenerator,
    contract_id: int,
    *,
    config: ReplayExperimentConfig = DEFAULT_REPLAY_CONFIG,
) -> ReplayContractOutcome:
    """
    Evaluate one synthetic contract through production checker functions.
    """
    _LOG.debug(hprint.to_str("scenario.name contract_id"))
    contract = config.make_contract(contract_id)
    observations = generator.generate()
    policy = config.make_policy()
    evidence = rnochevi.build_evidence(observations, contract.terms, policy)
    targets_reached = (
        scenario.buyer_requests >= config.target_requests
        and scenario.canaries >= config.target_canaries
    )
    elapsed_seconds = (
        _TARGET_REACHED_SECONDS if targets_reached else config.maximum_seconds
    )
    now = _WINDOW_START + datetime.timedelta(seconds=elapsed_seconds)
    readiness = rnochwin.assess_window_readiness(contract, evidence, now)
    hdbg.dassert(
        readiness.ready,
        "Replay scenario '%s' did not close its contract window",
        scenario.name,
    )
    decision = rnochver.decide_verdict(evidence, contract.terms, policy)
    result = ReplayContractOutcome(contract_id, readiness, decision)
    _LOG.debug("return=%s", result)
    return result


def run_replay_scenario(
    scenario: rnochrep.ReplayScenario,
    seed: int,
    contracts: int,
    *,
    config: ReplayExperimentConfig = DEFAULT_REPLAY_CONFIG,
) -> ReplayRunSummary:
    """
    Run and aggregate repeated synthetic contracts for one scenario and seed.
    """
    _LOG.debug(hprint.to_str("scenario.name seed contracts"))
    hdbg.dassert_lte(1, contracts, "contracts must be positive")
    generator = rnochrep.ReplayObservationGenerator(
        scenario,
        seed,
        latency_threshold_ms=config.l_max_s * 1000,
    )
    verdict_counts = {verdict.value: 0 for verdict in rnochmod.Verdict}
    latency_failure_count = 0
    quality_failure_count = 0
    both_failure_count = 0
    for contract_id in range(1, contracts + 1):
        outcome = evaluate_replay_contract(
            scenario,
            generator,
            contract_id,
            config=config,
        )
        decision = outcome.decision
        verdict_counts[decision.verdict.value] += 1
        latency_failed = decision.latency.conclusive_failure
        quality_failed = decision.quality.conclusive_failure
        latency_failure_count += latency_failed
        quality_failure_count += quality_failed
        both_failure_count += latency_failed and quality_failed
    result = ReplayRunSummary(
        scenario_name=scenario.name,
        seed=seed,
        contracts=contracts,
        verdict_counts=verdict_counts,
        latency_failure_count=latency_failure_count,
        quality_failure_count=quality_failure_count,
        both_failure_count=both_failure_count,
        observed_failure_rate=(
            verdict_counts[rnochmod.Verdict.FAILED.value] / contracts
        ),
        analytic_rates=analytic_verdict_rates(scenario, config=config),
    )
    _LOG.debug("return=%s", result)
    return result
