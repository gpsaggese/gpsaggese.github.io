"""
Run blinded behavioral-fingerprint controls with Model Equality Testing.

Import as:

import research.Noesis.monitor.monitor_fingerprint_experiment as rnmonfiex
"""

import dataclasses
import enum
import hashlib
import math
import os
import random
import statistics
import subprocess
import sys
import types
from typing import Callable, Optional, Sequence, Tuple


_SEED_VERSION = "fingerprint-replay-v1"


class FingerprintExpectation(str, enum.Enum):
    """
    Describe the acceptance rule for one planted fingerprint scenario.
    """

    NO_ALERT = "NO_ALERT"
    DETECT = "DETECT"
    MEASURE = "MEASURE"


@dataclasses.dataclass(frozen=True)
class CompletionTriple:
    """
    Hold two independent reference draws and one alternative draw.
    """

    record_id: str
    prompt_id: int
    reference_a: str
    reference_b: str
    alternative: str

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id must not be empty")
        if self.prompt_id < 0:
            raise ValueError("prompt_id must not be negative")
        for name in ("reference_a", "reference_b", "alternative"):
            if not getattr(self, name):
                raise ValueError("%s must not be empty" % name)


@dataclasses.dataclass(frozen=True)
class CompletionPool:
    """
    Store the detector-visible prompt and completion arrays.
    """

    label: str
    prompt_ids: Tuple[int, ...]
    completions: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("pool label must not be empty")
        if not self.prompt_ids or len(self.prompt_ids) != len(self.completions):
            raise ValueError("pool arrays must have one or more aligned rows")
        if any(prompt_id < 0 for prompt_id in self.prompt_ids):
            raise ValueError("prompt ids must not be negative")
        if any(not completion for completion in self.completions):
            raise ValueError("completions must not be empty")


@dataclasses.dataclass(frozen=True)
class FingerprintProtocol:
    """
    Freeze one behavioral-fingerprint validation protocol.
    """

    trials_per_scenario: int = 100
    prompt_count: int = 5
    samples_per_prompt: int = 12
    character_length: int = 1000
    permutations: int = 1000
    alpha: float = 0.05
    implementation_commit: str = (
        "fd2ee24d75c9fef87debff8caefa0c04d4a5d374"
    )

    def __post_init__(self) -> None:
        for name in (
            "trials_per_scenario",
            "prompt_count",
            "samples_per_prompt",
            "character_length",
            "permutations",
        ):
            if getattr(self, name) < 1:
                raise ValueError("%s must be positive" % name)
        if self.samples_per_prompt < 2:
            raise ValueError("each prompt needs at least two samples")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if len(self.implementation_commit) != 40:
            raise ValueError("implementation_commit must be a full Git SHA")

    @property
    def sample_size(self) -> int:
        """
        Return the number of completions in each comparison pool.
        """
        return self.prompt_count * self.samples_per_prompt


@dataclasses.dataclass(frozen=True)
class FingerprintScenario:
    """
    Describe one exact-count alternative-completion mixture.
    """

    name: str
    substitution_fraction: float
    expectation: FingerprintExpectation

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must not be empty")
        if not 0 <= self.substitution_fraction <= 1:
            raise ValueError("substitution_fraction must be between zero and one")


@dataclasses.dataclass(frozen=True)
class FingerprintTestResult:
    """
    Store the output of one two-sample test.
    """

    p_value: float
    statistic: float

    def __post_init__(self) -> None:
        if not 0 <= self.p_value <= 1:
            raise ValueError("p_value must be between zero and one")


FingerprintTest = Callable[
    [CompletionPool, CompletionPool, FingerprintProtocol, int],
    FingerprintTestResult,
]


@dataclasses.dataclass(frozen=True)
class FingerprintTrialResult:
    """
    Store one seeded, blinded test outcome.
    """

    scenario_name: str
    seed: int
    substituted_count: int
    p_value: float
    statistic: float
    detected: bool
    false_alarm: bool


@dataclasses.dataclass(frozen=True)
class FingerprintScenarioResult:
    """
    Aggregate repeated trials for one scenario.
    """

    scenario: FingerprintScenario
    total_trials: int
    substituted_count: int
    detection_rate: float
    false_alarm_rate: float
    median_p_value: float
    accepted: Optional[bool]


@dataclasses.dataclass(frozen=True)
class FingerprintExperimentResult:
    """
    Store every result from the fingerprint control experiment.
    """

    protocol: FingerprintProtocol
    scenarios: Tuple[FingerprintScenarioResult, ...]
    trials: Tuple[FingerprintTrialResult, ...]

    @property
    def passed(self) -> bool:
        """
        Return whether all scenarios with gates passed.
        """
        gated = [item.accepted for item in self.scenarios if item.accepted is not None]
        return bool(gated) and all(gated)


DEFAULT_FINGERPRINT_SCENARIOS = (
    FingerprintScenario("healthy_a_a", 0.0, FingerprintExpectation.NO_ALERT),
    FingerprintScenario("full_model_swap", 1.0, FingerprintExpectation.DETECT),
    FingerprintScenario(
        "partial_substitution_10", 0.10, FingerprintExpectation.MEASURE
    ),
    FingerprintScenario(
        "partial_substitution_25", 0.25, FingerprintExpectation.MEASURE
    ),
    FingerprintScenario(
        "partial_substitution_50", 0.50, FingerprintExpectation.DETECT
    ),
)


def _stream_seed(seed: int, stream: str) -> int:
    material = "%s|%s|%s" % (_SEED_VERSION, seed, stream)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _selection_score(seed: int, value: str) -> bytes:
    """Return a stable pseudo-random rank for one mixture candidate."""
    material = "%s|%s|mixture|%s" % (_SEED_VERSION, seed, value)
    return hashlib.sha256(material.encode("utf-8")).digest()


def _completion_text(prompt_id: int, rng: random.Random, alternative: bool) -> str:
    if alternative:
        phrases = (
            "A terse shortcut gives the result",
            "The alternate system returns this answer",
            "Using a compact heuristic, the output is",
        )
        alphabet = "uvwxyz"
    else:
        phrases = (
            "A careful analysis gives the result",
            "Working through each step, the answer is",
            "Using the stated constraints, the output is",
        )
        alphabet = "abcdef"
    noise = "".join(rng.choice(alphabet) for _ in range(24))
    value = rng.randrange(1000)
    return "prompt-%02d: %s %03d [%s]" % (
        prompt_id,
        rng.choice(phrases),
        value,
        noise,
    )


def generate_synthetic_completion_triples(
    protocol: FingerprintProtocol,
    seed: int,
) -> Tuple[CompletionTriple, ...]:
    """
    Generate deterministic controls without making provider calls.

    The strings validate experiment plumbing only. They are not evidence of
    detection power on real model outputs.
    """
    reference_a_rng = random.Random(_stream_seed(seed, "reference-a"))
    reference_b_rng = random.Random(_stream_seed(seed, "reference-b"))
    alternative_rng = random.Random(_stream_seed(seed, "alternative"))
    records = []
    for prompt_id in range(protocol.prompt_count):
        for sample_index in range(protocol.samples_per_prompt):
            records.append(
                CompletionTriple(
                    record_id="p%03d-s%03d" % (prompt_id, sample_index),
                    prompt_id=prompt_id,
                    reference_a=_completion_text(
                        prompt_id, reference_a_rng, alternative=False
                    ),
                    reference_b=_completion_text(
                        prompt_id, reference_b_rng, alternative=False
                    ),
                    alternative=_completion_text(
                        prompt_id, alternative_rng, alternative=True
                    ),
                )
            )
    return tuple(records)


def build_comparison_pools(
    records: Sequence[CompletionTriple],
    substitution_fraction: float,
    seed: int,
) -> Tuple[CompletionPool, CompletionPool, Tuple[str, ...]]:
    """
    Build independent reference and exact-count candidate pools.

    The detector receives only the two pools. The returned substituted record
    identifiers are evaluator truth and must not be passed to it.
    """
    if not records:
        raise ValueError("records must not be empty")
    if not 0 <= substitution_fraction <= 1:
        raise ValueError("substitution_fraction must be between zero and one")
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("record ids must be unique")
    substituted_count = round(len(records) * substitution_fraction)
    prompt_indices = {}
    for index, record in enumerate(records):
        prompt_indices.setdefault(record.prompt_id, []).append(index)
    prompt_quotas = {
        prompt_id: math.floor(len(indices) * substitution_fraction)
        for prompt_id, indices in prompt_indices.items()
    }
    remaining = substituted_count - sum(prompt_quotas.values())
    prompt_order = sorted(
        prompt_indices,
        key=lambda prompt_id: (
            -(
                len(prompt_indices[prompt_id]) * substitution_fraction
                - prompt_quotas[prompt_id]
            ),
            _selection_score(seed, "prompt-%s" % prompt_id),
        ),
    )
    for prompt_id in prompt_order[:remaining]:
        prompt_quotas[prompt_id] += 1
    substituted_indices = set()
    for prompt_id, indices in prompt_indices.items():
        ranked = sorted(
            indices,
            key=lambda index: _selection_score(seed, records[index].record_id),
        )
        substituted_indices.update(ranked[: prompt_quotas[prompt_id]])
    reference = CompletionPool(
        label="reference-a",
        prompt_ids=tuple(record.prompt_id for record in records),
        completions=tuple(record.reference_a for record in records),
    )
    candidate = CompletionPool(
        label="candidate",
        prompt_ids=tuple(record.prompt_id for record in records),
        completions=tuple(
            record.alternative
            if index in substituted_indices
            else record.reference_b
            for index, record in enumerate(records)
        ),
    )
    substituted_ids = tuple(
        records[index].record_id for index in sorted(substituted_indices)
    )
    return reference, candidate, substituted_ids


def run_official_hamming_mmd(
    reference: CompletionPool,
    candidate: CompletionPool,
    protocol: FingerprintProtocol,
    seed: int,
) -> FingerprintTestResult:
    """
    Run the pinned reference package's Hamming-MMD permutation test.

    The caller must install the package from the commit frozen in the campaign
    manifest. Imports stay inside this adapter so the rest of Noesis has no
    runtime dependency on the research package.
    """
    _prepare_reference_import(protocol.implementation_commit)
    try:
        import numpy as np
        import torch
        import model_equality_testing.distribution as met_distribution
        import model_equality_testing.tests as met_tests
        from model_equality_testing.algorithm import run_two_sample_test
        from model_equality_testing.distribution import CompletionSample
        from model_equality_testing.utils import pad_to_length, tokenize_unicode
    except ImportError as error:
        raise RuntimeError(
            "The pinned model-equality-testing reference package is required"
        ) from error
    if reference.prompt_ids != candidate.prompt_ids:
        raise ValueError("comparison pools must use the same prompt layout")
    maximum_prompt = max(reference.prompt_ids)
    prompt_count = maximum_prompt + 1

    def _completion_sample(pool: CompletionPool) -> object:
        texts = [text[: protocol.character_length] for text in pool.completions]
        tokenized = tokenize_unicode(texts)
        padded = pad_to_length(tokenized, L=protocol.character_length)
        return CompletionSample(
            prompts=np.asarray(pool.prompt_ids),
            completions=padded,
            m=prompt_count,
        )

    reference_sample = _completion_sample(reference)
    candidate_sample = _completion_sample(candidate)
    torch_state = torch.random.get_rng_state()
    official_statistic = met_tests.IMPLEMENTED_TESTS["mmd_hamming"]
    clear_sequence_cache = met_distribution.CompletionSample.sequences.fget.cache_clear

    def _memory_safe_hamming_mmd(*args, **kwargs):
        """Release the reference package's instance cache after every permutation."""
        try:
            return official_statistic(*args, **kwargs)
        finally:
            clear_sequence_cache()

    try:
        # The pinned reference package decorates an instance property with
        # functools.cache. A permutation test creates 2,000 short-lived sample
        # objects, so that global cache otherwise retains every tensor until the
        # process exits. Keep the exact statistic but clear the cache after each
        # permutation to bound resident memory.
        met_tests.IMPLEMENTED_TESTS["mmd_hamming"] = _memory_safe_hamming_mmd
        torch.manual_seed(_stream_seed(seed, "permutation") % (2**63 - 1))
        p_value, statistic = run_two_sample_test(
            reference_sample,
            candidate_sample,
            pvalue_type="permutation_pvalue",
            stat_type="mmd_hamming",
            b=protocol.permutations,
        )
    finally:
        met_tests.IMPLEMENTED_TESTS["mmd_hamming"] = official_statistic
        clear_sequence_cache()
        torch.random.set_rng_state(torch_state)
    return FingerprintTestResult(float(p_value), float(statistic))


def _prepare_reference_import(expected_commit: str) -> None:
    """
    Load the reference package only from a checkout at the frozen commit.

    The authors' repository nests its Python package under
    ``model_equality_testing/src``. Loading it as a namespace avoids copying or
    modifying the reference implementation.
    """
    existing = sys.modules.get("model_equality_testing")
    if existing is not None:
        loaded_commit = getattr(existing, "__noesis_commit__", None)
        if loaded_commit != expected_commit:
            raise RuntimeError("Loaded reference package has an unknown commit")
        return
    repository = os.environ.get("NOESIS_MODEL_EQUALITY_REPO", "")
    if not repository:
        raise RuntimeError(
            "Set NOESIS_MODEL_EQUALITY_REPO to the pinned reference checkout"
        )
    revision = subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected_commit:
        raise RuntimeError(
            "Reference checkout is %s, expected %s"
            % (revision, expected_commit)
        )
    source_dir = os.path.join(repository, "model_equality_testing", "src")
    if not os.path.isfile(os.path.join(source_dir, "algorithm.py")):
        raise RuntimeError("Reference checkout has an unexpected layout")
    package = types.ModuleType("model_equality_testing")
    package.__path__ = [source_dir]
    package.__package__ = "model_equality_testing"
    package.__noesis_commit__ = revision
    sys.modules["model_equality_testing"] = package


def run_fingerprint_trial(
    protocol: FingerprintProtocol,
    scenario: FingerprintScenario,
    seed: int,
    test_func: FingerprintTest = run_official_hamming_mmd,
) -> FingerprintTrialResult:
    """
    Run one seeded trial without exposing planted labels to the detector.
    """
    records = generate_synthetic_completion_triples(protocol, seed)
    reference, candidate, substituted_ids = build_comparison_pools(
        records, scenario.substitution_fraction, seed
    )
    test_result = test_func(reference, candidate, protocol, seed)
    detected = test_result.p_value < protocol.alpha
    false_alarm = detected and scenario.expectation == FingerprintExpectation.NO_ALERT
    return FingerprintTrialResult(
        scenario_name=scenario.name,
        seed=seed,
        substituted_count=len(substituted_ids),
        p_value=test_result.p_value,
        statistic=test_result.statistic,
        detected=detected,
        false_alarm=false_alarm,
    )


def _scenario_acceptance(
    scenario: FingerprintScenario,
    detection_rate: float,
    false_alarm_rate: float,
    alpha: float,
) -> Optional[bool]:
    if scenario.expectation == FingerprintExpectation.MEASURE:
        return None
    if scenario.expectation == FingerprintExpectation.NO_ALERT:
        return false_alarm_rate <= alpha
    return detection_rate >= 0.80


def run_fingerprint_experiment(
    protocol: FingerprintProtocol,
    scenarios: Sequence[FingerprintScenario] = DEFAULT_FINGERPRINT_SCENARIOS,
    test_func: FingerprintTest = run_official_hamming_mmd,
) -> FingerprintExperimentResult:
    """
    Run deterministic synthetic controls through a supplied two-sample test.
    """
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    scenario_results = []
    all_trials = []
    for scenario in scenarios:
        trials = tuple(
            run_fingerprint_trial(protocol, scenario, seed, test_func)
            for seed in range(protocol.trials_per_scenario)
        )
        all_trials.extend(trials)
        detection_rate = sum(trial.detected for trial in trials) / len(trials)
        false_alarm_rate = sum(trial.false_alarm for trial in trials) / len(trials)
        scenario_results.append(
            FingerprintScenarioResult(
                scenario=scenario,
                total_trials=len(trials),
                substituted_count=trials[0].substituted_count,
                detection_rate=detection_rate,
                false_alarm_rate=false_alarm_rate,
                median_p_value=statistics.median(
                    trial.p_value for trial in trials
                ),
                accepted=_scenario_acceptance(
                    scenario,
                    detection_rate,
                    false_alarm_rate,
                    protocol.alpha,
                ),
            )
        )
    return FingerprintExperimentResult(
        protocol=protocol,
        scenarios=tuple(scenario_results),
        trials=tuple(all_trials),
    )
