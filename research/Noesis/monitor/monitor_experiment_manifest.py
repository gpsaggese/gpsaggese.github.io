"""
Freeze the reviewed Phase 0 model-drift campaign configuration.

Import as:

import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
"""

import dataclasses
import hashlib
import json
from typing import Dict, Tuple


@dataclasses.dataclass(frozen=True)
class RouteSpec:
    """
    Identify one provider-pinned route and its research role.
    """

    name: str
    provider_slug: str
    expected_provider_name: str
    model_id: str
    role: str
    precision: str
    allow_fallbacks: bool = False

    def __post_init__(self) -> None:
        for name, value in dataclasses.asdict(self).items():
            if name == "allow_fallbacks":
                continue
            if not value.strip():
                raise ValueError("%s must not be empty" % name)
        if self.allow_fallbacks:
            raise ValueError("Phase 0 routes must disable fallbacks")


@dataclasses.dataclass(frozen=True)
class BenchmarkFamily:
    """
    Freeze one capability family and its provisional source mix.
    """

    name: str
    bank_items: int
    items_per_window: int
    sources: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("family name must not be empty")
        if self.bank_items < 1 or self.items_per_window < 1:
            raise ValueError("benchmark item counts must be positive")
        if self.items_per_window > self.bank_items:
            raise ValueError("window cannot exceed the family bank")
        if not self.sources or any(not item.strip() for item in self.sources):
            raise ValueError("benchmark sources must not be empty")


@dataclasses.dataclass(frozen=True)
class BudgetStage:
    """
    Bound calls and spend for one separately approved live stage.
    """

    name: str
    max_calls: int
    max_cost_usd: float
    requires_gp_approval: bool

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("budget stage name must not be empty")
        if self.max_calls < 0 or self.max_cost_usd < 0:
            raise ValueError("budget limits must not be negative")


@dataclasses.dataclass(frozen=True)
class StopPolicy:
    """
    Freeze conditions that halt a live campaign before more calls.
    """

    timeout_count: int = 2
    consecutive_provider_errors: int = 2
    maximum_error_rate: float = 0.20
    require_attribution: bool = True
    require_usage_metadata: bool = True
    require_durable_persistence: bool = True

    def __post_init__(self) -> None:
        if self.timeout_count < 1 or self.consecutive_provider_errors < 1:
            raise ValueError("stop counts must be positive")
        if not 0 <= self.maximum_error_rate <= 1:
            raise ValueError("maximum_error_rate must be between zero and one")


@dataclasses.dataclass(frozen=True)
class FingerprintPolicy:
    """
    Freeze the first Model Equality Testing configuration.
    """

    implementation_commit: str
    statistic: str = "mmd_hamming"
    pvalue_method: str = "permutation_pvalue"
    character_length: int = 1000
    permutations: int = 1000
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if len(self.implementation_commit) != 40:
            raise ValueError("implementation_commit must be a full Git SHA")
        if self.character_length < 1 or self.permutations < 1:
            raise ValueError("fingerprint sizes must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")


@dataclasses.dataclass(frozen=True)
class PhaseZeroManifest:
    """
    Store the complete reviewed inputs for the Phase 0 research campaign.
    """

    schema_version: str
    campaign_id: str
    bank_version: str
    routes: Tuple[RouteSpec, ...]
    benchmark_families: Tuple[BenchmarkFamily, ...]
    budget_stages: Tuple[BudgetStage, ...]
    stop_policy: StopPolicy
    fingerprint_policy: FingerprintPolicy
    capability_window_size: int = 60
    execution_enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("schema_version", "campaign_id", "bank_version"):
            if not getattr(self, name).strip():
                raise ValueError("%s must not be empty" % name)
        if not self.routes or not self.benchmark_families:
            raise ValueError("routes and benchmark families must not be empty")
        if len({route.name for route in self.routes}) != len(self.routes):
            raise ValueError("route names must be unique")
        family_window = sum(
            family.items_per_window for family in self.benchmark_families
        )
        if family_window != self.capability_window_size:
            raise ValueError(
                "family window counts must sum to capability_window_size"
            )


def phase_zero_manifest_to_dict(manifest: PhaseZeroManifest) -> Dict[str, object]:
    """
    Convert a manifest into a stable JSON-compatible record.
    """
    return dataclasses.asdict(manifest)


def render_phase_zero_manifest(manifest: PhaseZeroManifest) -> str:
    """
    Render a manifest as deterministic JSON.
    """
    return json.dumps(
        phase_zero_manifest_to_dict(manifest), indent=2, sort_keys=True
    ) + "\n"


def phase_zero_manifest_hash(manifest: PhaseZeroManifest) -> str:
    """
    Return the SHA-256 identity of the canonical manifest.
    """
    encoded = json.dumps(
        phase_zero_manifest_to_dict(manifest),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


DEFAULT_PHASE_ZERO_MANIFEST = PhaseZeroManifest(
    schema_version="noesis-model-drift-phase0-v1",
    campaign_id="noesis-qos-phase0-controlled-v1",
    bank_version="five-family-pilot-v1",
    routes=(
        RouteSpec(
            "cloudflare-llama-3.3-70b",
            "cloudflare",
            "Cloudflare",
            "meta-llama/llama-3.3-70b-instruct",
            "primary-reference",
            "fp8",
        ),
        RouteSpec(
            "cloudflare-llama-3.1-8b",
            "cloudflare",
            "Cloudflare",
            "meta-llama/llama-3.1-8b-instruct",
            "controlled-substitution",
            "provider-reported",
        ),
        RouteSpec(
            "cloudflare-llama-3.2-3b",
            "cloudflare",
            "Cloudflare",
            "meta-llama/llama-3.2-3b-instruct",
            "controlled-substitution",
            "provider-reported",
        ),
        RouteSpec(
            "coreweave-llama-3.3-70b",
            "coreweave",
            "CoreWeave",
            "meta-llama/llama-3.3-70b-instruct",
            "independent-provider-control",
            "fp16",
        ),
    ),
    benchmark_families=(
        BenchmarkFamily(
            "math", 60, 12, ("GSM-Symbolic-style", "hidden-generated")
        ),
        BenchmarkFamily(
            "expert-science", 60, 12, ("GPQA", "hidden-equivalents")
        ),
        BenchmarkFamily(
            "code-tracing", 60, 12, ("CRUXEval-O", "hidden-generated")
        ),
        BenchmarkFamily(
            "instruction-following", 60, 12, ("IFEval", "hidden-constraints")
        ),
        BenchmarkFamily(
            "multi-step-reasoning", 60, 12, ("BBH-style", "hidden-generated")
        ),
    ),
    budget_stages=(
        BudgetStage("offline-validation", 0, 0.0, False),
        BudgetStage("attribution-smoke", 15, 0.25, True),
        BudgetStage("controlled-warm-up", 240, 10.0, True),
    ),
    stop_policy=StopPolicy(),
    fingerprint_policy=FingerprintPolicy(
        implementation_commit="fd2ee24d75c9fef87debff8caefa0c04d4a5d374"
    ),
)
