"""
Build and validate the paid Lane 2 behavioral-fingerprint pilot.

Import as:

import research.Noesis.monitor.monitor_live_fingerprint as rnmonlifp
"""

import dataclasses
import hashlib
import json
from typing import Dict, Mapping, Sequence, Tuple

import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
import research.Noesis.monitor.monitor_fingerprint_experiment as rnmonfiex
import research.Noesis.monitor.monitor_paid_execution as rnmonpaex


REFERENCE_ROUTE = "cloudflare-llama-3.3-70b"
ALTERNATIVE_ROUTE = "cloudflare-llama-3.1-8b"
POOL_ROUTES = (
    ("reference-a", REFERENCE_ROUTE),
    ("reference-b", REFERENCE_ROUTE),
    ("alternative", ALTERNATIVE_ROUTE),
)

LIVE_FINGERPRINT_MANIFEST = dataclasses.replace(
    rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST,
    schema_version="noesis-model-drift-phase0-live-fingerprint-v1",
    campaign_id="noesis-qos-phase0-live-fingerprint-v1",
    bank_version="fingerprint-five-prompt-v1",
    budget_stages=rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST.budget_stages
    + (
        rnmonexma.BudgetStage(
            "live-fingerprint-pilot", 180, 2.0, True
        ),
    ),
)


@dataclasses.dataclass(frozen=True)
class LiveFingerprintPrompt:
    """Store one reviewed prompt in the live fingerprint distribution."""

    prompt_id: str
    prompt: str

    def __post_init__(self) -> None:
        if not self.prompt_id.strip() or not self.prompt.strip():
            raise ValueError("fingerprint prompt fields must not be empty")
        if len(self.prompt) > 20000:
            raise ValueError("fingerprint prompts must not exceed 20000 characters")


@dataclasses.dataclass(frozen=True)
class LiveFingerprintProtocol:
    """Freeze the first paid fingerprint collection policy."""

    prompt_count: int = 5
    samples_per_prompt: int = 12
    pool_count: int = 3
    max_tokens: int = 384
    timeout_s: float = 60.0
    sampling_temperature: float = 1.0
    maximum_cost_usd: float = 2.0
    per_call_reserve_usd: float = 0.01

    def __post_init__(self) -> None:
        for name in ("prompt_count", "samples_per_prompt", "pool_count", "max_tokens"):
            if getattr(self, name) < 1:
                raise ValueError("%s must be positive" % name)
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not 0 <= self.sampling_temperature <= 2:
            raise ValueError("sampling_temperature must be between zero and two")
        if self.pool_count != len(POOL_ROUTES):
            raise ValueError("live fingerprint pilot requires exactly three pools")

    @property
    def pool_size(self) -> int:
        """Return the planned responses in each comparison pool."""
        return self.prompt_count * self.samples_per_prompt

    @property
    def call_count(self) -> int:
        """Return the exact provider-call count."""
        return self.pool_size * self.pool_count

    def make_execution_policy(self) -> rnmonpaex.PaidExecutionPolicy:
        """Build the shared paid-execution safety policy."""
        return rnmonpaex.PaidExecutionPolicy(
            maximum_calls=self.call_count,
            maximum_cost_usd=self.maximum_cost_usd,
            per_call_reserve_usd=self.per_call_reserve_usd,
        )

    def make_detector_protocol(self) -> rnmonfiex.FingerprintProtocol:
        """Build the detector settings frozen by the Phase 0 manifest."""
        return rnmonfiex.FingerprintProtocol(
            trials_per_scenario=100,
            prompt_count=self.prompt_count,
            samples_per_prompt=self.samples_per_prompt,
            character_length=1000,
            permutations=1000,
            alpha=0.05,
        )


def load_live_fingerprint_bank(path: str) -> Tuple[LiveFingerprintPrompt, ...]:
    """Load a reviewed JSONL fingerprint prompt bank."""
    prompts = []
    with open(path, encoding="utf-8") as file_:
        for line_number, line in enumerate(file_, start=1):
            if not line.strip():
                continue
            try:
                prompts.append(LiveFingerprintPrompt(**json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    "invalid fingerprint bank row %s: %s" % (line_number, error)
                ) from error
    return tuple(prompts)


def validate_live_fingerprint_bank(
    prompts: Sequence[LiveFingerprintPrompt],
    protocol: LiveFingerprintProtocol,
) -> None:
    """Require the exact five-prompt live distribution."""
    if len(prompts) != protocol.prompt_count:
        raise ValueError(
            "fingerprint bank must contain exactly %s prompts"
            % protocol.prompt_count
        )
    prompt_ids = [item.prompt_id for item in prompts]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("fingerprint prompt ids must be unique")


def live_fingerprint_bank_hash(
    prompts: Sequence[LiveFingerprintPrompt],
) -> str:
    """Return the stable identity of the ordered prompt bank."""
    encoded = json.dumps(
        [dataclasses.asdict(item) for item in prompts],
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _order_key(
    bank_hash: str,
    pool_name: str,
    prompt_id: str,
    sample_index: int,
) -> bytes:
    material = "live-fingerprint-v1|%s|%s|%s|%s" % (
        bank_hash,
        pool_name,
        prompt_id,
        sample_index,
    )
    return hashlib.sha256(material.encode("utf-8")).digest()


def build_live_fingerprint_calls(
    prompts: Sequence[LiveFingerprintPrompt],
    manifest: rnmonexma.PhaseZeroManifest = LIVE_FINGERPRINT_MANIFEST,
    protocol: LiveFingerprintProtocol = LiveFingerprintProtocol(),
) -> Tuple[rnmonpaex.PaidCallSpec, ...]:
    """Build and deterministically interleave the three 60-response pools."""
    validate_live_fingerprint_bank(prompts, protocol)
    routes = {route.name: route for route in manifest.routes}
    required_routes = {route_name for _, route_name in POOL_ROUTES}
    if not required_routes.issubset(routes):
        raise ValueError("campaign manifest is missing a fingerprint route")
    bank_hash = live_fingerprint_bank_hash(prompts)
    unordered = []
    for pool_name, route_name in POOL_ROUTES:
        route = routes[route_name]
        for prompt_index, item in enumerate(prompts):
            for sample_index in range(protocol.samples_per_prompt):
                call_id = "fingerprint-%s-%s-s%02d" % (
                    pool_name,
                    item.prompt_id,
                    sample_index,
                )
                unordered.append(
                    rnmonpaex.PaidCallSpec(
                        call_id=call_id,
                        batch=1,
                        route_name=route.name,
                        provider_slug=route.provider_slug,
                        expected_provider_name=route.expected_provider_name,
                        model_id=route.model_id,
                        prompt=item.prompt,
                        max_tokens=protocol.max_tokens,
                        timeout_s=protocol.timeout_s,
                        metadata=(
                            ("pool", pool_name),
                            ("prompt_id", item.prompt_id),
                            ("prompt_index", str(prompt_index)),
                            ("sample_index", str(sample_index)),
                            ("bank_hash", bank_hash),
                            (
                                "sampling_temperature",
                                str(protocol.sampling_temperature),
                            ),
                        ),
                    )
                )
    ordered = sorted(
        unordered,
        key=lambda call: _order_key(
            bank_hash,
            dict(call.metadata)["pool"],
            dict(call.metadata)["prompt_id"],
            int(dict(call.metadata)["sample_index"]),
        ),
    )
    calls = tuple(
        dataclasses.replace(call, batch=(index // 60) + 1)
        for index, call in enumerate(ordered)
    )
    rnmonpaex.validate_paid_plan(calls, protocol.make_execution_policy())
    return calls


def summarize_live_fingerprint_plan(
    calls: Sequence[rnmonpaex.PaidCallSpec],
) -> Mapping[str, object]:
    """Summarize routes, pools, and prompt counts in the paid plan."""
    routes: Dict[str, int] = {}
    pools: Dict[str, int] = {}
    prompts: Dict[str, int] = {}
    for call in calls:
        metadata = dict(call.metadata)
        routes[call.route_name] = routes.get(call.route_name, 0) + 1
        pool = metadata["pool"]
        pools[pool] = pools.get(pool, 0) + 1
        prompt_id = metadata["prompt_id"]
        prompts[prompt_id] = prompts.get(prompt_id, 0) + 1
    return {
        "total_calls": len(calls),
        "route_counts": dict(sorted(routes.items())),
        "pool_counts": dict(sorted(pools.items())),
        "prompt_counts": dict(sorted(prompts.items())),
    }


def build_unattempted_live_fingerprint_calls(
    calls: Sequence[rnmonpaex.PaidCallSpec],
    prior_observations: Sequence[Mapping[str, object]],
) -> Tuple[rnmonpaex.PaidCallSpec, ...]:
    """Resume only the unattempted suffix of an exact frozen plan."""
    if len(prior_observations) >= len(calls):
        raise ValueError("the frozen fingerprint plan has no unattempted calls")
    observed_ids = [str(row.get("call_id", "")) for row in prior_observations]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("prior fingerprint observations contain duplicate call ids")
    expected_prefix = [call.call_id for call in calls[: len(observed_ids)]]
    if observed_ids != expected_prefix:
        raise ValueError("prior observations are not an exact plan prefix")
    return tuple(calls[len(observed_ids) :])
