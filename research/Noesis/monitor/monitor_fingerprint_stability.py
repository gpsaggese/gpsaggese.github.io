"""Build the compressed three-pool Lane 2 stability campaign."""

import dataclasses
import hashlib
from typing import Dict, Mapping, Sequence, Tuple

import research.Noesis.monitor.monitor_live_fingerprint as rnmonlifp
import research.Noesis.monitor.monitor_paid_execution as rnmonpaex


NEW_POOL_NAMES = ("reference-c", "reference-d", "reference-e")


@dataclasses.dataclass(frozen=True)
class FingerprintStabilityProtocol:
    """Freeze the compressed same-day A/A collection bounds."""

    prompt_count: int = 5
    samples_per_prompt: int = 12
    pool_count: int = 3
    max_tokens: int = 384
    timeout_s: float = 60.0
    sampling_temperature: float = 1.0
    maximum_cost_usd: float = 0.50
    per_call_reserve_usd: float = 0.002

    def __post_init__(self) -> None:
        if self.pool_count != len(NEW_POOL_NAMES):
            raise ValueError("stability campaign requires exactly three new pools")
        if self.prompt_count < 1 or self.samples_per_prompt < 2:
            raise ValueError("prompt and sample counts must be positive")
        if self.max_tokens < 1 or self.timeout_s <= 0:
            raise ValueError("call bounds must be positive")
        if not 0 <= self.sampling_temperature <= 2:
            raise ValueError("sampling_temperature must be between zero and two")

    @property
    def pool_size(self) -> int:
        return self.prompt_count * self.samples_per_prompt

    @property
    def call_count(self) -> int:
        return self.pool_size * self.pool_count

    def make_execution_policy(self) -> rnmonpaex.PaidExecutionPolicy:
        return rnmonpaex.PaidExecutionPolicy(
            maximum_calls=self.call_count,
            maximum_cost_usd=self.maximum_cost_usd,
            per_call_reserve_usd=self.per_call_reserve_usd,
        )


def _order_key(
    bank_hash: str, pool: str, prompt_id: str, sample_index: int
) -> bytes:
    material = "fingerprint-stability-v1|%s|%s|%s|%s" % (
        bank_hash,
        pool,
        prompt_id,
        sample_index,
    )
    return hashlib.sha256(material.encode("utf-8")).digest()


def build_stability_calls(
    prompts: Sequence[rnmonlifp.LiveFingerprintPrompt],
    protocol: FingerprintStabilityProtocol = FingerprintStabilityProtocol(),
) -> Tuple[rnmonpaex.PaidCallSpec, ...]:
    """Build three deterministically interleaved Cloudflare 70B pools."""
    rnmonlifp.validate_live_fingerprint_bank(
        prompts,
        rnmonlifp.LiveFingerprintProtocol(
            prompt_count=protocol.prompt_count,
            samples_per_prompt=protocol.samples_per_prompt,
        ),
    )
    manifest = rnmonlifp.LIVE_FINGERPRINT_MANIFEST
    route = next(
        item
        for item in manifest.routes
        if item.name == rnmonlifp.REFERENCE_ROUTE
    )
    bank_hash = rnmonlifp.live_fingerprint_bank_hash(prompts)
    unordered = []
    for pool in NEW_POOL_NAMES:
        for prompt_index, item in enumerate(prompts):
            for sample_index in range(protocol.samples_per_prompt):
                unordered.append(
                    rnmonpaex.PaidCallSpec(
                        call_id="stability-%s-%s-s%02d"
                        % (pool, item.prompt_id, sample_index),
                        batch=1,
                        route_name=route.name,
                        provider_slug=route.provider_slug,
                        expected_provider_name=route.expected_provider_name,
                        model_id=route.model_id,
                        prompt=item.prompt,
                        max_tokens=protocol.max_tokens,
                        timeout_s=protocol.timeout_s,
                        metadata=(
                            ("pool", pool),
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


def summarize_stability_plan(
    calls: Sequence[rnmonpaex.PaidCallSpec],
) -> Mapping[str, object]:
    pools: Dict[str, int] = {}
    prompts: Dict[str, int] = {}
    for call in calls:
        metadata = dict(call.metadata)
        pool = metadata["pool"]
        prompt = metadata["prompt_id"]
        pools[pool] = pools.get(pool, 0) + 1
        prompts[prompt] = prompts.get(prompt, 0) + 1
    return {
        "total_calls": len(calls),
        "pool_counts": dict(sorted(pools.items())),
        "prompt_counts": dict(sorted(prompts.items())),
    }
