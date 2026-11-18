"""
Build the $10 four-route, five-family controlled warm-up campaign.

Import as:

import research.Noesis.monitor.monitor_controlled_warmup as rnmoncowar
"""

import dataclasses
import hashlib
import json
import re
from typing import Dict, Mapping, Sequence, Tuple

import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
import research.Noesis.monitor.monitor_paid_execution as rnmonpaex


@dataclasses.dataclass(frozen=True)
class WarmupItem:
    """
    Store one frozen capability prompt and its grader identity.
    """

    item_id: str
    family: str
    prompt: str
    grader_id: str
    grader_config_hash: str

    def __post_init__(self) -> None:
        for name in (
            "item_id",
            "family",
            "prompt",
            "grader_id",
            "grader_config_hash",
        ):
            if not getattr(self, name).strip():
                raise ValueError("%s must not be empty" % name)
        if len(self.prompt) > 20000:
            raise ValueError("warm-up prompts must not exceed 20000 characters")
        if re.fullmatch(r"[0-9a-f]{64}", self.grader_config_hash) is None:
            raise ValueError(
                "grader_config_hash must be a lowercase SHA-256 digest"
            )


@dataclasses.dataclass(frozen=True)
class WarmupProtocol:
    """
    Freeze the controlled live campaign bounds.
    """

    items_per_family: int = 12
    family_count: int = 5
    route_count: int = 4
    max_tokens: int = 512
    timeout_s: float = 60.0
    maximum_cost_usd: float = 10.0
    per_call_reserve_usd: float = 0.04

    @property
    def item_count(self) -> int:
        return self.items_per_family * self.family_count

    @property
    def call_count(self) -> int:
        return self.item_count * self.route_count

    def make_execution_policy(self) -> rnmonpaex.PaidExecutionPolicy:
        """
        Build the shared paid-execution safety policy.
        """
        return rnmonpaex.PaidExecutionPolicy(
            maximum_calls=self.call_count,
            maximum_cost_usd=self.maximum_cost_usd,
            per_call_reserve_usd=self.per_call_reserve_usd,
        )


def load_warmup_bank(path: str) -> Tuple[WarmupItem, ...]:
    """
    Load a JSONL bank without interpreting grader configuration.
    """
    items = []
    with open(path, encoding="utf-8") as file_:
        for line_number, line in enumerate(file_, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                items.append(WarmupItem(**record))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    "invalid warm-up bank row %s: %s" % (line_number, error)
                ) from error
    return tuple(items)


def validate_warmup_bank(
    items: Sequence[WarmupItem],
    manifest: rnmonexma.PhaseZeroManifest,
    protocol: WarmupProtocol,
) -> None:
    """
    Require the reviewed 60-item, five-family balanced window.
    """
    if len(items) != protocol.item_count:
        raise ValueError(
            "warm-up bank must contain exactly %s items" % protocol.item_count
        )
    item_ids = [item.item_id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("warm-up item ids must be unique")
    expected_families = {family.name for family in manifest.benchmark_families}
    actual_families = {item.family for item in items}
    if actual_families != expected_families:
        raise ValueError("warm-up bank families do not match the campaign")
    family_counts: Dict[str, int] = {}
    for item in items:
        family_counts[item.family] = family_counts.get(item.family, 0) + 1
    if set(family_counts.values()) != {protocol.items_per_family}:
        raise ValueError("each warm-up family must contain exactly 12 items")


def warmup_bank_hash(items: Sequence[WarmupItem]) -> str:
    """
    Return a stable identity for the ordered prompt and grader bank.
    """
    encoded = json.dumps(
        [dataclasses.asdict(item) for item in items],
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _order_key(bank_hash: str, route_name: str, item_id: str) -> bytes:
    material = "controlled-warmup-v1|%s|%s|%s" % (
        bank_hash,
        route_name,
        item_id,
    )
    return hashlib.sha256(material.encode("utf-8")).digest()


def build_warmup_calls(
    items: Sequence[WarmupItem],
    manifest: rnmonexma.PhaseZeroManifest = rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST,
    protocol: WarmupProtocol = WarmupProtocol(),
) -> Tuple[rnmonpaex.PaidCallSpec, ...]:
    """
    Cross 60 prompts with four routes and deterministically randomize order.
    """
    validate_warmup_bank(items, manifest, protocol)
    if len(manifest.routes) != protocol.route_count:
        raise ValueError("campaign manifest must contain exactly four routes")
    bank_hash = warmup_bank_hash(items)
    unordered = []
    for route in manifest.routes:
        for item in items:
            call_id = "warmup-%s-%s" % (route.name, item.item_id)
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
                        ("item_id", item.item_id),
                        ("family", item.family),
                        ("grader_id", item.grader_id),
                        ("grader_config_hash", item.grader_config_hash),
                        ("bank_hash", bank_hash),
                    ),
                )
            )
    ordered = sorted(
        unordered,
        key=lambda call: _order_key(
            bank_hash,
            call.route_name,
            dict(call.metadata)["item_id"],
        ),
    )
    calls = tuple(
        dataclasses.replace(call, batch=(index // 60) + 1)
        for index, call in enumerate(ordered)
    )
    rnmonpaex.validate_paid_plan(calls, protocol.make_execution_policy())
    return calls


def summarize_warmup_plan(
    calls: Sequence[rnmonpaex.PaidCallSpec],
) -> Mapping[str, object]:
    """
    Summarize the balanced route and family call counts.
    """
    routes: Dict[str, int] = {}
    families: Dict[str, int] = {}
    for call in calls:
        routes[call.route_name] = routes.get(call.route_name, 0) + 1
        family = dict(call.metadata)["family"]
        families[family] = families.get(family, 0) + 1
    return {
        "total_calls": len(calls),
        "route_counts": dict(sorted(routes.items())),
        "family_counts": dict(sorted(families.items())),
    }


def build_unattempted_calls(
    calls: Sequence[rnmonpaex.PaidCallSpec],
    prior_observations: Sequence[Mapping[str, object]],
) -> Tuple[rnmonpaex.PaidCallSpec, ...]:
    """
    Resume only an exact persisted prefix of the frozen call order.

    Failed attempts remain evidence and are never silently retried.
    """
    if len(prior_observations) >= len(calls):
        raise ValueError("the frozen warm-up plan has no unattempted calls")
    observed_ids = [str(row.get("call_id", "")) for row in prior_observations]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("prior warm-up observations contain duplicate call ids")
    expected_prefix = [call.call_id for call in calls[: len(observed_ids)]]
    if observed_ids != expected_prefix:
        raise ValueError("prior observations are not an exact plan prefix")
    return tuple(calls[len(observed_ids) :])
