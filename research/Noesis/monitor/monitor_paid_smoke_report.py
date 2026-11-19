"""
Persist the paid attribution-smoke manifest and observations to local files.

Import as:

import research.Noesis.monitor.monitor_paid_smoke_report as rnmonpasr
"""

import dataclasses
import json
import os
from typing import Sequence

import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
import research.Noesis.monitor.monitor_paid_smoke as rnmonpasm


@dataclasses.dataclass(frozen=True)
class SmokeArtifactPaths:
    """
    Store the attribution-smoke artifact paths.
    """

    plan_path: str
    observations_path: str
    summary_path: str
    failure_path: str


def _write_durable_text(path: str, content: str, *, immutable: bool) -> None:
    """
    Flush one artifact and refuse to replace a different frozen plan.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if immutable and os.path.exists(path):
        with open(path, encoding="utf-8") as file_:
            if file_.read() == content:
                return
        raise ValueError("refusing to replace a different smoke plan")
    with open(path, "w", encoding="utf-8", newline="") as file_:
        file_.write(content)
        file_.flush()
        os.fsync(file_.fileno())


def make_artifact_paths(output_dir: str) -> SmokeArtifactPaths:
    """
    Return the stable file layout for one smoke execution directory.
    """
    return SmokeArtifactPaths(
        plan_path=os.path.join(output_dir, "plan.json"),
        observations_path=os.path.join(output_dir, "observations.jsonl"),
        summary_path=os.path.join(output_dir, "summary.json"),
        failure_path=os.path.join(output_dir, "failure.json"),
    )


def render_smoke_plan(
    calls: Sequence[rnmonpasm.SmokeCallSpec],
    policy: rnmonpasm.SmokePolicy,
    campaign_manifest: rnmonexma.PhaseZeroManifest,
) -> str:
    """
    Render the exact paid plan and its parent campaign identity.
    """
    record = {
        "schema_version": "noesis-attribution-smoke-v1",
        "campaign_id": campaign_manifest.campaign_id,
        "campaign_manifest_hash": rnmonexma.phase_zero_manifest_hash(
            campaign_manifest
        ),
        "provider_calls": len(calls),
        "fallbacks_allowed": False,
        "policy": dataclasses.asdict(policy),
        "calls": [dataclasses.asdict(call) for call in calls],
    }
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def write_smoke_plan(
    paths: SmokeArtifactPaths,
    calls: Sequence[rnmonpasm.SmokeCallSpec],
    policy: rnmonpasm.SmokePolicy,
    campaign_manifest: rnmonexma.PhaseZeroManifest,
) -> None:
    """
    Persist the reviewed plan before any provider call.
    """
    _write_durable_text(
        paths.plan_path,
        render_smoke_plan(calls, policy, campaign_manifest),
        immutable=True,
    )


class JsonlObservationStore:
    """
    Durably append each observation before the next paid call.
    """

    def __init__(self, path: str) -> None:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            raise ValueError("observation file already contains a prior run")
        self._path = path

    async def record(
        self, observation: rnmonpasm.SmokeObservation
    ) -> None:
        """
        Append, flush, and fsync one observation.
        """
        line = json.dumps(dataclasses.asdict(observation), sort_keys=True) + "\n"
        with open(self._path, "a", encoding="utf-8", newline="") as file_:
            file_.write(line)
            file_.flush()
            os.fsync(file_.fileno())


def write_smoke_summary(
    path: str, execution: rnmonpasm.SmokeExecution
) -> None:
    """
    Write the final state after all calls have stopped.
    """
    content = json.dumps(
        rnmonpasm.summarize_smoke(execution),
        indent=2,
        sort_keys=True,
    ) + "\n"
    _write_durable_text(path, content, immutable=False)


def write_smoke_failure(path: str, reason: str) -> None:
    """
    Record an orchestration or persistence failure without secrets.
    """
    content = json.dumps(
        {"state": "HALTED", "halt_reason": reason},
        indent=2,
        sort_keys=True,
    ) + "\n"
    _write_durable_text(path, content, immutable=False)
