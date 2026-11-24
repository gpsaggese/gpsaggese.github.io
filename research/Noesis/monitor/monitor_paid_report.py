"""
Persist generic paid-campaign plans, observations, and summaries.

Import as:

import research.Noesis.monitor.monitor_paid_report as rnmonpare
"""

import dataclasses
import json
import os
from typing import Mapping, Sequence

import research.Noesis.monitor.monitor_paid_execution as rnmonpaex


@dataclasses.dataclass(frozen=True)
class PaidArtifactPaths:
    """
    Store the common paid-campaign artifact paths.
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
        raise ValueError("refusing to replace a different paid plan")
    with open(path, "w", encoding="utf-8", newline="") as file_:
        file_.write(content)
        file_.flush()
        os.fsync(file_.fileno())


def make_paid_artifact_paths(output_dir: str) -> PaidArtifactPaths:
    return PaidArtifactPaths(
        plan_path=os.path.join(output_dir, "plan.json"),
        observations_path=os.path.join(output_dir, "observations.jsonl"),
        summary_path=os.path.join(output_dir, "summary.json"),
        failure_path=os.path.join(output_dir, "failure.json"),
    )


def write_paid_plan(
    path: str,
    schema_version: str,
    calls: Sequence[rnmonpaex.PaidCallSpec],
    policy: rnmonpaex.PaidExecutionPolicy,
    metadata: Mapping[str, object],
) -> None:
    """
    Persist an exact paid plan before the first provider call.
    """
    record = {
        "schema_version": schema_version,
        "provider_calls": len(calls),
        "fallbacks_allowed": False,
        "policy": dataclasses.asdict(policy),
        "metadata": dict(metadata),
        "calls": [dataclasses.asdict(call) for call in calls],
    }
    _write_durable_text(
        path,
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        immutable=True,
    )


class JsonlPaidObservationStore:
    """
    Durably append each observation before the next paid call.
    """

    def __init__(self, path: str) -> None:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            raise ValueError("observation file already contains a prior run")
        self._path = path

    async def record(self, observation: rnmonpaex.PaidObservation) -> None:
        line = json.dumps(dataclasses.asdict(observation), sort_keys=True) + "\n"
        with open(self._path, "a", encoding="utf-8", newline="") as file_:
            file_.write(line)
            file_.flush()
            os.fsync(file_.fileno())


def write_paid_summary(path: str, execution: rnmonpaex.PaidExecution) -> None:
    content = json.dumps(
        rnmonpaex.summarize_paid_execution(execution),
        indent=2,
        sort_keys=True,
    ) + "\n"
    _write_durable_text(path, content, immutable=False)


def write_paid_failure(path: str, reason: str) -> None:
    content = json.dumps(
        {"state": "HALTED", "halt_reason": reason},
        indent=2,
        sort_keys=True,
    ) + "\n"
    _write_durable_text(path, content, immutable=False)
