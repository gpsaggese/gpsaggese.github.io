"""Agent state tracking for the EDA agent loop."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

_LOG = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Tracks what the agent knows and has done."""

    dataset_path: str
    session_id: str
    research_direction: str | None = None

    # Discovered during ingestion
    dataset_loaded: bool = False
    row_count: int = 0
    col_count: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    time_col: str | None = None
    target_cols: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    secondary_keys: list[str] = field(default_factory=list)
    series_type: str = "single"  # single, multiple, multivariate

    # Analysis findings
    findings: list[dict] = field(default_factory=list)
    errors_encountered: list[dict] = field(default_factory=list)

    # Progress tracking
    phases_completed: list[str] = field(default_factory=list)
    cell_count: int = 0
    cell_phases: dict[str, str] = field(default_factory=dict)  # cell_id -> phase
    cell_registry: list[dict] = field(default_factory=list)  # ordered cell records
    knowledge_graph: object = None  # KnowledgeGraph instance, set after investigation
    loop_count: int = 0
    subagent_run_count: int = 0
    max_subagents: int = 3
    max_loops: int = 2
    plan_steps: list[dict] = field(default_factory=list)  # current execution plan for REST endpoint
    steering_notes: list[dict] = field(default_factory=list)
    consumed_steering_ids: list[str] = field(default_factory=list)

    def add_finding(self, phase: str, finding: str):
        self.findings.append({"phase": phase, "finding": finding})

    def add_error(self, phase: str, error: str, fix: str):
        self.errors_encountered.append({"phase": phase, "error": error, "fix": fix})

    def mark_phase_done(self, phase: str):
        if phase not in self.phases_completed:
            self.phases_completed.append(phase)

    def next_cell_id(self) -> str:
        self.cell_count += 1
        return f"cell_{self.cell_count}"

    def register_cell(self, cell_id: str, cell_type: str, source: str, outputs: list[dict] | None = None, notebook_id: str = "main"):
        """Record a cell for later notebook serialization."""
        # Overwrite if cell already exists (e.g. after fix/backtrack)
        for entry in self.cell_registry:
            if entry["id"] == cell_id:
                entry["source"] = source
                entry["cell_type"] = cell_type
                entry["notebook_id"] = notebook_id
                if outputs is not None:
                    entry["outputs"] = outputs
                return
        self.cell_registry.append({
            "id": cell_id,
            "cell_type": cell_type,
            "source": source,
            "outputs": outputs or [],
            "notebook_id": notebook_id,
        })

    def run_guidance(self) -> str:
        """Return compact user guidance for prompts."""
        parts: list[str] = []
        if self.research_direction:
            parts.append(f"Initial research direction: {self.research_direction}")
        if self.steering_notes:
            notes = "\n".join(
                f"- {item.get('content', '')}"
                for item in self.steering_notes[-5:]
                if item.get("content")
            )
            if notes:
                parts.append(f"Recent user steering:\n{notes}")
        return "\n\n".join(parts)

    def summarize(self) -> str:
        lines = [f"EDA complete on {self.dataset_path}"]
        lines.append(f"Dataset: {self.row_count} rows x {self.col_count} cols")
        if self.time_col:
            lines.append(f"Time column: {self.time_col}")
        lines.append(f"Phases completed: {', '.join(self.phases_completed)}")
        lines.append(f"Findings: {len(self.findings)}")
        if self.errors_encountered:
            lines.append(f"Errors encountered and fixed: {len(self.errors_encountered)}")
        for f in self.findings:
            lines.append(f"  - [{f['phase']}] {f['finding']}")
        return "\n".join(lines)
