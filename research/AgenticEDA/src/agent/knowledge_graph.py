"""Knowledge graph for accumulating EDA findings with evidence chains."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import math

_LOG = logging.getLogger(__name__)


def compute_finding_confidence(
    has_p_value: bool = False,
    p_value: float | None = None,
    sample_size: int = 0,
    num_evidence_lines: int = 1,
    has_visual_confirmation: bool = False,
    confirmed_by_multiple_methods: bool = False,
) -> float:
    """Composite confidence score based on statistical evidence quality."""
    score = 0.0

    # Statistical test strength (0-0.35)
    if has_p_value and p_value is not None:
        if p_value < 0.001:
            score += 0.35
        elif p_value < 0.01:
            score += 0.30
        elif p_value < 0.05:
            score += 0.25
        elif p_value < 0.10:
            score += 0.15
        else:
            score += 0.05
    else:
        score += 0.10  # heuristic/descriptive only

    # Sample size adequacy (0-0.25)
    if sample_size >= 1000:
        score += 0.25
    elif sample_size >= 100:
        score += 0.20
    elif sample_size >= 30:
        score += 0.15
    elif sample_size >= 10:
        score += 0.08
    else:
        score += 0.03

    # Evidence convergence (0-0.20) — diminishing returns
    score += 0.20 * (1 - math.exp(-0.5 * num_evidence_lines))

    # Visual confirmation (0-0.10)
    if has_visual_confirmation:
        score += 0.10

    # Multi-method confirmation (0-0.10)
    if confirmed_by_multiple_methods:
        score += 0.10

    return round(min(1.0, score), 3)


def extract_p_values(text: str) -> list[float]:
    """Extract p-values from statistical output text."""
    import re
    p_vals = []
    # Match patterns like p=0.05, p<0.001, p-value: 0.023, (p=1.04e-19)
    for match in re.finditer(r'p[\s_-]*(?:value)?[\s:=<]*([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)', text, re.IGNORECASE):
        try:
            val = float(match.group(1))
            if 0 <= val <= 1:
                p_vals.append(val)
        except ValueError:
            pass
    return p_vals


@dataclass
class KGEdge:
    """A typed, weighted edge between two knowledge nodes."""
    source_id: str
    target_id: str
    type: str  # "supports", "contradicts", "derived_from", "tested_by", "visualized_by", "supersedes"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeNode:
    """A single node in the knowledge graph."""
    id: str
    type: str  # "fact", "hypothesis", "evidence", "conclusion"
    text: str
    phase: str = ""
    cell_ids: list[str] = field(default_factory=list)
    plot_cell_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    hypothesis_id: str | None = None
    children: list[str] = field(default_factory=list)  # child node IDs
    metadata: dict[str, Any] = field(default_factory=dict)
    loop_number: int = 0
    superseded_by: str | None = None
    is_latest: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "phase": self.phase,
            "cell_ids": self.cell_ids,
            "plot_cell_ids": self.plot_cell_ids,
            "confidence": self.confidence,
            "hypothesis_id": self.hypothesis_id,
            "children": self.children,
            "metadata": self.metadata,
            "loop_number": self.loop_number,
            "superseded_by": self.superseded_by,
            "is_latest": self.is_latest,
        }


class KnowledgeGraph:
    """Accumulates findings from all phases and investigations."""

    def __init__(self):
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: list[KGEdge] = []
        self._counter = 0

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _next_id(self, prefix: str = "n") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    # ------------------------------------------------------------------
    # Core mutators (backward-compatible signatures)
    # ------------------------------------------------------------------

    def add_fact(
        self,
        text: str,
        phase: str,
        cell_id: str | None = None,
        *,
        columns: list[str] | None = None,
        loop_number: int = 0,
    ) -> str:
        """Add a data fact from the initial EDA pass."""
        nid = self._next_id("fact")
        meta: dict[str, Any] = {}
        if columns:
            meta["columns"] = columns
        self.nodes[nid] = KnowledgeNode(
            id=nid, type="fact", text=text, phase=phase,
            cell_ids=[cell_id] if cell_id else [],
            metadata=meta,
            loop_number=loop_number,
        )
        return nid

    def add_investigation(
        self,
        hypothesis_id: str,
        hypothesis_title: str,
        finding: str,
        evidence_cells: list[str],
        plot_cells: list[str],
        confidence: float = 0.8,
        sub_findings: list[dict] | None = None,
        *,
        columns: list[str] | None = None,
        analysis_type: str | None = None,
        loop_number: int = 0,
    ) -> str:
        """Add results from a hypothesis investigation."""
        nid = self._next_id("inv")
        meta: dict[str, Any] = {}
        if columns:
            meta["columns"] = columns
        if analysis_type:
            meta["analysis_type"] = analysis_type

        node = KnowledgeNode(
            id=nid, type="conclusion", text=finding,
            phase=f"Investigation: {hypothesis_title}",
            cell_ids=evidence_cells, plot_cell_ids=plot_cells,
            confidence=confidence, hypothesis_id=hypothesis_id,
            metadata=meta,
            loop_number=loop_number,
        )

        # Add sub-findings as children and create SUPPORTS edges
        if sub_findings:
            for sf in sub_findings:
                child_id = self._next_id("sub")
                self.nodes[child_id] = KnowledgeNode(
                    id=child_id, type="evidence", text=sf.get("finding", ""),
                    phase=node.phase,
                    cell_ids=sf.get("cell_ids", []),
                    plot_cell_ids=sf.get("plot_cells", []),
                    confidence=sf.get("confidence", 0.7),
                    hypothesis_id=hypothesis_id,
                )
                node.children.append(child_id)
                # Auto-create supports edge from evidence to conclusion
                self.edges.append(KGEdge(
                    source_id=child_id,
                    target_id=nid,
                    type="supports",
                ))

        self.nodes[nid] = node
        return nid

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(self, edge: KGEdge) -> None:
        """Append an edge to the graph."""
        self.edges.append(edge)

    # ------------------------------------------------------------------
    # Confidence / supersession
    # ------------------------------------------------------------------

    def reinforce(self, node_id: str) -> None:
        """Increase a node's confidence: min(0.99, conf + 0.1*(1-conf))."""
        node = self.nodes[node_id]
        node.confidence = min(0.99, node.confidence + 0.1 * (1 - node.confidence))

    def supersede(self, old_id: str, new_id: str) -> None:
        """Mark *old_id* as superseded by *new_id* and add a SUPERSEDES edge."""
        old = self.nodes[old_id]
        old.is_latest = False
        old.superseded_by = new_id
        self.edges.append(KGEdge(
            source_id=new_id,
            target_id=old_id,
            type="supersedes",
        ))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query_by_type(self, node_type: str) -> list[KnowledgeNode]:
        """Return latest nodes of the given type."""
        return [n for n in self.nodes.values() if n.type == node_type and n.is_latest]

    def query_by_columns(self, columns: list[str]) -> list[KnowledgeNode]:
        """Find nodes whose metadata columns overlap or whose text mentions the columns."""
        col_set = {c.lower() for c in columns}
        results: list[KnowledgeNode] = []
        for node in self.nodes.values():
            if not node.is_latest:
                continue
            # Check metadata columns
            node_cols = {c.lower() for c in node.metadata.get("columns", [])}
            if node_cols & col_set:
                results.append(node)
                continue
            # Check text mentions
            text_lower = node.text.lower()
            if any(c in text_lower for c in col_set):
                results.append(node)
        return results

    def get_evidence_chain(self, conclusion_id: str) -> list[KnowledgeNode]:
        """Follow supports/contradicts edges backward from *conclusion_id*."""
        supporters: list[KnowledgeNode] = []
        visited: set[str] = set()
        queue = [conclusion_id]
        while queue:
            current = queue.pop(0)
            for edge in self.edges:
                if edge.target_id == current and edge.type in ("supports", "contradicts"):
                    if edge.source_id not in visited:
                        visited.add(edge.source_id)
                        node = self.nodes.get(edge.source_id)
                        if node:
                            supporters.append(node)
                            queue.append(edge.source_id)
        return supporters

    def find_duplicate_investigation(
        self, columns: list[str], analysis_type: str
    ) -> KnowledgeNode | None:
        """SHA-256 hash match on sorted columns + analysis_type."""
        target_hash = self._investigation_hash(columns, analysis_type)
        for node in self.nodes.values():
            if node.type != "conclusion" or not node.is_latest:
                continue
            node_cols = node.metadata.get("columns", [])
            node_at = node.metadata.get("analysis_type", "")
            if self._investigation_hash(node_cols, node_at) == target_hash:
                return node
        return None

    @staticmethod
    def _investigation_hash(columns: list[str], analysis_type: str) -> str:
        key = json.dumps({"columns": sorted(c.lower() for c in columns), "analysis_type": analysis_type.lower()})
        return hashlib.sha256(key.encode()).hexdigest()

    def find_similar_hypothesis(
        self, hypothesis, threshold: float = 0.5
    ) -> KnowledgeNode | None:
        """Column-overlap ratio between hypothesis words and node metadata columns + text."""
        # Accept either a string or a Hypothesis object with .title and .relevant_cols
        if hasattr(hypothesis, 'relevant_cols'):
            hyp_text = hypothesis.title
            hyp_cols = set(c.lower() for c in (hypothesis.relevant_cols or []))
        else:
            hyp_text = str(hypothesis)
            hyp_cols = set()
        hyp_words = set(hyp_text.lower().split()) | hyp_cols
        best_node: KnowledgeNode | None = None
        best_score = 0.0
        for node in self.nodes.values():
            if node.type != "conclusion" or not node.is_latest:
                continue
            node_words = set(node.text.lower().split())
            node_cols = {c.lower() for c in node.metadata.get("columns", [])}
            combined = node_words | node_cols
            if not combined or not hyp_words:
                continue
            overlap = len(hyp_words & combined) / len(hyp_words)
            if overlap > best_score:
                best_score = overlap
                best_node = node
        if best_score >= threshold:
            return best_node
        return None

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def get_context_for_hypothesis_generation(self) -> str:
        """Text summary of facts + conclusions + visual insights."""
        parts: list[str] = []
        facts = self.query_by_type("fact")
        if facts:
            parts.append("## Known Facts")
            for f in facts:
                parts.append(f"- {f.text}")

        conclusions = self.query_by_type("conclusion")
        if conclusions:
            parts.append("\n## Conclusions So Far")
            for c in conclusions:
                parts.append(f"- [{c.confidence:.0%}] {c.text}")

        evidence = self.query_by_type("evidence")
        visual = [e for e in evidence if e.plot_cell_ids]
        if visual:
            parts.append("\n## Visual Insights")
            for v in visual:
                parts.append(f"- {v.text}")

        return "\n".join(parts)

    def get_context_for_chat(self, question: str) -> str:
        """Word-overlap scoring; return top 8 relevant nodes as context."""
        q_words = set(question.lower().split())
        if not q_words:
            return ""

        scored: list[tuple[float, KnowledgeNode]] = []
        for node in self.nodes.values():
            node_words = set(node.text.lower().split())
            if not node_words:
                continue
            overlap = len(q_words & node_words)
            if overlap > 0:
                scored.append((overlap, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:8]
        if not top:
            return ""

        lines = ["Relevant knowledge:"]
        for score, node in top:
            lines.append(f"- [{node.type}] {node.text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Story / summary (backward compatible)
    # ------------------------------------------------------------------

    def get_story_sections(self) -> list[dict]:
        """Generate story sections from the knowledge graph."""
        phase_nodes: dict[str, list[KnowledgeNode]] = {}
        for node in self.nodes.values():
            phase_nodes.setdefault(node.phase, []).append(node)

        sections = []

        # First: initial EDA facts
        for phase in ["Data Loading", "Data Cleaning", "Univariate Analysis",
                       "Time Series", "Dynamics", "Correlations", "Train/Test Split"]:
            nodes = phase_nodes.get(phase, [])
            if not nodes:
                continue
            all_cells: list[str] = []
            all_plots: list[str] = []
            findings: list[str] = []
            for n in nodes:
                findings.append(n.text)
                all_cells.extend(n.cell_ids)
                all_plots.extend(n.plot_cell_ids)
            sections.append({
                "phase": phase,
                "title": phase,
                "content": "\n".join(f"- {f}" for f in findings),
                "cell_ids": list(set(all_cells)),
                "plot_cell_ids": list(set(all_plots)),
                "type": "initial",
            })

        # Then: investigation conclusions
        investigations = [n for n in self.nodes.values() if n.type == "conclusion"]
        investigations.sort(key=lambda n: n.confidence, reverse=True)

        for inv in investigations:
            children_text: list[str] = []
            child_cells: list[str] = []
            child_plots: list[str] = []
            for child_id in inv.children:
                child = self.nodes.get(child_id)
                if child:
                    children_text.append(f"  - {child.text}")
                    child_cells.extend(child.cell_ids)
                    child_plots.extend(child.plot_cell_ids)

            content = inv.text
            if children_text:
                content += "\n\nSupporting evidence:\n" + "\n".join(children_text)

            # Build plot metadata for story builder curation
            plot_metadata = []
            cell_sources = inv.metadata.get("cell_sources", {})
            cell_outputs = inv.metadata.get("cell_outputs", {})
            all_plot_ids = list(set(inv.plot_cell_ids + child_plots))
            for pcid in all_plot_ids:
                plot_metadata.append({
                    "cell_id": pcid,
                    "cell_source": cell_sources.get(pcid, ""),
                    "cell_output": cell_outputs.get(pcid, ""),
                })

            sections.append({
                "phase": inv.phase,
                "title": inv.phase.replace("Investigation: ", ""),
                "content": content,
                "cell_ids": list(set(inv.cell_ids + child_cells)),
                "plot_cell_ids": all_plot_ids,
                "confidence": inv.confidence,
                "type": "investigation",
                "plot_metadata": plot_metadata,
            })

        return sections

    def get_top_conclusions(self, n: int = 5) -> list[str]:
        """Get the top N conclusions by confidence for the executive summary."""
        conclusions = [
            node for node in self.nodes.values()
            if node.type == "conclusion"
        ]
        conclusions.sort(key=lambda node: node.confidence, reverse=True)
        return [c.text for c in conclusions[:n]]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeGraph:
        kg = cls()
        for nid, nd in data.get("nodes", {}).items():
            kg.nodes[nid] = KnowledgeNode(**nd)
            kg._counter = max(kg._counter, int(nid.split("_")[-1]) if "_" in nid else 0)
        for ed in data.get("edges", []):
            kg.edges.append(KGEdge(**ed))
        return kg
