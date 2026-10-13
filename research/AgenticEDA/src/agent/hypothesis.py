"""Hypothesis generator — formulates investigation questions from initial EDA findings."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

_LOG = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A single investigation hypothesis."""
    id: str
    title: str
    description: str
    priority: int  # 1 = highest
    relevant_cols: list[str] = field(default_factory=list)
    eda_rules: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "relevant_cols": self.relevant_cols,
            "eda_rules": self.eda_rules,
        }


def generate_hypotheses(
    columns: list[str],
    numeric_cols: list[str],
    time_col: str | None,
    findings: list[dict],
    row_count: int,
    col_count: int,
    kg_context: str = "",
    run_guidance: str = "",
) -> list[Hypothesis]:
    """
    Use the LLM to generate investigation hypotheses from initial EDA findings.

    Returns up to 10 ranked hypotheses.
    """
    findings_text = "\n".join(
        f"- [{f.get('phase', '')}] {f.get('finding', '')}" for f in findings
    )

    kg_section = ""
    if kg_context:
        kg_section = f"\n\nPrevious investigation results (do NOT re-investigate these):\n{kg_context}\n\nIMPORTANT: Generate NOVEL hypotheses that are NOT already covered above.\n"
    guidance_section = (
        f"\n\nUser guidance for this run:\n{run_guidance}\n"
        if run_guidance.strip()
        else ""
    )

    prompt = f"""You are an expert data analyst. Based on these initial EDA findings,
generate up to 10 hypotheses worth investigating further.
Prefer 3 to 5 strong hypotheses if enough evidence exists.

Dataset: {row_count} rows x {col_count} cols
Columns: {', '.join(columns[:20])}
Numeric columns: {', '.join(numeric_cols[:15])}
Time column: {time_col or 'none'}

Initial findings:
{findings_text}
{kg_section}
{guidance_section}
For each hypothesis, consider:
- Unexpected correlations that need deeper analysis
- Anomalies/outliers that need explanation (are they clustered? regime-dependent?)
- Seasonal patterns to decompose (multiple seasonalities? changing over time?)
- Possible causal relationships between variables
- Non-linear relationships hidden by linear correlation
- Feature interactions worth exploring
- Distribution characteristics that affect modeling

Respond with a JSON array (no markdown fencing):
[{{"id": "h1", "title": "short title", "description": "what to investigate and why", "priority": 1, "relevant_cols": ["col1", "col2"], "eda_rules": [25, 27]}}]

Rules:
- Maximum 10 hypotheses
- Prefer 3 to 5 hypotheses when the findings support it
- Priority 1 = most important
- eda_rules = which EDA rules from the rulebook this addresses
- Use ONLY column names from the list above
- Each hypothesis should be SPECIFIC and TESTABLE with code
- Don't include vague hypotheses like "explore the data more"
"""

    def _dedupe_and_sort(items: list[Hypothesis]) -> list[Hypothesis]:
        seen: set[tuple[str, tuple[str, ...]]] = set()
        deduped: list[Hypothesis] = []
        for hyp in items:
            key = (hyp.title.strip().lower(), tuple(hyp.relevant_cols))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hyp)
        deduped.sort(key=lambda h: h.priority)
        return deduped

    def _fallback_hypotheses() -> list[Hypothesis]:
        fallback: list[Hypothesis] = []

        if len(numeric_cols) >= 2:
            fallback.append(Hypothesis(
                id="h1",
                title=f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                description=(
                    f"Test whether {numeric_cols[0]} and {numeric_cols[1]} have a linear, nonlinear, "
                    "or regime-dependent relationship that the initial summary may have obscured."
                ),
                priority=1,
                relevant_cols=numeric_cols[:2],
                eda_rules=[25, 26],
            ))

        if time_col and numeric_cols:
            fallback.append(Hypothesis(
                id="h2",
                title=f"Temporal behavior in {numeric_cols[0]}",
                description=(
                    f"Check whether {numeric_cols[0]} shows trend, seasonality, or local regime shifts over time."
                ),
                priority=2,
                relevant_cols=[numeric_cols[0], time_col],
                eda_rules=[14, 18, 21],
            ))

        if numeric_cols:
            focus_col = numeric_cols[min(2, len(numeric_cols) - 1)]
            fallback.append(Hypothesis(
                id="h3",
                title=f"Outlier structure in {focus_col}",
                description=(
                    f"Investigate whether extreme values in {focus_col} cluster around specific conditions "
                    "or form a separate operating regime."
                ),
                priority=3,
                relevant_cols=[focus_col],
                eda_rules=[8, 9, 10],
            ))

        if len(fallback) < 3 and len(numeric_cols) >= 1:
            fallback.append(Hypothesis(
                id=f"h{len(fallback)+1}",
                title=f"Distribution shape of {numeric_cols[0]}",
                description=(
                    f"Assess whether {numeric_cols[0]} is skewed, multimodal, or heavy-tailed enough to change "
                    "the interpretation of downstream modeling."
                ),
                priority=len(fallback) + 1,
                relevant_cols=[numeric_cols[0]],
                eda_rules=[3, 4, 5],
            ))

        return _dedupe_and_sort(fallback)

    try:
        from src.config.config import get_chat_model
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_chat_model()
        response = llm.invoke([
            SystemMessage(content="You generate specific, testable data analysis hypotheses. Always respond with valid JSON. Never use emojis or special unicode symbols."),
            HumanMessage(content=prompt),
        ])

        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        raw = json.loads(text)
        hypotheses: list[Hypothesis] = []
        for h in raw[:10]:
            hypotheses.append(Hypothesis(
                id=h.get("id", f"h{len(hypotheses)+1}"),
                title=h.get("title", ""),
                description=h.get("description", ""),
                priority=h.get("priority", len(hypotheses)+1),
                relevant_cols=h.get("relevant_cols", []),
                eda_rules=h.get("eda_rules", []),
            ))

        hypotheses = _dedupe_and_sort(hypotheses)
        if len(hypotheses) < 3:
            fallback = _fallback_hypotheses()
            existing_keys = {(h.title.strip().lower(), tuple(h.relevant_cols)) for h in hypotheses}
            for hyp in fallback:
                key = (hyp.title.strip().lower(), tuple(hyp.relevant_cols))
                if key not in existing_keys:
                    hypotheses.append(hyp)
                    existing_keys.add(key)
                if len(hypotheses) >= 3:
                    break
            hypotheses = _dedupe_and_sort(hypotheses)

        _LOG.info("Generated %d hypotheses", len(hypotheses))
        return hypotheses

    except Exception as exc:
        _LOG.warning("Hypothesis generation failed: %s", exc)
        return _fallback_hypotheses()


def hypothesis_from_user_question(
    question: str,
    columns: list[str],
    numeric_cols: list[str],
    time_col: str | None,
) -> Hypothesis | None:
    """Convert a user's chat question into a formal Hypothesis."""
    try:
        from src.config.config import get_chat_model
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_chat_model()
        response = llm.invoke([
            SystemMessage(content=f"""Convert the user's question into a testable data hypothesis.
Available columns: {', '.join(columns[:20])}
Time column: {time_col or 'none'}

Respond with JSON (no markdown):
{{"id": "user_h1", "title": "short title", "description": "what to test", "relevant_cols": ["col1"], "eda_rules": []}}

If the question is NOT a hypothesis (just a chat question), respond with: {{"id": null}}

Never use emojis or special unicode symbols."""),
            HumanMessage(content=question),
        ])

        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)
        if not data.get("id"):
            return None

        return Hypothesis(
            id=data["id"],
            title=data.get("title", question[:50]),
            description=data.get("description", question),
            priority=1,
            relevant_cols=data.get("relevant_cols", []),
            eda_rules=data.get("eda_rules", []),
        )
    except Exception as exc:
        _LOG.warning("hypothesis_from_user_question failed: %s", exc)
        return None
