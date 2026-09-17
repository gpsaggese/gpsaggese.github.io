"""Chat agent for interactive EDA follow-up questions."""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

_LOG = logging.getLogger(__name__)
SESSIONS_DIR = pathlib.Path(__file__).resolve().parents[2] / "sessions"


class ChatContext:
    """Holds session state for the chat agent."""

    def __init__(self, session_id: str, state: dict | None = None, kg=None):
        self.session_id = session_id
        self.state = state or {}
        self.kg = kg

    def get_state_value(self, key: str) -> Any:
        return self.state.get(key)

    def get_summary(self, question: str | None = None) -> str:
        """Build a context string the LLM can use to answer questions."""
        parts = []

        # Dataset info
        rc = self.state.get("row_count", 0)
        cc = self.state.get("col_count", 0)
        tc = self.state.get("time_col")
        nc = self.state.get("numeric_cols", [])
        parts.append(f"Dataset: {rc} rows x {cc} cols.")
        if tc:
            parts.append(f"Time column: {tc}")
        if nc:
            parts.append(f"Numeric columns: {', '.join(nc[:10])}")

        # Phases completed
        phases = self.state.get("phases_completed", [])
        if phases:
            parts.append(f"Phases completed: {', '.join(phases)}")

        # Findings
        if self.kg:
            conclusions = self.kg.get_top_conclusions(5)
            if conclusions:
                parts.append("Key findings:")
                for c in conclusions:
                    parts.append(f"  - {c}")
            # Add KG context relevant to the question
            if question:
                try:
                    kg_context = self.kg.get_context_for_hypothesis_generation()
                    if kg_context:
                        parts.append(f"Knowledge graph context:\n{kg_context[:1500]}")
                except Exception:
                    pass
        else:
            findings = self.state.get("findings") or self.state.get("insights") or []
            if findings:
                parts.append("Key findings:")
                for f in findings:
                    if isinstance(f, dict):
                        parts.append(f"  - [{f.get('phase', '')}] {f.get('finding', f.get('description', ''))}")
                    else:
                        parts.append(f"  - {f}")

        # Decision summary
        ds = self.state.get("decision_summary", {})
        if isinstance(ds, dict) and ds.get("summary"):
            parts.append(f"Summary: {ds['summary'][:500]}")

        return "\n".join(parts) if parts else "No analysis results available yet."


def build_chat_agent(session_id: str, state: dict | None = None, kg=None):
    """
    Build a chat agent for a session.

    Tries to use the configured LLM first. Falls back to a deterministic
    responder if no API key is available.
    """
    context = ChatContext(session_id, state, kg=kg)

    # Try to build an LLM-powered agent
    llm = _try_get_llm()

    def process_message(user_message: str) -> dict:
        response: dict = {"role": "agent", "type": "text", "content": "", "action_code": None}

        if llm is not None:
            raw = _llm_respond(llm, context, user_message)
        else:
            raw = _fallback_respond(context, user_message)

        # Parse structured action blocks
        response = _parse_action_response(raw, response)
        return response

    return process_message


def _parse_action_response(raw: str, response: dict) -> dict:
    """Parse LLM response for structured action blocks."""
    import re

    # ACTION_EDIT: <cell_id>
    edit_match = re.search(r'ACTION_EDIT:\s*(\S+)\s*\n```python\n(.*?)```', raw, re.DOTALL)
    if edit_match:
        before = raw[:edit_match.start()].strip()
        response["content"] = before or "Editing cell..."
        response["type"] = "action"
        response["action_type"] = "edit_cell"
        response["target_cell_id"] = edit_match.group(1).strip()
        response["action_code"] = edit_match.group(2).strip()
        return response

    # ACTION_DELETE: <cell_id_1>, <cell_id_2>
    delete_match = re.search(r'ACTION_DELETE:\s*(.+)', raw)
    if delete_match:
        before = raw[:delete_match.start()].strip()
        cell_ids = [cid.strip() for cid in delete_match.group(1).split(",") if cid.strip()]
        response["content"] = before or "Deleting cells..."
        response["type"] = "action"
        response["action_type"] = "delete_cells"
        response["target_cell_ids"] = cell_ids
        return response

    # ACTION_REORDER: <cell_id> <up|down>
    reorder_match = re.search(r'ACTION_REORDER:\s*(\S+)\s+(up|down)', raw, re.IGNORECASE)
    if reorder_match:
        before = raw[:reorder_match.start()].strip()
        response["content"] = before or "Reordering cell..."
        response["type"] = "action"
        response["action_type"] = "reorder_cell"
        response["target_cell_id"] = reorder_match.group(1).strip()
        response["direction"] = reorder_match.group(2).lower()
        return response

    # ACTION_NEW: (or legacy ACTION:)
    new_match = re.search(r'ACTION(?:_NEW)?:\s*\n```python\n(.*?)```', raw, re.DOTALL)
    if new_match:
        before = raw[:new_match.start()].strip()
        response["content"] = before or "Running analysis..."
        response["type"] = "action"
        response["action_type"] = "new_cell"
        response["action_code"] = new_match.group(1).strip()
        return response

    # No action block
    response["content"] = raw
    return response


def _try_get_llm():
    """Try to get a configured LLM. Returns None if not available."""
    try:
        from src.config.config import get_chat_model
        model = get_chat_model()
        return model
    except Exception as exc:
        _LOG.warning("LLM not available for chat: %s", exc)
        return None


def _llm_respond(llm, context: ChatContext, user_message: str) -> str:
    """Use the LLM to respond, with EDA context in the system prompt."""
    summary = context.get_summary(question=user_message)

    cols = context.state.get("columns", context.state.get("numeric_cols", []))
    col_list = ", ".join(f'"{c}"' for c in cols) if cols else "unknown"
    time_col = context.state.get("time_col", "")

    # Build a cell listing for the LLM so it knows what exists
    cells = context.state.get("cells", [])
    cell_listing = ""
    if cells:
        lines = []
        for i, c in enumerate(cells):
            src_preview = (c.get("source") or "")[:80].replace("\n", " ")
            lines.append(f"  [{i}] id={c['id']}  type={c.get('cell_type','code')}  | {src_preview}")
        cell_listing = "\n".join(lines)
    else:
        cell_listing = "  (no cells available — the notebook may not have been synced to chat state yet)"

    system_prompt = f"""You are an EDA assistant. The user has a dataset loaded as `df` in a Jupyter kernel.
The user's notebook currently has these cells:
{cell_listing}

Analysis state:
{summary}

Available columns: [{col_list}]
Time column: {time_col or "not identified"}

You can perform these actions by including ONE of the following blocks in your response.
The block MUST appear at the end, after any explanation text.

1. CREATE a new cell (new analysis / plot):
ACTION_NEW:
```python
# code here
```

2. EDIT an existing cell (change its code and re-run it):
ACTION_EDIT: <cell_id>
```python
# replacement code here
```

3. DELETE one or more cells:
ACTION_DELETE: <cell_id_1>, <cell_id_2>

4. REORDER a cell (move up or down):
ACTION_REORDER: <cell_id> <up|down>

Rules:
- If the user asks to "plot", "visualize", "show", "check", "test", "compare",
  "investigate", "analyze" → include an ACTION_NEW or ACTION_EDIT block
- When the user asks to CHANGE an existing plot/analysis, prefer ACTION_EDIT over ACTION_NEW
- Use ONLY these column names: [{col_list}]
- The variable `df` is already loaded with the dataset in the kernel
- Always plt.show() after matplotlib plots
- Keep code focused and readable
- You may only include ONE action block per response
- If you're unsure which cell the user means, ask for clarification instead of guessing"""

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as exc:
        _LOG.warning("LLM chat failed: %s", exc)
        return _fallback_respond(context, user_message)


def _fallback_respond(context: ChatContext, user_message: str) -> str:
    """Deterministic fallback when LLM is not available."""
    summary = context.get_summary()

    if "No analysis results" in summary:
        return "The EDA pipeline hasn't produced results yet. Upload a dataset and wait for the analysis to complete."

    # For any question, return the summary
    return f"Here's what I know about your data:\n\n{summary}\n\nFor more specific questions, try asking about specific columns, trends, or patterns."
