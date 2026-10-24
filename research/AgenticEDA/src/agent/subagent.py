"""Subagent runner — investigates a single hypothesis with code cells."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

_LOG = logging.getLogger(__name__)


@dataclass
class InvestigationResult:
    """Result of a hypothesis investigation."""
    hypothesis_id: str
    hypothesis_title: str
    finding: str
    cell_ids: list[str] = field(default_factory=list)
    plot_cell_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    # Explicit execution status (confidence is a separate "how sure" score).
    status: str = "complete"  # "complete" | "failed" | "timeout"
    sub_findings: list[dict] = field(default_factory=list)
    images: dict[str, list[str]] = field(default_factory=dict)  # cell_id -> [base64 png]
    relevant_cols: list[str] = field(default_factory=list)
    cell_sources: dict[str, str] = field(default_factory=dict)  # cell_id -> source code
    cell_outputs: dict[str, str] = field(default_factory=dict)  # cell_id -> stdout text

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_title": self.hypothesis_title,
            "finding": self.finding,
            "cell_ids": self.cell_ids,
            "plot_cell_ids": self.plot_cell_ids,
            "confidence": self.confidence,
            "status": self.status,
            "sub_findings": self.sub_findings,
            "cell_sources": self.cell_sources,
            "cell_outputs": self.cell_outputs,
        }


def _auto_fix(code: str, error: str) -> str | None:
    """Try to fix common errors without an LLM call. Returns fixed code or None."""
    import re as _re

    # NameError: name 'X' is not defined → add missing import
    m = _re.search(r"NameError: name '(\w+)' is not defined", error)
    if m:
        name = m.group(1)
        imports = {
            "np": "import numpy as np",
            "pd": "import pandas as pd",
            "plt": "import matplotlib.pyplot as plt",
            "sns": "import seaborn as sns",
            "stats": "from scipy import stats",
            "pearsonr": "from scipy.stats import pearsonr",
            "spearmanr": "from scipy.stats import spearmanr",
            "ttest_ind": "from scipy.stats import ttest_ind",
            "chi2_contingency": "from scipy.stats import chi2_contingency",
            "mannwhitneyu": "from scipy.stats import mannwhitneyu",
            "f_oneway": "from scipy.stats import f_oneway",
            "norm": "from scipy.stats import norm",
            "zscore": "from scipy.stats import zscore",
            "linregress": "from scipy.stats import linregress",
            "adfuller": "from statsmodels.tsa.stattools import adfuller",
            "OLS": "import statsmodels.api as sm",
            "sm": "import statsmodels.api as sm",
        }
        if name in imports:
            return imports[name] + "\n" + code

    # KeyError: column name not found → try fuzzy match
    m = _re.search(r"KeyError: ['\"](.+?)['\"]", error)
    if m:
        bad_col = m.group(1)
        # Can't fix without knowing all columns — return None to let LLM handle
        return None

    # ValueError: could not convert string to float → add .dropna() or numeric coercion
    if "could not convert string to float" in error:
        return code.replace(".values", ".dropna().values").replace(
            "df[", "pd.to_numeric(df["
        ).replace(".dropna()", "], errors='coerce').dropna()")

    # ModuleNotFoundError → wrap in try/except with fallback
    if "ModuleNotFoundError" in error or "No module named" in error:
        return f"try:\n    {code.replace(chr(10), chr(10) + '    ')}\nexcept ImportError:\n    print('Module not available, skipping')"

    return None


def run_subagent(
    hypothesis_id: str,
    hypothesis_title: str,
    hypothesis_description: str,
    relevant_cols: list[str],
    all_columns: list[str],
    time_col: str | None,
    session_id: str,
    push_event: Callable[[str, dict], None],
    execute_code: Callable[[str, str, int], tuple[list[dict], str | None]],
    cell_counter: list[int],  # mutable counter [current_count]
    max_cells: int = 5,
    kernel_id: str | None = None,
    notebook_id: str = "main",
    kg_context: str = "",
    run_guidance: str = "",
    deadline: float = 0,
) -> InvestigationResult:
    """
    Investigate a hypothesis by writing and executing notebook cells.

    :param hypothesis_id: unique ID for this hypothesis
    :param hypothesis_title: short title
    :param hypothesis_description: what to investigate
    :param relevant_cols: columns relevant to this hypothesis
    :param all_columns: all available column names
    :param time_col: time column name (or None)
    :param session_id: session for kernel execution
    :param push_event: callback to stream events to frontend
    :param execute_code: callback to execute code in kernel
    :param cell_counter: shared mutable counter for cell IDs
    :param max_cells: maximum cells this subagent can write
    :param deadline: absolute time.time() by which this subagent must finish (0 = no limit)
    """
    result = InvestigationResult(
        hypothesis_id=hypothesis_id,
        hypothesis_title=hypothesis_title,
        finding="",
    )
    result.relevant_cols = relevant_cols

    col_list = ", ".join(f'"{c}"' for c in all_columns)
    relevant_str = ", ".join(f'"{c}"' for c in relevant_cols)

    def _next_cell_id() -> str:
        cell_counter[0] += 1
        return f"{hypothesis_id}_cell_{cell_counter[0]}"

    _first_cell_executed = [False]

    def _write_and_execute(
        code: str,
        cell_type: str = "code",
        *,
        cell_id: str | None = None,
        overwrite: bool = False,
    ) -> tuple[str, list[dict], str | None]:
        """Write a cell, execute it, stream events, return (cell_id, outputs, error)."""
        original_code = code  # capture before preamble injection
        # Prepend safe imports only to the FIRST code cell (not every cell)
        if cell_type == "code" and not overwrite and not _first_cell_executed[0]:
            code = (
                "import numpy as np\nimport pandas as pd\n"
                "import matplotlib.pyplot as plt\n"
                "import warnings; warnings.filterwarnings('ignore')\n"
            ) + code
            _first_cell_executed[0] = True
        cell_id = cell_id or _next_cell_id()
        replacement = overwrite and cell_id is not None
        if replacement:
            push_event(session_id, {"type": "cell_delete", "cell_id": cell_id})
            time.sleep(0.05)
        push_event(session_id, {
            "type": "cell_write",
            "cell_id": cell_id,
            "cell_type": cell_type,
            "source": code,
            "overwrite": replacement,
            "notebook_id": notebook_id,
        })
        time.sleep(0.05)

        if cell_type == "markdown":
            return cell_id, [], None

        push_event(session_id, {"type": "cell_executing", "cell_id": cell_id, "notebook_id": notebook_id})
        exec_id = kernel_id if kernel_id else session_id
        cell_timeout = min(30, int(_time_left())) if deadline else 30
        outputs, error = execute_code(exec_id, code, max(cell_timeout, 10), cell_id=cell_id)

        if error:
            push_event(session_id, {
                "type": "cell_error", "cell_id": cell_id, "error": error,
                "notebook_id": notebook_id,
            })
        else:
            push_event(session_id, {
                "type": "cell_output", "cell_id": cell_id, "outputs": outputs,
                "notebook_id": notebook_id,
            })
            # Check if outputs contain plots
            for o in outputs:
                data = o.get("data", {})
                if data.get("image/png") or data.get("application/vnd.plotly.v1+json"):
                    result.plot_cell_ids.append(cell_id)
                    break
            # Extract images for vision analysis by main agent
            for o in outputs:
                img = o.get("data", {}).get("image/png")
                if img:
                    result.images.setdefault(cell_id, []).append(img)

        result.cell_ids.append(cell_id)
        result.cell_sources[cell_id] = original_code
        time.sleep(0.05)
        return cell_id, outputs, error

    def _extract_text(outputs: list[dict]) -> str:
        parts = []
        for o in outputs:
            if o.get("text"):
                parts.append(o["text"])
            if o.get("data", {}).get("text/plain"):
                parts.append(str(o["data"]["text/plain"]))
            if o.get("data", {}).get("image/png"):
                parts.append("[plot generated]")
        return "\n".join(parts)

    def _time_left() -> float:
        """Seconds remaining until deadline (inf if no deadline)."""
        if not deadline:
            return float("inf")
        return max(0, deadline - time.time())

    def _past_deadline() -> bool:
        return deadline > 0 and time.time() >= deadline

    # --- Adaptive investigation loop ---
    # Each step: LLM generates ONE cell → execute → feed output back → LLM decides next.
    # Budget: up to max_cells steps (default 5). Each step sees all previous output.
    all_outputs: list[str] = []
    cells_executed = 0
    try:
        from src.config.config import get_chat_model, get_subagent_model
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_chat_model(model=get_subagent_model(), max_retries=1)
        guidance_section = (
            f"\n\nUser guidance for this investigation:\n{run_guidance}"
            if run_guidance.strip()
            else ""
        )

        system_prompt = f"""You are investigating a data hypothesis step by step. Write ONE Python code cell at a time, see its output, then decide the next step.

Available columns: [{col_list}]
Relevant columns: [{relevant_str}]
Time column: {time_col or 'none'}
Variable `df` is already loaded.
{f"Previous findings (don't repeat): " + kg_context[:500] if kg_context else ""}
{guidance_section}

Rules:
- Write ONE code cell per response
- Use ONLY columns from the available list
- Include matplotlib plots where relevant (always plt.show())
- Print findings clearly with numbers
- Include statistical tests where appropriate (scipy.stats)
- Never use emojis or special unicode symbols

Respond with JSON (no markdown fencing):
{{"code": "python code for ONE cell", "reasoning": "why this step", "done": false}}

Set "done": true when you have enough evidence to conclude. When done, set "code" to a final print statement summarizing the key finding."""

        conversation: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Hypothesis: {hypothesis_title}\nDescription: {hypothesis_description}\n\nGenerate the first analysis step."),
        ]

        for step in range(max_cells):
            if _past_deadline():
                _LOG.warning("Subagent %s hit deadline at step %d", hypothesis_id, step+1)
                break

            push_event(session_id, {
                "type": "thinking",
                "content": f"Investigation step {step+1}/{max_cells} for: {hypothesis_title}" + (
                    f" (last: {all_outputs[-1][:60]}...)" if all_outputs else ""
                ),
                "notebook_id": notebook_id,
            })

            try:
                step_response = llm.invoke(conversation)
                step_text = step_response.content.strip()
                if step_text.startswith("```"):
                    step_text = step_text.split("\n", 1)[1] if "\n" in step_text else step_text[3:]
                    if step_text.endswith("```"):
                        step_text = step_text[:-3]
                    step_text = step_text.strip()

                step_data = json.loads(step_text)
                code = step_data.get("code", "")
                is_done = step_data.get("done", False)
            except Exception as exc:
                _LOG.warning("Subagent step %d LLM failed: %s", step+1, exc)
                break

            if not code:
                break

            cell_id, outputs, error = _write_and_execute(code)
            cells_executed += 1

            if not error:
                output_text = _extract_text(outputs)
                all_outputs.append(output_text)
                result.cell_outputs[cell_id] = output_text

                # Build multimodal feedback — include plot image if one was generated
                feedback_text = f"Cell {step+1} output:\n{output_text[:1000]}"
                plot_b64 = None
                for o in outputs:
                    img = o.get("data", {}).get("image/png")
                    if img:
                        plot_b64 = img
                        break

                conversation.append(step_response)
                if is_done:
                    break

                if plot_b64:
                    # Multimodal feedback — LLM sees the plot + text
                    conversation.append(HumanMessage(content=[
                        {"type": "text", "text": f"{feedback_text}\n\nWhat should the next analysis step be? (step {step+2}/{max_cells})"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{plot_b64}", "detail": "low"}},
                    ]))
                else:
                    conversation.append(HumanMessage(content=f"{feedback_text}\n\nWhat should the next analysis step be? (step {step+2}/{max_cells})"))
            else:
                # Try auto-fix first (instant), then LLM fix
                fixed = _auto_fix(code, error)
                if fixed and fixed != code:
                    push_event(session_id, {
                        "type": "backtrack",
                        "reason": f"Auto-fixing: {error[:60]}",
                        "cell_id": cell_id,
                        "notebook_id": notebook_id,
                    })
                    _, fix_outputs, fix_error = _write_and_execute(fixed, cell_id=cell_id, overwrite=True)
                    if not fix_error:
                        output_text = _extract_text(fix_outputs)
                        all_outputs.append(output_text)
                        result.cell_outputs[cell_id] = output_text
                        conversation.append(step_response)
                        conversation.append(HumanMessage(content=f"Cell {step+1} output (after auto-fix):\n{output_text[:1000]}\n\nWhat next? (step {step+2}/{max_cells})"))
                        continue

                # LLM fix fallback
                if _time_left() < 20:
                    _LOG.warning("Subagent %s skipping LLM fix — low on time", hypothesis_id)
                    continue
                try:
                    fix_response = llm.invoke([
                        SystemMessage(content=f"Fix this Python error. Available columns: [{col_list}]. Respond with ONLY corrected code."),
                        HumanMessage(content=f"Error: {error}\nOriginal code:\n{code}"),
                    ])
                    fixed_code = fix_response.content.strip()
                    if fixed_code.startswith("```"):
                        fixed_code = fixed_code.split("\n", 1)[1] if "\n" in fixed_code else fixed_code[3:]
                        if fixed_code.endswith("```"):
                            fixed_code = fixed_code[:-3]
                        fixed_code = fixed_code.strip()
                    push_event(session_id, {
                        "type": "backtrack",
                        "reason": f"LLM fixing: {error[:80]}",
                        "cell_id": cell_id,
                        "notebook_id": notebook_id,
                    })
                    _, fix_outputs, fix_error = _write_and_execute(fixed_code, cell_id=cell_id, overwrite=True)
                    if not fix_error:
                        output_text = _extract_text(fix_outputs)
                        all_outputs.append(output_text)
                        result.cell_outputs[cell_id] = output_text
                        conversation.append(step_response)
                        conversation.append(HumanMessage(content=f"Cell {step+1} output (after fix):\n{output_text[:1000]}\n\nWhat next? (step {step+2}/{max_cells})"))
                except Exception as fix_exc:
                    _LOG.warning("Subagent error fix failed: %s", fix_exc)

    except Exception as exc:
        _LOG.warning("Subagent adaptive loop failed: %s", exc)
        # Fallback: basic analysis
        if relevant_cols and not all_outputs:
            col = relevant_cols[0]
            _, fallback_outputs, fallback_error = _write_and_execute(f'print(df["{col}"].describe())')
            if fallback_error:
                all_outputs.append(f"Fallback describe for {col} failed: {fallback_error}")
            else:
                output_text = _extract_text(fallback_outputs)
                all_outputs.append(f"Described {col}:\n{output_text}")

    # --- Synthesize conclusion ---
    combined_output = "\n---\n".join(all_outputs[:5])
    if _past_deadline():
        _LOG.warning("Subagent %s hit deadline before conclusion synthesis", hypothesis_id)
        result.finding = f"Investigation of '{hypothesis_title}' produced {len(all_outputs)} analysis steps but ran out of time."
        result.confidence = 0.2
        result.status = "timeout"
        return result

    push_event(session_id, {
        "type": "thinking",
        "content": f"Synthesizing conclusion for: {hypothesis_title} ({len(all_outputs)} evidence steps, {len(result.images)} plots)",
        "notebook_id": notebook_id,
    })

    img_count = 0
    try:
        from src.config.config import get_chat_model
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_chat_model()

        conclusion_prompt = (
            "You are concluding a data investigation. Based on the text evidence"
            " AND any plots shown, write ONE specific, quantitative paragraph about"
            " what was found. Include numbers. If the hypothesis was confirmed, say"
            " so with evidence. If refuted, say so. If plots are shown, describe"
            " what visual patterns you see (clusters, trends, outliers, regime changes)."
            " Format with markdown. For math notation, always use proper LaTeX delimiters:"
            " $x$ for inline math (e.g., $r = 0.95$, $p < 0.05$, $\\Delta$, $\\chi^2$)."
            " Never write raw LaTeX without $ delimiters."
            " Never use emojis or special unicode symbols."
        )

        # Build multimodal content: text + up to 2 plots (detail=low for speed)
        _LOG.info("Subagent %s: synthesizing conclusion (with %d plot images)", hypothesis_id, len(result.images))
        content_parts: list = [
            {"type": "text", "text": (
                f"Hypothesis: {hypothesis_title}\n\n"
                f"Evidence from {len(all_outputs)} analysis steps:\n{combined_output[:3000]}"
            )}
        ]
        for cell_id_img, imgs in result.images.items():
            for img_b64 in imgs:
                if img_count >= 5:
                    break
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "low"},
                })
                img_count += 1
            if img_count >= 2:
                break

        # Single call — if images present it's multimodal, otherwise just text
        if img_count > 0:
            conclusion_response = llm.invoke([
                SystemMessage(content=conclusion_prompt),
                HumanMessage(content=content_parts),
            ])
        else:
            conclusion_response = llm.invoke([
                SystemMessage(content=conclusion_prompt),
                HumanMessage(content=content_parts[0]["text"]),
            ])
        result.finding = conclusion_response.content.strip()
        _LOG.info("Subagent %s: conclusion done (vision=%s): %s", hypothesis_id, img_count > 0, result.finding[:80])

        # Extract p-values from outputs for confidence scoring
        from src.agent.knowledge_graph import compute_finding_confidence, extract_p_values
        all_text = combined_output
        p_vals = extract_p_values(all_text)
        min_p = min(p_vals) if p_vals else None

        import re
        n_match = re.search(r'(?:n|N|rows|observations)\s*[=:]\s*([0-9,]+)', all_text)
        sample_n = int(n_match.group(1).replace(',', '')) if n_match else 100

        result.confidence = compute_finding_confidence(
            has_p_value=bool(p_vals),
            p_value=min_p,
            sample_size=sample_n,
            num_evidence_lines=len(all_outputs),
            has_visual_confirmation=img_count > 0,
            confirmed_by_multiple_methods=len(p_vals) > 1,
        )

    except Exception as exc:
        _LOG.warning("Conclusion synthesis failed: %s", exc)
        if all_outputs:
            result.finding = (
                f"Investigation of '{hypothesis_title}' produced {len(all_outputs)} analysis steps. "
                f"Evidence: {combined_output[:1000]}"
            )
        else:
            result.finding = f"Could not investigate '{hypothesis_title}' due to errors."
        result.confidence = 0.15
        result.status = "failed"

    push_event(session_id, {
        "type": "thinking",
        "content": f"Conclusion ready for: {hypothesis_title} (confidence: {result.confidence:.0%})",
        "notebook_id": notebook_id,
    })
    _LOG.info("Subagent %s complete: %s (confidence=%.2f)",
              hypothesis_id, result.finding[:80], result.confidence)
    return result
