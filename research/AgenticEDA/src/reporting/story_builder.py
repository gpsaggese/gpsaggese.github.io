"""Agentic story builder — curates plots, writes captions, structures narrative."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

_LOG = logging.getLogger(__name__)


def curate_section_plots(
    section: dict[str, Any],
    max_plots: int = 3,
) -> list[dict[str, str]]:
    """Select the best plots for a section and generate unique captions.

    Args:
        section: Section dict with title, content, confidence, plot_metadata.
        max_plots: Maximum plots to include.

    Returns:
        List of {cell_id, caption} dicts for selected plots.
    """
    plot_metadata = section.get("plot_metadata", [])
    if not plot_metadata:
        return []

    if len(plot_metadata) <= max_plots:
        return _generate_captions(section, plot_metadata)

    return _llm_curate(section, plot_metadata, max_plots)


def _generate_captions(
    section: dict[str, Any],
    plots: list[dict],
) -> list[dict[str, str]]:
    """Generate unique captions for each plot using LLM."""
    try:
        from src.config.config import get_chat_model
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_chat_model()

        plots_desc = "\n".join(
            f"Plot {i+1} (cell {p['cell_id']}):\n"
            f"  Code: {p.get('cell_source', 'unknown')[:200]}\n"
            f"  Output: {p.get('cell_output', '')[:200]}"
            for i, p in enumerate(plots)
        )

        response = llm.invoke([
            SystemMessage(content=(
                "You are writing figure captions for a data analysis report. "
                "For each plot, write ONE specific caption (1-2 sentences) that describes "
                "what the plot shows and what pattern is visible. Reference specific numbers "
                "from the output where possible. Use $...$ for inline math. "
                "Never use emojis or special unicode symbols.\n\n"
                "Respond with JSON (no markdown fencing):\n"
                '[{"cell_id": "...", "caption": "..."}, ...]'
            )),
            HumanMessage(content=(
                f"Section: {section.get('title', '')}\n"
                f"Finding: {section.get('content', '')[:500]}\n\n"
                f"Plots to caption:\n{plots_desc}"
            )),
        ])

        text = _strip_markdown_fences(response.content.strip())
        result = json.loads(text)
        valid = []
        for item in result:
            if isinstance(item, dict) and "cell_id" in item and "caption" in item:
                valid.append({"cell_id": item["cell_id"], "caption": item["caption"]})
        return valid if valid else _fallback_captions(section, plots)

    except Exception as exc:
        _LOG.warning("Caption generation failed: %s", exc)
        return _fallback_captions(section, plots)


def _llm_curate(
    section: dict[str, Any],
    plots: list[dict],
    max_plots: int,
) -> list[dict[str, str]]:
    """Use LLM to select best plots and generate captions."""
    try:
        from src.config.config import get_chat_model
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_chat_model()

        plots_desc = "\n".join(
            f"Plot {i+1} (cell {p['cell_id']}):\n"
            f"  Code: {p.get('cell_source', 'unknown')[:300]}\n"
            f"  Output: {p.get('cell_output', '')[:300]}"
            for i, p in enumerate(plots)
        )

        response = llm.invoke([
            SystemMessage(content=(
                f"You are curating figures for a data analysis report. "
                f"Select the {max_plots} most informative and visually distinct plots. "
                f"Skip plots that are redundant (e.g., same scatter with minor variations). "
                f"Prefer: (1) plots showing the key finding, (2) statistical test visualizations, "
                f"(3) plots with different chart types (scatter vs histogram vs bar). "
                f"For each selected plot, write a specific caption (1-2 sentences) with numbers. "
                f"Never use emojis or special unicode symbols.\n\n"
                f"Respond with JSON (no markdown fencing):\n"
                f'[{{"cell_id": "...", "caption": "..."}}]\n\n'
                f"Select exactly {max_plots} plots. Use $...$ for math in captions."
            )),
            HumanMessage(content=(
                f"Section: {section.get('title', '')}\n"
                f"Finding: {section.get('content', '')[:800]}\n"
                f"Confidence: {section.get('confidence', 'N/A')}\n\n"
                f"Candidate plots ({len(plots)} total):\n{plots_desc}"
            )),
        ])

        text = _strip_markdown_fences(response.content.strip())
        result = json.loads(text)
        valid = []
        for item in result:
            if isinstance(item, dict) and "cell_id" in item and "caption" in item:
                valid.append({"cell_id": item["cell_id"], "caption": item["caption"]})
        if valid:
            return valid[:max_plots]
        return _fallback_captions(section, plots[:max_plots])

    except Exception as exc:
        _LOG.warning("Plot curation failed: %s", exc)
        return _fallback_captions(section, plots[:max_plots])


def _fallback_captions(
    section: dict[str, Any],
    plots: list[dict],
) -> list[dict[str, str]]:
    """Generate simple captions without LLM (deterministic fallback)."""
    title = section.get("title", "Analysis")
    result = []
    for p in plots:
        source = p.get("cell_source", "")
        caption = title
        if "plt.title(" in source:
            m = re.search(r"plt\.title\(['\"](.+?)['\"]\)", source)
            if m:
                caption = m.group(1)
        elif "scatter" in source:
            caption = f"Scatter plot: {title}"
        elif "hist" in source:
            caption = f"Distribution: {title}"
        elif "heatmap" in source or "imshow" in source:
            caption = f"Heatmap: {title}"
        elif "bar" in source:
            caption = f"Comparison: {title}"
        elif "plot" in source:
            caption = f"Trend: {title}"
        result.append({"cell_id": p["cell_id"], "caption": caption})
    return result


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def build_story_abstract(
    executive_summary: str,
    max_sentences: int = 5,
) -> str:
    """Condense executive summary into a concise abstract for the PDF."""
    try:
        from src.config.config import get_chat_model
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_chat_model()
        response = llm.invoke([
            SystemMessage(content=(
                f"Condense this executive summary into exactly {max_sentences} sentences "
                f"for a paper abstract. Keep the most important quantitative findings. "
                f"Use $...$ for math notation. Do not add information not in the original. "
                f"Never use emojis or special unicode symbols."
            )),
            HumanMessage(content=executive_summary[:3000]),
        ])
        return response.content.strip()
    except Exception as exc:
        _LOG.warning("Abstract condensation failed: %s", exc)
        sentences = executive_summary.replace("\n", " ").split(". ")
        return ". ".join(sentences[:max_sentences]) + "."


def build_curated_story(
    sections: list[dict],
    executive_summary: str,
    dataset_name: str = "dataset",
    max_plots_per_section: int = 3,
) -> dict:
    """Build a curated story with selected plots and unique captions.

    Args:
        sections: Raw sections from KG with plot_metadata.
        executive_summary: Full executive summary text.
        dataset_name: Name of the dataset file.
        max_plots_per_section: Max figures per section.

    Returns:
        Curated story dict ready for story.json.
    """
    import datetime

    curated_sections = []
    for section in sections:
        curated = dict(section)

        if section.get("plot_metadata"):
            selected = curate_section_plots(section, max_plots=max_plots_per_section)
            curated["selected_plots"] = selected
            selected_ids = {s["cell_id"] for s in selected}
            curated["plot_cell_ids"] = [pid for pid in curated.get("plot_cell_ids", []) if pid in selected_ids]
            curated["plot_captions"] = {s["cell_id"]: s["caption"] for s in selected}
        curated_sections.append(curated)

    abstract = build_story_abstract(executive_summary)

    return {
        "title": f"EDA Report: {dataset_name}",
        "executive_summary": executive_summary,
        "abstract": abstract,
        "sections": curated_sections,
        "generated_at": datetime.datetime.now().isoformat(),
    }
