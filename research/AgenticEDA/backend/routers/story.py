"""Story router: fetch, export, and regenerate the narrative story."""
from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import re
import io
import unicodedata
import tempfile
import textwrap
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response

import nbformat

from backend.routers.stream import push_event

from backend.services.session_manager import get_session_dir
from src.reporting.plot_contract import (
    build_report_viz_spec,
    infer_plot_semantics,
    plot_artifacts_from_outputs,
    plot_specs_by_cell,
)

_LOG = logging.getLogger(__name__)
router = APIRouter(tags=["story"])


def atomic_write_json(path: pathlib.Path, data: dict) -> None:
    """Write JSON atomically — write to temp file then rename (prevents corruption)."""
    content = json.dumps(data, default=str, indent=2)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, content.encode())
        os.close(fd)
        os.replace(tmp, str(path))
    except Exception:
        os.close(fd) if not os.get_inheritable(fd) else None
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _load_story(session_id: str) -> dict:
    """Load story.json for a session, or raise 404."""
    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    story_file = session_dir / "story.json"
    if not story_file.exists():
        raise HTTPException(status_code=404, detail="Story not generated yet")

    try:
        return json.loads(story_file.read_text())
    except json.JSONDecodeError as exc:
        _LOG.warning("Corrupt story.json for session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Story data is corrupted. Try regenerating.")


def _default_plot_display(plot_type: str, index: int = 0) -> dict[str, Any]:
    if plot_type == "plotly":
        return {
            "width": 960,
            "height": 540,
            "aspect_ratio": 960 / 540,
            "fit": "contain",
            "max_width": 960,
            "max_height": 540,
            "slot": index + 1,
        }
    return {
        "width": 960,
        "height": 540,
        "aspect_ratio": 960 / 540,
        "fit": "contain",
        "max_width": 960,
        "max_height": 540,
        "slot": index + 1,
    }


def _clean_base64_payload(value: str) -> str:
    return value.replace("\n", "").replace("\r", "").replace(" ", "")


def _coerce_dimension(value: Any) -> int | None:
    try:
        if value is None:
            return None
        numeric = int(float(value))
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return max(240, min(numeric, 1600))


def _frame_from_dimensions(plot_type: str, width: int | None, height: int | None, index: int = 0) -> dict[str, Any]:
    frame = _default_plot_display(plot_type, index)
    if width and height:
        ratio = max(1.1, min(width / height, 2.1))
        frame["aspect_ratio"] = ratio
        frame["height"] = max(300, min(height, 540))
        frame["width"] = max(320, min(width, 960))
        frame["native_width"] = width
        frame["native_height"] = height
    return frame


def _normalize_plotly_payload(payload: Any, index: int = 0) -> tuple[str, dict[str, Any]]:
    try:
        parsed = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
    except Exception:
        return str(payload), _default_plot_display("plotly", index)

    layout = parsed.get("layout") or {}
    if not isinstance(layout, dict):
        layout = {}
        parsed["layout"] = layout

    width = _coerce_dimension(layout.pop("width", None))
    height = _coerce_dimension(layout.pop("height", None))
    frame = _frame_from_dimensions("plotly", width, height, index)

    layout["autosize"] = True
    layout["paper_bgcolor"] = "rgba(0,0,0,0)"
    layout["plot_bgcolor"] = "rgba(0,0,0,0)"
    if not isinstance(layout.get("margin"), dict):
        layout["margin"] = {"t": 28, "r": 20, "b": 52, "l": 60}

    return json.dumps(parsed), frame


def _image_frame_from_output_metadata(metadata: Any, index: int = 0) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return _default_plot_display("image", index)

    image_meta = metadata.get("image/png")
    width = None
    height = None
    if isinstance(image_meta, dict):
        width = _coerce_dimension(image_meta.get("width"))
        height = _coerce_dimension(image_meta.get("height"))

    width = width or _coerce_dimension(metadata.get("width"))
    height = height or _coerce_dimension(metadata.get("height"))
    return _frame_from_dimensions("image", width, height, index)


def _normalize_plot_entry(
    plot: Any,
    *,
    section: dict[str, Any],
    cell_id: str | None = None,
    index: int = 0,
) -> dict[str, Any]:
    title = section.get("visual_title") or section.get("title") or section.get("phase") or "Figure"
    caption = section.get("visual_caption") or ""

    if isinstance(plot, dict):
        normalized = dict(plot)
        source = normalized.get("source") or normalized.get("payload") or normalized.get("path") or ""
        plot_kind = str(normalized.get("kind") or normalized.get("type") or "image").strip().lower()
        display = normalized.get("display") or normalized.get("frame")

        if plot_kind == "plotly":
            source, display_from_source = _normalize_plotly_payload(source, index)
            display = display or display_from_source
        else:
            source = str(source)
            if source and not source.startswith("http") and not source.startswith("data:"):
                cleaned = _clean_base64_payload(source)
                if len(cleaned) > 120 and re.fullmatch(r"[A-Za-z0-9+/=]+", cleaned):
                    source = f"data:image/png;base64,{cleaned}"
            display = display or _default_plot_display("image", index)

        normalized["kind"] = "plotly" if plot_kind == "plotly" else "image"
        normalized["mime_type"] = normalized.get("mime_type") or (
            "application/vnd.plotly.v1+json" if normalized["kind"] == "plotly" else "image/png"
        )
        normalized["source"] = source
        normalized["title"] = normalized.get("title") or title
        normalized["caption"] = normalized.get("caption") or caption
        normalized["source_cell_id"] = normalized.get("source_cell_id") or cell_id or ""
        normalized["display"] = display
        semantic = infer_plot_semantics(
            normalized["kind"],
            str(source),
            title=str(normalized["title"]),
            caption=str(normalized["caption"]),
        )
        for field in ("chart_family", "semantic_intent", "x_axis_role", "y_axis_role", "semantic_confidence"):
            if normalized.get(field) in (None, "", "unknown"):
                normalized[field] = semantic[field]
        normalized["viz_spec"] = build_report_viz_spec(normalized).model_dump()
        normalized.pop("payload", None)
        normalized.pop("frame", None)
        return normalized

    raw = str(plot).strip()
    if not raw:
        return {
            "kind": "image",
            "mime_type": "image/png",
            "source": "",
            "title": title,
            "caption": caption,
            "source_cell_id": cell_id or "",
            "display": _default_plot_display("image", index),
            "viz_spec": build_report_viz_spec(
                {
                    "kind": "image",
                    "source": "",
                    "title": title,
                    "caption": caption,
                    "display": _default_plot_display("image", index),
                    "chart_family": "image",
                    "semantic_intent": "unknown",
                    "x_axis_role": "unknown",
                    "y_axis_role": "unknown",
                    "semantic_confidence": 1.0,
                }
            ).model_dump(),
        }

    if raw.startswith("http") or raw.startswith("data:"):
        semantic = infer_plot_semantics("image", raw, title=title, caption=caption)
        return {
            "kind": "image",
            "mime_type": "image/png",
            "source": raw,
            "title": title,
            "caption": caption,
            "source_cell_id": cell_id or "",
            "display": _default_plot_display("image", index),
            "viz_spec": build_report_viz_spec(
                {
                    "kind": "image",
                    "source": raw,
                    "title": title,
                    "caption": caption,
                    "display": _default_plot_display("image", index),
                    **semantic,
                }
            ).model_dump(),
            **semantic,
        }

    cleaned = _clean_base64_payload(raw)
    if cleaned.startswith("{") or cleaned.startswith("["):
        source, display = _normalize_plotly_payload(raw, index)
        semantic = infer_plot_semantics("plotly", source, title=title, caption=caption)
        return {
            "kind": "plotly",
            "mime_type": "application/vnd.plotly.v1+json",
            "source": source,
            "title": title,
            "caption": caption,
            "source_cell_id": cell_id or "",
            "display": display,
            "viz_spec": build_report_viz_spec(
                {
                    "kind": "plotly",
                    "source": source,
                    "title": title,
                    "caption": caption,
                    "display": display,
                    **semantic,
                }
            ).model_dump(),
            **semantic,
        }

    if len(cleaned) > 120 and re.fullmatch(r"[A-Za-z0-9+/=]+", cleaned):
        semantic = infer_plot_semantics("image", cleaned, title=title, caption=caption)
        return {
            "kind": "image",
            "mime_type": "image/png",
            "source": f"data:image/png;base64,{cleaned}",
            "title": title,
            "caption": caption,
            "source_cell_id": cell_id or "",
            "display": _default_plot_display("image", index),
            "viz_spec": build_report_viz_spec(
                {
                    "kind": "image",
                    "source": f"data:image/png;base64,{cleaned}",
                    "title": title,
                    "caption": caption,
                    "display": _default_plot_display("image", index),
                    **semantic,
                }
            ).model_dump(),
            **semantic,
        }

    semantic = infer_plot_semantics("image", raw, title=title, caption=caption)
    return {
        "kind": "image",
        "mime_type": "image/png",
        "source": raw,
        "title": title,
        "caption": caption,
        "source_cell_id": cell_id or "",
        "display": _default_plot_display("image", index),
        "viz_spec": build_report_viz_spec(
            {
                "kind": "image",
                "source": raw,
                "title": title,
                "caption": caption,
                "display": _default_plot_display("image", index),
                **semantic,
            }
        ).model_dump(),
        **semantic,
    }


def _augment_story_plot_metadata(session_id: str, story: dict) -> dict:
    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        return story

    nb_path = session_dir / "notebook.ipynb"
    cell_map: dict[str, Any] = {}
    if nb_path.exists():
        try:
            nb = nbformat.read(str(nb_path), as_version=4)
            for cell in nb.cells:
                cell_id = cell.get("id")
                if cell_id:
                    cell_map[str(cell_id)] = cell
        except Exception:
            cell_map = {}

    plot_specs_map = plot_specs_by_cell(session_dir)

    sections = story.get("sections") or []
    for section in sections:
        if not isinstance(section, dict):
            continue

        plot_metadata: list[dict[str, Any]] = []
        plot_cell_ids = section.get("plot_cell_ids") or section.get("cell_ids") or []
        raw_plots = section.get("plots") or []

        for index, plot in enumerate(raw_plots):
            cell_id = plot_cell_ids[index] if index < len(plot_cell_ids) else None
            # Attach hidden plot specs from sidecar JSONL when not already present
            if cell_id and isinstance(plot, dict) and not plot.get("plot_spec"):
                cell_specs = plot_specs_map.get(str(cell_id), [])
                if cell_specs:
                    plot = dict(plot)
                    plot["plot_spec"] = cell_specs[0]
            plot_metadata.append(
                _normalize_plot_entry(
                    plot,
                    section=section,
                    cell_id=cell_id,
                    index=index,
                )
            )

        if not plot_metadata and plot_cell_ids:
            for index, cell_id in enumerate(plot_cell_ids):
                cell = cell_map.get(str(cell_id))
                if cell:
                    plot_metadata.extend(
                        plot_artifacts_from_outputs(
                            cell.get("outputs", []) or [],
                            title=section.get("title", ""),
                            caption=section.get("visual_caption", ""),
                            source_cell_id=str(cell_id),
                            plot_specs=plot_specs_map.get(str(cell_id), []),
                        )
                    )
                else:
                    # No notebook on disk — build artifacts directly from
                    # hidden plot specs in plot_specs.jsonl (only specs with
                    # actual Plotly data, not metadata-only stubs)
                    cell_specs = plot_specs_map.get(str(cell_id), [])
                    seen_sources = set()
                    unique_specs = []
                    for spec in cell_specs:
                        if not isinstance(spec, dict):
                            continue
                        source = spec.get("source")
                        if not source:
                            continue
                        if isinstance(source, dict) and "data" not in source:
                            continue
                        source_key = (
                            json.dumps(source, sort_keys=True, default=str)
                            if isinstance(source, dict)
                            else str(source)
                        )
                        if source_key in seen_sources:
                            continue
                        seen_sources.add(source_key)
                        unique_specs.append(spec)
                    for spec in unique_specs:
                        source = spec.get("source")
                        source_str = json.dumps(source, default=str) if isinstance(source, dict) else str(source)
                        plot_metadata.append(
                            _normalize_plot_entry(
                                {
                                    "kind": str(spec.get("kind", "plotly")).lower(),
                                    "mime_type": spec.get("mime_type", "application/vnd.plotly.v1+json"),
                                    "source": source_str,
                                    "title": spec.get("title") or section.get("title", ""),
                                    "caption": spec.get("caption") or section.get("visual_caption", ""),
                                    "source_cell_id": str(cell_id),
                                    "chart_family": spec.get("chart_family", "unknown"),
                                    "semantic_intent": spec.get("semantic_intent", "unknown"),
                                    "x_axis_role": spec.get("x_axis_role", "unknown"),
                                    "y_axis_role": spec.get("y_axis_role", "unknown"),
                                    "semantic_confidence": float(spec.get("semantic_confidence", 0.85)),
                                    "plot_spec": spec,
                                },
                                section=section,
                                cell_id=str(cell_id),
                                index=index,
                            )
                        )

        # Filter to curated selection if available
        selected_plots_list = section.get("selected_plots")
        if selected_plots_list and plot_metadata:
            selected_ids = {s["cell_id"] for s in selected_plots_list}
            caption_map = section.get("plot_captions", {})
            filtered = []
            for p in plot_metadata:
                src_id = p.get("source_cell_id", "")
                if src_id in selected_ids:
                    if src_id in caption_map:
                        p["caption"] = caption_map[src_id]
                        p["title"] = caption_map[src_id]
                    filtered.append(p)
            plot_metadata = filtered if filtered else plot_metadata[:3]

        if plot_metadata:
            section["plots"] = plot_metadata

    return story


def _story_to_markdown(story: dict) -> str:
    """Convert story JSON to markdown string."""
    lines = [f"# {story.get('title', 'EDA Report')}\n"]
    summary = story.get("executive_summary", "")
    if summary:
        lines.append(f"{summary}\n")
    for section in story.get("sections", []):
        lines.append(f"## {section.get('title', section.get('phase', ''))}\n")
        lines.append(f"{section.get('content', '')}\n")

        plot_metadata = section.get("plots") or []
        if isinstance(plot_metadata, list) and plot_metadata:
            lines.append("\n**Figures:**\n")
            for plot in plot_metadata:
                if isinstance(plot, dict):
                    payload = str(plot.get("source", "")).strip()
                    caption = str(plot.get("caption") or plot.get("title") or "Figure").strip()
                    if plot.get("kind") == "image" and payload:
                        lines.append(f"![{caption}]({payload})\n")
                    elif payload:
                        lines.append(f"- {caption}\n")
                else:
                    lines.append(f"- {plot}\n")
    return "\n".join(lines)


def _story_to_pdf_lines(story: dict) -> list[tuple[str, str]]:
    """Flatten a story dict into styled text lines for the PDF fallback."""
    lines: list[tuple[str, str]] = []

    def add_heading(text: str, level: str = "h1") -> None:
        if text:
            lines.append((level, text.strip()))

    def add_paragraph(text: str) -> None:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            lines.append(("blank", ""))
            return
        for wrapped in textwrap.wrap(cleaned, width=90, break_long_words=False, break_on_hyphens=False):
            lines.append(("p", wrapped))

    add_heading(story.get("title", "EDA Report"), "h1")
    generated_at = story.get("generated_at")
    if generated_at:
        add_paragraph(f"Generated {generated_at}")

    summary = story.get("executive_summary", "")
    if summary:
        add_heading("Executive Summary", "h2")
        for paragraph in summary.split("\n"):
            add_paragraph(paragraph)

    for section in story.get("sections", []):
        title = section.get("title", section.get("phase", "Section"))
        add_heading(title, "h2")
        content = section.get("content", "")
        if content:
            for paragraph in content.split("\n"):
                add_paragraph(paragraph)

        insights = section.get("insights") or []
        if isinstance(insights, list) and insights:
            add_heading("Key Takeaways", "h3")
            for insight in insights:
                if isinstance(insight, dict):
                    desc = str(insight.get("description", "")).strip()
                    label = str(insight.get("type", "")).strip()
                    parts = [p for p in [label, desc] if p]
                    if parts:
                        add_paragraph(f"- {' - '.join(parts)}")
                else:
                    add_paragraph(f"- {insight}")

        plots = section.get("plots") or []
        if isinstance(plots, list) and plots:
            add_heading("Visuals", "h3")
            for plot in plots[:6]:
                if isinstance(plot, dict):
                    caption = str(plot.get("caption") or plot.get("title") or "Figure").strip()
                    payload = str(plot.get("source") or "").strip()
                    if plot.get("kind") == "image" and payload:
                        add_paragraph(f"- {caption}: {payload}")
                    else:
                        add_paragraph(f"- {caption}")
                else:
                    add_paragraph(f"- {plot}")

        lines.append(("blank", ""))

    return lines


def _escape_pdf_text(text: str) -> str:
    safe = unicodedata.normalize("NFKD", text).encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_minimal_pdf(story: dict) -> bytes:
    """Build a simple text PDF without third-party rendering dependencies."""
    width = 612
    height = 792
    margin_x = 54
    margin_top = 54
    line_height = 14
    bottom_margin = 54
    max_lines_per_page = max(1, (height - margin_top - bottom_margin) // line_height)

    flattened = _story_to_pdf_lines(story)
    pages: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    for kind, text in flattened:
        if current and len(current) >= max_lines_per_page:
            pages.append(current)
            current = []
        current.append((kind, text))
    if current:
        pages.append(current)

    def render_page(page_lines: list[tuple[str, str]]) -> bytes:
        content = io.StringIO()
        content.write("BT\n")
        y = height - margin_top
        for kind, text in page_lines:
            if kind == "blank":
                y -= line_height
                continue

            font = "/F2" if kind in {"h1", "h2", "h3"} else "/F1"
            size = "18" if kind == "h1" else "14" if kind == "h2" else "12" if kind == "h3" else "11"
            content.write(f"{font} {size} Tf\n")
            content.write(f"1 0 0 1 {margin_x} {y} Tm\n")
            content.write(f"({_escape_pdf_text(text)}) Tj\n")
            y -= line_height if kind == "p" else line_height + 4
        content.write("ET\n")
        return content.getvalue().encode("latin-1", "replace")

    object_defs: list[tuple[int, bytes]] = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, b""),
        (3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        (4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"),
    ]

    next_obj = 5
    page_nums: list[int] = []
    for page_lines in pages or [[("h1", story.get("title", "EDA Report"))]]:
        content_stream = render_page(page_lines)
        content_obj_num = next_obj + 1
        page_nums.append(next_obj)
        object_defs.append(
            (
                next_obj,
                (
                    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 "
                    + str(width).encode("ascii")
                    + b" "
                    + str(height).encode("ascii")
                    + b"] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents "
                    + str(content_obj_num).encode("ascii")
                    + b" 0 R >>"
                ),
            )
        )
        object_defs.append(
            (
                content_obj_num,
                b"<< /Length "
                + str(len(content_stream)).encode("ascii")
                + b" >>\nstream\n"
                + content_stream
                + b"endstream",
            )
        )
        next_obj += 2

    object_defs[1] = (
        2,
        b"<< /Type /Pages /Kids ["
        + b" ".join(f"{num} 0 R".encode("ascii") for num in page_nums)
        + b"] /Count "
        + str(len(page_nums)).encode("ascii")
        + b" >>",
    )

    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj_num, obj_bytes in sorted(object_defs, key=lambda item: item[0]):
        offsets.append(buffer.tell())
        buffer.write(f"{obj_num} 0 obj\n".encode("ascii"))
        buffer.write(obj_bytes)
        buffer.write(b"\nendobj\n")

    xref_start = buffer.tell()
    buffer.write(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    buffer.write(
        b"trailer\n<< /Size "
        + str(len(offsets)).encode("ascii")
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_start).encode("ascii")
        + b"\n%%EOF"
    )
    return buffer.getvalue()


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("**", "")  # Strip markdown bold
        .replace("*", "")   # Strip markdown italic
    )


import re as _re

# LaTeX command → Unicode mappings for PDF export
_LATEX_UNICODE = {
    "\\times": "×", "\\approx": "≈", "\\geq": "≥", "\\leq": "≤",
    "\\pm": "±", "\\neq": "≠", "\\sim": "∼", "\\propto": "∝",
    "\\infty": "∞", "\\cdot": "·", "\\ldots": "…", "\\equiv": "≡",
    "\\Delta": "Δ", "\\Sigma": "Σ", "\\Omega": "Ω", "\\Gamma": "Γ",
    "\\Lambda": "Λ", "\\Pi": "Π",
    "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
    "\\epsilon": "ε", "\\eta": "η", "\\theta": "θ", "\\lambda": "λ",
    "\\mu": "μ", "\\nu": "ν", "\\xi": "ξ", "\\pi": "π", "\\rho": "ρ",
    "\\sigma": "σ", "\\tau": "τ", "\\phi": "φ", "\\chi": "χ",
    "\\psi": "ψ", "\\omega": "ω", "\\kappa": "κ", "\\zeta": "ζ",
    "\\ll": "≪", "\\gg": "≫", "\\ge": "≥", "\\le": "≤",
}

# Bare commands (backslash eaten by Python)
_BARE_COMMANDS = {
    "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω", "Gamma": "Γ", "Lambda": "Λ",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "eta": "η", "theta": "θ", "lambda": "λ", "mu": "μ", "rho": "ρ",
    "sigma": "σ", "chi": "χ", "phi": "φ", "psi": "ψ", "omega": "ω",
    "times": "×", "approx": "≈", "infty": "∞", "neq": "≠", "sim": "∼",
    "propto": "∝", "equiv": "≡",
}


def _latex_math_to_html(math_content: str) -> str:
    """Convert a LaTeX math expression to HTML with Unicode symbols."""
    s = math_content
    # \text{...} → plain text
    s = _re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = _re.sub(r"\\textbf\{([^}]*)\}", r"<b>\1</b>", s)
    s = _re.sub(r"\\textit\{([^}]*)\}", r"<i>\1</i>", s)
    s = _re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    # Bare text commands (backslash eaten): textFoo → Foo
    s = _re.sub(r"(?<![a-zA-Z\\])text([A-Z][a-zA-Z_]*)", r"\1", s)
    s = _re.sub(r"(?<![a-zA-Z\\])text\{([^}]*)\}", r"\1", s)
    # \frac{a}{b} → a/b
    s = _re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1/\2)", s)
    # Replace backslash commands with Unicode
    for cmd, uni in _LATEX_UNICODE.items():
        s = s.replace(cmd, uni)
    # Replace bare commands (Python ate backslash) — more aggressive inside math
    # Uppercase Greek: DeltaP → ΔP, SigmaX → ΣX
    for cmd in ("Delta", "Sigma", "Omega", "Gamma", "Lambda", "Pi"):
        if cmd in _BARE_COMMANDS:
            s = _re.sub(rf"(?<![a-zA-Z\\]){cmd}", _BARE_COMMANDS[cmd], s)
    # Lowercase Greek and operators
    for cmd, uni in _BARE_COMMANDS.items():
        if cmd[0].isupper():
            continue  # already handled above
        s = _re.sub(rf"(?<![a-zA-Z\\]){cmd}(?=[^a-z]|$)", uni, s)
    # x^{exp} → x<sup>exp</sup>
    s = _re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", s)
    s = _re.sub(r"\^(-?\d+)", r"<sup>\1</sup>", s)
    # x_{sub} → x<sub>sub</sub>
    s = _re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", s)
    # Single-char subscript, but NOT underscores in variable names like Power_Curve
    s = _re.sub(r"(?<=[a-zA-Z])_(?=[a-zA-Z]{2,})", "_", s)  # keep word_word underscores
    s = _re.sub(r"_([a-zA-Z0-9])(?![a-zA-Z])", r"<sub>\1</sub>", s)  # only single-char subscripts
    # Clean up remaining backslashes and braces
    s = s.replace("\\", "").replace("{", "").replace("}", "")
    return s


def _render_math_in_text(text: str) -> str:
    """Convert LaTeX $...$ and $$...$$ in text to HTML with Unicode math."""
    # Fix double-escaped backslashes: \\text → \text, \\rho → \rho, etc.
    text = _re.sub(
        r"\\\\(text|textit|textbf|textrm|texttt|mathrm|mathit|mathbf|frac|sqrt|"
        r"bar|hat|tilde|vec|overline|underline|times|approx|g?eq|leq|pm|"
        r"Delta|Sigma|Omega|Gamma|Lambda|Pi|"
        r"delta|sigma|omega|gamma|lambda|alpha|beta|eta|rho|chi|mu|theta|phi|psi|"
        r"epsilon|kappa|tau|pi|zeta|nu|xi|"
        r"infty|neq|sim|propto|cdot|ldots|ll|gg|equiv|[,;!>]|quad|qquad"
        r")(?![a-zA-Z])",
        r"\\\1", text,
    )
    # $$...$$ display math
    text = _re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: _latex_math_to_html(m.group(1)), text)
    # $...$ inline math
    text = _re.sub(r"\$([^$\n]+?)\$", lambda m: _latex_math_to_html(m.group(1)), text)
    # Fix bare commands outside math (Python ate backslash)
    for cmd, uni in _BARE_COMMANDS.items():
        text = _re.sub(rf"(?<![a-zA-Z\\]){cmd}(?=[^a-z]|$)", uni, text)
    # Handle bare exponents/subscripts outside math: 10^{-8} → 10<sup>-8</sup>
    text = _re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", text)
    text = _re.sub(r"\^(-?\d+)", r"<sup>\1</sup>", text)
    text = _re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", text)
    return text


def _roman(n: int) -> str:
    """Convert integer to Roman numeral."""
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    result = ""
    for v, s in vals:
        while n >= v:
            result += s
            n -= v
    return result


def _story_to_ieee_html_legacy(story_data: dict) -> str:
    """Convert story data to IEEE-style HTML for PDF rendering (legacy, uses column-count)."""
    title = story_data.get("title", "EDA Report")
    summary = story_data.get("executive_summary", "")
    sections = story_data.get("sections", [])
    generated_at = story_data.get("generated_at", "")

    # Build sections HTML
    sections_html = ""
    fig_counter = 0
    for i, section in enumerate(sections):
        sec_title = section.get("title", f"Section {i+1}")
        content = section.get("content", "")
        # Convert markdown-like content to HTML paragraphs
        content_html = ""
        for para in content.split("\n"):
            para = para.strip()
            if not para:
                continue
            if para.startswith("- "):
                content_html += f"<li>{_render_math_in_text(_escape_html(para[2:]))}</li>\n"
            elif para.startswith("**") and para.endswith("**"):
                content_html += f"<p><strong>{_render_math_in_text(_escape_html(para[2:-2]))}</strong></p>\n"
            else:
                content_html += f"<p>{_render_math_in_text(_escape_html(para))}</p>\n"

        # Add plots as figures
        plots_html = ""
        for plot in section.get("plots", []):
            fig_counter += 1
            img_src = ""
            caption = plot.get("caption", plot.get("title", f"Figure {fig_counter}"))
            if plot.get("kind") == "image" and plot.get("source"):
                src = plot["source"]
                if not src.startswith("data:"):
                    src = f"data:image/png;base64,{src}"
                img_src = src
            elif isinstance(plot.get("source"), str) and plot.get("source", "").startswith("data:"):
                img_src = plot["source"]
            else:
                # Plotly JSON or other non-image — skip for PDF
                continue
            if img_src:
                plots_html += f'''
                <div class="figure">
                    <img src="{img_src}" alt="{_escape_html(str(caption))}" />
                    <p class="fig-caption">Fig. {fig_counter}. {_escape_html(str(caption))}</p>
                </div>'''

        section_num = _roman(i + 1)
        sections_html += f'''
        <div class="section">
            <h2>{section_num}. {_escape_html(sec_title)}</h2>
            {content_html}
            {plots_html}
        </div>'''

    return f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: letter;
    margin: 1in 0.75in;
}}
body {{
    font-family: "Times New Roman", Times, serif;
    font-size: 10pt;
    line-height: 1.4;
    color: #000;
    column-count: 2;
    column-gap: 0.25in;
    text-align: justify;
    max-width: none;
}}
h1 {{
    column-span: all;
    text-align: center;
    font-size: 22pt;
    font-weight: bold;
    margin: 0 0 4pt 0;
    line-height: 1.2;
}}
.authors {{
    column-span: all;
    text-align: center;
    font-size: 10pt;
    margin-bottom: 8pt;
    color: #333;
}}
.date {{
    column-span: all;
    text-align: center;
    font-size: 9pt;
    color: #666;
    margin-bottom: 16pt;
}}
.abstract {{
    column-span: all;
    margin: 0 0.5in 16pt 0.5in;
    font-size: 9pt;
}}
.abstract h3 {{
    font-size: 10pt;
    font-style: italic;
    font-weight: bold;
    margin: 0 0 4pt 0;
    text-align: center;
}}
.abstract p {{
    text-align: justify;
    margin: 0;
}}
h2 {{
    font-size: 11pt;
    font-weight: bold;
    text-transform: uppercase;
    text-align: center;
    margin: 12pt 0 6pt 0;
}}
h3 {{
    font-size: 10pt;
    font-style: italic;
    font-weight: bold;
    margin: 8pt 0 4pt 0;
}}
p {{
    margin: 0 0 6pt 0;
    text-indent: 0.25in;
}}
p:first-child {{
    text-indent: 0;
}}
li {{
    margin: 2pt 0;
    font-size: 9pt;
}}
.figure {{
    break-inside: avoid;
    text-align: center;
    margin: 8pt 0;
    padding: 4pt;
    border: 0.5pt solid #ccc;
}}
.figure img {{
    max-width: 100%;
    height: auto;
    max-height: 3in;
}}
.fig-caption {{
    font-size: 8pt;
    text-indent: 0;
    margin: 4pt 0 0 0;
    text-align: center;
    font-style: italic;
}}
.section {{
    break-inside: auto;
}}
.footer {{
    column-span: all;
    text-align: center;
    font-size: 8pt;
    color: #999;
    margin-top: 16pt;
    border-top: 0.5pt solid #ccc;
    padding-top: 4pt;
}}
strong {{ font-weight: bold; }}
em {{ font-style: italic; }}
</style>
</head>
<body>
<h1>{_escape_html(title)}</h1>
<p class="authors">AgenticEDA Automated Analysis</p>
<p class="date">Generated: {_escape_html(generated_at[:10] if generated_at else "")}</p>

<div class="abstract">
    <h3>Abstract</h3>
    <p>{_render_math_in_text(_escape_html(summary[:2000]))}</p>
</div>

{sections_html}

<div class="footer">
    This report was automatically generated by AgenticEDA.
</div>
</body>
</html>'''


def _story_to_ieee_markdown(story_data: dict) -> str:
    """Convert story data to clean IEEE-style markdown."""
    lines = []
    title = story_data.get("title", "EDA Report")
    summary = story_data.get("executive_summary", "")
    sections = story_data.get("sections", [])
    generated_at = story_data.get("generated_at", "")

    lines.append(f"# {title}\n")
    lines.append(f"*AgenticEDA Automated Analysis — {generated_at[:10] if generated_at else ''}*\n")
    lines.append(f"## Abstract\n")
    lines.append(f"{summary}\n")

    fig_counter = 0
    for i, section in enumerate(sections):
        sec_title = section.get("title", f"Section {i+1}")
        content = section.get("content", "")
        section_num = _roman(i + 1)
        lines.append(f"## {section_num}. {sec_title}\n")
        lines.append(f"{content}\n")

        for plot in section.get("plots", []):
            fig_counter += 1
            caption = plot.get("caption", plot.get("title", f"Figure {fig_counter}"))
            if plot.get("kind") == "image" and plot.get("source"):
                src = plot["source"]
                lines.append(f"\n![Fig. {fig_counter}. {caption}]({src[:80]}...)\n")
                lines.append(f"*Fig. {fig_counter}. {caption}*\n")

    lines.append(f"\n---\n*This report was automatically generated by AgenticEDA.*\n")
    return "\n".join(lines)


def _story_pdf_filename(story: dict) -> str:
    """Build a safe PDF filename from the story title."""
    title = str(story.get("title", "eda-report")).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", title).strip("-")
    return f"{slug or 'eda-report'}.pdf"


def _export_story_pdf(story: dict) -> bytes:
    """Export story as IEEE-style PDF using LaTeX (tectonic) with the IEEEtran template."""
    import base64
    import shutil
    import subprocess
    import tempfile

    import shutil as _shutil_find
    TECTONIC = _shutil_find.which("tectonic") or "/tmp/tectonic"
    IEEE_CLS = pathlib.Path(__file__).resolve().parents[2] / "ieee_template"

    temp_dir = tempfile.mkdtemp(prefix="eda_pdf_")
    try:
        # Copy IEEEtran.cls if the ieee_template dir has one; tectonic auto-downloads it otherwise
        for f in IEEE_CLS.glob("*.cls"):
            shutil.copy(f, temp_dir)

        # Save plot images as PNG files
        fig_counter = 0
        fig_paths: dict[int, str] = {}
        for section in story.get("sections", []):
            for plot in section.get("plots", []):
                fig_counter += 1
                source = plot.get("source", "")
                if not source or not isinstance(source, str):
                    continue
                if source.startswith("data:image"):
                    source = source.split(",", 1)[-1]
                try:
                    img_bytes = base64.b64decode(source)
                    img_path = os.path.join(temp_dir, f"fig{fig_counter}.png")
                    with open(img_path, "wb") as fh:
                        fh.write(img_bytes)
                    fig_paths[fig_counter] = f"fig{fig_counter}.png"
                except Exception:
                    pass

        # Build LaTeX document
        tex = _build_ieee_latex(story, fig_paths)
        tex_path = os.path.join(temp_dir, "report.tex")
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write(tex)

        # Compile with tectonic
        try:
            result = subprocess.run(
                [TECTONIC, "--chatter", "minimal", tex_path],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            pdf_path = os.path.join(temp_dir, "report.pdf")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as fh:
                    return fh.read()
            _LOG.warning("tectonic failed: %s\n%s", result.stdout, result.stderr)
        except (FileNotFoundError, OSError) as exc:
            _LOG.warning("tectonic not available: %s", exc)
        except subprocess.TimeoutExpired:
            _LOG.warning("tectonic timed out after 120s")

        # Fallback to WeasyPrint
        try:
            import weasyprint
            html_content = _story_to_ieee_html_legacy(story)
            return weasyprint.HTML(string=html_content).write_pdf()
        except Exception:
            return _build_minimal_pdf(story)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _tex_esc_raw(text: str) -> str:
    """Escape special LaTeX characters in a plain text fragment (no math)."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _tex_esc(text: str) -> str:
    """Escape special LaTeX characters, preserving $...$ math blocks."""
    if not text:
        return ""
    # Strip markdown bold/italic
    text = text.replace("**", "").replace("*", "")

    # Fix stray LaTeX commands outside $ delimiters (common LLM error)
    # e.g., \text{foo} → foo, \bar{x} → x, \geq → >=, \times → x
    import re as _re
    def _fix_stray_latex(t: str) -> str:
        t = _re.sub(r'\\text\{([^}]*)\}', r'\1', t)
        t = _re.sub(r'\\bar\{([^}]*)\}', r'\1', t)
        t = _re.sub(r'\\hat\{([^}]*)\}', r'\1', t)
        t = t.replace('\\geq', '>=').replace('\\leq', '<=')
        t = t.replace('\\times', 'x').replace('\\approx', '~')
        t = t.replace('\\rho', 'rho').replace('\\chi', 'chi')
        return t

    # Split on $...$ math blocks, escape only the non-math parts
    parts = re.split(r'(\$[^$]+\$)', text)
    result = []
    for part in parts:
        if part.startswith('$') and part.endswith('$') and len(part) > 1:
            # Math block — pass through unchanged
            result.append(part)
        else:
            # Fix stray LaTeX in non-math text, then escape
            result.append(_tex_esc_raw(_fix_stray_latex(part)))
    return "".join(result)


def _build_ieee_latex(story: dict, fig_paths: dict[int, str]) -> str:
    """Generate IEEE conference paper LaTeX from story data."""
    title = story.get("title", "EDA Report")
    summary = story.get("executive_summary", "")
    sections = story.get("sections", [])
    generated_at = story.get("generated_at", "")
    date_str = generated_at[:10] if generated_at else ""

    # Build section bodies
    sec_tex = ""
    fig_counter = 0
    for i, section in enumerate(sections):
        sec_title = section.get("title", f"Section {i+1}")
        content = section.get("content", "")

        # Clean markdown artifacts from content
        content = re.sub(r'^#{1,4}\s+.*$', '', content, flags=re.MULTILINE)  # Strip markdown headers
        content = content.replace("- Key finding:", "-")  # Remove verbose prefix
        content = content.replace("- - Key finding:", "-")
        content = content.strip()

        # Convert content lines
        body_lines = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("- "):
                body_lines.append(f"\\item {_tex_esc(line[2:])}")
            else:
                body_lines.append(_tex_esc(line))

        body_tex = ""
        in_itemize = False
        for bl in body_lines:
            if bl.startswith("\\item "):
                if not in_itemize:
                    body_tex += "\\begin{itemize}\n"
                    in_itemize = True
                body_tex += f"  {bl}\n"
            else:
                if in_itemize:
                    body_tex += "\\end{itemize}\n"
                    in_itemize = False
                body_tex += f"{bl}\n\n"
        if in_itemize:
            body_tex += "\\end{itemize}\n"

        # Figures — capped at 3 per section, use curated captions
        figs_tex = ""
        section_fig_count = 0
        caption_map = section.get("plot_captions", {})
        for plot in section.get("plots", []):
            if section_fig_count >= 3:
                break
            fig_counter += 1
            section_fig_count += 1

            # Use curated caption if available, else fallback
            src_cell = plot.get("source_cell_id", "")
            caption = caption_map.get(src_cell, plot.get("caption", plot.get("title", f"Figure {fig_counter}")))

            if fig_counter in fig_paths:
                # Wide figures for heatmaps, correlation matrices, multi-panel plots
                caption_lower = caption.lower()
                is_wide = any(kw in caption_lower for kw in [
                    "matrix", "heatmap", "correlation", "pairwise", "grid", "panel",
                ])
                fig_env = "figure*" if is_wide else "figure"
                width = "0.85\\textwidth" if is_wide else "0.95\\columnwidth"

                figs_tex += f"""
\\begin{{{fig_env}}}[htbp]
\\centerline{{\\includegraphics[width={width}]{{{fig_paths[fig_counter]}}}}}
\\caption{{{_tex_esc(str(caption))}}}
\\label{{fig{fig_counter}}}
\\end{{{fig_env}}}
"""

        sec_tex += f"\\section{{{_tex_esc(sec_title)}}}\n{body_tex}\n{figs_tex}\n"

    return f"""\\documentclass[conference]{{IEEEtran}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{textcomp}}
\\usepackage{{xcolor}}
\\usepackage[utf8]{{inputenc}}

\\begin{{document}}

\\title{{{_tex_esc(title)}}}

\\author{{\\IEEEauthorblockN{{AgenticEDA}}
\\IEEEauthorblockA{{Automated Exploratory Data Analysis\\\\
Generated: {_tex_esc(date_str)}}}}}

\\maketitle

\\begin{{abstract}}
{_tex_esc(story.get("abstract", summary[:500]))}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
exploratory data analysis, automated EDA, machine learning, data science
\\end{{IEEEkeywords}}

{sec_tex}

\\section*{{Acknowledgment}}
This report was automatically generated by AgenticEDA.

\\end{{document}}
"""


@router.get("/story/{session_id}")
async def get_story(session_id: str, format: str = Query("json")):
    """Return the story in the requested format (json, md, pdf)."""
    story = _load_story(session_id)

    story = _augment_story_plot_metadata(session_id, story)

    if format == "md":
        md = _story_to_ieee_markdown(story)
        return PlainTextResponse(md, media_type="text/markdown")

    if format == "pdf":
        pdf_bytes = _export_story_pdf(story)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{_story_pdf_filename(story)}"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )

    # Default: return JSON — sanitize NaN/Inf that can't be serialized
    import math

    def _sanitize(obj):
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    return _sanitize(story)


@router.post("/story/{session_id}/regenerate")
async def regenerate_story(session_id: str):
    """Re-read the current notebook and regenerate the story via LLM."""
    import asyncio
    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    # Run in thread so the event loop can flush WebSocket plan_update events
    return await asyncio.to_thread(_regenerate_story_sync, session_id, session_dir)


def _regenerate_story_sync(session_id: str, session_dir):
    """Synchronous story regeneration — runs in a thread."""

    # Load existing KG
    from src.agent.knowledge_graph import KnowledgeGraph
    kg = None
    existing_story_path = session_dir / "story.json"
    old_data = {}
    if existing_story_path.exists():
        try:
            old_data = json.loads(existing_story_path.read_text())
            if "knowledge_graph" in old_data:
                kg = KnowledgeGraph.from_dict(old_data["knowledge_graph"])
        except Exception as exc:
            _LOG.warning("Failed to load KG for regeneration: %s", exc)

    # Load agent state
    state_path = session_dir / "agent_state.json"
    agent_state = {}
    if state_path.exists():
        try:
            agent_state = json.loads(state_path.read_text())
        except Exception:
            pass

    # Find dataset name
    uploads_dir = session_dir / "uploads"
    dataset_name = "dataset"
    if uploads_dir.is_dir():
        files = list(uploads_dir.iterdir())
        if files:
            dataset_name = files[0].name

    # Build sections from KG + merge with existing story sections.
    # KG provides the canonical sections, but story.json may have chat investigation
    # sections that were appended but not yet in the KG. Merge both, dedup by title.
    if kg is not None:
        kg_sections = kg.get_story_sections()
        conclusions = kg.get_top_conclusions(5)
    else:
        kg_sections = []
        conclusions = []
        for f in agent_state.get("findings", []):
            conclusions.append(f.get("finding", ""))
        conclusions = conclusions[:5]

    # Emit regen progress via stream
    def _regen_progress(details: list[dict]):
        push_event(session_id, {"type": "plan_update", "steps": [
            {"phase": "Regenerating Report", "status": "current", "details": details},
        ]})

    _regen_progress([{"label": "Loading knowledge graph", "status": "complete"}, {"label": "Building sections", "status": "current"}])

    # Merge: start with KG sections, then append any story.json sections not in KG
    kg_titles = {s.get("title", "") for s in kg_sections}
    existing_sections = old_data.get("sections", [])
    merged_extra = []
    for sec in existing_sections:
        if sec.get("superseded"):
            continue  # Skip superseded sections
        if sec.get("title", "") not in kg_titles:
            merged_extra.append(sec)
    sections = kg_sections + merged_extra

    # Attach plot artifacts from notebook
    nb_path = session_dir / "notebook.ipynb"
    if nb_path.exists():
        try:
            nb = nbformat.read(str(nb_path), as_version=4)
            cell_map = {}
            for cell in nb.cells:
                cell_id = cell.get("id")
                if cell_id:
                    cell_map[str(cell_id)] = cell

            plot_specs_map = plot_specs_by_cell(session_dir)

            for section in sections:
                plot_cell_ids = section.get("plot_cell_ids") or section.get("cell_ids") or []
                plots: list[dict] = []
                for cell_id in plot_cell_ids:
                    cell = cell_map.get(str(cell_id))
                    if not cell:
                        continue
                    plots.extend(
                        plot_artifacts_from_outputs(
                            cell.get("outputs", []) or [],
                            title=section.get("title", ""),
                            caption=section.get("visual_caption", ""),
                            source_cell_id=str(cell_id),
                            plot_specs=plot_specs_map.get(str(cell_id), []),
                        )
                    )

                # Fallback for investigation plots from KG metadata
                if not plots and section.get("type") == "investigation" and kg:
                    kg_data = kg.to_dict()
                    for nid_key, node_data in kg_data.get("nodes", {}).items():
                        if (node_data.get("type") == "conclusion" and
                            node_data.get("phase", "").replace("Investigation: ", "") == section.get("title", "")):
                            plot_images = node_data.get("metadata", {}).get("plot_images", [])
                            for pi in plot_images:
                                plots.append({
                                    "kind": "image",
                                    "mime_type": "image/png",
                                    "source": pi["image_png"],
                                    "title": section.get("title", ""),
                                    "caption": f"Investigation: {section.get('title', '')}",
                                    "source_cell_id": pi.get("cell_id", ""),
                                })
                            break

                if plots:
                    section["plots"] = plots
        except Exception as exc:
            _LOG.warning("Plot extraction during regeneration failed: %s", exc)

    _regen_progress([
        {"label": "Loading knowledge graph", "status": "complete"},
        {"label": "Building sections", "status": "complete"},
        {"label": "Extracting plots", "status": "complete"},
        {"label": "Writing executive summary", "status": "current"},
    ])
    # Generate executive summary via LLM
    narrative = ""
    try:
        from src.config.config import get_chat_model
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = get_chat_model()

        conclusions_text = "\n".join(f"- {c}" for c in conclusions)
        findings_text = "\n".join(
            f"- [{f.get('phase', '')}] {f.get('finding', '')}"
            for f in agent_state.get("findings", [])
        )
        kg_context = ""
        if kg:
            kg_context = kg.get_context_for_hypothesis_generation()

        dataset_info = f"{agent_state.get('row_count', '?')} rows x {agent_state.get('col_count', '?')} cols"
        cols = ', '.join(agent_state.get('numeric_cols', [])[:15])
        time_col = agent_state.get('time_col', 'N/A')

        resp = llm.invoke([
            SystemMessage(content="Write 2-3 paragraphs of flowing prose for an EDA report executive summary. Describe: what the data contains, key patterns, notable anomalies, investigated hypotheses and their conclusions, and recommended next steps. Be specific with numbers. Do NOT use bullet points. Format with markdown. For math, use proper LaTeX delimiters: $x$ for inline (e.g., $r = 0.95$, $p < 0.05$). Never write raw LaTeX without $ delimiters. Never use emojis or special unicode symbols."),
            HumanMessage(content=f"Dataset: {dataset_name}\n{dataset_info}\nColumns: {cols}\nTime column: {time_col}\n\nKnowledge graph:\n{kg_context[:2000]}\n\nTop conclusions:\n{conclusions_text}\n\nAll findings:\n{findings_text[:3000]}"),
        ])
        narrative = resp.content.strip()
    except Exception as exc:
        _LOG.warning("LLM narrative regeneration failed: %s", exc)
        narrative = old_data.get("executive_summary", "")
        if not narrative:
            narrative = "Key findings: " + "; ".join(s.get("title", "") for s in sections[:10])

    _regen_progress([
        {"label": "Loading knowledge graph", "status": "complete"},
        {"label": "Building sections", "status": "complete"},
        {"label": "Extracting plots", "status": "complete"},
        {"label": "Writing executive summary", "status": "complete"},
        {"label": "Curating plots and captions", "status": "current"},
    ])
    from src.reporting.story_builder import build_curated_story

    story_data = build_curated_story(
        sections=sections,
        executive_summary=narrative,
        dataset_name=dataset_name,
        max_plots_per_section=3,
    )
    if kg is not None:
        story_data["knowledge_graph"] = kg.to_dict()

    _regen_progress([
        {"label": "Loading knowledge graph", "status": "complete"},
        {"label": "Building sections", "status": "complete"},
        {"label": "Extracting plots", "status": "complete"},
        {"label": "Writing executive summary", "status": "complete"},
        {"label": "Curating plots and captions", "status": "complete"},
        {"label": "Saving report", "status": "current"},
    ])
    story_path = session_dir / "story.json"
    atomic_write_json(story_path, story_data)

    # Clear the regen plan
    push_event(session_id, {"type": "plan_update", "steps": []})

    _LOG.info("Story regenerated for session %s: %d sections, kg=%s", session_id, len(sections), kg is not None)

    return story_data
