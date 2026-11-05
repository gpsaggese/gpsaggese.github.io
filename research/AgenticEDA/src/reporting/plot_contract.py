"""Shared plot artifact contract for story/report rendering."""
from __future__ import annotations

import base64
import io
import json
import pathlib
import re
from collections import defaultdict
from typing import Any, Literal

import pydantic

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
DEFAULT_ASPECT_RATIO = DEFAULT_WIDTH / DEFAULT_HEIGHT
REPORT_NATIVE_FAMILIES = {"scatter", "line", "bar", "histogram"}
PLOT_SPEC_MIME = "application/vnd.agenticeda.plot-spec+json"
PLOT_SPEC_FILENAME = "plot_specs.jsonl"
PLOT_SPEC_VERSION = 1

ChartFamily = Literal["scatter", "line", "bar", "histogram", "box", "violin", "heatmap", "image", "unknown"]
SemanticIntent = Literal["trend", "comparison", "distribution", "relationship", "matrix", "residual", "unknown"]
AxisRole = Literal["time", "category", "ordinal", "numeric", "index", "measure", "count", "density", "unknown"]
RendererKind = Literal["report_d3", "plotly", "image", "unknown"]
PlotMark = Literal["point", "line", "bar", "rect", "box", "violin", "heatmap", "image", "unknown"]
PlotOrientation = Literal["vertical", "horizontal", "matrix", "none"]
ScaleKind = Literal["linear", "band", "time", "ordinal", "log", "unknown"]
LegendPosition = Literal["top", "right", "bottom", "none"]
InteractionKind = Literal["hover", "focus", "zoom", "pan", "source_link"]


class PlotDisplay(pydantic.BaseModel):
    """Standardized display hints for story/report visuals."""

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    aspect_ratio: float = DEFAULT_ASPECT_RATIO
    fit: Literal["contain", "cover"] = "contain"
    native_width: int | None = None
    native_height: int | None = None
    max_width: int = DEFAULT_WIDTH
    max_height: int = DEFAULT_HEIGHT


class PlotVizChannel(pydantic.BaseModel):
    axis: Literal["x", "y"]
    role: AxisRole = "unknown"
    scale: ScaleKind = "unknown"
    label: str = ""


class PlotLegend(pydantic.BaseModel):
    show: bool = False
    position: LegendPosition = "none"


class PlotVizSpec(pydantic.BaseModel):
    renderer: RendererKind = "unknown"
    fallback_renderer: RendererKind = "unknown"
    chart_family: ChartFamily = "unknown"
    semantic_intent: SemanticIntent = "unknown"
    mark: PlotMark = "unknown"
    orientation: PlotOrientation = "none"
    trace_count: int = 0
    report_native: bool = False
    x: PlotVizChannel = pydantic.Field(default_factory=lambda: PlotVizChannel(axis="x"))
    y: PlotVizChannel = pydantic.Field(default_factory=lambda: PlotVizChannel(axis="y"))
    legend: PlotLegend = pydantic.Field(default_factory=PlotLegend)
    interactions: list[InteractionKind] = pydantic.Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    display: PlotDisplay = pydantic.Field(default_factory=PlotDisplay)


class PlotArtifact(pydantic.BaseModel):
    """Normalized plot payload for story sections and exports."""

    kind: Literal["image", "plotly"]
    mime_type: str
    source: str
    title: str = ""
    caption: str = ""
    source_path: str | None = None
    source_cell_id: str | None = None
    display: PlotDisplay = pydantic.Field(default_factory=PlotDisplay)
    chart_family: ChartFamily = "unknown"
    semantic_intent: SemanticIntent = "unknown"
    x_axis_role: AxisRole = "unknown"
    y_axis_role: AxisRole = "unknown"
    semantic_confidence: float = 0.0
    plot_spec: dict[str, Any] | None = None
    viz_spec: PlotVizSpec = pydantic.Field(default_factory=PlotVizSpec)


def _is_probable_base64(value: str) -> bool:
    cleaned = re.sub(r"\s+", "", value)
    return len(cleaned) > 120 and bool(re.fullmatch(r"[A-Za-z0-9+/=]+", cleaned))


def _looks_like_plotly_json(value: Any) -> bool:
    if isinstance(value, dict):
        return "data" in value and "layout" in value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            return False
        return '"data"' in stripped or "'data'" in stripped
    return False


def _coerce_plot_spec(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, PlotArtifact):
        value = value.plot_spec
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except Exception:
            return {"raw": stripped}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    if isinstance(value, dict):
        return dict(value)
    return {"raw": value}


def _normalize_role(value: str, allowed: set[str], fallback: str = "unknown") -> str:
    cleaned = value.strip().lower()
    return cleaned if cleaned in allowed else fallback


def _normalize_axis_role(value: str) -> AxisRole:
    return _normalize_role(
        value,
        {"time", "category", "ordinal", "numeric", "index", "measure", "count", "density", "unknown"},
    )  # type: ignore[return-value]


def _normalize_chart_family(value: str) -> ChartFamily:
    return _normalize_role(
        value,
        {"scatter", "line", "bar", "histogram", "box", "violin", "heatmap", "image", "unknown"},
    )  # type: ignore[return-value]


def _normalize_semantic_intent(value: str) -> SemanticIntent:
    return _normalize_role(
        value,
        {"trend", "comparison", "distribution", "relationship", "matrix", "residual", "unknown"},
    )  # type: ignore[return-value]


def _normalize_renderer(value: str) -> RendererKind:
    return _normalize_role(value, {"report_d3", "plotly", "image", "unknown"})  # type: ignore[return-value]


def _normalize_mark(value: str) -> PlotMark:
    return _normalize_role(
        value,
        {"point", "line", "bar", "rect", "box", "violin", "heatmap", "image", "unknown"},
    )  # type: ignore[return-value]


def _normalize_orientation(value: str) -> PlotOrientation:
    return _normalize_role(value, {"vertical", "horizontal", "matrix", "none"})  # type: ignore[return-value]


def _normalize_scale(value: str) -> ScaleKind:
    return _normalize_role(value, {"linear", "band", "time", "ordinal", "log", "unknown"})  # type: ignore[return-value]


def _normalize_legend_position(value: str) -> LegendPosition:
    return _normalize_role(value, {"top", "right", "bottom", "none"})  # type: ignore[return-value]


def _chart_mark(chart_family: ChartFamily) -> PlotMark:
    return {
        "scatter": "point",
        "line": "line",
        "bar": "bar",
        "histogram": "bar",
        "box": "box",
        "violin": "violin",
        "heatmap": "heatmap",
        "image": "image",
    }.get(chart_family, "unknown")


def _chart_orientation(chart_family: ChartFamily) -> PlotOrientation:
    if chart_family == "heatmap":
        return "matrix"
    if chart_family in {"scatter", "line"}:
        return "none"
    if chart_family in {"bar", "histogram", "box", "violin"}:
        return "vertical"
    return "none"


def _axis_scale(role: AxisRole, *, chart_family: ChartFamily, axis: Literal["x", "y"]) -> ScaleKind:
    if role == "time":
        return "time"
    if role in {"category", "ordinal"}:
        return "band"
    if chart_family == "heatmap":
        return "band"
    if chart_family == "histogram" and axis == "x":
        return "linear"
    if role in {"numeric", "index", "measure", "count", "density"}:
        return "linear"
    return "unknown"


def _renderer_for_family(
    chart_family: ChartFamily,
    confidence: float,
    kind: Literal["image", "plotly"],
    *,
    has_plot_spec: bool = False,
) -> RendererKind:
    if kind == "image":
        if has_plot_spec and chart_family in REPORT_NATIVE_FAMILIES and confidence >= 0.65:
            return "report_d3"
        return "image"
    if chart_family in REPORT_NATIVE_FAMILIES and confidence >= 0.65:
        return "report_d3"
    if chart_family in {"box", "violin", "heatmap"}:
        return "plotly"
    return "plotly"


def _fallback_renderer(renderer: RendererKind, kind: Literal["image", "plotly"]) -> RendererKind:
    if kind == "image":
        return "image"
    if renderer == "report_d3":
        return "plotly"
    return "image"


def _interactions_for_renderer(renderer: RendererKind, chart_family: ChartFamily) -> list[InteractionKind]:
    if renderer == "image":
        return ["source_link"]
    if renderer == "report_d3":
        interactions: list[InteractionKind] = ["hover", "focus", "source_link"]
        if chart_family in {"scatter", "line", "histogram"}:
            interactions.insert(2, "zoom")
        return interactions
    return ["hover", "zoom", "source_link"]


def build_report_viz_spec(value: PlotArtifact | dict[str, Any]) -> PlotVizSpec:
    """Build a renderer-ready spec from a normalized plot artifact."""
    plot_spec = _coerce_plot_spec(value.plot_spec if isinstance(value, PlotArtifact) else value.get("plot_spec"))
    if isinstance(value, PlotArtifact):
        source = value.source
        kind = value.kind
        title = value.title
        caption = value.caption
        display = value.display
        chart_family = value.chart_family
        semantic_intent = value.semantic_intent
        x_axis_role = value.x_axis_role
        y_axis_role = value.y_axis_role
        confidence = value.semantic_confidence
    else:
        source = str(value.get("source") or value.get("payload") or "")
        kind = "plotly" if str(value.get("kind") or value.get("type") or "").strip().lower() == "plotly" else "image"
        title = str(value.get("title") or "")
        caption = str(value.get("caption") or "")
        display_value = value.get("display")
        display = PlotDisplay(**display_value) if isinstance(display_value, dict) else PlotDisplay()
        chart_family = _normalize_chart_family(str(value.get("chart_family") or "unknown"))
        semantic_intent = _normalize_semantic_intent(str(value.get("semantic_intent") or "unknown"))
        x_axis_role = _normalize_axis_role(str(value.get("x_axis_role") or "unknown"))
        y_axis_role = _normalize_axis_role(str(value.get("y_axis_role") or "unknown"))
        confidence = float(value.get("semantic_confidence") or 0.0)

    if plot_spec:
        chart_family = _normalize_chart_family(str(plot_spec.get("chart_family") or chart_family))
        semantic_intent = _normalize_semantic_intent(str(plot_spec.get("semantic_intent") or semantic_intent))
        x_axis_role = _normalize_axis_role(str(plot_spec.get("x_axis_role") or plot_spec.get("x", {}).get("role") if isinstance(plot_spec.get("x"), dict) else x_axis_role))
        y_axis_role = _normalize_axis_role(str(plot_spec.get("y_axis_role") or plot_spec.get("y", {}).get("role") if isinstance(plot_spec.get("y"), dict) else y_axis_role))
        confidence = float(plot_spec.get("semantic_confidence") or plot_spec.get("confidence") or confidence)

    trace_count = 0
    x_label = ""
    y_label = ""
    try:
        parsed = json.loads(source) if kind == "plotly" and source else {}
    except Exception:
        parsed = {}

    layout = parsed.get("layout") if isinstance(parsed, dict) else {}
    if not isinstance(layout, dict):
        layout = {}
    traces = parsed.get("data") if isinstance(parsed, dict) else []
    traces = traces if isinstance(traces, list) else []
    trace_count = len(traces) if kind == "plotly" else 0

    def _axis_title(axis: str) -> str:
        axis_value = layout.get(f"{axis}axis") if axis in {"x", "y"} else {}
        if not isinstance(axis_value, dict):
            return ""
        title_value = axis_value.get("title")
        if isinstance(title_value, dict):
            return str(title_value.get("text") or "").strip()
        return str(title_value or "").strip()

    x_label = _axis_title("x")
    y_label = _axis_title("y")

    if plot_spec:
        display_value = plot_spec.get("display")
        if isinstance(display_value, dict):
            try:
                display = PlotDisplay(**display_value)
            except Exception:
                pass
        x_val = plot_spec.get("x")
        y_val = plot_spec.get("y")
        x_label = str(
            plot_spec.get("x_axis_label")
            or (x_val.get("label") if isinstance(x_val, dict) else None)
            or x_label
        ).strip()
        y_label = str(
            plot_spec.get("y_axis_label")
            or (y_val.get("label") if isinstance(y_val, dict) else None)
            or y_label
        ).strip()
        plot_trace_count = plot_spec.get("trace_count")
        if plot_trace_count is None:
            for key in ("series", "facets", "traces"):
                value_count = plot_spec.get(key)
                if isinstance(value_count, list):
                    plot_trace_count = len(value_count)
                    break
        if plot_trace_count is not None:
            try:
                trace_count = max(trace_count, int(plot_trace_count))
            except Exception:
                pass

    renderer = _renderer_for_family(chart_family, confidence, kind, has_plot_spec=bool(plot_spec))
    fallback_renderer = _fallback_renderer(renderer, kind)
    legend_show = bool(trace_count > 1 and kind == "plotly" and chart_family != "heatmap")
    if plot_spec and trace_count > 1:
        legend_show = bool(plot_spec.get("legend", {}).get("show")) if isinstance(plot_spec.get("legend"), dict) else True
    legend_position = "right" if legend_show else "none"
    if chart_family == "heatmap":
        legend_position = "none"
    if kind == "image":
        legend_show = False
        if plot_spec and renderer == "report_d3":
            legend_show = bool(trace_count > 1)

    return PlotVizSpec(
        renderer=renderer,
        fallback_renderer=fallback_renderer,
        chart_family=chart_family,
        semantic_intent=semantic_intent,
        mark=_chart_mark(chart_family),
        orientation=_chart_orientation(chart_family),
        trace_count=trace_count,
        report_native=renderer == "report_d3",
        x=PlotVizChannel(
            axis="x",
            role=x_axis_role,
            scale=_axis_scale(x_axis_role, chart_family=chart_family, axis="x"),
            label=x_label,
        ),
        y=PlotVizChannel(
            axis="y",
            role=y_axis_role,
            scale=_axis_scale(y_axis_role, chart_family=chart_family, axis="y"),
            label=y_label,
        ),
        legend=PlotLegend(show=legend_show, position=_normalize_legend_position(legend_position)),
        interactions=_interactions_for_renderer(renderer, chart_family),
        confidence=round(float(confidence), 3),
        reason=(
            "hidden structured plot spec from notebook cell" if plot_spec and renderer == "report_d3"
            else "report-native chart family" if renderer == "report_d3"
            else "plotly fallback for unsupported or lower-confidence chart family"
            if kind == "plotly"
            else "static image artifact"
        ),
        display=display,
    )


def _text_blob(*parts: str) -> str:
    return " ".join(part for part in parts if part).strip().lower()


def _looks_like_time_series(values: list[Any], axis_title: str = "") -> bool:
    title = axis_title.lower()
    if re.search(r"\b(time|date|datetime|timestamp|day|month|year|hour|minute|second)\b", title):
        return True

    date_like = 0
    total = 0
    for value in values:
        if value is None:
            continue
        total += 1
        if isinstance(value, str):
            try:
                import datetime

                datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
                date_like += 1
            except Exception:
                continue
    return total > 0 and date_like >= max(1, total // 2)


def _infer_semantics_from_plotly_payload(
    source: str,
    *,
    title: str = "",
    caption: str = "",
    kind: ChartFamily = "unknown",
) -> dict[str, Any]:
    try:
        parsed = json.loads(source)
    except Exception:
        parsed = {}

    layout = parsed.get("layout") if isinstance(parsed, dict) else {}
    if not isinstance(layout, dict):
        layout = {}
    traces = parsed.get("data") if isinstance(parsed, dict) else []
    traces = traces if isinstance(traces, list) else []

    trace_types = [str((trace or {}).get("type") or "scatter").lower() for trace in traces if isinstance(trace, dict)]
    modes = [str((trace or {}).get("mode") or "").lower() for trace in traces if isinstance(trace, dict)]
    title_text = _text_blob(
        title,
        caption,
        str((layout.get("title") or {}).get("text") if isinstance(layout.get("title"), dict) else layout.get("title") or ""),
        str((layout.get("xaxis") or {}).get("title") if isinstance((layout.get("xaxis") or {}).get("title"), str) else ((layout.get("xaxis") or {}).get("title") or {}).get("text", "")),
        str((layout.get("yaxis") or {}).get("title") if isinstance((layout.get("yaxis") or {}).get("title"), str) else ((layout.get("yaxis") or {}).get("title") or {}).get("text", "")),
    )

    chart_family: ChartFamily = "unknown"
    semantic_intent: SemanticIntent = "unknown"
    x_axis_role: AxisRole = "unknown"
    y_axis_role: AxisRole = "measure"
    confidence = 0.35

    if traces and all(trace_type == "histogram" for trace_type in trace_types):
        chart_family = "histogram"
        semantic_intent = "distribution"
        x_axis_role = "numeric"
        y_axis_role = "count"
        confidence = 0.95
    elif traces and all(trace_type == "heatmap" for trace_type in trace_types):
        chart_family = "heatmap"
        semantic_intent = "matrix"
        x_axis_role = "category"
        y_axis_role = "category"
        confidence = 0.96
    elif traces and all(trace_type == "box" for trace_type in trace_types):
        chart_family = "box"
        semantic_intent = "distribution"
        x_axis_role = "category"
        y_axis_role = "measure"
        confidence = 0.91
    elif traces and all(trace_type == "violin" for trace_type in trace_types):
        chart_family = "violin"
        semantic_intent = "distribution"
        x_axis_role = "category"
        y_axis_role = "measure"
        confidence = 0.91
    elif traces and all(trace_type == "bar" for trace_type in trace_types):
        chart_family = "bar"
        semantic_intent = "comparison"
        first_trace = traces[0] if isinstance(traces[0], dict) else {}
        x_values = list(first_trace.get("x") or [])
        x_axis_role = "category" if any(isinstance(v, str) for v in x_values) else "ordinal"
        y_axis_role = "measure"
        confidence = 0.92
    elif traces and all(trace_type in {"scatter", "scattergl"} for trace_type in trace_types):
        has_lines = any("line" in mode for mode in modes)
        has_markers = any("marker" in mode for mode in modes)
        chart_family = "line" if has_lines else "scatter"
        if re.search(r"\b(residual|residuals|outlier|error|bias|deviation|fit)\b", title_text):
            semantic_intent = "residual"
        elif chart_family == "line" and _looks_like_time_series(
            list((traces[0] or {}).get("x") or []),
            str((layout.get("xaxis") or {}).get("title") or ""),
        ):
            semantic_intent = "trend"
            x_axis_role = "time"
        elif has_markers and not has_lines:
            semantic_intent = "relationship"
        else:
            semantic_intent = "trend" if chart_family == "line" else "relationship"

        first_trace = traces[0] if isinstance(traces[0], dict) else {}
        x_values = list(first_trace.get("x") or [])
        if x_axis_role == "unknown":
            x_axis_role = "time" if _looks_like_time_series(x_values, str((layout.get("xaxis") or {}).get("title") or "")) else "numeric"
        if re.search(r"\b(count|frequency|occurrence)\b", title_text):
            y_axis_role = "count"
        confidence = 0.88 if chart_family == "line" else 0.84
    elif kind == "image":
        chart_family = "image"
        semantic_intent = "unknown"
        x_axis_role = "unknown"
        y_axis_role = "unknown"
        confidence = 1.0

    if re.search(r"\b(residual|residuals|outlier|error|bias|deviation|fit)\b", title_text):
        semantic_intent = "residual"

    if chart_family == "unknown" and kind != "image":
        chart_family = kind

    return {
        "chart_family": _normalize_chart_family(chart_family),
        "semantic_intent": _normalize_semantic_intent(semantic_intent),
        "x_axis_role": _normalize_axis_role(x_axis_role),
        "y_axis_role": _normalize_axis_role(y_axis_role),
        "semantic_confidence": round(float(confidence), 3),
    }


def infer_plot_semantics(
    kind: Literal["image", "plotly"],
    source: str,
    *,
    title: str = "",
    caption: str = "",
) -> dict[str, Any]:
    """Return grounded semantic hints for a normalized plot artifact."""
    if kind == "image":
        return {
            "chart_family": "image",
            "semantic_intent": "unknown",
            "x_axis_role": "unknown",
            "y_axis_role": "unknown",
            "semantic_confidence": 1.0,
        }
    return _infer_semantics_from_plotly_payload(source, title=title, caption=caption, kind="plotly")


def _merge_semantic_overrides(base: dict[str, Any], value: Any) -> dict[str, Any]:
    if isinstance(value, PlotArtifact):
        for field in ("chart_family", "semantic_intent", "x_axis_role", "y_axis_role", "semantic_confidence"):
            existing = getattr(value, field, None)
            if existing not in (None, "", "unknown"):
                base[field] = existing
        return base

    if isinstance(value, dict):
        for field in ("chart_family", "semantic_intent", "x_axis_role", "y_axis_role", "semantic_confidence"):
            existing = value.get(field)
            if existing not in (None, "", "unknown"):
                base[field] = existing
    return base


def _with_viz_spec(artifact: PlotArtifact) -> PlotArtifact:
    artifact.viz_spec = build_report_viz_spec(artifact)
    return artifact


def plot_spec_path(session_dir: str | pathlib.Path) -> pathlib.Path:
    """Return the sidecar path used for hidden plot-spec emissions."""
    return pathlib.Path(session_dir) / PLOT_SPEC_FILENAME


def append_plot_spec(session_dir: str | pathlib.Path, plot_spec: dict[str, Any]) -> dict[str, Any]:
    """Append a structured plot spec to the session sidecar."""
    path = plot_spec_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = dict(plot_spec)
    record.setdefault("plot_spec_version", PLOT_SPEC_VERSION)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str))
        fh.write("\n")
    return record


def load_plot_specs(session_dir: str | pathlib.Path) -> list[dict[str, Any]]:
    """Load hidden plot specs from a session sidecar."""
    path = plot_spec_path(session_dir)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    except OSError:
        return []
    return records


def plot_specs_by_cell(session_dir: str | pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    """Group hidden plot specs by source cell ID."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_plot_specs(session_dir):
        cell_id = str(record.get("source_cell_id") or record.get("cell_id") or "").strip()
        if not cell_id:
            continue
        grouped[cell_id].append(record)
    return dict(grouped)


def plot_artifacts_from_plot_specs(
    plot_specs: list[dict[str, Any]] | None,
    *,
    title: str = "",
    caption: str = "",
    source_cell_id: str | None = None,
) -> list[dict]:
    """Normalize hidden plot specs into report-ready plot artifacts."""
    artifacts: list[dict] = []
    for spec in plot_specs or []:
        if not isinstance(spec, dict):
            continue
        if not any(key in spec for key in ("source", "payload", "value", "kind", "type", "mime_type", "mimeType", "data")):
            continue
        cell_id = str(spec.get("source_cell_id") or spec.get("cell_id") or source_cell_id or "").strip() or None
        artifact = normalize_plot_artifact(
            spec,
            title=str(spec.get("title") or title or ""),
            caption=str(spec.get("caption") or caption or ""),
            source_cell_id=cell_id,
        )
        if artifact is not None:
            artifacts.append(artifact.model_dump())
    return artifacts


def plot_artifacts_from_session_plot_specs(
    session_dir: str | pathlib.Path,
    *,
    title: str = "",
    caption: str = "",
    source_cell_id: str | None = None,
) -> list[dict]:
    """Load and normalize hidden plot specs for a session."""
    records = load_plot_specs(session_dir)
    if source_cell_id:
        records = [
            record
            for record in records
            if str(record.get("source_cell_id") or record.get("cell_id") or "").strip() == source_cell_id
        ]
    return plot_artifacts_from_plot_specs(
        records,
        title=title,
        caption=caption,
        source_cell_id=source_cell_id,
    )


def _plot_artifact_key(artifact: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(artifact.get("kind", "")),
        str(artifact.get("source", "")),
        str(artifact.get("source_cell_id", "")),
    )


def merge_plot_artifact_lists(*lists: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Merge plot artifacts while preserving order and dropping duplicates."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for items in lists:
        for artifact in items or []:
            key = _plot_artifact_key(artifact)
            if key in seen:
                continue
            seen.add(key)
            merged.append(artifact)
    return merged


def _image_dimensions_from_bytes(blob: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(blob)) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def _image_dimensions_from_source(source: str, source_path: str | None = None) -> tuple[int, int] | None:
    if source_path:
        path = pathlib.Path(source_path)
        if path.exists():
            try:
                from PIL import Image  # type: ignore

                with Image.open(path) as image:
                    return int(image.width), int(image.height)
            except Exception:
                pass

    if source.startswith("data:"):
        try:
            raw = source.split(",", 1)[1]
            blob = base64.b64decode(raw)
            return _image_dimensions_from_bytes(blob)
        except Exception:
            return None

    if _is_probable_base64(source):
        try:
            blob = base64.b64decode(re.sub(r"\s+", "", source))
            return _image_dimensions_from_bytes(blob)
        except Exception:
            return None

    return None


def _build_display(kind: Literal["image", "plotly"], *, source: str, source_path: str | None = None) -> PlotDisplay:
    if kind == "plotly":
        return PlotDisplay()

    dims = _image_dimensions_from_source(source, source_path)
    if not dims:
        return PlotDisplay()

    width, height = dims
    aspect_ratio = float(width / height) if height else DEFAULT_ASPECT_RATIO
    return PlotDisplay(
        width=min(width, DEFAULT_WIDTH),
        height=min(height, DEFAULT_HEIGHT),
        aspect_ratio=aspect_ratio,
        native_width=width,
        native_height=height,
        max_width=DEFAULT_WIDTH,
        max_height=DEFAULT_HEIGHT,
    )


def normalize_plot_artifact(
    value: Any,
    *,
    title: str = "",
    caption: str = "",
    source_cell_id: str | None = None,
    source_path: str | None = None,
) -> PlotArtifact | None:
    """Convert a raw notebook/stored plot payload into a normalized artifact."""
    if value is None:
        return None

    if isinstance(value, PlotArtifact):
        if title and not value.title:
            value.title = title
        if caption and not value.caption:
            value.caption = caption
        if source_cell_id and not value.source_cell_id:
            value.source_cell_id = source_cell_id
        if source_path and not value.source_path:
            value.source_path = source_path
        return _with_viz_spec(value)

    if isinstance(value, dict):
        kind = str(value.get("kind") or value.get("type") or "").strip().lower()
        mime_type = str(value.get("mime_type") or value.get("mimeType") or "").strip()
        source = value.get("source") or value.get("payload") or value.get("value") or ""
        plot_spec = _coerce_plot_spec(value.get("plot_spec") or value.get("plotSpec"))
        if not source and "data" in value and _looks_like_plotly_json(value["data"]):
            source = json.dumps(value["data"], default=str)
        if not kind:
            kind = "plotly" if mime_type == "application/vnd.plotly.v1+json" or _looks_like_plotly_json(source) else "image"
        if not mime_type:
            mime_type = (
                "application/vnd.plotly.v1+json"
                if kind == "plotly"
                else "image/png"
            )
        source_str = source if isinstance(source, str) else json.dumps(source, default=str)
        display = value.get("display")
        if plot_spec and isinstance(plot_spec.get("display"), dict):
            plot_display = PlotDisplay(**plot_spec["display"])
        elif isinstance(display, dict):
            plot_display = PlotDisplay(**display)
        else:
            plot_display = _build_display(kind, source=source_str, source_path=source_path or value.get("source_path"))
        semantic = infer_plot_semantics(
            "plotly" if kind == "plotly" else "image",
            source_str,
            title=str(value.get("title") or title or ""),
            caption=str(value.get("caption") or caption or ""),
        )
        semantic = _merge_semantic_overrides(semantic, value)
        semantic = _merge_semantic_overrides(semantic, plot_spec or {})
        return _with_viz_spec(PlotArtifact(
            kind="plotly" if kind == "plotly" else "image",
            mime_type=mime_type,
            source=source_str,
            title=str(value.get("title") or title or ""),
            caption=str(value.get("caption") or caption or ""),
            source_path=str(value.get("source_path") or source_path) if (value.get("source_path") or source_path) else None,
            source_cell_id=str(value.get("source_cell_id") or source_cell_id) if (value.get("source_cell_id") or source_cell_id) else None,
            display=plot_display,
            chart_family=_normalize_chart_family(str(semantic.get("chart_family", "unknown"))),
            semantic_intent=_normalize_semantic_intent(str(semantic.get("semantic_intent", "unknown"))),
            x_axis_role=_normalize_axis_role(str(semantic.get("x_axis_role", "unknown"))),
            y_axis_role=_normalize_axis_role(str(semantic.get("y_axis_role", "unknown"))),
            semantic_confidence=float(semantic.get("semantic_confidence", 0.0) or 0.0),
            plot_spec=plot_spec,
        ))

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("data:image/"):
            semantic = infer_plot_semantics("image", stripped, title=title, caption=caption)
            return _with_viz_spec(PlotArtifact(
                kind="image",
                mime_type=stripped.split(";", 1)[0].replace("data:", ""),
                source=stripped,
                title=title,
                caption=caption,
                source_path=source_path,
                source_cell_id=source_cell_id,
                display=_build_display("image", source=stripped, source_path=source_path),
                chart_family=_normalize_chart_family(str(semantic["chart_family"])),
                semantic_intent=_normalize_semantic_intent(str(semantic["semantic_intent"])),
                x_axis_role=_normalize_axis_role(str(semantic["x_axis_role"])),
                y_axis_role=_normalize_axis_role(str(semantic["y_axis_role"])),
                semantic_confidence=float(semantic["semantic_confidence"]),
            ))

        if _looks_like_plotly_json(stripped):
            semantic = infer_plot_semantics("plotly", stripped, title=title, caption=caption)
            return _with_viz_spec(PlotArtifact(
                kind="plotly",
                mime_type="application/vnd.plotly.v1+json",
                source=stripped,
                title=title,
                caption=caption,
                source_path=source_path,
                source_cell_id=source_cell_id,
                display=_build_display("plotly", source=stripped, source_path=source_path),
                chart_family=_normalize_chart_family(str(semantic["chart_family"])),
                semantic_intent=_normalize_semantic_intent(str(semantic["semantic_intent"])),
                x_axis_role=_normalize_axis_role(str(semantic["x_axis_role"])),
                y_axis_role=_normalize_axis_role(str(semantic["y_axis_role"])),
                semantic_confidence=float(semantic["semantic_confidence"]),
            ))

        cleaned = re.sub(r"\s+", "", stripped)
        if _is_probable_base64(cleaned):
            semantic = infer_plot_semantics("image", cleaned, title=title, caption=caption)
            return _with_viz_spec(PlotArtifact(
                kind="image",
                mime_type="image/png",
                source=cleaned,
                title=title,
                caption=caption,
                source_path=source_path,
                source_cell_id=source_cell_id,
                display=_build_display("image", source=cleaned, source_path=source_path),
                chart_family=_normalize_chart_family(str(semantic["chart_family"])),
                semantic_intent=_normalize_semantic_intent(str(semantic["semantic_intent"])),
                x_axis_role=_normalize_axis_role(str(semantic["x_axis_role"])),
                y_axis_role=_normalize_axis_role(str(semantic["y_axis_role"])),
                semantic_confidence=float(semantic["semantic_confidence"]),
            ))

        path = pathlib.Path(stripped)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            semantic = infer_plot_semantics("image", stripped, title=title, caption=caption)
            return _with_viz_spec(PlotArtifact(
                kind="image",
                mime_type="image/png",
                source=stripped,
                title=title,
                caption=caption,
                source_path=stripped,
                source_cell_id=source_cell_id,
                display=_build_display("image", source=stripped, source_path=stripped),
                chart_family=_normalize_chart_family(str(semantic["chart_family"])),
                semantic_intent=_normalize_semantic_intent(str(semantic["semantic_intent"])),
                x_axis_role=_normalize_axis_role(str(semantic["x_axis_role"])),
                y_axis_role=_normalize_axis_role(str(semantic["y_axis_role"])),
                semantic_confidence=float(semantic["semantic_confidence"]),
            ))

        if path.suffix.lower() == ".json":
            semantic = infer_plot_semantics("plotly", stripped, title=title, caption=caption)
            return _with_viz_spec(PlotArtifact(
                kind="plotly",
                mime_type="application/vnd.plotly.v1+json",
                source=stripped,
                title=title,
                caption=caption,
                source_path=stripped,
                source_cell_id=source_cell_id,
                display=_build_display("plotly", source=stripped, source_path=stripped),
                chart_family=_normalize_chart_family(str(semantic["chart_family"])),
                semantic_intent=_normalize_semantic_intent(str(semantic["semantic_intent"])),
                x_axis_role=_normalize_axis_role(str(semantic["x_axis_role"])),
                y_axis_role=_normalize_axis_role(str(semantic["y_axis_role"])),
                semantic_confidence=float(semantic["semantic_confidence"]),
            ))

        semantic = infer_plot_semantics("image", stripped, title=title, caption=caption)
        return _with_viz_spec(PlotArtifact(
            kind="image",
            mime_type="image/png",
            source=stripped,
            title=title,
            caption=caption,
            source_path=source_path,
            source_cell_id=source_cell_id,
            display=_build_display("image", source=stripped, source_path=source_path),
            chart_family=_normalize_chart_family(str(semantic["chart_family"])),
            semantic_intent=_normalize_semantic_intent(str(semantic["semantic_intent"])),
            x_axis_role=_normalize_axis_role(str(semantic["x_axis_role"])),
            y_axis_role=_normalize_axis_role(str(semantic["y_axis_role"])),
            semantic_confidence=float(semantic["semantic_confidence"]),
        ))

    if _looks_like_plotly_json(value):
        source = json.dumps(value, default=str)
        semantic = infer_plot_semantics("plotly", source, title=title, caption=caption)
        return _with_viz_spec(PlotArtifact(
            kind="plotly",
            mime_type="application/vnd.plotly.v1+json",
            source=source,
            title=title,
            caption=caption,
            source_path=source_path,
            source_cell_id=source_cell_id,
            display=_build_display("plotly", source=source, source_path=source_path),
            chart_family=_normalize_chart_family(str(semantic["chart_family"])),
            semantic_intent=_normalize_semantic_intent(str(semantic["semantic_intent"])),
            x_axis_role=_normalize_axis_role(str(semantic["x_axis_role"])),
            y_axis_role=_normalize_axis_role(str(semantic["y_axis_role"])),
            semantic_confidence=float(semantic["semantic_confidence"]),
        ))

    return None


def normalize_plot_artifacts(
    values: list[Any] | tuple[Any, ...] | None,
    *,
    title: str = "",
    caption: str = "",
    source_cell_id: str | None = None,
    source_path: str | None = None,
) -> list[dict]:
    """Normalize a sequence of raw plot values into serializable dicts."""
    artifacts: list[dict] = []
    for value in values or []:
        artifact = normalize_plot_artifact(
            value,
            title=title,
            caption=caption,
            source_cell_id=source_cell_id,
            source_path=source_path,
        )
        if artifact is not None:
            artifacts.append(artifact.model_dump())
    return artifacts


def _extract_plot_spec_from_output(output: Any) -> dict[str, Any] | None:
    if not isinstance(output, dict):
        return None

    for container in (output.get("data"), output.get("metadata")):
        if not isinstance(container, dict):
            continue
        raw = container.get(PLOT_SPEC_MIME)
        if raw is None:
            raw = container.get("plot_spec") or container.get("plotSpec")
        spec = _coerce_plot_spec(raw)
        if spec:
            return spec
    return None


def _compose_plot_specs(plot_specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    specs = [spec for spec in plot_specs if isinstance(spec, dict)]
    if not specs:
        return None
    if len(specs) == 1:
        return specs[0]

    first = dict(specs[0])
    first.setdefault("plot_spec_kind", "composite")
    first["subplots"] = specs
    first["trace_count"] = max(
        int(first.get("trace_count") or 0),
        len(specs),
    )
    for field in ("chart_family", "semantic_intent", "x_axis_role", "y_axis_role", "semantic_confidence"):
        if first.get(field) in (None, "", "unknown"):
            for spec in specs:
                candidate = spec.get(field)
                if candidate not in (None, "", "unknown"):
                    first[field] = candidate
                    break
    if "display" not in first:
        for spec in specs:
            display = spec.get("display")
            if isinstance(display, dict):
                first["display"] = display
                break
    return first


def plot_artifacts_from_outputs(
    outputs: list[Any] | None,
    *,
    title: str = "",
    caption: str = "",
    source_cell_id: str | None = None,
    plot_specs: list[Any] | None = None,
) -> list[dict]:
    """Extract normalized plot artifacts from notebook cell outputs."""
    artifacts: list[dict] = []
    collected_specs: list[dict[str, Any]] = [
        spec for spec in (_coerce_plot_spec(value) for value in (plot_specs or [])) if spec
    ]
    pending_specs: list[dict[str, Any]] = list(collected_specs)

    def _attach_plot_spec(artifact: dict[str, Any], plot_spec: dict[str, Any] | None) -> dict[str, Any]:
        if not plot_spec:
            return artifact
        artifact = dict(artifact)
        artifact["plot_spec"] = plot_spec
        normalized = normalize_plot_artifact(
            artifact,
            title=title,
            caption=caption,
            source_cell_id=source_cell_id,
        )
        return normalized.model_dump() if normalized is not None else artifact

    for output in outputs or []:
        hidden_spec = _extract_plot_spec_from_output(output)
        if hidden_spec:
            collected_specs.append(hidden_spec)
        data = output.get("data") if isinstance(output, dict) else None
        if not isinstance(data, dict):
            if hidden_spec:
                pending_specs.append(hidden_spec)
            continue

        plotly_value = data.get("application/vnd.plotly.v1+json")
        if plotly_value is not None:
            plot_spec = hidden_spec or (pending_specs.pop(0) if pending_specs else None)
            artifact = normalize_plot_artifact(
                {
                    "kind": "plotly",
                    "mime_type": "application/vnd.plotly.v1+json",
                    "source": plotly_value,
                    "title": title,
                    "caption": caption,
                    "source_cell_id": source_cell_id,
                    "plot_spec": plot_spec,
                },
                title=title,
                caption=caption,
                source_cell_id=source_cell_id,
            )
            if artifact is not None:
                artifacts.append(artifact.model_dump())
            continue

        img_value = data.get("image/png")
        if img_value is not None:
            plot_spec = hidden_spec or (pending_specs.pop(0) if pending_specs else None)
            artifact = normalize_plot_artifact(
                {
                    "kind": "image",
                    "mime_type": "image/png",
                    "source": img_value,
                    "title": title,
                    "caption": caption,
                    "source_cell_id": source_cell_id,
                    "plot_spec": plot_spec,
                },
                title=title,
                caption=caption,
                source_cell_id=source_cell_id,
            )
            if artifact is not None:
                artifacts.append(artifact.model_dump())
            continue

        if hidden_spec:
            pending_specs.append(hidden_spec)

    if pending_specs and artifacts:
        # Attach orphaned specs to the most recent artifact from the same cell.
        for plot_spec in pending_specs:
            artifacts[-1] = _attach_plot_spec(artifacts[-1], plot_spec)

    if len(artifacts) == 1 and len(collected_specs) > 1:
        composite_spec = _compose_plot_specs(collected_specs)
        artifacts[0] = _attach_plot_spec(artifacts[0], composite_spec)
    return artifacts


def plot_artifacts_from_paths(
    paths: list[str] | None,
    *,
    title: str = "",
    caption: str = "",
) -> list[dict]:
    """Normalize plot paths into artifact dictionaries."""
    return normalize_plot_artifacts(paths or [], title=title, caption=caption)
