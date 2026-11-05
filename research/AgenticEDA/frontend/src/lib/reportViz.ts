import type {
  StoryPlot,
  StoryPlotDisplay,
  StoryPlotVizChannel,
  StoryPlotVizLegend,
  StoryPlotVizSpec,
  StoryPlotVizSpecInput,
} from "@/lib/types";

export type ReportRendererKind = "d3" | "plotly" | "image";
export type ReportChartKind = StoryPlotVizSpec["chart_family"];
export type ReportAxisKind = StoryPlotVizChannel["role"];
export type ReportPlotKind = "plotly" | "image";

export interface PlotlyTrace {
  type?: string;
  mode?: string;
  name?: string;
  x?: unknown[];
  y?: unknown[];
  z?: unknown[] | unknown[][];
  orientation?: string;
  fill?: string;
  stackgroup?: string;
  marker?: { color?: unknown; size?: unknown; symbol?: unknown };
  line?: { color?: unknown; dash?: string; width?: unknown };
  nbinsx?: number;
  colorscale?: string | Array<[number, string]>;
  zmin?: number;
  zmax?: number;
}

export interface PlotlyShape {
  type?: string;
  x0?: number;
  x1?: number;
  y0?: number;
  y1?: number;
  xref?: string;
  yref?: string;
  line?: { color?: string; width?: number; dash?: string };
}

export interface PlotlyFigure {
  data?: PlotlyTrace[];
  layout?: {
    title?: string | { text?: string };
    xaxis?: { title?: string | { text?: string } };
    yaxis?: { title?: string | { text?: string } };
    showlegend?: boolean;
    barmode?: string;
    grid?: unknown;
    annotations?: unknown[];
    shapes?: PlotlyShape[];
    images?: unknown[];
    xaxis2?: unknown;
    yaxis2?: unknown;
  };
}

export interface PlotlyTraceClassification {
  kind: Exclude<ReportChartKind, "image" | "unknown">;
  name: string;
  x: Array<number | string>;
  y: number[];
  xKind: ReportAxisKind;
  traceType: string;
  samples?: number[];
  z?: number[][];
  colorscale?: string;
}

export interface PlotlyFigureClassification {
  kind: Exclude<ReportChartKind, "image" | "unknown">;
  title: string;
  xLabel?: string;
  yLabel?: string;
  traceCount: number;
  multiSeries: boolean;
  traces: PlotlyTraceClassification[];
  reason: string;
  /** True when the layout has features D3 can't render (subplots, annotations) */
  complexLayout?: boolean;
}

export interface ReportVizContext {
  sectionTitle?: string;
  fallbackTitle?: string;
  fallbackCaption?: string;
  plotCellId?: string;
  index?: number;
}

export interface ReportVizResolution {
  plot: StoryPlot;
  renderer: ReportRendererKind;
  title: string;
  caption: string;
  display: StoryPlotDisplay;
  viz: StoryPlotVizSpec;
}

interface RendererRule {
  renderer: Exclude<ReportRendererKind, "unknown">;
  fallbackRenderer: Exclude<ReportRendererKind, "unknown">;
  matches: (input: {
    normalized: StoryPlot;
    figure: PlotlyFigure | null;
    classification: PlotlyFigureClassification | null;
  }) => boolean;
  reason: (input: {
    normalized: StoryPlot;
    figure: PlotlyFigure | null;
    classification: PlotlyFigureClassification | null;
  }) => string;
}

const DEFAULT_WIDTH = 960;
const DEFAULT_HEIGHT = 540;
const DEFAULT_ASPECT_RATIO = DEFAULT_WIDTH / DEFAULT_HEIGHT;
const MAX_SUPPORTED_TRACES = 16;
const D3_SUPPORTED_KINDS = new Set<Exclude<ReportChartKind, "image" | "unknown">>([
  "scatter",
  "line",
  "bar",
  "histogram",
  "heatmap",
]);
const REPORT_RENDERER_REGISTRY: RendererRule[] = [
  {
    renderer: "d3",
    fallbackRenderer: "plotly",
    matches: ({ normalized, classification }) =>
      (normalized.kind ?? normalized.type) === "plotly" &&
      Boolean(classification && D3_SUPPORTED_KINDS.has(classification.kind)) &&
      !classification?.complexLayout,
    reason: ({ classification }) => classification?.reason || "report-native D3 renderer",
  },
  {
    renderer: "plotly",
    fallbackRenderer: "image",
    matches: ({ normalized, classification }) =>
      (normalized.kind ?? normalized.type) === "plotly" &&
      (!classification || !D3_SUPPORTED_KINDS.has(classification.kind)),
    reason: () => "interactive Plotly fallback",
  },
  {
    renderer: "image",
    fallbackRenderer: "image",
    matches: ({ normalized }) => (normalized.kind ?? normalized.type) !== "plotly",
    reason: () => "static image fallback",
  },
];

function normalizeText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && "text" in value) {
    const text = (value as { text?: unknown }).text;
    return typeof text === "string" ? text : "";
  }
  return "";
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function isCategoryValue(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isHttpUrl(value: string): boolean {
  return value.startsWith("http://") || value.startsWith("https://");
}

function isDataUrl(value: string): boolean {
  return value.startsWith("data:");
}

function isProbablyBase64(value: string): boolean {
  const cleaned = value.replace(/\s+/g, "");
  return cleaned.length > 120 && /^[A-Za-z0-9+/=]+$/.test(cleaned);
}

function parsePlotlyJsonString(value: string): PlotlyFigure | null {
  const trimmed = value.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return null;
  try {
    const parsed = JSON.parse(trimmed) as PlotlyFigure;
    if (!parsed || !Array.isArray(parsed.data)) return null;
    return parsed;
  } catch {
    return null;
  }
}

function resolveImageSource(value: string): string {
  if (!value) return "";
  if (isHttpUrl(value) || isDataUrl(value)) return value;
  const cleaned = value.replace(/\s+/g, "");
  if (isProbablyBase64(cleaned)) {
    return `data:image/png;base64,${cleaned}`;
  }
  return value;
}

export function defaultPlotDisplay(kind: ReportPlotKind = "image"): StoryPlotDisplay {
  return {
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
    aspect_ratio: DEFAULT_ASPECT_RATIO,
    fit: "contain",
    native_width: kind === "plotly" ? DEFAULT_WIDTH : null,
    native_height: kind === "plotly" ? DEFAULT_HEIGHT : null,
    max_width: DEFAULT_WIDTH,
    max_height: DEFAULT_HEIGHT,
  };
}

function normalizeRole(value: string, allowed: Set<string>, fallback: ReportAxisKind = "unknown"): ReportAxisKind {
  const cleaned = value.trim().toLowerCase();
  return allowed.has(cleaned) ? (cleaned as ReportAxisKind) : fallback;
}

function normalizeChartFamily(value: string): ReportChartKind {
  return normalizeRole(
    value,
    new Set(["scatter", "line", "bar", "histogram", "box", "violin", "heatmap", "image", "unknown"]),
    "unknown"
  ) as ReportChartKind;
}

function normalizeSemanticIntent(value: string): StoryPlotVizSpec["semantic_intent"] {
  return normalizeRole(
    value,
    new Set(["trend", "comparison", "distribution", "relationship", "matrix", "residual", "unknown"]),
    "unknown"
  ) as StoryPlotVizSpec["semantic_intent"];
}

function normalizeAxisRole(value: string): ReportAxisKind {
  return normalizeRole(
    value,
    new Set(["time", "category", "ordinal", "numeric", "index", "measure", "count", "density", "unknown"]),
    "unknown"
  );
}

function isSimplePlotlyLayout(layout: PlotlyFigure["layout"]): boolean {
  if (!layout) return true;
  if (layout.grid) return false;
  // Allow shapes (reference lines) and annotations (text labels)
  if (layout.images && layout.images.length > 0) return false;
  // Allow secondary axes — they'll route to Plotly fallback, not D3
  if (layout.barmode && !["group", "overlay", "stack", ""].includes(layout.barmode)) return false;
  return true;
}

function inferTraceKind(traceType: string, mode?: string): Exclude<ReportChartKind, "image" | "unknown"> | null {
  const normalizedType = traceType.toLowerCase();
  if (normalizedType === "scatter" || normalizedType === "scattergl") {
    const normalizedMode = String(mode || "lines").toLowerCase();
    return normalizedMode.includes("line") ? "line" : "scatter";
  }
  if (normalizedType === "bar") return "bar";
  if (normalizedType === "histogram") return "histogram";
  if (normalizedType === "box") return "box";
  if (normalizedType === "violin") return "violin";
  if (normalizedType === "heatmap") return "heatmap";
  return null;
}

function classifyTrace(trace: PlotlyTrace): PlotlyTraceClassification | null {
  const traceType = String(trace.type || "scatter").toLowerCase();
  const kind = inferTraceKind(traceType, trace.mode);
  if (!kind) return null;
  // Allow horizontal bars — the renderer handles orientation
  if (trace.fill && trace.fill.toLowerCase() !== "none" && kind !== "histogram") return null;
  if (trace.stackgroup) return null;

  const xs = asArray(trace.x);
  const ys = asArray(trace.y);

  if (kind === "histogram") {
    const numericValues = xs.map(toNumber).filter((value): value is number => value != null);
    if (numericValues.length === 0) return null;
    return {
      kind,
      name: trace.name?.trim() || "Series",
      x: numericValues,
      y: [],
      xKind: "numeric",
      traceType,
    };
  }

  if (kind === "box" || kind === "violin") {
    const values = (ys.length > 0 ? ys : xs).map(toNumber).filter((value): value is number => value != null);
    if (values.length === 0) return null;
    return {
      kind,
      name: trace.name?.trim() || "Series",
      x: [],
      y: values,
      xKind: "category",
      traceType,
      samples: values,
    };
  }

  if (kind === "heatmap") {
    const zData = (trace.z ?? []) as number[][];
    const xLabels = asArray(trace.x).map(String);
    const yLabels = asArray(trace.y).map(String);
    return {
      kind,
      name: trace.name?.trim() || "Series",
      x: xLabels,
      y: yLabels.map((v) => toNumber(v) ?? 0),
      xKind: "category",
      traceType,
      z: zData,
      colorscale: typeof (trace as PlotlyTrace).colorscale === "string" ? (trace as PlotlyTrace).colorscale as string : undefined,
    };
  }

  if (xs.length === 0 || ys.length === 0 || xs.length !== ys.length) return null;

  if (kind === "bar") {
    const labels = xs.map((value) => (isCategoryValue(value) ? value.trim() : "")).filter(Boolean);
    const values = ys.map(toNumber);
    if (labels.length !== xs.length || values.some((value) => value == null)) return null;
    return {
      kind,
      name: trace.name?.trim() || "Series",
      x: labels,
      y: values.map((value) => value as number),
      xKind: "category",
      traceType,
    };
  }

  const xValues = xs.map(toNumber);
  const yValues = ys.map(toNumber);
  if (xValues.some((value) => value == null) || yValues.some((value) => value == null)) return null;
  return {
    kind,
    name: trace.name?.trim() || "Series",
    x: xValues.map((value) => value as number),
    y: yValues.map((value) => value as number),
    xKind: "numeric",
    traceType,
  };
}

function classifyTraceFamily(traces: PlotlyTraceClassification[]): Exclude<ReportChartKind, "image" | "unknown"> | null {
  if (traces.length === 0) return null;
  const kinds = new Set(traces.map((trace) => trace.kind));
  if (kinds.size === 1) return traces[0]?.kind ?? null;

  // Common composite patterns — pick dominant kind
  // scatter + line = scatter (reference lines over scatter)
  // line + scatter = line (annotations over trend)
  if (kinds.size === 2 && kinds.has("scatter") && kinds.has("line")) {
    const scatterCount = traces.filter((t) => t.kind === "scatter").length;
    const lineCount = traces.filter((t) => t.kind === "line").length;
    return scatterCount >= lineCount ? "scatter" : "line";
  }
  // bar + line = bar (trend line over bar chart)
  if (kinds.size === 2 && kinds.has("bar") && kinds.has("line")) return "bar";
  // histogram + line = histogram (density overlay)
  if (kinds.size === 2 && kinds.has("histogram") && kinds.has("line")) return "histogram";

  return null;
}

function plotDisplay(kind: ReportPlotKind): StoryPlotDisplay {
  return {
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
    aspect_ratio: DEFAULT_ASPECT_RATIO,
    fit: "contain",
    max_width: DEFAULT_WIDTH,
    max_height: DEFAULT_HEIGHT,
    native_width: kind === "plotly" ? DEFAULT_WIDTH : null,
    native_height: kind === "plotly" ? DEFAULT_HEIGHT : null,
  };
}

function coerceDimension(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  return null;
}

function frameFromDimensions(
  kind: ReportPlotKind,
  width: number | null,
  height: number | null
): StoryPlotDisplay {
  const frame = plotDisplay(kind);
  if (width && height) {
    frame.aspect_ratio = Math.max(1.1, Math.min(width / height, 2.1));
    frame.height = Math.max(300, Math.min(height, DEFAULT_HEIGHT));
    frame.width = Math.max(320, Math.min(width, DEFAULT_WIDTH));
    frame.native_width = width;
    frame.native_height = height;
  }
  return frame;
}

function normalizePlotlySource(source: unknown): { source: string; display: StoryPlotDisplay } {
  if (typeof source !== "string") {
    return { source: JSON.stringify(source ?? {}), display: plotDisplay("plotly") };
  }

  const parsed = parsePlotlyJsonString(source);
  if (!parsed) {
    return { source, display: plotDisplay("plotly") };
  }

  const layout = parsed.layout ?? {};
  const width = coerceDimension((layout as Record<string, unknown>).width);
  const height = coerceDimension((layout as Record<string, unknown>).height);
  const display = frameFromDimensions("plotly", width, height);

  const normalizedLayout = {
    ...layout,
    width: undefined,
    height: undefined,
    autosize: true,
    paper_bgcolor: "rgba(248,250,252,0.95)",
    plot_bgcolor: "rgba(248,250,252,0.95)",
    margin:
      typeof layout === "object" &&
      layout &&
      !Array.isArray(layout) &&
      typeof (layout as Record<string, unknown>).margin === "object"
        ? (layout as Record<string, unknown>).margin
        : { t: 28, r: 20, b: 52, l: 60 },
  };

  return {
    source: JSON.stringify({ ...parsed, layout: normalizedLayout }),
    display,
  };
}

function normalizeImageSource(source: string): string {
  if (!source) return "";
  if (isHttpUrl(source) || isDataUrl(source)) return source;
  const cleaned = source.replace(/\s+/g, "");
  if (cleaned.length > 120 && /^[A-Za-z0-9+/=]+$/.test(cleaned)) {
    return `data:image/png;base64,${cleaned}`;
  }
  return source;
}

function inferChartFamily(plot: StoryPlot, classification: PlotlyFigureClassification | null): ReportChartKind {
  if (plot.chart_family && plot.chart_family !== "unknown") return plot.chart_family;
  if (classification) return classification.kind;
  return (plot.kind ?? plot.type) === "image" ? "image" : "unknown";
}

function inferSemanticIntent(plot: StoryPlot, classification: PlotlyFigureClassification | null): StoryPlotVizSpec["semantic_intent"] {
  if (plot.semantic_intent && plot.semantic_intent !== "unknown") return plot.semantic_intent;
  if (!classification) return "unknown";
  if (classification.kind === "line") return "trend";
  if (classification.kind === "scatter") return "relationship";
  if (classification.kind === "bar") return "comparison";
  if (classification.kind === "histogram" || classification.kind === "box" || classification.kind === "violin") return "distribution";
  if (classification.kind === "heatmap") return "matrix";
  return "unknown";
}

function buildChannel(axis: "x" | "y", role: ReportAxisKind, label: string): StoryPlotVizChannel {
  const scale =
    role === "time"
      ? "time"
      : role === "category" || role === "ordinal"
        ? "band"
        : role === "numeric" || role === "index" || role === "measure" || role === "count" || role === "density"
          ? "linear"
          : "unknown";
  return { axis, role, scale, label };
}

function buildLegend(renderer: ReportRendererKind, classification: PlotlyFigureClassification | null): StoryPlotVizLegend {
  const show = renderer !== "image" && Boolean(classification?.traceCount && classification.traceCount > 1);
  return { show, position: show ? "top" : "none" };
}

function getBackendVizSpec(plot: StoryPlot): StoryPlotVizSpecInput | null {
  const vizSpec = plot.viz_spec;
  const plotSpec = plot.plot_spec;
  if (!vizSpec && !plotSpec) return null;
  if (!plotSpec) return vizSpec ?? null;
  if (!vizSpec) return plotSpec ?? null;
  // Merge: plot_spec has raw data (source, chart_family), viz_spec has
  // computed decisions (renderer, fallback_renderer). Viz_spec wins on overlap.
  return { ...plotSpec, ...vizSpec };
}

/**
 * When a plot is an image but has a hidden structured spec (plot_spec) with
 * Plotly JSON in its `source` field, extract that JSON so D3 can render it.
 */
function extractPlotlySourceFromSpec(plot: StoryPlot): string | null {
  const rawSpec = plot.plot_spec;
  if (!rawSpec) return null;

  const specSource = (rawSpec as Record<string, unknown>).source;
  if (!specSource) return null;

  if (typeof specSource === "string") {
    return parsePlotlyJsonString(specSource) ? specSource : null;
  }

  if (typeof specSource === "object" && specSource !== null) {
    const json = JSON.stringify(specSource);
    return parsePlotlyJsonString(json) ? json : null;
  }

  return null;
}

function normalizeBackendVizSpec(plot: StoryPlot, spec: StoryPlotVizSpecInput | null): StoryPlotVizSpec | null {
  if (!spec) return null;
  const renderer =
    spec.renderer === "report_d3"
      ? "d3"
      : spec.renderer === "d3" || spec.renderer === "plotly" || spec.renderer === "image"
        ? spec.renderer
        : "image";
  const fallbackRenderer =
    spec.fallback_renderer === "report_d3"
      ? "d3"
      : spec.fallback_renderer === "d3" ||
          spec.fallback_renderer === "plotly" ||
          spec.fallback_renderer === "image"
        ? spec.fallback_renderer
        : renderer === "d3"
          ? "plotly"
          : "image";
  const base = buildVizSpec(plot, renderer, null, typeof spec.reason === "string" ? spec.reason : "backend plot spec");
  return {
    ...base,
    ...spec,
    renderer: spec.renderer === "report_d3" ? "report_d3" : renderer,
    fallback_renderer: fallbackRenderer,
    display: spec.display && typeof spec.display === "object" ? (spec.display as StoryPlotDisplay) : base.display,
    reason: typeof spec.reason === "string" ? spec.reason : base.reason,
  };
}

function buildVizSpec(
  normalized: StoryPlot,
  renderer: ReportRendererKind,
  classification: PlotlyFigureClassification | null,
  reason: string
): StoryPlotVizSpec {
  const chartFamily = inferChartFamily(normalized, classification);
  const semanticIntent = inferSemanticIntent(normalized, classification);
  const xRole = normalizeAxisRole(
    String(normalized.x_axis_role ? normalized.x_axis_role : classification?.xLabel ? "category" : "unknown")
  );
  const yRole = normalizeAxisRole(
    String(normalized.y_axis_role ? normalized.y_axis_role : classification?.yLabel ? "measure" : "unknown")
  );
  const traceCount = classification?.traceCount ?? 1;
  return {
    renderer,
    fallback_renderer: renderer === "d3" ? "plotly" : renderer === "plotly" ? "image" : "image",
    chart_family: chartFamily,
    semantic_intent: semanticIntent,
    mark:
      chartFamily === "line"
        ? "line"
        : chartFamily === "scatter"
          ? "point"
          : chartFamily === "bar"
            ? "bar"
            : chartFamily === "histogram"
              ? "rect"
              : chartFamily === "box"
                ? "box"
                : chartFamily === "violin"
                  ? "violin"
                  : chartFamily === "heatmap"
                    ? "heatmap"
                    : chartFamily === "image"
                      ? "image"
                      : "unknown",
    orientation: chartFamily === "heatmap" ? "matrix" : chartFamily === "bar" ? "vertical" : "none",
    trace_count: traceCount,
    report_native: renderer === "d3",
    x: buildChannel("x", xRole, classification?.xLabel || "X"),
    y: buildChannel("y", yRole, classification?.yLabel || "Y"),
    legend: buildLegend(renderer, classification),
    interactions: renderer === "image" ? ["source_link"] : ["hover", "focus", "zoom", "source_link"],
    confidence: Math.max(normalized.semantic_confidence ?? 0, classification ? 0.82 : 0.4),
    reason,
    display: normalized.display ?? defaultPlotDisplay(normalized.kind === "plotly" ? "plotly" : "image"),
  };
}

function routeRenderer(
  normalized: StoryPlot,
  figure: PlotlyFigure | null,
  classification: PlotlyFigureClassification | null,
  backendViz: StoryPlotVizSpec | null
): { renderer: Exclude<ReportRendererKind, "unknown">; fallbackRenderer: Exclude<ReportRendererKind, "unknown">; reason: string } {
  // Force static image fallback for box/violin — D3/Plotly garble matplotlib boxplot data
  if (classification?.kind === "box" || classification?.kind === "violin") {
    return {
      renderer: "image" as const,
      fallbackRenderer: "image" as const,
      reason: "static image fallback for box/violin plots",
    };
  }

  if (backendViz?.renderer === "image") {
    return {
      renderer: "image",
      fallbackRenderer: "image",
      reason: backendViz.reason || "static image fallback",
    };
  }

  if (backendViz?.renderer === "plotly") {
    return {
      renderer: "plotly",
      fallbackRenderer: "image",
      reason: backendViz.reason || "interactive Plotly fallback",
    };
  }

  if (backendViz?.renderer === "report_d3") {
    if (classification && D3_SUPPORTED_KINDS.has(classification.kind)) {
      return {
        renderer: "d3",
        fallbackRenderer: "plotly",
        reason: backendViz.reason || classification.reason || "report-native D3 renderer",
      };
    }
    if (figure) {
      return {
        renderer: "plotly",
        fallbackRenderer: "image",
        reason: backendViz.reason || "report-native spec fell back to Plotly",
      };
    }
  }

  for (const rule of REPORT_RENDERER_REGISTRY) {
    if (rule.matches({ normalized, figure, classification })) {
      return {
        renderer: rule.renderer,
        fallbackRenderer: rule.fallbackRenderer,
        reason: rule.reason({ normalized, figure, classification }),
      };
    }
  }

  return {
    renderer: "image",
    fallbackRenderer: "image",
    reason: "static image fallback",
  };
}

export function normalizeReportPlot(
  plot: StoryPlot | string,
  context: ReportVizContext = {}
): StoryPlot {
  if (typeof plot === "string") {
    const raw = plot.trim();
    if (!raw) {
      return {
        kind: "image",
        mime_type: "image/png",
        source: "",
        title: context.fallbackTitle || context.sectionTitle || "Figure",
        caption: context.fallbackCaption || "",
        source_cell_id: context.plotCellId || "",
        display: defaultPlotDisplay("image"),
      };
    }

    const parsed = parsePlotlyJsonString(raw);
    if (parsed) {
      const { source, display } = normalizePlotlySource(raw);
      return {
        kind: "plotly",
        mime_type: "application/vnd.plotly.v1+json",
        source,
        title: context.fallbackTitle || context.sectionTitle || "Figure",
        caption: context.fallbackCaption || "",
        source_cell_id: context.plotCellId || "",
        display,
      };
    }

    return {
      kind: "image",
      mime_type: "image/png",
      source: normalizeImageSource(raw),
      title: context.fallbackTitle || context.sectionTitle || "Figure",
      caption: context.fallbackCaption || "",
      source_cell_id: context.plotCellId || "",
      display: defaultPlotDisplay("image"),
    };
  }

  const normalized: StoryPlot = { ...plot };
  const kind = (normalized.kind ?? normalized.type ?? "image") as ReportPlotKind;
  if (kind === "plotly") {
    const sourceValue = normalized.source ?? normalized.payload ?? "";
    const { source, display } = normalizePlotlySource(
      typeof sourceValue === "string" ? sourceValue : JSON.stringify(sourceValue ?? {})
    );
    normalized.source = source;
    normalized.display = normalized.display ?? display;
    normalized.mime_type = "application/vnd.plotly.v1+json";
  } else {
    const sourceValue = String(normalized.source ?? normalized.payload ?? "");
    normalized.source = normalizeImageSource(sourceValue);
    normalized.mime_type = "image/png";
    normalized.display = normalized.display ?? defaultPlotDisplay("image");
  }

  normalized.title = normalized.title || context.fallbackTitle || context.sectionTitle || "Figure";
  normalized.caption = normalized.caption || context.fallbackCaption || "";
  normalized.source_cell_id = normalized.source_cell_id || context.plotCellId || "";
  normalized.chart_family = normalizeChartFamily(normalized.chart_family || "unknown");
  normalized.semantic_intent = normalizeSemanticIntent(normalized.semantic_intent || "unknown");
  normalized.x_axis_role = normalizeAxisRole(normalized.x_axis_role || "unknown");
  normalized.y_axis_role = normalizeAxisRole(normalized.y_axis_role || "unknown");
  normalized.semantic_confidence =
    typeof normalized.semantic_confidence === "number" && Number.isFinite(normalized.semantic_confidence)
      ? Math.max(0, Math.min(1, normalized.semantic_confidence))
      : kind === "plotly"
        ? 0.35
        : 1;
  normalized.display = normalized.display ?? defaultPlotDisplay(kind);
  return normalized;
}

export function parsePlotlyFigure(source: string): PlotlyFigure | null {
  return parsePlotlyJsonString(source);
}

export function classifyPlotlyFigure(figure: PlotlyFigure): PlotlyFigureClassification | null {
  if (!figure?.data?.length) return null;
  if (!isSimplePlotlyLayout(figure.layout)) return null;
  if (figure.data.length > MAX_SUPPORTED_TRACES) return null;

  const traces = figure.data
    .map((trace) => classifyTrace(trace))
    .filter((trace): trace is PlotlyTraceClassification => Boolean(trace));

  if (traces.length === 0 || traces.length !== figure.data.length) return null;
  let kind = classifyTraceFamily(traces);
  if (!kind) {
    // Allow heatmap + scatter overlay (e.g., annotated heatmaps)
    const kinds = new Set(traces.map((t) => t.kind));
    if (kinds.has("heatmap")) kind = "heatmap";
  }
  if (!kind) return null;

  if ((kind === "line" || kind === "scatter") && !traces.every((trace) => trace.xKind === "numeric" || trace.xKind === "category")) return null;
  if (kind === "bar") {
    const categorySet = traces[0]?.x.join("||") ?? "";
    const sameCategories = traces.every((trace) => trace.xKind === "category" && trace.x.join("||") === categorySet);
    if (!sameCategories) return null;
  }
  if (kind === "histogram") {
    const anyValues = traces.some((trace) => trace.x.length > 0);
    if (!anyValues) return null;
  }

  const multiSeries = traces.length > 1;
  return {
    kind,
    title: normalizeText(figure.layout?.title),
    xLabel: normalizeText(figure.layout?.xaxis?.title) || undefined,
    yLabel: normalizeText(figure.layout?.yaxis?.title) || undefined,
    traceCount: traces.length,
    multiSeries,
    traces,
    reason:
      kind === "line"
        ? multiSeries
          ? "multi-series trend figure"
          : "single-series trend figure"
        : kind === "scatter"
          ? multiSeries
            ? "multi-series relationship figure"
            : "single-series relationship figure"
          : kind === "bar"
            ? "category comparison figure"
            : kind === "histogram"
              ? "distribution figure"
              : kind === "box" || kind === "violin"
                ? "spread distribution figure"
                : kind === "heatmap"
                  ? "matrix figure"
                  : "supported plotly figure",
    complexLayout: Boolean(
      figure.layout?.xaxis2 || figure.layout?.yaxis2 ||
      figure.layout?.grid ||
      (figure.layout?.annotations && figure.layout.annotations.length > 0) ||
      // Detect boxplot artifacts: many short traces (kernel extracted boxplot components as lines)
      (traces.length >= 5 && traces.every((t) => t.x.length <= 5 && t.y.length <= 5))
    ),
  };
}

export function resolveReportVizSpec(
  plot: StoryPlot | string,
  context: ReportVizContext = {}
): ReportVizResolution {
  const normalized = normalizeReportPlot(plot, context);
  const title = normalized.title || context.fallbackTitle || context.sectionTitle || "Figure";
  const caption = normalized.caption || context.fallbackCaption || "";
  const backendViz = normalizeBackendVizSpec(normalized, getBackendVizSpec(normalized));

  if (normalized.kind !== "plotly" && backendViz?.renderer !== "report_d3" && backendViz?.renderer !== "plotly") {
    const viz = buildVizSpec(normalized, "image", null, "static image fallback");
    return {
      plot: normalized,
      renderer: "image",
      title,
      caption,
      display: viz.display,
      viz,
    };
  }

  // When source is an image but backend recommends D3/Plotly, try to extract
  // the Plotly JSON from the hidden structured plot spec instead.
  if (normalized.kind !== "plotly") {
    const specSource = extractPlotlySourceFromSpec(normalized);
    if (specSource) {
      const { source: plotlySource, display: plotlyDisplay } = normalizePlotlySource(specSource);
      normalized.source = plotlySource;
      normalized.kind = "plotly";
      normalized.mime_type = "application/vnd.plotly.v1+json";
      normalized.display = plotlyDisplay;
    }
  }

  const source = normalized.source ?? normalized.payload ?? "";
  const plotlyFigure = parsePlotlyFigure(String(source));
  const plotlyClassification = plotlyFigure ? classifyPlotlyFigure(plotlyFigure) : null;
  if (!plotlyFigure) {
    const viz: StoryPlotVizSpec = backendViz
      ? {
          ...backendViz,
          renderer: "image",
          fallback_renderer: "image",
          reason: "plotly payload could not be parsed",
          display: backendViz.display ?? defaultPlotDisplay("image"),
        }
      : buildVizSpec(normalized, "image", null, "plotly payload could not be parsed");
    return {
      plot: normalized,
      renderer: "image",
      title,
      caption,
      display: viz.display,
      viz,
    };
  }

  const routed = routeRenderer(normalized, plotlyFigure, plotlyClassification, backendViz);

  const viz: StoryPlotVizSpec = backendViz
    ? {
        ...backendViz,
        renderer: routed.renderer === "d3" ? "report_d3" : routed.renderer,
        fallback_renderer: routed.fallbackRenderer === "plotly" ? "plotly" : routed.fallbackRenderer,
        display: backendViz.display ?? defaultPlotDisplay(normalized.kind === "plotly" ? "plotly" : "image"),
        reason: routed.reason,
      }
    : buildVizSpec(normalized, routed.renderer, plotlyClassification, routed.reason);
  if (backendViz) {
    viz.display = backendViz.display ?? viz.display;
    viz.reason = routed.reason;
    viz.fallback_renderer = routed.fallbackRenderer === "plotly" ? "plotly" : routed.fallbackRenderer;
  }
  return {
    plot: normalized,
    renderer: routed.renderer,
    title,
    caption,
    display: viz.display,
    viz,
  };
}

export function canRenderD3StoryPlot(plot: StoryPlot): boolean {
  return resolveReportVizSpec(plot).renderer === "d3";
}

export function getFigureTitle(fig: PlotlyFigure, fallback: string): string {
  return normalizeText(fig.layout?.title) || fallback;
}

export function getAxisTitle(
  axis: { title?: string | { text?: string } } | undefined,
  fallback: string
): string {
  return normalizeText(axis?.title) || fallback;
}
