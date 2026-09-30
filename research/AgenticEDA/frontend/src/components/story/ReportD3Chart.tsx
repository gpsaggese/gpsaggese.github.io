"use client";

import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import * as d3 from "d3";
import type { StoryPlotDisplay, StoryPlotVizSpec } from "@/lib/types";
import {
  classifyPlotlyFigure,
  getAxisTitle,
  getFigureTitle,
  parsePlotlyFigure,
  type PlotlyFigure,
  type PlotlyFigureClassification,
  type PlotlyShape,
  type ReportChartKind,
} from "@/lib/reportViz";

interface Props {
  plotlyJson: string;
  display?: StoryPlotDisplay;
  spec?: StoryPlotVizSpec;
  title?: string;
  className?: string;
}

interface Point {
  x: number;
  y: number;
  label: string;
  traceName: string;
  key: string;
}

interface LegendEntry {
  name: string;
  color: string;
  kind: ReportChartKind;
}

type ActivePoint = {
  key: string;
  title: string;
  series: string;
  x: string;
  y: string;
  kind: ReportChartKind;
  color: string;
} | null;

// Neural Slate palette — primary blue action thread + cold-to-electric spectrum
const PALETTE = [
  "#004ac6", "#f28e2b", "#1d6f8a", "#e15759", "#59a14f",
  "#8b5cf6", "#0891b2", "#b45309", "#64748b", "#6366f1",
];
const AXIS_TEXT = "#475569";       // slate-600
const GRID_TEXT = "rgba(71, 85, 105, 0.10)";  // subtle grid
const PANEL_TEXT = "#0f172a";      // slate-900

function smartFormat(domain: [number, number]): (n: number) => string {
  const range = Math.abs(domain[1] - domain[0]);
  if (!Number.isFinite(range) || range === 0) return d3.format(",.3~f");
  if (range > 1e6) return d3.format(".2s");
  if (range > 1000) return d3.format(",.0f");
  if (range > 1) return d3.format(",.2~f");
  if (range > 0.01) return d3.format(".3f");
  return d3.format(".2e");
}

function fmtValue(value: number): string {
  return Number.isInteger(value) ? d3.format(",")(value) : d3.format(",.3~f")(value);
}

function clampText(value: string, limit = 18): string {
  return value.length <= limit ? value : `${value.slice(0, Math.max(1, limit - 1))}…`;
}

function safeExtent(values: number[], padRatio = 0.08): [number, number] {
  const [min, max] = d3.extent(values) as [number | undefined, number | undefined];
  if (typeof min !== "number" || typeof max !== "number") return [0, 1];
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min - 1, max + 1];
  const span = max - min;
  const pad = span * padRatio;
  return [min - pad, max + pad];
}

function readPoints(trace: PlotlyFigureClassification["traces"][number]): Point[] {
  if (trace.kind === "bar") return [];
  const xValues = trace.x.filter((value): value is number => typeof value === "number");
  return xValues.map((x, index) => ({
    x,
    y: trace.y[index] ?? 0,
    label: `${trace.name} • ${index + 1}`,
    traceName: trace.name,
    key: `${trace.name}:${index}`,
  }));
}

function buildBarSeries(classification: PlotlyFigureClassification): {
  categories: string[];
  series: Array<{
    name: string;
    values: number[];
  }>;
} {
  const categories = classification.traces[0]?.x.map(String) ?? [];
  const series = classification.traces.map((trace) => ({
    name: trace.name,
    values: trace.y,
  }));
  return { categories, series };
}

function chartKindLabel(kind: ReportChartKind): string {
  switch (kind) {
    case "line":
      return "Trend";
    case "scatter":
      return "Comparison";
    case "bar":
      return "Category";
    case "histogram":
      return "Distribution";
    case "heatmap":
      return "Matrix";
    case "box":
      return "Spread";
    case "violin":
      return "Distribution";
    default:
      return "Chart";
  }
}

function computeBoxStats(values: number[]) {
  const sorted = values.slice().sort(d3.ascending);
  const q1 = d3.quantile(sorted, 0.25)!;
  const median = d3.quantile(sorted, 0.5)!;
  const q3 = d3.quantile(sorted, 0.75)!;
  const iqr = q3 - q1;
  const whiskerLo = Math.max(d3.min(sorted)!, q1 - 1.5 * iqr);
  const whiskerHi = Math.min(d3.max(sorted)!, q3 + 1.5 * iqr);
  const outliers = sorted.filter((v) => v < q1 - 1.5 * iqr || v > q3 + 1.5 * iqr);
  return { whiskerLo, q1, median, q3, whiskerHi, outliers };
}

function renderLegendEntry(entry: LegendEntry, activeSeries?: string) {
  const active = !activeSeries || activeSeries === entry.name;

  let swatch: ReactNode;
  if (entry.kind === "heatmap") {
    // For heatmap, show a small gradient swatch
    swatch = (
      <span
        className="inline-block h-2.5 w-4 rounded-sm"
        style={{
          background: "linear-gradient(to right, #2166ac, #f7f7f7, #d6604d)",
          opacity: active ? 1 : 0.38,
        }}
      />
    );
  } else if (entry.kind === "line") {
    swatch = (
      <span
        className="inline-block h-0.5 w-4 rounded-full"
        style={{ backgroundColor: entry.color, opacity: active ? 1 : 0.38 }}
      />
    );
  } else if (entry.kind === "bar" || entry.kind === "box") {
    swatch = (
      <span
        className="inline-block h-2.5 w-2.5 rounded-[0.28rem]"
        style={{ backgroundColor: entry.color, opacity: active ? 1 : 0.38 }}
      />
    );
  } else {
    // scatter, histogram, violin, and everything else → circle
    swatch = (
      <span
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: entry.color, opacity: active ? 1 : 0.38 }}
      />
    );
  }

  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-outline-variant/30 bg-white/88 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600 shadow-sm backdrop-blur"
      style={{ opacity: active ? 1 : 0.62 }}
    >
      {swatch}
      <span className="max-w-[14rem] truncate">{entry.name}</span>
    </span>
  );
}

export default function ReportD3Chart({
  plotlyJson,
  display,
  spec,
  title,
  className = "",
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const uid = useId().replace(/:/g, "-");
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [activePoint, setActivePoint] = useState<ActivePoint>(null);

  const figure = useMemo(() => parsePlotlyFigure(plotlyJson), [plotlyJson]);
  const classification = useMemo(
    () => (figure ? classifyPlotlyFigure(figure) : null),
    [figure]
  );

  // Raw figure traces for chart types that need data not captured by classification
  // Extract layout shapes (reference lines) for rendering
  const layoutShapes = useMemo((): PlotlyShape[] => {
    return (figure?.layout?.shapes ?? []).filter(
      (s): s is PlotlyShape => s != null && typeof s === "object"
    );
  }, [figure]);

  // Extract per-trace marker color from raw Plotly data
  const traceColors = useMemo((): (string | null)[] => {
    if (!figure?.data) return [];
    return figure.data.map((trace) => {
      const mc = trace.marker?.color;
      if (typeof mc === "string") return mc;
      const lc = trace.line?.color;
      if (typeof lc === "string") return lc;
      return null;
    });
  }, [figure]);

  const rawTraces = useMemo(() => {
    if (!figure?.data) return [];
    return figure.data;
  }, [figure]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const update = () => {
      const rect = el.getBoundingClientRect();
      setSize({
        width: Math.max(320, rect.width),
        height: Math.max(260, rect.height),
      });
    };

    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  if (!figure || !classification) return null;

  const renderModel = classification;
  const resolvedTitle = getFigureTitle(figure as PlotlyFigure, title || "Figure");
  const resolvedXLabel = spec?.x.label || getAxisTitle(figure.layout?.xaxis, "X");
  const resolvedYLabel = spec?.y.label || getAxisTitle(figure.layout?.yaxis, "Y");
  const width = size.width || display?.width || 960;
  const height = size.height || display?.height || 540;

  // Responsive margins scale with chart size; heatmap gets extra right space for legend
  const isHeatmap = renderModel.kind === "heatmap";
  // Estimate left margin from y-axis data range to avoid label truncation
  const allYValues = renderModel.traces.flatMap((t) =>
    (t.samples ?? t.y).filter((v): v is number => typeof v === "number")
  );
  const yMax = Math.max(Math.abs(d3.min(allYValues) ?? 0), Math.abs(d3.max(allYValues) ?? 0));
  const yLabelWidth = yMax > 1e6 ? 72 : yMax > 1000 ? 64 : yMax > 10 ? 56 : 48;
  const margin = {
    top: Math.max(36, Math.round(height * 0.07)),
    right: isHeatmap ? 80 : Math.max(24, Math.round(width * 0.03)),
    bottom: Math.max(64, Math.round(height * 0.12)),
    left: Math.max(yLabelWidth, Math.round(width * 0.07)),
  };
  const innerWidth = Math.max(120, width - margin.left - margin.right);
  const innerHeight = Math.max(120, height - margin.top - margin.bottom);
  const xTickCount = Math.max(3, Math.floor(innerWidth / 90));
  const yTickCount = Math.max(3, Math.floor(innerHeight / 60));
  const chartId = `report-chart-${uid}`;
  const bgGradientId = `${chartId}-bg`;
  const areaGradientId = `${chartId}-area`;
  const clipId = `${chartId}-clip`;
  const palette = PALETTE;
  const legendEntries: LegendEntry[] = renderModel.traces.map((trace, index) => ({
    name: trace.name,
    color: palette[index % palette.length],
    kind: renderModel.kind,
  }));
  const focusClass =
    "outline-none transition-[filter,transform,opacity,stroke,fill] focus-visible:drop-shadow-[0_0_0_3px_rgba(29,78,216,0.24)] focus-visible:brightness-105";
  const activeSeries = activePoint?.series;

  const showPoint = (point: NonNullable<ActivePoint>) => setActivePoint(point);

  let chartContent: ReactNode = null;

  if (renderModel.kind === "bar") {
    const { categories, series } = buildBarSeries(renderModel);
    const x = d3.scaleBand<string>().domain(categories).range([0, innerWidth]).padding(0.24);
    const group = d3
      .scaleBand<string>()
      .domain(series.map((entry) => entry.name))
      .range([0, x.bandwidth()])
      .padding(0.14);
    const values = series.flatMap((entry) => entry.values);
    const y = d3.scaleLinear().domain([0, d3.max(values) ?? 1]).nice().range([innerHeight, 0]);
    const yTicks = y.ticks(yTickCount);

    chartContent = (
      <>
        <g clipPath={`url(#${clipId})`}>
          {yTicks.map((tick) => (
            <g key={tick} transform={`translate(0,${y(tick)})`}>
              <line x1={0} x2={innerWidth} stroke={GRID_TEXT} strokeWidth={1} />
            </g>
          ))}
          {categories.map((category, categoryIndex) => {
            const bandX = x(category);
            if (bandX == null) return null;
            const centerX = bandX + x.bandwidth() / 2;
            return (
              <g key={category}>
                <line
                  x1={centerX}
                  x2={centerX}
                  y1={0}
                  y2={innerHeight}
                  stroke={GRID_TEXT}
                  strokeWidth={1}
                  opacity={0.32}
                />
                {series.map((entry, seriesIndex) => {
                  const value = entry.values[categoryIndex] ?? 0;
                  const barX = group(entry.name);
                  if (barX == null) return null;
                  const color = palette[seriesIndex % palette.length];
                  const barHeight = innerHeight - y(value);
                  const key = `bar:${entry.name}:${category}`;
                  const point: NonNullable<ActivePoint> = {
                    key,
                    title: resolvedTitle,
                    series: entry.name,
                    x: category,
                    y: fmtValue(value),
                    kind: "bar",
                    color,
                  };
                  const isActive = activePoint?.key ? activePoint.key === key : false;
                  return (
                    <rect
                      key={`${category}-${entry.name}`}
                      x={bandX + barX}
                      y={y(value)}
                      width={Math.max(1, group.bandwidth())}
                      height={Math.max(1, barHeight)}
                      rx={3}
                      fill={color}
                      fillOpacity={isActive ? 0.98 : 0.82}
                      stroke="rgba(255,255,255,0.92)"
                      strokeWidth={isActive ? 1.4 : 0.75}
                      tabIndex={0}
                      role="button"
                      aria-label={`${entry.name} bar ${category}: ${fmtValue(value)}`}
                      className={focusClass}
                      onFocus={() => showPoint(point)}
                      onBlur={() => setActivePoint(null)}
                      onPointerEnter={() => showPoint(point)}
                      onPointerLeave={() => setActivePoint(null)}
                    >
                      <title>{`${entry.name} • ${category}: ${value}`}</title>
                    </rect>
                  );
                })}
              </g>
            );
          })}
        </g>
        <g transform={`translate(0,${innerHeight})`} className="text-slate-500">
          <line x1={0} x2={innerWidth} stroke="currentColor" strokeOpacity="0.22" />
          {categories.map((category) => {
            const bandX = x(category);
            if (bandX == null) return null;
            return (
              <text
                key={category}
                x={bandX + x.bandwidth() / 2}
                y={36}
                textAnchor="middle"
                fontSize="11"
                fill={AXIS_TEXT}
              >
                <title>{category}</title>
                {clampText(category, 14)}
              </text>
            );
          })}
        </g>
        <g className="text-slate-500">
          {yTicks.map((tick) => (
            <text
              key={`y-${tick}`}
              x={-14}
              y={y(tick) + 4}
              textAnchor="end"
              fontSize="11"
              fill={AXIS_TEXT}
              style={{ fontFeatureSettings: '"tnum" 1' }}
            >
              {fmtValue(tick)}
            </text>
          ))}
        </g>
      </>
    );
  } else if (renderModel.kind === "histogram") {
    const values = renderModel.traces.flatMap((trace) =>
      trace.x.filter((value): value is number => typeof value === "number")
    );
    const extent = safeExtent(values, 0.04);
    const binCount = Math.max(12, Math.min(24, Math.round(innerWidth / 42)));
    const bins = d3.bin().domain(extent).thresholds(binCount)(values);
    const x = d3.scaleLinear().domain(extent).nice().range([0, innerWidth]);
    const y = d3
      .scaleLinear()
      .domain([0, d3.max(bins, (bin) => bin.length) ?? 1])
      .nice()
      .range([innerHeight, 0]);
    const xTicks = x.ticks(xTickCount);
    const yTicks = y.ticks(yTickCount);
    const histogramName = renderModel.traces[0]?.name || "Series";
    const color = palette[0];

    chartContent = (
      <>
        <g clipPath={`url(#${clipId})`}>
          {yTicks.map((tick) => (
            <g key={tick} transform={`translate(0,${y(tick)})`}>
              <line x1={0} x2={innerWidth} stroke={GRID_TEXT} strokeWidth={1} />
            </g>
          ))}
          {xTicks.map((tick) => (
            <line
              key={tick}
              x1={x(tick)}
              x2={x(tick)}
              y1={0}
              y2={innerHeight}
              stroke={GRID_TEXT}
              strokeWidth={1}
              opacity={0.4}
            />
          ))}
          {bins.map((bin, index) => {
            const x0 = x(bin.x0 ?? 0);
            const x1 = x(bin.x1 ?? 0);
            if (x0 == null || x1 == null) return null;
            const widthPx = Math.max(1, x1 - x0 - 2);
            const heightPx = innerHeight - y(bin.length);
            const key = `hist:${bin.x0}:${bin.x1}:${index}`;
            const point: NonNullable<ActivePoint> = {
              key,
              title: resolvedTitle,
              series: histogramName,
              x: `${bin.x0?.toFixed(2) ?? ""} to ${bin.x1?.toFixed(2) ?? ""}`,
              y: `${bin.length} values`,
              kind: "histogram",
              color,
            };
            const isActive = activePoint?.key === key;
            return (
              <rect
                key={`${bin.x0}-${bin.x1}-${index}`}
                x={x0 + 1}
                y={y(bin.length)}
                width={widthPx}
                height={Math.max(1, heightPx)}
                rx={3}
                fill={color}
                fillOpacity={isActive ? 0.96 : 0.78}
                stroke="rgba(255,255,255,0.92)"
                strokeWidth={isActive ? 1.3 : 0.8}
                tabIndex={0}
                role="button"
                aria-label={`${histogramName} histogram bin ${bin.x0?.toFixed(2) ?? ""} to ${bin.x1?.toFixed(2) ?? ""}: ${bin.length} values`}
                className={focusClass}
                onFocus={() => showPoint(point)}
                onBlur={() => setActivePoint(null)}
                onPointerEnter={() => showPoint(point)}
                onPointerLeave={() => setActivePoint(null)}
              >
                <title>{`${bin.x0?.toFixed(2)} – ${bin.x1?.toFixed(2)}: ${bin.length}`}</title>
              </rect>
            );
          })}
        </g>
        <g transform={`translate(0,${innerHeight})`} className="text-slate-500">
          <line x1={0} x2={innerWidth} stroke="currentColor" strokeOpacity="0.22" />
          {xTicks.map((tick) => (
            <text
              key={tick}
              x={x(tick)}
              y={36}
              textAnchor="middle"
              fontSize="11"
              fill={AXIS_TEXT}
              style={{ fontFeatureSettings: '"tnum" 1' }}
            >
              {fmtValue(tick)}
            </text>
          ))}
        </g>
        <g className="text-slate-500">
          {yTicks.map((tick) => (
            <text
              key={`y-${tick}`}
              x={-14}
              y={y(tick) + 4}
              textAnchor="end"
              fontSize="11"
              fill={AXIS_TEXT}
              style={{ fontFeatureSettings: '"tnum" 1' }}
            >
              {fmtValue(tick)}
            </text>
          ))}
        </g>
      </>
    );
  } else if (renderModel.kind === "heatmap") {
    // ── HEATMAP ──────────────────────────────────────────────────────────────
    const heatTrace = rawTraces[0];
    const zData = (heatTrace?.z ?? []) as number[][];
    const xLabels = (heatTrace?.x ?? []) as string[];
    const yLabels = (heatTrace?.y ?? []) as string[];

    // Flatten z for domain computation
    const zFlat = zData.flatMap((row) => row.filter((v) => typeof v === "number" && Number.isFinite(v)));
    const zMin = d3.min(zFlat) ?? 0;
    const zMax = d3.max(zFlat) ?? 1;

    const isDiverging = zMin < 0 && zMax > 0;
    const colorScale = isDiverging
      ? d3.scaleSequential(d3.interpolateRdBu).domain([zMax, zMin]) // reversed so red=high, blue=low
      : d3.scaleSequential(d3.interpolateViridis).domain([zMin, zMax]);

    const numCols = xLabels.length || (zData[0]?.length ?? 0);
    const numRows = yLabels.length || zData.length;

    const x = d3
      .scaleBand<string>()
      .domain(xLabels.length ? xLabels : Array.from({ length: numCols }, (_, i) => String(i)))
      .range([0, innerWidth])
      .padding(0.04);

    const y = d3
      .scaleBand<string>()
      .domain(yLabels.length ? yLabels : Array.from({ length: numRows }, (_, i) => String(i)))
      .range([0, innerHeight])
      .padding(0.04);

    const cellW = x.bandwidth();
    const cellH = y.bandwidth();
    const showCellText = cellW > 28 && cellH > 18;

    // Color legend bar — positioned to the right of the chart
    const legendBarId = `${chartId}-hm-legend`;
    const legendX = innerWidth + 12;
    const legendBarH = Math.min(innerHeight, 200);
    const legendBarW = 12;
    const legendBarY = (innerHeight - legendBarH) / 2;

    const legendStops = Array.from({ length: 11 }, (_, i) => {
      const t = i / 10;
      const val = zMin + t * (zMax - zMin);
      return { offset: `${t * 100}%`, color: colorScale(val) };
    });

    const hmXDomain = xLabels.length
      ? xLabels
      : Array.from({ length: numCols }, (_, i) => String(i));
    const hmYDomain = yLabels.length
      ? yLabels
      : Array.from({ length: numRows }, (_, i) => String(i));

    chartContent = (
      <>
        <defs>
          <linearGradient id={legendBarId} x1="0" x2="0" y1="0" y2="1">
            {/* Gradient goes top (zMax) → bottom (zMin) */}
            {legendStops.map((stop, i) => (
              <stop key={i} offset={stop.offset} stopColor={colorScale(zMax - (i / 10) * (zMax - zMin))} />
            ))}
          </linearGradient>
        </defs>
        <g clipPath={`url(#${clipId})`}>
          {zData.map((row, rowIndex) => {
            const rowLabel = hmYDomain[rowIndex] ?? String(rowIndex);
            const cellY = y(rowLabel);
            if (cellY == null) return null;
            return row.map((cellVal, colIndex) => {
              const colLabel = hmXDomain[colIndex] ?? String(colIndex);
              const cellX = x(colLabel);
              if (cellX == null) return null;
              const fill = colorScale(cellVal);
              const key = `hm:${rowIndex}:${colIndex}`;
              const isActive = activePoint?.key === key;
              const point: NonNullable<ActivePoint> = {
                key,
                title: resolvedTitle,
                series: `(${colLabel}, ${rowLabel})`,
                x: colLabel,
                y: rowLabel,
                kind: "heatmap",
                color: fill,
              };

              // Determine text contrast: parse fill to luminance
              let textFill = "#ffffff";
              try {
                const rgb = d3.rgb(fill);
                const luminance =
                  0.2126 * (rgb.r / 255) + 0.7152 * (rgb.g / 255) + 0.0722 * (rgb.b / 255);
                textFill = luminance > 0.45 ? "#0f172a" : "#ffffff";
              } catch {
                textFill = "#ffffff";
              }

              return (
                <g key={key}>
                  <rect
                    x={cellX}
                    y={cellY}
                    width={cellW}
                    height={cellH}
                    rx={3}
                    fill={fill}
                    fillOpacity={isActive ? 1 : 0.9}
                    stroke={isActive ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.35)"}
                    strokeWidth={isActive ? 2 : 0.8}
                    tabIndex={0}
                    role="button"
                    aria-label={`${colLabel}, ${rowLabel}: ${fmtValue(cellVal)}`}
                    className={focusClass}
                    onFocus={() => showPoint(point)}
                    onBlur={() => setActivePoint(null)}
                    onPointerEnter={() => showPoint(point)}
                    onPointerLeave={() => setActivePoint(null)}
                  >
                    <title>{`(${colLabel}, ${rowLabel}): ${cellVal}`}</title>
                  </rect>
                  {showCellText && (
                    <text
                      x={cellX + cellW / 2}
                      y={cellY + cellH / 2 + 4}
                      textAnchor="middle"
                      fontSize={Math.min(10, cellH * 0.38)}
                      fill={textFill}
                      style={{ pointerEvents: "none", userSelect: "none" }}
                    >
                      {fmtValue(cellVal)}
                    </text>
                  )}
                </g>
              );
            });
          })}
        </g>

        {/* X axis labels */}
        <g transform={`translate(0,${innerHeight})`} className="text-slate-500">
          <line x1={0} x2={innerWidth} stroke="currentColor" strokeOpacity="0.22" />
          {hmXDomain.map((label) => {
            const bx = x(label);
            if (bx == null) return null;
            return (
              <text
                key={label}
                x={bx + cellW / 2}
                y={36}
                textAnchor="middle"
                fontSize="10"
                fill={AXIS_TEXT}
              >
                <title>{label}</title>
                {clampText(label, 10)}
              </text>
            );
          })}
        </g>

        {/* Y axis labels */}
        <g className="text-slate-500">
          {hmYDomain.map((label) => {
            const by = y(label);
            if (by == null) return null;
            return (
              <text
                key={label}
                x={-8}
                y={by + cellH / 2 + 4}
                textAnchor="end"
                fontSize="10"
                fill={AXIS_TEXT}
              >
                <title>{label}</title>
                {clampText(label, 12)}
              </text>
            );
          })}
        </g>

        {/* Color legend bar */}
        <g transform={`translate(${legendX}, ${legendBarY})`}>
          <rect
            x={0}
            y={0}
            width={legendBarW}
            height={legendBarH}
            rx={4}
            fill={`url(#${legendBarId})`}
            stroke="rgba(148,163,184,0.3)"
            strokeWidth={0.8}
          />
          {/* Top label (max) */}
          <text x={legendBarW + 4} y={10} fontSize="9" fill={AXIS_TEXT} style={{ fontFeatureSettings: '"tnum" 1' }}>
            {fmtValue(zMax)}
          </text>
          {/* Bottom label (min) */}
          <text x={legendBarW + 4} y={legendBarH} fontSize="9" fill={AXIS_TEXT} style={{ fontFeatureSettings: '"tnum" 1' }}>
            {fmtValue(zMin)}
          </text>
        </g>
      </>
    );
  } else if (renderModel.kind === "box") {
    // ── BOX PLOT ─────────────────────────────────────────────────────────────
    const traceNames = renderModel.traces.map((t) => t.name);
    const x = d3.scaleBand<string>().domain(traceNames).range([0, innerWidth]).padding(0.3);

    const allValues = renderModel.traces.flatMap((t) => t.samples ?? t.y);
    const yDomain = safeExtent(allValues, 0.06);
    const y = d3.scaleLinear().domain(yDomain).nice().range([innerHeight, 0]);
    const yTicks = y.ticks(yTickCount);
    const boxWidth = Math.min(x.bandwidth() * 0.6, 60);

    chartContent = (
      <>
        <g clipPath={`url(#${clipId})`}>
          {yTicks.map((tick) => (
            <g key={tick} transform={`translate(0,${y(tick)})`}>
              <line x1={0} x2={innerWidth} stroke={GRID_TEXT} strokeWidth={1} />
            </g>
          ))}

          {renderModel.traces.map((trace, traceIndex) => {
            const values = trace.samples ?? trace.y;
            if (values.length === 0) return null;
            const stats = computeBoxStats(values);
            const color = palette[traceIndex % palette.length];
            const bx = x(trace.name);
            if (bx == null) return null;
            const cx = bx + x.bandwidth() / 2;
            const bxLeft = cx - boxWidth / 2;

            const whiskerLoY = y(stats.whiskerLo);
            const whiskerHiY = y(stats.whiskerHi);
            const q1Y = y(stats.q1);
            const q3Y = y(stats.q3);
            const medY = y(stats.median);

            const boxKey = `box:${trace.name}`;
            const boxPoint: NonNullable<ActivePoint> = {
              key: boxKey,
              title: resolvedTitle,
              series: trace.name,
              x: trace.name,
              y: `Q1: ${fmtValue(stats.q1)}, Median: ${fmtValue(stats.median)}, Q3: ${fmtValue(stats.q3)}`,
              kind: "box",
              color,
            };
            const isActive = activePoint?.key === boxKey;

            return (
              <g key={trace.name}>
                {/* Low whisker vertical line */}
                <line x1={cx} x2={cx} y1={whiskerLoY} y2={q1Y} stroke={color} strokeWidth={1.5} strokeDasharray="3 2" />
                {/* High whisker vertical line */}
                <line x1={cx} x2={cx} y1={q3Y} y2={whiskerHiY} stroke={color} strokeWidth={1.5} strokeDasharray="3 2" />
                {/* Whisker caps */}
                <line x1={bxLeft + boxWidth * 0.2} x2={bxLeft + boxWidth * 0.8} y1={whiskerLoY} y2={whiskerLoY} stroke={color} strokeWidth={1.5} />
                <line x1={bxLeft + boxWidth * 0.2} x2={bxLeft + boxWidth * 0.8} y1={whiskerHiY} y2={whiskerHiY} stroke={color} strokeWidth={1.5} />

                {/* IQR box */}
                <rect
                  x={bxLeft}
                  y={q3Y}
                  width={boxWidth}
                  height={Math.max(1, q1Y - q3Y)}
                  rx={4}
                  fill={color}
                  fillOpacity={isActive ? 0.86 : 0.68}
                  stroke={color}
                  strokeWidth={isActive ? 1.8 : 1.2}
                  tabIndex={0}
                  role="button"
                  aria-label={`${trace.name} box plot: median ${fmtValue(stats.median)}, Q1 ${fmtValue(stats.q1)}, Q3 ${fmtValue(stats.q3)}`}
                  className={focusClass}
                  onFocus={() => showPoint(boxPoint)}
                  onBlur={() => setActivePoint(null)}
                  onPointerEnter={() => showPoint(boxPoint)}
                  onPointerLeave={() => setActivePoint(null)}
                >
                  <title>{`${trace.name}: Q1=${fmtValue(stats.q1)}, Median=${fmtValue(stats.median)}, Q3=${fmtValue(stats.q3)}`}</title>
                </rect>

                {/* Median line */}
                <line
                  x1={bxLeft}
                  x2={bxLeft + boxWidth}
                  y1={medY}
                  y2={medY}
                  stroke="white"
                  strokeWidth={2}
                  strokeLinecap="round"
                />

                {/* Outlier circles */}
                {stats.outliers.map((val, oi) => {
                  const outlierKey = `${boxKey}:outlier:${oi}`;
                  const outlierPoint: NonNullable<ActivePoint> = {
                    key: outlierKey,
                    title: resolvedTitle,
                    series: trace.name,
                    x: trace.name,
                    y: fmtValue(val),
                    kind: "box",
                    color,
                  };
                  const isOutlierActive = activePoint?.key === outlierKey;
                  return (
                    <circle
                      key={outlierKey}
                      cx={cx}
                      cy={y(val)}
                      r={isOutlierActive ? 4.5 : 3.2}
                      fill={color}
                      fillOpacity={isOutlierActive ? 1 : 0.72}
                      stroke="white"
                      strokeWidth={1.2}
                      tabIndex={0}
                      role="button"
                      aria-label={`${trace.name} outlier: ${fmtValue(val)}`}
                      className={focusClass}
                      onFocus={() => showPoint(outlierPoint)}
                      onBlur={() => setActivePoint(null)}
                      onPointerEnter={() => showPoint(outlierPoint)}
                      onPointerLeave={() => setActivePoint(null)}
                    >
                      <title>{`${trace.name} outlier: ${val}`}</title>
                    </circle>
                  );
                })}
              </g>
            );
          })}
        </g>

        {/* X axis */}
        <g transform={`translate(0,${innerHeight})`} className="text-slate-500">
          <line x1={0} x2={innerWidth} stroke="currentColor" strokeOpacity="0.22" />
          {traceNames.map((name) => {
            const bx = x(name);
            if (bx == null) return null;
            return (
              <text
                key={name}
                x={bx + x.bandwidth() / 2}
                y={36}
                textAnchor="middle"
                fontSize="11"
                fill={AXIS_TEXT}
              >
                <title>{name}</title>
                {clampText(name, 14)}
              </text>
            );
          })}
        </g>

        {/* Y axis */}
        <g className="text-slate-500">
          {yTicks.map((tick) => (
            <text
              key={`y-${tick}`}
              x={-14}
              y={y(tick) + 4}
              textAnchor="end"
              fontSize="11"
              fill={AXIS_TEXT}
              style={{ fontFeatureSettings: '"tnum" 1' }}
            >
              {fmtValue(tick)}
            </text>
          ))}
        </g>
      </>
    );
  } else if (renderModel.kind === "violin") {
    // ── VIOLIN PLOT ───────────────────────────────────────────────────────────
    const traceNames = renderModel.traces.map((t) => t.name);
    const x = d3.scaleBand<string>().domain(traceNames).range([0, innerWidth]).padding(0.3);

    const allValues = renderModel.traces.flatMap((t) => t.samples ?? t.y);
    const yDomain = safeExtent(allValues, 0.06);
    const y = d3.scaleLinear().domain(yDomain).nice().range([innerHeight, 0]);
    const yTicks = y.ticks(yTickCount);

    chartContent = (
      <>
        <g clipPath={`url(#${clipId})`}>
          {yTicks.map((tick) => (
            <g key={tick} transform={`translate(0,${y(tick)})`}>
              <line x1={0} x2={innerWidth} stroke={GRID_TEXT} strokeWidth={1} />
            </g>
          ))}

          {renderModel.traces.map((trace, traceIndex) => {
            const values = trace.samples ?? trace.y;
            if (values.length === 0) return null;
            const color = palette[traceIndex % palette.length];
            const bx = x(trace.name);
            if (bx == null) return null;
            const cx = bx + x.bandwidth() / 2;

            // Bin values for KDE approximation
            const bins = d3
              .bin()
              .domain(y.domain() as [number, number])
              .thresholds(y.ticks(20))(values);

            const maxBinLen = d3.max(bins, (b) => b.length) || 1;
            const xNum = d3
              .scaleLinear()
              .domain([-maxBinLen, maxBinLen])
              .range([0, x.bandwidth()]);

            // Build violin path using area generator
            const areaGen = d3
              .area<d3.Bin<number, number>>()
              .x0((b) => xNum(-b.length))
              .x1((b) => xNum(b.length))
              .y((b) => y((b.x0! + b.x1!) / 2))
              .curve(d3.curveCatmullRom);

            const violinPath = areaGen(bins);

            // Median
            const sorted = values.slice().sort(d3.ascending);
            const medianVal = d3.quantile(sorted, 0.5) ?? values[0] ?? 0;
            const medY = y(medianVal);

            const violinKey = `violin:${trace.name}`;
            const violinPoint: NonNullable<ActivePoint> = {
              key: violinKey,
              title: resolvedTitle,
              series: trace.name,
              x: trace.name,
              y: `Median: ${fmtValue(medianVal)}`,
              kind: "violin",
              color,
            };
            const isActive = activePoint?.key === violinKey;

            return (
              <g key={trace.name} transform={`translate(${bx}, 0)`}>
                <path
                  d={violinPath ?? ""}
                  fill={color}
                  fillOpacity={isActive ? 0.82 : 0.6}
                  stroke={color}
                  strokeWidth={isActive ? 1.6 : 1}
                  strokeOpacity={0.9}
                  tabIndex={0}
                  role="button"
                  aria-label={`${trace.name} violin: median ${fmtValue(medianVal)}`}
                  className={focusClass}
                  onFocus={() => showPoint(violinPoint)}
                  onBlur={() => setActivePoint(null)}
                  onPointerEnter={() => showPoint(violinPoint)}
                  onPointerLeave={() => setActivePoint(null)}
                >
                  <title>{`${trace.name}: median=${fmtValue(medianVal)}, n=${values.length}`}</title>
                </path>

                {/* Median line inside violin */}
                <line
                  x1={xNum(-maxBinLen * 0.3)}
                  x2={xNum(maxBinLen * 0.3)}
                  y1={medY}
                  y2={medY}
                  stroke="white"
                  strokeWidth={2}
                  strokeLinecap="round"
                />
              </g>
            );
          })}
        </g>

        {/* X axis */}
        <g transform={`translate(0,${innerHeight})`} className="text-slate-500">
          <line x1={0} x2={innerWidth} stroke="currentColor" strokeOpacity="0.22" />
          {traceNames.map((name) => {
            const bx = x(name);
            if (bx == null) return null;
            return (
              <text
                key={name}
                x={bx + x.bandwidth() / 2}
                y={36}
                textAnchor="middle"
                fontSize="11"
                fill={AXIS_TEXT}
              >
                <title>{name}</title>
                {clampText(name, 14)}
              </text>
            );
          })}
        </g>

        {/* Y axis */}
        <g className="text-slate-500">
          {yTicks.map((tick) => (
            <text
              key={`y-${tick}`}
              x={-14}
              y={y(tick) + 4}
              textAnchor="end"
              fontSize="11"
              fill={AXIS_TEXT}
              style={{ fontFeatureSettings: '"tnum" 1' }}
            >
              {fmtValue(tick)}
            </text>
          ))}
        </g>
      </>
    );
  } else {
    const allPoints = renderModel.traces.flatMap((entry) => readPoints(entry));
    const xs = allPoints.map((point) => point.x);
    const ys = allPoints.map((point) => point.y);
    const x = d3.scaleLinear().domain(safeExtent(xs)).nice().range([0, innerWidth]);
    const y = d3.scaleLinear().domain(safeExtent(ys)).nice().range([innerHeight, 0]);
    const xTicks = x.ticks(xTickCount);
    const yTicks = y.ticks(yTickCount);
    const lineColor = palette[0];

    chartContent = (
      <>
        <defs>
          <linearGradient id={areaGradientId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={lineColor} stopOpacity="0.22" />
            <stop offset="100%" stopColor={lineColor} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <g clipPath={`url(#${clipId})`}>
          {yTicks.map((tick) => (
            <g key={tick} transform={`translate(0,${y(tick)})`}>
              <line x1={0} x2={innerWidth} stroke={GRID_TEXT} strokeWidth={1} />
            </g>
          ))}
          {xTicks.map((tick) => (
            <line
              key={tick}
              x1={x(tick)}
              x2={x(tick)}
              y1={0}
              y2={innerHeight}
              stroke={GRID_TEXT}
              strokeWidth={1}
              opacity={0.35}
            />
          ))}
          {/* Layout shapes — horizontal/vertical reference lines */}
          {layoutShapes.map((shape, shapeIndex) => {
            const shapeColor = shape.line?.color || "#334155";
            const shapeDash = shape.line?.dash === "dash" || shape.line?.dash === "dot" ? "6 4" : undefined;
            const shapeWidth = shape.line?.width ?? 1;
            // Horizontal line (y0 === y1)
            if (shape.y0 != null && shape.y0 === shape.y1) {
              const yPos = y(shape.y0);
              return (
                <line key={`shape-${shapeIndex}`} x1={0} x2={innerWidth}
                  y1={yPos} y2={yPos} stroke={shapeColor}
                  strokeWidth={shapeWidth} strokeDasharray={shapeDash}
                  opacity={0.7} />
              );
            }
            // Vertical line (x0 === x1)
            if (shape.x0 != null && shape.x0 === shape.x1) {
              const xPos = x(shape.x0);
              return (
                <line key={`shape-${shapeIndex}`} x1={xPos} x2={xPos}
                  y1={0} y2={innerHeight} stroke={shapeColor}
                  strokeWidth={shapeWidth} strokeDasharray={shapeDash}
                  opacity={0.7} />
              );
            }
            return null;
          })}
          {renderModel.traces.map((series, seriesIndex) => {
            const points = readPoints(series).sort((a, b) => a.x - b.x);
            const color = traceColors[seriesIndex] || palette[seriesIndex % palette.length];
            const rawTrace = rawTraces[seriesIndex];
            const lineDash = rawTrace?.line?.dash;
            const dashArray = lineDash === "dash" ? "8 4" : lineDash === "dot" ? "2 4" : lineDash === "dashdot" ? "8 4 2 4" : undefined;
            const markerSymbol = String(rawTrace?.marker?.symbol ?? "");
            const isOpenMarker = markerSymbol.includes("open");
            const rawSize = typeof rawTrace?.marker?.size === "number" ? rawTrace.marker.size : null;
            const ptRadius = rawSize ? Math.max(3, rawSize * 0.55) : series.kind === "scatter" ? 4.7 : 3.8;
            // For line traces with many points, hide dots — show line only, reveal dots on hover
            const showDots = series.kind !== "line" || points.length <= 30;
            const lineGen = d3
              .line<Point>()
              .x((d) => x(d.x))
              .y((d) => y(d.y))
              .curve(d3.curveMonotoneX);
            const areaGen =
              renderModel.kind === "line" && renderModel.traces.length === 1
                ? d3
                    .area<Point>()
                    .x((d) => x(d.x))
                    .y0(innerHeight)
                    .y1((d) => y(d.y))
                    .curve(d3.curveMonotoneX)
                : null;

            return (
              <g key={`${series.name}-${seriesIndex}`}>
                {areaGen && points.length > 1 && (
                  <path d={areaGen(points) ?? ""} fill={`url(#${areaGradientId})`} />
                )}
                {series.kind === "line" && points.length > 1 && (
                  <>
                    <path
                      d={lineGen(points) ?? ""}
                      fill="none"
                      stroke={color}
                      strokeOpacity={0.12}
                      strokeWidth={7}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeDasharray={dashArray}
                    />
                    <path
                      d={lineGen(points) ?? ""}
                      fill="none"
                      stroke={color}
                      strokeWidth={2.8}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeDasharray={dashArray}
                    />
                  </>
                )}
                {points.map((point, pointIndex) => {
                  const key = point.key;
                  const pointActive = activePoint?.key === key;
                  const pointColor = color;
                  const displayPoint: NonNullable<ActivePoint> = {
                    key,
                    title: resolvedTitle,
                    series: series.name,
                    x: fmtValue(point.x),
                    y: fmtValue(point.y),
                    kind: series.kind,
                    color: pointColor,
                  };
                  return (
                    <circle
                      key={`${series.name}-${pointIndex}`}
                      cx={x(point.x)}
                      cy={y(point.y)}
                      r={pointActive ? ptRadius + 1.35 : showDots ? ptRadius : 0}
                      fill={isOpenMarker ? "none" : pointActive ? pointColor : showDots ? pointColor : "transparent"}
                      fillOpacity={isOpenMarker ? 0 : pointActive ? 1 : showDots ? 0.92 : 0}
                      stroke={pointActive ? (isOpenMarker ? pointColor : "white") : isOpenMarker ? pointColor : "none"}
                      strokeWidth={isOpenMarker ? 2 : pointActive ? 2 : 0}
                      tabIndex={showDots ? 0 : -1}
                      role="button"
                      aria-label={`${series.name} point ${point.label || `${fmtValue(point.x)}, ${fmtValue(point.y)}`}`}
                      className={focusClass}
                      style={{ pointerEvents: "all", cursor: "crosshair" }}
                      onFocus={() => showPoint(displayPoint)}
                      onBlur={() => setActivePoint(null)}
                      onPointerEnter={() => showPoint(displayPoint)}
                      onPointerLeave={() => setActivePoint(null)}
                    >
                      <title>{point.label || `${point.x}, ${point.y}`}</title>
                    </circle>
                  );
                })}
              </g>
            );
          })}
        </g>
        <g transform={`translate(0,${innerHeight})`} className="text-slate-500">
          <line x1={0} x2={innerWidth} stroke="currentColor" strokeOpacity="0.22" />
          {xTicks.map((tick) => (
            <text
              key={tick}
              x={x(tick)}
              y={36}
              textAnchor="middle"
              fontSize="11"
              fill={AXIS_TEXT}
              style={{ fontFeatureSettings: '"tnum" 1' }}
            >
              {fmtValue(tick)}
            </text>
          ))}
        </g>
        <g className="text-slate-500">
          {yTicks.map((tick) => (
            <text
              key={`y-${tick}`}
              x={-14}
              y={y(tick) + 4}
              textAnchor="end"
              fontSize="11"
              fill={AXIS_TEXT}
              style={{ fontFeatureSettings: '"tnum" 1' }}
            >
              {fmtValue(tick)}
            </text>
          ))}
        </g>
      </>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`relative h-full w-full ${className}`}
      style={{ minHeight: Math.max(360, display?.height ?? 360) }}
    >
      {legendEntries.length > 1 && (
        <div className="pointer-events-none absolute right-4 top-4 z-10 flex max-w-[56%] flex-wrap justify-end gap-2">
          {legendEntries.map((entry, idx) => (
            <span key={`${entry.name}-${idx}`}>{renderLegendEntry(entry, activeSeries)}</span>
          ))}
        </div>
      )}

      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={resolvedTitle}
        className="block h-full w-full overflow-visible"
      >
        <defs>
          <linearGradient id={bgGradientId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#f8fafc" />
            <stop offset="100%" stopColor="#f1f5f9" />
          </linearGradient>
          <clipPath id={clipId}>
            <rect x={0} y={0} width={innerWidth} height={innerHeight} rx={4} />
          </clipPath>
        </defs>

        <rect x={0} y={0} width={width} height={height} rx={6} fill={`url(#${bgGradientId})`} />
        <rect
          x={margin.left - 8}
          y={margin.top - 8}
          width={innerWidth + 16}
          height={innerHeight + 16}
          rx={6}
          fill="rgba(255,255,255,0.72)"
          stroke="rgba(148, 163, 184, 0.10)"
        />

        <g transform={`translate(${margin.left},${margin.top})`}>
          <text x={0} y={-12} fontSize="14" fontWeight="700" fill={PANEL_TEXT}>
            {resolvedTitle}
          </text>

          {chartContent}

          <g transform={`translate(0,${innerHeight})`}>
            <text
              x={innerWidth / 2}
              y={50}
              textAnchor="middle"
              fontSize="12"
              fontWeight="600"
              fill={AXIS_TEXT}
            >
              {resolvedXLabel}
            </text>
          </g>

          <g>
            <text
              x={-(margin.left - 12)}
              y={innerHeight / 2}
              transform={`rotate(-90 ${-(margin.left - 12)} ${innerHeight / 2})`}
              textAnchor="middle"
              fontSize="12"
              fontWeight="600"
              fill={AXIS_TEXT}
            >
              {resolvedYLabel}
            </text>
          </g>
        </g>
      </svg>

      {activePoint && (
        <div className="pointer-events-none absolute bottom-4 left-4 z-10 max-w-[18rem] rounded-2xl border border-outline-variant/20 bg-white/92 px-3 py-2.5 text-xs shadow-[0_14px_40px_rgba(15,23,42,0.12)] backdrop-blur">
          <div className="flex items-center gap-2">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: activePoint.color }}
            />
            <div className="font-semibold text-on-surface">{activePoint.title}</div>
          </div>
          <div className="mt-1 text-on-surface-variant">{activePoint.series}</div>
          <div className="mt-1.5 text-on-surface-variant">
            <span className="font-medium text-on-surface">{activePoint.x}</span> · {activePoint.y}
          </div>
        </div>
      )}
    </div>
  );
}
