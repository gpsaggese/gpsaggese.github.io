"use client";

import dynamic from "next/dynamic";
import { useMemo, type CSSProperties, type ComponentType } from "react";

type PlotComponentProps = {
  data: unknown[];
  layout: Record<string, unknown>;
  config: Record<string, unknown>;
  useResizeHandler?: boolean;
  style?: CSSProperties;
};

const Plot = dynamic(
  () =>
    Promise.all([
      import("plotly.js-dist-min"),
      import("react-plotly.js/factory"),
    ]).then(([Plotly, factoryMod]) => {
      const createPlotlyComponent = factoryMod.default as (
        plotly: unknown
      ) => ComponentType<PlotComponentProps>;
      return createPlotlyComponent(Plotly);
    }),
  { ssr: false, loading: () => <div className="p-4 text-on-surface/50 text-sm">Loading chart...</div> }
);

interface Props {
  plotlyJson: string;
  className?: string;
  style?: CSSProperties;
  minHeight?: number;
  maxHeight?: number;
  width?: number;
  height?: number;
}

export default function InteractivePlot({
  plotlyJson,
  className = "",
  style,
  minHeight = 320,
  maxHeight,
  width,
  height,
}: Props) {
  const parsed = useMemo(() => {
    try {
      const obj = typeof plotlyJson === "string" ? JSON.parse(plotlyJson) : plotlyJson;
      return { data: obj.data ?? [], layout: obj.layout ?? {} };
    } catch (e) {
      return { error: String(e) };
    }
  }, [plotlyJson]);

  if ("error" in parsed) {
    return (
      <div className="p-4 text-error text-sm font-mono">
        Failed to parse Plotly JSON: {parsed.error}
      </div>
    );
  }

  // Neural Slate design system: primary blue #004ac6, slate text, Inter font
  const neuralSlateColorway = [
    "#004ac6", "#f28e2b", "#1d6f8a", "#e15759", "#59a14f",
    "#8b5cf6", "#0891b2", "#b45309", "#64748b", "#6366f1",
  ];
  const layout = {
    ...parsed.layout,
    ...(width ? { width } : {}),
    ...(height ? { height } : {}),
    autosize: true,
    paper_bgcolor: "rgba(248, 250, 252, 0.95)",
    plot_bgcolor: "rgba(248, 250, 252, 0.95)",
    colorway: parsed.layout?.colorway || neuralSlateColorway,
    margin:
      typeof parsed.layout?.margin === "object" && parsed.layout.margin !== null
        ? parsed.layout.margin
        : { t: 28, r: 20, b: 52, l: 60 },
    font: {
      family: "Inter, system-ui, sans-serif",
      size: 12,
      color: "#475569",
      ...(typeof parsed.layout?.font === "object" ? parsed.layout.font : {}),
    },
    xaxis: {
      ...(typeof parsed.layout?.xaxis === "object" ? parsed.layout.xaxis : {}),
      gridcolor: "rgba(71, 85, 105, 0.10)",
      linecolor: "rgba(71, 85, 105, 0.10)",
      zerolinecolor: "rgba(71, 85, 105, 0.20)",
    },
    yaxis: {
      ...(typeof parsed.layout?.yaxis === "object" ? parsed.layout.yaxis : {}),
      gridcolor: "rgba(71, 85, 105, 0.10)",
      linecolor: "rgba(71, 85, 105, 0.10)",
      zerolinecolor: "rgba(71, 85, 105, 0.20)",
    },
  };

  return (
    <div
      className={`w-full rounded-lg overflow-hidden ${className}`}
      style={{ minHeight, maxHeight, ...style }}
    >
      <Plot
        data={parsed.data}
        layout={layout}
        config={{ responsive: true, displayModeBar: true, displaylogo: false }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
