"use client";

import dynamic from "next/dynamic";
import type { StoryPlot, StorySection } from "@/lib/types";
import { resolveStoryPlotRenderer } from "@/lib/reportEngine";
import ReportD3Chart from "./ReportD3Chart";

const InteractivePlot = dynamic(
  () => import("@/components/notebook/InteractivePlot"),
  { ssr: false }
);

interface Props {
  plot: StoryPlot | string;
  section: Pick<StorySection, "title" | "visual_title" | "visual_caption">;
  index?: number;
  plotCellId?: string;
  className?: string;
}

export default function StoryPlotRenderer({
  plot,
  section,
  index = 0,
  plotCellId,
  className = "",
}: Props) {
  const resolved = resolveStoryPlotRenderer({
    plot,
    context: {
      sectionTitle: section.title,
      fallbackTitle: section.visual_title || section.title,
      fallbackCaption: section.visual_caption,
      plotCellId,
      index,
    },
  });
  const plotSource = resolved.plot.source ?? resolved.plot.payload ?? "";

  if (resolved.renderer === "d3") {
    return (
      <ReportD3Chart
        plotlyJson={plotSource}
        display={resolved.display}
        spec={resolved.viz}
        title={resolved.title}
        className={className}
      />
    );
  }

  if (resolved.renderer === "plotly") {
    return (
      <InteractivePlot
        plotlyJson={plotSource}
        className={className}
        width={resolved.display.width ?? 960}
        height={resolved.display.height ?? 540}
        minHeight={resolved.display.height ?? 540}
      />
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={plotSource}
      alt={resolved.caption || `${section.title} plot ${resolved.title || ""}`.trim()}
      className={`h-full w-full object-contain ${className}`}
    />
  );
}
