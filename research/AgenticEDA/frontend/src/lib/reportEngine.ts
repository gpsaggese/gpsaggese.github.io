import type { StoryPlot } from "@/lib/types";
import {
  resolveReportVizSpec,
  type ReportVizContext,
  type ReportVizResolution,
} from "./reportViz";

export interface StoryPlotRenderRequest {
  plot: StoryPlot | string;
  context?: ReportVizContext;
}

export function resolveStoryPlotRenderer({
  plot,
  context = {},
}: StoryPlotRenderRequest): ReportVizResolution {
  return resolveReportVizSpec(plot, context);
}
