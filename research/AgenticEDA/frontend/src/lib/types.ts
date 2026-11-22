export interface Session {
  session_id: string;
  original_filename: string;
  source_format: string;
  created_at: string;
  row_count: number;
  col_count: number;
}

export interface CellOutput {
  output_type: "stream" | "execute_result" | "display_data" | "error";
  text?: string;
  data?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  ename?: string;
  evalue?: string;
  traceback?: string[];
}

export interface Cell {
  id: string;
  cell_type: "code" | "markdown";
  source: string;
  outputs: CellOutput[];
  execution_count: number | null;
  metadata?: Record<string, unknown>;
  baselineOutputs?: CellOutput[];
  executing?: boolean;
  error?: string | null;
  thinking?: string | null;
}

export interface StorySection {
  phase: string;
  title: string;
  content: string;
  plots?: Array<StoryPlot | string>;
  insights?: InsightCard[];
  cell_ids?: string[];
  plot_cell_ids?: string[];
  subtitle?: string;
  summary?: string;
  tags?: string[];
  visual_title?: string;
  visual_caption?: string;
}

export interface InsightCard {
  type: string;
  description: string;
  phase: string;
  rule: number;
  confidence?: number;
  mini_chart?: string;
}

export interface StoryTakeaway {
  id: string;
  title: string;
  body: string;
  points: string[];
  icon: string;
  index: number;
}

export interface StoryPlot {
  kind?: "plotly" | "image";
  mime_type?: string;
  source?: string;
  type?: "plotly" | "image";
  payload?: string;
  caption?: string;
  title?: string;
  source_path?: string;
  source_cell_id?: string;
  display?: StoryPlotDisplay;
  chart_family?: "scatter" | "line" | "bar" | "histogram" | "box" | "violin" | "heatmap" | "image" | "unknown";
  semantic_intent?: "trend" | "comparison" | "distribution" | "relationship" | "matrix" | "residual" | "unknown";
  x_axis_role?: "time" | "category" | "ordinal" | "numeric" | "index" | "measure" | "count" | "density" | "unknown";
  y_axis_role?: "time" | "category" | "ordinal" | "numeric" | "index" | "measure" | "count" | "density" | "unknown";
  semantic_confidence?: number;
  plot_spec?: StoryPlotVizSpecInput;
  viz_spec?: StoryPlotVizSpecInput;
}

export interface StoryPlotDisplay {
  width: number;
  height: number;
  aspect_ratio: number;
  fit: "contain" | "cover";
  native_width?: number | null;
  native_height?: number | null;
  max_width?: number;
  max_height?: number;
}

export interface StoryPlotVizChannel {
  axis: "x" | "y";
  role: "time" | "category" | "ordinal" | "numeric" | "index" | "measure" | "count" | "density" | "unknown";
  scale: "linear" | "band" | "time" | "ordinal" | "log" | "unknown";
  label: string;
}

export interface StoryPlotVizLegend {
  show: boolean;
  position: "top" | "right" | "bottom" | "none";
}

export interface StoryPlotVizSpec {
  renderer: "report_d3" | "d3" | "plotly" | "image" | "unknown";
  fallback_renderer: "report_d3" | "d3" | "plotly" | "image" | "unknown";
  chart_family: "scatter" | "line" | "bar" | "histogram" | "box" | "violin" | "heatmap" | "image" | "unknown";
  semantic_intent: "trend" | "comparison" | "distribution" | "relationship" | "matrix" | "residual" | "unknown";
  mark: "point" | "line" | "bar" | "rect" | "box" | "violin" | "heatmap" | "image" | "unknown";
  orientation: "vertical" | "horizontal" | "matrix" | "none";
  trace_count: number;
  report_native: boolean;
  x: StoryPlotVizChannel;
  y: StoryPlotVizChannel;
  legend: StoryPlotVizLegend;
  interactions: Array<"hover" | "focus" | "zoom" | "pan" | "source_link">;
  confidence: number;
  reason: string;
  display: StoryPlotDisplay;
}

export type StoryPlotVizSpecInput = Record<string, unknown> & Partial<StoryPlotVizSpec>;

export interface ChatMessage {
  id: string;
  role: "user" | "agent";
  content: string;
  type: "text" | "cell_ref" | "action";
  cell_id?: string;
  meta?: {
    kind?: "investigation" | "action" | "info" | "steering";
    notebook_id?: string;
    hypothesis_id?: string;
    title?: string;
    status?: "started" | "running" | "complete" | "failed" | "timeout" | "queued" | "read";
    confidence?: number;
  };
  timestamp: string;
}

export interface Version {
  version_id: number;
  trigger: string;
  timestamp: string;
}
