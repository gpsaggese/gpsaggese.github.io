"use client";

import { useEffect, useMemo, useState, useCallback, useRef, type CSSProperties } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { useStoryStore } from "@/stores/storyStore";
import { useNotebookStore } from "@/stores/notebookStore";
import { getStory } from "@/lib/api";
import { API_BASE } from "@/lib/backend";
import {
  defaultPlotDisplay,
  normalizeReportPlot,
} from "@/lib/reportViz";
import type { Cell, CellOutput, StoryPlot, StoryPlotVizSpecInput } from "@/lib/types";
import StorySectionCard from "./StorySectionCard";

/** Inline markdown — renders as a <span> without wrapping <p> tags */
function InlineMd({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{ p: ({ children: c }) => <>{c}</> }}
    >
      {children}
    </ReactMarkdown>
  );
}

function ReportMarkdown({ children, className = "" }: { children: string; className?: string }) {
  return (
    <div className={`report-prose prose prose-sm max-w-none text-on-surface-variant ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
        {children}
      </ReactMarkdown>
    </div>
  );
}

function iconForText(text: string): string {
  const lower = text.toLowerCase();
  if (lower.includes("correlat")) return "hub";
  if (lower.includes("trend") || lower.includes("rise") || lower.includes("fall")) return "trending_up";
  if (lower.includes("outlier") || lower.includes("anomal")) return "warning";
  if (lower.includes("distribution") || lower.includes("spread")) return "bar_chart";
  if (lower.includes("missing") || lower.includes("null")) return "report";
  if (lower.includes("cluster") || lower.includes("segment")) return "workspaces";
  if (lower.includes("wind") || lower.includes("speed")) return "air";
  if (lower.includes("power") || lower.includes("energy")) return "electric_bolt";
  if (lower.includes("next") || lower.includes("recommend")) return "task_alt";
  return "insights";
}

function extractPlotsFromOutputs(outputs: CellOutput[], sectionTitle = ""): StoryPlot[] {
  const plots: StoryPlot[] = [];
  for (const output of outputs || []) {
    const data = output.data || {};
    const metadata = output.metadata || {};
    const plotSpec =
      (metadata.plot_spec as StoryPlotVizSpecInput | undefined) ||
      (metadata.viz_spec as StoryPlotVizSpecInput | undefined) ||
      (data["application/vnd.agenticeda.plot-spec+json"] as StoryPlotVizSpecInput | undefined) ||
      null;
    const plotlyData = data["application/vnd.plotly.v1+json"];
    if (plotlyData !== undefined) {
      plots.push(
        normalizeReportPlot(
          {
            kind: "plotly",
            mime_type: "application/vnd.plotly.v1+json",
            source: typeof plotlyData === "string" ? plotlyData : JSON.stringify(plotlyData),
            title: sectionTitle,
            display: defaultPlotDisplay("plotly"),
            plot_spec: plotSpec ?? undefined,
            viz_spec: plotSpec ?? undefined,
          },
          { sectionTitle }
        )
      );
      continue;
    }

    const imgData = data["image/png"];
    if (imgData !== undefined) {
      const raw = typeof imgData === "string" ? imgData : JSON.stringify(imgData);
      plots.push(
        normalizeReportPlot(
          {
            kind: "image",
            mime_type: "image/png",
            source: raw.startsWith("data:") ? raw : `data:image/png;base64,${raw.replace(/\s+/g, "")}`,
            title: sectionTitle,
            display: defaultPlotDisplay("image"),
            plot_spec: plotSpec ?? undefined,
            viz_spec: plotSpec ?? undefined,
          },
          { sectionTitle }
        )
      );
    }
  }
  return plots;
}

function formatGeneratedAt(value: string): string {
  if (!value) return "";
  return new Date(value).toLocaleString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

interface StoryPaneProps {
  sessionId: string;
  onOpenNotebookCell?: (cellId: string) => void;
}

export default function StoryPane({ sessionId, onOpenNotebookCell }: StoryPaneProps) {
  const {
    title,
    executiveSummary,
    summaryBlocks,
    takeaways,
    sections,
    generatedAt,
    loading,
    error,
  } = useStoryStore();
  const pipelineRunning = useNotebookStore((s) => s.pipelineRunning);
  const planSteps = useNotebookStore((s) => s.planSteps);
  const cells = useNotebookStore((s) => s.cells);
  const setStory = useStoryStore((s) => s.setStory);
  const setLoading = useStoryStore((s) => s.setLoading);
  const setError = useStoryStore((s) => s.setError);
  const setActiveTab = useNotebookStore((s) => s.setActiveTab);
  const [regenerating, setRegenerating] = useState(false);
  const [exportingPdf, setExportingPdf] = useState(false);
  const [exportingMd, setExportingMd] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [changesDetected, setChangesDetected] = useState(false);

  // Compute a simple hash of MAIN notebook cell sources only.
  // Investigation notebooks (agent_N, chat_*) are separate tabs — their
  // changes should not trigger "changes detected" on the story.
  const contentHash = useMemo(() => {
    let hash = 0;
    for (const cell of cells) {
      for (let i = 0; i < cell.source.length; i++) {
        hash = ((hash << 5) - hash + cell.source.charCodeAt(i)) | 0;
      }
    }
    return hash;
  }, [cells]);

  const storyGeneratedAt = generatedAt;

  const baselineHashRef = useRef<number | null>(null);
  const baselineSetRef = useRef(false);

  useEffect(() => {
    // Set baseline when pipeline finishes
    if (!pipelineRunning && storyGeneratedAt && !baselineSetRef.current && contentHash !== 0) {
      const timer = setTimeout(() => {
        baselineHashRef.current = contentHash;
        baselineSetRef.current = true;
      }, 3000);
      return () => clearTimeout(timer);
    }
    // Detect changes after baseline
    if (baselineSetRef.current && !pipelineRunning && baselineHashRef.current !== null) {
      if (contentHash !== baselineHashRef.current) {
        setChangesDetected(true);
      }
    }
  }, [contentHash, storyGeneratedAt, pipelineRunning]);

  useEffect(() => {
    if (!title && !loading && !pipelineRunning) {
      setLoading(true);
      getStory(sessionId, "json")
        .then((data) => {
          if (data) setStory(data);
          else setError("not_found");
        })
        .catch(() => setError("not_found"));
    }
  }, [sessionId, title, loading, pipelineRunning, setStory, setLoading, setError]);

  const sectionData = useMemo(() => {
    return sections.map((section) => {
      const sectionCellIds = section.cell_ids || [];
      const sectionTitle = section.visual_title || section.title;
      const declaredPlots = (section.plots || []).map((plot) =>
        normalizeReportPlot(plot, {
          sectionTitle,
          fallbackTitle: sectionTitle,
          fallbackCaption: section.visual_caption,
        })
      );
      const plots: StoryPlot[] = declaredPlots.length > 0 ? declaredPlots : [];

      if (plots.length === 0) {
        cells.forEach((cell) => {
          if (sectionCellIds.includes(cell.id) && cell.outputs) {
            plots.push(...extractPlotsFromOutputs(cell.outputs, sectionTitle));
          }
        });
      }

      return {
        section,
        plots,
        plotCellIds: (() => {
          const sourceIds = plots
            .map((plot) => plot.source_cell_id)
            .filter((cellId): cellId is string => Boolean(cellId));
          return sourceIds.length > 0 ? sourceIds : section.plot_cell_ids || sectionCellIds;
        })(),
      };
    });
  }, [sections, cells]);

  const openNotebookCell = useCallback((cellId: string) => {
    if (onOpenNotebookCell) {
      onOpenNotebookCell(cellId);
      return;
    }

    // Find which notebook owns this cell and switch to it
    const store = useNotebookStore.getState();
    const notebooks = store.notebooks;

    // Check main notebook first
    let targetNotebook = "main";
    if (!store.cells.some((c: Cell) => c.id === cellId)) {
      // Search investigation notebooks
      for (const [nbId, nb] of Object.entries(notebooks)) {
        if (nbId !== "main" && nb.cells.some((c: Cell) => c.id === cellId)) {
          targetNotebook = nbId;
          break;
        }
      }
    }

    store.setActiveNotebook(targetNotebook);
    setActiveTab("notebook");

    // Wait for notebook switch to render, then scroll to cell
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        const el = document.getElementById(`cell-${cellId}`);
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "center" });
          el.classList.add("ring-2", "ring-primary/40", "ring-offset-2", "activity-highlight");
          window.setTimeout(
            () => el.classList.remove("ring-2", "ring-primary/40", "ring-offset-2", "activity-highlight"),
            1400
          );
        }
      });
    });
  }, [onOpenNotebookCell, setActiveTab]);

  const handleRegenerate = async () => {
    setRegenerating(true);
    try {
      const res = await fetch(`${API_BASE}/story/${sessionId}/regenerate`, {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        setStory(data);
        setChangesDetected(false);
        // Reset baseline to current hash so it doesn't immediately re-trigger
        baselineHashRef.current = contentHash;
      }
    } catch {
      // ignore
    } finally {
      setRegenerating(false);
    }
  };

  const handleExport = async (format: "pdf" | "md") => {
    if (format === "pdf") setExportingPdf(true);
    else setExportingMd(true);
    try {
      const blob =
        format === "pdf"
          ? await getStory(sessionId, "pdf")
          : new Blob([await getStory(sessionId, "md")], {
              type: "text/markdown",
            });

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `eda-report.${format}`;
      a.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1500);
    } catch {
      // ignore export errors
    } finally {
      if (format === "pdf") setExportingPdf(false);
      else setExportingMd(false);
    }
  };

  // Auto-retry polling when story hasn't been generated yet
  useEffect(() => {
    if (error !== "not_found" || title || loading || pipelineRunning) return;
    if (retryCount >= 15) return;
    const timer = window.setTimeout(() => {
      setError(null);
      setLoading(true);
      getStory(sessionId, "json")
        .then((d) => {
          if (d) setStory(d);
          else setError("not_found");
        })
        .catch(() => setError("not_found"));
      setRetryCount((c) => c + 1);
    }, 2000);
    return () => window.clearTimeout(timer);
  }, [error, title, loading, pipelineRunning, retryCount, sessionId, setError, setLoading, setStory]);

  // Animate sections into view as they scroll into the viewport
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("animate-rise-in");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -50px 0px" }
    );
    document.querySelectorAll("[data-animate]").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [sectionData]);

  // Derive report generation status from the execution plan
  const reportStep = planSteps.find((s) => s.phase === "Report Generation");
  const reportDone = reportStep?.status === "complete";
  const reportRunning = reportStep?.status === "current";

  if (pipelineRunning) {
    // Show real plan steps if available, otherwise generic message
    const stepsToShow = planSteps.filter((s) => s.status !== "skipped");
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 text-on-surface-variant">
        <div className="h-7 w-7 rounded-full border-2 border-primary border-t-transparent animate-spin" />
        <p className="text-sm font-body">Agent is analyzing your data...</p>
        {stepsToShow.length > 0 ? (
          <div className="flex flex-col gap-1.5 mt-1 text-xs text-on-surface-variant font-body">
            {stepsToShow.map((step) => (
              <div key={step.phase} className="flex items-center gap-2">
                {step.status === "complete" ? (
                  <span className="material-symbols-outlined text-sm text-primary">check_circle</span>
                ) : step.status === "current" ? (
                  <div className="h-3.5 w-3.5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                ) : (
                  <span className="material-symbols-outlined text-sm text-outline-variant/40">circle</span>
                )}
                <span className={step.status === "upcoming" ? "opacity-40" : ""}>{step.phase}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-outline">Story will be generated when analysis completes</p>
        )}
      </div>
    );
  }

  if (error === "not_found" && !title) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 text-on-surface-variant">
        {retryCount < 15 ? (
          <>
            {reportDone ? (
              <div className="h-7 w-7 rounded-full border-2 border-primary border-t-transparent animate-spin" />
            ) : (
              <div className="h-7 w-7 rounded-full border-2 border-primary border-t-transparent animate-spin" />
            )}
            <p className="text-sm font-body">{reportDone ? "Loading story report..." : "Generating story report..."}</p>
            {reportStep && (
              <div className="flex flex-col gap-1.5 mt-3 text-xs text-on-surface-variant font-body">
                {(reportStep.details || []).map((d, i) => (
                  <div key={i} className="flex items-center gap-2">
                    {d.status === "complete" ? (
                      <span className="material-symbols-outlined text-sm text-primary">check_circle</span>
                    ) : (
                      <div className="h-3.5 w-3.5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                    )}
                    <span>{d.label}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <>
            <span className="material-symbols-outlined text-4xl text-outline">auto_stories</span>
            <p className="text-sm font-body">No story available yet</p>
            <button
              onClick={() => {
                setRetryCount(0);
                setError(null);
                setLoading(true);
                getStory(sessionId, "json")
                  .then((d) => {
                    if (d) setStory(d);
                    else setError("not_found");
                  })
                  .catch(() => setError("not_found"));
              }}
              className="text-xs font-body text-primary hover:underline"
            >
              Retry
            </button>
          </>
        )}
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-on-surface-variant">
        <div className="h-6 w-6 rounded-full border-2 border-primary border-t-transparent animate-spin" />
      </div>
    );
  }

  const leadParagraph = summaryBlocks[0] || executiveSummary;
  const visibleTakeaways = takeaways.length > 0 ? takeaways : [];

  return (
    <div className="story-shell relative h-full overflow-y-auto">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-80 bg-gradient-to-b from-primary/10 via-transparent to-transparent" />
      <div className="pointer-events-none absolute right-[-6rem] top-24 h-72 w-72 rounded-full bg-tertiary/10 blur-3xl animate-soft-drift" />
      <div className="pointer-events-none absolute left-[-5rem] top-48 h-64 w-64 rounded-full bg-primary/10 blur-3xl animate-soft-drift" />

      <div className="sticky top-0 z-20 border-b border-outline-variant/60 bg-surface/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-2 sm:px-6 lg:px-10">
          <div className="flex items-center gap-2 text-xs text-on-surface-variant font-body">
            <span className="material-symbols-outlined text-base">article</span>
            <span>Story report</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative group">
              <button
                onClick={handleRegenerate}
                disabled={regenerating}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1 text-xs font-body transition-all disabled:opacity-50 ${
                  changesDetected
                    ? "bg-green-50 text-green-700 border border-green-300 hover:bg-green-100 shadow-[0_0_8px_rgba(34,197,94,0.3)]"
                    : "bg-surface-container text-on-surface-variant hover:bg-surface-container-high"
                }`}
                title={changesDetected ? "Notebook cells changed — regenerate to update story" : "Regenerate story from current notebook"}
              >
                {regenerating ? (
                  <div className="h-3.5 w-3.5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                ) : (
                  <span className="material-symbols-outlined text-sm">auto_fix_high</span>
                )}
                {regenerating ? "Regenerating..." : "Regenerate"}
              </button>
            </div>
            <button
              onClick={() => handleExport("pdf")}
              disabled={exportingPdf}
              className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1 text-xs font-body text-on-primary transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {exportingPdf ? (
                <div className="h-3.5 w-3.5 rounded-full border-2 border-on-primary border-t-transparent animate-spin" />
              ) : (
                <span className="material-symbols-outlined text-sm">picture_as_pdf</span>
              )}
              {exportingPdf ? "Generating PDF..." : "Export PDF"}
            </button>
            <button
              onClick={() => handleExport("md")}
              disabled={exportingMd}
              className="flex items-center gap-1.5 rounded-lg border border-outline-variant px-3 py-1 text-xs font-body text-on-surface-variant transition-colors hover:bg-surface-container disabled:opacity-50"
            >
              {exportingMd ? (
                <div className="h-3.5 w-3.5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
              ) : (
                <span className="material-symbols-outlined text-sm">share</span>
              )}
              {exportingMd ? "Generating..." : "Export Markdown"}
            </button>
          </div>
        </div>
      </div>

      <article className="relative mx-auto flex max-w-4xl flex-col gap-12 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <section className="story-lead animate-rise-in border-b border-outline-variant/18 pb-10">
          <div className="max-w-4xl">
            <div className="flex flex-wrap items-center gap-2 text-xs font-body text-on-surface-variant">
              <span className="story-kicker text-[10px] font-bold text-primary">Narrative report</span>
              {generatedAt && (
                <span>{formatGeneratedAt(generatedAt)}</span>
              )}
            </div>

            <h1 className="mt-4 text-4xl font-extrabold tracking-tight text-on-surface sm:text-5xl">
              {title || "EDA Report"}
            </h1>

            {leadParagraph && (
              <ReportMarkdown className="mt-5 text-[15px] leading-7 text-on-surface-variant">
                {leadParagraph}
              </ReportMarkdown>
            )}
          </div>

          {visibleTakeaways.length > 0 && (
            <section className="mt-8 -mx-4 px-4 overflow-x-auto">
              <div className="flex items-center gap-2 mb-4">
                <span className="material-symbols-outlined text-primary">summarize</span>
                <h2 className="text-lg font-bold tracking-tight text-on-surface">
                  Key Takeaways
                </h2>
              </div>
              <div className="flex gap-4 pb-4" style={{ minWidth: "min-content" }}>
                {visibleTakeaways.map((takeaway, index) => (
                  <article
                    key={takeaway.id}
                    className="w-72 shrink-0 rounded-2xl border border-outline-variant/20 bg-white/80 backdrop-blur px-5 py-4 shadow-sm"
                    style={{ "--story-delay": `${index * 70}ms` } as CSSProperties}
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-surface-container">
                        <span className="material-symbols-outlined text-[18px] text-primary">
                          {takeaway.icon || iconForText(takeaway.body)}
                        </span>
                      </div>
                      <div className="min-w-0 flex-1">
                        <span className="story-kicker text-[10px] font-bold text-primary">
                          Takeaway {String(takeaway.index).padStart(2, "0")}
                        </span>
                        <h3 className="mt-0.5 text-sm font-semibold text-on-surface leading-snug">
                          {takeaway.title}
                        </h3>
                      </div>
                    </div>
                    <ul className="mt-3 space-y-2 text-[13px] leading-6 text-on-surface-variant">
                      {takeaway.points.map((point, pointIndex) => (
                        <li key={pointIndex} className="flex gap-2">
                          <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary/70" />
                          <span className="min-w-0"><InlineMd>{point}</InlineMd></span>
                        </li>
                      ))}
                    </ul>
                  </article>
                ))}
              </div>
            </section>
          )}
        </section>

        {sectionData.length > 0 && (
          <section className="space-y-8">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary">insights</span>
              <h2 className="text-lg font-bold tracking-tight text-on-surface">
                Analysis
              </h2>
            </div>
            <div className="space-y-0 border-t border-outline-variant/20 pt-2">
              {sectionData.map(({ section, plots, plotCellIds }, index) => (
                <div key={`${section.phase}-${index}`}>
                  {index > 0 && (
                    <div className="flex justify-center py-2">
                      <div className="h-8 w-px bg-gradient-to-b from-outline-variant/30 to-transparent" />
                    </div>
                  )}
                  <div data-animate className="opacity-0">
                    <StorySectionCard
                      section={section}
                      plots={plots}
                      plotCellIds={plotCellIds}
                      onOpenCell={openNotebookCell}
                      index={index}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <footer className="pt-6 pb-4 text-xs text-outline">
          <div className="flex flex-wrap items-center gap-2">
            <span className="material-symbols-outlined text-base">smart_toy</span>
            <span>Generated by AgenticEDA</span>
          </div>
        </footer>
      </article>
    </div>
  );
}
