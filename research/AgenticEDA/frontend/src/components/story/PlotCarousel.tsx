"use client";

import { useState, useEffect, useCallback } from "react";
import type { StoryPlot, StorySection } from "@/lib/types";
import StoryPlotRenderer from "./StoryPlotRenderer";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

interface PlotCarouselProps {
  plots: Array<StoryPlot | string>;
  section: Pick<StorySection, "title" | "visual_title" | "visual_caption">;
  plotCellIds?: string[];
  onOpenCell?: (cellId: string) => void;
}

export default function PlotCarousel({
  plots,
  section,
  plotCellIds = [],
  onOpenCell,
}: PlotCarouselProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [lightboxOpen, setLightboxOpen] = useState(false);

  const count = plots.length;

  const prev = useCallback(() => {
    setActiveIndex((i) => (i - 1 + count) % count);
  }, [count]);

  const next = useCallback(() => {
    setActiveIndex((i) => (i + 1) % count);
  }, [count]);

  useEffect(() => {
    if (count <= 1) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") prev();
      else if (e.key === "ArrowRight") next();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [count, prev, next]);

  // Single plot — render inline with no carousel chrome
  if (count === 1) {
    return (
      <StoryPlotRenderer
        plot={plots[0]}
        section={section}
        index={0}
        plotCellId={plotCellIds[0]}
      />
    );
  }

  const activePlot = plots[activeIndex];
  const activePlotObj = typeof activePlot === "string" ? null : activePlot;
  const sourceCellId =
    activePlotObj?.source_cell_id || plotCellIds[activeIndex] || "";
  const canOpenCell = Boolean(sourceCellId && onOpenCell);

  const caption =
    activePlotObj?.caption ||
    section.visual_caption ||
    "";
  const title =
    activePlotObj?.title ||
    section.visual_title ||
    section.title ||
    "";

  return (
    <div className="relative w-full select-none">
      {/* Slide track */}
      <div className="relative overflow-hidden rounded-lg border border-outline-variant/10 bg-surface-container-lowest shadow-[0_4px_12px_rgba(11,28,48,0.04)]">
        {/* Slides container */}
        <div
          className="flex"
          style={{
            transform: `translateX(-${activeIndex * 100}%)`,
            transition: "transform 400ms ease",
          }}
        >
          {plots.map((plot, i) => {
            const plotObj = typeof plot === "string" ? null : plot;
            const base =
              plotCellIds[i] ||
              plotObj?.source_cell_id ||
              plotObj?.source_path ||
              plotObj?.title ||
              `carousel-${section.title}`;
            const key = `${base}-${i}`;
            return (
              <div key={key} className="relative min-w-full max-w-full overflow-hidden">
                <StoryPlotRenderer
                  plot={plot}
                  section={section}
                  index={i}
                  plotCellId={plotCellIds[i]}
                  className="h-full w-full"
                />
                {/* Expand overlay — click to open lightbox */}
                <button
                  className="absolute inset-0 z-10 cursor-zoom-in opacity-0 hover:opacity-100 transition-opacity flex items-end justify-end p-3"
                  onClick={() => setLightboxOpen(true)}
                  tabIndex={i === activeIndex ? 0 : -1}
                  aria-label={`View plot ${i + 1} of ${count} in full screen`}
                >
                  <span className="flex items-center gap-1.5 rounded-full bg-black/60 px-3 py-1.5 text-[11px] font-medium text-white backdrop-blur-sm shadow-lg">
                    <span className="material-symbols-outlined text-[14px]">open_in_full</span>
                    Expand
                  </span>
                </button>
              </div>
            );
          })}
        </div>

        {/* Left chevron */}
        <button
          onClick={(e) => { e.stopPropagation(); prev(); }}
          aria-label="Previous plot"
          className="absolute left-3 top-1/2 z-10 -translate-y-1/2 rounded-full border border-outline-variant/30 bg-surface-container-lowest/80 p-1.5 text-on-surface shadow-sm backdrop-blur-sm transition-colors hover:bg-surface-container hover:border-outline-variant/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
        >
          <span className="material-symbols-outlined text-[20px] leading-none">
            chevron_left
          </span>
        </button>

        {/* Right chevron */}
        <button
          onClick={(e) => { e.stopPropagation(); next(); }}
          aria-label="Next plot"
          className="absolute right-3 top-1/2 z-10 -translate-y-1/2 rounded-full border border-outline-variant/30 bg-surface-container-lowest/80 p-1.5 text-on-surface shadow-sm backdrop-blur-sm transition-colors hover:bg-surface-container hover:border-outline-variant/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
        >
          <span className="material-symbols-outlined text-[20px] leading-none">
            chevron_right
          </span>
        </button>
      </div>

      {/* Dot indicators */}
      <div className="mt-3 flex items-center justify-center gap-1.5">
        {plots.map((_, i) => (
          <button
            key={i}
            onClick={() => setActiveIndex(i)}
            aria-label={`Go to plot ${i + 1}`}
            className={`transition-all duration-200 rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 ${
              i === activeIndex
                ? "h-2 w-5 bg-primary"
                : "h-2 w-2 border border-outline-variant bg-transparent hover:bg-outline-variant/40"
            }`}
          />
        ))}
      </div>

      {/* Caption / title row */}
      {(title || caption || canOpenCell) && (
        <div className="mt-3 flex items-start justify-between gap-4 px-1">
          <div className="min-w-0">
            {title && (
              <div className="text-sm font-semibold tracking-tight text-on-surface [&_p]:m-0">
                <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                  {title}
                </ReactMarkdown>
              </div>
            )}
            {caption && (
              <div className="mt-0.5 text-[13px] italic leading-5 text-on-surface-variant [&_p]:m-0">
                <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                  {caption}
                </ReactMarkdown>
              </div>
            )}
          </div>
          {canOpenCell && (
            <button
              onClick={() => onOpenCell?.(sourceCellId)}
              aria-label={`Open notebook source for ${title}`}
              className="shrink-0 inline-flex items-center gap-1.5 rounded-full border border-outline-variant/30 bg-white/75 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-primary transition-colors hover:border-outline-variant/50 hover:bg-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
            >
              Open source
              <span className="material-symbols-outlined text-[14px]">open_in_new</span>
            </button>
          )}
        </div>
      )}

      {/* Lightbox rendered inline to avoid import cycle */}
      {lightboxOpen && (
        <PlotLightboxInline
          plots={plots}
          section={section}
          plotCellIds={plotCellIds}
          initialIndex={activeIndex}
          onClose={() => setLightboxOpen(false)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline lightbox sub-component (avoids a circular import with PlotLightbox)
// ---------------------------------------------------------------------------

interface InlineLightboxProps {
  plots: Array<StoryPlot | string>;
  section: Pick<StorySection, "title" | "visual_title" | "visual_caption">;
  plotCellIds?: string[];
  initialIndex: number;
  onClose: () => void;
}

function PlotLightboxInline({
  plots,
  section,
  plotCellIds = [],
  initialIndex,
  onClose,
}: InlineLightboxProps) {
  const [index, setIndex] = useState(initialIndex);
  const count = plots.length;

  const prev = useCallback(() => setIndex((i) => (i - 1 + count) % count), [count]);
  const next = useCallback(() => setIndex((i) => (i + 1) % count), [count]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft") prev();
      else if (e.key === "ArrowRight") next();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose, prev, next]);

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center animate-agent-modal-in"
      style={{ animation: "agent-modal-in 200ms ease-out both" }}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-md"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Dialog */}
      <div
        className="relative z-10 flex w-full max-w-[1200px] flex-col px-4"
        role="dialog"
        aria-modal="true"
        aria-label="Plot lightbox"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <div className="mb-3 flex justify-end">
          <button
            onClick={onClose}
            aria-label="Close lightbox"
            className="text-white hover:bg-white/20 rounded-full p-2 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40"
          >
            <span className="material-symbols-outlined text-[24px] leading-none">close</span>
          </button>
        </div>

        {/* Plot */}
        <div className="relative flex items-center gap-3">
          {count > 1 && (
            <button
              onClick={prev}
              aria-label="Previous plot"
              className="shrink-0 bg-white/20 hover:bg-white/30 rounded-full p-3 text-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40"
            >
              <span className="material-symbols-outlined text-[24px] leading-none">chevron_left</span>
            </button>
          )}

          <div className="min-w-0 flex-1 rounded-2xl overflow-hidden border border-white/10 bg-black/30">
            <StoryPlotRenderer
              plot={plots[index]}
              section={section}
              index={index}
              plotCellId={plotCellIds[index]}
              className="w-full h-full"
            />
          </div>

          {count > 1 && (
            <button
              onClick={next}
              aria-label="Next plot"
              className="shrink-0 bg-white/20 hover:bg-white/30 rounded-full p-3 text-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40"
            >
              <span className="material-symbols-outlined text-[24px] leading-none">chevron_right</span>
            </button>
          )}
        </div>

        {/* Counter */}
        {count > 1 && (
          <div className="mt-3 flex justify-center">
            <span className="text-white/80 text-sm">
              {index + 1} of {count}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
