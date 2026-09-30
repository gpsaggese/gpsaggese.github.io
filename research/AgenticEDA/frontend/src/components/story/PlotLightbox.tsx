"use client";

import { useState, useEffect, useCallback } from "react";
import type { StoryPlot, StorySection } from "@/lib/types";
import StoryPlotRenderer from "./StoryPlotRenderer";

interface PlotLightboxProps {
  plots: Array<StoryPlot | string>;
  section: Pick<StorySection, "title" | "visual_title" | "visual_caption">;
  plotCellIds?: string[];
  initialIndex: number;
  onClose: () => void;
}

export default function PlotLightbox({
  plots,
  section,
  plotCellIds = [],
  initialIndex,
  onClose,
}: PlotLightboxProps) {
  const [index, setIndex] = useState(initialIndex);
  const [visible, setVisible] = useState(false);
  const count = plots.length;

  // Trigger fade-in after mount
  useEffect(() => {
    const id = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(id);
  }, []);

  const handleClose = useCallback(() => {
    setVisible(false);
    // Wait for fade-out transition before calling onClose
    const id = setTimeout(onClose, 200);
    return () => clearTimeout(id);
  }, [onClose]);

  const prev = useCallback(() => setIndex((i) => (i - 1 + count) % count), [count]);
  const next = useCallback(() => setIndex((i) => (i + 1) % count), [count]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleClose();
      else if (e.key === "ArrowLeft") prev();
      else if (e.key === "ArrowRight") next();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleClose, prev, next]);

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "scale(1)" : "scale(0.97)",
        transition: "opacity 200ms ease, transform 200ms ease",
      }}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-md"
        onClick={handleClose}
        aria-hidden="true"
      />

      {/* Dialog container */}
      <div
        className="relative z-10 flex w-full max-w-[1200px] flex-col px-4"
        role="dialog"
        aria-modal="true"
        aria-label="Plot lightbox"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top bar: close button */}
        <div className="mb-3 flex items-center justify-between">
          <div />
          <button
            onClick={handleClose}
            aria-label="Close lightbox"
            className="text-white hover:bg-white/20 rounded-full p-2 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40"
          >
            <span className="material-symbols-outlined text-[24px] leading-none">
              close
            </span>
          </button>
        </div>

        {/* Plot row with optional navigation arrows */}
        <div className="relative flex items-center gap-3">
          {count > 1 && (
            <button
              onClick={prev}
              aria-label="Previous plot"
              className="shrink-0 bg-white/20 hover:bg-white/30 rounded-full p-3 text-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40"
            >
              <span className="material-symbols-outlined text-[24px] leading-none">
                chevron_left
              </span>
            </button>
          )}

          {/* Plot frame */}
          <div
            className="min-w-0 flex-1 overflow-hidden rounded-2xl border border-white/10 bg-black/30"
            style={{ maxWidth: "100%" }}
          >
            {/* Slide track for smooth transition between plots */}
            <div
              className="flex"
              style={{
                transform: `translateX(-${index * 100}%)`,
                transition: "transform 400ms ease",
              }}
            >
              {plots.map((plot, i) => {
                const plotObj = typeof plot === "string" ? null : plot;
                const key =
                  plotCellIds[i] ||
                  plotObj?.source_cell_id ||
                  plotObj?.source_path ||
                  plotObj?.title ||
                  `lightbox-${section.title}-${i}`;
                return (
                  <div key={key} className="min-w-full">
                    <StoryPlotRenderer
                      plot={plot}
                      section={section}
                      index={i}
                      plotCellId={plotCellIds[i]}
                      className="w-full h-full"
                    />
                  </div>
                );
              })}
            </div>
          </div>

          {count > 1 && (
            <button
              onClick={next}
              aria-label="Next plot"
              className="shrink-0 bg-white/20 hover:bg-white/30 rounded-full p-3 text-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40"
            >
              <span className="material-symbols-outlined text-[24px] leading-none">
                chevron_right
              </span>
            </button>
          )}
        </div>

        {/* Bottom bar: counter + dot indicators */}
        {count > 1 && (
          <div className="mt-3 flex flex-col items-center gap-2">
            {/* Dot indicators */}
            <div className="flex items-center gap-1.5">
              {plots.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setIndex(i)}
                  aria-label={`Go to plot ${i + 1}`}
                  className={`transition-all duration-200 rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40 ${
                    i === index
                      ? "h-2 w-5 bg-white"
                      : "h-2 w-2 border border-white/50 bg-transparent hover:bg-white/30"
                  }`}
                />
              ))}
            </div>

            {/* Counter label */}
            <span className="text-white/80 text-sm">
              {index + 1} of {count}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
