"use client";

import { useState } from "react";

interface ThinkingBlockProps {
  content: string;
}

export default function ThinkingBlock({ content }: ThinkingBlockProps) {
  const [expanded, setExpanded] = useState(true);

  if (!content) return null;

  // Parse progress info from [N/M] prefix
  const progressMatch = content.match(/^\[(\d+)\/(\d+)\]\s*(.*)/);
  const step = progressMatch ? parseInt(progressMatch[1]) : null;
  const total = progressMatch ? parseInt(progressMatch[2]) : null;
  const label = progressMatch ? progressMatch[3] : content;
  const pct = step && total ? Math.round((step / total) * 100) : null;

  return (
    <div className="mx-4 my-2 overflow-hidden rounded-2xl border border-primary/10 bg-gradient-to-r from-primary/10 via-primary/5 to-transparent shadow-[0_1px_0_rgba(37,99,235,0.04)] animate-thought-rise">
      <button
        type="button"
        className="w-full px-3 py-2.5 text-left cursor-pointer"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        <div className="flex items-center gap-2">
          <span className="relative flex h-5 w-5 items-center justify-center">
            <span className="absolute inline-flex h-full w-full rounded-full bg-primary/20 animate-thought-pulse" />
            <span className="material-symbols-outlined relative text-primary text-[18px]">psychology</span>
          </span>
          <span className="text-xs text-primary font-body font-semibold tracking-wide uppercase">
            {pct !== null ? "Accumulating results" : "Model thinking"}
          </span>
          {pct !== null && (
            <span className="text-xs text-primary/70 font-mono font-medium">{step}/{total}</span>
          )}
          <span className="ml-auto flex items-center gap-1 text-primary/60">
            <span className={`h-1.5 w-1.5 rounded-full bg-primary animate-thought-pulse`} />
            <span className={`h-1.5 w-1.5 rounded-full bg-primary/70 animate-thought-pulse [animation-delay:150ms]`} />
            <span className={`h-1.5 w-1.5 rounded-full bg-primary/40 animate-thought-pulse [animation-delay:300ms]`} />
            <span className={`material-symbols-outlined text-[16px] transition-transform duration-300 ${expanded ? "rotate-180" : ""}`}>
              expand_more
            </span>
          </span>
        </div>
        {pct !== null && (
          <div className="mt-2 h-1.5 w-full rounded-full bg-primary/10 overflow-hidden">
            <div
              className="h-full rounded-full bg-primary transition-all duration-500 ease-out"
              style={{ width: `${pct}%` }}
            />
          </div>
        )}
      </button>

      <div
        className={`grid transition-all duration-300 ease-out ${expanded ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"}`}
      >
        <div className="overflow-hidden px-3 pb-3">
          <div className="rounded-xl border border-primary/10 bg-surface-container-low/80 px-3 py-2">
            <p className="text-sm text-on-surface-variant leading-relaxed font-body italic animate-fade-in-up">
              {label}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
