"use client";

import { useState } from "react";
import AboutPane from "./AboutPane";

export default function AboutLauncher() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="fixed right-5 top-5 z-20 inline-flex items-center gap-2 rounded-lg border border-outline-variant bg-white px-3 py-2 text-sm font-semibold text-on-surface shadow-sm transition-colors hover:border-primary hover:text-primary"
      >
        <span className="material-symbols-outlined text-[18px]">info</span>
        About
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/30 backdrop-blur-sm">
          <button
            type="button"
            aria-label="Close About panel"
            className="absolute inset-0 cursor-default"
            onClick={() => setOpen(false)}
          />
          <aside className="relative flex h-full w-full max-w-4xl flex-col border-l border-outline-variant bg-surface shadow-2xl">
            <div className="flex h-14 shrink-0 items-center justify-between border-b border-outline-variant/60 bg-white px-5">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-primary">info</span>
                <span className="text-sm font-bold text-on-surface font-headline">About AgenticEDA</span>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-lg p-2 text-on-surface-variant transition-colors hover:bg-surface-container hover:text-on-surface"
                aria-label="Close About panel"
              >
                <span className="material-symbols-outlined text-[20px]">close</span>
              </button>
            </div>
            <AboutPane compact />
          </aside>
        </div>
      )}
    </>
  );
}
