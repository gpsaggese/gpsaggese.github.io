"use client";

import { useState } from "react";
import { API_BASE } from "@/lib/backend";

interface Props {
  sessionId: string;
}

export default function ExportDialog({ sessionId }: Props) {
  const [open, setOpen] = useState(false);

  const downloadPdf = async () => {
    const res = await fetch(
      `${API_BASE}/story/${sessionId}?format=pdf`
    );
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `story_${sessionId}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
    setOpen(false);
  };

  const downloadMarkdown = async () => {
    const res = await fetch(
      `${API_BASE}/story/${sessionId}?format=md`
    );
    if (!res.ok) return;
    const data = await res.json();
    const text = data.markdown || data.content || "";
    const blob = new Blob([text], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `story_${sessionId}.md`;
    a.click();
    URL.revokeObjectURL(url);
    setOpen(false);
  };

  return (
    <div className="relative inline-block">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 px-4 py-2 text-sm rounded-xl bg-primary text-on-primary hover:opacity-90 transition-opacity font-body"
      >
        <span className="material-symbols-outlined text-base">download</span>
        Export
      </button>

      {open && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setOpen(false)}
          />
          <div className="absolute right-0 bottom-full mb-2 z-50 bg-surface-container-lowest border border-outline-variant rounded-xl shadow-lg p-1.5 min-w-[200px] code-shadow">
            <button
              onClick={downloadPdf}
              className="w-full flex items-center gap-2.5 text-left px-3 py-2.5 text-sm text-on-surface hover:bg-surface-container-low rounded-lg transition-colors font-body"
            >
              <span className="material-symbols-outlined text-base text-on-surface-variant">picture_as_pdf</span>
              Download PDF
            </button>
            <button
              onClick={downloadMarkdown}
              className="w-full flex items-center gap-2.5 text-left px-3 py-2.5 text-sm text-on-surface hover:bg-surface-container-low rounded-lg transition-colors font-body"
            >
              <span className="material-symbols-outlined text-base text-on-surface-variant">markdown</span>
              Download Markdown
            </button>
          </div>
        </>
      )}
    </div>
  );
}
