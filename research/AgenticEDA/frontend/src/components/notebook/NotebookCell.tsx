"use client";

import React, { useState, useCallback, useEffect, useRef } from "react";
import type { Cell } from "@/lib/types";
import { useNotebookStore } from "@/stores/notebookStore";
import CellToolbar from "./CellToolbar";
import CodeEditor from "./CodeEditor";
import CellOutput from "./CellOutput";
import ThinkingBlock from "./ThinkingBlock";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

interface Props {
  cell: Cell;
  isActive: boolean;
  onFocus: () => void;
  onRun: (cellId: string, source: string) => void;
}

function NotebookCellInner({ cell, isActive, onFocus, onRun }: Props) {
  const { updateCellSource, deleteCell, moveCell } = useNotebookStore();
  const fixedCellIds = useNotebookStore((s) => s.fixedCellIds);
  const [editingMarkdown, setEditingMarkdown] = useState(false);
  const [showFixed, setShowFixed] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const prevFixedRef = useRef(false);

  // Show "Fixed" badge when this cell appears in fixedCellIds
  const isFixed = fixedCellIds.has(cell.id);
  useEffect(() => {
    let showHandle: number | null = null;
    let hideHandle: number | null = null;

    if (isFixed && !prevFixedRef.current) {
      showHandle = window.requestAnimationFrame(() => {
        setShowFixed(true);
      });
      hideHandle = window.setTimeout(() => setShowFixed(false), 2000);
    }
    prevFixedRef.current = isFixed;

    return () => {
      if (showHandle !== null) {
        window.cancelAnimationFrame(showHandle);
      }
      if (hideHandle !== null) {
        window.clearTimeout(hideHandle);
      }
    };
  }, [isFixed]);

  const handleRun = useCallback(() => {
    onRun(cell.id, cell.source);
  }, [cell.id, cell.source, onRun]);

  const handleDelete = useCallback(() => {
    if (isDeleting) return;
    setIsDeleting(true);
    window.setTimeout(() => deleteCell(cell.id), 180);
  }, [cell.id, deleteCell, isDeleting]);

  const execLabel = cell.executing
    ? "[*]:"
    : cell.execution_count !== null
    ? `[${cell.execution_count}]:`
    : "[ ]:";

  const wrapperClass = cell.cell_type === "markdown"
    ? "border border-outline-variant/10 rounded-2xl overflow-hidden shadow-[0_1px_0_rgba(11,28,48,0.03)]"
    : cell.error
    ? "rounded-2xl overflow-hidden code-shadow ring-2 ring-error/30"
    : cell.executing
    ? "rounded-2xl overflow-hidden code-shadow ring-2 ring-primary/30"
    : "rounded-2xl overflow-hidden code-shadow border border-outline-variant/10";

  return (
    <>
      {cell.thinking && <ThinkingBlock content={cell.thinking} />}
      <div
        id={`cell-${cell.id}`}
        className={`group relative mb-4 transition-all duration-300 ${wrapperClass} ${showFixed ? "animate-cell-resolve" : ""} ${isDeleting ? "animate-cell-delete pointer-events-none opacity-0 scale-[0.985]" : ""}`}
        onClick={onFocus}
      >
        {/* Executing spinner overlay */}
        {cell.executing && (
          <div className="absolute top-2 right-2 z-10">
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin shadow-[0_0_0_4px_rgba(37,99,235,0.08)]" />
          </div>
        )}

        {/* Fixed badge */}
        {showFixed && (
          <div className="absolute top-2 right-2 z-10 animate-fade-out">
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-500/15 text-green-700 border border-green-500/25 shadow-sm backdrop-blur-sm">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Fixed
            </span>
          </div>
        )}

        {/* Toolbar: visible on hover or when active */}
        <div className={`${isActive ? "opacity-100" : "opacity-0 group-hover:opacity-100"} transition-opacity duration-200`}>
          <CellToolbar
            cellType={cell.cell_type}
            onRun={handleRun}
            onDelete={handleDelete}
            onMoveUp={() => moveCell(cell.id, "up")}
            onMoveDown={() => moveCell(cell.id, "down")}
          />
        </div>

        {/* Cell body */}
        {cell.cell_type === "code" ? (
          <div className="flex">
            {/* Execution label gutter */}
            <div className="flex-shrink-0 w-12 bg-gradient-to-b from-surface-container-highest to-surface-container-high pt-3 flex justify-center select-none">
              <span className="bg-surface-container px-2 py-0.5 rounded-md text-[10px] font-mono text-outline h-fit shadow-[0_1px_0_rgba(11,28,48,0.04)]">
                {execLabel}
              </span>
            </div>
            {/* Code area */}
            <div className="flex-grow min-w-0 bg-surface-container-low">
              <CodeEditor
                cellId={cell.id}
                value={cell.source}
                onChange={updateCellSource}
                onRun={handleRun}
              />
            </div>
          </div>
        ) : (
          /* Markdown cell */
          <div className="px-4 py-3 bg-surface/95">
            {editingMarkdown ? (
              <textarea
                className="w-full min-h-[60px] p-3 text-sm font-mono border border-outline-variant/20 rounded-xl bg-surface-container-low text-on-surface resize-y focus:outline-none focus:ring-2 focus:ring-primary/20"
                value={cell.source}
                onChange={(e) => updateCellSource(cell.id, e.target.value)}
                onBlur={() => setEditingMarkdown(false)}
                autoFocus
              />
            ) : (
              <div
                className="prose prose-sm max-w-none cursor-text min-h-[40px] text-on-surface prose-headings:text-on-surface prose-h1:text-xl prose-h1:font-headline prose-h1:font-bold prose-h1:border-b prose-h1:border-outline-variant/20 prose-h1:pb-2 prose-h2:text-lg prose-h2:font-headline prose-h2:font-semibold prose-h3:text-base prose-h3:font-headline prose-h3:font-semibold prose-hr:my-4 prose-hr:border-outline-variant/20 prose-blockquote:border-l-primary prose-blockquote:bg-primary-container/20 prose-blockquote:py-1 prose-blockquote:px-3 prose-blockquote:rounded-r prose-table:text-xs"
                onDoubleClick={() => setEditingMarkdown(true)}
              >
                {cell.source ? (
                  <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                    {cell.source.replace(/\\n/g, "\n")}
                  </ReactMarkdown>
                ) : (
                  <p className="text-on-surface-variant italic">Double-click to edit markdown</p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Output area (code cells only) */}
        {cell.cell_type === "code" && <CellOutput outputs={cell.outputs} />}
      </div>
    </>
  );
}

export default React.memo(NotebookCellInner, (prev, next) => {
  return (
    prev.cell.id === next.cell.id &&
    prev.cell.source === next.cell.source &&
    prev.cell.outputs === next.cell.outputs &&
    prev.cell.executing === next.cell.executing &&
    prev.cell.error === next.cell.error &&
    prev.isActive === next.isActive
  );
});
