"use client";

import { useNotebookStore, type NotebookData } from "@/stores/notebookStore";

export default function NotebookTabs() {
  const notebooks = useNotebookStore((s) => s.notebooks);
  const activeId = useNotebookStore((s) => s.activeNotebookId);
  const setActive = useNotebookStore((s) => s.setActiveNotebook);

  const tabs: NotebookData[] = Object.values(notebooks);
  if (tabs.length <= 1) return null;

  return (
    <div className="flex items-center gap-1 px-3 py-1.5 border-b border-outline-variant/30 bg-surface-container-low shrink-0 min-h-[40px] overflow-x-auto">
      {tabs.map((nb) => {
        const isActive = nb.id === activeId;
        const isMain = nb.id === "main";
        return (
          <button
            key={nb.id}
            onClick={() => setActive(nb.id)}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium whitespace-nowrap transition-colors font-body ${
              isActive
                ? "bg-primary/10 text-primary"
                : "text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface"
            }`}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
              {isMain ? "description" : "science"}
            </span>
            {(() => {
              const agentMatch = nb.id.match(/^agent_(\d+)$/);
              const label = nb.id === "main" ? "Main" : agentMatch ? `Agent ${agentMatch[1]}` : nb.title;
              return (
                <>
                  <span className="max-w-[180px] truncate">
                    {label}
                  </span>
                  {agentMatch && nb.title && (
                    <span className="max-w-[120px] truncate text-[10px] text-on-surface-variant opacity-70">
                      {nb.title}
                    </span>
                  )}
                </>
              );
            })()}
            {!isMain && nb.status !== "idle" && (
              <span
                className="material-symbols-outlined"
                style={{ fontSize: "14px" }}
              >
                {nb.status === "running"
                  ? "schedule"
                  : nb.status === "complete"
                  ? "check_circle"
                  : nb.status === "failed"
                  ? "error"
                  : "timer_off"}
              </span>
            )}
            {!isMain && (
              <span className="text-[10px] text-on-surface-variant">
                ({nb.cells.length})
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
