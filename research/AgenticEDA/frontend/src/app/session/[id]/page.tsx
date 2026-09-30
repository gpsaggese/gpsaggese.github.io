"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import NotebookPane from "@/components/notebook/NotebookPane";
import StoryPane from "@/components/story/StoryPane";
import HistoryPanel from "@/components/history/HistoryPanel";
import ChatSidebar from "@/components/chat/ChatSidebar";
import AboutPane from "@/components/about/AboutPane";
import AgentActivityBadge from "../../../components/layout/AgentActivityBadge";
import { useNotebookStore } from "@/stores/notebookStore";
import { useStoryStore } from "@/stores/storyStore";
import { useChatStore } from "@/stores/chatStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useAgentStream } from "@/hooks/useAgentStream";
import { getNotebook, getStory, listSessions } from "@/lib/api";

/** Circular progress indicator estimating LLM context window usage. */
function ContextWindowIndicator({ cellCount, findingsCount }: { cellCount: number; findingsCount: number }) {
  // Estimate: ~500 tokens per cell, ~200 per finding, 128k context window
  const maxTokens = 128_000;
  const estimatedUsed = cellCount * 500 + findingsCount * 200 + 2000; // 2k base
  const ratio = Math.min(1, estimatedUsed / maxTokens);
  const pct = Math.round(ratio * 100);

  const radius = 12;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - ratio);

  const color = pct > 80 ? "#ef4444" : pct > 50 ? "#f59e0b" : "#6366f1";

  return (
    <div className="flex items-center gap-2 px-2 py-1.5" title={`~${Math.round(estimatedUsed / 1000)}k / ${maxTokens / 1000}k tokens`}>
      <svg width="28" height="28" viewBox="0 0 28 28">
        <circle cx="14" cy="14" r={radius} fill="none" stroke="#e5e7eb" strokeWidth="3" />
        <circle
          cx="14" cy="14" r={radius}
          fill="none" stroke={color} strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          transform="rotate(-90 14 14)"
          style={{ transition: "stroke-dashoffset 0.5s ease" }}
        />
      </svg>
      <div className="flex flex-col">
        <span className="text-[9px] font-medium text-on-surface">{pct}% context</span>
        <span className="text-[8px] text-on-surface-variant">~{Math.round(estimatedUsed / 1000)}k tokens</span>
      </div>
    </div>
  );
}
import type { Cell, CellOutput, Session } from "@/lib/types";

type SessionTab = "notebook" | "story" | "history" | "about";

function normalizeText(value: unknown): string {
  if (Array.isArray(value)) return value.join("");
  return typeof value === "string" ? value : "";
}

function normalizeOutputs(value: unknown): CellOutput[] {
  if (!Array.isArray(value)) return [];
  return value.map((output) => {
    const entry = (output ?? {}) as Record<string, unknown>;
    return {
      output_type: (entry.output_type as CellOutput["output_type"]) ?? "stream",
      text: typeof entry.text === "string" ? entry.text : undefined,
      data: entry.data && typeof entry.data === "object" ? (entry.data as Record<string, string>) : undefined,
      metadata:
        entry.metadata && typeof entry.metadata === "object"
          ? (entry.metadata as Record<string, unknown>)
          : undefined,
      ename: typeof entry.ename === "string" ? entry.ename : undefined,
      evalue: typeof entry.evalue === "string" ? entry.evalue : undefined,
      traceback: Array.isArray(entry.traceback)
        ? entry.traceback.flatMap((line) => (typeof line === "string" ? [line] : []))
        : undefined,
    };
  });
}

function normalizeNotebookCells(payload: Record<string, unknown> | null): Cell[] {
  const cells = Array.isArray(payload?.cells) ? payload.cells : [];
  return cells.map((rawCell, index) => {
    const entry = (rawCell ?? {}) as Record<string, unknown>;
    const cellType = entry.cell_type === "markdown" ? "markdown" : "code";
    return {
      id: typeof entry.id === "string" ? entry.id : `nb-${index + 1}`,
      cell_type: cellType,
      source: normalizeText(entry.source),
      outputs: normalizeOutputs(entry.outputs),
      execution_count:
        typeof entry.execution_count === "number" ? entry.execution_count : null,
      metadata:
        entry.metadata && typeof entry.metadata === "object"
          ? (entry.metadata as Record<string, unknown>)
          : undefined,
      executing: false,
      error: null,
    };
  });
}

function scrollToNotebookCell(cellId?: string) {
  if (!cellId) return;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const el = document.getElementById(`cell-${cellId}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        el.classList.add("ring-2", "ring-primary/30", "ring-offset-2");
        window.setTimeout(() => {
          el.classList.remove("ring-2", "ring-primary/30", "ring-offset-2");
        }, 1400);
      }
    });
  });
}

export default function SessionPage() {
  const params = useParams<{ id: string | string[] }>();
  const router = useRouter();
  const sessionId = Array.isArray(params.id) ? params.id[0] : params.id;
  const [activeTab, setActiveTabState] = useState<SessionTab>("notebook");
  const [explorerOpen, setExplorerOpen] = useState(true);
  const [chatOpen, setChatOpen] = useState(true);
  const contentPaneRef = useRef<HTMLElement | null>(null);
  const scrollPositionsRef = useRef<Record<string, number>>({});
  const cellCount = useNotebookStore((s) => s.cells.length);
  const hypothesisGroups = useNotebookStore((s) => s.hypothesisGroups);
  const activeNotebookId = useNotebookStore((s) => s.activeNotebookId);
  const notebooks = useNotebookStore((s) => s.notebooks);
  const activityLog = useNotebookStore((s) => s.activityLog);
  const pipelineRunning = useNotebookStore((s) => s.pipelineRunning);
  const currentPhase = useNotebookStore((s) => s.currentPhase);
  const storyTitle = useStoryStore((s) => s.title);
  const storySections = useStoryStore((s) => s.sections);
  const setActiveSession = useSessionStore((s) => s.setActiveSession);
  const sessions = useSessionStore((s) => s.sessions);
  const setSessions = useSessionStore((s) => s.setSessions);
  const sessionMeta =
    sessions.find((session: Session) => session.session_id === sessionId) ?? null;

  // Don't override user's tab choice — just use activeTab directly
  const resolvedTab = activeTab;

  const findScrollContainer = useCallback((tab: string): HTMLElement | null => {
    const root = contentPaneRef.current;
    if (!root) return null;
    if (tab === "notebook") return root.querySelector<HTMLElement>("[data-notebook-scroll]");
    if (tab === "story") return root.querySelector<HTMLElement>(".story-shell");
    if (tab === "history") return root.querySelector<HTMLElement>(".history-scroll-pane");
    if (tab === "about") return root.querySelector<HTMLElement>(".about-scroll-pane");
    return null;
  }, []);

  const setActiveTab = useCallback((tab: SessionTab) => {
    setActiveTabState((prev) => {
      const container = findScrollContainer(prev);
      if (container) {
        scrollPositionsRef.current[prev] = container.scrollTop;
        try { sessionStorage.setItem(`scroll_${sessionId}_${prev}`, String(container.scrollTop)); } catch {}
      }
      return tab;
    });
    useNotebookStore.getState().setActiveTab(tab);
  }, [findScrollContainer, sessionId]);

  useEffect(() => {
    // Restore scroll positions from sessionStorage on mount
    for (const tab of ["notebook", "story", "history", "about"]) {
      try {
        const saved = sessionStorage.getItem(`scroll_${sessionId}_${tab}`);
        if (saved) scrollPositionsRef.current[tab] = parseInt(saved, 10);
      } catch {}
    }
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    useNotebookStore.getState().resetForNewSession();
    useStoryStore.getState().clear();
    useChatStore.getState().clear();
    setActiveSession(sessionId);

    void Promise.allSettled([
      listSessions(),
      getNotebook(sessionId),
      getStory(sessionId, "json"),
      fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"}/run/${sessionId}/status`).then(r => r.json()).catch(() => null),
      fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"}/run/${sessionId}/plan`).then(r => r.json()).catch(() => null),
    ]).then(([sessionsResult, notebookResult, storyResult, statusResult, planResult]) => {
      if (cancelled) return;

      if (sessionsResult.status === "fulfilled") {
        setSessions(sessionsResult.value.sessions || []);
      }

      // If run is already completed/failed, clear stale pipelineRunning
      if (statusResult.status === "fulfilled" && statusResult.value) {
        const runStatus = statusResult.value.status;
        if (runStatus === "completed" || runStatus === "failed" || runStatus === "idle") {
          useNotebookStore.getState().setPipelineRunning(false);
        }
      }

      // Hydrate execution plan from persisted state
      if (planResult.status === "fulfilled" && planResult.value?.steps?.length) {
        useNotebookStore.getState().setPlanSteps(planResult.value.steps);
      }

      const notebook =
        notebookResult.status === "fulfilled" ? notebookResult.value : null;
      const normalizedCells = normalizeNotebookCells(notebook);
      if (normalizedCells.length > 0) {
        useNotebookStore.getState().setCells(normalizedCells);
      }

      if (storyResult.status === "fulfilled" && storyResult.value) {
        useStoryStore.getState().setStory(storyResult.value);
        if (normalizedCells.length === 0) {
          setActiveTab("story");
        }
      }
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Restore scroll position instantly when tab changes
  useEffect(() => {
    const saved = scrollPositionsRef.current[resolvedTab] ?? 0;
    // Try immediately — container may already be in DOM
    const container = findScrollContainer(resolvedTab);
    if (container) {
      container.scrollTop = saved;
      return;
    }
    // Fallback: single rAF for cases where DOM hasn't painted yet
    const frame = requestAnimationFrame(() => {
      const el = findScrollContainer(resolvedTab);
      if (el) el.scrollTop = saved;
    });
    return () => cancelAnimationFrame(frame);
  }, [findScrollContainer, resolvedTab]);

  useAgentStream(sessionId);

  const tabClass = (tab: SessionTab) =>
    resolvedTab === tab
      ? "text-blue-700 border-b-2 border-blue-700 font-semibold"
      : "text-slate-500 hover:bg-slate-50";

  const sideNavClass = (tab: string) =>
    resolvedTab === tab
      ? "text-blue-700 border-l-2 border-blue-700"
      : "text-slate-400 hover:text-slate-900";

  const focusNotebook = useCallback(() => {
    setActiveTab("notebook");
    requestAnimationFrame(() => {
      const firstCell = useNotebookStore.getState().cells.find((cell) => cell.cell_type === "code" || cell.cell_type === "markdown");
      if (firstCell) {
        scrollToNotebookCell(firstCell.id);
      }
    });
  }, [setActiveTab]);

  const focusInvestigation = useCallback(
    (groupId: string) => {
      const store = useNotebookStore.getState();
      const group = store.hypothesisGroups.find((g) => g.id === groupId);
      setActiveTab("notebook");
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          const targetCellId =
            group?.cellIds?.[0] ||
            activityLog.find((entry) => entry.hypothesisId === groupId && entry.cellId)?.cellId ||
            store.cells.find((cell) => cell.cell_type === "code")?.id;

          if (targetCellId) {
            scrollToNotebookCell(targetCellId);
            return;
          }
          const container = document.querySelector<HTMLElement>("[data-notebook-scroll]");
          container?.scrollTo({ top: 0, behavior: "smooth" });
        });
      });
    },
    [activityLog, setActiveTab]
  );

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* ── TopNavBar ─────────────────────────────────────── */}
      <header className="w-full h-12 flex items-center justify-between px-4 bg-white border-b border-slate-200 z-50 shrink-0">
        <div className="flex items-center gap-6">
          <button
            onClick={() => router.push("/")}
            className="text-lg font-extrabold text-slate-900 tracking-tight font-headline hover:opacity-80 transition-opacity"
          >
            AgenticEDA
          </button>
          <nav className="hidden md:flex items-center gap-1 h-12">
            <button
              onClick={() => setActiveTab("notebook")}
              className={`px-3 h-full flex items-center text-sm tracking-tight font-headline transition-colors ${tabClass("notebook")}`}
            >
              Notebook
            </button>
            <button
              onClick={() => setActiveTab("story")}
              className={`px-3 h-full flex items-center text-sm tracking-tight font-headline transition-colors ${tabClass("story")}`}
            >
              Story
            </button>
            <button
              onClick={() => setActiveTab("history")}
              className={`px-3 h-full flex items-center text-sm tracking-tight font-headline transition-colors ${tabClass("history")}`}
            >
              History
            </button>
            <button
              onClick={() => setActiveTab("about")}
              className={`px-3 h-full flex items-center text-sm tracking-tight font-headline transition-colors ${tabClass("about")}`}
            >
              About
            </button>
          </nav>
        </div>
        <div className="flex items-center gap-3">
          {pipelineRunning && (
            <span className="flex items-center gap-1.5 text-xs text-primary font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              {currentPhase || "Analyzing..."}
            </span>
          )}
          {!chatOpen && (
            <button
              onClick={() => setChatOpen(true)}
              className="flex items-center gap-1 text-xs text-slate-500 hover:text-primary transition-colors"
              title="Open chat"
            >
              <span className="material-symbols-outlined text-[18px]">chat</span>
            </button>
          )}
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* ── SideNavRail ───────────────────────────────────── */}
        <aside className="w-16 bg-slate-50 flex flex-col items-center py-4 border-r border-slate-200 shrink-0 z-40">
          <div className="flex flex-col gap-6 items-center w-full">
            <button
              onClick={() => setExplorerOpen((v) => !v)}
              className={`w-full flex flex-col items-center py-2 transition-all duration-150 ${explorerOpen ? "text-blue-700 border-l-2 border-blue-700" : "text-slate-400 hover:text-slate-900"}`}
            >
              <span className="material-symbols-outlined" style={explorerOpen ? { fontVariationSettings: '"FILL" 1' } : undefined}>
                folder
              </span>
              <span className="text-[10px] font-semibold mt-1">Explorer</span>
            </button>
            <button
              onClick={() => setActiveTab("story")}
              className={`w-full flex flex-col items-center py-2 transition-all duration-150 ${sideNavClass("story")}`}
            >
              <span className="material-symbols-outlined">auto_stories</span>
              <span className="text-[10px] font-semibold mt-1">Story</span>
            </button>
            <button
              onClick={() => setActiveTab("history")}
              className={`w-full flex flex-col items-center py-2 transition-all duration-150 ${sideNavClass("history")}`}
            >
              <span className="material-symbols-outlined" style={resolvedTab === "history" ? { fontVariationSettings: '"FILL" 1' } : undefined}>
                history
              </span>
              <span className="text-[10px] font-semibold mt-1">History</span>
            </button>
            <button
              onClick={() => setActiveTab("about")}
              className={`w-full flex flex-col items-center py-2 transition-all duration-150 ${sideNavClass("about")}`}
            >
              <span className="material-symbols-outlined" style={resolvedTab === "about" ? { fontVariationSettings: '"FILL" 1' } : undefined}>
                info
              </span>
              <span className="text-[10px] font-semibold mt-1">About</span>
            </button>
          </div>
        </aside>

        {/* ── Explorer Panel ────────────────────────────────── */}
        <section
          className={`bg-surface-container-low flex flex-col shrink-0 border-r border-outline-variant/20 transition-[width] duration-200 overflow-hidden ${explorerOpen ? "w-64" : "w-0 border-r-0"}`}
        >
          <div className="p-4 flex items-center justify-between min-w-[15rem]">
            <h2 className="font-headline font-bold text-xs uppercase tracking-widest text-on-surface-variant">
              Explorer
            </h2>
            <button onClick={() => setExplorerOpen(false)} className="material-symbols-outlined text-sm cursor-pointer hover:text-primary">
              chevron_left
            </button>
          </div>
          <div className="px-2 space-y-1 overflow-y-auto flex-1">
            {/* Datasets */}
            <div className="group">
              <button
                type="button"
                onClick={focusNotebook}
                className="flex w-full items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-surface-container text-left text-xs font-semibold text-on-surface"
              >
                <span className="material-symbols-outlined text-[16px]">keyboard_arrow_down</span>
                <span>DATASETS</span>
              </button>
              {sessionMeta && (
                <div className="ml-4 space-y-0.5 mt-1">
                  <button
                    type="button"
                    onClick={focusNotebook}
                    className="flex w-full items-center gap-2 px-2 py-1.5 rounded-lg bg-surface-container-highest/50 text-left text-xs text-primary font-medium hover:bg-surface-container-highest/70"
                  >
                    <span className="material-symbols-outlined text-[16px] text-blue-500">table_chart</span>
                    <span className="truncate">{sessionMeta.original_filename}</span>
                  </button>
                  <div className="px-2 text-[10px] text-on-surface-variant ml-6">
                    {sessionMeta.row_count} rows &middot; {sessionMeta.col_count} cols
                  </div>
                </div>
              )}
            </div>

            {/* Agents */}
            <div className="group mt-4">
              <button
                type="button"
                className="flex w-full items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-surface-container text-left text-xs font-semibold text-on-surface"
              >
                <span className="material-symbols-outlined text-[16px]">keyboard_arrow_down</span>
                <span>AGENTS</span>
              </button>
              <div className="ml-4 space-y-0.5 mt-1">
                {/* Main notebook */}
                <button
                  type="button"
                  className={`flex w-full items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer text-xs text-left transition-colors ${
                    activeNotebookId === "main"
                      ? "bg-primary/10 text-primary font-medium"
                      : "text-on-surface-variant hover:bg-surface-container-high"
                  }`}
                  onClick={() => {
                    useNotebookStore.getState().setActiveNotebook("main");
                    useNotebookStore.getState().setActiveTab("notebook");
                    setActiveTab("notebook");
                  }}
                >
                  <span className="material-symbols-outlined text-[16px]" style={{ fontVariationSettings: '"FILL" 1' }}>description</span>
                  <span>Main</span>
                </button>
                {/* Agent notebooks */}
                {hypothesisGroups.map((g) => {
                  const nb = notebooks[g.id];
                  const agentMatch = g.id.match(/^agent_(\d+)$/);
                  const label = agentMatch ? `Agent ${agentMatch[1]}` : g.id;
                  const status = nb?.status || "idle";
                  return (
                    <button
                      key={g.id}
                      type="button"
                      className={`flex w-full items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer text-xs text-left transition-colors ${
                        activeNotebookId === g.id
                          ? "bg-primary/10 text-primary font-medium"
                          : "text-on-surface-variant hover:bg-surface-container-high"
                      }`}
                      onClick={() => {
                        useNotebookStore.getState().setActiveNotebook(g.id);
                        useNotebookStore.getState().setActiveTab("notebook");
                        setActiveTab("notebook");
                      }}
                    >
                      <span className="material-symbols-outlined text-[16px] text-primary">science</span>
                      <div className="flex flex-col min-w-0">
                        <span className="truncate">{label}</span>
                        {nb?.title && (
                          <span className="truncate text-[10px] text-on-surface-variant opacity-70">{nb.title}</span>
                        )}
                      </div>
                      {status === "running" && (
                        <span className="material-symbols-outlined ml-auto text-primary" style={{ fontSize: "14px" }}>schedule</span>
                      )}
                      {status === "complete" && (
                        <span className="material-symbols-outlined ml-auto text-green-600" style={{ fontSize: "14px" }}>check_circle</span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Agent Activity */}
            <AgentActivityBadge />
          </div>

          {/* Kernel Status + Context Window */}
          <div className="p-4 border-t border-outline-variant/10 space-y-2">
            <div className="flex items-center justify-between p-2 rounded-lg bg-primary/5 border border-primary/10">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-primary text-[16px]">cloud_done</span>
                <span className="text-[10px] font-bold text-on-surface truncate">Py Kernel</span>
              </div>
              <div className={`w-1.5 h-1.5 rounded-full ${pipelineRunning ? "bg-green-500 animate-pulse" : "bg-green-500"}`} />
            </div>
            <ContextWindowIndicator cellCount={cellCount} findingsCount={hypothesisGroups.length} />
          </div>
        </section>

        {/* ── Main Content Area ─────────────────────────────── */}
        <main
          ref={(node) => {
            contentPaneRef.current = node;
          }}
          className="flex-1 overflow-hidden flex flex-col"
        >
          {/* Keep all tabs mounted — hide with CSS for instant switching */}
          <div className={resolvedTab === "notebook" ? "flex flex-col flex-1 overflow-hidden" : "hidden"}>
            <NotebookPane />
          </div>
          <div className={resolvedTab === "story" ? "flex flex-col flex-1 overflow-hidden" : "hidden"}>
            <StoryPane
              sessionId={sessionId}
              onOpenNotebookCell={(cellId) => {
                setActiveTab("notebook");
                scrollToNotebookCell(cellId);
              }}
            />
          </div>
          <div className={resolvedTab === "history" ? "flex flex-col flex-1 overflow-hidden" : "hidden"}>
            <HistoryPanel sessionId={sessionId} onClose={() => setActiveTab("notebook")} />
          </div>
          <div className={resolvedTab === "about" ? "flex flex-col flex-1 overflow-hidden" : "hidden"}>
            <AboutPane />
          </div>
        </main>

        {/* ── Right Chat Sidebar ────────────────────────────── */}
        <ChatSidebar sessionId={sessionId} collapsed={!chatOpen} onToggle={() => setChatOpen((v) => !v)} />
      </div>
    </div>
  );
}
