import { create } from "zustand";
import type { Cell } from "@/lib/types";
import { v4 as uuidv4 } from "uuid";

export type AgentActivity = "idle" | "thinking" | "generating" | "executing" | "fixing" | "backtracking" | "complete" | "failed";
export type WorkspaceTab = "notebook" | "story" | "history" | "about";

export interface PlanStepDetail {
  label: string;
  status: "complete" | "current" | "skipped" | "upcoming";
}

export interface PlanStep {
  phase: string;
  status: "complete" | "current" | "skipped" | "upcoming";
  details?: PlanStepDetail[];
}

export interface NotebookData {
  id: string;
  title: string;
  cells: Cell[];
  status: "idle" | "running" | "complete" | "failed" | "timeout";
}

export interface ActivityLogEntry {
  id: number;
  activity: AgentActivity;
  detail: string;
  phase: string;
  ts: number;
  cellId?: string;
  cellType?: "code" | "markdown";
  hypothesisId?: string;
  notebookId?: string;
  chatAction?: string; // "new_cell" | "edit_cell" | "delete_cells" | "reorder_cell"
}

interface NotebookState {
  cells: Cell[];
  dirty: boolean;
  pipelineRunning: boolean;
  currentPhase: string;
  executionCounter: number;
  agentActivity: AgentActivity;
  agentActivityDetail: string;
  activityLog: ActivityLogEntry[];
  fixedCellIds: Set<string>;
  setCells: (cells: Cell[]) => void;
  addCell: (index: number, type: "code" | "markdown") => void;
  appendCell: (cell: Cell, options?: { markFixed?: boolean }) => void;
  overwriteCell: (id: string, source: string) => void;
  updateCellSource: (id: string, source: string) => void;
  updateCellOutputs: (id: string, outputs: Cell["outputs"]) => void;
  deleteCell: (id: string) => void;
  moveCell: (id: string, direction: "up" | "down") => void;
  setDirty: (dirty: boolean) => void;
  setPipelineRunning: (running: boolean) => void;
  setCurrentPhase: (phase: string) => void;
  setCellExecuting: (id: string, executing: boolean) => void;
  setCellError: (id: string, error: string | null) => void;
  setCellThinking: (id: string, thinking: string | null) => void;
  latestThinking: string;
  setLatestThinking: (text: string) => void;
  hypothesisGroups: { id: string; title: string; cellIds: string[] }[];
  addHypothesisGroup: (id: string, title: string) => void;
  setAgentActivity: (activity: AgentActivity, detail: string, extra?: Partial<Pick<ActivityLogEntry, "cellId" | "cellType" | "hypothesisId" | "notebookId">>) => void;
  addChatAction: (detail: string, actionType: string) => void;
  planSteps: PlanStep[];
  setPlanSteps: (steps: PlanStep[]) => void;
  clearActivityLog: () => void;
  resetForNewSession: () => void;
  activeTab: WorkspaceTab;
  setActiveTab: (tab: WorkspaceTab) => void;
  notebooks: Record<string, NotebookData>;
  activeNotebookId: string;
  ensureNotebook: (id: string, title: string) => void;
  setActiveNotebook: (id: string) => void;
  setNotebookStatus: (id: string, status: NotebookData["status"]) => void;
  appendCellToNotebook: (notebookId: string, cell: Cell, options?: { markFixed?: boolean }) => void;
  clearNotebookCells: (id: string) => void;
  getActiveNotebookCells: () => Cell[];
}

export const useNotebookStore = create<NotebookState>((set) => ({
  cells: [],
  dirty: false,
  pipelineRunning: false,
  currentPhase: "",
  executionCounter: 0,
  latestThinking: "",
  agentActivity: "idle" as AgentActivity,
  agentActivityDetail: "",
  activityLog: [],
  planSteps: [],
  fixedCellIds: new Set<string>(),
  setCells: (cells) => {
    // Deduplicate by ID, keeping last occurrence
    const seen = new Map<string, Cell>();
    for (const cell of cells) seen.set(cell.id, cell);
    const nextCells = Array.from(seen.values());
    const executionCounter = nextCells.reduce(
      (max, cell) => Math.max(max, cell.execution_count ?? 0),
      0,
    );
    return set({ cells: nextCells, dirty: false, executionCounter });
  },
  addCell: (index, type) =>
    set((s) => {
      const cell: Cell = { id: uuidv4(), cell_type: type, source: "", outputs: [], execution_count: null };
      const cells = [...s.cells];
      cells.splice(index, 0, cell);
      return { cells, dirty: true };
    }),
  appendCell: (cell, options) =>
    set((s) => {
      if (s.cells.some((c) => c.id === cell.id)) return s;
      // Filter out intermediary agent status markdown cells
      if (cell.cell_type === "markdown") {
        const src = cell.source.trim();
        const skipPatterns = [
          /^Attempting fix/i,
          /^Backtracking/i,
          /^Retrying/i,
          /^Error:/i,
          /^Fixing/i,
          /^Let me (try|fix|correct)/i,
          /^I('ll| will) (try|fix|correct|retry)/i,
          /^##\s+Conclusions[\s\S]*Synthesizing all findings/i,
          /^##\s+Investigation Phase[\s\S]*Generating hypotheses/i,
        ];
        if (skipPatterns.some((p) => p.test(src))) return s;
      }
      const nextState: Partial<NotebookState> = { cells: [...s.cells, cell] };
      if (options?.markFixed) {
        const fixedCellIds = new Set(s.fixedCellIds);
        fixedCellIds.add(cell.id);
        nextState.fixedCellIds = fixedCellIds;
        setTimeout(() => {
          useNotebookStore.setState((prev) => {
            const next = new Set(prev.fixedCellIds);
            next.delete(cell.id);
            return { fixedCellIds: next };
          });
        }, 2000);
      }
      return nextState as { cells: Cell[] } & Partial<NotebookState>;
    }),
  overwriteCell: (id, source) =>
    set((s) => {
      const idx = s.cells.findIndex((c) => c.id === id);
      if (idx < 0) {
        const fixedCellIds = new Set(s.fixedCellIds);
        fixedCellIds.add(id);
        setTimeout(() => {
          useNotebookStore.setState((prev) => {
            const next = new Set(prev.fixedCellIds);
            next.delete(id);
            return { fixedCellIds: next };
          });
        }, 2000);
        return {
          cells: [
            ...s.cells,
            {
              id,
              cell_type: "code",
              source,
              outputs: [],
              execution_count: null,
              error: null,
              executing: false,
            },
          ],
          dirty: true,
          fixedCellIds,
        };
      }
      const updated = {
        ...s.cells[idx],
        source,
        outputs: [] as Cell["outputs"],
        error: null,
        execution_count: null,
        executing: false,
      };
      const cells = [...s.cells];
      cells[idx] = updated;
      // Track this cell as recently fixed
      const fixedCellIds = new Set(s.fixedCellIds);
      fixedCellIds.add(id);
      // Auto-clear after 2 seconds
      setTimeout(() => {
        useNotebookStore.setState((prev) => {
          const next = new Set(prev.fixedCellIds);
          next.delete(id);
          return { fixedCellIds: next };
        });
      }, 2000);
      return { cells, dirty: true, fixedCellIds };
    }),
  updateCellSource: (id, source) =>
    set((s) => ({
      cells: s.cells.map((c) => (c.id === id ? { ...c, source } : c)),
      dirty: true,
    })),
  updateCellOutputs: (id, outputs) =>
    set((s) => {
      const newCounter = s.executionCounter + 1;
      return {
        executionCounter: newCounter,
        cells: s.cells.map((c) =>
          c.id === id ? { ...c, outputs, execution_count: newCounter } : c
        ),
        dirty: true,
      };
    }),
  deleteCell: (id) =>
    set((s) => ({ cells: s.cells.filter((c) => c.id !== id), dirty: true })),
  moveCell: (id, direction) =>
    set((s) => {
      const idx = s.cells.findIndex((c) => c.id === id);
      if (idx < 0) return s;
      const newIdx = direction === "up" ? idx - 1 : idx + 1;
      if (newIdx < 0 || newIdx >= s.cells.length) return s;
      const cells = [...s.cells];
      [cells[idx], cells[newIdx]] = [cells[newIdx], cells[idx]];
      return { cells, dirty: true };
    }),
  setDirty: (dirty) => set({ dirty }),
  setPipelineRunning: (pipelineRunning) => set({ pipelineRunning }),
  setCurrentPhase: (currentPhase) => set({ currentPhase }),
  setCellExecuting: (id, executing) =>
    set((s) => ({
      cells: s.cells.map((c) => (c.id === id ? { ...c, executing } : c)),
    })),
  setCellError: (id, error) =>
    set((s) => ({
      cells: s.cells.map((c) => (c.id === id ? { ...c, error } : c)),
    })),
  setCellThinking: (id, thinking) =>
    set((s) => ({
      cells: s.cells.map((c) => (c.id === id ? { ...c, thinking } : c)),
    })),
  setLatestThinking: (latestThinking) => set({ latestThinking }),
  hypothesisGroups: [],
  addHypothesisGroup: (id, title) =>
    set((s) => ({
      hypothesisGroups: s.hypothesisGroups.some((g) => g.id === id)
        ? s.hypothesisGroups
        : [...s.hypothesisGroups, { id, title, cellIds: [] }],
    })),
  setAgentActivity: (activity, detail, extra) =>
    set((s) => {
      const entry: ActivityLogEntry = {
        id: s.activityLog.length + 1,
        activity,
        detail,
        phase: s.currentPhase,
        ts: Date.now(),
        ...extra,
      };
      return {
        agentActivity: activity,
        agentActivityDetail: detail,
        activityLog: [...s.activityLog, entry],
      };
    }),
  addChatAction: (detail, actionType) =>
    set((s) => {
      const entry: ActivityLogEntry = {
        id: s.activityLog.length + 1,
        activity: "generating",
        detail,
        phase: "Chat",
        ts: Date.now(),
        chatAction: actionType,
      };
      return { activityLog: [...s.activityLog, entry] };
    }),
  setPlanSteps: (planSteps) => set({ planSteps }),
  clearActivityLog: () =>
    set({ activityLog: [], agentActivity: "idle" as AgentActivity, agentActivityDetail: "" }),
  resetForNewSession: () =>
    set({
      cells: [],
      dirty: false,
      pipelineRunning: false,
      currentPhase: "",
      executionCounter: 0,
      latestThinking: "",
      agentActivity: "idle" as AgentActivity,
      agentActivityDetail: "",
      activityLog: [],
      planSteps: [],
      hypothesisGroups: [],
      fixedCellIds: new Set<string>(),
      activeTab: "notebook" as WorkspaceTab,
      notebooks: { main: { id: "main", title: "Main", cells: [], status: "idle" as const } },
      activeNotebookId: "main",
    }),
  activeTab: "notebook" as WorkspaceTab,
  setActiveTab: (activeTab) => set({ activeTab }),
  notebooks: { main: { id: "main", title: "Main", cells: [], status: "idle" as const } },
  activeNotebookId: "main",

  ensureNotebook: (id, title) =>
    set((s) => {
      if (s.notebooks[id]) {
        // Update title for recycled tabs
        const existing = s.notebooks[id];
        if (existing.title === title) return s;
        return {
          notebooks: {
            ...s.notebooks,
            [id]: { ...existing, title },
          },
        };
      }
      return {
        notebooks: {
          ...s.notebooks,
          [id]: { id, title, cells: [], status: "idle" as const },
        },
      };
    }),

  setActiveNotebook: (id) => set({ activeNotebookId: id }),

  setNotebookStatus: (id, status) =>
    set((s) => {
      const nb = s.notebooks[id];
      if (!nb) return s;
      return {
        notebooks: {
          ...s.notebooks,
          [id]: { ...nb, status },
        },
      };
    }),

  appendCellToNotebook: (notebookId, cell, options) =>
    set((s) => {
      const nb = s.notebooks[notebookId] || s.notebooks["main"];
      const targetId = s.notebooks[notebookId] ? notebookId : "main";
      const existing = nb.cells.findIndex((c) => c.id === cell.id);
      let newCells: Cell[];
      if (existing >= 0) {
        newCells = [...nb.cells];
        // Only merge non-empty fields to avoid overwriting source with ""
        const nonEmptyUpdates = Object.fromEntries(
          Object.entries(cell).filter(([, value]) => value !== "" && value !== undefined && value !== null),
        ) as Partial<Cell>;
        const merged: Cell = { ...newCells[existing], ...nonEmptyUpdates };
        // Always merge outputs and error even if "empty" (they represent real state)
        if (cell.outputs !== undefined) merged.outputs = cell.outputs;
        if (cell.error !== undefined) merged.error = cell.error;
        if (cell.executing !== undefined) merged.executing = cell.executing;
        newCells[existing] = merged;
      } else {
        newCells = [...nb.cells, cell];
      }
      const fixedCellIds = options?.markFixed
        ? new Set([...s.fixedCellIds, cell.id])
        : s.fixedCellIds;
      return {
        notebooks: {
          ...s.notebooks,
          [targetId]: { ...nb, cells: newCells },
        },
        fixedCellIds,
      };
    }),

  clearNotebookCells: (id) =>
    set((s) => {
      const nb = s.notebooks[id];
      if (!nb) return s;
      return {
        notebooks: {
          ...s.notebooks,
          [id]: { ...nb, cells: [], status: "idle" },
        },
      };
    }),

  getActiveNotebookCells: (): Cell[] => {
    // NOTE: call useNotebookStore.getState() at invocation time, not init time
    const s = useNotebookStore.getState();
    const nb = s.notebooks[s.activeNotebookId];
    return nb ? nb.cells : [];
  },
}));
