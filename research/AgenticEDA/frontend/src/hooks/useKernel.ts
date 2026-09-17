"use client";

import { useCallback } from "react";
import { useNotebookStore } from "@/stores/notebookStore";
import type { CellOutput } from "@/lib/types";
import { API_BASE } from "@/lib/backend";

type KernelStatus = "disconnected" | "connected" | "busy";

export interface ExecuteCellResult {
  outputs: CellOutput[];
  error: string | null;
}

async function readError(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  return text.trim() || res.statusText || `HTTP ${res.status}`;
}

export function useKernel(sessionId: string) {
  // Derive kernel status from pipeline state instead of hardcoding
  const pipelineRunning = useNotebookStore((s) => s.pipelineRunning);
  const agentActivity = useNotebookStore((s) => s.agentActivity);

  let status: KernelStatus = "disconnected";
  if (pipelineRunning) {
    status = agentActivity === "executing" ? "busy" : "connected";
  } else if (agentActivity === "complete" || agentActivity === "idle") {
    // Pipeline finished — kernel is alive but idle
    status = "connected";
  }

  const executeCell = useCallback(
    async (cellId: string, source: string): Promise<ExecuteCellResult> => {
      if (!sessionId) {
        throw new Error("No active session is selected.");
      }
      if (!source.trim()) {
        return { outputs: [], error: null };
      }

      const res = await fetch(`${API_BASE}/kernel/${sessionId}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cell_id: cellId,
          code: source,
          timeout: 60,
        }),
      });

      if (!res.ok) {
        throw new Error(`Cell execution failed: ${await readError(res)}`);
      }

      const data = (await res.json()) as Partial<ExecuteCellResult>;
      return {
        outputs: Array.isArray(data.outputs) ? data.outputs : [],
        error: typeof data.error === "string" ? data.error : null,
      };
    },
    [sessionId]
  );

  return { status, executeCell };
}
