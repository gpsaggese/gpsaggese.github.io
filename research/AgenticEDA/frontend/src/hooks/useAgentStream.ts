"use client";
import { useEffect, useRef } from "react";
import { useNotebookStore } from "@/stores/notebookStore";
import { useStoryStore } from "@/stores/storyStore";
import { useChatStore } from "@/stores/chatStore";
import type { Cell, CellOutput } from "@/lib/types";
import { API_BASE, WS_BASE } from "@/lib/backend";

type StreamEvent =
  | { type: "thinking"; content?: string }
  | { type: "cell_write"; cell_id: string; cell_type?: "code" | "markdown"; source?: string; overwrite?: boolean; notebook_id?: string }
  | { type: "cell_executing"; cell_id: string; notebook_id?: string }
  | { type: "cell_output"; cell_id: string; outputs?: CellOutput[]; notebook_id?: string }
  | { type: "cell_error"; cell_id: string; error?: string; traceback?: string[] | string[][]; notebook_id?: string }
  | { type: "cell_update"; cell_id: string; source?: string }
  | { type: "cell_delete"; cell_id: string }
  | { type: "cell_reorder"; cell_id: string; direction?: "up" | "down" }
  | { type: "chat_action"; detail?: string; action?: string; cell_id?: string }
  | { type: "phase_transition"; phase?: string; notebook_id?: string }
  | { type: "backtrack"; reason?: string; cell_id?: string }
  | { type: "chat_investigation_complete"; hypothesis_id?: string; finding?: string; confidence?: number }
  | { type: "subagent_start"; hypothesis_id?: string; notebook_id?: string; title?: string }
  | { type: "subagent_complete"; hypothesis_id?: string; notebook_id?: string; finding?: string; confidence?: number; status?: "complete" | "failed" }
  | { type: "subagent_timeout"; hypothesis_id?: string; notebook_id?: string }
  | { type: "subagents_dispatched"; notebook_id?: string; loop_number?: number; count?: number; hypothesis_ids?: string[]; titles?: string[] }
  | { type: "subagents_returned"; notebook_id?: string; loop_number?: number; results_count?: number }
  | { type: "loop_start"; loop_number?: number; total_loops?: number }
  | { type: "loop_complete"; loop_number?: number }
  | { type: "accumulation_progress"; step?: number; total?: number; label?: string; notebook_id?: string }
  | { type: "notebook_clear"; notebook_id?: string }
  | { type: "notebook_focus"; notebook_id?: string }
  | { type: "plan_update"; steps?: Array<{ phase: string; status: string; details?: Array<{ label: string; status: string }> }> }
  | { type: "steering_queued"; message_id: string; content?: string; status?: "queued" }
  | { type: "steering_read"; message_id: string; content?: string; read_at?: string; checkpoint?: string }
  | { type: "complete"; summary?: string };

export function useAgentStream(sessionId: string) {
  const connected = useRef<string | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const thinkingTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (connected.current === sessionId) return;
    // Close previous WebSocket if switching sessions
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
    connected.current = sessionId;

    try {
      const socket = new WebSocket(`${WS_BASE}/stream/${sessionId}`);
      ws.current = socket;

      socket.onopen = () => {
        // Don't set pipelineRunning here — the page checks run status on load.
        // pipelineRunning will be set on the first agent event (phase_transition, cell_write, etc.)
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data) as StreamEvent;
        const store = useNotebookStore.getState();
        const chat = useChatStore.getState();

        switch (data.type) {
          case "thinking":
            // Debounce thinking updates to reduce re-renders
            if (thinkingTimer.current) clearTimeout(thinkingTimer.current);
            thinkingTimer.current = setTimeout(() => {
              store.setLatestThinking(data.content || "");
            }, 150);
            store.setAgentActivity("thinking", data.content || "Thinking...", { notebookId: "main" });
            break;

          case "steering_queued":
            if (data.message_id) {
              chat.updateMessageMeta(data.message_id, { kind: "steering", status: "queued" });
            }
            break;

          case "steering_read":
            if (data.message_id) {
              chat.updateMessageMeta(data.message_id, { kind: "steering", status: "read" });
            }
            break;

          case "cell_write": {
            const notebookId = data.notebook_id || "main";
            // Extract hypothesis ID first (needed for activity log)
            const hypMatch = data.cell_id?.match(/^(h\d+)_/);
            const hypId = hypMatch ? hypMatch[1] : undefined;

            const cell: Cell = {
              id: data.cell_id,
              cell_type: data.cell_type || "code",
              source: data.source || "",
              outputs: [],
              execution_count: null,
              executing: false,
              error: null,
            };
            if (notebookId !== "main") {
              store.appendCellToNotebook(notebookId, cell, { markFixed: !!data.overwrite });
            } else {
              store.appendCell(cell, { markFixed: !!data.overwrite });
            }
            store.setLatestThinking("");  // clear thinking when cell appears
            store.setAgentActivity(data.overwrite ? "fixing" : "generating", data.overwrite ? "Redoing failed cell with corrected code" : (data.source || ""), {
              cellId: data.cell_id,
              cellType: (data.cell_type as "code" | "markdown") || "code",
              hypothesisId: hypId,
              notebookId,
            });

            // Track hypothesis grouping
            if (hypId) {
              const groups = store.hypothesisGroups;
              const group = groups.find((g: { id: string }) => g.id === hypId);
              if (group) {
                group.cellIds.push(data.cell_id);
              }
            }
            break;
          }

          case "cell_executing": {
            const notebookId_exec = data.notebook_id || "main";
            if (notebookId_exec === "main") {
              store.setCellExecuting(data.cell_id, true);
            }
            // For investigation notebooks, the executing state is handled by the tab badge
            store.setAgentActivity("executing", "Running cell...", { cellId: data.cell_id, notebookId: notebookId_exec });
            break;
          }

          case "cell_output": {
            const notebookId_out = data.notebook_id || "main";
            if (notebookId_out !== "main") {
              // Update cell in investigation notebook
              store.appendCellToNotebook(notebookId_out, {
                id: data.cell_id,
                cell_type: "code",
                source: "",
                outputs: data.outputs || [],
                execution_count: null,
                executing: false,
                error: null,
              });
            } else {
              store.setCellExecuting(data.cell_id, false);
              store.updateCellOutputs(data.cell_id, data.outputs || []);
            }
            // Removed cosmetic setLatestThinking call to reduce re-renders
            break;
          }

          case "cell_error": {
            const notebookId_err = data.notebook_id || "main";
            if (notebookId_err === "main") {
              store.setCellExecuting(data.cell_id, false);
              store.setCellError(data.cell_id, data.error || "Unknown error");
              if (data.traceback) {
                store.updateCellOutputs(data.cell_id, [{
                  output_type: "error",
                  ename: "Error",
                  evalue: data.error,
                  traceback: Array.isArray(data.traceback) ? data.traceback.flat() : [],
                }]);
              }
            }
            // Investigation notebook errors are tracked in the subagent's result
            store.setAgentActivity("fixing", `Fixing: ${data.error || "Unknown error"}`, {
              cellId: data.cell_id,
              cellType: "code",
              notebookId: notebookId_err,
            });
            break;
          }

          case "cell_update":
            store.overwriteCell(data.cell_id, data.source || "");
            store.setAgentActivity("fixing", "Overwrote cell with corrected code", {
              cellId: data.cell_id,
              cellType: "code",
              notebookId: "main",
            });
            break;

          case "cell_delete":
            store.setAgentActivity("backtracking", "Removed failed cell before retry", {
              cellId: data.cell_id,
              cellType: "code",
              notebookId: "main",
            });
            store.deleteCell(data.cell_id);
            break;

          case "cell_reorder":
            store.moveCell(data.cell_id, data.direction || "down");
            break;

          case "chat_action":
            store.setAgentActivity("generating", data.detail || "User action", {
              cellId: data.cell_id,
              notebookId: "main",
            });
            store.addChatAction(data.detail || "Action", data.action || "unknown");
            break;

          case "phase_transition": {
            // Chat investigations (notebook_id starts with "chat_") should NOT
            // set pipelineRunning — that hides the story pane with a spinner.
            const phaseNotebookId = data.notebook_id || "main";
            const isChatInvestigation = phaseNotebookId.startsWith("chat_");
            if (!isChatInvestigation) {
              store.setPipelineRunning(true);
            }
            store.setCurrentPhase(data.phase || "");
            store.setAgentActivity("thinking", data.phase || "", {
              hypothesisId: data.notebook_id || undefined,
              notebookId: phaseNotebookId,
            });
            if (data.notebook_id) {
              store.addHypothesisGroup(data.notebook_id, data.phase || "");
            }
            break;
          }

          case "backtrack":
            store.setLatestThinking(`⟲ Correcting: ${data.reason}`);
            store.setAgentActivity("backtracking", `Correcting: ${data.reason || "retrying failed cell"}`, {
              cellId: data.cell_id,
              cellType: "code",
              notebookId: "main",
            });
            break;

          case "chat_investigation_complete":
            store.setPipelineRunning(false);
            store.setCurrentPhase("");
            store.setLatestThinking("");
            store.setAgentActivity("complete", `Investigation complete: ${data.finding || ""}`, { notebookId: "main" });
            break;

          case "notebook_clear":
            if (data.notebook_id) {
              store.clearNotebookCells(data.notebook_id);
            }
            break;

          case "notebook_focus":
            if (data.notebook_id) {
              store.setActiveNotebook(data.notebook_id);
              store.setActiveTab("notebook");
            }
            break;

          case "subagent_start":
            if (data.notebook_id && data.title) {
              store.ensureNotebook(data.notebook_id, data.title);
              store.setNotebookStatus(data.notebook_id, "running");
              store.addHypothesisGroup(data.notebook_id, data.title);
            }
            if (data.notebook_id && String(data.notebook_id).startsWith("chat_")) {
              const alreadyAnnounced = chat.messages?.some(
                (m) =>
                  m.meta?.kind === "investigation" &&
                  m.meta?.notebook_id === data.notebook_id &&
                  m.meta?.status === "started",
              );
              if (!alreadyAnnounced) {
                chat.addAgentMessage(
                  "Investigation started. Streaming progress to the new notebook tab.",
                  "action",
                  undefined,
                  {
                    kind: "investigation",
                    notebook_id: data.notebook_id,
                    hypothesis_id: data.hypothesis_id,
                    title: data.title,
                    status: "started",
                  },
                );
              }
            }
            store.setAgentActivity("thinking", `Starting: ${data.title || ""}`, {
              hypothesisId: data.hypothesis_id,
              notebookId: data.notebook_id || "main",
            });
            break;

          case "subagent_complete": {
            const status = data.status === "failed" ? "failed" : "complete";
            if (data.notebook_id) {
              store.setNotebookStatus(data.notebook_id, status);
            }
            // For chat-triggered investigations (notebook_id starts with "chat_"), reset pipeline state
            if (data.notebook_id && String(data.notebook_id).startsWith("chat_")) {
              store.setPipelineRunning(false);
              store.setCurrentPhase("");
              store.setLatestThinking("");
              chat.addAgentMessage(
                (data.finding || "").trim() ? (data.finding as string) : "Investigation complete.",
                "text",
                undefined,
                {
                  kind: "investigation",
                  notebook_id: data.notebook_id,
                  hypothesis_id: data.hypothesis_id,
                  status,
                  confidence: data.confidence,
                },
              );
            }
            if (status === "failed") {
              store.setAgentActivity("failed", `Failed: ${(data.finding || "").slice(0, 80)}`, {
                hypothesisId: data.hypothesis_id,
                notebookId: data.notebook_id || "main",
              });
            } else {
              store.setAgentActivity("complete", `Done: ${(data.finding || "").slice(0, 80)}`, {
                hypothesisId: data.hypothesis_id,
                notebookId: data.notebook_id || "main",
              });
            }
            break;
          }

          case "subagent_timeout":
            if (data.notebook_id) {
              store.setNotebookStatus(data.notebook_id, "timeout");
            }
            if (data.notebook_id && String(data.notebook_id).startsWith("chat_")) {
              store.setPipelineRunning(false);
              store.setCurrentPhase("");
              store.setLatestThinking("");
              chat.addAgentMessage(
                "Investigation timed out before producing a result.",
                "text",
                undefined,
                {
                  kind: "investigation",
                  notebook_id: data.notebook_id,
                  hypothesis_id: data.hypothesis_id,
                  status: "timeout",
                },
              );
            }
            store.setAgentActivity("fixing", "Investigation timed out", {
              hypothesisId: data.hypothesis_id,
              notebookId: data.notebook_id || "main",
            });
            break;

          case "subagents_dispatched":
            store.setAgentActivity("thinking", `Dispatched ${data.count || 0} subagents...`, { notebookId: "main" });
            break;

          case "subagents_returned":
            store.setAgentActivity("thinking", `${data.results_count || 0} investigations complete. Compiling...`, { notebookId: "main" });
            break;

          case "accumulation_progress": {
            const pct = data.total ? Math.round(((data.step || 0) / data.total) * 100) : 0;
            store.setLatestThinking(`[${data.step}/${data.total}] ${data.label || "Processing..."}`);
            store.setAgentActivity("generating", `Accumulating results (${pct}%): ${data.label || ""}`, { notebookId: "main" });
            break;
          }

          case "loop_start":
            store.setCurrentPhase(`Investigation Loop ${data.loop_number || 1}/${data.total_loops || 1}`);
            break;

          case "loop_complete":
            store.setAgentActivity("thinking", `Loop ${data.loop_number} complete, analyzing results...`, { notebookId: "main" });
            break;

          case "plan_update":
            if (data.steps) {
              store.setPlanSteps(
                data.steps.map((s) => ({
                  phase: s.phase,
                  status: s.status as "complete" | "current" | "skipped" | "upcoming",
                  details: s.details?.map((d) => ({
                    label: d.label,
                    status: d.status as "complete" | "current" | "skipped" | "upcoming",
                  })),
                })),
              );
            }
            break;

          case "complete":
            // Guard: only handle once (ignore duplicate complete events)
            if (!store.pipelineRunning) break;
            store.setPipelineRunning(false);
            store.setCurrentPhase("");
            store.setLatestThinking("");
            store.setAgentActivity("complete", "EDA complete", { notebookId: "main" });
            chat.addAgentMessage(
              "EDA complete! Switch to the **Story** tab for the full narrative report, or ask me questions about the data.",
              "text",
            );
            // Fetch story after a short delay to give backend time to write it
            setTimeout(() => {
              fetch(`${API_BASE}/story/${sessionId}?format=json`)
                .then((r) => (r.ok ? r.json() : null))
                .then((story) => {
                  if (story) useStoryStore.getState().setStory(story);
                })
                .catch(() => {});
            }, 1000);
            break;
        }
      };

      socket.onclose = () => {
        ws.current = null;
      };

      socket.onerror = () => {
        // WebSocket may fail on first try if agent hasn't started yet — retry once
        setTimeout(() => {
          if (!ws.current || ws.current.readyState === WebSocket.CLOSED) {
            connected.current = null;
            // The useEffect will re-run on next render cycle
          }
        }, 2000);
      };
    } catch {
      console.warn("Stream WebSocket not available");
    }

    return () => {
      // Don't close the socket on cleanup from strict-mode remount
      // Only close if component is truly unmounting
    };
  }, [sessionId]);
}
