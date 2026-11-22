"use client";
import { useCallback } from "react";
import { useChatStore } from "@/stores/chatStore";
import { useNotebookStore } from "@/stores/notebookStore";
import { useStoryStore } from "@/stores/storyStore";
import { sendSteering } from "@/lib/api";
import { API_BASE } from "@/lib/backend";

export function useChat(sessionId: string) {
  const sendMessage = useCallback(
    async (content: string) => {
      const notebookState = useNotebookStore.getState();
      if (notebookState.pipelineRunning) {
        const id = useChatStore.getState().addUserMessage(content, {
          kind: "steering",
          status: "queued",
        });
        try {
          await sendSteering(sessionId, content, id);
        } catch {
          useChatStore.getState().updateMessageMeta(id, {
            kind: "steering",
            status: "failed",
          });
        }
        return;
      }

      useChatStore.getState().addUserMessage(content);
      useChatStore.getState().setTyping(true);

      try {
        // Send current cells so the LLM knows what's in the notebook
        const cells = useNotebookStore.getState().cells.map((c) => ({
          id: c.id,
          cell_type: c.cell_type,
          source: c.source,
        }));

        // Use REST endpoint instead of WebSocket for reliability
        const res = await fetch(`${API_BASE}/chat/${sessionId}/message`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content, cells }),
        });

        if (res.ok) {
          const data = await res.json();
          useChatStore.getState().addAgentMessage(
            data.content,
            data.type || "text",
            data.cell_id,
            (data.notebook_id || data.hypothesis_id)
              ? {
                  kind: data.notebook_id ? "investigation" : "info",
                  notebook_id: data.notebook_id,
                  hypothesis_id: data.hypothesis_id,
                  title: data.title,
                  status: data.status,
                  confidence: data.confidence,
                }
              : undefined
          );

          // If chat executed code, the cell should already be in the stream
          // but also trigger a scroll to make it visible
          if (data.type === "action" && data.action_code) {
            setTimeout(() => {
              const cells = useNotebookStore.getState().cells;
              const lastCell = cells[cells.length - 1];
              if (lastCell) {
                const el = document.getElementById(`cell-${lastCell.id}`);
                if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
              }
            }, 500);

            // Retrigger story generation
            setTimeout(() => {
              fetch(`${API_BASE}/story/${sessionId}?format=json`)
                .then((r) => (r.ok ? r.json() : null))
                .then((story) => {
                  if (story) useStoryStore.getState().setStory(story);
                })
                .catch(() => {});
            }, 2000);
          }
        } else {
          useChatStore.getState().addAgentMessage(
            "Unable to process that request.",
            "text"
          );
        }
      } catch {
        useChatStore.getState().addAgentMessage(
          "Chat is not connected. Make sure the backend is running.",
          "text"
        );
      }
    },
    [sessionId]
  );

  return { sendMessage };
}
