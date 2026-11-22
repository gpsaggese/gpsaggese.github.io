"use client";

import { useRef, useEffect, useCallback } from "react";
import { useChatStore } from "@/stores/chatStore";
import { useNotebookStore } from "@/stores/notebookStore";
import { useChat } from "@/hooks/useChat";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import ExecutionPlan from "./ExecutionPlan";

interface ChatSidebarProps {
  sessionId: string;
  collapsed?: boolean;
  onToggle?: () => void;
}

export default function ChatSidebar({ sessionId, collapsed, onToggle }: ChatSidebarProps) {
  const messages = useChatStore((s) => s.messages);
  const isTyping = useChatStore((s) => s.isTyping);
  const pipelineRunning = useNotebookStore((s) => s.pipelineRunning);
  const { sendMessage } = useChat(sessionId);
  const setActiveNotebook = useNotebookStore((s) => s.setActiveNotebook);
  const setActiveTab = useNotebookStore((s) => s.setActiveTab);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, isTyping]);

  const handleViewCell = useCallback((cellId: string) => {
    const el = document.getElementById(`cell-${cellId}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.add("ring-2", "ring-primary");
      setTimeout(() => el.classList.remove("ring-2", "ring-primary"), 2000);
    }
  }, []);

  const handleOpenNotebook = useCallback((notebookId: string) => {
    setActiveNotebook(notebookId);
    setActiveTab("notebook");
  }, [setActiveNotebook, setActiveTab]);

  return (
    <div className={`bg-surface-container-low border-l border-outline-variant/20 flex flex-col shrink-0 h-full transition-[width] duration-200 overflow-hidden ${collapsed ? "w-0 border-l-0" : "w-80"}`}>
      {/* Header */}
      <div className="p-4 flex items-center justify-between border-b border-outline-variant/10 min-w-[19rem]">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary text-xl">smart_toy</span>
          <h2 className="text-sm font-semibold font-headline text-on-surface">AgenticEDA</h2>
        </div>
        <div className="flex items-center gap-2">
          {pipelineRunning && (
            <span className="flex items-center gap-1.5 rounded-full bg-primary/8 px-2.5 py-1 text-[10px] font-medium text-primary">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              Analyzing
            </span>
          )}
          {onToggle && (
            <button onClick={onToggle} className="material-symbols-outlined text-sm text-on-surface-variant hover:text-primary cursor-pointer">
              chevron_right
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <ExecutionPlan />
        {messages.length === 0 && !pipelineRunning && (
          <div className="mt-8 space-y-4">
            <div className="text-center text-on-surface-variant text-sm">
              <p className="font-body">Ask questions about your data</p>
              <p className="text-xs mt-1">or try a suggested action below</p>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => sendMessage("Clean up and fix data quality issues")}
                className="flex items-center gap-2 p-2.5 rounded-xl bg-surface-container-lowest border border-outline-variant/10 text-xs text-on-surface hover:bg-surface-container transition-colors"
              >
                <span className="material-symbols-outlined text-primary text-base">auto_fix_high</span>
                Cleanup Data
              </button>
              <button
                onClick={() => sendMessage("Explain the key variables in this dataset")}
                className="flex items-center gap-2 p-2.5 rounded-xl bg-surface-container-lowest border border-outline-variant/10 text-xs text-on-surface hover:bg-surface-container transition-colors"
              >
                <span className="material-symbols-outlined text-primary text-base">query_stats</span>
                Explain Var
              </button>
            </div>
          </div>
        )}
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} onViewCell={handleViewCell} onOpenNotebook={handleOpenNotebook} />
        ))}
        {isTyping && (
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 rounded bg-primary/10 flex items-center justify-center shrink-0">
              <span className="material-symbols-outlined text-primary text-sm">smart_toy</span>
            </div>
            <div className="bg-surface-container-lowest rounded-2xl rounded-tl-none chat-shadow border border-outline-variant/5 px-3 py-2">
              <div className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-on-surface-variant/60 animate-pulse" />
                <span className="w-1.5 h-1.5 rounded-full bg-on-surface-variant/60 animate-pulse [animation-delay:150ms]" />
                <span className="w-1.5 h-1.5 rounded-full bg-on-surface-variant/60 animate-pulse [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <ChatInput onSend={sendMessage} />
    </div>
  );
}
