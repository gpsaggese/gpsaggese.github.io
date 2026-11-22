"use client";

import { useState, useCallback } from "react";
import { useChatStore } from "@/stores/chatStore";
import { useNotebookStore } from "@/stores/notebookStore";

interface ChatInputProps {
  onSend: (content: string) => void;
}

export default function ChatInput({ onSend }: ChatInputProps) {
  const [value, setValue] = useState("");
  const isTyping = useChatStore((s) => s.isTyping);
  const pipelineRunning = useNotebookStore((s) => s.pipelineRunning);

  const submit = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || isTyping) return;
    onSend(trimmed);
    setValue("");
  }, [value, isTyping, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="p-4 border-t border-outline-variant/10 bg-surface-container-lowest">
      <div className="relative">
        <textarea
          className="w-full bg-surface-container-low border border-outline-variant/20 rounded-xl p-3 pr-10 text-xs focus:ring-2 focus:ring-primary focus:border-transparent resize-none font-body text-on-surface placeholder:text-on-surface-variant/50 outline-none"
          rows={2}
          placeholder={pipelineRunning ? "Steer the current run..." : "Ask about your data..."}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isTyping}
        />
        <button
          className="absolute bottom-3 right-3 p-1.5 bg-primary text-white rounded-lg disabled:opacity-40 transition-colors hover:bg-primary/90"
          onClick={submit}
          disabled={isTyping || !value.trim()}
        >
          <span className="material-symbols-outlined text-sm">arrow_upward</span>
        </button>
      </div>
      <div className="flex items-center justify-between mt-2">
        <p className="text-[10px] text-on-surface-variant font-body">
          {isTyping ? "Agent is typing..." : "Connected to GPT-5.4 Nano"}
        </p>
        <p className="text-[10px] text-on-surface-variant font-body">
          Shift + Enter to send
        </p>
      </div>
    </div>
  );
}
