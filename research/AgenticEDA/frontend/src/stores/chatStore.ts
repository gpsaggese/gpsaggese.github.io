"use client";
import { create } from "zustand";
import type { ChatMessage } from "@/lib/types";
import { v4 as uuidv4 } from "uuid";

interface ChatState {
  messages: ChatMessage[];
  isTyping: boolean;
  addUserMessage: (content: string, meta?: ChatMessage["meta"]) => string;
  addAgentMessage: (
    content: string,
    type?: "text" | "cell_ref" | "action",
    cellId?: string,
    meta?: ChatMessage["meta"],
  ) => void;
  updateMessageMeta: (id: string, meta: Partial<NonNullable<ChatMessage["meta"]>>) => void;
  setTyping: (typing: boolean) => void;
  clear: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isTyping: false,
  addUserMessage: (content, meta) => {
    const id = uuidv4();
    const msg: ChatMessage = { id, role: "user", content, type: "text", meta, timestamp: new Date().toISOString() };
    set((s) => ({ messages: [...s.messages, msg] }));
    return id;
  },
  addAgentMessage: (content, type = "text", cellId, meta) => {
    const msg: ChatMessage = { id: uuidv4(), role: "agent", content, type, cell_id: cellId, meta, timestamp: new Date().toISOString() };
    set((s) => ({ messages: [...s.messages, msg], isTyping: false }));
  },
  updateMessageMeta: (id, meta) =>
    set((s) => ({
      messages: s.messages.map((msg) =>
        msg.id === id ? { ...msg, meta: { ...(msg.meta || {}), ...meta } } : msg,
      ),
    })),
  setTyping: (isTyping) => set({ isTyping }),
  clear: () => set({ messages: [], isTyping: false }),
}));
