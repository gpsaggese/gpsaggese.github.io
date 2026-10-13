import { create } from "zustand";
import type { Session } from "@/lib/types";

interface SessionState {
  activeSessionId: string | null;
  sessions: Session[];
  setActiveSession: (id: string | null) => void;
  setSessions: (sessions: Session[]) => void;
  addSession: (session: Session) => void;
  removeSession: (id: string) => void;
}

export const useSessionStore = create<SessionState>((set) => ({
  activeSessionId: null,
  sessions: [],
  setActiveSession: (id) => set({ activeSessionId: id }),
  setSessions: (sessions) => set({ sessions }),
  addSession: (session) => set((s) => ({ sessions: [...s.sessions, session] })),
  removeSession: (id) =>
    set((s) => ({
      sessions: s.sessions.filter((sess) => sess.session_id !== id),
      activeSessionId: s.activeSessionId === id ? null : s.activeSessionId,
    })),
}));
