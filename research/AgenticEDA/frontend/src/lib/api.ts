import type { Cell, Session, StorySection } from "./types";
import { API_BASE } from "./backend";

export interface UploadResponse {
  session_id: string;
  dataset_preview: {
    columns: string[];
    dtypes: Record<string, string>;
    first_5_rows: Record<string, unknown>[];
    row_count: number;
    col_count: number;
  };
  source_format: string;
  load_warnings: string[];
}

export interface SessionListResponse {
  sessions: Session[];
}

export interface StoryResponse {
  title: string;
  executive_summary: string;
  sections: StorySection[];
  generated_at: string;
}

export interface RunResponse {
  status: string;
  session_id: string;
  message?: string;
}

async function readError(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  return text.trim() || res.statusText || `HTTP ${res.status}`;
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}

export async function listSessions(): Promise<SessionListResponse> {
  const res = await fetch(`${API_BASE}/sessions`);
  return res.json();
}

export async function getNotebook(sessionId: string): Promise<Record<string, unknown> | null> {
  const res = await fetch(`${API_BASE}/notebook/${sessionId}`);
  if (!res.ok) return null;
  return res.json();
}

export async function patchNotebook(sessionId: string, cells: Cell[]): Promise<{ status: string; cell_count: number }> {
  const res = await fetch(`${API_BASE}/notebook/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cells }),
  });
  if (!res.ok) {
    throw new Error(`Notebook save failed: ${await readError(res)}`);
  }
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/session/${sessionId}`, { method: "DELETE" });
}

export async function runEda(
  sessionId: string,
  config?: {
    max_subagents?: number;
    max_loops?: number;
    loop_timeout?: number;
    seed?: number;
    research_direction?: string;
  },
): Promise<RunResponse> {
  const res = await fetch(`${API_BASE}/run/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config || {}),
  });
  return res.json();
}

export interface SteeringResponse {
  id: string;
  session_id: string;
  content: string;
  created_at: string;
  status: "queued" | "read";
  read_at?: string | null;
}

export async function sendSteering(
  sessionId: string,
  content: string,
  messageId: string,
): Promise<SteeringResponse> {
  const res = await fetch(`${API_BASE}/run/${sessionId}/steering`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, message_id: messageId }),
  });
  if (!res.ok) {
    throw new Error(`Steering failed: ${await readError(res)}`);
  }
  return res.json();
}

export async function getStory(sessionId: string, format?: "json"): Promise<StoryResponse>;
export async function getStory(sessionId: string, format: "md"): Promise<string>;
export async function getStory(sessionId: string, format: "pdf"): Promise<Blob>;
export async function getStory(
  sessionId: string,
  format: "json" | "md" | "pdf" = "json"
): Promise<StoryResponse | Blob | string> {
  const res = await fetch(`${API_BASE}/story/${sessionId}?format=${format}`, {
    headers: format === "pdf" ? { Accept: "application/pdf" } : undefined,
  });
  if (!res.ok) {
    throw new Error(`Story fetch failed: ${await readError(res)}`);
  }
  if (format === "pdf") {
    const blob = await res.blob();
    if (!blob.size) {
      throw new Error("Story PDF export returned an empty file");
    }
    return blob;
  }
  if (format === "md") {
    return res.text();
  }
  return res.json();
}
