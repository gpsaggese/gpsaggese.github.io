"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { sanitizeLatex } from "@/components/story/StoryMarkdown";
import type { ChatMessage as ChatMessageType } from "@/lib/types";

interface ChatMessageProps {
  message: ChatMessageType;
  onViewCell?: (cellId: string) => void;
  onOpenNotebook?: (notebookId: string) => void;
}

type InvestigationStatus = NonNullable<ChatMessageType["meta"]>["status"];

function statusBadge(status?: InvestigationStatus) {
  const s = status || "running";
  const base = "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-semibold";
  if (s === "complete") return { label: "Complete", className: `${base} bg-emerald-500/10 text-emerald-700` };
  if (s === "failed") return { label: "Failed", className: `${base} bg-red-500/10 text-red-700` };
  if (s === "timeout") return { label: "Timed out", className: `${base} bg-amber-500/10 text-amber-700` };
  if (s === "started") return { label: "Started", className: `${base} bg-primary/10 text-primary` };
  return { label: "Running", className: `${base} bg-primary/10 text-primary` };
}

const TRUNCATE_LENGTH = 250;

function ChatContent({ content, isInvestigation }: { content: string; isInvestigation: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const needsTruncation = isInvestigation && content.length > TRUNCATE_LENGTH;
  const displayText = needsTruncation && !expanded
    ? content.slice(0, TRUNCATE_LENGTH).replace(/\s+\S*$/, "") + "..."
    : content;
  const sanitized = sanitizeLatex(displayText);

  return (
    <div>
      <div className="text-sm font-body text-on-surface prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-li:my-0.5 prose-pre:my-2 prose-code:rounded prose-code:bg-surface-container prose-code:px-1.5 prose-code:py-0.5 prose-code:before:content-none prose-code:after:content-none">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
          {sanitized}
        </ReactMarkdown>
      </div>
      {needsTruncation && !expanded && (
        <button
          onClick={() => setExpanded(true)}
          className="mt-1 text-xs text-primary hover:underline"
        >
          Show full finding
        </button>
      )}
    </div>
  );
}

export default function ChatMessage({ message, onViewCell, onOpenNotebook }: ChatMessageProps) {
  const isUser = message.role === "user";
  const isAction = message.type === "action";
  const isInvestigation = message.meta?.kind === "investigation";
  const isSteering = message.meta?.kind === "steering";
  const badge = isInvestigation ? statusBadge(message.meta?.status) : null;
  const steeringStatus = message.meta?.status === "read" ? "Read" : message.meta?.status === "failed" ? "Failed" : "Queued";

  return (
    <div className={`flex items-start gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      {/* Avatar */}
      {isUser ? (
        <div className="w-6 h-6 rounded bg-secondary-container flex items-center justify-center shrink-0">
          <span className="material-symbols-outlined text-on-secondary-container text-sm">person</span>
        </div>
      ) : (
        <div className="w-6 h-6 rounded bg-primary/10 flex items-center justify-center shrink-0">
          <span className="material-symbols-outlined text-primary text-sm">smart_toy</span>
        </div>
      )}

      {/* Bubble */}
      <div
        className={`max-w-[85%] p-3 ${
          isUser
            ? "bg-primary text-white rounded-2xl rounded-tr-none chat-shadow"
            : isAction
            ? "bg-surface-container-lowest rounded-2xl rounded-tl-none chat-shadow border border-outline-variant/5"
            : "bg-surface-container-lowest rounded-2xl rounded-tl-none chat-shadow border border-outline-variant/5"
        }`}
      >
        {isUser ? (
          <p className="text-sm font-body leading-6 whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="space-y-2">
            <div className="space-y-1">
              <div className="flex items-center gap-2 flex-wrap">
                {isInvestigation ? (
                  <>
                    <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-primary">
                      <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                      Investigation
                    </div>
                    {badge && <span className={badge.className}>{badge.label}</span>}
                  </>
                ) : isAction ? (
                  <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.2em] text-primary">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Agent update
                  </div>
                ) : null}
              </div>
              {typeof message.meta?.confidence === "number" && (
                <div className="flex items-center gap-1.5">
                  <div className="w-16 h-1.5 rounded-full bg-outline-variant/20 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${Math.round(message.meta.confidence * 100)}%` }}
                    />
                  </div>
                  <span className="text-[9px] font-medium text-on-surface-variant">
                    {Math.round(message.meta.confidence * 100)}% confidence
                  </span>
                </div>
              )}
            </div>

            {isInvestigation && message.meta?.title && (
              <div className="text-xs font-semibold text-on-surface">
                {message.meta.title}
              </div>
            )}
            <ChatContent content={message.content} isInvestigation={isInvestigation} />
          </div>
        )}

        {message.meta?.notebook_id && onOpenNotebook && (
          <button
            onClick={() => onOpenNotebook(message.meta!.notebook_id!)}
            className="mt-2 inline-flex items-center gap-1.5 rounded-lg border border-outline-variant/20 bg-surface-container px-2 py-1 text-xs text-on-surface hover:bg-surface-container-high transition-colors"
          >
            <span className="material-symbols-outlined text-[16px] text-primary">science</span>
            Open investigation tab
          </button>
        )}

        {message.cell_id && onViewCell && (
          <button
            onClick={() => onViewCell(message.cell_id!)}
            className="mt-1.5 inline-flex items-center gap-1 text-xs text-primary hover:underline"
          >
            <span className="material-symbols-outlined text-xs">description</span>
            View in notebook
          </button>
        )}

        <div className={`mt-1 flex items-center gap-2 text-[10px] ${isUser ? "justify-end text-white/60" : "text-on-surface-variant"}`}>
          <span>{new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
          {isUser && isSteering && (
            <span className="inline-flex items-center gap-1">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  message.meta?.status === "read"
                    ? "bg-emerald-200"
                    : message.meta?.status === "failed"
                    ? "bg-red-200"
                    : "bg-white/60 animate-pulse"
                }`}
              />
              {steeringStatus}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
