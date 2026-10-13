"use client";

import React, { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type { CellOutput as CellOutputType } from "@/lib/types";
import AnsiToHtml from "ansi-to-html";

const InteractivePlot = dynamic(() => import("./InteractivePlot"), {
  ssr: false,
  loading: () => <div className="p-4 text-on-surface/50 text-sm">Loading chart...</div>,
});

const ansiConverter = new AnsiToHtml({ escapeXML: true });

interface Props {
  outputs: CellOutputType[];
}

function asText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  return JSON.stringify(value);
}

function ErrorOutput({ output }: { output: CellOutputType }) {
  const [tbExpanded, setTbExpanded] = useState(false);
  const traceback = output.traceback ?? [];
  const lastLine = traceback[traceback.length - 1] || output.evalue || "";
  const fullHtml = traceback.map((line) => ansiConverter.toHtml(line)).join("\n");
  const lastLineHtml = ansiConverter.toHtml(lastLine);

  return (
    <div className="animate-output-pop border-t border-error/15 bg-error-container/20">
      <div className="flex items-center justify-between px-4 pt-3 text-[10px] uppercase tracking-widest text-error/80">
        <span>Traceback</span>
        <div className="flex items-center gap-2">
          {traceback.length > 1 && (
            <button
              onClick={() => setTbExpanded((v) => !v)}
              className="rounded-full bg-error/10 px-2 py-0.5 text-[9px] font-semibold text-error hover:bg-error/20 transition-colors"
            >
              {tbExpanded ? "Hide traceback" : "Show full traceback"}
            </button>
          )}
          <span className="rounded-full bg-error/10 px-2 py-0.5 text-[9px] font-semibold text-error">Error</span>
        </div>
      </div>
      {tbExpanded ? (
        <pre
          className="font-mono text-[13px] p-4 whitespace-pre-wrap overflow-x-auto text-error"
          dangerouslySetInnerHTML={{ __html: fullHtml }}
        />
      ) : (
        <pre
          className="font-mono text-[13px] p-4 whitespace-pre-wrap overflow-x-auto text-error"
          dangerouslySetInnerHTML={{ __html: lastLineHtml }}
        />
      )}
    </div>
  );
}

function SingleOutput({ output }: { output: CellOutputType }) {
  if (output.output_type === "stream") {
    return (
      <div className="animate-output-pop border-t border-outline-variant/10 bg-surface/70">
        <div className="flex items-center justify-between px-4 pt-3 text-[10px] uppercase tracking-widest text-on-surface-variant">
          <span>Output</span>
          <span className="text-primary/70">stdout</span>
        </div>
        <pre className="bg-surface-container-low font-mono text-[13px] p-4 whitespace-pre-wrap overflow-x-auto text-on-surface">
          {output.text}
        </pre>
      </div>
    );
  }

  if (output.output_type === "error") {
    return <ErrorOutput output={output} />;
  }

  // execute_result or display_data
  const data = output.data;
  if (!data) return null;

  if (data["text/html"]) {
    const html = asText(data["text/html"]);
    return (
      <div className="animate-output-pop border-t border-outline-variant/10 bg-surface/80">
        <div className="px-4 pt-3 text-[10px] uppercase tracking-widest text-on-surface-variant">Rendered Output</div>
        <div
          className="overflow-x-auto p-4 text-sm text-on-surface bg-surface"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>
    );
  }

  if (data["application/vnd.plotly.v1+json"]) {
    const plotlyData = asText(data["application/vnd.plotly.v1+json"]);
    return (
      <div className="animate-output-pop border-t border-outline-variant/10 bg-surface/80 p-4">
        <div className="mb-3 flex items-center justify-between text-[10px] uppercase tracking-widest text-on-surface-variant">
          <span>Interactive chart</span>
          <span className="text-primary">Plotly</span>
        </div>
        <div className="rounded-2xl border border-outline-variant/10 bg-white/60 shadow-[0_8px_24px_rgba(11,28,48,0.05)] backdrop-blur-sm">
          <InteractivePlot plotlyJson={plotlyData} />
        </div>
      </div>
    );
  }

  if (data["image/png"]) {
    const imageData = asText(data["image/png"]);
    return (
      <div className="animate-output-pop border-t border-outline-variant/10 bg-surface/80 p-4">
        <div className="mb-3 flex items-center justify-between text-[10px] uppercase tracking-widest text-on-surface-variant">
          <span>Static figure</span>
          <span className="text-primary/70">PNG</span>
        </div>
        <div className="overflow-hidden rounded-2xl border border-outline-variant/10 bg-white shadow-[0_8px_24px_rgba(11,28,48,0.05)]">
          <img src={`data:image/png;base64,${imageData}`} alt="output" className="block w-full" />
        </div>
      </div>
    );
  }

  if (data["text/plain"]) {
    const text = asText(data["text/plain"]);
    return (
      <div className="animate-output-pop border-t border-outline-variant/10 bg-surface/70">
        <div className="flex items-center justify-between px-4 pt-3 text-[10px] uppercase tracking-widest text-on-surface-variant">
          <span>Result</span>
          <span className="text-primary/70">text</span>
        </div>
        <pre className="bg-surface-container-low font-mono text-[13px] p-4 whitespace-pre-wrap overflow-x-auto text-on-surface">
          {text}
        </pre>
      </div>
    );
  }

  return null;
}

const CellOutput = React.memo(function CellOutput({ outputs }: Props) {
  const filtered = useMemo(
    () => outputs.filter((o) => o.text || o.data || o.traceback),
    [outputs]
  );

  if (filtered.length === 0) return null;

  return (
    <div className="border-t border-outline-variant/10 overflow-hidden">
      {filtered.map((output, i) => (
        <SingleOutput key={i} output={output} />
      ))}
    </div>
  );
});

export default CellOutput;
