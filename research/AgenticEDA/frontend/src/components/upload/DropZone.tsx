"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useRouter } from "next/navigation";
import { uploadFile, runEda } from "@/lib/api";

const ACCEPTED: Record<string, string[]> = {
  "text/csv": [".csv"],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
  "application/vnd.ms-excel": [".xls"],
  "application/json": [".json"],
  "application/x-ndjson": [".jsonl", ".ndjson"],
  "application/octet-stream": [".parquet"],
  "text/plain": [".log", ".txt", ".tsv"],
};

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DropZone() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [maxSubagents, setMaxSubagents] = useState(3);
  const [maxLoops, setMaxLoops] = useState(2);
  const [seedInput, setSeedInput] = useState("");
  const [researchDirection, setResearchDirection] = useState("");

  const onDrop = useCallback((accepted: File[]) => {
    setError(null);
    if (accepted.length > 0) setFile(accepted[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    multiple: false,
  });

  const handleRun = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const data = await uploadFile(file);
      const parsedSeed = seedInput.trim() !== "" ? Number(seedInput.trim()) : null;
      runEda(data.session_id, {
        max_subagents: maxSubagents,
        max_loops: maxLoops,
        seed: parsedSeed != null && Number.isFinite(parsedSeed) ? parsedSeed : undefined,
        research_direction: researchDirection.trim() || undefined,
      }).catch(() => {});
      router.push(`/session/${data.session_id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setUploading(false);
    }
  };

  return (
    <div className="flex flex-col items-center gap-4 w-full max-w-lg">
      <div
        {...getRootProps()}
        className={`w-full border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
          isDragActive
            ? "border-primary bg-primary/5"
            : "border-outline-variant hover:border-outline"
        } bg-surface-container-lowest`}
      >
        <input {...getInputProps()} />
        {file ? (
          <div>
            <span className="material-symbols-outlined text-primary text-3xl mb-2">description</span>
            <p className="font-medium text-on-surface font-body">{file.name}</p>
            <p className="text-sm text-on-surface-variant font-body">{formatSize(file.size)}</p>
          </div>
        ) : (
          <div>
            <span className="material-symbols-outlined text-on-surface-variant text-3xl mb-2">upload_file</span>
            <p className="text-on-surface-variant font-body">
              {isDragActive ? "Drop the file here..." : "Drag & drop a dataset, or click to browse"}
            </p>
          </div>
        )}
      </div>

      {file && (
        <div className="w-full border border-outline-variant rounded-lg p-4 bg-surface-container-lowest">
          <p className="text-sm font-medium text-on-surface mb-3 font-body">Agent Configuration</p>
          <div className="flex gap-6">
            <label className="flex flex-col gap-1 text-sm text-on-surface-variant font-body">
              Subagents per loop
              <select
                value={maxSubagents}
                onChange={(e) => setMaxSubagents(Number(e.target.value))}
                className="px-3 py-1.5 rounded border border-outline-variant bg-surface text-on-surface font-body"
              >
                {[1, 2, 3, 4, 5, 6].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-sm text-on-surface-variant font-body">
              Reasoning depth
              <select
                value={maxLoops}
                onChange={(e) => setMaxLoops(Number(e.target.value))}
                className="px-3 py-1.5 rounded border border-outline-variant bg-surface text-on-surface font-body"
              >
                <option value={1}>Quick</option>
                <option value={2}>Standard</option>
                <option value={5}>Deep</option>
              </select>
            </label>
            <label className="flex flex-col gap-1 text-sm text-on-surface-variant font-body">
              Seed (optional)
              <input
                type="text"
                inputMode="numeric"
                value={seedInput}
                onChange={(e) => setSeedInput(e.target.value.replace(/[^0-9]/g, ""))}
                placeholder="Leave blank for random"
                className="px-3 py-1.5 rounded border border-outline-variant bg-surface text-on-surface font-body w-44"
              />
            </label>
          </div>
          <label className="mt-4 flex flex-col gap-1 text-sm text-on-surface-variant font-body">
            Research direction (optional)
            <textarea
              value={researchDirection}
              onChange={(e) => setResearchDirection(e.target.value)}
              rows={3}
              placeholder="Focus the analysis on anomalies, temporal shifts, operational risks, or a specific business question."
              className="px-3 py-2 rounded border border-outline-variant bg-surface text-on-surface font-body resize-none"
            />
          </label>
        </div>
      )}

      {error && <p className="text-error text-sm font-body">{error}</p>}

      <button
        onClick={handleRun}
        disabled={!file || uploading}
        className="px-6 py-2.5 rounded-lg bg-primary text-on-primary font-medium hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-body"
      >
        {uploading ? "Uploading..." : "Run EDA"}
      </button>
    </div>
  );
}
