"use client";

import { useCallback, useRef } from "react";
import Editor, { type OnMount } from "@monaco-editor/react";

interface Props {
  cellId: string;
  value: string;
  onChange: (id: string, value: string) => void;
  onRun: () => void;
}

export default function CodeEditor({ cellId, value, onChange, onRun }: Props) {
  const editorRef = useRef<Parameters<OnMount>[0] | null>(null);

  const lineCount = value.split("\n").length;
  const height = Math.min(Math.max(lineCount * 20, 60), 400);

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    // Shift+Enter to run
    editor.addCommand(monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => {
      onRun();
    });
  };

  const handleChange = useCallback(
    (val: string | undefined) => {
      onChange(cellId, val ?? "");
    },
    [cellId, onChange]
  );

  return (
    <div className="bg-surface-container-low p-4 font-mono text-[13px]">
      <Editor
        height={height}
        language="python"
        theme="vs-light"
        value={value}
        onChange={handleChange}
        onMount={handleMount}
        options={{
          minimap: { enabled: false },
          lineNumbers: "on",
          scrollBeyondLastLine: false,
          wordWrap: "on",
          fontSize: 13,
          fontFamily: "'JetBrains Mono', monospace",
          automaticLayout: true,
          overviewRulerLanes: 0,
          hideCursorInOverviewRuler: true,
          scrollbar: { vertical: "hidden", horizontal: "hidden", alwaysConsumeMouseWheel: false },
          renderLineHighlight: "none",
          padding: { top: 0, bottom: 0 },
          tabSize: 4,
          folding: false,
          glyphMargin: false,
          lineDecorationsWidth: 0,
          lineNumbersMinChars: 3,
        }}
      />
    </div>
  );
}
