"use client";

import { useState, useRef, useCallback, useEffect, type ReactNode } from "react";

interface Props {
  left: ReactNode;
  center: ReactNode;
  right: ReactNode;
}

const LEFT_DEFAULT = 200;
const LEFT_MIN = 140;
const LEFT_MAX = 420;
const LEFT_COLLAPSE = 80;

const RIGHT_DEFAULT = 320;
const RIGHT_MIN = 220;
const RIGHT_MAX = 500;
const RIGHT_COLLAPSE = 100;

export default function ThreeColumnLayout({ left, center, right }: Props) {
  const [leftWidth, setLeftWidth] = useState(LEFT_DEFAULT);
  const [rightWidth, setRightWidth] = useState(RIGHT_DEFAULT);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);

  // Refs to avoid per-pixel re-renders during drag
  const dragging = useRef<"left" | "right" | null>(null);
  const leftRef = useRef<HTMLElement>(null);
  const rightRef = useRef<HTMLElement>(null);
  const savedLeft = useRef(LEFT_DEFAULT);
  const savedRight = useRef(RIGHT_DEFAULT);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (dragging.current === "left" && leftRef.current) {
      const w = Math.max(0, e.clientX);
      leftRef.current.style.width = `${Math.min(LEFT_MAX, Math.max(0, w))}px`;
    } else if (dragging.current === "right" && rightRef.current) {
      const w = Math.max(0, window.innerWidth - e.clientX);
      rightRef.current.style.width = `${Math.min(RIGHT_MAX, Math.max(0, w))}px`;
    }
  }, []);

  const onMouseUp = useCallback((e: MouseEvent) => {
    document.body.style.cursor = "";
    document.body.style.userSelect = "";

    if (dragging.current === "left") {
      const w = Math.max(0, e.clientX);
      if (w < LEFT_COLLAPSE) {
        setLeftCollapsed(true);
        setLeftWidth(0);
      } else {
        const clamped = Math.min(LEFT_MAX, Math.max(LEFT_MIN, w));
        setLeftCollapsed(false);
        setLeftWidth(clamped);
        savedLeft.current = clamped;
      }
    } else if (dragging.current === "right") {
      const w = Math.max(0, window.innerWidth - e.clientX);
      if (w < RIGHT_COLLAPSE) {
        setRightCollapsed(true);
        setRightWidth(0);
      } else {
        const clamped = Math.min(RIGHT_MAX, Math.max(RIGHT_MIN, w));
        setRightCollapsed(false);
        setRightWidth(clamped);
        savedRight.current = clamped;
      }
    }

    dragging.current = null;
    document.removeEventListener("mousemove", onMouseMove);
  }, [onMouseMove]);

  const startDrag = useCallback((side: "left" | "right") => {
    dragging.current = side;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp, { once: true });
  }, [onMouseMove, onMouseUp]);

  const toggleLeft = useCallback(() => {
    if (leftCollapsed) {
      setLeftCollapsed(false);
      setLeftWidth(savedLeft.current);
    } else {
      setLeftCollapsed(true);
      setLeftWidth(0);
    }
  }, [leftCollapsed]);

  const toggleRight = useCallback(() => {
    if (rightCollapsed) {
      setRightCollapsed(false);
      setRightWidth(savedRight.current);
    } else {
      setRightCollapsed(true);
      setRightWidth(0);
    }
  }, [rightCollapsed]);

  // Sync refs on state change (for after drag commits)
  useEffect(() => {
    if (leftRef.current && !dragging.current) {
      leftRef.current.style.width = leftCollapsed ? "0px" : `${leftWidth}px`;
    }
  }, [leftWidth, leftCollapsed]);

  useEffect(() => {
    if (rightRef.current && !dragging.current) {
      rightRef.current.style.width = rightCollapsed ? "0px" : `${rightWidth}px`;
    }
  }, [rightWidth, rightCollapsed]);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Left sidebar */}
      <aside
        ref={leftRef}
        className="flex-shrink-0 border-r border-gray-200 bg-gray-50 overflow-y-auto overflow-x-hidden transition-[width] duration-200"
        style={{ width: leftCollapsed ? 0 : leftWidth }}
      >
        {!leftCollapsed && <div className="p-3">{left}</div>}
      </aside>

      {/* Left drag handle */}
      <div
        className="flex-shrink-0 w-1.5 cursor-col-resize flex items-center justify-center hover:bg-gray-200 group relative"
        onMouseDown={() => startDrag("left")}
        onDoubleClick={toggleLeft}
      >
        <div className="w-0.5 h-8 rounded-full bg-gray-300 group-hover:bg-gray-400 transition-colors" />
        {leftCollapsed && (
          <button
            onClick={toggleLeft}
            className="absolute -right-3 top-1/2 -translate-y-1/2 w-5 h-8 flex items-center justify-center text-gray-400 hover:text-gray-600 text-xs z-10"
            aria-label="Expand left sidebar"
          >
            &#9654;
          </button>
        )}
      </div>

      {/* Center */}
      <main className="flex-grow overflow-y-auto min-w-0">{center}</main>

      {/* Right drag handle */}
      <div
        className="flex-shrink-0 w-1.5 cursor-col-resize flex items-center justify-center hover:bg-gray-200 group relative"
        onMouseDown={() => startDrag("right")}
        onDoubleClick={toggleRight}
      >
        <div className="w-0.5 h-8 rounded-full bg-gray-300 group-hover:bg-gray-400 transition-colors" />
        {rightCollapsed && (
          <button
            onClick={toggleRight}
            className="absolute -left-3 top-1/2 -translate-y-1/2 w-5 h-8 flex items-center justify-center text-gray-400 hover:text-gray-600 text-xs z-10"
            aria-label="Expand right sidebar"
          >
            &#9664;
          </button>
        )}
      </div>

      {/* Right sidebar */}
      <aside
        ref={rightRef}
        className="flex-shrink-0 border-l border-gray-200 bg-gray-50 overflow-y-auto overflow-x-hidden transition-[width] duration-200"
        style={{ width: rightCollapsed ? 0 : rightWidth }}
      >
        {!rightCollapsed && <div className="p-3">{right}</div>}
      </aside>
    </div>
  );
}
