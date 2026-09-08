"use client";

interface Props {
  cellType: "code" | "markdown";
  onRun: () => void;
  onDelete: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
}

export default function CellToolbar({
  cellType,
  onRun,
  onDelete,
  onMoveUp,
  onMoveDown,
}: Props) {
  const btnClass =
    "p-1 rounded text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors";

  return (
    <div className="flex items-center gap-0.5 py-1 px-2">
      <button className={`${btnClass} hover:text-primary`} onClick={onRun} title="Run cell">
        <span className="material-symbols-outlined text-[18px]">play_arrow</span>
      </button>
      <button className={btnClass} onClick={onMoveUp} title="Move up">
        <span className="material-symbols-outlined text-[18px]">keyboard_arrow_up</span>
      </button>
      <button className={btnClass} onClick={onMoveDown} title="Move down">
        <span className="material-symbols-outlined text-[18px]">keyboard_arrow_down</span>
      </button>
      <button className={`${btnClass} hover:text-error`} onClick={onDelete} title="Delete cell">
        <span className="material-symbols-outlined text-[18px]">delete</span>
      </button>
      <span className="ml-auto text-[10px] text-outline font-body uppercase tracking-wide">
        {cellType}
      </span>
    </div>
  );
}
