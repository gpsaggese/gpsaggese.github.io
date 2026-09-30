"use client";

interface Props {
  versionId: number;
  trigger: string;
  timestamp: string;
  isCurrent: boolean;
  onRestore: () => void;
  restoring: boolean;
}

export default function VersionItem({
  versionId,
  trigger,
  timestamp,
  isCurrent,
  onRestore,
  restoring,
}: Props) {
  const triggerLower = trigger.toLowerCase();
  const isSystem = triggerLower.includes("system") || triggerLower.includes("auto");
  const isStable = triggerLower.includes("stable");

  return (
    <div className="flex items-start gap-4 relative pl-2">
      {/* Circle indicator */}
      <div
        className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 z-10 ${
          isCurrent
            ? "bg-primary-container text-on-primary-container"
            : "bg-surface-container-low text-outline"
        }`}
      >
        <span className="material-symbols-outlined text-lg">
          {isCurrent ? "star" : "history"}
        </span>
      </div>

      {/* Card */}
      <div
        className={`flex-1 min-w-0 rounded-xl p-4 transition-colors ${
          isCurrent
            ? "bg-white border-l-4 border-primary shadow-sm"
            : "bg-surface-container-low hover:bg-surface-container-high"
        }`}
      >
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-sm font-semibold text-on-surface font-headline">
              Version {versionId}
            </span>
            {isCurrent && (
              <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-primary-container text-on-primary-container">
                Current
              </span>
            )}
          </div>

          {!isCurrent && (
            <button
              onClick={onRestore}
              disabled={restoring}
              className="flex items-center gap-1 text-xs text-primary hover:underline disabled:text-outline disabled:no-underline transition-colors shrink-0"
            >
              <span className="material-symbols-outlined text-sm">settings_backup_restore</span>
              {restoring ? "Restoring..." : "Restore"}
            </button>
          )}
        </div>

        <p className="text-xs text-on-surface-variant mt-1 truncate font-body">{trigger}</p>

        <div className="flex items-center gap-2 mt-2">
          <span className="text-[10px] text-on-surface-variant font-mono">
            {new Date(timestamp).toLocaleString()}
          </span>

          {/* Tags */}
          <div className="flex items-center gap-1">
            {isSystem ? (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-container text-on-surface-variant">
                System
              </span>
            ) : (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-container-highest text-on-surface">
                User
              </span>
            )}
            {isStable && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-secondary-fixed text-on-secondary-container">
                Stable
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
