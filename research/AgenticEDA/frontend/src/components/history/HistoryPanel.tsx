"use client";

import { useEffect, useState } from "react";
import { useHistory } from "@/hooks/useHistory";
import { useStoryStore } from "@/stores/storyStore";
import VersionItem from "./VersionItem";

interface Props {
  sessionId: string;
  onClose: () => void;
}

export default function HistoryPanel({ sessionId, onClose }: Props) {
  const { versions, loading, fetchVersions, restoreVersion } =
    useHistory(sessionId);
  const [restoringId, setRestoringId] = useState<number | null>(null);
  const clearStory = useStoryStore((s) => s.clear);

  useEffect(() => {
    fetchVersions();
  }, [fetchVersions]);

  const handleRestore = async (versionId: number) => {
    setRestoringId(versionId);
    try {
      await restoreVersion(versionId);
      clearStory(); // Force story to reload
    } catch {
      // Error handling could be improved
    } finally {
      setRestoringId(null);
    }
  };

  const currentVersionId =
    versions.length > 0
      ? Math.max(...versions.map((v) => v.version_id))
      : -1;

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/20">
      <div className="w-full max-w-md max-h-[80vh] bg-white rounded-xl shadow-xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
          <h2 className="text-sm font-semibold text-gray-800">
            Version History
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-lg leading-none"
            aria-label="Close history panel"
          >
            &times;
          </button>
        </div>

        {/* Content */}
        <div className="flex-grow overflow-y-auto px-4 py-2">
          {loading && versions.length === 0 ? (
            <div className="flex items-center justify-center py-8">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
            </div>
          ) : versions.length === 0 ? (
            <p className="text-sm text-gray-400 text-center py-8">
              No versions yet
            </p>
          ) : (
            versions
              .slice()
              .sort((a, b) => b.version_id - a.version_id)
              .map((v) => (
                <VersionItem
                  key={v.version_id}
                  versionId={v.version_id}
                  trigger={v.trigger}
                  timestamp={v.timestamp}
                  isCurrent={v.version_id === currentVersionId}
                  onRestore={() => handleRestore(v.version_id)}
                  restoring={restoringId === v.version_id}
                />
              ))
          )}
        </div>
      </div>
    </div>
  );
}
